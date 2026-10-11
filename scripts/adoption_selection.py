"""Single-source TensorBoard checkpoint selection for strict adoption."""
from __future__ import annotations

import bisect
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import statistics
import struct
from typing import Iterable

from deployment.contracts import ContractError, path_contains_link_or_reparse


_CKPT_RE = re.compile(r"model_([0-9]+)\.pt$")


def checkpoint_iteration(path: Path) -> int:
    match = _CKPT_RE.fullmatch(path.name)
    if match is None:
        raise ContractError(f"checkpoint name is not model_<iteration>.pt: {path.name}")
    return int(match.group(1))


def scalar_at_or_before(steps: list[int], values: list[float], iteration: int) -> float:
    index = bisect.bisect_right(steps, iteration) - 1
    if index < 0:
        raise ContractError(f"no scalar value exists at/before iteration {iteration}")
    value = float(values[index])
    if not math.isfinite(value):
        raise ContractError("checkpoint selection scalar is non-finite")
    return value


def trailing_median_at_checkpoint(
    steps: list[int], values: list[float], iteration: int, window: int = 200,
) -> float:
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ContractError("checkpoint iteration must be a non-negative integer")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ContractError("behavior-loss window must be a positive integer")
    if len(steps) != len(values):
        raise ContractError("behavior-loss scalar steps/values length mismatch")

    # RSL-RL writes model_0.pt after iteration zero, so early checkpoints have
    # fewer than ``window`` loss samples by construction.  Bind each checkpoint
    # to every sample that can exist from iteration zero through that checkpoint:
    # [max(0, iteration-window+1), iteration].  Missing/duplicated steps are a
    # provenance failure rather than permission to substitute an older sample.
    first_step = max(0, iteration - window + 1)
    begin = bisect.bisect_left(steps, first_step)
    end = bisect.bisect_right(steps, iteration)
    expected_steps = list(range(first_step, iteration + 1))
    selected_steps = steps[begin:end]
    if selected_steps != expected_steps:
        raise ContractError(
            f"checkpoint {iteration} behavior-loss window is missing, duplicated, or unordered"
        )
    selected = [float(value) for value in values[begin:end]]
    if any(not math.isfinite(value) for value in selected):
        raise ContractError("behavior-loss trailing window contains non-finite values")
    return float(statistics.median(selected))


def select_teacher_candidates(
    checkpoints: Iterable[Path], steps: list[int], values: list[float], *,
    count: int = 5, separation: int = 500, window: int = 200,
) -> list[dict]:
    """placed_success trailing-median 상위 checkpoint를 고른다 (★F70★).

    ⚠️순간 Train/mean_reward 선발은 금지다. 성공 에피소드는 placed_success DoneTerm으로
    3-step 만에 조기 종료되어 에피소드 보상 합이 깎이므로, mean_reward는 성공률이 오르기
    ★직전★에 정점을 찍는다(F69 실측: 정점 5200=108.8, 학습 placed 0.468 — 후기 0.92
    구간보다 높음). 그 선발이 점화 중간 checkpoint 5개만 게이트에 올려 채택을 전멸시켰다
    (평가 placed 74~81% vs 게이트 95.1%). F68은 점화(7000)와 정점이 우연히 겹쳐 살았을 뿐.
    ⇒ 게이트가 재는 지표 그대로(trailing-200 median) 선발한다. 형태는 student 선발과
    동일(최종 checkpoint 고정 + separation)이고 방향만 반대다(높을수록 좋음).
    """
    checkpoint_rows = []
    for path in checkpoints:
        iteration = checkpoint_iteration(path)
        checkpoint_rows.append({
            "checkpoint": path,
            "iteration": iteration,
            "selection_value": trailing_median_at_checkpoint(steps, values, iteration, window),
        })
    if not checkpoint_rows:
        return []
    final = max(checkpoint_rows, key=lambda row: row["iteration"])
    selected = [final]
    ranked = sorted(
        (row for row in checkpoint_rows if row is not final),
        key=lambda row: (-row["selection_value"], -row["iteration"]),
    )
    for row in ranked:
        if all(abs(row["iteration"] - prior["iteration"]) >= separation for prior in selected):
            selected.append(row)
            if len(selected) == count:
                break
    return selected


def select_student_candidates(
    checkpoints: Iterable[Path], steps: list[int], values: list[float], *,
    count: int = 5, separation: int = 500, window: int = 200,
) -> list[dict]:
    checkpoint_rows = []
    for path in checkpoints:
        iteration = checkpoint_iteration(path)
        checkpoint_rows.append({
            "checkpoint": path,
            "iteration": iteration,
            "selection_value": trailing_median_at_checkpoint(steps, values, iteration, window),
        })
    if not checkpoint_rows:
        return []
    final = max(checkpoint_rows, key=lambda row: row["iteration"])
    selected = [final]
    ranked = sorted(
        (row for row in checkpoint_rows if row is not final),
        key=lambda row: (row["selection_value"], -row["iteration"]),
    )
    for row in ranked:
        if all(abs(row["iteration"] - prior["iteration"]) >= separation for prior in selected):
            selected.append(row)
            if len(selected) == count:
                break
    return selected


def _read_receipt_bound_event(
    run_dir: Path, name: str, binding: object,
) -> bytes:
    """Read one event once and bind both parsing and hashing to that handle."""
    if not isinstance(name, str) or not name.startswith("events.out.tfevents.") \
            or Path(name).name != name:
        raise ContractError("TensorBoard event inventory contains an invalid name")
    if not isinstance(binding, dict) or set(binding) != {"sha256", "size"}:
        raise ContractError("TensorBoard event inventory binding is malformed")
    expected_sha256 = binding.get("sha256")
    expected_size = binding.get("size")
    if re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)) is None \
            or isinstance(expected_size, bool) \
            or not isinstance(expected_size, int) or expected_size <= 0:
        raise ContractError("TensorBoard event inventory binding is malformed")

    event_path = run_dir / name
    if path_contains_link_or_reparse(event_path.absolute()):
        raise ContractError("TensorBoard event path traverses a symlink or reparse point")
    try:
        with event_path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != expected_size:
                raise ContractError("TensorBoard event size/type differs from its receipt")
            payload = stream.read()
            closed_over = os.fstat(stream.fileno())
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot read receipt-bound TensorBoard event: {exc}") from exc
    if not stat.S_ISREG(closed_over.st_mode) or closed_over.st_size != expected_size \
            or len(payload) != expected_size:
        raise ContractError("TensorBoard event changed while it was read")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ContractError("TensorBoard event bytes differ from their training receipt")
    return payload


def _iter_event_protos(payload: bytes, *, name: str):
    """Decode exact in-memory TFRecord bytes with the same CRC contract as TensorBoard."""
    from google.protobuf.message import DecodeError
    from tensorboard.compat.proto import event_pb2
    from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

    offset = 0
    records = 0
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise ContractError(f"TensorBoard event has a truncated TFRecord header: {name}")
        length_bytes = payload[offset:offset + 8]
        expected_length_crc = struct.unpack_from("<I", payload, offset + 8)[0]
        if masked_crc32c(length_bytes) != expected_length_crc:
            raise ContractError(f"TensorBoard event has an invalid TFRecord header CRC: {name}")
        record_size = struct.unpack("<Q", length_bytes)[0]
        offset += 12
        if record_size > len(payload) - offset - 4:
            raise ContractError(f"TensorBoard event has a truncated TFRecord payload: {name}")
        record = payload[offset:offset + record_size]
        offset += record_size
        expected_record_crc = struct.unpack_from("<I", payload, offset)[0]
        offset += 4
        if masked_crc32c(record) != expected_record_crc:
            raise ContractError(f"TensorBoard event has an invalid TFRecord payload CRC: {name}")
        try:
            yield event_pb2.Event.FromString(record)
        except DecodeError as exc:
            raise ContractError(f"TensorBoard event protobuf is invalid: {name}") from exc
        records += 1
    if records == 0:
        raise ContractError(f"TensorBoard event contains no TFRecord records: {name}")


def load_scalar_series(
    run_dir: Path, tag: str, tensorboard_events: object,
) -> tuple[list[int], list[float]]:
    """Load one scalar solely from receipt-bound, same-open event bytes.

    The restart/purge behavior below mirrors TensorBoard's legacy scalar
    accumulation semantics without invoking its path-based file loader.
    """
    from tensorboard.backend.event_processing import event_util
    from tensorboard.compat.proto import event_pb2

    raw_run_dir = Path(run_dir).expanduser().absolute()
    if path_contains_link_or_reparse(raw_run_dir) or not raw_run_dir.is_dir():
        raise ContractError("TensorBoard run directory is missing or traverses a reparse point")
    if not isinstance(tag, str) or not tag:
        raise ContractError("TensorBoard scalar tag is invalid")
    if not isinstance(tensorboard_events, dict) or not tensorboard_events:
        raise ContractError("TensorBoard scalar selection requires a receipt event inventory")

    scalar_events: list[tuple[int, float]] = []
    file_version: float | None = None
    most_recent_step = -1
    for name in sorted(tensorboard_events):
        payload = _read_receipt_bound_event(raw_run_dir, name, tensorboard_events[name])
        for event in _iter_event_protos(payload, name=name):
            if event.HasField("file_version"):
                try:
                    file_version = event_util.ParseFileVersion(event.file_version)
                except ValueError as exc:
                    raise ContractError("TensorBoard event file_version is invalid") from exc

            if file_version is not None and file_version >= 2:
                if event.HasField("session_log") \
                        and event.session_log.status == event_pb2.SessionLog.START:
                    scalar_events = [row for row in scalar_events if row[0] < event.step]
            elif event.step < most_recent_step and event.HasField("summary"):
                if any(value.tag == tag for value in event.summary.value):
                    scalar_events = [row for row in scalar_events if row[0] < event.step]
            else:
                most_recent_step = event.step

            if event.HasField("summary"):
                for value in event.summary.value:
                    if value.tag == tag and value.HasField("simple_value"):
                        scalar_events.append((int(event.step), float(value.simple_value)))

    steps = [step for step, _value in scalar_events]
    values = [value for _step, value in scalar_events]
    if not steps or any(right <= left for left, right in zip(steps, steps[1:])) \
            or any(not math.isfinite(value) for value in values):
        raise ContractError(f"TensorBoard scalar series is empty, unordered, or non-finite: {tag}")
    return steps, values
