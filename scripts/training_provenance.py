"""Fail-closed provenance receipts for Teacher and Vision training runs."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tempfile
from typing import Iterable

from deployment.contracts import ContractError, path_contains_link_or_reparse, sha256_file
from common.file_snapshot import FileSnapshotError, read_regular_file_snapshot
from common.runtime_authority import (
    RuntimeAuthorityError,
    validate_runtime_authority_binding,
)
from common.nurec_provenance import load_nurec_provenance, validate_sealed_nurec
from deployment.release import (
    REQUIRED_RELEASE_CONFIGS,
    publish_regular_file_no_replace,
    require_clean_git_commit,
    training_stack_provenance,
)


TRAINING_RECEIPT_SCHEMA_VERSION = 2
TRAINING_RECEIPT_NAME = "run_manifest.json"
TRAINING_RECEIPT_TYPE = "ur5e_training_run"


def _read_regular_file_once(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> bytes:
    """Read one stable regular-file payload through a single open handle."""
    try:
        return read_regular_file_snapshot(
            path,
            label=label,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        ).payload
    except FileSnapshotError as exc:
        raise ContractError(str(exc)) from exc


def canonical_config_hashes(project_root: Path | str) -> dict[str, str]:
    root = Path(project_root).resolve()
    result: dict[str, str] = {}
    for relative in sorted(REQUIRED_RELEASE_CONFIGS):
        path = root / relative
        if path_contains_link_or_reparse(path.absolute()) or not path.is_file():
            raise ContractError(f"canonical training config is missing/non-regular: {relative}")
        try:
            result[relative] = read_regular_file_snapshot(
                path, label=f"canonical training config {relative}"
            ).sha256
        except FileSnapshotError as exc:
            raise ContractError(str(exc)) from exc
    return result


def capture_training_context(project_root: Path | str, *, require_nurec: bool = False) -> dict:
    """Capture a clean, exact context immediately before/after one run."""
    root = Path(project_root).resolve()
    return {
        "schema_version": 1,
        "project_git_commit": require_clean_git_commit(root),
        "training_stack": training_stack_provenance(root),
        "canonical_configs": canonical_config_hashes(root),
        "optional_assets": (
            {"nurec": load_nurec_provenance(root)} if require_nurec else {}
        ),
    }


def _validate_context(value: object) -> dict:
    # Reuse the release validator through a deliberately local import.  This
    # keeps the exact Isaac/ML contract single-sourced without exposing private
    # paths in the receipt.
    from deployment.release import _validate_training_stack_provenance

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ContractError("training context is missing or has wrong schema")
    commit = value.get("project_git_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ContractError("training context requires an exact clean Git commit")
    _validate_training_stack_provenance(value.get("training_stack"))
    configs = value.get("canonical_configs")
    if not isinstance(configs, dict) or set(configs) != REQUIRED_RELEASE_CONFIGS:
        raise ContractError("training context canonical config set mismatch")
    if any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in configs.values()):
        raise ContractError("training context contains an invalid config digest")
    optional = value.get("optional_assets")
    if not isinstance(optional, dict) or not set(optional).issubset({"nurec"}):
        raise ContractError("training context optional asset set is invalid")
    if "nurec" in optional:
        validate_sealed_nurec(optional["nurec"])
    return value


def _relative_run_root(run_dir: Path, project_root: Path, experiment_name: str) -> str:
    try:
        relative = run_dir.relative_to(project_root)
    except ValueError as exc:
        raise ContractError("training run is outside the project root") from exc
    parts = relative.parts
    if (
        len(parts) != 4
        or parts[:3] != ("logs", "rsl_rl", experiment_name)
        or parts[3].startswith("_")
    ):
        raise ContractError("training run is not a canonical timestamped experiment directory")
    return relative.as_posix()


def _sanitize_arguments(arguments: Iterable[str]) -> list[str]:
    sanitized: list[str] = []
    for raw in arguments:
        value = str(raw)
        path = Path(value).expanduser()
        if path.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value):
            # PureWindowsPath는 \와 / 모두 구분자로 본다 — Linux에서도 Windows 경로의 디렉터리가 새지 않는다
            sanitized.append(f"<external>/{PureWindowsPath(value).name}")
        else:
            sanitized.append(value.replace("\\", "/"))
    return sanitized


def _tensorboard_event_inventory(run_dir: Path) -> dict[str, dict[str, int | str]]:
    """Return the exact root-level TensorBoard event inventory for one run."""
    raw_events = sorted(run_dir.glob("events.out.tfevents.*"), key=lambda path: path.name)
    if not raw_events:
        raise ContractError("training run has no TensorBoard event artifact")
    inventory: dict[str, dict[str, int | str]] = {}
    for path in raw_events:
        if path_contains_link_or_reparse(path.absolute()) or not path.is_file():
            raise ContractError("training TensorBoard event is missing/non-regular or linked")
        size = path.stat().st_size
        if size <= 0:
            raise ContractError("training TensorBoard event is empty")
        inventory[path.name] = {"sha256": sha256_file(path), "size": size}
    return inventory


def _verify_tensorboard_event_inventory(
    run_dir: Path, value: object, *, required: bool,
) -> dict[str, dict[str, int | str]] | None:
    """Validate a receipt's event binding against the exact current inventory."""
    if value is None and not required:
        # Compatibility for schema-2 receipts created before event binding was
        # introduced.  Such a receipt remains usable as training lineage, but
        # adoption callers pass ``required=True`` and reject it.
        return None
    if not isinstance(value, dict) or not value:
        raise ContractError("training receipt lacks required TensorBoard event binding")
    for name, item in value.items():
        if not isinstance(name, str) or not name.startswith("events.out.tfevents.") \
                or Path(name).name != name:
            raise ContractError("training receipt TensorBoard event name is invalid")
        if not isinstance(item, dict) or set(item) != {"sha256", "size"} \
                or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None \
                or isinstance(item.get("size"), bool) \
                or not isinstance(item.get("size"), int) or item["size"] <= 0:
            raise ContractError("training receipt TensorBoard event binding is malformed")
    current = _tensorboard_event_inventory(run_dir)
    if value != current:
        raise ContractError("training TensorBoard event inventory/hash drift")
    return current


def _verify_checkpoint_inventory(
    run_dir: Path, value: object, *, required: bool,
) -> dict[str, dict[str, int | str]] | None:
    """Validate every receipt-bound checkpoint when selection needs the full set."""
    if not required:
        return None
    if not isinstance(value, dict) or not value:
        raise ContractError("training receipt checkpoint inventory is missing")
    for name, item in value.items():
        if re.fullmatch(r"model_[0-9]+\.pt", str(name)) is None or Path(str(name)).name != name:
            raise ContractError("training receipt checkpoint name is invalid")
        if not isinstance(item, dict) or set(item) != {"sha256", "size"} \
                or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None \
                or isinstance(item.get("size"), bool) \
                or not isinstance(item.get("size"), int) or item["size"] <= 0:
            raise ContractError("training receipt checkpoint binding is malformed")
    current_paths = sorted(run_dir.glob("model_[0-9]*.pt"), key=lambda path: path.name)
    if {path.name for path in current_paths} != set(value):
        raise ContractError("training checkpoint inventory drift")
    for path in current_paths:
        if path_contains_link_or_reparse(path.absolute()) or not path.is_file():
            raise ContractError("training checkpoint inventory contains an unsafe path")
        expected = value[path.name]
        if expected != {"sha256": sha256_file(path), "size": path.stat().st_size}:
            raise ContractError("training checkpoint inventory/hash drift")
    return value


def verify_training_checkpoint_receipt(
    checkpoint_path: Path | str,
    *,
    project_root: Path | str,
    expected_experiment: str | None = None,
    require_tensorboard_events: bool = False,
    require_all_checkpoints: bool = False,
) -> dict:
    """Verify a parent/selected checkpoint against its immutable run receipt."""
    root = Path(project_root).resolve()
    raw = Path(checkpoint_path).expanduser().absolute()
    if path_contains_link_or_reparse(raw):
        raise ContractError("training checkpoint path contains a symlink/reparse point")
    checkpoint = raw.resolve()
    if not checkpoint.is_file() or re.fullmatch(r"model_[0-9]+\.pt", checkpoint.name) is None:
        raise ContractError("training checkpoint must be a canonical model_<iteration>.pt file")
    run_dir = checkpoint.parent
    receipt_path = run_dir / TRAINING_RECEIPT_NAME
    if path_contains_link_or_reparse(receipt_path.absolute()) or not receipt_path.is_file():
        raise ContractError("checkpoint has no trusted schema-2 training receipt")
    receipt_bytes = _read_regular_file_once(receipt_path, label="training receipt")
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode training receipt: {exc}") from exc
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != TRAINING_RECEIPT_SCHEMA_VERSION
        or receipt.get("receipt_type") != TRAINING_RECEIPT_TYPE
    ):
        raise ContractError("legacy/malformed training receipt is not deployable")
    experiment = receipt.get("experiment_name")
    if not isinstance(experiment, str) or (expected_experiment and experiment != expected_experiment):
        raise ContractError("training receipt experiment mismatch")
    expected_run_root = _relative_run_root(run_dir, root, experiment)
    if receipt.get("run_root") != expected_run_root:
        raise ContractError("training receipt run-root binding mismatch")
    context = _validate_context(receipt.get("source"))
    expected_purpose = (
        "teacher-training"
        if experiment == "ur5e_binpick_teacher"
        else "vision-training"
    )
    try:
        validate_runtime_authority_binding(
            receipt.get("runtime_authority"),
            expected_git_commit=context["project_git_commit"],
            expected_configs=context["canonical_configs"],
            expected_purpose=expected_purpose,
        )
    except RuntimeAuthorityError as exc:
        raise ContractError(f"training receipt runtime authority is invalid: {exc}") from exc
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContractError("training receipt artifacts are missing")
    checkpoints = artifacts.get("checkpoints")
    if not isinstance(checkpoints, dict) or checkpoint.name not in checkpoints:
        raise ContractError("selected checkpoint is not bound by its training receipt")
    selected = checkpoints[checkpoint.name]
    if (
        not isinstance(selected, dict)
        or selected.get("sha256") != sha256_file(checkpoint)
        or selected.get("size") != checkpoint.stat().st_size
    ):
        raise ContractError("selected checkpoint differs from its training receipt")
    parameters = artifacts.get("parameters")
    if not isinstance(parameters, dict) or set(parameters) != {"params/agent.yaml", "params/env.yaml"}:
        raise ContractError("training receipt parameter set mismatch")
    for relative, digest in parameters.items():
        path = run_dir / PurePosixPath(relative)
        if path_contains_link_or_reparse(path.absolute()) or not path.is_file():
            raise ContractError(f"training receipt parameter is missing: {relative}")
        try:
            _read_regular_file_once(
                path,
                label=f"training receipt parameter {relative}",
                expected_sha256=digest,
            )
        except ContractError as exc:
            raise ContractError(f"training receipt parameter drift: {relative}: {exc}") from exc
    tensorboard_events = _verify_tensorboard_event_inventory(
        run_dir,
        artifacts.get("tensorboard_events"),
        required=require_tensorboard_events,
    )
    checkpoint_inventory = _verify_checkpoint_inventory(
        run_dir, checkpoints, required=require_all_checkpoints
    )
    return {
        "schema_version": 1,
        "run_root": expected_run_root,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": selected["sha256"],
        "receipt_sha256": receipt_sha256,
        "task_id": receipt.get("task_id"),
        "mode": receipt.get("mode"),
        "source": context,
        "parameters": dict(parameters),
        "tensorboard_events": tensorboard_events,
        "checkpoints": checkpoint_inventory,
    }


def finalize_training_receipt(
    run_dir: Path | str,
    *,
    project_root: Path | str,
    experiment_name: str,
    task_id: str,
    mode: str,
    started_at: str,
    completed_at: str,
    arguments: Iterable[str],
    start_context: dict,
    end_context: dict,
    inputs: Iterable[dict] = (),
    runtime_authority: dict | None = None,
    accept_parent_commit_drift_from: str | None = None,
) -> Path:
    """Atomically bind one completed run; any start/end drift is fatal.

    ★F95-b(2026-09-22)★ `accept_parent_commit_drift_from`에 부모(스승) 체크포인트 run의 ★정확한 40자
    commit★을 주면, 그 부모 입력에 한해 ★project_git_commit만★ 다른 컨텍스트를 허용한다(나머지 필드가
    하나라도 다르면 그대로 거부). 허용 사실은 receipt의 그 입력에 `context_drift`로 남는다.
    `scripts/adoption.py::verify_adopted_checkpoint`의 같은 이름 옵션과 ★한 쌍★이다 — 증류 시작에서만
    허용하고 봉인에서 거부하면 16시간 학습이 ★끝에서★ 버려진다(512 env 프로브에서 실측).
    """
    root = Path(project_root).resolve()
    raw_run = Path(run_dir).absolute()
    if path_contains_link_or_reparse(raw_run) or not raw_run.is_dir():
        raise ContractError("training run path is missing or contains a symlink/reparse point")
    run = raw_run.resolve()
    run_root = _relative_run_root(run, root, experiment_name)
    start = _validate_context(start_context)
    end = _validate_context(end_context)
    if start != end:
        raise ContractError("Git/config/Isaac training context changed while the run was active")
    expected_purpose = (
        "teacher-training"
        if experiment_name == "ur5e_binpick_teacher"
        else "vision-training"
    )
    try:
        frozen_runtime_authority = validate_runtime_authority_binding(
            runtime_authority,
            expected_git_commit=start["project_git_commit"],
            expected_configs=start["canonical_configs"],
            expected_purpose=expected_purpose,
        )
    except RuntimeAuthorityError as exc:
        raise ContractError(f"training runtime authority is invalid: {exc}") from exc
    params = {
        relative: run / PurePosixPath(relative)
        for relative in ("params/agent.yaml", "params/env.yaml")
    }
    if any(path_contains_link_or_reparse(path.absolute()) or not path.is_file() for path in params.values()):
        raise ContractError("completed training run has incomplete parameter snapshots")
    checkpoints = sorted(run.glob("model_[0-9]*.pt"))
    if not checkpoints or any(path_contains_link_or_reparse(path.absolute()) for path in checkpoints):
        raise ContractError("completed training run has no safe checkpoint artifacts")
    tensorboard_events = _tensorboard_event_inventory(run)
    input_records = list(inputs)
    for _index, item in enumerate(input_records):
        if not isinstance(item, dict) or re.fullmatch(r"[0-9a-f]{64}", str(item.get("receipt_sha256", ""))) is None:
            raise ContractError("training input is not backed by a verified parent receipt")
        parent_context = _validate_context(item.get("source"))
        compatible_parent = parent_context
        if mode == "nurec" and not parent_context["optional_assets"]:
            compatible_parent = {**parent_context, "optional_assets": start["optional_assets"]}
        if compatible_parent != start and accept_parent_commit_drift_from is not None:
            # ★F95-b★ 부모 commit을 정확히 지목했고 ★commit만★ 다를 때에 한해 허용하고 기록한다.
            parent_commit = parent_context["project_git_commit"]
            if re.fullmatch(r"[0-9a-f]{40}", str(accept_parent_commit_drift_from)) is not None \
                    and accept_parent_commit_drift_from == parent_commit \
                    and {**compatible_parent, "project_git_commit": start["project_git_commit"]} == start:
                input_records[_index] = {**item, "context_drift": {
                    "parent_commit": parent_commit,
                    "run_commit": start["project_git_commit"],
                    "accepted_by": "explicit accept_parent_commit_drift_from (F95)",
                }}
                compatible_parent = start
        if compatible_parent != start:
            raise ContractError(
                "parent checkpoint Git/config/Isaac context is incompatible with this run"
            )
    if mode == "nurec":
        if set(start["optional_assets"]) != {"nurec"} or len(input_records) != 1:
            raise ContractError("NuRec training requires one parent and sealed NuRec provenance")
        if input_records[0].get("task_id") not in {
            "Isaac-BinPick-UR5e-Vision-Distill-v0",
            "Isaac-BinPick-UR5e-Vision-NuRec-v0",
        } or input_records[0].get("mode") not in {"distillation", "nurec"}:
            raise ContractError("NuRec parent must be a sealed distillation/NuRec student")
        adoption = input_records[0].get("adoption")
        if not isinstance(adoption, dict) or set(adoption) != {
            "schema_version", "role", "path", "sha256", "checkpoint_name",
            "checkpoint_sha256", "training_receipt_sha256",
        }:
            raise ContractError("NuRec requires one verified Student adoption binding")
        if (
            adoption.get("schema_version") != 1
            or adoption.get("role") != "student"
            or adoption.get("training_receipt_sha256") != input_records[0]["receipt_sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", str(adoption.get("sha256", ""))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(adoption.get("checkpoint_sha256", ""))) is None
            or re.fullmatch(r"model_[0-9]+\.pt", str(adoption.get("checkpoint_name", ""))) is None
            or not isinstance(adoption.get("path"), str)
            or PurePosixPath(adoption["path"]).is_absolute()
            or ".." in PurePosixPath(adoption["path"]).parts
        ):
            raise ContractError("NuRec Student adoption binding is malformed or inconsistent")
        if input_records[0].get("checkpoint_name") not in (None, adoption["checkpoint_name"]) \
                or input_records[0].get("checkpoint_sha256") not in (
                    None, adoption["checkpoint_sha256"]
                ):
            raise ContractError("NuRec parent and Student adoption select different checkpoints")
    elif mode == "distillation":
        if len(input_records) != 1 or input_records[0].get("task_id") != (
            "Isaac-BinPick-UR5e-Teacher-v0"
        ):
            raise ContractError("distillation requires one sealed Teacher parent")
        adoption = input_records[0].get("adoption")
        if not isinstance(adoption, dict) or set(adoption) != {
            "schema_version", "role", "path", "sha256", "checkpoint_name",
            "checkpoint_sha256", "training_receipt_sha256",
        }:
            raise ContractError("distillation requires one verified Teacher adoption binding")
        if (
            adoption.get("schema_version") != 1
            or adoption.get("role") != "teacher"
            or adoption.get("training_receipt_sha256") != input_records[0]["receipt_sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", str(adoption.get("sha256", ""))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(adoption.get("checkpoint_sha256", ""))) is None
            or re.fullmatch(r"model_[0-9]+\.pt", str(adoption.get("checkpoint_name", ""))) is None
            or not isinstance(adoption.get("path"), str)
            or PurePosixPath(adoption["path"]).is_absolute()
            or ".." in PurePosixPath(adoption["path"]).parts
        ):
            raise ContractError("distillation Teacher adoption binding is malformed or inconsistent")
        if input_records[0].get("checkpoint_name") not in (None, adoption["checkpoint_name"]) \
                or input_records[0].get("checkpoint_sha256") not in (
                    None, adoption["checkpoint_sha256"]
                ):
            raise ContractError("distillation parent and Teacher adoption select different checkpoints")
    elif mode == "resume":
        if len(input_records) != 1 or input_records[0].get("task_id") != task_id:
            raise ContractError("resume requires one sealed parent from the same task")
    elif mode in {"scratch", "ppo"}:
        if input_records:
            raise ContractError(f"{mode} training may not consume a parent checkpoint")
    else:
        raise ContractError(f"unsupported training receipt mode: {mode}")
    if mode != "nurec" and start["optional_assets"]:
        raise ContractError("non-NuRec training may not seal optional NuRec assets")
    payload = {
        "schema_version": TRAINING_RECEIPT_SCHEMA_VERSION,
        "receipt_type": TRAINING_RECEIPT_TYPE,
        "experiment_name": experiment_name,
        "run_root": run_root,
        "task_id": task_id,
        "mode": mode,
        "started_at": started_at,
        "completed_at": completed_at,
        "arguments": _sanitize_arguments(arguments),
        "source": start,
        "runtime_authority": frozen_runtime_authority,
        "inputs": input_records,
        "artifacts": {
            "parameters": {
                relative: hashlib.sha256(
                    _read_regular_file_once(
                        path, label=f"completed training parameter {relative}"
                    )
                ).hexdigest()
                for relative, path in params.items()
            },
            "checkpoints": {
                path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
                for path in checkpoints
            },
            "tensorboard_events": tensorboard_events,
        },
    }
    destination = run / TRAINING_RECEIPT_NAME
    if destination.exists() or path_contains_link_or_reparse(destination.absolute()):
        raise ContractError("training receipt already exists or is unsafe")
    fd, temporary_name = tempfile.mkstemp(prefix=".run-manifest-", suffix=".json", dir=run)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_regular_file_no_replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
    return destination


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
