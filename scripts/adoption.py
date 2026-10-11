"""Fail-closed checkpoint adoption receipts shared by training and export.

An ``adoption.json`` is deliberately separate from the immutable schema-2
``run_manifest.json``.  The training receipt proves where a checkpoint came
from; the adoption receipt proves that the exact checkpoint passed the fixed
multi-seed evaluation protocol.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Iterable

from deployment.contracts import ContractError, path_contains_link_or_reparse, sha256_file
from scripts.training_provenance import capture_training_context, verify_training_checkpoint_receipt
from scripts.adoption_selection import (
    checkpoint_iteration,
    load_scalar_series,
    select_student_candidates,
    select_teacher_candidates,
)
from scripts.evaluation_provenance import (
    EVALUATION_RECEIPT_NAME,
    validate_evaluation_receipt,
)


ADOPTION_SCHEMA_VERSION = 1
ADOPTION_RECORD_TYPE = "ur5e_checkpoint_adoption"
ADOPTION_NAME = "adoption.json"
EVALUATION_SEEDS = (42, 43, 44, 45)
EPISODES_PER_SEED = 512
MIN_SUCCESSES_PER_SEED = 487
POOLED_EPISODES = EPISODES_PER_SEED * len(EVALUATION_SEEDS)
MIN_WILSON_LOWER = 0.95
# ★F90(2026-09-19) 역할별 채택 게이트★ — Teacher(증류의 정답원)는 종전 95% 기준을 유지하고,
#   Student(배포 대상)는 사용자 배포 기준(2026-09-19 지시: "Task 성공률 적어도 75% 이상")을 쓴다:
#   각 seed ≥ 384/512(75%) ∧ pooled 2048 Wilson 95% 하한 ≥ 0.75. 게이트는 protocol에 봉인되어
#   receipt 검증(verify_adopted_checkpoint)이 같은 값을 요구한다.
ROLE_GATES: dict[str, tuple[int, float]] = {
    "teacher": (MIN_SUCCESSES_PER_SEED, MIN_WILSON_LOWER),
    "student": (384, 0.75),
}
METRICS = ("grasp", "carry", "placed")

_ROLE_CONTRACTS = {
    "teacher": {
        "experiment": "ur5e_binpick_teacher",
        "task_id": "Isaac-BinPick-UR5e-Teacher-v0",
        "modes": {"scratch"},
        "result_phase": "Phase1",
        "result_dir": "top5_evaluation",
        "selection_metric": "Episode_Termination/placed_success trailing-200 median",
        "eval_task": "Isaac-BinPick-UR5e-Teacher-Eval-v0",
        "eval_num_envs": 512,
        "selection_tag": "Episode_Termination/placed_success",
        "require_zero_target_infeasible": False,
    },
    "student": {
        "experiment": "ur5e_binpick_vision",
        "task_id": "Isaac-BinPick-UR5e-Vision-Distill-v0",
        "modes": {"distillation"},
        "result_phase": "Phase2",
        "result_dir": "posttrain_evaluation",
        "selection_metric": "Loss/behavior trailing-200 median",
        "eval_task": "Isaac-BinPick-UR5e-Vision-Eval-v0",
        "eval_num_envs": 128,
        "selection_tag": "Loss/behavior",
        "require_zero_target_infeasible": True,
    },
}


def _read_regular_file_once(path: Path, *, label: str) -> tuple[bytes, str]:
    """Read and hash one exact regular-file payload through one open handle."""
    absolute = path.absolute()
    if path_contains_link_or_reparse(absolute) or not absolute.is_file():
        raise ContractError(f"{label} is missing, non-regular, or traverses a reparse point")
    try:
        with absolute.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(f"{label} is not a regular file")
            payload = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ContractError(f"cannot read {label}: {exc}") from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ContractError(f"{label} changed while it was being read")
    return payload, hashlib.sha256(payload).hexdigest()


def _read_adoption_receipt_once(path: Path) -> tuple[object, str]:
    """Parse and hash the exact adoption bytes from one regular-file handle."""

    payload, digest = _read_regular_file_once(path, label="adoption receipt")
    try:
        receipt = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode adoption receipt: {exc}") from exc
    return receipt, digest


def _protocol(role: str) -> dict:
    contract = _role_contract(role)
    return {
        "candidate_count": 5,
        "seeds": list(EVALUATION_SEEDS),
        "episodes_per_seed": EPISODES_PER_SEED,
        "min_successes_per_seed": ROLE_GATES[role][0],
        "pooled_episodes": POOLED_EPISODES,
        "wilson_confidence": 0.95,
        "min_wilson_lower": ROLE_GATES[role][1],
        "metrics": list(METRICS),
        "require_finite": True,
        "require_stage_invariants": True,
        "require_zero_target_infeasible": contract["require_zero_target_infeasible"],
        "winner_order": [
            "min_seed_placed", "min_seed_carry", "min_seed_grasp",
            "pooled_placed", "pooled_carry", "pooled_grasp",
            ("higher_Episode_Termination/placed_success_trailing_200_median"
             if role == "teacher"
             else "lower_Loss/behavior_trailing_200_median"),
            "higher_iteration",
        ],
    }


def _role_contract(role: str) -> dict:
    try:
        return _ROLE_CONTRACTS[role]
    except KeyError as exc:
        raise ContractError(f"unsupported adoption role: {role!r}") from exc


def _evaluation_protocol(role: str, seed: int) -> dict:
    contract = _role_contract(role)
    return {
        "task_id": contract["eval_task"],
        "seed": int(seed),
        "episodes": EPISODES_PER_SEED,
        "num_envs": contract["eval_num_envs"],
        "max_steps": 6000,
        "goal_radius_m": 0.08,
        "place_tolerance_m": 0.03,
        "headless": True,
        "video": False,
        "disable_fabric": False,
    }


def wilson_lower(k: int, n: int, z: float = 1.959963984540054) -> float:
    if isinstance(k, bool) or isinstance(n, bool) or not (0 <= k <= n) or n <= 0:
        raise ContractError("Wilson input must satisfy 0 <= k <= n and n > 0")
    p = k / n
    denominator = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return max(0.0, (center - half) / denominator)


def canonical_adoption_path(project_root: Path | str, checkpoint: Path | str, role: str) -> Path:
    root_raw = Path(project_root).expanduser().absolute()
    checkpoint_raw = Path(checkpoint).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw) or path_contains_link_or_reparse(checkpoint_raw):
        raise ContractError("adoption root/checkpoint traverses a symlink or reparse point")
    root = root_raw.resolve()
    checkpoint_path = checkpoint_raw.resolve()
    contract = _role_contract(role)
    expected_parent = root / "logs" / "rsl_rl" / contract["experiment"]
    if checkpoint_path.parent.parent != expected_parent.resolve():
        raise ContractError("adoption checkpoint is outside its canonical experiment store")
    if re.fullmatch(r"model_[0-9]+\.pt", checkpoint_path.name) is None:
        raise ContractError("adoption checkpoint must be an original model_<iteration>.pt")
    output_raw = (
        root_raw / contract["result_phase"] / "Result" / checkpoint_path.parent.name
        / contract["result_dir"] / ADOPTION_NAME
    )
    if path_contains_link_or_reparse(output_raw):
        raise ContractError("canonical adoption output traverses a symlink or reparse point")
    output = output_raw.resolve()
    expected_root = (root / contract["result_phase"] / "Result").resolve()
    if expected_root not in output.parents:
        raise ContractError("canonical adoption output escapes its phase result root")
    return output


def _relative_regular_file(path: Path, root: Path, label: str) -> str:
    raw = path.absolute()
    if path_contains_link_or_reparse(raw) or not raw.is_file():
        raise ContractError(f"{label} is missing, non-regular, or traverses a reparse point")
    resolved = raw.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{label} escapes the project root") from exc
    return relative.as_posix()


def adoption_file_binding(path: Path | str, *, project_root: Path | str) -> dict[str, str]:
    root_raw = Path(project_root).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw):
        raise ContractError("adoption project root traverses a symlink or reparse point")
    root = root_raw.resolve()
    adoption = Path(path).expanduser().absolute()
    relative = _relative_regular_file(adoption, root, "adoption receipt")
    _, adoption_sha256 = _read_adoption_receipt_once(adoption)
    return {"path": relative, "sha256": adoption_sha256}


def _evaluation_artifact_from_snapshots(
    eval_dir: Path | str,
    *,
    project_root: Path | str,
    checkpoint_sha256: str,
    checkpoint_name: str,
    seed: int,
    role: str,
    expected_receipt_sha256: str,
    official_snapshots: dict[str, bytes],
    receipt_bytes: bytes,
) -> dict:
    """Bind the exact official bytes that the evaluator semantics consumed."""

    root_raw = Path(project_root).expanduser().absolute()
    directory_raw = Path(eval_dir).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw) \
            or path_contains_link_or_reparse(directory_raw) \
            or not directory_raw.is_dir():
        raise ContractError("evaluation root/directory traverses a symlink or reparse point")
    root = root_raw.resolve()
    directory = directory_raw.resolve()
    try:
        relative_dir = directory.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("evaluation directory escapes the project root") from exc
    contract = _role_contract(role)
    expected_prefix = PurePosixPath(contract["result_phase"], "Result")
    relative_parts = PurePosixPath(relative_dir)
    if relative_parts.parent != expected_prefix or not relative_parts.name.startswith("eval_"):
        raise ContractError("evaluation directory is outside the canonical phase result store")
    official_names = {"episodes.csv", "eval_stats.json", "summary.txt"}
    if set(official_snapshots) != official_names:
        raise ContractError("evaluation official byte snapshot set is incomplete")
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise ContractError("evaluation checkpoint digest is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", str(expected_receipt_sha256)) is None:
        raise ContractError("evaluation receipt anchor digest is invalid")
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    if receipt_sha256 != expected_receipt_sha256:
        raise ContractError("evaluation receipt differs from the parent-process SHA anchor")
    files = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in official_snapshots.items()
    }
    files[EVALUATION_RECEIPT_NAME] = receipt_sha256
    return {
        "directory": relative_dir,
        "files": files,
        "evaluation_receipt_sha256": expected_receipt_sha256,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": int(seed),
        "episodes": EPISODES_PER_SEED,
        "task_id": contract["eval_task"],
    }


def evaluation_artifact_binding(
    eval_dir: Path | str,
    *,
    project_root: Path | str,
    checkpoint_sha256: str,
    checkpoint_name: str,
    seed: int,
    role: str,
    expected_receipt_sha256: str,
) -> dict:
    """Hash-bind one completed evaluation to its exact protocol inputs."""
    directory_raw = Path(eval_dir).expanduser().absolute()
    official_snapshots = {}
    for name in ("episodes.csv", "eval_stats.json", "summary.txt"):
        payload, _ = _read_regular_file_once(
            directory_raw / name, label=f"evaluation {name}"
        )
        official_snapshots[name] = payload
    receipt_bytes, _ = _read_regular_file_once(
        directory_raw / EVALUATION_RECEIPT_NAME,
        label=f"evaluation {EVALUATION_RECEIPT_NAME}",
    )
    return _evaluation_artifact_from_snapshots(
        directory_raw,
        project_root=project_root,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_name=checkpoint_name,
        seed=seed,
        role=role,
        expected_receipt_sha256=expected_receipt_sha256,
        official_snapshots=official_snapshots,
        receipt_bytes=receipt_bytes,
    )


def _exact_csv_integer(raw: object, label: str) -> int:
    if raw is None or not isinstance(raw, str) or not raw.strip():
        raise ContractError(f"evaluation CSV official field is blank: {label}")
    try:
        value = Decimal(raw.strip())
    except InvalidOperation as exc:
        raise ContractError(f"evaluation CSV official field is not numeric: {label}") from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ContractError(f"evaluation CSV official field is not an exact integer: {label}")
    return int(value)


def parse_evaluation(
    eval_dir: Path | str,
    *,
    project_root: Path | str,
    checkpoint_name: str,
    checkpoint_sha256: str,
    checkpoint_path: Path | str,
    training: dict,
    seed: int,
    role: str,
    expected_receipt_sha256: str,
) -> dict:
    """Validate one exact 512-episode evaluation and return its seed row."""
    if seed not in EVALUATION_SEEDS:
        raise ContractError("evaluation seed is outside the fixed 42..45 protocol")
    root_raw = Path(project_root).expanduser().absolute()
    directory_raw = Path(eval_dir).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw) or path_contains_link_or_reparse(directory_raw):
        raise ContractError("evaluation root/directory traverses a symlink or reparse point")
    root = root_raw.resolve()
    directory = directory_raw.resolve()
    checkpoint = Path(checkpoint_path).expanduser().absolute()
    if path_contains_link_or_reparse(checkpoint) or not checkpoint.is_file():
        raise ContractError("evaluation checkpoint path is missing/non-regular or linked")
    checkpoint = checkpoint.resolve()
    if checkpoint.name != checkpoint_name or sha256_file(checkpoint) != checkpoint_sha256 \
            or training.get("checkpoint_name") != checkpoint_name \
            or training.get("checkpoint_sha256") != checkpoint_sha256:
        raise ContractError("evaluation checkpoint arguments differ from verified training lineage")
    receipt_path = directory / EVALUATION_RECEIPT_NAME
    try:
        receipt_bytes, receipt_sha256 = _read_regular_file_once(
            receipt_path, label="evaluation provenance receipt"
        )
        if receipt_sha256 != expected_receipt_sha256:
            raise ContractError("evaluation receipt differs from its in-memory SHA anchor")
        provenance = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode evaluation provenance receipt: {exc}") from exc
    snapshots = validate_evaluation_receipt(
        provenance,
        project_root=root,
        checkpoint_path=checkpoint,
        training=training,
        expected_protocol=_evaluation_protocol(role, seed),
        evaluation_dir=directory,
    )
    artifact = _evaluation_artifact_from_snapshots(
        directory,
        project_root=root,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_name=checkpoint_name,
        seed=seed,
        role=role,
        expected_receipt_sha256=expected_receipt_sha256,
        official_snapshots=snapshots,
        receipt_bytes=receipt_bytes,
    )
    try:
        summary = snapshots["summary.txt"].decode("utf-8")
        csv_text = snapshots["episodes.csv"].decode("utf-8")
        stats_text = snapshots["eval_stats.json"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("evaluation official artifact is not valid UTF-8") from exc
    contract = _role_contract(role)
    if checkpoint_name not in summary \
            or f"task           : {contract['eval_task']}" not in summary \
            or f"seed           : {seed}" not in summary:
        raise ContractError("evaluation summary does not bind checkpoint/task/seed")
    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    fieldnames = reader.fieldnames
    rows = list(reader)
    if len(rows) != EPISODES_PER_SEED:
        raise ContractError(f"evaluation must contain exactly {EPISODES_PER_SEED} episodes")
    required = {
        "episode", "grasp_success", "carry_latch_success", "placed_success",
        "target_infeasible", "ep_steps",
    }
    if not rows or not isinstance(fieldnames, list) or len(fieldnames) != len(set(fieldnames)) \
            or not required.issubset(fieldnames):
        raise ContractError("evaluation CSV lacks official gate columns")
    counts = {name: 0 for name in METRICS}
    infeasible = 0
    for index, row in enumerate(rows):
        # All non-empty evaluator fields are numeric by contract.  Reject NaN,
        # infinity, and silently changed text columns before deriving metrics.
        for key, raw in row.items():
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ContractError(f"evaluation CSV contains non-numeric {key}") from exc
            if not math.isfinite(value):
                raise ContractError(f"evaluation CSV contains non-finite {key}")
        official = {
            key: _exact_csv_integer(row.get(key), key)
            for key in required
        }
        if official["episode"] != index or official["ep_steps"] <= 0:
            raise ContractError("evaluation episode sequence/length invariant failed")
        values = {
            "grasp": official["grasp_success"],
            "carry": official["carry_latch_success"],
            "placed": official["placed_success"],
        }
        if any(value not in (0, 1) for value in values.values()):
            raise ContractError("evaluation success columns must be binary")
        if not values["placed"] <= values["carry"] <= values["grasp"]:
            raise ContractError("evaluation grasp/carry/placed stage invariant failed")
        for metric, value in values.items():
            counts[metric] += value
        target_infeasible = official["target_infeasible"]
        if target_infeasible not in (0, 1):
            raise ContractError("target_infeasible must be binary")
        infeasible += target_infeasible
    if contract["require_zero_target_infeasible"] and infeasible:
        raise ContractError("student evaluation includes target-infeasible episodes")
    try:
        stats = json.loads(stats_text)
    except json.JSONDecodeError as exc:
        raise ContractError(f"cannot decode evaluation stats: {exc}") from exc
    runs = stats.get("runs") if isinstance(stats, dict) else None
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict) \
            or runs[0].get("n_episodes") != EPISODES_PER_SEED:
        raise ContractError("evaluation stats must contain one exact 512-episode run")
    metrics = runs[0].get("metrics")
    if not isinstance(metrics, dict):
        raise ContractError("evaluation stats metrics are missing")
    for name in METRICS:
        value = metrics.get(name)
        if not isinstance(value, dict) or value.get("k") != counts[name] \
                or value.get("n") != EPISODES_PER_SEED:
            raise ContractError(f"evaluation stats/CSV {name} count mismatch")
        rate = value.get("rate")
        ci95 = value.get("ci95")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) \
                or not math.isfinite(float(rate)) \
                or not math.isclose(float(rate), counts[name] / EPISODES_PER_SEED, abs_tol=1e-12) \
                or not isinstance(ci95, list) or len(ci95) != 2 \
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(float(item)) for item in ci95):
            raise ContractError(f"evaluation stats {name} rate/CI is invalid")
    return {
        "seed": seed,
        "episodes": EPISODES_PER_SEED,
        "metrics": counts,
        "target_infeasible_count": infeasible,
        "finite": True,
        "stage_invariants": True,
        "artifact": artifact,
    }


def _metric_summary(evaluations: Iterable[dict], role: str = "teacher") -> tuple[dict, bool]:
    _role_contract(role)  # 알 수 없는 역할은 여기서 거부된다
    min_seed, min_wilson = ROLE_GATES[role]
    rows = list(evaluations)
    if len(rows) != len(EVALUATION_SEEDS):
        raise ContractError("candidate must contain exactly four seed evaluations")
    if sorted(int(row.get("seed", -1)) for row in rows) != list(EVALUATION_SEEDS):
        raise ContractError("candidate evaluation seed set mismatch")
    pooled: dict[str, dict] = {}
    all_seed_counts_pass = True
    for metric in METRICS:
        counts: list[int] = []
        for row in rows:
            if row.get("episodes") != EPISODES_PER_SEED:
                raise ContractError("evaluation episode count is not exactly 512")
            metrics = row.get("metrics")
            if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
                raise ContractError("evaluation metric set mismatch")
            value = metrics[metric]
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= EPISODES_PER_SEED:
                raise ContractError(f"invalid {metric} success count")
            counts.append(value)
        total = sum(counts)
        lower = wilson_lower(total, POOLED_EPISODES)
        pooled[metric] = {
            "successes": total,
            "episodes": POOLED_EPISODES,
            "rate": total / POOLED_EPISODES,
            "wilson_lower_95": lower,
            "seed_counts": counts,
        }
        all_seed_counts_pass &= all(value >= min_seed for value in counts)
    passed = all_seed_counts_pass and all(
        pooled[metric]["wilson_lower_95"] >= min_wilson for metric in METRICS
    )
    return pooled, passed


def finalize_candidate(candidate: dict, role: str = "teacher") -> dict:
    """Recompute pooled statistics and the role-gated pass bit from seed evidence (F90)."""
    required = {"checkpoint_name", "checkpoint_sha256", "iteration", "selection_value", "evaluations"}
    if not isinstance(candidate, dict) or set(candidate) != required:
        raise ContractError("candidate adoption record has the wrong schema")
    if re.fullmatch(r"model_([0-9]+)\.pt", str(candidate["checkpoint_name"])) is None:
        raise ContractError("candidate checkpoint name is invalid")
    match = re.fullmatch(r"model_([0-9]+)\.pt", candidate["checkpoint_name"])
    assert match is not None
    if int(match.group(1)) != candidate["iteration"]:
        raise ContractError("candidate iteration/name mismatch")
    if re.fullmatch(r"[0-9a-f]{64}", str(candidate["checkpoint_sha256"])) is None:
        raise ContractError("candidate checkpoint digest is invalid")
    selection_value = candidate["selection_value"]
    if isinstance(selection_value, bool) or not isinstance(selection_value, (int, float)) \
            or not math.isfinite(float(selection_value)):
        raise ContractError("candidate selection value is not finite")
    pooled, passed = _metric_summary(candidate["evaluations"], role)
    return {**candidate, "selection_value": float(selection_value), "pooled": pooled, "passed": passed}


def _winner_sort_key(role: str, candidate: dict) -> tuple:
    pooled = candidate["pooled"]
    metric_order = ("placed", "carry", "grasp")
    minimum_seed = tuple(min(int(value) for value in pooled[name]["seed_counts"])
                         for name in metric_order)
    pooled_counts = tuple(int(pooled[name]["successes"]) for name in metric_order)
    selection = float(candidate["selection_value"])
    if role == "student":
        selection = -selection
    return (*minimum_seed, *pooled_counts, selection, int(candidate["iteration"]))


def make_adoption_receipt(
    *,
    role: str,
    run_root: str,
    training_receipt_sha256: str,
    candidates: Iterable[dict],
    created_at: str,
) -> dict:
    contract = _role_contract(role)
    candidate_rows = [finalize_candidate(row, role) for row in candidates]
    if len(candidate_rows) != 5:
        raise ContractError("adoption requires exactly five checkpoint candidates")
    names = [row["checkpoint_name"] for row in candidate_rows]
    if len(set(names)) != 5:
        raise ContractError("adoption candidates must be unique")
    passing = [row for row in candidate_rows if row["passed"]]
    winner = max(passing, key=lambda row: _winner_sort_key(role, row)) if passing else None
    selection = None if winner is None else {
        "checkpoint_name": winner["checkpoint_name"],
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "iteration": winner["iteration"],
    }
    return {
        "schema_version": ADOPTION_SCHEMA_VERSION,
        "record_type": ADOPTION_RECORD_TYPE,
        "role": role,
        "status": "adopted" if selection else "rejected",
        "source": {
            "experiment_name": contract["experiment"],
            "run_root": run_root,
            "training_receipt_sha256": training_receipt_sha256,
        },
        "selection_metric": contract["selection_metric"],
        "protocol": _protocol(role),
        "selection": selection,
        "candidates": candidate_rows,
        "created_at": created_at,
    }


def _validate_evaluation_binding(
    evaluation: dict, *, checkpoint_name: str, checkpoint_sha256: str, role: str,
    root: Path, checkpoint_path: Path, training: dict,
) -> None:
    expected_keys = {
        "seed", "episodes", "metrics", "target_infeasible_count", "finite",
        "stage_invariants", "artifact",
    }
    if not isinstance(evaluation, dict) or set(evaluation) != expected_keys:
        raise ContractError("seed evaluation has the wrong schema")
    artifact = evaluation["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {
        "directory", "files", "evaluation_receipt_sha256", "checkpoint_name",
        "checkpoint_sha256", "seed", "episodes", "task_id",
    }:
        raise ContractError("evaluation artifact binding has the wrong schema")
    if artifact["checkpoint_name"] != checkpoint_name \
            or artifact["checkpoint_sha256"] != checkpoint_sha256 \
            or artifact["seed"] != evaluation["seed"] \
            or artifact["episodes"] != EPISODES_PER_SEED:
        raise ContractError("evaluation artifact/checkpoint protocol binding mismatch")
    directory = root / PurePosixPath(str(artifact["directory"]))
    expected_artifact = evaluation_artifact_binding(
        directory,
        project_root=root,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_name=checkpoint_name,
        seed=int(evaluation["seed"]),
        role=role,
        expected_receipt_sha256=str(artifact["evaluation_receipt_sha256"]),
    )
    if artifact != expected_artifact:
        raise ContractError("evaluation artifacts changed after adoption")
    reparsed = parse_evaluation(
        directory,
        project_root=root,
        checkpoint_name=checkpoint_name,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_path=checkpoint_path,
        training=training,
        seed=int(evaluation["seed"]),
        role=role,
        expected_receipt_sha256=str(artifact["evaluation_receipt_sha256"]),
    )
    if evaluation != reparsed:
        raise ContractError("adoption seed metrics differ from their evaluation artifacts")
    infeasible = evaluation["target_infeasible_count"]
    if isinstance(infeasible, bool) or not isinstance(infeasible, int) or not 0 <= infeasible <= EPISODES_PER_SEED:
        raise ContractError("target infeasible count is invalid")
    if evaluation["finite"] is not True or evaluation["stage_invariants"] is not True:
        raise ContractError("evaluation finite/invariant gate did not pass")
    if _role_contract(role)["require_zero_target_infeasible"] and infeasible != 0:
        raise ContractError("student evaluation contains infeasible targets")


# ★F95★ 마지막으로 허용된 commit 드리프트(진단·로그용). binding 형상을 바꾸지 않기 위해 분리한다.
last_accepted_context_drift: dict = {}


def verify_adopted_checkpoint(
    checkpoint_path: Path | str,
    *,
    project_root: Path | str,
    role: str,
    accept_commit_drift_from: str | None = None,
) -> dict:
    """Return an immutable adoption binding, or reject the checkpoint.

    ★F95(2026-09-22) 명시적 commit-drift 허용★ — `accept_commit_drift_from`에 채택 run의 ★정확한 40자
    commit★을 주면, 현재 컨텍스트와 채택 컨텍스트가 ★project_git_commit만★ 다르고 canonical config
    해시·training_stack이 같을 때에 한해 통과시킨다. 반환 binding에 `context_drift`를 기록해 호출측
    receipt에 남긴다. 사용자가 "이번만" 허용한 예외(README F95)이며 기본은 여전히 완전 동일 컨텍스트다.
    """
    root_raw = Path(project_root).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw):
        raise ContractError("adoption project root traverses a symlink or reparse point")
    root = root_raw.resolve()
    checkpoint = Path(checkpoint_path).expanduser().absolute()
    if path_contains_link_or_reparse(checkpoint) or not checkpoint.is_file():
        raise ContractError("adopted checkpoint is missing/non-regular or traverses a reparse point")
    checkpoint = checkpoint.resolve()
    contract = _role_contract(role)
    training = verify_training_checkpoint_receipt(
        checkpoint,
        project_root=root,
        expected_experiment=contract["experiment"],
        require_tensorboard_events=True,
        require_all_checkpoints=True,
    )
    if training.get("task_id") != contract["task_id"] or training.get("mode") not in contract["modes"]:
        raise ContractError("checkpoint role/task/mode is not eligible for adoption")
    current_context = capture_training_context(root)
    context_drift = None
    if current_context != training["source"]:
        adopted_commit = training["source"].get("project_git_commit")
        if accept_commit_drift_from is None:
            raise ContractError("current Git/config/training-stack context differs from adopted run")
        if not isinstance(accept_commit_drift_from, str) \
                or re.fullmatch(r"[0-9a-f]{40}", accept_commit_drift_from) is None \
                or accept_commit_drift_from != adopted_commit:
            raise ContractError(
                "accept_commit_drift_from must name the adopted run's exact 40-hex commit "
                f"({adopted_commit}); got {accept_commit_drift_from!r}"
            )
        if {**current_context, "project_git_commit": adopted_commit} != training["source"]:
            raise ContractError(
                "context drift is limited to the Git commit — canonical configs/training stack differ"
            )
        context_drift = {
            "adopted_commit": adopted_commit,
            "current_commit": current_context["project_git_commit"],
            "accepted_by": "explicit accept_commit_drift_from (F95)",
        }
        # ★F95★ binding ★형상★은 고정이다(7키) — 배포/봉인 검증이 key set을 엄격히 본다.
        #   드리프트 사실은 학습 receipt의 입력 레코드(F95-b finalize_training_receipt)에 기록된다.
        last_accepted_context_drift.clear()
        last_accepted_context_drift.update(context_drift)
        print(f"[adoption][warn] ★F95★ teacher commit drift accepted explicitly: "
              f"adopted {adopted_commit[:12]} → current {current_context['project_git_commit'][:12]} "
              "(canonical configs/training stack identical)")
    adoption_path = canonical_adoption_path(root, checkpoint, role)
    if path_contains_link_or_reparse(adoption_path.absolute()) or not adoption_path.is_file():
        raise ContractError("checkpoint has no canonical adoption.json")
    receipt, adoption_sha256 = _read_adoption_receipt_once(adoption_path)
    expected_keys = {
        "schema_version", "record_type", "role", "status", "source", "selection_metric",
        "protocol", "selection", "candidates", "created_at",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys \
            or receipt.get("schema_version") != ADOPTION_SCHEMA_VERSION \
            or receipt.get("record_type") != ADOPTION_RECORD_TYPE \
            or receipt.get("role") != role \
            or receipt.get("status") != "adopted" \
            or receipt.get("protocol") != _protocol(role) \
            or receipt.get("selection_metric") != contract["selection_metric"]:
        raise ContractError("adoption receipt is rejected, malformed, or uses another protocol")
    source = receipt.get("source")
    if source != {
        "experiment_name": contract["experiment"],
        "run_root": training["run_root"],
        "training_receipt_sha256": training["receipt_sha256"],
    }:
        raise ContractError("adoption source differs from the immutable training receipt")
    candidates = receipt.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ContractError("adoption candidate matrix is incomplete")
    candidate_names = [item.get("checkpoint_name") for item in candidates if isinstance(item, dict)]
    if len(candidate_names) != 5 or len(set(candidate_names)) != 5:
        raise ContractError("adoption requires five unique checkpoint candidates")
    checkpoint_inventory = training.get("checkpoints")
    if not isinstance(checkpoint_inventory, dict) or not checkpoint_inventory:
        raise ContractError("adoption requires a receipt-bound full checkpoint inventory")
    recorded_selection_evidence = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, dict) or set(raw_candidate) != {
            "checkpoint_name", "checkpoint_sha256", "iteration", "selection_value",
            "evaluations", "pooled", "passed",
        }:
            raise ContractError("adoption candidate has the wrong finalized schema")
        candidate_name = raw_candidate["checkpoint_name"]
        if not isinstance(candidate_name, str) \
                or re.fullmatch(r"model_[0-9]+\.pt", candidate_name) is None \
                or candidate_name not in checkpoint_inventory:
            raise ContractError("adoption candidate is outside the receipt checkpoint inventory")
        iteration = raw_candidate["iteration"]
        selection_value = raw_candidate["selection_value"]
        if isinstance(iteration, bool) or not isinstance(iteration, int) \
                or iteration != checkpoint_iteration(Path(candidate_name)) \
                or isinstance(selection_value, bool) \
                or not isinstance(selection_value, (int, float)) \
                or not math.isfinite(float(selection_value)):
            raise ContractError("adoption candidate selection evidence is malformed")
        recorded_selection_evidence.append({
            "checkpoint_name": candidate_name,
            "iteration": iteration,
            "selection_value": float(selection_value),
        })
    all_checkpoints = sorted(
        (checkpoint.parent / name for name in checkpoint_inventory),
        key=checkpoint_iteration,
    )
    steps, values = load_scalar_series(
        checkpoint.parent, contract["selection_tag"], training["tensorboard_events"]
    )
    expected_candidates = (
        select_teacher_candidates(all_checkpoints, steps, values)
        if role == "teacher"
        else select_student_candidates(all_checkpoints, steps, values)
    )
    if len(expected_candidates) != 5:
        raise ContractError("TensorBoard evidence does not produce exactly five candidates")
    expected_selection_evidence = [{
        "checkpoint_name": row["checkpoint"].name,
        "iteration": row["iteration"],
        "selection_value": float(row["selection_value"]),
    } for row in expected_candidates]
    if recorded_selection_evidence != expected_selection_evidence:
        raise ContractError("adoption candidate set/value differs from TensorBoard selection evidence")
    if role == "student":
        final_iteration = max(checkpoint_iteration(path) for path in all_checkpoints)
        if expected_candidates[0]["iteration"] != final_iteration:
            raise ContractError("Student adoption candidate set does not include final checkpoint first")
    normalized = []
    for raw_candidate in candidates:
        original = checkpoint.parent / raw_candidate["checkpoint_name"]
        verified = verify_training_checkpoint_receipt(
            original,
            project_root=root,
            expected_experiment=contract["experiment"],
            require_tensorboard_events=True,
        )
        if verified["run_root"] != training["run_root"] \
                or verified["receipt_sha256"] != training["receipt_sha256"] \
                or verified["source"] != training["source"] \
                or verified["tensorboard_events"] != training["tensorboard_events"]:
            raise ContractError("adoption candidates do not share one immutable training run")
        if verified["checkpoint_sha256"] != raw_candidate["checkpoint_sha256"]:
            raise ContractError("adoption candidate differs from its original run checkpoint")
        for evaluation in raw_candidate.get("evaluations", []):
            _validate_evaluation_binding(
                evaluation,
                checkpoint_name=raw_candidate["checkpoint_name"],
                checkpoint_sha256=raw_candidate["checkpoint_sha256"],
                role=role,
                root=root,
                checkpoint_path=original,
                training=verified,
            )
        source_candidate = {key: raw_candidate[key] for key in (
            "checkpoint_name", "checkpoint_sha256", "iteration", "selection_value", "evaluations"
        )}
        recomputed = finalize_candidate(source_candidate, role)
        if raw_candidate != recomputed:
            raise ContractError("adoption pooled metrics/pass result was not deterministically derived")
        normalized.append(recomputed)
    passing = [row for row in normalized if row["passed"]]
    expected_winner = max(passing, key=lambda row: _winner_sort_key(role, row)) if passing else None
    expected_selection = None if expected_winner is None else {
        "checkpoint_name": expected_winner["checkpoint_name"],
        "checkpoint_sha256": expected_winner["checkpoint_sha256"],
        "iteration": expected_winner["iteration"],
    }
    if receipt.get("selection") != expected_selection or expected_selection is None:
        raise ContractError("adoption selection is missing or not the deterministic passing winner")
    if expected_selection["checkpoint_name"] != checkpoint.name \
            or expected_selection["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ContractError("requested checkpoint is not the adopted checkpoint")
    final_training = verify_training_checkpoint_receipt(
        checkpoint,
        project_root=root,
        expected_experiment=contract["experiment"],
        require_tensorboard_events=True,
        require_all_checkpoints=True,
    )
    if final_training != training:
        raise ContractError("training receipt/event inventory changed during adoption verification")
    if capture_training_context(root) != current_context:   # ★F95★ 재확인 기준은 검증 시작 시점의 현재 컨텍스트
        raise ContractError("current context changed during adoption verification")
    return {
        "schema_version": 1,
        "role": role,
        "path": adoption_path.relative_to(root).as_posix(),
        "sha256": adoption_sha256,
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": expected_selection["checkpoint_sha256"],
        "training_receipt_sha256": training["receipt_sha256"],
    }


def digest_bytes(data: bytes) -> str:
    """Small test/helper API kept independent from filesystem path semantics."""
    return hashlib.sha256(data).hexdigest()
