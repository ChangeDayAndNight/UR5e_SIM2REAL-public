"""Immutable provenance contract for raw post-training evaluation artifacts."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping

from deployment.contracts import ContractError, path_contains_link_or_reparse
from deployment.release import publish_regular_file_no_replace
from common.runtime_authority import (
    RuntimeAuthorityError,
    runtime_authority_binding,
    validate_runtime_authority_binding,
)
from scripts.training_provenance import (
    capture_training_context,
    verify_training_checkpoint_receipt,
)


EVALUATION_RECEIPT_NAME = "evaluation_receipt.json"
EVALUATION_RECEIPT_SCHEMA_VERSION = 2
EVALUATION_RECEIPT_TYPE = "ur5e_checkpoint_evaluation"
OFFICIAL_EVALUATION_ARTIFACTS = ("episodes.csv", "eval_stats.json", "summary.txt")
FORBIDDEN_ENV_PREFIXES = ("BINPICK_",)
FORBIDDEN_ENV_KEYS = ("BP_CURRICULUM_STEP_OFFSET",)

_TASK_CONTRACTS = {
    "Isaac-BinPick-UR5e-Teacher-Eval-v0": {
        "experiment": "ur5e_binpick_teacher",
        "training_task": "Isaac-BinPick-UR5e-Teacher-v0",
        "training_mode": "scratch",
        "result_phase": "Phase1",
    },
    "Isaac-BinPick-UR5e-Vision-Eval-v0": {
        "experiment": "ur5e_binpick_vision",
        "training_task": "Isaac-BinPick-UR5e-Vision-Distill-v0",
        "training_mode": "distillation",
        "result_phase": "Phase2",
    },
}


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def forbidden_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {
        str(key): str(value)
        for key, value in sorted(source.items())
        if key in FORBIDDEN_ENV_KEYS or any(key.startswith(prefix) for prefix in FORBIDDEN_ENV_PREFIXES)
    }


def sanitized_evaluation_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Copy the ambient environment while deleting every policy/distribution override."""
    result = {
        str(key): str(value)
        for key, value in environ.items()
        if key not in FORBIDDEN_ENV_KEYS
        and not any(key.startswith(prefix) for prefix in FORBIDDEN_ENV_PREFIXES)
    }
    result.pop("PYTHONPATH", None)
    result["PYTHONNOUSERSITE"] = "1"
    result["OMNI_KIT_ACCEPT_EULA"] = "YES"
    return result


def _canonical_checkpoint(path: Path | str, root: Path) -> tuple[Path, str]:
    raw = Path(path).expanduser().absolute()
    if path_contains_link_or_reparse(raw) or not raw.is_file():
        raise ContractError("evaluation checkpoint is missing/non-regular or linked")
    checkpoint = raw.resolve()
    try:
        relative = checkpoint.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("evaluation checkpoint escapes the project root") from exc
    if re.fullmatch(
        r"logs/rsl_rl/(ur5e_binpick_teacher|ur5e_binpick_vision)/[^/]+/model_[0-9]+\.pt",
        relative,
    ) is None:
        raise ContractError("evaluation checkpoint is not an original canonical run checkpoint")
    return checkpoint, relative


def _canonical_evaluation_directory(
    evaluation_dir: Path | str, *, project_root: Path, task_id: str,
) -> tuple[Path, str]:
    raw = Path(evaluation_dir).expanduser().absolute()
    if path_contains_link_or_reparse(raw) or not raw.is_dir():
        raise ContractError("evaluation directory is missing or traverses a link/reparse point")
    directory = raw.resolve()
    contract = _TASK_CONTRACTS.get(task_id)
    if contract is None:
        raise ContractError("evaluation task is outside the adoption contract")
    expected_parent = (
        project_root / str(contract["result_phase"]) / "Result"
    ).resolve()
    if directory.parent != expected_parent or not directory.name.startswith("eval_"):
        raise ContractError("evaluation directory is outside its canonical result store")
    return directory, directory.relative_to(project_root).as_posix()


def _regular_file_snapshot(path: Path, *, project_root: Path) -> tuple[dict, bytes]:
    """Read one immutable-in-call byte snapshot and return its exact binding."""
    raw = path.absolute()
    if path_contains_link_or_reparse(raw) or not raw.is_file():
        raise ContractError(f"evaluation official artifact is missing/unsafe: {path.name}")
    resolved = raw.resolve()
    try:
        relative = resolved.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ContractError("evaluation official artifact escapes the project root") from exc
    try:
        with resolved.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError(
                    f"evaluation official artifact is not regular: {path.name}"
                )
            data = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise ContractError(
            f"cannot read evaluation official artifact {path.name}: {exc}"
        ) from exc
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(data) != before.st_size:
        raise ContractError(f"evaluation official artifact changed while reading: {path.name}")
    if not data:
        raise ContractError(f"evaluation official artifact is empty: {path.name}")
    return {
        "path": relative,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }, data


def snapshot_evaluation_artifacts(
    evaluation_dir: Path | str, *, project_root: Path | str, task_id: str,
) -> tuple[dict[str, dict[str, int | str]], dict[str, bytes]]:
    """Snapshot the three official outputs from one canonical evaluation directory."""
    root_raw = Path(project_root).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw) or not root_raw.is_dir():
        raise ContractError("evaluation project root is missing or unsafe")
    root = root_raw.resolve()
    directory, _ = _canonical_evaluation_directory(
        evaluation_dir, project_root=root, task_id=task_id
    )
    inventory: dict[str, dict[str, int | str]] = {}
    snapshots: dict[str, bytes] = {}
    for name in OFFICIAL_EVALUATION_ARTIFACTS:
        record, data = _regular_file_snapshot(directory / name, project_root=root)
        inventory[name] = record
        snapshots[name] = data
    return inventory, snapshots


def evaluation_artifact_inventory(
    evaluation_dir: Path | str, *, project_root: Path | str, task_id: str,
) -> dict[str, dict[str, int | str]]:
    inventory, _ = snapshot_evaluation_artifacts(
        evaluation_dir, project_root=project_root, task_id=task_id
    )
    return inventory


def _validate_protocol(protocol: object) -> dict:
    keys = {
        "task_id", "seed", "episodes", "num_envs", "max_steps",
        "goal_radius_m", "place_tolerance_m", "headless", "video", "disable_fabric",
    }
    if not isinstance(protocol, dict) or set(protocol) != keys:
        raise ContractError("evaluation protocol record is malformed")
    for key in ("seed", "episodes", "num_envs", "max_steps"):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError(f"evaluation protocol {key} must be an integer")
    if protocol["seed"] < 0 or protocol["episodes"] <= 0 \
            or protocol["num_envs"] <= 0 or protocol["max_steps"] <= 0:
        raise ContractError("evaluation protocol contains a non-positive count")
    for key in ("goal_radius_m", "place_tolerance_m"):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not 0.0 < float(value) < 1.0:
            raise ContractError(f"evaluation protocol {key} is invalid")
    if any(protocol[key] is not expected for key, expected in {
        "headless": True, "video": False, "disable_fabric": False,
    }.items()):
        raise ContractError("adoption evaluation must be headless without video/fabric override")
    task_id = protocol["task_id"]
    if task_id not in _TASK_CONTRACTS:
        raise ContractError("evaluation task is outside the adoption contract")
    return {
        **protocol,
        "goal_radius_m": float(protocol["goal_radius_m"]),
        "place_tolerance_m": float(protocol["place_tolerance_m"]),
    }


def begin_evaluation(
    *, project_root: Path | str, checkpoint_path: Path | str, protocol: dict,
) -> dict:
    """Capture and verify the exact source immediately before simulation starts."""
    root_raw = Path(project_root).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw):
        raise ContractError("evaluation project root traverses a symlink/reparse point")
    root = root_raw.resolve()
    checkpoint, checkpoint_relative = _canonical_checkpoint(checkpoint_path, root)
    normalized_protocol = _validate_protocol(protocol)
    forbidden = forbidden_environment()
    if forbidden:
        raise ContractError(
            "evaluation environment contains forbidden policy overrides: "
            + ", ".join(sorted(forbidden))
        )
    task_contract = _TASK_CONTRACTS[normalized_protocol["task_id"]]
    training = verify_training_checkpoint_receipt(
        checkpoint,
        project_root=root,
        expected_experiment=task_contract["experiment"],
        require_tensorboard_events=True,
    )
    if training.get("task_id") != task_contract["training_task"] \
            or training.get("mode") != task_contract["training_mode"]:
        raise ContractError("evaluation task and training receipt role/mode differ")
    context = capture_training_context(root)
    if context != training["source"]:
        raise ContractError("evaluation context differs from the source training receipt")
    try:
        runtime_authority = validate_runtime_authority_binding(
            runtime_authority_binding(),
            expected_git_commit=context["project_git_commit"],
            expected_configs=context["canonical_configs"],
            expected_purpose="evaluation",
        )
    except RuntimeAuthorityError as exc:
        raise ContractError(f"evaluation runtime authority is invalid: {exc}") from exc
    return {
        "root": root,
        "checkpoint": checkpoint,
        "checkpoint_relative": checkpoint_relative,
        "protocol": normalized_protocol,
        "training": training,
        "context": context,
        "runtime_authority": runtime_authority,
    }


def finalize_evaluation(session: dict, *, evaluation_dir: Path | str) -> dict:
    """Revalidate the source and create the machine-readable raw-eval receipt."""
    root = session["root"]
    checkpoint = session["checkpoint"]
    task_contract = _TASK_CONTRACTS[session["protocol"]["task_id"]]
    if forbidden_environment():
        raise ContractError("forbidden policy override appeared during evaluation")
    end_context = capture_training_context(root)
    if end_context != session["context"]:
        raise ContractError("Git/config/training-stack context changed during evaluation")
    end_training = verify_training_checkpoint_receipt(
        checkpoint,
        project_root=root,
        expected_experiment=task_contract["experiment"],
        require_tensorboard_events=True,
    )
    if end_training != session["training"]:
        raise ContractError("checkpoint/run receipt artifacts changed during evaluation")
    try:
        current_runtime_authority = validate_runtime_authority_binding(
            runtime_authority_binding(),
            expected_git_commit=session["context"]["project_git_commit"],
            expected_configs=session["context"]["canonical_configs"],
            expected_purpose="evaluation",
        )
    except RuntimeAuthorityError as exc:
        raise ContractError(f"evaluation runtime authority changed: {exc}") from exc
    if current_runtime_authority != session["runtime_authority"]:
        raise ContractError("evaluation runtime authority changed during execution")
    training = session["training"]
    context = session["context"]
    artifacts, _ = snapshot_evaluation_artifacts(
        evaluation_dir,
        project_root=root,
        task_id=session["protocol"]["task_id"],
    )
    return {
        "schema_version": EVALUATION_RECEIPT_SCHEMA_VERSION,
        "receipt_type": EVALUATION_RECEIPT_TYPE,
        "checkpoint": {
            "path": session["checkpoint_relative"],
            "sha256": training["checkpoint_sha256"],
            "training_run_root": training["run_root"],
            "training_receipt": {
                "path": f"{training['run_root']}/run_manifest.json",
                "sha256": training["receipt_sha256"],
            },
        },
        "protocol": session["protocol"],
        "evaluation_context": {
            "sha256": canonical_json_sha256(context),
            "value": context,
        },
        "runtime_authority": current_runtime_authority,
        "environment": {
            "forbidden_prefixes": list(FORBIDDEN_ENV_PREFIXES),
            "forbidden_keys": list(FORBIDDEN_ENV_KEYS),
            "observed_forbidden": {},
            "allowed_override_map": {},
        },
        "artifacts": artifacts,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def write_evaluation_receipt(path: Path | str, receipt: dict) -> Path:
    destination = Path(path).absolute()
    if path_contains_link_or_reparse(destination) or destination.exists() \
            or not destination.parent.is_dir():
        raise ContractError("evaluation receipt destination exists or is unsafe")
    payload = (json.dumps(
        receipt, indent=2, sort_keys=True, allow_nan=False
    ) + "\n").encode("utf-8")
    handle, temporary_name = tempfile.mkstemp(
        prefix=".evaluation-receipt-", suffix=".json", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        publish_regular_file_no_replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def validate_evaluation_receipt(
    value: object,
    *,
    project_root: Path | str,
    checkpoint_path: Path | str,
    training: dict,
    expected_protocol: dict,
    evaluation_dir: Path | str,
) -> dict[str, bytes]:
    """Validate lineage/output bindings and return the exact bytes that were hashed."""
    keys = {
        "schema_version", "receipt_type", "checkpoint", "protocol",
        "evaluation_context", "runtime_authority", "environment", "artifacts", "created_at",
    }
    if not isinstance(value, dict) or set(value) != keys \
            or value.get("schema_version") != EVALUATION_RECEIPT_SCHEMA_VERSION \
            or value.get("receipt_type") != EVALUATION_RECEIPT_TYPE:
        raise ContractError("evaluation provenance receipt has the wrong schema")
    root = Path(project_root).resolve()
    checkpoint, relative = _canonical_checkpoint(checkpoint_path, root)
    del checkpoint
    expected_checkpoint = {
        "path": relative,
        "sha256": training["checkpoint_sha256"],
        "training_run_root": training["run_root"],
        "training_receipt": {
            "path": f"{training['run_root']}/run_manifest.json",
            "sha256": training["receipt_sha256"],
        },
    }
    if value.get("checkpoint") != expected_checkpoint:
        raise ContractError("evaluation receipt checkpoint/run binding mismatch")
    protocol = _validate_protocol(value.get("protocol"))
    if protocol != _validate_protocol(expected_protocol):
        raise ContractError("evaluation receipt protocol differs from the fixed adoption protocol")
    context = training.get("source")
    expected_context = {
        "sha256": canonical_json_sha256(context),
        "value": context,
    }
    if value.get("evaluation_context") != expected_context:
        raise ContractError("evaluation receipt context/hash differs from training")
    try:
        validate_runtime_authority_binding(
            value.get("runtime_authority"),
            expected_git_commit=context["project_git_commit"],
            expected_configs=context["canonical_configs"],
            expected_purpose="evaluation",
        )
    except RuntimeAuthorityError as exc:
        raise ContractError(f"evaluation receipt runtime authority is invalid: {exc}") from exc
    expected_environment = {
        "forbidden_prefixes": list(FORBIDDEN_ENV_PREFIXES),
        "forbidden_keys": list(FORBIDDEN_ENV_KEYS),
        "observed_forbidden": {},
        "allowed_override_map": {},
    }
    if value.get("environment") != expected_environment:
        raise ContractError("evaluation receipt contains an ambient override")
    current_artifacts, snapshots = snapshot_evaluation_artifacts(
        evaluation_dir,
        project_root=root,
        task_id=protocol["task_id"],
    )
    if value.get("artifacts") != current_artifacts:
        raise ContractError("evaluation official artifact path/size/hash drift")
    created_at = value.get("created_at")
    if not isinstance(created_at, str):
        raise ContractError("evaluation receipt creation time is missing")
    try:
        parsed = dt.datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise ContractError("evaluation receipt creation time is invalid") from exc
    if parsed.tzinfo is None:
        raise ContractError("evaluation receipt creation time lacks a timezone")
    return snapshots
