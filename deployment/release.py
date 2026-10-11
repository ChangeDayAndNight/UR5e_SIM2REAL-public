"""Student-only deployment release manifest and artifact verification."""
from __future__ import annotations

import json
import math
import hashlib
import io
import importlib.metadata
import importlib.util
import os
import platform
import re
import secrets
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable

import numpy as np
import torch

from common.gripper_command import (
    GRIPPER_COMMAND_TRANSITION_S,
    GRIPPER_COMMAND_TRANSITION_STEPS,
    gripper_command_spec,
)
from common.config_io import (
    load_yaml_snapshot,
    validate_phase3_target_geometry,
)
from common.file_snapshot import FileSnapshotError, read_regular_file_snapshot
from common.runtime_authority import (
    RuntimeAuthorityError,
    validate_runtime_authority_execution_receipt,
)
from common.types import ARM_CLAMP_COMPARISON_ATOL, ARM_COMMAND_CHANNELS
from common.training_stack_contract import (
    CUDA_RUNTIME,
    EXPECTED_TRAINING_DISTRIBUTIONS,
    ISAAC_LAB_COMMIT,
    ISAAC_LAB_DISTRIBUTIONS,
    ISAAC_LAB_RELEASE,
    ISAAC_LAB_TAG,
    ISAAC_SIM_METADATA,
    MIN_TRAINING_VRAM_MIB,
    MIN_WINDOWS_BUILD,
    TRAINING_OS,
    TRAINING_PYTHON_IMPLEMENTATION,
    TRAINING_PYTHON_VERSION,
)
from deployment.contracts import (
    ContractError,
    is_link_or_reparse,
    path_contains_link_or_reparse,
    sha256_file,
    validate_release_calibrations,
)

RELEASE_SCHEMA_VERSION = 5
RELEASE_MANIFEST_NAME = "release_manifest.json"
CANDIDATE_SCHEMA_VERSION = 2
CANDIDATE_MANIFEST_NAME = "candidate_manifest.json"
CANDIDATE_RECORD_TYPE = "ur5e_export_candidate"
CAPTURE_SCHEMA_VERSION = 6
CAPTURE_MANIFEST_NAME = "capture_manifest.json"
CAPTURE_RECORD_TYPE = "ur5e_parity_capture"
CAPTURE_OWNER_NAME = ".capture_owner.json"
CAPTURE_OWNER_TYPE = "ur5e_parity_capture_owner"
PHASE3_RUNTIME_AUTHORITY_NAME = "runtime_authority.json"
PHASE3_RUNTIME_REQUIRED_CONFIGS = (
    "configs/common.yaml",
    "configs/dr.yaml",
    "configs/safety.yaml",
    "configs/sim.yaml",
)
MIN_PARITY_SAMPLES = 10_000
PHASE3_CAPTURE_SEED = 45
MIN_PHASE3_EPISODES = 100
MAX_PHASE3_EPISODE_STEPS = 200   # ★F93★ Teacher v2 학습 에피소드 10 s × 20 Hz (구 5 s = 100)
MIN_LATENCY_SAMPLES = 10_000
MAX_PARITY_ERROR = 1.0e-5
PARITY_AUDIT_ROWS = 512
MIN_PARITY_UNIQUE_RATIO = 0.90
MIN_BLOCK_VARYING_FRACTION = 0.01
MAX_ARM_CLAMP_FRACTION = 0.01
TRAINING_STACK = {
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "torchaudio": "2.7.0+cu128",
    "rsl-rl-lib": "3.1.2",
}
REQUIRED_ENVIRONMENT = {
    "python", "platform", "system", "machine", "torch", "numpy",
    "pyyaml", "torchvision", "torchaudio", "rsl-rl-lib",
}
INVALID_ENVIRONMENT_VALUES = {"", "not-installed", "unknown", "none", "null"}
PROJECT_DISTRIBUTION_VERSION = "0.1.0"
ALLOWED_STUDENT_POLICY_CLASSES = {
    "VisionStudentTeacher",
    "VisionStudentTeacherResNet",
}
ALLOWED_DISTILLATION_CLASSES = {"Distillation", "DistillationAux"}
REQUIRED_RELEASE_CONFIGS = {
    "configs/common.yaml",
    "configs/dr.yaml",
    "configs/safety.yaml",
    "configs/sim.yaml",
    "configs/calib/hand_eye.yaml",
    "configs/calib/nurec_landmarks.yaml",
    "configs/calib/scene_layout.yaml",
    "configs/calib/scene_registration.yaml",
    "configs/calib/tcp_offset.yaml",
    "configs/calib/tool_chain.yaml",
}


class _VerifiedCandidateManifest(dict):
    """A schema-compatible manifest carrying its privately verified byte hash.

    ``verify_candidate`` historically returned a plain ``dict`` and callers
    rely on that API.  A dict subclass preserves that interface while keeping
    the digest of the exact bytes that were parsed out of the public JSON
    schema.  ``candidate_receipt_binding`` uses the private anchor to reject a
    path that was swapped (even to semantically identical JSON) between the
    two operations.
    """

    def __init__(self, value: dict, *, manifest_sha256: str) -> None:
        super().__init__(value)
        self._verified_manifest_sha256 = manifest_sha256


class _VerifiedReleaseManifest(dict):
    """Release mapping carrying the hash of the exact parsed manifest bytes."""

    def __init__(self, value: dict, *, manifest_sha256: str) -> None:
        super().__init__(value)
        self._verified_manifest_sha256 = manifest_sha256


def _read_authority_regular_file_once(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[bytes, str]:
    """Read and hash one exact regular-file payload through one open handle."""

    source = Path(path).absolute()
    if path_contains_link_or_reparse(source):
        raise ContractError(f"{label} contains a symlink/reparse path")
    try:
        with source.open("rb") as stream:
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
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ContractError(f"{label} hash mismatch")
    return payload, digest


def _read_json_regular_file_once(
    path: Path | str,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[object, str]:
    """Parse and hash one exact JSON payload through one open handle."""

    payload, digest = _read_authority_regular_file_once(
        path, label=label, expected_sha256=expected_sha256
    )
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode {label}: {exc}") from exc
    return decoded, digest


def _phase3_runtime_authority_record(
    path: Path,
    *,
    candidate_manifest: dict,
    project_root: Path,
    expected_record: object | None = None,
) -> dict:
    """Consume and validate the exact Phase 3 authority sidecar bytes once."""
    payload, digest = _read_authority_regular_file_once(
        path,
        label="Phase 3 runtime authority receipt",
    )
    try:
        receipt = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode Phase 3 runtime authority receipt: {exc}") from exc
    source = candidate_manifest.get("source")
    if not isinstance(source, dict):
        raise ContractError("candidate source is missing for Phase 3 runtime authority")
    training_run = _validate_training_run_provenance(source.get("training_run"))
    training_source = training_run["source"]
    try:
        frozen = validate_runtime_authority_execution_receipt(
            receipt,
            expected_stage="phase3",
            expected_purpose="phase3-rehearsal",
            expected_git_commit=training_source["project_git_commit"],
            expected_configs=training_source["canonical_configs"],
            required_configs=PHASE3_RUNTIME_REQUIRED_CONFIGS,
            project_root=project_root,
        )
    except RuntimeAuthorityError as exc:
        raise ContractError(f"Phase 3 runtime authority receipt is invalid: {exc}") from exc
    record = {
        "path": PHASE3_RUNTIME_AUTHORITY_NAME,
        "sha256": digest,
        "bytes": len(payload),
        "receipt": frozen,
    }
    if expected_record is not None and expected_record != record:
        raise ContractError(
            "Phase 3 runtime authority sidecar differs from the capture manifest"
        )
    return record


def _canonical_config_path(config_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ContractError(f"unsafe canonical config path: {relative!r}")
    raw = config_root.joinpath(*pure.parts).absolute()
    if path_contains_link_or_reparse(raw):
        raise ContractError(f"canonical config contains a symlink/reparse path: {relative}")
    path = raw.resolve()
    if config_root not in path.parents or not path.is_file():
        raise ContractError(f"canonical config is missing or escapes root: {relative}")
    return path


def _snapshot_release_config_hashes(
    config_root: Path | str,
    config_paths: Iterable[Path | str],
    *,
    expected_hashes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Hash each canonical config from one opened regular-file snapshot."""
    root = Path(config_root).resolve()
    if expected_hashes is not None and (
        not isinstance(expected_hashes, dict)
        or set(expected_hashes) != REQUIRED_RELEASE_CONFIGS
    ):
        raise ContractError("expected canonical config hash set is incomplete")
    result: dict[str, str] = {}
    for supplied in config_paths:
        raw = Path(supplied).absolute()
        if path_contains_link_or_reparse(raw):
            raise ContractError("canonical config path contains a symlink/reparse point")
        try:
            relative = raw.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ContractError("canonical config path escapes config root") from exc
        path = _canonical_config_path(root, relative)
        expected = expected_hashes.get(relative) if expected_hashes is not None else None
        try:
            snapshot = read_regular_file_snapshot(
                path,
                label=f"canonical config {relative}",
                expected_sha256=expected,
            )
        except FileSnapshotError as exc:
            raise ContractError(str(exc)) from exc
        if relative in result:
            raise ContractError(f"duplicate canonical config path: {relative}")
        result[relative] = snapshot.sha256
    return result


def _load_canonical_yaml_configs(
    config_root: Path | str,
    expected_hashes: object,
    relatives: Iterable[str],
    *,
    label: str,
) -> dict[str, dict]:
    """Parse hash-anchored canonical YAML from the exact hashed snapshots."""
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != REQUIRED_RELEASE_CONFIGS:
        raise ContractError(f"{label} canonical config hash set is incomplete")
    root = Path(config_root).resolve()
    decoded: dict[str, dict] = {}
    for relative in relatives:
        expected = expected_hashes.get(relative)
        if re.fullmatch(r"[0-9a-f]{64}", str(expected or "")) is None:
            raise ContractError(f"{label} canonical config hash is invalid: {relative}")
        path = _canonical_config_path(root, relative)
        try:
            value, _ = load_yaml_snapshot(
                path,
                expected_sha256=expected,
                label=f"{label} canonical config {relative}",
            )
        except (FileSnapshotError, OSError, ValueError) as exc:
            raise ContractError(f"cannot load {label} canonical config {relative}: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"{label} canonical config must be a mapping: {relative}")
        decoded[relative] = value
    return decoded


def _validated_gripper_command_obs_spec(obs_spec: object) -> dict:
    """Reject every legacy physical-finger observation contract."""
    if not isinstance(obs_spec, dict):
        raise ContractError("obs_spec must be an object")
    expected = gripper_command_spec(
        transition_s=GRIPPER_COMMAND_TRANSITION_S,
        transition_steps=GRIPPER_COMMAND_TRANSITION_STEPS,
    )
    if obs_spec.get("gripper_command_state") != expected:
        raise ContractError("obs_spec gripper_command_state semantics are missing or stale")
    layout = obs_spec.get("layout")
    if (
        not isinstance(layout, list)
        or {"name": "gripper_command_state", "size": 1} not in layout
        or any(isinstance(item, dict) and item.get("name") == "gripper_pos" for item in layout)
    ):
        raise ContractError("obs_spec layout is not the 1D gripper_command_state contract")
    return expected


def rollout_safety_contract() -> dict:
    """Return the immutable clamp contract bound by every export candidate."""

    return {
        "arm_channel_count": ARM_COMMAND_CHANNELS,
        "minimum_denominator": MIN_PARITY_SAMPLES,
        "maximum_clamp_fraction_exclusive": MAX_ARM_CLAMP_FRACTION,
        "raw_command_field": "raw_arm_command",
        "safety_command_field": "safety_arm_command",
        "issued_command_field": "arm_command",
        "semantics": "same-delay-mapped-vs-guarded-vs-actual-applied-union",
        "receipt_sequence_field": "arm_command_receipt_sequence",
        "same_lineage_required": True,
        "comparison_atol": ARM_CLAMP_COMPARISON_ATOL,
        "successful_command_only": True,
    }


def _canonical_mapping_sha256(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rollout_safety_from_counts(counts: Iterable[int], denominator: int) -> dict:
    count_list = [int(value) for value in counts]
    evidence = {
        "arm_channel_count": ARM_COMMAND_CHANNELS,
        "denominator": int(denominator),
        "clamp_counts": count_list,
        "clamp_fractions": [int(value) / int(denominator) for value in count_list]
        if int(denominator) > 0 else [0.0] * ARM_COMMAND_CHANNELS,
        "maximum_fraction_exclusive": MAX_ARM_CLAMP_FRACTION,
        "all_channels_below_limit": bool(
            int(denominator) > 0
            and all(int(value) * 100 < int(denominator) for value in count_list)
        ),
    }
    return _validate_rollout_safety(evidence)


def _validate_rollout_safety(value: object) -> dict:
    expected_keys = {
        "arm_channel_count", "denominator", "clamp_counts", "clamp_fractions",
        "maximum_fraction_exclusive", "all_channels_below_limit",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ContractError("rollout safety evidence has an incomplete/unexpected field set")
    denominator = value.get("denominator")
    counts = value.get("clamp_counts")
    fractions = value.get("clamp_fractions")
    if (
        value.get("arm_channel_count") != ARM_COMMAND_CHANNELS
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < MIN_PARITY_SAMPLES
        or not isinstance(counts, list)
        or len(counts) != ARM_COMMAND_CHANNELS
        or not isinstance(fractions, list)
        or len(fractions) != ARM_COMMAND_CHANNELS
        or value.get("maximum_fraction_exclusive") != MAX_ARM_CLAMP_FRACTION
    ):
        raise ContractError("rollout safety evidence does not meet the fixed clamp contract")
    normalized_counts: list[int] = []
    normalized_fractions: list[float] = []
    for channel, (count, fraction) in enumerate(zip(counts, fractions, strict=True)):
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= denominator:
            raise ContractError(f"rollout safety channel {channel} clamp count is invalid")
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) \
                or not math.isfinite(float(fraction)):
            raise ContractError(f"rollout safety channel {channel} clamp fraction is invalid")
        expected_fraction = count / denominator
        if float(fraction) != expected_fraction:
            raise ContractError(f"rollout safety channel {channel} fraction/count mismatch")
        # Integer arithmetic makes the exclusive boundary unambiguous: an
        # exact 100/10,000 (1%) is rejected, never rounded below the gate.
        if count * 100 >= denominator:
            raise ContractError(
                f"rollout safety channel {channel} clamp fraction must be strictly below 1%"
            )
        normalized_counts.append(count)
        normalized_fractions.append(expected_fraction)
    expected = {
        "arm_channel_count": ARM_COMMAND_CHANNELS,
        "denominator": denominator,
        "clamp_counts": normalized_counts,
        "clamp_fractions": normalized_fractions,
        "maximum_fraction_exclusive": MAX_ARM_CLAMP_FRACTION,
        "all_channels_below_limit": True,
    }
    if value != expected:
        raise ContractError("rollout safety evidence is not canonically encoded")
    return value


def _validate_environment(environment: object) -> dict[str, str]:
    if not isinstance(environment, dict) or not REQUIRED_ENVIRONMENT.issubset(environment):
        raise ContractError("release environment/platform provenance is incomplete")
    for name in sorted(REQUIRED_ENVIRONMENT):
        value = environment.get(name)
        if not isinstance(value, str) or value.strip().lower() in INVALID_ENVIRONMENT_VALUES:
            raise ContractError(
                f"release environment {name} must be an exact non-placeholder value; got {value!r}"
            )
    return environment


def package_versions() -> dict[str, str]:
    import importlib.metadata

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    for package in ("PyYAML", "torchvision", "torchaudio", "rsl-rl-lib"):
        try:
            versions[package.lower()] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package.lower()] = "not-installed"
    return versions


def _metadata_version(name: str) -> str:
    try:
        value = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ContractError(f"training provenance distribution is missing: {name}") from exc
    if not value or value.strip().lower() in INVALID_ENVIRONMENT_VALUES:
        raise ContractError(f"training provenance distribution has invalid version: {name}={value!r}")
    return value


def _module_origin(name: str) -> Path:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ContractError(f"cannot resolve training import origin for {name}: {exc}") from exc
    candidates: list[str] = []
    if spec is not None and spec.origin and spec.origin not in {"built-in", "frozen"}:
        candidates.append(str(spec.origin))
    if spec is not None and spec.submodule_search_locations:
        candidates.extend(str(item) for item in spec.submodule_search_locations)
    for candidate in candidates:
        path = Path(candidate).resolve()
        if path.exists():
            return path
    raise ContractError(f"training import origin is missing/unreadable: {name}")


def _relative_module_origin(name: str, project_root: Path, expected_root: Path) -> str:
    origin = _module_origin(name)
    try:
        origin.relative_to(expected_root.resolve())
        return origin.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(
            f"{name} import origin is outside the pinned checkout: "
            f"actual={origin}, expected_under={expected_root.resolve()}"
        ) from exc


def _site_packages_module_origin(name: str) -> str:
    origin = _module_origin(name)
    roots = {
        Path(value).resolve()
        for key in ("purelib", "platlib")
        if (value := sysconfig.get_path(key))
    }
    for root in roots:
        try:
            relative = origin.relative_to(root)
        except ValueError:
            continue
        return (PurePosixPath("site-packages") / PurePosixPath(relative.as_posix())).as_posix()
    raise ContractError(
        f"{name} import origin is outside Python purelib/platlib: {origin}"
    )


def _git_checkout_value(checkout: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=checkout, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(
            f"cannot establish Isaac Lab Git provenance ({' '.join(args)}): {exc}"
        ) from exc


def _nvidia_gpu_inventory() -> list[dict[str, str | float]]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"cannot record NVIDIA driver provenance: {exc}") from exc
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not rows:
        raise ContractError(
            f"cannot record NVIDIA GPU provenance: exit={proc.returncode} {proc.stdout.strip()}"
        )
    devices: list[dict[str, str | float]] = []
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1]:
            raise ContractError(f"invalid nvidia-smi GPU provenance row: {row!r}")
        try:
            memory_mib = float(fields[2])
        except ValueError as exc:
            raise ContractError(f"invalid nvidia-smi GPU memory: {fields[2]!r}") from exc
        if memory_mib <= 0:
            raise ContractError(f"invalid nvidia-smi GPU memory: {memory_mib}")
        devices.append({
            "driver_version": fields[0],
            "name": fields[1],
            "memory_total_mib": memory_mib,
        })
    return devices


def _training_host_provenance() -> dict[str, str | int]:
    if platform.system() != TRAINING_OS or not hasattr(sys, "getwindowsversion"):
        raise ContractError("training provenance requires a native Windows 11 host")
    return {
        "os": platform.system(),
        "windows_build": int(sys.getwindowsversion().build),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def _torch_gpu_provenance() -> dict[str, str | int | float]:
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise ContractError("training provenance requires an available CUDA device")
    props = torch.cuda.get_device_properties(0)
    return {
        "index": 0,
        "device_count": int(torch.cuda.device_count()),
        "name": str(props.name),
        "memory_total_mib": float(props.total_memory / (1024**2)),
    }


def _validate_relative_origin(value: object, prefix: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"training provenance {label} import origin is missing")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not (normalized == prefix or normalized.startswith(prefix + "/"))
    ):
        raise ContractError(
            f"training provenance {label} import origin must be under {prefix}: {value!r}"
        )
    return normalized


def _validate_training_stack_provenance(value: object) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ContractError("release training_stack provenance is missing or has wrong schema")
    sim = value.get("isaac_sim")
    lab = value.get("isaac_lab")
    project = value.get("project")
    ml = value.get("ml_stack")
    gpu = value.get("gpu")
    host = value.get("host")
    distributions = value.get("distributions")
    if not all(isinstance(item, dict) for item in (sim, lab, project, ml, gpu, host, distributions)):
        raise ContractError("release training_stack provenance is incomplete")
    if host != {
        "os": TRAINING_OS,
        "windows_build": host.get("windows_build"),
        "machine": "AMD64",
        "python": TRAINING_PYTHON_VERSION,
        "python_implementation": TRAINING_PYTHON_IMPLEMENTATION,
    }:
        raise ContractError("release training host must be exact Windows/CPython 3.11 x64")
    windows_build = host.get("windows_build")
    if isinstance(windows_build, bool) or not isinstance(windows_build, int) \
            or windows_build < MIN_WINDOWS_BUILD:
        raise ContractError("release training host is below the Windows 11 build contract")
    if set(distributions) != set(EXPECTED_TRAINING_DISTRIBUTIONS):
        raise ContractError("release training distribution set is incomplete or contains extras")
    for name, accepted in EXPECTED_TRAINING_DISTRIBUTIONS.items():
        actual = distributions.get(name)
        if not isinstance(actual, str) or actual not in accepted:
            raise ContractError(
                f"release training distribution {name} must be one of {accepted}; got {actual!r}"
            )
    if sim.get("release_version") != "5.1.0" or sim.get("metadata") != ISAAC_SIM_METADATA:
        raise ContractError("release Isaac Sim provenance must be exactly 5.1.0 / 5.1.0.0 metadata")
    _validate_relative_origin(
        sim.get("import_origin"), "site-packages/isaacsim", "Isaac Sim"
    )
    if (
        lab.get("release_version") != ISAAC_LAB_RELEASE
        or lab.get("git_commit") != ISAAC_LAB_COMMIT
        or lab.get("git_tag") != ISAAC_LAB_TAG
        or lab.get("checkout") != "external/IsaacLab"
        or lab.get("distributions") != ISAAC_LAB_DISTRIBUTIONS
    ):
        raise ContractError("release Isaac Lab v2.3.2 checkout provenance mismatch")
    lab_origins = lab.get("import_origins")
    if not isinstance(lab_origins, dict) or set(lab_origins) != set(ISAAC_LAB_DISTRIBUTIONS):
        raise ContractError("release Isaac Lab import-origin set is incomplete")
    for module, origin in lab_origins.items():
        _validate_relative_origin(
            origin, f"external/IsaacLab/source/{module}", f"Isaac Lab {module}"
        )
    if project.get("distribution_version") != PROJECT_DISTRIBUTION_VERSION:
        raise ContractError("release project distribution provenance mismatch")
    _validate_relative_origin(
        project.get("import_origin"), "source/ur5e_binpick", "ur5e_binpick"
    )
    if ml != {"torch": TRAINING_STACK["torch"], "rsl-rl-lib": TRAINING_STACK["rsl-rl-lib"]}:
        raise ContractError("release Torch/RSL training provenance mismatch")
    inventory = gpu.get("nvidia_smi")
    selected = gpu.get("torch_device")
    if (
        gpu.get("cuda_runtime") != CUDA_RUNTIME
        or not isinstance(inventory, list)
        or not inventory
        or not isinstance(selected, dict)
    ):
        raise ContractError("release CUDA/NVIDIA GPU provenance is incomplete")
    for device in inventory:
        if not isinstance(device, dict) or set(device) != {
            "driver_version", "name", "memory_total_mib"
        }:
            raise ContractError("release nvidia-smi GPU inventory is invalid")
        memory = device.get("memory_total_mib")
        if (
            re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", str(device.get("driver_version", ""))) is None
            or not isinstance(device.get("name"), str)
            or not device["name"].strip()
            or isinstance(memory, bool)
            or not isinstance(memory, (int, float))
            or not math.isfinite(float(memory))
            or float(memory) <= 0
        ):
            raise ContractError("release nvidia-smi GPU inventory values are invalid")
    if set(selected) != {"index", "device_count", "name", "memory_total_mib"}:
        raise ContractError("release selected Torch GPU provenance is invalid")
    selected_memory = selected.get("memory_total_mib")
    if (
        selected.get("index") != 0
        or isinstance(selected.get("device_count"), bool)
        or not isinstance(selected.get("device_count"), int)
        or selected["device_count"] <= 0
        or not isinstance(selected.get("name"), str)
        or not selected["name"].strip()
        or isinstance(selected_memory, bool)
        or not isinstance(selected_memory, (int, float))
        or not math.isfinite(float(selected_memory))
        or float(selected_memory) < MIN_TRAINING_VRAM_MIB
    ):
        raise ContractError("release selected training GPU name/VRAM is below contract")
    return value


def training_stack_provenance(project_root: Path | str) -> dict:
    """Capture exact export-host Isaac SDK/Git/import provenance, fail closed."""
    root = Path(project_root).resolve()
    checkout = root / "external" / "IsaacLab"
    version_file = checkout / "VERSION"
    if not version_file.is_file() or version_file.read_text(encoding="utf-8").strip() != ISAAC_LAB_RELEASE:
        raise ContractError("Isaac Lab external/IsaacLab/VERSION must be exactly 2.3.2")
    commit = _git_checkout_value(checkout, "rev-parse", "HEAD")
    tag = _git_checkout_value(checkout, "describe", "--tags", "--exact-match")
    dirty = _git_checkout_value(checkout, "status", "--porcelain", "--untracked-files=normal")
    if commit != ISAAC_LAB_COMMIT or tag != ISAAC_LAB_TAG or dirty:
        raise ContractError(
            "Isaac Lab checkout must be clean v2.3.2 at " + ISAAC_LAB_COMMIT
        )
    installed_distributions = {
        name: _metadata_version(name) for name in EXPECTED_TRAINING_DISTRIBUTIONS
    }
    sim_metadata = {name: installed_distributions[name] for name in ISAAC_SIM_METADATA}
    lab_distributions = {name: installed_distributions[name] for name in ISAAC_LAB_DISTRIBUTIONS}
    record = {
        "schema_version": 2,
        "host": _training_host_provenance(),
        "distributions": installed_distributions,
        "isaac_sim": {
            "release_version": "5.1.0",
            "metadata": sim_metadata,
            "import_origin": _site_packages_module_origin("isaacsim"),
        },
        "isaac_lab": {
            "release_version": ISAAC_LAB_RELEASE,
            "git_commit": commit,
            "git_tag": tag,
            "checkout": "external/IsaacLab",
            "distributions": lab_distributions,
            "import_origins": {
                module: _relative_module_origin(
                    module, root, checkout / "source" / module
                )
                for module in ISAAC_LAB_DISTRIBUTIONS
            },
        },
        "project": {
            "distribution_version": installed_distributions["ur5e_binpick"],
            "import_origin": _relative_module_origin(
                "ur5e_binpick", root, root / "source" / "ur5e_binpick"
            ),
        },
        "ml_stack": {
            "torch": installed_distributions["torch"],
            "rsl-rl-lib": installed_distributions["rsl-rl-lib"],
        },
        "gpu": {
            "cuda_runtime": str(torch.version.cuda or ""),
            "nvidia_smi": _nvidia_gpu_inventory(),
            "torch_device": _torch_gpu_provenance(),
        },
    }
    return _validate_training_stack_provenance(record)


def _validate_training_run_provenance(value: object) -> dict:
    from common.nurec_provenance import validate_sealed_nurec

    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ContractError("release training_run provenance is missing or has wrong schema")
    run_root = value.get("run_root")
    checkpoint_name = value.get("checkpoint_name")
    configs = value.get("configs")
    contract = value.get("agent_contract")
    source = value.get("source")
    if not isinstance(run_root, str):
        raise ContractError("release training run root is missing")
    run_path = PurePosixPath(run_root.replace("\\", "/"))
    if (
        run_path.is_absolute()
        or ".." in run_path.parts
        or len(run_path.parts) != 4
        or run_path.parts[:3] != ("logs", "rsl_rl", "ur5e_binpick_vision")
    ):
        raise ContractError("release checkpoint must come from one canonical vision run root")
    if not isinstance(checkpoint_name, str) or re.fullmatch(r"model_[0-9]+\.pt", checkpoint_name) is None:
        raise ContractError("release checkpoint must be a canonical model_<iteration>.pt")
    for label in ("checkpoint_sha256", "receipt_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(value.get(label, ""))) is None:
            raise ContractError(f"release training run {label} is invalid")
    task_mode = (value.get("task_id"), value.get("mode"))
    if task_mode not in {
        ("Isaac-BinPick-UR5e-Vision-Distill-v0", "distillation"),
        ("Isaac-BinPick-UR5e-Vision-NuRec-v0", "nurec"),
    }:
        raise ContractError("release checkpoint is not from an approved distillation run")
    expected_configs = {"params/agent.yaml", "params/env.yaml"}
    if not isinstance(configs, dict) or set(configs) != expected_configs:
        raise ContractError("release training run must bind agent.yaml and env.yaml")
    if any(re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None for digest in configs.values()):
        raise ContractError("release training config hash is invalid")
    if not isinstance(contract, dict) or contract != {
        "runner_class_name": "DistillationRunner",
        "policy_class_name": contract.get("policy_class_name"),
        "algorithm_class_name": contract.get("algorithm_class_name"),
        "activation": "elu",
        "student_obs_normalization": False,
        "teacher_obs_normalization": False,
        "empirical_normalization": False,
    }:
        raise ContractError("release training agent activation/normalization contract mismatch")
    if contract["policy_class_name"] not in ALLOWED_STUDENT_POLICY_CLASSES:
        raise ContractError("release training policy class is not an approved student model")
    if contract["algorithm_class_name"] not in ALLOWED_DISTILLATION_CLASSES:
        raise ContractError("release training algorithm class is not approved distillation")
    if not isinstance(source, dict) or source.get("schema_version") != 1:
        raise ContractError("release training source context is missing")
    if re.fullmatch(r"[0-9a-f]{40}", str(source.get("project_git_commit", ""))) is None:
        raise ContractError("release training source Git commit is invalid")
    _validate_training_stack_provenance(source.get("training_stack"))
    canonical_configs = source.get("canonical_configs")
    if not isinstance(canonical_configs, dict) or set(canonical_configs) != REQUIRED_RELEASE_CONFIGS:
        raise ContractError("release training canonical config set mismatch")
    if any(re.fullmatch(r"[0-9a-f]{64}", str(item)) is None for item in canonical_configs.values()):
        raise ContractError("release training canonical config hash is invalid")
    optional_assets = source.get("optional_assets")
    if task_mode[1] == "distillation":
        if optional_assets != {}:
            raise ContractError("plain distillation release must not bind optional NuRec assets")
    else:
        if not isinstance(optional_assets, dict) or set(optional_assets) != {"nurec"}:
            raise ContractError("NuRec release must bind exactly one sealed NuRec asset lineage")
        validate_sealed_nurec(optional_assets["nurec"])
    return value


def collect_training_run_provenance(
    checkpoint_path: Path | str,
    store_root: Path | str,
) -> dict:
    """Bind a checkpoint to the exact RSL-RL run snapshots used to create it."""
    from scripts.training_provenance import verify_training_checkpoint_receipt

    checkpoint_raw = Path(checkpoint_path).absolute()
    store_raw = Path(store_root).absolute()
    if path_contains_link_or_reparse(checkpoint_raw) or path_contains_link_or_reparse(store_raw):
        raise ContractError("checkpoint/run paths must not contain symlinks/reparse points")
    checkpoint = checkpoint_raw.resolve()
    store = store_raw.resolve()
    run_root = checkpoint.parent
    if run_root.parent != store or re.fullmatch(r"model_[0-9]+\.pt", checkpoint.name) is None:
        raise ContractError("checkpoint must be model_<iteration>.pt directly under a canonical run")
    if not checkpoint.is_file():
        raise ContractError(f"checkpoint is missing: {checkpoint}")
    receipt = verify_training_checkpoint_receipt(
        checkpoint,
        project_root=store.parent.parent.parent,
        expected_experiment="ur5e_binpick_vision",
    )
    params = run_root / "params"
    paths = {name: params / name for name in ("agent.yaml", "env.yaml")}
    decoded: dict[str, dict] = {}
    parameter_hashes = receipt.get("parameters")
    if not isinstance(parameter_hashes, dict) or set(parameter_hashes) != {
        "params/agent.yaml", "params/env.yaml",
    }:
        raise ContractError("training receipt parameter hash binding is missing")
    consumed_hashes: dict[str, str] = {}
    for name, path in paths.items():
        if path_contains_link_or_reparse(path) or not path.is_file():
            raise ContractError(f"training run snapshot is missing/non-regular: {path}")
        relative = f"params/{name}"
        try:
            doc, snapshot = load_yaml_snapshot(
                path,
                expected_sha256=parameter_hashes[relative],
                label=f"training run snapshot {relative}",
            )
        except (FileSnapshotError, OSError, ValueError) as exc:
            raise ContractError(f"cannot load sealed training run snapshot {relative}: {exc}") from exc
        if not isinstance(doc, dict):
            raise ContractError(f"training run snapshot must contain a mapping: {path}")
        decoded[name] = doc
        consumed_hashes[relative] = snapshot.sha256
    agent = decoded["agent.yaml"]
    policy = agent.get("policy")
    algorithm = agent.get("algorithm")
    if not isinstance(policy, dict) or not isinstance(algorithm, dict):
        raise ContractError("agent.yaml must contain policy and algorithm mappings")
    student_norm = policy.get("student_obs_normalization")
    teacher_norm = policy.get("teacher_obs_normalization")
    if "empirical_normalization" in agent and agent["empirical_normalization"] is not False:
        raise ContractError("agent.yaml empirical_normalization must be false")
    record = {
        "schema_version": 2,
        "run_root": run_root.relative_to(store.parent.parent.parent).as_posix(),
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "task_id": receipt["task_id"],
        "mode": receipt["mode"],
        "source": receipt["source"],
        "configs": consumed_hashes,
        "agent_contract": {
            "runner_class_name": agent.get("class_name"),
            "policy_class_name": policy.get("class_name"),
            "algorithm_class_name": algorithm.get("class_name"),
            "activation": policy.get("activation"),
            "student_obs_normalization": student_norm,
            "teacher_obs_normalization": teacher_norm,
            "empirical_normalization": not (
                student_norm is False and teacher_norm is False
            ),
        },
    }
    return _validate_training_run_provenance(record)


def _validated_parity_layout(layout: object, expected_dim: int) -> list[dict[str, int | str]]:
    if layout is None:
        return [{"name": "observations", "size": expected_dim}]
    if not isinstance(layout, list) or not layout:
        raise ContractError("parity observation layout must be a non-empty list")
    normalized: list[dict[str, int | str]] = []
    names: set[str] = set()
    for item in layout:
        if not isinstance(item, dict):
            raise ContractError("parity observation layout entries must be mappings")
        name = item.get("name")
        size = item.get("size")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise ContractError("parity observation layout has an invalid/duplicate block")
        names.add(name)
        normalized.append({"name": name, "size": size})
    if sum(int(item["size"]) for item in normalized) != expected_dim:
        raise ContractError("parity observation layout size differs from observation_dim")
    return normalized


def _parity_representativeness(
    array: np.ndarray,
    *,
    layout: object = None,
) -> dict:
    """Reject synthetic/repeated fixtures while keeping the audit memory bounded."""
    normalized_layout = _validated_parity_layout(layout, int(array.shape[1]))
    sample_count = min(PARITY_AUDIT_ROWS, int(array.shape[0]))
    sample_indices = np.linspace(
        0, array.shape[0] - 1, num=sample_count, dtype=np.int64
    )
    # Advanced indexing intentionally materializes only the fixed-size audit sample
    # (about 98 MiB for the current 50,206-D contract), never another full dataset.
    sampled = np.ascontiguousarray(array[sample_indices], dtype=np.float32)
    unique_rows = len({
        hashlib.sha256(row.tobytes()).digest() for row in sampled
    })
    required_unique = int(math.ceil(sample_count * MIN_PARITY_UNIQUE_RATIO))
    if unique_rows < required_unique:
        raise ContractError(
            "parity observations lack representative row diversity: "
            f"unique={unique_rows}/{sample_count}, required={required_unique}"
        )

    block_stats: dict[str, dict[str, int | float]] = {}
    offset = 0
    for item in normalized_layout:
        name = str(item["name"])
        size = int(item["size"])
        block = sampled[:, offset : offset + size]
        offset += size
        varying = int(np.count_nonzero(np.ptp(block, axis=0) > 1.0e-6))
        required_varying = max(1, int(math.ceil(size * MIN_BLOCK_VARYING_FRACTION)))
        if varying < required_varying:
            raise ContractError(
                f"parity observation block {name!r} is not representative: "
                f"varying_features={varying}, required={required_varying}"
            )
        block_min = float(block.min())
        block_max = float(block.max())
        if name in {"depth", "rgb"} and (block_min < -1.0e-6 or block_max > 1.0 + 1.0e-6):
            raise ContractError(
                f"parity observation block {name!r} is outside normalized [0,1] bounds"
            )
        block_stats[name] = {
            "size": size,
            "varying_features": varying,
            "required_varying_features": required_varying,
            "min": block_min,
            "max": block_max,
        }
    return {
        "sampled_rows": sample_count,
        "unique_sampled_rows": unique_rows,
        "minimum_unique_ratio": MIN_PARITY_UNIQUE_RATIO,
        "blocks": block_stats,
    }


def load_parity_observations(
    path: Path | str,
    expected_dim: int,
    *,
    layout: object = None,
) -> torch.Tensor:
    src_raw = Path(path).expanduser().absolute()
    if path_contains_link_or_reparse(src_raw):
        raise ContractError(f"parity observations path contains a symlink/reparse point: {src_raw}")
    src = src_raw.resolve()
    if not src.is_file():
        raise ContractError(f"parity observations not found: {src}")
    mmap_owner: np.memmap | None = None
    try:
        if src.suffix == ".npy":
            loaded = np.load(src, allow_pickle=False, mmap_mode="c")
            if isinstance(loaded, np.memmap):
                mmap_owner = loaded
            array = loaded
        elif src.suffix == ".npz":
            with np.load(src, allow_pickle=False) as data:
                key = "obs" if "obs" in data.files else (data.files[0] if len(data.files) == 1 else "")
                if not key:
                    raise ContractError("parity npz must contain an 'obs' array")
                array = data[key]
        else:
            raise ContractError("parity observations must be .npy or .npz")
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != expected_dim:
            raise ContractError(
                f"parity observations must have exact shape [N,{expected_dim}]; got {array.shape}"
            )
        if array.shape[0] < MIN_PARITY_SAMPLES:
            raise ContractError(
                f"parity dataset has {array.shape[0]} samples; at least {MIN_PARITY_SAMPLES} required"
            )
        if not np.isfinite(array).all():
            raise ContractError("parity observations contain NaN/Inf")
        _parity_representativeness(array, layout=layout)
        # ``mmap_mode='c'`` is writable copy-on-write, so torch can view a multi-GB
        # current RGB-D dataset without materializing a second full host allocation.
        tensor = torch.from_numpy(array)
        if mmap_owner is not None and np.shares_memory(array, mmap_owner):
            # Keep and expose the owner so Windows callers can close the mapped
            # file deterministically after parity/latency checks have finished.
            tensor._parity_mmap_owner = mmap_owner
            mmap_owner = None
        return tensor
    finally:
        if mmap_owner is not None:
            memory_map = getattr(mmap_owner, "_mmap", None)
            if memory_map is not None:
                memory_map.close()


def close_parity_observations(observations: torch.Tensor | None) -> None:
    """Release a parity loader's mmap after its final consumer is done.

    Keeping this explicit avoids Windows file-handle leaks without ever
    materializing the current multi-gigabyte RGB-D parity dataset.
    """

    if observations is None:
        return
    owner = getattr(observations, "_parity_mmap_owner", None)
    if owner is None:
        return
    observations._parity_mmap_owner = None
    memory_map = getattr(owner, "_mmap", None)
    if memory_map is not None:
        memory_map.close()


PARITY_SIDECAR_SCHEMA_VERSION = 5
PARITY_SIDECAR_TYPE = "ur5e_parity_dataset"


def parity_sidecar_path(dataset_path: Path | str) -> Path:
    dataset = Path(dataset_path)
    return dataset.with_name(dataset.name + ".provenance.json")


def _regular_file_identity(path: Path | str, *, label: str) -> tuple[int, int]:
    """Return the filesystem identity of one non-reparse regular file."""

    raw = Path(path).absolute()
    if path_contains_link_or_reparse(raw) or is_link_or_reparse(raw):
        raise ContractError(f"{label} contains a symlink/reparse point")
    try:
        metadata = raw.stat(follow_symlinks=False)
    except OSError as exc:
        raise ContractError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_ino) <= 0:
        raise ContractError(f"{label} has no trustworthy regular-file identity")
    return int(metadata.st_dev), int(metadata.st_ino)


def _remove_owned_published_file(
    path: Path | str,
    *,
    expected_identity: tuple[int, int],
    quarantine_parent: Path | str | None = None,
) -> None:
    """Quarantine only the exact regular file published by this caller.

    Keeping the failed artifact under an unpredictable private sibling avoids
    a second check/delete race: no path-based unlink is attempted after the
    ownership check.  If the public name was exchanged before the rename, the
    unowned file is restored when possible and is never deleted.
    """

    raw = Path(path).absolute()
    if not raw.exists() and not is_link_or_reparse(raw):
        return
    current = _regular_file_identity(raw, label="published artifact cleanup target")
    if current != expected_identity:
        raise ContractError(
            "published artifact identity changed; refusing to delete an unowned file"
        )
    target_parent = (
        raw.parent
        if quarantine_parent is None
        else Path(quarantine_parent).absolute()
    )
    if (
        target_parent not in {raw.parent, raw.parent.parent}
        or path_contains_link_or_reparse(target_parent)
        or not target_parent.is_dir()
    ):
        raise ContractError("published artifact quarantine parent is unsafe")
    prefix = (
        f".{raw.name}.failed-"
        if target_parent == raw.parent
        else f".{raw.parent.name}-{raw.name}.failed-"
    )
    quarantine = target_parent / (
        f"{prefix}{secrets.token_hex(16)}"
    )
    try:
        os.rename(raw, quarantine)
    except OSError as exc:
        raise ContractError(f"cannot quarantine owned published file: {exc}") from exc
    moved = _regular_file_identity(
        quarantine, label="quarantined published artifact"
    )
    if moved != expected_identity:
        if not raw.exists() and not is_link_or_reparse(raw):
            try:
                os.rename(quarantine, raw)
            except OSError:
                pass
        raise ContractError(
            "publication cleanup raced with an unowned file; nothing was deleted"
        )


def publish_regular_file_no_replace(
    source: Path | str,
    destination: Path | str,
    *,
    quarantine_parent: Path | str | None = None,
) -> tuple[int, int]:
    """Atomically publish one complete same-filesystem file without overwriting.

    ``Path.replace`` has a check/use race and overwrites an artifact published by
    another process.  A hard-link is an atomic no-replace directory operation;
    the private staging name is removed only after the public name exists.
    """
    src = Path(source).absolute()
    dst = Path(destination).absolute()
    direct_sibling = src.parent == dst.parent
    private_child = (
        src.parent.parent == dst.parent
        and src.parent.name.startswith(".parity-build-")
    )
    if (not (direct_sibling or private_child)) \
            or path_contains_link_or_reparse(src) \
            or path_contains_link_or_reparse(dst):
        raise ContractError("atomic artifact publication requires safe sibling paths")
    if not src.is_file() or is_link_or_reparse(src):
        raise ContractError("atomic artifact source is not a regular file")
    source_identity = _regular_file_identity(src, label="atomic artifact source")
    published = False
    try:
        os.link(src, dst)
        published = True
    except FileExistsError as exc:
        raise ContractError(f"artifact already exists; refusing overwrite: {dst}") from exc
    except OSError as exc:
        raise ContractError(f"atomic no-replace publication failed for {dst}: {exc}") from exc
    try:
        if _regular_file_identity(dst, label="published atomic artifact") != source_identity:
            raise ContractError("published artifact is not the staged regular file")
        if _regular_file_identity(src, label="atomic artifact source") != source_identity:
            raise ContractError("atomic artifact source changed during publication")
        src.unlink()
        if _regular_file_identity(dst, label="published atomic artifact") != source_identity:
            raise ContractError("published artifact identity changed during publication")
    except BaseException as exc:
        if published:
            try:
                _remove_owned_published_file(
                    dst,
                    expected_identity=source_identity,
                    quarantine_parent=quarantine_parent,
                )
            except BaseException as cleanup_exc:
                raise ContractError(
                    f"atomic artifact publication failed and owned cleanup failed: {cleanup_exc}"
                ) from exc
        raise
    return source_identity


def scan_parity_source_dumps(
    capture_manifest_path: Path | str,
    *,
    expected_dim: int,
    candidate_dir: Path | str,
    project_root: Path | str,
) -> tuple[dict, list[Path]]:
    """Verify one explicit sealed capture; never discover sibling/stale dumps."""
    return verify_parity_capture(
        capture_manifest_path,
        candidate_dir=candidate_dir,
        project_root=project_root,
        expected_dim=expected_dim,
        require_current_context=True,
    )


def verified_capture_records_from_manifest_bytes(
    manifest_path: Path,
    *,
    expected_sha256: str,
    ordered_files: list[Path],
) -> list[dict]:
    """Read and align the exact capture-manifest bytes returned by verification."""

    payload, _ = _read_safe_regular_bytes(manifest_path, label="capture manifest")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ContractError("capture manifest changed after verification")
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("verified capture manifest cannot be decoded") from exc
    records = manifest.get("files")
    if not isinstance(records, dict):
        raise ContractError("verified capture manifest file inventory is missing")
    capture = manifest_path.parent.resolve()
    expected_relatives = sorted(records)
    try:
        actual_relatives = [
            path.resolve().relative_to(capture).as_posix() for path in ordered_files
        ]
    except ValueError as exc:
        raise ContractError("verified capture file escaped its manifest root") from exc
    if actual_relatives != expected_relatives:
        raise ContractError("verified capture file order differs from manifest inventory")
    return [records[relative] for relative in expected_relatives]


class _HashingReader:
    """Minimal reader that hashes exactly the bytes consumed by the NPY parser."""

    def __init__(self, stream, digest: "hashlib._Hash"):
        self._stream = stream
        self._digest = digest

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self._digest.update(data)
        return data


def _read_exact_hashed(stream: _HashingReader, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(size)
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ContractError("parity dataset ended before its declared observation rows")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stream_verify_parity_dataset_content(
    dataset: Path,
    observations: torch.Tensor,
    *,
    source_files: list[Path],
    source_records: list[dict],
    expected_dim: int,
    expected_path: str,
) -> dict:
    """Bind evaluated NPY rows bit-for-bit to sealed capture observations.

    Only one capture member (at most the fixed 100-row episode horizon) is in
    memory at a time.  The NPY header/data and SHA-256 are consumed from the
    same open file descriptor, and every row is also compared with the tensor
    that was used for actor/TorchScript parity.
    """

    if len(source_files) != len(source_records) or not source_files:
        raise ContractError("parity source file/record inventory is empty or misaligned")
    raw = dataset.absolute()
    if path_contains_link_or_reparse(raw):
        raise ContractError("parity dataset contains a symlink/reparse path")
    digest = hashlib.sha256()
    try:
        with raw.open("rb") as file_stream:
            before = os.fstat(file_stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError("parity dataset is not a regular file")
            stream = _HashingReader(file_stream, digest)
            try:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                        stream
                    )
                elif version == (2, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                        stream
                    )
                else:
                    raise ContractError(
                        f"parity dataset uses unsupported NPY version {version!r}"
                    )
            except (EOFError, ValueError) as exc:
                raise ContractError(f"cannot decode parity dataset NPY header: {exc}") from exc
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or shape[0] < MIN_PARITY_SAMPLES
                or shape[1] != expected_dim
                or fortran_order
                or np.dtype(dtype) != np.dtype(np.float32)
            ):
                raise ContractError(
                    "parity dataset must be exact C-order float32 with shape "
                    f"[N,{expected_dim}], N>={MIN_PARITY_SAMPLES}; got "
                    f"shape={shape}, dtype={dtype}, fortran_order={fortran_order}"
                )
            if observations.device.type != "cpu" \
                    or observations.dtype != torch.float32 \
                    or tuple(observations.shape) != shape:
                raise ContractError(
                    "evaluated parity tensor differs from the exact dataset dtype/shape/device"
                )

            row_offset = 0
            for path, record in zip(source_files, source_records, strict=True):
                block = read_verified_capture_observation_block(
                    path, record, expected_dim=expected_dim
                )
                row_count = int(block.shape[0])
                raw_block = _read_exact_hashed(stream, int(block.nbytes))
                source_bytes = block.tobytes(order="C")
                if raw_block != source_bytes:
                    dataset_block = np.frombuffer(raw_block, dtype=np.float32).reshape(
                        block.shape
                    )
                    max_diff = float(
                        np.max(
                            np.abs(
                                dataset_block.astype(np.float64)
                                - block.astype(np.float64)
                            )
                        )
                    )
                    raise ContractError(
                        "parity dataset rows differ from sealed capture observations: "
                        f"{path.name}, max_abs_diff={max_diff:.9g}"
                    )
                evaluated = (
                    observations[row_offset : row_offset + row_count]
                    .detach()
                    .contiguous()
                    .numpy()
                )
                if evaluated.tobytes(order="C") != raw_block:
                    raise ContractError(
                        "evaluated parity tensor differs from the same-opened dataset rows"
                    )
                row_offset += row_count
            if row_offset != shape[0]:
                raise ContractError(
                    "parity dataset row count differs from canonical capture order"
                )
            if stream.read(1):
                raise ContractError("parity dataset contains trailing bytes")
            after = os.fstat(file_stream.fileno())
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(getattr(before, name, None) != getattr(after, name, None)
                   for name in stable_fields):
                raise ContractError("parity dataset changed while it was verified")
    except OSError as exc:
        raise ContractError(f"cannot stream parity dataset: {dataset}: {exc}") from exc

    return {
        "path": expected_path,
        "sha256": digest.hexdigest(),
        "bytes": int(before.st_size),
        "format": "npy",
        "rows": int(observations.shape[0]),
        "observation_dim": int(observations.shape[1]),
    }


def _prepare_parity_sidecar(
    value: object,
    *,
    dataset: Path,
    expected_rows: int | None,
    expected_dim: int,
    layout: object,
    obs_cfg_hash: str,
    training_run: dict,
    candidate_dir: Path,
    candidate_manifest: dict,
    capture_manifest_path: Path,
    project_root: Path,
) -> tuple[dict, str, list[Path], list[dict]]:
    """Validate authority metadata and return the exact capture inventory."""
    expected_keys = {
        "schema_version", "record_type", "dataset", "observation_contract",
        "training", "candidate", "source_dumps",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ContractError("parity sidecar has an incomplete/unexpected field set")
    if (
        value.get("schema_version") != PARITY_SIDECAR_SCHEMA_VERSION
        or value.get("record_type") != PARITY_SIDECAR_TYPE
    ):
        raise ContractError("parity sidecar schema/type mismatch")
    try:
        dataset_relative = dataset.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ContractError("parity dataset must be inside the project ignored logs tree") from exc
    dataset_record = value.get("dataset")
    if not isinstance(dataset_record, dict) or set(dataset_record) != {
        "path", "sha256", "bytes", "format", "rows", "observation_dim",
    } or dataset_record.get("path") != dataset_relative \
            or re.fullmatch(r"[0-9a-f]{64}", str(dataset_record.get("sha256", ""))) is None \
            or type(dataset_record.get("bytes")) is not int \
            or dataset_record.get("bytes") <= 0 \
            or dataset_record.get("format") != "npy" \
            or type(dataset_record.get("rows")) is not int \
            or dataset_record.get("rows") < MIN_PARITY_SAMPLES \
            or (expected_rows is not None and dataset_record.get("rows") != expected_rows) \
            or type(dataset_record.get("observation_dim")) is not int \
            or dataset_record.get("observation_dim") != expected_dim:
        raise ContractError("parity sidecar dataset byte/shape binding is malformed")
    normalized_layout = _validated_parity_layout(layout, expected_dim)
    if value.get("observation_contract") != {
        "cfg_hash": obs_cfg_hash,
        "layout": normalized_layout,
        "action_mode": "ik_rel",
    }:
        raise ContractError("parity sidecar observation config/layout contract mismatch")
    frozen_run = _validate_training_run_provenance(training_run)
    if value.get("training") != {
        "receipt_sha256": frozen_run["receipt_sha256"],
        "checkpoint_sha256": frozen_run["checkpoint_sha256"],
        "checkpoint_name": frozen_run["checkpoint_name"],
        "run_root": frozen_run["run_root"],
    }:
        raise ContractError("parity sidecar is not bound to the selected checkpoint/run receipt")
    if candidate_manifest["source"]["training_run"] != frozen_run:
        raise ContractError("parity candidate is not bound to the selected training receipt")
    expected_candidate = candidate_receipt_binding(
        candidate_dir, candidate_manifest, project_root=project_root
    )
    if value.get("candidate") != expected_candidate:
        raise ContractError("parity sidecar is not bound to the exact export candidate")
    source_summary, source_files = scan_parity_source_dumps(
        capture_manifest_path,
        expected_dim=expected_dim,
        candidate_dir=candidate_dir,
        project_root=project_root,
    )
    if value.get("source_dumps") != source_summary:
        raise ContractError("parity sidecar source dump count/aggregate binding mismatch")
    if source_summary["total_rows"] != dataset_record["rows"]:
        raise ContractError("parity dataset row count differs from its source dumps")
    source_records = verified_capture_records_from_manifest_bytes(
        capture_manifest_path,
        expected_sha256=source_summary["capture_manifest_sha256"],
        ordered_files=source_files,
    )
    return value, dataset_relative, source_files, source_records


def _validate_parity_sidecar(
    value: object,
    *,
    dataset: Path,
    observations: torch.Tensor,
    layout: object,
    obs_cfg_hash: str,
    training_run: dict,
    candidate_dir: Path,
    candidate_manifest: dict,
    capture_manifest_path: Path,
    project_root: Path,
) -> dict:
    if observations.ndim != 2:
        raise ContractError("parity tensor must be two-dimensional")
    sealed, dataset_relative, source_files, source_records = _prepare_parity_sidecar(
        value,
        dataset=dataset,
        expected_rows=int(observations.shape[0]),
        expected_dim=int(observations.shape[1]),
        layout=layout,
        obs_cfg_hash=obs_cfg_hash,
        training_run=training_run,
        candidate_dir=candidate_dir,
        candidate_manifest=candidate_manifest,
        capture_manifest_path=capture_manifest_path,
        project_root=project_root,
    )
    exact_dataset = _stream_verify_parity_dataset_content(
        dataset,
        observations,
        source_files=source_files,
        source_records=source_records,
        expected_dim=int(observations.shape[1]),
        expected_path=dataset_relative,
    )
    if sealed["dataset"] != exact_dataset:
        raise ContractError("parity sidecar differs from same-opened dataset bytes")
    return sealed


def _parity_provenance_projection(
    sealed: dict,
    *,
    sidecar_sha256: str,
    obs_cfg_hash: str,
    candidate_manifest: dict,
    representativeness: dict,
) -> dict:
    source = sealed["source_dumps"]
    return {
        "dataset_sha256": sealed["dataset"]["sha256"],
        "dataset_bytes": sealed["dataset"]["bytes"],
        "dataset_format": sealed["dataset"]["format"],
        "dataset_rows": sealed["dataset"]["rows"],
        "observation_dim": sealed["dataset"]["observation_dim"],
        "dataset_sidecar_sha256": sidecar_sha256,
        "observation_cfg_hash": obs_cfg_hash,
        "source_training_receipt_sha256": sealed["training"]["receipt_sha256"],
        "source_checkpoint_sha256": sealed["training"]["checkpoint_sha256"],
        "source_run_root": sealed["training"]["run_root"],
        "candidate_path": sealed["candidate"]["path"],
        "candidate_manifest_sha256": sealed["candidate"]["manifest_sha256"],
        "candidate_policy_sha256": sealed["candidate"]["policy_sha256"],
        "candidate_obs_spec_sha256": sealed["candidate"]["obs_spec_sha256"],
        "source_dump_root": source["root"],
        "source_capture_id": source["capture_id"],
        "source_capture_manifest_sha256": source["capture_manifest_sha256"],
        "source_capture_episode_count": source["episode_count"],
        "source_dump_file_count": source["file_count"],
        "source_dump_aggregate_sha256": source["aggregate_sha256"],
        "rollout_safety_contract": candidate_manifest["rollout_safety_contract"],
        "source_rollout_safety": source["rollout_safety"],
        "source_rollout_safety_sha256": source["rollout_safety_sha256"],
        "source_capture_seed": source["seed"],
        "source_episode_outcomes": source["episode_outcomes"],
        "source_episode_outcomes_sha256": source["episode_outcomes_sha256"],
        "source_runtime_authority_sha256": source["runtime_authority_sha256"],
        "source_runtime_authority_bytes": source["runtime_authority_bytes"],
        "source_runtime_authority": source["runtime_authority"],
        "representativeness": representativeness,
    }


def _load_parity_authority_context(
    path: Path | str,
    *,
    candidate_dir: Path | str,
    capture_manifest_path: Path | str,
    project_root: Path | str,
) -> tuple[Path, Path, dict, str, Path, dict, Path]:
    root = Path(project_root).resolve()
    src_raw = Path(path).expanduser().absolute()
    sidecar_raw = parity_sidecar_path(src_raw).absolute()
    if path_contains_link_or_reparse(src_raw) or path_contains_link_or_reparse(sidecar_raw):
        raise ContractError("parity dataset/sidecar path contains a symlink/reparse point")
    src = src_raw.resolve()
    sidecar = sidecar_raw.resolve()
    allowed_parent = (root / "logs/parity").resolve()
    if src.parent != allowed_parent or sidecar.parent != allowed_parent:
        raise ContractError(f"parity dataset and sidecar must be directly under {allowed_parent}")
    if not src.is_file() or src.suffix.lower() != ".npy" or not sidecar.is_file():
        raise ContractError("parity release requires a regular .npy plus .provenance.json sidecar")
    sidecar_payload, _ = _read_safe_regular_bytes(
        sidecar, label="parity provenance sidecar"
    )
    sidecar_sha256 = hashlib.sha256(sidecar_payload).hexdigest()
    try:
        decoded = json.loads(sidecar_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode parity provenance sidecar: {exc}") from exc
    candidate_raw = Path(candidate_dir).expanduser().absolute()
    if path_contains_link_or_reparse(candidate_raw):
        raise ContractError("parity candidate path contains a symlink/reparse point")
    candidate_path = candidate_raw.resolve()
    candidate_manifest = verify_candidate(
        candidate_path,
        project_root=root,
        require_current_context=True,
    )
    capture_raw = Path(capture_manifest_path).expanduser().absolute()
    if path_contains_link_or_reparse(capture_raw):
        raise ContractError("parity capture-manifest path contains a symlink/reparse point")
    return (
        root,
        src,
        decoded,
        sidecar_sha256,
        candidate_path,
        candidate_manifest,
        capture_raw.resolve(),
    )


def parity_dataset_provenance(
    path: Path | str,
    observations: torch.Tensor,
    *,
    layout: object,
    obs_cfg_hash: str,
    training_run: dict,
    candidate_dir: Path | str,
    capture_manifest_path: Path | str,
    project_root: Path | str,
) -> dict:
    """Verify, never invent, collection/build-time representative-data provenance."""
    if observations.ndim != 2:
        raise ContractError("parity tensor must be two-dimensional")
    (
        root,
        src,
        decoded,
        sidecar_sha256,
        candidate_path,
        candidate_manifest,
        capture_path,
    ) = _load_parity_authority_context(
        path,
        candidate_dir=candidate_dir,
        capture_manifest_path=capture_manifest_path,
        project_root=project_root,
    )
    sealed = _validate_parity_sidecar(
        decoded,
        dataset=src,
        observations=observations,
        layout=layout,
        obs_cfg_hash=obs_cfg_hash,
        training_run=training_run,
        candidate_dir=candidate_path,
        candidate_manifest=candidate_manifest,
        capture_manifest_path=capture_path,
        project_root=root,
    )
    array = observations.detach().cpu().numpy()
    representativeness = _parity_representativeness(array, layout=layout)
    return _parity_provenance_projection(
        sealed,
        sidecar_sha256=sidecar_sha256,
        obs_cfg_hash=obs_cfg_hash,
        candidate_manifest=candidate_manifest,
        representativeness=representativeness,
    )


def _validate_parity_provenance(
    parity: object,
    *,
    training_run: dict,
    observation_dim: int,
    layout: object,
    obs_cfg_hash: str,
    policy_sha256: str,
    obs_spec_sha256: str,
    config_root: Path,
) -> dict:
    if not isinstance(parity, dict):
        raise ContractError("release parity evidence must be a mapping")
    for field in (
        "dataset_sha256",
        "dataset_sidecar_sha256",
        "source_dump_aggregate_sha256",
        "source_training_receipt_sha256",
        "source_checkpoint_sha256",
        "candidate_manifest_sha256",
        "candidate_policy_sha256",
        "candidate_obs_spec_sha256",
        "source_capture_manifest_sha256",
        "source_rollout_safety_sha256",
        "source_episode_outcomes_sha256",
        "source_runtime_authority_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(parity.get(field, ""))) is None:
            raise ContractError(f"release parity {field} is missing or invalid")
    if parity["source_training_receipt_sha256"] != training_run["receipt_sha256"]:
        raise ContractError("parity dataset is not bound to the selected training receipt")
    if parity["source_checkpoint_sha256"] != training_run["checkpoint_sha256"]:
        raise ContractError("parity dataset is not bound to the selected checkpoint")
    if parity.get("source_run_root") != training_run["run_root"]:
        raise ContractError("parity dataset source run differs from the selected training run")
    if parity["candidate_policy_sha256"] != policy_sha256 \
            or parity["candidate_obs_spec_sha256"] != obs_spec_sha256:
        raise ContractError("parity candidate artifacts differ from the released artifacts")
    candidate_path = str(parity.get("candidate_path", ""))
    expected_candidate_id = training_run["checkpoint_sha256"][:16]
    if candidate_path != f"logs/export_candidates/{expected_candidate_id}":
        raise ContractError("parity candidate path differs from the selected checkpoint")
    if parity.get("dataset_format") not in {"npy", "npz"}:
        raise ContractError("release parity dataset format must be npy or npz")
    for name in ("dataset_bytes", "dataset_rows", "observation_dim"):
        value = parity.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError(f"release parity {name} must be a positive integer")
    if parity["dataset_rows"] != parity.get("samples"):
        raise ContractError("release parity dataset row count differs from parity sample count")
    if parity["observation_dim"] != observation_dim:
        raise ContractError("release parity dataset observation_dim differs from obs_spec")
    if parity.get("observation_cfg_hash") != obs_cfg_hash \
            or re.fullmatch(r"[0-9a-f]{16}", str(obs_cfg_hash)) is None:
        raise ContractError("release parity observation config hash differs from obs_spec")
    source_root = str(parity.get("source_dump_root", ""))
    candidate_manifest_id = parity["candidate_manifest_sha256"][:16]
    capture_id = str(parity.get("source_capture_id", ""))
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", capture_id) is None \
            or source_root != (
                f"Phase3/logs/obs_dump_sim/{candidate_manifest_id}/{capture_id}"
            ):
        raise ContractError("release parity source capture root is not canonical")
    episode_count = parity.get("source_capture_episode_count")
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) \
            or episode_count <= 0:
        raise ContractError("release parity capture episode count is invalid")
    source_count = parity.get("source_dump_file_count")
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count <= 0:
        raise ContractError("release parity source dump count is invalid")
    if parity.get("rollout_safety_contract") != rollout_safety_contract():
        raise ContractError("release parity rollout-safety contract differs from candidate")
    rollout_safety = _validate_rollout_safety(parity.get("source_rollout_safety"))
    if parity["source_rollout_safety_sha256"] != _canonical_mapping_sha256(rollout_safety):
        raise ContractError("release parity rollout-safety evidence hash mismatch")
    if rollout_safety["denominator"] != parity["dataset_rows"]:
        raise ContractError("release parity rollout-safety denominator differs from dataset rows")
    outcome_receipt = _validate_embedded_phase3_outcomes(
        parity.get("source_episode_outcomes"),
        project_root=config_root,
        expected_config_hashes=training_run["source"]["canonical_configs"],
    )
    if type(parity.get("source_capture_seed")) is not int \
            or parity.get("source_capture_seed") != PHASE3_CAPTURE_SEED:
        raise ContractError("release parity source seed is not exact Phase 3 seed 45")
    if parity["source_episode_outcomes_sha256"] != _canonical_mapping_sha256(
        outcome_receipt
    ):
        raise ContractError("release parity episode-outcome receipt hash mismatch")
    if outcome_receipt["episode_count"] != episode_count \
            or outcome_receipt["total_steps"] != parity["dataset_rows"]:
        raise ContractError("release parity episode outcomes differ from dataset/capture totals")
    authority_bytes = parity.get("source_runtime_authority_bytes")
    if isinstance(authority_bytes, bool) or not isinstance(authority_bytes, int) \
            or authority_bytes <= 0:
        raise ContractError("release parity runtime-authority byte count is invalid")
    try:
        validate_runtime_authority_execution_receipt(
            parity.get("source_runtime_authority"),
            expected_stage="phase3",
            expected_purpose="phase3-rehearsal",
            expected_git_commit=training_run["source"]["project_git_commit"],
            expected_configs=training_run["source"]["canonical_configs"],
            required_configs=PHASE3_RUNTIME_REQUIRED_CONFIGS,
            project_root=config_root,
        )
    except RuntimeAuthorityError as exc:
        raise ContractError(f"release parity runtime authority is invalid: {exc}") from exc

    normalized_layout = _validated_parity_layout(layout, observation_dim)
    evidence = parity.get("representativeness")
    if not isinstance(evidence, dict):
        raise ContractError("release parity representativeness evidence is missing")
    sampled_rows = evidence.get("sampled_rows")
    unique_rows = evidence.get("unique_sampled_rows")
    minimum_ratio = evidence.get("minimum_unique_ratio")
    if (
        isinstance(sampled_rows, bool)
        or not isinstance(sampled_rows, int)
        or sampled_rows < PARITY_AUDIT_ROWS
        or sampled_rows > parity["dataset_rows"]
        or isinstance(unique_rows, bool)
        or not isinstance(unique_rows, int)
        or unique_rows < math.ceil(sampled_rows * MIN_PARITY_UNIQUE_RATIO)
        or unique_rows > sampled_rows
        or minimum_ratio != MIN_PARITY_UNIQUE_RATIO
    ):
        raise ContractError("release parity row-diversity evidence is below contract")
    blocks = evidence.get("blocks")
    expected_blocks = {str(item["name"]): int(item["size"]) for item in normalized_layout}
    if not isinstance(blocks, dict) or set(blocks) != set(expected_blocks):
        raise ContractError("release parity block-diversity evidence is incomplete")
    for name, size in expected_blocks.items():
        item = blocks[name]
        required = max(1, int(math.ceil(size * MIN_BLOCK_VARYING_FRACTION)))
        if (
            not isinstance(item, dict)
            or item.get("size") != size
            or item.get("required_varying_features") != required
            or isinstance(item.get("varying_features"), bool)
            or not isinstance(item.get("varying_features"), int)
            or item["varying_features"] < required
            or item["varying_features"] > size
        ):
            raise ContractError(f"release parity block {name!r} diversity is below contract")
        for bound in ("min", "max"):
            value = item.get(bound)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                raise ContractError(f"release parity block {name!r} bounds are invalid")
        if float(item["min"]) > float(item["max"]):
            raise ContractError(f"release parity block {name!r} bounds are reversed")
        if name in {"depth", "rgb"} and (
            float(item["min"]) < -1.0e-6 or float(item["max"]) > 1.0 + 1.0e-6
        ):
            raise ContractError(f"release parity block {name!r} is outside [0,1]")
    return parity


@torch.inference_mode()
def compare_actor_and_export(
    actor: torch.nn.Module,
    scripted: torch.jit.ScriptModule,
    observations: torch.Tensor,
    *,
    batch_size: int = 256,
) -> dict[str, float | int]:
    max_abs = 0.0
    sum_abs = 0.0
    count = 0
    for start in range(0, observations.shape[0], batch_size):
        obs = observations[start : start + batch_size]
        reference = actor(obs).detach().cpu()
        exported = scripted(obs).detach().cpu()
        expected_shape = (obs.shape[0], 7)
        if tuple(reference.shape) != expected_shape or tuple(exported.shape) != expected_shape:
            raise ContractError(
                f"export parity shape mismatch: actor={tuple(reference.shape)}, "
                f"scripted={tuple(exported.shape)}"
            )
        if not torch.isfinite(reference).all() or not torch.isfinite(exported).all():
            raise ContractError("actor/export parity output contains NaN/Inf")
        error = torch.abs(reference - exported)
        max_abs = max(max_abs, float(error.max().item()))
        sum_abs += float(error.sum().item())
        count += error.numel()
    result = {
        "samples": int(observations.shape[0]),
        "max_abs_error": max_abs,
        "mean_abs_error": sum_abs / max(count, 1),
        "threshold": MAX_PARITY_ERROR,
    }
    if max_abs > MAX_PARITY_ERROR:
        raise ContractError(f"TorchScript parity error {max_abs:g} exceeds {MAX_PARITY_ERROR:g}")
    return result


@torch.inference_mode()
def benchmark_policy(
    module: torch.jit.ScriptModule,
    observations: torch.Tensor,
    *,
    warmup: int = 20,
    samples: int = MIN_LATENCY_SAMPLES,
    allow_repeated_rows: bool = False,
) -> dict[str, float | int]:
    import time

    if observations.device.type != "cpu" or any(
        parameter.device.type != "cpu" for parameter in module.parameters()
    ):
        raise ContractError("policy release latency must be measured on CPU")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ContractError("policy benchmark warmup must be a non-negative integer")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < MIN_LATENCY_SAMPLES:
        raise ContractError(
            f"policy benchmark requires at least {MIN_LATENCY_SAMPLES} measured samples"
        )
    if observations.ndim != 2 or observations.shape[0] < 1:
        raise ContractError("policy benchmark requires a non-empty 2D observation tensor")
    if observations.shape[0] < MIN_LATENCY_SAMPLES and not allow_repeated_rows:
        raise ContractError(
            f"policy benchmark requires at least {MIN_LATENCY_SAMPLES} observation rows"
        )
    rows = observations[: min(observations.shape[0], samples)]
    for i in range(warmup):
        module(rows[i % rows.shape[0] : i % rows.shape[0] + 1])
    elapsed: list[float] = []
    for i in range(samples):
        obs = rows[i % rows.shape[0] : i % rows.shape[0] + 1]
        t0 = time.perf_counter()
        out = module(obs)
        elapsed.append((time.perf_counter() - t0) * 1000.0)
        if tuple(out.shape) != (1, 7) or not torch.isfinite(out).all():
            raise ContractError("policy benchmark produced invalid output")
    values = np.asarray(elapsed)
    return {
        "samples": int(samples),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": float(values.max()),
    }


@torch.inference_mode()
def _stream_evaluate_sealed_parity_dataset(
    dataset: Path,
    actor: torch.nn.Module,
    scripted: torch.jit.ScriptModule,
    *,
    source_files: list[Path],
    source_records: list[dict],
    expected_dim: int,
    expected_path: str,
    layout: object,
) -> tuple[dict, dict, dict, dict]:
    """Measure only private copies of rows read and authenticated on one handle.

    The former export path exposed a writable copy-on-write mmap to parity and
    latency callers, then reopened the dataset for provenance.  A caller could
    therefore evaluate bytes B, restore bytes A in the mapping, and have the
    later provenance check authenticate A.  This authority path instead reads,
    hashes, capture-compares, and evaluates every row before the single dataset
    descriptor is closed.  Neither policy receives the authenticated buffer:
    each invocation gets a private tensor copy whose immutability is checked.
    """

    import time

    if isinstance(expected_dim, bool) or not isinstance(expected_dim, int) \
            or expected_dim <= 0:
        raise ContractError("parity observation dimension must be a positive integer")
    if len(source_files) != len(source_records) or not source_files:
        raise ContractError("parity source file/record inventory is empty or misaligned")
    if any(parameter.device.type != "cpu" for parameter in actor.parameters()) \
            or any(buffer.device.type != "cpu" for buffer in actor.buffers()) \
            or any(parameter.device.type != "cpu" for parameter in scripted.parameters()) \
            or any(buffer.device.type != "cpu" for buffer in scripted.buffers()):
        raise ContractError("sealed parity and latency evaluation must run on CPU")

    raw = dataset.absolute()
    if path_contains_link_or_reparse(raw):
        raise ContractError("parity dataset contains a symlink/reparse path")
    digest = hashlib.sha256()
    max_abs = 0.0
    sum_abs = 0.0
    error_count = 0
    latency_ms: list[float] = []
    warmup_complete = False
    audit_rows: np.ndarray | None = None
    audit_indices: np.ndarray | None = None
    audit_cursor = 0
    before: os.stat_result | None = None
    shape: tuple[int, int] | None = None

    try:
        with raw.open("rb") as file_stream:
            before = os.fstat(file_stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ContractError("parity dataset is not a regular file")
            stream = _HashingReader(file_stream, digest)
            try:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    decoded_shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_1_0(stream)
                    )
                elif version == (2, 0):
                    decoded_shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_2_0(stream)
                    )
                else:
                    raise ContractError(
                        f"parity dataset uses unsupported NPY version {version!r}"
                    )
            except (EOFError, ValueError) as exc:
                raise ContractError(f"cannot decode parity dataset NPY header: {exc}") from exc
            if (
                not isinstance(decoded_shape, tuple)
                or len(decoded_shape) != 2
                or decoded_shape[0] < MIN_PARITY_SAMPLES
                or decoded_shape[1] != expected_dim
                or fortran_order
                or np.dtype(dtype) != np.dtype(np.float32)
            ):
                raise ContractError(
                    "parity dataset must be exact C-order float32 with shape "
                    f"[N,{expected_dim}], N>={MIN_PARITY_SAMPLES}; got "
                    f"shape={decoded_shape}, dtype={dtype}, "
                    f"fortran_order={fortran_order}"
                )
            shape = (int(decoded_shape[0]), int(decoded_shape[1]))
            normalized_layout = _validated_parity_layout(layout, expected_dim)
            audit_count = min(PARITY_AUDIT_ROWS, shape[0])
            audit_indices = np.linspace(
                0, shape[0] - 1, num=audit_count, dtype=np.int64
            )
            audit_rows = np.empty((audit_count, expected_dim), dtype=np.float32)

            row_offset = 0
            for path, record in zip(source_files, source_records, strict=True):
                capture_block = read_verified_capture_observation_block(
                    path, record, expected_dim=expected_dim
                )
                row_count = int(capture_block.shape[0])
                raw_block = _read_exact_hashed(stream, int(capture_block.nbytes))
                if raw_block != capture_block.tobytes(order="C"):
                    dataset_block = np.frombuffer(
                        raw_block, dtype=np.float32
                    ).reshape(capture_block.shape)
                    max_diff = float(
                        np.max(
                            np.abs(
                                dataset_block.astype(np.float64)
                                - capture_block.astype(np.float64)
                            )
                        )
                    )
                    raise ContractError(
                        "parity dataset rows differ from sealed capture observations: "
                        f"{path.name}, max_abs_diff={max_diff:.9g}"
                    )

                # ``bytes`` owns immutable block storage.  Every tensor passed
                # to policy code is then copied again, so a policy cannot alter
                # the bytes authenticated above or another policy's input.
                trusted = np.frombuffer(raw_block, dtype=np.float32).reshape(
                    capture_block.shape
                )
                block_end = row_offset + row_count
                while audit_cursor < len(audit_indices) \
                        and int(audit_indices[audit_cursor]) < block_end:
                    global_index = int(audit_indices[audit_cursor])
                    if global_index < row_offset:
                        raise ContractError("parity audit row order is inconsistent")
                    audit_rows[audit_cursor] = trusted[global_index - row_offset]
                    audit_cursor += 1

                for batch_start in range(0, row_count, 256):
                    trusted_tensor = torch.from_numpy(
                        trusted[batch_start : batch_start + 256].copy()
                    )
                    actor_input = trusted_tensor.clone()
                    scripted_input = trusted_tensor.clone()
                    reference = actor(actor_input)
                    exported = scripted(scripted_input)
                    if not torch.equal(actor_input, trusted_tensor) \
                            or not torch.equal(scripted_input, trusted_tensor):
                        raise ContractError("policy mutated an authenticated parity input")
                    reference = reference.detach().cpu().clone()
                    exported = exported.detach().cpu().clone()
                    expected_shape = (trusted_tensor.shape[0], 7)
                    if tuple(reference.shape) != expected_shape \
                            or tuple(exported.shape) != expected_shape:
                        raise ContractError(
                            "export parity shape mismatch: "
                            f"actor={tuple(reference.shape)}, "
                            f"scripted={tuple(exported.shape)}"
                        )
                    if not torch.isfinite(reference).all() \
                            or not torch.isfinite(exported).all():
                        raise ContractError("actor/export parity output contains NaN/Inf")
                    error = torch.abs(reference - exported)
                    max_abs = max(max_abs, float(error.max().item()))
                    sum_abs += float(error.sum().item())
                    error_count += error.numel()

                if not warmup_complete:
                    trusted_row = torch.from_numpy(trusted[0:1].copy())
                    for _ in range(20):
                        warm_input = trusted_row.clone()
                        warm_output = scripted(warm_input)
                        if not torch.equal(warm_input, trusted_row) \
                                or tuple(warm_output.shape) != (1, 7) \
                                or not torch.isfinite(warm_output).all():
                            raise ContractError(
                                "policy benchmark mutated its input or produced invalid output"
                            )
                    warmup_complete = True

                remaining_latency = MIN_LATENCY_SAMPLES - len(latency_ms)
                for local_index in range(min(row_count, remaining_latency)):
                    trusted_row = torch.from_numpy(
                        trusted[local_index : local_index + 1].copy()
                    )
                    latency_input = trusted_row.clone()
                    started = time.perf_counter()
                    output = scripted(latency_input)
                    latency_ms.append((time.perf_counter() - started) * 1000.0)
                    if not torch.equal(latency_input, trusted_row) \
                            or tuple(output.shape) != (1, 7) \
                            or not torch.isfinite(output).all():
                        raise ContractError(
                            "policy benchmark mutated its input or produced invalid output"
                        )
                row_offset = block_end

            if row_offset != shape[0]:
                raise ContractError(
                    "parity dataset row count differs from canonical capture order"
                )
            if len(latency_ms) != MIN_LATENCY_SAMPLES:
                raise ContractError(
                    "sealed parity dataset lacks the required distinct latency rows"
                )
            if audit_rows is None or audit_indices is None \
                    or audit_cursor != len(audit_indices):
                raise ContractError("parity representativeness sample is incomplete")
            if stream.read(1):
                raise ContractError("parity dataset contains trailing bytes")
            after = os.fstat(file_stream.fileno())
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
            if any(
                getattr(before, name, None) != getattr(after, name, None)
                for name in stable_fields
            ):
                raise ContractError("parity dataset changed while it was evaluated")
    except OSError as exc:
        raise ContractError(f"cannot stream parity dataset: {dataset}: {exc}") from exc

    if before is None or shape is None or audit_rows is None:
        raise ContractError("parity dataset evaluation did not produce authenticated rows")
    parity = {
        "samples": shape[0],
        "max_abs_error": max_abs,
        "mean_abs_error": sum_abs / max(error_count, 1),
        "threshold": MAX_PARITY_ERROR,
    }
    if max_abs > MAX_PARITY_ERROR:
        raise ContractError(
            f"TorchScript parity error {max_abs:g} exceeds {MAX_PARITY_ERROR:g}"
        )
    latency_values = np.asarray(latency_ms, dtype=np.float64)
    latency = {
        "samples": MIN_LATENCY_SAMPLES,
        "p50_ms": float(np.percentile(latency_values, 50)),
        "p95_ms": float(np.percentile(latency_values, 95)),
        "p99_ms": float(np.percentile(latency_values, 99)),
        "max_ms": float(latency_values.max()),
    }
    exact_dataset = {
        "path": expected_path,
        "sha256": digest.hexdigest(),
        "bytes": int(before.st_size),
        "format": "npy",
        "rows": shape[0],
        "observation_dim": shape[1],
    }
    representativeness = _parity_representativeness(
        audit_rows, layout=normalized_layout
    )
    return exact_dataset, parity, latency, representativeness


def evaluate_sealed_parity_dataset(
    path: Path | str,
    actor: torch.nn.Module,
    scripted: torch.jit.ScriptModule,
    *,
    expected_dim: int,
    layout: object,
    obs_cfg_hash: str,
    training_run: dict,
    candidate_dir: Path | str,
    capture_manifest_path: Path | str,
    project_root: Path | str,
) -> tuple[dict, dict]:
    """Authenticate capture bytes and produce parity/latency in one authority call."""

    (
        root,
        src,
        decoded,
        sidecar_sha256,
        candidate_path,
        candidate_manifest,
        capture_path,
    ) = _load_parity_authority_context(
        path,
        candidate_dir=candidate_dir,
        capture_manifest_path=capture_manifest_path,
        project_root=project_root,
    )
    sealed, dataset_relative, source_files, source_records = _prepare_parity_sidecar(
        decoded,
        dataset=src,
        expected_rows=None,
        expected_dim=expected_dim,
        layout=layout,
        obs_cfg_hash=obs_cfg_hash,
        training_run=training_run,
        candidate_dir=candidate_path,
        candidate_manifest=candidate_manifest,
        capture_manifest_path=capture_path,
        project_root=root,
    )
    exact_dataset, metrics, latency, representativeness = (
        _stream_evaluate_sealed_parity_dataset(
            src,
            actor,
            scripted,
            source_files=source_files,
            source_records=source_records,
            expected_dim=expected_dim,
            expected_path=dataset_relative,
            layout=layout,
        )
    )
    if sealed["dataset"] != exact_dataset:
        raise ContractError("parity sidecar differs from same-opened dataset bytes")
    parity = _parity_provenance_projection(
        sealed,
        sidecar_sha256=sidecar_sha256,
        obs_cfg_hash=obs_cfg_hash,
        candidate_manifest=candidate_manifest,
        representativeness=representativeness,
    )
    parity.update(metrics)
    return parity, latency


def make_release_manifest(
    *,
    policy_path: Path,
    obs_spec_path: Path,
    checkpoint_path: Path,
    actor_kind: str,
    config_paths: Iterable[Path],
    config_root: Path,
    parity: dict,
    latency: dict,
    max_inference_p99_ms: float,
    git_commit: str,
    control_hz: float,
    horizon_steps: int,
    training_stack: dict | None = None,
    training_run: dict | None = None,
) -> dict:
    if actor_kind != "student":
        raise ContractError(f"only student policies may be released; got {actor_kind!r}")
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ContractError("release requires an exact 40-hex Git commit")
    parity_samples = parity.get("samples", 0)
    parity_max = parity.get("max_abs_error", float("inf"))
    parity_mean = parity.get("mean_abs_error", float("inf"))
    parity_threshold = parity.get("threshold", MAX_PARITY_ERROR)
    latency_samples = latency.get("samples", 0)
    latency_p99 = latency.get("p99_ms", float("inf"))
    numeric_values = (parity_samples, parity_max, parity_mean, parity_threshold,
                      latency_samples, latency_p99, max_inference_p99_ms)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) or float(value) < 0 for value in numeric_values):
        raise ContractError("release parity/latency evidence must contain finite non-negative values")
    if not isinstance(parity_samples, int) or not isinstance(latency_samples, int):
        raise ContractError("release parity/latency sample counts must be integers")
    if float(max_inference_p99_ms) <= 0 or float(max_inference_p99_ms) > 10.0:
        raise ContractError("release inference p99 contract must be in (0,10] ms")
    if int(parity_samples) < MIN_PARITY_SAMPLES:
        raise ContractError("release parity sample count is below contract minimum")
    if float(parity_max) > MAX_PARITY_ERROR or float(parity_mean) > float(parity_max):
        raise ContractError("release parity error exceeds contract")
    if float(parity_threshold) != MAX_PARITY_ERROR:
        raise ContractError("release parity threshold differs from the contract")
    if int(latency_samples) < MIN_LATENCY_SAMPLES \
            or float(latency_p99) > float(max_inference_p99_ms):
        raise ContractError("export-host inference p99 exceeds release latency contract")
    environment = _validate_environment(package_versions())
    for name, expected in TRAINING_STACK.items():
        if environment.get(name) != expected:
            raise ContractError(
                f"training export stack {name} must be exactly {expected}; "
                f"got {environment.get(name)!r}"
            )
    frozen_export_stack = _validate_training_stack_provenance(
        training_stack if training_stack is not None else training_stack_provenance(config_root)
    )
    frozen_training_run = _validate_training_run_provenance(training_run)
    if frozen_training_run["checkpoint_name"] != checkpoint_path.name:
        raise ContractError("training run checkpoint name differs from released checkpoint")
    files = {
        "policy.pt": sha256_file(policy_path),
        "obs_spec.json": sha256_file(obs_spec_path),
    }
    config_root = config_root.resolve()
    training_source = frozen_training_run["source"]
    configs = _snapshot_release_config_hashes(
        config_root,
        config_paths,
        expected_hashes=training_source["canonical_configs"],
    )
    if set(configs) != REQUIRED_RELEASE_CONFIGS:
        raise ContractError(
            f"release config set mismatch: expected={sorted(REQUIRED_RELEASE_CONFIGS)}, "
            f"actual={sorted(configs)}"
        )
    validate_release_calibrations(config_root, expected_hashes=configs)
    if training_source["project_git_commit"] != git_commit:
        raise ContractError(
            "export Git commit differs from the checkpoint training commit; "
            "export from the exact clean training revision"
        )
    if training_source["training_stack"] != frozen_export_stack:
        raise ContractError("export Isaac/ML stack differs from the checkpoint training stack")
    if training_source["canonical_configs"] != configs:
        raise ContractError("current release configs differ from checkpoint training configs")
    if frozen_training_run["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ContractError("released checkpoint differs from the sealed training artifact")
    obs_spec = json.loads(obs_spec_path.read_text(encoding="utf-8"))
    gripper_obs_contract = _validated_gripper_command_obs_spec(obs_spec)
    expected_layout = obs_spec.get("layout")
    if not isinstance(expected_layout, list) or not expected_layout:
        raise ContractError("obs_spec has no observation layout")
    if int(obs_spec.get("action_dim", -1)) != 7:
        raise ContractError("release action contract must be exactly 7D")
    if obs_spec.get("action_scheme") != "ik_rel":
        raise ContractError("release action mode must be ik_rel")
    _validate_parity_provenance(
        parity,
        training_run=frozen_training_run,
        observation_dim=int(obs_spec["obs_dim"]),
        layout=expected_layout,
        obs_cfg_hash=str(obs_spec.get("cfg_hash", "")),
        policy_sha256=files["policy.pt"],
        obs_spec_sha256=files["obs_spec.json"],
        config_root=config_root,
    )
    if control_hz != 20.0 or horizon_steps != 100:
        raise ContractError("release control contract must be 20 Hz / 100 steps (5 seconds)")
    return {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "model_role": "student",
        "actor_kind": actor_kind,
        "source": {
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "git_commit": training_source["project_git_commit"],
            "training_run": frozen_training_run,
        },
        "export": {
            "git_commit": git_commit,
            "training_stack": frozen_export_stack,
        },
        "files": files,
        "configs": configs,
        "runtime_contract": {
            "observation_dim": int(obs_spec["obs_dim"]),
            "observation_layout": expected_layout,
            "gripper_command_state": gripper_obs_contract,
            "action_mode": obs_spec["action_scheme"],
            "action_dim": 7,
            "control_hz": float(control_hz),
            "horizon_steps": int(horizon_steps),
            "horizon_seconds": float(horizon_steps / control_hz),
        },
        "parity": parity,
        "latency_contract": {
            "max_inference_p99_ms": float(max_inference_p99_ms),
            "export_host": latency,
        },
        "environment": environment,
        "training_stack": training_source["training_stack"],
    }


def save_release_manifest(path: Path | str, manifest: dict) -> str:
    """Write deterministic manifest bytes and return their in-memory SHA-256.

    Publication callers retain this digest instead of reopening the pathname
    after the write.  A staging-path swap can therefore never redefine which
    manifest bytes are authorized for the subsequent atomic rename.
    """

    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    Path(path).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def candidate_output_path(project_root: Path | str, checkpoint_sha256: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}", checkpoint_sha256) is None:
        raise ContractError("candidate path requires an exact checkpoint SHA-256")
    return Path(project_root).resolve() / "logs/export_candidates" / checkpoint_sha256[:16]


def make_candidate_manifest(
    *,
    policy_path: Path,
    obs_spec_path: Path,
    checkpoint_path: Path,
    git_commit: str,
    training_stack: dict,
    training_run: dict,
    student_adoption: dict,
    config_paths: Iterable[Path],
    config_root: Path,
) -> dict:
    frozen_run = _validate_training_run_provenance(training_run)
    frozen_stack = _validate_training_stack_provenance(training_stack)
    if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
        raise ContractError("candidate requires an exact clean Git commit")
    if frozen_run["checkpoint_name"] != checkpoint_path.name \
            or frozen_run["checkpoint_sha256"] != sha256_file(checkpoint_path):
        raise ContractError("candidate checkpoint differs from its training receipt")
    expected_adoption_keys = {
        "schema_version", "role", "path", "sha256", "checkpoint_name",
        "checkpoint_sha256", "training_receipt_sha256",
    }
    if (
        not isinstance(student_adoption, dict)
        or set(student_adoption) != expected_adoption_keys
        or student_adoption.get("schema_version") != 1
        or student_adoption.get("role") != "student"
        or student_adoption.get("checkpoint_name") != frozen_run["checkpoint_name"]
        or student_adoption.get("checkpoint_sha256") != frozen_run["checkpoint_sha256"]
        or student_adoption.get("training_receipt_sha256") != frozen_run["receipt_sha256"]
        or re.fullmatch(r"[0-9a-f]{64}", str(student_adoption.get("sha256", ""))) is None
    ):
        raise ContractError("candidate requires the exact adopted Student checkpoint")
    from scripts.adoption import verify_adopted_checkpoint

    current_student_adoption = verify_adopted_checkpoint(
        checkpoint_path, project_root=config_root, role="student"
    )
    if current_student_adoption != student_adoption:
        raise ContractError("Student adoption changed before candidate manifest creation")
    source = frozen_run["source"]
    if source["project_git_commit"] != git_commit or source["training_stack"] != frozen_stack:
        raise ContractError("candidate export host differs from checkpoint training context")
    root = Path(config_root).resolve()
    configs = _snapshot_release_config_hashes(
        root,
        config_paths,
        expected_hashes=source["canonical_configs"],
    )
    if set(configs) != REQUIRED_RELEASE_CONFIGS \
            or source["canonical_configs"] != configs:
        raise ContractError("candidate configs differ from sealed checkpoint configs")
    obs_spec = json.loads(obs_spec_path.read_text(encoding="utf-8"))
    gripper_obs_contract = _validated_gripper_command_obs_spec(obs_spec)
    if (
        obs_spec.get("model_role") != "student"
        or obs_spec.get("empirical_normalization") is not False
        or obs_spec.get("source_checkpoint") != checkpoint_path.name
        or obs_spec.get("policy_sha256") != sha256_file(policy_path)
        or obs_spec.get("action_scheme") != "ik_rel"
        or int(obs_spec.get("action_dim", -1)) != 7
    ):
        raise ContractError("candidate obs_spec is not a sealed 7D ik_rel student contract")
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "record_type": CANDIDATE_RECORD_TYPE,
        "approved": False,
        "source": {
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_sha256": frozen_run["checkpoint_sha256"],
            "training_receipt_sha256": frozen_run["receipt_sha256"],
            "run_root": frozen_run["run_root"],
            "training_run": frozen_run,
            "student_adoption": dict(student_adoption),
        },
        "export": {"git_commit": git_commit, "training_stack": frozen_stack},
        "configs": configs,
        "files": {
            "policy.pt": sha256_file(policy_path),
            "obs_spec.json": sha256_file(obs_spec_path),
        },
        "observation_contract": {
            "cfg_hash": obs_spec["cfg_hash"],
            "observation_dim": int(obs_spec["obs_dim"]),
            "layout": obs_spec["layout"],
            "gripper_command_state": gripper_obs_contract,
            "action_mode": "ik_rel",
            "action_dim": 7,
        },
        "rollout_safety_contract": rollout_safety_contract(),
    }


def verify_candidate(
    candidate_dir: Path | str,
    *,
    project_root: Path | str,
    require_current_context: bool = True,
    expected_manifest_sha256: str | None = None,
) -> dict:
    root = Path(project_root).resolve()
    raw = Path(candidate_dir).expanduser().absolute()
    if path_contains_link_or_reparse(raw):
        raise ContractError("candidate path contains a symlink/reparse point")
    candidate = raw.resolve()
    if candidate.parent != (root / "logs/export_candidates").resolve() \
            or re.fullmatch(r"[0-9a-f]{16}", candidate.name) is None:
        raise ContractError("candidate is outside the canonical ignored candidate store")
    entries = list(candidate.rglob("*")) if candidate.is_dir() else []
    if any(is_link_or_reparse(path) for path in entries):
        raise ContractError("candidate contains a symlink/reparse path")
    actual = {path.relative_to(candidate).as_posix() for path in entries if path.is_file()}
    expected = {"policy.pt", "obs_spec.json", CANDIDATE_MANIFEST_NAME}
    if actual != expected:
        raise ContractError(f"candidate file set mismatch: {sorted(actual)}")
    manifest_path = candidate / CANDIDATE_MANIFEST_NAME
    manifest, manifest_sha256 = _read_json_regular_file_once(
        manifest_path,
        label="candidate manifest",
        expected_sha256=expected_manifest_sha256,
    )
    expected_manifest_keys = {
        "schema_version", "record_type", "approved", "source", "export",
        "configs", "files", "observation_contract", "rollout_safety_contract",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_manifest_keys
        or
        manifest.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or manifest.get("record_type") != CANDIDATE_RECORD_TYPE
        or manifest.get("approved") is not False
    ):
        raise ContractError("candidate must be explicitly unapproved with the exact schema")
    source = manifest.get("source")
    export = manifest.get("export")
    files = manifest.get("files")
    if not isinstance(source, dict) or not isinstance(export, dict) or not isinstance(files, dict):
        raise ContractError("candidate source/export/file provenance is incomplete")
    training_run = _validate_training_run_provenance(source.get("training_run"))
    from scripts.adoption import verify_adopted_checkpoint

    checkpoint_path = root / training_run["run_root"] / training_run["checkpoint_name"]
    student_adoption = verify_adopted_checkpoint(
        checkpoint_path, project_root=root, role="student"
    )
    if source != {
        "checkpoint_name": training_run["checkpoint_name"],
        "checkpoint_sha256": training_run["checkpoint_sha256"],
        "training_receipt_sha256": training_run["receipt_sha256"],
        "run_root": training_run["run_root"],
        "training_run": training_run,
        "student_adoption": student_adoption,
    }:
        raise ContractError("candidate source differs from its sealed training receipt")
    if candidate != candidate_output_path(root, training_run["checkpoint_sha256"]):
        raise ContractError("candidate directory name differs from checkpoint hash")
    export_stack = _validate_training_stack_provenance(export.get("training_stack"))
    if (
        export.get("git_commit") != training_run["source"]["project_git_commit"]
        or export_stack != training_run["source"]["training_stack"]
    ):
        raise ContractError("candidate export/training context mismatch")
    if set(files) != {"policy.pt", "obs_spec.json"}:
        raise ContractError("candidate must bind exactly policy.pt and obs_spec.json")
    for relative, digest in files.items():
        if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
            raise ContractError(f"candidate artifact hash mismatch: {relative}")
    if sha256_file(candidate / "policy.pt") != files["policy.pt"]:
        raise ContractError("candidate artifact hash mismatch: policy.pt")
    obs_spec, _ = _read_json_regular_file_once(
        candidate / "obs_spec.json",
        label="candidate artifact obs_spec.json",
        expected_sha256=files["obs_spec.json"],
    )
    gripper_obs_contract = _validated_gripper_command_obs_spec(obs_spec)
    if manifest.get("observation_contract") != {
        "cfg_hash": obs_spec.get("cfg_hash"),
        "observation_dim": int(obs_spec.get("obs_dim", -1)),
        "layout": obs_spec.get("layout"),
        "gripper_command_state": gripper_obs_contract,
        "action_mode": obs_spec.get("action_scheme"),
        "action_dim": int(obs_spec.get("action_dim", -1)),
    } or obs_spec.get("policy_sha256") != files["policy.pt"]:
        raise ContractError("candidate observation/artifact contract mismatch")
    if manifest.get("rollout_safety_contract") != rollout_safety_contract():
        raise ContractError("candidate rollout-safety contract mismatch")
    configs = manifest.get("configs")
    if configs != training_run["source"]["canonical_configs"]:
        raise ContractError("candidate config hashes differ from training receipt")
    if require_current_context:
        if require_clean_git_commit(root) != export["git_commit"]:
            raise ContractError("current clean Git HEAD differs from candidate")
        if training_stack_provenance(root) != export_stack:
            raise ContractError("current training stack differs from candidate")
        current_configs = _snapshot_release_config_hashes(
            root,
            [root / relative for relative in sorted(configs)],
            expected_hashes=configs,
        )
        if current_configs != configs:
            raise ContractError("current configs differ from candidate")
    return _VerifiedCandidateManifest(
        manifest, manifest_sha256=manifest_sha256
    )


def candidate_receipt_binding(
    candidate_dir: Path | str,
    candidate_manifest: dict,
    *,
    project_root: Path | str,
) -> dict[str, str]:
    """Return the immutable candidate identity embedded by capture/dataset receipts."""
    root = Path(project_root).resolve()
    candidate = Path(candidate_dir).resolve()
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("candidate receipt path escapes the project") from exc
    files = candidate_manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"policy.pt", "obs_spec.json"}:
        raise ContractError("candidate receipt has an invalid artifact set")
    manifest_path = candidate / CANDIDATE_MANIFEST_NAME
    parsed_manifest, manifest_sha256 = _read_json_regular_file_once(
        manifest_path, label="candidate manifest"
    )
    if parsed_manifest != candidate_manifest:
        raise ContractError(
            "candidate manifest bytes differ from the supplied verified manifest"
        )
    verified_sha256 = getattr(
        candidate_manifest, "_verified_manifest_sha256", None
    )
    if verified_sha256 is not None and verified_sha256 != manifest_sha256:
        raise ContractError("candidate manifest changed after verification")
    return {
        "path": relative,
        "manifest_sha256": manifest_sha256,
        "policy_sha256": str(files["policy.pt"]),
        "obs_spec_sha256": str(files["obs_spec.json"]),
    }


def require_canonical_candidate_output(
    project_root: Path | str,
    checkpoint_sha256: str,
) -> Path:
    """Validate the ignored, checkpoint-keyed candidate destination before writing."""
    root = Path(project_root).absolute()
    output = candidate_output_path(root, checkpoint_sha256).absolute()
    if path_contains_link_or_reparse(root) or path_contains_link_or_reparse(output):
        raise ContractError("candidate output path contains a symlink/reparse point")
    candidate_files = ("policy.pt", "obs_spec.json", CANDIDATE_MANIFEST_NAME)
    staging_probe = output.parent / f".{output.name}.staging-ignore-probe"
    for path in (
        *(output / name for name in candidate_files),
        *(staging_probe / name for name in candidate_files),
    ):
        relative = path.relative_to(root.resolve()).as_posix()
        try:
            checked = subprocess.run(
                ["git", "check-ignore", "-q", "--", relative],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContractError(f"cannot verify candidate ignore policy: {exc}") from exc
        if checked.returncode != 0:
            raise ContractError(f"candidate artifact is not ignored: {relative}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if path_contains_link_or_reparse(output.parent.absolute()):
        raise ContractError("candidate output parent became a symlink/reparse point")
    return output.resolve()


def capture_output_path(
    project_root: Path | str,
    candidate_manifest_sha256: str,
    capture_id: str,
) -> Path:
    if re.fullmatch(r"[0-9a-f]{64}", candidate_manifest_sha256) is None:
        raise ContractError("capture path requires an exact candidate-manifest SHA-256")
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}", capture_id) is None:
        raise ContractError("capture id must be UTC YYYYMMDDTHHMMSSZ- plus 8 lowercase hex")
    return (
        Path(project_root).resolve()
        / "Phase3/logs/obs_dump_sim"
        / candidate_manifest_sha256[:16]
        / capture_id
    )


def create_parity_capture(
    *,
    candidate_dir: Path | str,
    project_root: Path | str,
    capture_id: str,
) -> tuple[Path, dict]:
    """Create one new ignored capture root; existing/partial captures are immutable."""
    root = Path(project_root).resolve()
    candidate_raw = Path(candidate_dir).expanduser().absolute()
    if path_contains_link_or_reparse(candidate_raw):
        raise ContractError("capture candidate path contains a symlink/reparse point")
    candidate = candidate_raw.resolve()
    manifest = verify_candidate(candidate, project_root=root, require_current_context=True)
    binding = candidate_receipt_binding(candidate, manifest, project_root=root)
    capture = capture_output_path(root, binding["manifest_sha256"], capture_id)
    if path_contains_link_or_reparse(capture.absolute()):
        raise ContractError("capture destination contains a symlink/reparse point")
    relative_probe = (capture / "ep000/step_0000.npz").relative_to(root).as_posix()
    try:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", relative_probe],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ContractError(f"cannot verify capture ignore policy: {exc}") from exc
    if ignored.returncode != 0:
        raise ContractError(f"capture artifacts are not ignored: {relative_probe}")
    capture.parent.mkdir(parents=True, exist_ok=True)
    if path_contains_link_or_reparse(capture.parent.absolute()):
        raise ContractError("capture parent became a symlink/reparse point")
    try:
        capture.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ContractError("capture id already exists; choose a new immutable capture id") from exc
    owner = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "record_type": CAPTURE_OWNER_TYPE,
        "candidate": binding,
        "candidate_id": binding["manifest_sha256"][:16],
        "capture_id": capture_id,
    }
    owner_path = capture / CAPTURE_OWNER_NAME
    try:
        with owner_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(owner, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if owner_path.is_file() and owner_path.parent == capture:
            owner_path.unlink()
        if capture.is_dir() and not any(capture.iterdir()):
            capture.rmdir()
        raise
    return capture, manifest


def _validate_capture_owner(capture: Path, binding: dict) -> tuple[dict, str]:
    owner_path = capture / CAPTURE_OWNER_NAME
    if is_link_or_reparse(owner_path) or not owner_path.is_file():
        raise ContractError("capture ownership marker is missing or unsafe")
    owner, owner_sha256 = _read_json_regular_file_once(
        owner_path, label="capture ownership marker"
    )
    expected = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "record_type": CAPTURE_OWNER_TYPE,
        "candidate": binding,
        "candidate_id": binding["manifest_sha256"][:16],
        "capture_id": capture.name,
    }
    if owner != expected:
        raise ContractError("capture ownership marker differs from candidate/path identity")
    return owner, owner_sha256


def _read_safe_regular_bytes(path: Path, *, label: str) -> tuple[bytes, os.stat_result]:
    """Read one regular file once so validation and hashing share exact bytes."""

    absolute = path.absolute()
    if path_contains_link_or_reparse(absolute):
        raise ContractError(f"{label} contains a symlink/reparse path")
    try:
        with absolute.open("rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(f"{label} is not a regular file")
            payload = stream.read()
    except OSError as exc:
        raise ContractError(f"cannot read {label}: {path}: {exc}") from exc
    return payload, metadata


def _decode_capture_observation_bytes(
    path: Path,
    *,
    expected_dim: int,
    expected_record: dict | None = None,
) -> tuple[np.ndarray, dict, tuple[int, ...], tuple[int, ...]]:
    """Decode, hash, and validate one capture member from one byte string."""

    payload, metadata = _read_safe_regular_bytes(path, label="capture member")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as decoded:
            required = {
                "obs",
                "raw_arm_command",
                "safety_arm_command",
                "arm_command",
                "command_issued",
                "arm_command_receipt_sequence",
            }
            if not required.issubset(decoded.files):
                raise ContractError(
                    "capture member lacks raw/guarded/applied/receipt command evidence: "
                    f"{path.name}"
                )
            obs = decoded["obs"]
            if obs.ndim != 2 or obs.shape[0] <= 0 or obs.shape[1] != expected_dim:
                raise ContractError(f"capture member observation shape mismatch: {obs.shape}")
            if obs.dtype != np.dtype(np.float32) or not obs.flags.c_contiguous:
                raise ContractError(
                    "capture member observations must be exact C-order float32"
                )
            if not np.isfinite(obs).all():
                raise ContractError("capture member observations are non-finite")
            # Copy before the NPZ owner closes.  This is bounded by the fixed
            # <=100-row episode horizon, never the full parity dataset.
            observation_block = np.array(obs, dtype=np.float32, order="C", copy=True)
            rows = int(obs.shape[0])
            raw_command = np.asarray(decoded["raw_arm_command"])
            safety_command = np.asarray(decoded["safety_arm_command"])
            issued_command = np.asarray(decoded["arm_command"])
            command_issued = np.asarray(decoded["command_issued"])
            receipt_sequence = np.asarray(decoded["arm_command_receipt_sequence"])
            if rows == 1 and raw_command.shape == (ARM_COMMAND_CHANNELS,):
                raw_command = raw_command.reshape(1, ARM_COMMAND_CHANNELS)
            if rows == 1 and safety_command.shape == (ARM_COMMAND_CHANNELS,):
                safety_command = safety_command.reshape(1, ARM_COMMAND_CHANNELS)
            if rows == 1 and issued_command.shape == (ARM_COMMAND_CHANNELS,):
                issued_command = issued_command.reshape(1, ARM_COMMAND_CHANNELS)
            if rows == 1 and command_issued.shape == ():
                command_issued = command_issued.reshape(1)
            if rows == 1 and receipt_sequence.shape == ():
                receipt_sequence = receipt_sequence.reshape(1)
            if raw_command.shape != (rows, ARM_COMMAND_CHANNELS) \
                    or safety_command.shape != (rows, ARM_COMMAND_CHANNELS) \
                    or issued_command.shape != (rows, ARM_COMMAND_CHANNELS):
                raise ContractError(
                    "capture raw/guarded/applied arm command shape does not match obs rows"
                )
            if not np.issubdtype(raw_command.dtype, np.floating) \
                    or not np.issubdtype(safety_command.dtype, np.floating) \
                    or not np.issubdtype(issued_command.dtype, np.floating) \
                    or not np.isfinite(raw_command).all() \
                    or not np.isfinite(safety_command).all() \
                    or not np.isfinite(issued_command).all():
                raise ContractError(
                    "capture raw/guarded/applied arm commands are non-finite/non-numeric"
                )
            if command_issued.shape != (rows,) or command_issued.dtype != np.bool_ \
                    or not bool(command_issued.all()):
                raise ContractError(
                    "capture may contain only successfully issued simulation command rows"
                )
            if receipt_sequence.shape != (rows,) \
                    or receipt_sequence.dtype != np.dtype(np.int64):
                raise ContractError(
                    "capture arm command receipt sequence must be exact int64 scalar/row"
                )
            if bool((receipt_sequence <= 0).any()) or bool(
                (receipt_sequence[1:] <= receipt_sequence[:-1]).any()
            ):
                raise ContractError(
                    "capture arm command receipt sequence must be positive and strictly increasing"
                )
            # Copy the validated same-lineage identifier before the NPZ owner
            # closes.  Clamp counts below are eligible only after this check.
            receipt_sequences = tuple(int(value) for value in receipt_sequence)
            outer_clamped = np.greater(
                np.abs(raw_command - safety_command), ARM_CLAMP_COMPARISON_ATOL
            )
            inner_clamped = np.greater(
                np.abs(safety_command - issued_command), ARM_CLAMP_COMPARISON_ATOL
            )
            clamp_counts = tuple(
                int(value)
                for value in np.logical_or(outer_clamped, inner_clamped).sum(axis=0)
            )
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot decode capture member {path}: {exc}") from exc
    record = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": int(metadata.st_size),
        "rows": rows,
        "observation_dim": expected_dim,
        "command_rows": rows,
        "arm_clamp_counts": list(clamp_counts),
        "receipt_sequence_first": receipt_sequences[0],
        "receipt_sequence_last": receipt_sequences[-1],
        "receipt_sequence_sha256": hashlib.sha256(
            np.asarray(receipt_sequences, dtype=np.int64).tobytes(order="C")
        ).hexdigest(),
    }
    if len(payload) != metadata.st_size:
        raise ContractError("capture member size changed while it was read")
    if expected_record is not None and record != expected_record:
        raise ContractError(f"capture source bytes changed before dataset assembly: {path}")
    return observation_block, record, clamp_counts, receipt_sequences


def read_verified_capture_observation_block(
    path: Path,
    expected_record: dict,
    *,
    expected_dim: int,
) -> np.ndarray:
    """Return the exact observation block bound by one manifest record."""

    block, _, _, _ = _decode_capture_observation_bytes(
        path,
        expected_dim=expected_dim,
        expected_record=expected_record,
    )
    return block


def _read_capture_observation(
    path: Path, *, expected_dim: int
) -> tuple[np.ndarray, dict, tuple[int, ...], tuple[int, ...]]:
    block, record, clamp_counts, receipt_sequences = _decode_capture_observation_bytes(
        path, expected_dim=expected_dim
    )
    return block, record, clamp_counts, receipt_sequences


def _phase3_target_observation_contract(
    candidate_manifest: dict, project_root: Path,
) -> tuple[slice, float]:
    observation = candidate_manifest.get("observation_contract")
    if not isinstance(observation, dict):
        raise ContractError("candidate observation contract is missing")
    dimension = int(observation.get("observation_dim", -1))
    layout = _validated_parity_layout(observation.get("layout"), dimension)
    offset = 0
    target_slice = None
    for item in layout:
        size = int(item["size"])
        if item["name"] == "target_pos":
            if size != 3 or target_slice is not None:
                raise ContractError("candidate target_pos observation block must be unique 3D")
            target_slice = slice(offset, offset + size)
        offset += size
    if target_slice is None:
        raise ContractError("candidate observation layout has no target_pos block")
    common = _load_canonical_yaml_configs(
        project_root,
        candidate_manifest.get("configs"),
        ("configs/common.yaml",),
        label="Phase 3 target observation",
    )["configs/common.yaml"]
    try:
        scale = float(common["proprio"]["tcp_pos_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("canonical target observation scale is missing") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ContractError("canonical target observation scale must be finite and positive")
    return target_slice, scale


def _record_episode_observation_target(
    observed: dict[str, np.ndarray], episode: str, block: np.ndarray, target_slice: slice,
) -> None:
    target_rows = np.asarray(block[:, target_slice], dtype=np.float32)
    first = np.array(target_rows[0], dtype=np.float32, copy=True)
    if not np.array_equal(target_rows, np.broadcast_to(first, target_rows.shape)):
        raise ContractError(f"capture target_pos observation changes within {episode}")
    previous = observed.get(episode)
    if previous is not None and not np.array_equal(previous, first):
        raise ContractError(f"capture target_pos observation differs between files in {episode}")
    observed[episode] = first


def _validate_episode_observation_targets(
    observed: dict[str, np.ndarray], outcome_receipt: dict, *, scale: float,
) -> None:
    rows = outcome_receipt.get("episodes")
    if not isinstance(rows, list) or len(rows) != len(observed):
        raise ContractError("capture target observations differ from outcome episode inventory")
    for outcome in rows:
        episode = f"ep{int(outcome['episode']):03d}"
        actual = observed.get(episode)
        expected = np.asarray(outcome["target_pos"], dtype=np.float32) / np.float32(scale)
        if actual is None or not np.array_equal(actual, expected):
            raise ContractError(
                f"capture target_pos observation is not bound to the reported outcome: {episode}"
            )


def _advance_capture_receipt_sequence(
    previous: int | None,
    sequences: tuple[int, ...],
    *,
    label: str,
) -> int:
    """Bind one decoded member into the capture-wide receipt lineage."""
    if not sequences:
        raise ContractError(f"{label} has no arm command receipt sequence")
    if previous is not None and sequences[0] <= previous:
        raise ContractError(
            "capture arm command receipt sequence must be strictly increasing "
            f"across members: {label}"
        )
    return sequences[-1]


def _scan_one_unsealed_capture(
    capture_dir: Path,
    *,
    expected_dim: int,
    target_slice: slice,
) -> tuple[
    dict[str, dict], dict[str, dict], list[Path], tuple[int, ...], int,
    dict[str, np.ndarray],
]:
    """Scan only one explicit capture root while sealing it; sibling discovery is forbidden."""
    entries = list(capture_dir.rglob("*"))
    if any(is_link_or_reparse(path) for path in entries):
        raise ContractError("capture contains a symlink/reparse path")
    directories = sorted(path for path in entries if path.is_dir())
    if any(path.parent != capture_dir or re.fullmatch(r"ep[0-9]{3,}", path.name) is None
           for path in directories):
        raise ContractError("capture may contain only direct epNNN directories")
    episode_numbers = [int(path.name[2:]) for path in directories]
    if episode_numbers != list(range(len(episode_numbers))):
        raise ContractError("capture episode directories must be contiguous from ep000")
    files = sorted(
        path
        for path in entries
        if path.is_file()
        and path.name not in {CAPTURE_OWNER_NAME, PHASE3_RUNTIME_AUTHORITY_NAME}
    )
    if not files:
        raise ContractError("capture contains no observation dump files")
    records: dict[str, dict] = {}
    episodes: dict[str, dict] = {}
    clamp_counts = np.zeros(ARM_COMMAND_CHANNELS, dtype=np.int64)
    command_denominator = 0
    last_receipt_sequence: int | None = None
    observed_targets: dict[str, np.ndarray] = {}
    for episode in directories:
        members = sorted(path for path in files if path.parent == episode)
        if not members or any(re.fullmatch(r"step_[0-9]{4,}\.npz", path.name) is None
                              for path in members):
            raise ContractError(f"capture episode has an invalid/empty file set: {episode.name}")
        step_numbers = [int(path.stem.split("_")[1]) for path in members]
        if step_numbers != list(range(len(step_numbers))):
            raise ContractError(f"capture steps must be contiguous from zero: {episode.name}")
        episode_rows = 0
        for path in members:
            relative = path.relative_to(capture_dir).as_posix()
            block, record, member_counts, member_sequences = _read_capture_observation(
                path, expected_dim=expected_dim
            )
            last_receipt_sequence = _advance_capture_receipt_sequence(
                last_receipt_sequence, member_sequences, label=relative
            )
            rows = int(block.shape[0])
            _record_episode_observation_target(
                observed_targets, episode.name, block, target_slice
            )
            records[relative] = record
            episode_rows += rows
            command_denominator += rows
            clamp_counts += np.asarray(member_counts, dtype=np.int64)
        episodes[episode.name] = {"file_count": len(members), "rows": episode_rows}
    if len(records) != len(files):
        raise ContractError("capture contains files outside its episode directories")
    return (
        records,
        episodes,
        files,
        tuple(int(value) for value in clamp_counts),
        command_denominator,
        observed_targets,
    )


def phase3_outcome_contract() -> dict:
    """Return the immutable episode-level Phase 3 acceptance contract."""
    return {
        "seed": PHASE3_CAPTURE_SEED,
        "minimum_distinct_episodes": MIN_PHASE3_EPISODES,
        "maximum_steps_per_episode": MAX_PHASE3_EPISODE_STEPS,
        "all_placed_success": True,
        "maximum_safety_trips": 0,
        "maximum_target_infeasible": 0,
        "finite_required": True,
    }


def _validate_phase3_episode_outcomes(
    reported: object,
    *,
    seed: int,
    episodes: dict[str, dict],
    project_root: Path,
    expected_config_hashes: object,
) -> dict:
    """Normalize exact outcomes and derive feasibility from canonical geometry."""
    if type(seed) is not int or seed != PHASE3_CAPTURE_SEED:
        raise ContractError("sealed Phase 3 outcomes require exact seed 45")
    if not isinstance(reported, list):
        raise ContractError("Phase 3 episode outcomes must be an ordered list")
    if len(episodes) < MIN_PHASE3_EPISODES or len(reported) != len(episodes):
        raise ContractError("Phase 3 requires at least 100 distinct episode outcomes")
    geometry = _load_canonical_yaml_configs(
        project_root,
        expected_config_hashes,
        ("configs/common.yaml", "configs/sim.yaml", "configs/safety.yaml"),
        label="Phase 3 target geometry",
    )
    common_yaml = geometry["configs/common.yaml"]
    sim_yaml = geometry["configs/sim.yaml"]
    safety_yaml = geometry["configs/safety.yaml"]
    normalized: list[dict] = []
    expected_keys = {
        "episode", "steps", "placed_success", "tripped", "trip_reason", "target_pos",
    }
    for expected_index, item in enumerate(reported):
        episode_name = f"ep{expected_index:03d}"
        inventory = episodes.get(episode_name)
        if not isinstance(item, dict) or set(item) != expected_keys:
            raise ContractError("Phase 3 episode outcome field set is not exact")
        episode_index = item.get("episode")
        steps = item.get("steps")
        if type(episode_index) is not int or episode_index != expected_index:
            raise ContractError("Phase 3 episode outcomes must be distinct and contiguous")
        if isinstance(steps, bool) or not isinstance(steps, int) \
                or not 1 <= steps <= MAX_PHASE3_EPISODE_STEPS:
            raise ContractError("each Phase 3 episode must contain 1..100 control steps")
        if not isinstance(inventory, dict) or inventory.get("rows") != steps:
            raise ContractError("Phase 3 outcome steps differ from dump episode rows")
        if item.get("placed_success") is not True:
            raise ContractError("every sealed Phase 3 episode must end in placed_success")
        if item.get("tripped") is not False or item.get("trip_reason") != "":
            raise ContractError("sealed Phase 3 outcomes require zero safety trips")
        target = item.get("target_pos")
        if not isinstance(target, (list, tuple)) or len(target) != 3 \
                or any(isinstance(value, bool) for value in target):
            raise ContractError("Phase 3 outcome target must be exact finite XYZ")
        try:
            feasible_target = validate_phase3_target_geometry(
                target,
                common_yaml,
                sim_yaml,
                safety_yaml,
                tag=f"phase3:{episode_name}",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                f"Phase 3 target infeasible by canonical config geometry: {episode_name}: {exc}"
            ) from exc
        normalized.append({
            "episode": expected_index,
            "dump_episode": episode_name,
            "steps": steps,
            "placed_success": True,
            "safety_trip": False,
            "finite": True,
            "target_pos": [float(value) for value in feasible_target],
            # This value is derived above.  It is intentionally not accepted
            # from the caller as a manually editable boolean.
            "target_infeasible": False,
        })
    total_steps = sum(item["steps"] for item in normalized)
    receipt = {
        "schema_version": 1,
        "record_type": "ur5e_phase3_episode_outcomes",
        "contract": phase3_outcome_contract(),
        "seed": PHASE3_CAPTURE_SEED,
        "episode_count": len(normalized),
        "total_steps": total_steps,
        "episodes": normalized,
    }
    return receipt


def _verify_phase3_outcome_receipt(
    value: object,
    *,
    episodes: dict[str, dict],
    project_root: Path,
    expected_config_hashes: object,
) -> dict:
    _validate_embedded_phase3_outcomes(
        value,
        project_root=project_root,
        expected_config_hashes=expected_config_hashes,
    )
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "record_type", "contract", "seed",
        "episode_count", "total_steps", "episodes",
    } or value.get("schema_version") != 1 \
            or value.get("record_type") != "ur5e_phase3_episode_outcomes" \
            or value.get("contract") != phase3_outcome_contract():
        raise ContractError("Phase 3 episode outcome receipt schema/contract mismatch")
    sealed_episodes = value.get("episodes")
    if not isinstance(sealed_episodes, list):
        raise ContractError("Phase 3 episode receipt rows are missing")
    normalized_keys = {
        "episode", "dump_episode", "steps", "placed_success", "safety_trip",
        "finite", "target_pos", "target_infeasible",
    }
    reported: list[dict] = []
    for expected_index, item in enumerate(sealed_episodes):
        episode_index = item.get("episode") if isinstance(item, dict) else None
        if not isinstance(item, dict) or set(item) != normalized_keys \
                or type(episode_index) is not int or episode_index != expected_index \
                or item.get("dump_episode") != f"ep{expected_index:03d}" \
                or item.get("finite") is not True \
                or item.get("target_infeasible") is not False:
            raise ContractError("Phase 3 normalized outcome row was tampered")
        reported.append({
            "episode": item["episode"],
            "steps": item["steps"],
            "placed_success": item["placed_success"],
            "tripped": item["safety_trip"],
            "trip_reason": "",
            "target_pos": item["target_pos"],
        })
    recalculated = _validate_phase3_episode_outcomes(
        reported,
        seed=value.get("seed"),
        episodes=episodes,
        project_root=project_root,
        expected_config_hashes=expected_config_hashes,
    )
    if value != recalculated:
        raise ContractError("Phase 3 episode outcome receipt differs from dump/config evidence")
    return recalculated


def _validate_embedded_phase3_outcomes(
    value: object,
    *,
    project_root: Path,
    expected_config_hashes: object,
) -> dict:
    """Validate embedded outcomes against the exact canonical scene geometry."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "record_type", "contract", "seed",
        "episode_count", "total_steps", "episodes",
    } or value.get("schema_version") != 1 \
            or value.get("record_type") != "ur5e_phase3_episode_outcomes" \
            or value.get("contract") != phase3_outcome_contract() \
            or type(value.get("seed")) is not int \
            or value.get("seed") != PHASE3_CAPTURE_SEED \
            or type(value.get("episode_count")) is not int \
            or type(value.get("total_steps")) is not int:
        raise ContractError("embedded Phase 3 outcome receipt contract mismatch")
    rows = value.get("episodes")
    if not isinstance(rows, list) or len(rows) < MIN_PHASE3_EPISODES \
            or value.get("episode_count") != len(rows):
        raise ContractError("embedded Phase 3 receipt has fewer than 100 episodes")
    geometry = _load_canonical_yaml_configs(
        project_root,
        expected_config_hashes,
        ("configs/common.yaml", "configs/sim.yaml", "configs/safety.yaml"),
        label="embedded Phase 3 target geometry",
    )
    common_yaml = geometry["configs/common.yaml"]
    sim_yaml = geometry["configs/sim.yaml"]
    safety_yaml = geometry["configs/safety.yaml"]
    total = 0
    expected_keys = {
        "episode", "dump_episode", "steps", "placed_success", "safety_trip",
        "finite", "target_pos", "target_infeasible",
    }
    for index, row in enumerate(rows):
        steps = row.get("steps") if isinstance(row, dict) else None
        target = row.get("target_pos") if isinstance(row, dict) else None
        if not isinstance(row, dict) or set(row) != expected_keys \
                or type(row.get("episode")) is not int or row.get("episode") != index \
                or row.get("dump_episode") != f"ep{index:03d}" \
                or isinstance(steps, bool) or not isinstance(steps, int) \
                or not 1 <= steps <= MAX_PHASE3_EPISODE_STEPS \
                or row.get("placed_success") is not True \
                or row.get("safety_trip") is not False \
                or row.get("finite") is not True \
                or row.get("target_infeasible") is not False \
                or not isinstance(target, list) or len(target) != 3 \
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or not math.isfinite(float(item)) for item in target):
            raise ContractError("embedded Phase 3 episode outcome is invalid")
        try:
            canonical_target = validate_phase3_target_geometry(
                target,
                common_yaml,
                sim_yaml,
                safety_yaml,
                tag=f"embedded-phase3:ep{index:03d}",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError(
                f"embedded Phase 3 target is infeasible by canonical config geometry: "
                f"ep{index:03d}: {exc}"
            ) from exc
        if row["target_pos"] != [float(item) for item in canonical_target]:
            raise ContractError("embedded Phase 3 target is not canonically encoded")
        total += steps
    if value.get("total_steps") != total or total < MIN_PARITY_SAMPLES:
        raise ContractError("embedded Phase 3 outcome total is below/mismatches 10,000")
    return value


def seal_parity_capture(
    capture_dir: Path | str,
    *,
    candidate_dir: Path | str,
    project_root: Path | str,
    expected_dim: int,
    rollout_safety: dict,
    seed: int,
    episode_outcomes: list[dict],
) -> Path:
    """Seal one >=10k-row, >=100-episode successful Phase 3 capture."""
    root = Path(project_root).resolve()
    candidate_raw = Path(candidate_dir).expanduser().absolute()
    if path_contains_link_or_reparse(candidate_raw):
        raise ContractError("capture candidate path contains a symlink/reparse point")
    candidate = candidate_raw.resolve()
    candidate_manifest = verify_candidate(
        candidate, project_root=root, require_current_context=True
    )
    binding = candidate_receipt_binding(candidate, candidate_manifest, project_root=root)
    raw = Path(capture_dir).expanduser().absolute()
    if path_contains_link_or_reparse(raw) or not raw.is_dir():
        raise ContractError("capture root is missing or contains a symlink/reparse point")
    capture = raw.resolve()
    expected = capture_output_path(root, binding["manifest_sha256"], capture.name)
    if capture != expected:
        raise ContractError("capture root is outside the exact candidate-specific store")
    _, owner_sha256 = _validate_capture_owner(capture, binding)
    runtime_authority = _phase3_runtime_authority_record(
        capture / PHASE3_RUNTIME_AUTHORITY_NAME,
        candidate_manifest=candidate_manifest,
        project_root=root,
    )
    manifest_path = capture / CAPTURE_MANIFEST_NAME
    if manifest_path.exists():
        raise ContractError("capture is already sealed; refusing overwrite")
    target_slice, target_scale = _phase3_target_observation_contract(
        candidate_manifest, root
    )
    (
        records,
        episodes,
        _,
        clamp_counts,
        command_denominator,
        observed_targets,
    ) = _scan_one_unsealed_capture(
        capture, expected_dim=expected_dim, target_slice=target_slice
    )
    if len(episodes) < MIN_PHASE3_EPISODES:
        raise ContractError("Phase 3 capture requires at least 100 distinct episodes")
    if any(
        isinstance(item.get("rows"), bool)
        or not isinstance(item.get("rows"), int)
        or not 1 <= item["rows"] <= MAX_PHASE3_EPISODE_STEPS
        for item in episodes.values()
    ):
        raise ContractError("each Phase 3 dump episode must contain at most 100 rows")
    total_rows = sum(int(record["rows"]) for record in records.values())
    if total_rows < MIN_PARITY_SAMPLES:
        raise ContractError(
            f"capture has {total_rows} rows; at least {MIN_PARITY_SAMPLES} required"
        )
    if command_denominator != total_rows:
        raise ContractError("capture command denominator differs from observation rows")
    reported_rollout_safety = _validate_rollout_safety(rollout_safety)
    recalculated_rollout_safety = _rollout_safety_from_counts(
        clamp_counts, command_denominator
    )
    if reported_rollout_safety != recalculated_rollout_safety:
        raise ContractError(
            "reported rollout safety differs from raw/guarded/applied command dumps"
        )
    rollout_safety = recalculated_rollout_safety
    rollout_safety_sha256 = _canonical_mapping_sha256(rollout_safety)
    outcome_receipt = _validate_phase3_episode_outcomes(
        episode_outcomes,
        seed=seed,
        episodes=episodes,
        project_root=root,
        expected_config_hashes=candidate_manifest.get("configs"),
    )
    if outcome_receipt["total_steps"] != total_rows:
        raise ContractError("Phase 3 outcome receipt total differs from dump rows")
    _validate_episode_observation_targets(
        observed_targets, outcome_receipt, scale=target_scale
    )
    outcome_receipt_sha256 = _canonical_mapping_sha256(outcome_receipt)
    aggregate = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    relative_root = capture.relative_to(root).as_posix()
    payload = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "record_type": CAPTURE_RECORD_TYPE,
        "candidate": binding,
        "capture": {
            "candidate_id": binding["manifest_sha256"][:16],
            "capture_id": capture.name,
            "root": relative_root,
            "expected_episode_count": len(episodes),
            "expected_file_count": len(records),
            "expected_rows": total_rows,
            "aggregate_sha256": aggregate,
            "owner_sha256": owner_sha256,
            "rollout_safety_sha256": rollout_safety_sha256,
            "episode_outcomes_sha256": outcome_receipt_sha256,
        },
        "observation_contract": candidate_manifest["observation_contract"],
        "runtime_authority": runtime_authority,
        "rollout_safety": rollout_safety,
        "episode_outcomes": outcome_receipt,
        "episodes": episodes,
        "files": records,
    }
    manifest_payload = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{CAPTURE_MANIFEST_NAME}.", suffix=".tmp", dir=capture
    )
    temporary = Path(temporary_name)
    published_identity: tuple[int, int] | None = None
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(manifest_payload)
            stream.flush()
            os.fsync(stream.fileno())
        published_identity = publish_regular_file_no_replace(
            temporary,
            manifest_path,
            quarantine_parent=capture.parent,
        )
        verify_parity_capture(
            manifest_path,
            candidate_dir=candidate,
            project_root=root,
            expected_dim=expected_dim,
            require_current_context=True,
        )
        final_payload, _ = _read_authority_regular_file_once(
            manifest_path,
            label="rechecked capture manifest",
            expected_sha256=manifest_sha256,
        )
        if final_payload != manifest_payload:
            raise ContractError("capture manifest changed during postverification")
    except BaseException as exc:
        if temporary.exists() and temporary.parent == capture:
            temporary.unlink()
        if published_identity is not None:
            try:
                _remove_owned_published_file(
                    manifest_path,
                    expected_identity=published_identity,
                    quarantine_parent=capture.parent,
                )
            except BaseException as cleanup_exc:
                raise ContractError(
                    "capture sealing failed and owned manifest cleanup failed: "
                    f"{cleanup_exc}"
                ) from exc
        raise
    return manifest_path


def verify_parity_capture(
    capture_manifest_path: Path | str,
    *,
    candidate_dir: Path | str,
    project_root: Path | str,
    expected_dim: int,
    require_current_context: bool = True,
) -> tuple[dict, list[Path]]:
    """Verify one manifest-enumerated capture and exclude every sibling capture."""
    root = Path(project_root).resolve()
    candidate_raw = Path(candidate_dir).expanduser().absolute()
    if path_contains_link_or_reparse(candidate_raw):
        raise ContractError("capture candidate path contains a symlink/reparse point")
    candidate = candidate_raw.resolve()
    candidate_manifest = verify_candidate(
        candidate,
        project_root=root,
        require_current_context=require_current_context,
    )
    binding = candidate_receipt_binding(candidate, candidate_manifest, project_root=root)
    raw_manifest = Path(capture_manifest_path).expanduser().absolute()
    if raw_manifest.name != CAPTURE_MANIFEST_NAME \
            or path_contains_link_or_reparse(raw_manifest):
        raise ContractError("capture manifest path is non-canonical or contains a link")
    manifest_path = raw_manifest.resolve()
    capture = manifest_path.parent
    expected = capture_output_path(root, binding["manifest_sha256"], capture.name)
    if capture != expected or not manifest_path.is_file():
        raise ContractError("capture manifest is outside the exact candidate-specific store")
    _, owner_sha256 = _validate_capture_owner(capture, binding)
    manifest_payload, _ = _read_safe_regular_bytes(
        manifest_path, label="capture manifest"
    )
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode capture manifest: {exc}") from exc
    expected_keys = {
        "schema_version", "record_type", "candidate", "capture",
        "observation_contract", "runtime_authority", "rollout_safety", "episode_outcomes",
        "episodes", "files",
    }
    if set(manifest) != expected_keys \
            or manifest.get("schema_version") != CAPTURE_SCHEMA_VERSION \
            or manifest.get("record_type") != CAPTURE_RECORD_TYPE:
        raise ContractError("capture manifest schema/type/field set mismatch")
    if manifest.get("candidate") != binding:
        raise ContractError("capture manifest is not bound to the exact candidate")
    if manifest.get("observation_contract") != candidate_manifest["observation_contract"] \
            or int(candidate_manifest["observation_contract"]["observation_dim"]) != expected_dim:
        raise ContractError("capture observation contract differs from candidate")
    runtime_authority = _phase3_runtime_authority_record(
        capture / PHASE3_RUNTIME_AUTHORITY_NAME,
        candidate_manifest=candidate_manifest,
        project_root=root,
        expected_record=manifest.get("runtime_authority"),
    )
    records = manifest.get("files")
    episodes = manifest.get("episodes")
    if not isinstance(records, dict) or not records or not isinstance(episodes, dict) or not episodes:
        raise ContractError("capture manifest file/episode inventory is empty")
    expected_files = {
        CAPTURE_MANIFEST_NAME,
        CAPTURE_OWNER_NAME,
        PHASE3_RUNTIME_AUTHORITY_NAME,
        *records.keys(),
    }
    entries = list(capture.rglob("*"))
    if any(is_link_or_reparse(path) for path in entries):
        raise ContractError("sealed capture contains a symlink/reparse path")
    actual_files = {path.relative_to(capture).as_posix() for path in entries if path.is_file()}
    if actual_files != expected_files:
        raise ContractError("sealed capture exact file set differs from its manifest")
    recalculated_episodes: dict[str, dict] = {}
    ordered_files: list[Path] = []
    recalculated_clamp_counts = np.zeros(ARM_COMMAND_CHANNELS, dtype=np.int64)
    command_denominator = 0
    last_receipt_sequence: int | None = None
    target_slice, target_scale = _phase3_target_observation_contract(
        candidate_manifest, root
    )
    observed_targets: dict[str, np.ndarray] = {}
    for relative, expected_record in sorted(records.items()):
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or len(pure.parts) != 2
            or re.fullmatch(r"ep[0-9]{3,}", pure.parts[0]) is None
            or re.fullmatch(r"step_[0-9]{4,}\.npz", pure.parts[1]) is None
        ):
            raise ContractError(f"unsafe capture inventory path: {relative!r}")
        path = capture.joinpath(*pure.parts)
        block, actual_record, member_counts, member_sequences = _read_capture_observation(
            path, expected_dim=expected_dim
        )
        last_receipt_sequence = _advance_capture_receipt_sequence(
            last_receipt_sequence, member_sequences, label=relative
        )
        rows = int(block.shape[0])
        if expected_record != actual_record:
            raise ContractError(f"capture member hash/shape drift: {relative}")
        episode = pure.parts[0]
        _record_episode_observation_target(
            observed_targets, episode, block, target_slice
        )
        item = recalculated_episodes.setdefault(episode, {"file_count": 0, "rows": 0})
        item["file_count"] += 1
        item["rows"] += rows
        command_denominator += rows
        recalculated_clamp_counts += np.asarray(member_counts, dtype=np.int64)
        ordered_files.append(path)
    if episodes != dict(sorted(recalculated_episodes.items())):
        raise ContractError("capture episode inventory differs from its file inventory")
    episode_numbers = [int(name[2:]) for name in sorted(episodes)]
    if episode_numbers != list(range(len(episode_numbers))):
        raise ContractError("sealed capture episodes are not contiguous from ep000")
    if len(episodes) < MIN_PHASE3_EPISODES:
        raise ContractError("sealed Phase 3 capture has fewer than 100 distinct episodes")
    if any(not 1 <= int(item["rows"]) <= MAX_PHASE3_EPISODE_STEPS
           for item in episodes.values()):
        raise ContractError("sealed Phase 3 episode exceeds the 100-step horizon")
    total_rows = sum(int(record["rows"]) for record in records.values())
    if total_rows < MIN_PARITY_SAMPLES:
        raise ContractError("sealed capture has fewer than 10,000 observation rows")
    if command_denominator != total_rows:
        raise ContractError("sealed capture command denominator differs from observation rows")
    rollout_safety = _rollout_safety_from_counts(
        recalculated_clamp_counts, command_denominator
    )
    if manifest.get("rollout_safety") != rollout_safety:
        raise ContractError(
            "capture rollout safety differs from raw/guarded/applied command dumps"
        )
    rollout_safety_sha256 = _canonical_mapping_sha256(rollout_safety)
    outcome_receipt = _verify_phase3_outcome_receipt(
        manifest.get("episode_outcomes"),
        episodes=episodes,
        project_root=root,
        expected_config_hashes=candidate_manifest.get("configs"),
    )
    if outcome_receipt["total_steps"] != total_rows:
        raise ContractError("capture outcome receipt total differs from observation rows")
    _validate_episode_observation_targets(
        observed_targets, outcome_receipt, scale=target_scale
    )
    outcome_receipt_sha256 = _canonical_mapping_sha256(outcome_receipt)
    aggregate = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_capture = {
        "candidate_id": binding["manifest_sha256"][:16],
        "capture_id": capture.name,
        "root": capture.relative_to(root).as_posix(),
        "expected_episode_count": len(episodes),
        "expected_file_count": len(records),
        "expected_rows": total_rows,
        "aggregate_sha256": aggregate,
        "owner_sha256": owner_sha256,
        "rollout_safety_sha256": rollout_safety_sha256,
        "episode_outcomes_sha256": outcome_receipt_sha256,
    }
    if manifest.get("capture") != expected_capture:
        raise ContractError("capture summary differs from its exact file inventory")
    return ({
        "root": expected_capture["root"],
        "capture_id": capture.name,
        "capture_manifest_sha256": manifest_sha256,
        "candidate_manifest_sha256": binding["manifest_sha256"],
        "episode_count": len(episodes),
        "file_count": len(records),
        "aggregate_sha256": aggregate,
        "total_rows": total_rows,
        "observation_dim": expected_dim,
        "rollout_safety": rollout_safety,
        "rollout_safety_sha256": rollout_safety_sha256,
        "seed": outcome_receipt["seed"],
        "episode_outcomes": outcome_receipt,
        "episode_outcomes_sha256": outcome_receipt_sha256,
        "runtime_authority_sha256": runtime_authority["sha256"],
        "runtime_authority_bytes": runtime_authority["bytes"],
        "runtime_authority": runtime_authority["receipt"],
    }, ordered_files)


def _directory_identity(path: Path, *, label: str) -> tuple[int, int]:
    """Return a stable directory identity used to constrain failure cleanup."""

    if path_contains_link_or_reparse(path) or is_link_or_reparse(path):
        raise ContractError(f"{label} contains a symlink/reparse point")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise ContractError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode) or int(metadata.st_ino) <= 0:
        raise ContractError(f"{label} has no trustworthy regular-directory identity")
    return int(metadata.st_dev), int(metadata.st_ino)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename one directory without authorizing an overwrite.

    The release workflow targets Windows, where ``os.rename`` is an atomic
    no-replace operation.  Both pre- and postconditions are still checked so a
    destination that appears concurrently is never accepted as our artifact.
    """

    if destination.exists() or is_link_or_reparse(destination):
        raise ContractError(f"artifact output already exists; refusing overwrite: {destination}")
    try:
        os.rename(source, destination)
    except FileExistsError as exc:
        raise ContractError(
            f"artifact output appeared during publish; refusing overwrite: {destination}"
        ) from exc
    except OSError as exc:
        raise ContractError(f"atomic artifact directory publish failed: {exc}") from exc


def _remove_owned_published_directory(
    output: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Quarantine only the exact directory just renamed by this publisher."""

    if not output.exists() and not is_link_or_reparse(output):
        return
    current = _directory_identity(output, label="published artifact cleanup target")
    if current != expected_identity:
        raise ContractError(
            "published artifact identity changed; refusing to delete an unowned directory"
        )
    quarantine_prefix = (
        f".{output.name}.staging-failed-"
        if output.name == "exported"
        else f".{output.name}.failed-"
    )
    quarantine = output.with_name(
        f"{quarantine_prefix}{secrets.token_hex(16)}"
    )
    try:
        os.rename(output, quarantine)
    except OSError as exc:
        raise ContractError(
            f"cannot quarantine owned published directory: {exc}"
        ) from exc
    moved = _directory_identity(
        quarantine, label="quarantined published artifact"
    )
    if moved != expected_identity:
        if not output.exists() and not is_link_or_reparse(output):
            try:
                os.rename(quarantine, output)
            except OSError:
                pass
        raise ContractError(
            "publication cleanup raced with an unowned directory; nothing was deleted"
        )


def _verify_staged_manifest_artifacts(
    staging: Path,
    *,
    manifest_name: str,
    expected_manifest_sha256: str,
) -> dict:
    """Bind one staging tree to exact manifest bytes and artifact snapshots."""

    if re.fullmatch(r"[0-9a-f]{64}", str(expected_manifest_sha256)) is None:
        raise ContractError("publication requires an exact expected manifest SHA-256")
    manifest, digest = _read_json_regular_file_once(
        staging / manifest_name,
        label=f"staged {manifest_name}",
        expected_sha256=expected_manifest_sha256,
    )
    if digest != expected_manifest_sha256 or not isinstance(manifest, dict):
        raise ContractError("staged artifact manifest differs from its publication authority")
    files = manifest.get("files")
    expected_artifacts = {"policy.pt", "obs_spec.json"}
    if not isinstance(files, dict) or set(files) != expected_artifacts:
        raise ContractError("staged artifact manifest has an invalid policy/spec file set")
    expected_tree = {*expected_artifacts, manifest_name}
    entries = list(staging.rglob("*"))
    if any(is_link_or_reparse(path) for path in entries) \
            or any(path.is_dir() for path in staging.iterdir()):
        raise ContractError("staged artifact tree contains a directory or reparse point")
    actual = {path.relative_to(staging).as_posix() for path in entries if path.is_file()}
    if actual != expected_tree:
        raise ContractError("staged artifact tree differs from its exact manifest file set")
    for relative in sorted(expected_artifacts):
        expected = files.get(relative)
        if re.fullmatch(r"[0-9a-f]{64}", str(expected or "")) is None:
            raise ContractError(f"staged artifact manifest hash is invalid: {relative}")
        _read_authority_regular_file_once(
            staging / relative,
            label=f"staged artifact {relative}",
            expected_sha256=str(expected),
        )
    return manifest


def publish_candidate_staging(
    staging_dir: Path | str,
    output_dir: Path | str,
    *,
    project_root: Path | str,
    expected_manifest_sha256: str,
) -> Path:
    """Preverify, atomically publish, and postverify one exact candidate."""

    staging, output = _require_safe_release_staging(staging_dir, output_dir)
    identity = _directory_identity(staging, label="candidate staging")
    _verify_staged_manifest_artifacts(
        staging,
        manifest_name=CANDIDATE_MANIFEST_NAME,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if _directory_identity(staging, label="candidate staging") != identity:
        raise ContractError("candidate staging directory changed during prepublication verify")
    _rename_directory_no_replace(staging, output)
    try:
        if _directory_identity(output, label="published candidate") != identity:
            raise ContractError("published candidate is not the renamed staging directory")
        published = verify_candidate(
            output,
            project_root=project_root,
            require_current_context=True,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        final_manifest = _verify_staged_manifest_artifacts(
            output,
            manifest_name=CANDIDATE_MANIFEST_NAME,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if getattr(published, "_verified_manifest_sha256", None) \
                != expected_manifest_sha256 or published != final_manifest:
            raise ContractError("published candidate manifest digest differs from authority")
    except BaseException as exc:
        try:
            _remove_owned_published_directory(output, expected_identity=identity)
        except BaseException as cleanup_exc:
            raise ContractError(
                f"published candidate verification failed and owned cleanup failed: {cleanup_exc}"
            ) from exc
        raise
    return output


def require_canonical_release_output(
    output_dir: Path | str,
    project_root: Path | str,
) -> Path:
    """Allow export only to the repository's ignored ``exported`` artifact root."""
    root_raw = Path(project_root).absolute()
    output_raw = Path(output_dir).expanduser().absolute()
    canonical_raw = (root_raw / "exported").absolute()
    if path_contains_link_or_reparse(root_raw) or path_contains_link_or_reparse(output_raw):
        raise ContractError("project/release output path must not contain symlinks/reparse points")
    root = root_raw.resolve()
    output = output_raw.resolve()
    canonical = canonical_raw.resolve()
    if output != canonical or canonical.parent != root:
        raise ContractError(f"release output must be exactly {canonical}")
    for relative in (
        "exported/policy.pt",
        "exported/obs_spec.json",
        f"exported/{RELEASE_MANIFEST_NAME}",
        ".exported.staging-ignore-probe/policy.pt",
        ".exported.staging-ignore-probe/obs_spec.json",
        f".exported.staging-ignore-probe/{RELEASE_MANIFEST_NAME}",
    ):
        try:
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", "--", relative],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ContractError(f"cannot verify release artifact ignore policy: {exc}") from exc
        if ignored.returncode != 0:
            raise ContractError(
                f"release artifact is not ignored and would pollute source control: {relative}"
            )
    return canonical


def create_release_staging(output_dir: Path | str) -> Path:
    """Create a sibling staging directory without touching an accepted release."""
    output_raw = Path(output_dir).absolute()
    if path_contains_link_or_reparse(output_raw):
        raise ContractError(f"release output path must not contain symlinks/reparse points: {output_raw}")
    output = output_raw.resolve()
    if output_raw.exists() or is_link_or_reparse(output_raw):
        raise ContractError(f"release output must be absent: {output_raw}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)).absolute()


def _require_safe_release_staging(staging_dir: Path | str, output_dir: Path | str) -> tuple[Path, Path]:
    staging_raw = Path(staging_dir).absolute()
    output_raw = Path(output_dir).absolute()
    if (
        path_contains_link_or_reparse(staging_raw)
        or path_contains_link_or_reparse(output_raw)
    ):
        raise ContractError("release staging/output path must not contain symlinks")
    staging = staging_raw.resolve()
    output = output_raw.resolve()
    if (
        staging.parent != output.parent
        or not staging.name.startswith(f".{output.name}.staging-")
        or staging == output
    ):
        raise ContractError("release staging is not a validated sibling temporary directory")
    return staging, output


def cleanup_release_staging(staging_dir: Path | str, output_dir: Path | str) -> None:
    staging, _ = _require_safe_release_staging(staging_dir, output_dir)
    if staging.exists():
        shutil.rmtree(staging)


def publish_release_staging(
    staging_dir: Path | str,
    output_dir: Path | str,
    *,
    config_root: Path | str,
    expected_manifest_sha256: str,
    expected_obs_dim: int | None = None,
) -> Path:
    """Preverify, atomically publish, and postverify one exact release."""

    staging, output = _require_safe_release_staging(staging_dir, output_dir)
    identity = _directory_identity(staging, label="release staging")
    _verify_staged_manifest_artifacts(
        staging,
        manifest_name=RELEASE_MANIFEST_NAME,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    staged = verify_release(
        staging,
        config_root=config_root,
        expected_obs_dim=expected_obs_dim,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if getattr(staged, "_verified_manifest_sha256", None) != expected_manifest_sha256:
        raise ContractError("staged release manifest digest differs from authority")
    final_staged_manifest = _verify_staged_manifest_artifacts(
        staging,
        manifest_name=RELEASE_MANIFEST_NAME,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if staged != final_staged_manifest:
        raise ContractError("staged release changed during semantic verification")
    if _directory_identity(staging, label="release staging") != identity:
        raise ContractError("release staging directory changed during prepublication verify")
    _rename_directory_no_replace(staging, output)
    try:
        if _directory_identity(output, label="published release") != identity:
            raise ContractError("published release is not the renamed staging directory")
        published = verify_release(
            output,
            config_root=config_root,
            expected_obs_dim=expected_obs_dim,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        final_manifest = _verify_staged_manifest_artifacts(
            output,
            manifest_name=RELEASE_MANIFEST_NAME,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        if getattr(published, "_verified_manifest_sha256", None) \
                != expected_manifest_sha256 \
                or published != staged \
                or published != final_manifest:
            raise ContractError("published release differs from verified staging authority")
    except BaseException as exc:
        try:
            _remove_owned_published_directory(output, expected_identity=identity)
        except BaseException as cleanup_exc:
            raise ContractError(
                f"published release verification failed and owned cleanup failed: {cleanup_exc}"
            ) from exc
        raise
    return output


def verify_release(
    release_dir: Path | str,
    *,
    config_root: Path | str | None = None,
    benchmark: bool = False,
    expected_obs_dim: int | None = None,
    expected_action_dim: int = 7,
    expected_manifest_sha256: str | None = None,
) -> dict:
    root_raw = Path(release_dir).absolute()
    if path_contains_link_or_reparse(root_raw):
        raise ContractError("release path must not contain symlinks/reparse points")
    root = root_raw.resolve()
    manifest_path = root / RELEASE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ContractError(f"release manifest missing: {manifest_path}")
    allowed_files = {"policy.pt", "obs_spec.json", RELEASE_MANIFEST_NAME}
    entries = list(root.rglob("*"))
    if any(is_link_or_reparse(path) for path in entries):
        raise ContractError("release contains a symlink/reparse path")
    actual_files = {path.relative_to(root).as_posix() for path in entries if path.is_file()}
    if actual_files != allowed_files:
        raise ContractError(
            f"release file set mismatch: expected={sorted(allowed_files)}, "
            f"actual={sorted(actual_files)}"
        )
    manifest, release_manifest_sha256 = _read_json_regular_file_once(
        manifest_path,
        label="release manifest",
        expected_sha256=expected_manifest_sha256,
    )
    if not isinstance(manifest, dict):
        raise ContractError("release manifest must be an object")
    manifest = _VerifiedReleaseManifest(
        manifest, manifest_sha256=release_manifest_sha256
    )
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ContractError("release manifest schema mismatch")
    if manifest.get("model_role") != "student" or manifest.get("actor_kind") != "student":
        raise ContractError("release manifest is not a student policy")
    source = manifest.get("source", {})
    if (
        not isinstance(source, dict)
        or re.fullmatch(r"[0-9a-f]{64}", str(source.get("checkpoint_sha256", ""))) is None
        or not str(source.get("checkpoint_name", ""))
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit", ""))) is None
    ):
        raise ContractError("release source/checkpoint provenance is incomplete")
    training_run = _validate_training_run_provenance(source.get("training_run"))
    expected_source_projection = {
        "checkpoint_name": training_run["checkpoint_name"],
        "checkpoint_sha256": training_run["checkpoint_sha256"],
        "git_commit": training_run["source"]["project_git_commit"],
        "training_run": training_run,
    }
    if source != expected_source_projection:
        raise ContractError("release training run/checkpoint provenance mismatch")
    environment = manifest.get("environment", {})
    environment = _validate_environment(environment)
    for name, expected in TRAINING_STACK.items():
        if environment.get(name) != expected:
            raise ContractError(
                f"release training stack {name} must be {expected}; got {environment.get(name)!r}"
            )
    training_stack = _validate_training_stack_provenance(manifest.get("training_stack"))
    export = manifest.get("export")
    if not isinstance(export, dict) or export.get("git_commit") != source.get("git_commit"):
        raise ContractError("release export/training Git provenance mismatch")
    export_stack = _validate_training_stack_provenance(export.get("training_stack"))
    if training_run["source"]["project_git_commit"] != source.get("git_commit"):
        raise ContractError("release source differs from sealed training Git commit")
    if training_run["source"]["training_stack"] != training_stack or export_stack != training_stack:
        raise ContractError("release training/export stack provenance mismatch")
    latency_contract = manifest.get("latency_contract", {})
    raw_limit = latency_contract.get("max_inference_p99_ms")
    if (isinstance(raw_limit, bool) or not isinstance(raw_limit, (int, float))
            or not math.isfinite(float(raw_limit))):
        raise ContractError("release inference p99 contract must be finite numeric")
    limit = float(raw_limit)
    if limit <= 0 or limit > 10.0:
        raise ContractError("release inference p99 contract must be in (0,10] ms")
    if not isinstance(manifest.get("configs"), dict) or not manifest["configs"]:
        raise ContractError("release must hash its configuration set")
    parity = manifest.get("parity", {})
    latency_host = latency_contract.get("export_host", {})
    checked_values = {
        "parity.samples": parity.get("samples"),
        "parity.max_abs_error": parity.get("max_abs_error"),
        "parity.mean_abs_error": parity.get("mean_abs_error"),
        "parity.threshold": parity.get("threshold"),
        "latency.samples": latency_host.get("samples"),
        "latency.p99_ms": latency_host.get("p99_ms"),
    }
    for name, value in checked_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) < 0:
            raise ContractError(f"release {name} must be finite and non-negative")
    if not isinstance(parity["samples"], int) or not isinstance(latency_host["samples"], int):
        raise ContractError("release parity/latency sample counts must be integers")
    if int(parity["samples"]) < MIN_PARITY_SAMPLES:
        raise ContractError("release parity evidence has too few samples")
    if (float(parity["max_abs_error"]) > MAX_PARITY_ERROR
            or float(parity["mean_abs_error"]) > float(parity["max_abs_error"])):
        raise ContractError("release parity evidence exceeds tolerance")
    if float(parity["threshold"]) != MAX_PARITY_ERROR:
        raise ContractError("release parity threshold differs from contract")
    if int(latency_host["samples"]) < MIN_LATENCY_SAMPLES \
            or float(latency_host["p99_ms"]) > limit:
        raise ContractError("release export-host inference p99 exceeds contract")
    if set(manifest.get("files", {})) != {"policy.pt", "obs_spec.json"}:
        raise ContractError("release manifest must bind exactly policy.pt and obs_spec.json")
    for relative, expected in manifest["files"].items():
        if re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
            raise ContractError(f"release file hash mismatch: {relative}")
    policy_payload, _ = _read_authority_regular_file_once(
        root / "policy.pt",
        label="release artifact policy.pt",
        expected_sha256=manifest["files"]["policy.pt"],
    )
    obs_spec, _ = _read_json_regular_file_once(
        root / "obs_spec.json",
        label="release artifact obs_spec.json",
        expected_sha256=manifest["files"]["obs_spec.json"],
    )
    if config_root is None:
        raise ContractError(
            "release verification requires the canonical config root for target geometry"
        )
    base = Path(config_root).resolve()
    if set(manifest["configs"]) != REQUIRED_RELEASE_CONFIGS:
        raise ContractError("release config set is incomplete or contains unexpected files")
    current_configs = _snapshot_release_config_hashes(
        base,
        [base / relative for relative in sorted(manifest["configs"])],
        expected_hashes=manifest["configs"],
    )
    if current_configs != manifest["configs"]:
        raise ContractError("release config hash mismatch")
    validate_release_calibrations(base, expected_hashes=manifest["configs"])
    if training_run["source"]["canonical_configs"] != manifest["configs"]:
        raise ContractError("release configs differ from sealed training config hashes")
    policy = torch.jit.load(io.BytesIO(policy_payload), map_location="cpu")
    policy.eval()
    gripper_obs_contract = _validated_gripper_command_obs_spec(obs_spec)
    if (
        obs_spec.get("model_role") != "student"
        or obs_spec.get("empirical_normalization") is not False
        or obs_spec.get("policy_sha256") != manifest["files"]["policy.pt"]
        or obs_spec.get("source_checkpoint") != source["checkpoint_name"]
    ):
        raise ContractError("obs_spec student/normalization/policy/checkpoint provenance mismatch")
    obs_dim = int(obs_spec.get("obs_dim", -1))
    action_dim = int(obs_spec.get("action_dim", -1))
    _validate_parity_provenance(
        parity,
        training_run=training_run,
        observation_dim=obs_dim,
        layout=obs_spec.get("layout"),
        obs_cfg_hash=str(obs_spec.get("cfg_hash", "")),
        policy_sha256=manifest["files"]["policy.pt"],
        obs_spec_sha256=manifest["files"]["obs_spec.json"],
        config_root=base,
    )
    runtime_contract = manifest.get("runtime_contract", {})
    if runtime_contract != {
        "observation_dim": obs_dim,
        "observation_layout": obs_spec.get("layout"),
        "gripper_command_state": gripper_obs_contract,
        "action_mode": obs_spec.get("action_scheme"),
        "action_dim": action_dim,
        "control_hz": 20.0,
        "horizon_steps": 100,
        "horizon_seconds": 5.0,
    }:
        raise ContractError("release runtime/observation/action/horizon contract mismatch")
    if action_dim != 7 or obs_spec.get("action_scheme") != "ik_rel":
        raise ContractError("release must be a 7D ik_rel student policy")
    if expected_obs_dim is not None and obs_dim != expected_obs_dim:
        raise ContractError(f"release obs_dim mismatch: {obs_dim} != {expected_obs_dim}")
    if action_dim != expected_action_dim:
        raise ContractError(f"release action_dim mismatch: {action_dim} != {expected_action_dim}")
    with torch.inference_mode():
        output = policy(torch.zeros(1, obs_dim))
    if tuple(output.shape) != (1, action_dim) or not torch.isfinite(output).all():
        raise ContractError("release policy smoke output is invalid")
    if benchmark:
        # Deployment-host timing is a single-observation online latency contract.
        # Reusing one shape-valid smoke observation is intentional here, but the
        # export-host release evidence above is still measured on >=10k actual rows.
        measured = benchmark_policy(
            policy,
            torch.zeros(1, obs_dim),
            samples=MIN_LATENCY_SAMPLES,
            allow_repeated_rows=True,
        )
        if float(measured["p99_ms"]) > limit:
            raise ContractError(
                f"deployment-host inference p99 {measured['p99_ms']:.3f} ms exceeds {limit:.3f} ms"
            )
        manifest["deployment_host_latency"] = measured
    return manifest


def git_commit_or_unknown(root: Path) -> str:
    import subprocess

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if dirty:
            return "unknown"
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else "unknown"
    except Exception:
        return "unknown"


def require_clean_git_commit(root: Path) -> str:
    """Return a clean exact source commit or reject release generation."""
    commit = git_commit_or_unknown(root)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ContractError(
            "release export requires a clean Git worktree and exact 40-hex HEAD"
        )
    return commit
