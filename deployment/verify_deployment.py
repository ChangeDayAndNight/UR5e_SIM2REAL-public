#!/usr/bin/env python
"""Verify an immutable deployment bundle and collect receive-only HIL evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config_io import (  # noqa: E402
    build_action_cfg,
    build_preprocess_cfg,
    build_safety_cfg,
    load_yaml,
    load_yaml_snapshot,
)
from common.file_snapshot import (  # noqa: E402
    FileSnapshotError,
    RegularFileSnapshot,
    read_regular_file_snapshot,
)
from common.obs_preprocess import obs_dim  # noqa: E402
from common.policy_io import PolicyRunner  # noqa: E402
from common.safety import SafetyGuard  # noqa: E402
from deployment.contracts import (  # noqa: E402
    ContractError,
    is_link_or_reparse,
    load_real_config_snapshot,
    path_contains_link_or_reparse,
    real_contract_sha256,
    require_bundle_local_real_config,
    sha256_file,
)
from deployment.release import verify_release  # noqa: E402

_LOCK_LINE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)(?:\s*\\)?$")
_MIN_NO_MOTION_SECONDS = 600.0
_MIN_NO_MOTION_CYCLES = 10_000


class _VerifiedBundleManifest(dict):
    """Schema-compatible bundle mapping carrying exact parsed-byte anchors."""

    def __init__(
        self,
        value: dict,
        *,
        manifest_sha256: str,
        release_manifest: dict,
        release_manifest_sha256: str,
    ) -> None:
        super().__init__(value)
        self._verified_manifest_sha256 = manifest_sha256
        self._verified_release_manifest = release_manifest
        self._verified_release_manifest_sha256 = release_manifest_sha256


def _snapshot(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> RegularFileSnapshot:
    """Capture one exact authority payload and expose ContractError at this layer."""
    if path_contains_link_or_reparse(path):
        raise ContractError(f"{label} contains a symlink/reparse path")
    try:
        return read_regular_file_snapshot(
            path, label=label, expected_sha256=expected_sha256
        )
    except FileSnapshotError as exc:
        raise ContractError(str(exc)) from exc


def _json_snapshot(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[object, RegularFileSnapshot]:
    snapshot = _snapshot(path, label=label, expected_sha256=expected_sha256)
    try:
        value = json.loads(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode {label}: {exc}") from exc
    return value, snapshot


def _verified_bundle_manifest_sha256(manifest: dict) -> str:
    digest = getattr(manifest, "_verified_manifest_sha256", None)
    if re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None:
        raise ContractError("bundle manifest has no exact verified byte hash")
    return str(digest)


def _verified_release_manifest_sha256(release: dict) -> str:
    digest = getattr(release, "_verified_manifest_sha256", None)
    if re.fullmatch(r"[0-9a-f]{64}", str(digest or "")) is None:
        raise ContractError("release manifest has no exact verified byte hash")
    return str(digest)


class RobotProbe(Protocol):
    def sample(self): ...
    @property
    def controller_version(self) -> str: ...
    def close(self) -> None: ...


class CameraProbe(Protocol):
    def sample(self): ...
    @property
    def serial(self) -> str: ...
    @property
    def firmware_version(self) -> str: ...
    def close(self) -> None: ...


class ReadOnlyRobotProbe:
    """Composition facade that intentionally exposes no motion/IO method."""

    __slots__ = ("__adapter",)

    def __init__(self, real_cfg: dict, tcp_offset_path: Path) -> None:
        from real.real_robot import RealRobot

        self.__adapter = RealRobot(real_cfg, tcp_offset_path, receive_only=True)

    def sample(self):
        return self.__adapter.get_state(), self.__adapter.get_health_snapshot()

    @property
    def controller_version(self) -> str:
        return self.__adapter.get_controller_version()

    def close(self) -> None:
        self.__adapter.close()


class ReadOnlyCameraProbe:
    """Read-only RealSense facade; only frame acquisition and close are public."""

    __slots__ = ("__adapter",)

    def __init__(self, camera_cfg: dict, *, enable_rgb: bool) -> None:
        from real.real_camera import RealCamera

        self.__adapter = RealCamera(camera_cfg, enable_rgb=enable_rgb)
        self.__adapter.start()

    def sample(self):
        return self.__adapter.get_frame()

    @property
    def serial(self) -> str:
        return self.__adapter.actual_serial

    @property
    def firmware_version(self) -> str:
        return self.__adapter.firmware_version

    def close(self) -> None:
        self.__adapter.stop()


def _resolve_bundle_root(bundle_root: Path | str) -> Path:
    root_raw = Path(bundle_root).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw):
        raise ContractError("bundle root must not contain symlinks/reparse points")
    return root_raw.resolve()


def verify_bundle_manifest(bundle_root: Path) -> dict:
    bundle_root = _resolve_bundle_root(bundle_root)
    path = bundle_root / "bundle_manifest.json"
    if is_link_or_reparse(path):
        raise ContractError("bundle_manifest.json must not be a symlink")
    if not path.is_file():
        raise ContractError("bundle_manifest.json is missing; run deployment/build_bundle.py")
    manifest, manifest_snapshot = _json_snapshot(path, label="bundle manifest")
    if not isinstance(manifest, dict):
        raise ContractError("invalid bundle manifest")
    if manifest.get("schema_version") != 2 or not isinstance(manifest.get("files"), dict):
        raise ContractError("invalid bundle manifest")
    source_git_commit = manifest.get("source_git_commit")
    if re.fullmatch(r"[0-9a-f]{40}", str(source_git_commit or "")) is None:
        raise ContractError("bundle manifest has no exact source Git commit")
    expected = manifest["files"]
    verified_sizes: dict[str, int] = {}
    for relative, expected_hash in expected.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContractError(f"unsafe bundle manifest path: {relative!r}")
        if re.fullmatch(r"[0-9a-f]{64}", str(expected_hash or "")) is None:
            raise ContractError(f"invalid bundle file hash: {relative!r}")

    def mutable(path_: Path) -> bool:
        relative = path_.relative_to(bundle_root).parts
        return (
            (len(relative) >= 2 and relative[:2] == ("deployment", "evidence"))
            or relative == ("configs", "real.local.yaml")
        )

    entries = list(bundle_root.rglob("*"))
    if any(is_link_or_reparse(item) for item in entries):
        raise ContractError("bundle contains a symlink/reparse path")
    actual = {
        item.relative_to(bundle_root).as_posix()
        for item in entries
        if item.is_file() and item.name != "bundle_manifest.json" and not mutable(item)
    }
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ContractError(f"bundle file set mismatch: missing={missing[:5]}, extra={extra[:5]}")
    release_relative = "exported/release_manifest.json"
    release_expected_sha256 = expected.get(release_relative)
    for relative, expected_hash in expected.items():
        if relative == release_relative:
            # Parsed below from one exact manifest-anchored payload.
            continue
        bound = bundle_root / relative
        if is_link_or_reparse(bound) or not bound.is_file():
            raise ContractError(f"bundle hash mismatch: {relative}")
        try:
            snapshot = _snapshot(
                bound,
                label=f"bundle file {relative}",
                expected_sha256=expected_hash,
            )
            verified_sizes[relative] = snapshot.size
        except ContractError as exc:
            raise ContractError(f"bundle hash mismatch: {relative}") from exc
    if release_expected_sha256 is None:
        raise ContractError("bundle manifest does not bind embedded release manifest")
    release_manifest, release_snapshot = _json_snapshot(
        bundle_root / release_relative,
        label="bundled release manifest",
        expected_sha256=release_expected_sha256,
    )
    if not isinstance(release_manifest, dict):
        raise ContractError("bundled release manifest must be an object")
    embedded_files = release_manifest.get("files")
    if embedded_files != {
        "policy.pt": expected.get("exported/policy.pt"),
        "obs_spec.json": expected.get("exported/obs_spec.json"),
    }:
        raise ContractError("bundle policy/spec hashes differ from embedded release manifest")
    wheelhouse = manifest.get("wheelhouse")
    if not isinstance(wheelhouse, dict) or not wheelhouse.get("files"):
        raise ContractError("bundle manifest has no hashed wheelhouse provenance")
    wheel_files = wheelhouse["files"]
    if not isinstance(wheel_files, dict):
        raise ContractError("bundle wheelhouse file provenance is malformed")
    expected_wheel_paths = {
        relative for relative in expected
        if relative.startswith("deployment/wheelhouse/")
    }
    recorded_wheel_paths: set[str] = set()
    for filename, record in wheel_files.items():
        relative = f"deployment/wheelhouse/{filename}"
        if (
            Path(str(filename)).name != filename
            or not str(filename).lower().endswith(".whl")
            or not isinstance(record, dict)
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or record.get("sha256") != expected.get(relative)
            or record["bytes"] != verified_sizes.get(relative)
        ):
            raise ContractError("bundle wheelhouse file provenance is malformed")
        recorded_wheel_paths.add(relative)
    if recorded_wheel_paths != expected_wheel_paths:
        raise ContractError("bundle wheelhouse file provenance is incomplete")
    aggregate = hashlib.sha256(
        json.dumps(wheel_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if wheelhouse.get("aggregate_sha256") != aggregate:
        raise ContractError("bundle wheelhouse aggregate hash is inconsistent")
    if (
        wheelhouse.get("provenance_manifest_sha256")
        != expected.get("deployment/wheelhouse.manifest.json")
        or wheelhouse.get("lock_sha256")
        != expected.get("deployment/requirements-win11.lock.txt")
    ):
        raise ContractError("bundle wheelhouse provenance/lock hashes are inconsistent")
    if release_manifest.get("source", {}).get("git_commit") != source_git_commit:
        raise ContractError("bundle source Git commit differs from embedded release provenance")
    return _VerifiedBundleManifest(
        manifest,
        manifest_sha256=manifest_snapshot.sha256,
        release_manifest=release_manifest,
        release_manifest_sha256=release_snapshot.sha256,
    )


def _parse_lock_payload(payload: bytes) -> dict[str, str]:
    """Parse an already anchored pip-tools ``--require-hashes`` payload."""
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ContractError(f"cannot decode dependency lock: {exc}") from exc
    if not any(line.strip() == "--require-hashes" for line in lines):
        raise ContractError("dependency lock must enable --require-hashes")
    locked: dict[str, str] = {}
    hashes_for: dict[str, int] = {}
    current: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--extra-index-url"):
            continue
        if line == "--require-hashes":
            continue
        match = _LOCK_LINE.fullmatch(line)
        if match:
            name, version = match.groups()
            current = name.lower().replace("_", "-")
            locked[current] = version
            hashes_for[current] = 0
            continue
        if line.startswith("--hash=sha256:"):
            if current is None or not re.fullmatch(r"--hash=sha256:[0-9a-f]{64}(?:\s*\\)?", line):
                raise ContractError(f"invalid hash continuation: {line!r}")
            hashes_for[current] += 1
            continue
        raise ContractError(f"unlocked/unsupported requirement line: {line!r}")
    if not locked or any(count == 0 for count in hashes_for.values()):
        raise ContractError("every locked distribution must have at least one SHA-256 hash")
    return locked


def parse_lock(path: Path, *, expected_sha256: str | None = None) -> dict[str, str]:
    """Parse and optionally authenticate one exact dependency-lock snapshot."""
    snapshot = _snapshot(
        Path(path), label="dependency lock", expected_sha256=expected_sha256
    )
    return _parse_lock_payload(snapshot.payload)


def verify_runtime(
    bundle_root: Path,
    *,
    expected_runtime_sha256: str | None = None,
    expected_lock_sha256: str | None = None,
) -> dict:
    runtime_path = bundle_root / "deployment/runtime-win11.json"
    runtime, runtime_snapshot = _json_snapshot(
        runtime_path,
        label="deployment runtime contract",
        expected_sha256=expected_runtime_sha256,
    )
    if not isinstance(runtime, dict):
        raise ContractError("deployment runtime contract must be an object")
    if platform.system() != runtime["os"]:
        raise ContractError(f"deployment OS must be {runtime['os']}; got {platform.system()}")
    if platform.python_implementation() != runtime["python_implementation"]:
        raise ContractError("Python implementation mismatch")
    actual_python = platform.python_version()
    if actual_python != runtime["python"]:
        raise ContractError(f"Python must be exactly {runtime['python']}; got {actual_python}")
    if platform.machine().upper() != str(runtime["architecture"]).upper():
        raise ContractError(f"architecture must be {runtime['architecture']}; got {platform.machine()}")
    if os.name != "nt" or not hasattr(sys, "getwindowsversion"):
        raise ContractError("Windows build information unavailable")
    build = int(sys.getwindowsversion().build)
    if build < int(runtime["minimum_build"]):
        raise ContractError(f"Windows build {build} is below {runtime['minimum_build']}")
    lock_path = bundle_root / "deployment/requirements-win11.lock.txt"
    lock_snapshot = _snapshot(
        lock_path,
        label="dependency lock",
        expected_sha256=expected_lock_sha256,
    )
    lock = _parse_lock_payload(lock_snapshot.payload)
    installed = {
        dist.metadata["Name"].lower().replace("_", "-"): dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    }
    for name, expected in lock.items():
        actual = installed.get(name)
        if actual != expected:
            raise ContractError(f"dependency {name} must be {expected}; got {actual!r}")
    extras = sorted(set(installed) - set(lock))
    if runtime.get("allowed_bootstrap_packages") != []:
        raise ContractError("runtime must allow zero distributions outside the exact lock")
    if extras:
        raise ContractError(
            "deployment Python contains distributions outside the exact lock: "
            + ", ".join(extras)
        )
    return {
        "windows_build": build,
        "python": actual_python,
        "runtime_sha256": runtime_snapshot.sha256,
        "lock_sha256": lock_snapshot.sha256,
        "locked_packages": lock,
        "additional_packages": extras,
        "allowed_bootstrap_packages": [],
    }


def verify_offline(bundle_root: Path) -> dict:
    bundle_root = _resolve_bundle_root(bundle_root)
    manifest = verify_bundle_manifest(bundle_root)
    files = manifest["files"]
    required = {
        "deployment/runtime-win11.json",
        "deployment/requirements-win11.lock.txt",
        "configs/common.yaml",
        "exported/release_manifest.json",
        "exported/policy.pt",
        "exported/obs_spec.json",
    }
    missing = sorted(required - set(files))
    if missing:
        raise ContractError(f"bundle manifest lacks offline authority files: {missing}")
    runtime = verify_runtime(
        bundle_root,
        expected_runtime_sha256=files["deployment/runtime-win11.json"],
        expected_lock_sha256=files["deployment/requirements-win11.lock.txt"],
    )
    try:
        common, common_snapshot = load_yaml_snapshot(
            bundle_root / "configs/common.yaml",
            expected_sha256=files["configs/common.yaml"],
            label="bundled common config",
        )
    except FileSnapshotError as exc:
        raise ContractError(str(exc)) from exc
    if not isinstance(common, dict):
        raise ContractError("bundled common config must be a mapping")
    pre_cfg = build_preprocess_cfg(common)
    action_cfg = build_action_cfg(common)
    release = verify_release(
        bundle_root / "exported",
        config_root=bundle_root,
        benchmark=True,
        expected_obs_dim=obs_dim(pre_cfg),
        expected_action_dim=action_cfg.action_dim,
        expected_manifest_sha256=files["exported/release_manifest.json"],
    )
    release_manifest_sha256 = _verified_release_manifest_sha256(release)
    if release_manifest_sha256 != files["exported/release_manifest.json"]:
        raise ContractError("verified release manifest differs from bundle authority")
    if release.get("files") != {
        "policy.pt": files["exported/policy.pt"],
        "obs_spec.json": files["exported/obs_spec.json"],
    }:
        raise ContractError("verified release policy/spec differ from bundle authority")
    runner = PolicyRunner(
        bundle_root / "exported/policy.pt",
        bundle_root / "exported/obs_spec.json",
        pre_cfg,
        action_cfg,
        device="cpu",
        expected_policy_sha256=files["exported/policy.pt"],
        expected_obs_spec_sha256=files["exported/obs_spec.json"],
    )
    action = runner.act(np.zeros((1, obs_dim(pre_cfg)), dtype=np.float32))
    if action.shape != (1, action_cfg.action_dim) or not np.isfinite(action).all():
        raise ContractError("policy smoke inference failed")
    authority = {
        "bundle_manifest_sha256": _verified_bundle_manifest_sha256(manifest),
        "release_manifest_sha256": release_manifest_sha256,
        "policy_sha256": files["exported/policy.pt"],
        "obs_spec_sha256": files["exported/obs_spec.json"],
        "common_sha256": common_snapshot.sha256,
    }
    return {
        "manifest": manifest,
        "runtime": runtime,
        "release": release,
        "common": common,
        "authority": authority,
    }


def _validate_host_arrival(
    name: str,
    value: object,
    *,
    call_started_s: float,
    call_ended_s: float,
) -> float:
    """Require an adapter-recorded arrival time inside its acquisition call."""
    try:
        arrival = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} host-arrival timestamp is unavailable") from exc
    tolerance = 1.0e-6
    if (
        not math.isfinite(arrival)
        or arrival < call_started_s - tolerance
        or arrival > call_ended_s + tolerance
    ):
        raise ContractError(
            f"{name} host-arrival timestamp is outside its acquisition call: "
            f"arrival={arrival!r}, call=[{call_started_s}, {call_ended_s}]"
        )
    return arrival


def _analyze_source_clock(
    name: str,
    *,
    source_timestamps: list[float],
    source_domains: list[str],
    host_arrivals: list[float],
    period_s: float,
    allowed_domains: set[str],
) -> dict:
    """Validate source monotonicity/cadence without inventing a clock offset."""
    if not source_timestamps or len(source_timestamps) != len(host_arrivals):
        raise ContractError(f"{name} source timestamp series is incomplete")
    unique_domains = set(source_domains)
    if len(unique_domains) != 1 or not unique_domains.issubset(allowed_domains):
        raise ContractError(
            f"{name} source timestamp domain must be one exact physical SDK domain; "
            f"got {sorted(unique_domains)!r}"
        )
    source = np.asarray(source_timestamps, dtype=np.float64)
    arrivals = np.asarray(host_arrivals, dtype=np.float64)
    if (
        not (np.isfinite(source).all() and np.isfinite(arrivals).all())
        or np.any(source < 0.0)
    ):
        raise ContractError(f"{name} source/arrival timestamp is non-finite or negative")
    source_delta = np.diff(source)
    arrival_delta = np.diff(arrivals)
    if np.any(source_delta <= 0.0) or np.any(arrival_delta <= 0.0):
        raise ContractError(f"non-monotonic {name} source/arrival timestamp")
    host_span = float(arrivals[-1] - arrivals[0])
    source_span = float(source[-1] - source[0])
    if host_span <= 0.0 or source_span <= 0.0:
        raise ContractError(f"{name} timestamp span did not advance")
    rate_ratio = source_span / host_span
    if not 0.95 <= rate_ratio <= 1.05:
        raise ContractError(
            f"{name} source clock cadence diverged from host acquisition: ratio={rate_ratio:.6f}"
        )
    max_gap = float(np.max(source_delta))
    if max_gap > 4.0 * period_s:
        raise ContractError(
            f"{name} source timestamp gap exceeds four control periods: {max_gap:.6f}s"
        )

    return {
        "domain": next(iter(unique_domains)),
        "mapping": "none_raw_clock_only",
        "source_span_s": source_span,
        "host_arrival_span_s": host_span,
        "source_to_host_rate_ratio": rate_ratio,
        "source_delta_p99_s": float(np.percentile(source_delta, 99)),
        "source_delta_max_s": max_gap,
    }


def _absolute_sensor_metrics(
    *,
    robot_lower_bounds: list[float],
    robot_upper_bounds: list[float],
    camera_source_timestamps: list[float],
    camera_host_walls: list[float],
    camera_host_arrivals: list[float],
    cycle_ends: list[float],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute conservative absolute age/skew without arbitrary clock fitting.

    The RTDE controller epoch is not comparable to the host, so freshness is
    proven by the new-package arrival interval captured by ``RealRobot``.  The
    RealSense global/system timestamp is host-wall epoch; its absolute age at
    adapter return is mapped to monotonic time using the paired wall/monotonic
    readings stored in ``CameraFrame``.  Skew is the worst-case distance from
    camera capture time to either end of the robot package-arrival interval.
    """
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (
            robot_lower_bounds,
            robot_upper_bounds,
            camera_source_timestamps,
            camera_host_walls,
            camera_host_arrivals,
            cycle_ends,
        )
    ]
    if not arrays[0].size or len({array.size for array in arrays}) != 1:
        raise ContractError("absolute sensor timestamp series is incomplete")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ContractError("absolute sensor timestamp series contains non-finite values")
    robot_lower, robot_upper, camera_source, host_wall, camera_arrival, ends = arrays
    if np.any(robot_lower > robot_upper):
        raise ContractError("RTDE package acquisition bounds are inverted")
    if np.any(robot_upper > camera_arrival + 1.0e-6):
        raise ContractError("RTDE package acquisition bound is after camera acquisition")

    camera_age_at_return = host_wall - camera_source
    if np.any(camera_age_at_return < -0.002):
        raise ContractError("RealSense source timestamp is in the host-wall future")
    camera_age_at_return = np.maximum(camera_age_at_return, 0.0)
    camera_capture_mono = camera_arrival - camera_age_at_return
    camera_age_at_cycle_end = ends - camera_capture_mono
    robot_age_upper = ends - robot_lower
    ages = np.maximum(robot_age_upper, camera_age_at_cycle_end)
    skew = np.maximum(
        np.abs(camera_capture_mono - robot_lower),
        np.abs(camera_capture_mono - robot_upper),
    )
    if np.any(ages < 0.0):
        raise ContractError("physical source acquisition maps into the future")

    # A wall-clock step would invalidate the epoch-to-monotonic conversion.
    wall_minus_mono = host_wall - camera_arrival
    wall_offset_span = float(np.max(wall_minus_mono) - np.min(wall_minus_mono))
    if wall_offset_span > 0.005:
        raise ContractError(
            f"host wall/monotonic clock offset changed by {wall_offset_span:.6f}s"
        )
    return ages, skew, {
        "method": "absolute_camera_epoch_and_fresh_rtde_arrival_bounds",
        "camera_epoch": "host_wall_clock",
        "robot_epoch": "unmapped_controller_clock",
        "robot_freshness": "new_package_after_request_same_timestamp_bracket",
        "skew_bound": "max_distance_to_rtde_acquisition_interval",
        "host_wall_monotonic_offset_span_s": wall_offset_span,
        "camera_age_at_return_p99_s": float(
            np.percentile(camera_age_at_return, 99)
        ),
        "rtde_acquisition_bound_width_p99_s": float(
            np.percentile(robot_upper - robot_lower, 99)
        ),
    }


def collect_no_motion(
    robot: RobotProbe,
    camera: CameraProbe,
    *,
    real_cfg: dict,
    safety_cfg,
    control_hz: float,
    duration_s: float,
    use_rgb: bool,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Collect long-running read-only HIL measurements; raises on first unsafe sample."""
    if duration_s < _MIN_NO_MOTION_SECONDS:
        raise ContractError(f"no-motion duration must be >= {_MIN_NO_MOTION_SECONDS:.0f} seconds")
    required_cycles = max(_MIN_NO_MOTION_CYCLES, math.ceil(duration_s * control_hz))
    if duration_s * control_hz < _MIN_NO_MOTION_CYCLES:
        raise ContractError("duration/control_hz contract provides fewer than 10,000 cycles")
    period = 1.0 / control_hz
    guard = SafetyGuard(safety_cfg)
    start = clock()
    deadline = start + duration_s
    next_scheduled_start = start
    previous_state_arrival = previous_frame_arrival = None
    previous_state_source = previous_frame_source = None
    previous_sequence = None
    previous_cycle_start = None
    robot_source_timestamps: list[float] = []
    camera_source_timestamps: list[float] = []
    robot_source_domains: list[str] = []
    camera_source_domains: list[str] = []
    robot_host_arrivals: list[float] = []
    camera_host_arrivals: list[float] = []
    robot_acquisition_lower_bounds: list[float] = []
    robot_acquisition_upper_bounds: list[float] = []
    camera_host_walls: list[float] = []
    cycle_ends: list[float] = []
    deadline_lateness: list[float] = []
    inter_cycle_intervals: list[float] = []
    cycle_budget_times: list[float] = []
    cycle_work_times: list[float] = []
    overrun_cycles = skipped_periods = 0
    consecutive = max_consecutive = catch_up_bursts = 0
    robot_modes: set[int] = set()
    safety_modes: set[int] = set()
    runtime_states: set[int] = set()
    expected_robot_mode = int(real_cfg["robot"].get("expected_robot_mode", 7))
    expected_safety_mode = int(real_cfg["robot"].get("expected_safety_mode", 1))
    allowed_runtime = {
        int(value) for value in real_cfg["robot"].get("no_motion_runtime_states", [1])
    }
    actual_camera_serial = str(camera.serial)
    if actual_camera_serial != str(real_cfg["camera"]["serial"]):
        raise ContractError(
            f"connected camera serial {actual_camera_serial!r} differs from real config"
        )
    controller_version = str(robot.controller_version)
    camera_firmware_version = str(camera.firmware_version)
    if controller_version != str(real_cfg["robot"]["controller_version"]):
        raise ContractError("controller version differs from configured exact version")
    if camera_firmware_version != str(real_cfg["camera"]["firmware_version"]):
        raise ContractError("camera firmware differs from configured exact version")
    for index in range(required_cycles):
        now = clock()
        if now < next_scheduled_start:
            sleep(next_scheduled_start - now)
        cycle_start = clock()
        if cycle_start < next_scheduled_start - 1.0e-6:
            raise ContractError("scheduler sleep returned before the planned cycle start")
        raw_lateness = max(0.0, cycle_start - next_scheduled_start)
        skipped_now = int(math.floor(raw_lateness / period))
        if skipped_now:
            # Advance the schedule over every missed slot.  Never execute a
            # burst of overdue acquisitions to catch up.
            skipped_periods += skipped_now
            next_scheduled_start += skipped_now * period
        lateness = max(0.0, cycle_start - next_scheduled_start)
        deadline_lateness.append(lateness)
        if previous_cycle_start is not None:
            interval = cycle_start - previous_cycle_start
            inter_cycle_intervals.append(interval)
            if interval < 0.5 * period:
                catch_up_bursts += 1
        previous_cycle_start = cycle_start

        robot_call_start = clock()
        state, health = robot.sample()
        robot_call_end = clock()
        camera_call_start = robot_call_end
        frame = camera.sample()
        cycle_end = clock()
        state_arrival = _validate_host_arrival(
            "robot",
            state.timestamp,
            call_started_s=robot_call_start,
            call_ended_s=robot_call_end,
        )
        frame_arrival = _validate_host_arrival(
            "camera",
            frame.timestamp,
            call_started_s=camera_call_start,
            call_ended_s=cycle_end,
        )
        cycle_work_time = cycle_end - cycle_start
        cycle_budget_time = cycle_end - next_scheduled_start
        cycle_work_times.append(cycle_work_time)
        cycle_budget_times.append(cycle_budget_time)
        missed_now = skipped_now
        if cycle_budget_time > period:
            overrun_cycles += 1
            missed_now += 1
        if missed_now:
            consecutive += missed_now
            # Count adjacent-overrun events beyond the first isolated overrun.
            # Thus 0 means no back-to-back overruns while <1% isolated overruns
            # remain permitted by the separate fraction contract.
            max_consecutive = max(max_consecutive, consecutive - 1)
        else:
            consecutive = 0
        next_scheduled_start += period

        if not health.get("connected") or any(
            health.get(key) for key in ("protective_stopped", "emergency_stopped", "safeguard_stopped")
        ):
            raise ContractError(f"unsafe/disconnected robot health at cycle {index}: {health}")
        robot_mode = int(health["robot_mode"])
        safety_mode = int(health["safety_mode"])
        runtime_state = int(health["runtime_state"])
        robot_modes.add(robot_mode)
        safety_modes.add(safety_mode)
        runtime_states.add(runtime_state)
        if robot_mode != expected_robot_mode or safety_mode != expected_safety_mode:
            raise ContractError(f"unexpected robot/safety mode at cycle {index}: {health}")
        if runtime_state not in allowed_runtime:
            raise ContractError(f"unexpected runtime state at cycle {index}: {runtime_state}")

        arrays = (state.joint_pos, state.joint_vel, state.tcp_pose, state.tcp_force, frame.depth_m)
        if state.tcp_force is None or not all(
            np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in arrays
        ):
            raise ContractError(f"non-finite state/frame or missing force at cycle {index}")
        if use_rgb and (frame.rgb is None or not np.isfinite(np.asarray(frame.rgb)).all()):
            raise ContractError(f"missing/non-finite RGB frame at cycle {index}")
        verdict = guard.check(state)
        if not verdict.ok:
            raise ContractError(f"safety violation at cycle {index}: {verdict.reason}")
        try:
            state_source = float(state.source_timestamp_s)
            frame_source = float(frame.source_timestamp_s)
            robot_source_lower = float(state.source_acquisition_host_lower_s)
            robot_source_upper = float(state.source_acquisition_host_upper_s)
            camera_host_wall = float(frame.host_wall_timestamp_s)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                f"physical source timestamp/bounds are required at cycle {index}; "
                "host-arrival wrappers are not evidence"
            ) from exc
        state_domain = str(state.source_timestamp_domain or "")
        frame_domain = str(frame.source_timestamp_domain or "")
        if (
            not math.isfinite(robot_source_lower)
            or not math.isfinite(robot_source_upper)
            or robot_source_lower < robot_call_start - 1.0e-6
            or robot_source_upper > robot_call_end + 1.0e-6
            or robot_source_lower > robot_source_upper
        ):
            raise ContractError(
                f"RTDE fresh-package acquisition bounds are invalid at cycle {index}"
            )
        if not math.isfinite(camera_host_wall):
            raise ContractError(f"camera host-wall timestamp is invalid at cycle {index}")
        if previous_state_arrival is not None and state_arrival <= previous_state_arrival:
            raise ContractError(f"non-monotonic robot host-arrival timestamp at cycle {index}")
        if previous_frame_arrival is not None and frame_arrival <= previous_frame_arrival:
            raise ContractError(f"non-monotonic camera host-arrival timestamp at cycle {index}")
        if previous_state_source is not None and state_source <= previous_state_source:
            raise ContractError(f"non-monotonic robot source timestamp at cycle {index}")
        if previous_frame_source is not None and frame_source <= previous_frame_source:
            raise ContractError(f"non-monotonic camera source timestamp at cycle {index}")
        if frame.sequence_id is None:
            raise ContractError("camera sequence_id is required")
        if previous_sequence is not None and frame.sequence_id <= previous_sequence:
            raise ContractError(f"stale/duplicate camera sequence at cycle {index}")
        previous_state_arrival = state_arrival
        previous_frame_arrival = frame_arrival
        previous_state_source = state_source
        previous_frame_source = frame_source
        previous_sequence = int(frame.sequence_id)
        robot_source_timestamps.append(state_source)
        camera_source_timestamps.append(frame_source)
        robot_source_domains.append(state_domain)
        camera_source_domains.append(frame_domain)
        robot_host_arrivals.append(state_arrival)
        camera_host_arrivals.append(frame_arrival)
        robot_acquisition_lower_bounds.append(robot_source_lower)
        robot_acquisition_upper_bounds.append(robot_source_upper)
        camera_host_walls.append(camera_host_wall)
        cycle_ends.append(cycle_end)
    if clock() < deadline:
        sleep(deadline - clock())
    elapsed = clock() - start
    robot_clock = _analyze_source_clock(
        "robot",
        source_timestamps=robot_source_timestamps,
        source_domains=robot_source_domains,
        host_arrivals=robot_host_arrivals,
        period_s=period,
        allowed_domains={"ur_rtde_controller_clock"},
    )
    camera_clock = _analyze_source_clock(
        "camera",
        source_timestamps=camera_source_timestamps,
        source_domains=camera_source_domains,
        host_arrivals=camera_host_arrivals,
        period_s=period,
        allowed_domains={"realsense_global_time", "realsense_system_time"},
    )
    ages, skews, absolute_time = _absolute_sensor_metrics(
        robot_lower_bounds=robot_acquisition_lower_bounds,
        robot_upper_bounds=robot_acquisition_upper_bounds,
        camera_source_timestamps=camera_source_timestamps,
        camera_host_walls=camera_host_walls,
        camera_host_arrivals=camera_host_arrivals,
        cycle_ends=cycle_ends,
    )
    max_age = float(np.max(ages))
    max_skew = float(np.max(skews))
    age_p99 = float(np.percentile(ages, 99))
    skew_p99 = float(np.percentile(skews, 99))
    overruns = overrun_cycles + skipped_periods
    overrun_fraction = overruns / (required_cycles + skipped_periods)
    if elapsed < duration_s or required_cycles < _MIN_NO_MOTION_CYCLES:
        raise ContractError("no-motion acquisition ended before its duration/cycle contract")
    timing_thresholds_met = (
        overrun_fraction < 0.01 and max_consecutive == 0 and catch_up_bursts == 0
    )
    delivery_thresholds_met = (
        max_age <= 2.0 * period
        and max_skew <= period
        and age_p99 <= 0.05
        and skew_p99 <= 0.05
    )
    return {
        "duration_s": elapsed,
        "samples": required_cycles,
        "required_samples": required_cycles,
        "control_hz": control_hz,
        "force_available": True,
        "read_failures": 0,
        "disconnects": 0,
        "nonfinite_samples": 0,
        "stale_samples": 0,
        "max_sensor_age_s": max_age,
        "max_sensor_skew_s": max_skew,
        "sensor_age_p99_s": age_p99,
        "sensor_skew_p99_s": skew_p99,
        "overruns": overruns,
        "overrun_cycles": overrun_cycles,
        "skipped_periods": skipped_periods,
        "overrun_fraction": overrun_fraction,
        "max_consecutive_overruns": max_consecutive,
        "catch_up_bursts": catch_up_bursts,
        "diagnostic_timing_thresholds_met": timing_thresholds_met,
        "diagnostic_delivery_thresholds_met": delivery_thresholds_met,
        "diagnostic_thresholds_met": timing_thresholds_met and delivery_thresholds_met,
        # This includes scheduler lateness against the planned start, unlike a
        # work-only stopwatch around the SDK calls.
        "max_cycle_time_s": float(max(cycle_budget_times)),
        "max_cycle_work_time_s": float(max(cycle_work_times)),
        "deadline_lateness_p99_s": float(np.percentile(deadline_lateness, 99)),
        "max_deadline_lateness_s": float(max(deadline_lateness)),
        "inter_cycle_interval_p01_s": float(np.percentile(inter_cycle_intervals, 1)),
        "inter_cycle_interval_p99_s": float(np.percentile(inter_cycle_intervals, 99)),
        "min_inter_cycle_interval_s": float(min(inter_cycle_intervals)),
        "max_inter_cycle_interval_s": float(max(inter_cycle_intervals)),
        "source_timestamps_verified": True,
        "source_timestamp_monotonic": True,
        "source_cadence_verified": True,
        # Explicitly fail closed: camera timestamps are absolute, but the UR
        # controller only exposes time-since-boot. Fresh-package bracketing
        # proves delivery freshness, not the physical acquisition instant or
        # an exact cross-device clock relationship.
        "timestamp_sync_verified": False,
        "timestamp_sync_unverified": True,
        "physical_sensor_age_verified": False,
        "physical_sensor_skew_verified": False,
        "evidence_issuance_blocked": True,
        "diagnostic_measurement_scope": (
            "absolute RealSense age plus fresh RTDE delivery bounds; "
            "not physical cross-clock acquisition age/skew"
        ),
        "source_timestamp_domains": {
            "robot": robot_clock["domain"],
            "camera": camera_clock["domain"],
        },
        "source_clock_alignment": {
            "robot": robot_clock,
            "camera": camera_clock,
        },
        "absolute_time_validation": absolute_time,
        "camera_sequence_monotonic": True,
        "robot_modes": sorted(robot_modes),
        "safety_modes": sorted(safety_modes),
        "runtime_states": sorted(runtime_states),
        "camera_serial": actual_camera_serial,
        "controller_version": controller_version,
        "camera_firmware_version": camera_firmware_version,
    }


def verify_no_motion(
    bundle_root: Path,
    real_config_path: Path,
    report_out: Path,
    *,
    duration_s: float = _MIN_NO_MOTION_SECONDS,
    robot_probe: RobotProbe | None = None,
    camera_probe: CameraProbe | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    bundle_root = _resolve_bundle_root(bundle_root)
    exact_config = require_bundle_local_real_config(bundle_root, real_config_path)
    report_out = _require_safe_report_path(bundle_root, report_out)
    offline = verify_offline(bundle_root)
    real_cfg, _config_path, real_config_snapshot = load_real_config_snapshot(exact_config)
    common = offline["common"]
    pre_cfg = build_preprocess_cfg(common)
    safety_expected = offline["manifest"]["files"].get("configs/safety.yaml")
    if safety_expected is None:
        raise ContractError("bundle manifest does not bind configs/safety.yaml")
    try:
        safety_yaml, _ = load_yaml_snapshot(
            bundle_root / "configs/safety.yaml",
            expected_sha256=safety_expected,
            label="bundled safety config",
        )
    except FileSnapshotError as exc:
        raise ContractError(str(exc)) from exc
    if not isinstance(safety_yaml, dict):
        raise ContractError("bundled safety config must be a mapping")
    safety_cfg = build_safety_cfg(safety_yaml)
    control_hz = float(common["control"]["control_hz"])
    own_robot = robot_probe is None
    own_camera = camera_probe is None
    if own_robot:
        robot_probe = ReadOnlyRobotProbe(real_cfg, bundle_root / "configs/calib/tcp_offset.yaml")
    try:
        if own_camera:
            camera_probe = ReadOnlyCameraProbe(real_cfg["camera"], enable_rgb=pre_cfg.use_rgb)
        try:
            measurements = collect_no_motion(
                robot_probe,
                camera_probe,
                real_cfg=real_cfg,
                safety_cfg=safety_cfg,
                control_hz=control_hz,
                duration_s=duration_s,
                use_rgb=pre_cfg.use_rgb,
                clock=clock,
                sleep=sleep,
            )
        finally:
            if own_camera and camera_probe is not None:
                camera_probe.close()
    finally:
        if own_robot and robot_probe is not None:
            robot_probe.close()
    now = dt.datetime.now(dt.timezone.utc)
    authority = offline["authority"]
    measurements.update({
        "release_verified": True,
        "dependencies_verified": True,
        "offline_tests_passed": True,
        "hardware_receive_only": True,
        "command_methods_exposed": False,
        "bundle_manifest_sha256": authority["bundle_manifest_sha256"],
        "release_checkpoint_sha256": offline["release"]["source"]["checkpoint_sha256"],
        "release_manifest_sha256": authority["release_manifest_sha256"],
        "policy_sha256": authority["policy_sha256"],
        "obs_spec_sha256": authority["obs_spec_sha256"],
        "real_config_path": "configs/real.local.yaml",
        "real_config_sha256": real_config_snapshot.sha256,
        "runtime": offline["runtime"],
    })
    report = {
        "schema_version": 1,
        "kind": "no_motion_diagnostic",
        "passed": False,
        "status": "timestamp_sync_unverified",
        "evidence_issuance_blocked": True,
        "blocker": (
            "physical RTDE/RealSense acquisition timestamps are not bound to "
            "one verified clock; no-motion evidence cannot be issued"
        ),
        "contract_sha256": real_contract_sha256(real_cfg),
        "recorded_at": now.isoformat(),
        "measurements": measurements,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _default_report(mode: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = "no_motion_diagnostic" if mode == "no-motion" else mode.replace("-", "_")
    return ROOT / "deployment/evidence" / f"{stem}_{stamp}.json"


def _require_safe_report_path(bundle_root: Path, report_out: Path) -> Path:
    root = _resolve_bundle_root(bundle_root)
    raw_evidence_root = root / "deployment" / "evidence"
    if is_link_or_reparse(raw_evidence_root) or is_link_or_reparse(root / "deployment"):
        raise ContractError("bundle evidence directory must not be a symlink")
    report_raw = Path(report_out).expanduser().absolute()
    if is_link_or_reparse(report_raw):
        raise ContractError("evidence report path must not be a symlink/reparse point")
    report = report_raw.resolve()
    evidence_root = raw_evidence_root.resolve()
    if evidence_root != report.parent and evidence_root not in report.parents:
        raise ContractError(f"report output must stay under {evidence_root}")
    if report.exists():
        raise ContractError(f"refusing to overwrite evidence: {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("offline", "no-motion"))
    parser.add_argument("--bundle-root", type=Path, required=True,
                        help="generated bundle root (canonical source tree is not a bundle)")
    parser.add_argument("--real-config", type=Path)
    parser.add_argument("--duration-s", type=float, default=_MIN_NO_MOTION_SECONDS)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    bundle_root = args.bundle_root
    default_report = bundle_root / "deployment/evidence" / _default_report(args.mode).name
    try:
        report_out = _require_safe_report_path(
            bundle_root, args.report_out or default_report
        )
        if args.mode == "offline":
            result = verify_offline(bundle_root)
            report_out.parent.mkdir(parents=True, exist_ok=True)
            report_out.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "offline",
                        "passed": True,
                        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "bundle_manifest_sha256": result["authority"]["bundle_manifest_sha256"],
                        "runtime": result["runtime"],
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
        else:
            if args.real_config is None:
                raise ContractError("--real-config is required for --mode no-motion")
            verify_no_motion(
                bundle_root,
                args.real_config,
                report_out,
                duration_s=args.duration_s,
            )
            print(f"[verify][BLOCKED] no-motion diagnostic: {report_out}")
            print(f"[verify] sha256={sha256_file(report_out)}")
            print(
                "[verify] timestamp_sync_unverified: no kind=no_motion / "
                "passed=true evidence was issued"
            )
            return 2
    except BaseException as exc:
        print(f"[verify][FAIL] {type(exc).__name__}: {exc}")
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    print(f"[verify][PASS] {args.mode} report: {report_out}")
    print(f"[verify] sha256={sha256_file(report_out)}")
    if args.mode == "no-motion":
        print("[verify] receive-only RTDE/RealSense HIL completed; no command interface was opened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
