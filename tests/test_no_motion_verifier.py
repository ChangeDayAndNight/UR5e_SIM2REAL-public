from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from common.config_io import build_safety_cfg, load_yaml
from common.types import CameraFrame, RobotState
from deployment.contracts import ContractError, _validate_no_motion, sha256_file
from deployment.verify_deployment import collect_no_motion, verify_bundle_manifest
from deployment.verify_deployment import _require_safe_report_path


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += max(0.0, float(seconds))


class OversleepClock(Clock):
    def __init__(self, oversleep_call: int, extra_s: float):
        super().__init__()
        self.sleep_calls = 0
        self.oversleep_call = oversleep_call
        self.extra_s = extra_s

    def sleep(self, seconds):
        self.sleep_calls += 1
        super().sleep(seconds)
        if self.sleep_calls == self.oversleep_call:
            self.now += self.extra_s


class Robot:
    def __init__(self, clock, *, force=True):
        self.clock = clock
        self.force = force

    def sample(self):
        state = RobotState(
            joint_pos=np.zeros(6),
            joint_vel=np.zeros(6),
            tcp_pose=np.array([-0.45, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
            tcp_force=np.zeros(6) if self.force else None,
            timestamp=self.clock(),
            source_timestamp_s=self.clock(),
            source_timestamp_domain="ur_rtde_controller_clock",
            source_acquisition_host_lower_s=self.clock(),
            source_acquisition_host_upper_s=self.clock(),
        )
        return state, {
            "connected": True,
            "protective_stopped": False,
            "emergency_stopped": False,
            "safeguard_stopped": False,
            "robot_mode": 7,
            "safety_mode": 1,
            "runtime_state": 1,
        }

    def close(self):
        pass

    @property
    def controller_version(self):
        return "5.14.6.0"


class Camera:
    def __init__(self, clock, *, stale=False):
        self.clock = clock
        self.count = 0
        self.stale = stale
        self.serial = "TEST_CAMERA_SERIAL"
        self.firmware_version = "5.16.0.1"

    def sample(self):
        self.count += 1
        return CameraFrame(
            depth_m=np.ones((2, 2), dtype=np.float32),
            rgb=np.ones((2, 2, 3), dtype=np.uint8),
            timestamp=self.clock(),
            sequence_id=1 if self.stale else self.count,
            source_timestamp_s=self.clock(),
            source_timestamp_domain="realsense_global_time",
            host_wall_timestamp_s=self.clock(),
        )

    def close(self):
        pass


class OverrunCamera(Camera):
    def __init__(self, clock, overrun_cycles):
        super().__init__(clock)
        self.overrun_cycles = set(overrun_cycles)

    def sample(self):
        frame = super().sample()
        if self.count in self.overrun_cycles:
            self.clock.now += 0.051
        return frame


class BufferedCamera(Camera):
    def __init__(self, clock, delay_s):
        super().__init__(clock)
        self.delay_s = float(delay_s)

    def sample(self):
        frame = super().sample()
        frame.source_timestamp_s -= self.delay_s
        return frame


def _config():
    return {
        "robot": {
            "expected_robot_mode": 7,
            "expected_safety_mode": 1,
            "no_motion_runtime_states": [1],
            "controller_version": "5.14.6.0",
        },
        "camera": {"serial": "TEST_CAMERA_SERIAL", "firmware_version": "5.16.0.1"},
    }


def _safety():
    return build_safety_cfg(load_yaml("configs/safety.yaml"))


def test_no_motion_collects_600_seconds_and_12000_samples_without_commands():
    clock = Clock()
    robot = Robot(clock)
    camera = Camera(clock)
    result = collect_no_motion(
        robot,
        camera,
        real_cfg=_config(),
        safety_cfg=_safety(),
        control_hz=20.0,
        duration_s=600.0,
        use_rgb=True,
        clock=clock,
        sleep=clock.sleep,
    )
    assert result["samples"] == 12_000
    assert result["duration_s"] >= 600.0
    assert result["camera_sequence_monotonic"] is True
    assert result["source_timestamps_verified"] is True
    assert result["source_timestamp_domains"]["robot"] == "ur_rtde_controller_clock"
    assert result["source_timestamp_domains"]["camera"] == "realsense_global_time"
    assert result["catch_up_bursts"] == 0
    assert not hasattr(robot, "send_joint_target")
    assert not hasattr(camera, "set_output")


def test_no_motion_allows_isolated_overrun_but_rejects_back_to_back_overruns():
    clock = Clock()
    result = collect_no_motion(
        Robot(clock), OverrunCamera(clock, {10}), real_cfg=_config(), safety_cfg=_safety(),
        control_hz=20.0, duration_s=600.0, use_rgb=True, clock=clock, sleep=clock.sleep,
    )
    assert result["overruns"] == 1
    assert result["max_consecutive_overruns"] == 0

    clock = Clock()
    result = collect_no_motion(
        Robot(clock), OverrunCamera(clock, {10, 11}), real_cfg=_config(),
        safety_cfg=_safety(), control_hz=20.0, duration_s=600.0, use_rgb=True,
        clock=clock, sleep=clock.sleep,
    )
    assert result["max_consecutive_overruns"] == 1
    assert result["diagnostic_timing_thresholds_met"] is False


def test_no_motion_measures_and_skips_a_missed_period_without_catch_up():
    clock = OversleepClock(oversleep_call=10, extra_s=0.051)
    result = collect_no_motion(
        Robot(clock), Camera(clock), real_cfg=_config(), safety_cfg=_safety(),
        control_hz=20.0, duration_s=600.0, use_rgb=True, clock=clock, sleep=clock.sleep,
    )
    assert result["skipped_periods"] == 1
    assert result["overruns"] == 1
    assert result["catch_up_bursts"] == 0
    assert result["max_inter_cycle_interval_s"] > 0.1


def test_no_motion_rejects_host_now_wrapper_without_physical_source_timestamp():
    clock = Clock()
    robot = Robot(clock)
    original_sample = robot.sample

    def wrapped_sample():
        state, health = original_sample()
        state.source_timestamp_s = None
        state.source_timestamp_domain = "host_monotonic"
        return state, health

    robot.sample = wrapped_sample
    with pytest.raises(ContractError, match="physical source timestamp"):
        collect_no_motion(
            robot, Camera(clock), real_cfg=_config(), safety_cfg=_safety(),
            control_hz=20.0, duration_s=600.0, use_rgb=True,
            clock=clock, sleep=clock.sleep,
        )


def test_no_motion_does_not_erase_a_fixed_camera_buffer_delay():
    clock = Clock()
    clock.now = 1_000.0
    result = collect_no_motion(
        Robot(clock), BufferedCamera(clock, 0.075),
        real_cfg=_config(), safety_cfg=_safety(), control_hz=20.0,
        duration_s=600.0, use_rgb=True, clock=clock, sleep=clock.sleep,
    )
    assert result["sensor_age_p99_s"] > 0.074
    assert result["sensor_skew_p99_s"] > 0.074
    assert result["diagnostic_delivery_thresholds_met"] is False


def test_no_motion_rejects_short_duration():
    clock = Clock()
    with pytest.raises(ContractError, match="duration"):
        collect_no_motion(
            Robot(clock), Camera(clock), real_cfg=_config(), safety_cfg=_safety(),
            control_hz=20.0, duration_s=599.0, use_rgb=True,
            clock=clock, sleep=clock.sleep,
        )


def test_no_motion_rejects_missing_force():
    clock = Clock()
    with pytest.raises(ContractError, match="missing force"):
        collect_no_motion(
            Robot(clock, force=False), Camera(clock), real_cfg=_config(), safety_cfg=_safety(),
            control_hz=20.0, duration_s=600.0, use_rgb=True,
            clock=clock, sleep=clock.sleep,
        )


def test_no_motion_rejects_duplicate_camera_frame():
    clock = Clock()
    with pytest.raises(ContractError, match="stale/duplicate"):
        collect_no_motion(
            Robot(clock), Camera(clock, stale=True), real_cfg=_config(), safety_cfg=_safety(),
            control_hz=20.0, duration_s=600.0, use_rgb=True,
            clock=clock, sleep=clock.sleep,
        )


def test_no_motion_rejects_firmware_mismatch():
    clock = Clock()
    camera = Camera(clock)
    camera.firmware_version = "unexpected"
    with pytest.raises(ContractError, match="firmware"):
        collect_no_motion(
            Robot(clock), camera, real_cfg=_config(), safety_cfg=_safety(),
            control_hz=20.0, duration_s=600.0, use_rgb=True,
            clock=clock, sleep=clock.sleep,
        )


def test_no_motion_evidence_is_bound_to_current_release_and_bundle():
    release = {
        "source": {"checkpoint_sha256": "a" * 64},
        "files": {"policy.pt": "b" * 64, "obs_spec.json": "c" * 64},
    }
    context = {
        "release": release,
        "release_manifest_sha256": "d" * 64,
        "bundle_manifest_sha256": "e" * 64,
    }
    measurements = {
        "release_verified": True,
        "dependencies_verified": True,
        "offline_tests_passed": True,
        "hardware_receive_only": True,
        "camera_sequence_monotonic": True,
        "command_methods_exposed": False,
        "duration_s": 600.0,
        "samples": 12_000,
        "read_failures": 0,
        "disconnects": 0,
        "nonfinite_samples": 0,
        "stale_samples": 0,
        "overrun_fraction": 0.0,
        "max_consecutive_overruns": 0,
        "sensor_age_p99_s": 0.01,
        "sensor_skew_p99_s": 0.01,
        "catch_up_bursts": 0,
        "overruns": 0,
        "skipped_periods": 0,
        "source_timestamps_verified": True,
        "source_timestamp_monotonic": True,
        "source_cadence_verified": True,
        "source_timestamp_domains": {
            "robot": "ur_rtde_controller_clock",
            "camera": "realsense_global_time",
        },
        "source_clock_alignment": {
            "robot": {"mapping": "none_raw_clock_only"},
            "camera": {"mapping": "none_raw_clock_only"},
        },
        "absolute_time_validation": {
            "method": "absolute_camera_epoch_and_fresh_rtde_arrival_bounds",
            "robot_freshness": "new_package_after_request_same_timestamp_bracket",
        },
        "camera_serial": "TEST_CAMERA_SERIAL",
        "release_checkpoint_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "obs_spec_sha256": "c" * 64,
        "release_manifest_sha256": "d" * 64,
        "bundle_manifest_sha256": "old",
    }
    with pytest.raises(ContractError, match="current bundle/release"):
        _validate_no_motion(measurements, _config(), context)

    measurements["bundle_manifest_sha256"] = "e" * 64
    measurements["overrun_fraction"] = 0.01
    with pytest.raises(ContractError, match="<1%"):
        _validate_no_motion(measurements, _config(), context)

    measurements["overrun_fraction"] = 0.0
    measurements["max_consecutive_overruns"] = 1
    with pytest.raises(ContractError, match="zero consecutive"):
        _validate_no_motion(measurements, _config(), context)

    measurements["max_consecutive_overruns"] = 0
    with pytest.raises(ContractError, match="timestamp_sync_unverified"):
        _validate_no_motion(measurements, _config(), context)


def test_bundle_allows_only_exact_local_config_and_evidence_tree(tmp_path):
    stable = tmp_path / "stable.txt"
    stable.write_text("stable", encoding="utf-8")
    policy_path = tmp_path / "exported" / "policy.pt"
    release_path = tmp_path / "exported" / "release_manifest.json"
    release_path.parent.mkdir()
    policy_path.write_bytes(b"policy")
    spec_path = tmp_path / "exported" / "obs_spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    release_path.write_text(
        json.dumps({
            "source": {"git_commit": "a" * 40},
            "files": {
                "policy.pt": sha256_file(policy_path),
                "obs_spec.json": sha256_file(spec_path),
            },
        }),
        encoding="utf-8",
    )
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    provenance_path = deployment / "wheelhouse.manifest.json"
    provenance_path.write_text("{}", encoding="utf-8")
    lock_path = deployment / "requirements-win11.lock.txt"
    lock_path.write_text("--require-hashes\n", encoding="utf-8")
    wheel_dir = deployment / "wheelhouse"
    wheel_dir.mkdir()
    wheel_path = wheel_dir / "wheel.whl"
    wheel_path.write_bytes(b"wheel")
    wheel_files = {
        "wheel.whl": {
            "bytes": wheel_path.stat().st_size,
            "sha256": sha256_file(wheel_path),
        }
    }
    manifest = {
        "schema_version": 2,
        "source_git_commit": "a" * 40,
        "files": {
            "stable.txt": sha256_file(stable),
            "exported/policy.pt": sha256_file(policy_path),
            "exported/obs_spec.json": sha256_file(spec_path),
            "exported/release_manifest.json": sha256_file(release_path),
            "deployment/wheelhouse.manifest.json": sha256_file(provenance_path),
            "deployment/requirements-win11.lock.txt": sha256_file(lock_path),
            "deployment/wheelhouse/wheel.whl": sha256_file(wheel_path),
        },
        "wheelhouse": {
            "provenance_manifest_sha256": sha256_file(provenance_path),
            "lock_sha256": sha256_file(lock_path),
            "aggregate_sha256": hashlib.sha256(
                json.dumps(wheel_files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "files": wheel_files,
        },
    }
    (tmp_path / "bundle_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "real.local.yaml").write_text("local", encoding="utf-8")
    evidence = tmp_path / "deployment" / "evidence"
    evidence.mkdir(parents=True)
    (evidence / "result.json").write_text("{}", encoding="utf-8")
    verify_bundle_manifest(tmp_path)
    (configs / "unexpected.yaml").write_text("extra", encoding="utf-8")
    with pytest.raises(ContractError, match="file set mismatch"):
        verify_bundle_manifest(tmp_path)


def test_no_motion_rejects_external_config_before_offline_or_hardware(tmp_path, monkeypatch):
    from deployment.verify_deployment import verify_no_motion

    outside = tmp_path / "outside.yaml"
    outside.write_text("not used", encoding="utf-8")
    monkeypatch.setattr(
        "deployment.verify_deployment.verify_offline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("offline reached")),
    )
    with pytest.raises(ContractError, match="resolve exactly"):
        verify_no_motion(tmp_path, outside, tmp_path / "report.json")


def test_completed_no_motion_run_writes_diagnostic_but_never_evidence(tmp_path, monkeypatch):
    from deployment import verify_deployment as verifier_module
    from deployment.verify_deployment import verify_no_motion

    configs = tmp_path / "configs"
    configs.mkdir()
    real_cfg = load_yaml("configs/real.example.yaml")
    real_cfg["robot"]["ip"] = "192.0.2.10"
    real_cfg["robot"]["controller_version"] = "5.14.6.0"
    real_cfg["camera"]["serial"] = "TEST_CAMERA_SERIAL"
    real_cfg["camera"]["firmware_version"] = "5.16.0.1"
    real_path = configs / "real.local.yaml"
    real_path.write_text(yaml.safe_dump(real_cfg), encoding="utf-8")
    trusted_path_bytes = real_path.read_bytes()
    consumed_config_bytes = trusted_path_bytes + b"# transient consumed snapshot\n"
    real_path.write_bytes(consumed_config_bytes)
    real_loader = verifier_module.load_real_config_snapshot

    def restore_path_after_snapshot(path, **kwargs):
        value = real_loader(path, **kwargs)
        # Recreate a B -> A swap-back after validation.  Evidence must bind B,
        # the bytes actually parsed, even though the pathname now exposes A.
        real_path.write_bytes(trusted_path_bytes)
        return value

    monkeypatch.setattr(
        verifier_module, "load_real_config_snapshot", restore_path_after_snapshot
    )
    (configs / "safety.yaml").write_text(
        Path("configs/safety.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    safety_sha256 = sha256_file(configs / "safety.yaml")
    exported = tmp_path / "exported"
    exported.mkdir()
    for name in ("release_manifest.json", "policy.pt", "obs_spec.json"):
        (exported / name).write_text(name, encoding="utf-8")
    (tmp_path / "bundle_manifest.json").write_text("bundle", encoding="utf-8")
    monkeypatch.setattr(
        "deployment.verify_deployment.verify_offline",
        lambda _root: {
            "common": load_yaml("configs/common.yaml"),
            "runtime": {"verified": True},
            "release": {"source": {"checkpoint_sha256": "a" * 64}},
            "manifest": {"files": {"configs/safety.yaml": safety_sha256}},
            "authority": {
                "bundle_manifest_sha256": "b" * 64,
                "release_manifest_sha256": "c" * 64,
                "policy_sha256": "d" * 64,
                "obs_spec_sha256": "e" * 64,
            },
        },
    )
    clock = Clock()
    report_path = tmp_path / "deployment/evidence/no_motion_diagnostic.json"
    report = verify_no_motion(
        tmp_path,
        real_path,
        report_path,
        duration_s=600.0,
        robot_probe=Robot(clock),
        camera_probe=Camera(clock),
        clock=clock,
        sleep=clock.sleep,
    )
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert report == persisted
    assert report["kind"] == "no_motion_diagnostic"
    assert report["passed"] is False
    assert report["status"] == "timestamp_sync_unverified"
    assert report["measurements"]["bundle_manifest_sha256"] == "b" * 64
    assert report["measurements"]["release_manifest_sha256"] == "c" * 64
    assert report["measurements"]["policy_sha256"] == "d" * 64
    assert report["measurements"]["obs_spec_sha256"] == "e" * 64
    assert report["measurements"]["real_config_sha256"] == hashlib.sha256(
        consumed_config_bytes
    ).hexdigest()
    assert real_path.read_bytes() == trusted_path_bytes
    assert report["evidence_issuance_blocked"] is True
    assert report["measurements"]["physical_sensor_skew_verified"] is False


def test_report_output_is_unique_and_stays_in_bundle_evidence(tmp_path):
    evidence = tmp_path / "deployment" / "evidence"
    candidate = evidence / "offline.json"
    assert _require_safe_report_path(tmp_path, candidate) == candidate.resolve()
    candidate.parent.mkdir(parents=True)
    candidate.write_text("accepted", encoding="utf-8")
    with pytest.raises(ContractError, match="overwrite"):
        _require_safe_report_path(tmp_path, candidate)
    with pytest.raises(ContractError, match="must stay under"):
        _require_safe_report_path(tmp_path, tmp_path / "outside.json")


def test_cli_preserves_raw_bundle_root_for_link_validation():
    source = Path("deployment/verify_deployment.py").read_text(encoding="utf-8")
    main = source[source.index("def main()") :]
    assert "bundle_root = args.bundle_root.resolve()" not in main
    assert "bundle_root = args.bundle_root" in main
    no_motion = source[
        source.index("def verify_no_motion(") : source.index("def _default_report(")
    ]
    assert "bundle_root = _resolve_bundle_root(bundle_root)" in no_motion
