"""Fail-closed explicit real-config/readiness contract tests."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from deployment.contracts import (
    ContractError,
    _require_evidence_path,
    _number,
    _validate_motion_safety,
    load_real_config,
    real_contract_sha256,
    require_bundle_local_real_config,
    verify_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _local_config(tmp_path: Path) -> Path:
    config = yaml.safe_load((ROOT / "configs/real.example.yaml").read_text(encoding="utf-8"))
    config["robot"]["ip"] = "192.0.2.10"
    config["robot"]["controller_version"] = "5.14.6.0"
    config["camera"]["serial"] = "123456789"
    config["camera"]["firmware_version"] = "5.16.0.1"
    path = tmp_path / "real.local.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_repository_example_cannot_be_used_for_hardware():
    with pytest.raises(ContractError, match="template"):
        load_real_config(ROOT / "configs/real.example.yaml")


def test_real_config_is_always_explicit():
    with pytest.raises(ContractError, match="--real-config"):
        load_real_config(None)


def test_local_identity_is_required(tmp_path):
    path = _local_config(tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["camera"]["serial"] = ""
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ContractError, match="camera.serial"):
        load_real_config(path)


def test_valid_local_config_loads_but_empty_evidence_remains_closed(tmp_path):
    path = _local_config(tmp_path)
    config, resolved = load_real_config(path)
    assert resolved == path.resolve()
    assert all(not ref["path"] and not ref["sha256"] for ref in config["evidence"].values())


def test_bundle_real_config_path_is_exact_and_not_a_symlink(tmp_path):
    exact = tmp_path / "configs" / "real.local.yaml"
    exact.parent.mkdir()
    exact.write_text("local", encoding="utf-8")
    assert require_bundle_local_real_config(tmp_path, exact) == exact.resolve()
    outside = tmp_path / "outside.yaml"
    outside.write_text("local", encoding="utf-8")
    with pytest.raises(ContractError, match="resolve exactly"):
        require_bundle_local_real_config(tmp_path, outside)


def test_evidence_path_must_stay_under_bundle_evidence_and_not_link(tmp_path):
    evidence_root = tmp_path / "deployment" / "evidence"
    evidence_root.mkdir(parents=True)
    allowed = evidence_root / "record.json"
    allowed.write_text("{}", encoding="utf-8")
    assert _require_evidence_path(allowed, evidence_root) == allowed.resolve()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ContractError, match="must stay under"):
        _require_evidence_path(outside, evidence_root)


def test_evidence_hash_and_json_are_from_the_same_open_handle(tmp_path, monkeypatch):
    target = tmp_path / "tcp-command.json"
    trusted_path = tmp_path / "trusted-tcp-command.json"
    real_cfg = {"evidence": {}}
    contract_hash = real_contract_sha256(real_cfg)
    common = {
        "schema_version": 1,
        "kind": "tcp_command",
        "contract_sha256": contract_hash,
        "recorded_at": "2026-08-15T00:00:00+00:00",
        "valid_until": "2026-08-16T00:00:00+00:00",
        "measurements": {
            "controller_tcp_zero": True,
            "base_frame_roundtrip_passed": True,
            "servol_signature_verified": True,
        },
    }
    trusted = json.dumps({**common, "passed": False}, sort_keys=True).encode("utf-8")
    attacker = json.dumps({**common, "passed": True}, sort_keys=True).encode("utf-8")
    trusted_path.write_bytes(trusted)
    target.write_bytes(attacker)
    real_cfg["evidence"]["tcp_command"] = {
        "path": str(target),
        "sha256": hashlib.sha256(trusted).hexdigest(),
    }
    real_open = Path.open
    redirected = False

    def split_open(path: Path, *args, **kwargs):
        nonlocal redirected
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == target and mode == "rb" and not redirected:
            redirected = True
            return real_open(trusted_path, "rb")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", split_open)
    with pytest.raises(ContractError, match="tcp_command kind/pass mismatch"):
        verify_evidence(
            real_cfg,
            tmp_path / "real.local.yaml",
            common_cfg={},
            dr_cfg={},
            now=dt.datetime(2026, 8, 15, 12, tzinfo=dt.timezone.utc),
        )
    assert redirected is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_evidence_numbers_must_be_finite(value):
    with pytest.raises(ContractError, match="finite"):
        _number({"value": value}, "value")


def test_motion_safety_evidence_is_a_hard_readiness_gate():
    measurements = {
        key: True for key in (
            "joint_position_limits_verified", "singularity_manipulability_verified",
            "self_collision_verified", "cell_collision_verified",
            "tcp_accel_jerk_torque_verified", "command_ack_verified",
            "low_speed_ladder_verified", "stop_distance_verified",
            "fault_injection_verified", "no_post_fault_commands",
        )
    }
    measurements.update({
        "minimum_joint_limit_margin_rad": 0.1,
        "minimum_manipulability": 0.01,
        "max_tcp_accel_m_s2": 0.5,
        "tcp_accel_limit_m_s2": 1.0,
        "max_tcp_jerk_m_s3": 1.0,
        "tcp_jerk_limit_m_s3": 2.0,
        "max_joint_torque_ratio": 0.5,
        "command_ack_failures": 0,
        "low_speed_levels_tested": 3,
        "max_tested_speed_scale": 0.2,
        "max_stop_distance_m": 0.01,
        "stop_distance_limit_m": 0.02,
        "max_stop_time_s": 0.1,
        "stop_time_limit_s": 0.2,
        "fault_cases_tested": 5,
        "post_fault_commands": 0,
        "release_checkpoint_sha256": "a" * 64,
        "release_manifest_sha256": "b" * 64,
        "policy_sha256": "c" * 64,
        "obs_spec_sha256": "d" * 64,
        "bundle_manifest_sha256": "e" * 64,
    })
    ctx = {
        "release": {
            "source": {"checkpoint_sha256": "a" * 64},
            "files": {"policy.pt": "c" * 64, "obs_spec.json": "d" * 64},
        },
        "release_manifest_sha256": "b" * 64,
        "bundle_manifest_sha256": "e" * 64,
    }
    _validate_motion_safety(measurements, {}, ctx)
    measurements["self_collision_verified"] = False
    with pytest.raises(ContractError, match="must be true"):
        _validate_motion_safety(measurements, {}, ctx)


def test_motion_safety_evidence_is_bound_to_current_policy_bundle():
    measurements = {
        key: True for key in (
            "joint_position_limits_verified", "singularity_manipulability_verified",
            "self_collision_verified", "cell_collision_verified",
            "tcp_accel_jerk_torque_verified", "command_ack_verified",
            "low_speed_ladder_verified", "stop_distance_verified",
            "fault_injection_verified", "no_post_fault_commands",
        )
    }
    measurements.update({
        "minimum_joint_limit_margin_rad": 0.1, "minimum_manipulability": 0.01,
        "max_tcp_accel_m_s2": 0.5, "tcp_accel_limit_m_s2": 1.0,
        "max_tcp_jerk_m_s3": 1.0, "tcp_jerk_limit_m_s3": 2.0,
        "max_joint_torque_ratio": 0.5, "command_ack_failures": 0,
        "low_speed_levels_tested": 3, "max_tested_speed_scale": 0.2,
        "max_stop_distance_m": 0.01, "stop_distance_limit_m": 0.02,
        "max_stop_time_s": 0.1, "stop_time_limit_s": 0.2,
        "fault_cases_tested": 5, "post_fault_commands": 0,
        "release_checkpoint_sha256": "0" * 64,
        "release_manifest_sha256": "b" * 64, "policy_sha256": "c" * 64,
        "obs_spec_sha256": "d" * 64, "bundle_manifest_sha256": "e" * 64,
    })
    ctx = {
        "release": {"source": {"checkpoint_sha256": "a" * 64},
                    "files": {"policy.pt": "c" * 64, "obs_spec.json": "d" * 64}},
        "release_manifest_sha256": "b" * 64, "bundle_manifest_sha256": "e" * 64,
    }
    with pytest.raises(ContractError, match="release_checkpoint_sha256"):
        _validate_motion_safety(measurements, {}, ctx)
