from __future__ import annotations

import base64
import json
import copy
import hashlib
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from unittest.mock import patch

from common.config_io import build_action_cfg, build_preprocess_cfg, load_yaml
from common.gripper_command import gripper_command_spec
from common.obs_preprocess import obs_dim
from common.policy_io import make_obs_spec, save_obs_spec
from common.runtime_authority import runtime_authority_session_contract
from common.training_stack_contract import EXPECTED_TRAINING_DISTRIBUTIONS
from deployment import release as release_module
from deployment.contracts import ContractError, validate_release_calibrations
from deployment.release import (
    ISAAC_LAB_COMMIT,
    CAPTURE_SCHEMA_VERSION,
    MIN_LATENCY_SAMPLES,
    MIN_PARITY_SAMPLES,
    MIN_PHASE3_EPISODES,
    PHASE3_CAPTURE_SEED,
    PARITY_SIDECAR_SCHEMA_VERSION,
    REQUIRED_RELEASE_CONFIGS,
    benchmark_policy,
    cleanup_release_staging,
    close_parity_observations,
    collect_training_run_provenance,
    compare_actor_and_export,
    create_release_staging,
    evaluate_sealed_parity_dataset,
    load_parity_observations,
    CANDIDATE_MANIFEST_NAME,
    candidate_receipt_binding,
    candidate_output_path,
    capture_output_path,
    make_release_manifest,
    make_candidate_manifest,
    parity_dataset_provenance,
    parity_sidecar_path,
    publish_candidate_staging,
    publish_release_staging,
    publish_regular_file_no_replace,
    seal_parity_capture,
    require_canonical_candidate_output,
    require_canonical_release_output,
    save_release_manifest,
    scan_parity_source_dumps,
    rollout_safety_contract,
    training_stack_provenance,
    verify_release,
    verify_candidate,
    verify_parity_capture,
)
from deployment.release import _validate_training_run_provenance


class ZeroPolicy(torch.nn.Module):
    def forward(self, x):
        return x[:, :7] * 0.0


class BadPolicy(torch.nn.Module):
    def forward(self, x):
        return x[:, :6] * 0.0


class OnePolicy(torch.nn.Module):
    def forward(self, x):
        return x[:, :7] * 0.0 + 1.0


class SevenZeroPolicy(torch.nn.Module):
    def forward(self, x):
        return x[:, :1].repeat(1, 7) * 0.0


class PoisonGatePolicy(torch.nn.Module):
    def forward(self, x):
        return (x[:, 3:4] - 0.5).repeat(1, 7) * 0.01


def _training_stack() -> dict:
    distributions = {
        name: ("5.1.0.0" if name == "isaacsim" else versions[0])
        for name, versions in EXPECTED_TRAINING_DISTRIBUTIONS.items()
    }
    return {
        "schema_version": 2,
        "host": {
            "os": "Windows",
            "windows_build": 22631,
            "machine": "AMD64",
            "python": "3.11.15",
            "python_implementation": "CPython",
        },
        "distributions": distributions,
        "isaac_sim": {
            "release_version": "5.1.0",
            "metadata": {
                "isaacsim": "5.1.0.0",
                "isaacsim-core": "5.1.0.0",
                "isaacsim-kernel": "5.1.0.0",
            },
            "import_origin": "site-packages/isaacsim/__init__.py",
        },
        "isaac_lab": {
            "release_version": "2.3.2",
            "git_commit": ISAAC_LAB_COMMIT,
            "git_tag": "v2.3.2",
            "checkout": "external/IsaacLab",
            "distributions": {
                "isaaclab": "0.54.2",
                "isaaclab_assets": "0.2.4",
                "isaaclab_rl": "0.4.7",
                "isaaclab_tasks": "0.11.12",
            },
            "import_origins": {
                name: f"external/IsaacLab/source/{name}/{name}/__init__.py"
                for name in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks")
            },
        },
        "project": {
            "distribution_version": "0.1.0",
            "import_origin": "source/ur5e_binpick/ur5e_binpick/__init__.py",
        },
        "ml_stack": {"torch": "2.7.0+cu128", "rsl-rl-lib": "3.1.2"},
        "gpu": {
            "cuda_runtime": "12.8",
            "nvidia_smi": [{
                "driver_version": "591.74", "name": "NVIDIA Test GPU",
                "memory_total_mib": 16384.0,
            }],
            "torch_device": {
                "index": 0, "device_count": 1, "name": "NVIDIA Test GPU",
                "memory_total_mib": 16384.0,
            },
        },
    }


def _write_approved_release_calibrations(root: Path) -> None:
    calibration_dir = root / "configs/calib"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    tcp = copy.deepcopy(identity)
    tcp[2][3] = 0.2425
    values = {
        "tcp_offset.yaml": {
            "T_flange_tcp": tcp,
            "meta": {
                "calibrated": True,
                "gripper_model": "onrobot_2fg7",
                "method": "fixed_point_multi_orientation",
                "date": "2026-08-14",
                "residual_m": 0.001,
            },
        },
        "hand_eye.yaml": {
            "T_flange_cam": identity,
            "meta": {
                "calibrated": True,
                "mount": "eye_in_hand",
                "method": "opencv_calibrateHandEye_park",
                "date": "2026-08-14",
                "residual_m": 0.008,
                "median_error_m": 0.008,
                "p95_error_m": 0.018,
                "post_correction_verified": True,
                "dr_coverage_verified": True,
            },
        },
        "nurec_landmarks.yaml": {
            "pairs": [
                {"name": "origin", "p_nurec": [0.0, 0.0, 0.0], "p_base": [0.0, 0.0, 0.0]},
                {"name": "x", "p_nurec": [1.0, 0.0, 0.0], "p_base": [1.0, 0.0, 0.0]},
                {"name": "y", "p_nurec": [0.0, 1.0, 0.0], "p_base": [0.0, 1.0, 0.0]},
            ],
            "meta": {"measured": True, "date": "2026-08-14"},
        },
        "scene_registration.yaml": {
            "T_base_nurec": identity,
            "scale": 1.0,
            "meta": {
                "calibrated": True,
                "method": "umeyama_landmarks",
                "date": "2026-08-14",
                "residual_m": 0.015,
            },
        },
        "tool_chain.yaml": {
            "elements": [
                {"name": "tool_flange_plate", "height_m": 0.005, "measured": True},
                {"name": "magbot_TCW1", "height_m": 0.0865, "measured": True},
                {"name": "onrobot_QC_RS_v3", "height_m": 0.01125, "measured": True},
                {"name": "onrobot_2FG7", "height_to_grip_m": 0.13975, "measured": True},
            ],
            "verification": {
                "expected_tcp_z_m": 0.2425,
                "tcp_offset_yaml_z_m": 0.2425,
                "qc_missing": False,
            },
        },
    }
    for name, value in values.items():
        (calibration_dir / name).write_text(
            yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
        )


def test_release_calibration_gate_accepts_only_measured_approved_configs(tmp_path):
    _write_approved_release_calibrations(tmp_path)
    result = validate_release_calibrations(tmp_path)
    assert result["hand_eye_median_error_m"] == 0.008
    assert result["hand_eye_p95_error_m"] == 0.018
    assert result["nurec_landmark_pairs"] == 3


def test_release_calibration_hash_and_yaml_use_the_same_opened_bytes(
    tmp_path, monkeypatch,
):
    _write_approved_release_calibrations(tmp_path)
    relatives = (
        "configs/calib/tcp_offset.yaml",
        "configs/calib/hand_eye.yaml",
        "configs/calib/scene_registration.yaml",
        "configs/calib/nurec_landmarks.yaml",
        "configs/calib/tool_chain.yaml",
    )
    expected = {
        relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        for relative in relatives
    }
    tcp_path = tmp_path / "configs/calib/tcp_offset.yaml"
    trusted = tcp_path.read_bytes()
    attacker = trusted + b"attacker_extension: accepted-by-yaml\n"
    attacker_path = tmp_path / "attacker-tcp.yaml"
    attacker_path.write_bytes(attacker)
    real_open = Path.open
    attacked = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal attacked
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == tcp_path and mode == "rb" and not attacked:
            attacked = True
            opened = real_open(attacker_path, "rb")
            with real_open(tcp_path, "wb") as stream:
                stream.write(trusted)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    with pytest.raises(ContractError, match="sha256 mismatch"):
        validate_release_calibrations(tmp_path, expected_hashes=expected)
    assert attacked is True


@pytest.mark.parametrize(
    ("relative", "field", "value", "message"),
    [
        ("tcp_offset.yaml", "calibrated", False, "not calibrated"),
        ("tcp_offset.yaml", "residual_m", None, "must be numeric"),
        ("hand_eye.yaml", "post_correction_verified", False, "must be true"),
        ("hand_eye.yaml", "p95_error_m", 0.020001, "p95 <= 20 mm"),
        ("scene_registration.yaml", "residual_m", 0.020001, "no greater than"),
    ],
)
def test_release_calibration_gate_rejects_false_null_or_out_of_threshold(
    tmp_path, relative, field, value, message
):
    _write_approved_release_calibrations(tmp_path)
    path = tmp_path / "configs/calib" / relative
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["meta"][field] = value
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match=message):
        validate_release_calibrations(tmp_path)


def _training_run(
    checkpoint_name: str = "model_100.pt",
    *,
    checkpoint_sha256: str = "d" * 64,
    canonical_configs: dict[str, str] | None = None,
) -> dict:
    return {
        "schema_version": 2,
        "run_root": "logs/rsl_rl/ur5e_binpick_vision/2026-08-14_00-00-00",
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_sha256,
        "receipt_sha256": "e" * 64,
        "task_id": "Isaac-BinPick-UR5e-Vision-Distill-v0",
        "mode": "distillation",
        "source": {
            "schema_version": 1,
            "project_git_commit": "a" * 40,
            "training_stack": _training_stack(),
            "canonical_configs": canonical_configs
            or {relative: "c" * 64 for relative in REQUIRED_RELEASE_CONFIGS},
            "optional_assets": {},
        },
        "configs": {"params/agent.yaml": "a" * 64, "params/env.yaml": "b" * 64},
        "agent_contract": {
            "runner_class_name": "DistillationRunner",
            "policy_class_name": "VisionStudentTeacher",
            "algorithm_class_name": "Distillation",
            "activation": "elu",
            "student_obs_normalization": False,
            "teacher_obs_normalization": False,
            "empirical_normalization": False,
        },
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _phase3_runtime_authority_fixture(root: Path, run: dict) -> tuple[dict, bytes]:
    configs = run["source"]["canonical_configs"]
    files = {}
    yaml_files = {}
    for relative, expected_sha256 in sorted(configs.items()):
        payload = (root / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        git_blob = hashlib.sha1(
            f"blob {len(payload)}\0".encode("ascii") + payload
        ).hexdigest()
        files[relative] = {
            "git_blob": git_blob,
            "sha256": expected_sha256,
            "size": len(payload),
        }
        yaml_files[relative] = {
            "payload_b64": base64.b64encode(payload).decode("ascii"),
            "sha256": expected_sha256,
            "size": len(payload),
        }
    # Runtime authority requires the exact standard Teacher/Vision asset
    # closure.  Release fixtures are offline and therefore use deterministic
    # stand-in bytes, but their manifest shape and aggregate binding must be
    # identical to a real authority receipt.
    for relative in runtime_authority_session_contract()["tracked_runtime_assets"]:
        payload = f"release-fixture:{relative}\n".encode("utf-8")
        files[relative] = {
            "git_blob": hashlib.sha1(
                f"blob {len(payload)}\0".encode("ascii") + payload
            ).hexdigest(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
    manifest = {
        "schema_version": 1,
        "record_type": "ur5e_tracked_runtime_authority",
        "project_root": str(root.resolve()),
        "purpose": "phase3-rehearsal",
        "git_commit": run["source"]["project_git_commit"],
        "runtime_files": files,
        "runtime_files_sha256": hashlib.sha256(_canonical_json_bytes(files)).hexdigest(),
        "canonical_yaml": yaml_files,
        "canonical_yaml_sha256": hashlib.sha256(
            _canonical_json_bytes(yaml_files)
        ).hexdigest(),
        "nonce": "4" * 64,
    }
    manifest_sha256 = hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest()
    binding = {
        "schema_version": 1,
        "record_type": "ur5e_tracked_runtime_authority",
        "purpose": "phase3-rehearsal",
        "manifest_sha256": manifest_sha256,
        "git_commit": run["source"]["project_git_commit"],
        "runtime_file_count": len(files),
        "runtime_files_sha256": manifest["runtime_files_sha256"],
        "canonical_yaml_sha256": manifest["canonical_yaml_sha256"],
        "canonical_yaml": dict(configs),
    }
    consumed = sorted(configs)
    sentinel = {
        "schema_version": 1,
        "record_type": "ur5e_runtime_authority_child",
        "stage": "phase3",
        "pid": 1234,
        "nonce": manifest["nonce"],
        "binding": binding,
        "consumed_canonical_yaml": consumed,
    }
    receipt = {
        "schema_version": 1,
        "record_type": "ur5e_runtime_authority_execution",
        "stage": "phase3",
        "required_canonical_yaml": [
            "configs/common.yaml",
            "configs/dr.yaml",
            "configs/safety.yaml",
            "configs/sim.yaml",
        ],
        "consumed_canonical_yaml": consumed,
        "binding": binding,
        "runtime_manifest": manifest,
        "runtime_manifest_sha256": manifest_sha256,
        "sentinel": sentinel,
        "sentinel_sha256": hashlib.sha256(_canonical_json_bytes(sentinel)).hexdigest(),
        "session_contract": runtime_authority_session_contract(),
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return receipt, encoded


def _parity_evidence(
    dim: int,
    layout: list[dict],
    run: dict,
    cfg_hash: str,
    policy_sha256: str,
    obs_spec_sha256: str,
    project_root: Path,
) -> dict:
    blocks = {}
    for item in layout:
        size = int(item["size"])
        required = max(1, int(np.ceil(size * 0.01)))
        blocks[item["name"]] = {
            "size": size,
            "varying_features": required,
            "required_varying_features": required,
            "min": 0.0,
            "max": 1.0,
        }
    runtime_receipt, runtime_bytes = _phase3_runtime_authority_fixture(
        project_root, run
    )
    return {
        "samples": MIN_PARITY_SAMPLES,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "threshold": 1.0e-5,
        "dataset_sha256": "f" * 64,
        "dataset_sidecar_sha256": "9" * 64,
        "dataset_bytes": 123456,
        "dataset_format": "npy",
        "dataset_rows": MIN_PARITY_SAMPLES,
        "observation_dim": dim,
        "observation_cfg_hash": cfg_hash,
        "source_training_receipt_sha256": run["receipt_sha256"],
        "source_checkpoint_sha256": run["checkpoint_sha256"],
        "source_run_root": run["run_root"],
        "candidate_path": f"logs/export_candidates/{run['checkpoint_sha256'][:16]}",
        "candidate_manifest_sha256": "7" * 64,
        "candidate_policy_sha256": policy_sha256,
        "candidate_obs_spec_sha256": obs_spec_sha256,
        "source_dump_root": (
            "Phase3/logs/obs_dump_sim/" + "7" * 16
            + "/20260814T000000Z-deadbeef"
        ),
        "source_capture_id": "20260814T000000Z-deadbeef",
        "source_capture_manifest_sha256": "6" * 64,
        "source_capture_episode_count": 100,
        "source_dump_file_count": 10,
        "source_dump_aggregate_sha256": "8" * 64,
        "rollout_safety_contract": rollout_safety_contract(),
        "source_rollout_safety": _rollout_safety(MIN_PARITY_SAMPLES),
        "source_rollout_safety_sha256": hashlib.sha256(
            json.dumps(
                _rollout_safety(MIN_PARITY_SAMPLES),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "source_capture_seed": PHASE3_CAPTURE_SEED,
        "source_episode_outcomes": _episode_outcome_receipt(),
        "source_episode_outcomes_sha256": hashlib.sha256(
            json.dumps(
                _episode_outcome_receipt(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "source_runtime_authority_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "source_runtime_authority_bytes": len(runtime_bytes),
        "source_runtime_authority": runtime_receipt,
        "representativeness": {
            "sampled_rows": 512,
            "unique_sampled_rows": 512,
            "minimum_unique_ratio": 0.90,
            "blocks": blocks,
        },
    }


def _sealed_nurec() -> dict:
    return {
        "metadata": {
            "schema_version": 1,
            "asset_path": "assets/Scene/Workplace.usdz",
            "asset_sha256": "1" * 64,
            "asset_bytes": 100,
            "source_path": "assets/Scene/Workplace.ply",
            "source_sha256": "2" * 64,
            "source_bytes": 200,
            "capture_id": "capture-001",
            "conversion_tool": "Isaac-Sim-5.1",
            "created_at": "2026-08-14T00:00:00+09:00",
        },
        "provenance_sha256": "3" * 64,
        "provenance_bytes": 300,
    }


def _write_capture_owner(capture: Path, binding: dict, candidate: dict) -> None:
    (capture / ".capture_owner.json").write_text(
        json.dumps({
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "record_type": "ur5e_parity_capture_owner",
            "candidate": binding,
            "candidate_id": binding["manifest_sha256"][:16],
            "capture_id": capture.name,
        }),
        encoding="utf-8",
    )
    _, encoded = _phase3_runtime_authority_fixture(
        capture.parents[4], candidate["source"]["training_run"]
    )
    (capture / "runtime_authority.json").write_bytes(encoded)


def _rollout_safety(
    denominator: int,
    counts: tuple[int, ...] = (0, 0, 0, 0, 0, 0),
) -> dict:
    fractions = [count / denominator for count in counts]
    return {
        "arm_channel_count": 6,
        "denominator": denominator,
        "clamp_counts": list(counts),
        "clamp_fractions": fractions,
        "maximum_fraction_exclusive": 0.01,
        "all_channels_below_limit": all(value < 0.01 for value in fractions),
    }


def _episode_outcome_receipt() -> dict:
    episodes = [
        {
            "episode": index,
            "dump_episode": f"ep{index:03d}",
            "steps": 100,
            "placed_success": True,
            "safety_trip": False,
            "finite": True,
            "target_pos": [-0.5, 0.0, -0.0125],
            "target_infeasible": False,
        }
        for index in range(MIN_PHASE3_EPISODES)
    ]
    return {
        "schema_version": 1,
        "record_type": "ur5e_phase3_episode_outcomes",
        "contract": {
            "seed": PHASE3_CAPTURE_SEED,
            "minimum_distinct_episodes": MIN_PHASE3_EPISODES,
            "maximum_steps_per_episode": 200,   # ★F93★ Teacher v2 학습 에피소드 10 s × 20 Hz
            "all_placed_success": True,
            "maximum_safety_trips": 0,
            "maximum_target_infeasible": 0,
            "finite_required": True,
        },
        "seed": PHASE3_CAPTURE_SEED,
        "episode_count": MIN_PHASE3_EPISODES,
        "total_steps": MIN_PARITY_SAMPLES,
        "episodes": episodes,
    }


def _student_adoption(run: dict) -> dict:
    return {
        "schema_version": 1,
        "role": "student",
        "path": "Phase2/Result/run/posttrain_evaluation/adoption.json",
        "sha256": "1" * 64,
        "checkpoint_name": run["checkpoint_name"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "training_receipt_sha256": run["receipt_sha256"],
    }


def _write_capture_dump(
    path: Path,
    observations: np.ndarray,
    counts: tuple[int, ...] = (0, 0, 0, 0, 0, 0),
    *,
    receipt_sequence_start: int = 1,
) -> None:
    rows = int(observations.shape[0])
    raw = np.zeros((rows, 6), dtype=np.float64)
    issued = raw.copy()
    for channel, count in enumerate(counts):
        issued[:count, channel] = 1.0
    np.savez(
        path,
        obs=observations,
        raw_arm_command=raw,
        safety_arm_command=raw.copy(),
        arm_command=issued,
        command_issued=np.ones(rows, dtype=np.bool_),
        arm_command_receipt_sequence=np.arange(
            receipt_sequence_start,
            receipt_sequence_start + rows,
            dtype=np.int64,
        ),
    )


def _write_small_receipt_member(path: Path, receipt_sequence) -> None:
    rows = 2
    raw = np.zeros((rows, 6), dtype=np.float64)
    applied = raw.copy()
    applied[0, 1] = 2.0e-12
    values = {
        "obs": np.zeros((rows, 6), dtype=np.float32),
        "raw_arm_command": raw,
        "safety_arm_command": raw.copy(),
        "arm_command": applied,
        "command_issued": np.ones(rows, dtype=np.bool_),
    }
    if receipt_sequence is not None:
        values["arm_command_receipt_sequence"] = receipt_sequence
    np.savez(path, **values)


def test_rollout_safety_contract_binds_actual_applied_receipt_semantics():
    contract = rollout_safety_contract()
    assert (
        contract["semantics"]
        == "same-delay-mapped-vs-guarded-vs-actual-applied-union"
    )
    assert contract["raw_command_field"] == "raw_arm_command"
    assert contract["safety_command_field"] == "safety_arm_command"
    assert contract["issued_command_field"] == "arm_command"
    assert contract["receipt_sequence_field"] == "arm_command_receipt_sequence"
    assert contract["same_lineage_required"] is True


@pytest.mark.parametrize(
    ("receipt_sequence", "message"),
    [
        (None, "receipt command evidence"),
        (np.array([1.0, np.nan], dtype=np.float64), "exact int64"),
    ],
)
def test_capture_decoder_rejects_missing_or_nan_receipt_sequence(
    tmp_path, receipt_sequence, message,
):
    from deployment import release as release_module

    member = tmp_path / "member.npz"
    _write_small_receipt_member(member, receipt_sequence)
    with pytest.raises(ContractError, match=message):
        release_module._read_capture_observation(member, expected_dim=6)


@pytest.mark.parametrize(
    "receipt_sequence",
    [
        np.array([4, 4], dtype=np.int64),
        np.array([5, 4], dtype=np.int64),
    ],
)
def test_capture_decoder_rejects_duplicate_or_decreasing_receipt_sequence(
    tmp_path, receipt_sequence,
):
    from deployment import release as release_module

    member = tmp_path / "member.npz"
    _write_small_receipt_member(member, receipt_sequence)
    with pytest.raises(ContractError, match="strictly increasing"):
        release_module._read_capture_observation(member, expected_dim=6)


def test_capture_decoder_binds_valid_receipt_lineage_before_clamp_count(tmp_path):
    from deployment import release as release_module

    member = tmp_path / "member.npz"
    _write_small_receipt_member(member, np.array([7, 9], dtype=np.int64))
    block, record, counts, sequences = release_module._read_capture_observation(
        member, expected_dim=6
    )
    assert block.shape == (2, 6)
    assert counts == (0, 1, 0, 0, 0, 0)
    assert sequences == (7, 9)
    assert record["receipt_sequence_first"] == 7
    assert record["receipt_sequence_last"] == 9
    assert record["receipt_sequence_sha256"] == hashlib.sha256(
        np.array([7, 9], dtype=np.int64).tobytes(order="C")
    ).hexdigest()


def test_capture_decoder_counts_reversing_clamps_as_union(tmp_path):
    from deployment import release as release_module

    member = tmp_path / "reversal.npz"
    raw = np.zeros((2, 6), dtype=np.float64)
    guarded = raw.copy()
    guarded[0, 0] = 1.0
    # The adapter reverses the first row back to the mapped raw value.  A
    # raw-vs-applied-only verifier would incorrectly report zero clamps.
    np.savez(
        member,
        obs=np.zeros((2, 6), dtype=np.float32),
        raw_arm_command=raw,
        safety_arm_command=guarded,
        arm_command=raw.copy(),
        command_issued=np.ones(2, dtype=np.bool_),
        arm_command_receipt_sequence=np.array([1, 2], dtype=np.int64),
    )
    _, _, counts, _ = release_module._read_capture_observation(
        member, expected_dim=6
    )
    assert counts == (1, 0, 0, 0, 0, 0)


def _write_phase3_geometry(root: Path) -> None:
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "common.yaml").write_text(
        "proprio:\n  tcp_pos_scale: 1.0\n"
        "target:\n  sample_range:\n    x: [-0.725, -0.275]\n"
        "    y: [-0.3, 0.3]\n    z: [-0.0125, -0.0125]\n",
        encoding="utf-8",
    )
    (configs / "sim.yaml").write_text(
        "scene:\n  belt_size_m: [0.58, 1.25, 0.04]\n"
        "  belt_center_xy_m: [-0.5, 0.0]\n  belt_yaw_deg: 180.0\n"
        "  object_size_m: 0.05\n  table_height_m: -0.0375\n",
        encoding="utf-8",
    )
    (configs / "safety.yaml").write_text(
        "workspace:\n  min: [-0.75, -0.35, -0.0375]\n"
        "  max: [-0.21, 0.35, 0.60]\n",
        encoding="utf-8",
    )
    for relative in sorted(REQUIRED_RELEASE_CONFIGS):
        path = root / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture: {relative}\n", encoding="utf-8")


def _phase3_canonical_hashes(root: Path) -> dict[str, str]:
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in REQUIRED_RELEASE_CONFIGS
    }


def _write_phase3_reach_test_geometry(root: Path) -> None:
    """Allow a target through range/support/workspace so reach is the rejecting gate."""
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "common.yaml").write_text(
        "proprio:\n  tcp_pos_scale: 1.0\n"
        "target:\n  sample_range:\n    x: [-1.0, 1.0]\n"
        "    y: [-1.0, 1.0]\n    z: [-0.0125, -0.0125]\n",
        encoding="utf-8",
    )
    (configs / "sim.yaml").write_text(
        "scene:\n  belt_size_m: [3.0, 3.0, 0.04]\n"
        "  belt_center_xy_m: [0.0, 0.0]\n  belt_yaw_deg: 0.0\n"
        "  object_size_m: 0.05\n  table_height_m: -0.0375\n",
        encoding="utf-8",
    )
    (configs / "safety.yaml").write_text(
        "workspace:\n  min: [-2.0, -2.0, -0.0375]\n"
        "  max: [2.0, 2.0, 0.60]\n",
        encoding="utf-8",
    )


def _write_valid_phase3_capture(
    capture: Path,
    observations: np.ndarray,
    counts: tuple[int, ...] = (0, 0, 0, 0, 0, 0),
) -> list[dict]:
    """Use 100 episode batches while preserving the real per-episode row gate."""
    chunks = np.array_split(observations, MIN_PHASE3_EPISODES)
    remaining = list(counts)
    outcomes = []
    receipt_sequence_start = 1
    for index, chunk in enumerate(chunks):
        episode = capture / f"ep{index:03d}"
        episode.mkdir(parents=True, exist_ok=True)
        local_counts = tuple(min(int(chunk.shape[0]), remaining[channel]) for channel in range(6))
        remaining = [remaining[channel] - local_counts[channel] for channel in range(6)]
        alpha = index / max(1, MIN_PHASE3_EPISODES - 1)
        target_values = [
            -0.70 + 0.40 * alpha,
            -0.25 + 0.50 * alpha,
            -0.0125,
        ]
        target = np.asarray(target_values, dtype=np.float32)
        chunk[:, :3] = target
        _write_capture_dump(
            episode / "step_0000.npz",
            chunk,
            local_counts,
            receipt_sequence_start=receipt_sequence_start,
        )
        receipt_sequence_start += int(chunk.shape[0])
        outcomes.append({
            "episode": index,
            "steps": int(chunk.shape[0]),
            "placed_success": True,
            "tripped": False,
            "trip_reason": "",
            # Preserve the canonical YAML-domain value (not the float32
            # round-trip) in the outcome receipt.  The verifier converts this
            # value to float32 when it compares it with the actor observation.
            "target_pos": [float(value) for value in target_values],
        })
    assert remaining == [0] * 6
    return outcomes


def _capture_test_context(tmp_path: Path, *, observation_dim: int = 6):
    _write_phase3_geometry(tmp_path)
    run = _training_run(canonical_configs=_phase3_canonical_hashes(tmp_path))
    candidate_dir = tmp_path / "logs/export_candidates" / run["checkpoint_sha256"][:16]
    candidate_dir.mkdir(parents=True)
    candidate = {
        "source": {
            "training_receipt_sha256": run["receipt_sha256"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "checkpoint_name": run["checkpoint_name"],
            "run_root": run["run_root"],
            "training_run": run,
        },
        "files": {"policy.pt": "1" * 64, "obs_spec.json": "2" * 64},
        "observation_contract": {
            "cfg_hash": "a" * 16,
            "observation_dim": observation_dim,
            "layout": [
                {"name": "target_pos", "size": 3},
                *(
                    [{"name": "observations", "size": observation_dim - 3}]
                    if observation_dim > 3 else []
                ),
            ],
            "action_mode": "ik_rel",
            "action_dim": 7,
        },
        "rollout_safety_contract": rollout_safety_contract(),
        "configs": run["source"]["canonical_configs"],
    }
    (candidate_dir / CANDIDATE_MANIFEST_NAME).write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    binding = candidate_receipt_binding(candidate_dir, candidate, project_root=tmp_path)
    capture = capture_output_path(
        tmp_path, binding["manifest_sha256"], "20260814T010203Z-deadbeef"
    )
    episode = capture / "ep000"
    capture.mkdir(parents=True)
    _write_capture_owner(capture, binding, candidate)
    return candidate_dir, candidate, capture, episode


def test_release_training_run_requires_mode_specific_optional_assets():
    distill = _training_run()
    assert _validate_training_run_provenance(distill) is distill
    invalid = copy.deepcopy(distill)
    invalid["source"]["optional_assets"] = {"nurec": _sealed_nurec()}
    with pytest.raises(ContractError, match="must not bind"):
        _validate_training_run_provenance(invalid)

    nurec = copy.deepcopy(distill)
    nurec["task_id"] = "Isaac-BinPick-UR5e-Vision-NuRec-v0"
    nurec["mode"] = "nurec"
    nurec["source"]["optional_assets"] = {"nurec": _sealed_nurec()}
    assert _validate_training_run_provenance(nurec) is nurec
    nurec["source"]["optional_assets"] = {}
    with pytest.raises(ContractError, match="must bind exactly"):
        _validate_training_run_provenance(nurec)


def test_parity_loader_requires_exact_2d_shape(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((100, 2, 3), dtype=np.float32))
    with pytest.raises(ContractError, match="exact shape"):
        load_parity_observations(path, 6)


def test_parity_loader_rejects_repeated_zero_fixture(tmp_path):
    path = tmp_path / "zeros.npy"
    np.save(path, np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32))
    with pytest.raises(ContractError, match="diversity"):
        load_parity_observations(path, 6)


def test_atomic_artifact_publish_never_replaces_concurrent_destination(tmp_path):
    source = tmp_path / ".artifact.tmp"
    destination = tmp_path / "artifact.json"
    source.write_bytes(b"ours")
    destination.write_bytes(b"concurrent")
    with pytest.raises(ContractError, match="refusing overwrite"):
        publish_regular_file_no_replace(source, destination)
    assert destination.read_bytes() == b"concurrent"
    assert source.read_bytes() == b"ours"


def test_atomic_artifact_publish_accepts_only_private_parity_child(tmp_path):
    staging = tmp_path / ".parity-build-test"
    staging.mkdir()
    source = staging / "data.npy"
    destination = tmp_path / "data.npy"
    source.write_bytes(b"complete")
    publish_regular_file_no_replace(source, destination)
    assert destination.read_bytes() == b"complete"
    assert not source.exists()
    unsafe = tmp_path / "arbitrary" / "data.npy"
    unsafe.parent.mkdir()
    unsafe.write_bytes(b"unsafe")
    with pytest.raises(ContractError, match="safe sibling"):
        publish_regular_file_no_replace(unsafe, tmp_path / "other.npy")


def test_parity_dataset_digest_and_size_share_one_open_handle(tmp_path, monkeypatch):
    from scripts import build_parity_dataset as builder

    dataset = tmp_path / "dataset.npy"
    trusted = b"trusted-parity-bytes"
    swapped = b"different-size-attacker-payload"
    alternate = tmp_path / "trusted-opened-file"
    dataset.write_bytes(trusted)
    alternate.write_bytes(trusted)
    real_open = Path.open
    attacked = False

    def swap_after_open(path: Path, *args, **kwargs):
        nonlocal attacked
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == dataset and mode == "rb" and not attacked:
            attacked = True
            opened = real_open(alternate, "rb")
            dataset.write_bytes(swapped)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_after_open)
    digest, size = builder._sha256_size_regular_file(
        dataset, label="test parity dataset"
    )
    assert attacked is True
    assert digest == hashlib.sha256(trusted).hexdigest()
    assert size == len(trusted)
    assert dataset.read_bytes() == swapped


def test_candidate_receipt_rejects_swap_back_manifest_bytes(tmp_path, monkeypatch):
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    manifest_path = candidate_dir / CANDIDATE_MANIFEST_NAME
    trusted = {
        "files": {"policy.pt": "1" * 64, "obs_spec.json": "2" * 64},
        "marker": "trusted",
    }
    attacker = {**trusted, "marker": "attacker"}
    trusted_bytes = json.dumps(trusted, sort_keys=True).encode("utf-8")
    attacker_bytes = json.dumps(attacker, sort_keys=True).encode("utf-8")
    alternate = tmp_path / "attacker-candidate-manifest"
    alternate.write_bytes(attacker_bytes)
    manifest_path.write_bytes(attacker_bytes)
    real_open = Path.open
    attacked = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal attacked
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == manifest_path and mode in {"r", "rb"} and not attacked:
            attacked = True
            opened = real_open(alternate, *args, **kwargs)
            with real_open(manifest_path, "wb") as stream:
                stream.write(trusted_bytes)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    with pytest.raises(ContractError, match="supplied verified manifest"):
        candidate_receipt_binding(candidate_dir, trusted, project_root=tmp_path)
    assert attacked is True
    assert manifest_path.read_bytes() == trusted_bytes


def test_capture_owner_hash_is_from_the_same_parsed_handle(tmp_path, monkeypatch):
    from deployment import release as release_module

    capture = tmp_path / "20260814T010203Z-deadbeef"
    capture.mkdir()
    binding = {
        "path": "logs/export_candidates/deadbeefdeadbeef",
        "manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "obs_spec_sha256": "c" * 64,
    }
    owner = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "record_type": release_module.CAPTURE_OWNER_TYPE,
        "candidate": binding,
        "candidate_id": "a" * 16,
        "capture_id": capture.name,
    }
    owner_path = capture / release_module.CAPTURE_OWNER_NAME
    trusted_bytes = (json.dumps(owner, indent=2, sort_keys=True) + "\n").encode("utf-8")
    consumed_bytes = json.dumps(owner, separators=(",", ":"), sort_keys=True).encode("utf-8")
    alternate = tmp_path / "alternate-owner"
    alternate.write_bytes(consumed_bytes)
    owner_path.write_bytes(consumed_bytes)
    real_open = Path.open
    attacked = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal attacked
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == owner_path and mode in {"r", "rb"} and not attacked:
            attacked = True
            opened = real_open(alternate, *args, **kwargs)
            with real_open(owner_path, "wb") as stream:
                stream.write(trusted_bytes)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    parsed, digest = release_module._validate_capture_owner(capture, binding)
    assert parsed == owner
    assert digest == hashlib.sha256(consumed_bytes).hexdigest()
    assert attacked is True
    assert owner_path.read_bytes() == trusted_bytes


def test_candidate_is_explicitly_unapproved_hash_bound_and_student_only(
    tmp_path, monkeypatch,
):
    checkpoint = tmp_path / "logs/rsl_rl/ur5e_binpick_vision/run/model_100.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"sealed student checkpoint")
    policy_module = torch.jit.trace(ZeroPolicy(), torch.zeros(1, 12))
    staging = tmp_path / "staging"
    staging.mkdir()
    policy_path = staging / "policy.pt"
    policy_module.save(str(policy_path))
    spec_path = staging / "obs_spec.json"
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    spec = {
        "model_role": "student",
        "empirical_normalization": False,
        "source_checkpoint": checkpoint.name,
        "policy_sha256": policy_hash,
        "action_scheme": "ik_rel",
        "action_dim": 7,
        "obs_dim": 12,
        "cfg_hash": "a" * 16,
        "layout": [
            {"name": "gripper_command_state", "size": 1},
            {"name": "observations", "size": 11},
        ],
        "gripper_command_state": gripper_command_spec(
            transition_s=1.20, transition_steps=24
        ),
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    config_paths = []
    for relative in sorted(REQUIRED_RELEASE_CONFIGS):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        config_paths.append(path)
    config_hashes = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config_paths
    }
    run = _training_run(
        checkpoint.name,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        canonical_configs=config_hashes,
    )
    with patch(
        "scripts.adoption.verify_adopted_checkpoint", return_value=_student_adoption(run)
    ):
        manifest = make_candidate_manifest(
            policy_path=policy_path,
            obs_spec_path=spec_path,
            checkpoint_path=checkpoint,
            git_commit="a" * 40,
            training_stack=_training_stack(),
            training_run=run,
            student_adoption=_student_adoption(run),
            config_paths=config_paths,
            config_root=tmp_path,
        )
    drifted_adoption = {**_student_adoption(run), "sha256": "2" * 64}
    with patch(
        "scripts.adoption.verify_adopted_checkpoint", return_value=drifted_adoption
    ), pytest.raises(ContractError, match="changed before candidate"):
        make_candidate_manifest(
            policy_path=policy_path,
            obs_spec_path=spec_path,
            checkpoint_path=checkpoint,
            git_commit="a" * 40,
            training_stack=_training_stack(),
            training_run=run,
            student_adoption=_student_adoption(run),
            config_paths=config_paths,
            config_root=tmp_path,
        )
    assert manifest["approved"] is False
    candidate_dir = candidate_output_path(tmp_path, run["checkpoint_sha256"])
    candidate_dir.mkdir(parents=True)
    for source in (policy_path, spec_path):
        (candidate_dir / source.name).write_bytes(source.read_bytes())
    (candidate_dir / CANDIDATE_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with patch(
        "scripts.adoption.verify_adopted_checkpoint", return_value=_student_adoption(run)
    ):
        assert verify_candidate(
            candidate_dir, project_root=tmp_path, require_current_context=False
        ) == manifest
    assert not (candidate_dir / "release_manifest.json").exists()

    # A same-schema byte swap cannot detach the manifest hash used by the
    # capture receipt from the exact bytes consumed by candidate verification.
    manifest_path = candidate_dir / CANDIDATE_MANIFEST_NAME
    trusted_manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    consumed_manifest_bytes = json.dumps(
        manifest, separators=(",", ":")
    ).encode("utf-8")
    alternate_manifest = tmp_path / "alternate-candidate-manifest"
    alternate_manifest.write_bytes(consumed_manifest_bytes)
    manifest_path.write_bytes(consumed_manifest_bytes)
    real_open = Path.open
    manifest_attacked = False

    def swap_back_manifest_open(path: Path, *args, **kwargs):
        nonlocal manifest_attacked
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == manifest_path and mode in {"r", "rb"} and not manifest_attacked:
            manifest_attacked = True
            opened = real_open(alternate_manifest, *args, **kwargs)
            with real_open(manifest_path, "wb") as stream:
                stream.write(trusted_manifest_bytes)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_manifest_open)
    with patch(
        "scripts.adoption.verify_adopted_checkpoint", return_value=_student_adoption(run)
    ):
        verified = verify_candidate(
            candidate_dir, project_root=tmp_path, require_current_context=False
        )
    with pytest.raises(ContractError, match="changed after verification"):
        candidate_receipt_binding(candidate_dir, verified, project_root=tmp_path)
    assert manifest_attacked is True
    assert manifest_path.read_bytes() == trusted_manifest_bytes

    # The obs-spec hash and its schema validation must consume one byte
    # payload.  This reproduces the old hash-A / parse-B / restore-A window.
    obs_path = candidate_dir / "obs_spec.json"
    trusted_obs_bytes = obs_path.read_bytes()
    swapped_obs_bytes = (json.dumps(spec, indent=2) + "\n").encode("utf-8")
    assert hashlib.sha256(swapped_obs_bytes).hexdigest() != manifest["files"]["obs_spec.json"]
    obs_path.write_bytes(swapped_obs_bytes)
    from deployment import release as release_module
    real_sha256_file = release_module.sha256_file
    real_read_text = Path.read_text

    def old_split_hash(path):
        if Path(path) == obs_path:
            obs_path.write_bytes(trusted_obs_bytes)
            digest = real_sha256_file(obs_path)
            obs_path.write_bytes(swapped_obs_bytes)
            return digest
        return real_sha256_file(path)

    def old_split_parse(path: Path, *args, **kwargs):
        value = real_read_text(path, *args, **kwargs)
        if path == obs_path:
            obs_path.write_bytes(trusted_obs_bytes)
        return value

    try:
        with patch("deployment.release.sha256_file", side_effect=old_split_hash), \
                patch.object(Path, "read_text", old_split_parse), \
                patch(
                    "scripts.adoption.verify_adopted_checkpoint",
                    return_value=_student_adoption(run),
                ), pytest.raises(ContractError, match="obs_spec.json hash mismatch"):
            verify_candidate(
                candidate_dir, project_root=tmp_path, require_current_context=False
            )
    finally:
        obs_path.write_bytes(trusted_obs_bytes)

    for field, tampered_value in (
        ("maximum_clamp_fraction_exclusive", 0.02),
        ("semantics", "current-request-vs-applied"),
    ):
        invalid_contract = copy.deepcopy(manifest)
        invalid_contract["rollout_safety_contract"][field] = tampered_value
        (candidate_dir / CANDIDATE_MANIFEST_NAME).write_text(
            json.dumps(invalid_contract), encoding="utf-8"
        )
        with patch(
            "scripts.adoption.verify_adopted_checkpoint", return_value=_student_adoption(run)
        ), pytest.raises(ContractError, match="rollout-safety contract"):
            verify_candidate(candidate_dir, project_root=tmp_path, require_current_context=False)
    (candidate_dir / CANDIDATE_MANIFEST_NAME).write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    (candidate_dir / "policy.pt").write_bytes(b"tampered")
    with patch(
        "scripts.adoption.verify_adopted_checkpoint", return_value=_student_adoption(run)
    ), pytest.raises(ContractError, match="artifact hash"):
        verify_candidate(candidate_dir, project_root=tmp_path, require_current_context=False)


def test_build_parity_dataset_atomically_publishes_candidate_bound_pair(tmp_path, monkeypatch):
    from scripts import build_parity_dataset as builder

    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    observations[:, 3] = np.linspace(-1.0, 1.0, MIN_PARITY_SAMPLES, dtype=np.float32)
    _write_phase3_geometry(tmp_path)
    layout = [
        {"name": "target_pos", "size": 3},
        {"name": "observations", "size": 3},
    ]
    candidate_obs_spec = {
        "obs_dim": 6,
        "action_scheme": "ik_rel",
        "layout": layout,
        "cfg_hash": "a" * 16,
    }
    candidate_obs_spec_bytes = json.dumps(candidate_obs_spec).encode("utf-8")
    run = _training_run(canonical_configs=_phase3_canonical_hashes(tmp_path))
    candidate_dir = tmp_path / "logs/export_candidates" / run["checkpoint_sha256"][:16]
    candidate_dir.mkdir(parents=True)
    candidate = {
        "source": {
            "training_receipt_sha256": run["receipt_sha256"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "checkpoint_name": run["checkpoint_name"],
            "run_root": run["run_root"],
            "training_run": run,
        },
        "files": {
            "policy.pt": "1" * 64,
            "obs_spec.json": hashlib.sha256(candidate_obs_spec_bytes).hexdigest(),
        },
        "observation_contract": {
            "cfg_hash": "a" * 16,
            "observation_dim": 6,
            "layout": layout,
            "action_mode": "ik_rel",
            "action_dim": 7,
        },
        "rollout_safety_contract": rollout_safety_contract(),
        "configs": run["source"]["canonical_configs"],
    }
    (candidate_dir / CANDIDATE_MANIFEST_NAME).write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    candidate_spec_path = candidate_dir / "obs_spec.json"
    candidate_spec_path.write_bytes(candidate_obs_spec_bytes)
    binding = candidate_receipt_binding(candidate_dir, candidate, project_root=tmp_path)
    capture = capture_output_path(
        tmp_path, binding["manifest_sha256"], "20260814T010203Z-deadbeef"
    )
    capture.mkdir(parents=True)
    _write_capture_owner(capture, binding, candidate)
    outcomes = _write_valid_phase3_capture(capture, observations)
    with patch("deployment.release.verify_candidate", return_value=candidate):
        capture_manifest = seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    output_path = tmp_path / "logs/parity/release.npy"
    with patch.object(builder, "verify_candidate", return_value=candidate), \
            patch("deployment.release.verify_candidate", return_value=candidate):
        # Deterministic old-race reproduction: A was verified, B is visible
        # only while the builder copies, then A returns before the rescan.
        victim = capture / "ep000/step_0000.npz"
        original_bytes = victim.read_bytes()
        replacement = tmp_path / "replacement.npz"
        _write_capture_dump(replacement, np.full((100, 6), 0.25, dtype=np.float32))
        replacement_bytes = replacement.read_bytes()
        original_reader = builder.read_verified_capture_observation_block
        attacked = False

        def swap_during_copy(path, record, *, expected_dim):
            nonlocal attacked
            if not attacked:
                attacked = True
                path.write_bytes(replacement_bytes)
                try:
                    return original_reader(path, record, expected_dim=expected_dim)
                finally:
                    path.write_bytes(original_bytes)
            return original_reader(path, record, expected_dim=expected_dim)

        monkeypatch.setattr(builder, "read_verified_capture_observation_block", swap_during_copy)
        with pytest.raises(ContractError, match="changed before dataset assembly"):
            builder.build_dataset(candidate_dir, capture_manifest,
                                  tmp_path / "logs/parity/attack.npy")
        assert not (tmp_path / "logs/parity/attack.npy").exists()
        assert not parity_sidecar_path(tmp_path / "logs/parity/attack.npy").exists()
        monkeypatch.setattr(builder, "read_verified_capture_observation_block", original_reader)
        # If the public dataset changes after its pre-sidecar hash, the
        # sidecar commit marker must not make the pair look successfully
        # published.  Both exact objects are rolled back to private quarantine.
        pair_output = tmp_path / "logs/parity/pair-race.npy"
        real_publish = builder.publish_regular_file_no_replace
        publish_calls = 0

        def mutate_dataset_before_sidecar(source, destination):
            nonlocal publish_calls
            publish_calls += 1
            if publish_calls == 2:
                pair_output.write_bytes(b"attacker-dataset")
            return real_publish(source, destination)

        monkeypatch.setattr(
            builder, "publish_regular_file_no_replace", mutate_dataset_before_sidecar
        )
        with pytest.raises(ContractError, match="published parity dataset differs"):
            builder.build_dataset(candidate_dir, capture_manifest, pair_output)
        assert publish_calls == 2
        assert not pair_output.exists()
        assert not parity_sidecar_path(pair_output).exists()
        assert list(pair_output.parent.glob(f".{pair_output.name}.failed-*"))
        assert list(
            pair_output.parent.glob(
                f".{parity_sidecar_path(pair_output).name}.failed-*"
            )
        )
        monkeypatch.setattr(builder, "publish_regular_file_no_replace", real_publish)
        # The candidate verifier consumed A.  Swap the pathname to a valid B
        # immediately after the anchored A snapshot is returned.  The builder
        # must keep using A's manifest-bound cfg/layout for the sidecar.
        swapped_spec = {**candidate_obs_spec, "cfg_hash": "b" * 16}
        trusted_spec_reader = builder.read_anchored_regular_file

        def swap_after_anchored_spec(path, *, expected_sha256, label):
            payload = trusted_spec_reader(
                path, expected_sha256=expected_sha256, label=label
            )
            candidate_spec_path.write_text(json.dumps(swapped_spec), encoding="utf-8")
            return payload

        monkeypatch.setattr(
            builder, "read_anchored_regular_file", swap_after_anchored_spec
        )
        output, sidecar = builder.build_dataset(
            candidate_dir, capture_manifest, output_path
        )
    assert output == output_path.resolve()
    assert sidecar == parity_sidecar_path(output)
    decoded = json.loads(sidecar.read_text(encoding="utf-8"))
    assert decoded["candidate"] == binding
    assert decoded["observation_contract"]["cfg_hash"] == "a" * 16
    assert json.loads(candidate_spec_path.read_text(encoding="utf-8"))["cfg_hash"] == "b" * 16
    assert decoded["source_dumps"]["file_count"] == MIN_PHASE3_EPISODES
    assert decoded["source_dumps"]["total_rows"] == MIN_PARITY_SAMPLES
    assert np.load(output, allow_pickle=False).shape == (MIN_PARITY_SAMPLES, 6)


def test_capture_postseal_failure_removes_public_commit_marker(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, _ = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations)
    manifest_path = capture / "capture_manifest.json"
    with patch("deployment.release.verify_candidate", return_value=candidate), \
            patch(
                "deployment.release.verify_parity_capture",
                side_effect=ContractError("injected capture postverify failure"),
            ), pytest.raises(ContractError, match="postverify failure"):
        seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )
    assert not manifest_path.exists()
    assert list(
        capture.parent.glob(
            f".{capture.name}-capture_manifest.json.failed-*"
        )
    )
    assert (capture / "ep000/step_0000.npz").is_file()
    with patch("deployment.release.verify_candidate", return_value=candidate):
        retried = seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )
    assert retried == manifest_path
    assert manifest_path.is_file()


def test_capture_rejects_exact_one_percent_and_reported_bypass(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, episode = _capture_test_context(tmp_path)
    exact_boundary = (100, 0, 0, 0, 0, 0)
    outcomes = _write_valid_phase3_capture(capture, observations, exact_boundary)

    # A caller cannot round 1.00% down in its reported evidence: sealing always
        # recomputes the counts from the raw/guarded/applied arrays first.
    reported_below_boundary = _rollout_safety(
        MIN_PARITY_SAMPLES, (99, 0, 0, 0, 0, 0)
    )
    with patch("deployment.release.verify_candidate", return_value=candidate):
        with pytest.raises(ContractError, match="strictly below 1%"):
            seal_parity_capture(
                capture,
                candidate_dir=candidate_dir,
                project_root=tmp_path,
                expected_dim=6,
                rollout_safety=reported_below_boundary,
                seed=PHASE3_CAPTURE_SEED,
                episode_outcomes=outcomes,
            )


def test_capture_seal_rejects_receipt_sequence_replay_across_members(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, _ = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations)
    # ep000 owns 1..100.  Replaying 100 as ep001's first receipt is invalid
    # even though ep001 is internally strictly increasing.
    _write_capture_dump(
        capture / "ep001/step_0000.npz",
        observations[100:200],
        receipt_sequence_start=100,
    )
    with patch("deployment.release.verify_candidate", return_value=candidate), \
            pytest.raises(ContractError, match="strictly increasing across members"):
        seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )


def test_capture_verify_rejects_receipt_sequence_tamper_after_seal(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, _ = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations)
    with patch("deployment.release.verify_candidate", return_value=candidate):
        manifest = seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )
    _write_capture_dump(
        capture / "ep001/step_0000.npz",
        observations[100:200],
        receipt_sequence_start=100,
    )
    with patch("deployment.release.verify_candidate", return_value=candidate), \
            pytest.raises(ContractError, match="strictly increasing across members"):
        verify_parity_capture(
            manifest,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
        )


def test_capture_rejects_one_batched_episode_even_with_ten_thousand_rows(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, episode = _capture_test_context(tmp_path)
    episode.mkdir()
    _write_capture_dump(episode / "step_0000.npz", observations)
    one_outcome = [{
        "episode": 0,
        "steps": MIN_PARITY_SAMPLES,
        "placed_success": True,
        "tripped": False,
        "trip_reason": "",
        "target_pos": [-0.5, 0.0, -0.0125],
    }]
    with patch("deployment.release.verify_candidate", return_value=candidate), \
            pytest.raises(ContractError, match="100 distinct episodes"):
        seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=one_outcome,
        )


def test_capture_outcomes_are_exact_geometry_derived_and_hash_sealed(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, _ = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations)
    safety = _rollout_safety(MIN_PARITY_SAMPLES)
    with patch("deployment.release.verify_candidate", return_value=candidate):
        with pytest.raises(ContractError, match="seed 45"):
            seal_parity_capture(
                capture, candidate_dir=candidate_dir, project_root=tmp_path,
                expected_dim=6, rollout_safety=safety, seed=44,
                episode_outcomes=outcomes,
            )
        failed = copy.deepcopy(outcomes)
        failed[0]["placed_success"] = False
        with pytest.raises(ContractError, match="placed_success"):
            seal_parity_capture(
                capture, candidate_dir=candidate_dir, project_root=tmp_path,
                expected_dim=6, rollout_safety=safety, seed=45,
                episode_outcomes=failed,
            )
        manual_bypass = copy.deepcopy(outcomes)
        manual_bypass[0]["target_infeasible"] = False
        with pytest.raises(ContractError, match="field set"):
            seal_parity_capture(
                capture, candidate_dir=candidate_dir, project_root=tmp_path,
                expected_dim=6, rollout_safety=safety, seed=45,
                episode_outcomes=manual_bypass,
            )
        infeasible = copy.deepcopy(outcomes)
        infeasible[0]["target_pos"] = [-0.8, 0.0, -0.0125]
        with pytest.raises(ContractError, match="canonical config geometry"):
            seal_parity_capture(
                capture, candidate_dir=candidate_dir, project_root=tmp_path,
                expected_dim=6, rollout_safety=safety, seed=45,
                episode_outcomes=infeasible,
            )
        mismatched_observation = copy.deepcopy(outcomes)
        mismatched_observation[0]["target_pos"] = [-0.5, 0.0, -0.0125]
        with pytest.raises(ContractError, match="not bound to the reported outcome"):
            seal_parity_capture(
                capture, candidate_dir=candidate_dir, project_root=tmp_path,
                expected_dim=6, rollout_safety=safety, seed=45,
                episode_outcomes=mismatched_observation,
            )
        manifest_path = seal_parity_capture(
            capture, candidate_dir=candidate_dir, project_root=tmp_path,
            expected_dim=6, rollout_safety=safety, seed=45,
            episode_outcomes=outcomes,
        )
        sealed = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert sealed["episode_outcomes"]["episode_count"] == MIN_PHASE3_EPISODES
        assert sealed["capture"]["episode_outcomes_sha256"] == hashlib.sha256(
            json.dumps(
                sealed["episode_outcomes"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        sealed["episode_outcomes"]["episodes"][0]["placed_success"] = False
        manifest_path.write_text(json.dumps(sealed), encoding="utf-8")
        with pytest.raises(ContractError, match="outcome"):
            verify_parity_capture(
                manifest_path, candidate_dir=candidate_dir, project_root=tmp_path,
                expected_dim=6,
            )


def test_phase3_geometry_yaml_is_anchored_to_candidate_config_hash(
    tmp_path, monkeypatch,
):
    candidate_dir, candidate, capture, _ = _capture_test_context(tmp_path)
    common_path = tmp_path / "configs/common.yaml"
    trusted = common_path.read_bytes()
    attacker = trusted.replace(b"tcp_pos_scale: 1.0", b"tcp_pos_scale: 2.0")
    attacker_path = tmp_path / "attacker-common.yaml"
    attacker_path.write_bytes(attacker)
    real_open = Path.open
    attacked = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal attacked
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == common_path and mode == "rb" and not attacked:
            attacked = True
            opened = real_open(attacker_path, "rb")
            with real_open(common_path, "wb") as stream:
                stream.write(trusted)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    with patch("deployment.release.verify_candidate", return_value=candidate), \
            pytest.raises(ContractError, match="sha256 mismatch"):
        seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=[],
        )
    assert attacked is True


def test_capture_runtime_authority_uses_one_exact_sidecar_snapshot(
    tmp_path, monkeypatch,
):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, _ = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations)
    with patch("deployment.release.verify_candidate", return_value=candidate):
        manifest = seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )

    authority_path = capture / "runtime_authority.json"
    trusted = authority_path.read_bytes()
    decoded = json.loads(trusted.decode("utf-8"))
    alternate_bytes = _canonical_json_bytes(decoded)
    assert hashlib.sha256(alternate_bytes).hexdigest() != hashlib.sha256(trusted).hexdigest()
    alternate = tmp_path / "alternate-runtime-authority.json"
    alternate.write_bytes(alternate_bytes)
    real_open = Path.open
    attacked = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal attacked
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == authority_path and mode == "rb" and not attacked:
            attacked = True
            opened = real_open(alternate, "rb")
            with real_open(authority_path, "wb") as stream:
                stream.write(trusted)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    with patch("deployment.release.verify_candidate", return_value=candidate), \
            pytest.raises(ContractError, match="sidecar differs from the capture manifest"):
        verify_parity_capture(
            manifest,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
        )
    assert attacked is True
    assert authority_path.read_bytes() == trusted


def test_capture_rejects_command_dump_tamper_after_sealing(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, episode = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations)
    dump = episode / "step_0000.npz"
    with patch("deployment.release.verify_candidate", return_value=candidate):
        manifest = seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )

    _write_capture_dump(dump, observations[:100], (1, 0, 0, 0, 0, 0))
    with patch("deployment.release.verify_candidate", return_value=candidate):
        with pytest.raises(ContractError, match="hash/shape drift"):
            verify_parity_capture(
                manifest,
                candidate_dir=candidate_dir,
                project_root=tmp_path,
                expected_dim=6,
            )


def test_capture_rejects_manual_report_that_disagrees_with_dumps(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, episode = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations, (1, 0, 0, 0, 0, 0))
    with patch("deployment.release.verify_candidate", return_value=candidate):
        with pytest.raises(ContractError, match="reported rollout safety differs"):
            seal_parity_capture(
                capture,
                candidate_dir=candidate_dir,
                project_root=tmp_path,
                expected_dim=6,
                rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
                seed=PHASE3_CAPTURE_SEED,
                episode_outcomes=outcomes,
            )


def test_capture_rejects_receive_only_dump_rows(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, episode = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations)
    raw = np.zeros((100, 6), dtype=np.float64)
    np.savez(
        episode / "step_0000.npz",
        obs=observations[:100],
        raw_arm_command=raw,
        safety_arm_command=raw,
        arm_command=raw,
        command_issued=np.zeros(100, dtype=np.bool_),
        arm_command_receipt_sequence=np.arange(1, 101, dtype=np.int64),
    )
    with patch("deployment.release.verify_candidate", return_value=candidate):
        with pytest.raises(ContractError, match="successfully issued"):
            seal_parity_capture(
                capture,
                candidate_dir=candidate_dir,
                project_root=tmp_path,
                expected_dim=6,
                rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
                seed=PHASE3_CAPTURE_SEED,
                episode_outcomes=outcomes,
            )


def test_capture_recalculation_uses_same_fixed_tolerance_as_rollout(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    candidate_dir, candidate, capture, episode = _capture_test_context(tmp_path)
    outcomes = _write_valid_phase3_capture(capture, observations)
    raw = np.zeros((100, 6), dtype=np.float64)
    issued = raw.copy()
    issued[:, 0] = 0.5e-12  # ignored on every row
    issued[0, 1] = 2.0e-12  # one counted clamp
    np.savez(episode / "step_0000.npz", obs=observations[:100], raw_arm_command=raw,
             safety_arm_command=raw, arm_command=issued,
             command_issued=np.ones(100, dtype=np.bool_),
             arm_command_receipt_sequence=np.arange(1, 101, dtype=np.int64))
    expected = _rollout_safety(MIN_PARITY_SAMPLES, (0, 1, 0, 0, 0, 0))
    with patch("deployment.release.verify_candidate", return_value=candidate):
        manifest = seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=expected,
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )
    assert json.loads(manifest.read_text(encoding="utf-8"))["rollout_safety"] == expected


def test_parity_loader_accepts_deterministically_varied_rows(tmp_path):
    path = tmp_path / "varied.npy"
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    observations[:, 0] = np.linspace(-1.0, 1.0, MIN_PARITY_SAMPLES, dtype=np.float32)
    np.save(path, observations)
    loaded = load_parity_observations(path, 6)
    assert tuple(loaded.shape) == (MIN_PARITY_SAMPLES, 6)


def test_parity_loader_rejects_sparse_single_outlier_fixture(tmp_path):
    path = tmp_path / "outlier.npy"
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    observations[-1, :] = 1.0
    np.save(path, observations)
    with pytest.raises(ContractError, match="row diversity"):
        load_parity_observations(path, 6)


def test_parity_dataset_requires_preexisting_verified_sidecar(tmp_path):
    observations = np.zeros((MIN_PARITY_SAMPLES, 6), dtype=np.float32)
    observations[:, 3] = np.linspace(-1.0, 1.0, MIN_PARITY_SAMPLES, dtype=np.float32)
    _write_phase3_geometry(tmp_path)
    dataset_dir = tmp_path / "logs/parity"
    dataset_dir.mkdir(parents=True)
    dataset = dataset_dir / "student.npy"
    layout = [
        {"name": "target_pos", "size": 3},
        {"name": "observations", "size": 3},
    ]
    run = _training_run(canonical_configs=_phase3_canonical_hashes(tmp_path))
    candidate_dir = tmp_path / "logs/export_candidates" / run["checkpoint_sha256"][:16]
    candidate_dir.mkdir(parents=True)
    candidate = {
        "source": {"training_run": run},
        "files": {"policy.pt": "1" * 64, "obs_spec.json": "2" * 64},
        "observation_contract": {
            "cfg_hash": "a" * 16,
            "observation_dim": 6,
            "layout": layout,
            "action_mode": "ik_rel",
            "action_dim": 7,
        },
        "rollout_safety_contract": rollout_safety_contract(),
        "configs": run["source"]["canonical_configs"],
    }
    (candidate_dir / CANDIDATE_MANIFEST_NAME).write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    binding = candidate_receipt_binding(candidate_dir, candidate, project_root=tmp_path)
    capture = capture_output_path(
        tmp_path, binding["manifest_sha256"], "20260814T010203Z-deadbeef"
    )
    capture.mkdir(parents=True)
    _write_capture_owner(capture, binding, candidate)
    outcomes = _write_valid_phase3_capture(capture, observations)
    # The capture helper writes each episode's canonical target into the
    # observation rows.  Publish the parity tensor from those exact sealed
    # source rows, not from the pre-capture template.
    np.save(dataset, observations)
    tensor = load_parity_observations(dataset, 6)
    # A stale sibling must never enter the manifest-enumerated capture.
    stale = capture.parent / "20260814T010204Z-cafebabe" / "ep000"
    stale.mkdir(parents=True)
    _write_capture_dump(stale / "step_0000.npz", np.full_like(observations, 0.5))
    with patch("deployment.release.verify_candidate", return_value=candidate):
        capture_manifest = seal_parity_capture(
            capture,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
            expected_dim=6,
            rollout_safety=_rollout_safety(MIN_PARITY_SAMPLES),
            seed=PHASE3_CAPTURE_SEED,
            episode_outcomes=outcomes,
        )
    with pytest.raises(ContractError, match="sidecar"):
        with patch("deployment.release.verify_candidate", return_value=candidate):
            parity_dataset_provenance(
                dataset, tensor, layout=layout, obs_cfg_hash="a" * 16,
                training_run=run, candidate_dir=candidate_dir,
                capture_manifest_path=capture_manifest, project_root=tmp_path,
            )

    with patch("deployment.release.verify_candidate", return_value=candidate):
        source_summary, source_files = scan_parity_source_dumps(
            capture_manifest,
            expected_dim=6,
            candidate_dir=candidate_dir,
            project_root=tmp_path,
        )
    assert len(source_files) == MIN_PHASE3_EPISODES
    assert source_files[0] == (capture / "ep000/step_0000.npz").resolve()
    payload = {
        "schema_version": PARITY_SIDECAR_SCHEMA_VERSION,
        "record_type": "ur5e_parity_dataset",
        "dataset": {
            "path": "logs/parity/student.npy",
            "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "bytes": dataset.stat().st_size,
            "format": "npy",
            "rows": MIN_PARITY_SAMPLES,
            "observation_dim": 6,
        },
        "observation_contract": {
            "cfg_hash": "a" * 16,
            "layout": layout,
            "action_mode": "ik_rel",
        },
        "training": {
            "receipt_sha256": run["receipt_sha256"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "checkpoint_name": run["checkpoint_name"],
            "run_root": run["run_root"],
        },
        "candidate": binding,
        "source_dumps": source_summary,
    }
    sidecar = parity_sidecar_path(dataset)
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    actor = SevenZeroPolicy()
    poisoned_scripted = torch.jit.trace(PoisonGatePolicy(), torch.zeros(1, 6))
    original_gate_column = tensor[:, 3].clone()
    tensor[:, 3].fill_(0.5)
    forged_metrics = compare_actor_and_export(actor, poisoned_scripted, tensor)
    assert forged_metrics["max_abs_error"] == 0.0
    tensor[:, 3].copy_(original_gate_column)
    with patch("deployment.release.verify_candidate", return_value=candidate):
        sealed = parity_dataset_provenance(
            dataset, tensor, layout=layout, obs_cfg_hash="a" * 16,
            training_run=run, candidate_dir=candidate_dir,
            capture_manifest_path=capture_manifest, project_root=tmp_path,
        )
    assert sealed["source_dump_file_count"] == MIN_PHASE3_EPISODES
    assert sealed["dataset_sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()

    # The legacy split calls above can authenticate A after measuring writable
    # mmap bytes B.  The final export authority re-reads A only once and feeds
    # private copies of those exact authenticated rows to both policies.
    with pytest.raises(ContractError, match="TorchScript parity error"):
        with patch("deployment.release.verify_candidate", return_value=candidate):
            evaluate_sealed_parity_dataset(
                dataset, actor, poisoned_scripted,
                expected_dim=6, layout=layout, obs_cfg_hash="a" * 16,
                training_run=run, candidate_dir=candidate_dir,
                capture_manifest_path=capture_manifest, project_root=tmp_path,
            )
    trusted_scripted = torch.jit.trace(SevenZeroPolicy(), torch.zeros(1, 6))
    with patch("deployment.release.verify_candidate", return_value=candidate):
        authoritative, latency = evaluate_sealed_parity_dataset(
            dataset, actor, trusted_scripted,
            expected_dim=6, layout=layout, obs_cfg_hash="a" * 16,
            training_run=run, candidate_dir=candidate_dir,
            capture_manifest_path=capture_manifest, project_root=tmp_path,
        )
    assert authoritative["dataset_sha256"] == payload["dataset"]["sha256"]
    assert authoritative["samples"] == MIN_PARITY_SAMPLES
    assert authoritative["max_abs_error"] == 0.0
    assert latency["samples"] == MIN_LATENCY_SAMPLES

    # Coordinated dataset+sidecar rehashing must not sever the actual capture
    # content link.  Keep B diverse/representative so content equality is the
    # rejecting gate, and close mmap owners before Windows replaces the file.
    close_parity_observations(tensor)
    del tensor
    tampered = np.zeros_like(observations)
    tampered[:, 3] = np.linspace(
        0.25, 0.75, MIN_PARITY_SAMPLES, dtype=np.float32
    )
    assert float(np.max(np.abs(tampered - observations))) == pytest.approx(1.25)
    np.save(dataset, tampered)
    payload["dataset"]["sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    payload["dataset"]["bytes"] = dataset.stat().st_size
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    tampered_tensor = load_parity_observations(dataset, 6)
    try:
        with pytest.raises(ContractError, match=r"max_abs_diff=1\.25"):
            with patch("deployment.release.verify_candidate", return_value=candidate):
                parity_dataset_provenance(
                    dataset, tampered_tensor, layout=layout, obs_cfg_hash="a" * 16,
                    training_run=run, candidate_dir=candidate_dir,
                    capture_manifest_path=capture_manifest, project_root=tmp_path,
                )
    finally:
        close_parity_observations(tampered_tensor)
        del tampered_tensor

    np.save(dataset, observations)
    payload["dataset"]["sha256"] = hashlib.sha256(dataset.read_bytes()).hexdigest()
    payload["dataset"]["bytes"] = dataset.stat().st_size
    payload["training"]["receipt_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    tensor = load_parity_observations(dataset, 6)
    with pytest.raises(ContractError, match="selected checkpoint/run receipt"):
        try:
            with patch("deployment.release.verify_candidate", return_value=candidate):
                parity_dataset_provenance(
                    dataset, tensor, layout=layout, obs_cfg_hash="a" * 16,
                    training_run=run, candidate_dir=candidate_dir,
                    capture_manifest_path=capture_manifest, project_root=tmp_path,
                )
        finally:
            close_parity_observations(tensor)


def test_export_latency_requires_10000_real_rows_but_deployment_smoke_may_repeat_one():
    scripted = torch.jit.trace(ZeroPolicy(), torch.zeros(1, 12))
    with pytest.raises(ContractError, match="observation rows"):
        benchmark_policy(scripted, torch.zeros(1, 12))
    measured = benchmark_policy(
        scripted,
        torch.zeros(1, 12),
        allow_repeated_rows=True,
    )
    assert measured["samples"] == MIN_LATENCY_SAMPLES


def test_parity_requires_exact_7d_outputs():
    obs = torch.zeros(MIN_PARITY_SAMPLES, 12)
    actor = BadPolicy()
    scripted = torch.jit.trace(BadPolicy(), torch.zeros(1, 12))
    with pytest.raises(ContractError, match="shape mismatch"):
        compare_actor_and_export(actor, scripted, obs)


def test_student_release_binds_runtime_and_artifacts(tmp_path):
    common = load_yaml("configs/common.yaml")
    pre = build_preprocess_cfg(common)
    action = build_action_cfg(common)
    dim = obs_dim(pre)
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    policy = torch.jit.trace(ZeroPolicy(), torch.zeros(1, dim))
    policy_path = release_dir / "policy.pt"
    policy.save(str(policy_path))
    spec_path = release_dir / "obs_spec.json"
    checkpoint = tmp_path / "model_100.pt"
    checkpoint.write_bytes(b"student checkpoint")
    spec = make_obs_spec(pre, action, action_scheme="ik_rel")
    spec.update({
        "empirical_normalization": False,
        "model_role": "student",
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "source_checkpoint": checkpoint.name,
    })
    save_obs_spec(spec_path, spec)
    config_paths = []
    for relative in sorted(REQUIRED_RELEASE_CONFIGS):
        config = tmp_path / relative
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"source: {relative}\n", encoding="utf-8")
        config_paths.append(config)
    _write_approved_release_calibrations(tmp_path)
    _write_phase3_reach_test_geometry(tmp_path)
    canonical_configs = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in config_paths
    }
    environment = {
        "python": "3.11.15", "platform": "test", "system": "Windows", "machine": "AMD64",
        "torch": "2.7.0+cu128", "numpy": "2.2.6", "pyyaml": "6.0.2",
        "torchvision": "0.22.0+cu128", "torchaudio": "2.7.0+cu128", "rsl-rl-lib": "3.1.2",
    }
    training_run = _training_run(
        checkpoint.name,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        canonical_configs=canonical_configs,
    )
    with patch("deployment.release.package_versions", return_value=environment):
        manifest = make_release_manifest(
            policy_path=policy_path,
            obs_spec_path=spec_path,
            checkpoint_path=checkpoint,
            actor_kind="student",
            config_paths=config_paths,
            config_root=tmp_path,
                parity=_parity_evidence(
                    dim,
                    spec["layout"],
                    training_run,
                    spec["cfg_hash"],
                    hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                    hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                    tmp_path,
                ),
            latency={"samples": MIN_LATENCY_SAMPLES, "p99_ms": 1.0},
            max_inference_p99_ms=10.0,
            git_commit="a" * 40,
            control_hz=20.0,
            horizon_steps=100,
            training_stack=_training_stack(),
            training_run=training_run,
        )
    save_release_manifest(release_dir / "release_manifest.json", manifest)
    release_manifest_sha256 = hashlib.sha256(
        (release_dir / "release_manifest.json").read_bytes()
    ).hexdigest()
    checked = verify_release(
        release_dir,
        config_root=tmp_path,
        expected_obs_dim=dim,
        expected_manifest_sha256=release_manifest_sha256,
    )
    with pytest.raises(ContractError, match="release manifest hash mismatch"):
        verify_release(
            release_dir,
            config_root=tmp_path,
            expected_obs_dim=dim,
            expected_manifest_sha256="0" * 64,
        )
    assert checked["model_role"] == "student"
    assert checked["runtime_contract"]["action_dim"] == 7
    assert checked["runtime_contract"]["horizon_seconds"] == 5.0
    assert checked["files"] == manifest["files"]
    assert checked["training_stack"]["isaac_lab"]["release_version"] == "2.3.2"
    assert checked["source"]["training_run"]["configs"]["params/agent.yaml"] == "a" * 64
    assert "C:/Users" not in json.dumps(checked["training_stack"])

    # Rehashing every downstream reference cannot turn a different execution
    # role into Phase 3 authority.  The full manifest/sentinel remain
    # semantically checked against the fixed phase3-rehearsal boundary.
    rebound = copy.deepcopy(manifest)
    receipt = rebound["parity"]["source_runtime_authority"]
    receipt["runtime_manifest"]["purpose"] = "evaluation"
    rebound_manifest_sha = hashlib.sha256(
        _canonical_json_bytes(receipt["runtime_manifest"])
    ).hexdigest()
    receipt["runtime_manifest_sha256"] = rebound_manifest_sha
    receipt["binding"]["purpose"] = "evaluation"
    receipt["binding"]["manifest_sha256"] = rebound_manifest_sha
    receipt["sentinel"]["binding"] = copy.deepcopy(receipt["binding"])
    receipt["sentinel_sha256"] = hashlib.sha256(
        _canonical_json_bytes(receipt["sentinel"])
    ).hexdigest()
    rebound_authority_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    rebound["parity"]["source_runtime_authority_sha256"] = hashlib.sha256(
        rebound_authority_bytes
    ).hexdigest()
    rebound["parity"]["source_runtime_authority_bytes"] = len(
        rebound_authority_bytes
    )
    rebound["parity"]["source_capture_manifest_sha256"] = "4" * 64
    rebound["parity"]["dataset_sidecar_sha256"] = "5" * 64
    save_release_manifest(release_dir / "release_manifest.json", rebound)
    with pytest.raises(ContractError, match="runtime authority is invalid"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)
    save_release_manifest(release_dir / "release_manifest.json", manifest)

    # Reproduce the former release authority split: each path hash sees the
    # trusted artifact, then a different (still schema/shape-valid) payload is
    # exposed only to path-based JSON/JIT loading before the trusted bytes are
    # restored.  The hardened verifier never performs either second open.
    manifest_path = release_dir / "release_manifest.json"
    trusted_policy_bytes = policy_path.read_bytes()
    trusted_spec_bytes = spec_path.read_bytes()
    swapped_spec_bytes = (json.dumps(spec, indent=4) + "\n").encode("utf-8")
    assert hashlib.sha256(swapped_spec_bytes).hexdigest() != manifest["files"]["obs_spec.json"]
    alternate_policy_path = tmp_path / "alternate-policy.pt"
    torch.jit.trace(OnePolicy(), torch.zeros(1, dim)).save(str(alternate_policy_path))
    swapped_policy_bytes = alternate_policy_path.read_bytes()
    assert hashlib.sha256(swapped_policy_bytes).hexdigest() != manifest["files"]["policy.pt"]
    real_sha256_file = __import__("deployment.release", fromlist=["sha256_file"]).sha256_file
    real_read_text = Path.read_text
    real_jit_load = torch.jit.load
    manifest_path_read = False
    obs_path_read = False
    policy_path_load = False

    def old_split_release_hash(path):
        target = Path(path)
        if target == policy_path:
            digest = real_sha256_file(policy_path)
            policy_path.write_bytes(swapped_policy_bytes)
            return digest
        if target == spec_path:
            digest = real_sha256_file(spec_path)
            spec_path.write_bytes(swapped_spec_bytes)
            return digest
        return real_sha256_file(path)

    def old_split_release_json(path: Path, *args, **kwargs):
        nonlocal manifest_path_read, obs_path_read
        if path == manifest_path:
            manifest_path_read = True
        value = real_read_text(path, *args, **kwargs)
        if path == spec_path:
            obs_path_read = True
            spec_path.write_bytes(trusted_spec_bytes)
        return value

    def old_split_release_policy(source, *args, **kwargs):
        nonlocal policy_path_load
        if isinstance(source, (str, Path)):
            policy_path_load = True
            loaded = real_jit_load(source, *args, **kwargs)
            policy_path.write_bytes(trusted_policy_bytes)
            return loaded
        return real_jit_load(source, *args, **kwargs)

    try:
        with patch("deployment.release.sha256_file", side_effect=old_split_release_hash), \
                patch.object(Path, "read_text", old_split_release_json), \
                patch("deployment.release.torch.jit.load", side_effect=old_split_release_policy):
            same_open_checked = verify_release(
                release_dir, config_root=tmp_path, expected_obs_dim=dim
            )
    finally:
        policy_path.write_bytes(trusted_policy_bytes)
        spec_path.write_bytes(trusted_spec_bytes)
    assert same_open_checked["files"] == manifest["files"]
    assert manifest_path_read is False
    assert obs_path_read is False
    assert policy_path_load is False

    # Hash rebinding is provenance, not approval: even an attacker who edits a
    # tracked calibration and recomputes every manifest/config hash cannot turn
    # calibrated:false/null semantics into a final release.
    tcp_path = tmp_path / "configs/calib/tcp_offset.yaml"
    approved_tcp = tcp_path.read_bytes()
    tcp_data = yaml.safe_load(approved_tcp.decode("utf-8"))
    tcp_data["meta"]["calibrated"] = False
    tcp_data["meta"]["residual_m"] = None
    tcp_path.write_text(yaml.safe_dump(tcp_data, sort_keys=False), encoding="utf-8")
    rebound_hash = hashlib.sha256(tcp_path.read_bytes()).hexdigest()
    invalid = copy.deepcopy(manifest)
    invalid["configs"]["configs/calib/tcp_offset.yaml"] = rebound_hash
    invalid["source"]["training_run"]["source"]["canonical_configs"][
        "configs/calib/tcp_offset.yaml"
    ] = rebound_hash
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="not calibrated"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)
    tcp_path.write_bytes(approved_tcp)
    save_release_manifest(release_dir / "release_manifest.json", manifest)

    invalid = copy.deepcopy(manifest)
    invalid["configs"].pop("configs/dr.yaml")
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="config set"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)
    save_release_manifest(release_dir / "release_manifest.json", manifest)

    for field, tampered in (
        ("checkpoint_sha256", "0" * 64),
        ("checkpoint_name", "model_other.pt"),
        ("git_commit", "b" * 40),
    ):
        invalid = copy.deepcopy(manifest)
        invalid["source"][field] = tampered
        save_release_manifest(release_dir / "release_manifest.json", invalid)
        with pytest.raises(ContractError, match="provenance mismatch"):
            verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)
    invalid = copy.deepcopy(manifest)
    invalid["source"]["unexpected"] = "not allowed"
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="provenance mismatch"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)
    save_release_manifest(release_dir / "release_manifest.json", manifest)

    extra = release_dir / "extra.bin"
    extra.write_bytes(b"unexpected")
    with pytest.raises(ContractError, match="file set mismatch"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)
    extra.unlink()

    invalid = copy.deepcopy(manifest)
    invalid["environment"]["numpy"] = "not-installed"
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="exact non-placeholder"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)

    invalid = copy.deepcopy(manifest)
    invalid["parity"]["max_abs_error"] = float("nan")
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="finite"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)

    invalid = copy.deepcopy(manifest)
    invalid["parity"]["source_rollout_safety"]["clamp_counts"][0] = 1
    invalid["parity"]["source_rollout_safety"]["clamp_fractions"][0] = 0.0001
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="evidence hash mismatch"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)

    invalid = copy.deepcopy(manifest)
    outcomes = invalid["parity"]["source_episode_outcomes"]
    outcomes["episodes"][0]["placed_success"] = False
    # Even an attacker who recomputes the outer digest cannot turn a failed
    # per-episode result into valid final-release evidence.
    invalid["parity"]["source_episode_outcomes_sha256"] = hashlib.sha256(
        json.dumps(outcomes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="episode outcome"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)

    invalid = copy.deepcopy(manifest)
    outcomes = invalid["parity"]["source_episode_outcomes"]
    # This remains inside the test's trained range, support polygon and safety
    # workspace; only the canonical UR5e reach check can reject it.
    outcomes["episodes"][0]["target_pos"] = [0.9, 0.0, -0.0125]
    invalid["parity"]["source_episode_outcomes_sha256"] = hashlib.sha256(
        json.dumps(outcomes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="canonical config geometry"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)

    invalid = copy.deepcopy(manifest)
    invalid["latency_contract"]["export_host"]["p99_ms"] = float("nan")
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="finite"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)

    bad_spec = dict(spec)
    bad_spec["model_role"] = "teacher"
    save_obs_spec(spec_path, bad_spec)
    invalid = copy.deepcopy(manifest)
    invalid["files"]["obs_spec.json"] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="obs_spec student"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)

    save_obs_spec(spec_path, spec)
    invalid = copy.deepcopy(manifest)
    invalid["training_stack"]["isaac_lab"]["release_version"] = "0.54.2"
    invalid["files"]["obs_spec.json"] = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    save_release_manifest(release_dir / "release_manifest.json", invalid)
    with pytest.raises(ContractError, match="Isaac Lab"):
        verify_release(release_dir, config_root=tmp_path, expected_obs_dim=dim)


@pytest.mark.parametrize("placeholder", ["", "not-installed", "unknown", "None", "null"])
def test_release_generation_rejects_environment_placeholders(tmp_path, placeholder):
    dummy = tmp_path / "x"
    dummy.write_bytes(b"x")
    environment = {
        "python": "3.11.15", "platform": "test", "system": "Windows", "machine": "AMD64",
        "torch": "2.7.0+cu128", "numpy": "2.2.6", "pyyaml": placeholder,
        "torchvision": "0.22.0+cu128", "torchaudio": "2.7.0+cu128", "rsl-rl-lib": "3.1.2",
    }
    with patch("deployment.release.package_versions", return_value=environment):
        with pytest.raises(ContractError, match="exact non-placeholder"):
            make_release_manifest(
                policy_path=dummy, obs_spec_path=dummy, checkpoint_path=dummy,
                actor_kind="student", config_paths=[dummy], config_root=tmp_path,
                parity={"samples": 10_000, "max_abs_error": 0.0,
                        "mean_abs_error": 0.0, "threshold": 1.0e-5},
                latency={"samples": MIN_LATENCY_SAMPLES, "p99_ms": 1.0},
                max_inference_p99_ms=10.0,
                git_commit="a" * 40, control_hz=20.0, horizon_steps=100,
            )


def test_teacher_release_is_forbidden(tmp_path):
    dummy = tmp_path / "x"
    dummy.write_bytes(b"x")
    with pytest.raises(ContractError, match="only student"):
        make_release_manifest(
            policy_path=dummy, obs_spec_path=dummy, checkpoint_path=dummy,
            actor_kind="teacher", config_paths=[dummy], config_root=tmp_path,
            parity={"samples": 10_000, "max_abs_error": 0.0},
            latency={"p99_ms": 1.0}, max_inference_p99_ms=10.0,
            git_commit="a" * 40, control_hz=20.0, horizon_steps=100,
        )


def test_release_rejects_unknown_git_commit(tmp_path):
    dummy = tmp_path / "x"
    dummy.write_bytes(b"x")
    with pytest.raises(ContractError, match="40-hex"):
        make_release_manifest(
            policy_path=dummy, obs_spec_path=dummy, checkpoint_path=dummy,
            actor_kind="student", config_paths=[dummy], config_root=tmp_path,
            parity={"samples": 10_000, "max_abs_error": 0.0}, latency={"p99_ms": 1.0},
            max_inference_p99_ms=10.0, git_commit="unknown", control_hz=20.0,
            horizon_steps=100,
        )


def test_git_commit_provenance_rejects_dirty_tree(tmp_path):
    from deployment.release import git_commit_or_unknown

    with patch("subprocess.check_output", return_value=" M deployment/release.py\n") as command:
        assert git_commit_or_unknown(tmp_path) == "unknown"
        command.assert_called_once()


def test_export_captures_clean_commit_before_writing_artifacts():
    source = Path("Phase2/Export_Policy.py").read_text(encoding="utf-8")
    capture = source.index("source_git_commit = require_clean_git_commit(ROOT)")
    first_write = source.index("out_dir = create_release_staging(requested_out_dir)")
    manifest = source.index("git_commit=source_git_commit")
    assert capture < first_write < manifest
    assert "git_commit=git_commit_or_unknown(ROOT)" not in source
    assert "requested_out_dir = Path(args.out_dir).resolve()" not in source
    assert "requested = Path(args.out_dir or (ROOT / \"exported\")).expanduser()" in source


def test_export_calibration_gate_is_final_only_and_precedes_artifact_staging():
    source = Path("Phase2/Export_Policy.py").read_text(encoding="utf-8")
    final_only = source.index("if not args.candidate_only:", source.index("config_paths = ["))
    calibration = source.index("validate_release_calibrations(ROOT)", final_only)
    staging = source.index("out_dir = create_release_staging(requested_out_dir)", calibration)
    assert final_only < calibration < staging
    assert source.count("validate_release_calibrations(ROOT)") == 1


def test_final_export_benchmarks_only_manifest_anchored_candidate_bytes():
    source = Path("Phase2/Export_Policy.py").read_text(encoding="utf-8")
    final_branch = source.index(
        'candidate_policy_bytes = read_anchored_regular_file('
    )
    load = source.index("torch.jit.load(", final_branch)
    byte_stream = source.index("io.BytesIO(candidate_policy_bytes)", load)
    publish = source.index('with policy_path.open("xb") as stream:', byte_stream)
    parity = source.index("evaluate_sealed_parity_dataset(", publish)
    assert final_branch < load < byte_stream < publish < parity
    final_body = source[final_branch:parity]
    assert "torch.jit.load(str(policy_path)" not in final_body
    assert "shutil.copy2(candidate_path" not in final_body
    assert 'candidate_spec_bytes = read_anchored_regular_file(' in final_body
    release_body = source[parity:source.index("make_release_manifest(", parity)]
    assert "load_parity_observations(" not in release_body
    assert "compare_actor_and_export(" not in release_body
    assert "parity_dataset_provenance(" not in release_body
    assert "benchmark_policy(" not in release_body


def test_final_export_rechecks_exact_candidate_manifest_byte_identity():
    from deployment.release import _VerifiedCandidateManifest

    # Dict equality alone cannot carry a manifest's byte identity.
    first = _VerifiedCandidateManifest({"same": True}, manifest_sha256="a" * 64)
    second = _VerifiedCandidateManifest({"same": True}, manifest_sha256="b" * 64)
    assert first == second
    assert first._verified_manifest_sha256 != second._verified_manifest_sha256

    source = Path("Phase2/Export_Policy.py").read_text(encoding="utf-8")
    initial_binding = source.index(
        "selected_candidate_binding = candidate_receipt_binding("
    )
    parity = source.index("evaluate_sealed_parity_dataset(", initial_binding)
    parity_binding = source.index(
        'parity.get("candidate_manifest_sha256")', parity
    )
    final_verify = source.index("current_candidate = verify_candidate(", parity_binding)
    final_binding = source.index(
        "current_candidate_binding = candidate_receipt_binding(", final_verify
    )
    exact_compare = source.index(
        "current_candidate_binding != selected_candidate_binding", final_binding
    )
    assert initial_binding < parity < parity_binding < final_verify < final_binding < exact_compare


def test_release_output_is_restricted_to_ignored_project_exported_tree(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitignore").write_text(
        "exported/\n.exported.staging-*/\n", encoding="utf-8"
    )
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    canonical = require_canonical_release_output(project / "exported", project)
    assert canonical == (project / "exported").resolve()
    with pytest.raises(ContractError, match="exactly"):
        require_canonical_release_output(project / "elsewhere", project)


def _init_clean_export_git(project: Path, ignore_rules: str) -> str:
    import subprocess

    project.mkdir()
    (project / ".gitignore").write_text(ignore_rules, encoding="utf-8")
    (project / "tracked.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=project, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=project, check=True)
    subprocess.run(["git", "add", "."], cwd=project, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=project, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()


def test_final_release_staging_artifacts_do_not_dirty_clean_head(tmp_path):
    from deployment.release import require_clean_git_commit

    project = tmp_path / "project"
    head = _init_clean_export_git(
        project, "exported/\n.exported.staging-*/\n"
    )
    output = require_canonical_release_output(project / "exported", project)
    staging = create_release_staging(output)
    for name in ("policy.pt", "obs_spec.json", "release_manifest.json"):
        (staging / name).write_text(name, encoding="utf-8")
    assert require_clean_git_commit(project) == head
    cleanup_release_staging(staging, output)


def test_candidate_staging_artifacts_do_not_dirty_clean_head(tmp_path):
    from deployment.release import require_clean_git_commit

    project = tmp_path / "project"
    head = _init_clean_export_git(project, "**/logs/\n")
    output = require_canonical_candidate_output(project, "a" * 64)
    staging = create_release_staging(output)
    for name in ("policy.pt", "obs_spec.json", CANDIDATE_MANIFEST_NAME):
        (staging / name).write_text(name, encoding="utf-8")
    assert require_clean_git_commit(project) == head
    cleanup_release_staging(staging, output)


def test_release_gate_rejects_unignored_staging_json(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".gitignore").write_text("exported/\n", encoding="utf-8")
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    with pytest.raises(ContractError, match="not ignored"):
        require_canonical_release_output(project / "exported", project)


@pytest.mark.parametrize(
    "target",
    ["scripts_builder_candidate", "scripts_builder_capture", "release_candidate", "release_capture"],
)
def test_parity_cli_aliases_are_rejected_before_resolution(tmp_path, monkeypatch, target):
    from scripts import build_parity_dataset as builder

    candidate = tmp_path / "logs/export_candidates" / ("a" * 16)
    capture = (
        tmp_path / "Phase3/logs/obs_dump_sim" / ("b" * 16)
        / "20260814T010203Z-deadbeef/capture_manifest.json"
    )
    candidate.mkdir(parents=True)
    capture.parent.mkdir(parents=True)
    capture.write_text("{}", encoding="utf-8")
    output = tmp_path / "logs/parity/test.npy"

    if target == "scripts_builder_candidate":
        monkeypatch.setattr(builder, "ROOT", tmp_path)
        with patch("scripts.build_parity_dataset.path_contains_link_or_reparse", return_value=True):
            with pytest.raises(ContractError, match="candidate CLI path"):
                builder.build_dataset(candidate, capture, output)
    elif target == "scripts_builder_capture":
        monkeypatch.setattr(builder, "ROOT", tmp_path)
        checks = iter([False, True])
        with patch(
            "scripts.build_parity_dataset.path_contains_link_or_reparse",
            side_effect=lambda _path: next(checks),
        ), patch.object(builder, "verify_candidate", side_effect=ContractError("should not reach")):
            # Candidate validation occurs before capture validation, so mock the
            # verified candidate only far enough to exercise the raw capture gate.
            fake = {
                "observation_contract": {"observation_dim": 6},
                "files": {"policy.pt": "1" * 64, "obs_spec.json": "2" * 64},
            }
            with patch.object(builder, "verify_candidate", return_value=fake), \
                    patch.object(builder, "candidate_receipt_binding", return_value={} ), \
                    patch.object(Path, "read_text", return_value='{"obs_dim":6,"action_scheme":"ik_rel"}'):
                with pytest.raises(ContractError, match="capture-manifest CLI path"):
                    builder.build_dataset(candidate, capture, output)
    elif target == "release_candidate":
        with patch("deployment.release.path_contains_link_or_reparse", return_value=True):
            with pytest.raises(ContractError, match="candidate path"):
                verify_parity_capture(
                    capture, candidate_dir=candidate, project_root=tmp_path, expected_dim=6
                )
    else:
        calls = iter([False, True])
        with patch(
            "deployment.release.path_contains_link_or_reparse",
            side_effect=lambda _path: next(calls),
        ), patch("deployment.release.verify_candidate", return_value={
            "files": {"policy.pt": "1" * 64, "obs_spec.json": "2" * 64}
        }), patch("deployment.release.candidate_receipt_binding", return_value={
            "manifest_sha256": "b" * 64
        }):
            with pytest.raises(ContractError, match="capture manifest path"):
                verify_parity_capture(
                    capture, candidate_dir=candidate, project_root=tmp_path, expected_dim=6
                )


def test_require_clean_git_commit_validates_before_export(tmp_path):
    from deployment.release import require_clean_git_commit

    with patch("deployment.release.git_commit_or_unknown", return_value="a" * 40):
        assert require_clean_git_commit(tmp_path) == "a" * 40
    with patch("deployment.release.git_commit_or_unknown", return_value="unknown"):
        with pytest.raises(ContractError, match="clean Git worktree"):
            require_clean_git_commit(tmp_path)


def test_training_stack_provenance_binds_metadata_checkout_origins_and_gpu(tmp_path):
    checkout = tmp_path / "external" / "IsaacLab"
    checkout.mkdir(parents=True)
    (checkout / "VERSION").write_text("2.3.2\n", encoding="utf-8")
    origins = {
        name: checkout / "source" / name / name / "__init__.py"
        for name in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks")
    }
    origins["ur5e_binpick"] = tmp_path / "source/ur5e_binpick/ur5e_binpick/__init__.py"
    origins["isaacsim"] = Path(sysconfig.get_path("purelib")) / "isaacsim/__init__.py"
    versions = _training_stack()["distributions"]
    git_values = [ISAAC_LAB_COMMIT, "v2.3.2", ""]
    with (
        patch("deployment.release.importlib.metadata.version", side_effect=lambda name: versions[name]),
        patch("deployment.release._module_origin", side_effect=lambda name: origins[name]),
        patch("deployment.release._git_checkout_value", side_effect=git_values),
        patch(
            "deployment.release._training_host_provenance",
            return_value=_training_stack()["host"],
        ),
        patch(
            "deployment.release._nvidia_gpu_inventory",
            return_value=_training_stack()["gpu"]["nvidia_smi"],
        ),
        patch(
            "deployment.release._torch_gpu_provenance",
            return_value=_training_stack()["gpu"]["torch_device"],
        ),
        patch.object(torch.version, "cuda", "12.8"),
    ):
        record = training_stack_provenance(tmp_path)
    assert record == _training_stack()
    assert record["isaac_lab"]["release_version"] != record["isaac_lab"]["distributions"]["isaaclab"]
    assert not Path(record["isaac_sim"]["import_origin"]).is_absolute()


def test_training_run_provenance_binds_exact_agent_and_env_snapshots(
    tmp_path, monkeypatch,
):
    store = tmp_path / "logs/rsl_rl/ur5e_binpick_vision"
    run = store / "2026-08-14_00-00-00"
    params = run / "params"
    params.mkdir(parents=True)
    checkpoint = run / "model_100.pt"
    checkpoint.write_bytes(b"checkpoint")
    (params / "agent.yaml").write_text(
        "class_name: DistillationRunner\n"
        "policy:\n  class_name: VisionStudentTeacher\n  activation: elu\n"
        "  student_obs_normalization: false\n  teacher_obs_normalization: false\n"
        "algorithm:\n  class_name: Distillation\n",
        encoding="utf-8",
    )
    (params / "env.yaml").write_text("scene:\n  num_envs: 512\n", encoding="utf-8")
    receipt = {
        "schema_version": 2,
        "receipt_type": "ur5e_training_run",
        "experiment_name": "ur5e_binpick_vision",
        "run_root": "logs/rsl_rl/ur5e_binpick_vision/2026-08-14_00-00-00",
        "task_id": "Isaac-BinPick-UR5e-Vision-Distill-v0",
        "mode": "distillation",
            "source": {
                "schema_version": 1,
                "project_git_commit": "a" * 40,
                "training_stack": _training_stack(),
                "canonical_configs": {relative: "c" * 64 for relative in REQUIRED_RELEASE_CONFIGS},
                "optional_assets": {},
            },
        "runtime_authority": {
            "schema_version": 1,
            "record_type": "ur5e_tracked_runtime_authority",
            "purpose": "vision-training",
            "manifest_sha256": "1" * 64,
            "git_commit": "a" * 40,
            "runtime_file_count": 100,
            "runtime_files_sha256": "2" * 64,
            "canonical_yaml_sha256": "3" * 64,
            "canonical_yaml": {
                relative: "c" * 64 for relative in REQUIRED_RELEASE_CONFIGS
            },
        },
        "artifacts": {
            "parameters": {
                "params/agent.yaml": hashlib.sha256((params / "agent.yaml").read_bytes()).hexdigest(),
                "params/env.yaml": hashlib.sha256((params / "env.yaml").read_bytes()).hexdigest(),
            },
            "checkpoints": {
                checkpoint.name: {
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "size": checkpoint.stat().st_size,
                }
            },
        },
    }
    (run / "run_manifest.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record = collect_training_run_provenance(checkpoint, store)
    assert record["run_root"] == "logs/rsl_rl/ur5e_binpick_vision/2026-08-14_00-00-00"
    assert record["agent_contract"]["empirical_normalization"] is False
    assert record["source"]["project_git_commit"] == "a" * 40
    assert all(len(value) == 64 for value in record["configs"].values())

    # Reproduce the old verify-A / parse-B / hash-A swap-back window.  Receipt
    # verification sees the trusted env bytes, while the subsequent pathname
    # open is redirected to a different, still-valid YAML mapping and the
    # public path is restored before any later hash.  The collector must bind
    # its YAML decoder to the receipt hash on that same second open.
    env_path = params / "env.yaml"
    trusted_env = env_path.read_bytes()
    attacker_env = b"scene:\n  num_envs: 999\n"
    trusted_copy = tmp_path / "trusted-env.yaml"
    attacker_copy = tmp_path / "attacker-env.yaml"
    trusted_copy.write_bytes(trusted_env)
    attacker_copy.write_bytes(attacker_env)
    env_path.write_bytes(attacker_env)
    real_open = Path.open
    env_opens = 0

    def swap_back_env_open(path: Path, *args, **kwargs):
        nonlocal env_opens
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == env_path and mode == "rb":
            env_opens += 1
            if env_opens == 1:
                return real_open(trusted_copy, "rb")
            if env_opens == 2:
                opened = real_open(attacker_copy, "rb")
                with real_open(env_path, "wb") as stream:
                    stream.write(trusted_env)
                return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_env_open)
    with pytest.raises(ContractError, match="snapshot.*sha256 mismatch"):
        collect_training_run_provenance(checkpoint, store)
    assert env_opens == 2

    (params / "agent.yaml").unlink()
    with pytest.raises(ContractError, match="parameter"):
        collect_training_run_provenance(checkpoint, store)


def test_release_staging_refuses_overwrite_and_publishes_exact_trio(tmp_path):
    output = tmp_path / "exported"
    output.mkdir()
    (output / "accepted.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ContractError, match="must be absent"):
        create_release_staging(output)
    (output / "accepted.txt").unlink()
    output.rmdir()
    staging = create_release_staging(output)
    policy = staging / "policy.pt"
    spec = staging / "obs_spec.json"
    policy.write_bytes(b"policy")
    spec.write_bytes(b"spec")
    manifest = {
        "files": {
            "policy.pt": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "obs_spec.json": hashlib.sha256(spec.read_bytes()).hexdigest(),
        }
    }
    manifest_sha256 = save_release_manifest(
        staging / "release_manifest.json", manifest
    )

    class Verified(dict):
        def __init__(self):
            super().__init__(manifest)
            self._verified_manifest_sha256 = manifest_sha256

    with patch("deployment.release.verify_release", return_value=Verified()):
        published = publish_release_staging(
            staging,
            output,
            config_root=tmp_path,
            expected_manifest_sha256=manifest_sha256,
        )
    assert published == output.resolve()
    assert {path.name for path in output.iterdir()} == {
        "policy.pt", "obs_spec.json", "release_manifest.json"
    }


def _write_publication_fixture(staging: Path, manifest_name: str) -> tuple[dict, str]:
    policy = staging / "policy.pt"
    spec = staging / "obs_spec.json"
    policy.write_bytes(b"trusted-policy")
    spec.write_bytes(b"trusted-spec")
    manifest = {
        "files": {
            "policy.pt": hashlib.sha256(policy.read_bytes()).hexdigest(),
            "obs_spec.json": hashlib.sha256(spec.read_bytes()).hexdigest(),
        }
    }
    digest = save_release_manifest(staging / manifest_name, manifest)
    return manifest, digest


def test_candidate_postpublish_mutation_is_quarantined(monkeypatch, tmp_path):
    output = tmp_path / "logs/export_candidates" / ("a" * 16)
    output.parent.mkdir(parents=True)
    staging = output.parent / f".{output.name}.staging-test"
    staging.mkdir()
    manifest, manifest_sha256 = _write_publication_fixture(
        staging, CANDIDATE_MANIFEST_NAME
    )

    class Verified(dict):
        def __init__(self):
            super().__init__(manifest)
            self._verified_manifest_sha256 = manifest_sha256

    real_rename = release_module._rename_directory_no_replace

    def rename_then_mutate(source, destination):
        real_rename(source, destination)
        (destination / "policy.pt").write_bytes(b"attacker-policy")

    def verify_public(candidate, **_kwargs):
        if (Path(candidate) / "policy.pt").read_bytes() != b"trusted-policy":
            raise ContractError("injected candidate postpublish mutation")
        return Verified()

    monkeypatch.setattr(
        release_module, "_rename_directory_no_replace", rename_then_mutate
    )
    monkeypatch.setattr(release_module, "verify_candidate", verify_public)
    with pytest.raises(ContractError, match="postpublish mutation"):
        publish_candidate_staging(
            staging,
            output,
            project_root=tmp_path,
            expected_manifest_sha256=manifest_sha256,
        )
    assert not output.exists()
    quarantined = list(output.parent.glob(f".{output.name}.failed-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "policy.pt").read_bytes() == b"attacker-policy"


def test_release_staging_swap_before_rename_is_quarantined(monkeypatch, tmp_path):
    output = tmp_path / "exported"
    staging = create_release_staging(output)
    manifest, manifest_sha256 = _write_publication_fixture(
        staging, "release_manifest.json"
    )

    class Verified(dict):
        def __init__(self):
            super().__init__(manifest)
            self._verified_manifest_sha256 = manifest_sha256

    def verify_at(path, **_kwargs):
        if (Path(path) / "policy.pt").read_bytes() != b"trusted-policy":
            raise ContractError("injected release staging swap")
        return Verified()

    real_rename = release_module._rename_directory_no_replace

    def mutate_then_rename(source, destination):
        (source / "policy.pt").write_bytes(b"attacker-policy")
        real_rename(source, destination)

    monkeypatch.setattr(release_module, "verify_release", verify_at)
    monkeypatch.setattr(
        release_module, "_rename_directory_no_replace", mutate_then_rename
    )
    with pytest.raises(ContractError, match="staging swap"):
        publish_release_staging(
            staging,
            output,
            config_root=tmp_path,
            expected_manifest_sha256=manifest_sha256,
        )
    assert not output.exists()
    quarantined = list(output.parent.glob(f".{output.name}.staging-failed-*"))
    assert len(quarantined) == 1
    assert (quarantined[0] / "policy.pt").read_bytes() == b"attacker-policy"


def test_owned_cleanup_never_deletes_directory_swapped_after_identity_check(
    monkeypatch, tmp_path
):
    output = tmp_path / "published"
    output.mkdir()
    (output / "ours.txt").write_text("ours", encoding="utf-8")
    expected_identity = release_module._directory_identity(output, label="fixture")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "foreign.txt").write_text("foreign", encoding="utf-8")
    displaced_owned = tmp_path / "displaced-owned"
    real_identity = release_module._directory_identity
    attacked = False

    def swap_after_identity(path, *, label):
        nonlocal attacked
        identity = real_identity(path, label=label)
        if Path(path) == output and not attacked:
            attacked = True
            output.rename(displaced_owned)
            foreign.rename(output)
        return identity

    monkeypatch.setattr(release_module, "_directory_identity", swap_after_identity)
    with pytest.raises(ContractError, match="raced with an unowned directory"):
        release_module._remove_owned_published_directory(
            output, expected_identity=expected_identity
        )
    assert attacked is True
    assert (output / "foreign.txt").read_text(encoding="utf-8") == "foreign"
    assert (displaced_owned / "ours.txt").read_text(encoding="utf-8") == "ours"


def test_release_staging_cleanup_rejects_unrelated_path(tmp_path):
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ContractError, match="validated sibling"):
        cleanup_release_staging(unrelated, tmp_path / "exported")
    assert unrelated.is_dir()
