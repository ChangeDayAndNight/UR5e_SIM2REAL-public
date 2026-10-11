from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from deployment.contracts import ContractError
from deployment.release import REQUIRED_RELEASE_CONFIGS
from common.nurec_provenance import load_nurec_provenance
from common.training_stack_contract import EXPECTED_TRAINING_DISTRIBUTIONS
from scripts import training_provenance
from scripts import evaluation_provenance
from scripts.training_provenance import (
    canonical_config_hashes,
    finalize_training_receipt,
    verify_training_checkpoint_receipt,
)


def _stack() -> dict:
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
            "git_commit": "37ddf626871758333d6ed89cf64ad702aef127d0",
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


def _context(commit: str = "a" * 40) -> dict:
    return {
        "schema_version": 1,
        "project_git_commit": commit,
        "training_stack": _stack(),
        "canonical_configs": {relative: "c" * 64 for relative in REQUIRED_RELEASE_CONFIGS},
        "optional_assets": {},
    }


def _run(tmp_path: Path, experiment: str = "ur5e_binpick_vision") -> tuple[Path, Path]:
    run = tmp_path / "logs" / "rsl_rl" / experiment / "2026-08-14_00-00-00"
    params = run / "params"
    params.mkdir(parents=True)
    (params / "agent.yaml").write_text("agent: true\n", encoding="utf-8")
    (params / "env.yaml").write_text("env: true\n", encoding="utf-8")
    checkpoint = run / "model_1.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run / "events.out.tfevents.test").write_bytes(b"tensorboard-events")
    return run, checkpoint


def _teacher_adoption(receipt_sha256: str = "f" * 64) -> dict:
    return {
        "schema_version": 1,
        "role": "teacher",
        "path": "Phase1/Result/run/top5_evaluation/adoption.json",
        "sha256": "1" * 64,
        "checkpoint_name": "model_1.pt",
        "checkpoint_sha256": "2" * 64,
        "training_receipt_sha256": receipt_sha256,
    }


def _runtime_authority(
    context: dict | None = None, *, purpose: str = "vision-training"
) -> dict:
    source = _context() if context is None else context
    return {
        "schema_version": 1,
        "record_type": "ur5e_tracked_runtime_authority",
        "purpose": purpose,
        "manifest_sha256": "1" * 64,
        "git_commit": source["project_git_commit"],
        "runtime_file_count": 100,
        "runtime_files_sha256": "2" * 64,
        "canonical_yaml_sha256": "3" * 64,
        "canonical_yaml": dict(source["canonical_configs"]),
    }


def test_finalize_receipt_is_atomic_and_binds_checkpoint_params_and_source(tmp_path):
    run, checkpoint = _run(tmp_path)
    receipt = finalize_training_receipt(
        run,
        project_root=tmp_path,
        experiment_name="ur5e_binpick_vision",
        task_id="Isaac-BinPick-UR5e-Vision-Distill-v0",
        mode="distillation",
        started_at="2026-08-14T00:00:00+00:00",
        completed_at="2026-08-14T00:01:00+00:00",
        arguments=["--distill-from", r"C:\private\teacher\model_1.pt"],
        start_context=_context(),
        end_context=_context(),
        inputs=[{
            "receipt_sha256": "f" * 64,
            "source": _context(),
            "task_id": "Isaac-BinPick-UR5e-Teacher-v0",
            "mode": "scratch",
            "adoption": _teacher_adoption(),
        }],
        runtime_authority=_runtime_authority(),
    )
    assert receipt.name == "run_manifest.json"
    assert not list(run.glob(".run-manifest-*"))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["source"]["project_git_commit"] == "a" * 40
    assert payload["arguments"][-1] == "<external>/model_1.pt"
    assert payload["artifacts"]["tensorboard_events"] == {
        "events.out.tfevents.test": {
            "sha256": hashlib.sha256(b"tensorboard-events").hexdigest(),
            "size": len(b"tensorboard-events"),
        }
    }
    verified = verify_training_checkpoint_receipt(
        checkpoint, project_root=tmp_path, expected_experiment="ur5e_binpick_vision"
    )
    assert verified["checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert payload["inputs"][0]["adoption"] == _teacher_adoption()


def test_parent_commit_drift_is_rejected_unless_explicitly_accepted(tmp_path):
    """★F95-b★ 부모(스승) commit만 다른 입력은 명시 허용 시에만 봉인된다 — 그 외는 그대로 거부.

    [왜 필수인가] 증류 ★시작★에서만 드리프트를 허용하고 ★봉인★에서 거부하면 16시간 학습이 끝에서
    통째로 버려진다(512 env 프로브에서 실측). 두 게이트는 항상 같은 인자를 받아야 한다.
    """
    parent = {
        "receipt_sha256": "f" * 64,
        "source": _context(commit="a" * 40),
        "task_id": "Isaac-BinPick-UR5e-Teacher-v0",
        "mode": "scratch",
        "adoption": _teacher_adoption(),
    }
    kwargs = dict(
        project_root=tmp_path,
        experiment_name="ur5e_binpick_vision",
        task_id="Isaac-BinPick-UR5e-Vision-Distill-v0",
        mode="distillation",
        started_at="2026-08-14T00:00:00+00:00",
        completed_at="2026-08-14T00:01:00+00:00",
        arguments=["--distill-from", "model_1.pt"],
        start_context=_context(commit="b" * 40),
        end_context=_context(commit="b" * 40),
        inputs=[parent],
        runtime_authority=_runtime_authority(_context(commit="b" * 40)),
    )
    run, _ = _run(tmp_path)
    with pytest.raises(ContractError, match="incompatible with this run"):
        finalize_training_receipt(run, **kwargs)
    for bad in ("b" * 40, "a" * 12, ""):
        with pytest.raises(ContractError, match="incompatible with this run"):
            finalize_training_receipt(run, accept_parent_commit_drift_from=bad, **kwargs)
    # config까지 다르면 정확한 commit을 줘도 거부한다.
    other_cfg = {**_context(commit="a" * 40), "canonical_configs": {relative: "d" * 64 for relative in REQUIRED_RELEASE_CONFIGS}}
    with pytest.raises(ContractError, match="incompatible with this run"):
        finalize_training_receipt(
            run, accept_parent_commit_drift_from="a" * 40,
            **{**kwargs, "inputs": [{**parent, "source": other_cfg}]},
        )
    receipt = finalize_training_receipt(run, accept_parent_commit_drift_from="a" * 40, **kwargs)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["inputs"][0]["context_drift"] == {
        "parent_commit": "a" * 40,
        "run_commit": "b" * 40,
        "accepted_by": "explicit accept_parent_commit_drift_from (F95)",
    }


def test_training_receipt_json_and_digest_come_from_the_same_open_handle(
    tmp_path, monkeypatch,
) -> None:
    run, checkpoint = _run(tmp_path)
    receipt = finalize_training_receipt(
        run,
        project_root=tmp_path,
        experiment_name="ur5e_binpick_vision",
        task_id="Isaac-BinPick-UR5e-Vision-v0",
        mode="ppo",
        started_at="start",
        completed_at="end0",
        arguments=[],
        start_context=_context(),
        end_context=_context(),
        runtime_authority=_runtime_authority(),
    )
    original_bytes = receipt.read_bytes()
    alternate_payload = json.loads(original_bytes.decode("utf-8"))
    alternate_payload["completed_at"] = "end1"
    alternate_bytes = (
        json.dumps(alternate_payload, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    alternate = tmp_path / "alternate-run-manifest.json"
    alternate.write_bytes(alternate_bytes)
    real_open = Path.open
    opened_alternate = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal opened_alternate
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == receipt and mode == "rb" and not opened_alternate:
            opened_alternate = True
            return real_open(alternate, "rb")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    verified = verify_training_checkpoint_receipt(
        checkpoint, project_root=tmp_path
    )
    assert opened_alternate is True
    assert verified["receipt_sha256"] == hashlib.sha256(alternate_bytes).hexdigest()
    assert verified["receipt_sha256"] != hashlib.sha256(original_bytes).hexdigest()


def test_canonical_config_digest_uses_the_exact_opened_snapshot(
    tmp_path, monkeypatch,
) -> None:
    for relative in REQUIRED_RELEASE_CONFIGS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"trusted: {relative}\n".encode("utf-8"))
    target = tmp_path / "configs/common.yaml"
    trusted = target.read_bytes()
    alternate_bytes = b"attacker: snapshot\n"
    alternate = tmp_path / "alternate-common.yaml"
    alternate.write_bytes(alternate_bytes)
    real_open = Path.open
    redirected = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal redirected
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == target and mode == "rb" and not redirected:
            redirected = True
            opened = real_open(alternate, "rb")
            with real_open(target, "wb") as stream:
                stream.write(trusted)
            return opened
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    hashes = canonical_config_hashes(tmp_path)
    assert redirected is True
    assert hashes["configs/common.yaml"] == hashlib.sha256(alternate_bytes).hexdigest()
    assert hashes["configs/common.yaml"] != hashlib.sha256(trusted).hexdigest()


def test_evaluation_artifact_snapshot_hashes_the_same_opened_bytes(
    tmp_path, monkeypatch,
) -> None:
    result_dir = tmp_path / "Phase1" / "Result" / "eval_test"
    result_dir.mkdir(parents=True)
    artifact = result_dir / "episodes.csv"
    trusted = b"episode,placed_success\n0,1\n"
    alternate_bytes = b"episode,placed_success\n0,0\n"
    assert len(trusted) == len(alternate_bytes)
    artifact.write_bytes(trusted)
    alternate = tmp_path / "alternate-episodes.csv"
    alternate.write_bytes(alternate_bytes)
    real_open = Path.open
    opened_alternate = False

    def swap_back_open(path: Path, *args, **kwargs):
        nonlocal opened_alternate
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == artifact and mode == "rb" and not opened_alternate:
            opened_alternate = True
            return real_open(alternate, "rb")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", swap_back_open)
    record, payload = evaluation_provenance._regular_file_snapshot(
        artifact, project_root=tmp_path
    )
    assert opened_alternate is True
    assert payload == alternate_bytes
    assert record["sha256"] == hashlib.sha256(alternate_bytes).hexdigest()


def test_finalize_receipt_atomic_publish_never_replaces_competing_manifest(
    tmp_path, monkeypatch,
) -> None:
    run, _ = _run(tmp_path)
    destination = run / "run_manifest.json"
    competing = b'{"owner":"competing-run"}\n'
    real_mkstemp = training_provenance.tempfile.mkstemp

    def racing_mkstemp(*args, **kwargs):
        handle, name = real_mkstemp(*args, **kwargs)
        destination.write_bytes(competing)
        return handle, name

    monkeypatch.setattr(training_provenance.tempfile, "mkstemp", racing_mkstemp)
    with pytest.raises(ContractError, match="already exists; refusing overwrite"):
        finalize_training_receipt(
            run,
            project_root=tmp_path,
            experiment_name="ur5e_binpick_vision",
            task_id="Isaac-BinPick-UR5e-Vision-v0",
            mode="ppo",
            started_at="start",
            completed_at="end",
            arguments=[],
            start_context=_context(),
            end_context=_context(),
            runtime_authority=_runtime_authority(),
        )
    assert destination.read_bytes() == competing
    assert not list(run.glob(".run-manifest-*"))


def test_adoption_grade_training_verification_rejects_event_drift(tmp_path):
    run, checkpoint = _run(tmp_path)
    finalize_training_receipt(
        run,
        project_root=tmp_path,
        experiment_name="ur5e_binpick_vision",
        task_id="Isaac-BinPick-UR5e-Vision-v0",
        mode="ppo",
        started_at="start",
        completed_at="end",
        arguments=[],
        start_context=_context(),
        end_context=_context(),
        runtime_authority=_runtime_authority(),
    )
    verify_training_checkpoint_receipt(
        checkpoint,
        project_root=tmp_path,
        require_tensorboard_events=True,
        require_all_checkpoints=True,
    )
    (run / "events.out.tfevents.test").write_bytes(b"tampered-events")
    with pytest.raises(ContractError, match="event inventory/hash drift"):
        verify_training_checkpoint_receipt(
            checkpoint,
            project_root=tmp_path,
            require_tensorboard_events=True,
            require_all_checkpoints=True,
        )


def test_finalize_distillation_rejects_unadopted_teacher(tmp_path):
    run, _ = _run(tmp_path)
    with pytest.raises(ContractError, match="Teacher adoption"):
        finalize_training_receipt(
            run,
            project_root=tmp_path,
            experiment_name="ur5e_binpick_vision",
            task_id="Isaac-BinPick-UR5e-Vision-Distill-v0",
            mode="distillation",
            started_at="start",
            completed_at="end",
            arguments=[],
            start_context=_context(),
            end_context=_context(),
            inputs=[{
                "receipt_sha256": "f" * 64,
                "source": _context(),
                "task_id": "Isaac-BinPick-UR5e-Teacher-v0",
                "mode": "scratch",
            }],
            runtime_authority=_runtime_authority(),
        )


def test_finalize_rejects_context_drift_without_writing_receipt(tmp_path):
    run, _ = _run(tmp_path)
    with pytest.raises(ContractError, match="changed while"):
        finalize_training_receipt(
            run,
            project_root=tmp_path,
            experiment_name="ur5e_binpick_vision",
            task_id="Isaac-BinPick-UR5e-Vision-Distill-v0",
            mode="distillation",
            started_at="start",
            completed_at="end",
            arguments=[],
            start_context=_context("a" * 40),
            end_context=_context("b" * 40),
            runtime_authority=_runtime_authority(_context("a" * 40)),
        )
    assert not (run / "run_manifest.json").exists()


def test_finalize_rejects_parent_checkpoint_from_incompatible_context(tmp_path):
    run, _ = _run(tmp_path)
    with pytest.raises(ContractError, match="parent checkpoint.*incompatible"):
        finalize_training_receipt(
            run,
            project_root=tmp_path,
            experiment_name="ur5e_binpick_vision",
            task_id="Isaac-BinPick-UR5e-Vision-Distill-v0",
            mode="distillation",
            started_at="start",
            completed_at="end",
            arguments=[],
            start_context=_context("a" * 40),
            end_context=_context("a" * 40),
            inputs=[{
                "receipt_sha256": "f" * 64,
                "source": _context("b" * 40),
                "task_id": "Isaac-BinPick-UR5e-Teacher-v0",
                "mode": "scratch",
            }],
            runtime_authority=_runtime_authority(_context("a" * 40)),
        )
    assert not (run / "run_manifest.json").exists()


def test_checkpoint_or_parameter_byte_drift_is_rejected(tmp_path):
    run, checkpoint = _run(tmp_path)
    finalize_training_receipt(
        run,
        project_root=tmp_path,
        experiment_name="ur5e_binpick_vision",
        task_id="Isaac-BinPick-UR5e-Vision-Distill-v0",
        mode="ppo",
        started_at="start",
        completed_at="end",
        arguments=[],
        start_context=_context(),
        end_context=_context(),
        runtime_authority=_runtime_authority(),
    )
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="differs"):
        verify_training_checkpoint_receipt(checkpoint, project_root=tmp_path)


def test_legacy_or_missing_receipt_is_not_deployable(tmp_path):
    run, checkpoint = _run(tmp_path)
    with pytest.raises(ContractError, match="no trusted"):
        verify_training_checkpoint_receipt(checkpoint, project_root=tmp_path)
    (run / "run_manifest.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    with pytest.raises(ContractError, match="legacy"):
        verify_training_checkpoint_receipt(checkpoint, project_root=tmp_path)


def _write_nurec_lineage(root: Path) -> dict:
    scene = root / "assets" / "Scene"
    scene.mkdir(parents=True)
    source = scene / "Workplace.ply"
    asset = scene / "Workplace.usdz"
    source.write_bytes(b"ply-capture")
    asset.write_bytes(b"usdz-conversion")
    metadata = {
        "schema_version": 1,
        "asset_path": "assets/Scene/Workplace.usdz",
        "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
        "asset_bytes": asset.stat().st_size,
        "source_path": "assets/Scene/Workplace.ply",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_bytes": source.stat().st_size,
        "capture_id": "test-capture-001",
        "conversion_tool": "Isaac-Sim-5.1-ply-to-usdz",
        "created_at": "2026-08-14T00:00:00+09:00",
    }
    (scene / "Workplace.provenance.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    return load_nurec_provenance(root)


def test_nurec_receipt_binds_local_asset_and_accepts_compatible_student_parent(tmp_path):
    run, _ = _run(tmp_path)
    start = _context()
    start["optional_assets"] = {"nurec": _write_nurec_lineage(tmp_path)}
    parent = {
        "receipt_sha256": "f" * 64,
        "source": _context(),
        "task_id": "Isaac-BinPick-UR5e-Vision-Distill-v0",
        "mode": "distillation",
        "checkpoint_name": "model_7.pt",
        "checkpoint_sha256": "e" * 64,
        "adoption": {
            "schema_version": 1,
            "role": "student",
            "path": "logs/rsl_rl/ur5e_binpick_vision/run/adoption.json",
            "sha256": "d" * 64,
            "checkpoint_name": "model_7.pt",
            "checkpoint_sha256": "e" * 64,
            "training_receipt_sha256": "f" * 64,
        },
    }
    receipt = finalize_training_receipt(
        run,
        project_root=tmp_path,
        experiment_name="ur5e_binpick_vision",
        task_id="Isaac-BinPick-UR5e-Vision-NuRec-v0",
        mode="nurec",
        started_at="start",
        completed_at="end",
        arguments=[],
        start_context=start,
        end_context=start,
        inputs=[parent],
        runtime_authority=_runtime_authority(start),
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["source"]["optional_assets"]["nurec"]["metadata"]["asset_bytes"] > 0


def test_nurec_lineage_rejects_byte_drift(tmp_path):
    _write_nurec_lineage(tmp_path)
    (tmp_path / "assets/Scene/Workplace.usdz").write_bytes(b"changed")
    with pytest.raises(ContractError, match="byte binding mismatch"):
        load_nurec_provenance(tmp_path)
