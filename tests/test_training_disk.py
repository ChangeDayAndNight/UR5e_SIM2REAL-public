from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment.contracts import ContractError
from scripts.check_training_disk import GIB, require_free_space


def _vision_train_module():
    path = Path(__file__).parents[1] / "Phase2" / "VisionPick_Train.py"
    spec = importlib.util.spec_from_file_location("vision_train_disk_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_free_space_gate_accepts_exact_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.check_training_disk.shutil.disk_usage",
        lambda path: SimpleNamespace(total=100 * GIB, used=75 * GIB, free=25 * GIB),
    )
    assert require_free_space(tmp_path, 25.0) == 25.0


def test_free_space_gate_rejects_below_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.check_training_disk.shutil.disk_usage",
        lambda path: SimpleNamespace(
            total=100 * GIB, used=75 * GIB + 1, free=25 * GIB - 1
        ),
    )
    with pytest.raises(RuntimeError, match="free-space gate failed"):
        require_free_space(tmp_path, 25.0)


def test_train_launcher_applies_disk_gate_only_to_nontrivial_distill():
    source = (Path(__file__).parents[1] / "scripts/train.ps1").read_text(encoding="utf-8")
    assert '$Preset -eq "Distill"' in source
    assert '$MaxIterations -gt 1' in source
    assert '"--minimum-gib", "25"' in source
    assert 'RequiredSentinel "FREE_SPACE_OK"' in source

    direct = (Path(__file__).parents[1] / "Phase2/VisionPick_Train.py").read_text(
        encoding="utf-8"
    )
    assert "_effective_max_iterations(user_args) > 1" in direct
    gate = "require_free_space(Path(ROOT), DISTILL_MINIMUM_FREE_GIB)"
    assert direct.count(gate) == 2
    copied = direct.index("cmd, seed_dir = _build_cmd")
    launch_gate = direct.index(gate, copied)
    launched = direct.index("rc, run_dir = _run", launch_gate)
    assert copied < launch_gate < launched


def test_nontrivial_distill_rechecks_disk_after_warmstart_copy_before_run(
    monkeypatch, tmp_path,
) -> None:
    module = _vision_train_module()
    train_script = tmp_path / "train.py"
    train_script.write_text("# test\n", encoding="utf-8")
    teacher = tmp_path / "model_1.pt"
    teacher.write_bytes(b"teacher")
    context = {"schema_version": 1, "context": "same"}
    adoption = {
        "schema_version": 1,
        "role": "teacher",
        "checkpoint_sha256": "a" * 64,
    }
    events: list[str] = []

    monkeypatch.setattr(module, "ROOT", str(tmp_path))
    monkeypatch.setattr(module, "TRAIN_SCRIPT", str(train_script))
    monkeypatch.setattr(module, "LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(module, "_require_full_preflight", lambda **_kwargs: None)
    monkeypatch.setattr(module, "capture_training_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        module,
        "verify_training_checkpoint_receipt",
        lambda *_args, **_kwargs: {
            "source": context,
            "checkpoint_name": "model_1.pt",
            "checkpoint_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(module, "verify_adopted_checkpoint", lambda *_args, **_kwargs: adoption)

    def build(*_args, **_kwargs):
        events.append("warmstart_copy")
        return ["python", "train.py"], None

    monkeypatch.setattr(module, "_build_cmd", build)
    monkeypatch.setattr(
        module,
        "require_free_space",
        lambda *_args, **_kwargs: events.append("disk_gate") or 25.0,
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: events.append("run") or (1, None),
    )
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["VisionPick_Train.py", "--distill-from", str(teacher), "--max_iterations", "2"],
    )

    assert module.main() == 1
    assert events == ["disk_gate", "warmstart_copy", "disk_gate", "run"]


def test_direct_distill_iteration_resolution_is_exact_and_fail_closed():
    module = _vision_train_module()
    assert module._effective_max_iterations([]) == 20_000
    assert module._effective_max_iterations(["--max_iterations", "1"]) == 1
    assert module._effective_max_iterations(["--max_iterations=20000"]) == 20_000
    with pytest.raises(ContractError, match="only once"):
        module._effective_max_iterations([
            "--max_iterations", "1", "--max_iterations=20000",
        ])
    with pytest.raises(ContractError, match="canonical integer"):
        module._effective_max_iterations(["--max_iterations", "+2"])


def test_only_distill_checkpoint_interval_is_reduced_for_storage_policy():
    source = (
        Path(__file__).parents[1]
        / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    teacher = source[source.index("class BinPickTeacherRunnerCfg"):source.index(
        "class BinPickVisionRunnerCfg"
    )]
    vision = source[source.index("class BinPickVisionRunnerCfg"):source.index(
        "class BinPickDistillRunnerCfg"
    )]
    distill = source[source.index("class BinPickDistillRunnerCfg"):]
    assert "save_interval = 50" in teacher
    assert "save_interval = 50" in vision
    assert "save_interval = 500" in distill
