from pathlib import Path
import os

import pytest

from scripts.validate_training_run import TrainingArtifactError, validate_training_run


def _complete_run(path: Path) -> Path:
    (path / "params").mkdir(parents=True)
    (path / "model_0.pt").write_bytes(b"checkpoint")
    (path / "params" / "env.yaml").write_text("scene: {}\n", encoding="utf-8")
    (path / "params" / "agent.yaml").write_text("runner: {}\n", encoding="utf-8")
    (path / "events.out.tfevents.TEST").write_bytes(b"tensorboard")
    return path


def test_complete_training_run_passes(tmp_path: Path):
    artifacts = validate_training_run(_complete_run(tmp_path / "run"))
    assert [path.name for path in artifacts.checkpoints] == ["model_0.pt"]
    assert [path.name for path in artifacts.parameter_files] == ["env.yaml", "agent.yaml"]
    assert [path.name for path in artifacts.tensorboard_events] == [
        "events.out.tfevents.TEST"
    ]


@pytest.mark.parametrize(
    "relative",
    (
        "model_0.pt",
        "params/env.yaml",
        "params/agent.yaml",
        "events.out.tfevents.TEST",
    ),
)
def test_missing_required_training_artifact_fails(tmp_path: Path, relative: str):
    run = _complete_run(tmp_path / "run")
    (run / relative).unlink()
    with pytest.raises(TrainingArtifactError):
        validate_training_run(run)


@pytest.mark.parametrize(
    "relative",
    (
        "model_0.pt",
        "params/env.yaml",
        "params/agent.yaml",
        "events.out.tfevents.TEST",
    ),
)
def test_empty_required_training_artifact_fails(tmp_path: Path, relative: str):
    run = _complete_run(tmp_path / "run")
    (run / relative).write_bytes(b"")
    with pytest.raises(TrainingArtifactError, match="empty"):
        validate_training_run(run)


def test_linked_run_path_fails_closed(tmp_path: Path):
    real_run = _complete_run(tmp_path / "real")
    linked_run = tmp_path / "linked"
    try:
        os.symlink(real_run, linked_run, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink unavailable: {exc}")
    with pytest.raises(TrainingArtifactError, match="link/reparse"):
        validate_training_run(linked_run)


def test_run_must_stay_below_expected_experiment_root(tmp_path: Path):
    expected = tmp_path / "expected"
    expected.mkdir()
    outside = _complete_run(tmp_path / "outside")
    with pytest.raises(TrainingArtifactError, match="escaped expected experiment root"):
        validate_training_run(outside, expected_root=expected)
