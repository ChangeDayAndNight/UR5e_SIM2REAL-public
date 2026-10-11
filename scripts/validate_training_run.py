#!/usr/bin/env python
"""Fail-closed validation of artifacts emitted by one RSL-RL training run."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import stat


class TrainingArtifactError(RuntimeError):
    """A successful training process did not leave a complete usable run."""


@dataclass(frozen=True)
class TrainingArtifacts:
    run_dir: Path
    checkpoints: tuple[Path, ...]
    parameter_files: tuple[Path, ...]
    tensorboard_events: tuple[Path, ...]


def _require_nonempty_file(path: Path, label: str) -> None:
    _reject_linked_path(Path(os.path.abspath(path)))
    if not path.is_file():
        raise TrainingArtifactError(f"missing/non-regular {label}: {path}")
    if path.stat().st_size <= 0:
        raise TrainingArtifactError(f"empty {label}: {path}")


def _require_group(run_dir: Path, pattern: str, label: str) -> tuple[Path, ...]:
    matches = tuple(sorted(run_dir.glob(pattern)))
    if not matches:
        raise TrainingArtifactError(f"no {label} matching {pattern!r}: {run_dir}")
    for path in matches:
        _require_nonempty_file(path, label)
    return matches


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse_flag = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(info.st_mode) or bool(reparse_flag and attributes & reparse_flag)


def _reject_linked_path(path: Path) -> None:
    """Reject a link/reparse point in the run path before resolving it."""

    current = path
    while True:
        if current.exists() and _is_link_or_reparse(current):
            raise TrainingArtifactError(f"training run path contains link/reparse point: {current}")
        if current.parent == current:
            return
        current = current.parent


def validate_training_run(
    run_dir: Path | str, *, expected_root: Path | str | None = None
) -> TrainingArtifacts:
    """Require checkpoint, both parameter YAMLs, and TensorBoard output."""

    lexical = Path(os.path.abspath(Path(run_dir).expanduser()))
    _reject_linked_path(lexical)
    if expected_root is not None:
        root = Path(os.path.abspath(Path(expected_root).expanduser()))
        _reject_linked_path(root)
        try:
            lexical.relative_to(root)
        except ValueError as exc:
            raise TrainingArtifactError(
                f"training run escaped expected experiment root {root}: {lexical}"
            ) from exc
    if not lexical.is_dir():
        raise TrainingArtifactError(f"training run directory does not exist: {lexical}")
    run_dir = lexical.resolve(strict=True)
    checkpoints = _require_group(run_dir, "model_*.pt", "checkpoint")
    parameter_files = tuple(run_dir / "params" / name for name in ("env.yaml", "agent.yaml"))
    for path in parameter_files:
        _require_nonempty_file(path, "parameter YAML")
    tensorboard_events = _require_group(
        run_dir, "events.out.tfevents.*", "TensorBoard event file"
    )
    return TrainingArtifacts(run_dir, checkpoints, parameter_files, tensorboard_events)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate one completed RSL-RL run")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-root", type=Path)
    args = parser.parse_args(argv)
    try:
        artifacts = validate_training_run(args.run_dir, expected_root=args.expected_root)
    except (OSError, TrainingArtifactError) as exc:
        print(f"[training-artifacts][FAIL] {exc}")
        return 1
    print(
        "[training-artifacts] TRAINING_ARTIFACTS_OK "
        f"run={artifacts.run_dir} checkpoints={len(artifacts.checkpoints)} "
        f"params={len(artifacts.parameter_files)} "
        f"tensorboard={len(artifacts.tensorboard_events)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
