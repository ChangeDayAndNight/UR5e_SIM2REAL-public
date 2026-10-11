"""Shared, stdlib-only resolver for the canonical Isaac training interpreter."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


ENV_NAME = "isaac51"
NATIVE_TRAINING_MODULES = ("torch", "tensordict", "rsl_rl")


def preload_native_training_modules() -> list[str]:
    """Initialize native RL dependencies before Isaac Kit on Windows.

    Isaac Sim 5.1 can terminate with ``0xC0000005`` when TensorDict's native
    extension is loaded for the first time after Kit.  These modules do not
    require Omni APIs, so loading them first is safe and deterministic.
    """

    loaded: list[str] = []
    for name in NATIVE_TRAINING_MODULES:
        importlib.import_module(name)
        loaded.append(name)
    return loaded


def _python_in_env(env_dir: Path) -> Path:
    return env_dir / ("python.exe" if os.name == "nt" else "bin/python")


def _normal_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def conda_executable_candidates(environ: Mapping[str, str] | None = None) -> list[Path]:
    """Return conda executables discoverable from the shell and standard homes."""

    env = os.environ if environ is None else environ
    candidates: list[Path] = []
    explicit = env.get("CONDA_EXE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    on_path = shutil.which("conda", path=env.get("PATH"))
    if on_path:
        candidates.append(Path(on_path))

    home = Path(env.get("USERPROFILE", "")).expanduser() if env.get("USERPROFILE") else Path.home()
    for root_name in ("miniconda3", "anaconda3", "miniforge3"):
        root = home / root_name
        candidates.extend((root / "Scripts/conda.exe", root / "bin/conda"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normal_key(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _conda_environment_paths(
    conda_executable: Path,
    environ: Mapping[str, str],
) -> list[Path]:
    """Read conda's authoritative environment list, including custom envs_dirs."""

    try:
        proc = subprocess.run(
            [str(conda_executable), "env", "list", "--json"],
            env=dict(environ),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    try:
        payload = json.loads(proc.stdout)
    except (TypeError, json.JSONDecodeError):
        return []
    envs = payload.get("envs", [])
    if not isinstance(envs, list):
        return []
    return [Path(value).expanduser() for value in envs if isinstance(value, str) and value.strip()]


def training_python_candidates(environ: Mapping[str, str] | None = None) -> list[Path]:
    """Return canonical ``isaac51`` candidates from overrides and conda metadata."""

    env = os.environ if environ is None else environ
    candidates: list[Path] = []

    override = env.get("ISAACLAB_ENV_PY", "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    conda_prefix = env.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        prefix = Path(conda_prefix).expanduser()
        if prefix.name.casefold() == ENV_NAME.casefold():
            candidates.append(_python_in_env(prefix))

    for conda_executable in conda_executable_candidates(env):
        if not conda_executable.is_file():
            continue
        for env_path in _conda_environment_paths(conda_executable, env):
            if env_path.name.casefold() == ENV_NAME.casefold():
                candidates.append(_python_in_env(env_path))

    # Last-resort standard layouts are useful when conda itself is temporarily
    # unavailable, while custom envs_dirs are covered by ``conda env list``.
    home = Path(env.get("USERPROFILE", "")).expanduser() if env.get("USERPROFILE") else Path.home()
    for root_name in ("miniconda3", "anaconda3", "miniforge3"):
        candidates.append(_python_in_env(home / root_name / "envs" / ENV_NAME))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = _normal_key(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _is_canonical_env_python(path: Path) -> bool:
    env_dir = path.parent if os.name == "nt" else path.parent.parent
    return env_dir.name.casefold() == ENV_NAME.casefold()


def resolve_training_python(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the sole supported ``isaac51`` interpreter, failing closed."""

    env = os.environ if environ is None else environ
    override = env.get("ISAACLAB_ENV_PY", "").strip()
    if override:
        override_path = Path(override).expanduser()
        if not override_path.is_file():
            raise FileNotFoundError(f"ISAACLAB_ENV_PY does not exist: {override}")
        if not _is_canonical_env_python(override_path):
            raise RuntimeError(f"ISAACLAB_ENV_PY must belong to the {ENV_NAME} environment: {override}")
        return str(override_path.resolve())

    existing = [candidate.resolve() for candidate in training_python_candidates(env) if candidate.is_file()]
    unique = { _normal_key(candidate): candidate for candidate in existing }
    if len(unique) == 1:
        return str(next(iter(unique.values())))
    if len(unique) > 1:
        choices = ", ".join(sorted(str(path) for path in unique.values()))
        raise RuntimeError(
            f"multiple {ENV_NAME} environments found; set ISAACLAB_ENV_PY to the canonical one: {choices}"
        )

    current = Path(sys.executable)
    if Path(sys.prefix).name.casefold() == ENV_NAME.casefold() and _is_canonical_env_python(current):
        return str(current.resolve())
    raise FileNotFoundError(
        f"{ENV_NAME} training Python was not found. Run "
        "powershell -ExecutionPolicy Bypass -File scripts/bootstrap_training.ps1 first."
    )


def reexec_in_training_python(
    script_file: str | os.PathLike[str],
    argv: Sequence[str] | None = None,
) -> None:
    """Re-execute an entry point under the canonical interpreter when needed."""

    try:
        if Path(sys.argv[0]).resolve() != Path(script_file).resolve():
            return
    except (OSError, ValueError):
        return

    target = resolve_training_python()
    current = _normal_key(Path(sys.executable))
    desired = _normal_key(Path(target))
    if current == desired and not os.environ.get("PYTHONPATH"):
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        return

    child_env = dict(os.environ)
    child_env.pop("PYTHONPATH", None)
    child_env["PYTHONNOUSERSITE"] = "1"
    child_env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    args = list(sys.argv[1:] if argv is None else argv)
    return_code = subprocess.run(
        [target, "-u", os.path.abspath(script_file), *args],
        env=child_env,
        check=False,
    ).returncode
    raise SystemExit(return_code)


def configure_utf8_stdio() -> None:
    """Keep diagnostics printable in Windows consoles with legacy code pages."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resolve the canonical Isaac training environment")
    parser.add_argument("--resolve-python", action="store_true", help="print the canonical isaac51 Python path")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.resolve_python:
        _build_parser().error("--resolve-python is required")
    try:
        resolved = resolve_training_python()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"training environment resolution failed: {exc}", file=sys.stderr)
        return 1
    print(resolved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
