from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import training_env


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_extension_wheel", ROOT / "scripts" / "validate_extension_wheel.py"
)
assert SPEC and SPEC.loader
wheel_validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wheel_validator
SPEC.loader.exec_module(wheel_validator)


def test_within_rejects_editable_source_outside_install_target(tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    inside = target / "ur5e_binpick" / "module.py"
    inside.parent.mkdir()
    inside.write_text("VALUE = 1\n", encoding="utf-8")
    outside = tmp_path / "source" / "module.py"
    outside.parent.mkdir()
    outside.write_text("VALUE = 2\n", encoding="utf-8")
    assert wheel_validator._within(inside, target)
    assert not wheel_validator._within(outside, target)


def test_missing_wheel_target_fails_closed(tmp_path: Path):
    with pytest.raises(RuntimeError, match="does not exist"):
        wheel_validator.validate_installed_tree(tmp_path / "missing")


def _write_minimal_distribution(target: Path, *, editable: bool = False) -> None:
    package = target / "demo_package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "child.py").write_text("VALUE = 2\n", encoding="utf-8")
    metadata = target / "demo_package-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: demo-package\nVersion: 1.0\n",
        encoding="utf-8",
    )
    if editable:
        (metadata / "direct_url.json").write_text(
            '{"dir_info": {"editable": true}, "url": "file:///placeholder"}\n',
            encoding="utf-8",
        )


def test_isolated_distribution_imports_entire_package_tree(tmp_path: Path):
    target = tmp_path / "target"
    _write_minimal_distribution(target)
    imported = wheel_validator.validate_installed_tree(
        target,
        package="demo_package",
        distribution="demo-package",
        expected_modules=2,
    )
    assert imported == ["demo_package", "demo_package.child"]


def test_editable_distribution_is_rejected(tmp_path: Path):
    target = tmp_path / "target"
    _write_minimal_distribution(target, editable=True)
    with pytest.raises(RuntimeError, match="editable"):
        wheel_validator.validate_installed_tree(
            target,
            package="demo_package",
            distribution="demo-package",
        )


def test_native_training_modules_are_preloaded_in_safe_order(monkeypatch):
    imported: list[str] = []
    monkeypatch.setattr(
        training_env.importlib,
        "import_module",
        lambda name: imported.append(name),
    )
    assert wheel_validator.preload_native_training_modules() == [
        "torch",
        "tensordict",
        "rsl_rl",
    ]
    assert imported == ["torch", "tensordict", "rsl_rl"]


def test_native_preload_happens_before_app_launcher_import():
    source = (ROOT / "scripts" / "validate_extension_wheel.py").read_text(encoding="utf-8")
    run_source = source[
        source.index("def _run_validation(") : source.index("def _run_isolated_kit_child(")
    ]
    assert run_source.index("preload_native_training_modules()") < run_source.index(
        "from isaaclab.app import AppLauncher"
    )
    assert run_source.index("_emit_validation_result(imported") < run_source.index(
        "simulation_app.close()"
    )


def test_launch_app_gate_spawns_explicit_isolated_child(monkeypatch, tmp_path: Path):
    target = tmp_path / "target"
    target.mkdir()
    observed: list[list[str]] = []

    def fake_run(command, **kwargs):
        observed.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="[wheel-import] WHEEL_IMPORT_OK modules=14\nWHEEL_KIT_CHILD_OK\n",
        )

    monkeypatch.setattr(wheel_validator.subprocess, "run", fake_run)
    args = SimpleNamespace(target=target, support_root=None, expected_modules=14)
    assert wheel_validator._run_isolated_kit_child(args) == 0
    assert observed[0][:3] == [
        sys.executable,
        "-I",
        str((ROOT / "scripts" / "validate_extension_wheel.py").resolve()),
    ]
    assert "--app-child" in observed[0]
    assert "--launch-app" not in observed[0]


@pytest.mark.parametrize(
    "returncode,output",
    [
        (0, "[wheel-import] WHEEL_IMPORT_OK modules=14\n"),
        (3221225477, ""),
        (0, "[wheel-import][FAIL] RuntimeError: import failed\nWHEEL_KIT_CHILD_FAIL\n"),
    ],
)
def test_launch_app_gate_rejects_missing_sentinel_crash_and_masked_failure(
    monkeypatch, tmp_path: Path, returncode: int, output: str
):
    target = tmp_path / "target"
    target.mkdir()
    monkeypatch.setattr(
        wheel_validator.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=returncode, stdout=output),
    )
    args = SimpleNamespace(target=target, support_root=None, expected_modules=14)
    assert wheel_validator._run_isolated_kit_child(args) == 1


def test_wheel_child_flushes_verdict_before_app_close(monkeypatch, tmp_path: Path):
    events: list[str] = []

    class FakeApp:
        def close(self) -> None:
            events.append("close")

    class FakeLauncher:
        def __init__(self, *, headless: bool) -> None:
            assert headless
            self.app = FakeApp()

    isaaclab = ModuleType("isaaclab")
    isaaclab.__path__ = []
    isaaclab_app = ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = FakeLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app)
    monkeypatch.setattr(
        wheel_validator,
        "preload_native_training_modules",
        lambda: events.append("preload") or ["torch", "tensordict", "rsl_rl"],
    )
    monkeypatch.setattr(
        wheel_validator,
        "validate_installed_tree",
        lambda *_args, **_kwargs: events.append("validate") or ["ur5e_binpick"],
    )
    monkeypatch.setattr(
        wheel_validator,
        "_emit_validation_result",
        lambda _imported, *, child: events.append("verdict-flushed"),
    )
    args = SimpleNamespace(target=tmp_path, support_root=None, expected_modules=1)
    assert wheel_validator._run_validation(args, launch_app=True, child=True) == 0
    assert events == ["preload", "validate", "verdict-flushed", "close"]


def test_manual_gpu_workflow_uses_isolated_wheel_gate():
    workflow = (ROOT / ".github/workflows/isaac-gpu-manual.yml").read_text(encoding="utf-8")
    assert "timeout-minutes: 120" in workflow
    assert "pip wheel --no-deps --no-build-isolation" in workflow
    assert "pip install --no-deps --target" in workflow
    assert "python -I scripts/validate_extension_wheel.py" in workflow
    assert "--expected-modules 14 --launch-app" in workflow
