"""Isaac/GPU 없이 검사하는 학습 bootstrap·preflight·smoke 계약."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
from types import ModuleType, SimpleNamespace
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _training_doc() -> str:
    """학습 문서(원장) 본문 — 학습 저장소는 README.md, 배포 저장소는 PROJECT_LOG.md다(★F97★).

    배포 패키지에서는 README.md가 '받아서 바로 실행' 안내가 되고 원장은 PROJECT_LOG.md로
    실린다. 두 저장소가 같은 tests/를 공유하므로 문서 위치만 여기서 흡수한다 — 검사 내용은 같다.
    """
    for name in ("README.md", "PROJECT_LOG.md"):
        path = ROOT / name
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if "## 2. GitHub clone" in text or "`external/IsaacLab`" in text:
                return text
    raise AssertionError("학습 문서(README.md 또는 PROJECT_LOG.md)를 찾지 못했다")


def test_asset_manifest_contains_exact_canonical_bundle() -> None:
    manifest = json.loads(_source("assets/asset_manifest.json"))
    paths = {entry["path"] for entry in manifest["bundled_assets"]}
    assert len(paths) == 7
    assert paths == {
        "assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd",
        "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_base.usd",
        "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_physics.usd",
        "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_robot.usd",
        "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_sensor.usd",
        "assets/d435i/d435i.usd",
        "assets/d435i/Props/instanceable_meshes.usd",
    }
    for entry in manifest["bundled_assets"]:
        assert len(entry["sha256"]) == 64
        assert entry["size"] > 0


def test_static_preflight_has_hash_stage_and_git_gates() -> None:
    src = _source("scripts/preflight_training.py")
    assert "git\", \"ls-files\", \"--error-unmatch" in src
    assert "Usd.Stage.Open" in src
    assert "UsdUtils.ComputeAllDependencies" in src
    assert "AppLauncher(headless=True)" in src
    assert "Sdf.Layer.FindOrOpen" in src
    assert "ExportToString" in src
    assert "platform.python_version() == TRAINING_PYTHON_VERSION" in src
    assert "Isaac Lab checkout clean" in src


def test_static_preflight_seals_training_runtime_contract() -> None:
    path = ROOT / "scripts" / "preflight_training.py"
    spec = importlib.util.spec_from_file_location("training_preflight_contract_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.CheckReport()
    module.validate_runtime_contract(report)
    assert not report.errors
    assert "timing contract=120Hz/20Hz/decimation6" in report.ok
    assert any(message.startswith("action contract=ik_rel/7D") for message in report.ok)
    assert any(message.startswith("camera/preprocess contract=112x112 RGB-D") for message in report.ok)
    assert "policy observation contract=50,206D" in report.ok


def test_smoke_is_fail_closed_and_checks_joint_order() -> None:
    src = _source("Phase0/smoke_test.py")
    assert "set(robot.joint_names)" not in src
    assert "list(robot.joint_names) != expected_names" in src
    assert "UR5E_SMOKE_OK" in src
    assert 'f"{sentinel} robot={robot_mode} joints={num_joints} steps={n_steps} "' in src
    assert "ISAAC_BOOT_SMOKE_OK" in src
    assert "raise FileNotFoundError" in src
    assert "raise RuntimeError" in src
    assert 'for state_name in ("joint_pos", "joint_vel")' in src
    assert "torch.isfinite(getattr(robot.data, state_name)).all()" in src


def test_train_smoke_contract_and_postprocess_strip() -> None:
    ps = _source("scripts/train.ps1")
    wrapper = _source("Phase1/BinPick_Train.py")
    assert '$NumEnvs = 16' in ps
    assert '$MaxIterations = 1' in ps
    assert '$NumEnvs -ne 16' in ps
    assert '$MaxIterations -ne 1' in ps
    assert '$Seed -ne 42' in ps
    assert "-Smoke rejects all ExtraArgs" in ps
    assert "task/resume/checkpoint/run/video" in ps
    assert "SkipPreflight" not in ps
    assert 'Invoke-Python -Arguments $preflight' in ps
    assert '-RequiredSentinel "UR5E_SMOKE_OK robot=ur5e_cfg joints=8"' in ps
    assert "without required sentinel" in ps
    assert '$arguments += "--skip-postprocess"' in ps
    assert ".Value" not in ps
    assert 'skip_postprocess = "--skip-postprocess" in raw_args' in wrapper
    assert 'arg != "--skip-postprocess"' in wrapper
    assert wrapper.index("if rc != 0:") < wrapper.index("export_metrics(", wrapper.index("def main"))
    assert "process exited 0 without exactly one RSL_RL_RUN_DIR receipt" in wrapper
    assert 'line.startswith("RSL_RL_RUN_DIR=")' in wrapper
    assert "validate_training_run(run_dir, expected_root=RUNS_ROOT)" in wrapper
    assert "TRAINING_ARTIFACTS_OK" in wrapper

    vision_wrapper = _source("Phase2/VisionPick_Train.py")
    main = vision_wrapper.index("def main")
    assert vision_wrapper.index("if rc != 0:", main) < vision_wrapper.index("export_metrics(", main)
    assert "validate_training_run(run_dir, expected_root=RUNS_ROOT)" in vision_wrapper
    assert 'skip_postprocess = "--skip-postprocess" in raw_args' in vision_wrapper
    assert 'arg != "--skip-postprocess"' in vision_wrapper
    assert "TRAINING_ARTIFACTS_OK" in vision_wrapper
    assert 'print(f"RSL_RL_RUN_DIR={log_dir}"' in _source("scripts/rsl_rl/train.py")
    assert 'strftime("%Y-%m-%d_%H-%M-%S-%f")' in _source("scripts/rsl_rl/train.py")
    workflow = _source(".github/workflows/isaac-gpu-manual.yml")
    assert "-SkipPostprocess" in workflow
    assert 'OMNI_KIT_ACCEPT_EULA: "YES"' in workflow
    assert "scripts/training_env.py --resolve-python" in workflow
    assert "conda info --base" not in workflow
    assert "$env:ISAACLAB_ENV_PY scripts/training_env.py --resolve-python" in workflow


def test_bootstrap_finds_conda_outside_path() -> None:
    bootstrap = _source("scripts/bootstrap_training.ps1")
    assert "function Resolve-CondaExecutable" in bootstrap
    assert "$env:CONDA_EXE" in bootstrap
    assert "OSVersion.Version.Build" in bootstrap
    assert 'Join-Path $env:USERPROFILE "$root\\Scripts\\conda.exe"' in bootstrap
    for root in ("miniconda3", "anaconda3", "miniforge3"):
        assert f'"{root}"' in bootstrap
    assert "SkipPreflight" not in bootstrap
    assert '[string]$EnvName' not in bootstrap
    assert '[string]$IsaacLabDir' not in bootstrap
    assert '"ISAACLAB_ENV_PY=$Python" | Out-File -FilePath $env:GITHUB_ENV' in bootstrap


def test_train_finds_standard_non_path_conda_after_child_bootstrap() -> None:
    source = _source("scripts/train.ps1")
    assert 'foreach ($rootName in @("miniconda3", "anaconda3", "miniforge3"))' in source
    assert 'Join-Path $root "envs\\isaac51\\python.exe"' in source
    assert 'Join-Path $root "Scripts\\conda.exe"' in source


def test_isaaclab_default_checkout_is_canonical_external_path(monkeypatch) -> None:
    bootstrap = _source("scripts/bootstrap_training.ps1")
    preflight_path = ROOT / "scripts" / "preflight_training.py"
    spec = importlib.util.spec_from_file_location("training_preflight_path_test", preflight_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert 'Join-Path $RepoRoot "external\\IsaacLab"' in bootstrap
    assert "IsaacLab_v2.3.2" not in bootstrap
    assert "`external/IsaacLab`" in _training_doc()
    monkeypatch.delenv("ISAACLAB_DIR", raising=False)
    expected = (ROOT / "external" / "IsaacLab").resolve()
    assert module.resolve_isaaclab_dir() == expected

    assert "--isaaclab-dir" not in _source("scripts/preflight_training.py")


def test_nonstatic_preflight_checks_live_gym_registry() -> None:
    src = _source("scripts/preflight_training.py")
    assert "simulation_app = AppLauncher(headless=True).app" in src
    assert "import ur5e_binpick.tasks" in src
    assert '"Isaac-BinPick-UR5e-Vision-NuRec-v0"' in src
    assert "gym.spec(task_id)" in src
    assert "runtime Gym task 미등록" in src


def test_preflight_allows_only_exact_isaac_runtime_mdl_identifier() -> None:
    module = _load_preflight_module("training_preflight_runtime_mdl_test")
    allowed, unexpected = module._partition_unresolved_assets(
        ["OmniPBR.mdl", "materials/OmniPBR.mdl", "MissingTexture.png"]
    )
    assert allowed == ["OmniPBR.mdl"]
    assert unexpected == ["MissingTexture.png", "materials/OmniPBR.mdl"]


def _load_preflight_module(name: str):
    path = ROOT / "scripts" / "preflight_training.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kit_preflight_parent_accepts_only_complete_child_verdict(monkeypatch) -> None:
    module = _load_preflight_module("training_preflight_child_success_test")
    payload = json.dumps(
        {"ok": ["stage graph checked"], "warnings": ["optional asset absent"], "errors": []}
    )

    def fake_run(command, **kwargs):
        assert command[:2] == [sys.executable, "-u"]
        assert command[-1] == "--kit-child"
        return SimpleNamespace(
            returncode=0,
            stdout=f"{module._KIT_CHILD_RESULT_PREFIX}{payload}\n{module._KIT_CHILD_OK}\n",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    report = module.CheckReport()
    module.validate_usd_dependencies(report)
    assert report.errors == []
    assert report.ok == ["stage graph checked"]
    assert report.warnings == ["optional asset absent"]


@pytest.mark.parametrize(
    "returncode,output,expected",
    [
        (0, "UR5E_KIT_PREFLIGHT_RESULT={\"ok\":[],\"warnings\":[],\"errors\":[]}\n", "missing exactly one success sentinel"),
        (3221225477, "", "exited nonzero"),
        (
            0,
            "UR5E_KIT_PREFLIGHT_RESULT={\"ok\":[],\"warnings\":[],\"errors\":[\"stage failed\"]}\n"
            "UR5E_KIT_PREFLIGHT_FAIL\n",
            "failure sentinel",
        ),
    ],
)
def test_kit_preflight_parent_rejects_missing_sentinel_crash_and_masked_failure(
    monkeypatch, returncode, output, expected
) -> None:
    module = _load_preflight_module(f"training_preflight_child_failure_test_{returncode}")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=returncode, stdout=output),
    )
    report = module.CheckReport()
    module.validate_usd_dependencies(report)
    assert report.errors
    assert any(expected in error for error in report.errors)


def test_kit_child_flush_callback_precedes_app_close(monkeypatch) -> None:
    module = _load_preflight_module("training_preflight_child_lifecycle_test")
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
    pxr = ModuleType("pxr")
    pxr.Sdf = SimpleNamespace()
    pxr.Usd = SimpleNamespace()
    pxr.UsdUtils = SimpleNamespace()
    gymnasium = ModuleType("gymnasium")
    isaaclab_tasks = ModuleType("isaaclab_tasks")
    ur5e_binpick = ModuleType("ur5e_binpick")
    ur5e_binpick.__path__ = []
    ur5e_tasks = ModuleType("ur5e_binpick.tasks")
    for name, fake in {
        "isaaclab": isaaclab,
        "isaaclab.app": isaaclab_app,
        "pxr": pxr,
        "gymnasium": gymnasium,
        "isaaclab_tasks": isaaclab_tasks,
        "ur5e_binpick": ur5e_binpick,
        "ur5e_binpick.tasks": ur5e_tasks,
    }.items():
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(module, "EXPECTED_TASKS", set())
    monkeypatch.setattr(module, "load_asset_manifest", lambda: {"bundled_assets": []})

    module._validate_usd_dependencies_in_kit_child(
        module.CheckReport(), lambda _report: events.append("verdict-flushed")
    )
    assert events == ["verdict-flushed", "close"]


def test_all_training_reexecs_use_common_resolver() -> None:
    offenders: list[str] = []
    tracked_python = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for relative in tracked_python:
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {
                "_resolve_env_py",
                "_env_py",
            }:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders
    for relative in (
        "scripts/rsl_rl/train.py",
        "Phase1/BinPick_Train.py",
        "Phase1/BinPick_Env.py",
        "Phase2/VisionPick_Train.py",
        "Phase2/VisionPick_Env.py",
    ):
        assert "reexec_in_training_python(__file__)" in _source(relative)


def test_extension_package_discovery_and_actuator_fields() -> None:
    setup_src = _source("source/ur5e_binpick/setup.py")
    actuator_src = _source(
        "source/ur5e_binpick/ur5e_binpick/tasks/binpick/mdp/ur5e_assets.py"
    )
    assert "packages=find_packages()" in setup_src
    assert 'python_requires=">=3.11,<3.12"' in setup_src
    assert "velocity_limit_sim=3.1416" in actuator_src
    assert "effort_limit_sim=150.0" in actuator_src
    assert "effort_limit_sim=140.0" in actuator_src
    assert "velocity_limit=3.1416" not in actuator_src
    grasp_physics = _source("Phase0/Grasp_PhysicsTest.py")
    assert "effort_limit_sim=140.0" in grasp_physics
    assert "effort_limit=140.0" not in grasp_physics


def test_preflight_module_imports_without_isaac() -> None:
    path = ROOT / "scripts" / "preflight_training.py"
    spec = importlib.util.spec_from_file_location("training_preflight_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.EXPECTED_DISTRIBUTIONS["setuptools"] == {"81.0.0"}
    assert module.EXPECTED_DISTRIBUTIONS["wheel"] == {"0.45.1"}
    assert module.EXPECTED_DISTRIBUTIONS["matplotlib"] == {"3.10.3"}
    assert module.EXPECTED_DISTRIBUTIONS["openpyxl"] == {"3.1.5"}
    assert module.EXPECTED_DISTRIBUTIONS["pandas"] == {"2.2.3"}
    assert module.EXPECTED_DISTRIBUTIONS["tensordict"] == {"0.13.0"}
    assert 'sys.getwindowsversion().build' in _source("scripts/preflight_training.py")


def test_nvidia_smi_query_is_parsed_without_gpu(monkeypatch) -> None:
    path = ROOT / "scripts" / "preflight_training.py"
    spec = importlib.util.spec_from_file_location("training_preflight_smi_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="580.88, NVIDIA GeForce RTX 5080, 16376\n",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module.query_nvidia_smi() == ("580.88", "NVIDIA GeForce RTX 5080", 16376.0)
    assert calls == [[
        "nvidia-smi",
        "--query-gpu=driver_version,name,memory.total",
        "--format=csv,noheader,nounits",
    ]]


def test_nvidia_smi_query_fails_closed_on_bad_output(monkeypatch) -> None:
    path = ROOT / "scripts" / "preflight_training.py"
    spec = importlib.util.spec_from_file_location("training_preflight_smi_bad_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="not,csv\n"),
    )
    try:
        module.query_nvidia_smi()
    except RuntimeError as exc:
        assert "unexpected nvidia-smi output" in str(exc)
    else:
        raise AssertionError("malformed nvidia-smi output must fail closed")


def test_authored_absolute_usd_asset_regex() -> None:
    path = ROOT / "scripts" / "preflight_training.py"
    spec = importlib.util.spec_from_file_location("training_preflight_asset_path_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    absolute = [
        r"@C:\\Users\\name\\asset.usd@",
        "@/home/name/asset.usd@",
        "@/opt/project/asset.usd@",
        "@/mnt/data/asset.usd@",
        "@file:///C:/asset.usd@",
        r"@\\\\server\\share\\asset.usd@",
    ]
    for authored in absolute:
        assert module._AUTHORED_ABSOLUTE_ASSET_RE.search(authored), authored
    for authored in ("@./asset.usd@", "@../asset.usd@", "@configuration/base.usd@"):
        assert not module._AUTHORED_ABSOLUTE_ASSET_RE.search(authored), authored


def test_training_python_resolver_fails_closed_without_env(monkeypatch) -> None:
    from scripts import training_env

    monkeypatch.setattr(sys, "prefix", str(ROOT / "system-python"))

    try:
        training_env.resolve_training_python(
            {"PATH": "", "USERPROFILE": str(ROOT / "missing-home")}
        )
    except FileNotFoundError as exc:
        assert "bootstrap_training.ps1" in str(exc)
    else:
        raise AssertionError("system Python fallback must fail closed")


def test_training_reexec_is_inert_when_entrypoint_module_is_imported(monkeypatch, tmp_path) -> None:
    from scripts import training_env

    script = tmp_path / "train_entry.py"
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["pytest", "-q"])
    monkeypatch.setattr(
        training_env,
        "resolve_training_python",
        lambda: (_ for _ in ()).throw(AssertionError("resolver must not run on import")),
    )
    training_env.reexec_in_training_python(script)


def test_training_python_candidates_use_conda_json_custom_envs_dir(monkeypatch, tmp_path) -> None:
    from scripts import training_env

    conda = tmp_path / "portable" / "Scripts" / "conda.exe"
    conda.parent.mkdir(parents=True)
    conda.touch()
    custom_env = tmp_path / "custom-envs" / "isaac51"
    custom_python = training_env._python_in_env(custom_env)
    custom_python.parent.mkdir(parents=True)
    custom_python.touch()

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"envs": [str(custom_env)]}),
            stderr="",
        )

    monkeypatch.setattr(training_env.subprocess, "run", fake_run)
    candidates = training_env.training_python_candidates(
        {
            "CONDA_EXE": str(conda),
            "PATH": "",
            "USERPROFILE": str(tmp_path / "home"),
        }
    )
    assert custom_python.resolve() in [path.resolve() for path in candidates]
    assert calls == [[str(conda), "env", "list", "--json"]]


def test_training_python_resolver_rejects_ambiguous_named_envs(monkeypatch, tmp_path) -> None:
    from scripts import training_env

    envs = []
    for root_name in ("one", "two"):
        env_dir = tmp_path / root_name / "isaac51"
        python = training_env._python_in_env(env_dir)
        python.parent.mkdir(parents=True)
        python.touch()
        envs.append(env_dir)
    monkeypatch.setattr(
        training_env,
        "training_python_candidates",
        lambda _environ=None: [training_env._python_in_env(env_dir) for env_dir in envs],
    )
    try:
        training_env.resolve_training_python({"PATH": ""})
    except RuntimeError as exc:
        assert "multiple isaac51 environments found" in str(exc)
    else:
        raise AssertionError("ambiguous canonical environment names must fail closed")


def test_training_python_resolver_rejects_noncanonical_override(tmp_path) -> None:
    from scripts import training_env

    wrong_env = tmp_path / "other-env"
    wrong_python = training_env._python_in_env(wrong_env)
    wrong_python.parent.mkdir(parents=True)
    wrong_python.touch()
    try:
        training_env.resolve_training_python({"ISAACLAB_ENV_PY": str(wrong_python)})
    except RuntimeError as exc:
        assert "must belong to the isaac51 environment" in str(exc)
    else:
        raise AssertionError("noncanonical explicit interpreter must fail closed")


def test_train_smoke_rejects_semantic_extra_args_before_environment_resolution() -> None:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is not installed on this pure-test host")
    proc = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "train.ps1"),
            "-Smoke",
            "--resume",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "-Smoke rejects all ExtraArgs" in proc.stdout


def test_train_forbids_skip_preflight_alias_before_environment_resolution() -> None:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is not installed on this pure-test host")
    proc = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "train.ps1"),
            "-SkipPreflight",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "preflight bypass is forbidden" in proc.stdout


@pytest.mark.parametrize(
    "preset,argument,expected",
    [
        ("Teacher", "--task=Other-v0", "preset-owned training identity"),
        ("Vision", "--resume", "warm-start flags are owned"),
        ("Distill", "--nurec-finetune=x.pt", "preset-owned training identity"),
        ("Vision", "--device=cpu", "outside the sealed launcher allowlist"),
    ],
)
def test_train_rejects_preset_identity_overrides_before_environment_resolution(
    preset, argument, expected
) -> None:
    shell = shutil.which("powershell") or shutil.which("pwsh")
    if shell is None:
        pytest.skip("PowerShell is not installed on this pure-test host")
    proc = subprocess.run(
        [
            shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            str(ROOT / "scripts" / "train.ps1"), "-Preset", preset, argument,
        ],
        text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30,
    )
    assert proc.returncode != 0
    assert expected in proc.stdout


def test_python_wrappers_reject_task_identity_override() -> None:
    teacher = _source("Phase1/BinPick_Train.py")
    vision = _source("Phase2/VisionPick_Train.py")
    assert "Teacher wrapper task is fixed" in teacher
    assert 'reserved = {"--task", "--experiment_name", "--resume", "--load_run", "--checkpoint"}' in vision
    assert "Vision wrapper owns task/experiment/warm-start flag" in vision
    for wrapper in (teacher, vision):
        assert "_require_full_preflight" in wrapper
    low_level = _source("scripts/rsl_rl/train.py")
    assert "mandatory low-level training preflight failed" in low_level
    assert '"preflight_training.py"' in low_level
    assert "UR5E_TRAINING_PREFLIGHT_OK" not in low_level
    assert low_level.index("preflight_training.py") < low_level.index("from isaaclab.app import AppLauncher")


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/rsl_rl/train.py",
        "scripts/rsl_rl/play.py",
        "Phase1/BinPick_Eval.py",
        "Phase1/BinPick_PlacedProbe.py",
        "Phase1/BinPick_Progression.py",
        "diagnostics/render_pickplace.py",
        "diagnostics/rollout_probe.py",
        "diagnostics/rgbd_failure_probe.py",
    ],
)
def test_rsl_app_launchers_preload_native_modules_before_kit(relative: str) -> None:
    source = _source(relative)
    assert source.index("preload_native_training_modules()") < source.index(
        "from isaaclab.app import AppLauncher"
    )


def test_low_level_train_preloads_only_after_mandatory_preflight() -> None:
    source = _source("scripts/rsl_rl/train.py")
    assert source.index("mandatory low-level training preflight failed") < source.index(
        "preload_native_training_modules()"
    )


def test_evaluation_output_directory_collision_fails_without_reusing_artifacts(
    tmp_path,
) -> None:
    source = _source("Phase1/BinPick_Eval.py")
    parsed = ast.parse(source)
    creator_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_create_evaluation_output_directory"
    )

    class _FixedStamp:
        def strftime(self, pattern: str) -> str:
            assert pattern == "%Y-%m-%d_%H-%M-%S"
            return "2026-08-15_12-34-56"

    class _FixedDatetime:
        @staticmethod
        def now() -> _FixedStamp:
            return _FixedStamp()

    namespace = {"os": os, "datetime": _FixedDatetime}
    isolated = ast.Module(body=[creator_node], type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, "BinPick_Eval.py", "exec"), namespace)
    create_output = namespace["_create_evaluation_output_directory"]

    phase_dir = tmp_path / "Phase1"
    result_dir = phase_dir / "Result"
    result_dir.mkdir(parents=True)
    output = Path(create_output(str(phase_dir)))
    assert output == result_dir / "eval_2026-08-15_12-34-56"
    sentinel = output / "episodes.csv"
    sentinel.write_bytes(b"sealed-prior-evaluation")

    with pytest.raises(FileExistsError):
        create_output(str(phase_dir))
    assert sentinel.read_bytes() == b"sealed-prior-evaluation"


def test_vision_warmstart_seed_is_exclusive_hashed_and_narrowly_cleaned(
    monkeypatch, tmp_path
) -> None:
    path = ROOT / "Phase2" / "VisionPick_Train.py"
    spec = importlib.util.spec_from_file_location("vision_seed_safety_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = tmp_path / "logs" / "rsl_rl" / "ur5e_binpick_vision"
    monkeypatch.setattr(module, "RUNS_ROOT", str(runs))
    source = tmp_path / "model_7.pt"
    source.write_bytes(b"sealed-checkpoint")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    name_a, seed_a = module._seed_warmstart_ckpt(
        str(source), "distill", expected_sha256=digest
    )
    name_b, seed_b = module._seed_warmstart_ckpt(
        str(source), "distill", expected_sha256=digest
    )
    assert name_a != name_b
    assert (seed_a / "model_0.pt").read_bytes() == b"sealed-checkpoint"
    assert (seed_b / "model_0.pt").read_bytes() == b"sealed-checkpoint"
    assert module._verify_warmstart_seed(seed_a, digest) == seed_a / "model_0.pt"
    (seed_b / "model_0.pt").write_bytes(b"tampered-after-copy")
    with pytest.raises(module.ContractError, match="seed changed"):
        module._verify_warmstart_seed(seed_b, digest)
    module._remove_warmstart_seed(seed_a)
    module._remove_warmstart_seed(seed_b)
    assert not seed_a.exists() and not seed_b.exists()


def test_vision_warmstart_rejects_opened_bytes_from_a_swap_back_race(
    monkeypatch, tmp_path,
) -> None:
    path = ROOT / "Phase2" / "VisionPick_Train.py"
    spec = importlib.util.spec_from_file_location("vision_seed_race_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runs = tmp_path / "logs" / "rsl_rl" / "ur5e_binpick_vision"
    monkeypatch.setattr(module, "RUNS_ROOT", str(runs))
    source = tmp_path / "model_7.pt"
    trusted = b"adopted-checkpoint"
    source.write_bytes(trusted)
    digest = hashlib.sha256(trusted).hexdigest()
    real_open = Path.open

    def raced_open(self, mode="r", *args, **kwargs):
        if self == source and mode == "rb":
            # Return the bytes present during the vulnerable copy, while making
            # the pathname look trusted again before any later path recheck.
            source.write_bytes(trusted)
            return io.BytesIO(b"unadopted-checkpoint")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raced_open)
    with pytest.raises(module.ContractError, match="opened bytes differ"):
        module._seed_warmstart_ckpt(
            str(source), "distill", expected_sha256=digest
        )
    with real_open(source, "rb") as stream:
        assert hashlib.sha256(stream.read()).hexdigest() == digest
    assert not list(runs.glob("_distill-warmstart-*"))


def test_vision_warmstart_is_rehashed_immediately_before_child_run() -> None:
    source = _source("Phase2/VisionPick_Train.py")
    parent_gate = source.index("pre_run_teacher != teacher_input")
    seed_gate = source.index("_verify_warmstart_seed(seed_dir, warmstart_sha256)", parent_gate)
    run_call = source.index("rc, run_dir = _run", seed_gate)
    assert parent_gate < seed_gate < run_call


def test_vision_warmstart_validates_only_the_anchored_seed() -> None:
    source = _source("Phase2/VisionPick_Train.py")
    builder = source.index("def _build_cmd(")
    distill_copy = source.index(
        'distill_from, "distill", expected_sha256=distill_checkpoint_sha256',
        builder,
    )
    distill_shape = source.index(
        "_verify_teacher_ckpt(str(seeded_checkpoint))", distill_copy
    )
    nurec_copy = source.index(
        'nurec_from, "nurec", expected_sha256=nurec_checkpoint_sha256',
        distill_shape,
    )
    nurec_shape = source.index(
        "_verify_student_ckpt(str(seeded_checkpoint))", nurec_copy
    )
    assert distill_copy < distill_shape < nurec_copy < nurec_shape
    build_source = source[builder : source.index("def _validate_forwarded_args", builder)]
    assert "_verify_teacher_ckpt(distill_from)" not in build_source
    assert "_verify_student_ckpt(nurec_from)" not in build_source


def test_export_checkpoint_snapshot_rejects_swap_back_opened_bytes(
    monkeypatch, tmp_path,
) -> None:
    path = ROOT / "scripts" / "checkpoint_io.py"
    spec = importlib.util.spec_from_file_location("checkpoint_io_race_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    store = tmp_path / "logs" / "rsl_rl" / "ur5e_binpick_vision"
    run = store / "run"
    run.mkdir(parents=True)
    source = run / "model_7.pt"
    trusted = b"adopted-student-checkpoint"
    source.write_bytes(trusted)
    digest = hashlib.sha256(trusted).hexdigest()
    real_open = Path.open

    def raced_open(self, mode="r", *args, **kwargs):
        if self == source and mode == "rb":
            source.write_bytes(trusted)
            return io.BytesIO(b"unadopted-student-checkpoint")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raced_open)
    with pytest.raises(module.ContractError, match="opened bytes differ"):
        module.load_anchored_checkpoint(
            source,
            expected_sha256=digest,
            allowed_root=store,
            torch_module=object(),
        )
    assert not list(run.glob("_verified-checkpoint-*"))


@pytest.mark.parametrize(
    "relative,invalid",
    [
        ("Phase1/BinPick_Train.py", ["env.decimation=1"]),
        ("Phase1/BinPick_Train.py", ["--device", "cpu"]),
        ("Phase2/VisionPick_Train.py", ["sim.dt=0.1"]),
        ("Phase2/VisionPick_Train.py", ["--distributed"]),
    ],
)
def test_python_wrappers_reject_unsealed_semantic_arguments(relative, invalid) -> None:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("semantic_arg_gate_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(Exception, match="unsupported/Hydra"):
        module._validate_forwarded_args(invalid)


def test_readme_exposes_only_public_training_launcher() -> None:
    readme = _training_doc()
    assert "내부 구현이며 공개 진입점이 아니다" in readme
    assert "python Phase1/BinPick_Train.py" not in readme
    assert "python Phase2/VisionPick_Train.py" not in readme
    for preset in ("Teacher", "Vision", "Distill", "NuRec"):
        assert f"scripts/train.ps1 -Preset {preset}" in readme
    launcher = _source("scripts/train.ps1")
    assert '[ValidateSet("Teacher", "Vision", "Distill", "NuRec")]' in launcher
    assert '"--nurec-finetune", $NuRecFrom' in launcher
    assert '$preflight += "--require-nurec"' in launcher
