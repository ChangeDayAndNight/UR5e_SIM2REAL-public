#!/usr/bin/env python
"""Isaac Sim 5.1.0 / Isaac Lab 2.3.2 학습 환경과 자산을 fail-fast 검증한다.

표준 실행::

    python scripts/preflight_training.py

`--static`은 Isaac/GPU import 없이 저장소·manifest·태스크 등록 소스만 검사하므로 CI와
개발 중 사용할 수 있다. 표준 실행은 학습을 시작해도 되는 환경만 성공(0)으로 판정한다.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

# A bootstrap invoked with ``powershell -File`` cannot activate the caller's
# next shell.  For the documented non-static CLI, re-exec while imports are
# still stdlib-only; common.nurec_provenance reaches PyYAML through the release
# contracts and must not run in an arbitrary base Python first.
if __name__ == "__main__" and "--static" not in sys.argv[1:]:
    reexec_in_training_python(__file__)

from common.nurec_provenance import load_nurec_provenance
from deployment.contracts import ContractError
from common.training_stack_contract import (
    EXPECTED_TRAINING_DISTRIBUTIONS,
    ISAAC_LAB_COMMIT,
    MIN_TRAINING_VRAM_MIB,
    MIN_WINDOWS_BUILD,
    TRAINING_OS,
    TRAINING_PYTHON_VERSION,
)


MANIFEST_PATH = REPO_ROOT / "assets" / "asset_manifest.json"
ISAACLAB_COMMIT = ISAAC_LAB_COMMIT
EXPECTED_TASKS = {
    "Isaac-BinPick-UR5e-Teacher-v0",
    "Isaac-BinPick-UR5e-Vision-v0",
    "Isaac-BinPick-UR5e-Vision-Distill-v0",
    "Isaac-BinPick-UR5e-Vision-NuRec-v0",
}
EXPECTED_ASSET_PATHS = {
    "assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd",
    "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_base.usd",
    "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_physics.usd",
    "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_robot.usd",
    "assets/ur5e_2fg7/configuration/ur5e_tcw1_2fg7_sensor.usd",
    "assets/d435i/d435i.usd",
    "assets/d435i/Props/instanceable_meshes.usd",
}
EXPECTED_DISTRIBUTIONS = {
    name: set(versions) for name, versions in EXPECTED_TRAINING_DISTRIBUTIONS.items()
}

_KIT_CHILD_RESULT_PREFIX = "UR5E_KIT_PREFLIGHT_RESULT="
_KIT_CHILD_OK = "UR5E_KIT_PREFLIGHT_OK"
_KIT_CHILD_FAIL = "UR5E_KIT_PREFLIGHT_FAIL"
_RUNTIME_PROVIDED_MDL_ASSETS = frozenset({"OmniPBR.mdl"})


_AUTHORED_ABSOLUTE_ASSET_RE = re.compile(
    r"@(?:[A-Za-z]:[\\/]|/|file://|\\\\)[^@]*@",
    re.IGNORECASE,
)


class CheckReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.ok: list[str] = []

    def require(self, condition: bool, success: str, failure: str) -> None:
        (self.ok if condition else self.errors).append(success if condition else failure)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def render(self) -> None:
        for message in self.ok:
            print(f"[preflight][ok] {message}")
        for message in self.warnings:
            print(f"[preflight][warn] {message}")
        for message in self.errors:
            print(f"[preflight][error] {message}")
        if self.errors:
            print(f"[preflight] 실패: error={len(self.errors)} warn={len(self.warnings)}")
        else:
            print(f"[preflight] TRAINING_PREFLIGHT_OK warn={len(self.warnings)}")
        sys.stdout.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partition_unresolved_assets(unresolved: Iterable[object]) -> tuple[list[str], list[str]]:
    """Separate exact Isaac runtime MDL identifiers from broken asset paths."""

    paths = sorted(str(path) for path in unresolved)
    runtime_mdl = [path for path in paths if path in _RUNTIME_PROVIDED_MDL_ASSETS]
    unexpected = [path for path in paths if path not in _RUNTIME_PROVIDED_MDL_ASSETS]
    return runtime_mdl, unexpected


def load_asset_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema_version") != 1:
        raise ValueError(f"지원하지 않는 asset manifest schema: {data.get('schema_version')!r}")
    return data


def validate_assets(report: CheckReport) -> None:
    try:
        manifest = load_asset_manifest()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.errors.append(f"자산 manifest를 읽을 수 없음: {MANIFEST_PATH}: {exc}")
        return

    entries = manifest.get("bundled_assets", [])
    report.require(bool(entries), "자산 manifest 항목 존재", "자산 manifest가 비어 있음")
    seen: set[str] = set()
    for entry in entries:
        relative = str(entry.get("path", ""))
        rel_path = Path(relative)
        safe = bool(relative) and not rel_path.is_absolute() and ".." not in rel_path.parts
        if not safe:
            report.errors.append(f"manifest에 안전하지 않은 경로: {relative!r}")
            continue
        if relative in seen:
            report.errors.append(f"manifest 중복 경로: {relative}")
            continue
        seen.add(relative)
        path = REPO_ROOT / rel_path
        if not path.is_file():
            report.errors.append(f"필수 USD 없음: {relative}")
            continue
        expected_size = int(entry.get("size", -1))
        report.require(
            path.stat().st_size == expected_size,
            f"자산 크기 일치: {relative}",
            f"자산 크기 불일치: {relative} expected={expected_size} actual={path.stat().st_size}",
        )
        actual_hash = _sha256(path)
        expected_hash = str(entry.get("sha256", "")).lower()
        report.require(
            actual_hash == expected_hash,
            f"자산 SHA256 일치: {relative}",
            f"자산 SHA256 불일치: {relative} expected={expected_hash} actual={actual_hash}",
        )
    report.require(
        seen == EXPECTED_ASSET_PATHS,
        "canonical USD bundle 7개 manifest 고정",
        "canonical USD manifest 경로 불일치: "
        f"missing={sorted(EXPECTED_ASSET_PATHS - seen)} extra={sorted(seen - EXPECTED_ASSET_PATHS)}",
    )

    # 실수로 다시 ignore하거나 index에 추가하지 않으면 fresh clone에서 manifest만 남고
    # 자산이 사라진다. 두 조건을 독립적으로 검사한다.
    if (REPO_ROOT / ".git").exists():
        for relative in ["assets/asset_manifest.json", *sorted(seen)]:
            proc = subprocess.run(
                ["git", "check-ignore", "-q", "--", relative],
                cwd=REPO_ROOT,
                check=False,
            )
            if proc.returncode == 0:
                report.errors.append(f"필수 자산이 .gitignore 대상임: {relative}")
            tracked = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", relative],
                cwd=REPO_ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            report.require(
                tracked.returncode == 0,
                f"Git 추적 자산: {relative}",
                f"필수 자산이 Git index에 없음(fresh clone에서 누락됨): {relative}",
            )


def _validate_usd_dependencies_in_kit_child(
    report: CheckReport,
    before_close: Callable[[CheckReport], None],
) -> None:
    """Launch Kit, verify the live Gym registry, and resolve every USD layer."""

    simulation_app = None
    try:
        # Always launch Kit. Importing registration source or parsing its AST is
        # not proof that the installed extension registered tasks at runtime.
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(headless=True).app
        from pxr import Sdf, Usd, UsdUtils
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import ur5e_binpick.tasks  # noqa: F401
    except Exception as exc:  # pragma: no cover - 실제 Isaac 환경 진단
        report.errors.append(f"Isaac AppLauncher/runtime import 실패: {exc}")
        before_close(report)
        if simulation_app is not None:
            simulation_app.close()
        return

    for task_id in sorted(EXPECTED_TASKS):
        try:
            spec = gym.spec(task_id)
        except Exception as exc:
            report.errors.append(f"runtime Gym task 미등록: {task_id}: {exc}")
        else:
            report.require(
                spec is not None and spec.id == task_id,
                f"runtime Gym task 등록: {task_id}",
                f"runtime Gym task spec 불일치: {task_id}: {spec!r}",
            )

    try:
        entries = load_asset_manifest().get("bundled_assets", [])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.errors.append(f"USD dependency 검사 전 manifest 로드 실패: {exc}")
        before_close(report)
        if simulation_app is not None:
            simulation_app.close()
        return

    try:
        for entry in entries:
            relative = str(entry.get("path", ""))
            usd_path = REPO_ROOT / relative
            if not usd_path.is_file():
                continue
            try:
                stage = Usd.Stage.Open(str(usd_path))
            except Exception as exc:
                report.errors.append(f"USD Stage.Open 예외: {relative}: {exc}")
                continue
            report.require(
                stage is not None and stage.GetPseudoRoot().IsValid(),
                f"USD root stage 유효: {relative}",
                f"USD root stage를 열 수 없음: {relative}",
            )
            layer = Sdf.Layer.FindOrOpen(str(usd_path))
            if layer is None:
                report.errors.append(f"USD authored layer를 열 수 없음: {relative}")
            else:
                authored_text = layer.ExportToString()
                authored_absolutes = sorted(set(_AUTHORED_ABSOLUTE_ASSET_RE.findall(authored_text)))
                report.require(
                    not authored_absolutes,
                    f"USD authored absolute asset paths=0: {relative}",
                    f"USD authored absolute asset paths: {relative}: {authored_absolutes}",
                )
            try:
                _layers, _assets, unresolved = UsdUtils.ComputeAllDependencies(str(usd_path))
            except Exception as exc:
                report.errors.append(f"USD dependency 계산 실패: {relative}: {exc}")
                continue
            runtime_mdl, unresolved_paths = _partition_unresolved_assets(unresolved)
            if runtime_mdl:
                report.ok.append(
                    f"USD runtime-provided MDL identifiers: {relative}: {runtime_mdl}"
                )
            report.require(
                not unresolved_paths,
                f"USD unresolved dependencies=0: {relative}",
                f"USD unresolved references/payloads/assets: {relative}: {unresolved_paths}",
            )
    except Exception as exc:  # pragma: no cover - actual USD runtime diagnosis
        report.errors.append(f"unexpected USD dependency validation failure: {exc}")
    finally:
        before_close(report)
        if simulation_app is not None:
            simulation_app.close()


def _emit_kit_child_result(report: CheckReport) -> None:
    """Publish the complete check result before Kit shutdown changes control flow."""

    payload = json.dumps(
        {"ok": report.ok, "warnings": report.warnings, "errors": report.errors},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    print(_KIT_CHILD_RESULT_PREFIX + payload, flush=True)
    print(_KIT_CHILD_FAIL if report.errors else _KIT_CHILD_OK, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def _run_kit_child() -> int:
    """Run checks and let their callback publish a verdict before closing Kit."""

    report = CheckReport()
    try:
        _validate_usd_dependencies_in_kit_child(report, _emit_kit_child_result)
    except Exception as exc:  # pragma: no cover - actual Kit shutdown diagnosis
        # The normal and validation-failure paths already flushed their verdict
        # before close(). Reaching this handler therefore means close itself (or
        # an unexpected lifecycle error) failed and must remain non-zero.
        print(f"[preflight-kit][close-fail] {type(exc).__name__}: {exc}", flush=True)
        return 2
    return 1 if report.errors else 0


def _merge_kit_child_payload(report: CheckReport, output: str) -> bool:
    payload_lines = [
        line[len(_KIT_CHILD_RESULT_PREFIX) :]
        for line in output.splitlines()
        if line.startswith(_KIT_CHILD_RESULT_PREFIX)
    ]
    if len(payload_lines) != 1:
        report.errors.append(
            "Kit preflight child emitted an invalid result payload count: "
            f"expected=1 actual={len(payload_lines)}"
        )
        return False
    try:
        payload = json.loads(payload_lines[0])
    except json.JSONDecodeError as exc:
        report.errors.append(f"Kit preflight child result payload is invalid JSON: {exc}")
        return False
    if not isinstance(payload, dict) or any(
        not isinstance(payload.get(key), list)
        or any(not isinstance(item, str) for item in payload[key])
        for key in ("ok", "warnings", "errors")
    ):
        report.errors.append("Kit preflight child result payload has an invalid schema")
        return False
    report.ok.extend(payload["ok"])
    report.warnings.extend(payload["warnings"])
    report.errors.extend(payload["errors"])
    return True


def validate_usd_dependencies(report: CheckReport) -> None:
    """Run all Kit-owned checks behind a sentinel-verifying process gate."""

    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--kit-child"]
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.errors.append(f"Kit preflight child invocation failed: {exc}")
        return

    output = proc.stdout or ""
    if output:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    payload_ok = _merge_kit_child_payload(report, output)
    lines = [line.strip() for line in output.splitlines()]
    success_count = sum(line == _KIT_CHILD_OK for line in lines)
    failure_count = sum(line == _KIT_CHILD_FAIL for line in lines)
    if proc.returncode != 0:
        report.errors.append(f"Kit preflight child exited nonzero: exit={proc.returncode}")
    if success_count != 1:
        report.errors.append(
            "Kit preflight child missing exactly one success sentinel: "
            f"actual={success_count}"
        )
    if failure_count:
        report.errors.append(
            f"Kit preflight child emitted failure sentinel(s): count={failure_count}"
        )
    if not payload_ok:
        return


def _registered_task_ids(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "register":
            continue
        for keyword in node.keywords:
            if keyword.arg == "id" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    found.add(keyword.value.value)
    return found


def validate_source_assets(report: CheckReport) -> None:
    """★F93(리뷰 T-04)★ URDF↔USD 결합 — URDF만 고치고 USD를 재변환하지 않으면(또는 그 반대) 여기서 걸린다.

    F91의 교훈: 변환 산출물(USD)만 해시로 묶으면 URDF 편집이 조용히 지나간다. 소스(URDF)도 manifest에
    묶어, 한쪽만 바뀐 상태로 학습이 시작되지 못하게 한다.
    """
    try:
        entries = load_asset_manifest().get("source_assets", [])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report.errors.append(f"자산 manifest를 읽을 수 없음: {MANIFEST_PATH}: {exc}")
        return
    report.require(
        bool(entries),
        "source_assets(URDF) manifest 항목 존재",
        "asset_manifest.json에 source_assets가 없음 — URDF↔USD 결합이 풀렸다(F93/T-04)",
    )
    for entry in entries:
        relative = str(entry.get("path", ""))
        path = REPO_ROOT / relative
        if not path.is_file():
            report.errors.append(f"소스 자산 없음: {relative}")
            continue
        expected_size = int(entry.get("size", -1))
        report.require(
            path.stat().st_size == expected_size,
            f"소스 자산 크기 일치: {relative}",
            f"소스 자산 크기 불일치: {relative} expected={expected_size} actual={path.stat().st_size} — "
            "URDF를 고쳤으면 convert_2fg7_usd.py로 USD를 재변환하고 manifest의 USD·URDF 항목을 함께 갱신할 것",
        )
        actual_hash = _sha256(path)
        expected_hash = str(entry.get("sha256", "")).lower()
        report.require(
            actual_hash == expected_hash,
            f"소스 자산 SHA256 일치: {relative}",
            f"소스 자산 SHA256 불일치: {relative} — URDF↔USD 결합 위반(F93/T-04)",
        )


def validate_repository(report: CheckReport) -> None:
    task_file = (
        REPO_ROOT
        / "source"
        / "ur5e_binpick"
        / "ur5e_binpick"
        / "tasks"
        / "binpick"
        / "__init__.py"
    )
    try:
        registered = _registered_task_ids(task_file)
    except (OSError, SyntaxError) as exc:
        report.errors.append(f"태스크 등록 소스를 검사할 수 없음: {exc}")
        return
    missing = EXPECTED_TASKS - registered
    report.require(
        not missing,
        "Teacher/Vision/Distill/NuRec 태스크 ID 등록 확인",
        f"필수 태스크 ID 미등록: {sorted(missing)}",
    )

    source_files = (
        REPO_ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf",
        REPO_ROOT / "assets" / "d435i" / "meshes" / "d435.dae",
        REPO_ROOT / "scripts" / "rsl_rl" / "train.py",
        REPO_ROOT / "Phase1" / "BinPick_Train.py",
    )
    for path in source_files:
        report.require(path.is_file(), f"소스 존재: {path.relative_to(REPO_ROOT)}", f"소스 없음: {path}")


def validate_runtime_contract(
    report: CheckReport,
    *,
    common_path: Path | None = None,
    sim_path: Path | None = None,
) -> None:
    """Seal the training-facing timing, observation, and action contract.

    This deliberately runs in ``--static`` mode too. A fresh clone must reject
    config drift before starting Isaac or reserving GPU memory.
    """

    from common.config_io import build_action_cfg, build_preprocess_cfg, load_yaml
    from common.timing import derive_decimation

    common_path = common_path or REPO_ROOT / "configs" / "common.yaml"
    sim_path = sim_path or REPO_ROOT / "configs" / "sim.yaml"
    try:
        common_yaml = load_yaml(common_path)
        sim_yaml = load_yaml(sim_path)
        action_cfg = build_action_cfg(common_yaml)
        preprocess_cfg = build_preprocess_cfg(common_yaml)
        physics_hz = float(common_yaml["control"]["physics_hz"])
        control_hz = float(common_yaml["control"]["control_hz"])
        decimation = derive_decimation(physics_hz, control_hz)
        camera_width = int(sim_yaml["camera"]["width"])
        camera_height = int(sim_yaml["camera"]["height"])
        camera_contract = {
            "vertical_fov_deg": float(sim_yaml["camera"]["vertical_fov_deg"]),
            "horizontal_fov_deg": float(sim_yaml["camera"]["horizontal_fov_deg"]),
            "clipping_near_m": float(sim_yaml["camera"]["clipping_near_m"]),
        }
    except (KeyError, TypeError, ValueError, OSError) as exc:
        report.errors.append(f"training runtime contract load failed: {exc}")
        return

    report.require(
        physics_hz == 120.0 and control_hz == 20.0 and decimation == 6,
        "timing contract=120Hz/20Hz/decimation6",
        "timing contract mismatch: "
        f"physics={physics_hz} control={control_hz} decimation={decimation}",
    )
    report.require(
        action_cfg.mode == "ik_rel"
        and action_cfg.num_arm_joints == 6
        and action_cfg.action_dim == 7
        and action_cfg.action_scale_rad == 0.5
        # ★Teacher v2(F88)★ 스케일 = 배포 SafetyGuard 스텝 상한(0.25 m/s·0.5 rad/s × 0.05 s)
        and action_cfg.task_pos_scale_m == 0.0125
        and action_cfg.task_rot_scale_rad == 0.025
        and action_cfg.gripper_close_threshold == 0.0,
        "action contract=ik_rel/7D, pos/rot scale=0.0125/0.025, close<0",
        "action contract mismatch: "
        f"mode={action_cfg.mode} arm={action_cfg.num_arm_joints} dim={action_cfg.action_dim}",
    )
    # ★Student v2(F89)★ 렌더 격자 = FOV 종횡비(등방 픽셀): 112 × round(112·tan(21.25°)/tan(34.7°)) = 112×63.
    import math as _math

    expected_camera_height = int(round(
        preprocess_cfg.out_width
        * _math.tan(_math.radians(camera_contract["vertical_fov_deg"]) / 2.0)
        / _math.tan(_math.radians(camera_contract["horizontal_fov_deg"]) / 2.0)
    ))
    camera_matches = (
        preprocess_cfg.out_width == 112
        and preprocess_cfg.out_height == 112
        and preprocess_cfg.use_rgb
        and preprocess_cfg.depth_near_m == 0.20
        and preprocess_cfg.depth_far_m == 1.20
        and preprocess_cfg.crop_fraction == 1.0
        and preprocess_cfg.frame_mode == "full"
        and camera_width == preprocess_cfg.out_width
        and camera_height == expected_camera_height == 63
        and camera_contract == {
            "vertical_fov_deg": 42.5,
            "horizontal_fov_deg": 69.4,
            "clipping_near_m": 0.05,
        }
    )
    report.require(
        camera_matches,
        "camera/preprocess contract=112x112 RGB-D full (render 112x63), depth[0.20,1.20], FOV69.4x42.5",
        "camera/preprocess contract mismatch: "
        f"preprocess={preprocess_cfg.out_width}x{preprocess_cfg.out_height} "
        f"rgb={preprocess_cfg.use_rgb} camera={camera_width}x{camera_height}",
    )
    image_dim = preprocess_cfg.out_width * preprocess_cfg.out_height * (
        4 if preprocess_cfg.use_rgb else 1
    )
    proprio_dim = 6 + 6 + 1 + 7 + 3 + action_cfg.action_dim
    observation_dim = image_dim + proprio_dim
    report.require(
        observation_dim == 50_206,
        "policy observation contract=50,206D",
        f"policy observation contract mismatch: dim={observation_dim}",
    )


def _run_git(isaaclab_dir: Path, *args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["git", "-C", str(isaaclab_dir), *args],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def resolve_isaaclab_dir() -> Path:
    """Return the only supported in-repository Isaac Lab checkout."""

    return (REPO_ROOT / "external" / "IsaacLab").resolve()


def validate_isaaclab_checkout(report: CheckReport, isaaclab_dir: Path) -> None:
    version_file = isaaclab_dir / "VERSION"
    report.require(isaaclab_dir.is_dir(), f"Isaac Lab checkout: {isaaclab_dir}", f"Isaac Lab checkout 없음: {isaaclab_dir}")
    if not isaaclab_dir.is_dir():
        return
    version = version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else ""
    report.require(version == "2.3.2", "Isaac Lab VERSION=2.3.2", f"Isaac Lab VERSION 불일치: {version!r}")
    rc, commit = _run_git(isaaclab_dir, "rev-parse", "HEAD")
    report.require(rc == 0 and commit == ISAACLAB_COMMIT, "Isaac Lab commit=37ddf62", f"Isaac Lab commit 불일치: {commit}")
    rc, tag = _run_git(isaaclab_dir, "describe", "--tags", "--exact-match")
    report.require(rc == 0 and tag == "v2.3.2", "Isaac Lab tag=v2.3.2", f"Isaac Lab tag 불일치: {tag}")
    rc, dirty = _run_git(isaaclab_dir, "status", "--porcelain")
    report.require(
        rc == 0 and not dirty,
        "Isaac Lab checkout clean",
        f"Isaac Lab checkout이 clean하지 않음: {dirty or 'git status 실패'}",
    )


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def validate_distributions(report: CheckReport, isaaclab_dir: Path) -> None:
    for distribution, accepted in EXPECTED_DISTRIBUTIONS.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            report.errors.append(f"필수 distribution 미설치: {distribution}")
            continue
        report.require(
            installed in accepted,
            f"{distribution}={installed}",
            f"{distribution} 버전 불일치: {installed}; expected={sorted(accepted)}",
        )

    origins = {
        "isaaclab": isaaclab_dir / "source" / "isaaclab",
        "isaaclab_assets": isaaclab_dir / "source" / "isaaclab_assets",
        "isaaclab_rl": isaaclab_dir / "source" / "isaaclab_rl",
        "isaaclab_tasks": isaaclab_dir / "source" / "isaaclab_tasks",
        "ur5e_binpick": REPO_ROOT / "source" / "ur5e_binpick",
    }
    for module_name, expected_root in origins.items():
        spec = importlib.util.find_spec(module_name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        report.require(
            origin is not None and _within(origin, expected_root),
            f"{module_name} import origin={origin}",
            f"{module_name}가 다른 checkout을 가리킴: actual={origin} expected_under={expected_root}",
        )


def validate_pip(report: CheckReport) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    allowed = re.compile(
        r"^isaaclab 0\.54\.2 has requirement starlette==0\.49\.1, "
        r"but you have starlette 0\.45\.3\.$"
    )
    unexpected = [line for line in lines if line != "No broken requirements found." and not allowed.match(line)]
    if unexpected:
        report.errors.append("pip dependency 충돌: " + " | ".join(unexpected))
    elif any(allowed.match(line) for line in lines):
        report.warn("업스트림 metadata 충돌 1건 허용: Isaac Lab starlette 0.49.1 vs Isaac Sim/FastAPI 0.45.3")
    else:
        report.ok.append("pip check 통과")


def query_nvidia_smi() -> tuple[str, str, float]:
    """Return ``(driver_version, gpu_name, memory_total_mib)`` fail-closed."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"nvidia-smi invocation failed: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi exit={proc.returncode}: {proc.stdout.strip()}")

    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3 or not fields[0] or not fields[1]:
        raise RuntimeError(f"unexpected nvidia-smi output: {rows[0]!r}")
    try:
        memory_mib = float(fields[2])
    except ValueError as exc:
        raise RuntimeError(f"invalid nvidia-smi memory.total: {fields[2]!r}") from exc
    if memory_mib <= 0.0:
        raise RuntimeError(f"invalid nvidia-smi memory.total: {memory_mib}")
    return fields[0], fields[1], memory_mib


def validate_gpu(report: CheckReport) -> None:
    try:
        driver, smi_name, smi_memory_mib = query_nvidia_smi()
    except RuntimeError as exc:
        report.errors.append(str(exc))
    else:
        report.ok.append(
            f"NVIDIA driver={driver}, GPU={smi_name}, VRAM={smi_memory_mib / 1024.0:.1f}GiB (nvidia-smi)"
        )

    try:
        import torch
    except Exception as exc:  # pragma: no cover - 실제 환경 진단
        report.errors.append(f"torch import 실패: {exc}")
        return
    report.require(torch.version.cuda == "12.8", "PyTorch CUDA runtime=12.8", f"PyTorch CUDA runtime 불일치: {torch.version.cuda}")
    if not torch.cuda.is_available():
        report.errors.append("CUDA GPU를 사용할 수 없음")
        return
    count = torch.cuda.device_count()
    report.require(count > 0, f"CUDA device count={count}", "CUDA device가 없음")
    if count:
        props = torch.cuda.get_device_properties(0)
        total_gib = props.total_memory / (1024**3)
        report.require(
            props.total_memory / (1024**2) >= MIN_TRAINING_VRAM_MIB,
            f"GPU={props.name}, VRAM={total_gib:.1f}GiB",
            f"학습 GPU VRAM 부족: {props.name}, {total_gib:.1f}GiB (<7.5GiB)",
        )
        if 7.5 <= total_gib < 16.0:
            report.warn(
                f"GPU VRAM={total_gib:.1f}GiB: project Teacher smoke threshold is met, "
                "but Isaac Sim 5.1's published minimum is 16GiB; Vision/full-scale settings require "
                "an explicit low-env validation before launch"
            )


def validate_python_host(report: CheckReport) -> None:
    report.require(
        platform.python_version() == TRAINING_PYTHON_VERSION,
        f"Python={TRAINING_PYTHON_VERSION}",
        f"Python {TRAINING_PYTHON_VERSION} 필요: {platform.python_version()}",
    )
    is_windows = platform.system() == TRAINING_OS and hasattr(sys, "getwindowsversion")
    report.require(is_windows, "OS=Windows", f"Windows 11 native 필요: {platform.platform()}")
    if is_windows:
        build = int(sys.getwindowsversion().build)
        report.require(
            build >= MIN_WINDOWS_BUILD,
            f"Windows 11 build={build}",
            f"Windows 11 build>=22000 필요: build={build}",
        )


def run_checks(*, static: bool, isaaclab_dir: Path, require_nurec: bool, skip_gpu: bool) -> CheckReport:
    report = CheckReport()
    validate_repository(report)
    validate_runtime_contract(report)
    validate_assets(report)
    validate_source_assets(report)
    if require_nurec:
        try:
            nurec = load_nurec_provenance(REPO_ROOT)
        except ContractError as exc:
            report.errors.append(f"NuRec 자산/provenance 검증 실패: {exc}")
        else:
            report.ok.append(
                "NuRec PLY→USDZ provenance 봉인="
                + nurec["provenance_sha256"][:12]
            )
    elif not (REPO_ROOT / "assets" / "Scene" / "Workplace.usdz").is_file():
        report.warn("NuRec Workplace.usdz 없음(선택 기능이며 Teacher/Vision 표준 학습에는 영향 없음)")

    if not static:
        validate_python_host(report)
        validate_isaaclab_checkout(report, isaaclab_dir)
        validate_distributions(report, isaaclab_dir)
        validate_pip(report)
        validate_usd_dependencies(report)
        if not skip_gpu:
            validate_gpu(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac 5.1/Lab 2.3.2 학습 preflight")
    parser.add_argument("--static", action="store_true", help="Isaac/GPU 없이 저장소와 asset manifest만 검사")
    parser.add_argument("--require-nurec", action="store_true", help="선택 NuRec Workplace.usdz도 필수로 검사")
    parser.add_argument("--skip-gpu", action="store_true", help="패키지/checkout은 검사하되 CUDA probe 생략")
    parser.add_argument("--kit-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    forwarded_args = list(argv) if argv is not None else None
    args = build_parser().parse_args(forwarded_args)
    configure_utf8_stdio()
    if args.kit_child:
        return _run_kit_child()
    # Bootstrap은 자식 PowerShell 안에서만 환경 변수를 설정한다. 따라서 문서의 표준
    # `python scripts/preflight_training.py` 명령도 isaac51 Python으로 스스로 재실행해야 한다.
    # 정적 검사는 CI/개발 Python에서도 실행 가능하도록 재실행하지 않는다.
    if not args.static:
        reexec_in_training_python(__file__, forwarded_args)
    report = run_checks(
        static=args.static,
        isaaclab_dir=resolve_isaaclab_dir(),
        require_nurec=args.require_nurec,
        skip_gpu=args.skip_gpu,
    )
    report.render()
    return 1 if report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
