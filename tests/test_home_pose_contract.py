"""홈 포즈 계약 — 안전·동기·도달성 회귀 방지 (sim 불필요, README 검증 계약).

배경(2026-07-28 실사고): 손목 카메라 가시율을 위해 홈 포즈를 (0,-90,+45,-32,-90,0)으로 교체했더니
      홈 TCP z가 0.5249가 되어 ★safety.yaml workspace 상한(0.50)을 넘었다★. SafetyGuard.check()는
      TCP AABB 이탈을 즉시 trip·latch하므로(common/safety.py:70), 배포·리허설이 ★홈 자세에서
      기동조차 못 하는★ 상태가 됐다. 테스트가 없어 pytest 290개가 전부 통과한 채로 커밋됐다.

여기서 고정하는 계약(전부 순수 계산 — isaac 불필요):
  ① 홈 TCP ⊂ safety workspace AABB          (없으면 배포 즉시 trip)
  ② robot_spec 홈 == configs/real.example.yaml home_joints_rad  (손 동기 구조라 어긋나기 쉽다)
  ③ 홈 FK == events_dr._PREGRASP_GRIP        (test_urdf_fk와 중복이지만 홈 변경 시 함께 깨져야 함)
  ④ target.sample_range ⊂ safety workspace AABB
  ⑤ 홈·타겟이 UR5e 물리 도달 범위 안
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_URDF = _ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf"

# scripts/tools는 패키지가 아님 — 파일 경로로 로드(test_urdf_fk.py와 동일 패턴)
_spec = importlib.util.spec_from_file_location("urdf_fk", _ROOT / "scripts" / "tools" / "urdf_fk.py")
urdf_fk = importlib.util.module_from_spec(_spec)
sys.modules["urdf_fk"] = urdf_fk
_spec.loader.exec_module(urdf_fk)

from common.calibration import load_tcp_offset  # noqa: E402
from common.config_io import build_safety_cfg, configs_dir, load_yaml  # noqa: E402
from common.robot_spec import UR5E_ARM_JOINT_NAMES, UR5E_HOME_JOINT_POS  # noqa: E402

# UR5e 링크 길이(URDF) — 어깨~손목 최대 도달
_UPPER_ARM = 0.425
_FOREARM = 0.3922


@pytest.fixture(scope="module")
def home_tcp() -> np.ndarray:
    """robot_spec 홈 포즈의 파지중심(base frame) — FK × tcp_offset."""
    if not _URDF.is_file():
        pytest.skip(f"URDF 없음: {_URDF}")
    joints = urdf_fk.load_joints(_URDF)
    poses = urdf_fk.fk(joints, dict(UR5E_HOME_JOINT_POS))
    t_ft = load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")
    return (poses["wrist_3_link"] @ t_ft)[:3, 3]


@pytest.fixture(scope="module")
def safety():
    return build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))


def test_home_tcp_inside_safety_workspace(home_tcp: np.ndarray, safety) -> None:
    """★핵심★ 홈 TCP가 safety workspace AABB 안이어야 한다.

    밖이면 SafetyGuard가 시작 즉시 trip·latch → 배포/리허설 기동 불가(2026-07-28 실사고).
    """
    lo = np.asarray(safety.workspace_min)
    hi = np.asarray(safety.workspace_max)
    assert np.all(home_tcp >= lo) and np.all(home_tcp <= hi), (
        f"홈 TCP {np.round(home_tcp, 4).tolist()}가 workspace "
        f"[{lo.tolist()}, {hi.tolist()}] 밖 — configs/safety.yaml 또는 홈 포즈를 조정할 것"
    )


def test_home_tcp_has_margin_from_workspace_bounds(home_tcp: np.ndarray, safety) -> None:
    """경계에 아슬아슬하면 센서 노이즈/오차만으로 trip한다 — 최소 1cm 여유 요구."""
    lo = np.asarray(safety.workspace_min)
    hi = np.asarray(safety.workspace_max)
    margin = float(min(np.min(home_tcp - lo), np.min(hi - home_tcp)))
    assert margin >= 0.01, f"홈 TCP가 workspace 경계에서 {margin*1000:.1f}mm — 여유 부족(≥10mm 필요)"


def test_robot_spec_home_matches_real_yaml() -> None:
    """robot_spec 홈 ↔ configs/real.example.yaml home_joints_rad 값 동기(손 동기 구조 — 어긋나기 쉽다).

    어긋나면 sim이 학습한 시작 자세와 실기 홈 복귀 자세가 달라진다(robot_spec.py:20 경고).
    """
    real = load_yaml(configs_dir() / "real.example.yaml")["robot"]["home_joints_rad"]
    spec = [float(UR5E_HOME_JOINT_POS[j]) for j in UR5E_ARM_JOINT_NAMES]
    assert len(real) == len(spec) == 6
    assert np.allclose(np.asarray(real, dtype=float), np.asarray(spec), atol=1e-6), (
        f"홈 불일치 — real.example.yaml {list(real)} vs robot_spec {spec}"
    )


def test_home_tcp_matches_pregrasp_constant(home_tcp: np.ndarray) -> None:
    """홈 FK 파지중심 == events_dr._PREGRASP_GRIP (isaaclab import 없이 소스에서 파싱).

    홈만 바꾸고 이 상수를 안 고치면 pregrasp 보조가 엉뚱한 좌표에 큐브를 놓는다.
    """
    import re

    _p = (_ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
          / "mdp" / "events_dr.py")
    if not _p.is_file():
        # 생성 deployment bundle에는 sim 확장이 없다 — 대조할 sim 쪽이 없으므로 skip.
        pytest.skip("sim 확장(source/) 없음 — 배포 번들에서는 대조 불가")
    src = _p.read_text(encoding="utf-8")
    m = re.search(r"^_PREGRASP_GRIP\s*=\s*\(([^)]*)\)", src, re.M)
    assert m, "_PREGRASP_GRIP 상수를 찾지 못함"
    grip = np.array([float(v) for v in m.group(1).split(",")[:3]])
    assert np.linalg.norm(home_tcp - grip) < 3e-3, (
        f"홈 FK {np.round(home_tcp, 4).tolist()} ≠ _PREGRASP_GRIP {grip.tolist()} "
        "— 홈 포즈를 바꿨으면 이 상수도 재산출할 것"
    )


def test_target_range_inside_safety_workspace(safety) -> None:
    """place 목표 범위 전체가 safety AABB 안 — 밖이면 목표에 도달하는 순간 trip한다."""
    tgt = load_yaml(configs_dir() / "common.yaml")["target"]["sample_range"]
    lo = np.asarray(safety.workspace_min)
    hi = np.asarray(safety.workspace_max)
    for corner_x in tgt["x"]:
        for corner_y in tgt["y"]:
            for corner_z in tgt["z"]:
                p = np.array([float(corner_x), float(corner_y), float(corner_z)])
                assert np.all(p >= lo) and np.all(p <= hi), f"타겟 코너 {p.tolist()}가 workspace 밖"


def test_home_and_targets_within_reach(home_tcp: np.ndarray) -> None:
    """홈·타겟이 UR5e 물리 도달 범위(상완+전완) 안 — IK가 풀리지 않는 목표를 주지 않게."""
    reach = _UPPER_ARM + _FOREARM  # 0.8172 m (어깨 기준)
    assert float(np.linalg.norm(home_tcp[:2])) < reach, "홈이 도달 범위 밖"
    tgt = load_yaml(configs_dir() / "common.yaml")["target"]["sample_range"]
    far = math.hypot(max(abs(float(v)) for v in tgt["x"]), max(abs(float(v)) for v in tgt["y"]))
    assert far < reach, f"최원 타겟 반경 {far:.3f}m ≥ 도달 한계 {reach:.3f}m"


def test_home_gripper_starts_open() -> None:
    """홈은 '파지 전' 상태여야 한다 — 그리퍼가 닫힌 채 시작하면 리셋 직후 큐브를 밀어낸다."""
    _p = (_ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
          / "mdp" / "ur5e_assets.py")
    if not _p.is_file():
        pytest.skip("sim 확장(source/) 없음 — 배포 번들에서는 대조 불가")
    src = _p.read_text(encoding="utf-8")
    assert "GRIPPER_OPEN_M" in src and '"left_finger_joint": GRIPPER_OPEN_M' in src, (
        "홈 초기 그리퍼가 열림(GRIPPER_OPEN_M)으로 배선돼 있어야 한다"
    )
