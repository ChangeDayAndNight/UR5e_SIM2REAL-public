"""UR 컨트롤러 Base ↔ URDF base_link 정합 (감사 C-2) — 순수 수학 검증, sim/실기 불필요.

배경(2026-07-28): URDF `base_link-base_fixed_joint`가 rpy=(0,0,π)로 명시하듯,
      ★UR 컨트롤러가 보고하는 Base 프레임 = base_link를 z축 180° 돌린 것★이다.
      sim tcp_pose는 base_link 기준(scene.py ee_frame source)이고 real RTDE
      getActualTCPPose()는 컨트롤러 Base 기준이라, 변환 없이 쓰면 ★x·y 부호가 반대★가 되어
      정책이 거울상 좌표를 받는다(감사 C-2 = 배포 차단).

여기서 고정하는 계약:
  ① 위치: (x,y,z) → (−x,−y,z)
  ② 자세: R → Rz(π)·R  (유효 회전 유지: 직교·det=+1)
  ③ 대합(involution): 두 번 적용 = 원복
  ④ wrench: force·torque 모두 같은 회전 (한쪽만 돌리면 안전판정이 뒤집힌 축으로 이뤄진다)
  ⑤ URDF의 선언과 실제로 일치(문서-코드 드리프트 방지)
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from common.calibration import (
    mat_to_quat,
    pose7_to_mat,
    quat_to_mat,
    ur_base_to_base_link_pose7,
    ur_base_to_base_link_wrench,
)

_ROOT = Path(__file__).resolve().parent.parent
_URDF = _ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf"


def _pose7(x, y, z, quat=(1.0, 0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([x, y, z, *quat], dtype=np.float64)


def test_position_x_y_negated_z_kept() -> None:
    """★핵심★ 위치는 x·y만 부호가 뒤집히고 z는 유지된다."""
    out = ur_base_to_base_link_pose7(_pose7(0.30, 0.13, 0.52))
    assert np.allclose(out[:3], [-0.30, -0.13, 0.52], atol=1e-12)


def test_identity_orientation_becomes_z180() -> None:
    """항등 자세는 z축 180° 회전이 된다(quat wxyz ≈ (0,0,0,±1))."""
    out = ur_base_to_base_link_pose7(_pose7(0.0, 0.0, 0.0))
    R = quat_to_mat(out[3:])
    assert np.allclose(R, np.diag([-1.0, -1.0, 1.0]), atol=1e-12)


def test_rotation_stays_valid(  ) -> None:
    """변환 결과가 유효 회전(직교·det=+1) — 거울반사가 섞이면 안 된다."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        out = ur_base_to_base_link_pose7(_pose7(*rng.normal(size=3), quat=tuple(q)))
        R = quat_to_mat(out[3:])
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_involution_roundtrip() -> None:
    """두 번 적용하면 원복 — 관례 변환이 대합이라는 성질(잘못 두 번 적용해도 티가 난다)."""
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        p = _pose7(*rng.normal(size=3), quat=tuple(q))
        back = ur_base_to_base_link_pose7(ur_base_to_base_link_pose7(p))
        assert np.allclose(back[:3], p[:3], atol=1e-9)
        # quat은 ±q가 같은 회전이므로 회전행렬로 비교
        assert np.allclose(quat_to_mat(back[3:]), quat_to_mat(p[3:]), atol=1e-9)


def test_matches_explicit_rz_pi_matrix() -> None:
    """정의대로 T_out = Rz(π)·T_in 인지 행렬 수준에서 확인."""
    rng = np.random.default_rng(2)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    p = _pose7(0.4, -0.2, 0.3, quat=tuple(q))
    rz = np.diag([-1.0, -1.0, 1.0])
    T_in = pose7_to_mat(p)
    expect = np.eye(4)
    expect[:3, :3] = rz @ T_in[:3, :3]
    expect[:3, 3] = rz @ T_in[:3, 3]
    got = pose7_to_mat(ur_base_to_base_link_pose7(p))
    assert np.allclose(got[:3, 3], expect[:3, 3], atol=1e-12)
    assert np.allclose(quat_to_mat(mat_to_quat(got[:3, :3])), expect[:3, :3], atol=1e-9)


def test_wrench_force_and_torque_both_rotated() -> None:
    """★force만 돌리고 torque를 두면 안전판정 축이 어긋난다★ — 둘 다 회전해야 한다."""
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = ur_base_to_base_link_wrench(w)
    assert np.allclose(out, [-1.0, -2.0, 3.0, -4.0, -5.0, 6.0], atol=1e-12)


def test_wrench_norm_preserved() -> None:
    """순수 회전이므로 힘·토크 크기는 보존된다(SafetyGuard의 norm 판정이 불변)."""
    rng = np.random.default_rng(3)
    w = rng.normal(size=6)
    out = ur_base_to_base_link_wrench(w)
    assert np.isclose(np.linalg.norm(w[:3]), np.linalg.norm(out[:3]), atol=1e-12)
    assert np.isclose(np.linalg.norm(w[3:]), np.linalg.norm(out[3:]), atol=1e-12)


@pytest.mark.skipif(not _URDF.is_file(), reason="URDF 없음")
def test_urdf_declares_z180_between_base_link_and_base() -> None:
    """★문서-코드 드리프트 방지★ — 이 변환의 근거가 URDF에 실제로 있는지 확인.

    URDF가 바뀌어 회전이 사라지면(또는 각도가 달라지면) 이 테스트가 먼저 깨진다.
    """
    src = _URDF.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'name="base_link-base_fixed_joint".*?<origin\s+rpy="([^"]+)"',
                  src, re.S)
    assert m, "URDF에서 base_link-base_fixed_joint의 origin rpy를 찾지 못함"
    rpy = [float(v) for v in m.group(1).split()]
    assert np.allclose(rpy[:2], [0.0, 0.0], atol=1e-9), f"roll/pitch가 0이 아님: {rpy}"
    assert np.isclose(abs(rpy[2]), math.pi, atol=1e-6), (
        f"yaw가 ±π가 아님({rpy[2]}) — common.calibration의 Rz(π) 가정이 깨졌다"
    )


def test_config_default_is_ur_base() -> None:
    """기본값이 'ur_base'(=변환 적용)여야 한다 — URDF 근거에 따른 안전한 기본.

    실기 실측으로 '변환 불필요'가 확인되면 그때 base_link로 바꾼다(주석 절차 참조).
    """
    from common.config_io import configs_dir, load_yaml

    conv = load_yaml(configs_dir() / "real.example.yaml")["robot"].get("tcp_frame_convention")
    assert conv in ("ur_base", "base_link"), f"허용되지 않는 값: {conv}"
    assert conv == "ur_base", (
        "기본이 ur_base가 아니다 — 실기 실측 근거 없이 변환을 끄면 거울상 좌표가 된다"
    )
