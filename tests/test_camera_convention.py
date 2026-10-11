"""손목 카메라 프레임 관례(ros ↔ usd/opengl) 변환 골든 — sim 불필요(README 검증 계약).

배경(2026-07-28): place_workspace_gui.py가 고스트 손목에 '정책과 같은 화각'의 USD 카메라를
      붙여 실시간 미리보기를 제공한다. hand_eye.yaml의 T_flange_cam은 ★ros 관례★
      (+Z 전방, +Y 아래 — TiledCameraCfg(convention="ros")가 소비)인데 USD 카메라는
      ★opengl 관례★(−Z 전방, +Y 위)라 X축 180° 후곱이 필요하다. 이 변환이 틀리면
      미리보기가 정반대(하늘)를 비추고, 사용자가 그걸 보고 홈 포즈를 잘못 고르게 된다.

여기서 고정하는 계약:
  R_usd = R_ros @ diag(1, -1, -1)  ⇒  USD 시선(−Z) == ROS 광축(+Z),  USD 위(+Y) == ROS 위(−Y)
  (isaaclab.utils.math.convert_camera_frame_orientation_convention(origin="ros",
   target="opengl")과 동일 정의. 그 함수는 isaaclab.utils가 pxr을 끌어와 순수 테스트가
   불가하므로 등가 정의를 여기서 골든으로 못박는다.)
"""
from __future__ import annotations

import numpy as np
import pytest

from common.calibration import load_hand_eye, mat_to_quat, quat_to_mat
from common.config_io import configs_dir

# GUI 코드와 동일한 후곱 행렬 (place_workspace_gui.py 카메라 부착부)
ROS_TO_USD = np.diag([1.0, -1.0, -1.0])


def _r_usd(r_ros: np.ndarray) -> np.ndarray:
    return r_ros @ ROS_TO_USD


@pytest.fixture(scope="module")
def r_ros() -> np.ndarray:
    """실제 배포에 쓰이는 hand_eye.yaml의 회전부 — 합성 값이 아니라 진짜 계약으로 검증."""
    return load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")[:3, :3]


def test_ros_to_usd_is_valid_rotation(r_ros: np.ndarray) -> None:
    """변환 결과가 정상 회전행렬(직교·det=+1)이어야 한다 — 아니면 쿼터니언 저작이 깨진다."""
    R = _r_usd(r_ros)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9), "직교성 위반"
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9), "det≠+1 (거울 반사)"


def test_usd_view_direction_matches_ros_optical_axis(r_ros: np.ndarray) -> None:
    """★핵심★ USD 카메라 시선(−Z)이 ROS 광축(+Z)과 같은 방향을 향해야 한다.

    이게 깨지면 미리보기가 정반대를 비춘다(사용자가 '하늘만 보인다'고 느끼는 그 증상).
    """
    fwd_ros = r_ros @ np.array([0.0, 0.0, 1.0])
    fwd_usd = _r_usd(r_ros) @ np.array([0.0, 0.0, -1.0])
    assert np.allclose(fwd_ros, fwd_usd, atol=1e-12), f"시선 불일치 ros={fwd_ros} usd={fwd_usd}"


def test_usd_up_matches_ros_up(r_ros: np.ndarray) -> None:
    """USD의 위(+Y)가 ROS의 위(−Y)와 일치 — 어긋나면 상하 반전 이미지가 된다."""
    up_ros = r_ros @ np.array([0.0, -1.0, 0.0])
    up_usd = _r_usd(r_ros) @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(up_ros, up_usd, atol=1e-12)


def test_right_axis_preserved(r_ros: np.ndarray) -> None:
    """이미지 우측(+X)은 두 관례에서 동일 — 좌우 반전이 없어야 한다."""
    assert np.allclose(r_ros @ np.array([1.0, 0.0, 0.0]),
                       _r_usd(r_ros) @ np.array([1.0, 0.0, 0.0]), atol=1e-12)


def test_involution(r_ros: np.ndarray) -> None:
    """두 번 적용하면 원복 — 관례 변환이 대합(involution)이라는 성질."""
    assert np.allclose(_r_usd(_r_usd(r_ros)), r_ros, atol=1e-12)


def test_quaternion_roundtrip_survives_conversion(r_ros: np.ndarray) -> None:
    """GUI는 변환 결과를 quaternion으로 저작한다 — 왕복에서 회전이 보존돼야 한다."""
    R = _r_usd(r_ros)
    assert np.allclose(quat_to_mat(mat_to_quat(R)), R, atol=1e-9)


def test_optical_axis_points_downward_for_current_calib(r_ros: np.ndarray) -> None:
    """현행 캘리브에서 광축은 아래(−z, base frame)를 향한다 — 손목 카메라가 작업면을 본다는 전제.

    (홈 포즈에 따라 실제 지향은 달라지지만, flange 기준 장착 방향 자체가 뒤집혀 있으면
     어떤 자세로도 작업면을 못 본다 — 그 회귀를 여기서 잡는다.)
    """
    # T_flange_cam의 광축을 flange 프레임에서 본 방향: 공구축(+z_flange)과 이루는 각
    axis = r_ros @ np.array([0.0, 0.0, 1.0])
    cos_tool = float(axis @ np.array([0.0, 0.0, 1.0]))
    assert cos_tool > 0.9, (
        f"카메라 광축이 공구축과 {np.degrees(np.arccos(cos_tool)):.1f}° 벌어짐 — "
        "브라켓 장착 방향(hand_eye.yaml) 확인 필요"
    )
