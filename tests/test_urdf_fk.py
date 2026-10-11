"""urdf_fk(배치 GUI FK) 검증 — 실측 골든값 대조 (sim 불필요, README 검증 계약).

★골든★: 홈 포즈 FK로 계산한 TCP(파지중심)가 events_dr._PREGRASP_GRIP와 일치해야 FK가 옳다
— GUI 고스트 관절 저작의 전제. 현행 골든 = (−0.2975, −0.1328, 0.5091)
  (2026-07-30 툴 체인 5단 정정 반영: tool flange 5mm + QC-RS v3 11.25mm 추가).
목적은 '값이 무엇인가'가 아니라 ★robot_spec 홈 ↔ tcp_offset ↔ _PREGRASP_GRIP 3자 정합★이다.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
_URDF = _ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf"

# scripts/tools는 패키지가 아님 — 파일 경로로 로드(dataclass 요구사항: sys.modules 등록 선행)
_spec = importlib.util.spec_from_file_location("urdf_fk", _ROOT / "scripts" / "tools" / "urdf_fk.py")
urdf_fk = importlib.util.module_from_spec(_spec)
sys.modules["urdf_fk"] = urdf_fk
_spec.loader.exec_module(urdf_fk)

from common.robot_spec import UR5E_HOME_JOINT_POS  # noqa: E402 (pure)

# sim 실측 골든(TCW1 체인, 검증홈): events_dr._PREGRASP_GRIP — 해당 모듈은 isaaclab을
# import하므로 여기선 값을 고정 복사(변경 시 이 테스트도 갱신).
# ★2026-07-28 홈 포즈 교체와 함께 갱신★ (구: 0.4919, 0.1328, 0.2617 — 구 홈 -90/+90/-90)
#   이 테스트의 목적은 '값이 무엇인가'가 아니라 ★robot_spec 홈 ↔ events_dr._PREGRASP_GRIP의
#   정합★이다 — 한쪽만 바꾸면 pregrasp 보조가 엉뚱한 곳에 큐브를 놓는다(과거 사고 지점).
# ★2026-07-28(2차) 배치 180°★ — 홈 shoulder_pan 0→π로 (x,y)→(−x,−y). z 불변.
# ★2026-07-30 툴 체인 5단 정정★ tcp_offset 0.2262→0.2425 (tool flange+QC 추가)
# ★2026-08-30 홈 교체(0/-76/-103/-96/+93/+4)에 맞춰 재산출★ — 구값은
#   (-0.2975, -0.1328, 0.5091)로 구 홈(180/-90/+45/-32/-90/0)의 파지중심이었다.
_PREGRASP_GRIP_GOLDEN = np.array([-0.3588, 0.1159, 0.2327])


def _joints():
    return urdf_fk.load_joints(_URDF)


def _tcp_offset() -> np.ndarray:
    rows = yaml.safe_load((_ROOT / "configs" / "calib" / "tcp_offset.yaml").read_text(encoding="utf-8"))[
        "T_flange_tcp"
    ]
    return np.array(rows, dtype=float)


def test_fk_home_matches_pregrasp_grip():
    """★골든★ 홈 포즈 FK ∘ tcp_offset ≈ 실측 파지점 (3mm 이내)."""
    joints = _joints()
    poses = urdf_fk.fk(joints, dict(UR5E_HOME_JOINT_POS))
    t_off = _tcp_offset()
    # tcp_offset은 ee_frame(FrameTransformer, wrist_3_link 기준) 관례 — 후보 프레임 대조
    errs = {}
    for frame in ("wrist_3_link", "tool0", "flange"):
        grip = (poses[frame] @ t_off)[:3, 3]
        errs[frame] = float(np.linalg.norm(grip - _PREGRASP_GRIP_GOLDEN))
    best = min(errs, key=errs.get)
    assert errs[best] < 3e-3, f"홈 FK 파지점 불일치: {errs} (기대 {_PREGRASP_GRIP_GOLDEN.tolist()})"


def test_fk_zero_pose_horizontal_reach():
    """관절 전부 0 = UR 표준 0-포즈(팔이 수평으로 쭉 뻗음) — 수평 도달 ≈0.82m, z 낮음."""
    poses = urdf_fk.fk(_joints(), {})
    p = poses["wrist_3_link"][:3, 3]
    assert np.linalg.norm(p[:2]) > 0.7  # 수평 도달(0.425+0.3922 등)
    assert -0.1 < p[2] < 0.3  # 낮은 높이(수직 스택 아님)


def test_fk_covers_usd_links():
    """USD(평면 계층)에 존재하는 가동 링크가 모두 FK 결과에 있어야 저작 가능."""
    poses = urdf_fk.fk(_joints(), {})
    for name in (
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
        "left_finger_link",
        "right_finger_link",
    ):
        assert name in poses, name


def test_fk_gripper_independent_symmetric_close():
    """★F91★ 두 핑거는 독립 관절이며 같은 목표를 받아 서로를 향해 대칭 접근한다(mimic 제거)."""
    joints = _joints()
    p0 = urdf_fk.fk(joints, {"left_finger_joint": 0.0, "right_finger_joint": 0.0})
    p1 = urdf_fk.fk(joints, {"left_finger_joint": 0.020, "right_finger_joint": 0.020})
    gap0 = np.linalg.norm(p0["left_finger_link"][:3, 3] - p0["right_finger_link"][:3, 3])
    gap1 = np.linalg.norm(p1["left_finger_link"][:3, 3] - p1["right_finger_link"][:3, 3])
    assert gap1 < gap0 - 0.03  # 양쪽 각 20mm 접근 → 40mm 감소
    # 한쪽만 움직이면(편측 파지 = F91 사고) 20mm만 줄어든다 — 그 회귀를 막는다
    p_one = urdf_fk.fk(joints, {"left_finger_joint": 0.020})
    gap_one = np.linalg.norm(p_one["left_finger_link"][:3, 3] - p_one["right_finger_link"][:3, 3])
    assert abs((gap0 - gap_one) - 0.020) < 0.002


def test_fk_shoulder_pan_rotates_wrist():
    """shoulder_pan 90° 회전 → 손목 xy가 z축 기준으로 회전해야 한다(체인 전파)."""
    joints = _joints()
    p0 = urdf_fk.fk(joints, dict(UR5E_HOME_JOINT_POS))
    q = dict(UR5E_HOME_JOINT_POS)
    q["shoulder_pan_joint"] += math.pi / 2
    p1 = urdf_fk.fk(joints, q)
    a, b = p0["wrist_3_link"][:2, 3], p1["wrist_3_link"][:2, 3]
    rot = np.array([[0.0, -1.0], [1.0, 0.0]]) @ a  # z축 +90° 회전 기대값
    assert np.linalg.norm(b - rot) < 1e-6
