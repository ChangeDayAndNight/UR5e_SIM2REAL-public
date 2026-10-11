"""action 매핑(정책 출력 → 관절 목표/그리퍼) 단위테스트 — 시뮬레이터 불필요.

수식 계약:
  ik_rel(현행): task delta = action[:6] * (position/rotation scale)
  joint_absolute(legacy fixture): q_target = home + action[:6] * action_scale_rad
  joint_delta(legacy fixture): q_target = q_cur + clip(action[:6], -1, 1) * action_scale_rad
  gripper_closed       = action[6] < gripper_close_threshold  (strict <)
    ★2026-07-09 부호 정정: Isaac Lab BinaryJointPositionAction 관례(actions<0=close,
    binary_joint_actions.py L124-129)를 전체 계약으로 채택 — sim=real 일치(§4-3).

실행: python -m pytest tests/test_action_mapping.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import numpy as np
import pytest

from common.action_mapping import ActionMapper
from common.config_io import build_action_cfg, configs_dir, load_yaml
from common.types import ActionCfg

# UR5E_HOME_JOINT_POS 순서(shoulder_pan, shoulder_lift, elbow, wrist_1/2/3)
_HOME = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
CFG = ActionCfg(
    mode="joint_absolute",
    num_arm_joints=6,
    action_scale_rad=0.5,
    gripper_close_threshold=0.0,
    home_joint_pos=_HOME,
)


def test_absolute_formula():
    """절대 목표 수식: q_target == home + action[:6] * scale (q_cur 무시)."""
    mapper = ActionMapper(CFG)
    action = np.array([0.5, -0.5, 1.0, -1.0, 0.1, 0.0, -1.0])
    q_cur = np.full(6, 9.0)  # ★무시되어야 함★
    q_target, closed = mapper.map(action, q_cur)
    expected = np.asarray(_HOME) + action[:6] * CFG.action_scale_rad
    np.testing.assert_allclose(q_target, expected, atol=1e-12)
    assert closed is True  # -1.0 < 0.0 → close (upstream 관례)


def test_absolute_no_clip():
    """절대 모드는 clip 없음 — 큰 action도 선형 스케일(sim clip_actions=None과 정합)."""
    mapper = ActionMapper(CFG)
    action = np.array([5.0, -7.0, 1.0, -1.0, 0.0, 100.0, 0.5])
    q_target, closed = mapper.map(action, np.zeros(6))
    expected = np.asarray(_HOME) + action[:6] * CFG.action_scale_rad
    np.testing.assert_allclose(q_target, expected, atol=1e-12)  # 5.0·0.5=2.5 그대로(무클립)
    assert closed is False  # 0.5 < 0.0 아님 → open


def test_absolute_zero_action_gives_home():
    """action=0 → 목표 = home (자세 유지)."""
    mapper = ActionMapper(CFG)
    q_target, _ = mapper.map(np.zeros(7), np.full(6, 3.0))
    np.testing.assert_allclose(q_target, np.asarray(_HOME), atol=1e-12)


def test_qcur_ignored_in_absolute():
    """절대 모드: q_cur이 달라도 같은 목표(절대 목표라 현재와 무관)."""
    mapper = ActionMapper(CFG)
    action = np.array([0.2, 0.3, -0.4, 0.1, 0.0, 0.0, 0.0])
    t1, _ = mapper.map(action, np.zeros(6))
    t2, _ = mapper.map(action, np.full(6, 3.0))
    np.testing.assert_allclose(t1, t2, atol=1e-12)


def test_legacy_delta_formula():
    """legacy joint_delta: q_target = q_cur + clip(action,-1,1)*scale (상한 = scale)."""
    cfg = ActionCfg(mode="joint_delta", num_arm_joints=6, action_scale_rad=0.05,
                    gripper_close_threshold=0.0)
    mapper = ActionMapper(cfg)
    action = np.array([5.0, -7.0, 1.0, -1.0, 0.0, 0.5, 0.0])  # ±1 초과 → clip
    q_cur = np.zeros(6)
    q_target, _ = mapper.map(action, q_cur)
    expected = np.array([1.0, -1.0, 1.0, -1.0, 0.0, 0.5]) * cfg.action_scale_rad
    np.testing.assert_allclose(q_target, expected, atol=1e-12)
    assert np.max(np.abs(q_target - q_cur)) <= cfg.action_scale_rad + 1e-12


@pytest.mark.parametrize(
    "grip_val,expected_closed",
    [
        (0.0, False),    # 경계: threshold와 같으면 strict < 라서 open
        (1e-9, False),   # 경계 바로 위 → open
        (-1e-9, True),   # 경계 바로 아래 → close (upstream: actions<0=close)
        (1.0, False),
        (-1.0, True),
    ],
)
def test_gripper_threshold_boundary(grip_val, expected_closed):
    """그리퍼 임계 경계(threshold=0.0): strict < 판정 — upstream 관례(actions<0=close)."""
    mapper = ActionMapper(CFG)
    action = np.concatenate([np.zeros(6), [grip_val]])
    _, closed = mapper.map(action, np.zeros(6))
    assert closed is expected_closed


def test_gripper_threshold_nonzero():
    """threshold != 0 케이스도 같은 strict < 규칙."""
    cfg = ActionCfg(mode="joint_absolute", num_arm_joints=6, action_scale_rad=0.5,
                    gripper_close_threshold=0.5, home_joint_pos=_HOME)
    mapper = ActionMapper(cfg)
    _, closed_at = mapper.map(np.concatenate([np.zeros(6), [0.5]]), np.zeros(6))
    _, closed_below = mapper.map(np.concatenate([np.zeros(6), [0.4]]), np.zeros(6))
    assert closed_at is False and closed_below is True


@pytest.mark.parametrize("bad_len", [6, 8, 1])
def test_action_dim_mismatch_raises(bad_len):
    """action 길이 != action_dim(7) → ValueError."""
    mapper = ActionMapper(CFG)
    with pytest.raises(ValueError):
        mapper.map(np.zeros(bad_len), np.zeros(6))


@pytest.mark.parametrize("bad_len", [5, 7])
def test_qcur_dim_mismatch_raises(bad_len):
    """q_cur 길이 != num_arm_joints(6) → ValueError (절대 모드도 시그니처 계약상 검증)."""
    mapper = ActionMapper(CFG)
    with pytest.raises(ValueError):
        mapper.map(np.zeros(7), np.zeros(bad_len))


def test_absolute_home_length_mismatch_raises():
    """joint_absolute인데 home 길이 != num_arm_joints → 생성 시 ValueError."""
    with pytest.raises(ValueError):
        ActionMapper(ActionCfg(mode="joint_absolute", num_arm_joints=6,
                               action_scale_rad=0.5, gripper_close_threshold=0.0,
                               home_joint_pos=(0.0, 0.0)))


def test_unsupported_mode_raises():
    """mode not in {absolute, delta}는 확장 지점 — 생성 시 NotImplementedError."""
    with pytest.raises(NotImplementedError):
        ActionMapper(ActionCfg(mode="unknown", num_arm_joints=6,
                               action_scale_rad=0.5, gripper_close_threshold=0.0))


def test_ik_rel_matches_isaac_scale_contract():
    cfg = ActionCfg(
        mode="ik_rel", num_arm_joints=6, action_scale_rad=0.5,
        task_pos_scale_m=0.2, task_rot_scale_rad=0.2,
        gripper_close_threshold=0.0,
    )
    action = np.array([1.0, -0.5, 0.25, -1.0, 0.5, -0.25, -1.0])
    command, closed = ActionMapper(cfg).map(action, np.zeros(6))
    np.testing.assert_allclose(command, action[:6] * 0.2)
    assert closed is True


# ---------- SpeedLimitedMapper (Phase4 저속 램프 — 실행측 스텝당 이동 상한) ----------

def _speed_limited_mapper_cls():
    """Phase4/_deploy_common.py는 순수 모듈(§8-2)이나 패키지가 아니라 경로 주입 필요."""
    import sys
    from pathlib import Path

    phase4 = Path(__file__).resolve().parent.parent / "Phase4"
    if str(phase4) not in sys.path:
        sys.path.insert(0, str(phase4))
    from _deploy_common import SpeedLimitedMapper

    return SpeedLimitedMapper


def test_speed_limited_clamps_step():
    """|q_target − q_cur| > max_step → 방향 유지한 채 max_step으로 잘린다."""
    SpeedLimitedMapper = _speed_limited_mapper_cls()
    mapper = SpeedLimitedMapper(CFG, max_step_rad=0.01)
    action = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    q_cur = np.zeros(6)
    q_full, closed_full = ActionMapper(CFG).map(action, q_cur)
    q_lim, closed_lim = mapper.map(action, q_cur)
    np.testing.assert_allclose(
        q_lim, q_cur + np.clip(q_full - q_cur, -0.01, 0.01), atol=1e-12
    )
    assert np.max(np.abs(q_lim - q_cur)) <= 0.01 + 1e-12
    assert closed_lim is closed_full  # 그리퍼 판정은 불변


def test_speed_limited_identity_when_within_limit():
    """스텝 이동이 상한 이내면 원본 매핑과 항등 (목표 분포 불변 계약)."""
    SpeedLimitedMapper = _speed_limited_mapper_cls()
    mapper = SpeedLimitedMapper(CFG, max_step_rad=10.0)  # 사실상 무제한
    action = np.array([0.2, -0.3, 0.4, -0.1, 0.05, 0.0, 0.7])
    q_cur = np.asarray(_HOME)
    q_full, _ = ActionMapper(CFG).map(action, q_cur)
    q_lim, _ = mapper.map(action, q_cur)
    np.testing.assert_allclose(q_lim, q_full, atol=1e-12)


def test_speed_limited_converges_to_target():
    """같은 목표를 반복 추종하면 여러 스텝에 걸쳐 원 목표에 수렴 (순수 감속 의미론)."""
    SpeedLimitedMapper = _speed_limited_mapper_cls()
    mapper = SpeedLimitedMapper(CFG, max_step_rad=0.1)
    action = np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    q_full, _ = ActionMapper(CFG).map(action, np.zeros(6))
    q = np.asarray(_HOME, dtype=np.float64).copy()
    for _ in range(100):
        q, _ = mapper.map(action, q)
    np.testing.assert_allclose(q, q_full, atol=1e-9)


def test_speed_limited_rejects_nonpositive_limit():
    """max_step_rad <= 0 → 생성 시 ValueError."""
    SpeedLimitedMapper = _speed_limited_mapper_cls()
    with pytest.raises(ValueError):
        SpeedLimitedMapper(CFG, max_step_rad=0.0)


def test_real_config_builds_ik_rel_mapper():
    """★end-to-end 계약★: 실 common.yaml → config_io → mapper.
    mode=ik_rel, action=0 → task-space zero delta."""
    cfg = build_action_cfg(load_yaml(configs_dir() / "common.yaml"))
    assert cfg.mode == "ik_rel"
    assert len(cfg.home_joint_pos) == cfg.num_arm_joints == 6
    mapper = ActionMapper(cfg)
    delta, _ = mapper.map(np.zeros(cfg.action_dim), np.zeros(cfg.num_arm_joints))
    np.testing.assert_allclose(delta, np.zeros(6), atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
