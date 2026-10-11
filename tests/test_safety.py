"""요소5(SafetyGuard) 단위테스트 — 시뮬레이터/실기 불필요.

clamp 수식, workspace AABB 판정(경계 안/밖), 속도/힘 한계, trip latch,
e-stop hook 호출(중복 trip 무시)을 계약 그대로 검증한다.

실행: python -m pytest tests/test_safety.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import numpy as np
import pytest

from common.safety import SafetyGuard
from common.types import RobotState, SafetyCfg

# configs/safety.yaml과 같은 수치 (테스트 자체는 파일과 독립)
CFG = SafetyCfg(
    workspace_min=(0.20, -0.45, 0.00),
    workspace_max=(0.70, 0.45, 0.50),
    joint_vel_max_rad_s=1.0,
    tcp_force_max_n=50.0,
)


def _state(
    tcp_pos=(0.45, 0.0, 0.25),
    joint_vel=(0.0,) * 6,
    tcp_force=None,
) -> RobotState:
    """workspace 중앙의 기본 안전 상태 (인자로 위반 상황 주입)."""
    return RobotState(
        joint_pos=np.zeros(6),
        joint_vel=np.asarray(joint_vel, dtype=np.float64),
        tcp_pose=np.array([*tcp_pos, 1.0, 0.0, 0.0, 0.0]),
        tcp_force=None if tcp_force is None else np.asarray(tcp_force, dtype=np.float64),
        timestamp=0.0,
    )


# ---------------------------------------------------------------- clamp ----


def test_clamp_joint_target_formula():
    """스텝당 이동량 |Δq| <= joint_vel_max*dt, 초과분은 잘리고 부호 유지."""
    guard = SafetyGuard(CFG)
    dt = 0.05
    max_step = CFG.joint_vel_max_rad_s * dt  # = 0.05
    q_cur = np.zeros(6)
    q_target = np.array([0.2, -0.2, 0.01, -0.01, 0.05, 0.0])
    out = guard.clamp_joint_target(q_cur, q_target, dt)
    expected = np.array([max_step, -max_step, 0.01, -0.01, 0.05, 0.0])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_clamp_relative_to_qcur():
    """clamp는 q_cur 기준 상대량 — q_cur가 0이 아니어도 수식 유지."""
    guard = SafetyGuard(CFG)
    q_cur = np.array([1.0, -1.0, 0.5, 0.0, 2.0, -2.0])
    q_target = q_cur + 0.3  # 전 관절 +0.3 (max_step=0.02 초과)
    out = guard.clamp_joint_target(q_cur, q_target, dt=0.02)
    np.testing.assert_allclose(out, q_cur + 0.02, atol=1e-12)


# ------------------------------------------------------------ workspace ----


def test_check_ok_inside_and_on_boundary():
    """AABB 내부와 경계 위(min/max 자체)는 ok — strict 비교 계약."""
    guard = SafetyGuard(CFG)
    assert guard.check(_state()).ok
    assert guard.check(_state(tcp_pos=CFG.workspace_min)).ok
    assert guard.check(_state(tcp_pos=CFG.workspace_max)).ok


@pytest.mark.parametrize(
    "tcp_pos",
    [
        (0.19, 0.0, 0.25),   # x < min
        (0.71, 0.0, 0.25),   # x > max
        (0.45, -0.46, 0.25), # y < min
        (0.45, 0.46, 0.25),  # y > max
        (0.45, 0.0, -0.01),  # z < min
        (0.45, 0.0, 0.51),   # z > max
    ],
)
def test_check_workspace_violation(tcp_pos):
    """AABB 6개 면 각각의 이탈 검출."""
    guard = SafetyGuard(CFG)
    verdict = guard.check(_state(tcp_pos=tcp_pos))
    assert not verdict.ok
    assert "workspace" in verdict.reason


# ---------------------------------------------------------- vel / force ----


def test_check_joint_vel_limit():
    """|joint_vel|의 최대값이 한계 초과면 not-ok (경계값은 ok — strict >)."""
    guard = SafetyGuard(CFG)
    assert guard.check(_state(joint_vel=(0, 0, 0, 0, 0, 1.0))).ok  # 정확히 한계
    verdict = guard.check(_state(joint_vel=(0, 0, -1.5, 0, 0, 0)))  # 음수 방향도 검출
    assert not verdict.ok and "속도" in verdict.reason


def test_check_tcp_force_limit_and_none():
    """힘 norm(앞 3성분) 초과 검출 + tcp_force=None이면 검사 생략(ok)."""
    guard = SafetyGuard(CFG)
    assert guard.check(_state(tcp_force=None)).ok  # 미지원 센서 허용
    assert guard.check(_state(tcp_force=(30, 40, 0, 0, 0, 0))).ok  # norm=50, strict >
    verdict = guard.check(_state(tcp_force=(60, 0, 0, 0, 0, 0)))  # norm=60 > 50
    assert not verdict.ok and "힘" in verdict.reason


def test_check_force_uses_first3_only():
    """wrench 뒤 3성분(토크)은 힘 norm에 미포함 — 계약 회귀 가드."""
    guard = SafetyGuard(CFG)
    assert guard.check(_state(tcp_force=(0, 0, 0, 999, 999, 999))).ok


# ----------------------------------------------------------- trip latch ----


def test_trip_latch_blocks_clamp_and_check():
    """trip 후: clamp는 RuntimeError, check는 not-ok(latch) — 재기동 전 명령 불가."""
    guard = SafetyGuard(CFG)
    guard.trip("테스트 수동 trip")
    assert guard.tripped and guard.trip_reason == "테스트 수동 trip"
    with pytest.raises(RuntimeError):
        guard.clamp_joint_target(np.zeros(6), np.zeros(6), dt=0.05)
    verdict = guard.check(_state())  # 완전히 안전한 상태여도
    assert not verdict.ok and "tripped" in verdict.reason


def test_estop_hook_called_with_reason():
    """trip → estop_hook(reason) 정확히 1회 호출 (mock callable로 확인)."""
    calls: list[str] = []
    guard = SafetyGuard(CFG, estop_hook=calls.append)
    guard.trip("workspace 이탈 테스트")
    assert calls == ["workspace 이탈 테스트"]


def test_duplicate_trip_ignored():
    """중복 trip은 무시 — hook 재호출 없음, 최초 reason 유지."""
    calls: list[str] = []
    guard = SafetyGuard(CFG, estop_hook=calls.append)
    guard.trip("첫번째")
    guard.trip("두번째")
    assert calls == ["첫번째"]
    assert guard.trip_reason == "첫번째"


def test_no_hook_trip_does_not_crash():
    """estop_hook 미지정(None)이어도 trip은 latch만 하고 예외 없음."""
    guard = SafetyGuard(CFG, estop_hook=None)
    guard.trip("hook 없음")
    assert guard.tripped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
