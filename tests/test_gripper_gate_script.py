"""Canonical deployment gripper verifier logic tests; no robot required.

왜 필요한가: 이 게이트는 ★실기에서 단 한 번★ 돌리는 안전 판정 스크립트다. 판정 로직이
틀려 있으면 (a) 배선 이상을 못 잡고 통과시키거나 (b) 정상인데 중단시킨다 — 둘 다 실기
세션에서야 드러난다. 로봇 없이 mock으로 ★두 방향 모두★ 미리 확인한다.

이 프로젝트는 "게이트를 만들어 두고 실제로 작동하는지 확인하지 않아" 실패한 이력이
여러 건 있다(F60: 검증이 수정 이전 프로세스를 본 오귀속 / 조기 게이트 미실행으로 5시간 소모).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from common.config_io import build_safety_cfg, configs_dir, load_yaml
from common.safety import SafetyGuard
from common.types import RobotState

_ROOT = Path(__file__).resolve().parent.parent
_GATE = _ROOT / "deployment" / "Verify_Gripper2FG7.py"


def test_command_capable_commissioning_tools_have_outer_emergency_stop_guards():
    for relative in (
        "deployment/Verify_Gripper2FG7.py",
        "deployment/Verify_JointParity.py",
        "deployment/Calib_HandEye.py",
        "deployment/Calib_TcpOffset.py",
    ):
        source = (_ROOT / relative).read_text(encoding="utf-8")
        assert "except BaseException:" in source, relative
        assert "robot.emergency_stop()" in source, relative


@pytest.fixture(scope="module")
def gate():
    """게이트 스크립트를 모듈로 로드 (main 미실행 — 함수만 쓴다)."""
    if not _GATE.is_file():
        pytest.skip(f"게이트 스크립트 없음: {_GATE}")
    spec = importlib.util.spec_from_file_location("verify_gripper_2fg7", _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeRobot:
    """set_gripper 호출을 기록하고, 지정한 만큼 관절을 '움직이는' mock.

    joint_drift > 0 이면 DO 펄스가 팔을 움직이는 배선 이상을 재현한다.
    """

    def __init__(self, joint_drift: float = 0.0, tcp_xyz=(-0.45, 0.0, 0.20), force=True):
        self._q = np.zeros(6)
        self._drift = float(joint_drift)
        self._tcp = np.array([*tcp_xyz, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.calls: list[bool] = []
        self.estop_calls = 0
        self._force = force

    def set_gripper(self, closed: bool) -> None:
        self.calls.append(bool(closed))
        self._q = self._q + self._drift

    def get_state(self) -> RobotState:
        return RobotState(
            joint_pos=self._q.copy(),
            joint_vel=np.zeros(6),
            tcp_pose=self._tcp.copy(),
            tcp_force=np.zeros(6) if self._force else None,
            timestamp=0.0,
        )

    def emergency_stop(self) -> None:
        self.estop_calls += 1


def _guard(robot: _FakeRobot) -> SafetyGuard:
    cfg = build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))
    return SafetyGuard(cfg, estop_hook=lambda reason: robot.emergency_stop())


def test_gate_passes_when_joints_stay_still(gate):
    """정상 케이스 — 관절이 안 움직이면 통과하고 기록을 남긴다."""
    robot = _FakeRobot(joint_drift=0.0)
    records: list = []
    gate._one_action(robot, _guard(robot), closed=True, settle_s=0.0,
                     q_tol=1e-3, records=records)
    assert robot.calls == [True]
    assert len(records) == 1
    r = records[0]
    assert r["ok_q"] is True and r["safe"] is True
    assert r["dq"] == pytest.approx(0.0)
    assert "close" in r["action"] and "DO5" in r["action"]   # 닫기=DO5 (실측 매핑)


def test_gate_aborts_when_gripper_action_moves_joints(gate):
    """★핵심 안전 판정★ — DO 펄스가 팔을 움직이면 즉시 중단 + trip + e-stop.

    legacy 실측 노이즈는 5.7e-05 rad(README에 보존)이므로, 그보다 훨씬 큰 변화는
    배선/프로그램 오배치를 뜻한다 — 이 게이트가 그것을 잡지 못하면 존재 이유가 없다.
    """
    robot = _FakeRobot(joint_drift=0.01)      # 10 mrad — 기준(1e-3)의 10배
    guard = _guard(robot)
    records: list = []
    with pytest.raises(SystemExit) as exc:
        gate._one_action(robot, guard, closed=True, settle_s=0.0,
                         q_tol=1e-3, records=records)
    assert "moved arm joints" in str(exc.value)
    assert guard.tripped is True
    assert robot.estop_calls == 1              # latch + e-stop hook 실제 발화
    assert records[-1]["ok_q"] is False


def test_gate_tolerance_accepts_measured_noise_level(gate):
    """실측 노이즈(5.7e-05 rad)는 통과해야 한다 — 기준이 과민하면 정상을 막는다."""
    robot = _FakeRobot(joint_drift=5.7e-05)
    records: list = []
    gate._one_action(robot, _guard(robot), closed=False, settle_s=0.0,
                     q_tol=1e-3, records=records)
    assert records[0]["ok_q"] is True
    assert "open" in records[0]["action"] and "DO6" in records[0]["action"]


def test_gate_aborts_when_safety_verdict_fails_before_action(gate):
    """Excessive force blocks the pulse before command transmission."""
    robot = _FakeRobot()
    robot._force = True
    original = robot.get_state
    def high_force_state():
        state = original()
        state.tcp_force = np.array([1e6, 0, 0, 0, 0, 0], dtype=float)
        return state
    robot.get_state = high_force_state
    guard = _guard(robot)
    records: list = []
    with pytest.raises(SystemExit) as exc:
        gate._one_action(robot, guard, closed=True, settle_s=0.0,
                         q_tol=1e-3, records=records)
    assert "safety check failed" in str(exc.value)
    assert robot.calls == []                          # ★펄스 미전송★
    assert guard.tripped is True


def test_gate_report_renders_with_verdicts(gate, tmp_path):
    """리포트가 판정 결과를 담아 생성된다(실기 세션의 유일한 증적)."""
    real_cfg = load_yaml(configs_dir() / "real.example.yaml")
    records = [
        {"action": "open(DO6)", "call_ms": 0.4, "dq": 3e-05, "ok_q": True, "safe": True, "reason": ""},
        {"action": "close(DO5)", "call_ms": 0.5, "dq": 5e-05, "ok_q": True, "safe": True, "reason": ""},
        {"action": "open(DO6)", "call_ms": 0.4, "dq": 3e-05, "ok_q": True, "safe": True, "reason": ""},
    ]
    gate._write_report(tmp_path, real_cfg, records, cycles=1, q_tol=1e-3, settle_s=1.2)
    text = (tmp_path / "report.txt").read_text(encoding="utf-8")
    report = __import__("json").loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "max joint delta" in text and "max non-blocking call" in text
    assert report["passed"] is True
    assert report["limitations"]
    assert [r["action"] for r in report["records"]] == ["open(DO6)", "close(DO5)", "open(DO6)"]


def test_gate_requires_force_feedback_before_command(gate):
    robot = _FakeRobot(force=False)
    with pytest.raises(RuntimeError, match="force feedback"):
        gate._one_action(robot, _guard(robot), closed=True, settle_s=0.0,
                         q_tol=1e-3, records=[])
    assert robot.calls == []
