"""real/real_robot.py 순수 수학(rotvec_to_quat / ur_pose_to_pose7) 단위테스트 — 순수 numpy.

real_robot.py docstring이 명시한 '단위 테스트 계약'을 그대로 검증한다:
  - 영벡터 → [1, 0, 0, 0] (항등 회전)
  - [pi, 0, 0] → [0, 1, 0, 0] (x축 180°)
  - [pi/2, 0, 0] → [cos(pi/4), sin(pi/4), 0, 0]
  - 반환 quat은 단위 노름 & w >= 0 정규화 (common.calibration 관례와 동일)
  - 왕복 일관성: quat_to_mat(rotvec_to_quat(r)) == Rodrigues(r)
  - ur_pose_to_pose7: 위치 통과 + 회전부 rotvec_to_quat 합성

⚠️ real.real_robot의 하드웨어 SDK(ur_rtde)는 __init__에서 지연 import되므로
   이 모듈 import 자체는 SDK 없는 어느 PC에서도 안전하다 (README 코드 경계 계약).

실행: python -m pytest tests/test_real_robot_math.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import ast
import numpy as np
import pytest
from pathlib import Path
import threading
import types
import sys

from common.calibration import quat_to_mat
from real.real_robot import (
    RealRobot,
    apply_task_delta_pose7,
    quat_to_rotvec,
    rotvec_to_quat,
    ur_pose_to_pose7,
)


def test_global_command_lock_contains_no_blocking_sdk_wait_or_join_calls():
    """The state-publication lock must never inherit transport latency."""
    source_path = Path(__file__).parents[1] / "real" / "real_robot.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {
        "setStandardDigitalOut",
        "servoJ",
        "servoL",
        "moveJ",
        "servoStop",
        "speedStop",
        "stopJ",
        "stopL",
        "stopScript",
        "start",
        "wait",
        "join",
    }
    violations: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        owns_command_lock = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "_get_command_lock"
            for item in node.items
        )
        if not owns_command_lock:
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            name = child.func.attr if isinstance(child.func, ast.Attribute) else None
            if name in forbidden:
                violations.append((child.lineno, name))

    assert violations == []


def test_emergency_stop_continues_when_gripper_io_lock_is_wedged():
    """A hung DO worker must never delay the arm stop sequence indefinitely."""
    calls: list[str] = []

    class Control:
        def servoStop(self): calls.append("servoStop")
        def speedStop(self): calls.append("speedStop")
        def stopJ(self, _a): calls.append("stopJ")
        def stopL(self, _a): calls.append("stopL")
        def stopScript(self): calls.append("stopScript")

    robot = object.__new__(RealRobot)
    robot._estopped = False
    robot._grip_stop_event = threading.Event()
    robot._grip_threads = set()
    robot._grip_worker_error = None
    robot._grip_stop_incomplete = False
    robot._grip_io_lock = threading.Lock()
    assert robot._grip_io_lock.acquire(blocking=False)
    robot._rtde_io = object()
    robot._grip_do_open = 6
    robot._grip_do_close = 5
    robot._rtde_c = Control()

    robot.emergency_stop()

    assert calls == ["servoStop", "speedStop", "stopJ", "stopL", "stopScript"]
    assert robot._grip_stop_incomplete is True
    robot._grip_io_lock.release()


def test_blocking_stop_sdk_never_owns_global_command_lock():
    entered_stop = threading.Event()
    release_stop = threading.Event()

    class Control:
        def servoStop(self):
            entered_stop.set()
            assert release_stop.wait(timeout=1.0)
            return True
        def speedStop(self): return True
        def stopJ(self, _a): return True
        def stopL(self, _a): return True
        def stopScript(self): return True

    robot = object.__new__(RealRobot)
    robot._command_lock = threading.RLock()
    robot._estopped = False
    robot._receive_only = False
    robot._grip_stop_event = threading.Event()
    robot._grip_threads = set()
    robot._grip_worker_error = None
    robot._grip_stop_incomplete = False
    robot._grip_io_lock = threading.Lock()
    robot._rtde_io = None
    robot._rtde_c = Control()
    robot._stop_sequence_active = False

    stopper = threading.Thread(target=robot.emergency_stop)
    stopper.start()
    assert entered_stop.wait(timeout=1.0)
    assert robot._estopped is True
    assert robot._get_command_lock().acquire(timeout=0.02)
    robot._get_command_lock().release()
    with pytest.raises(RuntimeError, match="e-stop latch"):
        robot._assert_alive()

    release_stop.set()
    stopper.join(timeout=1.0)
    assert not stopper.is_alive()


def test_close_escalates_when_gripper_worker_cannot_join():
    robot = object.__new__(RealRobot)
    robot._estopped = False
    robot._grip_stop_event = threading.Event()
    robot._grip_stop_incomplete = False
    robot._rtde_c = None
    robot._rtde_r = None
    robot._rtde_io = None
    escalated: list[bool] = []

    class WedgedWorker:
        def join(self, timeout):
            assert timeout == 0.2

        def is_alive(self):
            return True

    robot._grip_threads = {WedgedWorker()}
    robot.emergency_stop = lambda: (
        escalated.append(True),
        setattr(robot, "_estopped", True),
    )
    robot._disconnect_rtde_resources = lambda: None

    robot.close()

    assert robot._grip_stop_event.is_set()
    assert robot._grip_stop_incomplete is True
    assert escalated == [True]


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("send_joint_target", (np.full(6, np.nan), 0.05)),
        ("send_task_delta", (np.full(6, np.nan), np.array([0, 0, 0, 1, 0, 0, 0]), 0.05)),
        ("move_to_joints", (np.full(6, np.nan), 0.1)),
    ],
)
def test_command_adapter_latches_before_sdk_on_nonfinite_input(method, args):
    robot = object.__new__(RealRobot)
    robot._estopped = False
    robot._receive_only = False
    robot._tcp_command_verified = True
    tripped: list[bool] = []
    robot.emergency_stop = lambda: (tripped.append(True), setattr(robot, "_estopped", True))

    with pytest.raises(ValueError, match="non-finite/invalid"):
        kwargs = {"mapped_raw_delta": np.zeros(6)} if method == "send_task_delta" else {}
        getattr(robot, method)(*args, **kwargs)
    assert tripped == [True]


def _command_robot() -> RealRobot:
    robot = object.__new__(RealRobot)
    robot._estopped = False
    robot._receive_only = False
    robot._tcp_command_verified = True
    robot._servoj_lookahead = 0.1
    robot._servoj_gain = 300.0
    robot._T_flange_tcp = np.eye(4)
    robot._needs_base_rot = False
    robot._movej_timeout_s = 0.1
    robot._command_lock = threading.RLock()
    return robot


def test_gripper_fault_after_entry_check_blocks_servoj_sdk_call():
    """A background fault published after the first check wins before SDK I/O."""
    robot = _command_robot()
    sdk_calls: list[str] = []
    first_check_passed = threading.Event()
    fault_published = threading.Event()

    class Control:
        def servoJ(self, *_args):
            sdk_calls.append("servoJ")
            return True

    robot._rtde_c = Control()
    robot.emergency_stop = lambda: setattr(robot, "_estopped", True)
    original_assert_alive = robot._assert_alive
    check_count = 0

    def coordinated_assert_alive():
        nonlocal check_count
        original_assert_alive()
        check_count += 1
        if check_count == 1:
            first_check_passed.set()
            assert fault_published.wait(timeout=1.0)

    robot._assert_alive = coordinated_assert_alive

    def publish_fault():
        assert first_check_passed.wait(timeout=1.0)
        with robot._get_command_lock():
            robot._grip_worker_error = OSError("DO worker failed")
            robot._estopped = True
        fault_published.set()

    worker = threading.Thread(target=publish_fault)
    worker.start()
    with pytest.raises(RuntimeError, match="gripper worker failed"):
        robot.send_joint_target(np.zeros(6), 0.05)
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert sdk_calls == []


def test_get_state_preserves_rtde_source_timestamp_separate_from_host(monkeypatch):
    robot = object.__new__(RealRobot)
    robot._T_flange_tcp = np.eye(4)
    robot._needs_base_rot = False
    robot._rtde_frequency_hz = 500.0

    class Receive:
        def __init__(self): self.timestamp_calls = 0
        def getActualQ(self): return [0.0] * 6
        def getActualQd(self): return [0.0] * 6
        def getActualTCPPose(self): return [0.0] * 6
        def getActualTCPForce(self): return [0.0] * 6
        def getTimestamp(self):
            self.timestamp_calls += 1
            return 123.0 if self.timestamp_calls == 1 else 123.5

    robot._rtde_r = Receive()
    times = iter([456.0, 456.001, 456.002, 456.003, 456.004])
    monkeypatch.setattr("real.real_robot.time.perf_counter", lambda: next(times))
    state = robot.get_state()
    assert state.timestamp == 456.004
    assert state.source_timestamp_s == 123.5
    assert state.source_timestamp_domain == "ur_rtde_controller_clock"
    assert state.source_acquisition_host_lower_s == 456.001
    assert state.source_acquisition_host_upper_s == 456.003


def test_force_read_failure_latches_instead_of_returning_missing_force():
    robot = object.__new__(RealRobot)
    robot._T_flange_tcp = np.eye(4)
    robot._needs_base_rot = False
    robot._rtde_frequency_hz = 500.0
    tripped: list[bool] = []
    robot.emergency_stop = lambda: tripped.append(True)

    class Receive:
        def __init__(self): self.timestamp_calls = 0
        def getTimestamp(self):
            self.timestamp_calls += 1
            return 1.0 if self.timestamp_calls == 1 else 1.002
        def getActualQ(self): return [0.0] * 6
        def getActualQd(self): return [0.0] * 6
        def getActualTCPPose(self): return [0.0] * 6
        def getActualTCPForce(self): raise OSError("force channel lost")

    robot._rtde_r = Receive()
    with pytest.raises(OSError, match="force channel lost"):
        robot.get_state()
    assert tripped == [True]


@pytest.mark.parametrize("failure", [False, OSError("servoJ socket lost")])
def test_servoj_sdk_failure_latches_before_propagation(failure):
    robot = _command_robot()
    tripped: list[bool] = []
    robot.emergency_stop = lambda: (tripped.append(True), setattr(robot, "_estopped", True))

    class Control:
        def servoJ(self, *_args):
            if isinstance(failure, BaseException):
                raise failure
            return failure

    robot._rtde_c = Control()
    with pytest.raises((RuntimeError, OSError)):
        robot.send_joint_target(np.zeros(6), 0.05)
    assert tripped == [True]


@pytest.mark.parametrize("failure", [False, OSError("servoL socket lost")])
def test_servol_sdk_failure_latches_before_propagation(failure):
    robot = _command_robot()
    tripped: list[bool] = []
    robot.emergency_stop = lambda: (tripped.append(True), setattr(robot, "_estopped", True))

    class Control:
        def servoL(self, *_args):
            if isinstance(failure, BaseException):
                raise failure
            return failure

    robot._rtde_c = Control()
    robot._rtde_r = object()
    with pytest.raises((RuntimeError, OSError)):
        robot.send_task_delta(
            np.zeros(6),
            np.array([0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
            0.05,
            mapped_raw_delta=np.zeros(6),
        )
    assert tripped == [True]


def test_servol_success_returns_identity_applied_command_receipt():
    robot = _command_robot()
    calls: list[list[float]] = []
    delta = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06])
    expected_delta = delta.copy()
    mapped_raw = delta * 2.0
    expected_mapped_raw = mapped_raw.copy()

    class Control:
        def servoL(self, pose, *_args):
            calls.append(pose)
            # A hostile/concurrent owner may mutate its input while the SDK
            # call is in flight; the adapter receipt must retain its snapshot.
            delta[:] = 99.0
            mapped_raw[:] = 88.0
            return True

    robot._rtde_c = Control()
    receipt = robot.send_task_delta(
        delta,
        np.array([0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
        0.05,
        mapped_raw_delta=mapped_raw,
    )
    assert receipt.sequence == 1
    np.testing.assert_array_equal(receipt.delayed_mapped_raw, expected_mapped_raw)
    np.testing.assert_array_equal(receipt.post_delay_pre_floor, expected_delta)
    np.testing.assert_array_equal(receipt.applied, expected_delta)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("delta", "current"),
    [
        (np.zeros((2, 3)), np.array([0.4, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])),
        (np.zeros(6), np.zeros((1, 7))),
    ],
)
def test_servol_rejects_wrong_shape_before_sdk(delta, current):
    robot = _command_robot()
    tripped: list[bool] = []
    robot.emergency_stop = lambda: (tripped.append(True), setattr(robot, "_estopped", True))

    class Control:
        def servoL(self, *_args):
            raise AssertionError("SDK must not be called")

    robot._rtde_c = Control()
    with pytest.raises(ValueError, match="non-finite/invalid"):
        robot.send_task_delta(delta, current, 0.05, mapped_raw_delta=np.zeros(6))
    assert tripped == [True]


@pytest.mark.parametrize("failure", [False, OSError("moveJ socket lost")])
def test_movej_initial_sdk_failure_latches_before_propagation(failure):
    robot = _command_robot()
    tripped: list[bool] = []
    robot.emergency_stop = lambda: (tripped.append(True), setattr(robot, "_estopped", True))

    class Control:
        def moveJ(self, *_args):
            if isinstance(failure, BaseException):
                raise failure
            return failure

    robot._rtde_c = Control()
    with pytest.raises((RuntimeError, OSError)):
        robot.move_to_joints(np.zeros(6), 0.1)
    assert tripped == [True]


def test_constructor_disconnects_partial_control_when_receive_connect_fails(monkeypatch):
    disconnected: list[str] = []

    class Control:
        def __init__(self, *_args):
            pass

        def disconnect(self):
            disconnected.append("control")

    class Receive:
        def __init__(self, *_args):
            raise OSError("receive refused")

    monkeypatch.setitem(
        sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=Control)
    )
    monkeypatch.setitem(
        sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=Receive)
    )
    monkeypatch.setattr(
        "deployment.contracts.require_motion_authorization",
        lambda *args, **kwargs: None,   # ★F74★ scope 인자를 받는 새 시그니처
    )
    monkeypatch.setattr("real.real_robot.load_tcp_offset", lambda _path: np.eye(4))
    cfg = {
        "robot": {
            "ip": "192.0.2.10",
            "servoj_lookahead_time": 0.1,
            "servoj_gain": 300,
            "rtde_frequency_hz": 500,
            "movej_timeout_s": 30,
            "tcp_command_verified": False,
            "home_joints_rad": [0.0] * 6,
            "tcp_frame_convention": "ur_base",
        },
        "gripper": {"driver": "do_pulse", "do_open": 6, "do_close": 5, "pulse_s": 0.3},
    }

    with pytest.raises(RuntimeError, match="UR5e 연결 실패"):
        RealRobot(cfg, "unused.yaml", receive_only=False)
    assert disconnected == ["control"]


def test_constructor_disconnects_receive_when_tcp_readback_fails(monkeypatch):
    disconnected: list[str] = []

    class Receive:
        def __init__(self, *_args):
            pass

        def getTCPOffset(self):
            raise OSError("readback lost")

        def disconnect(self):
            disconnected.append("receive")

    monkeypatch.setitem(
        sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=Receive)
    )
    monkeypatch.setattr("real.real_robot.load_tcp_offset", lambda _path: np.eye(4))
    cfg = {
        "robot": {
            "ip": "192.0.2.10",
            "servoj_lookahead_time": 0.1,
            "servoj_gain": 300,
            "rtde_frequency_hz": 500,
            "movej_timeout_s": 30,
            "tcp_command_verified": False,
            "home_joints_rad": [0.0] * 6,
            "tcp_frame_convention": "ur_base",
        },
        "gripper": {"driver": "do_pulse", "do_open": 6, "do_close": 5, "pulse_s": 0.3},
    }

    with pytest.raises(OSError, match="readback lost"):
        RealRobot(cfg, "unused.yaml", receive_only=True)
    assert disconnected == ["receive"]


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    """독립 구현 기준값: rotation-vector → 회전행렬 (Rodrigues 공식, 순수 numpy).

    R = I + sin(θ)·K + (1 − cos(θ))·K²  (K = 단위축 skew 행렬)
    """
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.eye(3)
    kx, ky, kz = r / angle
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _random_rotvecs(n: int, seed: int = 0) -> list[np.ndarray]:
    """다양한 축/각도의 랜덤 rotation-vector (각도는 (0, pi) 범위 — 부호 모호 회피)."""
    rng = np.random.default_rng(seed)
    vecs = []
    for _ in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.05, np.pi - 0.05)
        vecs.append(axis * angle)
    return vecs


RANDOM_ROTVECS = _random_rotvecs(20)


# ------------------------------------------------------- rotvec_to_quat ----


def test_zero_rotvec_is_identity():
    """영벡터 → 항등 quat [1,0,0,0] (docstring 계약 1)."""
    np.testing.assert_allclose(rotvec_to_quat(np.zeros(3)), [1.0, 0.0, 0.0, 0.0], atol=1e-12)


def test_pi_about_x():
    """[pi,0,0] → [0,1,0,0] — x축 180° (docstring 계약 2)."""
    np.testing.assert_allclose(rotvec_to_quat([np.pi, 0.0, 0.0]), [0.0, 1.0, 0.0, 0.0], atol=1e-12)


def test_half_pi_about_x():
    """[pi/2,0,0] → [cos(pi/4), sin(pi/4), 0, 0] (docstring 계약 3)."""
    expected = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    np.testing.assert_allclose(rotvec_to_quat([np.pi / 2, 0.0, 0.0]), expected, atol=1e-12)


@pytest.mark.parametrize("r", RANDOM_ROTVECS)
def test_unit_norm_and_w_nonneg(r):
    """반환 quat은 단위 노름 & w >= 0 (docstring 계약 4 — calibration 관례 동일)."""
    q = rotvec_to_quat(r)
    assert q.shape == (4,)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-12
    assert q[0] >= 0.0


@pytest.mark.parametrize("r", RANDOM_ROTVECS)
def test_roundtrip_vs_rodrigues(r):
    """quat_to_mat(rotvec_to_quat(r)) == Rodrigues(r) — common.calibration과 왕복 일관성."""
    np.testing.assert_allclose(quat_to_mat(rotvec_to_quat(r)), _rodrigues(r), atol=1e-9)


def test_w_nonneg_beyond_pi():
    """각도 > pi(quat w가 음수가 되는 영역)에서도 w >= 0 정규화가 유지된다."""
    r = np.array([0.0, 0.0, 1.0]) * (1.5 * np.pi)  # z축 270°
    q = rotvec_to_quat(r)
    assert q[0] >= 0.0
    # 회전 자체는 동일해야 한다 (q와 -q는 같은 회전)
    np.testing.assert_allclose(quat_to_mat(q), _rodrigues(r), atol=1e-9)


# ------------------------------------------------------- ur_pose_to_pose7 ----


@pytest.mark.parametrize("r", RANDOM_ROTVECS[:5])
def test_ur_pose_to_pose7_composition(r):
    """(6,) UR 포즈 → (7,) pos+quat(wxyz): 위치는 통과, 회전부는 rotvec_to_quat과 일치."""
    pos = np.array([0.45, -0.10, 0.30])
    pose7 = ur_pose_to_pose7(np.concatenate([pos, r]))
    assert pose7.shape == (7,)
    np.testing.assert_allclose(pose7[:3], pos, atol=1e-12)
    np.testing.assert_allclose(pose7[3:], rotvec_to_quat(r), atol=1e-12)


def test_ur_pose_to_pose7_shape_error():
    """길이 6이 아닌 입력은 reshape 단계에서 ValueError."""
    with pytest.raises(ValueError):
        ur_pose_to_pose7(np.zeros(5))


@pytest.mark.parametrize("r", RANDOM_ROTVECS[:5])
def test_quat_to_rotvec_roundtrip(r):
    np.testing.assert_allclose(
        _rodrigues(quat_to_rotvec(rotvec_to_quat(r))), _rodrigues(r), atol=1e-9
    )


def test_task_delta_uses_base_translation_and_left_rotation():
    pose = np.array([0.4, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0])
    delta = np.array([0.01, -0.02, 0.03, 0.0, 0.0, np.pi / 2])
    target = apply_task_delta_pose7(pose, delta)
    np.testing.assert_allclose(target[:3], pose[:3] + delta[:3], atol=1e-12)
    np.testing.assert_allclose(quat_to_mat(target[3:]), _rodrigues(delta[3:]), atol=1e-9)


def test_real_gripper_observation_is_command_state_not_joint_echo():
    source = (Path(__file__).resolve().parents[1] / "real/real_robot.py").read_text(encoding="utf-8")
    assert "_grip_cmd_m" not in source
    assert "advance_gripper_command_state(" in source
    assert "GRIPPER_COMMAND_TRANSITION_S" in source
