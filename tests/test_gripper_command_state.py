"""Pure tests for the Student gripper-command observation contract."""
from __future__ import annotations

import importlib.util
import inspect
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from common.config_io import (
    build_action_cfg,
    build_preprocess_cfg,
    configs_dir,
    load_yaml,
)
from common.gripper_command import (
    GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S,
    GRIPPER_COMMAND_STEP,
    GRIPPER_COMMAND_TRANSITION_S,
    GRIPPER_COMMAND_TRANSITION_STEPS,
    advance_gripper_command_state,
    advance_gripper_command_state_tensor,
    gripper_command_spec,
    validate_gripper_command_config,
)
from common.obs_preprocess import build_policy_obs, preprocess_proprio
from common.policy_io import make_obs_spec, validate_obs_spec
from common.teacher_contract import TEACHER_OBS_DIM, TEACHER_OBS_LAYOUT
from common.types import PreprocessCfg
from real.real_robot import RealRobot

ROOT = Path(__file__).resolve().parents[1]


def test_common_contract_is_exactly_1200ms_at_20hz() -> None:
    common = load_yaml(configs_dir() / "common.yaml")
    pre = build_preprocess_cfg(common)
    assert float(common["control"]["control_hz"]) == 20.0
    assert pre.gripper_command_transition_s == GRIPPER_COMMAND_TRANSITION_S == 1.20
    assert pre.gripper_command_transition_steps == GRIPPER_COMMAND_TRANSITION_STEPS == 24
    assert validate_gripper_command_config(20.0, 1.20) == 24
    assert GRIPPER_COMMAND_STEP == float(np.float32(1.0 / 24.0))
    assert GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S == 0.040
    descriptor = gripper_command_spec(transition_s=1.20, transition_steps=24)
    assert descriptor["update_point"] == "after_delayed_or_successfully_dispatched_command"
    assert descriptor["visible_at"] == "next_observation"
    assert descriptor["sim_reset"] == "open"
    assert descriptor["real_startup"] == "unknown_until_explicit_open_settled"
    assert descriptor["real_dispatch_gate"] == "target_pin_high_sdk_success_bounded_ack"
    assert descriptor["real_sdk_success"] == "return_is_exactly_true"
    assert descriptor["real_dispatch_ack_timeout_s"] == 0.040
    assert descriptor["real_startup_settle_origin"] == "successful_pulse_completion"
    assert descriptor["rapid_reversal_while_pulse_active"] == "invalid_fail_closed"
    assert descriptor["late_high_after_sdk_entry"] == (
        "possible_safe_low_attempt_only_motion_p0"
    )
    assert descriptor["fault_or_estop"] == "invalid"
    assert descriptor["physical_width_feedback"] is False


def test_sim_and_real_command_traces_match_within_1e7() -> None:
    scalar = 0.0
    tensor = torch.zeros(1, dtype=torch.float32)
    scalar_trace: list[float] = []
    tensor_trace: list[float] = []
    for _ in range(24):
        scalar = advance_gripper_command_state(scalar, True)
        tensor = advance_gripper_command_state_tensor(tensor, torch.ones(1, dtype=torch.bool))
        scalar_trace.append(scalar)
        tensor_trace.append(float(tensor.item()))
    np.testing.assert_allclose(scalar_trace, tensor_trace, rtol=0.0, atol=1.0e-7)
    assert scalar == 1.0 and float(tensor.item()) == 1.0

    for _ in range(24):
        scalar = advance_gripper_command_state(scalar, False)
        tensor = advance_gripper_command_state_tensor(tensor, torch.zeros(1, dtype=torch.bool))
    assert scalar == 0.0 and float(tensor.item()) == 0.0


def test_command_state_reversal_is_bounded_and_moves_one_step_only() -> None:
    state = torch.zeros(3, dtype=torch.float32)
    for _ in range(5):
        state = advance_gripper_command_state_tensor(
            state, torch.tensor([True, True, False])
        )
    np.testing.assert_allclose(
        state.numpy(), [5.0 / 24.0, 5.0 / 24.0, 0.0], atol=1.0e-7
    )
    for _ in range(3):
        state = advance_gripper_command_state_tensor(
            state, torch.tensor([False, True, True])
        )
    np.testing.assert_allclose(
        state.numpy(), [2.0 / 24.0, 8.0 / 24.0, 3.0 / 24.0], atol=1.0e-7
    )
    assert bool(((state >= 0.0) & (state <= 1.0)).all())


@pytest.mark.parametrize("value", [None, np.nan, -0.01, 1.01])
def test_unknown_or_invalid_command_state_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="gripper command state|gripper_command_state"):
        advance_gripper_command_state(value, True)


def test_preprocess_consumes_normalized_command_state_without_width_scaling() -> None:
    cfg = PreprocessCfg(0.28, 1.2, 2, 2, np.pi, np.pi, 1.0, 7)
    q = np.zeros(6)
    tcp = np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    out = preprocess_proprio(
        q, q, tcp, np.zeros(3), np.zeros(7), cfg,
        gripper_command_state=0.5,
    )
    assert float(out[0, 12]) == 0.5
    for bad in (None, np.nan, -0.01, 1.01):
        with pytest.raises(ValueError, match="gripper_command_state"):
            preprocess_proprio(
                q, q, tcp, np.zeros(3), np.zeros(7), cfg,
                gripper_command_state=bad,
            )


def test_student_api_cannot_accept_physical_finger_q_and_teacher_is_34d() -> None:
    assert "gripper_pos" not in inspect.signature(build_policy_obs).parameters
    assert "gripper_command_state" in inspect.signature(build_policy_obs).parameters
    # ★Teacher v2★ Teacher도 실제 finger q 대신 같은 명령 상태를 관측한다(배포 가능 계약).
    assert TEACHER_OBS_DIM == 34
    assert ("gripper_command_state", 1) in TEACHER_OBS_LAYOUT
    assert all(name != "gripper_pos" for name, _w in TEACHER_OBS_LAYOUT)

    cfg = PreprocessCfg(0.28, 1.2, 2, 2, np.pi, np.pi, 1.0, 7)
    with pytest.raises(TypeError, match="gripper_pos"):
        build_policy_obs(
            np.ones((2, 2)), np.zeros(6), np.zeros(6),
            np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
            np.zeros(3), np.zeros(7), cfg,
            gripper_command_state=0.0,
            gripper_pos=0.019,  # type: ignore[call-arg]
        )


def test_old_physical_gripper_obs_spec_is_rejected() -> None:
    common = load_yaml(configs_dir() / "common.yaml")
    pre = build_preprocess_cfg(common)
    action = build_action_cfg(common)
    legacy = make_obs_spec(pre, action, action_scheme=action.mode)
    legacy.pop("gripper_command_state")
    for item in legacy["layout"]:
        if item["name"] == "gripper_command_state":
            item["name"] = "gripper_pos"
    with pytest.raises(ValueError, match="gripper_command_state"):
        validate_obs_spec(legacy, pre, action)


class _RecordingIo:
    def __init__(self) -> None:
        self.events: list[tuple[int, bool]] = []
        self.lock = threading.Lock()

    def setStandardDigitalOut(self, pin: int, level: bool) -> bool:
        with self.lock:
            self.events.append((int(pin), bool(level)))
        return True


def _threaded_real_adapter(
    io: object | None = None,
    *,
    pulse_s: float = 0.0,
    ack_timeout_s: float = 0.2,
) -> RealRobot:
    robot = RealRobot.__new__(RealRobot)
    robot._command_lock = threading.RLock()
    robot._estopped = False
    robot._receive_only = False
    robot._grip_worker_error = None
    robot._grip_stop_incomplete = False
    robot._grip_stop_event = threading.Event()
    robot._grip_io_lock = threading.Lock()
    robot._grip_threads = set()
    robot._gripper_driver = "do_pulse"
    robot._grip_do_open = 6
    robot._grip_do_close = 5
    robot._grip_pulse_s = float(pulse_s)
    robot._grip_dispatch_ack_timeout_s = float(ack_timeout_s)
    robot._rtde_io = io if io is not None else _RecordingIo()
    robot._rtde_c = None
    robot._gripper_closed_state = None
    robot._gripper_command_state = None
    robot._gripper_open_initialization_deadline_s = None
    robot._gripper_open_initialization_ticket = None
    robot._gripper_pending_dispatch = None
    robot._gripper_dispatch_generation = 0
    robot._arm_command_generation = 0
    robot._arm_command_inflight = None
    robot._stop_sequence_active = False
    robot._start_gripper_dispatch_watchdog()
    return robot


def test_real_startup_deadline_begins_after_late_pulse_completion(monkeypatch) -> None:
    entered_final_low = threading.Event()
    release_final_low = threading.Event()

    class LateIo(_RecordingIo):
        high_returned = False

        def setStandardDigitalOut(self, pin: int, level: bool) -> bool:
            if level:
                self.high_returned = True
            elif self.high_returned:
                entered_final_low.set()
                assert release_final_low.wait(timeout=1.0)
            return super().setStandardDigitalOut(pin, level)

    io = LateIo()
    robot = _threaded_real_adapter(io, ack_timeout_s=0.5)
    now = [100.0]
    monkeypatch.setattr("real.real_robot.time.perf_counter", lambda: now[0])

    errors: list[BaseException] = []
    caller = threading.Thread(
        target=lambda: _capture_thread_error(errors, robot.set_gripper, False)
    )
    caller.start()
    caller.join(timeout=1.0)
    assert not caller.is_alive() and errors == []
    assert entered_final_low.wait(timeout=1.0)
    # Advance beyond the old enqueue/HIGH-based deadline while pulse completion
    # is blocked.  The startup observation must remain invalid.
    now[0] += 5.0
    assert robot._gripper_command_state is None
    release_final_low.set()
    for worker in tuple(robot._grip_threads):
        worker.join(timeout=1.0)

    robot._refresh_gripper_command_state()
    assert robot._gripper_command_state is None
    now[0] += 1.20 - 1.0e-6
    robot._refresh_gripper_command_state()
    assert robot._gripper_command_state is None
    now[0] += 2.0e-6
    robot._refresh_gripper_command_state()
    assert robot._gripper_command_state == 0.0
    assert any(level for _, level in io.events)


def _capture_thread_error(errors: list[BaseException], func, *args) -> None:
    try:
        func(*args)
    except BaseException as exc:  # pragma: no cover - asserted by caller
        errors.append(exc)


def test_real_edge_state_waits_for_target_high_sdk_success() -> None:
    entered_high = threading.Event()
    release_high = threading.Event()

    class BlockingHighIo(_RecordingIo):
        def setStandardDigitalOut(self, pin: int, level: bool) -> bool:
            if level:
                entered_high.set()
                assert release_high.wait(timeout=1.0)
            return super().setStandardDigitalOut(pin, level)

    robot = _threaded_real_adapter(BlockingHighIo(), ack_timeout_s=0.5)
    robot._gripper_closed_state = False
    robot._gripper_command_state = 0.0
    errors: list[BaseException] = []
    caller = threading.Thread(
        target=lambda: _capture_thread_error(errors, robot.set_gripper, True)
    )
    caller.start()
    assert entered_high.wait(timeout=1.0)
    assert robot._gripper_command_state == 0.0
    assert robot._gripper_closed_state is False
    release_high.set()
    caller.join(timeout=1.0)
    assert not caller.is_alive() and errors == []
    assert robot._gripper_closed_state is True
    assert robot._gripper_command_state == pytest.approx(GRIPPER_COMMAND_STEP)


def test_real_normal_dispatched_trace_matches_sim_within_1e7() -> None:
    robot = _threaded_real_adapter()
    robot._gripper_closed_state = False
    robot._gripper_command_state = 0.0
    scalar_trace: list[float] = []
    tensor_trace: list[float] = []
    tensor = torch.zeros(1, dtype=torch.float32)

    for _ in range(24):
        robot.set_gripper(True)
        tensor = advance_gripper_command_state_tensor(
            tensor, torch.ones(1, dtype=torch.bool)
        )
        scalar_trace.append(float(robot._gripper_command_state))
        tensor_trace.append(float(tensor.item()))
    for worker in tuple(robot._grip_threads):
        worker.join(timeout=1.0)
    np.testing.assert_allclose(
        scalar_trace,
        tensor_trace,
        rtol=0.0,
        atol=1.0e-7,
    )
    assert abs(float(robot._gripper_command_state) - float(tensor.item())) <= 1.0e-7

    for _ in range(24):
        robot.set_gripper(False)
        tensor = advance_gripper_command_state_tensor(
            tensor, torch.zeros(1, dtype=torch.bool)
        )
    for worker in tuple(robot._grip_threads):
        worker.join(timeout=1.0)
    assert abs(float(robot._gripper_command_state) - float(tensor.item())) <= 1.0e-7


def test_real_rapid_reversal_while_previous_pulse_active_fails_closed() -> None:
    io = _RecordingIo()
    robot = _threaded_real_adapter(io, pulse_s=1.0)
    robot._gripper_closed_state = False
    robot._gripper_command_state = 0.0

    robot.set_gripper(True)
    assert robot._gripper_command_state == pytest.approx(GRIPPER_COMMAND_STEP)
    with pytest.raises(RuntimeError, match="previous pulse is active"):
        robot.set_gripper(False)
    assert robot._gripper_command_state is None
    assert robot._estopped is True
    # DO5 close HIGH was dispatched; the queued/reversed DO6 HIGH never was.
    assert (5, True) in io.events
    assert (6, True) not in io.events


def test_real_pre_sdk_io_lock_timeout_never_enters_high() -> None:
    io = _RecordingIo()
    robot = _threaded_real_adapter(io, ack_timeout_s=0.02)
    assert robot._grip_io_lock.acquire(timeout=0.1)
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="dispatch deadline|acknowledgement exceeded"):
        robot.set_gripper(False)
    elapsed = time.perf_counter() - started
    # This case is blocked before any SDK entry.  It does not make a claim
    # about HIGH calls that were already inside the external SDK at timeout.
    assert elapsed < 0.2
    assert robot._gripper_command_state is None
    assert robot._gripper_closed_state is None
    assert robot._estopped is True
    robot._grip_io_lock.release()
    for worker in tuple(robot._grip_threads):
        worker.join(timeout=1.0)
    assert not any(level for _, level in io.events)


def test_real_inflight_high_keeps_command_lock_free_and_blocks_arm_sdk() -> None:
    entered_high = threading.Event()
    release_high = threading.Event()

    class WedgedHighIo(_RecordingIo):
        def setStandardDigitalOut(self, pin: int, level: bool) -> bool:
            if level:
                entered_high.set()
                assert release_high.wait(timeout=1.0)
            return super().setStandardDigitalOut(pin, level)

    arm_sdk_calls: list[str] = []

    class Control:
        def servoJ(self, *_args):
            arm_sdk_calls.append("servoJ")
            return True
        def servoStop(self): return True
        def speedStop(self): return True
        def stopJ(self, _a): return True
        def stopL(self, _a): return True
        def stopScript(self): return True

    robot = _threaded_real_adapter(WedgedHighIo(), ack_timeout_s=0.5)
    robot._rtde_c = Control()
    robot._gripper_closed_state = False
    robot._gripper_command_state = 0.0
    errors: list[BaseException] = []
    caller = threading.Thread(
        target=lambda: _capture_thread_error(errors, robot.set_gripper, True)
    )
    caller.start()
    assert entered_high.wait(timeout=1.0)

    # A wedged RTDE-IO call owns only the IO lock.  The global state lock stays
    # available, while every arm entry sees the published dispatch generation.
    assert robot._get_command_lock().acquire(timeout=0.02)
    robot._get_command_lock().release()
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="dispatch acknowledgement is pending"):
        robot.send_joint_target(np.zeros(6), 0.05)
    assert time.perf_counter() - started < 0.1
    assert arm_sdk_calls == []
    assert robot._gripper_command_state == 0.0

    release_high.set()
    caller.join(timeout=1.0)
    assert not caller.is_alive() and errors == []
    assert robot._gripper_command_state == pytest.approx(GRIPPER_COMMAND_STEP)


def test_prestarted_watchdog_latches_during_blocked_worker_bootstrap(monkeypatch) -> None:
    """The 40 ms deadline is enforced even while per-edge Thread.start stalls."""
    entered_start = threading.Event()
    release_start = threading.Event()
    original_start = threading.Thread.start

    def blocking_grip_start(worker, *args, **kwargs):
        if worker.name == "grip-pulse":
            entered_start.set()
            assert release_start.wait(timeout=1.0)
        return original_start(worker, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", blocking_grip_start)
    arm_sdk_calls: list[str] = []

    class Control:
        def servoJ(self, *_args):
            arm_sdk_calls.append("servoJ")
            return True
        def servoStop(self): return True
        def speedStop(self): return True
        def stopJ(self, _a): return True
        def stopL(self, _a): return True
        def stopScript(self): return True

    # The watchdog is already ready before Thread.start is patched. Commands
    # may create a pulse worker, but they do not create their deadline monitor.
    robot = _threaded_real_adapter(ack_timeout_s=0.04)
    robot._rtde_c = Control()
    robot._servoj_lookahead = 0.1
    robot._servoj_gain = 300.0
    robot._gripper_closed_state = False
    robot._gripper_command_state = 0.0
    errors: list[BaseException] = []
    caller = threading.Thread(
        target=lambda: _capture_thread_error(errors, robot.set_gripper, True),
        name="gripper-api-caller",
    )
    caller.start()
    assert entered_start.wait(timeout=1.0)
    ticket = robot._gripper_pending_dispatch
    assert ticket is not None and ticket.status == "pending"

    assert robot._get_command_lock().acquire(timeout=0.02)
    robot._get_command_lock().release()
    # Keep Thread.start blocked beyond the absolute reservation deadline. The
    # caller is still inside an external runtime primitive, but logical stop
    # publication cannot be delayed by it.
    assert not release_start.wait(timeout=0.08)
    assert robot._estopped is True
    assert robot._gripper_command_state is None
    assert robot._gripper_closed_state is None
    assert ticket.status == "timeout"
    assert isinstance(ticket.error, TimeoutError)
    assert not any(level for _, level in robot._rtde_io.events)
    with pytest.raises(RuntimeError, match="worker failed|e-stop latch"):
        robot.send_joint_target(np.zeros(6), 0.05)
    assert arm_sdk_calls == []

    release_start.set()
    caller.join(timeout=1.0)
    assert not caller.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    assert not any(level for _, level in robot._rtde_io.events)
    assert ticket.status == "timeout"


def test_fault_stop_thread_bootstrap_never_owns_command_lock(monkeypatch) -> None:
    """Lock-held validation publishes only; stop-thread start happens unlocked."""
    entered_start = threading.Event()
    release_start = threading.Event()
    original_start = threading.Thread.start

    def blocking_stop_start(worker, *args, **kwargs):
        if worker.name == "gripper-dispatch-timeout-stop":
            entered_start.set()
            assert release_start.wait(timeout=1.0)
        return original_start(worker, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", blocking_stop_start)
    robot = _threaded_real_adapter()
    errors: list[BaseException] = []
    caller = threading.Thread(
        target=lambda: _capture_thread_error(errors, robot.set_gripper, True),
        name="invalid-gripper-api-caller",
    )
    caller.start()
    assert entered_start.wait(timeout=1.0)

    assert robot._get_command_lock().acquire(timeout=0.02)
    robot._get_command_lock().release()
    assert robot._estopped is True
    assert robot._gripper_command_state is None
    assert robot._gripper_closed_state is None

    release_start.set()
    caller.join(timeout=1.0)
    assert not caller.is_alive()
    assert len(errors) == 1
    assert "explicit open" in str(errors[0])


def test_real_inflight_arm_keeps_command_lock_free_and_blocks_gripper_sdk() -> None:
    entered_arm = threading.Event()
    release_arm = threading.Event()

    class BlockingControl:
        def servoJ(self, *_args):
            entered_arm.set()
            assert release_arm.wait(timeout=1.0)
            return True

    robot = _threaded_real_adapter()
    robot._rtde_c = BlockingControl()
    robot._servoj_lookahead = 0.1
    robot._servoj_gain = 300.0
    robot._gripper_closed_state = False
    robot._gripper_command_state = 0.0
    errors: list[BaseException] = []
    caller = threading.Thread(
        target=lambda: _capture_thread_error(
            errors, robot.send_joint_target, np.zeros(6), 0.05
        )
    )
    caller.start()
    assert entered_arm.wait(timeout=1.0)

    assert robot._get_command_lock().acquire(timeout=0.02)
    robot._get_command_lock().release()
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="arm command SDK call is in flight"):
        robot.set_gripper(True)
    assert time.perf_counter() - started < 0.1
    assert robot._rtde_io.events == []

    release_arm.set()
    caller.join(timeout=1.0)
    assert not caller.is_alive() and errors == []
    assert robot._arm_command_inflight is None


def test_real_late_high_after_timeout_is_bounded_safe_low_only_and_never_commits() -> None:
    entered_high = threading.Event()
    release_high = threading.Event()
    late_low_attempted = threading.Event()

    class LateReturningHighIo(_RecordingIo):
        high_returned = False

        def setStandardDigitalOut(self, pin: int, level: bool) -> bool:
            if level:
                entered_high.set()
                assert release_high.wait(timeout=1.0)
                result = super().setStandardDigitalOut(pin, level)
                self.high_returned = True
                return result
            result = super().setStandardDigitalOut(pin, level)
            if self.high_returned and pin == 5:
                late_low_attempted.set()
            return result

    arm_sdk_calls: list[str] = []

    class Control:
        def servoJ(self, *_args):
            arm_sdk_calls.append("servoJ")
            return True
        def servoStop(self): return True
        def speedStop(self): return True
        def stopJ(self, _a): return True
        def stopL(self, _a): return True
        def stopScript(self): return True

    io = LateReturningHighIo()
    robot = _threaded_real_adapter(io, ack_timeout_s=0.04)
    robot._rtde_c = Control()
    robot._gripper_closed_state = False
    robot._gripper_command_state = 0.0

    tickets = []
    original_start = robot._set_gripper_do_pulse

    def capture_ticket(closed):
        ticket = original_start(closed)
        tickets.append(ticket)
        return ticket

    robot._set_gripper_do_pulse = capture_ticket
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="acknowledgement exceeded"):
        robot.set_gripper(True)
    assert time.perf_counter() - started < 0.2
    assert entered_high.is_set()
    assert robot._estopped is True
    assert robot._gripper_command_state is None
    assert robot._gripper_closed_state is None
    assert len(tickets) == 1
    timeout_error = tickets[0].error
    assert tickets[0].status == "timeout"
    assert isinstance(timeout_error, TimeoutError)

    arm_started = time.perf_counter()
    with pytest.raises(RuntimeError, match="worker failed|e-stop latch"):
        robot.send_joint_target(np.zeros(6), 0.05)
    assert time.perf_counter() - arm_started < 0.1
    assert arm_sdk_calls == []

    # The timeout can be published by both the watchdog and the caller, so up to
    # two stop threads run emergency_stop(), and each one latches (clearing the
    # pending slot) first.  Let every stop thread finish before modelling the
    # later reservation; otherwise a late latch erases it and the checks below
    # race the scheduler (seen on Linux CI).
    stop_deadline = time.perf_counter() + 0.6  # 작업자의 release_high.wait(1.0) 안에 끝나야 한다
    stop_threads_seen = 0
    while time.perf_counter() < stop_deadline:
        stop_threads = [
            t for t in threading.enumerate() if t.name == "gripper-dispatch-timeout-stop"
        ]
        if not stop_threads:
            break
        stop_threads_seen += len(stop_threads)
        for thread in stop_threads:
            thread.join(timeout=max(0.0, stop_deadline - time.perf_counter()))
    assert stop_threads_seen >= 1
    assert robot._stop_sequence_active is False

    # Model a later reservation marker before the stale SDK returns.  The old
    # generation may update only its own ticket, never the new pending slot.
    later_generation = tickets[0].generation + 1
    later_pending = types.SimpleNamespace(
        status="acked",
        error=None,
        cancelled=threading.Event(),
    )
    with robot._get_command_lock():
        robot._gripper_dispatch_generation = later_generation
        robot._gripper_pending_dispatch = later_pending
        worker_error_before_late_return = robot._grip_worker_error

    # Once an external SDK call has already been entered, software cannot stop
    # it from physically applying HIGH late.  The adapter attempts LOW as soon
    # as the call returns but never turns that return into a logical observation.
    release_high.set()
    assert late_low_attempted.wait(timeout=1.0)
    assert (5, True) in io.events
    high_index = io.events.index((5, True))
    assert (5, False) in io.events[high_index + 1:]
    assert robot._gripper_command_state is None
    assert robot._gripper_closed_state is None
    assert tickets[0].status == "timeout"
    assert tickets[0].error is timeout_error
    assert robot._gripper_dispatch_generation == later_generation
    assert robot._gripper_pending_dispatch is later_pending
    assert robot._grip_worker_error is worker_error_before_late_return


@pytest.mark.parametrize("high_result", [False, None])
def test_real_non_true_high_sdk_result_fails_closed(high_result) -> None:
    class NonAcknowledgingIo(_RecordingIo):
        def setStandardDigitalOut(self, pin: int, level: bool):
            if level:
                return high_result
            return super().setStandardDigitalOut(pin, level)

    robot = _threaded_real_adapter(NonAcknowledgingIo())
    with pytest.raises(RuntimeError, match="HIGH.*not acknowledged|invalidated"):
        robot.set_gripper(False)
    assert robot._estopped is True
    assert robot._gripper_command_state is None
    assert robot._gripper_closed_state is None


def test_real_high_sdk_failure_never_commits_command_state() -> None:
    class FailingIo(_RecordingIo):
        def setStandardDigitalOut(self, pin: int, level: bool) -> bool:
            if level:
                raise OSError("HIGH dispatch failed")
            return super().setStandardDigitalOut(pin, level)

    robot = _threaded_real_adapter(FailingIo())
    with pytest.raises(RuntimeError, match="HIGH dispatch failed|invalidated"):
        robot.set_gripper(False)
    assert robot._gripper_command_state is None
    assert robot._gripper_closed_state is None
    assert robot._estopped is True

def test_real_close_first_startup_is_refused_and_fault_invalidates() -> None:
    robot = _threaded_real_adapter()
    robot.emergency_stop = lambda: (
        setattr(robot, "_estopped", True),
        robot._invalidate_gripper_command_state(),
    )
    with pytest.raises(RuntimeError, match="explicit open"):
        robot.set_gripper(True)
    assert not any(level for _, level in robot._rtde_io.events)
    assert robot._estopped is True
    assert robot._gripper_command_state is None

    robot._estopped = False
    robot._gripper_command_state = 0.5
    robot._grip_worker_error = OSError("DO fault")
    assert robot.is_ok() is False
    assert robot._gripper_command_state is None


def test_real_worker_fault_blocks_followup_without_state_update() -> None:
    robot = _threaded_real_adapter()
    robot._gripper_closed_state = False
    robot._gripper_command_state = 0.5
    robot._grip_worker_error = OSError("DO dispatch failed")
    robot.emergency_stop = lambda: (
        setattr(robot, "_estopped", True),
        robot._invalidate_gripper_command_state(),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        robot.set_gripper(False)
    assert not any(level for _, level in robot._rtde_io.events)
    assert robot._gripper_command_state is None
    assert robot._estopped is True


def _load_delayed_actions_with_test_doubles(monkeypatch):
    """Load the real action class with tiny Isaac Lab dependency doubles."""

    class BinaryJointPositionAction:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.num_envs = env.num_envs
            self.device = env.device
            self.last_processed = None

        def process_actions(self, actions):
            self.last_processed = actions.clone()

        def reset(self, env_ids=None):
            return None

    class DifferentialInverseKinematicsAction(BinaryJointPositionAction):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.action_dim = 6
            scale = torch.as_tensor(cfg.scale, dtype=torch.float32, device=env.device)
            self._scale = scale.reshape(1, 6).repeat(env.num_envs, 1)
            self._processed_actions = torch.zeros_like(self._scale)
            self._test_env = env

        @property
        def processed_actions(self):
            return self._processed_actions

        def _compute_frame_pose(self):
            return self._test_env.ee_pos, self._test_env.ee_quat

        def process_actions(self, actions):
            self.last_processed = actions.clone()
            self._processed_actions = actions * self._scale

    class DelayBuffer:
        def __init__(self, max_lag, num_envs, device):
            self.output = None
            self.last_input = None
            self.compute_calls = 0

        def compute(self, actions):
            self.last_input = actions.clone()
            self.compute_calls += 1
            return actions if self.output is None else self.output.clone()

        def set_time_lag(self, lag, batch_ids=None):
            return None

        def reset(self, batch_ids=None):
            return None

    class DummyCfg:
        pass

    modules = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": types.ModuleType("isaaclab.envs"),
        "isaaclab.envs.mdp": types.ModuleType("isaaclab.envs.mdp"),
        "isaaclab.envs.mdp.actions": types.ModuleType("isaaclab.envs.mdp.actions"),
        "isaaclab.envs.mdp.actions.binary_joint_actions": types.ModuleType(
            "isaaclab.envs.mdp.actions.binary_joint_actions"
        ),
        "isaaclab.envs.mdp.actions.task_space_actions": types.ModuleType(
            "isaaclab.envs.mdp.actions.task_space_actions"
        ),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.managers.action_manager": types.ModuleType(
            "isaaclab.managers.action_manager"
        ),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.buffers": types.ModuleType("isaaclab.utils.buffers"),
    }
    modules["isaaclab.envs.mdp"].DifferentialInverseKinematicsActionCfg = DummyCfg
    modules["isaaclab.envs.mdp"].BinaryJointPositionActionCfg = DummyCfg
    modules["isaaclab.envs.mdp.actions.binary_joint_actions"].BinaryJointPositionAction = (
        BinaryJointPositionAction
    )
    modules["isaaclab.envs.mdp.actions.task_space_actions"].DifferentialInverseKinematicsAction = (
        DifferentialInverseKinematicsAction
    )
    modules["isaaclab.managers.action_manager"].ActionTerm = object
    modules["isaaclab.utils"].configclass = lambda cls: cls
    modules["isaaclab.utils.buffers"].DelayBuffer = DelayBuffer
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/mdp/delayed_actions.py"
    spec = importlib.util.spec_from_file_location("_test_delayed_actions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gripper_command_is_policy_gated_by_height_and_subset_reset(monkeypatch) -> None:
    """★F68★ 그리퍼 명령 = ★정책의 닫기 명령 ∧ 높이 제약★ (전이만 막고 강제 개방은 없다).

    [계보] F67은 폐합 시점을 수퍼바이저가 ★탈취★했는데, 근거였던 "닫으면 유지된다"가
      사실이 아니었고(binary_joint_actions.py는 매 스텝 부호로 재계산) teacher 학습을
      F66의 20%로 망가뜨렸다. F68은 시점을 정책에 돌려주고 높이 제약만 건다.
      F68 초판은 제약을 '상태 지정'으로 써서 들어올리는 도중 강제 개방이 일어나
      배포등급이 63.67% -> 0.00%가 됐다. 그래서 제약은 ★open->close 전이에만★ 건다.

    [지금 지키는 것] ①낮은 곳의 닫기 명령은 통과 ②높은 곳의 닫기 명령은 차단
      ③이미 닫혀 있으면 높이와 무관하게 유지(들어올림 보존) ④리셋은 env별로만.
    """
    module = _load_delayed_actions_with_test_doubles(monkeypatch)
    env = types.SimpleNamespace(
        num_envs=2, device="cpu", step_dt=0.05, common_step_counter=0,
        action_manager=types.SimpleNamespace(action=torch.tensor([[-1.0], [-1.0]])),
    )
    cfg = types.SimpleNamespace(
        delay_steps=(0, 1), sample_lag=False, command_transition_steps=24
    )
    action = module.DelayedBinaryJointPositionAction(cfg, env)
    st = module._release_state(env)
    st["seq"] = int(env.common_step_counter)
    # env0 = 제약 통과해 닫힘, env1 = 아직 열림.
    st["closed_prev"] = torch.tensor([True, False])
    action._delay.output = torch.tensor([[1.0], [1.0]])
    action.process_actions(torch.tensor([[-1.0], [-1.0]]))
    # ★상태 규약: 0=open, 1=close★ — 닫는 쪽이 증가한다(configs/common.yaml 주석).
    #   env0(제약 통과 → close)은 0 -> 1/24로 전진, env1(열림 유지)은 0에 머문다.
    np.testing.assert_allclose(
        env._bp_gripper_command_state.numpy(), [1.0 / 24.0, 0.0], atol=1.0e-7
    )
    st["seq"] = int(env.common_step_counter)
    action.process_actions(torch.tensor([[-1.0], [-1.0]]))
    np.testing.assert_allclose(
        env._bp_gripper_command_state.numpy(), [2.0 / 24.0, 0.0], atol=1.0e-7
    )
    before_env1 = float(env._bp_gripper_command_state[1])
    action.reset(torch.tensor([0]))
    assert float(env._bp_gripper_command_state[1]) == before_env1
    assert float(env._bp_gripper_command_state[0]) == 0.0


def test_delayed_ik_receipt_binds_delay_one_to_current_z_floor(monkeypatch) -> None:
    module = _load_delayed_actions_with_test_doubles(monkeypatch)
    env = types.SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.05,
        common_step_counter=0,
        action_manager=types.SimpleNamespace(action=torch.zeros(1, 7)),
        ee_pos=torch.tensor([[0.0, 0.0, 0.08]], dtype=torch.float32),
        ee_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    cfg = types.SimpleNamespace(
        delay_steps=(1, 1),
        sample_lag=False,
        scale=(0.2, 0.2, 0.2, 0.3, 0.3, 0.3),
        tcp_z_floor_m=0.10,
    )
    action = module.DelayedDifferentialInverseKinematicsAction(cfg, env)
    mapped_request = torch.tensor([[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
    guarded_request = torch.tensor([[0.7, 0.6, 0.5, 0.4, 0.3, 0.2]])
    delayed_mapped = torch.tensor([[0.9, 0.8, -0.9, 0.7, 0.6, 0.5]])
    delayed_guarded = torch.tensor([[0.1, -0.2, -0.5, -0.4, 0.2, 0.3]])
    action._delay.output = torch.cat((delayed_mapped, delayed_guarded), dim=-1)

    action.process_actions(
        guarded_request, mapped_raw_actions=mapped_request
    )
    sequence, delayed_raw, pre_floor, applied = action.get_applied_command_receipt()

    assert sequence == 1
    assert action._delay.compute_calls == 1
    torch.testing.assert_close(
        action._delay.last_input,
        torch.cat((mapped_request, guarded_request), dim=-1),
    )
    torch.testing.assert_close(delayed_raw, delayed_mapped * action._scale)
    expected_pre = delayed_guarded * action._scale
    torch.testing.assert_close(pre_floor, expected_pre)
    torch.testing.assert_close(applied[:, :2], expected_pre[:, :2])
    torch.testing.assert_close(applied[:, 3:], expected_pre[:, 3:])
    torch.testing.assert_close(applied[:, 2], torch.tensor([0.02]), atol=1.0e-7, rtol=0.0)
    assert not torch.equal(pre_floor, guarded_request * action._scale)

    action._delay.output = torch.full((1, 12), torch.nan)
    with pytest.raises(ValueError, match="delayed raw/guarded action pair"):
        action.process_actions(
            guarded_request, mapped_raw_actions=mapped_request
        )
    assert action.applied_command_receipt_sequence == 1
    with pytest.raises(RuntimeError, match="no applied arm command receipt"):
        action.get_applied_command_receipt()


def test_delayed_ik_receipt_delay_zero_keeps_raw_and_guarded_same_lineage(monkeypatch) -> None:
    module = _load_delayed_actions_with_test_doubles(monkeypatch)
    env = types.SimpleNamespace(
        num_envs=1,
        device="cpu",
        step_dt=0.05,
        common_step_counter=0,
        action_manager=types.SimpleNamespace(action=torch.zeros(1, 7)),
        ee_pos=torch.tensor([[0.0, 0.0, 0.20]], dtype=torch.float32),
        ee_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    cfg = types.SimpleNamespace(
        delay_steps=(0, 0),
        sample_lag=False,
        scale=(0.2, 0.2, 0.2, 0.3, 0.3, 0.3),
        tcp_z_floor_m=None,
    )
    action = module.DelayedDifferentialInverseKinematicsAction(cfg, env)
    mapped = torch.tensor([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])
    guarded = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    action.process_actions(guarded, mapped_raw_actions=mapped)
    _, delayed_raw, pre_floor, applied = action.get_applied_command_receipt()
    assert action._delay.compute_calls == 1
    torch.testing.assert_close(
        action._delay.last_input, torch.cat((mapped, guarded), dim=-1)
    )
    torch.testing.assert_close(delayed_raw, mapped * action._scale)
    torch.testing.assert_close(pre_floor, guarded * action._scale)
    torch.testing.assert_close(applied, guarded * action._scale)

    action.reset()
    with pytest.raises(RuntimeError, match="no applied arm command receipt"):
        action.get_applied_command_receipt()


def test_sim_tracker_source_keeps_update_after_delay() -> None:
    source = (
        ROOT
        / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/mdp/delayed_actions.py"
    ).read_text(encoding="utf-8")
    body = source.split("class DelayedBinaryJointPositionAction(", 1)[1]
    assert body.index("self._delay.compute(actions)") < body.index(
        "advance_gripper_command_state_tensor"
    )
