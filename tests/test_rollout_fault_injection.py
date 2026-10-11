from __future__ import annotations

import numpy as np

from common.rollout import run_rollout
from common.safety import SafetyGuard
from test_rollout_mock import (
    TARGET_POS,
    MockCamera,
    MockRobot,
    _make_loop_parts,
    cfgs,
    policy,
)


class SensorFailureRobot(MockRobot):
    def get_state(self):
        raise OSError("RTDE disconnected")


class CommandFailureRobot(MockRobot):
    def send_task_delta(self, *args, **kwargs):
        raise OSError("servo transport failed")


class CameraFailure(MockCamera):
    def get_frame(self):
        raise OSError("RealSense disconnected")


class NoForceRobot(MockRobot):
    pass


class RaisingPolicy:
    def act(self, obs):
        raise RuntimeError("inference backend failed")


def test_sensor_exception_trips_before_policy_or_command(cfgs, policy):
    robot = SensorFailureRobot()
    pre, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, MockCamera(), policy, mapper, guard, rate, pre,
        target_pos=TARGET_POS, max_steps=3,
    )
    assert result.tripped and "sensor acquisition failed" in result.trip_reason
    assert robot.estop_calls == 1 and not robot.task_sent


def test_command_exception_trips_and_latches(cfgs, policy):
    robot = CommandFailureRobot()
    pre, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, MockCamera(), policy, mapper, guard, rate, pre,
        target_pos=TARGET_POS, max_steps=3,
    )
    assert result.rollout_safety.denominator == 0
    assert result.rollout_safety.arm_clamp_counts == (0, 0, 0, 0, 0, 0)
    assert result.tripped and "command transmission failed" in result.trip_reason
    assert robot.estop_calls == 1
    assert robot.task_sent == []
    assert robot.gripper_calls == [False]  # DO dispatch preceded transport fault; no later command


def test_camera_exception_trips_before_any_command(cfgs, policy):
    robot = MockRobot()
    pre, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, CameraFailure(), policy, mapper, guard, rate, pre,
        target_pos=TARGET_POS, max_steps=3,
    )
    assert result.tripped and "sensor acquisition failed" in result.trip_reason
    assert robot.estop_calls == 1
    assert robot.task_sent == [] and robot.gripper_calls == []


def test_force_none_trips_before_any_command(cfgs, policy):
    robot = NoForceRobot()
    pre, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, MockCamera(), policy, mapper, guard, rate, pre,
        target_pos=TARGET_POS, max_steps=3, require_force_feedback=True,
    )
    assert result.tripped and "force feedback unavailable" in result.trip_reason
    assert robot.estop_calls == 1
    assert robot.task_sent == [] and robot.gripper_calls == []


def test_policy_exception_trips_before_any_command(cfgs):
    robot = MockRobot()
    pre, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, MockCamera(), RaisingPolicy(), mapper, guard, rate, pre,
        target_pos=TARGET_POS, max_steps=3,
    )
    assert result.tripped and "preprocess/policy failed" in result.trip_reason
    assert robot.estop_calls == 1
    assert robot.task_sent == [] and robot.gripper_calls == []


def test_failed_estop_hook_does_not_clear_latch(cfgs):
    safety = cfgs[2]
    guard = SafetyGuard(safety, estop_hook=lambda reason: (_ for _ in ()).throw(OSError("stop")))
    guard.trip("fault")
    assert guard.tripped is True
    assert "emergency-stop hook failed" in guard.trip_reason
