from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sim.sim_robot import SimRobot


class _Termination:
    def __init__(self, calls: list[str], *, done: bool = False):
        self.calls = calls
        self.done = done
        self.terminated = torch.tensor([done])
        self.time_outs = torch.tensor([False])
        self._term_dones = torch.tensor([[done]])
        self._terminated_buf = self.terminated
        self._truncated_buf = self.time_outs

    @property
    def dones(self):
        return self._terminated_buf | self._truncated_buf

    def compute(self):
        self.calls.append("termination")
        self.terminated = torch.tensor([self.done])
        return self.terminated | self.time_outs


class _ComputeManager:
    def __init__(self, calls: list[str], name: str):
        self.calls = calls
        self.name = name

    def compute(self, *, dt: float):
        self.calls.append(self.name)
        return torch.tensor([0.0])


class _ObservationManager:
    def __init__(self, calls: list[str]):
        self.calls = calls

    def compute(self, *, update_history: bool):
        assert update_history is True
        self.calls.append("observation")
        return {"policy": torch.tensor([[0.0]])}


class _EventManager:
    available_modes = ["interval"]

    def __init__(self, calls: list[str]):
        self.calls = calls

    def apply(self, *, mode: str, dt: float):
        assert mode == "interval" and dt == 0.05
        self.calls.append("event")


def _robot_with_fake_env(*, done: bool = False) -> tuple[SimRobot, SimpleNamespace, list[str]]:
    calls: list[str] = []
    env = SimpleNamespace(
        episode_length_buf=torch.tensor([0], dtype=torch.int64),
        common_step_counter=0,
        step_dt=0.05,
        termination_manager=_Termination(calls, done=done),
        reward_manager=_ComputeManager(calls, "reward"),
        command_manager=_ComputeManager(calls, "command"),
        event_manager=_EventManager(calls),
        observation_manager=_ObservationManager(calls),
    )
    robot = SimRobot.__new__(SimRobot)
    robot._env = env
    return robot, env, calls


def test_direct_control_step_matches_isaaclab_232_post_physics_order_once():
    robot, env, calls = _robot_with_fake_env()
    robot._finish_control_step()
    assert env.episode_length_buf.tolist() == [1]
    assert env.common_step_counter == 1
    assert calls == ["termination", "reward", "command", "event", "observation"]
    assert env.reset_buf.tolist() == [False]
    assert env.reset_terminated.tolist() == [False]
    assert env.reset_time_outs.tolist() == [False]


def test_begin_episode_clears_stale_termination_without_compute_or_counter_advance():
    robot, env, calls = _robot_with_fake_env()
    env.reset_buf = torch.tensor([True])
    env.reset_terminated = torch.tensor([True])
    env.reset_time_outs = torch.tensor([True])
    robot.begin_episode()
    assert calls == []
    assert env.episode_length_buf.tolist() == [0]
    assert env.common_step_counter == 0
    assert env.reset_buf.tolist() == [False]
    assert env.reset_terminated.tolist() == [False]
    assert env.reset_time_outs.tolist() == [False]
    assert env.termination_manager._term_dones.tolist() == [[False]]


def test_direct_adapter_never_duplicates_manager_based_env_step_and_finishes_after_decimation():
    source = inspect.getsource(SimRobot)
    send_source = inspect.getsource(SimRobot.send_task_delta)
    assert "self._env.step(" not in source
    assert send_source.count("self._finish_control_step()") == 1
    assert send_source.index("for _ in range(n_steps):") < send_source.index(
        "self._finish_control_step()"
    )


class _ReceiptAction:
    def __init__(self):
        self._scale = torch.ones((1, 6), dtype=torch.float32)
        self.applied_command_receipt_sequence = 7
        self.apply_calls = 0

    def process_actions(self, raw, *, mapped_raw_actions):
        assert tuple(raw.shape) == (1, 6)
        assert tuple(mapped_raw_actions.shape) == (1, 6)
        self.applied_command_receipt_sequence += 1
        self._mapped_raw = mapped_raw_actions.clone()
        self._pre = torch.tensor([[0.0, 0.0, -0.03, 0.0, 0.0, 0.0]])
        self._applied = torch.tensor([[0.0, 0.0, -0.01, 0.0, 0.0, 0.0]])

    def get_applied_command_receipt(self):
        return (
            self.applied_command_receipt_sequence,
            self._mapped_raw,
            self._pre,
            self._applied,
        )

    def apply_actions(self):
        self.apply_calls += 1


class _GripperStub:
    """★F93★ send_task_delta는 팔 항 다음에 그리퍼 항을 1회 처리하고 물리 스텝마다 apply한다."""

    def __init__(self) -> None:
        self.process_calls = 0
        self.apply_calls = 0

    def process_actions(self, actions):
        assert tuple(actions.shape) == (1, 1)
        self.process_calls += 1

    def apply_actions(self):
        self.apply_calls += 1


def _receipt_robot() -> tuple[SimRobot, _ReceiptAction, list[str]]:
    calls: list[str] = []
    action = _ReceiptAction()
    robot = SimRobot.__new__(SimRobot)
    robot._estopped = False
    robot._arm_action = action
    robot._gripper_action = _GripperStub()
    # ★F93★ 팔 항의 수퍼바이저가 읽는 액션 버퍼(7D = 팔 6 + 그리퍼 1)를 리허설이 직접 채운다.
    robot._env = SimpleNamespace(
        device="cpu",
        action_manager=SimpleNamespace(_action=torch.zeros(1, 7), _prev_action=torch.zeros(1, 7)),
    )
    robot._physics_dt = 0.01
    robot._step_physics = lambda count: calls.append(f"physics:{count}")
    robot._finish_control_step = lambda: calls.append("finish")
    return robot, action, calls


def test_sim_send_returns_same_lineage_receipt_only_after_control_step_success():
    robot, action, calls = _receipt_robot()
    mapped_raw = np.array([0.1, 0.0, -0.2, 0.0, 0.0, 0.0])
    receipt = robot.send_task_delta(
        np.zeros(6),
        np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
        0.02,
        mapped_raw_delta=mapped_raw,
    )
    assert receipt.sequence == 8
    np.testing.assert_allclose(receipt.delayed_mapped_raw, mapped_raw)
    assert receipt.post_delay_pre_floor[2] == pytest.approx(-0.03)
    assert receipt.applied[2] == pytest.approx(-0.01)
    assert action.apply_calls == 2
    assert calls == ["physics:1", "physics:1", "finish"]


def test_sim_send_failure_never_completes_or_returns_success_receipt():
    robot, action, calls = _receipt_robot()

    def fail_physics(_count):
        calls.append("physics:failed")
        raise OSError("physics step failed")

    robot._step_physics = fail_physics
    with pytest.raises(OSError, match="physics step failed"):
        robot.send_task_delta(
            np.zeros(6),
            np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
            0.02,
            mapped_raw_delta=np.zeros(6),
        )
    assert action.apply_calls == 1
    assert calls == ["physics:failed"]


def test_sim_send_rejects_invalid_action_scale_before_processing_or_physics():
    robot, action, calls = _receipt_robot()
    action._scale[0, 3] = 0.0
    with pytest.raises(RuntimeError, match="action scale"):
        robot.send_task_delta(
            np.zeros(6),
            np.array([0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]),
            0.02,
            mapped_raw_delta=np.zeros(6),
        )
    assert action.applied_command_receipt_sequence == 7
    assert action.apply_calls == 0
    assert calls == []


def test_phase3_reset_target_and_stale_refresh_order_is_source_locked():
    root = Path(__file__).resolve().parents[1]
    source = (root / "Phase3/SimDeploy_Rehearsal.py").read_text(encoding="utf-8")
    reset = source.index("env.reset()")
    target = source.index("set_rollout_target(env, target)", reset)
    refresh = source.index("robot.begin_episode()", target)
    rollout = source.index("result = run_rollout(", refresh)
    assert reset < target < refresh < rollout


def test_phase3_candidate_argument_is_checked_before_app_launch():
    root = Path(__file__).resolve().parents[1]
    source = (root / "Phase3/SimDeploy_Rehearsal.py").read_text(encoding="utf-8")
    declaration = source.index('parser.add_argument("--candidate-dir"')
    add_launcher = source.index("AppLauncher.add_app_launcher_args(parser)")
    parse_args = source.index("args = parser.parse_args()", add_launcher)
    candidate_check = source.index("if args.candidate_dir is None:", parse_args)
    launch = source.index("app = AppLauncher(args).app", candidate_check)
    assert declaration < add_launcher < parse_args < candidate_check < launch
    assert 'parser.error("--candidate-dir is required")' in source
    assert "required=True" not in source[declaration:add_launcher]
