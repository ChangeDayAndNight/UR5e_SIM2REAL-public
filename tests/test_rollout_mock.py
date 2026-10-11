"""run_rollout 통합테스트(mock) — 시뮬레이터/실기 없이 배포 루프 전체 경로 검증.

Mock RobotInterface/CameraInterface(계약 준수) + torch.jit.trace로 만든
zeros 정책(policy.pt) + make_obs_spec 계약서로 PolicyRunner를 띄우고:
  (a) 정상 max_steps 완주 + obs dump 파일 수
  (b) dry_run=True → send_joint_target/set_gripper 미호출
  (c) workspace 밖 TCP → trip + emergency_stop 호출
  (d) is_ok=False → trip
RateLimiter는 1000Hz로 설정해 테스트 시간을 최소화한다.

실행: python -m pytest tests/test_rollout_mock.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest
import torch

from common.action_mapping import ActionMapper
from common.config_io import (
    build_action_cfg,
    build_preprocess_cfg,
    build_safety_cfg,
    configs_dir,
    load_yaml,
)
from common.interfaces import CameraInterface, RobotInterface
from common.obs_preprocess import obs_dim
from common.policy_io import PolicyRunner, make_obs_spec, save_obs_spec
from common.rollout import run_rollout
from common.safety import SafetyGuard
from common.timing import RateLimiter
from common.types import AppliedArmCommandReceipt, CameraFrame, Intrinsics, RobotState

# ★configs/safety.yaml에서 파생★ (2026-07-28(2차)): 구버전은 (0.45,0,0.25)를 하드코딩해서
#   배치를 −x로 옮기자(로봇이 벨트를 등지는 실셀 배치) 전부 workspace 이탈로 trip했다.
#   좌표를 복제하지 말고 계약 파일에서 뽑아야 배치 변경에 자동으로 따라온다.
_WS = load_yaml(configs_dir() / "safety.yaml")["workspace"]
_WS_MIN = np.asarray([float(v) for v in _WS["min"]])
_WS_MAX = np.asarray([float(v) for v in _WS["max"]])
TCP_INSIDE = tuple(float(v) for v in (_WS_MIN + _WS_MAX) / 2.0)   # AABB 중앙 — 항상 내부
TCP_OUTSIDE = tuple(float(v) for v in                             # x를 하한 밖으로 10cm
                    np.array([_WS_MIN[0] - 0.10, 0.0, (_WS_MIN[2] + _WS_MAX[2]) / 2.0]))
TARGET_POS = np.array([TCP_INSIDE[0], TCP_INSIDE[1], 0.02])


# ------------------------------------------------------------------ mocks ----


class MockRobot(RobotInterface):
    """계약 준수 mock: 호출 기록 + 상태/헬스 주입 가능."""

    def __init__(self, tcp_pos=TCP_INSIDE, ok: bool = True):
        self._tcp_pos = tcp_pos
        self._ok = ok
        self.sent: list[tuple[np.ndarray, float]] = []
        self.task_sent: list[tuple[np.ndarray, np.ndarray, float]] = []
        self.gripper_calls: list[bool] = []
        self.move_calls = 0
        self.estop_calls = 0
        self.closed_flag = False
        self._receipt_sequence = 0
        self.mapped_raw_sent: list[np.ndarray] = []

    def get_state(self) -> RobotState:
        return RobotState(
            joint_pos=np.zeros(6),
            joint_vel=np.zeros(6),
            tcp_pose=np.array([*self._tcp_pos, 1.0, 0.0, 0.0, 0.0]),
            tcp_force=None,
            timestamp=time.perf_counter(),
            gripper_command_state=0.0,
        )

    def send_joint_target(self, q_target: np.ndarray, duration_s: float) -> None:
        self.sent.append((np.asarray(q_target).copy(), duration_s))

    def send_task_delta(
        self,
        delta_pose: np.ndarray,
        current_tcp_pose: np.ndarray,
        duration_s: float,
        *,
        mapped_raw_delta: np.ndarray,
    ) -> AppliedArmCommandReceipt:
        command = np.array(delta_pose, dtype=np.float64, copy=True)
        mapped_raw = np.array(mapped_raw_delta, dtype=np.float64, copy=True)
        self.task_sent.append(
            (command.copy(), np.asarray(current_tcp_pose).copy(), duration_s)
        )
        self.mapped_raw_sent.append(mapped_raw.copy())
        self._receipt_sequence += 1
        return AppliedArmCommandReceipt.from_commands(
            sequence=self._receipt_sequence,
            delayed_mapped_raw=mapped_raw,
            post_delay_pre_floor=command,
            applied=command,
        )

    def move_to_joints(self, q: np.ndarray, speed_scale: float = 0.1) -> None:
        self.move_calls += 1

    def set_gripper(self, closed: bool) -> None:
        self.gripper_calls.append(bool(closed))

    def is_ok(self) -> bool:
        return self._ok

    def emergency_stop(self) -> None:
        self.estop_calls += 1

    def close(self) -> None:
        self.closed_flag = True


class MockCamera(CameraInterface):
    """계약 준수 mock: 0.5m 평면 depth + 회색 rgb 프레임 반환.

    ★2026-07-17 RGB-D 계약★: CameraFrame.rgb는 (H,W,3) uint8 — use_rgb=True
    (현행 common.yaml)면 build_policy_obs가 필수로 소비하고, False면 무시되므로
    항상 채워도 안전하다(계약은 cfg가 결정 — obs_preprocess docstring).
    """

    def __init__(self):
        self.frames_served = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def get_frame(self) -> CameraFrame:
        self.frames_served += 1
        return CameraFrame(
            depth_m=np.full((64, 64), 0.5, dtype=np.float32),
            timestamp=time.perf_counter(),
            rgb=np.full((64, 64, 3), 128, dtype=np.uint8),
        )

    def get_intrinsics(self) -> Intrinsics:
        return Intrinsics(fx=100.0, fy=100.0, cx=32.0, cy=32.0, width=64, height=64)

    def stop(self) -> None:
        self.started = False


class _ZeroPolicy(torch.nn.Module):
    """[N,obs_dim] → [N,7] zeros (trace 가능한 batch-무관 슬라이스 연산)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :7] * 0.0


class _ConstantPolicy:
    def __init__(self, action: np.ndarray):
        self._action = torch.as_tensor(action, dtype=torch.float32).reshape(1, 7)

    def act(self, _obs):
        return self._action.clone()


def test_cube_gate_hold_preserves_gripper_and_does_not_latch_release(cfgs):
    from Phase4.Run_Task import _CubeGate, _GatedPolicy

    gate = _CubeGate(None)
    policy = _GatedPolicy(_ConstantPolicy(np.array([0.1, 0, 0, 0, 0, 0, -1])), gate, 7)
    present = [True, True, False, False, True, True]
    heights = [0.02, 0.06, 0.06, 0.06, 0.06, 0.10]

    class MovingRobot(MockRobot):
        def get_state(self):
            self._tcp_pos = (TCP_INSIDE[0], TCP_INSIDE[1], heights[len(self.task_sent)])
            return super().get_state()

    class GatedCamera(MockCamera):
        def get_frame(self):
            gate.present = present[self.frames_served]
            return super().get_frame()

    robot = MovingRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(robot, GatedCamera(), policy, mapper, guard, rate, pre_cfg,
                         target_pos=TARGET_POS, max_steps=6)
    assert not result.tripped, result.trip_reason
    # ★F93★ hold 중에도 같은 상태를 재명령한다(edge-trigger라 물리 펄스는 없고 command-state만 전진).
    assert robot.gripper_calls == [True] * 6
    assert gate.blocked_steps == 2
    for i in (2, 3):
        np.testing.assert_array_equal(robot.task_sent[i][0], np.zeros(6))
    assert robot.task_sent[4][0][0] > 0, "hold must not permanently latch release"


def test_cube_gate_missing_rgb_fails_closed_without_opening_gripper(cfgs):
    from Phase4.Run_Task import _CubeGate, _GatedPolicy

    gate = _CubeGate(None)
    gate.judge(type("Frame", (), {"rgb": None})())
    assert not gate.present
    robot = MockRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    policy = _GatedPolicy(_ConstantPolicy(np.ones(7)), gate, 7)
    result = run_rollout(robot, MockCamera(), policy, mapper, guard, rate, pre_cfg,
                         target_pos=TARGET_POS, max_steps=2)
    assert not result.tripped
    # ★F93★ hold 중에도 수퍼바이저의 현재 상태(여기서는 한 번도 닫지 않았으므로 open)를 재명령한다 —
    #   어댑터는 edge-trigger라 물리 펄스는 없고 command-state만 학습처럼 전진한다. 닫힌 상태를 hold가
    #   열지 않는 성질은 test_cube_gate_hold_preserves_gripper_and_does_not_latch_release가 고정한다.
    assert robot.gripper_calls == [False, False]
    assert all(not command[0].any() for command in robot.task_sent)


def test_missing_required_rgb_trips_before_any_robot_command(cfgs):
    from Phase4.Run_Task import _CubeGate, _GatedCamera, _GatedPolicy

    class MissingRGBCamera(MockCamera):
        def get_frame(self):
            frame = super().get_frame()
            return CameraFrame(depth_m=frame.depth_m, timestamp=frame.timestamp, rgb=None)

    gate = _CubeGate(None)
    robot = MockRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    policy = _GatedPolicy(_ConstantPolicy(np.ones(7)), gate, 7)
    result = run_rollout(robot, _GatedCamera(MissingRGBCamera(), gate), policy,
                         mapper, guard, rate, pre_cfg, target_pos=TARGET_POS, max_steps=1)
    assert result.tripped and not gate.present
    assert not robot.gripper_calls and not robot.task_sent


# --------------------------------------------------------------- fixtures ----


@pytest.fixture(scope="module")
def cfgs():
    """실제 configs에서 cfg 생성 (config_io 통합 스모크 겸용)."""
    common = load_yaml(configs_dir() / "common.yaml")
    safety = load_yaml(configs_dir() / "safety.yaml")
    return (
        build_preprocess_cfg(common),
        build_action_cfg(common),
        build_safety_cfg(safety),
    )


@pytest.fixture
def policy(tmp_path, cfgs):
    """tiny TorchScript 정책 + obs_spec.json → PolicyRunner 로드."""
    pre_cfg, action_cfg, _ = cfgs
    dim = obs_dim(pre_cfg)
    module = torch.jit.trace(_ZeroPolicy(), torch.zeros(1, dim))
    policy_path = tmp_path / "policy.pt"
    module.save(str(policy_path))
    spec_path = tmp_path / "obs_spec.json"
    save_obs_spec(spec_path, make_obs_spec(pre_cfg, action_cfg,
                                       action_scheme=action_cfg.mode))
    return PolicyRunner(policy_path, spec_path, pre_cfg, action_cfg)


def _make_loop_parts(cfgs, robot: MockRobot):
    """mapper/guard(estop→robot.emergency_stop 연결)/rate 생성.

    ⚠️ SafetyGuard hook은 Callable[[str],None]이고 RobotInterface.emergency_stop은
    인자가 없다 — 배포 스크립트와 동일하게 lambda 어댑터로 잇는다.
    """
    pre_cfg, action_cfg, safety_cfg = cfgs
    mapper = ActionMapper(action_cfg)
    guard = SafetyGuard(safety_cfg, estop_hook=lambda reason: robot.emergency_stop())
    rate = RateLimiter(1000.0)  # 테스트 시간 최소화
    return pre_cfg, mapper, guard, rate


def test_policy_runner_manifest_hashes_bind_same_opened_bytes(tmp_path, cfgs):
    pre_cfg, action_cfg, _ = cfgs
    dim = obs_dim(pre_cfg)
    module = torch.jit.trace(_ZeroPolicy(), torch.zeros(1, dim))
    policy_path = tmp_path / "policy.pt"
    module.save(str(policy_path))
    spec_path = tmp_path / "obs_spec.json"
    save_obs_spec(
        spec_path,
        make_obs_spec(pre_cfg, action_cfg, action_scheme=action_cfg.mode),
    )
    policy_sha = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    PolicyRunner(
        policy_path,
        spec_path,
        pre_cfg,
        action_cfg,
        expected_policy_sha256=policy_sha,
        expected_obs_spec_sha256=spec_sha,
    )
    with pytest.raises(ValueError, match="manifest SHA-256"):
        PolicyRunner(
            policy_path,
            spec_path,
            pre_cfg,
            action_cfg,
            expected_policy_sha256="0" * 64,
            expected_obs_spec_sha256=spec_sha,
        )


# ------------------------------------------------------------------- tests ----


def test_rollout_completes_max_steps_and_dumps(tmp_path, cfgs, policy):
    """(a) 정상 상태에서 max_steps 완주 + 스텝별 dump(.npz) 파일 수 일치."""
    robot = MockRobot()
    camera = MockCamera()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    dump_dir = tmp_path / "dump"
    result = run_rollout(
        robot, camera, policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=5, dry_run=False, obs_dump_dir=dump_dir,
    )
    assert result.steps == 5
    assert result.tripped is False and result.trip_reason == ""
    assert result.rollout_safety.denominator == 5
    assert result.rollout_safety.arm_clamp_counts == (0, 0, 0, 0, 0, 0)
    dumps = sorted(dump_dir.glob("step_*.npz"))
    assert len(dumps) == 5
    with np.load(dumps[0]) as data:
        assert data["obs"].shape == (1, obs_dim(pre_cfg))  # 계약 파라미터(use_rgb)에 따름
        assert data["action"].shape == (mapper.cfg.action_dim,)
        assert data["raw_arm_command"].shape == (6,)
        assert data["safety_arm_command"].shape == (6,)
        assert bool(data["command_issued"]) is True
        assert int(data["arm_command_receipt_sequence"]) == 1
        assert data["q_target"].shape == (6,)
    # 명령 경로: 매 스텝 send + gripper, duration = 1/control_hz(테스트에선 1ms)
    assert len(robot.task_sent) == 5 and len(robot.gripper_calls) == 5
    for delta, tcp_pose, duration in robot.task_sent:
        assert delta.shape == (6,) and np.all(np.isfinite(delta))
        np.testing.assert_allclose(delta, 0.0)
        assert tcp_pose.shape == (7,)
        assert duration == pytest.approx(rate.period_s)
    assert robot.gripper_calls == [False] * 5  # action[-1]=0.0 > 0.0 아님
    assert robot.estop_calls == 0
    assert camera.frames_served == 5


def test_rollout_dry_run_sends_nothing(cfgs, policy):
    """(b) dry_run=True: 취득/추론/판정은 돌되 send_joint_target/set_gripper 미호출."""
    robot = MockRobot()
    camera = MockCamera()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, camera, policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=3, dry_run=True,
    )
    assert result.steps == 3 and result.tripped is False
    assert result.rollout_safety.denominator == 0
    assert result.rollout_safety.arm_clamp_counts == (0, 0, 0, 0, 0, 0)
    assert robot.sent == [] and robot.task_sent == [] and robot.gripper_calls == []
    assert camera.frames_served == 3  # 취득 경로는 전부 실행됐다


def test_rollout_clamp_counts_use_same_receipt_lineage_and_fixed_tolerance(
    tmp_path, cfgs, policy
):
    class ReceiptRobot(MockRobot):
        def send_task_delta(self, *args, **kwargs):
            receipt = super().send_task_delta(*args, **kwargs)
            applied = np.asarray(receipt.applied, dtype=np.float64).copy()
            applied[0] += 0.5e-12  # ignored: below the fixed tolerance
            applied[1] += 2.0e-12  # counted: above the fixed tolerance
            return AppliedArmCommandReceipt.from_commands(
                sequence=receipt.sequence,
                delayed_mapped_raw=receipt.delayed_mapped_raw,
                post_delay_pre_floor=receipt.post_delay_pre_floor,
                applied=applied,
            )

    robot = ReceiptRobot()
    camera = MockCamera()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, camera, policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=2, dry_run=False,
        obs_dump_dir=tmp_path / "receipt_dump",
    )
    assert result.tripped is False
    assert result.rollout_safety.denominator == 2
    assert result.rollout_safety.arm_clamp_counts == (0, 2, 0, 0, 0, 0)
    with np.load(tmp_path / "receipt_dump/step_0000.npz") as data:
        assert data["raw_arm_command"][1] == 0.0
        assert data["safety_arm_command"][1] == 0.0
        assert data["arm_command"][1] == pytest.approx(2.0e-12)
        assert int(data["arm_command_receipt_sequence"]) == 1


def test_rollout_safetyguard_only_clamp_is_counted_at_one_hundred_percent(
    tmp_path, cfgs
):
    action = np.zeros(7, dtype=np.float32)
    # ★Teacher v2★ 스케일(0.0125 m) = SafetyGuard 스텝 상한이라 ±1은 잘리지 않는다.
    #   2.0이면 매핑 델타 0.025 m > 상한 0.0125 m → SafetyGuard만 자르는 상황이 재현된다.
    action[0] = 2.0
    robot = MockRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot,
        MockCamera(),
        _ConstantPolicy(action),
        mapper,
        guard,
        rate,
        pre_cfg,
        target_pos=TARGET_POS,
        max_steps=2,
        dry_run=False,
        obs_dump_dir=tmp_path / "safety_only",
    )
    assert result.tripped is False
    assert result.rollout_safety.denominator == 2
    assert result.rollout_safety.arm_clamp_counts == (2, 0, 0, 0, 0, 0)
    with np.load(tmp_path / "safety_only/step_0000.npz") as data:
        assert data["raw_arm_command"][0] == pytest.approx(2.0 * mapper.cfg.task_pos_scale_m)
        assert data["safety_arm_command"][0] == pytest.approx(
            guard.cfg.tcp_linear_vel_max_m_s * rate.period_s
        )
        assert data["arm_command"][0] == data["safety_arm_command"][0]


def test_rollout_counts_outer_and_adapter_clamps_as_channel_union(cfgs):
    class BothRobot(MockRobot):
        def send_task_delta(self, *args, **kwargs):
            receipt = super().send_task_delta(*args, **kwargs)
            applied = np.asarray(receipt.applied, dtype=np.float64).copy()
            applied[1] += 2.0e-12
            return AppliedArmCommandReceipt.from_commands(
                sequence=receipt.sequence,
                delayed_mapped_raw=receipt.delayed_mapped_raw,
                post_delay_pre_floor=receipt.post_delay_pre_floor,
                applied=applied,
            )

    action = np.zeros(7, dtype=np.float32)
    action[0] = 1.0
    robot = BothRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot,
        MockCamera(),
        _ConstantPolicy(action),
        mapper,
        guard,
        rate,
        pre_cfg,
        target_pos=TARGET_POS,
        max_steps=1,
        dry_run=False,
    )
    assert result.tripped is False
    assert result.rollout_safety.arm_clamp_counts == (1, 1, 0, 0, 0, 0)


def test_rollout_clamp_reversal_cannot_cancel_intervention(tmp_path, cfgs):
    class ReversingRobot(MockRobot):
        def send_task_delta(self, *args, **kwargs):
            receipt = super().send_task_delta(*args, **kwargs)
            return AppliedArmCommandReceipt.from_commands(
                sequence=receipt.sequence,
                delayed_mapped_raw=receipt.delayed_mapped_raw,
                post_delay_pre_floor=receipt.post_delay_pre_floor,
                applied=receipt.delayed_mapped_raw,
            )

    action = np.zeros(7, dtype=np.float32)
    action[0] = 1.0
    robot = ReversingRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot,
        MockCamera(),
        _ConstantPolicy(action),
        mapper,
        guard,
        rate,
        pre_cfg,
        target_pos=TARGET_POS,
        max_steps=1,
        dry_run=False,
        obs_dump_dir=tmp_path / "reversal",
    )
    assert result.tripped is False
    assert result.rollout_safety.arm_clamp_counts == (1, 0, 0, 0, 0, 0)
    with np.load(tmp_path / "reversal/step_0000.npz") as data:
        assert data["raw_arm_command"][0] == data["arm_command"][0]
        assert data["raw_arm_command"][0] != data["safety_arm_command"][0]


def test_rollout_send_failure_never_counts_outer_clamp(cfgs):
    class FailingRobot(MockRobot):
        def send_task_delta(self, *args, **kwargs):
            raise OSError("simulated send failure")

    action = np.zeros(7, dtype=np.float32)
    action[0] = 1.0
    robot = FailingRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot,
        MockCamera(),
        _ConstantPolicy(action),
        mapper,
        guard,
        rate,
        pre_cfg,
        target_pos=TARGET_POS,
        max_steps=1,
        dry_run=False,
    )
    assert result.tripped is True
    assert "simulated send failure" in result.trip_reason
    assert result.rollout_safety.denominator == 0
    assert result.rollout_safety.arm_clamp_counts == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("mapped_raw", "pre_floor", "applied"),
    [
        (np.zeros((2, 3)), np.zeros(6), np.zeros(6)),
        (np.zeros(6), np.zeros((2, 3)), np.zeros(6)),
        (np.zeros(6), np.zeros(6), np.zeros((2, 3))),
        (np.full(6, np.nan), np.zeros(6), np.zeros(6)),
        (np.zeros(6), np.full(6, np.nan), np.zeros(6)),
        (np.zeros(6), np.zeros(6), np.full(6, np.inf)),
    ],
)
def test_applied_command_receipt_rejects_non_6d_or_nonfinite(
    mapped_raw, pre_floor, applied
):
    with pytest.raises(ValueError, match="finite 6D command"):
        AppliedArmCommandReceipt.from_commands(
            sequence=1,
            delayed_mapped_raw=mapped_raw,
            post_delay_pre_floor=pre_floor,
            applied=applied,
        )


def test_rollout_missing_applied_receipt_is_not_counted(cfgs, policy):
    class MissingReceiptRobot(MockRobot):
        def send_task_delta(self, *args, **kwargs):
            super().send_task_delta(*args, **kwargs)
            return None

    robot = MissingReceiptRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, MockCamera(), policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=2, dry_run=False,
    )
    assert result.tripped is True
    assert "applied-command receipt" in result.trip_reason
    assert result.rollout_safety.denominator == 0
    assert result.rollout_safety.arm_clamp_counts == (0, 0, 0, 0, 0, 0)


def test_rollout_replayed_applied_receipt_is_not_counted_twice(cfgs, policy):
    class ReplayReceiptRobot(MockRobot):
        def send_task_delta(self, *args, **kwargs):
            receipt = super().send_task_delta(*args, **kwargs)
            return AppliedArmCommandReceipt.from_commands(
                sequence=1,
                delayed_mapped_raw=receipt.delayed_mapped_raw,
                post_delay_pre_floor=receipt.post_delay_pre_floor,
                applied=receipt.applied,
            )

    robot = ReplayReceiptRobot()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, MockCamera(), policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=2, dry_run=False,
    )
    assert result.tripped is True
    assert "stale/non-monotonic receipt" in result.trip_reason
    assert result.rollout_safety.denominator == 1
    assert result.rollout_safety.arm_clamp_counts == (0, 0, 0, 0, 0, 0)


def test_command_intervals_never_catch_up_after_processing_overrun(
    monkeypatch, cfgs, policy
):
    """A slow step followed by a fast step must not emit back-to-back commands."""

    class FakeClock:
        now = 0.0

        def perf_counter(self):
            return self.now

        def sleep(self, duration):
            self.now += max(0.0, float(duration))

    clock = FakeClock()
    monkeypatch.setattr("common.timing.time.perf_counter", clock.perf_counter)
    monkeypatch.setattr("common.timing.time.sleep", clock.sleep)

    class DelayedPolicy:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.delays = iter((0.0, 0.070, 0.010))

        def act(self, obs):
            clock.now += next(self.delays)
            return self.wrapped.act(obs)

    class TimestampRobot(MockRobot):
        def __init__(self):
            super().__init__()
            self.command_times = []

        def send_task_delta(self, *args, **kwargs):
            self.command_times.append(clock.perf_counter())
            return super().send_task_delta(*args, **kwargs)

    robot = TimestampRobot()
    pre_cfg, mapper, guard, _ = _make_loop_parts(cfgs, robot)
    rate = RateLimiter(20.0)
    result = run_rollout(
        robot, MockCamera(), DelayedPolicy(policy), mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=3, dry_run=False,
    )
    assert result.tripped is False
    assert len(robot.command_times) == 3
    intervals = np.diff(robot.command_times)
    assert np.all(intervals >= rate.period_s - 1e-12), intervals


def test_rollout_workspace_violation_trips_and_estops(cfgs, policy):
    """(c) workspace 밖 TCP → 첫 스텝에서 trip + emergency_stop 1회, 명령 0건."""
    robot = MockRobot(tcp_pos=TCP_OUTSIDE)
    camera = MockCamera()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, camera, policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=10, dry_run=False,
    )
    assert result.tripped is True
    assert "workspace" in result.trip_reason
    assert result.steps == 0  # trip 스텝은 완주로 세지 않는다
    assert robot.estop_calls == 1
    assert robot.sent == [] and robot.gripper_calls == []  # 위반 시 명령 차단


def test_rollout_robot_not_ok_trips(cfgs, policy):
    """(d) is_ok=False → 'robot not ok' trip + emergency_stop, 추론 진입 전 중단."""
    robot = MockRobot(ok=False)
    camera = MockCamera()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, camera, policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=10, dry_run=False,
    )
    assert result.tripped is True
    assert "robot not ok" in result.trip_reason
    assert result.steps == 0
    assert robot.estop_calls == 1
    assert robot.sent == []


def test_rollout_nan_state_trips_before_any_command(cfgs, policy):
    """★감사 B2(2026-08-12) 회귀 고정★ — 비유한 상태는 ★루프 입구에서★ trip해야 한다.

    [사고] NaN은 모든 비교에서 False라 SafetyGuard의 clamp(무변화 통과)·AABB(ok=True)·
      관절속도 판정을 전부 통과했다 — 실행 재현으로 확인된 fail-open. 센서/드라이버가
      비유한값을 주면 servoL 인자까지 NaN이 도달할 수 있었다.
    """

    class NaNStateRobot(MockRobot):
        def get_state(self) -> RobotState:
            st = super().get_state()
            jp = st.joint_pos.copy()
            jp[2] = np.nan
            return RobotState(
                joint_pos=jp, joint_vel=st.joint_vel, tcp_pose=st.tcp_pose,
                tcp_force=st.tcp_force, timestamp=st.timestamp,
                gripper_command_state=st.gripper_command_state,
            )

    robot = NaNStateRobot()
    camera = MockCamera()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, camera, policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=10, dry_run=False,
    )
    assert result.tripped is True
    assert "non-finite state" in result.trip_reason
    assert result.steps == 0
    assert robot.estop_calls == 1
    assert robot.sent == [] and robot.task_sent == [] and robot.gripper_calls == []


def test_rollout_nan_action_trips_before_any_command(tmp_path, cfgs):
    """★감사 B2 회귀 고정★ — 정책이 NaN을 내면 명령 전송 전에 trip해야 한다."""

    class _NaNPolicy(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[:, :7] * torch.nan

    pre_cfg, action_cfg, _ = cfgs
    dim = obs_dim(pre_cfg)
    module = torch.jit.trace(_NaNPolicy(), torch.zeros(1, dim))
    policy_path = tmp_path / "nan_policy.pt"
    module.save(str(policy_path))
    spec_path = tmp_path / "obs_spec.json"
    save_obs_spec(spec_path, make_obs_spec(pre_cfg, action_cfg,
                                           action_scheme=action_cfg.mode))
    nan_policy = PolicyRunner(policy_path, spec_path, pre_cfg, action_cfg)

    robot = MockRobot()
    camera = MockCamera()
    pre_cfg, mapper, guard, rate = _make_loop_parts(cfgs, robot)
    result = run_rollout(
        robot, camera, nan_policy, mapper, guard, rate, pre_cfg,
        target_pos=TARGET_POS, max_steps=10, dry_run=False,
    )
    assert result.tripped is True
    assert "policy output must be finite" in result.trip_reason
    assert robot.task_sent == [] and robot.gripper_calls == []
    assert robot.estop_calls == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
