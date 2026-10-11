"""SimRobot — RobotInterface의 Isaac Lab 구현 (Phase3 SimDeploy_Rehearsal 전용).

목적: 요소6 쌍둥이의 sim 쪽 절반. common/rollout.py의 배포 루프가 실기 코드 경로
      그대로 sim 위에서 돌도록 ManagerBasedRLEnv의 "robot" articulation을
      RobotInterface 계약으로 감싼다. ★단일 env(num_envs=1) 리허설 전용★ —
      학습 경로(ActionManager)와는 무관하다.
실행: 직접 실행 없음 — Phase3/SimDeploy_Rehearsal.py가 생성해 run_rollout에 주입.
산출: SimRobot 클래스.

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from common.interfaces import RobotInterface
from common.types import AppliedArmCommandReceipt, RobotState

from common.robot_spec import UR5E_ARM_JOINT_NAMES


class SimRobot(RobotInterface):
    """Isaac Lab articulation 래퍼 — real/real_robot.py(RTDE)와 동일 계약.

    계약 대응표 (RobotState: quat=wxyz, base frame, depth 계약과 동일 문서):
      get_state         → articulation.data + ee_frame(FrameTransformer, 요소4 TCP)
      send_joint_target → set_joint_position_target + 물리 서브스텝(servoJ 대응)
      move_to_joints    → 상태 직접 기록(순간이동) — sim에서의 '블로킹 이동' 등가
      set_gripper       → 부호 기록 → send_task_delta가 학습 순서(팔 항→그리퍼 항)로 적용 (★F93★)
      is_ok             → 항상 True (sim에는 protective stop이 없음)
      emergency_stop    → 현재 자세 고정 + 이후 명령 차단(latch)
    """

    def __init__(self, env, robot_name: str = "robot", ee_frame_name: str = "ee_frame"):
        """env = 생성 완료된 ManagerBasedRLEnv (num_envs=1).

        ⚠️ env.reset()이 최소 1회 호출된 뒤 사용할 것 (센서/버퍼 초기화).
        """
        if env.num_envs != 1:
            raise ValueError(
                f"[sim] SimRobot은 단일 env 리허설 전용 — num_envs={env.num_envs}"
            )
        self._env = env
        self._robot = env.scene[robot_name]
        self._ee = env.scene[ee_frame_name]
        self._arm_ids, _ = self._robot.find_joints(UR5E_ARM_JOINT_NAMES, preserve_order=True)
        self._arm_action = env.action_manager.get_term("arm_action")
        self._gripper_action = env.action_manager.get_term("gripper_action")
        self._estopped = False
        self._physics_dt = float(env.cfg.sim.dt)  # 요소3: 1/physics_hz
        print(f"[sim] SimRobot 준비 완료 (arm joints={len(self._arm_ids)}, dt={self._physics_dt:.6f})")

    # ── 내부 유틸 ────────────────────────────────────────────────────────────
    def _q_tensor(self, q: np.ndarray) -> torch.Tensor:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != len(self._arm_ids):
            raise ValueError(f"[sim] 관절 수 불일치: 기대 {len(self._arm_ids)}, 실제 {q.shape[0]}")
        return torch.as_tensor(q, dtype=torch.float32, device=self._env.device).unsqueeze(0)

    def _step_physics(self, n_steps: int) -> None:
        """물리 n스텝 전진 + 씬 데이터 갱신 — 리허설 루프의 시간 전진 담당.

        ⚠️ TODO(확인 필요): 아래 시퀀스(write_data_to_sim → sim.step → scene.update)는
          Isaac Lab standalone 튜토리얼 관례이나, ManagerBasedRLEnv 내부 step과의
          정합(센서 update_period 처리 등)은 Phase3 스모크에서 검증할 것.
        """
        is_rendering = self._env.sim.has_gui() or self._env.sim.has_rtx_sensors()
        for _ in range(n_steps):
            # Keep the exact Isaac Lab 2.3.2 ManagerBasedRLEnv physics order.
            # Action terms are deliberately applied by the caller, so invoking
            # env.step()/action_manager.process_action here would issue twice.
            self._env._sim_step_counter += 1
            self._env.scene.write_data_to_sim()
            self._env.sim.step(render=False)
            if (
                self._env._sim_step_counter % self._env.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self._env.sim.render()
            self._env.scene.update(dt=self._physics_dt)

    def _finish_control_step(self) -> None:
        """Refresh RL managers exactly once after one control-period decimation.

        Phase 3 drives the Isaac action terms directly so the same deployment
        loop can be used in simulation.  This is the post-physics portion of
        Isaac Lab 2.3.2 ``ManagerBasedRLEnv.step`` up to (but deliberately not
        including) its automatic reset or a second action processing call.
        Keeping termination computation at this boundary makes
        ``placed_success`` current when ``run_rollout`` queries it.
        """
        env = self._env
        env.episode_length_buf += 1
        env.common_step_counter += 1
        env.reset_buf = env.termination_manager.compute()
        env.reset_terminated = env.termination_manager.terminated
        env.reset_time_outs = env.termination_manager.time_outs
        env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
        env.command_manager.compute(dt=env.step_dt)
        if "interval" in env.event_manager.available_modes:
            env.event_manager.apply(mode="interval", dt=env.step_dt)
        env.obs_buf = env.observation_manager.compute(update_history=True)

    def begin_episode(self) -> None:
        """Clear cached dones after the caller's normal ``env.reset()``.

        Isaac Lab 2.3.2 ``env.reset`` already calls
        ``termination_manager.reset`` (including stateful term resets), but the
        manager leaves its cached ``_term_dones``/net buffers untouched until
        the next compute.  Clear only those documented-v2.3.2 cache tensors;
        calling ``compute`` here could advance a stateful success term before a
        control step and would violate the exactly-once receipt boundary.
        """
        env = self._env
        if bool((env.episode_length_buf != 0).any().item()):
            raise RuntimeError("Phase 3 begin_episode requires a completed env.reset()")
        manager = env.termination_manager
        for name in ("_term_dones", "_terminated_buf", "_truncated_buf"):
            buffer = getattr(manager, name, None)
            if buffer is None or not hasattr(buffer, "zero_"):
                raise RuntimeError(
                    f"Isaac Lab 2.3.2 termination cache is unavailable: {name}"
                )
            buffer.zero_()
        env.reset_buf = manager.dones
        env.reset_terminated = manager.terminated
        env.reset_time_outs = manager.time_outs

    # ── RobotInterface 구현 ──────────────────────────────────────────────────
    def get_state(self) -> RobotState:
        """articulation/ee_frame 스냅샷 → RobotState (base frame, quat wxyz)."""
        joint_pos = self._robot.data.joint_pos[0, self._arm_ids].cpu().numpy().astype(np.float64)
        joint_vel = self._robot.data.joint_vel[0, self._arm_ids].cpu().numpy().astype(np.float64)
        # TCP 포즈: FrameTransformer의 source(=base_link) 기준 출력 — 요소4의
        # tcp_offset.yaml이 이미 반영된 값 (real의 apply_tcp_offset과 동일 의미).
        tcp_pos = self._ee.data.target_pos_source[0, 0, :].cpu().numpy().astype(np.float64)
        tcp_quat = self._ee.data.target_quat_source[0, 0, :].cpu().numpy().astype(np.float64)
        command_state = getattr(self._env, "_bp_gripper_command_state", None)
        command_value = None
        if command_state is not None:
            value = float(command_state[0].item())
            command_value = value if np.isfinite(value) and 0.0 <= value <= 1.0 else None
        return RobotState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            tcp_pose=np.concatenate([tcp_pos, tcp_quat]),
            # TODO: ContactSensor 활성화 시 손가락 접촉력 합력으로 대체 (지금은 미지원)
            tcp_force=None,
            timestamp=time.perf_counter(),
            # Same delayed-command tracker consumed by training policy_obs.
            gripper_command_state=command_value,
        )

    def send_joint_target(self, q_target: np.ndarray, duration_s: float) -> None:
        """스트리밍 추종(servoJ 대응): 목표 기록 후 duration_s만큼 물리 전진.

        real 대응: servoJ(q, time=duration_s) — sim에서는 PD(implicit actuator)가
        duration_s = 1/control_hz 동안 목표를 추종한다(요소3 정합).
        """
        if self._estopped:
            print("[sim] ⚠️ e-stop latch 상태 — send_joint_target 무시")
            return
        self._robot.set_joint_position_target(self._q_tensor(q_target), joint_ids=self._arm_ids)
        n_steps = max(1, round(duration_s / self._physics_dt))
        self._step_physics(n_steps)
        self._finish_control_step()

    def send_task_delta(
        self,
        delta_pose: np.ndarray,
        current_tcp_pose: np.ndarray,
        duration_s: float,
        *,
        mapped_raw_delta: np.ndarray,
    ) -> AppliedArmCommandReceipt:
        """Route mapped/raw and guarded commands through one delayed IK lineage."""
        if self._estopped:
            raise RuntimeError("[sim] e-stop latch blocks send_task_delta")
        delta_array = np.array(delta_pose, dtype=np.float64, copy=True)
        mapped_raw_array = np.array(mapped_raw_delta, dtype=np.float64, copy=True)
        current_array = np.array(current_tcp_pose, dtype=np.float64, copy=True)
        duration = float(duration_s)
        if (
            delta_array.shape != (6,)
            or mapped_raw_array.shape != (6,)
            or not np.isfinite(delta_array).all()
            or not np.isfinite(mapped_raw_array).all()
        ):
            raise ValueError("[sim] mapped/raw and guarded task deltas must be finite 6D vectors")
        if current_array.shape != (7,) or not np.isfinite(current_array).all():
            raise ValueError("[sim] current TCP pose must be a finite 7D vector")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("[sim] task command duration must be finite and positive")
        sequence_before = self._arm_action.applied_command_receipt_sequence
        if isinstance(sequence_before, bool) or not isinstance(sequence_before, int) \
                or sequence_before < 0:
            raise RuntimeError("[sim] invalid action-term receipt sequence")
        action_scale = self._arm_action._scale
        if tuple(action_scale.shape) != (1, 6) or not bool(
            torch.isfinite(action_scale).all().item()
        ) or bool((action_scale == 0).any().item()):
            raise RuntimeError("[sim] action scale must be finite, non-zero, and shape (1, 6)")
        delta = torch.as_tensor(
            delta_array.astype(np.float32, copy=False)[None, :],
            device=self._env.device,
        )
        mapped_raw_delta_tensor = torch.as_tensor(
            mapped_raw_array.astype(np.float32, copy=False)[None, :],
            device=self._env.device,
        )
        # ActionMapper already applied the same scale as the action term.
        raw = delta / action_scale
        mapped_raw = mapped_raw_delta_tensor / action_scale
        # ★F93(리뷰 AX-01)★ 팔 항의 릴리스 수퍼바이저는 정책 그리퍼 부호를 env.action_manager.action[:, -1]
        #   에서 읽는다(학습과 동일). 리허설은 액션 매니저를 거치지 않으므로 set_gripper가 기록한 부호를
        #   ★먼저★ 액션 버퍼에 써 넣고(팔 6D도 함께), 학습 순서대로 팔 항 → 그리퍼 항을 처리한다.
        #   종전에는 버퍼가 0(=open)으로 남아 sim 리허설은 그리퍼를 영영 닫지 못했다.
        gripper_sign = -1.0 if getattr(self, "_pending_gripper_closed", False) else 1.0
        manager = self._env.action_manager
        manager._prev_action[:] = manager._action
        manager._action[0, : raw.shape[1]] = raw[0]
        manager._action[0, -1] = gripper_sign
        self._arm_action.process_actions(raw, mapped_raw_actions=mapped_raw)
        sequence, delayed_mapped_raw, pre_floor, applied = (
            self._arm_action.get_applied_command_receipt()
        )
        if sequence != sequence_before + 1:
            raise RuntimeError("[sim] stale or non-monotonic action-term receipt")
        if (
            tuple(delayed_mapped_raw.shape) != (1, 6)
            or tuple(pre_floor.shape) != (1, 6)
            or tuple(applied.shape) != (1, 6)
        ):
            raise RuntimeError("[sim] action-term receipt must contain one 6D command")
        receipt = AppliedArmCommandReceipt.from_commands(
            sequence=sequence,
            delayed_mapped_raw=delayed_mapped_raw[0].detach().cpu().numpy(),
            post_delay_pre_floor=pre_floor[0].detach().cpu().numpy(),
            applied=applied[0].detach().cpu().numpy(),
        )
        # 그리퍼 항은 팔 항이 이번 스텝에 갱신한 수퍼바이저 상태를 소비한다(seq 가드) — 팔 항 ★다음★에,
        #   물리 전진 ★전★에 1회 처리해 command-state 추적기를 학습처럼 한 스텝 전진시킨다.
        gripper_cmd = torch.tensor([[gripper_sign]], dtype=torch.float32, device=self._env.device)
        self._gripper_action.process_actions(gripper_cmd)
        n_steps = max(1, round(duration / self._physics_dt))
        for _ in range(n_steps):
            self._arm_action.apply_actions()
            self._gripper_action.apply_actions()
            self._step_physics(1)
        self._finish_control_step()
        # Return only after the entire physics/manager step completed.
        return receipt

    def move_to_joints(self, q: np.ndarray, speed_scale: float = 0.1) -> None:
        """블로킹 절대 이동 — sim 등가 구현: 상태 직접 기록(순간이동) 후 안정화.

        캘리브레이션 포즈 순회/홈 복귀 전용(real=moveJ). sim에서는 경로 추종이
        검증 대상이 아니므로 teleport가 더 결정적이다. speed_scale은 무시(주석 계약).
        """
        if self._estopped:
            print("[sim] ⚠️ e-stop latch 상태 — move_to_joints 무시")
            return
        qt = self._q_tensor(q)
        zeros = torch.zeros_like(qt)
        self._robot.write_joint_state_to_sim(qt, zeros, joint_ids=self._arm_ids)
        self._robot.set_joint_position_target(qt, joint_ids=self._arm_ids)
        self._step_physics(5)  # 순간이동 직후 잔여 진동 안정화

    def set_gripper(self, closed: bool) -> None:
        """Record the accepted binary command; ``send_task_delta`` applies it (★F93, 리뷰 AX-01★).

        학습 순서(같은 스텝에 팔 항 → 그리퍼 항)를 지키려면 여기서 그리퍼 항을 바로 처리할 수 없다:
        그리퍼 항은 팔 항이 이번 스텝에 갱신한 수퍼바이저 상태(seq 가드)를 소비하고, 팔 항은 정책의
        그리퍼 부호를 액션 버퍼에서 읽는다. common.rollout은 set_gripper → send_task_delta 순으로
        부르므로 부호만 기록해 두고 send_task_delta가 버퍼 기록 → 팔 항 → 그리퍼 항 순으로 적용한다.
        """
        self._pending_gripper_closed = bool(closed)

    def is_ok(self) -> bool:
        """Sim has no protective stop, but the software e-stop remains latched."""
        return not self._estopped

    def emergency_stop(self) -> None:
        """요소5 e-stop hook: 현재 자세를 목표로 고정(사실상 정지) + latch.

        real 대응: servoStop(+stopScript 여부 TODO는 real 쪽 문서). sim 등가 =
        현재 관절각을 위치 목표로 재기록해 PD가 제자리를 유지하게 하고, 이후
        모든 명령을 무시한다(latch — SafetyGuard와 이중 안전).
        TODO: 물리 일시정지(sim.pause)까지 갈지는 리허설 UX 보고 결정.
        """
        if self._estopped:
            return
        self._estopped = True
        command_state = getattr(self._env, "_bp_gripper_command_state", None)
        if command_state is not None:
            command_state[:] = torch.nan
        hold = self._robot.data.joint_pos[:, self._arm_ids].clone()
        self._robot.set_joint_position_target(hold, joint_ids=self._arm_ids)
        print("[sim] ★EMERGENCY STOP★ 현재 자세 고정 + 명령 latch")

    def close(self) -> None:
        """자원 해제 — env 수명은 리허설 스크립트 소유라 여기서는 latch만 정리.

        ⚠️ env.close()/simulation_app.close()는 스크립트 쪽 책임
        (MEMORY: close() hang 이슈 → 검증된 종료 워크어라운드를 스크립트에서 적용).
        """
        print("[sim] SimRobot close (env 수명은 호출 스크립트 소유)")
