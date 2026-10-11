# ★배포 제어 루프의 유일한 구현★ — run_rollout().
# Phase3 SimDeploy_Rehearsal(sim 백엔드)과 Phase4 Deploy_DryRun/Deploy_Real(실기)이
# 전부 이 함수를 호출한다. "리허설에서 돌린 코드 경로 = 실기 코드 경로"가 이 파일의
# 존재 이유다 — 루프를 복사해 변형하는 순간 sim2real 검증이 무효가 된다.
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from common.action_mapping import ActionMapper
from common.interfaces import CameraInterface, RobotInterface
from common.obs_preprocess import build_policy_obs_from_state
from common.policy_io import PolicyRunner
from common.safety import SafetyGuard
from common.timing import RateLimiter
from common.types import (
    ARM_CLAMP_COMPARISON_ATOL,
    ARM_COMMAND_CHANNELS,
    AppliedArmCommandReceipt,
    PreprocessCfg,
    RolloutResult,
    RolloutSafetyStats,
)


def run_rollout(
    robot: RobotInterface,
    camera: CameraInterface,
    policy: PolicyRunner,
    mapper: ActionMapper,
    guard: SafetyGuard,
    rate: RateLimiter,
    pre_cfg: PreprocessCfg,
    target_pos: np.ndarray,
    max_steps: int,
    dry_run: bool = False,
    obs_dump_dir: Path | str | None = None,
    require_force_feedback: bool = False,
    max_sensor_age_s: float | None = None,
    max_sensor_skew_s: float | None = None,
    max_consecutive_overruns: int | None = None,
    termination_check=None,
    obs_builder=None,
) -> RolloutResult:
    """1 에피소드 실행. 루프 순서(고정):

      state/frame 취득 → is_ok 체크 → 공유 전처리(build_policy_obs) → 정책 추론
      → ActionMapper → SafetyGuard.clamp/check → rate.sleep → (dry_run 아니면)
      set_gripper + send_task_delta(ik_rel) 또는 send_joint_target(legacy) → dump

    dry_run=True: 명령 전송만 생략 — 취득/전처리/추론/안전판정은 전부 실행
                  (Deploy_DryRun이 실기 지터·분포를 측정하는 모드).
    obs_dump_dir: 스텝별 전처리 후 관측(.npz) 저장 → Verify_ObsParity 입력.
    obs_builder: ★모듈형 배포 v1(2026-09-15)★ — 관측 조립기 교체 훅. None이면 student 공유
                 전처리(build_policy_obs_from_state)를 쓴다. 주어지면
                 obs_builder(frame, state, target_pos, last_action) → 정책 입력 텐서이고,
                 obs_builder.hold_requested(있으면)가 정책의 hold_requested와 OR로 합쳐진다
                 (common.teacher_obs.TeacherObsBuilder: 인식 모듈 + Teacher v2 34D 조립).
                 루프의 나머지(안전·수퍼바이저·명령 전송·덤프)는 한 줄도 갈라지지 않는다.
    """
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    last_action = np.zeros(mapper.cfg.action_dim, dtype=np.float64)
    dump_dir: Path | None = None
    if obs_dump_dir is not None:
        dump_dir = Path(obs_dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

    steps_done = 0
    terminated = False
    success = False
    arm_clamp_counts = np.zeros(ARM_COMMAND_CHANNELS, dtype=np.int64)
    # ★★F81 — 그리퍼 수퍼바이저를 실기 경로에 이식 (리뷰 blocker #3)★★
    #   [사고] 정책의 그리퍼 채널은 사실상 '항상 닫아라' 포화 신호다(실측 중앙값 −1.6~−2.4,
    #     포화율 87~100%). sim에서는 액션텀(delayed_actions)의 수퍼바이저가 이 신호를
    #     걸러 주지만(폐합은 TCP가 낮을 때만·운반 latch·릴리스 후 팔 정지), 이 루프에는
    #     등가물이 ★없어서★ 실기 로봇이 큐브 근처에 가기도 전에 그리퍼를 닫았다(실기
    #     실증 2026-09-02). sim=real 제1원칙(§4-3) 위반이었다.
    #   [정정] delayed_actions와 ★같은 함수·같은 설정★(common/release_supervisor.py,
    #     configs/common.yaml release_supervisor)으로 같은 상태기계를 돌린다. 판정 입력은
    #     TCP z·TCP 속도·명령 이력뿐 — 물체 pose가 필요 없어 실기에서 그대로 성립한다.
    #   ⚠️Phase3 sim 리허설은 env 내부 액션텀도 같은 기계를 돌리므로 이중 적용되지만,
    #     같은 입력에 같은 결정(결정적 함수)이라 거동이 갈라지지 않는다.
    from common.release_supervisor import (
        carry_latch_update,
        edge_lockout_update,
        grasp_allowed,
        load_release_supervisor_cfg,
        release_decision,
        settle_counter_update,
    )

    rs_cfg = load_release_supervisor_cfg()
    rs_closed_prev = False
    rs_z_at_close = 0.0
    rs_carried = False
    rs_released = False
    rs_settle = 0                                   # ★F92★ 연속 정지 창 수
    rs_since_edge = int(rs_cfg.gripper_edge_lockout_steps)  # ★F92★ 리셋 = 잠금 길이(첫 전이 허용)
    rs_prev_tcp: np.ndarray | None = None
    was_policy_hold = False
    issued_command_steps = 0
    last_applied_receipt_sequence: int | None = None
    last_sequence_id: int | None = None
    for step in range(max_steps):
        applied_receipt_sequence = -1
        try:
            state = robot.get_state()
            frame = camera.get_frame()
        except BaseException as exc:
            guard.trip(f"sensor acquisition failed: {type(exc).__name__}: {exc}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            break

        try:
            robot_ok = robot.is_ok()
        except BaseException as exc:
            guard.trip(f"robot health check failed: {type(exc).__name__}: {exc}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            break
        if not robot_ok:
            guard.trip("robot not ok (protective stop / 통신 이상)")
            break

        now = time.perf_counter()
        if max_sensor_age_s is not None:
            state_age = now - float(state.timestamp)
            frame_age = now - float(frame.timestamp)
            if state_age < 0 or frame_age < 0:
                guard.trip("sensor timestamp is in the future or uses a different clock")
                break
            if state_age > max_sensor_age_s or frame_age > max_sensor_age_s:
                guard.trip(
                    f"stale sensor sample: state={state_age*1000:.1f}ms, "
                    f"frame={frame_age*1000:.1f}ms"
                )
                break
        if max_sensor_skew_s is not None:
            skew = abs(float(state.timestamp) - float(frame.timestamp))
            if skew > max_sensor_skew_s:
                guard.trip(f"robot/camera timestamp skew {skew*1000:.1f}ms")
                break
        if frame.sequence_id is not None:
            if last_sequence_id is not None and frame.sequence_id <= last_sequence_id:
                guard.trip(
                    f"stale/duplicate camera sequence: {frame.sequence_id} <= {last_sequence_id}"
                )
                break
            last_sequence_id = frame.sequence_id
        if require_force_feedback and state.tcp_force is None:
            guard.trip("TCP force feedback unavailable")
            break

        # ★★2026-08-12(감사 B2) 비유한값 fail-closed★★ — NaN/Inf는 SafetyGuard의 모든
        #   관문을 통과한다(NaN 비교는 전부 False → clamp 무변화·AABB ok=True, 실행 재현으로
        #   확인). 소프트웨어 안전층이 비유한값에 fail-open이었으므로 ★루프 입구에서 잡는다★.
        if state.gripper_command_state is None:
            guard.trip(
                "gripper command state unknown/invalid — explicit open settling or "
                "fault recovery is required"
            )
            break
        _vecs = [
            state.joint_pos,
            state.joint_vel,
            state.tcp_pose,
            [state.gripper_command_state],
        ]
        if state.tcp_force is not None:
            _vecs.append(state.tcp_force)
        if not all(np.isfinite(np.asarray(v, dtype=np.float64)).all() for v in _vecs):
            guard.trip("non-finite state (NaN/Inf) — 센서/드라이버 이상")
            break

        try:
            if obs_builder is None:
                obs = build_policy_obs_from_state(
                    frame, state, target_pos, last_action, pre_cfg
                )  # ★요소1: sim 학습과 동일 함수★
            else:
                obs = obs_builder(frame, state, target_pos, last_action)
            action = policy.act(obs)[0]
            policy_hold = bool(getattr(policy, "hold_requested", False)) or                 bool(getattr(obs_builder, "hold_requested", False))
        except BaseException as exc:
            guard.trip(f"preprocess/policy failed: {type(exc).__name__}: {exc}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            break
        # ★감사 B2★ 정책 출력도 동일 — NaN 관측/가중치 이상이 servoL 인자까지 도달하는 것을 차단.
        if not np.isfinite(np.asarray(action, dtype=np.float64)).all():
            guard.trip("non-finite action (NaN/Inf) — 정책 출력 이상")
            break

        try:
            arm_command, gripper_closed = mapper.map(action, state.joint_pos)

            # ★F81 수퍼바이저 판정 — delayed_actions의 상태기계를 그대로 미러★
            #   순서 계약: 폐합 게이트 → 폐합 기준높이 기록 → carry latch → 릴리스 판정
            #   → 릴리스 시 팔 정지 + 강제 개방. (한쪽만 바꾸면 sim/real이 갈라진다.)
            tcp_z = float(np.asarray(state.tcp_pose, dtype=np.float64).reshape(-1)[2])
            tcp_now = np.asarray(state.tcp_pose, dtype=np.float64).reshape(-1)[:3].copy()
            tcp_speed = (
                float(np.linalg.norm(tcp_now - rs_prev_tcp) / max(rate.period_s, 1e-6))
                if rs_prev_tcp is not None else 0.0
            )
            rs_prev_tcp = tcp_now
            if not policy_hold:
                may_close = bool(grasp_allowed(tcp_z, rs_cfg)) or rs_closed_prev
                closed_want = bool(gripper_closed) and may_close and not rs_released
                # ★F92★ 전이 잠금 — sim delayed_actions와 같은 함수·같은 자리(폐합 게이트 직후).
                closed_now, rs_since_edge = edge_lockout_update(
                    rs_closed_prev, closed_want, rs_since_edge, rs_cfg
                )
                closed_now = bool(closed_now)
                rs_since_edge = int(rs_since_edge)
                if closed_now and not rs_closed_prev:
                    rs_z_at_close = tcp_z
                rs_carried = bool(carry_latch_update(
                    rs_carried, closed_now, tcp_z - rs_z_at_close, rs_cfg
                ))
                # ★F92★ 연속 정지 창 — sim과 같은 카운터. 홀드 창은 세지 않는다(아래 else).
                rs_settle = int(settle_counter_update(rs_settle, tcp_speed, rs_cfg))
                # A detector-induced stop is not evidence of completed placement.
                if not was_policy_hold:
                    rs_released = rs_released or bool(release_decision(
                        rs_carried, tcp_z, rs_settle, rs_cfg
                    ))
                rs_closed_prev = closed_now
            else:
                rs_settle = 0   # 검출기 정지 중의 '정지'는 배치 완료 증거가 아니다
            # ★F93(2026-09-20, 리뷰 AX-07)★ 물리 그리퍼 명령은 ★수퍼바이저의 폐합 상태★ 그대로다.
            #   sim(delayed_actions)은 릴리스 판정 스텝에 closed_prev(=아직 닫힘)를 명령하고 ★다음★
            #   스텝에 closed_want(¬released)로 연다. 종전 `and not rs_released`는 실기만 한 스텝 먼저
            #   열어 sim/real 타이밍이 갈렸다. 정책은 sim 타이밍으로 학습됐으므로 실기를 sim에 맞춘다.
            gripper_closed = rs_closed_prev
            if rs_released or policy_hold:
                # 릴리스 후 팔 정지(delta 0) — 실측: 팔을 세우면 63.7%, 안 세우면 11.7%.
                arm_command = np.zeros_like(np.asarray(arm_command, dtype=np.float64))
                if mapper.cfg.mode != "ik_rel":
                    arm_command = np.asarray(state.joint_pos, dtype=np.float64).copy()
            if policy_hold:
                # Feedback records the hold, not the policy wrapper's placeholder.
                action = np.zeros(mapper.cfg.action_dim, dtype=np.float64)
                action[-1] = -1.0 if gripper_closed else 1.0
            was_policy_hold = policy_hold

            raw_arm_command = np.asarray(arm_command, dtype=np.float64).copy()
            if raw_arm_command.shape != (ARM_COMMAND_CHANNELS,) \
                    or not np.isfinite(raw_arm_command).all():
                raise ValueError("mapped arm command must be a finite 6D vector")
            if mapper.cfg.mode == "ik_rel":
                arm_command = guard.clamp_task_delta(
                    arm_command, rate.period_s, current_tcp_pose=state.tcp_pose
                )  # ★감사 B1★ z-floor 클램프에 현재 TCP 전달
            else:
                arm_command = guard.clamp_joint_target(
                    state.joint_pos, arm_command, rate.period_s
                )
            safety_arm_command = np.asarray(arm_command, dtype=np.float64).copy()
            verdict = guard.check(state)
            if verdict.ok and mapper.cfg.mode == "ik_rel":
                verdict = guard.check_task_target(state.tcp_pose, arm_command)
        except BaseException as exc:
            guard.trip(f"action/safety mapping failed: {type(exc).__name__}: {exc}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            break
        if not verdict.ok:
            guard.trip(verdict.reason)
            break

        # Gate immediately before command transmission.  Sleeping at the end
        # of the previous iteration permits a short-processing iteration to
        # emit a command immediately after an overrun.  This placement makes
        # every command pass the limiter and prevents catch-up bursts.
        try:
            rate.sleep()
        except BaseException as exc:
            guard.trip(f"rate limiter failed: {type(exc).__name__}: {exc}")
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            break
        if (
            max_consecutive_overruns is not None
            and rate.consecutive_overruns > max_consecutive_overruns
        ):
            guard.trip(
                f"consecutive control overruns {rate.consecutive_overruns} > "
                f"{max_consecutive_overruns}"
            )
            break

        if not dry_run:
            try:
                # ★F93(리뷰 P2)★ hold 중에도 그리퍼 명령을 보낸다 — 어댑터는 edge-trigger라 물리 펄스는
                #   같고, 24스텝 command-state 추적기만 학습과 똑같이 매 스텝 전진한다(정지 중 관측 동결 금지).
                robot.set_gripper(gripper_closed)
                if mapper.cfg.mode == "ik_rel":
                    receipt = robot.send_task_delta(
                        arm_command,
                        state.tcp_pose,
                        duration_s=rate.period_s,
                        mapped_raw_delta=raw_arm_command,
                    )
                    if not isinstance(receipt, AppliedArmCommandReceipt):
                        raise RuntimeError(
                            "arm adapter did not return an applied-command receipt"
                        )
                    if (
                        last_applied_receipt_sequence is not None
                        and receipt.sequence <= last_applied_receipt_sequence
                    ):
                        raise RuntimeError(
                            "arm adapter returned a stale/non-monotonic receipt"
                        )
                    raw_arm_command = np.asarray(
                        receipt.delayed_mapped_raw, dtype=np.float64
                    ).copy()
                    safety_arm_command = np.asarray(
                        receipt.post_delay_pre_floor, dtype=np.float64
                    ).copy()
                    arm_command = np.asarray(receipt.applied, dtype=np.float64).copy()
                    if raw_arm_command.shape != (ARM_COMMAND_CHANNELS,) \
                            or safety_arm_command.shape != (ARM_COMMAND_CHANNELS,) \
                            or arm_command.shape != (ARM_COMMAND_CHANNELS,) \
                            or not np.isfinite(raw_arm_command).all() \
                            or not np.isfinite(safety_arm_command).all() \
                            or not np.isfinite(arm_command).all():
                        raise RuntimeError(
                            "applied-command receipt must contain three finite 6D vectors"
                        )
                    last_applied_receipt_sequence = receipt.sequence
                    applied_receipt_sequence = receipt.sequence
                else:
                    robot.send_joint_target(arm_command, duration_s=rate.period_s)
                # Count only after the arm command has completed successfully.
                # Receive-only dry runs and transmission failures therefore
                # cannot contribute to deployable clamp evidence.  For ik_rel,
                # all three vectors now come from the same delay-selected
                # action lineage.  The union prevents two clamp stages that
                # numerically cancel/reverse from hiding an intervention.
                issued_command_steps += 1
                outer_clamped = np.greater(
                    np.abs(raw_arm_command - safety_arm_command),
                    ARM_CLAMP_COMPARISON_ATOL,
                )
                inner_clamped = np.greater(
                    np.abs(safety_arm_command - np.asarray(arm_command, dtype=np.float64)),
                    ARM_CLAMP_COMPARISON_ATOL,
                )
                arm_clamp_counts += np.logical_or(
                    outer_clamped, inner_clamped
                ).astype(np.int64)
            except BaseException as exc:
                guard.trip(f"command transmission failed: {type(exc).__name__}: {exc}")
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                break

        if dump_dir is not None:
            try:
                np.savez(
                    dump_dir / f"step_{step:04d}.npz",
                    obs=obs.cpu().numpy(),
                    action=action,
                    raw_arm_command=raw_arm_command,
                    safety_arm_command=safety_arm_command,
                    arm_command=arm_command,
                    q_target=arm_command,  # backward-compatible dump key
                    command_issued=np.asarray(not dry_run, dtype=np.bool_),
                    policy_hold=np.asarray(policy_hold, dtype=np.bool_),
                    supervisor_carried=np.asarray(rs_carried, dtype=np.bool_),
                    supervisor_released=np.asarray(rs_released, dtype=np.bool_),
                    gripper_closed_command=np.asarray(gripper_closed, dtype=np.bool_),
                    gripper_command_issued=np.asarray(not dry_run, dtype=np.bool_),
                    arm_command_receipt_sequence=np.asarray(
                        applied_receipt_sequence, dtype=np.int64
                    ),
                    timestamp=time.time(),
                    state_timestamp=state.timestamp,
                    frame_timestamp=frame.timestamp,
                    frame_sequence=(-1 if frame.sequence_id is None else frame.sequence_id),
                )
            except BaseException as exc:
                guard.trip(f"observation dump failed: {type(exc).__name__}: {exc}")
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                break

        last_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        steps_done = step + 1
        if termination_check is not None:
            try:
                outcome = termination_check()
            except BaseException as exc:
                guard.trip(f"termination check failed: {type(exc).__name__}: {exc}")
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                break
            if outcome is not None:
                terminated = True
                success = bool(outcome)
                break
    print(
        f"[rollout] 종료: steps={steps_done}, tripped={guard.tripped}"
        + (f" ({guard.trip_reason})" if guard.tripped else "")
        + f", overruns={rate.overruns}"
    )
    return RolloutResult(
        steps=steps_done,
        tripped=guard.tripped,
        trip_reason=guard.trip_reason,
        overruns=rate.overruns,
        terminated=terminated,
        success=success,
        rollout_safety=RolloutSafetyStats(
            denominator=issued_command_steps,
            arm_clamp_counts=tuple(int(value) for value in arm_clamp_counts),
        ),
    )
