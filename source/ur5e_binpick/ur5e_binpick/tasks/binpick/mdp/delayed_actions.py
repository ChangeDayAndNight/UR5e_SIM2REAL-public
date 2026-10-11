"""Isaac Lab action terms with per-environment command latency.

The sampled lag is applied to the six-dimensional relative IK command (arm
channel).  The gripper channel is deliberately *not* delayed: the release
supervisor (F66/F68/F92) recomputes the physical gripper command every step
from the undelayed policy sign, mirroring the real control loop where the
supervisor runs inside the loop after the policy (F92 docstring fix — the
earlier claim that both channels are delayed was never true in code).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer

from common.gripper_command import (
    GRIPPER_COMMAND_TRANSITION_STEPS,
    advance_gripper_command_state_tensor,
)
from common.release_supervisor import (
    carry_latch_update,
    edge_lockout_update,
    grasp_allowed,
    load_release_supervisor_cfg,
    release_decision,
    settle_counter_update,
)

_RELEASE_CFG = load_release_supervisor_cfg()


def _release_state(env):
    """에피소드별 릴리스 수퍼바이저 상태를 env에 지연 생성한다.

    ★두 액션 항이 공유한다★ — 팔 항이 판정하고(먼저 처리됨) 그리퍼 항이 소비한다.
    상태를 env에 두는 이유: ActionTerm 인스턴스가 둘이라 어느 한쪽에 두면 다른 쪽이 못 본다.
    """
    st = getattr(env, "_bp_release_state", None)
    if st is None or st.get("closed_prev") is None or st["carried"].shape[0] != env.num_envs:
        st = {
            "closed_prev": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "z_at_close": torch.zeros(env.num_envs, dtype=torch.float32, device=env.device),
            "carried": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "released": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            # ★F92★ 연속 정지 창 카운터 / 마지막 그리퍼 전이 이후 스텝 수(리셋 = 잠금 길이 → 첫 전이 허용)
            "settle_count": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "since_edge": torch.full(
                (env.num_envs,), int(_RELEASE_CFG.gripper_edge_lockout_steps),
                dtype=torch.long, device=env.device,
            ),
            "prev_tcp": None,
            "seq": -1,          # 판정이 갱신된 스텝 — 그리퍼 항의 staleness 가드
        }
        env._bp_release_state = st
    return st


def _env_ids(env, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    if torch.is_tensor(env_ids):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


class _DelayedActionMixin:
    cfg: "DelayedDifferentialInverseKinematicsActionCfg | DelayedBinaryJointPositionActionCfg"

    def _init_delay(self, env) -> None:
        lo, hi = self.cfg.delay_steps
        if lo < 0 or hi < lo:
            raise ValueError(f"invalid action delay range: {self.cfg.delay_steps}")
        self._delay_env = env
        self._delay = DelayBuffer(hi, env.num_envs, device=env.device)

    def _reset_delay(self, env_ids) -> None:
        ids = _env_ids(self._delay_env, env_ids)
        if ids.numel() == 0:
            return
        lag_state = getattr(self._delay_env, "_bp_action_delay_lags", None)
        if lag_state is None or lag_state.shape[0] != self._delay_env.num_envs:
            lag_state = torch.zeros(
                self._delay_env.num_envs, dtype=torch.int, device=self._delay_env.device
            )
            self._delay_env._bp_action_delay_lags = lag_state
        if self.cfg.sample_lag:
            lo, hi = self.cfg.delay_steps
            lag_state[ids] = torch.randint(
                lo, hi + 1, (ids.numel(),), dtype=torch.int, device=self._delay_env.device
            )
        self._delay.set_time_lag(lag_state[ids], batch_ids=ids.tolist())
        self._delay.reset(batch_ids=ids.tolist())


class DelayedDifferentialInverseKinematicsAction(
    _DelayedActionMixin, DifferentialInverseKinematicsAction
):
    """지연 IK 액션 + ★TCP 하강 바닥(z floor)★.

    ═══ 하강 바닥이 필요한 이유 (F51, 2026-08-09 실측) ═══
    수직 하강 스윕(`Phase1/BinPick_PlacedProbe.py --mode descend_sweep`)에서 손가락 패드가
    ★TCP z = 0.0174 m★부터 벨트에 닿았고 반력이 ★150~240 N★까지 올라갔다. 그런데 5cm 큐브의
    안착 중심(= place 타겟)은 z = 0.0150 m이라, 큐브를 안착 높이까지 내리려면 패드가
    ★벨트를 2.4mm 파고들어야★ 한다. 실제로 그 충격이 손가락을 도로 닫아 ★놓은 큐브를
    재포획★하고 0.25 m/s로 튕겨내, placed의 '정지' 조건이 성립하지 않았다.

    실물 로봇도 테이블을 긁도록 두지 않는다 — 하강 하한은 실기에도 존재하는 제약이므로
    sim에만 있는 편법이 아니다(요소5 안전과 같은 계열). 여기서 IK 목표 z를 바닥 위로
    clamp하면 벨트 충돌 자체가 사라지고, 큐브는 몇 mm 위에서 놓여 자연 낙하·안착한다.

    ⚠️ 파지도 이 바닥의 영향을 받는다: 큐브는 z ∈ [-0.010, 0.040]에 걸쳐 있으므로 바닥
       0.020에서도 패드는 큐브 몸통(중심보다 5mm 위)을 문다 — 파지 가능 영역 안이다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_delay(env)
        self._applied_command_receipt_sequence = 0
        self._delayed_mapped_raw_command: torch.Tensor | None = None
        self._post_delay_pre_floor_command: torch.Tensor | None = None
        self._post_floor_applied_command: torch.Tensor | None = None

    @property
    def applied_command_receipt_sequence(self) -> int:
        """Monotonic successful ``process_actions`` receipt sequence."""
        return self._applied_command_receipt_sequence

    def get_applied_command_receipt(
        self,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return an immutable snapshot of the latest same-lineage commands.

        The first tensor is the original mapped command, the second the
        SafetyGuard output, and the third the exact processed command handed to
        DifferentialIK after the floor and any parent action clip.  The first
        two are concatenated before one :class:`DelayBuffer.compute` call, so
        every environment selects the same lagged command lineage for both.
        """
        if (
            self._delayed_mapped_raw_command is None
            or self._post_delay_pre_floor_command is None
            or self._post_floor_applied_command is None
        ):
            raise RuntimeError("no applied arm command receipt is available")
        return (
            self._applied_command_receipt_sequence,
            self._delayed_mapped_raw_command.detach().clone(),
            self._post_delay_pre_floor_command.detach().clone(),
            self._post_floor_applied_command.detach().clone(),
        )

    def process_actions(
        self,
        actions: torch.Tensor,
        *,
        mapped_raw_actions: torch.Tensor | None = None,
    ):
        # Invalidate first: a failed process call must never leave an older
        # receipt available to the deployment adapter.
        self._delayed_mapped_raw_command = None
        self._post_delay_pre_floor_command = None
        self._post_floor_applied_command = None
        expected_shape = (self.num_envs, self.action_dim)
        if mapped_raw_actions is None:
            # Training has no outer deployment SafetyGuard.  Duplicating the
            # received action supplies an identity outer stage without changing
            # the six-dimensional action passed to the parent action term.
            mapped_raw_actions = actions
        if (
            tuple(actions.shape) != expected_shape
            or tuple(mapped_raw_actions.shape) != expected_shape
            or not bool(torch.isfinite(actions).all().item())
            or not bool(torch.isfinite(mapped_raw_actions).all().item())
        ):
            raise ValueError(
                "arm action and mapped-raw evidence must be finite with shape "
                f"{expected_shape}; got {tuple(actions.shape)} and "
                f"{tuple(mapped_raw_actions.shape)}"
            )
        # One DelayBuffer call is the lineage boundary.  Separate buffers, or
        # two calls to one buffer, could select/update different commands when
        # action_delay_steps contains both 0 and 1.
        delayed_pair = self._delay.compute(
            torch.cat((mapped_raw_actions, actions), dim=-1)
        )
        expected_pair_shape = (self.num_envs, self.action_dim * 2)
        if tuple(delayed_pair.shape) != expected_pair_shape or not bool(
            torch.isfinite(delayed_pair).all().item()
        ):
            raise ValueError(
                "delayed raw/guarded action pair must be finite with shape "
                f"{expected_pair_shape}, got {tuple(delayed_pair.shape)}"
            )
        delayed_mapped_raw_actions = delayed_pair[:, : self.action_dim]
        actions = delayed_pair[:, self.action_dim :]
        delayed_mapped_raw = delayed_mapped_raw_actions * self._scale
        post_delay_pre_floor = actions * self._scale
        if not bool(torch.isfinite(delayed_mapped_raw).all().item()) \
                or not bool(torch.isfinite(post_delay_pre_floor).all().item()):
            raise ValueError("scaled delayed raw/guarded arm actions must be finite")
        # ★★릴리스 수퍼바이저 판정(2026-08-16, F65 이후)★★ — 여기서 ★한 번★ 계산하고
        #   그리퍼 항이 소비한다. 팔 항이 먼저 처리되는 것(ActionsCfg 선언 순서)에 의존하므로
        #   그리퍼 항은 seq로 staleness를 검사한다.
        #   [왜 여기인가] 판정에 TCP pose/속도가 필요한데 그것을 가진 것이 이 항이다.
        #   [왜 정책에 맡기지 않는가] 프로브 실측 — 정책이 스스로 개방 0.0% vs 수퍼바이저 67.6%
        #     (배포 등급 ≥10스텝). 부호 전용 binary 액션은 policy-gradient가 학습하지 못한다
        #     (F49·F52·F53·F56·F57·F64가 지급식 상수를 다섯 번 고쳐 전부 실패).
        st = _release_state(self._delay_env)
        ee_pos_now, _ = self._compute_frame_pose()
        tcp_pos = ee_pos_now[:, :3]
        if st["prev_tcp"] is None or st["prev_tcp"].shape != tcp_pos.shape:
            st["prev_tcp"] = tcp_pos.clone()
        tcp_speed = (tcp_pos - st["prev_tcp"]).norm(dim=1) / float(self._delay_env.step_dt)
        st["prev_tcp"] = tcp_pos.clone()
        # 그리퍼 '닫힘 명령'은 ★정책의 원 액션 부호★다(실기도 같은 신호를 쓴다 —
        # 2FG7에 폭/파지 feedback이 없어 실측 파지로는 판정할 수 없다, README §5).
        # ★★F67(기각·철회) 폐합 시점 탈취안★★ — 아래 F68이 현행이다. 이력으로만 남긴다.
        #   [당시 근거] "BinaryJointPositionAction은 한번 닫으면 유지되므로 조기 폐합은 되돌릴 수
        #     없다" → ★사실이 아니었다★(binary_joint_actions.py:131-142가 매 스텝 부호로 재계산).
        #   [대가] 그 전제로 '언제 닫을지'까지 수퍼바이저가 정하게 했더니 teacher 학습이 F66의
        #     20%로 망가졌다(grasp 0-250 구간 0.031 vs 0.797). 시점은 정책이 정해야 한다.
        # ★★F68-b: 제약은 ★전이만★ 막는다 — 강제 개방이 아니다★★
        #   [사고] F67에서 들여온 `closed_cmd = (carried|grasped) & ~released`는 제약이 아니라
        #     ★상태 지정★이었다. carried는 TCP z > lift_z(0.10)까지 들어올려야 걸리는데, 그 전에
        #     grasp_allowed(z<임계)를 벗어나면 grasped가 false가 되어 ★그리퍼가 강제로 열려
        #     들다가 큐브를 놓쳤다★. 실측: 임계를 0.080까지 올려 제약이 거의 안 걸리는데도
        #     배포등급(≥10스텝)이 ★모든 임계에서 0.00%★였다(F66 기준선 63.67%).
        #   [정정] 높이 제약은 ★open→close 전이★에만 건다. 이미 닫혀 있으면 정책이 닫기를
        #     명령하는 한 유지된다(정책이 열면 열리고, 다시 닫으려면 내려가야 한다).
        #     ⇒ teacher의 '내려가며 닫고 그대로 든다'가 온전히 보존되고,
        #        student의 '높은 곳에서 닫은 채 접근'만 구조적으로 불가능해진다.
        # ★★F83(2026-09-03) — 수퍼바이저 ★시딩 금지★. F81·F82를 전면 철회한다★★
        #   [사고 연쇄] 리뷰가 "pregrasp 큐브가 1스텝에 강제 개방된다"를 결함으로 지목했고
        #     (사실 서술로는 맞다), F81이 closed_prev를, F82가 carried까지 시딩해 '고쳤다'.
        #     두 run 모두 placed가 ★정확히 0★이 됐다(0.0001/0.0004 vs 계보 0.804).
        #   [실측 서명(두 run 동일)] time_out 98%·평균 99/100스텝·grasp/lift/goal 전부
        #     성공 계보보다 ★높음★ — 큐브를 든 채 배회하며 goal을 farming하는 균형.
        #     closed_prev 시딩이 may_close를 참으로 만들어 ★홀드가 가능해진 것★이 원인.
        #   [구 동작의 진실] 성공한 모든 teacher(F80 계보 placed 0.93)는 시딩이 없어서
        #     may_close=False(closed_prev=False ∧ 홈 z 0.23 > grasp_z_max 0.04)로 pregrasp
        #     큐브가 ★1스텝에 반드시 떨어졌다★. 그 결과 전 함대가 iter 0부터 '떨어진
        #     큐브를 실제로 집는' 연습을 했고(평균 에피소드 60스텝 = placed 종료),
        #     EMA 게이트가 일찍 열려 커리큘럼이 정상 감쇠했다. 즉 이 프로젝트의 pregrasp
        #     커리큘럼은 설계 의도('쥔 채 시작')가 아니라 ★'홈 아래 결정적 위치에 큐브
        #     낙하'★로 기능해 왔고, 그것이 실제로 학습을 성립시키는 동역학이다.
        #   [교훈] '우연히 작동하는 결함'을 고칠 때는 그 결함이 받치는 하중을 먼저 실측할
        #     것. 여기서는 강제 낙하가 커리큘럼 전체의 하중을 받치고 있었다. 시딩을
        #     되살리려면 반드시 hold-farming 균형의 해소(예: 호버 페널티/조기 종료)와
        #     함께 설계하고, 게이트 3000에서 placed·time_out·에피소드 길이를 대조할 것.
        policy_close = self._delay_env.action_manager.action[:, -1] < 0.0
        may_close = grasp_allowed(tcp_pos[:, 2], _RELEASE_CFG) | st["closed_prev"]
        closed_want = policy_close & may_close & (~st["released"])
        # ★★F92(2026-09-20) 그리퍼 전이 잠금 — 실기 DO-pulse(0.30 s = 6스텝) 중 반전은 e-stop이다★★
        #   공통 판정(release_supervisor.edge_lockout_update)으로 전이 뒤 6스텝 동안 반전을 무시한다.
        #   실기 rollout.py가 같은 함수를 같은 순서로 호출한다 — 한쪽만 바꾸면 계약이 깨진다.
        closed_now, st["since_edge"] = edge_lockout_update(
            st["closed_prev"], closed_want, st["since_edge"], _RELEASE_CFG
        )
        # ★★F71(2026-08-24) F68-c 배선 완성 — carry latch에 ★상대★ 상승량을 넘긴다★★
        #   [사고] `carry_latch_update`의 계약은 3번째 인자가 ★폐합 시점 대비 상승량★이고
        #     docstring이 "호출측이 폐합 전이에서 기준값을 기록해야 한다"고 명시하는데,
        #     여기서는 ★절대 TCP z★를 넘기고 있었다. `z_at_close`는 할당·리셋만 되고
        #     기록도 사용도 되지 않는 죽은 상태였다 = F68-c는 문서·설정·README에만 있고
        #     실제로는 구 의미(절대 높이)로 동작했다.
        #   [영향] lift_rise_m=0.010을 절대 높이로 읽으면, 폐합 허용 높이(<0.040)에서 닫는
        #     순간 z>0.010이 거의 항상 참이라 ★들어올리기 전에★ carried가 굳는다. 그러면
        #     release_decision(carried ∧ z<0.075 ∧ 속도<0.02)이 ★파지 직후 느려지는 순간★
        #     성립해 팔이 얼고 그리퍼가 열린다 — 집은 자리에 그대로 떨어뜨린다.
        #     teacher는 닫자마자 들어올려 속도가 유지되므로 대체로 가려지지만, 느리고
        #     불안정한 student에서 터진다. 실측 병목과 정확히 일치한다:
        #     파지 88.5% → 배달 래치 68.4%(−20.1%p), hold 28.8% vs teacher 49.7%.
        #   [정정] open→close ★전이★에서 그 시점 TCP z를 기록하고, 이후 상승량을 넘긴다.
        #     열리면 기준값을 무효화한다(다음 폐합에서 새로 기록).
        newly_closed = closed_now & (~st["closed_prev"])
        st["z_at_close"] = torch.where(newly_closed, tcp_pos[:, 2], st["z_at_close"])
        st["closed_prev"] = closed_now
        # 운반 latch는 ★수퍼바이저의 폐합 상태★로 판정한다(정책 명령이 아니다) —
        #   sim/real 모두 같은 상태기계를 돌려야 계약이 성립한다.
        st["carried"] = carry_latch_update(
            st["carried"], closed_now, tcp_pos[:, 2] - st["z_at_close"], _RELEASE_CFG
        )
        # ★F92★ 정지는 ★연속 창★으로 센다 — 단일 창(1 mm/50 ms)은 운반 중 방향 전환 한 번으로
        #   비가역 릴리스를 굳힌다(실기 servoL 추종 지연에서 실제로 일어나는 형태).
        st["settle_count"] = settle_counter_update(st["settle_count"], tcp_speed, _RELEASE_CFG)
        st["released"] = st["released"] | release_decision(
            st["carried"], tcp_pos[:, 2], st["settle_count"], _RELEASE_CFG
        )
        st["seq"] = int(self._delay_env.common_step_counter)
        # 릴리스한 env는 팔을 ★정지★시킨다(ik_rel이므로 delta 0 = 현재 자세 유지).
        #   실측: 개방만 하고 팔이 계속 움직이면 11.7%, 팔을 세우면 63.7% (배포 등급 ≥10스텝).
        #   움직이는 팔이 자기가 놓은 큐브를 쳐서 무너뜨린다(F63이 본 현상의 정량).
        if bool(st["released"].any()):
            actions = actions.clone()
            actions[st["released"]] = 0.0

        # ★★F78(2026-08-31) 수직 파지 제약 — 접근축을 연직으로 고정★★
        #   [왜] 큐브는 작업면에 놓이므로 상면 법선이 월드 +z다. 그 면에 ★수직으로★
        #     내려가야 두 손가락이 마주보는 면을 문다. 회전 3자유도를 열어 두면 정책이
        #     기울어진 자세로 접근해 모서리를 긁거나 한쪽 손가락만 닿는 파지를 배운다.
        #   [무엇을 고정하나] axis-angle 델타의 x·y 성분(=기울임)만 0으로 만든다.
        #     ★yaw(z축 회전)는 남긴다★ — 5cm 큐브의 대각선은 70.7mm인데 그리퍼 개구는
        #     74mm뿐이라 여유가 3mm다. yaw가 없으면 45° 돌아간 큐브를 물리적으로 못 집는다.
        #   [실측] 새 홈에서 TCP 접근축은 (+0.087, −0.052, −0.995) = 연직에서 5.8°다.
        #     기울임 델타를 막으면 그 자세가 에피소드 내내 유지된다.
        #   ⚠️ sim/real 일치: 실기 경로는 common/action_mapping.py의 ik_rel 분기가
        #     ★같은 성분★을 0으로 만든다. 한쪽만 고치면 정책 거동이 갈라진다.
        if getattr(self.cfg, "vertical_grasp_only", False):
            actions = actions.clone()
            actions[:, 3:5] = 0.0

        # ★★Teacher v2(2026-09-18) 실행 제한 정합 — 배포 SafetyGuard.clamp_task_delta와 같은 clamp★★
        #   실기는 스텝당 TCP 변위 노름을 tcp_linear_vel_max·dt, 회전을 tcp_angular_vel_max·dt로
        #   자른다(configs/safety.yaml). 학습이 이 제한 없이 큰 목표를 내면 정책은 실기에서
        #   본 적 없는 응답을 만난다(README §5: clamp 비율 ≥1%면 같은 clamp를 학습 경로에 넣는다).
        #   순서도 실기와 같다: 노름 clamp → z 바닥(아래 floor). raw 단위로 환산해 자른다.
        for sl, step_max in ((slice(0, 3), getattr(self.cfg, "tcp_step_max_m", None)),
                             (slice(3, 6), getattr(self.cfg, "tcp_rot_step_max_rad", None))):
            if step_max is None:
                continue
            actions = actions.clone()
            delta = actions[:, sl] * self._scale[:, sl]
            norm = delta.norm(dim=1, keepdim=True).clamp_min(1.0e-9)
            actions[:, sl] = actions[:, sl] * (float(step_max) / norm).clamp(max=1.0)

        floor = getattr(self.cfg, "tcp_z_floor_m", None)
        if floor is not None:
            # 현재 TCP 위치(root=base 프레임 — TABLE_HEIGHT와 같은 프레임)
            ee_pos_curr, _ = self._compute_frame_pose()
            # raw 액션은 _scale이 곱해진 뒤 delta가 되므로, 바닥을 raw 단위로 환산해 clamp.
            scale_z = self._scale[:, 2].clamp_min(1.0e-9)
            min_raw_dz = (floor - ee_pos_curr[:, 2]) / scale_z
            actions = actions.clone()
            actions[:, 2] = torch.maximum(actions[:, 2], min_raw_dz)
        super().process_actions(actions)
        post_floor_applied = self.processed_actions
        if tuple(post_floor_applied.shape) != expected_shape or not bool(
            torch.isfinite(post_floor_applied).all().item()
        ):
            raise ValueError("processed arm action must be a finite 6D batch")
        self._delayed_mapped_raw_command = delayed_mapped_raw.detach().clone()
        self._post_delay_pre_floor_command = post_delay_pre_floor.detach().clone()
        self._post_floor_applied_command = post_floor_applied.detach().clone()
        self._applied_command_receipt_sequence += 1

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._reset_delay(env_ids)
        # ★릴리스 수퍼바이저 상태 해제★ — 안 하면 다음 에피소드가 '이미 놓은' 상태로
        #   시작해 팔이 영구 정지하고 그리퍼가 계속 열린 채가 된다(에피소드 전량 실패).
        st = _release_state(self._delay_env)
        ids = _env_ids(self._delay_env, env_ids)
        st["closed_prev"][ids] = False
        st["z_at_close"][ids] = 0.0
        st["carried"][ids] = False
        st["released"][ids] = False
        st["settle_count"][ids] = 0
        st["since_edge"][ids] = int(_RELEASE_CFG.gripper_edge_lockout_steps)
        if st["prev_tcp"] is not None:
            st["prev_tcp"][ids] = 0.0
        # A reset separates command lineages; no pre-reset receipt is valid.
        self._delayed_mapped_raw_command = None
        self._post_delay_pre_floor_command = None
        self._post_floor_applied_command = None


class DelayedBinaryJointPositionAction(_DelayedActionMixin, BinaryJointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_delay(env)
        if int(self.cfg.command_transition_steps) != GRIPPER_COMMAND_TRANSITION_STEPS:
            raise ValueError("gripper command transition must be exactly 24 policy steps")
        state = getattr(env, "_bp_gripper_command_state", None)
        if state is not None and tuple(state.shape) != (env.num_envs,):
            raise ValueError("existing gripper command state has an invalid shape")
        if state is None:
            state = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
            env._bp_gripper_command_state = state
        self._gripper_command_state = state

    def process_actions(self, actions: torch.Tensor):
        # Advance from the command *after* DelayBuffer, once per policy step.
        # process_actions is the one-shot acceptance point; apply_actions runs
        # at physics decimation and must never advance this observation.
        delayed = self._delay.compute(actions)
        # ★★릴리스 수퍼바이저 적용★★ — 팔 항이 이번 스텝에 판정한 결과를 소비한다.
        #   ⚠️지연 ★뒤에★ 덮어쓴다: 수퍼바이저는 실기에서 제어루프 안에서 도는 primitive라
        #     정책 액션 지연 DR의 대상이 아니다. 지연 앞에 두면 sim만 늦게 열린다.
        st = _release_state(self._delay_env)
        if st["seq"] != int(self._delay_env.common_step_counter):
            # 팔 항이 먼저 처리된다는 전제(ActionsCfg 선언 순서)가 깨진 것이다.
            # 조용히 지나가면 릴리스가 한 스텝 늦거나 영영 발화하지 않는다 → fail-closed.
            raise RuntimeError(
                "릴리스 판정이 이번 스텝에 갱신되지 않았다 — ActionsCfg에서 arm_action이 "
                "gripper_action보다 먼저 선언되어야 한다"
            )
        # ★F67★ 그리퍼 명령은 ★전적으로 수퍼바이저★가 만든다 — 정책 출력(delayed)은 무시한다.
        #   closed = 폐합 판정됨 ∧ 아직 릴리스 전.  그 외에는 개방.
        # 팔 항이 이번 스텝에 확정한 폐합 상태를 그대로 쓴다(제약이 반영된 정책 명령).
        closed_cmd = st["closed_prev"]
        delayed = torch.where(
            closed_cmd.reshape(-1, 1).expand_as(delayed),
            torch.full_like(delayed, -1.0),
            torch.full_like(delayed, 1.0),
        )
        super().process_actions(delayed)
        closed = delayed == 0 if delayed.dtype == torch.bool else delayed < 0
        closed = closed.reshape(self.num_envs)
        self._gripper_command_state[:] = advance_gripper_command_state_tensor(
            self._gripper_command_state, closed
        )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._reset_delay(env_ids)
        # Sim reset has a known explicit open state.  Teacher privileged
        # finger-q observations remain separate and are not modified here.
        self._gripper_command_state[_env_ids(self._delay_env, env_ids)] = 0.0


@configclass
class DelayedDifferentialInverseKinematicsActionCfg(
    mdp.DifferentialInverseKinematicsActionCfg
):
    class_type: type[ActionTerm] = DelayedDifferentialInverseKinematicsAction
    delay_steps: tuple[int, int] = (0, 0)
    sample_lag: bool = True
    # ★TCP z 하한 [m, base frame]★ — None이면 비활성. 근거·실측은 위 클래스 docstring.
    tcp_z_floor_m: float | None = None
    # ★F78 수직 파지★ — True면 axis-angle 델타의 x·y(기울임)를 0으로 고정하고 yaw만 남긴다.
    #   실기 등가물: common/action_mapping.py ik_rel 분기(configs/common.yaml action 절).
    vertical_grasp_only: bool = False
    # ★Teacher v2★ 스텝당 TCP 변위·회전 상한 [m, rad] — None이면 비활성. 실기 등가물:
    #   common/safety.py SafetyGuard.clamp_task_delta(configs/safety.yaml limits × dt).
    tcp_step_max_m: float | None = None
    tcp_rot_step_max_rad: float | None = None


@configclass
class DelayedBinaryJointPositionActionCfg(mdp.BinaryJointPositionActionCfg):
    class_type: type[ActionTerm] = DelayedBinaryJointPositionAction
    delay_steps: tuple[int, int] = (0, 0)
    sample_lag: bool = False
    command_transition_steps: int = GRIPPER_COMMAND_TRANSITION_STEPS
