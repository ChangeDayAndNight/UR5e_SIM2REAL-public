"""BinPick env cfg — BinPickTeacherEnvCfg(Phase1) / BinPickVisionEnvCfg(Phase2).

목적: ManagerBasedRLEnvCfg 스켈레톤 2종.
      teacher = 특권 상태 관측(양 그룹 동일), 카메라 없음, num_envs 4096.
      vision  = 비대칭(policy=단일 flat ObsTerm(요소1) / critic=특권), TiledCamera,
                num_envs 128. rsl_rl 3.1.2 StudentTeacher/DistillationRunner 대상.
실행: 직접 실행 없음 — 학습 스크립트가 gym.register(
        id="Isaac-BinPick-UR5e-Teacher-v0" / "Isaac-BinPick-UR5e-Vision-v0",
        kwargs={"env_cfg_entry_point": ...}) 로 등록해 사용(등록은 scripts 담당).
산출: BinPickTeacherEnvCfg, BinPickVisionEnvCfg (+Eval 변형, BinPickVisionNuRecEnvCfg).

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것.
★그리퍼 배선 완료(OnRobot 2FG7, 2026-07-06) → action_dim=7 (팔 6 + 그리퍼 1) = 계약 일치.
   vision env의 policy_obs(last_action 7차원)도 이제 정상. Phase1 teacher/Phase2 vision
   모두 그리퍼 포함 학습 가능. 결합 자산 = assets/ur5e_2fg7 (convert 후 사용).
"""

from __future__ import annotations

import math
import os

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand  # noqa: E402 (커스텀 서브클래스용)
from isaaclab.utils.math import subtract_frame_transforms

from common.calibration import load_tcp_offset, mat_to_quat
from common.config_io import configs_dir, load_yaml
from common.gripper_command import validate_gripper_command_config
from common.release_gate import grant_and_consume, open_gain, open_state_ratio
from common.timing import derive_decimation

from .action_scheme import SIM_ACTION_SCHEME  # noqa: F401 (obs_spec 계약 단일 소스)
from .mdp import delayed_actions as bp_actions
from .mdp import events_dr as dr
from .mdp import observations as bp_obs
from .mdp import perception_model as bp_perc
from .mdp.scene import (
    NUM_OBJECTS,
    OBJECT_SIZE,
    TABLE_HEIGHT,
    BinPickSceneCfg,
    add_scatter_objects,
    make_d435i_body_cfg,
    make_finger_contact_sensor_cfg,
    make_finger_contact_sensor_cfgs,
    make_nurec_background_cfg,
    make_nurec_belt_visual_cfg,
    make_tiled_camera_cfg,
    object_names,
)
from .mdp.ur5e_assets import (
    GRIPPER_CLOSE_COMMAND,
    GRIPPER_CLOSE_M,  # release_bias_bootstrap의 실제 개도 정규화 분모(open↔close 행정)
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND,
    GRIPPER_OPEN_M,
    UR5E_ARM_JOINT_NAMES,
)

# ─────────────────────────────────────────────────────────────────────────────
# yaml 단일 소스 (import 시 1회)
# ─────────────────────────────────────────────────────────────────────────────
_COMMON = load_yaml(configs_dir() / "common.yaml")
_DR_YAML = load_yaml(configs_dir() / "dr.yaml")
_SIM = load_yaml(configs_dir() / "sim.yaml")
_ACTION_DELAY_STEPS = tuple(int(v) for v in _DR_YAML["latency"]["action_delay_steps"])

_PHYSICS_HZ = float(_COMMON["control"]["physics_hz"])
_CONTROL_HZ = float(_COMMON["control"]["control_hz"])
_GRIPPER_COMMAND_TRANSITION_STEPS = validate_gripper_command_config(
    _CONTROL_HZ, float(_COMMON["gripper_command_state"]["transition_s"])
)
# ★task-space IK(ik_rel) 액션 상수★ — 정책 출력 = TCP pose delta [dx,dy,dz,dR,dP,dY]×scale.
#   Franka ik_rel 표준 scale=0.5. body_offset = tcp_offset.yaml(파지중심, 요소4 단일 소스) →
#   IK가 '그리퍼 파지중심'을 목표로 관절을 푼다(wrist_3_link가 아니라 TCP 제어 — reach/파지 정합).
_IK_SCALE = (float(_COMMON["action"]["task_pos_scale_m"]),) * 3 + (
    float(_COMMON["action"]["task_rot_scale_rad"]),
) * 3
# (2026-07-15 자세만 0.12 축소 완화책 시도 → grasp 미개선으로 원복. 원인은 자세 흔들림이 아님.)
_TCP_OFFSET_MAT = load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")
_TCP_OFFSET_POS = [float(v) for v in _TCP_OFFSET_MAT[:3, 3]]
_TCP_OFFSET_QUAT = [float(v) for v in mat_to_quat(_TCP_OFFSET_MAT[:3, :3])]  # wxyz
_TARGET_RANGE = _COMMON["target"]["sample_range"]
_TEACHER_NUM_ENVS = int(_SIM["training"]["teacher_num_envs"])
_STUDENT_NUM_ENVS = int(_SIM["training"]["student_num_envs"])
# ★Teacher v2★ 배포 SafetyGuard 한계(configs/safety.yaml) — sim 액션텀이 같은 스텝당 상한으로 clamp한다.
_SAFETY_LIMITS = load_yaml(configs_dir() / "safety.yaml")["limits"]

# 에피소드 길이 [s] — 20Hz 기준 160 제어 스텝. TODO: pick→place 소요 실측 후 조정.
# ★2026-08-05(P1-b ③) 8.0 → 5.0★: 정제 구간의 병목은 ★파지 시도 횟수★인데 시도는
#   에피소드당 1회다. 실측상 태스크는 ~25-40스텝에 끝나고 나머지는 안착 상태로 앉아 있다
#   (필름스트립: apex frame 23 → place frame 25). 100스텝이면 충분하고,
#   같은 계산량에서 ★에피소드(=파지 시도)가 1.6배★가 된다.
# ★★Teacher v2(2026-09-18) 5.0 → 10.0★★: 실행 제한 정합으로 TCP 최대 0.25 m/s가 되어
#   접근(~0.35 m, 1.4 s)+하강·폐합(1.7 s)+들기·운반(~0.3 m, 1.4 s)+하강·릴리스(1.7 s) ≈ 6 s.
#   5 s면 정상 경로가 시간 초과로 잘린다. 성공 종료(placed_success)가 있어 안착 후 낭비는 없다.
EPISODE_LENGTH_S = 10.0


# ═════════════════════════════════════════════════════════════════════════════
# Commands — ★가변 place 목표(사용자 결정 2026-07-04)★
#   결정: command manager(UniformPoseCommandCfg) 채택.
#   근거(vendored 검증): command_manager.reset()이 리셋 시 _resample을 호출
#   (command_manager.py L120-125, L334-339)하므로 "에피소드마다 균일 샘플"이
#   자동으로 성립하고, resampling_time_range를 에피소드보다 길게 잡으면 에피소드
#   중간 재샘플이 없다. event-term 방식(버퍼 수동 관리)보다 관측 배선이 단순
#   (mdp.generated_commands / observations.target_pos_command).
# ═════════════════════════════════════════════════════════════════════════════
# ★b5 목표거리 역커리큘럼 command (2026-07-23) — placed 직격 카드★
class TargetCurriculumPoseCommand(UniformPoseCommand):
    """★목표거리 역커리큘럼★ — BINPICK_TARGET_CURR 시 목표를 최근접 물체 근처 반경
    (env._bp_target_radius, curriculum이 warmup 후 확대)에서 샘플한다. 파지→운반 거리를
    초기에 짧게(carry→placed 전환을 자주 경험) 두고 점차 늘려 원거리 운반으로 전이
    (Florensa 2017 Reverse Curriculum / HER). b4 진단(파지 성공↑이나 placed 미개선,
    carry→placed 전환율 50→14%)을 직격한다.

    ★farming 회피(bootstrap-farming-dilemma — place bootstrap 5회 실패)★: pregrasp env의
    placed/place_fine/release_bias는 _real_pick_mask가 차단하므로 '잡힌 채+목표 근처=무료
    placed' 함정이 재현되지 않는다(자력 파지 env만 목표 커리큘럼 이득). 게이트
    (_bp_target_curr_on) off면 부모 UniformPoseCommand(절대 균일 샘플)와 완전히 동일."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)  # z·quat 설정 + 게이트 off 시 최종값(절대 균일)
        if not getattr(self._env, "_bp_target_curr_on", False):
            return
        if isinstance(env_ids, slice):
            return
        eid = (env_ids if torch.is_tensor(env_ids)
               else torch.tensor(list(env_ids), device=self.device))
        if eid.numel() == 0:
            return
        # 단일 큐브(object_0) 위치(world) → robot base frame (목표는 base frame 계약)
        obj_w = self._env.scene["object_0"].data.root_pos_w[eid]
        obj_b, _ = subtract_frame_transforms(
            self.robot.data.root_pos_w[eid], self.robot.data.root_quat_w[eid], obj_w)
        radius = float(getattr(self._env, "_bp_target_radius", 0.05))
        ang = torch.rand(eid.numel(), device=self.device) * (2.0 * math.pi)
        self.pose_command_b[eid, 0] = obj_b[:, 0] + radius * torch.cos(ang)
        self.pose_command_b[eid, 1] = obj_b[:, 1] + radius * torch.sin(ang)
        # z(표면 목표)·quat은 super가 설정한 값 유지 — 수평 반경만 물체 기준으로 대체


@configclass
class CommandsCfg:
    """place 목표 샘플러 — 범위는 configs/common.yaml target.sample_range.

    ★일반 UniformPoseCommandCfg★ — 모든 env가 에피소드마다 균일 랜덤 '표면' 타겟을 받는다.
    (한때 place bootstrap용 PlaceBootstrapPoseCommand 서브클래스가 있었으나 bootstrap 접근
    자체가 폐기됨(2026-07-11, farming) — 현행 릴리스 학습 = BINPICK_STAGE 2단계 + release_bias
    smooth. 경위는 README.md 결정 로그 참조.)
    """

    target_pos = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        # body_name은 디버그 시각화/메트릭 기준 바디일 뿐 샘플 분포와 무관.
        # ⚠️ TODO(확인 필요): UR5e USD의 실제 링크명("wrist_3_link" 가정) 검증.
        body_name="wrist_3_link",
        # 에피소드(8s)보다 훨씬 길게 → 리셋 시에만 재샘플 = "에피소드당 고정 목표".
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=tuple(float(v) for v in _TARGET_RANGE["x"]),
            pos_y=tuple(float(v) for v in _TARGET_RANGE["y"]),
            pos_z=tuple(float(v) for v in _TARGET_RANGE["z"]),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Actions — ★task-space IK (ik_rel, 2026-07-09)★
# ═════════════════════════════════════════════════════════════════════════════
@configclass
class ActionsCfg:
    """action = (7,) [TCP pose delta 6 (dx,dy,dz,dR,dP,dY) | 그리퍼 1].

    ★2026-07-09 task-space 상대 IK (DifferentialInverseKinematicsAction, ik_rel)★ — 관절 액션의
    딜레마 해소: 상대관절=reach○/holding✗(중력sag), 절대관절=holding○/reach✗(IK학습 어려움,
    정량 reach 0.12 막힘). task-space IK는 정책이 TCP pose를 명령 → reach가 Cartesian이라 쉽고
    (물체위치≈명령), 팔 중력 disable(ur5e_assets, Franka IK 표준)로 sag 없어 holding 성립.

    검증(vendored 2.1.0, differential_ik.py):
      - command_type="pose"+use_relative_mode=True → action_dim=6 (L81). +gripper=7 (obs 불변).
      - process L158: processed=raw×scale, set_command(delta, ee_cur). apply L175: IK→joint target.
      - body_offset = tcp_offset.yaml(파지중심) → IK가 TCP(그리퍼 파지점)를 제어(요소4 정합).
    ⚠️ real 계약(TCP pose delta → ur_rtde servoL): Phase4 reconcile(servoL 미검증 §9.4). 현재
       sim 학습·Phase2 증류는 이 task-space 액션으로 자기완결. obs last_action 차원은 7 불변.
    """

    arm_action = bp_actions.DelayedDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(UR5E_ARM_JOINT_NAMES),
        body_name="wrist_3_link",  # flange — body_offset로 TCP(파지중심)까지 확장
        controller=DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=True, ik_method="dls"
        ),
        scale=_IK_SCALE,
        # ★TCP 하강 바닥 (F51, 2026-08-09 실측)★
        #   descend_sweep 실측: 손가락 패드가 TCP z=0.0174 m(=TABLE_HEIGHT+0.0274)부터 벨트에
        #   닿고 반력 150~240 N. 5cm 큐브 안착 중심(place 타겟)은 0.0150 m이라, 안착 높이까지
        #   내리려면 패드가 벨트를 2.4mm 파고들어야 한다 → 그 충격이 손가락을 닫아 놓은 큐브를
        #   재포획·0.25 m/s로 튕겨냈다(placed '정지' 조건 붕괴의 실체).
        #   바닥 = TABLE_HEIGHT + 0.030 → 실측 충돌점보다 2.6mm 위. 큐브는 z∈[-0.010,0.040]에
        #   걸쳐 있어 이 높이에서도 패드가 몸통을 문다(파지 가능 영역 유지).
        #   ⚠️실물도 테이블을 긁도록 두지 않는다 — sim 전용 편법이 아니라 실기에도 있는 제약.
        tcp_z_floor_m=TABLE_HEIGHT + 0.030,
        # ★F78★ 수직 파지 — 기울임(axis-angle x·y)을 막고 yaw만 허용한다.
        #   실기 등가물은 configs/common.yaml action.vertical_grasp_only (양쪽 동시 설정).
        vertical_grasp_only=bool(_COMMON["action"].get("vertical_grasp_only", False)),
        body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=_TCP_OFFSET_POS, rot=_TCP_OFFSET_QUAT
        ),
        delay_steps=_ACTION_DELAY_STEPS,
        # ★Teacher v2★ 스텝당 TCP 변위/회전 상한 = 배포 SafetyGuard(safety.yaml limits × dt).
        #   학습이 실기와 같은 실행 제한 아래에서 이뤄져야 정책 거동이 전이된다(README §5 clamp 게이트).
        tcp_step_max_m=float(_SAFETY_LIMITS["tcp_linear_vel_max_m_s"]) / _CONTROL_HZ,
        tcp_rot_step_max_rad=float(_SAFETY_LIMITS["tcp_angular_vel_max_rad_s"]) / _CONTROL_HZ,
    )

    # ★그리퍼 배선 완료 (OnRobot 2FG7, 2026-07-06) — action 6→7★
    #   BinaryJointPositionAction = action 1차원 추가(±) → open/close 목표를 두 finger
    #   관절에 대칭 기록. ★부호 계약(2026-07-09 정정)★: action[-1] < 0 → close
    #   (upstream binary_joint_actions.py L124-129 관례 = common/action_mapping.py
    #   gripper_closed = action[-1] < gripper_close_threshold 와 동일 의미).
    #   open/close 값 = ur5e_assets.py.
    gripper_action = bp_actions.DelayedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND,
        close_command_expr=GRIPPER_CLOSE_COMMAND,
        delay_steps=_ACTION_DELAY_STEPS,
        command_transition_steps=_GRIPPER_COMMAND_TRANSITION_STEPS,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Observations
# ═════════════════════════════════════════════════════════════════════════════
@configclass
class PrivilegedObsCfg(ObsGroup):
    """★Teacher v2(2026-09-18)★ Teacher actor 관측 = ★실기에서 만들 수 있는 신호만★ (34D).

    순서·폭은 common/teacher_contract.TEACHER_OBS_LAYOUT과 글자 그대로 같아야 한다
    (tests/test_teacher_contract.py가 강제). 배포측 common/teacher_obs.TeacherObsBuilder가
    같은 레이아웃·같은 규칙으로 실기 신호를 조립한다.

    구 42D(실제 손가락 위치·물체 속도·접촉력 특권)는 실기에서 근사로 채워야 했고, 그 근사가
    학습 분포와 달라 배포 성능을 보장할 수 없었다(F69: 접촉을 뺀 teacher는 정밀도 의존 정책을
    배운다). v2는 반대로 ★sim 관측을 배포 인식 모듈 수준으로 낮춰★ 그 오차·결측·기하 제약
    아래에서 성공하는 정책을 학습한다(mdp/perception_model.py, configs/dr.yaml perception).

    ⚠️ vision policy 그룹은 오직 flat 단일 항(아래 VisionPolicyObsCfg)만 갖는다 — 이 그룹은
    teacher actor·증류 teacher 입력 전용이다.
    """

    joint_pos = ObsTerm(func=bp_obs.joint_pos_abs)
    joint_vel = ObsTerm(func=bp_obs.joint_vel_abs)
    # 실제 개도가 아니라 ★지연·전이 적분된 명령 상태★ — student flat 관측과 같은 신호.
    gripper_command_state = ObsTerm(func=bp_perc.gripper_command_state_obs)
    tcp_pose = ObsTerm(func=bp_obs.tcp_pose_b)
    target_pos = ObsTerm(func=bp_obs.target_pos_command)
    # ★인식 추정치★ — 참값 + 편향 + 노이즈, 카메라 기하·확률 결측, snap/held_offset/last_known 합성.
    object_pos = ObsTerm(func=bp_perc.object_pos_estimate)
    # 폐합 전이 시점 추정치-TCP 근접 latch — 배포 규칙과 동일(common/teacher_contract 상수).
    held = ObsTerm(func=bp_perc.held_estimate)
    # ★F81★ raw가 아니라 ★실행 가능 정규화★ 액션을 관측한다(clamp ±1 + 기울임 0).
    #   구 mdp.last_action(raw)은 ①무시 채널의 표류값(±7)을 자유 메모리로 되먹임했고
    #   ②증류에서 student 스케일과 갈라져 teacher를 분포 밖에서 질의하게 했다 —
    #   3연속 증류 실패의 blocker(리뷰 2026-09-03). 상세: bp_obs.executable_last_action.
    last_action = ObsTerm(func=bp_obs.executable_last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class TeacherCriticObsCfg(PrivilegedObsCfg):
    """★Teacher v2★ critic 전용 관측 = actor 34D + sim 참값(물체 위치·속도·접촉력).

    actor는 부분관측(추정치·결측)이지만 critic은 sim에서만 돌므로 참값을 더 받는다
    (비대칭 actor-critic — vision env의 critic=특권과 같은 원칙). 배포 대상이 아니다.
    """

    object_pos_true = ObsTerm(func=bp_obs.object_positions_b)
    object_vel = ObsTerm(func=bp_obs.object_velocities)
    contact_forces = ObsTerm(func=bp_obs.finger_contact_forces)


@configclass
class TeacherObservationsCfg:
    """Phase1 teacher — policy=배포 가능 34D / critic=34D+sim 참값 (카메라 없음)."""

    policy: PrivilegedObsCfg = PrivilegedObsCfg()
    critic: TeacherCriticObsCfg = TeacherCriticObsCfg()


@configclass
class VisionPolicyObsCfg(ObsGroup):
    """★요소1: vision policy 그룹 = 단일 flat ObsTerm — 절대 항을 추가하지 말 것★

    flat 레이아웃은 common.obs_preprocess.build_policy_obs 내부에서 구조적으로
    고정된다(= 배포 obs_spec.json과 동일). 여기 항을 추가하는 순간 sim/real 관측
    계약이 조용히 깨진다.
    """

    flat = ObsTerm(func=bp_obs.policy_obs, params={"delay_proprio": False})

    def __post_init__(self):
        self.enable_corruption = False  # 노이즈는 policy_obs 내부(depth) + DR가 담당
        self.concatenate_terms = True


@configclass
class VisionObservationsCfg:
    """Phase2 vision — 비대칭: policy=flat(배포 동일) / critic=특권 / teacher=특권(증류용).

    teacher 그룹: rsl_rl 3.1.2 DistillationRunner가 extras["observations"]["teacher"]에서
    teacher 입력을 읽는다(vendored on_policy_runner.py 줄 58-62 검증).
    Phase1 teacher actor의 관측 = PrivilegedObsCfg(TeacherObservationsCfg.policy)이므로
    같은 클래스를 그대로 노출하면 차원이 자동으로 일치한다 (critic과 항 중복 — 의도됨).
    PPO 본선(Vision-v0)에서는 이 그룹이 무시되므로 부작용 없음.
    """

    policy: VisionPolicyObsCfg = VisionPolicyObsCfg()
    critic: TeacherCriticObsCfg = TeacherCriticObsCfg()
    teacher: PrivilegedObsCfg = PrivilegedObsCfg()


# ═════════════════════════════════════════════════════════════════════════════
# Rewards — ★Franka Lift/Move 검증 구조(공중 목표 + 현재높이 게이트)★ (2026-07-07 재설계)
#   현재 채택 = Franka Phase4 "Lift/Move" 코어(TransportRewardsCfg): reach + lift(binary) +
#     goal-track(현재높이 게이트) + fine + 페널티. 목표가 공중이라 픽업(집기+들기+운반)을
#     물리적으로 강제 — 밀기는 goal 게이트 0으로 무보상. 이 프로젝트의 목적(전이 가능한 '픽업'
#     시연)에 정확히 부합하고, 밀기 익스플로잇을 원천 차단한다.
#   ⚠️ 향후 'place-down(목표 위에서 내려놓기)'을 학습으로 추가하려면 Franka Phase4의 어려운 교훈:
#     ① place 단계는 goal 게이트를 높이 아닌 grasp-latch로 (내려놓아도 보상 유지)
#     ② dense 보상 + 성공 '종료'를 섞으면 잔여 스트림 포기 모순 — 압도적 종료 보너스로 상회
#     ③ binary 그리퍼 release 탐색은 보상만으로 안 뚫림 — 역방향 커리큘럼 필요
#     → 현재는 내려놓기를 배포 시 스크립트(목표 위 그리퍼 open)로 처리하여 회피(권장).
# ═════════════════════════════════════════════════════════════════════════════
# ── 보상 기하 임계 (base frame [m]) — ★Franka Lift 검증 구조 이식 (2026-07-07 재설계)★
#   [배경] 직전 teacher(4096×2500)는 '집지 않고 바닥으로 밀기'로 수렴했다. 근본원인 2가지:
#     ① 목표 z=0.02(=큐브 안착높이) → 밀어서 (tx,ty)에 두면 place/placed 충족(픽업 불요).
#     ② grasp-latch: 한 번만 z>0.05 뜨면 래치 영구 ON → 이후 밀기에도 place 보상 지급.
#   [수정] Franka Lift/Move 구조로 전환 — 목표를 공중(common.yaml target.z=0.15)에 두고,
#     goal 보상을 '현재 높이 > MIN_LIFT'로 게이트한다. 들어올려 운반해야만 목표 거리가 줄어
#     보상, 밀기(바닥 z)는 게이트 0으로 무보상. latch·floor-target·approach-shaping 전부 제거.
#   객체 치수·벨트 높이는 아래 상대식이 자동 추종한다.
# ★2026-07-29 큐브 크기 독립화(사용자 결정)★: 구 `TABLE_HEIGHT + 0.04`는 ★절대 높이★라
#   큐브가 커지면 '들었다' 판정이 저절로 헐거워졌다 — 5cm 큐브면 안착 중심이 이미 상면+2.5cm라
#   실제 들어올림 요구가 2.5cm→1.5cm로 줄어, 파지 없이 밀기만 해도 lift가 발화할 수 있다
#   (과거 '밀기 수렴' 사고와 같은 계열). 안착 높이(상면+반변)를 기준으로 삼아 ★큐브 크기와
#   무관하게 실제 2.5cm 상승★을 요구한다.
#   현행 5cm에서는 작업면+2.5cm(반변)+2.5cm(상승)를 요구한다.
_LIFT_CLEARANCE = 0.025          # 안착 상태에서 추가로 들어올려야 하는 높이 [m]
MIN_LIFT = TABLE_HEIGHT + OBJECT_SIZE / 2.0 + _LIFT_CLEARANCE  # 최근접 객체 z > 이 값 → '들림'.
                                 #   5cm 큐브 안착 중심에서 실제 2.5cm 상승한 상태(밀기 배제).
GRASP_RADIUS = 0.06              # (부트스트랩 전용) 최근접 객체 TCP 근접 반경.
# ★파지 정렬(pregrasp_align) 상수 — Franka Phase4 grip_shift 대응 이식★. 검증 실측(2026-07-07):
#   grasp 부트스트랩(near+close)만으론 '닫기'는 배우나 lift 미발현 → 닫아도 큐브를 못 감쌈(정렬 불량)
#   가능성. TCP를 큐브 중심 바로 위(xy 정렬)로 유도해 그리퍼가 큐브를 감싸 닫히게 한다.
ALIGN_STD = 0.03                 # TCP-큐브 xy 정렬 tanh 폭[m] — 0.03이면 ~1.5cm 이내 급보상.
ALIGN_NEAR = 0.12                # 정렬 보상이 켜지는 TCP-큐브 3D 근접 반경[m] — 접근 최종 국면만 셰이핑.


def _nearest_object(env) -> tuple[torch.Tensor, torch.Tensor]:
    """최근접(TCP 기준) 객체의 위치(N,3, base)·거리(N,) — reach/lift/goal 보상이 공유.

    산란 큐브가 여럿이라 '지금 조작 중인 하나'를 최근접으로 특정한다. 큐브를 집어 들면
    그리퍼 안에 있어 계속 최근접이 되므로 reach→lift→goal이 같은 큐브를 일관 추종한다.
    base frame 전제(요소4): object_positions_b = root_pos_w − env_origin, z=world z
    (env_origin z=0) → 임계(MIN_LIFT, 벨트 상면 z=TABLE_HEIGHT 기준)와 같은 프레임.
    """
    n, k = env.num_envs, NUM_OBJECTS
    objs = bp_obs.object_positions_b(env).view(n, k, 3)          # (N,K,3) base
    tcp = bp_obs.tcp_pose_b(env)[:, :3]                          # (N,3)
    dist, idx = torch.norm(objs - tcp.unsqueeze(1), dim=-1).min(dim=1)
    pos = torch.gather(objs, 1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)
    return pos, dist


def reach_nearest_object(env, std: float) -> torch.Tensor:
    """reach: 1 − tanh(min_i‖TCP − object_i‖ / std). (Franka object_ee_distance, 최근접 객체)

    dense 신호라 그리퍼 없이도 teacher가 '물체로 팔을 뻗기'를 먼저 학습 → 그 위에서 랜덤
    그리퍼 닫힘+상승이 lift(아래)를 발화시켜 픽업이 부트스트랩된다.
    """
    _, dist = _nearest_object(env)
    return 1.0 - torch.tanh(dist / std)


# (구세대 무게이트 보상 lift_nearest_object / goal_distance_nearest / goal_distance_grasped 는
#  저글링 해킹(STAGE 3)·hover 설계 폐기(2026-07-10 Full Pick&Place)로 미사용이 되어 삭제 —
#  2026-07-11 검토. 정의가 필요하면 git 히스토리 참조.)


# ═════════════════════════════════════════════════════════════════════════════
# ★STAGE 3 (2026-07-09) — 파지 성립 게이트 (anti-reward-hacking)★
#   STAGE 0 프로브 확정: 정책이 '들림 중 양손접촉 0%'(왼손 타격 3회, 큐브 2.8-4.0 m/s,
#   최고 0.66m 비행) = 순수 저글링으로 lift/goal 보상을 해킹. 원인 = 높이 게이트만 있고
#   파지 게이트 부재. 아래 _grasped_mask는 프로브와 '동일 정의'(양손 접촉력>force_n ∧
#   큐브-EE 상대속도<rel_speed_mps) — 진단 지표 = 보상 게이트 정합으로 재해킹을 프로브가
#   즉시 재검출할 수 있다. 센서 = scene.make_finger_contact_sensor_cfgs() (프로브로 검증).
# ═════════════════════════════════════════════════════════════════════════════
def _grasped_mask(env, force_n: float, rel_speed_mps: float) -> torch.Tensor:
    """파지 성립(bool, N): 양손가락 접촉력>force_n ∧ (객체-EE 상대속도)<rel_speed_mps.

    force_matrix_w (N,1,M,3) — 손가락 prim이 env당 단일이라 filtered 접촉 보장
    (contact_sensor_cfg.py L47-51). 센서 부재/미가용 시 False(게이트 닫힘 — 보수적).
    M(=NUM_OBJECTS)개 객체 중 '양손이 동시에 잡고 있고 함께 움직이는' 객체가 하나라도
    있으면 True. 자유비행(저글링)은 상대속도 ~1.4 m/s(프로브 중앙값)라 게이트에 안 걸림.
    """
    sensors = env.scene.sensors
    if "left_finger_contact" not in sensors or "right_finger_contact" not in sensors:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fmL = sensors["left_finger_contact"].data.force_matrix_w   # (N,1,M,3)|None
    fmR = sensors["right_finger_contact"].data.force_matrix_w
    if fmL is None or fmR is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fL = fmL.squeeze(1).norm(dim=-1)   # (N,M) 객체별 접촉력 크기
    fR = fmR.squeeze(1).norm(dim=-1)
    both = (fL > force_n) & (fR > force_n)                      # (N,M)
    # 객체-EE 상대속도 (filter 순서 = Object_i 순서 = object_names 순서)
    robot = env.scene["robot"]
    ee_id = getattr(env, "_bp_ee_body_id", None)
    if ee_id is None:
        ids, _ = robot.find_bodies("wrist_3_link")              # articulation.py L209
        ee_id = ids[0]
        env._bp_ee_body_id = ee_id
    ee_v = robot.data.body_lin_vel_w[:, ee_id, :]               # (N,3) AD L794
    slow = torch.stack(
        [(env.scene[name].data.root_lin_vel_w - ee_v).norm(dim=-1) < rel_speed_mps
         for name in object_names(NUM_OBJECTS)], dim=1)         # (N,M)
    return (both & slow).any(dim=1)


def grasped_gate(env, force_n: float, rel_speed_mps: float) -> torch.Tensor:
    """grasp 스테핑스톤: 파지 성립 자체에 소량 보상 — reach와 lift 사이 사다리.

    ⚠️★2026-08-06(감사 P7-2)★ 여기 있던 '자력 파지 EMA 갱신' 부수효과를 ★제거★했다.
      `RewardManager.compute`는 `weight == 0.0`이면 항을 ★건너뛴다★(reward_manager.py L145-148).
      즉 grasp 가중치를 0으로 두는 순간 EMA가 얼어붙어 능력 게이트가 영구 차단됐을 것이다.
      EMA는 이제 보상과 무관한 전용 이벤트(`_update_grasp_ema`)가 담당한다.
    """
    return _grasped_mask(env, force_n, rel_speed_mps).float()


def lift_grasped(env, minimal_height: float, force_n: float, rel_speed_mps: float,
                 command_name: str = "target_pos", latch_radius: float = 0.15) -> torch.Tensor:
    """lift(gated): 파지 ∧ (z>minimal_height ∨ (배달 래치 ∧ 타겟≤latch_radius)) → 1.

    높이 단독 게이트(구 lift_nearest_object — 삭제됨)는 저글링(무파지 비행)에 뚫렸다
    (프로브 확정) — 실제 잡고 들어올린 경우에만 보상.

    ★★2026-07-31(F36) 최종 하강 예외 추가 — goal_distance_carry와 동일 가드★★
      [발견] `goal`은 2026-07-10 리뷰에서 '래치 ∧ 타겟 반경 내'면 z게이트를 면제해
        ★최종 하강 중에도 보상이 끊기지 않게★ 고쳤는데, ★lift만 그 수정에서 빠졌다★.
        그 결과 안착을 위해 2.5cm 하강하는 순간 lift(w5)만 사라져 스텝당 보상이 26.4 → 23.1
        (−12%)로 꺼진다 — 경로상 ★유일하게 남은 역인센티브★였다.
        v9 실측: hold_frac 0.876 · airborne_frac 0.908 · placed_frac 0.0000
        = 정책이 타겟 위 최저 높이에 떠서 에피소드를 마친다(호버-홀드 고착).
      [farming 안전성] 가드는 `goal`(w16 — lift의 3배)이 이미 쓰는 것과 ★완전히 동일★하므로
        새로운 수확 경로가 열리지 않는다. 2026-07-10 적대 리뷰의 매트릭스 재확인:
          밀기=파지게이트 / 저글링=접촉게이트 / 무들림 슬라이딩=래치 없음 /
          들고-떨구고-밀기=배달래치 해제(¬파지 10스텝 ∧ >0.10m) / 크로스벨트 드래그=반경 0.15 제한.
        추가 시나리오 '타겟에서 잡은 채 바닥에 얹어두기' = 28.1/step ≪ released 56.0/step (2.0×)
        → 놓는 것이 여전히 결정적으로 우월하다.
      [효과] 하강 구간 23.1 → 28.1로 ★밸리 소멸★: 호버 26.4 → 하강 28.1 → 릴리스 35.5 →
        안착 56.0 으로 ★단조 증가 경로★가 된다.
    """
    pos, _ = _nearest_object(env)
    target = env.command_manager.get_command(command_name)[:, :3]
    latch_near = _lifted_latch(env, force_n, rel_speed_mps, minimal_height) & (
        torch.norm(target - pos, dim=-1) < latch_radius
    )
    gate = _grasped_mask(env, force_n, rel_speed_mps) & (
        (pos[:, 2] > minimal_height) | latch_near
    )
    return gate.float()


# ═════════════════════════════════════════════════════════════════════════════
# ★Full Pick&Place (2026-07-10, 사용자 결정) — 표면 타겟 + ever_lifted 래치 + 릴리스★
#   [배경] 이전 설계는 목표=공중 hover(밀기 hacking 회피 shortcut)라 Place(내려놓기+릴리스)가
#   학습에서 빠졌다(사용자 지적). 이제 진짜파지 게이트(_grasped_mask)가 작동하므로:
#     · 타겟 = 벨트 표면 랜덤 좌표(common.yaml z=0.015)
#     · ever_lifted 래치 = "이 에피소드에서 진짜파지∧들림을 달성" — 옛 grasp-latch(순간 z만
#       확인 → 저글링에 뚫림, 2026-07-07 교훈②)와 달리 ★접촉 검증★이라 밀기/저글링으로 못 연다.
#     · carry(goal) 게이트 = 파지 ∧ (z>MIN_LIFT ∨ 래치): 들고 운반(z게이트) + 래치 후 최종
#       하강(z<MIN_LIFT)에도 보상 유지(밸리 제거). ★무들림 슬라이딩 차단★: 래치 없이는
#       바닥에서 잡고 끌어도 게이트 닫힘.
#     · placed(릴리스 성공) = 래치 ∧ 타겟 3cm ∧ 정지 ∧ ★무접촉★ — 놓아야만 최대 보상(w50)이
#       흐르는 흡수 상태(hover 26.2 ≪ released 57 — 아래 매트릭스 산술).
#   [hacking 매트릭스 — 적대적 리뷰(9에이전트) 산술검증 후 교정판 2026-07-10]
#     · 밀기(무파지) = 전 게이트 닫힘 / 저글링 = 접촉게이트 닫힘 / 무들림 슬라이딩 = 래치 없음.
#     · 들고-떨구고-밀기 = ★배달 래치★(¬파지 10스텝 ∧ 타겟>0.10m → 해제)가 차단 — 떨군 순간
#       0.5s 내 place 게이트 사멸, 재파지·재들기만이 복구 경로.
#     · 크로스벨트 드래그(래치 후 바닥 끌기) = carry 래치 분기의 타겟 반경(0.15m) 제한이 차단.
#     · 호버-홀드(타겟 위 z≈0.041 정지) ≈ 26.2/step(reach1+grasp2+lift5+carry14.6+fine3.6)
#       ≪ released ≈ 57/step(placed50+fine6+reach1) — 2.2× 마진으로 하강·릴리스가 결정적 우월.
#       (lift w15 시절엔 hover 36.2 ≈ released 37로 동률 = 릴리스 미학습 결함 — 리뷰 확정·교정.)
# ═════════════════════════════════════════════════════════════════════════════
def _nearest_object_vel(env) -> torch.Tensor:
    """최근접(TCP 기준) 객체의 선속도(N,3, world) — placed 정지 판정용."""
    n, k = env.num_envs, NUM_OBJECTS
    objs = bp_obs.object_positions_b(env).view(n, k, 3)
    tcp = bp_obs.tcp_pose_b(env)[:, :3]
    idx = torch.norm(objs - tcp.unsqueeze(1), dim=-1).min(dim=1).indices
    vels = torch.stack(
        [env.scene[nm].data.root_lin_vel_w for nm in object_names(k)], dim=1)  # (N,K,3)
    return torch.gather(vels, 1, idx.view(-1, 1, 1).expand(-1, 1, 3)).squeeze(1)


def _any_finger_contact(env, thr: float) -> torch.Tensor:
    """어느 한 손가락이라도 객체와 접촉(>thr[N])하는가(bool, N) — 릴리스(무접촉) 판정용.

    _grasped_mask(양손 동시)와 달리 단일 손가락 접촉(밀기/터치)도 True — placed는 이것의
    부정(¬)을 요구해 '손 뗀' 상태만 성공으로 친다. 센서 미가용 시 True(보수적 — placed 불허).
    """
    sensors = env.scene.sensors
    if "left_finger_contact" not in sensors or "right_finger_contact" not in sensors:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    fmL = sensors["left_finger_contact"].data.force_matrix_w
    fmR = sensors["right_finger_contact"].data.force_matrix_w
    if fmL is None or fmR is None:
        return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    fL = fmL.squeeze(1).norm(dim=-1)   # (N,M)
    fR = fmR.squeeze(1).norm(dim=-1)
    return (fL > thr).any(dim=1) | (fR > thr).any(dim=1)


def _lifted_latch(env, force_n: float = 0.1, rel_speed_mps: float = 0.2,
                  minimal_height: float = MIN_LIFT) -> torch.Tensor:
    """★배달(delivery) 래치(bool, N)★: (진짜파지 ∧ z>minimal_height) 달성 후 '배달 중/완료' 상태.

    place 단계 게이트의 핵심. 접촉 검증(_grasped_mask) 필수라 순간 들림/저글링/밀기로 열리지
    않는다. ★2026-07-10 리뷰 반영 — 영구 래치 → 배달 래치★: 파지를 잃고(¬grasped) 0.5s(10스텝)
    지났는데 큐브가 타겟에서 멀면(>0.10m) 래치 해제 — '한 번 들었다 떨구고 밀기'가 place 게이트를
    영구 개방하던 허점 차단. 타겟 근처 릴리스(≤0.10m)는 래치 유지(placed·place_fine 계속 지급 =
    의도된 종착 상태). 스텝당 1회만 상태 갱신(common_step_counter 가드 — 다항 호출 멱등,
    ungrasp 카운터 중복증가 방지). 매니저 초기화 중(버퍼 부재)은 전부 False.
    """
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # 버퍼 독립 초기화 — 보조 버퍼(_bp_ungrasp_steps·_bp_latch_step)를 각각 따로 가드(누락 방지).
    #   (place bootstrap의 _bp_place_sel 예외 분기는 bootstrap 폐기(2026-07-11)로 함께 삭제.)
    latch = getattr(env, "_bp_lift_latch", None)
    if latch is None or latch.shape[0] != env.num_envs:
        latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._bp_lift_latch = latch
    if getattr(env, "_bp_ungrasp_steps", None) is None or env._bp_ungrasp_steps.shape[0] != env.num_envs:
        env._bp_ungrasp_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if not hasattr(env, "_bp_latch_step"):
        env._bp_latch_step = -1
    step = int(env.common_step_counter)
    if env._bp_latch_step != step:               # ★스텝당 1회 갱신(멱등 가드)★
        env._bp_latch_step = step
        latch &= env.episode_length_buf > 1      # 리셋 직후 클리어
        pos, _ = _nearest_object(env)
        grasped = _grasped_mask(env, force_n, rel_speed_mps)
        latch |= (pos[:, 2] > minimal_height) & grasped
        # 배달 래치 해제: ¬파지 10스텝 초과 ∧ 타겟에서 멀다(>0.10m)
        cnt = env._bp_ungrasp_steps
        cnt[:] = torch.where(grasped, torch.zeros_like(cnt), cnt + 1)
        target = env.command_manager.get_command("target_pos")[:, :3]
        far = torch.norm(target - pos, dim=-1) > 0.10
        latch &= ~((cnt > 10) & far)
    return latch


def goal_distance_carry(
    env, std: float, minimal_height: float, force_n: float, rel_speed_mps: float,
    command_name: str = "target_pos", latch_radius: float = 0.15,
) -> torch.Tensor:
    """carry(운반) 보상: [파지 ∧ (z>minimal ∨ (래치 ∧ 타겟≤latch_radius))] · (1−tanh(d/std)).

    z게이트 단독은 표면 타겟 최종 하강 중 보상이 증발(밸리)하고, 파지게이트 단독은 '무들림
    슬라이딩'을 허용한다. ★2026-07-10 리뷰 반영★: 래치 분기에 타겟 반경(0.15m = place_fine 유효
    반경) 제한 추가 — 래치 후 '바닥에서 잡고 크로스벨트 드래그'가 carry(16)를 수집하던 허점 차단.
    의도된 용도(타겟 위 최종 하강)는 반경 내라 그대로 보존.
    """
    pos, _ = _nearest_object(env)
    target = env.command_manager.get_command(command_name)[:, :3]
    d = torch.norm(target - pos, dim=-1)
    latch_near = _lifted_latch(env, force_n, rel_speed_mps, minimal_height) & (d < latch_radius)
    gate = _grasped_mask(env, force_n, rel_speed_mps) & (
        (pos[:, 2] > minimal_height) | latch_near
    )
    return gate.float() * (1.0 - torch.tanh(d / std))


def _real_pick_mask(env) -> torch.Tensor:
    """★place-side farming 차단 마스크(2026-07-16)★ — pregrasp로 '잡힌 채 시작한' env는 1.0이
    아니라 0.0을 반환해 place '완성' 보상(placed·place_fine)에서 제외한다.
    (release_bias는 2026-07-16 마일스톤 2500 진단으로 마스크에서 해제 — 릴리스 부트스트랩 발판.
    해당 함수 주석 참조.)

    이유: pregrasp env는 리셋 즉시 배달 latch를 사실상 공짜로 얻어(events_dr._pregrasp_reset), 타겟이
    pregrasp 바로 아래면 '내려놓기'만으로 place 보상을 farm → grasp 학습이 죽는다(TCW1이 그립을
    8.6cm 낮춰 촉발, 통제 A/B로 확정). 이 세 항을 real-pick env(¬pregrasp)에서만 흘리면 공짜 latch가
    place 보상에 도달하는 경로가 원천 차단되고, pick/carry(grasp/lift/goal)는 전 env 유지되어 pregrasp의
    hold+carry 부트스트랩 목적은 보존된다. 플래그 부재(pregrasp off·eval)면 전부 real-pick(1.0).
    """
    m = getattr(env, "_bp_pregrasp_env", None)
    if m is None or m.shape[0] != env.num_envs:
        return torch.ones(env.num_envs, device=env.device)
    return (~m).float()


def place_fine(env, std: float, command_name: str = "target_pos") -> torch.Tensor:
    """place 정밀 유도: ever_lifted · (1 − tanh(‖목표−객체‖/std)). ★릴리스 후에도 흐름★.

    좁은 std(0.06)라 타겟 ~15cm 밖에선 ≈0 — 래치 후 '떨구고 밀기'로 수집 가능한 양이 미미
    (운반이 lift+carry로 압도). 릴리스(파지 해제) 후에도 큐브가 타겟에 있으면 계속 지급 →
    releases가 carry 스트림을 잃는 급락을 완충. ★pregrasp env 제외(_real_pick_mask, farming 차단)★.
    """
    pos, _ = _nearest_object(env)
    target = env.command_manager.get_command(command_name)[:, :3]
    d = torch.norm(target - pos, dim=-1)
    return _lifted_latch(env).float() * (1.0 - torch.tanh(d / std)) * _real_pick_mask(env)


# ═════════════════════════════════════════════════════════════════════════════
# ★★placed '정지' 임계 — 단일 소스★★
#   ⚠️★2026-08-01 F37 정정★: 앞서 이 값을 0.10 → 0.30으로 올렸으나 ★근거가 틀렸다★.
#     당시 "sim 접촉 지터가 0.154라 임계 0.10을 만족할 수 없다"고 판단했는데, 그 측정은
#     `Grasp_PhysicsTest`의 ★자체 큐브★(RigidBodyPropertiesCfg(disable_gravity=False)만 설정)
#     에서 나온 값이었다. ★학습 환경의 큐브★는 solver_position_iteration_count=16 등이 적용돼
#     실측 정지 속도가 ★0.0010 m/s★다(Pregrasp_Probe, teacher env, 40샘플 100%가 <0.01).
#     즉 임계 0.10은 ★100배 여유★로 충분했고, 0.30은 성공 기준을 부당하게 완화한 것이었다.
#   [값 근거] 문헌 표준(Isaac Lab pick&place 성공 기준)은 정지 판정 ★0.01 m/s★.
#     본 프로젝트는 0.10을 쓰며, 이 값으로 teacher v3가 placed 96.6%를 달성한 실적이 있다.
#     학습 환경 실측 정지 속도 0.0010과 100배 여유 → 판정력·달성가능성 모두 확보.
#   ⚠️★학습·평가가 반드시 같은 값을 써야 한다★ — 구버전은 이 값이 두 곳에 따로 박혀 있었다
#     (F30과 같은 부류의 드리프트). 이 상수가 유일한 정의이며 BinPick_Eval도 이를 import한다.
#   ⚠️교훈: ★진단 도구가 학습 환경과 같은 물리 설정을 쓰는지 먼저 확인하라★ —
#     이번 세션에서 같은 함정에 ★두 번★ 빠졌다(콜리전 형상 F30, 물체 물리 F37).
# ═════════════════════════════════════════════════════════════════════════════
# ★★★2026-08-12(F62) 0.02 → 0.10 복원 — 5/5 성공 vs 0/13 실패를 가른 유일한 변수★★★
#   [증거] 학습측 `Episode_Reward/placed`(eval 프로토콜과 무관한 지표) 18개 run 전수:
#     max_speed 0.10 → 5개 run 전부 ★20.4~23.3★ (placed 학습 성공)
#     max_speed 0.02 → 13개 run 전부 ★0.0000~0.0025★ (예외 0건)
#   [이 사고는 이미 기록돼 있었다] 바로 아래 F50 주석이 b6da816(2026-08-07)의 3중 강화가
#     20k run을 0으로 만든 것을 진단했는데, ★3개 중 2개만 되돌리고 속도는 남겼다★
#     ("한 번에 여러 변인을 바꾼 것이 사고 원인이므로 건드리지 않는다"). 그 판단이 오류였다 —
#     이후 13개 run이 전부 0이다.
#   [기전 — §9-5-1의 정확한 재발] 0.02가 ★물리적으로 불가능한 것은 아니다★
#     (94.7% run eval 실측 큐브 최저속도 중앙 0.0045). 문제는 ★학습 초기에 한 번도 표집되지
#     않는다★는 것이다 — 표집이 0이면 gradient가 0이고, 릴리스는 시작조차 못 한다.
#     0.10은 초기에 우연히 만족돼 복리로 자란다(08-06: duty 1.3e-3@2k → 0.256@4k).
#     ⇒ ★성공 판정으로서의 타당성★과 ★부트스트랩 신호로서의 표집 가능성★은 다른 축이다.
#   [문헌] 0.10은 업스트림보다 오히려 엄격하다 — Isaac Lab manipulation/place threshold 0.2.
#   ⚠️보고의 정직성: 학습은 이 값(0.10)으로 하되, ★배포 등급 판정★은 아래 STRICT 상수로
#     `BinPick_Eval`이 ★양쪽 다 보고★한다. 95% 주장 시 어느 기준인지 반드시 명시할 것.
PLACED_MAX_SPEED: float = 0.10

# ★배포 등급(엄격) 판정 상수 — 학습 보상이 아니라 ★평가 보고 전용★ (F62에서 분리).
#   P9(2026-08-11)가 도입했던 3중 조건을 여기 보존한다: 속도 0.02 · 표면 ±5mm · 10스텝 연속.
STRICT_PLACED_MAX_SPEED: float = 0.02
STRICT_PLACED_SURFACE_TOL_M: float = 0.005
STRICT_PLACED_MIN_CONSECUTIVE_STEPS: int = 10
# ★F62(2026-08-12) 표면 조건을 성공 마스크에서 제거★ — 5/5 성공 run에는 없던 조건이고,
#   도입(bc8eaea, 08-11)은 붕괴(08-08)보다 뒤라 붕괴 원인은 아니지만 현행 엄격도에 가중된다.
#   값은 STRICT_PLACED_SURFACE_TOL_M로 옮겨 평가 보고에서만 쓴다. (0.0 = 마스크에서 비활성)
PLACED_SURFACE_TOL_M: float = 0.0
# ★F62(2026-08-12) 10 → 1 복원★ — 5/5 성공 run은 전부 1스텝이었다(선례 없음은 F50이 이미 지적).
#   엄격 지속(10)은 STRICT_PLACED_MIN_CONSECUTIVE_STEPS로 옮겨 ★평가 보고 전용★이 됐다.
PLACED_MIN_CONSECUTIVE_STEPS: int = 1
PLACED_GRIPPER_OPEN_TOL_M: float = 0.003
# ═════════════════════════════════════════════════════════════════════════════
# ★★2026-08-09 (F50) placed 판정 재설계 — 문헌 대조 후 정정★★
#   [사고] 2026-08-07 커밋 b6da816이 성공 기준 3개를 ★한 커밋에서 동시★로 조였다
#     (속도 0.10→0.02 · 실제 개방 |q|≤3mm 신규 AND · 3스텝 지속 신규 AND).
#     그 논리곱이 성립 불가가 돼 20k run(22.5h) 내내 `Episode_Reward/placed`가 ★정확히 0★,
#     상위 5개 checkpoint 전부 placed 0.00%였다. 강제 개방 프로브에서도 0.00%.
#   [문헌 대조 — 이 프로젝트가 참조하는 표준 3종]
#     · Isaac Lab 2.3.2 `manipulation/place`(AgiBot)   : open_val 0.994 · threshold ★0.2★ (20%)
#     · Isaac Lab 2.3.2 `manipulation/stack`(Franka)   : open_val 0.04  · threshold ★0.005★ (12.5%)
#     · ManiSkill3 `StackCube-v1`                      : 성공 게이트는 관절 허용대가 아니라
#         ★접촉 기반 ¬is_grasping★이고, 관절 개도는 ★dense reward(ungrasp_reward)★로 준다.
#     ⇒ 우리 3mm/20mm = 15%는 ★비율로는 업스트림과 같은 계열★이다(내 초기 F50 기술을 정정).
#       진짜 문제는 임계값이 아니라 ①타겟존에서 손가락이 ★접촉에 막혀★ 열리지 않고
#       ②그 항이 ★성공 게이트★에 들어가 있어 막힘이 곧 실패로 직결된다는 구조다.
#     ⚠️Isaac Lab place/stack의 관절 허용대 판정은 ★teleop/Mimic 시연 태스크★용이라
#       '그리퍼가 항상 목표에 도달한다'가 전제다. 접촉으로 막히는 RL 환경에는 그대로 못 쓴다.
#   [실측 — Phase1/BinPick_PlacedProbe.py, 승자 model_17100]
#     · 무접촉(free-air): 개방 명령 후 ★8스텝(0.4s)에 q=0.0000 도달★ = 액추에이터는 정상.
#     · 타겟존: q_tgt=0(완전 개방 명령)인데 실제 q는 ★16.8mm에서 정체★, 30스텝 강제 개방에도
#       69%가 3mm 대역 진입 실패 → ★접촉 간섭(벨트/큐브)★이 원인.
#   [조치] ①`_gripper_open_mask`를 성공 AND에서 ★빼고★ 릴리스 게이트는 ManiSkill과 같이
#     ★¬접촉★ 단독으로 둔다(아래 placed_success_mask). ②지속 스텝은 ★선례 없음★이라 1로 되돌린다.
#     ③'실제로 열렸는가'는 게이트가 아니라 ★보상★으로 준다(release_bias_bootstrap의 상태항).
#   ⚠️속도 임계 0.02는 ★건드리지 않는다★ — ManiSkill StackCube의 물체 정지 판정 0.01과 같은
#     계열이라 근거가 있고, 한 번에 여러 변인을 바꾼 것이 이번 사고의 원인이기 때문이다.
#     단 F49의 placed 92.38%는 ★0.10에서 측정된 값★이라 run 간 직접 비교는 불가하다.
# ═════════════════════════════════════════════════════════════════════════════
# ★★2026-08-11(P9) 지속 1 → 10 스텝 (0.5 s)★★ — 문헌 + 실측 근거.
#   [문헌] 성공 판정에서 ★엄격한 속도는 예외 없이 긴 지속과 짝지어져 있다★:
#     HomeRobot OVMM(NeurIPS 2023) 5e-3 m/s ↔ ★50 연속 스텝★ / ManiSkill3 StackCube 1e-2 순간.
#     구 조합(2e-2 + 1스텝 + 표면조건 없음)은 세 축 모두에서 가장 약해, 튀어오르는 큐브가
#     정점에서 한 프레임만 걸려도 통과할 수 있었다.
#   [실측 달성률] `--mode settle_floor`: 자연 안착한 큐브는 60 스텝 연속 최대 속도가 0.0068 m/s
#     (임계 0.02의 1/3)다 → ★10 스텝 연속 통과는 보장★된다(F50의 '측정 없이 AND 추가' 재발 방지).
#   ↑ P9의 10스텝은 F62에서 STRICT_PLACED_MIN_CONSECUTIVE_STEPS로 이관(평가 보고 전용).

# ★★2026-08-11(P9) 표면 안착 조건★★ — OVMM의 "goal_receptacle 접촉 유지"에 대응.
#   접촉 센서 없이 ★높이★로 구현한다: 타겟 z가 곧 큐브 안착 중심이므로 |z − z_target| ≤ tol.
#   [실측] 자연 안착 큐브 z 중앙 = 타겟 z와 ★정확히 일치★(오차 0) → ±5mm는 넉넉하다.
#   [역할] ★공중에서 튀는 순간의 통과를 직접 차단하는 유일한 조건★ — 속도 임계로는 못 막는다
#     (마찰 정지거리 계산상 v=0.10에서도 0.42mm라 속도는 '이탈 방지' 장치가 아니다).
#   ↑ P9의 표면 ±5mm는 F62에서 STRICT_PLACED_SURFACE_TOL_M로 이관(평가 보고 전용).

# ★릴리스 보상의 '실제 개도' 반영 비중★ (0=구 명령전용, 1=상태전용).
#   [사용자 지시 2026-08-09] "실제로 Gripper의 열고 닫힘에 따라 보상을 주어야 한다".
#   [문헌] ManiSkill3 StackCube dense reward: `ungrasp_reward = Σ q_finger / gripper_width`
#     — 명령이 아니라 ★실제 finger 관절 위치★에 비례해 지급한다.
#   [왜 1.0(상태 전용)이 아닌가 — F49 재발 방지] 완전히 닫힌 개체는 상태항이 0이라
#     값·gradient가 모두 죽는다(F46/F49가 정확히 그 실패였다). 명령항(sigmoid)을 다리로
#     남겨 두고 상태항을 얹는다: 닫힌 채 open을 명령하면 0.5, 실제로 열면 ~1.0 →
#     ★'놓는 시늉'보다 '실제로 놓기'가 2배 이득★이 되면서 다리는 그대로 살아 있다.
RELEASE_STATE_WEIGHT: float = 0.5

# ★F47(2026-08-03) placed 에피소드 예산 [스텝]★ — '빨리 끝내는 이득'만 제거하는 상한.
#   에피소드는 160스텝(8.0s @20Hz)이다. 80으로 두면 ★스텝 80 이전에 안착하면 총 보상이 동일★해
#   서두를 유인이 사라지고(파지를 확실히 하는 데 시간을 써도 손해가 없다), 그보다 늦으면
#   여전히 손해라 '언젠가는 해야 한다'는 압력은 남는다.
#   지급 총량 = 80 × w50 × dt(0.05) = 200 — 대안(호버 홀드 등) 대비 여전히 압도적.
# ★F49(2026-08-06) 릴리스 shaping의 smooth 전환 파라미터★
#   `a_open = sigmoid(action / τ)`. τ가 작을수록 '닫힘'에서의 지급액이 작아지지만 기울기도 얕다.
#   τ=0.3 선택 근거(액션 평형 ≈ ±1.2 — F46 크기 페널티 기준):
#     a=−1.2 → 값 0.018 (지급 0.027/스텝 ≈ placed의 1% 수준) · 기울기 0.059 ★gradient 생존★
#     a=+2.6 → 값 ≈ 1.0 (이미 열림 — 추가 gradient 불필요)
#   구 `clamp(a,0,1)`은 a≤0에서 값·기울기가 모두 0이라 ★안 놓는 개체가 영원히 못 배웠다★.
RELEASE_OPEN_TEMP: float = 0.3
# ★★2026-08-11(F57) 예산 소모 임계 상수 ★삭제★ — 상수가 없으면 어긋날 수도 없다★★
#   F52→F53→F56, 그리고 F56의 수정까지 ★네 번 연속★ '지급>0 인데 소모=0'인 영역이 생겼다.
#   마지막이 결정적이다: F56에서 임계를 0.25→0.5로 올린 것은 그 영역을 ★넓힌★ 것이었고
#   (a_cmd<0.25 → a_cmd<0.5), 정책은 즉시 그 안(액션 −0.084)에 눌러앉아 또 open 명령 0.0%였다.
#   ⇒ 지급 계수 자체를 ★action ≤ 0에서 정확히 0★으로 만들어(common/release_gate.py)
#     '지급 > 0' ⟺ '개방 명령'이 성립하게 했다. 예산은 `pay > 0`으로만 소모하므로
#     ★비교할 임계도, 어긋날 두 양도 존재하지 않는다★.

PLACED_BUDGET_STEPS: int = 100  # ★에피소드 200스텝(Teacher v2, 10 s)의 0.5 — 8.0s/80과 같은 비율 유지★


def _gripper_open_mask(env, tol_m: float = PLACED_GRIPPER_OPEN_TOL_M) -> torch.Tensor:
    """두 손가락의 실제 관절 위치가 open 목표에 도달했는지 판정한다.

    ⚠️★2026-08-09(F50) 이후 이 함수는 '진단 지표' 전용이다★ — `placed_success_mask`의
      성공 AND에서 제거됐다(사유는 PLACED_MIN_CONSECUTIVE_STEPS 위 주석 블록).
      `Phase1/BinPick_PlacedProbe.py`가 개도 달성률을 따로 보고하는 데 쓴다.
      ★성공 게이트로 되돌리지 말 것★ — 접촉에 막힌 손가락이 곧 실패로 직결된다.
    """
    if tol_m < 0.0:
        raise ValueError("gripper open tolerance must be non-negative")
    robot = env.scene["robot"]
    joint_ids = getattr(env, "_bp_gripper_joint_ids", None)
    if joint_ids is None:
        joint_ids, _ = robot.find_joints(list(GRIPPER_JOINT_NAMES))
        env._bp_gripper_joint_ids = joint_ids
    q = robot.data.joint_pos[:, joint_ids]
    return (q - GRIPPER_OPEN_M).abs().amax(dim=1) <= tol_m


def placed_success_mask(
    env,
    tol: float,
    max_speed: float = PLACED_MAX_SPEED,
    contact_n: float = 0.05,
    command_name: str = "target_pos",
    delivery_latch: torch.Tensor | None = None,
) -> torch.Tensor:
    """학습 보상과 정량 평가가 공유하는 순간 placed 정의.

    ★2026-08-09(F50) 정정★: `_gripper_open_mask`(관절 허용대)를 AND에서 제거했다.
      릴리스 게이트는 ManiSkill3 StackCube와 같이 ★¬접촉(= ¬is_grasping)★ 단독이다.
      큐브가 목표 위에 ★정지★해 있고 손가락이 ★닿아 있지 않다면★ 그것이 곧 놓인 상태다 —
      그 위에 '손가락이 완전 개방 자세인가'까지 요구하면, 접촉에 막혀 손가락이 못 벌어질 때
      물리적으로 완료된 place가 실패로 집계된다(20k run에서 실제로 그렇게 됐다).
      '실제로 열렸는가'는 `release_bias_bootstrap`의 상태항이 ★보상★으로 계속 압박한다.
    """
    pos, _ = _nearest_object(env)
    target = env.command_manager.get_command(command_name)[:, :3]
    delivered = _lifted_latch(env) if delivery_latch is None else delivery_latch
    return (
        delivered
        & (torch.norm(target - pos, dim=-1) < tol)
        & (_nearest_object_vel(env).norm(dim=-1) < max_speed)
        & (~_any_finger_contact(env, contact_n))
        # ★P9(2026-08-11) 표면 안착★ — 타겟 z = 큐브 안착 중심. 공중 통과 차단.
        # ★F62★ PLACED_SURFACE_TOL_M=0.0이면 이 항은 항상 참(비활성) — 학습 보상용 마스크.
        #   배포 등급 판정은 STRICT_* 상수로 BinPick_Eval이 따로 계산한다.
        & ((pos[:, 2] - target[:, 2]).abs() < PLACED_SURFACE_TOL_M
           if PLACED_SURFACE_TOL_M > 0.0 else torch.ones_like(delivered))
    )


# ═════════════════════════════════════════════════════════════════════════════
# ★★★F63(2026-08-13) 성공 시 에피소드 종료 + 종단 보너스★★★
#   [측정으로 확정된 원인 — force_open 프로브 시계열, n=247]
#     강제로 열었을 때 placed는 ★4스텝 뒤 발화★하는데, 그 직후 큐브가 ★쓸려나간다★:
#       t=0 dist 0.029(파지) → t=4 0.021(placed 발화) → t=8 ★0.038(존 이탈)★ → t=28 ★0.109★
#       같은 구간 finger_z −0.006 → −0.021로 계속 하강, f_net ★26~79 N★(패드가 벨트를 찍는다).
#     ⇒ 팔이 하강을 멈추지 않은 채 손가락이 벌어지며 ★자기가 놓은 큐브를 밀어낸다★.
#   [귀결] 릴리스의 실제 지급은 예산 50스텝이 아니라 ★약 4스텝(=10 보상)★이고,
#     그 대가로 grasp2+lift5+goal16 = 23을 남은 ~50스텝 동안 잃는다(약 −58).
#     ★릴리스는 실제로 음의 기대값★이다 — 가중치(F61)나 속도 임계(F62)를 고쳐도 소용없었던 이유다.
#     지급 ★단가★가 아니라 지급 ★스텝 수★가 문제였다.
#   [왜 팔이 계속 내려가는가] 큐브를 쥔 채로는 타겟 높이에 닿을 수 없다(패드가 벨트 아래로
#     내려가야 하므로 tcp_z_floor가 막는다). 그래서 goal_fine의 거리 오차가 ★영원히 0이 되지 않아★
#     하강 압력이 계속 걸린다.
#   [조치] F47이 명시했으나 채택하지 않았던 ②안을 쓴다 — ★성공 시 에피소드 종료★.
#     종료하면 '놓은 뒤 쓸려나가는 구간' 자체가 사라지고, 성공의 가치가 팔의 이후 거동과
#     무관해진다. ⚠️rsl_rl은 ★time_out에만 부트스트랩★한다(ppo.py:177 — terminated는 잔여
#     가치 0). 따라서 ★종단 보너스로 포기하는 잔여 리턴을 반드시 메워야 한다★:
#       호버 지속 가치 ≈ 1.30/step × (1−0.98^75)/0.02 = ★약 51★  ⇒ 보너스 80으로 57% 마진.
#   [blip 차단] 종료·보너스는 ★3스텝 연속★을 요구한다. 운반 중 접촉이 한두 프레임 끊기는
#     지터(F59가 관측)로 에피소드가 끝나고 80을 받는 farming을 막는다. 실측상 진짜 릴리스는
#     t=4~8에 4~6스텝의 창이 있어 3연속은 충족 가능하다.
# ═════════════════════════════════════════════════════════════════════════════
PLACED_DONE_CONSECUTIVE_STEPS: int = 3


def _placed_done_mask(env, tol: float, max_speed: float, contact_n: float,
                      command_name: str = "target_pos") -> torch.Tensor:
    """★종료·보너스 공용 술어★ — placed 조건이 PLACED_DONE_CONSECUTIVE_STEPS 연속 성립.

    보상용 `placed_released`(1스텝)와 ★다른 카운터★를 쓴다 — 보상은 5/5 성공 구성(1스텝)을
    유지하고, 에피소드를 끝내는 판정만 blip 차단을 위해 더 엄격하게 둔다.
    """
    cand = placed_success_mask(
        env, tol=tol, max_speed=max_speed, contact_n=contact_n, command_name=command_name
    ) & (_real_pick_mask(env) > 0.0)
    st = getattr(env, "_bp_done_streak", None)
    if st is None or st.shape[0] != env.num_envs:
        st = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if hasattr(env, "episode_length_buf"):
        st = torch.where(env.episode_length_buf <= 1, torch.zeros_like(st), st)
    # ★스텝당 1회만 갱신★ — 종료 매니저와 보상 매니저가 같은 스텝에 둘 다 호출한다
    #   (manager_based_rl_env.py:204 종료 → :208 보상). 갱신을 두 번 하면 카운터가 2배로 뛴다.
    step = int(getattr(env, "common_step_counter", 0))
    if getattr(env, "_bp_done_streak_step", None) != step:
        env._bp_done_streak_step = step
        st = torch.where(cand, st + 1, torch.zeros_like(st))
        env._bp_done_streak = st
    return st >= PLACED_DONE_CONSECUTIVE_STEPS


def placed_success_done(env, tol: float, max_speed: float, contact_n: float,
                        command_name: str = "target_pos") -> torch.Tensor:
    """★성공 종료★ — 큐브를 타겟에 놓고 손을 뗀 상태가 확정되면 에피소드를 끝낸다."""
    return _placed_done_mask(env, tol, max_speed, contact_n, command_name)


def placed_success_bonus(env, tol: float, max_speed: float, contact_n: float,
                         command_name: str = "target_pos") -> torch.Tensor:
    """★종단 보너스★ — 위와 같은 술어. 종료와 같은 스텝에 1회 지급된다.

    보상 매니저가 종료 매니저보다 ★뒤에★ 계산되므로(:204 → :208) 종료 스텝에서도 발화한다.
    """
    return _placed_done_mask(env, tol, max_speed, contact_n, command_name).float()


def placed_released(
    env, tol: float, max_speed: float, contact_n: float, command_name: str = "target_pos",
) -> torch.Tensor:
    """★Place 성공(릴리스) 상태보상★: ever_lifted ∧ ‖목표−객체‖<tol ∧ 정지 ∧ ★무접촉★.

    '집어 들어 옮긴(래치)' 큐브가 타겟에 '놓여 있고(정지)' 그리퍼가 '손을 뗀' 상태 — full
    pick&place의 종착 상태에 per-step 보너스(성공 조기종료 미채택 관례: 교훈② 잔여 스트림).
    ★w50 근거(리뷰 산술검증)★: 최강 대안인 '타겟 위 호버-홀드'(≈26.2/step)를 released
    (50+6+1 ≈ 57/step)가 2.2× 압도해야 하강·릴리스가 학습된다(w30은 동률 = 결함이었음).
    """
    candidate = placed_success_mask(
        env, tol=tol, max_speed=max_speed, contact_n=contact_n, command_name=command_name
    )
    streak = getattr(env, "_bp_placed_streak", None)
    if streak is None or streak.shape[0] != env.num_envs:
        streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    if hasattr(env, "episode_length_buf"):
        streak = torch.where(env.episode_length_buf <= 1, torch.zeros_like(streak), streak)
    streak = torch.where(candidate, streak + 1, torch.zeros_like(streak))
    env._bp_placed_streak = streak
    stable = streak >= PLACED_MIN_CONSECUTIVE_STEPS
    # ★pregrasp env 제외★: 공짜 latch로 시작한 env가 '내려놓기'만으로 placed(w50)를 farm하는 것 차단.
    ok = stable.float() * _real_pick_mask(env)

    # ═══ ★F47(2026-08-03) 에피소드 예산 — '빨리 끝내는 이득' 제거★ ═══
    # 문제(실측): 이 항이 ★흡수 상태의 per-step 보상★이라 1스텝 빨리 안착할수록 +w·dt 이득이
    #   계속 쌓인다. 그 결과 정책이 ★파지 신뢰성을 속도와 맞바꾼다★ —
    #   학습 내내 hold_frac이 72%(iter 6k, grasp 98.4%) → 2.8%(iter 21k, grasp 93.6%)로 단조 감소.
    #   실패 1024ep 계측: 실패는 스폰 위치와 ★무관★(반경별 3.5~5.9% 평평)하고 100%가 큐브에
    #   ★접촉★하며(중앙 6스텝, 성공과 동일) hold_frac은 ★0.00%★ — 즉 6스텝에 잡아채고
    #   어긋나면 남은 150스텝 동안 ★재시도하지 않는다★.
    # 해법: placed를 ★에피소드당 최대 BUDGET 스텝★만 지급한다. 예산이 남아 있는 한
    #   '언제 안착했는가'가 총 보상을 바꾸지 않으므로 ★서두를 이유가 사라지고★,
    #   파지를 확실히 하는 데 스텝을 쓰는 것이 손해가 아니게 된다.
    #   ⚠️farming 검산: 예산은 ★에피소드당★이고 리셋으로만 회복된다 → 반복 수확 경로 없음.
    #     '해야 할 이유'는 그대로다(BUDGET×w×dt가 대안 대비 여전히 압도적).
    if not hasattr(env, "_bp_placed_budget") or env._bp_placed_budget.shape[0] != env.num_envs:
        env._bp_placed_budget = torch.full((env.num_envs,), float(PLACED_BUDGET_STEPS), device=env.device)
    if hasattr(env, "episode_length_buf"):
        env._bp_placed_budget = torch.where(
            env.episode_length_buf <= 1,
            torch.full_like(env._bp_placed_budget, float(PLACED_BUDGET_STEPS)),
            env._bp_placed_budget,
        )
    paying = (ok > 0.0) & (env._bp_placed_budget > 0.0)
    env._bp_placed_budget = torch.where(
        paying, env._bp_placed_budget - 1.0, env._bp_placed_budget
    )
    return torch.where(paying, ok, torch.zeros_like(ok))


def grasp_attempt_bootstrap(env, radius: float) -> torch.Tensor:
    """★grasp 부트스트랩★: 최근접 객체 근처(radius)에서 그리퍼 '닫기'(action[-1]>0)를 명령하면
    소량 보상 → 파지 행동을 유도.

    [필요성] 검증 실측(2026-07-07, 256env×400iter): reach는 학습(TCP가 큐브 ~12cm까지 접근)됐으나
    lift가 미발현(0.26 베이스라인=스폰 settling, 평탄). 최종 접근+닫기+들기가 sparse해 400iter 내
    무작위 탐색이 못 뚫음. reach가 TCP를 근처까지 데려오면 이 항이 '닫기'를 유도 → 랜덤 상승으로
    lift(w15) 발화 → 이후 lift/goal이 지배. Franka Lift(단일 큐브)엔 불필요했으나 2FG7+산란 4큐브엔
    필요(직전 flawed run에서도 approach-shaping이 grasp를 부트스트랩했음).
    ★밀기 안전★: 큰 보상(lift/carry/placed)이 전부 파지 게이트(_grasped_mask)·배달 래치로 잠겨
    있어 이 항으로 바닥밀기 지름길이 열리지 않는다 — 저가중(w2) 스테핑스톤.
    """
    if not hasattr(env, "episode_length_buf"):  # 매니저 초기화 중 차원 파악 가드
        return torch.zeros(env.num_envs, device=env.device)
    _, dist = _nearest_object(env)
    near = dist < radius
    # ★2026-07-09 부호 정정★: Isaac Lab BinaryJointPositionAction은 action<0=close
    #   (binary_joint_actions.py L124-129). 이전 '>0' 판정은 sim과 반대라 'open 유지'를
    #   보상했다(게이트 프로브 실증) — 저글링·게이트 불발의 최심층 원인.
    close_cmd = env.action_manager.action[:, -1] < 0.0  # upstream 관례 = 계약 (common/action_mapping)
    # ★2026-07-10 ¬래치 게이트★: place(릴리스) 국면에서 '큐브 근처 close'가 릴리스와 싸우지
    #   않도록, 이 부트스트랩은 첫 들어올림(ever_lifted) 전까지만 켠다 — pick 국면 전용 사다리.
    return (near & close_cmd & ~_lifted_latch(env)).float()


def _release_grace_mask(env, contact_n: float, grace_steps: int) -> torch.Tensor:
    """★릴리스 유예 마스크(2026-07-31, F34)★ — 접촉 중이거나 ★최근 grace_steps 내 접촉했던★ env.

    [왜 필요한가] F33에서 release_bias를 '현재 접촉 중'으로만 게이트했더니 farming은 막혔지만
      ★릴리스 다리가 너무 짧아졌다★: 그리퍼를 여는 순간 접촉력이 0.1N 아래로 떨어져
      grasp·lift·goal(합 ~29/step)이 전부 끊기는데, release_bias도 접촉과 함께 끊겨
      ★큐브가 낙하·정착하는 동안 지급 0인 골짜기★가 생긴다. 최종 상태(placed 50+goal_fine 6 = 56)가
      호버-홀드(29)보다 크지만 골짜기를 못 건너 정책이 ★타겟 위에서 놓지 않고 호버링★한다
      (v9 iter8000 실측: carry 87.9% / placed ★1.0%★ — 배달 후 안착 실패 86.9%).

    [왜 이 형태인가] farming(F33)과 정상 릴리스를 가르는 특징은 ★마지막 접촉으로부터의 경과★다:
      정상 릴리스 = 쥐고 있다 → 연다 → 접촉 상실 → 수 스텝 내 낙하·정착 → placed 인계
      farming    = 오래 전에 떨어뜨린 큐브 옆에서 ★무한정★ 손 벌리고 있기
      → 접촉 상실 후 유한 유예만 주면 둘이 정확히 갈린다. 바닥의 큐브는 유예가 지나 지급 0.

    [grace_steps 근거] 릴리스 높이 1~3cm의 자유낙하가 0.9~1.6 제어스텝(20Hz), 정착 여유를 더해 8스텝(0.4초).
    """
    contact = _any_finger_contact(env, contact_n)
    buf = getattr(env, "_bp_last_contact_age", None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.full((env.num_envs,), float(grace_steps + 1), device=env.device)
        env._bp_last_contact_age = buf
    # 접촉 중이면 0으로 리셋, 아니면 1스텝 증가 (리셋 이벤트는 _bp_last_contact_age를 건드리지 않아도
    # 새 에피소드에서 접촉이 없으면 자연히 유예가 만료된다 — 보수적 방향)
    buf = torch.where(contact, torch.zeros_like(buf), buf + 1.0)
    env._bp_last_contact_age = buf
    # ★★2026-08-01(F41) 유예 연장은 real-pick env에만★★
    #   [사고] scratch 학습 초기(frac=1.0, 전 env pregrasp)에서 유예가 ★기술 없이★ 지급됐다:
    #     래치는 리셋 즉시 공짜, 타겟은 커리큘럼상 물체 근처, 무작위 정책은 그리퍼를 자주 연다
    #     → rel_bias 3.24(예산 20스텝의 86% 소진)인데 grasp은 0.068 (v9 동일지점 ★1.09★의 1/16).
    #   [왜 해로운가] pregrasp env는 _real_pick_mask로 placed(w50)를 못 받는다. 즉 그 구간의 유예는
    #     ★릴리스로 이어질 보상 사슬이 없는데 지급만 되는★ 잡음이고, 파지 학습(w2)을 밀어낸다.
    #   [수정] 기본 접촉항은 전 env 유지하고 ★연장(유예)만★ real-pick으로 제한한다:
    #     · pregrasp env  → 접촉 중에만 지급 = ★v9와 동일 조건★(grasp 99.4% 달성)
    #     · real-pick env → 유예까지 지급 = 낙하·정착 다리 유지(placed가 실제로 열리는 env)
    #   ⚠️2026-07-16 실패(마스킹으로 릴리스 학습 사멸)와 다르다 — 그때는 ★항 전체★를 막았고
    #     여기는 ★연장분만★ 막으므로 부트스트랩 사슬(접촉→릴리스→placed)은 보존된다.
    grace_live = (buf <= float(grace_steps)) & (_real_pick_mask(env) > 0.5)
    return (contact | grace_live).float()


def release_bias_bootstrap(env, radius: float) -> torch.Tensor:
    """★release 부트스트랩(close_bias의 릴리스 대칭)★: 배달 래치 ∧ 큐브가 타겟 radius 내일 때
    그리퍼 'open'(action[-1]>0) 명령에 소량 보상 — 릴리스 절벽의 다리.

    [필요성 — 리뷰 확정] placed는 open을 보상하는 유일한 항인데, 그리퍼 개방은
    velocity_limit_sim=0.05 m/s로 무접촉까지 ~2-3 제어스텝 걸리고 그동안 파지 게이트 스트림
    (grasp/carry)이 전부 끊긴다(sparse 절벽). 이 항이 '타겟 위에서 open 명령 유지'에 dense
    gradient를 공급 → 개방 완료 → placed(50) 개시. close_bias(¬래치, pick 국면)와 정확히 대칭.
    """
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, device=env.device)
    pos, _ = _nearest_object(env)
    target = env.command_manager.get_command("target_pos")[:, :3]
    near = torch.norm(target - pos, dim=-1) < radius
    # ★2026-07-11 smooth 전환★: binary open_cmd(action>0 indicator)는 action에 gradient를 주지 않아
    #   정책이 open을 '탐험'해야만 발화했다 → entropy로 탐험 유도 시 팔 액션까지 흔들려 pick+carry
    #   붕괴(iter6501 carry0.14·drop45). ★open action(action[-1])에 '비례' 보상★ → gradient가 policy를
    #   open 방향으로 직접 민다(팔 무관, exploration 불필요). clamp: close(a<0)=0, open은 0~1 비례.
    # ═══ ★F49(2026-08-06)★ smooth 전환 — F46과 같은 기전이 잔여 개체군에 남아 있었다 ═══
    # 구: `a_open = clamp(action, 0, 1)`.
    # 문제: 그리퍼 액션이 ★음수인 env★(= 안 놓는 개체)는 clamp가 0이라 ★gradient가 정확히 0★.
    #   F46(액션 −9.9 포화)을 크기 페널티로 고쳐 액션이 ±1~2로 묶였지만 `clamp(−1.2,0,1)=0`은
    #   여전히 0 — ★릴리스 절벽을 잇는 다리가, 정작 그 다리가 필요한 개체에만 작동하지 않는다★.
    # 실측(10k run, 512ep): 정지 93.9% 도달인데 placed 85.2%, ★래치∧릴리스 86.3%★ =
    #   13.7%가 잡은 뒤 끝까지 놓지 않는다. 그 개체들의 액션은 음수라 위 이유로 학습이 막힌다.
    # 해법: sigmoid(a/τ). τ=0.3에서 a=−1.2 → 값 0.018(지급 미미)·기울기 0.059(★gradient 생존★).
    # ★F57★ a_cmd 단독 계산 폐지 — 지급 계수는 아래 open_gain()이 일괄 산출한다
    #   (닫힘 구간 shaping 은퇴 근거는 common/release_gate.py docstring 참조).
    # ═══ ★2026-08-09(F50) 실제 개도(관절 상태) 반영 — 사용자 지시 + ManiSkill3 선례★ ═══
    #   [문제] 구 `a_open`은 ★명령★(action[-1])만 봤다. F50 실측에서 명령은 완전 개방(q_tgt=0)인데
    #     손가락이 접촉에 막혀 실제로는 16.8mm에 정체한 상태가 ★만점으로 지급★됐다 —
    #     보상이 '놓으려는 시늉'과 '실제로 놓음'을 구분하지 못했다.
    #   [문헌] ManiSkill3 `StackCube-v1`: `ungrasp_reward = Σ q_finger / gripper_width`
    #     (실제 관절 위치 비례). 우리는 open=0·close=GRIPPER_CLOSE_M이라 부호가 반대다.
    #   [효과] 막힌 개체는 상태항이 낮게 남아 지급이 절반에 그치므로, 정책이 '명령만 열어두고
    #     버티기'로 수렴하지 않고 ★실제로 벌어지는 자세·경로★를 찾도록 압박이 걸린다.
    _grip_ids = getattr(env, "_bp_gripper_joint_ids", None)
    if _grip_ids is None:
        _grip_ids, _ = env.scene["robot"].find_joints(list(GRIPPER_JOINT_NAMES))
        env._bp_gripper_joint_ids = _grip_ids
    a_state = open_state_ratio(
        env.scene["robot"].data.joint_pos[:, _grip_ids], GRIPPER_OPEN_M, GRIPPER_CLOSE_M
    )
    # ★★2026-08-10(F52) 가산 → 곱셈 정정 — 가산은 farming 스트림을 만들었다★★
    #   [사고] 구현을 `(1-W)*a_cmd + W*a_state`로 두었더니, 50mm 큐브를 물면 손가락이
    #     q≈0.012에서 버텨 a_state≈0.40이 된다 → 그리퍼를 ★깊게 닫은 채로도★
    #     a_open≈0.20이 지급됐다(0.2×30×dt = 0.301/step). 게다가 0.20 < RELEASE_OPEN_ATTEMPT
    #     (0.25)라 ★예산이 소모되지 않아 무한 스트림★이 됐다.
    #     실측 결과(10k run `2026-08-09_21-20-15`): 타겟존 open 명령 에피소드가
    #     27.5% → ★2.13%★로 급감, 그리퍼 액션 중앙 −1.919(깊은 폐합), placed 0.20%.
    #     ⇒ '놓기'를 도우려던 항이 정확히 반대로 '쥐고 버티기'를 보상했다.
    #   [정정] 곱셈으로 바꾼다: 상태항은 명령항을 ★변조(modulate)★할 뿐 스스로 지급하지 않는다.
    #     쥐고 버티기 a_cmd=0.0017 → a_open=0.0012 (사실상 0, 구 clamp 거동과 동일)
    #     개방 명령·막힘  a_cmd=1.00 a_state=0.40 → 0.70
    #     개방 명령·성공  a_cmd=1.00 a_state=1.00 → 1.00  (막힘 대비 ★1.43배★)
    #   ⚠️F42 교훈("곱셈 shaping은 원래 역할을 약화시킨다") 점검: 배수 범위가
    #     [1−W, 1] = [0.5, 1.0]로 ★0이 되지 않으므로★ F49 명령항 다리는 최소 50% 보존된다.
    # ★★2026-08-11(F57) 지급식을 '개방 경계에서 0'인 램프로 재정의★★
    #   relu(tanh(action / 2τ)) · ((1−W) + W·a_state) — 항등식 2σ(x)−1 = tanh(x/2)에 의해
    #   ★기존 sigmoid 램프를 개방 경계에서 0이 되도록 평행이동·재정규화한 것★이다.
    #   개방 구간의 곡률·스케일은 보존하면서 ★닫힘 구간의 지급만 정확히 제거★한다:
    #     action −0.084(실측 중앙) → ★0.000★   action +0.3 → 0.323   +1.0·완전개방 → 0.931
    #   상태항은 여전히 곱으로 변조만 한다(F52 유지) — 배수 [1−W, 1] = [0.5, 1.0].
    a_open = open_gain(
        env.action_manager.action[:, -1], a_state, RELEASE_OPEN_TEMP, RELEASE_STATE_WEIGHT
    )
    # ★2026-07-16 마스크 해제 — 릴리스 발판 복원(마일스톤 2500 진단)★: release_bias까지 pregrasp
    #   env에서 차단하자 릴리스 학습의 발판이 사라짐 — 기준(99%) run은 pregrasp 탐험기에 이 항으로
    #   release를 부트스트랩(iter1000에 0.61 → 2500에 13.7 복리)했는데, 마스크 run은 40~380× 작게
    #   고착(frac=0 후 600 iter 평탄 0.03). hold 고착 후엔 비례 gradient가 죽는다(clamp: hold 정책은
    #   open을 샘플하지 않음 → 기대 gradient≈0). → 이 항만 전 env 지급 복원(기준 run과 동일).
    #   farming의 실제 지급원인 placed(w50)·place_fine(w6)은 _real_pick_mask 유지 — 중간해.
    # ★★2026-07-30 접촉 게이트 추가 (F33) — 위 '중간해'의 구멍을 막는다★★
    #   [발견] _lifted_latch는 ★래치★라 한 번 들리면 계속 참이고, pregrasp env는 리셋 즉시
    #     이를 공짜로 얻는다. 타겟은 커리큘럼상 물체 근처(5~35cm)에 샘플링되므로,
    #     ★큐브를 그냥 떨어뜨리면★ 타겟 12cm 안에 착지하고 그 뒤 ★손만 벌린 채 두면★
    #     w30을 매 스텝 수확할 수 있었다. 잡고 유지하면 grasp(2)+lift(5)+goal뿐이라
    #     ★놓는 쪽이 압도적으로 이득★ → v8은 frac=1.0인데도 grasp이 iter0부터 단조 감소했다.
    #   [수정] ★현재 손가락이 큐브에 닿아 있을 때만★ 지급한다. 이 항의 취지는
    #     '쥔 것을 타겟 위에서 놓는 동작'을 잇는 것이지, '바닥의 큐브 옆에서 손 벌리고 있기'가 아니다.
    #     접촉이 끊기면 지급이 멈추고 그 뒤는 placed(w50)가 받는다 — 부트스트랩 사슬은 그대로다.
    #   ⚠️_real_pick_mask는 ★쓰지 않는다★ — 2026-07-16에 시도했다가 릴리스 학습이 죽었다
    #     (기준 99% run이 pregrasp 탐험기의 이 항으로 release를 부트스트랩했다). 위 주석 참조.
    #     접촉 게이트는 farming만 끊고 부트스트랩은 보존하는 ★정확한 절개★다.
    # ★2026-07-31(F34) 유예 구간으로 다리 복원★ — F33의 '현재 접촉만' 게이트는 farming을 막았지만
    #   낙하·정착 구간의 지급이 끊겨 릴리스를 못 배웠다(carry 87.9% / placed 1.0%).
    #   접촉 상실 후 8스텝(0.4초)까지 유지 → 낙하·정착을 덮고 placed로 인계. farming은 유예 만료로 차단.
    live = _release_grace_mask(env, 0.05, 8)
    pay = (_lifted_latch(env) & near).float() * a_open * live

    # ★★2026-07-31(F39) 예산은 '실제 지급'으로만 소모★★ — F38 구현 버그 수정
    #   [버그] 예산 차감을 _release_grace_mask 안에서 `접촉∨유예`만 보고 했다. 그런데 지급은
    #     바깥에서 a_open을 곱한 뒤에 일어난다 → ★큐브를 쥐고만 있어도 매 스텝 예산이 깎여★
    #     에피소드 초반 20스텝에 소진되고, 정작 놓을 때는 예산 0이라 지급이 없었다.
    #     실측: rel_bias 0.62(v12 정상) → ★0.027(v13, 23배 감소)★ = 릴리스 부트스트랩 사멸.
    #   [수정] pay>0 인 스텝(= 래치∧근접∧open∧접촉/유예)에만 1씩 차감한다.
    #     예산 20 = 정상 릴리스 2회분. 상한 rel_bias = 20×30/160 = 3.75로 F38 farming(8.41) 차단 유지.
    budget_steps = 20
    bud = getattr(env, "_bp_release_budget", None)
    if bud is None or bud.shape[0] != env.num_envs:
        bud = torch.full((env.num_envs,), float(budget_steps), device=env.device)
        env._bp_release_budget = bud
    if hasattr(env, "episode_length_buf"):
        bud = torch.where(env.episode_length_buf <= 1,
                          torch.full_like(bud, float(budget_steps)), bud)
    # ★★2026-08-13(F64) 소모를 '스텝당 1'에서 ★지급액 비례★로★★
    #   [배경] F57은 소모 판정을 `pay > 0` 하나로 통합해 무한 스트림 영역을 없앴다. 그 구조는
    #     옳았으나, 그러려고 open_gain을 action≤0에서 0으로 만든 것이 ★릴리스 학습을 죽였다★
    #     (설정 동일·커밋 1개 A/B: |a|rms 0.588→1.850, release_bias 7.99→0.05).
    #   [F64] 두 결함은 ★독립★이다. 지급식은 F49형(닫힘 구간에도 소액, 0에 가까울수록 큼)으로
    #     되돌려 ★액션을 개방 경계로 끌어올리는 복원력★을 복구하고, 무한 스트림은 소모량을
    #     ★지급액 그 자체★에 묶어 막는다(비교할 임계 상수가 없으면 어긋날 수 없다).
    #   ⚠️여기서 '스텝당 1' 차감을 유지하면 닫힘 미세지급(0.0034)도 1씩 태워 20스텝에 예산이
    #     소진된다 = F39가 실측한 사고(rel_bias 0.62→0.027). 그래서 비례 차감이 ★필수 동반★이다.
    #     상한은 불변: 실제 개방(≈1.0/스텝)이면 20스텝에 소진 = Episode_Reward 6.0.
    grant, bud = grant_and_consume(pay, bud)
    env._bp_release_budget = bud
    return grant


def pregrasp_align(env, std: float, near_dist: float) -> torch.Tensor:
    """★파지 정렬★: TCP가 최근접 큐브 3D 근처(near_dist)일 때 TCP가 큐브 중심 바로 위(xy 정렬)일수록
    1−tanh(‖큐브xy − TCPxy‖/std). bounded[0,1].

    reach(3D 거리)는 옆에서 접근해도 충족되지만, 이 항은 TCP가 큐브 '위'에 정렬돼야 커진다 →
    그리퍼가 큐브를 감싸고 하강·폐합하도록 유도 → 파지 성공률↑(grasp 부트스트랩이 '닫기'만
    배우고 못 잡는 것 보완). Franka Phase4 pregrasp_align 이식. 밀기 안전(파지 게이트·래치가
    큰 보상을 잠금). ★현재 weight 0(RewardsCfg 참조)★ — 재활성 대비 보존.
    """
    pos, dist = _nearest_object(env)
    tcp = bp_obs.tcp_pose_b(env)[:, :3]
    xy_off = torch.norm(pos[:, :2] - tcp[:, :2], dim=-1)
    near = (dist < near_dist).float()
    return near * (1.0 - torch.tanh(xy_off / std))


# ═════════════════════════════════════════════════════════════════════════════
# ★F45(2026-08-01)★ 팔 전용 액션 저크 페널티 — 그리퍼 차원 제외
# ═════════════════════════════════════════════════════════════════════════════
_NUM_ARM_ACTIONS: int = len(UR5E_ARM_JOINT_NAMES)  # 액션 레이아웃: [팔 6, 그리퍼 1]

# ★F43/F45 동작 페널티 단계 강화 스케줄★ (스텝 = common_step_counter + resume 오프셋)
#   1 iter = num_steps_per_env(24) 스텝. 12000 iter 예산 기준.
#   근거: 업스트림 lift 태스크가 −1e-4 → −1e-1로 올리는 패턴(CurriculumCfg). 우리는 초기값이
#   이미 업스트림의 10배(−1e-3)이고, 실측상 −1e-1을 ★한 번에★ 걸면 정지는 해결되지만
#   (전환율 7.6% → 79.5%) ★carry가 97.3% → 82.0%로 떨어졌다★ — 적응 시간이 부족하다.
#   ⇒ 중간 단(−1e-2)을 둬 운반을 유지한 채 서서히 정밀·정지를 굳힌다.
_PENALTY_INIT: float = -1e-3
# ★2026-08-05(P1-b ②) 시점 전진★: placed는 −1e-1 이전엔 ★원천 불가★이므로 이 시점이
#   학습 시작의 하한이다. pregrasp decay 종료(iter 2500) 직후로 당긴다.
#   3번째 원소 = ★최소 자력 파지 EMA★(능력 게이트, P1-b ④). 미달이면 ★지연★된다.
_PENALTY_SCHEDULE: list[tuple[int, float, float]] = [
    (19_200, -1e-2, 0.00),   # ≈ iter 800  — 1차 강화(게이트 없음)
    (62_400, -1e-1, 0.30),   # ≈ iter 2600 — 업스트림 종점. ★자력 파지 EMA ≥ 0.30 이후에만★
]


def arm_action_rate_l2(env) -> torch.Tensor:
    """업스트림 `action_rate_l2`의 ★팔 6축 한정★ 판 — 그리퍼 개폐를 벌주지 않는다.

    업스트림(envs/mdp/rewards.py `action_rate_l2`)은 전 액션 차원의 (a_t − a_{t−1})²을
    더한다. 우리 액션은 [팔 IK 6, ★그리퍼 1★]이라 그 항이 커지면 ★릴리스 자체에 벌점★이
    붙는다 — 실측(F45): 페널티를 −1e-1로 올리자 정지→placed 전환율이 95.5% → ★3.3%★로
    붕괴하고 래치∧릴리스가 24.6% → 12.1%로 줄었다. 정책이 ★큐브를 쥔 채 얼어붙는 것★이
    최적해가 된다(잡고 있으면 grasp·lift·goal 스트림이 계속 들어오고, 놓으면 벌점 + 그 스트림 상실).

    페널티의 목적은 ★팔의 부드러운 동작★(저크 억제)이지 그리퍼 개폐 억제가 아니므로
    팔 차원만 본다. 그리퍼는 이산 개폐라 저크 개념 자체가 부적절하다.
    ⚠️액션 레이아웃(팔 앞·그리퍼 뒤)은 configs/common.yaml action 계약과 ActionsCfg 배선에
      묶여 있다 — 순서를 바꾸면 이 슬라이싱도 함께 고쳐야 한다(tests/test_penalty_scope.py가 강제).
    """
    d = env.action_manager.action[:, :_NUM_ARM_ACTIONS] - env.action_manager.prev_action[:, :_NUM_ARM_ACTIONS]
    return torch.sum(torch.square(d), dim=1)


def gripper_action_l2(env) -> torch.Tensor:
    """★F46(2026-08-01)★ 그리퍼 액션 크기 페널티 — ★비가역 포화 잠김★을 막는 복원력.

    실측한 결함: 타겟존에서 정책의 그리퍼 출력(action[-1]) 중앙값이 ★−9.9★였다
    (p90 −8.5, 정책 노이즈 std 0.37). 즉 open(>0)을 ★표본추출조차 못 한다★ —
    타겟에서 open을 명령한 에피소드가 ★0.6%★.

    왜 −10까지 가는가: `BinaryJointPositionAction`은 `actions < 0`으로 ★부호만★ 본다
    (binary_joint_actions.py L140). −1이든 −10이든 효과가 같아 ★크기에 복원력이 없다★.
    `ActionTermCfg.clip`은 raw 액션이 아니라 ★처리된 관절 명령★에 걸리므로(L143-146)
    이 드리프트를 막지 못한다. 그리고 `release_bias`는 `a_open = clamp(action, 0, 1)`을
    곱하는데 −10에서 clamp는 값도 0, ★gradient도 0★ — 한번 흘러가면 돌아올 길이 없다.

    ⇒ 이것이 릴리스가 학습되지 않은 진짜 이유다. 기존 '릴리스 성공'은 의도적 개방이 아니라
      ★팔이 요동쳐 큐브를 놓친 것★이었다(무접촉을 릴리스로 판정하므로).
      F33·F34·F38·F39·F41에서 release_bias 게이팅을 다섯 번 고쳤지만
      ★그 항의 gradient가 애초에 0이었다★.

    설계: 부호만 의미가 있으므로 크기를 키울 ★이득이 없다★ → 아주 약한 페널티로도 충분히
    −1~−2 근방에 묶인다. 그 범위에서는 파지 유지에 충분히 음수이면서(노이즈 4σ 이상)
    `release_bias`가 gradient를 되찾는다. 팔 차원은 건드리지 않는다(IK 델타는 크기가 의미를 가짐).
    """
    return torch.square(env.action_manager.action[:, _NUM_ARM_ACTIONS])


@configclass
class RewardsCfg:
    """보상 — ★Franka Lift 검증 구조(reach + lift + goal-track + fine + 페널티)★ (2026-07-07 재설계).

    ★2026-07-10 Full Pick&Place(사용자 결정)★: 공중 hover 목표(내려놓기 = 배포 스크립트 shortcut)
    → ★표면 랜덤 타겟에 안착+릴리스까지 학습★. 사다리:
      reach(1) → close_bias(2,¬래치) → grasp(2) → lift(5) → goal/carry(16, 파지∧(z∨래치·0.15m)) →
      place_fine(6, 배달래치) → release_bias(30, 타겟 0.12m 내 open 비례 — 릴리스 절벽 브릿지) →
      placed(50, 래치∧3cm∧정지∧무접촉).
    hacking 차단 근거는 위 'Full Pick&Place' 블록 주석(hacking 매트릭스) 참조 — 핵심은
    ever_lifted 래치가 ★접촉 검증★(_grasped_mask)이라 밀기/저글링/슬라이딩으로 열리지 않는 것.
    ★튜닝은 §9 규칙대로 한 번에 하나씩 PROGRESS 기록★. reach std는 UR5e 홈포즈가 벨트에서 멀어
    탐색 gradient 확보 위해 0.2 유지(Franka 0.1보다 완만).

    ★성공(릴리스) 후 팔 거동은 의도적으로 무규정(2026-07-12 문헌 조사 확정)★: 고정 horizon +
    absorbing 진영의 표준(Meta-World·Gym Fetch·Isaac Lab Lift 전부 성공 후 팔 무규정 — 호버링은
    수용된 귀결)이고, 실기 홈 복귀는 배포 상태기계 몫(Deploy_Real.py go_to_rest 패턴). placed의
    무접촉 요구가 재접촉·쳐냄을 구조적으로 차단(robosuite·RGB-Stacking과 동일 장치). ⚠️ 예비
    카드: 호버가 실질 문제로 확인되면 ManiSkill3식 placed 게이팅 static_reward
    (1−tanh(5·‖qvel‖), 그리퍼 관절 제외) 추가 — 계약(obs/action) 무변경 fine-tune 가능.
    """

    # ★2026-07-28 std 0.2 → 0.35 — 홈 포즈 교체(f9455d7)에 대한 ★기하 보정★(새 실험 아님)★
    #   홈 TCP가 (0.492,0.133,0.262)→(0.301,0.133,0.525)로 바뀌며 TCP–스폰큐브 평균거리가
    #   0.331m → 0.576m로 늘어, std=0.2에서 tanh가 포화(d/std 1.66 → 2.88)했다.
    #   실측(2만 표본): reach 평균 0.0829 → ★0.0070(-92%)★, 셰이핑 기울기도 동반 붕괴 →
    #   자력 파지 env의 유일한 dense 신호가 페널티(action_rate·joint_vel)와 동급이 된다.
    #   std=0.35면 d/std=1.65로 ★검증된 run의 비율(1.66)을 복원★하고 reach 평균 0.0741이 된다.
    #   ⚠️ 이 값은 홈 기하 종속 — 홈을 다시 바꾸면 d_mean/std 비를 재보정할 것.
    reach = RewTerm(func=reach_nearest_object, params={"std": 0.35}, weight=1.0)
    # ★2026-07-08: align·grasp 부트스트랩 weight 0으로★ — 역방향 커리큘럼(pre-grasp)이 파지를
    #   제공하므로 부트스트랩 불필요하고, 오히려 '큐브 근처서 정렬·닫힘 유지'를 보상해 hold-in-place
    #   국소최적(운반 안 함)을 만든다(검증: pre-grasp에서 lift 0.44·goal 평탄). reach+lift+goal만
    #   남기면 goal 경사가 잡은 큐브의 '운반'을 유도(carry가 hold보다 +10 보상). 파지 자체는
    #   pre-grasp + frac감소 curriculum이 담당. (부트스트랩 term은 재활성 대비 남겨두고 weight만 0.)
    align = RewTerm(func=pregrasp_align, params={"std": ALIGN_STD, "near_dist": ALIGN_NEAR}, weight=0.0)
    # ★★STAGE 3(2026-07-09) 파지 게이트 재설계★★ — 프로브 확정: 높이 게이트만으로는
    #   저글링(무파지 타격-비행)이 lift/goal을 해킹(들림 중 접촉 0%). lift/goal/goal_fine에
    #   파지 성립(_grasped_mask: 양손접촉>0.1N ∧ 상대속도<0.2m/s — 프로브와 동일 정의) 게이트를
    #   추가하고, grasp 항은 '파지 성립 자체' 스테핑스톤(w2)으로 교체(reach→grasp→lift 사다리).
    #   임계 근거: 접촉 0.1N = 프로브 판정 임계(30g 큐브 지지엔 ~0.3N 필요 — 여유), 상대속도
    #   0.2 m/s = 자유비행 중앙값 1.4 m/s 대비 7배 마진(파지 시 ≈0).
    grasp = RewTerm(
        func=grasped_gate, params={"force_n": 0.1, "rel_speed_mps": 0.2}, weight=2.0)
    # ★close 유지 부트스트랩 재활성(2026-07-09)★ — pregrasp를 켜도 게이트가 안 열린 원인:
    #   그리퍼 채널은 매 스텝 정책 명령이라, 미학습 정책이 ~50% 확률로 open → pre-grasped
    #   상태가 1-2스텝에 증발(검증: frac 0.75에서 grasp 0.0001). 이 항이 '근처(radius)에서
    #   close 명령'에 dense gradient를 공급 → close 유지 → 접촉 게이트 개방의 다리.
    #   과거 부작용(hold-in-place 국소최적)은 무게이트 lift 시대의 것 — 지금은 게이트된
    #   lift(15)+goal(16)이 근처-close(2)를 압도하므로 재발 없음.
    # ★★F67(2026-08-18) close_bias 폐지 — w2.0 → 0★★
    #   이 항은 `near ∧ 그리퍼 닫기 명령`에 지급한다. 그런데 F67에서 폐합 시점은 수퍼바이저가
    #   정하고 ★정책의 그리퍼 차원은 아예 사용되지 않는다★. 사용되지 않는 차원에 지급을
    #   유지하면 정책이 제어하지 않는 출력에 gradient가 흘러 ★순수 잡음★이 된다.
    #   (더 나쁘게는, '닫기 명령'을 내도록 학습시키는데 그 명령이 무시되므로 학습 신호가
    #    거짓이 된다 — 정책은 보상을 받지만 환경은 반응하지 않는다.)
    # ★F68★ 그리퍼 제어가 정책으로 돌아왔으므로 F66 값(2.0)을 복원한다.
    #   (F67-b가 0으로 내린 것은 '그리퍼 차원 미사용' 전제였고, 그 전제는 F68에서 사라졌다.)
    close_bias = RewTerm(
        func=grasp_attempt_bootstrap, params={"radius": GRASP_RADIUS}, weight=2.0)
    # ★2026-07-10 Full Pick&Place 재배선 + 적대적 리뷰(9에이전트) 산술검증 반영★
    #   [치명 결함 교정] lift w15·placed w30이면 '타겟 위 z≈0.041 호버-홀드'가
    #   1+2+15+14.6+3.6 ≈ 36.2/step으로 released(30+6+1 ≈ 37)와 동률 → 릴리스 미학습.
    #   → ★lift 15→5, placed 30→50★: hover ≈ 26.2/step ≪ released ≈ 57/step (2.2× 마진).
    #   lift는 carry(16, 파지 즉시 개방)가 사다리를 보강하므로 5로도 픽업 부트스트랩 유지.
    lift = RewTerm(
        func=lift_grasped,
        params={"minimal_height": MIN_LIFT, "force_n": 0.1, "rel_speed_mps": 0.2},
        weight=5.0,
    )
    #   goal: z게이트 → (z ∨ (래치∧타겟0.15m))게이트(carry — 표면 타겟 하강 연속, 드래그 차단).
    #   goal_fine → place_fine(배달 래치 게이트, 릴리스 후에도 흐름).
    #   placed(w50): 래치 ∧ 타겟 3cm ∧ 정지 ∧ 무접촉(릴리스) — full place 종착 상태.
    #   release_bias(w30, radius 0.12): 타겟 위 open '비례' 부트스트랩(릴리스 절벽 다리 —
    #     close_bias와 대칭, 2026-07-11 smooth 전환 — 아래 release_bias 주석 참조).
    # ★2026-07-28 std 0.3 → 0.55 — 홈 교체 + 타겟범위 확대에 대한 ★기하 보정★(새 실험 아님)★
    #   pregrasp 역커리큘럼 env는 큐브를 홈 파지점(_PREGRASP_GRIP)에 '이미 잡은' 상태로 시작하므로
    #   그 env의 사실상 유일한 학습 신호가 이 carry 항이다. 홈이 26cm 높아지고 타겟 박스가 2.6배
    #   넓어져 큐브–타겟 평균거리가 0.307m → 0.580m가 됐다.
    #   실측(2만 표본): goal 평균 3.795 → ★0.689(-82%)★ → 위치 무관 보상(reach+grasp+lift=8)의
    #   비중이 68%→92%로 올라 '잡고 가만히 있기' 국소최적(:646-648에 기록된 실패 모드)을 강화한다.
    #   std=0.55면 d/std=1.05로 ★검증된 비율(1.02)을 복원★하고 goal 평균 3.501이 된다.
    #   ⚠️ 릴리스 유인은 불변: 타겟 도달(d=0)에서 tanh(0)=0이라 std와 무관하게 goal=16이고,
    #      placed(w50)가 여전히 압도한다(hover≈30 < released≈57) — farming 가드 영향 없음.
    goal = RewTerm(
        func=goal_distance_carry,
        params={"std": 0.55, "minimal_height": MIN_LIFT, "force_n": 0.1,
                "rel_speed_mps": 0.2, "command_name": "target_pos", "latch_radius": 0.15},
        weight=16.0,
    )
    goal_fine = RewTerm(
        func=place_fine,
        params={"std": 0.06, "command_name": "target_pos"},
        weight=6.0,
    )
    # ★2026-07-11 eval 재진단 → release_bias 강화★: 완주 정책(model_2499) 정량 eval —
    #   grasp 80.8%·carry 71.7%(pick+carry 작동)·★placed 1.8%(릴리스 실패)★. ★근본 발견: 파지시
    #   평균 목표거리 5.6cm > release_bias radius 5cm → 정책이 발화 구역에 거의 못 들어감★
    #   (release_bias 실측 ≈0.02 = 사실상 죽은 항). 이전엔 weight만 2→18 올리고 radius 5cm를 안
    #   건드려 무의미했다. ★radius 0.05→0.12★로 정책 위치(5.6cm)를 확실히 덮고(carry 8cm 범위도
    #   포함), ★weight 18→30★으로 open 즉시 fine 4.1+30=34.1 ≫ 홀드 21(마진 13, 이전 1.1)이라
    #   탐험이 한 번 open하면 강하게 강화 → placed(50) shaping. 3D 거리라 공중(z 높음)선 여전히
    #   미발화 = 표면 근처 하강 후 open만 보상(성급한 공중 릴리스 방지).
    # ★★★F65(2026-08-15) release_bias 지급 폐지 — w30 → 0 (변인 1개)★★★
    #   [실측] model_9700(F64 run, iter 9705에서 중단) 프로브 256ep, 학습과 동일 판정 함수:
    #     · latch 96.9% · ★zone(타겟 3cm 이내) 87.5%★ · zone&settled 66.8%
    #     · ★zone&open 0.00%★ · placed 0.39% · ★타겟존 내 그리퍼 액션 최댓값 −0.026★
    #     대조군 force_open_smart(팔은 같은 정책, 개방만 강제): ★placed 64.8%★,
    #     강제 개방 후 placed까지 ★4스텝★. ⇒ 환경·판정·기하·운반은 전부 정상이고,
    #     결함은 ★릴리스 채널 단 하나★다(프로브 판정 지침의 '보상 지형 문제' 분기).
    #   [원인] F64가 넣은 두 장치가 ★서로를 무력화★한다.
    #     ① 복원력(닫힘 구간에도 sigmoid 지급, 0에 가까울수록 큼)이 액션을 개방 경계로 민다.
    #     ② 안전장치(소모 = 지급액 비례)는 정책이 ★깊은 닫힘★에 머문다는 전제에서만 성립한다.
    #     ①이 성공하는 순간 ②의 전제가 깨진다. 실측 −0.026에서 지급 0.2392/step =
    #     F64 docstring이 가정한 −1.7(0.0034)의 ★139배★ ⇒ 예산 20이 ★84스텝★에 소진.
    #     검산: 이론상한 20×30×0.05 = 30.0 vs 실측 release_bias 29.25 = ★예산의 97.5%를
    #     그리퍼를 한 번도 열지 않고 수확★. `grant_and_consume`은 '지급>0 ∧ 소모=0' 밴드를
    #     정확히 없앴지만, ★예산은 게이트가 아니라 수량★이라 닫힘 수확을 느리게 할 뿐 못 막는다.
    #   [일반 규칙] ★액션만의 함수인 지급은 어떤 형태든 수확 가능하다★ — 0으로 두면 gradient가
    #     죽고(F57), 0보다 크게 두면 경계에서 farming한다(F49·F64). F52·F53·F56·F57·F64가
    #     이 한 항의 상수를 다섯 번 고쳐 모두 실패했다. 지급식 조정으로는 닫히지 않는다.
    #   [해소] 릴리스 채널을 ★지급(payout)에서 페널티(penalty)로 전환★한다 — 페널티는 아래로
    #     유계라 수확할 대상 자체가 없다. 복원력은 이미 있는 `gripper_action` L2(w−0.05)가
    #     담당한다(실측 rms|a|≈0.57 — F46의 중앙 −9.9 표류와 무관한 상태). 릴리스 구동은
    #     placed(w50) + place_success_bonus(w1600)에 맡긴다: 종단 보너스 1600×0.05 = ★80★이
    #     잔여 hold 스트림(1.363/step, γ=0.98 할인 최대 54.6)을 ★모든 시점에서★ 압도한다.
    #   [문헌] ManiSkill3 StackCube·IsaacLab Franka Lift 모두 binary 그리퍼 액션을 shaping하지
    #     않고 객체 중심 보상 + 종단 성공으로 릴리스를 학습시킨다. release_bias는 그 표준에서
    #     벗어난 항이었다(F50 주석이 이미 ManiSkill3의 ¬접촉 단독 게이트를 근거로 인용한다).
    #   ⚠️조기 게이트: scratch에서 ★iter 3000에 zone&open > 5%★. 현재 0.00%라 즉시 갈린다.
    #     미달이면 이 방향(페널티 전환)이 틀린 것이므로 지급식 재도입이 아니라 재분석으로 간다.
    #   ⚠️weight=0.0이면 RewardManager가 항 자체를 건너뛴다(reward_manager.py:146) —
    #     `_bp_release_budget`은 이 함수 내부 전용이라 다른 항에 영향이 없다(참조처 3곳 전부 내부).
    release_bias = RewTerm(
        func=release_bias_bootstrap, params={"radius": 0.12}, weight=0.0)
    placed = RewTerm(
        func=placed_released,
        params={"tol": 0.03, "max_speed": PLACED_MAX_SPEED, "contact_n": 0.05,
                "command_name": "target_pos"},
        weight=50.0,
    )
    # ★★F63(2026-08-13) 종단 보너스★★ — 성공 종료로 포기하는 잔여 리턴을 메운다.
    #   [크기 근거] 호버 지속 가치 ≈ 1.30/step × (1−γ^75)/(1−γ), γ=0.98 ⇒ ★약 51★.
    #     보너스 = weight × func × dt = 1600 × 1.0 × 0.05 = ★80★ (마진 57%).
    #   ⚠️rsl_rl은 terminated에 부트스트랩하지 않으므로(ppo.py:177 time_outs 전용)
    #     이 보너스가 없으면 ★조기 종료가 오히려 손해★가 되어 정책이 성공을 회피한다.
    #   에피소드당 1회만 발화한다(발화 즉시 종료) — farming 경로 없음.
    # ★Teacher v2(2026-09-18) 1600 → 3000★: γ 0.98→0.99·에피소드 200스텝에서 포기 리턴
    #   상한 ≈ 1.30/step × (1−0.99^175)/0.01 ≈ 108 → ×1.2 마진 = 129 < 보너스 3000×0.05 = 150.
    # ★F92(2026-09-20) 3000 → 4000★ — 위 산식은 두 가지가 틀렸다(리뷰 RT-2).
    #   ① time_out은 부트스트랩된다(rsl_rl: rewards += γ·V·time_outs) → 호버 가치는 유한합이
    #     아니라 r/(1−γ)로 수렴한다. ② r_hover는 1.30이 아니라 RewardsCfg 가중치 합이다:
    #     reach 1 + grasp 2 + lift 5 + goal 16 + goal_fine 6 = 30/step × dt 0.05 = ★1.50/step★
    #     (타겟 위 저공 호버, 릴리스 없이도 전부 지급된다) → 150. ×1.2 마진 = 180 < 4000×0.05 = ★200★.
    #   run1/run2(3000)은 마진 1.03배로도 99.6%를 냈지만, 편측 파지(F91)로 무효가 됐고 재학습이
    #   필요하므로 이번에 바로잡는다. tests/test_teacher_contract.py가 같은 산식으로 고정한다.
    place_success_bonus = RewTerm(
        func=placed_success_bonus,
        params={"tol": 0.03, "max_speed": PLACED_MAX_SPEED, "contact_n": 0.05,
                "command_name": "target_pos"},
        weight=4000.0,
    )
    # ★★F59(2026-08-11) 릴리스 착지 부트스트랩 — P9가 닫은 낙수 채널을 복원한다★★
    #   판정(placed, w50)은 엄격한 채로 두고, ★97.7% run이 실제로 썼던 판정식★을 별도 항으로
    #   되살린다: speed<0.10 · ★연속 스텝 요구 없음★ · ★표면조건 없음★.
    # ★★F62(2026-08-12) release_land 제거 — placed가 다시 느슨해져 ★완전 중복★이 됐다★★
    #   F59가 이 항을 만든 이유는 "성공 판정은 엄격하게 두고 부트스트랩만 복원"이었는데,
    #   F62에서 학습 보상용 placed 자체를 5/5 성공 구성(속도 0.10·1스텝·표면조건 없음)으로
    #   되돌렸으므로 술어가 글자 그대로 같아졌다. 둘을 함께 두면 같은 사건에 100을 지급해
    #   ★성공 run(50)과 다른 구성★이 된다 — 복원의 의미가 사라진다.
    #   배포 등급 판정은 STRICT_* 상수로 `BinPick_Eval`이 별도 계산·보고한다.
    # 페널티(동작함 — upstream 검증: envs/mdp/rewards.py).
    # ★2026-07-09 "부드러운 동작" 강화(사용자 결정)★: 팔이 빠르게 스윙해 3cm 큐브를 벨트 밖으로
    #   쳐내 에피소드가 조기 종료(~40스텝)됐다. action_rate(저크)·joint_vel(속도) 페널티를 ×10 올려
    #   천천히 정밀하게 top-down 접근·파지하도록 유도(하드 속도캡 아님 → 학습된 거동이라 실기 전이).
    #   ik scale도 0.5→0.2로 낮춰(위 _IK_SCALE) 스텝당 TCP 이동을 줄임.
    # ★F43(2026-08-01) 동작 페널티 상향 — 업스트림 lift 태스크의 커리큘럼 종점 −1e-1★
    #   근거: Isaac Lab v2.3.2 manipulation/lift/lift_env_cfg.py CurriculumCfg가
    #   action_rate·joint_vel을 −1e-4 → ★−1e-1★(1000배)로 올린다. 우리는 −1e-3 고정 =
    #   업스트림 종점의 1/100이라 '정지'에 대한 유인이 보상의 0.6%뿐이었다(실측:
    #   action_rate −0.083 / joint_vel −0.017 vs goal +11.2).
    #   실측 효과(600 iter resume): 팔 관절속도 4.23 → ★0.043 rad/s★,
    #   래치∧3cm→정지 전환율 7.6% → ★79.5%★. '팔이 안 멈추는' 문제는 이걸로 해소됐다.
    #
    # ★F45(2026-08-01) 그리퍼를 페널티에서 제외★ — 위 −1e-1 적용 시 ★릴리스가 죽었다★
    #   (정지→placed 전환율 95.5% → 3.3%, 래치∧릴리스 24.6% → 12.1%).
    #   원인: 액션 7차원의 ★마지막이 그리퍼★인데 `action_rate_l2`는 전 차원의 (a_t−a_{t−1})²을
    #   더한다 → ★릴리스(그리퍼 값 반전)라는 태스크 필수 행동에 직접 벌점★이 붙었다.
    #   `joint_vel_l2`도 asset_cfg가 SceneEntityCfg("robot")이라 ★손가락 관절★을 포함했다.
    #   페널티의 목적은 ★팔의 부드러운 동작★이지 그리퍼 개폐 억제가 아니다 — 이건 튜닝이 아니라
    #   결함이다. 두 항 모두 ★팔 6축으로 한정★한다.
    #   ⚠️검증 후 CurriculumCfg(modify_reward_weight)로 옮길 것 — 초기부터 −1e-1이면
    #     도달·파지 학습 자체가 느려질 수 있어 업스트림도 커리큘럼으로 켠다.
    action_rate = RewTerm(func=arm_action_rate_l2, weight=_PENALTY_INIT)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=_PENALTY_INIT,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(UR5E_ARM_JOINT_NAMES))},
    )
    # ★F46★ 그리퍼 액션 크기 복원력 — 부호만 의미가 있는 차원이 −10까지 흘러가
    #   release_bias의 gradient를 0으로 만드는 ★비가역 잠김★을 막는다(실측 중앙값 −9.9).
    #   ★가중치 근거(평형점 분석 + F46 실측)★: w=−1e-2는 방향은 맞았으나 ★너무 느렸다★ —
    #   600 iter에 액션이 −9.9 → −6.29로만 회복(−1.5까지 1000 iter 이상 더 필요).
    #   정책은 두 한계비용을 저울질한다: 크기 벌점 w·a²·dt  vs  노이즈로 우연히 열려
    #   큐브를 놓칠 손실 (grasp+lift)·dt·Φ(−a/σ), σ=0.37(실측 노이즈 std).
    #   w=−5e-2에서 두 한계비용이 ★a≈1.1~1.3★에서 만난다 — 파지는 안전히 닫힌 채
    #   (스텝당 개방확률 0.1~0.5%) 에피소드당 한 번은 open을 표본추출해 ★탐색이 살아난다★.
    #   그 지점의 비용은 0.0056/스텝으로 파지 income(0.1/스텝)의 5.6% → 파지 무영향.
    # ★F68★ 정책이 다시 그리퍼를 제어하므로 F46('액션이 −9.9까지 표류해 개방을 표본추출조차
    #   못 함') 방어가 다시 필요하다 — F66 값(−0.05) 복원.
    gripper_action = RewTerm(func=gripper_action_l2, weight=-5e-2)


# ═════════════════════════════════════════════════════════════════════════════
# Terminations
# ═════════════════════════════════════════════════════════════════════════════
def any_object_dropped(env, minimum_height: float, num_objects: int = NUM_OBJECTS) -> torch.Tensor:
    """객체가 벨트 아래로 떨어지면 종료 (벨트 밖 낙하) — 순수 로직, 구현됨.

    벨트는 벽이 없어(평면) 밀어내면 가장자리 밖으로 떨어져 바닥(z=TABLE_HEIGHT−STAND_HEIGHT)까지
    낙하 → 이 임계(TABLE_HEIGHT−0.05)를 통과해 종료. 밀기를 회피시키는 물리적 근거 중 하나.
    """
    dropped = torch.stack(
        [env.scene[name].data.root_pos_w[:, 2] < minimum_height for name in object_names(num_objects)],
        dim=0,
    )
    return dropped.any(dim=0)


@configclass
class TerminationsCfg:
    """종료 — time_out + 객체 벨트 밖 낙하. ★성공 조기종료는 미채택★(교훈②: dense 보상 + 성공
    종료 = 잔여 스트림 포기 모순. gamma 0.98에서 일회성 종료 보너스로 상회 어려움).
    ★현행(2026-07-10 Full Pick&Place)★: 안착+릴리스 성공은 종료가 아니라 per-step 상태보상
    placed_released(w50)로 유도 — 성공 상태에 머무는 것이 잔여 스트림을 최대화하는 흡수 상태다.
    조기종료가 필요해지면 압도적 종료 보너스와 함께 켤 것(Franka Place-End 패턴 참조)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # ★★F63(2026-08-13) 성공 종료★★ — 놓고 손 뗀 상태가 3스텝 확정되면 에피소드를 끝낸다.
    #   [측정 근거] 릴리스 후 팔이 하강을 계속해 ★자기가 놓은 큐브를 밀어낸다★
    #     (force_open 프로브: dist 0.021@t4 → 0.038@t8 → 0.109@t28, f_net 26~79N).
    #     그래서 placed 지급이 예산 50스텝이 아니라 ★4스텝★에 그쳐 릴리스가 음의 EV였다.
    #     종료하면 그 구간 자체가 사라져 성공의 가치가 팔의 이후 거동과 무관해진다.
    #   ⚠️time_out=False(기본) — ★진짜 종료★다. 잔여 가치는 위 place_success_bonus가 메운다.
    placed_success = DoneTerm(
        func=placed_success_done,
        params={"tol": 0.03, "max_speed": PLACED_MAX_SPEED, "contact_n": 0.05,
                "command_name": "target_pos"},
    )
    object_dropped = DoneTerm(
        func=any_object_dropped, params={"minimum_height": TABLE_HEIGHT - 0.05}
    )


# ═════════════════════════════════════════════════════════════════════════════
# Curriculum — ★역방향 커리큘럼: pre-grasp 비율 감소 (2026-07-08)★
#   좁은 그립창 파지탐색을 우회: 처음엔 frac=0.75(대부분 잡힌 상태로 시작 → hold+운반 학습),
#   스텝 스케줄로 frac→0 감소시켜 점차 실제 파지를 요구. events_dr._pregrasp_reset이 소비.
# ═════════════════════════════════════════════════════════════════════════════
@configclass
class CurriculumCfg:
    pregrasp_frac = dr.make_pregrasp_curriculum()
    # ★F43/F45★ 동작 페널티 단계 강화 — 팔이 타겟 위에서 멈추도록(placed의 '정지' 조건).
    #   업스트림 lift 태스크의 modify_reward_weight 패턴을 다단계로 일반화(events_dr 참조).
    penalty_action_rate = dr.make_penalty_schedule_curriculum("action_rate", _PENALTY_SCHEDULE)
    penalty_joint_vel = dr.make_penalty_schedule_curriculum("joint_vel", _PENALTY_SCHEDULE)
    # (place-side 역커리큘럼 place_frac은 place bootstrap 폐기(2026-07-11, farming)와 함께 삭제 —
    #  릴리스 학습은 BINPICK_STAGE 2단계 + release_bias smooth가 담당. README.md 결정 로그 참조.)


# ═════════════════════════════════════════════════════════════════════════════
# Events — 기본 리셋만 정적 선언, 나머지(산란/DR)는 __post_init__에서 부착
# (객체 수가 configs/sim.yaml 소스라 동적 — EventManager는 cfg.__dict__를 순회
#  (event_manager.py L337 검증)하므로 인스턴스 setattr로 정상 등록된다)
# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# ★자력 파지 능력 EMA (감사 P7-2, 2026-08-06)★ — 보상 경로와 완전 분리
# ═════════════════════════════════════════════════════════════════════════════
GRASP_EMA_INTERVAL_S: float = 0.25    # 5 제어스텝(20Hz)마다 갱신
GRASP_EMA_ALPHA: float = 0.005        # 시정수 ≈ 200회 갱신 ≈ 50 s (구 구현의 1000스텝과 등가)


def _update_grasp_ema(env, env_ids, force_n: float, rel_speed_mps: float, alpha: float) -> None:
    """`env._bp_grasp_ema` 갱신 — 페널티 커리큘럼의 ★능력 게이트★가 읽는 값.

    ★왜 이벤트인가(P7-2)★: 구현 초판은 이 갱신을 `grasped_gate`(보상 함수) 안에 넣었다.
    그런데 `RewardManager.compute`는 `weight == 0.0`인 항을 ★건너뛴다★
    (vendored reward_manager.py L145-148 확인). grasp 가중치를 0으로 두는 실험을 하는 순간
    EMA가 얼어붙고 게이트가 ★영구 차단★된다 — 보상 설정이 커리큘럼을 조용히 망가뜨리는
    ★숨은 결합★이었다. 이벤트는 가중치와 무관하게 항상 실행되므로 그 결합이 끊긴다.

    ⚠️pregrasp 보조 env는 제외하고 ★real-pick env에서만★ 센다 — 그래야 '자력' 능력이다.
    """
    del env_ids  # 전 env 대상(전역 통계)
    m = _grasped_mask(env, force_n, rel_speed_mps)
    real = _real_pick_mask(env) > 0.5
    n_real = int(real.sum())
    if n_real == 0:
        return
    rate = float((m & real).sum()) / n_real
    prev = float(getattr(env, "_bp_grasp_ema", 0.0))
    env._bp_grasp_ema = (1.0 - alpha) * prev + alpha * rate


def make_grasp_ema_event() -> EventTerm:
    """일정 주기로 자력 파지 EMA를 갱신하는 이벤트(보상 가중치와 무관하게 항상 실행)."""
    return EventTerm(
        func=_update_grasp_ema,
        mode="interval",
        interval_range_s=(GRASP_EMA_INTERVAL_S, GRASP_EMA_INTERVAL_S),
        is_global_time=True,
        params={"force_n": 0.1, "rel_speed_mps": 0.2, "alpha": GRASP_EMA_ALPHA},
    )


@configclass
class EventCfg:
    """기본 리셋 이벤트 — DR 항은 env cfg __post_init__가 mdp/events_dr.py로 부착."""

    reset_all = dr.make_reset_scene_event()
    # ★P7-2★ 자력 파지 EMA — 능력 게이트의 입력. 보상 가중치와 무관하게 항상 갱신된다.
    grasp_ema = make_grasp_ema_event()


# ═════════════════════════════════════════════════════════════════════════════
# Env cfg 본체
# ═════════════════════════════════════════════════════════════════════════════
@configclass
class BinPickTeacherEnvCfg(ManagerBasedRLEnvCfg):
    """Phase1 teacher: 특권 상태 관측, 카메라 없음, num_envs 4096.

    Task ID: "Isaac-BinPick-UR5e-Teacher-v0" / experiment: ur5e_binpick_teacher
    (gym 등록은 학습 스크립트 담당 — 이 모듈은 cfg만 제공).
    """

    scene: BinPickSceneCfg = BinPickSceneCfg(num_envs=_TEACHER_NUM_ENVS, env_spacing=3.0)
    observations: TeacherObservationsCfg = TeacherObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()  # pre-grasp frac 감소 (enable_pregrasp=True일 때만 활성)

    # ★2026-07-09 pregrasp 재활성(STAGE 3 후속)★ — 게이트 보상에서 자력 grasp 탐색이 절벽
    #   (검증: 550 iter 동안 reach 0.86 포화·에피소드 만점 생존인데 grasp 발화 0). pregrasp가
    #   파지 성립 상태를 직접 경험시켜 게이트 스트림(grasp/lift/goal)을 개방하고, frac 0.75→0
    #   커리큘럼이 자력 잡기로 점진 전이시킨다. 이전 실패 원인 2개는 제거된 상태라 안전:
    #   ① hold-home 함정 = 절대관절 home-bias였음(현재 task-space ik_rel)
    #   ② pre-grasped 유지 실패 = 상대관절 중력 sag였음(현재 팔 중력 disable)
    #   (2026-07-08의 비활성 사유는 절대관절 시대의 기록 — git 히스토리 cffcf7e 참조.)
    enable_pregrasp: bool = True

    def __post_init__(self):
        # ★★요소3: sim dt ↔ 실기 control rate 정합 — 유일하게 허용된 배선★★
        # decimation은 어디에도 저장하지 않는다. configs/common.yaml의
        # physics_hz/control_hz에서 common.timing.derive_decimation으로 파생하며,
        # 정수로 나눠떨어지지 않으면 여기서 ValueError → 학습 시작 자체가 막힌다.
        # (sim.dt를 손으로 바꾸고 decimation을 안 바꾸는 사고를 원천 봉쇄)
        self.sim.dt = 1.0 / _PHYSICS_HZ
        self.decimation = derive_decimation(_PHYSICS_HZ, _CONTROL_HZ)
        self.sim.render_interval = self.decimation
        self.episode_length_s = EPISODE_LENGTH_S
        print(
            f"[cfg] ★요소3 정합★ physics={_PHYSICS_HZ}Hz control={_CONTROL_HZ}Hz "
            f"→ sim.dt={self.sim.dt:.6f}s decimation={self.decimation}"
        )

        # 산란 객체 (configs/sim.yaml num_objects)
        add_scatter_objects(self.scene, NUM_OBJECTS)

        # ★STAGE 3: 손가락별 ContactSensor 부착★ — 파지 게이트 보상(_grasped_mask)의 입력.
        #   setattr 부착의 유효성은 프로브 실행으로 검증됨(2026-07-09: 힘 값 실측).
        for _name, _cfg in make_finger_contact_sensor_cfgs().items():
            setattr(self.scene, _name, _cfg)

        # 이벤트: 객체 산란 + 물리 DR(요소2 — dr.yaml physics)
        for name, term in dr.make_object_scatter_events(NUM_OBJECTS).items():
            setattr(self.events, name, term)
        for name, term in dr.make_physics_dr_terms(NUM_OBJECTS).items():
            setattr(self.events, name, term)
        # ★Teacher v2★ 인식 모델 파라미터(오차·편향·검출률·최소거리)를 에피소드마다 재샘플.
        #   Eval cfg도 상속한다 — 평가는 배포 대표 조건(추정치 관측)으로만 한다.
        self.events.perception_resample = bp_perc.make_perception_resample_event()

        # ★★2026-08-11 사용자 지시★★ 큐브 색을 ★빨강/파랑 중 하나★로 (configs/sim.yaml 팔레트).
        #   mode="startup" — 기동 시 env마다 1회 정해지고 그 run 동안 유지된다.
        #   ⇒ 씬에 빨강·파랑 큐브가 섞여 나온다. 리셋마다 바꾸지 않는 이유:
        #     ① 색은 ★학습에 영향이 없다★ — teacher는 카메라가 없고 student는 depth 관측이다.
        #     ② per-env USD 쓰기라 4096 env에서 리셋마다 돌리면 비용이 크다.
        #   ⚠️upstream randomize_visual_color를 쓰지 않는다 — 그것은 replicate_physics=False를
        #     강제해 4096 env에서 VRAM·기동시간이 폭증한다(그래서 vision cfg의 128 env 전용이다).
        for name, term in dr.make_object_color_choice_events(NUM_OBJECTS).items():
            setattr(self.events, name, term)

        # ★2단계 학습 flag (BINPICK_STAGE 환경변수, 2026-07-11) — release_bias farming 딜레마 회피★.
        #   [배경] release_bias 강하면 '큐브 잡힌 채 시작'하는 보조 env(부트스트랩·pregrasp)에서 open
        #   farming으로 pick 붕괴, 약하면 exploration trap. 5회 실패 → 사용자 결정: 2단계 분리.
        #   ★stage 1★: place 보상(release_bias·placed) weight 0 → pick+carry 전용(farming 원천 무의미).
        #   ★stage 2★: pregrasp off → '잡힌 채 시작' 원천 제거, 1단계 완성 정책 resume + release_bias만
        #   fine-tune(farming 불가·자력 pick+carry 유지). 미설정=기본(통합, 단일학습 실험용).
        _stage = os.environ.get("BINPICK_STAGE", "").strip()
        if _stage == "1":
            self.rewards.release_bias.weight = 0.0
            self.rewards.placed.weight = 0.0
        elif _stage == "2":
            self.enable_pregrasp = False
        if _stage:
            print(f"[cfg] ★2단계 학습 BINPICK_STAGE={_stage}★ release_bias.w="
                  f"{self.rewards.release_bias.weight} placed.w={self.rewards.placed.weight} "
                  f"enable_pregrasp={self.enable_pregrasp}")

        # ★pre-grasp assist — enable_pregrasp(기본 True — 클래스 필드 주석 참조)에 따라 부착★
        #   켜면 scatter 이후 마지막에 붙어 일부 env를 '이미 잡힌 상태'로 시작(frac은 커리큘럼 감소).
        #   BINPICK_STAGE=2는 위 flag 처리에서 enable_pregrasp=False로 끈다('잡힌 채 시작' 원천 제거).
        #   curriculum term(pregrasp_frac)도 함께 켜고/끈다 — None-skip은 vendored 검증
        #   (curriculum_manager.py L188-189 `if term_cfg is None: continue`).
        #   ★place bootstrap 폐기(2026-07-11)★: v1(무료 placed)·v2(무료 release_bias) 둘 다 부트스트랩
        #   env가 place 보상을 정책 개입 0으로 farming → pick+carry 붕괴(iter601 grasp 1.0→0.01).
        #   → 릴리스는 release_bias smooth shaping + 2단계 학습이 담당(관련 코드 전부 삭제 — git 참조).
        if self.enable_pregrasp:
            self.events.pregrasp = dr.make_pregrasp_assist_event()
        else:
            self.curriculum.pregrasp_frac = None

        # ★b5 목표거리 역커리큘럼(BINPICK_TARGET_CURR, 2026-07-23) — placed 직격 카드★
        #   목표를 물체 근처(반경 curriculum 확대)에 둬 carry→placed 전환을 학습시킨다.
        #   command class_type을 커스텀으로 교체 + radius curriculum term 부착(= 게이트 on).
        #   ★farming 회피 = _real_pick_mask(pregrasp env placed 차단, 기존 장치) 유지★.
        #   미설정=기존 UniformPoseCommand(절대 균일) 재현. b4 진단 직격(carry→placed 50→14%).
        if os.environ.get("BINPICK_TARGET_CURR", "").strip().lower() in ("1", "true"):
            self.commands.target_pos.class_type = TargetCurriculumPoseCommand
            self.curriculum.target_radius = dr.make_target_radius_curriculum()
            # ⚠️★2026-08-11(F58) 이 커리큘럼은 현재 ★사용하지 말 것★★ — 타겟 xy를 클램프 없이
            #   `obj_b + r·(cosθ,sinθ)`로 덮어써 ★벨트(유일한 지지면) 밖★으로 나간다.
            #   r=0.35에서 타겟의 ★36.5%가 placed 원천 불가★(놓을 표면이 없어 |z−z_t|<5mm 불성립),
            #   ★74.6%가 선언 target range 밖★(= 관측 계약 위반). 상세 = README.md F58,
            #   지지면 클램프 재설계 원칙 = README.md F58/§10.
            # ★문구 정정(2026-08-11)★: decay 기본값은 40000이 아니라 ★100000★이다
            #   (make_target_radius_curriculum decay_steps=100000). 콘솔만 보고 스케줄을
            #   추정하면 반경 도달 시점을 2.5배 빠르게 오독한다.
            print("[cfg] ⚠️b5 목표거리 역커리큘럼 활성 — ★F58 미해결(지지면 밖 타겟)★ "
                  "target=물체근처 반경 0.05→0.35m(warmup 3000·decay 100000 step)")

        # ★2026-07-10 전역 기본 물리재질 = 고마찰★ — finger 콜라이더(USD 자산, 자체 재질 없음)에
        #   마찰을 부여해 파지 유지력 확보(큐브는 DR object_friction 1.2~2.5, 벨트/기타는 이 기본값).
        #   PhysX 기본은 저마찰이라 잡아도 큐브가 pad 사이를 미끄러져 낙하(렌더 확인)했다.
        #   ★2.0으로 상향★: 1.0에서 파지 성공률 50%(16env). finger-큐브 combine 마찰을 높여
        #   heavy 큐브(mass DR 상한 0.10kg=0.98N)·lift 가속에도 유지되도록 여유 확보.
        # ⚠️ TODO(Phase3 SysID — 리뷰 2026-07-10): 이 finger 마찰은 DR 미포함(고정). object_friction과
        #   combine되는 grip 레버의 절반이 un-bracketed 상태. 실 2FG7 pad 마찰 실측 후 dr.yaml에
        #   finger_friction 추가+randomize 배선 예정(teacher/student 동반 재학습). 상세=dr.yaml 주석.
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0, dynamic_friction=2.0, restitution=0.0
        )

        # PhysX 안정화 (upstream lift __post_init__ 승계 — L218-222)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # ★4096 env 대응★: 16*1024는 4096env(객체4+bin+그리퍼)에서 부족(~16400 필요) →
        #   PhysX "totalAggregatePairsCapacity" 에러(충돌 누락=물리 깨짐). 여유 있게 상향.
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024

        # 재생/영상용 뷰어 카메라 — 픽존(≈0.45,0) 클로즈업 3/4 뷰(2026-07-22 각도 개선).
        # ★headless 학습에는 무영향★ (뷰포트 없음). BinPick_Env.py --video 녹화 시
        # RecordVideo가 이 시점으로 렌더 → 그리퍼/객체 거동이 크게 보인다.
        # 카메라를 넘게 빼서(1.9m·z 1.25) 로봇 팔 전체 + 벨트 + 파지→운반→안착 전 과정을
        # 모두 담는다(2026-07-22 사용자 결정 — 팔이 높이 뻗어 근접 각도는 상단·운반 잘림).
        self.viewer.eye = (1.60, 1.08, 1.25)
        self.viewer.lookat = (0.40, 0.0, 0.30)

        # ★D435i 본체(시각 전용) 런타임 부착 — teacher/vision 공통(2026-07-15 v3)★:
        #   포즈 단일 소스 = hand_eye.yaml(T_flange_cam). URDF에는 굽지 않는다(브라켓만)
        #   — 사용자 GUI 배치(place_d435i_gui.py) 반복 시 yaml 수정만으로 본체·depth
        #   센서가 함께 이동. USD 미존재 시 None → 부착 생략.
        _body = make_d435i_body_cfg()
        if _body is not None:
            self.scene.d435i_body = _body


@configclass
class BinPickTeacherEvalEnvCfg(BinPickTeacherEnvCfg):
    """teacher 정량 평가용 — ★pre-grasp 보조 완전 제거(순수 자력 파지 성공률 측정)★.

    학습 cfg를 그대로 상속하되 pregrasp 리셋·curriculum term을 끈다(frac=0 등가). eval 시
    common_step_counter=0에서 curriculum이 frac을 warmup값(0.75)으로 되돌려 '대부분 이미 잡힌
    상태'로 평가돼 성공률이 부풀려지는 오염을 방지한다 — 모든 env가 벨트 위 산란 큐브를
    자력으로 파지해야 한다. (None 대입 시 매니저가 스킵함은 vendored 검증:
    event_manager.py L341-342 / curriculum_manager.py L188-189 — `if term_cfg is None: continue`.)

    Task ID: "Isaac-BinPick-UR5e-Teacher-Eval-v0". 러너 cfg는 teacher와 동일(같은 ckpt·구조).
    """

    def __post_init__(self):
        super().__post_init__()
        # ★모든 부트스트랩 보조 제거★ — scatter 이벤트만 남아 순수 자력 pick→carry→place 측정.
        self.events.pregrasp = None
        self.curriculum.pregrasp_frac = None


@configclass
class BinPickVisionEnvCfg(BinPickTeacherEnvCfg):
    """Phase2 vision student: 비대칭 관측 + TiledCamera, num_envs 128.

    Task ID: "Isaac-BinPick-UR5e-Vision-v0" / experiment: ur5e_binpick_vision.
    """

    observations: VisionObservationsCfg = VisionObservationsCfg()

    # ContactSensor(privileged 접촉력) 스위치 — 그리퍼 자산 확정 + VRAM 실측 전 OFF.
    enable_finger_contacts: bool = False

    def __post_init__(self):
        super().__post_init__()

        # VRAM: 현행 TiledCamera 112² RGB-D 계약. OOM 시 관측 계약은 유지하고
        # configs/sim.yaml의 num_envs 축소 순서(128→64→32)를 따른다.
        self.scene.num_envs = _STUDENT_NUM_ENVS

        # ★★F84 — 증류 커리큘럼 = 2상 프로파일: 0.9 고정(iter 5000) → 자연 감쇠★★
        #   [F81의 결함] min_grasp_ema=0.0으로 게이트만 열었더니 frac이 iter 600~2500에
        #     선형 소멸 → student가 carry를 익히기 ★전에★ carry 상태가 데이터에서 사라졌다.
        #     실측(F84 carry-probe, model_12000): latch 상태는 전체 스텝의 0.36%뿐이고,
        #     그 상태에서 cos(teacher,goal)=+0.54(타깃은 옳음) vs cos(student,goal)=−0.15
        #     (모방 백지) — MSE 0.135는 나머지 99.6% 스텝의 평균일 뿐이다. DAgger의
        #     상태-커버리지 가정 위반(Ross 2011): 라벨이 옳아도 상태가 없으면 못 배운다.
        #   [성공 계보의 실체] 08-21(0.131)·08-27(0.110)은 deadline 버그의 우연으로
        #     ★iter 5000까지 frac 0.9 고정★(carry 상태 풍부 → carry 먼저 마스터) 후 절벽
        #     → 이후 15k iter 자력 파지 성장. 그 2상 구조를 여기서 ★명시적으로★ 재현한다
        #     (Florensa 2017 역방향 커리큘럼: 목표 근방 상태에서 후행 스킬부터).
        #   [파라미터] warmup 120_000스텝 = iter 5000(0.9 고정); decay 43_200스텝 =
        #     max_drop 0.05×update_interval 2400의 자연 감쇠 상한과 정확히 일치(0.9→0을
        #     iter 5000→6800에 걸쳐, 절벽 대신 유계 감쇠 — F81의 '유계 갱신' 계약은 유지).
        #     min_grasp_ema=0.0은 F81대로 둔다(스케줄이 EMA에 막히지 않게).
        #   ⚠️teacher 학습(BinPickTeacherEnvCfg)은 기존 게이트·일정을 유지한다 — 이
        #     override는 Vision 계열에만 적용된다(변인 분리).
        self.curriculum.pregrasp_frac = dr.make_pregrasp_curriculum(
            min_grasp_ema=0.0, warmup_steps=120_000, decay_steps=43_200
        )

        # ★요소4: 카메라 포즈 = configs/calib/hand_eye.yaml (sim/real 단일 소스)★
        self.scene.tiled_camera = make_tiled_camera_cfg()
        # ★★F94(2026-09-22) RGB 타일 렌더 결함 회피 — DLSS 끔★★
        #   [실측] RTX 5070 Ti + Isaac Sim 5.1 headless 기본("performance" = DLSS)에서 타일 수 ≥128이면
        #     RGB 블록만 죽는다: 리셋 프레임 완전 검정(41%), 이후 스텝마다 검정/부분 렌더 교대. depth 정상.
        #     64 타일 이하 정상. antialiasing_mode="Off" 또는 --rendering_mode quality면 128·256 타일 모두 정상.
        #   [영향] 증류 1회(256 env, 20k iter)가 깨진 RGB로 학습됐고 채택 평가(128 env)가 3회 연속
        #     EVAL_RENDER_DEGENERATE로 거부해 드러났다. 실기 카메라에는 DLSS가 없으므로 끄는 쪽이 정합이다.
        #   [가드] common/render_guard.py를 train.py·BinPick_Eval.py가 첫 관측에서 실행한다.
        self.sim.render.antialiasing_mode = "Off"
        # Reset teleports objects/robot after the physics render. Refresh RGB-D
        # before pairing it with reset proprio and current teacher labels.
        self.num_rerenders_on_reset = 1
        # (D435i 본체 시각은 base cfg가 런타임 부착 — 2026-07-15 v3. depth 센서는
        #  위 tiled_camera, 둘 다 hand_eye.yaml 단일 소스라 포즈가 항상 일치.)

        # ★필수 DR 적용(2026-07-15, 사용자 지시 '필수만 적용')★
        #   ① 관측 지연 DR: 실기 루프 지연 강건성(배포 직결) — events_dr ①-d
        #   ② 카메라 pose DR: 캘리브 허용오차(±1cm/±1°) 강건성 — §4-5 재학습 판정 실효화
        #   ③ depth 노이즈 재샘플: 구 make_visual_dr_terms 묶음에서 개별 부착으로 전환
        #      (묶음의 색/텍스처 항만 제거, 이 항은 main8까지의 학습과 동일하게 유지)
        self.events.obs_delay = dr.make_obs_delay_event()
        self.events.camera_pose_dr = dr.make_camera_pose_event()
        self.events.depth_noise_resample = dr.make_depth_noise_resample_event()

        # ★RGB-D 시각 DR 재활성(2026-07-17 B안 — use_rgb 계약일 때만)★
        #   2026-07-15의 '색/조명 DR 실효 0' 판정은 depth-only 관측 시대의 것 — rgb 관측
        #   채택으로 반전(정책이 색·조명을 '본다'). use_rgb=false로 되돌리면 자동 미부착
        #   (실효 0 판정 복원) — 분기점은 configs/common.yaml preprocess.use_rgb 하나.
        _use_rgb = bool(_COMMON.get("preprocess", {}).get("use_rgb", False))
        if _use_rgb:
            # ① rgb 노이즈 파라미터 재샘플(리셋) — observations.policy_obs가
            #    env._binpick_rgb_noise를 소비(키 계약: brightness/contrast/pixel_sigma)
            self.events.rgb_noise_resample = dr.make_rgb_noise_resample_event()
            # ② 광량 DR — ★전역★(interval, is_global_time): /World/light가 전역 프림이라
            #    per-env 불가(events_dr.resample_light_intensity docstring)
            self.events.light_intensity = dr.make_light_intensity_event()
            # ③ 시각 색 DR — ★dr.yaml lighting.randomize_object_color 게이트
            #   (2026-07-18 기본 false — RGB-D r1/r2 실패 분석)★:
            #   색 DR은 replicate_physics=False를 강제해 VRAM 폭증 → num_envs 128 제한
            #   → 총 샘플이 검증된 depth-only 레시피(512env)의 1/3로 붕괴 + 소형
            #   Nature-CNN에 외형 불변성 난도 폭증(이중 타격). 문헌 대조: 색 DR 견고성은
            #   DextrAH-RGB(4×L40S·53h) 예산의 레시피 — 본 예산(8GB)은 SplatSim 노선
            #   (절차적 색 DR 없이 ★실사 렌더(NuRec 2단계)★로 외형 갭 해소)이 정합.
            #   rgb 지터·전역 광량 DR(위 ①②)은 유지(경량·replicate 무관).
            if bool(_DR_YAML.get("lighting", {}).get("randomize_object_color", False)):
                _color_terms = dr.make_rgb_visual_color_terms(NUM_OBJECTS)
                for _name, _term in _color_terms.items():
                    setattr(self.events, _name, _term)
                # ★replicate_physics=False 필수★ — stock randomize_visual_color 사용 시
                #   (vendored 2.3.2 envs/mdp/events.py L1611-1616 RuntimeError).
                if any(_n.startswith("color_object_") for _n in _color_terms):
                    self.scene.replicate_physics = False

        if self.enable_finger_contacts:
            # TODO: 그리퍼 자산 확정 후 — UR5E_CFG.spawn.activate_contact_sensors=True
            #   병행 필요 (mdp/scene.py make_finger_contact_sensor_cfg docstring).
            self.scene.finger_contacts = make_finger_contact_sensor_cfg()


@configclass
class BinPickVisionNuRecEnvCfg(BinPickVisionEnvCfg):
    """★NuRec 실셀 배경 파인튠 전용(2026-07-17 B안)★ — Eval 변형 없음(파인튠 전용 cfg).

    합성 배경에서 증류된 student를, 실셀 3DGS 스캔(Workplace.usdz) 배경 위에서 소규모
    fine-tune해 rgb 배경 도메인 갭을 줄인다. VisionPick_Train --nurec-finetune이 사용
    (Task ID: "Isaac-BinPick-UR5e-Vision-NuRec-v0", 러너 = BinPickDistillRunnerCfg 재사용
    — student ckpt resume으로 증류 연속).

    학습 cfg 대비 차이(전부 __post_init__):
      · num_envs=4 — NuRec 볼륨은 env 간 무공유(프로브 실측: 볼륨당 VRAM +0.4GB).
      · scene.nurec_bg 부착(scene_registration.yaml 정합 게이트 — 미정합 시 기동 거부).
      · 벨트/스탠드 색 DR off(실셀 배경 고정) / 조명 DR off(SH 베이크 불일치) —
        큐브 색 DR·rgb_noise 지터·depth 노이즈·카메라 pose·obs 지연 DR는 유지.
    """

    def __post_init__(self):
        super().__post_init__()
        # ★use_rgb 계약 게이트★: NuRec 가우시안은 depth annotator에 렌더되지 않음
        #   (2026-07-17 프로브 실증 — rgb에만 나타남). rgb 관측 없는 파인튠은 배경이
        #   정책 입력에 전혀 반영되지 않는 무의미한 run → 기동 거부가 정상 동작.
        if not bool(_COMMON.get("preprocess", {}).get("use_rgb", False)):
            raise RuntimeError(
                "[cfg] ★NuRec 파인튠 기동 거부★ — configs/common.yaml preprocess.use_rgb=false. "
                "NuRec 배경은 rgb 관측에만 렌더되므로(depth 미반영, 프로브 실증) "
                "use_rgb=true 계약에서만 의미가 있다."
            )

        # ★VRAM 스케일 사다리: num_envs 4 → 2 → 1★ (OOM 시 반감 — CLAUDE §3 규칙 준용).
        #   근거: NuRec 볼륨은 env 클론 간 공유되지 않아(프로브: 2개째 +0.4GB) env 수에
        #   비례해 VRAM 증가. 파인튠은 소규모 적응이라 4 env로 충분(증류 연속·짧은 run).
        self.scene.num_envs = 4

        # ★env_spacing 3.0 → 30.0(2026-07-17 적대 리뷰 수정)★ — 룸스케일 가우시안
        #   볼륨이 per-env로 스폰되는데 3m 간격이면 이웃 env의 볼륨·로봇이 wrist rgb
        #   배경에 겹쳐 렌더된다(rgb는 depth와 달리 far 마스크 없음, clipping 20m) —
        #   '실셀에 없는' 합성 이미지를 파인튠이 학습하는 오염. 관측·보상은 전부
        #   env origin/base frame 기준이라 spacing 확대는 학습에 무영향.
        #   ⚠️ 정합 후 실제 볼륨 extent가 30m를 넘보면 num_envs=1로 강등할 것.
        self.scene.env_spacing = 30.0

        # NuRec 배경 부착 — make_nurec_background_cfg 가드 4종(usdz 부재/미정합/잔차/
        # 항등 placeholder) 전부 RuntimeError = 기동 거부(2026-07-17 리뷰 — None 경로 폐지).
        self.scene.nurec_bg = make_nurec_background_cfg()

        # ★워크스페이스 시각 투명화(2026-07-17 사용자 지시 — 배포 갭 제거)★:
        #   sim 회색 벨트 플레이트가 RGB에서 스캔의 '실제' 벨트를 가리면 파인튠이
        #   실기에 없는 외형을 학습한다. 물리 벨트는 비가시화(콜라이더는 가시성과
        #   무관 — z=0 안착·타겟 계약 불변)하고, depth 대역은 3mm 침하 무충돌
        #   플레이트가 담당(가우시안이 위에서 덮어 RGB=실벨트, depth=실기와 3mm 이내
        #   — 근거·함정 분석은 scene.make_nurec_belt_visual_cfg docstring).
        #   ⚠️ 파인튠 시작 전 렌더 검증 필수: RGB에 회색 플레이트 소멸 + depth 벨트
        #   픽셀 유효(~0.4m) — diagnostics/render_nurec_env.py 재실행(GPU 유휴 시).
        self.scene.belt.spawn.visible = False
        self.scene.belt_visual_nurec = make_nurec_belt_visual_cfg()

        # ★벨트/스탠드 시각 색 DR off★ — 실셀 배경(NuRec) 위에 물리 벨트/스탠드가 겹쳐
        #   보이는 구성에서 이들의 색을 흔들면 '실셀에 없는 색 변화'를 학습한다. 파인튠
        #   목적 = 실셀 rgb 분포 적응이므로 배경 요소는 고정. ★큐브 색 DR(color_object_*)
        #   은 유지★ — 대상 부품 색은 실제로도 다양(일반화 유지).
        #   (None 대입 시 매니저 스킵 = vendored event_manager.py L341-342 검증.)
        self.events.color_belt = None
        self.events.color_belt_stand = None
        self.events.color_robot_stand = None

        # ★조명 DR off★ — NuRec(3DGS)은 스캔 시점 조명이 가우시안 SH에 '구워져' 있어
        #   런타임 라이트를 흔들면 메시(벨트/큐브/로봇 — 라이트에 반응)와 가우시안 배경
        #   (불변)의 조명 불일치를 만든다 — 실기에는 없는 인공 아티팩트를 학습하게 됨.
        #   전역 밝기 변동 모사는 rgb_noise 지터(brightness/contrast — 렌더 후 일괄
        #   후처리라 메시·가우시안에 동일 적용)가 유지 담당.
        self.events.light_intensity = None


@configclass
class BinPickVisionEvalEnvCfg(BinPickVisionEnvCfg):
    """vision student 정량 평가용 — ★pre-grasp 보조 완전 제거(순수 자력 성공률)★.

    BinPickTeacherEvalEnvCfg와 동일 원칙(2026-07-13 추가): 학습 cfg를 상속하되 pregrasp
    리셋·curriculum을 끈다 — eval 시 common_step_counter=0에서 frac이 warmup(0.75)으로
    되돌아 '잡힌 채 시작' 오염되는 것을 방지. 관측은 학습과 동일(RGB-D flat 50,206D —
    배포 계약, use_rgb 계약 준수). ★DR(색/조명/rgb·depth 노이즈/지연/카메라 pose)은
    학습 cfg 그대로 상속 = 평가도 배포 대표 조건(2026-07-15 결정과 동일 원칙)★ —
    다른 run과 성공률 비교 시 DR 조건 동일 여부를 명기할 것.

    Task ID: "Isaac-BinPick-UR5e-Vision-Eval-v0". 러너 cfg는 Distill과 동일(증류 ckpt 로드).
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.pregrasp = None
        self.curriculum.pregrasp_frac = None


@configclass
class BinPickVisionEvalNoLightDrEnvCfg(BinPickVisionEvalEnvCfg):
    """★진단 전용(2026-07-19) — 절차 Eval에서 광량 DR만 off★.

    목적: g3의 '절차 15.0% < 실사(NuRec) 21.9%' 역전 원인 격리. 절차 Eval과 NuRec Eval의
    실질 차이 변인은 ①전역 광량 DR(0.3~2.0, NuRec은 SH 베이크 불일치로 off) ②배경
    (근사색 씬 vs 실사 렌더) 둘뿐(색 DR은 dr.yaml 게이트 false로 양쪽 off, rgb 지터·
    지연·카메라 pose DR은 공통). 본 cfg = 절차 씬 + 광량 DR off → NuRec Eval과 'DR 구성
    동일·배경만 다름'인 대조를 만든다:
      · 성공률이 실사(21.9%)에 근접 → 갭 = 광량 DR 난이도(0.3배 극암 프레임이 주범 추정)
      · 15.0% 유지 → 갭 = 배경 자체(근사색 씬이 오히려 어려움 — 추가 진단)
    ⚠️ 배포 대표 조건이 아님 — 공식 성공률 비교에 사용 금지(진단 산출물에 조건 명기).

    Task ID: "Isaac-BinPick-UR5e-Vision-Eval-NoLightDR-v0".
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.light_intensity = None
