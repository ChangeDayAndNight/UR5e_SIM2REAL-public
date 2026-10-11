"""Bin-Picking 관측 함수 — 단일 policy ObsTerm(요소1) + privileged critic 항.

목적: (a) policy 그룹의 '유일한' ObsTerm인 policy_obs() — raw depth에 노이즈를
      입힌 뒤 common.obs_preprocess.build_policy_obs를 '그대로' 호출해 flat 관측을
      만든다(sim/real 분포 정렬의 심장). (b) critic(privileged) 그룹용 특권 관측
      함수들 — 팔/그리퍼 상태, 객체 포즈/속도, 좌·우 접촉력, 목표 원시값.
실행: 직접 실행 없음 — binpick_env_cfg.py의 ObsTerm(func=...)이 참조.
산출: policy_obs, joint_pos_abs, joint_vel_abs, tcp_pose_b, target_pos_command,
      object_positions_b, object_velocities, finger_contact_forces.

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것.
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from common.config_io import build_preprocess_cfg, configs_dir, load_yaml
from common.depth_noise import apply_depth_noise
from common.observation_delay import delay_policy_observation
from common.obs_preprocess import build_policy_obs, canonicalize_quat_wxyz, obs_dim
from common.rgb_noise import apply_rgb_noise

from .scene import NUM_OBJECTS, object_names
from .ur5e_assets import GRIPPER_JOINT_NAMES, UR5E_ARM_JOINT_NAMES

# 공유 전처리 파라미터 — configs/common.yaml 단일 소스 (요소1).
_PRE_CFG = build_preprocess_cfg(load_yaml(configs_dir() / "common.yaml"))
_DR_YAML = load_yaml(configs_dir() / "dr.yaml")

# depth 노이즈 fallback(리셋 이벤트가 아직 파라미터를 안 심었을 때) = dr.yaml 중심값.
_NOISE_FALLBACK = {
    "sigma_base_m": float(_DR_YAML["depth_noise"]["sigma_base_m"]),
    "sigma_scale_per_m2": float(_DR_YAML["depth_noise"]["sigma_scale_per_m2"]),
    "dropout_prob": float(sum(_DR_YAML["depth_noise"]["dropout_prob"]) / 2.0),
}

# rgb 노이즈 fallback — 동일 패턴(★2026-07-17 RGB-D 전환★, 범위 중심값).
_RGB_NOISE_FALLBACK = {
    "brightness_scale": float(sum(_DR_YAML["rgb_noise"]["brightness_scale"]) / 2.0),
    "contrast_scale": float(sum(_DR_YAML["rgb_noise"]["contrast_scale"]) / 2.0),
    "pixel_sigma": float(sum(_DR_YAML["rgb_noise"]["pixel_sigma"]) / 2.0),
}

# ★F81★ 수직 파지 플래그 — 증류 타깃 마스크(vision_policy)와 같은 단일 소스에서 읽는다.
_VERTICAL_GRASP_ONLY = bool(
    load_yaml(configs_dir() / "common.yaml")["action"].get("vertical_grasp_only", False)
)


def executable_last_action(env) -> torch.Tensor:
    """직전 정책 액션을 ★실행 가능 범위로 정규화해★ 관측한다 — raw가 아니다.

    ★★F81(2026-09-03) — last_action 되먹임 오염 수정 (리뷰 blocker #1)★★

    [사고] 종전 `mdp.last_action`은 클램프·마스크 ★이전★의 raw 정책 출력을 그대로
      관측 슬롯(42D 중 35:42)에 넣었다. 두 가지가 무너졌다:
      ① teacher PPO에서 gradient가 안 걸리는 채널(수직파지로 무시되는 기울임 3·4)의
        표류값(실측 ±7)이 자기 관측으로 되먹임 = ★사실상 자유 메모리 채널★.
      ② 증류(DAgger)에서는 env를 student가 구동하므로 teacher가 받는 last_action이
        teacher 학습분포(자기 raw, grip ±12)에서 student 스케일(~0..±1)로 갈라진다 —
        teacher가 분포 밖에서 질의되어 ★증류 타깃 자체가 오염★됐다. 채택 checkpoint
        실측: 이 슬롯 스케일만 바꿔도 출력이 채널 스케일의 33~86% 이동.

    관측 표현은 clamp ±1 + 수직파지 기울임 3·4 = 0으로 유지한다.
    이는 기존 Teacher가 학습한 관측 계약이지 실행 명령의 clipping이 아니다.
    증류 정답의 병진/yaw는 무계 명령을 보존해야 한다. 이름은 호환성 때문에 유지한다.

    ⚠️ 차원 불변(7D) — teacher 42D 계약 유지. 의미만 바뀌므로 teacher 재학습 대상.
    """
    action = env.action_manager.action.clone().clamp(-1.0, 1.0)
    if _VERTICAL_GRASP_ONLY and action.shape[-1] > 5:
        action[:, 3:5] = 0.0
    return action


def _tcp_pose_from_frame(ee) -> torch.Tensor:
    """Build base-frame pose with the sim/real canonical ``wxyz`` quaternion."""
    quat = canonicalize_quat_wxyz(ee.data.target_quat_source[:, 0, :])
    return torch.cat([ee.data.target_pos_source[:, 0, :], quat], dim=1)


def _arm_joint_ids(env, robot_name: str = "robot") -> list[int]:
    """UR5e 팔 6관절의 관절 인덱스(순서 = UR5E_ARM_JOINT_NAMES) — env에 캐시."""
    if not hasattr(env, "_binpick_arm_ids"):
        ids, _ = env.scene[robot_name].find_joints(
            UR5E_ARM_JOINT_NAMES, preserve_order=True
        )
        env._binpick_arm_ids = ids
    return env._binpick_arm_ids


# ═════════════════════════════════════════════════════════════════════════════
# ★★요소1: 정책 관측은 이 함수 '하나'로만 만든다★★
#
#   real 배포(common/rollout.py)와 sim 학습이 '문자 그대로 같은'
#   common.obs_preprocess.build_policy_obs를 호출한다. flat 레이아웃
#   [depth(H*W), rgb(H*W*3 — use_rgb 계약 시), joint_pos(6), joint_vel(6),
#    gripper_command_state(1), tcp_pose(7), target_pos(3), last_action(7)]은 ObservationManager의 항 나열
#   순서가 아니라 build_policy_obs 내부에서 구조적으로 고정된다 — policy 그룹에
#   ObsTerm을 '추가'하는 순간 sim/real 계약(obs_spec.json)이 깨진다. 절대 금지.
#
#   depth 노이즈는 '공유 전처리 이전'의 raw depth에만 주입한다(dr.yaml 계약):
#   real은 센서가 진짜 노이즈를 만들고, sim은 여기서 모사한다 → 전처리 이후
#   분포가 일치.
# ═════════════════════════════════════════════════════════════════════════════
def policy_obs(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    command_name: str = "target_pos",
    delay_proprio: bool = False,
) -> torch.Tensor:
    """policy 그룹의 단일 ObsTerm — flat [N, obs_dim] 반환 (vision env 전용).

    경로: TiledCamera raw depth[m] → apply_depth_noise(dr.yaml, 전처리 '이전')
          → build_policy_obs(공유 전처리 + 레이아웃 고정) → flat.
    """
    # 매니저 초기화(차원 파악) 단계 가드 — 센서 데이터가 아직 없을 수 있다
    # (Franka Phase4 _ever_grasped 검증 패턴).
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, obs_dim(_PRE_CFG), device=env.device)

    # ── ① raw depth [N,H,W] (TiledCamera 출력은 [N,H,W,1] — 검증됨) ──
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output["distance_to_image_plane"][..., 0]

    # ── ② depth 노이즈 (전처리 '이전', sim 전용) ──
    # 파라미터는 events_dr.resample_depth_noise_params(리셋 이벤트)가 에피소드마다
    # env._binpick_depth_noise에 심는다. 없으면 dr.yaml 중심값 fallback.
    noise = getattr(env, "_binpick_depth_noise", _NOISE_FALLBACK)
    depth = apply_depth_noise(
        depth,
        sigma_base_m=noise["sigma_base_m"],
        sigma_scale_per_m2=noise["sigma_scale_per_m2"],
        dropout_prob=noise["dropout_prob"],
    )

    # ── ②b ★raw RGB + 노이즈 (RGB-D 계약, 2026-07-17 B안)★ ──
    # TiledCamera 'rgb' annotator = [N,H,W,3] uint8(rgba 앞 3채널 alias —
    # tiled_camera.py L282-284 검증). float[0,1]로 변환 후 rgb 지터(전처리 '이전',
    # sim 전용 — real은 센서가 진짜 변동을 만든다). NuRec 배경 유무와 무관한 경로.
    rgb = None
    if _PRE_CFG.use_rgb:
        if "rgb" not in sensor.data.output:
            raise RuntimeError(
                "[obs] use_rgb 계약인데 TiledCamera에 'rgb' 출력이 없음 — "
                "scene.make_tiled_camera_cfg data_types 확인(계약 불일치)"
            )
        rgb = sensor.data.output["rgb"][..., :3].float() / 255.0
        jitter = getattr(env, "_binpick_rgb_noise", _RGB_NOISE_FALLBACK)
        rgb = apply_rgb_noise(
            rgb,
            brightness_scale=jitter["brightness_scale"],
            contrast_scale=jitter["contrast_scale"],
            pixel_sigma=jitter["pixel_sigma"],
        )

    # ── ③ proprio + 목표 + last_action ──
    robot = env.scene[robot_cfg.name]
    ids = _arm_joint_ids(env, robot_cfg.name)
    joint_pos = robot.data.joint_pos[:, ids]  # 절대각 [rad] — real getActualQ와 동일 의미
    joint_vel = robot.data.joint_vel[:, ids]

    # TCP 포즈(base frame): FrameTransformer의 source(=base_link) 기준 출력 사용.
    # target_pos_source/target_quat_source 필드 검증됨(frame_transformer_data.py L22-28).
    ee = env.scene[ee_frame_cfg.name]
    tcp_pose = _tcp_pose_from_frame(ee)

    # place 목표(base frame): command manager가 에피소드마다 균일 샘플(리셋 시 재샘플).
    target_pos = env.command_manager.get_command(command_name)[:, :3]

    # last_action: real 루프는 clip된 action을 기록하므로 sim도 clip해 정합 유지.
    last_action = env.action_manager.action.clamp(-1.0, 1.0)

    # ── ③b delayed binary command state ── action term이 DelayBuffer 이후
    # policy step마다 정확히 한 번 갱신한다. 실제 finger q/폭/접촉 신호가 아니다.
    gripper_command_state = getattr(env, "_bp_gripper_command_state", None)
    if gripper_command_state is None:
        raise RuntimeError("gripper command state tracker is unavailable")

    # ── ④ ★공유 전처리 + flat 조립 (sim/real 같은 함수)★ ──
    flat = build_policy_obs(
        depth, joint_pos, joint_vel, tcp_pose, target_pos, last_action, _PRE_CFG,
        gripper_command_state=gripper_command_state,
        rgb=rgb,
    )
    # ── ⑤ ★관측 지연 DR(2026-07-15)★ — events_dr._resample_obs_delay가 에피소드마다
    #   env별 지연(dr.yaml latency)을 재샘플한 DelayBuffer를 통과. 실기 루프 지연의
    #   최악 케이스 모델(전체 관측 일괄 지연). 버퍼 미생성(이벤트 미배선 env)이면 무지연.
    _delay = getattr(env, "_bp_obs_delay", None)
    image_dim = _PRE_CFG.out_height * _PRE_CFG.out_width * (4 if _PRE_CFG.use_rgb else 1)
    return delay_policy_observation(flat, _delay, image_dim, delay_proprio=delay_proprio)


# ═════════════════════════════════════════════════════════════════════════════
# privileged(critic 전용) 관측 — teacher policy/critic, vision critic이 사용.
# real에는 존재하지 않는 정보만 모아둔다 (actor에 새면 배포 불가).
# ═════════════════════════════════════════════════════════════════════════════
def joint_pos_abs(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """팔 6관절 절대각 [rad] (N,6) — teacher 관측용 (정규화는 안 함: rsl_rl
    empirical_normalization에 위임. ⚠️ policy_obs의 스케일 정규화와 다른 경로임을
    인지할 것 — teacher와 student 관측은 어차피 이종(distillation이 흡수))."""
    return env.scene[robot_cfg.name].data.joint_pos[:, _arm_joint_ids(env, robot_cfg.name)]


def joint_vel_abs(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """팔 6관절 각속도 [rad/s] (N,6)."""
    return env.scene[robot_cfg.name].data.joint_vel[:, _arm_joint_ids(env, robot_cfg.name)]


def gripper_pos_abs(env, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Actual left/right finger joint positions in metres, shape ``(N, 2)``."""
    robot = env.scene[robot_cfg.name]
    joint_ids, _ = robot.find_joints(list(GRIPPER_JOINT_NAMES))
    return robot.data.joint_pos[:, joint_ids]


def tcp_pose_b(env, ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """TCP 포즈 (N,7) pos+quat(wxyz), base frame — 요소4 tcp_offset 반영값."""
    ee = env.scene[ee_frame_cfg.name]
    return _tcp_pose_from_frame(ee)


def target_pos_command(env, command_name: str = "target_pos") -> torch.Tensor:
    """place 목표 위치 원시값 (N,3), base frame — 에피소드 가변(사용자 결정 2026-07-04)."""
    return env.command_manager.get_command(command_name)[:, :3]


def object_positions_b(env, num_objects: int = NUM_OBJECTS) -> torch.Tensor:
    """산란 객체 위치 (N, num_objects*3), env origin(=base — 요소4 전제) 기준.

    privileged: real에서는 관측 불가(depth만 존재) — critic 전용.
    """
    origins = env.scene.env_origins  # (N,3) — interactive_scene.py L297 검증
    cols = [
        env.scene[name].data.root_pos_w - origins for name in object_names(num_objects)
    ]
    return torch.cat(cols, dim=1)


def object_velocities(env, num_objects: int = NUM_OBJECTS) -> torch.Tensor:
    """산란 객체 선/각속도 (N, num_objects*6), world frame. privileged 전용."""
    cols = [
        torch.cat(
            [env.scene[name].data.root_lin_vel_w, env.scene[name].data.root_ang_vel_w],
            dim=1,
        )
        for name in object_names(num_objects)
    ]
    return torch.cat(cols, dim=1)


def finger_contact_forces(env, force_scale_n: float = 25.0) -> torch.Tensor:
    """Normalised left/right object-contact forces, shape ``(N, 2)``.

    The Teacher scene installs two filtered sensors.  Looking up the former
    ``finger_contacts`` aggregate silently produced a constant-zero observation.
    """
    if force_scale_n <= 0.0:
        raise ValueError("force_scale_n must be positive")
    values = []
    for name in ("left_finger_contact", "right_finger_contact"):
        sensor = env.scene.sensors.get(name)
        matrix = None if sensor is None else sensor.data.force_matrix_w
        if matrix is None:
            values.append(torch.zeros(env.num_envs, device=env.device))
            continue
        values.append(matrix.norm(dim=-1).flatten(start_dim=1).amax(dim=1))
    return torch.stack(values, dim=1).div(force_scale_n).clamp_(0.0, 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# 관측/액션 지연(latency) — dr.yaml latency 섹션 (요소2)
# ═════════════════════════════════════════════════════════════════════════════
# ✅ obs 지연: 2026-07-15 구현 완료 — events_dr._resample_obs_delay(리셋 이벤트,
#   DelayBuffer lazy 생성+set_time_lag)와 위 policy_obs ⑤(compute 통과)의 조합.
#   구 스텁 make_obs_delay_buffer는 삭제(구현과 이름 혼동 방지 — 스텁 셰도잉 사고 재발 방지).
# Action latency is implemented by mdp.delayed_actions and configured from dr.yaml.
