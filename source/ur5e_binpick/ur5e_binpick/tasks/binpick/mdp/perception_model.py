"""★Teacher v2(2026-09-18)★ sim 인식 모델 — 특권 큐브 위치를 '배포 인식 추정치'로 바꾼다.

목적: Teacher가 실기에서 받게 될 신호(common/teacher_obs.py TeacherObsBuilder)와 ★같은 규칙★으로
      관측을 만든다. 정책은 완벽한 큐브 위치가 아니라 오차·결측·기하 제약이 있는 추정치로 학습한다.
  · 검출 가능 = 카메라 기하(D435i 최소거리 밖 ∧ 실기 color FOV 안) ∧ 프레임별 확률 검출
    카메라 포즈 = configs/calib/hand_eye.yaml·tcp_offset.yaml(sim/real 단일 소스), FOV = sim.yaml camera.
  · 검출 시 추정치 = 참값 + 에피소드 고정 편향(hand-eye 잔차) + 가우시안 노이즈; 쥐지 않았으면 z는
    작업면 안착 중심으로 고정(실기 locator의 snap_z_on_table).
  · 미검출·held → TCP + held_offset / 미검출·¬held → 마지막 추정치 유지(없으면 TCP 아래 작업면).
  · held = 폐합 전이 시점에 추정치가 TCP 반경 안이면 latch. 열리면 해제. 닫힌 채 큐브가 멀리서
    검출되면 해제(헛집기 복구 신호).
파라미터 범위: configs/dr.yaml perception — "실측 인식 오차가 이 범위 안인가"를 Phase3 게이트가 대조한다.
실행: 직접 실행 없음 — binpick_env_cfg.PrivilegedObsCfg의 ObsTerm과 EventCfg가 참조.

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.math import quat_apply_inverse

from common.calibration import load_hand_eye, load_tcp_offset
from common.config_io import configs_dir, load_yaml
from common.teacher_contract import (
    GRIPPER_CLOSED_THRESHOLD,
    HELD_OFFSET_FROM_TCP_M,
    HELD_RADIUS_M,
)

from . import observations as bp_obs
from .scene import NUM_OBJECTS, OBJECT_SIZE, TABLE_HEIGHT

_DR = load_yaml(configs_dir() / "dr.yaml")["perception"]
_CAM = load_yaml(configs_dir() / "sim.yaml")["camera"]
_TAN_HALF_H = math.tan(math.radians(float(_CAM["horizontal_fov_deg"])) / 2.0)
_TAN_HALF_V = math.tan(math.radians(float(_CAM["vertical_fov_deg"])) / 2.0)
# 카메라 ← TCP 변환 (상수): T_tcp_cam = inv(T_flange_tcp) · T_flange_cam
_T_TCP_CAM = np.linalg.inv(load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")) @ load_hand_eye(
    configs_dir() / "calib" / "hand_eye.yaml"
)
TABLE_CENTER_Z: float = TABLE_HEIGHT + OBJECT_SIZE / 2.0

if NUM_OBJECTS != 1:
    raise RuntimeError("perception_model은 단일 큐브 계약이다(실기 locator와 동일)")


def _state(env) -> dict:
    """에피소드별 인식 파라미터·추정기 상태를 env에 지연 생성한다."""
    st = getattr(env, "_bp_perception", None)
    if st is None or st["est"].shape[0] != env.num_envs:
        n, dev = env.num_envs, env.device
        z3 = lambda: torch.zeros(n, 3, dtype=torch.float32, device=dev)  # noqa: E731
        z1 = lambda: torch.zeros(n, dtype=torch.float32, device=dev)  # noqa: E731
        st = {
            "bias": z3(), "xy_sigma": z1(), "z_sigma": z1(), "detect_prob": z1(), "blind_dist": z1(),
            "est": z3(), "has_est": torch.zeros(n, dtype=torch.bool, device=dev),
            "held": torch.zeros(n, dtype=torch.bool, device=dev),
            "closed_prev": torch.zeros(n, dtype=torch.bool, device=dev),
            "detected": torch.zeros(n, dtype=torch.bool, device=dev),
            "R_tcp_cam_T": torch.as_tensor(_T_TCP_CAM[:3, :3].T, dtype=torch.float32, device=dev),
            "t_tcp_cam": torch.as_tensor(_T_TCP_CAM[:3, 3], dtype=torch.float32, device=dev),
            "held_offset": torch.as_tensor(HELD_OFFSET_FROM_TCP_M, dtype=torch.float32, device=dev),
            "seq": -1,
        }
        env._bp_perception = st
    return st


def _uniform(lo_hi, n: int, device) -> torch.Tensor:
    lo, hi = (float(v) for v in lo_hi)
    return lo + (hi - lo) * torch.rand(n, device=device)


def resample_perception_params(
    env, env_ids, xy_sigma_m, z_sigma_m, bias_m, detect_prob, blind_dist_m
) -> None:
    """리셋된 env의 인식 파라미터를 재샘플하고 추정기 상태를 비운다(진행 중 에피소드는 불변)."""
    st = _state(env)
    ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=env.device)
    ids = ids.to(device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return
    k, dev = ids.numel(), env.device
    st["xy_sigma"][ids] = _uniform(xy_sigma_m, k, dev)
    st["z_sigma"][ids] = _uniform(z_sigma_m, k, dev)
    st["detect_prob"][ids] = _uniform(detect_prob, k, dev)
    st["blind_dist"][ids] = _uniform(blind_dist_m, k, dev)
    ang = torch.rand(k, device=dev) * (2.0 * math.pi)
    mag = _uniform(bias_m, k, dev)
    st["bias"][ids, 0] = mag * torch.cos(ang)
    st["bias"][ids, 1] = mag * torch.sin(ang)
    st["bias"][ids, 2] = 0.0
    st["est"][ids] = 0.0
    st["has_est"][ids] = False
    st["held"][ids] = False
    st["closed_prev"][ids] = False
    st["detected"][ids] = False


def make_perception_resample_event() -> EventTerm:
    """dr.yaml perception 범위로 에피소드마다 인식 파라미터를 재샘플하는 리셋 이벤트."""
    return EventTerm(
        func=resample_perception_params,
        mode="reset",
        params={
            "xy_sigma_m": tuple(float(v) for v in _DR["xy_sigma_m"]),
            "z_sigma_m": tuple(float(v) for v in _DR["z_sigma_m"]),
            "bias_m": tuple(float(v) for v in _DR["bias_m"]),
            "detect_prob": tuple(float(v) for v in _DR["detect_prob"]),
            "blind_dist_m": tuple(float(v) for v in _DR["blind_dist_m"]),
        },
    )


def cube_visible(st: dict, obj_pos: torch.Tensor, tcp_pos: torch.Tensor, tcp_quat: torch.Tensor) -> torch.Tensor:
    """카메라 기하 판정 — 큐브 ★상면 중심★이 D435i 최소 z 밖 ∧ 실기 color FOV 안(bool, N).

    ★F92(2026-09-20, 리뷰 P4)★ 실기 locator(common/cube_depth_locator.py)는 ★상면 픽셀★을 검출하고,
    RealSense Min-Z는 ★광축 z★ 기준이다(유클리드 거리가 아니다). 큐브 중심·노름으로 판정하면
    파지 직전(카메라-큐브 z 0.21~0.23 m, 노름 0.25~0.27 m)에 sim이 실기보다 ~2.5 cm + (노름−z)만큼
    낙관적으로 '보인다'고 판정한다. 상면 중심(+size/2)과 z로 판정한다.
    """
    top = obj_pos.clone()
    top[:, 2] += OBJECT_SIZE / 2.0                                       # 상면 중심 = locator가 보는 면
    v_tcp = quat_apply_inverse(tcp_quat, top - tcp_pos)                  # TCP 프레임으로
    v_cam = (v_tcp - st["t_tcp_cam"]) @ st["R_tcp_cam_T"].T               # → 카메라(optical) 프레임
    z = v_cam[:, 2]
    zc = z.clamp_min(1.0e-6)
    return (
        (z > 0.0)
        & (z >= st["blind_dist"])                                        # Min-Z는 광축 z 기준
        & ((v_cam[:, 0] / zc).abs() <= _TAN_HALF_H)
        & ((v_cam[:, 1] / zc).abs() <= _TAN_HALF_V)
    )


def _update(env) -> dict:
    """스텝당 1회 — 검출·노이즈·held latch·추정치 합성(멱등 가드: common_step_counter)."""
    st = _state(env)
    step = int(env.common_step_counter)
    if st["seq"] == step:
        return st
    st["seq"] = step
    obj_true = bp_obs.object_positions_b(env)[:, :3]
    tcp = bp_obs.tcp_pose_b(env)
    tcp_pos, tcp_quat = tcp[:, :3], tcp[:, 3:7]
    closed = env._bp_gripper_command_state >= GRIPPER_CLOSED_THRESHOLD

    visible = cube_visible(st, obj_true, tcp_pos, tcp_quat)
    detected = visible & (torch.rand(env.num_envs, device=env.device) < st["detect_prob"])
    noise = torch.randn_like(obj_true)
    noise[:, :2] *= st["xy_sigma"].unsqueeze(1)
    noise[:, 2] *= st["z_sigma"]
    center = obj_true + st["bias"] + noise
    near = (center - tcp_pos).norm(dim=1) <= HELD_RADIUS_M

    # ── held latch (실기 TeacherObsBuilder와 같은 규칙) ──
    fallback = torch.stack([tcp_pos[:, 0], tcp_pos[:, 1], torch.full_like(tcp_pos[:, 2], TABLE_CENTER_Z)], dim=1)
    est_prev = torch.where(st["has_est"].unsqueeze(1), st["est"], fallback)
    newly_closed = closed & ~st["closed_prev"]
    held = torch.where(newly_closed, (est_prev - tcp_pos).norm(dim=1) <= HELD_RADIUS_M, st["held"])
    held = torch.where(detected & closed, near, held)
    held = held & closed

    # ── 추정치 합성 ──
    snapped = center.clone()
    snapped[:, 2] = torch.where(held, center[:, 2], torch.full_like(center[:, 2], TABLE_CENTER_Z))
    held_est = tcp_pos + st["held_offset"]
    est = torch.where(
        detected.unsqueeze(1), snapped,
        torch.where(held.unsqueeze(1), held_est, est_prev),
    )
    st["est"] = est
    st["has_est"][:] = True
    st["held"] = held
    st["closed_prev"] = closed
    st["detected"] = detected
    return st


def _ready(env) -> bool:
    # 매니저 초기화(차원 파악) 단계: 버퍼가 아직 없다 — observations.policy_obs와 같은 가드.
    return hasattr(env, "episode_length_buf") and getattr(env, "_bp_gripper_command_state", None) is not None


def object_pos_estimate(env) -> torch.Tensor:
    """큐브 위치 ★추정치★ (N,3) base — Teacher v2 actor 관측(배포 인식 모듈과 같은 규칙)."""
    if not _ready(env):
        return torch.zeros(env.num_envs, 3, device=env.device)
    return _update(env)["est"]


def held_estimate(env) -> torch.Tensor:
    """'쥐었다' latch 추정치 (N,1) — 폐합 전이 시점의 추정치-TCP 근접으로 판정한 배포 규칙."""
    if not _ready(env):
        return torch.zeros(env.num_envs, 1, device=env.device)
    return _update(env)["held"].float().unsqueeze(1)


def gripper_command_state_obs(env) -> torch.Tensor:
    """지연·전이 적분된 그리퍼 명령 상태 (N,1) — student flat 관측과 같은 신호(common/gripper_command)."""
    state = getattr(env, "_bp_gripper_command_state", None)
    if state is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return state.unsqueeze(1)


def detection_rate(env) -> torch.Tensor:
    """진단용: 이번 스텝 검출 여부 (N,) float — 리허설·프로브가 검출률을 집계할 때 쓴다."""
    if not _ready(env):
        return torch.zeros(env.num_envs, device=env.device)
    return _update(env)["detected"].float()
