# ★요소1: 단일 공유 관측 전처리★
# sim(_binpick_obs.py의 단일 ObsTerm)과 real(rollout.py 루프)이 "문자 그대로
# 같은 함수"를 호출한다. 분포 정렬이 어긋나면 이 파일이 아니라 입력측
# (real.local.yaml 필터 / dr.yaml 범위)을 고친다 — 이 함수를 sim/real로 분기하는 것은 금지.
#
# 입력 유연성 계약:
#   - numpy/torch 모두 허용, batch 없음([H,W])/있음([N,H,W]) 모두 허용
#   - real은 N=1, sim은 N=num_envs — 완전히 같은 코드 경로를 지난다
# flat 레이아웃(순서 고정, obs_spec.json에 기록):
#   [ depth(h*w), (use_rgb 시) rgb(h*w*3, C-H-W 평탄화), joint_pos(6),
#     joint_vel(6), gripper_command_state(1), tcp_pose(7), target_pos(3),
#     last_action(action_dim) ]
# ★2026-07-17 RGB-D 확장(B안)★: rgb 블록은 cfg.use_rgb 계약 파라미터로 켠다 —
#   sim/real 분기가 아니라 양쪽이 같은 값을 읽는 계약(§4-3). 켜면 obs_dim이
#   바뀌는 '계약 변경'(재학습+재export). 현행 112² RGB-D + 그리퍼 관측 = 50,206.
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from common.gripper_command import validate_gripper_command_state
from common.types import CameraFrame, PreprocessCfg, RobotState

NUM_ARM_JOINTS = 6
TCP_POSE_DIM = 7
TARGET_DIM = 3
GRIPPER_COMMAND_DIM = 1


def obs_layout(cfg: PreprocessCfg) -> list[tuple[str, int]]:
    """flat 관측의 (이름, 크기) 순서 — 이 리스트가 sim-real 계약서의 본문이다."""
    layout: list[tuple[str, int]] = [
        ("depth", cfg.out_height * cfg.out_width),
    ]
    if cfg.use_rgb:
        # rgb 블록: [3,h,w] float[0,1]을 C-H-W 순서로 평탄화(split_flat_obs가 복원)
        layout.append(("rgb", cfg.out_height * cfg.out_width * 3))
    layout += [
        ("joint_pos", NUM_ARM_JOINTS),
        ("joint_vel", NUM_ARM_JOINTS),
        # Delayed/accepted binary command state only.  It is not physical
        # finger position, gripper width, or grasp feedback.
        ("gripper_command_state", GRIPPER_COMMAND_DIM),
        ("tcp_pose", TCP_POSE_DIM),
        ("target_pos", TARGET_DIM),
        ("last_action", cfg.action_dim),
    ]
    return layout


def obs_dim(cfg: PreprocessCfg) -> int:
    return sum(size for _, size in obs_layout(cfg))


def _as_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _ensure_batch_2d(x: torch.Tensor) -> torch.Tensor:
    """[H,W] → [1,H,W] / [N,H,W] 그대로."""
    if x.dim() == 2:
        return x.unsqueeze(0)
    if x.dim() == 3:
        return x
    raise ValueError(f"depth는 [H,W] 또는 [N,H,W]여야 함: {tuple(x.shape)}")


def _ensure_batch_1d(x: torch.Tensor, dim: int, name: str) -> torch.Tensor:
    """[D] → [1,D] / [N,D] 그대로 + 크기 검증."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2 or x.shape[1] != dim:
        raise ValueError(f"{name}: 기대 (N,{dim}), 실제 {tuple(x.shape)}")
    return x


def canonicalize_quat_wxyz(quat) -> torch.Tensor:
    """Return one deterministic representative of each ``q == -q`` rotation.

    The primary convention is ``w >= 0``.  For the exact 180-degree case
    (``w == 0``), the first non-zero vector component is made positive so that
    ``q`` and ``-q`` still produce byte-for-byte identical observations.
    """
    q = _as_tensor(quat)
    if q.shape[-1] != 4:
        raise ValueError(f"quat_wxyz: expected final dimension 4, got {tuple(q.shape)}")

    nonzero = q.ne(0)
    first_index = nonzero.to(torch.int64).argmax(dim=-1, keepdim=True)
    pivot = q.gather(-1, first_index)
    has_nonzero = nonzero.any(dim=-1, keepdim=True)
    flip = has_nonzero & pivot.lt(0)
    return torch.where(flip, -q, q)


def _center_crop_square(depth: torch.Tensor, crop_fraction: float = 1.0,
                        out_size: int = 0) -> torch.Tensor:
    """[N,H,W] → 정사각 center-crop (변 = min(h,w) × crop_fraction).

    FOV 정합 계약(configs/common.yaml preprocess.crop_fraction, 2026-07-14):
      · real → min변 × fraction 픽셀 크롭. 유효 화각은 ★tan-공간★으로 준다:
        vfov_eff = 2·atan(fraction·tan(vfov/2)) — 기준 vfov는 현행 계약(2026-07-17
        RGB-D)에서 D435i ★color★ 42.5°(sim.yaml vertical_fov_deg, rs.align 후).
      · sim은 같은 유효 FOV로 '직접' 협FOV 렌더(scene.make_tiled_camera_cfg —
        위 tan-공간 산식과 동일 공식 사용) — 렌더가 이미 크롭 등가이므로 여기서
        재크롭하면 이중 크롭이 된다. 이를 입력 크기 규칙으로 방지: ★크롭 변은
        out_size 미만으로 내려가지 않는다★ (112² sim 프레임 × fraction<1 → 112 유지
        = no-op). 순수 크기 규칙이라 sim/real 분기 없이 한 코드 경로(§4-3).
    """
    n, h, w = depth.shape
    side = min(h, w)
    if crop_fraction < 1.0:
        side = max(int(out_size) if out_size else 1, int(round(side * crop_fraction)))
        side = min(side, min(h, w))
    top = (h - side) // 2
    left = (w - side) // 2
    return depth[:, top : top + side, left : left + side]


def preprocess_depth(depth_m, cfg: PreprocessCfg) -> torch.Tensor:
    """raw depth[m] → [N,1,out_h,out_w] 정규화 텐서 ([0,1], 결측=1.0(far)).

    경로: 결측(<=0/NaN/inf)→far → clip(near,far) → 정사각 crop
          → resize(nearest: depth 경계 블리딩 방지·양쪽 결정적) → (d-near)/(far-near)
    """
    d = _ensure_batch_2d(_as_tensor(depth_m))
    far = cfg.depth_far_m
    # ★최소거리 무효화(2026-07-14, eye-in-hand 전환)★ — near 미만은 clamp가 아니라
    #   ★무효→far★로 처리한다: real D435i는 최소거리(~0.28m) 미만에서 depth=0(무효)을
    #   반환하므로, sim(렌더는 실거리 반환)도 같은 규칙을 적용해야 파지 최종 구간의
    #   'depth 결손' 분포가 sim=real 일치한다(§4-3). 구 동작(near로 clamp)은 sim만
    #   근접값을 유지해 실기와 어긋났다.
    invalid = ~torch.isfinite(d) | (d <= 0.0) | (d < cfg.depth_near_m)
    d = torch.where(invalid, torch.full_like(d, far), d)
    d = d.clamp(cfg.depth_near_m, far)
    # ★frame_mode(2026-07-18)★: "square"=정사각 crop(구 계약) / "full"=전체 프레임
    #   비등방 리사이즈(가로 FOV 69.4° 활용 — 왜곡이 sim/real 동일하므로 계약 성립).
    if cfg.frame_mode != "full":
        d = _center_crop_square(d, cfg.crop_fraction, min(cfg.out_height, cfg.out_width))
    d = F.interpolate(
        d.unsqueeze(1), size=(cfg.out_height, cfg.out_width), mode="nearest"
    )
    return (d - cfg.depth_near_m) / (far - cfg.depth_near_m)


def _ensure_batch_rgb(x: torch.Tensor) -> torch.Tensor:
    """[H,W,3] → [1,H,W,3] / [N,H,W,3] 그대로 + 채널 검증."""
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if x.dim() != 4 or x.shape[-1] != 3:
        raise ValueError(f"rgb는 [H,W,3] 또는 [N,H,W,3]이어야 함: {tuple(x.shape)}")
    return x


def preprocess_rgb(rgb, cfg: PreprocessCfg) -> torch.Tensor:
    """raw RGB → [N,3,out_h,out_w] 정규화 텐서 ([0,1]).

    입력 계약: uint8(0~255) 또는 float(이미 [0,1]) — RGB 채널 순서(BGR 금지).
      sim: TiledCamera 'rgb' annotator 출력([N,H,W,3] uint8) 또는 노이즈 주입 후
           float[0,1] / real: rs.align(depth→color) 후 color 프레임(uint8).
    경로: /255(uint8만) → clamp[0,1] → depth와 동일 정사각 crop
          → resize(bilinear+antialias: real 480→112 다운샘플 표준. sim도 현재
          카메라 원본에서 112×112 계약으로 맞춤 — Phase3
          Verify_ObsParity가 분포 대조로 검증) → [N,3,h,w].
    depth와 crop 규칙이 동일해야 화각 정합이 유지된다(같은 _center_crop_square).
    """
    # ★스케일 판별은 '원본 dtype'으로(2026-07-17 적대 리뷰 수정)★ — 값 휴리스틱
    #   (max>1.5)만 쓰면 극암 uint8 프레임(전 픽셀 ≤1: 렌즈 가림/노출 실패)이
    #   float[0,1]로 오판돼 /255가 생략, 픽셀 1이 255배 과대(1.0)로 주입된다
    #   (real 전용 — sim은 사전 /255라 원리적으로 미검출). 정수형 입력 = 8bit 계약.
    if isinstance(rgb, torch.Tensor):
        is_int = not torch.is_floating_point(rgb)
    else:
        is_int = np.issubdtype(np.asarray(rgb).dtype, np.integer)
    x = _ensure_batch_rgb(_as_tensor(rgb))
    if is_int or x.max() > 1.5:  # 값 휴리스틱은 '0~255 스케일 float' 방어용 폴백만
        x = x / 255.0
    x = x.clamp(0.0, 1.0)
    x = x.permute(0, 3, 1, 2)  # [N,3,H,W]
    n, c, h, w = x.shape
    # depth와 동일한 프레임 기하(frame_mode) — square면 동일 crop 산식 재사용
    if cfg.frame_mode != "full":
        x = _center_crop_square(
            x.reshape(n * c, h, w), cfg.crop_fraction, min(cfg.out_height, cfg.out_width)
        )
        side = x.shape[-1]
        x = x.reshape(n, c, side, side)
    return F.interpolate(
        x, size=(cfg.out_height, cfg.out_width), mode="bilinear",
        align_corners=False, antialias=True,
    ).clamp(0.0, 1.0)


def preprocess_proprio(
    joint_pos, joint_vel, tcp_pose, target_pos, last_action, cfg: PreprocessCfg,
    *, gripper_command_state,
) -> torch.Tensor:
    """proprioception + 목표 → [N, 30] 정규화 벡터 (스케일은 cfg·robot_spec 단일 소스).

    target_pos: 에피소드 가변 place 목표(base frame [m]) — 사용자 결정(2026-07-04)
    gripper_command_state: scalar or (N,)/(N,1), already in [0,1].
      0=open, 1=close.  This is a deterministic delayed-command ramp and not
      measured gripper opening/width.  Unknown, non-finite, and out-of-range
      values are rejected rather than clamped.
    """
    q = _ensure_batch_1d(_as_tensor(joint_pos), NUM_ARM_JOINTS, "joint_pos")
    qd = _ensure_batch_1d(_as_tensor(joint_vel), NUM_ARM_JOINTS, "joint_vel")
    tcp = _ensure_batch_1d(_as_tensor(tcp_pose), TCP_POSE_DIM, "tcp_pose")
    tgt = _ensure_batch_1d(_as_tensor(target_pos), TARGET_DIM, "target_pos")
    act = _ensure_batch_1d(_as_tensor(last_action), cfg.action_dim, "last_action")
    if gripper_command_state is None:
        raise ValueError("gripper_command_state is unknown/invalid")
    grip = _as_tensor(gripper_command_state)
    if grip.dim() == 0:
        grip = grip.reshape(1, 1)
    elif grip.dim() == 1:
        grip = grip.reshape(-1, 1)
    elif grip.dim() != 2 or grip.shape[1] != GRIPPER_COMMAND_DIM:
        raise ValueError(
            "gripper_command_state must be scalar, (N,), or (N,1), "
            f"got {tuple(grip.shape)}"
        )
    if grip.shape[0] == 1 and q.shape[0] > 1:
        grip = grip.expand(q.shape[0], GRIPPER_COMMAND_DIM)
    grip = _ensure_batch_1d(grip, GRIPPER_COMMAND_DIM, "gripper_command_state")
    if grip.shape[0] != q.shape[0]:
        raise ValueError(
            "gripper_command_state batch must be 1 or match joint_pos: "
            f"got {grip.shape[0]} vs {q.shape[0]}"
        )
    if not torch.isfinite(grip).all() or not ((grip >= 0.0) & (grip <= 1.0)).all():
        raise ValueError("gripper_command_state must be finite and within [0,1]")

    tcp_scaled = torch.cat(
        [tcp[:, :3] / cfg.tcp_pos_scale, canonicalize_quat_wxyz(tcp[:, 3:])], dim=1
    )  # quat은 이미 단위 크기 — 스케일 없음
    return torch.cat(
        [
            q / cfg.joint_pos_scale,
            qd / cfg.joint_vel_scale,
            grip,
            tcp_scaled,
            tgt / cfg.tcp_pos_scale,
            act,  # 이미 [-1,1]
        ],
        dim=1,
    )


def build_policy_obs(
    depth_m, joint_pos, joint_vel, tcp_pose, target_pos, last_action,
    cfg: PreprocessCfg, *, gripper_command_state, rgb=None,
) -> torch.Tensor:
    """★정책 입력 최종 조립 — flat [N, obs_dim] 반환.★

    sim: _binpick_obs.py의 '단일' ObsTerm이 이 함수를 그대로 호출해 flat 벡터를
         반환한다 → ObservationManager의 concat 순서에 의존하지 않고 레이아웃이
         구조적으로 고정된다.
    real: rollout.py가 build_policy_obs_from_state 경유로 같은 함수를 호출한다.
    rgb(키워드 전용 의미): cfg.use_rgb=True면 필수(RGB-D 계약), False면 무시
    (호출측이 항상 전달해도 안전 — 계약은 cfg가 결정).
    """
    depth = preprocess_depth(depth_m, cfg)  # [N,1,h,w]
    proprio = preprocess_proprio(
        joint_pos, joint_vel, tcp_pose, target_pos, last_action, cfg,
        gripper_command_state=gripper_command_state,
    )
    if depth.shape[0] != proprio.shape[0]:
        raise ValueError(
            f"depth batch({depth.shape[0]}) != proprio batch({proprio.shape[0]})"
        )
    parts = [depth.flatten(start_dim=1)]
    if cfg.use_rgb:
        if rgb is None:
            raise ValueError(
                "use_rgb 계약인데 rgb 입력이 없음 — sim은 TiledCamera 'rgb' 출력, "
                "real은 rs.align된 color 프레임을 전달해야 한다(CameraFrame.rgb)"
            )
        rgb_t = preprocess_rgb(rgb, cfg)  # [N,3,h,w]
        if rgb_t.shape[0] != depth.shape[0]:
            raise ValueError(
                f"rgb batch({rgb_t.shape[0]}) != depth batch({depth.shape[0]})"
            )
        parts.append(rgb_t.flatten(start_dim=1))  # C-H-W 순 평탄화(레이아웃 계약)
    parts.append(proprio)
    flat = torch.cat(parts, dim=1)
    expected = obs_dim(cfg)
    if flat.shape[1] != expected:
        raise ValueError(f"obs_dim 불일치: 기대 {expected}, 실제 {flat.shape[1]}")
    return flat


def build_policy_obs_from_state(
    frame: CameraFrame,
    state: RobotState,
    target_pos: np.ndarray,
    last_action: np.ndarray,
    cfg: PreprocessCfg,
) -> torch.Tensor:
    """real 편의 래퍼: (CameraFrame, RobotState) → build_policy_obs (N=1)."""
    return build_policy_obs(
        frame.depth_m,
        state.joint_pos,
        state.joint_vel,
        state.tcp_pose,
        target_pos,
        last_action,
        cfg,
        gripper_command_state=validate_gripper_command_state(
            state.gripper_command_state
        ),
        rgb=frame.rgb,
    )
