"""요소1 확장(RGB-D 계약, 2026-07-17 B안) 단위테스트 — 시뮬레이터 불필요(순수 torch/numpy).

use_rgb 계약 파라미터에 따른 obs_layout/obs_dim 전환(4126↔16414), preprocess_rgb의
uint8→[0,1]/float 패스스루/배치/크롭·리사이즈 형상, depth와의 crop 산식 동일성,
build_policy_obs의 rgb 포함 왕복(레이아웃 슬라이스 복원 — split_flat_obs 교차 검증),
use_rgb 위반 시 ValueError, CameraFrame.rgb 경유(build_policy_obs_from_state)를
★회귀 가드★로 잠근다. 기대값은 전부 계약(코드/common.yaml 주석)에서 계산했다.

실행: python -m pytest tests/test_obs_preprocess_rgb.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from common.obs_preprocess import (
    build_policy_obs,
    build_policy_obs_from_state,
    obs_dim,
    obs_layout,
    preprocess_depth,
    preprocess_rgb,
)
from common.types import CameraFrame, PreprocessCfg, RobotState

# 64x64 기준 cfg — test_obs_preprocess.py의 CFG와 동일 수치(파일 관례 미러)
CFG = PreprocessCfg(
    depth_near_m=0.15,
    depth_far_m=1.20,
    out_height=64,
    out_width=64,
    joint_pos_scale=np.pi,
    joint_vel_scale=np.pi,
    tcp_pos_scale=1.0,
    action_dim=7,
)
CFG_RGB = replace(CFG, use_rgb=True)

DEPTH_BLOCK = 64 * 64          # 4096
RGB_BLOCK = 64 * 64 * 3        # 12288
PROPRIO = 6 + 6 + 7 + 3 + 7    # 29


# ----------------------------------------------------- obs_layout / obs_dim ----


def test_obs_dim_contract_both_switches():
    """use_rgb=False → 4126 / True → 16414 (64² 기준 — common.yaml 주석 계약)."""
    assert obs_dim(CFG) == 4126
    assert obs_dim(CFG_RGB) == 16414
    assert obs_dim(CFG_RGB) - obs_dim(CFG) == RGB_BLOCK


def test_obs_layout_rgb_block_position():
    """rgb 블록은 ★depth 바로 다음★(인덱스 1) — 나머지 순서는 depth-only와 동일."""
    layout = obs_layout(CFG_RGB)
    assert [name for name, _ in layout] == [
        "depth", "rgb", "joint_pos", "joint_vel", "gripper_command_state", "tcp_pose", "target_pos", "last_action",
    ]
    assert dict(layout)["rgb"] == RGB_BLOCK
    assert dict(layout)["depth"] == DEPTH_BLOCK
    # use_rgb=False 레이아웃에서 rgb만 빠지고 순서는 동일
    names_no_rgb = [name for name, _ in obs_layout(CFG)]
    assert names_no_rgb == [n for n, _ in layout if n != "rgb"]
    assert obs_dim(CFG_RGB) == sum(s for _, s in layout)


# ----------------------------------------------------------- preprocess_rgb ----


def test_rgb_uint8_scaled_to_unit():
    """uint8(0~255) 입력 → /255 → [0,1] (0→0.0, 255→1.0, 128→128/255)."""
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[..., 0] = 0
    rgb[..., 1] = 128
    rgb[..., 2] = 255
    out = preprocess_rgb(rgb, CFG_RGB)
    assert out.shape == (1, 3, 64, 64)
    np.testing.assert_allclose(out[0, 0].numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 1].numpy(), 128.0 / 255.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 2].numpy(), 1.0, atol=1e-6)


def test_rgb_float01_passthrough():
    """float[0,1] 입력은 재스케일 없이 패스스루 (64→64 리사이즈는 항등 — 실측 0.0 오차)."""
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 1.0, size=(64, 64, 3)).astype(np.float32)
    out = preprocess_rgb(rgb, CFG_RGB)
    np.testing.assert_allclose(out[0].permute(1, 2, 0).numpy(), rgb, atol=1e-6)


def test_rgb_float_not_rescaled_even_near_one():
    """float 입력의 최댓값이 1.0이어도 /255를 타지 않는다 (판별 임계 1.5 계약)."""
    rgb = np.full((64, 64, 3), 1.0, dtype=np.float32)
    out = preprocess_rgb(rgb, CFG_RGB)
    np.testing.assert_allclose(out.numpy(), 1.0, atol=1e-6)


def test_rgb_batch_shapes():
    """[H,W,3] → [1,3,h,w] / [N,H,W,3] → [N,3,h,w] — 배치 유무 동일 경로."""
    rng = np.random.default_rng(1)
    frame = rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32)
    single = preprocess_rgb(frame, CFG_RGB)
    batch = preprocess_rgb(np.stack([frame, frame, frame]), CFG_RGB)
    assert single.shape == (1, 3, 64, 64) and batch.shape == (3, 3, 64, 64)
    for i in range(3):
        assert torch.allclose(batch[i], single[0], atol=1e-6)


def test_rgb_848x480_center_crop_resize():
    """real D435i 848x480 color → 중앙 480² 크롭 → 64² (depth 크롭 테스트 미러).

    좌/우 마진에 다른 마커를 넣고 중앙 정사각만 남는지 확인 —
    left-crop이면 좌측 마커(0.3)가 출력에 섞인다.
    """
    h, w = 480, 848
    left = (w - h) // 2
    rgb = np.full((h, w, 3), 0.3, dtype=np.float32)
    rgb[:, left + h :, :] = 0.7
    rgb[:, left : left + h, :] = 0.5
    out = preprocess_rgb(rgb, CFG_RGB)
    assert out.shape == (1, 3, 64, 64)
    assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-6), (
        "중앙 크롭이 아님 — 마진 마커가 출력에 섞임"
    )


def _expected_crop_side(h: int, w: int, crop_fraction: float, out: int) -> int:
    """_center_crop_square 계약 산식 미러: side=min(min(h,w), max(out, round(min·frac)))."""
    side = min(h, w)
    if crop_fraction < 1.0:
        side = max(out, int(round(side * crop_fraction)))
        side = min(side, min(h, w))
    return side


@pytest.mark.parametrize(
    "h,w,frac,out",
    [(480, 848, 1.0, 64), (480, 848, 0.75, 64), (100, 100, 0.5, 32), (64, 64, 0.6, 64)],
)
def test_rgb_and_depth_same_crop_rule(h, w, frac, out):
    """★depth와 crop 산식 동일성(화각 정합 계약)★ — 같은 crop_fraction에서 같은 side.

    계약 산식으로 계산한 크롭 영역에만 중앙 마커를 채우고, depth/rgb 양쪽 출력이
    마진 마커 없이 순수 중앙값이면 두 경로의 크롭 영역이 동일하다는 증명이 된다
    (한쪽이라도 다른 side를 쓰면 마진값이 섞여 실패). 64² 입력 × frac<1은
    out_size 하한으로 no-op(sim 이중 크롭 방지 규칙)인 것도 함께 커버.
    """
    cfg = PreprocessCfg(0.15, 1.20, out, out, np.pi, np.pi, 1.0, 7,
                        crop_fraction=frac, use_rgb=True)
    side = _expected_crop_side(h, w, frac, out)
    top, left = (h - side) // 2, (w - side) // 2

    margin_d, center_d = 0.9, 0.6  # depth [m]
    depth = np.full((h, w), margin_d, dtype=np.float32)
    depth[top : top + side, left : left + side] = center_d
    out_d = preprocess_depth(depth, cfg)
    exp_d = (center_d - cfg.depth_near_m) / (cfg.depth_far_m - cfg.depth_near_m)
    assert torch.allclose(out_d, torch.full_like(out_d, exp_d), atol=1e-6), (
        f"depth 크롭 side≠{side} 의심 — 마진값 혼입"
    )

    rgb = np.full((h, w, 3), 0.9, dtype=np.float32)
    rgb[top : top + side, left : left + side, :] = 0.4
    out_rgb = preprocess_rgb(rgb, cfg)
    assert torch.allclose(out_rgb, torch.full_like(out_rgb, 0.4), atol=1e-6), (
        f"rgb 크롭 side≠{side} 의심 — depth와 crop 산식이 어긋남"
    )


def test_rgb_out_of_range_float_clamped():
    """[0,1] 밖 float(노이즈 주입 후 등)은 clamp — 출력은 항상 [0,1]."""
    rgb = np.full((64, 64, 3), 1.4, dtype=np.float32)  # >1.5 아님 → /255 미적용
    out = preprocess_rgb(rgb, CFG_RGB)
    np.testing.assert_allclose(out.numpy(), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "shape", [(64, 64), (64, 64, 4), (2, 64, 64, 1), (1, 2, 64, 64, 3)]
)
def test_rgb_bad_shape_raises(shape):
    """rgb 형상이 [H,W,3]/[N,H,W,3]이 아니면 ValueError."""
    with pytest.raises(ValueError):
        preprocess_rgb(np.zeros(shape, dtype=np.float32), CFG_RGB)


# -------------------------------------------------------- build_policy_obs ----


def _mk_inputs_rgb(n_env: int | None = None):
    """마커 값이 든 입력 셋 (test_obs_preprocess._mk_inputs 미러 + rgb 채널 마커)."""
    depth_val = 0.15 + 0.25 * (1.20 - 0.15)  # 정규화 후 정확히 0.25
    depth = np.full((64, 64), depth_val, dtype=np.float32)
    rgb = np.zeros((64, 64, 3), dtype=np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.2, 0.5, 0.8  # 채널별 고유 마커
    q = np.full(6, 0.5 * np.pi)
    qd = np.full(6, -0.5 * np.pi)
    tcp = np.array([0.7, 0.7, 0.7, 1.0, 0.0, 0.0, 0.0])
    tgt = np.full(3, 0.9)
    act = np.full(7, 0.3)
    if n_env is not None:
        depth = np.stack([depth] * n_env)
        rgb = np.stack([rgb] * n_env)
        q, qd, tcp, tgt, act = (np.stack([a] * n_env) for a in (q, qd, tcp, tgt, act))
    return depth, rgb, q, qd, tcp, tgt, act


def test_build_policy_obs_rgb_layout_roundtrip():
    """★RGB 레이아웃 왕복 회귀 가드★ — 슬라이스 위치·C-H-W 평탄화 순서 고정.

    [ depth(4096) | rgb(12288: R 4096, G 4096, B 4096) | proprio(29) ]
    """
    depth, rgb, q, qd, tcp, tgt, act = _mk_inputs_rgb()
    flat = build_policy_obs(
        depth, q, qd, tcp, tgt, act, CFG_RGB,
        gripper_command_state=0.0, rgb=rgb,
    )
    assert flat.shape == (1, 16414)
    v = flat[0].numpy()
    i = 0
    np.testing.assert_allclose(v[i : i + DEPTH_BLOCK], 0.25, atol=1e-6)  # depth
    i += DEPTH_BLOCK
    # rgb: C-H-W 평탄화 → R 평면 4096, G 평면 4096, B 평면 4096 순
    np.testing.assert_allclose(v[i : i + 4096], 0.2, atol=1e-6)
    np.testing.assert_allclose(v[i + 4096 : i + 8192], 0.5, atol=1e-6)
    np.testing.assert_allclose(v[i + 8192 : i + 12288], 0.8, atol=1e-6)
    i += RGB_BLOCK
    np.testing.assert_allclose(v[i : i + 6], 0.5, atol=1e-6)  # joint_pos
    i += 6
    np.testing.assert_allclose(v[i : i + 6], -0.5, atol=1e-6)  # joint_vel
    i += 6
    np.testing.assert_allclose(v[i : i + 1], 0.0, atol=1e-6)  # command state open
    i += 1
    np.testing.assert_allclose(
        v[i : i + 7], [0.7, 0.7, 0.7, 1.0, 0.0, 0.0, 0.0], atol=1e-6
    )  # tcp_pose
    i += 7
    np.testing.assert_allclose(v[i : i + 3], 0.9, atol=1e-6)  # target_pos
    i += 3
    np.testing.assert_allclose(v[i : i + 7], 0.3, atol=1e-6)  # last_action
    assert i + 7 == 16414


def test_build_policy_obs_rgb_split_flat_obs_crosscheck():
    """★split_flat_obs 교차 검증★ — flat 왕복이 전처리 텐서를 그대로 복원.

    vision_policy.split_flat_obs(확장 패키지 — torch만 의존)는 obs_layout을
    소비하는 유일한 역변환. 배포 PC에는 확장이 없으므로(§8-2) 미설치면 skip.
    """
    vision_policy = pytest.importorskip(
        "ur5e_binpick.tasks.binpick.agents.vision_policy",
        reason="학습 확장 미설치(배포 PC) — sim-side 교차 검증은 학습 PC 전용",
    )
    depth, rgb, q, qd, tcp, tgt, act = _mk_inputs_rgb(n_env=2)
    flat = build_policy_obs(
        depth, q, qd, tcp, tgt, act, CFG_RGB,
        gripper_command_state=0.0, rgb=rgb,
    )
    parts = vision_policy.split_flat_obs(flat, CFG_RGB)
    assert set(parts) == {
        "depth", "rgb", "joint_pos", "joint_vel", "gripper_command_state", "tcp_pose", "target_pos", "last_action",
    }
    assert parts["depth"].shape == (2, 1, 64, 64)
    assert parts["rgb"].shape == (2, 3, 64, 64)
    # 복원 텐서 == 전처리 함수 직접 호출 결과 (평탄화·복원이 완전 왕복)
    assert torch.equal(parts["depth"], preprocess_depth(depth, CFG_RGB))
    assert torch.equal(parts["rgb"], preprocess_rgb(rgb, CFG_RGB))
    np.testing.assert_allclose(parts["last_action"].numpy(), np.full((2, 7), 0.3), atol=1e-6)


def test_build_policy_obs_use_rgb_without_rgb_raises():
    """use_rgb 계약인데 rgb=None → ValueError (조용한 진행 금지)."""
    depth, _, q, qd, tcp, tgt, act = _mk_inputs_rgb()
    with pytest.raises(ValueError, match="use_rgb"):
        build_policy_obs(
            depth, q, qd, tcp, tgt, act, CFG_RGB,
            gripper_command_state=0.0,
        )


def test_build_policy_obs_no_rgb_cfg_ignores_rgb():
    """use_rgb=False면 rgb를 넘겨도 무시 — depth-only(4126)와 완전 동일 출력."""
    depth, rgb, q, qd, tcp, tgt, act = _mk_inputs_rgb()
    with_rgb = build_policy_obs(
        depth, q, qd, tcp, tgt, act, CFG,
        gripper_command_state=0.0, rgb=rgb,
    )
    without = build_policy_obs(
        depth, q, qd, tcp, tgt, act, CFG, gripper_command_state=0.0
    )
    assert with_rgb.shape == (1, 4126)
    assert torch.equal(with_rgb, without)


def test_build_policy_obs_rgb_batch_mismatch_raises():
    """rgb batch(2) != depth batch(1) → ValueError."""
    depth, rgb, q, qd, tcp, tgt, act = _mk_inputs_rgb()
    rgb2 = np.stack([rgb, rgb])
    with pytest.raises(ValueError):
        build_policy_obs(
            depth, q, qd, tcp, tgt, act, CFG_RGB,
            gripper_command_state=0.0, rgb=rgb2,
        )


def test_build_policy_obs_rgb_batchN():
    """N=4 배치도 같은 함수·같은 결과 (sim 경로 등가)."""
    depth, rgb, q, qd, tcp, tgt, act = _mk_inputs_rgb(n_env=4)
    flat = build_policy_obs(
        depth, q, qd, tcp, tgt, act, CFG_RGB,
        gripper_command_state=0.0, rgb=rgb,
    )
    assert flat.shape == (4, 16414)
    for i in range(1, 4):
        assert torch.equal(flat[i], flat[0])


# ---------------------------------------------- build_policy_obs_from_state ----


def _mk_frame_state():
    """real 경로 입력(CameraFrame uint8 rgb + RobotState) 생성."""
    depth, rgb_f, q, qd, tcp, tgt, act = _mk_inputs_rgb()
    rgb_u8 = np.round(rgb_f * 255.0).astype(np.uint8)  # real: uint8 RGB 계약
    frame = CameraFrame(depth_m=depth, timestamp=0.0, rgb=rgb_u8)
    state = RobotState(
        joint_pos=q, joint_vel=qd, tcp_pose=tcp, tcp_force=None,
        timestamp=0.0, gripper_command_state=0.0,
    )
    return frame, state, tgt, act, rgb_u8


def test_build_policy_obs_from_state_rgb_path():
    """CameraFrame.rgb 경유(real N=1 경로) == build_policy_obs 직접 호출과 동일."""
    frame, state, tgt, act, rgb_u8 = _mk_frame_state()
    flat = build_policy_obs_from_state(frame, state, tgt, act, CFG_RGB)
    assert flat.shape == (1, 16414)
    direct = build_policy_obs(
        frame.depth_m, state.joint_pos, state.joint_vel, state.tcp_pose,
        tgt, act, CFG_RGB,
        gripper_command_state=state.gripper_command_state, rgb=rgb_u8,
    )
    assert torch.equal(flat, direct)


def test_build_policy_obs_from_state_rgb_none_raises():
    """use_rgb 계약인데 CameraFrame.rgb=None(구식 depth-only 프레임) → ValueError."""
    frame, state, tgt, act, _ = _mk_frame_state()
    frame.rgb = None
    with pytest.raises(ValueError, match="use_rgb"):
        build_policy_obs_from_state(frame, state, tgt, act, CFG_RGB)


def test_build_policy_obs_from_state_depth_only_allows_rgb_none():
    """use_rgb=False 계약이면 rgb=None 프레임도 정상 (depth-only ablation 경로)."""
    frame, state, tgt, act, _ = _mk_frame_state()
    frame.rgb = None
    flat = build_policy_obs_from_state(frame, state, tgt, act, CFG)
    assert flat.shape == (1, 4126)


def test_preprocess_rgb_near_black_uint8_still_divided_by_255():
    """★극암 uint8 프레임(전 픽셀 ≤1) 회귀 고정(2026-07-17 적대 리뷰)★.

    구 구현의 값 휴리스틱(max>1.5)은 렌즈 가림/노출 실패 프레임(uint8, max≤1)을
    float[0,1]로 오판해 /255를 생략 — 픽셀 1이 255배 과대(1.0)로 주입됐다.
    수정: 원본 dtype(정수형)으로 판별 → 극암 프레임도 1/255≈0.0039 스케일 유지.
    """
    from common.obs_preprocess import preprocess_rgb

    dark = np.zeros((480, 848, 3), dtype=np.uint8)
    dark[::2] = 1  # 값 1 픽셀 — 오염 시 1.0(최대 밝기)이 된다
    out = preprocess_rgb(dark, CFG_RGB)
    assert float(out.max()) < 0.01, "극암 uint8 프레임이 255배 과대 주입됨(/255 생략)"
    # float 0~255 입력은 폴백 휴리스틱으로 여전히 정규화(방어 경로 유지)
    f255 = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.float32)
    assert float(preprocess_rgb(f255, CFG_RGB).max()) <= 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
