"""요소1(공유 관측 전처리) 단위테스트 — 시뮬레이터 불필요(순수 torch/numpy).

preprocess_depth의 결측 처리/clip/center-crop/nearest resize 결정성,
preprocess_proprio의 스케일 수식, build_policy_obs의 flat 레이아웃(순서 고정)을
★회귀 가드★로 잠근다. 기대값은 전부 계약(코드/common.yaml 주석)에서 계산했다.

실행: python -m pytest tests/test_obs_preprocess.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from common.config_io import build_preprocess_cfg, configs_dir, load_yaml
from common.obs_preprocess import (
    NUM_ARM_JOINTS,
    TARGET_DIM,
    TCP_POSE_DIM,
    build_policy_obs,
    canonicalize_quat_wxyz,
    obs_dim,
    obs_layout,
    preprocess_depth,
    preprocess_proprio,
)
from common.types import PreprocessCfg

# 64x64 기준 cfg — configs/common.yaml과 동일 수치 (테스트 자체는 파일과 독립)
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


def _norm(d: float, cfg: PreprocessCfg = CFG) -> float:
    """계약 수식으로 기대 정규화값 계산: (clip(d)-near)/(far-near)."""
    d = min(max(d, cfg.depth_near_m), cfg.depth_far_m)
    return (d - cfg.depth_near_m) / (cfg.depth_far_m - cfg.depth_near_m)


# ── frame_mode=full 계약 (2026-07-20 감사 — 현행 생산 경로가 테스트 0회였던 구멍) ──

CFG_FULL = PreprocessCfg(
    depth_near_m=0.15,
    depth_far_m=1.20,
    out_height=64,
    out_width=64,
    joint_pos_scale=np.pi,
    joint_vel_scale=np.pi,
    tcp_pos_scale=1.0,
    action_dim=7,
    frame_mode="full",
)


def test_depth_full_mode_keeps_edges_no_crop():
    """★현행 생산 계약(frame_mode=full)★: 848×480 전체 프레임을 비등방 리사이즈 —
    정사각 center-crop이 일어나지 않아야 한다(좌우 가장자리 콘텐츠 보존이 증거).

    구성: 좌측 1/8 폭만 near(어두움), 나머지 far — square 모드였다면 crop 후에도
    좌측 밴드가 남지만 폭 비율이 달라진다. full 모드 기대: 출력 좌측 ~1/8 폭이
    near값, 즉 입력 가로 전체가 출력 가로에 매핑(비등방).
    """
    d = np.full((1, 480, 848), 1.0, dtype=np.float32)
    d[:, :, : 848 // 8] = 0.4  # 좌측 밴드
    out = preprocess_depth(d, CFG_FULL)
    assert out.shape == (1, 1, 64, 64)
    w8 = 64 // 8
    left = out[0, 0, :, : w8 - 1]
    right = out[0, 0, :, w8 + 1 :]
    assert torch.allclose(left, torch.full_like(left, _norm(0.4)), atol=1e-4), \
        "full 모드에서 좌측 밴드가 출력 좌측 1/8에 보존돼야 함(크롭 발생 의심)"
    assert torch.allclose(right, torch.full_like(right, _norm(1.0)), atol=1e-4)


def test_depth_full_vs_square_differ_on_wide_input():
    """같은 848×480 입력에서 full(전체 비등방)과 square(정사각 크롭) 결과가 달라야
    한다 — 두 모드가 같은 코드 경로로 합쳐지는 회귀를 차단."""
    rng = np.random.default_rng(7)
    d = rng.uniform(0.2, 1.1, size=(1, 480, 848)).astype(np.float32)
    out_full = preprocess_depth(d, CFG_FULL)
    out_sq = preprocess_depth(d, CFG)
    assert not torch.allclose(out_full, out_sq, atol=1e-5)


# ---------------------------------------------------------------- depth ----


def test_depth_output_shape_and_range():
    """[N,1,64,64] 형상 + 임의 입력(결측 포함)에서도 값이 [0,1] 범위.

    ⚠️ float32 한계: (far-near)를 분자(float32 뺄셈)와 분모(python float)가
    미세하게 다르게 보므로 상한이 1.0을 ~1e-7 넘을 수 있다(실측 1.0000001).
    frozen 코드의 실제 동작이므로 ε=1e-6 여유로 검정한다.
    """
    rng = np.random.default_rng(0)
    d = rng.uniform(-0.5, 3.0, size=(3, 48, 72)).astype(np.float32)
    d[0, 0, 0] = np.nan
    d[1, 1, 1] = np.inf
    d[2, 2, 2] = 0.0
    out = preprocess_depth(d, CFG)
    assert out.shape == (3, 1, CFG.out_height, CFG.out_width)
    eps = 1e-6
    assert torch.all(out >= -eps) and torch.all(out <= 1.0 + eps)


@pytest.mark.parametrize("bad", [0.0, -0.3, float("nan"), float("inf"), float("-inf")])
def test_depth_invalid_becomes_far(bad):
    """결측(<=0/NaN/inf) 픽셀은 far 대체 → 정규화 후 1.0."""
    # 10x10 → out 10x10 (resize 항등)으로 픽셀 위치를 그대로 추적한다
    cfg = PreprocessCfg(0.15, 1.20, 10, 10, np.pi, np.pi, 1.0, 7)
    d = np.full((10, 10), 0.5, dtype=np.float32)
    d[0, 0] = bad
    out = preprocess_depth(d, cfg)
    assert out[0, 0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert out[0, 0, 5, 5].item() == pytest.approx(_norm(0.5), abs=1e-6)


def test_depth_clip_boundaries():
    """★2026-07-14 최소거리 무효화 계약★: near 미만은 clamp가 아니라 무효→far(→1.0)
    (real D435i 최소거리 미만 = depth 0 무효와 동일 규칙 — eye-in-hand 파지 최종
    구간의 depth 결손을 sim/real 동일하게 모사). far 초과는 far로 clip."""
    cfg = PreprocessCfg(0.15, 1.20, 10, 10, np.pi, np.pi, 1.0, 7)
    d = np.full((10, 10), 0.5, dtype=np.float32)
    d[0, 0] = 0.05  # 0 < d < near → ★무효 → far(1.0)★ (구계약: near clamp → 0.0)
    d[0, 1] = 3.00  # > far → clip to far
    d[0, 2] = 0.15  # 정확히 near
    d[0, 3] = 1.20  # 정확히 far
    out = preprocess_depth(d, cfg)[0, 0]
    assert out[0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert out[0, 1].item() == pytest.approx(1.0, abs=1e-6)
    assert out[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert out[0, 3].item() == pytest.approx(1.0, abs=1e-6)


def test_depth_center_crop_848x480_asymmetric():
    """★크롭 위치 회귀 가드★ 848x480(W x H) → 중앙 480x480.

    좌/우 마진에 서로 다른 마커를 넣고 중앙 정사각만 남는지 확인 —
    left-crop이었다면 좌측 마커(0.3)가 출력에 섞인다.
    """
    h, w = 480, 848
    left = (w - h) // 2  # = 184
    d = np.full((h, w), 0.3, dtype=np.float32)  # 좌측 마진 마커
    d[:, left + h :] = 0.7  # 우측 마진 마커
    d[:, left : left + h] = 0.5  # 중앙 정사각 (crop 대상)
    out = preprocess_depth(d, CFG)
    assert out.shape == (1, 1, 64, 64)
    expected = _norm(0.5)
    assert torch.allclose(out, torch.full_like(out, expected), atol=1e-6), (
        "중앙 크롭이 아님 — 마진 마커가 출력에 섞임"
    )


def test_depth_center_crop_portrait():
    """세로가 긴 입력(H>W)도 짧은 변 기준 중앙 크롭 (상/하 마진 제거)."""
    h, w = 100, 60
    top = (h - w) // 2  # = 20
    d = np.full((h, w), 0.3, dtype=np.float32)
    d[top + w :, :] = 0.7
    d[top : top + w, :] = 0.5
    out = preprocess_depth(d, CFG)
    assert torch.allclose(out, torch.full_like(out, _norm(0.5)), atol=1e-6)


def test_depth_nearest_resize_deterministic_and_no_blending():
    """nearest resize: (1) 같은 입력 → 완전 동일 출력, (2) 값 블리딩 없음
    (출력의 모든 값이 입력 이산값 집합의 정규화값 중 하나)."""
    rng = np.random.default_rng(7)
    levels = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    d = levels[rng.integers(0, len(levels), size=(480, 848))]
    out1 = preprocess_depth(d, CFG)
    out2 = preprocess_depth(d.copy(), CFG)
    assert torch.equal(out1, out2), "nearest resize가 비결정적"
    expected_vals = torch.tensor([_norm(v) for v in levels])
    uniq = torch.unique(out1)
    for v in uniq:
        assert torch.min(torch.abs(expected_vals - v)).item() < 1e-6, (
            f"보간 블리딩 값 발견: {v.item()}"
        )


def test_depth_batch1_vs_batchN_identical():
    """[H,W] 단건과 [N,H,W] 배치가 완전히 같은 코드 경로(동일 출력)."""
    rng = np.random.default_rng(3)
    d = rng.uniform(0.0, 2.0, size=(48, 72)).astype(np.float32)
    d[0, 0] = np.nan
    single = preprocess_depth(d, CFG)  # [1,1,64,64]
    batch = preprocess_depth(np.stack([d, d, d]), CFG)  # [3,1,64,64]
    assert single.shape == (1, 1, 64, 64) and batch.shape == (3, 1, 64, 64)
    for i in range(3):
        assert torch.equal(batch[i], single[0])


def test_depth_numpy_vs_torch_input_identical():
    """numpy 입력과 torch 입력이 동일 결과 (입력 유연성 계약)."""
    rng = np.random.default_rng(11)
    d = rng.uniform(0.1, 1.5, size=(60, 90)).astype(np.float32)
    assert torch.equal(preprocess_depth(d, CFG), preprocess_depth(torch.from_numpy(d), CFG))


def test_depth_bad_ndim_raises():
    """depth 차원이 2/3이 아니면 ValueError."""
    with pytest.raises(ValueError):
        preprocess_depth(np.zeros((2, 3, 4, 5), dtype=np.float32), CFG)


# -------------------------------------------------------------- proprio ----


def test_proprio_scaling_exact():
    """스케일 수식 수치 검증: q/qps, qd/qvs, tcp_pos/tps, quat 그대로,
    target/tps, last_action 그대로 — 구분되는 스케일로 축별 확인."""
    cfg = PreprocessCfg(0.15, 1.20, 64, 64, 2.0, 4.0, 0.5, 7)
    rng = np.random.default_rng(5)
    q = rng.normal(size=6)
    qd = rng.normal(size=6)
    quat = rng.normal(size=4)
    quat /= np.linalg.norm(quat)
    tcp = np.concatenate([rng.normal(size=3), quat])
    tgt = rng.normal(size=3)
    act = rng.uniform(-1, 1, size=7)
    grip = 0.5  # delayed binary command ramp is already normalized to [0,1]
    out = preprocess_proprio(
        q, qd, tcp, tgt, act, cfg, gripper_command_state=grip
    )
    assert out.shape == (1, 6 + 6 + 1 + 7 + 3 + 7)
    expected = np.concatenate(
        [
            q / 2.0,
            qd / 4.0,
            [0.5],
            tcp[:3] / 0.5,
            canonicalize_quat_wxyz(tcp[3:]).numpy(),
            tgt / 0.5,
            act,
        ]
    )
    np.testing.assert_allclose(out[0].numpy(), expected, atol=1e-6)


@pytest.mark.parametrize(
    "quat",
    [
        np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32),
        np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),
    ],
)
def test_proprio_quaternion_sign_is_canonical(quat):
    """Equivalent q/-q poses must produce identical sim/real policy input."""
    q = np.zeros(6, dtype=np.float32)
    tcp_pos = np.array([0.4, -0.1, 0.3], dtype=np.float32)
    target = np.zeros(3, dtype=np.float32)
    action = np.zeros(CFG.action_dim, dtype=np.float32)

    positive = preprocess_proprio(
        q, q, np.concatenate([tcp_pos, quat]), target, action, CFG,
        gripper_command_state=0.0,
    )
    negative = preprocess_proprio(
        q, q, np.concatenate([tcp_pos, -quat]), target, action, CFG,
        gripper_command_state=0.0,
    )

    assert torch.equal(positive, negative)
    assert positive[0, 16] >= 0  # tcp quaternion w within the proprio layout


@pytest.mark.parametrize(
    "field,size",
    [("joint_pos", 5), ("joint_vel", 7), ("tcp_pose", 6), ("target_pos", 2), ("last_action", 6)],
)
def test_proprio_wrong_dim_raises(field, size):
    """proprio 각 입력의 크기 검증 (틀리면 ValueError)."""
    kw = {
        "joint_pos": np.zeros(6),
        "joint_vel": np.zeros(6),
        "tcp_pose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float64),
        "target_pos": np.zeros(3),
        "last_action": np.zeros(7),
    }
    kw[field] = np.zeros(size)
    with pytest.raises(ValueError):
        preprocess_proprio(
            kw["joint_pos"], kw["joint_vel"], kw["tcp_pose"], kw["target_pos"], kw["last_action"], CFG,
            gripper_command_state=0.0,
        )


# ------------------------------------------------------ build_policy_obs ----


def _mk_inputs(n_env: int | None = None):
    """마커 값이 든 build_policy_obs 입력 셋 (n_env=None이면 batch 없는 형상)."""
    depth_val = 0.15 + 0.25 * (1.20 - 0.15)  # 정규화 후 정확히 0.25
    depth = np.full((64, 64), depth_val, dtype=np.float32)
    q = np.full(6, 0.5 * np.pi)  # /pi → 0.5
    qd = np.full(6, -0.5 * np.pi)  # /pi → -0.5
    tcp = np.array([0.7, 0.7, 0.7, 1.0, 0.0, 0.0, 0.0])  # tps=1.0 → 그대로
    tgt = np.full(3, 0.9)
    act = np.full(7, 0.3)
    if n_env is not None:
        depth = np.stack([depth] * n_env)
        q, qd, tcp, tgt, act = (np.stack([a] * n_env) for a in (q, qd, tcp, tgt, act))
    return depth, q, qd, tcp, tgt, act


def test_build_policy_obs_dim_4126():
    """64x64 cfg 기준 obs_dim = 4096 + 6+6+1+7+3+7 = 4126 (common.yaml 계약)."""
    assert obs_dim(CFG) == 4126
    depth, q, qd, tcp, tgt, act = _mk_inputs()
    flat = build_policy_obs(
        depth, q, qd, tcp, tgt, act, CFG, gripper_command_state=0.0
    )
    assert flat.shape == (1, 4126)


def test_build_policy_obs_layout_order_regression():
    """★레이아웃 순서 회귀 가드★ 각 슬롯에 고유 마커를 넣고 flat 위치로 확인.

    [ depth(4096) | joint_pos(6) | joint_vel(6) | gripper(1) | tcp_pose(7) | target(3) | last_action(7) ]
    """
    depth, q, qd, tcp, tgt, act = _mk_inputs()
    flat = build_policy_obs(
        depth, q, qd, tcp, tgt, act, CFG, gripper_command_state=0.0
    )[0].numpy()
    i = 0
    np.testing.assert_allclose(flat[i : i + 4096], 0.25, atol=1e-6)  # depth
    i += 4096
    np.testing.assert_allclose(flat[i : i + 6], 0.5, atol=1e-6)  # joint_pos
    i += 6
    np.testing.assert_allclose(flat[i : i + 6], -0.5, atol=1e-6)  # joint_vel
    i += 6
    np.testing.assert_allclose(flat[i : i + 1], 0.0, atol=1e-6)  # command state open
    i += 1
    np.testing.assert_allclose(
        flat[i : i + 7], [0.7, 0.7, 0.7, 1.0, 0.0, 0.0, 0.0], atol=1e-6
    )  # tcp_pose (pos/1.0 + quat 그대로)
    i += 7
    np.testing.assert_allclose(flat[i : i + 3], 0.9, atol=1e-6)  # target_pos
    i += 3
    np.testing.assert_allclose(flat[i : i + 7], 0.3, atol=1e-6)  # last_action
    assert i + 7 == 4126


def test_build_policy_obs_batchN():
    """sim 경로(N=num_envs)도 같은 함수·같은 결과 — N=4 배치 확인."""
    depth, q, qd, tcp, tgt, act = _mk_inputs(n_env=4)
    flat = build_policy_obs(
        depth, q, qd, tcp, tgt, act, CFG, gripper_command_state=np.zeros(4)
    )
    assert flat.shape == (4, 4126)
    for i in range(1, 4):
        assert torch.equal(flat[i], flat[0])


def test_build_policy_obs_batch_mismatch_raises():
    """depth batch(2) != proprio batch(1) → ValueError."""
    depth, q, qd, tcp, tgt, act = _mk_inputs()
    depth2 = np.stack([depth, depth])  # [2,H,W], proprio는 N=1
    with pytest.raises(ValueError):
        build_policy_obs(
            depth2, q, qd, tcp, tgt, act, CFG, gripper_command_state=0.0
        )


def test_obs_layout_and_dim_consistency():
    """obs_layout 이름/순서/크기와 obs_dim 합의 일관성 (계약서 본문 고정)."""
    layout = obs_layout(CFG)
    assert [name for name, _ in layout] == [
        "depth", "joint_pos", "joint_vel", "gripper_command_state", "tcp_pose", "target_pos", "last_action",
    ]
    assert dict(layout) == {
        "depth": 64 * 64,
        "joint_pos": NUM_ARM_JOINTS,
        "joint_vel": NUM_ARM_JOINTS,
        "gripper_command_state": 1,
        "tcp_pose": TCP_POSE_DIM,
        "target_pos": TARGET_DIM,
        "last_action": CFG.action_dim,
    }
    assert obs_dim(CFG) == sum(s for _, s in layout)


def test_configs_common_yaml_obs_dim_contract():
    """실제 configs/common.yaml → build_preprocess_cfg → obs_dim 계약 검증 (통합 스모크).

    ★2026-07-17 RGB-D 전환(B안)★: use_rgb=true가 현행 계약 — obs_dim은
    depth(4096)+rgb(12288)+proprio(30)=16414. use_rgb 스위치와 무관하게 성립하는
    계약 산식(4126 + use_rgb×h*w*3)으로 검증하고, 현행 yaml 상태(True/16414)도
    회귀 가드로 잠근다(yaml이 바뀌면 여기가 먼저 울린다 = 계약 변경 알람).
    """
    cfg = build_preprocess_cfg(load_yaml(configs_dir() / "common.yaml"))
    # ★2026-07-19 h1 계약: 112² 상향(② 카드 — 64² 정체 병목 후보 해소)★
    assert cfg.out_height == 112 and cfg.out_width == 112 and cfg.action_dim == 7
    base = cfg.out_height * cfg.out_width + 30  # depth + proprio(6+6+1+7+3+7 — gripper 포함)
    expected = base + (cfg.out_height * cfg.out_width * 3 if cfg.use_rgb else 0)
    assert obs_dim(cfg) == expected
    # 현행 계약 고정: RGB-D(B안) 112² — 바꾸면 재학습+재export(계약 변경)
    assert cfg.use_rgb is True
    assert obs_dim(cfg) == 50206


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
