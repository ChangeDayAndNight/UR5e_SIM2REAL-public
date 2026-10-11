"""요소2(RGB 노이즈/지터 모델) 단위테스트 — 시뮬레이터 불필요(순수 torch).

계약: x*brightness → (x-mean)*contrast+mean(프레임별 mean) → +N(0,σ) → clamp[0,1].
결정성은 고정 seed generator로 검정한다 (test_depth_noise.py 스타일 미러).
범위 소스: configs/dr.yaml rgb_noise 섹션.

실행: python -m pytest tests/test_rgb_noise.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import pytest
import torch

from common.rgb_noise import apply_rgb_noise

# configs/dr.yaml rgb_noise 섹션 범위의 대표값
BRIGHT_HI = 1.3
CONTRAST_HI = 1.2
SIGMA_HI = 0.03


def _gen(seed: int = 42) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _rand_rgb(shape=(64, 64, 3), seed: int = 0) -> torch.Tensor:
    g = _gen(seed)
    return torch.rand(shape, generator=g)


# ------------------------------------------------------------- 항등/범위 ----


def test_identity_params_passthrough():
    """brightness=1, contrast=1, σ=0 → 입력 그대로 (항등 계약)."""
    x = _rand_rgb()
    out = apply_rgb_noise(x, 1.0, 1.0, 0.0)
    # (x-μ)·1+μ는 부동소수 재결합이라 완전 동일이 아닌 ~1e-7 오차까지 허용
    assert torch.allclose(out, x, atol=1e-6)


@pytest.mark.parametrize(
    "b,c,s",
    [(BRIGHT_HI, CONTRAST_HI, SIGMA_HI), (0.7, 0.8, 0.03), (5.0, 3.0, 0.5), (0.1, 0.1, 0.0)],
)
def test_output_always_clamped_unit_range(b, c, s):
    """어떤 파라미터 조합에도 출력은 [0,1] (clamp 계약) + shape 보존."""
    x = _rand_rgb(seed=3)
    out = apply_rgb_noise(x, b, c, s, generator=_gen(1))
    assert out.shape == x.shape
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


# ------------------------------------------------------------- 결정성 ----


def test_generator_determinism():
    """같은 seed generator → 완전 동일 / 다른 seed → 다른 출력 (재현성 계약)."""
    x = _rand_rgb(seed=5)
    out1 = apply_rgb_noise(x, BRIGHT_HI, CONTRAST_HI, SIGMA_HI, generator=_gen(7))
    out2 = apply_rgb_noise(x, BRIGHT_HI, CONTRAST_HI, SIGMA_HI, generator=_gen(7))
    assert torch.equal(out1, out2)
    out3 = apply_rgb_noise(x, BRIGHT_HI, CONTRAST_HI, SIGMA_HI, generator=_gen(8))
    assert not torch.equal(out1, out3), "seed가 달라도 같은 출력 — generator 미사용 의심"


def test_sigma_zero_fully_deterministic():
    """σ=0이면 generator 없이도 결정적 (난수 경로 자체를 타지 않음)."""
    x = _rand_rgb(seed=9)
    out1 = apply_rgb_noise(x, 1.1, 0.9, 0.0)
    out2 = apply_rgb_noise(x.clone(), 1.1, 0.9, 0.0)
    assert torch.equal(out1, out2)


# ------------------------------------------------------------- 수식 검증 ----


def test_brightness_formula_exact():
    """x*brightness (clamp 전 구간에서 정확 수식): 0.4×1.25 → 0.5."""
    x = torch.full((8, 8, 3), 0.4)
    out = apply_rgb_noise(x, 1.25, 1.0, 0.0)
    assert torch.allclose(out, torch.full_like(x, 0.5), atol=1e-7)


def test_contrast_formula_exact_global_mean():
    """비배치([H,W,3])는 전체 mean 기준: (x-μ)·c+μ 수식 그대로."""
    x = torch.zeros(2, 2, 3)
    x[0], x[1] = 0.2, 0.8  # μ = 0.5
    out = apply_rgb_noise(x, 1.0, 1.5, 0.0)
    # 0.2 → (0.2-0.5)*1.5+0.5 = 0.05 / 0.8 → 0.95
    assert torch.allclose(out[0], torch.full((2, 3), 0.05), atol=1e-6)
    assert torch.allclose(out[1], torch.full((2, 3), 0.95), atol=1e-6)


def test_contrast_per_frame_mean_batch():
    """[N,H,W,3] 배치는 ★프레임별★ mean — 다른 프레임의 밝기가 섞이면 안 된다."""
    x = torch.zeros(2, 4, 4, 3)
    x[0] = 0.2  # 프레임0: 균일 → 자기 mean=0.2 → contrast 불변
    x[1] = 0.8  # 프레임1: 균일 → 자기 mean=0.8 → contrast 불변
    out = apply_rgb_noise(x, 1.0, 2.0, 0.0)
    # 전역 mean(0.5)을 썼다면 0.2→-0.1(clamp 0), 0.8→1.1(clamp 1)로 왜곡된다
    assert torch.allclose(out[0], torch.full_like(out[0], 0.2), atol=1e-6)
    assert torch.allclose(out[1], torch.full_like(out[1], 0.8), atol=1e-6)


def test_pixel_noise_statistics():
    """σ=0.02 가우시안: 잔차 std ≈ σ(상대 5%), 평균 ≈ 0 (중앙값 0.5 → clamp 영향 미미)."""
    n = 500_000
    x = torch.full((n, 3), 0.5)
    out = apply_rgb_noise(x, 1.0, 1.0, 0.02, generator=_gen(11))
    resid = out - x
    assert float(resid.std()) == pytest.approx(0.02, rel=0.05)
    assert abs(float(resid.mean())) < 5 * 0.02 / (3 * n) ** 0.5


# ------------------------------------------------------------- 검증 에러 ----


@pytest.mark.parametrize("bad_b", [0.0, -1.0])
def test_brightness_nonpositive_raises(bad_b):
    """brightness_scale <= 0 → ValueError."""
    with pytest.raises(ValueError):
        apply_rgb_noise(torch.rand(4, 4, 3), bad_b, 1.0, 0.0)


@pytest.mark.parametrize("bad_c", [0.0, -0.5])
def test_contrast_nonpositive_raises(bad_c):
    """contrast_scale <= 0 → ValueError."""
    with pytest.raises(ValueError):
        apply_rgb_noise(torch.rand(4, 4, 3), 1.0, bad_c, 0.0)


@pytest.mark.parametrize("bad_s", [-0.001, -1.0])
def test_pixel_sigma_negative_raises(bad_s):
    """pixel_sigma < 0 → ValueError."""
    with pytest.raises(ValueError):
        apply_rgb_noise(torch.rand(4, 4, 3), 1.0, 1.0, bad_s)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
