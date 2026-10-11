"""요소2(D435i depth 노이즈 모델) 단위테스트 — 시뮬레이터 불필요(순수 torch).

계약: σ(z) = sigma_base_m + sigma_scale_per_m2 * z²(거리 제곱 비례),
      dropout은 valid 픽셀만 0.0(hole)으로, 결측(<=0)은 그대로 보존.
통계 검정은 큰 표본 + 고정 seed로 결정적으로 수행한다.

실행: python -m pytest tests/test_depth_noise.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import pytest
import torch

from common.depth_noise import apply_depth_noise

# configs/dr.yaml depth_noise 섹션과 같은 수치
SIGMA_BASE = 0.002
SIGMA_SCALE = 0.004


def _gen(seed: int = 42) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_reproducible_with_fixed_generator():
    """같은 seed generator → 완전 동일 출력 (재현성 계약)."""
    depth = torch.rand(64, 64) + 0.3
    out1 = apply_depth_noise(depth, SIGMA_BASE, SIGMA_SCALE, 0.05, generator=_gen(7))
    out2 = apply_depth_noise(depth, SIGMA_BASE, SIGMA_SCALE, 0.05, generator=_gen(7))
    assert torch.equal(out1, out2)
    out3 = apply_depth_noise(depth, SIGMA_BASE, SIGMA_SCALE, 0.05, generator=_gen(8))
    assert not torch.equal(out1, out3), "seed가 달라도 같은 출력 — generator 미사용 의심"


def test_sigma_grows_with_distance():
    """σ(z) 거리 의존 통계: 가까운(0.3m) vs 먼(1.0m) 픽셀의 잔차 표준편차 비교.

    기대: σ(0.3)=0.002+0.004*0.09=0.00236, σ(1.0)=0.006 (상대오차 5% 이내, n=4e5).
    """
    n = 400_000
    z_near, z_far = 0.3, 1.0
    d_near = torch.full((n,), z_near)
    d_far = torch.full((n,), z_far)
    noisy_near = apply_depth_noise(d_near, SIGMA_BASE, SIGMA_SCALE, 0.0, generator=_gen(1))
    noisy_far = apply_depth_noise(d_far, SIGMA_BASE, SIGMA_SCALE, 0.0, generator=_gen(2))
    std_near = float((noisy_near - d_near).std())
    std_far = float((noisy_far - d_far).std())
    exp_near = SIGMA_BASE + SIGMA_SCALE * z_near**2
    exp_far = SIGMA_BASE + SIGMA_SCALE * z_far**2
    assert std_near == pytest.approx(exp_near, rel=0.05)
    assert std_far == pytest.approx(exp_far, rel=0.05)
    assert std_far > std_near, "거리 제곱 비례 σ 위반 (먼 픽셀 분산이 더 커야 함)"


def test_gaussian_mean_unbiased():
    """가우시안 노이즈는 무편향 — 잔차 평균 ≈ 0 (σ/√n 스케일 허용)."""
    n = 400_000
    d = torch.full((n,), 0.8)
    noisy = apply_depth_noise(d, SIGMA_BASE, SIGMA_SCALE, 0.0, generator=_gen(3))
    sigma = SIGMA_BASE + SIGMA_SCALE * 0.8**2
    assert abs(float((noisy - d).mean())) < 5 * sigma / n**0.5


def test_dropout_fraction_approx():
    """dropout 비율 근사: p=0.05, n=1e6 → 0 픽셀 비율 0.05±0.005."""
    p = 0.05
    depth = torch.full((1000, 1000), 1.0)
    noisy = apply_depth_noise(depth, SIGMA_BASE, SIGMA_SCALE, p, generator=_gen(4))
    frac = float((noisy == 0.0).float().mean())
    assert frac == pytest.approx(p, abs=0.005)


def test_dropout_zero_means_no_holes():
    """dropout_prob=0.0이면 hole 생성 없음 (σ가 작아 0 도달 불가)."""
    depth = torch.full((512, 512), 1.0)
    noisy = apply_depth_noise(depth, SIGMA_BASE, SIGMA_SCALE, 0.0, generator=_gen(5))
    assert int((noisy == 0.0).sum()) == 0


def test_invalid_pixels_preserved():
    """결측(<=0) 픽셀은 노이즈/dropout 모두 미적용 — 값 그대로 보존."""
    depth = torch.full((100, 100), 0.7)
    depth[0, :50] = 0.0  # 결측 hole
    depth[1, :50] = -0.25  # 음수 결측(방어적)
    noisy = apply_depth_noise(depth, SIGMA_BASE, SIGMA_SCALE, 1.0, generator=_gen(6))
    # dropout_prob=1.0(경계 허용값): valid는 전부 0, invalid는 원값 유지
    assert torch.equal(noisy[0, :50], depth[0, :50])
    assert torch.equal(noisy[1, :50], depth[1, :50])
    assert torch.all(noisy[2:] == 0.0), "dropout_prob=1.0이면 valid 픽셀은 전부 hole"


def test_valid_pixels_actually_noised():
    """valid 픽셀에는 노이즈가 실제로 들어간다 (원본과 완전 동일하면 실패)."""
    depth = torch.full((64, 64), 0.7)
    noisy = apply_depth_noise(depth, SIGMA_BASE, SIGMA_SCALE, 0.0, generator=_gen(9))
    assert noisy.shape == depth.shape
    assert not torch.equal(noisy, depth)


@pytest.mark.parametrize("bad_p", [-0.01, 1.01, -1.0, 2.0])
def test_dropout_prob_out_of_range_raises(bad_p):
    """dropout_prob ∉ [0,1] → ValueError."""
    with pytest.raises(ValueError):
        apply_depth_noise(torch.ones(4, 4), SIGMA_BASE, SIGMA_SCALE, bad_p)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
