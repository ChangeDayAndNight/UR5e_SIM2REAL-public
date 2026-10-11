# 요소2(일부): D435i depth 노이즈 모델 — ★sim 학습에서만 적용★ (real은 센서가
# 진짜 노이즈를 만든다). 공유 전처리(obs_preprocess) '이전'의 raw depth에 적용한다.
# 순수 torch로 유지 → tests/test_depth_noise.py가 sim 없이 검증한다.
# 범위 소스: configs/dr.yaml depth_noise 섹션.
from __future__ import annotations

import torch

from common.noise_parameters import image_parameter


def apply_depth_noise(
    depth_m: torch.Tensor,
    sigma_base_m: float,
    sigma_scale_per_m2: float,
    dropout_prob: float | torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """raw depth[m]에 D435i 근사 노이즈 주입 → 같은 shape 반환.

    - 가우시안: σ(z) = sigma_base_m + sigma_scale_per_m2 * z²  (거리 제곱 비례)
    - dropout: 확률 dropout_prob로 0.0(결측 hole) — 전처리가 far로 채우므로
      "실기에서 구멍 난 depth"와 같은 최종 분포가 된다.
    - 결측(<=0) 픽셀은 노이즈를 덧씌우지 않고 결측 그대로 둔다.

    dropout_prob는 스칼라 또는 배치별 [N]이다. reset된 환경의 값만 재샘플한다.
    TODO(확장): 엣지 노이즈, fill-rate 저하, 지수형 quantization.
    """
    dropout_prob = image_parameter(dropout_prob, depth_m, "dropout_prob")
    if not bool(((dropout_prob >= 0.0) & (dropout_prob <= 1.0)).all()):
        raise ValueError(f"dropout_prob 범위 오류: {dropout_prob}")
    valid = depth_m > 0.0
    sigma = sigma_base_m + sigma_scale_per_m2 * depth_m.square()
    noise = torch.randn(
        depth_m.shape, generator=generator, device=depth_m.device, dtype=depth_m.dtype
    )
    noisy = torch.where(valid, depth_m + noise * sigma, depth_m)
    if bool((dropout_prob > 0.0).any()):
        drop = (
            torch.rand(
                depth_m.shape,
                generator=generator,
                device=depth_m.device,
                dtype=depth_m.dtype,
            )
            < dropout_prob
        )
        noisy = torch.where(drop & valid, torch.zeros_like(noisy), noisy)
    return noisy
