# 요소2(일부): RGB 노이즈/지터 모델 — ★sim 학습에서만 적용★ (real은 센서·조명이
# 진짜 변동을 만든다). 공유 전처리(obs_preprocess.preprocess_rgb) '이전'의 raw
# rgb(float[0,1])에 적용한다. 순수 torch → tests가 sim 없이 검증한다.
# 범위 소스: configs/dr.yaml rgb_noise 섹션.
# [근거] DextrAH-RGB(arXiv 2412.01791)의 color jitter DR 축약판(밝기/대비/픽셀 노이즈)
# — hue 변환은 배경·객체 색 DR(randomize_visual_color)이 담당(중복 회피).
from __future__ import annotations

import torch

from common.noise_parameters import image_parameter


def apply_rgb_noise(
    rgb: torch.Tensor,
    brightness_scale: float | torch.Tensor,
    contrast_scale: float | torch.Tensor,
    pixel_sigma: float | torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """raw rgb(float[0,1], [...,3] 또는 [N,H,W,3])에 지터 주입 → 같은 shape, [0,1] 유지.

    - 밝기: x * brightness_scale (전역 곱셈 — 조명 세기 변동 모사)
    - 대비: (x - mean) * contrast_scale + mean (mean = 프레임별 전체 평균)
    - 픽셀 노이즈: 가우시안 σ=pixel_sigma (센서 샷노이즈 근사)
    파라미터는 스칼라 또는 [N,H,W,3] 배치에 대한 [N] 값이다.
    reset된 환경의 값만 EventTerm에서 재샘플한다.
    """
    for value in (brightness_scale, contrast_scale, pixel_sigma):
        if torch.is_tensor(value) and value.ndim > 0 and rgb.ndim != 4:
            raise ValueError("per-frame RGB parameters require [N,H,W,3] input")
    brightness_scale = image_parameter(brightness_scale, rgb, "brightness_scale")
    contrast_scale = image_parameter(contrast_scale, rgb, "contrast_scale")
    pixel_sigma = image_parameter(pixel_sigma, rgb, "pixel_sigma")
    if bool((brightness_scale <= 0.0).any()) or bool((contrast_scale <= 0.0).any()):
        raise ValueError(
            f"scale은 양수여야 함: brightness={brightness_scale}, contrast={contrast_scale}"
        )
    if bool((pixel_sigma < 0.0).any()):
        raise ValueError(f"pixel_sigma 음수 불가: {pixel_sigma}")
    x = rgb * brightness_scale
    # 프레임(배치 원소)별 평균 대비 — [N,H,W,3]이면 (H,W,3) 축, 그 외엔 전체 평균
    if x.dim() == 4:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
    else:
        mean = x.mean()
    x = (x - mean) * contrast_scale + mean
    if bool((pixel_sigma > 0.0).any()):
        noise = torch.randn(
            x.shape, generator=generator, device=x.device, dtype=x.dtype
        )
        x = x + noise * pixel_sigma
    return x.clamp(0.0, 1.0)
