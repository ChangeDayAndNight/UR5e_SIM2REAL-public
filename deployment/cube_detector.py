"""큐브 인식 — 색분할 + depth 역투영으로 3D 위치 추정. [배포 PC][제어 루프 안]

═══════════════════════════════════════════════════════════════════════════════
 왜 이 파일이 필요한가
═══════════════════════════════════════════════════════════════════════════════
학습된 student 정책은 ★큐브를 인식하지 않는다★. 112x112 RGB-D와 관절상태를 받아 곧바로
7차원 액션을 내는 end-to-end 사상이고, "큐브를 찾았다"는 신호가 시스템 어디에도 없다.
그래서 큐브가 아닌 물체를 놓아도, 심지어 빈 장면에서도 신경망은 ★반드시 무언가를 출력★해
로봇이 움직인다(신경망은 어떤 입력에도 출력을 낸다).

이 모듈은 그 구멍을 배포측에서 막는다. 정책을 바꾸지 않고, 정책 ★앞에★ 게이트를 둔다:
큐브가 안 보이면 액션을 통과시키지 않는다.

═══════════════════════════════════════════════════════════════════════════════
 왜 SAM이 아니라 색분할인가 (설계 결정, 2026-08-31)
═══════════════════════════════════════════════════════════════════════════════
· 제어 주기는 50ms(20Hz)다. SAM ViT-B는 CPU에서 프레임당 2~5초로 ★40~100배 느리다★.
  GPU는 학습에 묶여 있고, 비어 있어도 ViT-B가 50~100ms라 여유가 없다.
· 대상이 ★단색 큐브★다(sim.yaml object_color_palette: 빨강 0.80/0.15/0.15,
  파랑 0.15/0.35/0.85). 색분할이 잘 맞는 조건이고 OpenCV로 1~2ms에 끝난다.
· SAM은 ★오프라인 검증용★으로 쓰는 것이 맞다 — 캡처한 프레임에서 SAM 마스크를 정답으로
  두고 이 빠른 경로의 오차를 재는 용도(validate_against_masks 참조).

★정직한 한계★
· 조명이 크게 바뀌면 HSV 범위를 재조정해야 한다(캘리브레이션 항목이다).
· 같은 색의 다른 물체는 구분하지 못한다 — 크기·정사각형성으로 걸러내지만 완벽하지 않다.
· 이것은 ★안전 게이트★이지 정밀 위치 추정기가 아니다. 정책은 여전히 자기 시각으로 움직인다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── 학습된 큐브 색 (configs/sim.yaml object_color_palette와 대응) ──────────────
#   OpenCV HSV: H∈[0,179], S/V∈[0,255]. 빨강은 H=0 경계를 넘으므로 두 구간으로 나눈다.
HSV_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "red_low":  ((0, 90, 60), (10, 255, 255)),
    "red_high": ((170, 90, 60), (179, 255, 255)),
    "blue":     ((100, 90, 60), (130, 255, 255)),
}

# ★면적 임계 근거★ — D435i color 848x480(fx≈604)에서 5cm 큐브의 화면 크기 실계산:
#   0.30m → 10,147 px² / 0.45m → 4,510 / 0.60m → 2,537 / 0.80m → 1,427
#   작업거리 상한(0.8m)에서도 잡히도록 그 절반인 700을 하한으로 둔다.
MIN_AREA_PX = 700
# 상한: 화면의 절반을 넘으면 큐브가 아니라 배경/조명이다.
MAX_AREA_FRACTION = 0.5
# 정사각형성 — 큐브 상면은 정사각형에 가깝다. 길쭉한 덩어리는 배제한다.
MAX_ASPECT_RATIO = 2.0
# 채움률 — 외곽 사각형 대비 마스크 면적. 큐브는 높고 링/테두리 형태는 낮다.
MIN_FILL_RATIO = 0.55


@dataclass(frozen=True)
class CubeDetection:
    """한 프레임의 큐브 판정 결과."""

    present: bool
    reason: str                       # 없을 때 왜 없는지 — 현장 디버깅용
    center_px: tuple[int, int] | None = None
    area_px: int = 0
    color: str | None = None          # "red" | "blue"
    position_cam: tuple[float, float, float] | None = None   # 카메라 광학 프레임 [m]
    position_base: tuple[float, float, float] | None = None  # 로봇 base 프레임 [m]
    depth_m: float | None = None


def _largest_colored_blob(rgb: np.ndarray) -> tuple[np.ndarray | None, str, str]:
    """빨강/파랑 마스크에서 가장 큰 덩어리를 고른다. (마스크, 색이름, 사유)"""
    import cv2

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    masks = {
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, np.array(HSV_RANGES["red_low"][0]), np.array(HSV_RANGES["red_low"][1])),
            cv2.inRange(hsv, np.array(HSV_RANGES["red_high"][0]), np.array(HSV_RANGES["red_high"][1])),
        ),
        "blue": cv2.inRange(
            hsv, np.array(HSV_RANGES["blue"][0]), np.array(HSV_RANGES["blue"][1])
        ),
    }
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    best = (0, None, "")
    for name, mask in masks.items():
        # 열기(잡음 제거) → 닫기(구멍 메움). 순서를 바꾸면 잡음이 덩어리로 자란다.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area > best[0]:
                best = (area, (labels == index).astype(np.uint8), name)
    if best[1] is None:
        return None, "", "색 마스크가 비었다(빨강/파랑 없음)"
    return best[1], best[2], ""


def detect_cube(
    rgb: np.ndarray,
    depth_m: np.ndarray | None = None,
    *,
    intrinsics: tuple[float, float, float, float] | None = None,  # fx, fy, cx, cy
    T_base_cam: np.ndarray | None = None,
    min_area_px: int = MIN_AREA_PX,
) -> CubeDetection:
    """RGB(+depth)에서 큐브를 찾고 가능하면 3D 위치까지 추정한다.

    depth/intrinsics가 없으면 존재 판정(게이트)만 하고 위치는 None으로 둔다 —
    게이트만 필요한 경우 카메라 캘리브레이션 없이도 쓸 수 있게 한 것이다.
    """
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return CubeDetection(False, f"RGB 형상이 아니다: {rgb.shape}")
    height, width = rgb.shape[:2]

    mask, color, why = _largest_colored_blob(rgb)
    if mask is None:
        return CubeDetection(False, why)

    area = int(mask.sum())
    if area < min_area_px:
        return CubeDetection(False, f"덩어리가 너무 작다 {area}px² < {min_area_px}")
    if area > MAX_AREA_FRACTION * height * width:
        return CubeDetection(False, f"덩어리가 화면의 절반을 넘는다 {area}px² — 큐브가 아니다")

    ys, xs = np.nonzero(mask)
    w_box, h_box = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    aspect = max(w_box, h_box) / max(1, min(w_box, h_box))
    if aspect > MAX_ASPECT_RATIO:
        return CubeDetection(False, f"길쭉하다 종횡비 {aspect:.1f} > {MAX_ASPECT_RATIO}")
    fill = area / float(w_box * h_box)
    if fill < MIN_FILL_RATIO:
        return CubeDetection(False, f"채움률이 낮다 {fill:.2f} < {MIN_FILL_RATIO} — 속 빈 형상")

    cx_px, cy_px = int(round(xs.mean())), int(round(ys.mean()))

    if depth_m is None or intrinsics is None:
        return CubeDetection(True, "", (cx_px, cy_px), area, color)

    # ★중앙값 depth★ — 마스크 안 유효 화소만 본다. 평균은 결측(0)과 경계 튐에 약하다.
    patch = depth_m[mask.astype(bool)]
    valid = patch[np.isfinite(patch) & (patch > 1e-3)]
    if valid.size < 10:
        return CubeDetection(False, "마스크 안에 유효 depth가 거의 없다", (cx_px, cy_px), area, color)
    z = float(np.median(valid))

    fx, fy, ppx, ppy = intrinsics
    x_cam = (cx_px - ppx) * z / fx
    y_cam = (cy_px - ppy) * z / fy
    p_cam = (float(x_cam), float(y_cam), z)

    p_base = None
    if T_base_cam is not None:
        homo = np.asarray(T_base_cam, dtype=np.float64) @ np.array([x_cam, y_cam, z, 1.0])
        p_base = (float(homo[0]), float(homo[1]), float(homo[2]))

    return CubeDetection(True, "", (cx_px, cy_px), area, color, p_cam, p_base, z)


def validate_against_masks(
    rgb_frames: list[np.ndarray], reference_masks: list[np.ndarray]
) -> dict:
    """오프라인 검증 — SAM 등으로 만든 정답 마스크와 이 검출기를 대조한다.

    GPU가 학습에서 풀리면 SAM으로 `reference_masks`를 만들어 이 함수에 넣는다.
    색분할이 놓치는 조건(조명 변화·그림자·같은 색 배경)을 정량으로 드러내기 위한 것이지,
    제어 루프에서 SAM을 쓰기 위한 것이 아니다(50ms 주기에 2~5초는 들어갈 수 없다).
    """
    if len(rgb_frames) != len(reference_masks):
        raise ValueError("프레임과 정답 마스크 개수가 다르다")
    ious, misses = [], 0
    for rgb, truth in zip(rgb_frames, reference_masks):
        found = detect_cube(rgb)
        truth_bool = np.asarray(truth, dtype=bool)
        if not found.present:
            misses += 1
            ious.append(0.0)
            continue
        mask, _, _ = _largest_colored_blob(rgb)
        got = np.asarray(mask, dtype=bool)
        union = np.logical_or(got, truth_bool).sum()
        ious.append(float(np.logical_and(got, truth_bool).sum() / union) if union else 0.0)
    return {
        "frames": len(rgb_frames),
        "missed": misses,
        "detection_rate": 1.0 - misses / max(1, len(rgb_frames)),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
    }
