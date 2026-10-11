# ★모듈형 배포 v1 — depth 기반 큐브 위치 추정★ [배포 PC][제어 루프 안, 순수 numpy(+cv2 선택)]
#
# 왜 색분할(deployment.cube_detector)만으로는 부족한가
#   색분할은 조명·배경색에 민감하다. 루프가 봉인한 sim 표본(112², 색·조명 DR 적용)에서 색분할
#   검출률은 11~34%였고, 잡힌 것도 DR로 붉어진 배경을 큐브로 오인한 경우가 섞여 중심 오차 중앙값이
#   6.5 cm였다. 실기 D435i에서도 조명이 바뀌면 HSV 범위를 다시 맞춰야 한다(그 파일의 자기 한계).
#   depth는 색·조명과 무관하다. 손목 카메라에서 작업면은 평면이고 큐브는 그 위 5 cm 덩어리이므로,
#   base 프레임으로 역투영한 점들 중 작업면 위 높이 띠에 있는 ★큐브 크기의 덩어리★를 찾으면 큐브다.
#
# 경로
#   depth[m](H,W) ─ 역투영(intrinsics) ─▶ 카메라 점 ─ T_base_cam ─▶ base 점
#     ─▶ 높이 띠(table_z + [z_min, z_max]) ∧ 작업영역 AABB ─▶ 후보 마스크(2D) ─▶ 연결요소들
#     ─▶ 각 요소의 base xy 폭·종횡비·상면 높이가 큐브 크기와 맞는 것만 남김 ─▶ 최대 면적 요소
#     ─▶ 상면 점들(최고 z − top_margin 이내)의 중앙값 xy, 중심 z = 상면 z − size/2
#
# 설계 근거(봉인 표본 2,048프레임 실측, 2026-09-15)
#   · 큐브 요소: xy 폭 5.0×3~5 cm, 상면 높이 5.3 cm(들면 7~8 cm), 픽셀 면적 75~143 px(112²)
#   · 작업면 잡음 파편: 폭 1.3 cm, 높이 1.3 cm, 8 px — 띠 하한 1 cm로는 새어 들어온다 → 하한 2.5 cm
#   · 손가락(카메라 0.2~0.26 m): 폭 ~1.5 cm × 5 cm, 종횡비 > 3 → 폭·종횡비 조건에서 탈락
#   · 픽셀 면적을 기하 기대치와 비교하는 필터는 경사 시야·가림으로 0.23~0.46배가 흔해 ★쓰지 않는다★
# 검증: scripts/tools/validate_teacher_perception_offline.py (봉인 표본 + 참값 대조)
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.types import Intrinsics


@dataclass(frozen=True)
class DepthLocatorCfg:
    object_size_m: float = 0.05
    table_height_m: float = -0.0375
    # 작업면 위 높이 띠 [m] — 하한은 면 잡음·캘리브 z 오차 여유(2.5 cm), 상한은 들어올림 여유
    z_band_m: tuple[float, float] = (0.025, 0.150)
    # 작업영역 AABB(base) — configs/safety.yaml workspace와 같은 값을 넘긴다
    workspace_min: tuple[float, float, float] = (-1.0, -1.0, -0.2)
    workspace_max: tuple[float, float, float] = (1.0, 1.0, 0.6)
    min_depth_m: float = 0.02          # 이하 = 결측
    max_depth_m: float = 1.50
    min_points: int = 12
    top_face_margin_m: float = 0.012   # 상면으로 간주할 최고 z 이내 두께
    # 큐브 요소 형상 조건(base xy) — object_size 배율
    footprint_min_frac: float = 0.35   # 작은 쪽 폭 ≥ 1.75 cm  (파편·손가락 탈락)
    footprint_max_frac: float = 1.60   # 큰 쪽 폭 ≤ 8 cm      (배경 덩어리 탈락)
    max_aspect: float = 2.5
    min_top_height_m: float = 0.035    # 상면 높이 ≥ 3.5 cm (5 cm 큐브)


@dataclass(frozen=True)
class DepthCubeEstimate:
    present: bool
    reason: str
    center_base: tuple[float, float, float] | None = None
    top_z_base: float | None = None
    n_points: int = 0
    footprint_m: tuple[float, float] | None = None
    centroid_px: tuple[int, int] | None = None
    n_candidates: int = 0


def deproject(depth_m: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """depth(H,W) → 카메라 점 (H,W,3) [x 우, y 하, z 전방]과 유한 마스크."""
    h, w = depth_m.shape
    if (w, h) != (int(intr.width), int(intr.height)):
        raise ValueError(f"depth {w}x{h} vs intrinsics {intr.width}x{intr.height}")
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = np.asarray(depth_m, dtype=np.float64)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.stack([x, y, z], axis=-1), np.isfinite(z)


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """8-연결 요소 마스크 목록 (OpenCV 있으면 사용, 없으면 numpy BFS)."""
    try:
        import cv2
        count, labels, _stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        return [labels == k for k in range(1, count)]
    except ImportError:
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        comps = []
        for sy, sx in zip(*np.nonzero(mask)):
            if seen[sy, sx]:
                continue
            comp = np.zeros_like(mask, dtype=bool)
            stack = [(sy, sx)]
            seen[sy, sx] = True
            while stack:
                y, x = stack.pop()
                comp[y, x] = True
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(comp)
        return comps


def locate_cube_from_depth(depth_m: np.ndarray, intr: Intrinsics, T_base_cam: np.ndarray,
                           cfg: DepthLocatorCfg, *, color_mask: np.ndarray | None = None) -> DepthCubeEstimate:
    """작업면 위 높이 띠에서 큐브 크기의 덩어리를 찾아 중심(base)을 추정한다."""
    T = np.asarray(T_base_cam, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_base_cam must be 4x4")
    depth = np.asarray(depth_m, dtype=np.float64)
    pts_cam, finite = deproject(depth, intr)
    valid = finite & (depth > cfg.min_depth_m) & (depth < cfg.max_depth_m)
    if not valid.any():
        return DepthCubeEstimate(False, "유효 depth 없음")
    flat = pts_cam.reshape(-1, 3)
    base = ((T[:3, :3] @ flat.T).T + T[:3, 3]).reshape(depth.shape + (3,))
    lo = np.asarray(cfg.workspace_min, dtype=np.float64); hi = np.asarray(cfg.workspace_max, dtype=np.float64)
    z0 = cfg.table_height_m
    band = valid & (base[..., 2] > z0 + cfg.z_band_m[0]) & (base[..., 2] < z0 + cfg.z_band_m[1])
    inside = np.all((base[..., :2] >= lo[:2]) & (base[..., :2] <= hi[:2]), axis=-1)
    cand = band & inside
    if color_mask is not None:
        cand &= np.asarray(color_mask, dtype=bool)
    n_cand = int(cand.sum())
    if n_cand < cfg.min_points:
        return DepthCubeEstimate(False, f"높이 띠 후보 {n_cand}px < {cfg.min_points}", n_points=n_cand)

    size = cfg.object_size_m
    best = None; best_area = 0; n_ok = 0; rejects = []
    for comp in connected_components(cand):
        area = int(comp.sum())
        if area < cfg.min_points:
            continue
        pts = base[comp]
        ext = pts[:, :2].max(axis=0) - pts[:, :2].min(axis=0)
        e_min, e_max = float(ext.min()), float(ext.max())
        top_h = float(pts[:, 2].max() - z0)
        if e_min < cfg.footprint_min_frac * size:
            rejects.append("좁음"); continue
        if e_max > cfg.footprint_max_frac * size:
            rejects.append("넓음"); continue
        if e_max / max(e_min, 1e-6) > cfg.max_aspect:
            rejects.append("길쭉"); continue
        if top_h < cfg.min_top_height_m:
            rejects.append("낮음"); continue
        n_ok += 1
        if area > best_area:
            best, best_area, best_ext, best_pts = comp, area, (e_min, e_max), pts
    if best is None:
        summary = ", ".join(f"{r}×{rejects.count(r)}" for r in dict.fromkeys(rejects)) or "요소 없음"
        return DepthCubeEstimate(False, f"큐브 형상 요소 없음 ({summary})", n_points=n_cand)
    top_z = float(best_pts[:, 2].max())
    top = best_pts[best_pts[:, 2] >= top_z - cfg.top_face_margin_m]
    ys, xs = np.nonzero(best)
    center = (float(np.median(top[:, 0])), float(np.median(top[:, 1])), float(np.median(top[:, 2])) - size / 2.0)
    return DepthCubeEstimate(True, "", center, float(np.median(top[:, 2])), best_area, best_ext,
                             (int(round(xs.mean())), int(round(ys.mean()))), n_ok)
