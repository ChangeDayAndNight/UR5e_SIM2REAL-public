# ★배포 v2 — 학습 화각에 실기 프레임을 맞추는 레터박스★ [배포 PC][공유 전처리 '이전']
#
# 무엇을 고치나
#   Isaac TiledCamera는 ★등방★으로 렌더한다. vendored tiled_camera.py가
#   f_x = W·focal/horizontal_aperture, f_y = f_x로 두므로(줄 596-597) sim.yaml의
#   vertical_fov_deg는 렌더에 반영되지 않는다. 즉 학습에서 student가 본 112² 프레임은
#   가로·세로 ★모두 69.4°★를 담는다.
#   실기 D435i color는 848×480에 가로 69.4°·세로 42.5°다. frame_mode="full"은 이 전체
#   프레임을 112²로 ★비등방 리사이즈★하므로, 같은 물체가 실기에서 세로로
#   tan(34.7°)/tan(21.25°) ≈ ★1.78배 크게★ 보인다. 학습 분포와 어긋난다.
#
# 어떻게 고치나 (공유 전처리는 한 줄도 건드리지 않는다)
#   전처리 ★이전★ 단계에서 원본 프레임을 캔버스에 레터박스한다. 캔버스는 원본과 같은
#   화소 밀도를 유지하되, 학습 화각이 덮는 각도만큼 넓다. 그 뒤 기존 full 리사이즈를
#   통과시키면 물체의 화면 점유가 학습과 일치한다. sim 프레임처럼 화각이 이미 같으면
#   캔버스가 원본과 같아 ★항등 변환★이 된다 — sim/real 분기가 아니라 '원본 화각'이라는
#   파라미터 하나로 양쪽을 같은 계약에 올린다.
#
# ★정직한 한계★
#   실기 카메라는 세로 42.5°밖에 못 본다. 학습이 본 69.4°의 나머지는 ★데이터가 없다★.
#   그 영역은 depth 0(결측 → 전처리가 far로 채움)·RGB 0(검정)으로 채운다. 학습 분포에는
#   그 자리에 실제 장면이 있었으므로 이 공백 자체가 새로운 괴리다. 기하(물체 크기·위치)를
#   맞추는 대신 주변 맥락을 잃는 거래이고, 어느 쪽이 유리한지는 ★실기 대조로만★ 정해진다.
#   그래서 배포 번들을 보정 적용/미적용 두 가지로 만든다.
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from common.interfaces import CameraInterface
from common.types import CameraFrame, Intrinsics


@dataclass(frozen=True)
class FovSpec:
    """프레임이 덮는 화각 [deg]. 수평·수직 각각 전체 각(半角이 아니다)."""

    horizontal_deg: float
    vertical_deg: float

    def __post_init__(self) -> None:
        for name, value in (("horizontal_deg", self.horizontal_deg), ("vertical_deg", self.vertical_deg)):
            if not (0.0 < float(value) < 180.0):
                raise ValueError(f"{name} must lie in (0, 180): {value}")

    @staticmethod
    def from_intrinsics(intr: Intrinsics) -> "FovSpec":
        return FovSpec(
            horizontal_deg=2.0 * math.degrees(math.atan(intr.width / (2.0 * intr.fx))),
            vertical_deg=2.0 * math.degrees(math.atan(intr.height / (2.0 * intr.fy))),
        )


def _half_tan(deg: float) -> float:
    return math.tan(math.radians(float(deg)) / 2.0)


def canvas_shape(height: int, width: int, source: FovSpec, target: FovSpec) -> tuple[int, int, int, int]:
    """(캔버스 h, 캔버스 w, 원본이 놓일 top, left). 화각이 같으면 원본 그대로."""
    if height <= 0 or width <= 0:
        raise ValueError("frame shape must be positive")
    scale_y = _half_tan(source.vertical_deg) / _half_tan(target.vertical_deg)
    scale_x = _half_tan(source.horizontal_deg) / _half_tan(target.horizontal_deg)
    if scale_y > 1.0 + 1e-9 or scale_x > 1.0 + 1e-9:
        raise ValueError(
            f"source FOV exceeds the training FOV (scale x={scale_x:.3f} y={scale_y:.3f}); "
            "cropping is out of scope for this transform"
        )
    canvas_h = max(height, int(round(height / scale_y)))
    canvas_w = max(width, int(round(width / scale_x)))
    return canvas_h, canvas_w, (canvas_h - height) // 2, (canvas_w - width) // 2


def letterbox_frame(frame: CameraFrame, source: FovSpec, target: FovSpec) -> CameraFrame:
    """원본 프레임을 학습 화각 캔버스 가운데에 놓는다(빈 곳: depth 0 = 결측, RGB 0)."""
    depth = np.asarray(frame.depth_m)
    if depth.ndim != 2:
        raise ValueError(f"depth must be (H, W): {depth.shape}")
    h, w = depth.shape
    canvas_h, canvas_w, top, left = canvas_shape(h, w, source, target)
    if (canvas_h, canvas_w) == (h, w):
        return frame  # 화각 동일 — 항등
    padded_depth = np.zeros((canvas_h, canvas_w), dtype=depth.dtype)
    padded_depth[top:top + h, left:left + w] = depth
    padded_rgb = None
    if frame.rgb is not None:
        rgb = np.asarray(frame.rgb)
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"rgb shape {rgb.shape[:2]} does not match depth {(h, w)}")
        padded_rgb = np.zeros((canvas_h, canvas_w, rgb.shape[2]), dtype=rgb.dtype)
        padded_rgb[top:top + h, left:left + w] = rgb
    return CameraFrame(
        depth_m=padded_depth, timestamp=frame.timestamp, rgb=padded_rgb,
        sequence_id=frame.sequence_id, source_timestamp_s=frame.source_timestamp_s,
        source_timestamp_domain=frame.source_timestamp_domain,
        host_wall_timestamp_s=frame.host_wall_timestamp_s,
    )


class FovMatchedCamera(CameraInterface):
    """CameraInterface 래퍼 — get_frame()마다 학습 화각으로 레터박스한다.

    get_intrinsics()는 ★원본★ intrinsics를 그대로 돌려준다. 역투영·캘리브레이션 검증은
    원본 격자 위에서 이뤄져야 하고, 이 변환은 정책 입력 경로에만 적용되기 때문이다.
    """

    def __init__(self, camera: CameraInterface, target: FovSpec, source: FovSpec | None = None) -> None:
        self._camera = camera
        self._target = target
        self._source = source
        self.frames = 0

    @property
    def source_fov(self) -> FovSpec:
        if self._source is None:
            self._source = FovSpec.from_intrinsics(self._camera.get_intrinsics())
        return self._source

    def start(self) -> None:
        self._camera.start()

    def stop(self) -> None:
        self._camera.stop()

    def get_intrinsics(self) -> Intrinsics:
        return self._camera.get_intrinsics()

    def get_frame(self) -> CameraFrame:
        self.frames += 1
        return letterbox_frame(self._camera.get_frame(), self.source_fov, self._target)

    def describe(self) -> dict:
        source = self.source_fov
        return {"source_fov_deg": [source.horizontal_deg, source.vertical_deg],
                "target_fov_deg": [self._target.horizontal_deg, self._target.vertical_deg],
                "frames": self.frames}
