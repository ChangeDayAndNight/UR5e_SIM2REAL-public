# RealCamera — pyrealsense2 기반 CameraInterface 실기 어댑터 (요소6).
"""RealCamera: RealSense D435i(★eye-in-hand — 손목 브래킷 장착★)를 CameraInterface 계약으로 감싼다.

목적: depth 프레임을 "meter 단위 float32 (H,W)"(CameraFrame 계약)로 노출한다.
      결측(0)은 그대로 반환 — 처리 책임은 common.obs_preprocess에 있다(요소1).
      ★RGB-D 확장(2026-07-17, B안)★: enable_rgb=True면 color 스트림을 함께 켜고
      rs.align(depth→color)으로 정렬한 (depth_m, rgb) 쌍을 CameraFrame으로 반환.
      enable_rgb 값은 configs/common.yaml preprocess.use_rgb(=PreprocessCfg.use_rgb)
      단일 소스에서 호출측이 넘긴다 — real.local.yaml에 별도 스위치를 두지 않는다(§4-3).
사용: env_ur_real(배포 PC) 전용 — Deploy_*.py / Calib_HandEye.py가 생성.
산출: 없음 (라이브러리).

⚠️ pyrealsense2 호출 경로는 구현되었지만 배포 PC의 exact 2.58.3.10794 API·
   D435i 스트림 프로파일 실기 검증이 남은 P0이다 (vendored 소스 없음).
⚠️ isaaclab / isaacsim import 금지. pyrealsense2는 start()에서 지연 import —
   SDK 없는 학습 PC에서도 모듈 import는 성공해야 한다.
⚠️ 필터 체인은 기본 OFF (real.local.yaml camera.enable_filters=false):
   off → sim이 raw 노이즈를 DR로 모사(현재 dr.yaml 가정)라는 프로젝트 결정과
   연동 — on/off 최종 결정은 deployment/Verify_ObsParity.py 결과로 내린다.
"""
from __future__ import annotations

import time

import numpy as np

from common.interfaces import CameraInterface
from common.types import CameraFrame, Intrinsics

# 오토익스포저/레이저 안정화용 워밍업 프레임 수 (~1초 @30fps).
_WARMUP_FRAMES = 30
_GLOBAL_TIME_WARMUP_S = 16.0

# wait_for_frames 타임아웃 [ms] — 제어주기(50ms)보다 길되 무한대기 방지.
_FRAME_TIMEOUT_MS = 100

_REALSENSE_TIMESTAMP_DOMAINS = {
    "hardware_clock",
    "system_time",
    "global_time",
}


def _source_timestamp_domain(value: object) -> str:
    """Normalize an SDK timestamp-domain enum without hiding unknown values."""
    name = getattr(value, "name", None)
    raw = str(name if name is not None else value).strip().lower()
    raw = raw.rsplit(".", 1)[-1].replace(" ", "_").replace("-", "_")
    if raw not in _REALSENSE_TIMESTAMP_DOMAINS:
        raise RuntimeError(f"[real] unsupported RealSense timestamp domain: {value!r}")
    return f"realsense_{raw}"


class RealCamera(CameraInterface):
    """D435i depth 스트림 어댑터 (스캐폴딩).

    생성 인자:
      camera_cfg: 명시적으로 선택한 configs/real.local.yaml의 camera 섹션 dict
                  (serial/width/height/fps/enable_filters)
      enable_rgb: RGB-D 계약 스위치 — ★호출측이 PreprocessCfg.use_rgb를 그대로
                  넘긴다★ (common.yaml preprocess.use_rgb 단일 소스, §4-3).
                  True면 color 스트림 + rs.align(depth→color) 활성.
    """

    def __init__(self, camera_cfg: dict, enable_rgb: bool = False) -> None:
        self._serial: str = str(camera_cfg.get("serial", "") or "")
        self._width: int = int(camera_cfg["width"])
        self._height: int = int(camera_cfg["height"])
        self._fps: int = int(camera_cfg["fps"])
        self._frame_timeout_ms: int = int(camera_cfg.get("frame_timeout_ms", _FRAME_TIMEOUT_MS))
        if not self._serial:
            raise ValueError("[real] camera.serial is required; first-device auto-selection is forbidden")
        if not 1 <= self._frame_timeout_ms <= 100:
            raise ValueError("[real] camera.frame_timeout_ms must be in [1,100]")
        self._enable_filters: bool = bool(camera_cfg.get("enable_filters", False))
        self._enable_rgb: bool = bool(enable_rgb)
        if camera_cfg.get("global_time_enabled") is not True:
            raise ValueError("[real] camera.global_time_enabled must be explicitly true")
        self._frames_queue_size = int(camera_cfg.get("frames_queue_size", 0))
        if self._frames_queue_size != 1:
            raise ValueError("[real] camera.frames_queue_size must be exactly 1")

        self._rs = None  # pyrealsense2 모듈 핸들 (start()에서 지연 import)
        self._pipeline = None
        self._profile = None
        self._depth_scale: float | None = None  # raw uint16 → meter 변환 계수
        self._filters: list = []
        self._align = None  # rs.align(depth→color) — enable_rgb일 때만 생성
        self._last_sequence: int | None = None
        self._actual_serial: str | None = None
        self._firmware_version: str | None = None

    # ----- CameraInterface 구현 -----

    def start(self) -> None:
        """pipeline 시작 + depth scale 취득 + 워밍업 프레임 소진."""
        try:
            import pyrealsense2 as rs  # 지연 import (모듈 최상위 금지)
        except ImportError as e:
            raise RuntimeError(
                "[real] pyrealsense2 미설치 — env_ur_real 활성화 후 bundle root에서 "
                "`python -m pip install --no-index --find-links "
                "deployment/wheelhouse --require-hashes -r "
                "deployment/requirements-win11.lock.txt` 후 사용할 것"
            ) from e
        self._rs = rs

        pipeline = rs.pipeline()
        config = rs.config()
        if self._serial:
            # ⚠️ TODO(확인 필요): config.enable_device(serial: str) 시그니처.
            config.enable_device(self._serial)
        # ⚠️ TODO(확인 필요): enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        #   인자 순서와 848x480@30이 D435i에서 유효한 depth 프로파일인지.
        config.enable_stream(
            rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
        )
        if self._enable_rgb:
            # ⚠️ TODO(확인 필요): enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
            #   인자 순서와 ★848x480@30이 D435i color 센서(RGB 모듈)의 유효 프로파일인지★
            #   — rs-enumerate-devices로 실기에서 확인할 것. 무효 프로파일이면
            #   pipeline.start가 실패한다(다른 해상도로 받으면 depth와 정합이 깨지므로
            #   해상도 변경이 아니라 프로파일 표에서 같은 848x480을 찾는 것이 우선).
            # rgb8 = ★RGB 채널 순서★ (CameraFrame 계약 — bgr8 금지).
            config.enable_stream(
                rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps
            )

        try:
            self._profile = pipeline.start(config)
        except Exception as e:
            raise RuntimeError(
                f"[real] RealSense 시작 실패 (serial={self._serial or '첫 장치'}) — "
                f"USB3 연결/점유 프로세스 확인: {e}"
            ) from e
        # Record ownership immediately after start succeeds.  Every later SDK
        # failure must be able to stop the USB stream through the one cleanup
        # path, including serial/depth-scale/filter/warmup failures.
        self._pipeline = pipeline
        try:
            # ⚠️ TODO(확인 필요): get_device().first_depth_sensor().get_depth_scale()
            #   경로 (D435i 계열 depth scale ≈ 0.001 m/unit 예상 — 실측값 로그 확인).
            depth_sensor = self._profile.get_device().first_depth_sensor()
            actual_serial = str(
                self._profile.get_device().get_info(rs.camera_info.serial_number)
            )
            if actual_serial != self._serial:
                raise RuntimeError(
                    f"[real] connected camera serial {actual_serial!r} != configured {self._serial!r}"
                )
            self._actual_serial = actual_serial
            self._firmware_version = str(
                self._profile.get_device().get_info(rs.camera_info.firmware_version)
            )
            self._depth_scale = float(depth_sensor.get_depth_scale())
            if not 1.0e-4 <= self._depth_scale <= 1.0e-2:
                raise RuntimeError(f"[real] implausible depth scale: {self._depth_scale}")

            # Absolute no-motion age evidence requires an epoch comparable to
            # the host wall clock and no application-visible frame backlog.
            # Do not accept the device hardware-clock epoch or a queue larger
            # than one.  Read both options back after setting them.
            for option_name, required_value in (
                ("global_time_enabled", 1.0),
                ("frames_queue_size", 1.0),
            ):
                option = getattr(rs.option, option_name, None)
                if option is None or not depth_sensor.supports(option):
                    raise RuntimeError(
                        f"[real] RealSense required option unavailable: {option_name}"
                    )
                depth_sensor.set_option(option, required_value)
                actual = float(depth_sensor.get_option(option))
                if actual != required_value:
                    raise RuntimeError(
                        f"[real] RealSense {option_name} readback={actual}, "
                        f"required={required_value}"
                    )

            if self._enable_rgb:
                # ⚠️ TODO(확인 필요): rs.align(rs.stream.color) 생성자 시그니처 —
                #   인자로 '정렬 대상(align-to)' 스트림을 받는 것으로 알려져 있으나 미검증.
                #   depth를 color 프레임에 재투영(align)하는 방향이 계약이다:
                #   hand_eye v7 캘리브가 ★color optical frame 기준★이므로 정렬 후
                #   depth/rgb 모두 color intrinsics 하나로 유효해야 한다.
                self._align = rs.align(rs.stream.color)

            if self._enable_filters:
                # ⚠️ TODO(확인 필요): 필터 클래스명/기본 파라미터 —
                #   rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()
                #   추정. 적용 순서 권장안(decimation→spatial→temporal→hole)도 문서 확인.
                # ⚠️ TODO(프로젝트 결정): 필터 on/off는 deployment/Verify_ObsParity.py 결과로 결정 —
                #   on으로 갈 경우 dr.yaml의 depth 노이즈 범위 재조정 필요.
                self._filters = [rs.spatial_filter(), rs.temporal_filter()]
                print("[real] ⚠️ depth 필터 체인 ON — sim DR 가정(raw)과 다름, ObsParity 필수")

            # 워밍업: 오토익스포저 안정화 전 프레임은 버린다.
            # librealsense documents early global timestamps as unstable.  A
            # 16-second discard window exceeds the documented 15-second
            # interval and is part of the evidence contract, not a tuning knob.
            warmup_frames = max(_WARMUP_FRAMES, int(self._fps * _GLOBAL_TIME_WARMUP_S))
            for _ in range(warmup_frames):
                # ⚠️ TODO(확인 필요): wait_for_frames(timeout_ms=...) 키워드 인자명.
                pipeline.wait_for_frames(self._frame_timeout_ms)
        except BaseException:
            self.stop()
            raise
        print(
            f"[real] 카메라 시작: {self._width}x{self._height}@{self._fps}fps, "
            f"depth_scale={self._depth_scale:.6f} m/unit, "
            f"filters={'on' if self._enable_filters else 'off'}, "
            f"rgb={'on(align→color)' if self._enable_rgb else 'off'}"
        )

    def get_frame(self) -> CameraFrame:
        """최신 프레임 → depth (H,W) float32 [m] (+enable_rgb면 rgb (H,W,3) uint8).

        enable_rgb=True면 rs.align으로 depth를 color 프레임에 재투영한 뒤 반환 —
        정렬 후 depth/rgb는 ★같은 color intrinsics★로 정합된다(CameraFrame 계약).
        결측(raw 0)은 0.0 그대로 반환 (처리 책임은 common.obs_preprocess).
        """
        if self._pipeline is None or self._depth_scale is None:
            raise RuntimeError("[real] 카메라 미시작 — start() 먼저 호출할 것")

        # ⚠️ TODO(확인 필요): wait_for_frames 타임아웃 초과 시 예외형 —
        #   RuntimeError로 나오는지 확인 후 rollout 쪽 trip 경로와 맞출 것.
        frames = self._pipeline.wait_for_frames(self._frame_timeout_ms)
        # Pair wall and monotonic clocks immediately when the SDK hands the
        # frameset to the application. Conversion/filter time remains visible
        # later as cycle_end - timestamp instead of being hidden.
        host_arrival_timestamp = time.perf_counter()
        host_wall_timestamp_s = time.time()

        rgb: np.ndarray | None = None
        if self._enable_rgb:
            # ⚠️ TODO(확인 필요): align.process(frames)가 frameset을 반환하고 그
            #   frameset의 get_depth_frame()/get_color_frame()이 정렬된 프레임인지,
            #   process 반환을 as_frameset()으로 재캐스팅해야 하는지 미검증.
            frames = self._align.process(frames)
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError("[real] color 프레임 없음 (enable_rgb인데 스트림 이상)")
            # ⚠️ TODO(확인 필요): rgb8 프로파일에서 get_data()가 (H,W,3) uint8로
            #   나오는지 (stride 패딩 없이 reshape되는지) 실기 확인.
            rgb = np.asanyarray(color.get_data())  # (H,W,3) uint8 ★RGB★

        depth = frames.get_depth_frame()
        if not depth:
            raise RuntimeError("[real] depth 프레임 없음 (스트림 이상)")
        sequence_id = int(depth.get_frame_number())
        try:
            # librealsense reports milliseconds.  Preserve the SDK metadata and
            # its clock domain before filters/conversion; neither value is a
            # host-arrival timestamp.
            source_timestamp_s = float(depth.get_timestamp()) / 1000.0
            source_timestamp_domain = _source_timestamp_domain(
                depth.get_frame_timestamp_domain()
            )
        except Exception as exc:
            raise RuntimeError("[real] RealSense frame timestamp metadata unavailable") from exc
        if not np.isfinite(source_timestamp_s) or source_timestamp_s < 0.0:
            raise RuntimeError(
                f"[real] invalid RealSense frame timestamp: {source_timestamp_s!r}"
            )
        if self._last_sequence is not None and sequence_id <= self._last_sequence:
            raise RuntimeError(
                f"[real] stale/duplicate depth frame: {sequence_id} <= {self._last_sequence}"
            )
        self._last_sequence = sequence_id

        for f in self._filters:
            # ⚠️ TODO(확인 필요): filter.process(frame) 반환이 get_data() 가능한
            #   depth frame인지, as_depth_frame() 재캐스팅이 필요한지.
            # ⚠️ TODO(확인 필요 — 순서 이슈): 현재는 align '이후' 필터 적용인데,
            #   librealsense 권장 순서(필터→align vs align→필터)를 문서로 확인할 것.
            #   align 후 depth는 재투영된 프레임이라 spatial/temporal 필터의 가정
            #   (원시 센서 격자)과 다를 수 있다 — enable_filters=true로 갈 경우 필수 확인.
            depth = f.process(depth)

        raw = np.asanyarray(depth.get_data())  # (H,W) uint16
        depth_m = raw.astype(np.float32) * np.float32(self._depth_scale)  # ★meter 계약★
        return CameraFrame(
            depth_m=depth_m,
            timestamp=host_arrival_timestamp,
            rgb=rgb,
            sequence_id=sequence_id,
            source_timestamp_s=source_timestamp_s,
            source_timestamp_domain=source_timestamp_domain,
            host_wall_timestamp_s=host_wall_timestamp_s,
        )

    def get_intrinsics(self) -> Intrinsics:
        """활성 프로파일의 원본 해상도 intrinsics (역투영/캘리브레이션용 — 요소4).

        ★계약★: enable_rgb=True면 ★color 스트림 intrinsics★를 반환한다 —
        rs.align(depth→color) 후 get_frame()의 depth/rgb는 둘 다 color 프레임
        격자 위에 있으므로 유효 intrinsics = color intrinsics이고,
        hand_eye v7 캘리브도 color optical frame 기준이다(단일 기준 유지).
        enable_rgb=False면 기존대로 depth 스트림 intrinsics.
        """
        if self._profile is None or self._rs is None:
            raise RuntimeError("[real] 카메라 미시작 — start() 먼저 호출할 것")
        # ⚠️ TODO(확인 필요): get_stream(rs.stream.depth|color).as_video_stream_profile()
        #   .get_intrinsics() 경로 및 필드명 fx/fy/ppx/ppy/width/height —
        #   color 스트림에서도 같은 경로가 유효한지 포함해 미검증.
        #   (distortion 계수는 Intrinsics 계약에 없음 — 필요해지면 계약 확장 논의)
        stream = self._rs.stream.color if self._enable_rgb else self._rs.stream.depth
        intr = (
            self._profile.get_stream(stream)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        return Intrinsics(
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            width=int(intr.width),
            height=int(intr.height),
        )

    @property
    def actual_serial(self) -> str:
        if self._actual_serial is None:
            raise RuntimeError("[real] camera serial unavailable before start()")
        return self._actual_serial

    @property
    def firmware_version(self) -> str:
        if self._firmware_version is None:
            raise RuntimeError("[real] camera firmware unavailable before start()")
        return self._firmware_version

    def stop(self) -> None:
        """스트림 종료 (여러 번 호출해도 안전)."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                print(f"[real][warn] 카메라 정지 실패(무시): {e}")
            self._pipeline = None
            self._profile = None
            self._depth_scale = None
            self._filters = []
            self._align = None
            self._last_sequence = None
            self._actual_serial = None
            self._firmware_version = None
            print("[real] 카메라 종료")
