"""SimCamera — CameraInterface의 Isaac Lab 구현 (Phase3 SimDeploy_Rehearsal 전용).

목적: 요소6 쌍둥이의 sim 쪽 카메라 절반. scene의 TiledCamera("tiled_camera")를
      CameraInterface 계약(depth[m], 결측 그대로, Intrinsics)으로 감싼다.
      real/real_camera.py(pyrealsense2 D435i)와 배포 루프에서 완전 호환.
실행: 직접 실행 없음 — Phase3/SimDeploy_Rehearsal.py가 생성해 run_rollout에 주입.
산출: SimCamera 클래스.

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것.
"""

from __future__ import annotations

import time

import numpy as np

from common.interfaces import CameraInterface
from common.types import CameraFrame, Intrinsics


class SimCamera(CameraInterface):
    """Isaac Lab TiledCamera 래퍼 — 단일 env 리허설 전용.

    depth 계약: (H,W) float32 [m], 결측(0/NaN)은 그대로 반환 — 전처리
    (common.obs_preprocess)가 처리한다. TiledCameraCfg.depth_clipping_behavior
    ="zero"(scene cfg)라 far 초과/무효 픽셀은 real D435i처럼 0으로 온다.

    ⚠️ 리허설에서는 depth 노이즈를 여기서 추가하지 않는다 — 노이즈 DR는 '학습'
      경로(_binpick_obs.policy_obs) 전용이고, 리허설은 clean sim depth로
      루프·안전·타이밍을 검증한다. (노이즈 리허설이 필요하면 rollout 앞단에서
      apply_depth_noise를 옵션으로 감쌀 것 — rollout.py 수정은 금지.)
    """

    def __init__(self, env, sensor_name: str = "tiled_camera"):
        """env = 생성 완료된 ManagerBasedRLEnv (BinPickVisionEnvCfg 기반, num_envs=1)."""
        if env.num_envs != 1:
            raise ValueError(f"[sim] SimCamera는 단일 env 전용 — num_envs={env.num_envs}")
        if sensor_name not in env.scene.sensors:
            raise KeyError(
                f"[sim] scene에 '{sensor_name}' 센서가 없음 — BinPickVisionEnvCfg"
                f"(카메라 포함)로 env를 만들었는지 확인 (teacher cfg에는 카메라가 없다)"
            )
        self._env = env
        self._sensor = env.scene.sensors[sensor_name]
        self._started = False

    def start(self) -> None:
        """스트림 시작 — sim에서는 센서 수명이 env 소유라 상태 플래그만 세운다.

        real 대응: pipeline.start + 워밍업 프레임 폐기. sim 등가 검증은
        '데이터 버퍼가 유효한가'로 대신한다 (env.reset() 이후여야 유효).
        """
        self._started = True
        print("[sim] SimCamera start (센서 수명은 env 소유 — no-op)")

    def get_frame(self) -> CameraFrame:
        """최신 depth(+rgb) 프레임 — depth [m] (H,W) float32, rgb (H,W,3) uint8.

        TiledCamera 출력 shape (N,H,W,1) 검증됨(tiled_camera.py L347-350) —
        N=1, 채널 squeeze 후 반환. 센서 갱신 주기는 TiledCameraCfg.update_period
        (=1/control_hz)로 scene.update가 관리한다(SimRobot._step_physics 경유).
        ★2026-07-17 RGB-D 전환(적대 리뷰 수정)★: use_rgb 계약이면 scene cfg가
        'rgb' annotator를 이미 켰으므로(단일 소스 common.yaml) 센서 출력 키
        존재 여부로 분기해 rgb도 채운다 — real(rs.align rgb)과 쌍둥이 계약 유지.
        누락 시 build_policy_obs가 ValueError로 리허설 첫 스텝에 죽는다.
        """
        if not self._started:
            raise RuntimeError("[sim] start() 이전에 get_frame() 호출됨")
        depth = self._sensor.data.output["distance_to_image_plane"]  # (1,H,W,1)
        depth_np = depth[0, :, :, 0].detach().cpu().numpy().astype(np.float32)
        rgb_np = None
        if "rgb" in self._sensor.data.output:  # use_rgb 계약 = cfg가 결정(분기 아님)
            rgb = self._sensor.data.output["rgb"][0, :, :, :3]  # (H,W,3) uint8
            rgb_np = rgb.detach().cpu().numpy()
            if rgb_np.dtype != np.uint8:  # 방어: annotator가 float를 주는 경우
                rgb_np = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
        return CameraFrame(
            depth_m=depth_np, timestamp=time.perf_counter(), rgb=rgb_np
        )

    def get_intrinsics(self) -> Intrinsics:
        """렌더 해상도 기준 intrinsics — CameraData.intrinsic_matrices(L43 검증) 사용.

        real과 달리 '원본=출력' 해상도(현행 112x112 렌더)라 그대로 반환.
        역투영 검증(Calib_Verify)에서 hand_eye.yaml과 함께 쓰인다(요소4).
        """
        K = self._sensor.data.intrinsic_matrices[0].detach().cpu().numpy()
        return Intrinsics(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            width=int(self._sensor.cfg.width),
            height=int(self._sensor.cfg.height),
        )

    def stop(self) -> None:
        """스트림 종료 — 플래그만 내린다 (센서/env 해제는 스크립트 책임)."""
        self._started = False
        print("[sim] SimCamera stop")
