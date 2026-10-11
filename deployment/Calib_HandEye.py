#!/usr/bin/env python
"""Phase 3 게이트 1 — Hand-Eye(★T_flange_cam★, eye-in-hand) 실측 캘리브레이션. [사람][배포PC] — Isaac 불필요.

★계약(2026-07-14 사용자 확정 — eye-in-hand)★
  D435i는 손목(플랜지 브래킷)에 장착되어 로봇과 ★함께 움직이고★,
  캘리브레이션 보드(ChArUco)는 ★작업공간에 고정★한다.
  산출 키 = configs/calib/hand_eye.yaml의 ★T_flange_cam★ (flange→camera COLOR optical, 4x4).
  (구 설계 'eye-to-hand: 고정 카메라 + 그리퍼 부착 보드'는 2026-07-17 적대 리뷰로 폐기 —
   solve 산출을 T_base_cam 의미로 만들고 save_hand_eye가 T_flange_cam 키로 저장하면
   ★의미가 뒤바뀐 값이 커밋되는 사고 경로★였다. 본 재작성으로 제거.)

역할 구분:
  - 실측 이력 v6(2026-07-16, 외부 도구 calibration/calibrate_handeye_final.py — ChArUco
    27뷰 + BA, 재투영 RMS 4.11px) → v7(링 동심 병진 보정)이 이미 yaml에 반영돼 있다.
  - 이 스크립트 = 배포 번들 안에서 도는 ★재실측/재검증 경로★ (hand_eye.yaml v7 주석의
    '실측 재검증 필수' 항목 지원). 정밀 최종값이 필요하면 v6 외부 도구(BA) 경로를
    병용할 수 있다 (보드 규격 상수는 그 도구와 동일하게 맞춰져 있음).

실행: command-capable collection is hard-blocked in this receive-only release.
  순수 포즈 목록 점검만 허용:
    python deployment\\Calib_HandEye.py --dry-run --real-config configs/real.local.yaml
산출: configs/calib/hand_eye.yaml 갱신 + deployment/logs/handeye_<시각>/samples.csv (포즈·검출 원장)

기대 출력(정상 시퀀스):
  "[calib] 유효 샘플 N/M" → AX=XB 잔차(회전/병진 median·max) → "★sanity 통과★ |t|≈0.07m"
  → 기존값(v7) 대비 Δpos/Δrot 출력 → 저장 확인(y) → "★갱신 완료★" + §4-5 커밋 안내
실패 시 확인할 로그:
  - 보드 미검출 다수 → deployment/logs/handeye_<시각>/samples.csv의 detected 열로 실패 포즈
    특정 (조명·거리·시야 이탈·모션 블러 점검 후 해당 포즈 교체)
  - sanity 실패(|t|>0.2m) → 프레임 혼동(보드가 세션 중 움직였거나 포즈 입력 오류) —
    저장은 자동 거부되므로 원인 제거 후 재실측
  - solve 예외 → 콘솔 [calib] 메시지 + 포즈 회전 다양성(비평행 회전축 모션 ≥2) 확인

절차([사람] — [Claude]는 코드 작성만, 실행·입회는 사람):
  1) 보드를 ★작업공간에 강체 고정★ (클램프 권장 — 세션 중 1mm라도 움직이면 전체 무효)
  2) 사전에 전 포즈에서 보드가 카메라 시야에 완전히 들어오는지 육안 확인
  3) 스크립트 실행 — CALIB_POSES_RAD 순회(move_to_joints 저속) → 정지 대기 → 촬영·검출
  4) solve → sanity 게이트 → 기존값(v7) 대비 차이 확인(y 입력) → yaml 갱신
  5) ★§4-5 커밋 의무★: 갱신된 hand_eye.yaml을 학습 PC로 회수해 반드시 커밋 —
     커밋 메시지에 잔차(residual) 기재. 실측-공칭 차가 configs/dr.yaml
     camera_pose(±1cm/±1°)를 벗어나면 ★재학습 판정★ — 조용히 진행하지 않고 사용자에게
     보고한다 (판정 게이트 자체는 게이트 3 Calib_Verify가 수행).
  6) 다음 게이트: Calib_TcpOffset.py(게이트 2) → Calib_Verify.py(게이트 3)

⚠️ 컨트롤러 TCP=0 전제(이중 offset 함정): RobotState.tcp_pose는 공칭 T_flange_tcp가
  적용된 값이므로, 같은 yaml의 역을 우측 곱해 flange 포즈를 '대수적으로 정확히' 복원한다
  (flange_mat_from_tcp_pose — 공칭 offset의 물리적 오차와 무관). 단 이 복원은
  real/real_robot.py의 전제(컨트롤러 설정 TCP=0 → getActualTCPPose=flange)를 공유한다 —
  v6 세션도 'TCP=0 확인' 후 실측했음. 배포 PC에서 setTcp([0]*6)/펜던트 확인 필수.

⚠️ 잔여 TODO(확인 필요):
  - 배포 PC(env_ur_real) cv2 버전: aruco Detector 클래스 체계(CharucoDetector)는 4.7+
    필요하다. canonical hash lock은 opencv-python==4.11.0.86이며, 본 파일의 cv2 호출은
    4.11 계열 docstring + v6 실사용 코드로 검증됐다. 배포 PC 설치본에서도 확인할 것.
  - 왜곡 계수: Intrinsics 계약에 distortion이 없어 solvePnP를 무왜곡 가정으로 호출 —
    detect_board_pose docstring 참조 (잔차가 v7보다 크면 v6 BA 경로 병용).
  - CALIB_POSES_RAD 실기 티칭 (현재 비어 있음 — 비면 스크립트가 스스로 중단).
  - real/real_camera.py의 pyrealsense2 TODO(848x480 color 프로파일, rs.align 등).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
# Canonical source와 build_bundle.py가 만든 hash bundle은 모두 deployment/의 부모를
# root로 사용한다. 생성 bundle의 사본 드리프트는 bundle_manifest.json 검증이 차단한다.
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.calibration import (  # noqa: E402
    load_hand_eye,
    load_tcp_offset,
    pose7_to_mat,
    save_hand_eye,
)
from common.config_io import configs_dir, load_yaml  # noqa: E402
from deployment.contracts import load_real_config, require_motion_authorization  # noqa: E402

# =====================================================================
# 상수
# =====================================================================

# ⚠️ TODO(실기 티칭): 캘리브레이션 포즈 세트 — 6관절 [rad], 10~20개 권장.
#   요건(eye-in-hand — 카메라가 움직이는 쪽):
#     ① ★고정 보드★가 모든 포즈에서 카메라 시야 안 (사전 육안 확인)
#     ② 회전 다양성 — 비평행 회전축 모션 ≥2 (cv2 4.11.0 docstring 최소 조건;
#        실전은 각 축 ±20° 이상 다양하게)
#     ③ 보드와의 거리 변화 (검출 스케일 다양성)
#     ④ workspace AABB(safety.yaml) 안쪽 + 브래킷/카메라 충돌 여유
#   비어 있으면 main()이 이동 없이 중단한다 (안전 기본값).
CALIB_POSES_RAD: list[list[float]] = [
    # 예시(홈 포즈 근방) — 실기 티칭으로 교체할 것:
    # [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
]

# 보드 규격 — ★v6 실측 세션(calibration/calibrate_handeye_final.py)과 동일★:
#   ChArUco 11x8 칸, 한 칸 15mm, 마커 11mm, DICT_4X4_50, LEGACY 패턴.
# ⚠️ 보드 실물을 교체하면 이 상수들을 실측으로 갱신할 것 (규격 불일치 = 조용한 오차).
BOARD_SQUARES_X = 11
BOARD_SQUARES_Y = 8
BOARD_SQUARE_M = 0.015
BOARD_MARKER_M = 0.011
BOARD_ARUCO_DICT = "DICT_4X4_50"
BOARD_LEGACY_PATTERN = True

SETTLE_S = 1.0        # 이동 정지 후 진동 감쇠 대기 [s] (손목 카메라 = 진동에 더 민감)
MIN_SAMPLES = 10      # solve에 요구하는 최소 유효 샘플 수
MIN_CORNERS = 6       # 뷰당 최소 ChArUco 코너 수 (v6 도구와 동일 기준)

# ★sanity 게이트★: eye-in-hand의 flange→camera 병진 = 브래킷 오프셋 = 수 cm 스케일
#   (v7 실측 |t|≈0.072m). 0.2m 초과 = 구 eye-to-hand 의미(T_base_cam, 0.5~1m 스케일)
#   같은 프레임 혼동 신호 → 저장 거부.
SANITY_MAX_TRANS_M = 0.2

# cv2 hand-eye method 이름 → cv2 상수명 (존재는 cv2 4.11.0에서 확인됨).
# 기본 park: v6 실측 도구의 초기해와 동일 (CALIB_HAND_EYE_PARK).
HAND_EYE_METHODS = {
    "tsai": "CALIB_HAND_EYE_TSAI",
    "park": "CALIB_HAND_EYE_PARK",
    "horaud": "CALIB_HAND_EYE_HORAUD",
    "andreff": "CALIB_HAND_EYE_ANDREFF",
    "daniilidis": "CALIB_HAND_EYE_DANIILIDIS",
}


# =====================================================================
# 순수 수학 헬퍼 (cv2/하드웨어 불필요 — tests/에서 검증 가능)
# =====================================================================


def flange_mat_from_tcp_pose(tcp_pose7: np.ndarray, T_flange_tcp: np.ndarray) -> np.ndarray:
    """RobotState.tcp_pose(공칭 TCP offset 적용 후) → T_base_flange (4,4) 복원.

    real_robot이 tcp_pose = flange ∘ T_flange_tcp(apply_tcp_offset)로 만들었으므로
    ★같은★ T_flange_tcp의 역을 우측 곱하면 flange 포즈가 대수적으로 정확히 복원된다 —
    공칭 offset의 물리적 오차와 무관 (같은 행렬을 곱했다 빼는 항등 연산).
    ★eye-in-hand 핵심★: calibrateHandEye의 gripper2base 자리에 ★flange★ 포즈를
    넣어야 출력 cam2gripper가 T_flange_cam이 된다 (TCP 포즈를 넣으면 T_tcp_cam이
    나와 키 의미가 어긋난다). 게이트 1이 게이트 2(TcpOffset)보다 앞이어도 되는 이유 =
    이 복원이 offset '정확도'가 아니라 '일관성'만 요구하기 때문.
    """
    T_ft = np.asarray(T_flange_tcp, dtype=np.float64)
    return pose7_to_mat(tcp_pose7) @ np.linalg.inv(T_ft)


def rot_angle_deg(R: np.ndarray) -> float:
    """회전행렬의 회전각 크기 [deg] (순수 수학)."""
    c = (float(np.trace(np.asarray(R)[:3, :3])) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def sanity_check_flange_cam(
    T_flange_cam: np.ndarray, max_trans_m: float = SANITY_MAX_TRANS_M
) -> tuple[bool, str]:
    """★프레임 혼동 조기 검출 게이트★ — 저장 직전 병진 노름 검사 (순수 수학).

    eye-in-hand의 flange→camera 병진은 수 cm 스케일(v7 |t|≈0.072m)이다.
    |t| > max_trans_m(0.2m)이면 T_base_cam(구 eye-to-hand 의미) 같은 잘못된 프레임을
    T_flange_cam 키로 저장하려는 상황일 가능성이 크므로 (False, 사유)를 반환한다 —
    호출측(main)은 이때 저장을 거부해야 한다.
    """
    t = np.asarray(T_flange_cam, dtype=np.float64)[:3, 3]
    norm = float(np.linalg.norm(t))
    if not np.isfinite(norm):
        return False, f"[calib] ⚠️ sanity 실패: 병진에 비유한값 t={t.tolist()} — 저장 거부"
    if norm > max_trans_m:
        return False, (
            f"[calib] ⚠️ sanity 실패: |t|={norm:.3f}m > {max_trans_m:.1f}m — eye-in-hand "
            "플랜지→카메라 오프셋 스케일(수 cm)이 아님. ★프레임 혼동(T_base_cam?) 의심★ "
            "— 저장 거부. 보드 고정/포즈 입력/검출 프레임을 점검할 것"
        )
    return True, f"[calib] ★sanity 통과★ |t|={norm:.3f}m (≤ {max_trans_m:.1f}m — 브래킷 오프셋 스케일)"


def axxb_residual(
    T_base_flange_list: list[np.ndarray],
    T_cam_target_list: list[np.ndarray],
    T_flange_cam: np.ndarray,
) -> dict:
    """AX=XB 정합 잔차 — 순수 numpy (cv2 불필요, tests 가능).

    cv2 4.11.0 docstring의 eye-in-hand 유도식과 동일 규약:
      A_k = inv(T_bf[k+1]) @ T_bf[k],  B_k = T_ct[k+1] @ inv(T_ct[k]),  A_k X = X B_k.
    연속 포즈쌍마다 D_k = (A_k X) vs (X B_k)의 회전각[deg]·병진[m] 차이를 모아
    median/max를 반환한다 → meta.residual_m 기록용 (완벽 데이터면 전부 0).
    """
    X = np.asarray(T_flange_cam, dtype=np.float64)
    rot_errs: list[float] = []
    trans_errs: list[float] = []
    for k in range(len(T_base_flange_list) - 1):
        A = np.linalg.inv(T_base_flange_list[k + 1]) @ T_base_flange_list[k]
        B = T_cam_target_list[k + 1] @ np.linalg.inv(T_cam_target_list[k])
        L = A @ X
        R = X @ B
        rot_errs.append(rot_angle_deg(L[:3, :3] @ R[:3, :3].T))
        trans_errs.append(float(np.linalg.norm(L[:3, 3] - R[:3, 3])))
    if not rot_errs:
        return {"n_pairs": 0, "rot_med_deg": None, "rot_max_deg": None,
                "trans_med_m": None, "trans_max_m": None}
    return {
        "n_pairs": len(rot_errs),
        "rot_med_deg": float(np.median(rot_errs)),
        "rot_max_deg": float(np.max(rot_errs)),
        "trans_med_m": float(np.median(trans_errs)),
        "trans_max_m": float(np.max(trans_errs)),
    }


# =====================================================================
# 취득 / 검출 / 솔브
# =====================================================================


def grab_color_image(camera) -> np.ndarray:
    """카메라에서 보드 검출용 컬러 이미지 1장 취득 → (H,W,3) uint8 ★RGB★.

    ★경로 확정(2026-07-17)★: RealCamera(enable_rgb=True) → camera.get_frame().rgb
    (CameraFrame 계약 — RGB 채널 순서, BGR 금지). 검출 함수는 이 배열을 받아
    ★COLOR_RGB2GRAY★로 변환한다 (cv2 함수에 BGR로 오인시키지 않도록 이 함수는
    변환 없이 RGB 그대로 반환 — 채널 의미는 CameraFrame 계약이 단일 소스).
    ⚠️ 잔여 TODO는 real/real_camera.py의 pyrealsense2 확인 항목(프로파일/rs.align).
    """
    frame = camera.get_frame()
    if frame.rgb is None:
        raise RuntimeError(
            "[calib] ⚠️ frame.rgb=None — RealCamera(enable_rgb=True)로 생성됐는지 확인 "
            "(캘리브레이션은 enable_rgb=True 고정)"
        )
    return frame.rgb


def detect_board_pose(rgb_image: np.ndarray, intr) -> np.ndarray | None:
    """RGB 프레임 → T_cam_target (4,4) = 보드(target)→카메라 변환. 검출 실패 시 None.

    파이프라인 = v6 실측 도구(calibration/calibrate_handeye_final.py)와 동일 규격:
      CharucoDetector.detectBoard → CharucoBoard.matchImagePoints → solvePnP → Rodrigues.
    ★채널 순서★: 입력은 CameraFrame 계약의 ★RGB★ — cv2는 BGR 관례이므로 그레이 변환은
      반드시 cv2.COLOR_RGB2GRAY (COLOR_BGR2GRAY를 쓰면 R/B 가중치가 뒤집힌다).
    ★intrinsics 기준★: intr = camera.get_intrinsics() = enable_rgb=True일 때
      ★color 스트림★ intrinsics — 검출 이미지·hand_eye v7(COLOR optical)과 같은 기준.

    API 검증 상태: CharucoDetector/CharucoBoard/setLegacyPattern/matchImagePoints/
      solvePnP/Rodrigues는 학습 PC isaac51 설치본 cv2 4.11.0에서 존재 확인 + v6 도구가
      동일 호출로 실측 성공(27뷰/1890코너).
    ⚠️ TODO(확인 필요): 배포 PC cv2 버전이 4.7+ (aruco Detector 클래스 체계)인지.
    ⚠️ TODO(확인 필요): 왜곡 계수 — Intrinsics 계약에 distortion이 없어 dist=None
      (무왜곡 가정)으로 solvePnP한다. D435i color는 왜곡이 0이 아니므로(v6에서
      calibrateCamera로 실측했음) AX=XB 잔차가 v7 residual(0.002m)보다 뚜렷이 크면:
      (a) pyrealsense2 intrinsics의 왜곡 계수 취득 방법 확정(Intrinsics 계약 확장 논의)
      또는 (b) 정밀값은 v6 외부 도구(BA) 경로 재사용.
    """
    import cv2  # 지연 import — 학습 PC 순수 pytest에서 모듈 import가 가능해야 함

    gray = cv2.cvtColor(np.ascontiguousarray(rgb_image), cv2.COLOR_RGB2GRAY)  # ★RGB→GRAY★
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, BOARD_ARUCO_DICT))
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y), BOARD_SQUARE_M, BOARD_MARKER_M, dictionary
    )
    board.setLegacyPattern(BOARD_LEGACY_PATTERN)
    detector = cv2.aruco.CharucoDetector(board)

    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_ids is None or len(charuco_ids) < MIN_CORNERS:
        return None
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or len(obj_pts) < MIN_CORNERS:
        return None

    K = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts.reshape(-1, 3).astype(np.float64),
        img_pts.reshape(-1, 2).astype(np.float64),
        K,
        None,  # ⚠️ 무왜곡 가정 (위 TODO 참조)
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T_cam_target = np.eye(4)
    T_cam_target[:3, :3] = R
    T_cam_target[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T_cam_target


def solve_hand_eye(
    T_base_flange_list: list[np.ndarray],
    T_cam_target_list: list[np.ndarray],
    method: str = "park",
) -> np.ndarray:
    """포즈 쌍 목록 → ★T_flange_cam★ (4,4) — cv2.calibrateHandEye 표준 eye-in-hand 규약.

    규약(cv2 4.11.0 설치본 docstring으로 검증):
      입력 R/t_gripper2base = T_base_gripper(로봇 FK)의 회전/병진 — 'gripper' 자리에
        ★flange★ 포즈(T_base_flange)를 넣으므로 출력 프레임도 flange가 된다.
      입력 R/t_target2cam   = T_cam_target(보드 검출)의 회전/병진.
      출력 R/t_cam2gripper  = T_gripper_cam = ★T_flange_cam 직접★ —
        구(eye-to-hand) 설계의 '역변환을 넣고 cam2base로 해석'하는 조작은 불필요(폐기).
    시그니처(검증됨): calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam,
      t_target2cam[, R_cam2gripper[, t_cam2gripper[, method]]]) -> R_cam2gripper, t_cam2gripper
    method 기본 park = v6 실측 도구의 초기해와 동일 (교차 확인은 --method).
    ⚠️ TODO(확인 필요): canonical hash lock의 opencv-python==4.11.0.86 배포 PC 설치본에서
      동일 시그니처인지 확인한다.
    """
    import cv2  # 지연 import

    if len(T_base_flange_list) != len(T_cam_target_list):
        raise ValueError(
            f"[calib] 포즈 쌍 길이 불일치: {len(T_base_flange_list)} vs {len(T_cam_target_list)}"
        )
    if len(T_base_flange_list) < 3:
        # cv2 4.11.0 docstring: 최소 3포즈(비평행 회전축 모션 2개) — 실전은 훨씬 많이.
        raise ValueError("[calib] 포즈 3개 미만 — hand-eye solve 불가 (비평행 회전축 모션 ≥2 필요)")
    if method not in HAND_EYE_METHODS:
        raise ValueError(f"[calib] 알 수 없는 method: {method!r} (선택지: {sorted(HAND_EYE_METHODS)})")

    R_gripper2base = [np.asarray(T, dtype=np.float64)[:3, :3] for T in T_base_flange_list]
    t_gripper2base = [np.asarray(T, dtype=np.float64)[:3, 3] for T in T_base_flange_list]
    R_target2cam = [np.asarray(T, dtype=np.float64)[:3, :3] for T in T_cam_target_list]
    t_target2cam = [np.asarray(T, dtype=np.float64)[:3, 3] for T in T_cam_target_list]

    R_cam2flange, t_cam2flange = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=getattr(cv2, HAND_EYE_METHODS[method]),
    )
    T_flange_cam = np.eye(4)
    T_flange_cam[:3, :3] = np.asarray(R_cam2flange, dtype=np.float64)
    T_flange_cam[:3, 3] = np.asarray(t_cam2flange, dtype=np.float64).reshape(3)
    return T_flange_cam


# =====================================================================
# 수집 루프
# =====================================================================


def collect_samples(
    robot, camera, poses: list[list[float]], log_dir: Path, T_flange_tcp: np.ndarray
):
    """포즈 세트 순회: move_to_joints(블로킹) → 정지 대기 → 촬영·검출 → 쌍 축적.

    로봇 포즈는 RobotState.tcp_pose에서 T_flange_tcp를 역적용해 복원한 ★flange 포즈★를
    쓴다 (flange_mat_from_tcp_pose docstring — eye-in-hand에서 출력이 T_flange_cam이
    되기 위한 필수 조건). samples.csv에는 관절각·flange 포즈(위치+quat wxyz)·검출 여부를
    남긴다 (실패 포즈 특정용 원장).
    """
    rows = []
    T_bf_list: list[np.ndarray] = []
    T_ct_list: list[np.ndarray] = []
    for i, q in enumerate(poses):
        print(f"[calib] 포즈 {i + 1}/{len(poses)} 이동 중...")
        robot.move_to_joints(np.asarray(q, dtype=np.float64), speed_scale=0.1)
        time.sleep(SETTLE_S)  # 손목 카메라 = 진동에 민감 — 감쇠 대기 후 촬영
        state = robot.get_state()
        T_bf = flange_mat_from_tcp_pose(state.tcp_pose, T_flange_tcp)
        rgb = grab_color_image(camera)
        # get_intrinsics(): enable_rgb=True → ★color 스트림 intrinsics★ —
        #   검출 이미지·hand_eye(COLOR optical)와 같은 기준 (real/real_camera.py 계약).
        T_ct = detect_board_pose(rgb, camera.get_intrinsics())
        detected = T_ct is not None
        if detected:
            T_bf_list.append(T_bf)
            T_ct_list.append(T_ct)
        else:
            print(f"[calib] ⚠️ 포즈 {i + 1}: 보드 미검출 — 샘플 제외 (조명/거리/시야 점검)")
        flange7_pos = T_bf[:3, 3]
        rows.append(
            {
                "pose_idx": i,
                **{f"q{j}": float(q[j]) for j in range(6)},
                **{f"flange_{ax}": float(flange7_pos[k]) for k, ax in enumerate("xyz")},
                **{f"tcp{k}": float(state.tcp_pose[k]) for k in range(7)},
                "detected": int(detected),
            }
        )
    with open(log_dir / "samples.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return T_bf_list, T_ct_list


# =====================================================================
# 메인
# =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hand-Eye(eye-in-hand, T_flange_cam) 실측 캘리브레이션 [사람][배포PC]"
    )
    parser.add_argument("--dry-run", action="store_true", help="포즈 목록 점검만 (이동/촬영 없음)")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--method", choices=sorted(HAND_EYE_METHODS), default="park",
                        help="cv2 hand-eye method (기본 park = v6 도구와 동일)")
    parser.add_argument("--yes", action="store_true",
                        help="저장 확인 프롬프트 생략 (sanity 게이트는 생략 불가)")
    parser.add_argument("--real-config", required=True, type=Path,
                        help="로컬 실기 설정 YAML")
    args = parser.parse_args()
    if not args.dry_run:
        require_motion_authorization("scripted_pose")   # 사전 정의 자세 저속 순회
    real_yaml, _ = load_real_config(args.real_config)

    hand_eye_path = configs_dir() / "calib" / "hand_eye.yaml"
    tcp_offset_path = configs_dir() / "calib" / "tcp_offset.yaml"
    log_dir = HERE / "logs" / f"handeye_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not CALIB_POSES_RAD:
        print("[calib] ⚠️ CALIB_POSES_RAD가 비어 있음 — 실기 티칭 후 상수를 채울 것. 중단.")
        return 2
    print(f"[calib] eye-in-hand: 보드는 작업공간 고정, 카메라는 손목 장착 (산출=T_flange_cam)")
    print(f"[calib] 포즈 {len(CALIB_POSES_RAD)}개, ChArUco {BOARD_SQUARES_X}x{BOARD_SQUARES_Y} "
          f"@ {BOARD_SQUARE_M * 1000:.1f}/{BOARD_MARKER_M * 1000:.1f}mm "
          f"({BOARD_ARUCO_DICT}, legacy={BOARD_LEGACY_PATTERN}), method={args.method}")
    if args.dry_run:
        for i, q in enumerate(CALIB_POSES_RAD):
            print(f"[calib]   #{i}: {np.round(q, 4).tolist()}")
        print("[calib] --dry-run 종료 (이동 없음)")
        return 0

    # --- 실기 연결 (배포 PC 전용 — real/ 번들 필요) ---
    try:
        from real.real_camera import RealCamera
        from real.real_robot import RealRobot
    except ImportError as e:
        print(f"[calib] ⚠️ real/ 모듈 import 실패 — 배포 번들에서 실행 중인지 확인: {e}")
        return 2

    # 생성자 계약(real/ 어댑터 확정, Phase0 테스트와 동일 호출 형태):
    #   RealRobot(real_cfg dict, tcp_offset.yaml 경로) / RealCamera(camera 섹션 dict, enable_rgb)
    # ★캘리브는 enable_rgb=True 고정★ — 보드 검출에 color 프레임이 정책 계약
    #   (PreprocessCfg.use_rgb)과 무관하게 항상 필요하고, get_intrinsics()가
    #   ★color 스트림 intrinsics★를 반환해야 solvePnP·hand_eye(COLOR optical 기준)와
    #   기준이 일치한다. (배포 루프의 enable_rgb는 use_rgb 단일 소스 —
    #   Phase4/_deploy_common.build_real_camera 참조.)
    robot = RealRobot(real_yaml, tcp_offset_path, motion_scope="scripted_pose")
    camera = None
    try:
        # Camera construction and every later startup operation belong to the
        # same fail-safe scope as robot motion.  If USB/config initialization
        # fails after the RTDE control socket opens, latch and close the robot.
        camera = RealCamera(real_yaml["camera"], enable_rgb=True)
        # flange 복원용 — ★RealRobot에 넘긴 것과 같은 파일★이어야 대수적 복원이 성립한다.
        T_flange_tcp = load_tcp_offset(tcp_offset_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        camera.start()
        T_bf_list, T_ct_list = collect_samples(
            robot, camera, CALIB_POSES_RAD, log_dir, T_flange_tcp
        )
    except BaseException:
        robot.emergency_stop()
        raise
    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        try:
            robot.close()
        except Exception:
            pass

    n = len(T_bf_list)
    print(f"[calib] 유효 샘플 {n}/{len(CALIB_POSES_RAD)} (원장: {log_dir / 'samples.csv'})")
    if n < args.min_samples:
        print(f"[calib] ⚠️ 유효 샘플 부족 (< {args.min_samples}) — 포즈 추가/조명 개선 후 재시도")
        return 1

    T_flange_cam = solve_hand_eye(T_bf_list, T_ct_list, method=args.method)

    # --- AX=XB 정합 잔차 (meta.residual_m 기록용 — 최종 판정은 게이트 3 Calib_Verify) ---
    res = axxb_residual(T_bf_list, T_ct_list, T_flange_cam)
    print(f"[calib] AX=XB 잔차 ({res['n_pairs']}쌍): "
          f"회전 med {res['rot_med_deg']:.3f}° / max {res['rot_max_deg']:.3f}°, "
          f"병진 med {res['trans_med_m'] * 1000:.2f}mm / max {res['trans_max_m'] * 1000:.2f}mm")

    # --- ★sanity 게이트★: 프레임 혼동 조기 검출 — 실패 시 저장 거부 (우회 금지) ---
    ok, msg = sanity_check_flange_cam(T_flange_cam)
    print(msg)
    if not ok:
        return 1

    # --- 기존값(v7 실측) 대비 차이 출력 → 저장 확인 요구 ---
    try:
        T_old = load_hand_eye(hand_eye_path)
        dpos_m = float(np.linalg.norm(T_flange_cam[:3, 3] - T_old[:3, 3]))
        drot_deg = rot_angle_deg(T_flange_cam[:3, :3] @ T_old[:3, :3].T)
        print(f"[calib] 기존 T_flange_cam 대비: Δpos={dpos_m * 100:.2f}cm, Δrot={drot_deg:.2f}°")
        print(f"[calib]   신규 t=[{', '.join(f'{v * 1000:+.1f}' for v in T_flange_cam[:3, 3])}]mm / "
              f"기존 t=[{', '.join(f'{v * 1000:+.1f}' for v in T_old[:3, 3])}]mm")
        print("[calib]   (참고: dr.yaml camera_pose ±1cm/±1° 초과면 재학습 판정 — 게이트 3에서 확정)")
    except Exception as e:
        print(f"[calib] ⚠️ 기존 hand_eye.yaml 비교 불가(신규 생성?): {e}")

    if not args.yes:
        try:
            answer = input("[calib] 위 값으로 hand_eye.yaml을 갱신할까요? ('y' 입력 시 저장): ")
        except (EOFError, KeyboardInterrupt):
            print("\n[calib] 입력 중단 — 저장하지 않음")
            return 2
        if answer.strip().lower() != "y":
            print("[calib] 저장 취소 (yaml 무변경)")
            return 2

    # --- 저장 (§4-5: 생성물이지만 반드시 커밋 — meta 필수 필드 채움) ---
    meta = {
        "calibrated": True,
        "mount": "eye_in_hand",  # ★2026-07-14 사용자 확정 계약 키★
        "method": f"opencv_calibrateHandEye_{args.method}",
        "date": _dt.date.today().isoformat(),
        # residual_m = AX=XB 병진 정합 잔차 median [m] (재투영 RMS가 아님 — v6/v7의
        # residual_m과 산출 방식이 다르므로 커밋 메시지에 지표명을 함께 기재할 것).
        # 게이트 3(Calib_Verify --write-residual)이 역투영 오차로 최종 갱신한다.
        "residual_m": round(res["trans_med_m"], 6),
        "residual_rot_deg": round(res["rot_med_deg"], 4),
        "n_samples": n,
    }
    save_hand_eye(hand_eye_path, T_flange_cam, meta)
    print(f"[calib] ★갱신 완료★ {hand_eye_path}")
    print("[calib] ⚠️ save_hand_eye는 yaml 주석(v1~v7 이력)을 보존하지 않는다 — "
          "이력은 git diff/과거 커밋으로 확인")
    print("[calib] ★§4-5 커밋 의무★: 이 파일은 sim/real 공유 단일 소스 — 학습 PC로 회수 후 "
          f"반드시 커밋할 것 (커밋 메시지에 잔차 기재: AX=XB trans med {res['trans_med_m'] * 1000:.2f}mm)")
    print("[calib] 다음 게이트: Calib_TcpOffset.py → Calib_Verify.py (재학습 판정 포함)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
