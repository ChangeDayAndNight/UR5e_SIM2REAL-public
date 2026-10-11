#!/usr/bin/env python
"""Phase 3 게이트 3 — 캘리브레이션 검증 (★합격 게이트★). Isaac 불필요 — 순수 numpy.

★eye-in-hand 계약(2026-07-14 사용자 확정)★: hand_eye.yaml의 키 = ★T_flange_cam★
  (flange→camera COLOR optical). 카메라는 손목(플랜지 브래킷) 장착 — base 기준
  카메라 포즈는 관측 시점의 로봇 포즈에 의존한다:
    ★T_base_cam(관측 시점) = T_base_flange(관측 포즈) @ T_flange_cam★
  따라서 ① 역투영 검증은 '관측 취득 시 로봇이 정지해 있던 flange 포즈'를 함께 받아야
  한다 (--flange-pose6 — 모든 관측점은 ★같은 정지 포즈★에서 취득할 것).

목적: 게이트 1·2의 결과를 독립 관측으로 교차 검증한다.
  ① 정확도: 기지(known) 점을 카메라로 관측 → 픽셀+depth 역투영 → T_base_flange @
     T_flange_cam으로 base 좌표 복원 → 실좌표와 오차 → CSV 리포트 +
     ★합격 판정: 오차 median ≤ 1cm, p95 ≤ 2cm★
  ② ★재학습 판정(요소4↔요소2 연결)★: 실측 T_flange_cam이 공칭값 대비
     configs/dr.yaml camera_pose 섭동 범위(±pos_perturb_m/±rot_perturb_deg)를
     벗어나면 학습된 정책의 DR 커버리지 밖 → '재학습 필요' 출력.
     (마운트 extrinsic끼리의 비교이므로 로봇 포즈와 무관 — flange 프레임에서 직접 비교)
실행: (관측 취득 [사람][배포PC], 분석은 아무 PC)
  CSV 모드(완전 구현): python deployment\\Calib_Verify.py --points-csv <관측.csv> --intr-json <intr.json>
                         --flange-pose6 x y z rx ry rz
  라이브 모드(TODO)  : python deployment\\Calib_Verify.py --live
산출: deployment/logs/calib_verify_<시각>/points.csv + 콘솔 판정
      (--write-residual 시 종합 합격한 configs/calib/hand_eye.yaml에 post-correction
       median/p95 승인 필드 갱신)

입력 CSV 형식 (헤더 필수): u,v,depth_m,x_base,y_base,z_base[,label]
  u,v      : 관측 픽셀 (원본 해상도 기준 — intrinsics와 같은 프레임)
  depth_m  : 해당 픽셀 depth [m] (distance_to_image_plane, 카메라 z)
  x/y/z_base: 그 점의 base frame 실좌표 [m]
    (권장 취득법: 로봇 TCP로 마커 중심을 터치 → get_state().tcp_pose[:3]이 실좌표.
     TCP offset이 게이트 2에서 실측돼 있어야 이 실좌표가 신뢰 가능 — 게이트 순서의 이유.
     ⚠️ 터치 후에는 반드시 '관측 포즈'로 복귀한 뒤 촬영 — 모든 u,v,depth는 같은 포즈에서)
--flange-pose6: 관측(촬영) 시점의 ★flange★ 포즈 [x,y,z,rx,ry,rz] (m, rotation-vector,
  base frame) — 컨트롤러 TCP=0일 때 펜던트/getActualTCPPose가 주는 그 형식.
  (real_robot의 tcp_pose(offset 적용 후)가 아니라 flange 포즈다 — 혼동 주의)
intr JSON 형식: {"fx":..., "fy":..., "cx":..., "cy":...}  (원본 해상도 기준 — enable_rgb=True
  경로에서는 ★color 스트림★ intrinsics = RealCamera.get_intrinsics() 반환과 같은 기준)

카메라 규약: OpenCV/ROS optical (전방 +Z, 우 +X, 하 +Y) — depth = 카메라 z.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
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

from common.calibration import camera_to_base, load_hand_eye, pose7_to_mat, save_hand_eye  # noqa: E402
from common.config_io import configs_dir, load_yaml  # noqa: E402
from deployment.contracts import (  # noqa: E402
    HAND_EYE_MAX_MEDIAN_ERROR_M,
    HAND_EYE_MAX_P95_ERROR_M,
    load_real_config,
)
from real.real_robot import ur_pose_to_pose7  # noqa: E402  (순수 수학 — SDK 불필요)

DEFAULT_GATE_CM = HAND_EYE_MAX_MEDIAN_ERROR_M * 100.0
P95_GATE_CM = HAND_EYE_MAX_P95_ERROR_M * 100.0

# 학습 시점 공칭 ★T_flange_cam★ — configs/calib/hand_eye.yaml의 학습 예정값 사본
# (★v7, 2026-07-16: v6 실기 hand-eye 실측 회전 + 링 동심 병진 보정★).
# 실측 캘리브레이션이 yaml을 덮어쓰므로 비교 기준을 여기 보존한다.
# (구 eye-to-hand 공칭 T_base_cam 상수는 2026-07-17 eye-in-hand 재작성으로 폐기.)
# ⚠️ 정석은 Export_Policy가 학습 시점 hand_eye.yaml을 exported/에 스냅샷으로 남기고
#   --nominal-yaml로 그것을 넘기는 것 (TODO: export 파이프라인과 조율).
#   이 상수는 그 스냅샷이 없을 때의 fallback — 학습 직전 값과 일치해야 의미가 있다.
NOMINAL_T_FLANGE_CAM = np.array(
    [
        [0.9995530733, -0.0251206647, 0.0162051171, -0.0011652429],
        [0.0286999091, 0.9580782886, -0.2850654454, -0.0698647569],
        [-0.0083647373, 0.2854031274, 0.9583710586, 0.0184224864],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


# =====================================================================
# ★순수 수학 — 완전 구현★
# =====================================================================


def unproject_pixel(
    u: float, v: float, depth_m: float, fx: float, fy: float, cx: float, cy: float
) -> np.ndarray:
    """픽셀(u,v)+depth(카메라 z) → 카메라 프레임 3D 점 (OpenCV/ROS optical 규약)."""
    if depth_m <= 0.0 or not np.isfinite(depth_m):
        raise ValueError(f"유효하지 않은 depth: {depth_m}")
    z = float(depth_m)
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])


def rotation_angle_deg(R: np.ndarray) -> float:
    """회전행렬의 회전각 크기 [deg]."""
    c = (float(np.trace(np.asarray(R))) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def compare_extrinsics(T_meas: np.ndarray, T_nom: np.ndarray) -> tuple[float, float]:
    """실측 vs 공칭 T_flange_cam(마운트 extrinsic) → (Δpos [m], Δrot [deg])."""
    T_meas = np.asarray(T_meas, dtype=np.float64)
    T_nom = np.asarray(T_nom, dtype=np.float64)
    dpos = float(np.linalg.norm(T_meas[:3, 3] - T_nom[:3, 3]))
    drot = rotation_angle_deg(T_meas[:3, :3] @ T_nom[:3, :3].T)
    return dpos, drot


def evaluate_points(records: list[dict], intr: dict, T_base_cam: np.ndarray) -> list[dict]:
    """관측 레코드 → 점별 (추정 base 좌표, 오차) 행 목록.

    records[i] 필수 키: u, v, depth_m, x_base, y_base, z_base (+선택 label)
    T_base_cam: ★관측 시점★ base←camera 변환 — eye-in-hand에서는 호출측이
      T_base_flange(관측 포즈) @ T_flange_cam으로 합성해 넘긴다 (파일에 저장된
      T_flange_cam을 그대로 넣으면 안 된다 — 프레임이 다르다).
    """
    rows = []
    for i, r in enumerate(records):
        p_cam = unproject_pixel(
            float(r["u"]), float(r["v"]), float(r["depth_m"]),
            float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"]),
        )
        p_est = camera_to_base(p_cam, T_base_cam)
        p_gt = np.array([float(r["x_base"]), float(r["y_base"]), float(r["z_base"])])
        err_m = float(np.linalg.norm(p_est - p_gt))
        rows.append(
            {
                "label": r.get("label", f"pt{i}"),
                "u": float(r["u"]),
                "v": float(r["v"]),
                "depth_m": float(r["depth_m"]),
                "est_x": round(float(p_est[0]), 5),
                "est_y": round(float(p_est[1]), 5),
                "est_z": round(float(p_est[2]), 5),
                "gt_x": float(r["x_base"]),
                "gt_y": float(r["y_base"]),
                "gt_z": float(r["z_base"]),
                "err_cm": round(err_m * 100.0, 3),
            }
        )
    return rows


# =====================================================================
# 관측 취득 (라이브 모드 — 스켈레톤 TODO)
# =====================================================================


def acquire_live_points() -> tuple[list[dict], dict]:
    """[사람][배포PC] 라이브 관측 취득 → (records, intr).

    ⚠️ TODO(확인 필요) — 미구현 스켈레톤. 확정할 것:
      - 마커 검출: 기지점 마커(ArUco 권장)의 픽셀 중심 검출 파이프라인
        (Calib_HandEye.detect_board_pose와 동일한 OpenCV 버전 이슈)
      - RGB/depth 정렬: RealCamera(enable_rgb=True)의 rs.align(depth→color) 경로 —
        pyrealsense2 API 미검증 (real/real_camera.py TODO와 동일 항목)
      - ★관측 포즈 기록(eye-in-hand)★: 촬영 시점의 flange 포즈를 함께 기록해야
        T_base_cam = T_base_flange @ T_flange_cam 합성이 가능 (--flange-pose6 자동화)
      - 실좌표 취득: 로봇 TCP 터치(get_state().tcp_pose[:3]) 절차 인터랙션
      - intr: RealCamera.get_intrinsics() → dict 변환 (enable_rgb=True → color 기준)
    당분간은 수동 취득 → CSV 작성 → --points-csv 경로를 쓸 것 (완전 구현됨).
    """
    raise NotImplementedError(
        "[verify] ⚠️ 라이브 취득 미구현 — --points-csv/--intr-json 경로를 사용할 것"
    )


# =====================================================================
# 메인 게이트
# =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="캘리브레이션 검증 게이트 (게이트 3)")
    parser.add_argument("--points-csv", type=Path, default=None, help="관측 CSV (형식은 docstring)")
    parser.add_argument("--intr-json", type=Path, default=None, help="intrinsics JSON (fx,fy,cx,cy)")
    parser.add_argument("--live", action="store_true", help="라이브 취득 모드 (TODO 미구현)")
    parser.add_argument("--gate-cm", type=float, default=DEFAULT_GATE_CM)
    parser.add_argument("--hand-eye-yaml", type=Path, default=None,
                        help="실측 T_flange_cam yaml (기본: configs/calib/hand_eye.yaml)")
    parser.add_argument("--nominal-yaml", type=Path, default=None,
                        help="공칭(학습 시점) T_flange_cam yaml — 미지정 시 내장 상수 사용")
    parser.add_argument("--flange-pose6", type=float, nargs=6, default=None,
                        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
                        help="관측(촬영) 시점 flange 포즈 [m, rotation-vector rad] — "
                             "eye-in-hand ① 역투영에 필수 (docstring 참조)")
    parser.add_argument("--write-residual", action="store_true",
                        help="종합 합격 시 hand_eye.yaml에 post-correction median/p95 승인 기록")
    parser.add_argument("--real-config", required=True, type=Path,
                        help="로컬 실기 설정 YAML")
    args = parser.parse_args()
    if not 0.0 < args.gate_cm <= DEFAULT_GATE_CM:
        parser.error(f"--gate-cm must be in (0, {DEFAULT_GATE_CM:g}] to preserve release readiness")
    load_real_config(args.real_config)

    hand_eye_path = args.hand_eye_yaml or (configs_dir() / "calib" / "hand_eye.yaml")
    T_flange_cam = load_hand_eye(hand_eye_path)  # ★eye-in-hand: flange→camera★
    hand_eye_raw = load_yaml(hand_eye_path)
    meta = dict(hand_eye_raw.get("meta", {}))
    if not meta.get("calibrated", False):
        print("[verify] ⚠️ hand_eye.yaml meta.calibrated=false — 게이트 1 미완료 상태의 공칭값을 "
              "검증하게 됨 (순서 위반 의심)")

    # --- 관측 취득 ---
    if args.live:
        records, intr = acquire_live_points()  # NotImplementedError (TODO)
    else:
        if args.points_csv is None or args.intr_json is None:
            parser.error("--points-csv와 --intr-json이 필요 (또는 --live)")
        if args.flange_pose6 is None:
            parser.error("--flange-pose6이 필요 — eye-in-hand에서는 관측 시점 flange 포즈 "
                         "없이 역투영(base 복원)이 성립하지 않음 (docstring 참조)")
        with open(args.points_csv, newline="", encoding="utf-8-sig") as f:
            records = list(csv.DictReader(f))
        with open(args.intr_json, encoding="utf-8") as f:
            intr = json.load(f)
        if not records:
            print(f"[verify] ⚠️ 관측 CSV가 비어 있음: {args.points_csv}")
            return 2

    # --- ① 역투영 정확도 ---
    # ★eye-in-hand 합성★: T_base_cam(관측 시점) = T_base_flange(관측 포즈) @ T_flange_cam
    #   (--flange-pose6 = UR [x,y,z,rx,ry,rz] → pose7 → 행렬. 모든 관측점이 같은
    #    정지 포즈에서 취득됐다는 전제 — docstring의 취득 절차 참조)
    T_base_flange = pose7_to_mat(ur_pose_to_pose7(np.asarray(args.flange_pose6)))
    T_base_cam = T_base_flange @ T_flange_cam
    rows = evaluate_points(records, intr, T_base_cam)
    errs = np.array([r["err_cm"] for r in rows])
    med, p95, mx = (
        float(np.median(errs)),
        float(np.percentile(errs, 95)),
        float(errs.max()),
    )

    out_dir = HERE / "logs" / f"calib_verify_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "points.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    acc_ok = med <= args.gate_cm and p95 <= P95_GATE_CM
    print(f"[verify] 점 {len(rows)}개 — 오차[cm]: median {med:.2f} · p95 {p95:.2f} · max {mx:.2f}")
    print(f"[verify] ① 정확도 게이트 (median ≤ {args.gate_cm:.1f}cm, "
          f"p95 ≤ {P95_GATE_CM:.1f}cm): "
          f"{'★통과★' if acc_ok else '⚠️ 미달'}")

    # --- ② 재학습 판정: 공칭 대비 ★T_flange_cam★(마운트 extrinsic) 차이 vs dr.yaml 범위 ---
    #   (로봇 포즈와 무관한 flange 프레임 비교 — dr.yaml camera_pose 섭동도 마운트 기준)
    if args.nominal_yaml is not None:
        T_nom = load_hand_eye(args.nominal_yaml)
        nom_src = str(args.nominal_yaml)
    else:
        T_nom = NOMINAL_T_FLANGE_CAM
        nom_src = "내장 상수 v7 (⚠️ 학습 시점 스냅샷이 있으면 --nominal-yaml 사용 권장)"
    dr = load_yaml(configs_dir() / "dr.yaml")["camera_pose"]
    pos_tol_m, rot_tol_deg = float(dr["pos_perturb_m"]), float(dr["rot_perturb_deg"])
    dpos_m, drot_deg = compare_extrinsics(T_flange_cam, T_nom)
    dr_ok = dpos_m <= pos_tol_m and drot_deg <= rot_tol_deg
    print(f"[verify] 공칭 기준: {nom_src}")
    print(f"[verify] 실측−공칭: Δpos {dpos_m * 100:.2f}cm (허용 ±{pos_tol_m * 100:.1f}cm) · "
          f"Δrot {drot_deg:.2f}° (허용 ±{rot_tol_deg:.1f}°)")
    if dr_ok:
        print("[verify] ② DR 커버리지: ★범위 이내★ — 학습된 정책이 실측 카메라 pose를 커버함")
    else:
        print("[verify] ② ★재학습 필요★ — 실측 T_flange_cam이 dr.yaml camera_pose 섭동 범위 밖. "
              "조치: (a) hand_eye.yaml 실측값 기준으로 sim 카메라 재배치 후 Phase2 재학습, "
              "또는 (b) dr.yaml 범위 확대 재학습. 이 상태로 배포 진행 금지.")

    verdict = acc_ok and dr_ok

    # --- 승인 기록 (release/readiness와 같은 median 10mm / p95 20mm 계약) ---
    if args.write_residual and verdict:
        median_m = round(med / 100.0, 8)
        p95_m = round(p95 / 100.0, 8)
        meta["residual_m"] = median_m
        meta["median_error_m"] = median_m
        meta["p95_error_m"] = p95_m
        meta["post_correction_verified"] = True
        meta["dr_coverage_verified"] = True
        save_hand_eye(hand_eye_path, T_flange_cam, meta)  # 행렬은 무변경 — meta만 갱신
        print(f"[verify] post-correction median={median_m}m / p95={p95_m}m 승인 기록 → "
              f"{hand_eye_path}")
    elif args.write_residual:
        print("[verify] 종합 게이트 불합격 — hand_eye 승인 필드를 기록하지 않음")

    print(f"[verify] 산출물: {out_dir}")
    print(f"[verify] 게이트 3 종합: {'★통과 — 게이트 4(SimDeploy_Rehearsal)로★' if verdict else '⚠️ 불합격'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
