#!/usr/bin/env python
"""Phase 3 게이트 2 — TCP offset(T_flange_tcp) 실측. [사람][배포PC] — Isaac 불필요.

목적: flange→TCP 변환(현재 OnRobot 2FG7 sim 측정치 z=0.1398m, calibrated:false)을
      '고정점 다자세 터치(4-점법)'로 실측하고 configs/calib/tcp_offset.yaml을 갱신한다.
      RobotState.tcp_pose 계약(TCP offset 적용 후)·workspace AABB(요소5)·place 판정이
      전부 이 값 위에 서 있다.
실행: command-capable calibration is hard-blocked in this receive-only release.
      솔버 자가 검증(로봇 불필요):   python deployment\\Calib_TcpOffset.py --self-test
산출: configs/calib/tcp_offset.yaml 갱신(meta.calibrated=true) + 콘솔 잔차 리포트

수학(★완전 구현 — 순수 numpy★):
  같은 공구 끝점(t, flange 기준)이 같은 고정점(p, base 기준)을 N개의 서로 다른
  자세로 터치하면:  R_i @ t + o_i = p  (T_base_flange_i = [R_i | o_i])
  → [R_i | -I] [t; p]ᵀ = -o_i 를 3N×6으로 쌓아 최소자승 → t, p 동시 추정.
  자세 다양성이 없으면(회전이 거의 같으면) 행렬이 퇴화 — 조건수로 경고한다.

⚠️ 실측 절차 TODO(확정 필요 — 스켈레톤):
  - 터치 방법: freedrive(수동 유도)로 공구 끝을 고정 첨점에 대고 Enter로 포즈 기록.
    ⚠️ TODO(확인 필요): ur_rtde freedrive API — RTDEControlInterface.teachMode()/
    endTeachMode()로 알려져 있으나 미검증 (RealRobot에 래퍼 필요 여부 포함).
  - 회전 성분: 4-점법은 병진만 관측 — 회전은 공칭(identity) 유지.
    TODO: 회전 실측이 필요해지면 기준 자세 정렬법 등 별도 절차 확정.
"""
from __future__ import annotations

import argparse
import datetime as _dt
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

from common.calibration import (  # noqa: E402
    load_tcp_offset,
    pose7_to_mat,
    quat_to_mat,
    save_tcp_offset,
)
from common.config_io import configs_dir  # noqa: E402
from deployment.contracts import load_real_config, require_motion_authorization  # noqa: E402

MIN_TOUCHES = 4          # 최소 터치 수 (미지수 6 → 4자세면 12식, 여유 확보)
COND_WARN = 1e4          # 자세 다양성 부족 경고 임계 (조건수)
DEFAULT_GATE_MM = 2.0    # 잔차 rms 합격 게이트 [mm] — TODO: 실측 후 확정


# =====================================================================
# ★순수 수학 — 완전 구현★
# =====================================================================


def solve_fixed_point_offset(
    T_base_flange_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """고정점 다자세 터치 최소자승 → (t_offset(3,), p_fixed(3,), residuals_m(N,)).

    t_offset: flange 프레임 기준 공구 끝점 병진 (T_flange_tcp의 병진 성분)
    p_fixed : base 프레임 기준 고정점 위치 (부산물 — 검증용)
    residuals_m: 터치별 |R_i t + o_i − p| [m] — rms가 게이트 지표.
    """
    Ts = [np.asarray(T, dtype=np.float64) for T in T_base_flange_list]
    n = len(Ts)
    if n < MIN_TOUCHES:
        raise ValueError(f"터치 수 부족: {n} < {MIN_TOUCHES}")
    for T in Ts:
        if T.shape != (4, 4):
            raise ValueError(f"T_base_flange 형상 오류: {T.shape}")

    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i, T in enumerate(Ts):
        A[3 * i : 3 * i + 3, :3] = T[:3, :3]
        A[3 * i : 3 * i + 3, 3:] = -np.eye(3)
        b[3 * i : 3 * i + 3] = -T[:3, 3]

    cond = float(np.linalg.cond(A))
    if cond > COND_WARN:
        print(f"[calib] ⚠️ 자세 다양성 부족 의심 (cond={cond:.1e}) — 회전을 더 크게 바꿔 재터치 권장")

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    t_offset, p_fixed = sol[:3], sol[3:]
    residuals = np.array(
        [np.linalg.norm(T[:3, :3] @ t_offset + T[:3, 3] - p_fixed) for T in Ts]
    )
    return t_offset, p_fixed, residuals


def flange_from_tcp_pose(tcp_pose7: np.ndarray, T_flange_tcp_nominal: np.ndarray) -> np.ndarray:
    """RobotState.tcp_pose(공칭 offset 적용 후) → T_base_flange (4,4).

    솔버 입력은 'flange' 포즈여야 한다. RobotState 계약상 tcp_pose에는 공칭
    T_flange_tcp가 이미 적용돼 있으므로 정확히 역산해 벗겨낸다 (순수 수학).
    """
    return pose7_to_mat(tcp_pose7) @ np.linalg.inv(np.asarray(T_flange_tcp_nominal))


def self_test(seed: int = 0) -> int:
    """합성 데이터로 솔버 무결성 검증 (로봇 불필요 — 어느 PC에서나)."""
    rng = np.random.default_rng(seed)
    t_true = np.array([0.003, -0.002, 0.152])
    p_true = np.array([0.45, 0.10, 0.02])
    Ts = []
    for _ in range(8):
        q = rng.normal(size=4)
        R = quat_to_mat(q / np.linalg.norm(q))
        o = p_true - R @ t_true  # 정확히 터치 조건을 만족하는 flange 원점
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, o
        Ts.append(T)
    t_est, p_est, res = solve_fixed_point_offset(Ts)
    err_t = float(np.linalg.norm(t_est - t_true))
    err_p = float(np.linalg.norm(p_est - p_true))
    ok = err_t < 1e-9 and err_p < 1e-9 and float(res.max()) < 1e-9
    print(f"[calib] self-test: err_t={err_t:.2e}m err_p={err_p:.2e}m "
          f"res_max={res.max():.2e}m → {'★통과★' if ok else '⚠️ 실패'}")
    return 0 if ok else 1


# =====================================================================
# 실측 절차 (스켈레톤 — 터치 절차 TODO 확정 후 완성)
# =====================================================================


def acquire_touch_poses(robot, num_touches: int) -> list[np.ndarray]:
    """[사람] 고정 첨점 터치 반복 → T_base_flange 목록.

    절차(안 — TODO 확정):
      1) 테이블에 고정 첨점(뾰족한 기준 핀)을 단단히 설치
      2) freedrive 진입 (⚠️ TODO(확인 필요): ur_rtde teachMode() 존재/시그니처 미검증,
         RealRobot 인터페이스에는 freedrive가 없음 — 계약 밖 보조 메서드 조율 필요)
      3) 공구 끝(그리퍼 닫힌 상태의 파지 중심 — TODO: 기준점 정의 확정)을
         첨점에 정확히 접촉 → Enter → 포즈 기록
      4) '자세를 크게 바꿔' 2~3을 num_touches회 반복 (병진만 바꾸면 퇴화)
    """
    T_nominal = load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")
    poses: list[np.ndarray] = []
    print(f"[calib] 고정점 터치 {num_touches}회 — 매회 '자세'를 크게 바꿀 것")
    for i in range(num_touches):
        # ⚠️ TODO: freedrive 진입/해제 래핑 (위 docstring 2) 참조)
        input(f"[calib] 터치 {i + 1}/{num_touches}: 공구 끝을 첨점에 접촉 후 Enter...")
        state = robot.get_state()
        poses.append(flange_from_tcp_pose(state.tcp_pose, T_nominal))
        print(f"[calib]   기록: tcp={np.round(state.tcp_pose[:3], 4).tolist()}")
    return poses


def main() -> int:
    parser = argparse.ArgumentParser(description="TCP offset 실측 (4-점법) [사람][배포PC]")
    parser.add_argument("--self-test", action="store_true", help="솔버 자가 검증 (로봇 불필요)")
    parser.add_argument("--touches", type=int, default=6, help="터치 횟수 (≥4, 다양한 자세)")
    parser.add_argument("--gate-mm", type=float, default=DEFAULT_GATE_MM)
    parser.add_argument("--real-config", type=Path,
                        help="로컬 실기 설정 YAML")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.real_config is None:
        parser.error("--real-config is required for hardware calibration")
    require_motion_authorization("freedrive")   # 사람이 손으로 유도 — 명령 모션 없음
    real_yaml, _ = load_real_config(args.real_config)

    tcp_path = configs_dir() / "calib" / "tcp_offset.yaml"

    try:
        from real.real_robot import RealRobot
    except ImportError as e:
        print(f"[calib] ⚠️ real/ 모듈 import 실패 — 배포 번들에서 실행 중인지 확인: {e}")
        return 2

    # 생성자 계약(real/real_robot.py 확정, deployment/rtde_connect_test.py와 동일 호출 형태):
    #   RealRobot(real_cfg dict, tcp_offset.yaml 경로) — 여기서 넘기는 공칭 tcp_offset은
    #   flange_from_tcp_pose가 정확히 역산해 벗겨내므로 실측 전 공칭치라도 무방하다.
    robot = RealRobot(real_yaml, tcp_path, motion_scope="freedrive")
    try:
        poses = acquire_touch_poses(robot, args.touches)
    except BaseException:
        robot.emergency_stop()
        raise
    finally:
        try:
            robot.close()
        except Exception:
            pass

    t_offset, p_fixed, residuals = solve_fixed_point_offset(poses)
    rms_mm = float(np.sqrt(np.mean(residuals**2))) * 1000.0
    print(f"[calib] t_offset(flange 기준) = {np.round(t_offset, 5).tolist()} m")
    print(f"[calib] 고정점(base 기준)    = {np.round(p_fixed, 5).tolist()} m")
    print(f"[calib] 잔차 rms = {rms_mm:.3f} mm (max {residuals.max() * 1000:.3f} mm)")

    gate_ok = rms_mm <= args.gate_mm
    print(f"[calib] 게이트 (rms ≤ {args.gate_mm}mm): {'★통과★' if gate_ok else '⚠️ 미달'}")
    if not gate_ok:
        print("[calib] 저장 생략 — 터치 정밀도/자세 다양성 개선 후 재시도")
        return 1

    # 회전은 공칭 유지(4-점법은 병진만 관측) — 병진만 실측치로 교체
    T_new = load_tcp_offset(tcp_path).copy()
    T_new[:3, 3] = t_offset
    meta = {
        "calibrated": True,
        "gripper_model": "onrobot_2fg7",
        "method": "fixed_point_multi_orientation",
        "date": _dt.date.today().isoformat(),
        "residual_m": round(rms_mm / 1000.0, 6),
    }
    save_tcp_offset(tcp_path, T_new, meta)
    print(f"[calib] ★갱신 완료★ {tcp_path}")
    print("[calib] ⚠️ 이 파일은 sim/real 공유 — 이 PC(학습측)로도 반드시 회수·동기화할 것")
    print("[calib] 다음 게이트: Calib_Verify.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
