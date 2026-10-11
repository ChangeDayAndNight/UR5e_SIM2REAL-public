"""Phase0 — [사람][배포PC] UR5e RTDE 연결 검증 (Isaac 불필요, env_ur_real 전용).

목적: 배포 PC에서 receive-only RealRobot으로 UR5e 상태(q, TCP pose)만 읽는다.
실행 (배포 PC, 프로젝트 루트 UR5e_Sim2Real에서, conda env: env_ur_real):
    python deployment\\rtde_connect_test.py --real-config configs/real.local.yaml
산출: stdout 상태 리포트 + 성공 sentinel "RTDE_CONNECT_OK".

⚠️ 이 스크립트는 isaaclab/isaacsim을 절대 import하지 않는다 (배포 번들 규칙).
이 릴리스에는 motion authorization이 없으며 이 도구는 제어/IO 인터페이스를 열지 않는다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
# Canonical self-contained bundle layout is produced by deployment/build_bundle.py
# and bound by bundle_manifest.json.  deployment/ sits directly under its root.
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from common.config_io import configs_dir
from deployment.contracts import load_real_config


def _fmt(arr, nd: int = 4) -> list[float]:
    return [round(float(v), nd) for v in np.asarray(arr).reshape(-1)]


def _print_state(state, tag: str = "rtde") -> None:
    """RobotState 계약(common/types.py) 필드를 사람이 읽기 좋게 출력."""
    q = np.asarray(state.joint_pos)
    print(f"[{tag}] joint_pos [rad]: {_fmt(q)}")
    print(f"[{tag}] joint_pos [deg]: {_fmt(np.degrees(q), 2)}")
    print(f"[{tag}] joint_vel [rad/s]: {_fmt(state.joint_vel)}")
    print(f"[{tag}] tcp pos [m] (base frame): {_fmt(state.tcp_pose[:3])}")
    print(f"[{tag}] tcp quat (wxyz): {_fmt(state.tcp_pose[3:])}")
    if state.tcp_force is not None:
        print(f"[{tag}] tcp wrench [N,Nm]: {_fmt(state.tcp_force, 2)}")
    else:
        # ⚠️ TODO(확인 필요): RTDEReceiveInterface.getActualTCPForce() 제공 여부/좌표계
        #    — real/real_robot.py가 미지원으로 None을 주면 SafetyGuard 힘 판정은 스킵됨.
        print(f"[{tag}] tcp wrench: None (미지원 — real_robot.py TODO 참조)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase0: UR5e RTDE 연결 검증 (배포 PC)")
    parser.add_argument("--real-config", required=True, type=Path,
                        help="로컬 실기 설정 YAML")
    args = parser.parse_args()

    real_cfg, _ = load_real_config(args.real_config)
    print(f"[rtde] 대상: {real_cfg['robot']['ip']} ({args.real_config})")

    try:
        from real.real_robot import RealRobot
    except ImportError as e:
        print(f"⚠️ [rtde] real/real_robot.py import 실패: {e}")
        print("⚠️ [rtde] canonical bundle/build manifest와 "
              "env_ur_real의 ur_rtde 설치를 확인할 것 (exit 2)")
        raise SystemExit(2)

    # 생성자 계약(real/real_robot.py): RealRobot(real_cfg dict, tcp_offset.yaml 경로)
    # — tcp_offset은 요소4: RobotState.tcp_pose 보정에 즉시 사용된다.
    try:
        robot = RealRobot(
            real_cfg,
            configs_dir() / "calib" / "tcp_offset.yaml",
            receive_only=True,
        )
    except Exception as e:
        print(f"⚠️ [rtde] 연결 실패: {e}")
        print("⚠️ [rtde] 점검: ① 로봇 전원/Remote Control 모드 ② IP/서브넷 "
              "③ 방화벽 ④ 다른 RTDE 클라이언트 점유 ⑤ env_ur_real에 ur_rtde 설치")
        raise SystemExit(1)

    try:
        if not robot.is_ok():
            raise RuntimeError("RTDE connected but robot health is not OK")
        print("[rtde] 연결 성공 — is_ok=True")
        state = robot.get_state()
        _print_state(state)
        print("RTDE_CONNECT_OK", flush=True)

    finally:
        robot.close()
        print("[rtde] 연결 종료")


if __name__ == "__main__":
    main()
