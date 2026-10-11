#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + 경로.
# ─────────────────────────────────────────────────────────────────────────────
"""OnRobot 2FG7 그리퍼 구동 테스트 — ★핑거를 실제로 여닫아 방향을 확인★ (isaac51, [사람]).

목적: convert_2fg7_usd.py 는 '정적 USD'만 만든다(스스로 안 움직임). 이 스크립트는
      변환된 USD를 Articulation으로 불러와 finger 관절을 열림↔닫힘으로 천천히 스윕한다.
      → GUI로 보면 두 핑거가 대칭으로 벌어졌다 오므라드는지, 어느 관절값이 '열림'인지
        눈으로 확인할 수 있다(open/close 커맨드 값 확정에 필요).
실행 (학습 PC, isaac51 — ★GUI로 봐야 하므로 --headless 없이★):
  python scripts\\tools\\test_2fg7_gripper.py
  python scripts\\tools\\test_2fg7_gripper.py --usd assets\\onrobot_2fg7\\onrobot_2fg7.usd
산출: 없음(육안 확인용). 콘솔에 관절명 + 매 스텝 커맨드값/실측값 출력.

⚠️ convert_2fg7_usd.py 를 먼저 실행해 USD가 있어야 한다(없으면 경로 에러).
⚠️ isaaclab은 AppLauncher 이후에만 import (아래 순서 고정).
"""
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()



_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# GUI 모드로 실행되므로 omni.sensors의 hdf5 DLL 충돌 회피 — Kit 기동 전 h5py 선로드 (§4-1 ③).
try:
    # ★2026-08-10(F54) h5py 선로드 제거★ — 실측상 ★이 import가 Isaac 센서 DLL을 깨뜨린다★.
    #   h5py 3.16.0(HDF5 2.0.0)이 Isaac 번들 hdf5.dll(1.x)보다 먼저 로드되면
    #   generic_mo_io.dll이 WinError 127로 실패하고 GUI가 access violation으로 죽는다.
    #   (구 h5py에서는 반대로 작동해 워크어라운드였다 — 근거·측정 = Failure_Check F54)
    pass
except Exception as _e:
    print("[test] h5py preimport skipped:", _e)

# ═══════════════════ AppLauncher (isaaclab import은 이 아래에서만) ═══════════════════
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="OnRobot 2FG7 그리퍼 구동 테스트.")
parser.add_argument(
    "--usd",
    default=os.path.join(_REPO_ROOT, "assets", "onrobot_2fg7", "onrobot_2fg7.usd"),
    help="테스트할 그리퍼 USD (기본: assets/onrobot_2fg7/onrobot_2fg7.usd).",
)
parser.add_argument("--period", type=float, default=2.5, help="한 번 여닫는 주기[s].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

# 2FG7 구동 관절 스트로크 (현행 URDF limit): left_finger_joint ∈ [0, 0.020] m.
FINGER_LOWER = 0.0
FINGER_UPPER = 0.020


def _build_gripper_cfg(usd_path: str) -> ArticulationCfg:
    """변환된 2FG7 USD → Articulation cfg. 중력 off로 제자리에 띄워 육안 관찰.

    finger 관절은 위치 제어(ImplicitActuator) — convert 시 심은 드라이브를 덮어쓴다.
    """
    return ArticulationCfg(
        prim_path="/World/Gripper",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            # 중력 off → 고정 없이도 제자리(핑거만 움직임). 테스트 전용.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*finger.*"],  # left/right_finger_joint 모두 포착
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )


def main() -> None:
    if not os.path.isfile(args_cli.usd):
        raise SystemExit(
            f"[test] USD가 없습니다: {args_cli.usd}\n"
            f"       먼저 변환하세요: python scripts\\tools\\convert_2fg7_usd.py --headless"
        )

    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device))
    sim.set_camera_view([0.4, 0.4, 0.7], [0.0, 0.0, 0.55])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    gripper = Articulation(_build_gripper_cfg(args_cli.usd))
    sim.reset()

    # 변환 결과 관절 구성 확인(mimic→독립2관절이면 2개, --keep-mimic면 1개).
    finger_ids, finger_names = gripper.find_joints([".*finger.*"], preserve_order=True)
    print("=" * 70)
    print(f"[test] 그리퍼 관절: {finger_names}  (구동 대상 {len(finger_ids)}개)")
    print(f"[test] 전체 관절수: {gripper.num_joints},  스트로크 [{FINGER_LOWER}, {FINGER_UPPER}] m")
    print("[test] ★핑거가 대칭으로 여닫히는지, 어느 값이 '열림'인지 GUI로 확인하세요★")
    print("=" * 70)

    sim_dt = sim.get_physics_dt()
    step = 0
    while simulation_app.is_running():
        # 0 ↔ 0.020 사인 스윕 (천천히)
        phase = 2.0 * math.pi * (step * sim_dt) / args_cli.period
        target_val = FINGER_LOWER + 0.5 * (FINGER_UPPER - FINGER_LOWER) * (1.0 - math.cos(phase))
        target = torch.full((1, len(finger_ids)), target_val, device=sim.device)
        gripper.set_joint_position_target(target, joint_ids=finger_ids)
        gripper.write_data_to_sim()
        sim.step()
        gripper.update(sim_dt)

        if step % 30 == 0:  # 육안+수치 동시 확인
            meas = gripper.data.joint_pos[0, finger_ids].tolist()
            state = "열림쪽" if target_val > (FINGER_UPPER * 0.5) else "닫힘쪽"
            print(f"[test] cmd={target_val:.4f} ({state})  실측={[round(v,4) for v in meas]}")
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
