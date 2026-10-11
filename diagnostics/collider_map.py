#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★손가락 콜라이더 벽 위치 실측(Y-sweep)★ — 그리퍼를 닫아 고정하고 큐브를 파지축(Y)으로
   쓸어(sweep) 접촉력이 발생하는 Y offset을 찾아 '실제 콜라이더 벽'이 어디인지 특정한다.

배경: 큐브가 손가락 사이(finger_mid) 정확히 놓였는데(거리 0.0001m) 닫아도 접촉력 ~0, 큐브가
      Z로 자유낙하 → 두 손가락 콜라이더가 큐브를 못 잡음. 구 계보 닫힘 q=0.019에서 pad 안쪽
      면은 ±3.3mm여야 하나, 실제 convex_hull 콜라이더가 어디 있는지 불명. 큐브를 Y로 쓸며
      접촉이 켜지는 지점 = 실제 콜라이더 벽. (기대: |dy|≈큐브반폭+pad면 근처에서 켜짐.)

방법: reset → 손가락 q=0.020(닫힘) write+target 고정 → 매 스텝 큐브를 finger_mid+dy에 핀 고정
      (dy: -0.05→+0.05, 21점) → 몇 스텝 안정 후 좌/우 접촉력 기록.
실행: python diagnostics\\collider_map.py --headless
산출: 콘솔 표 + Phase1/Result/diagnostics/collider_map_<ts>.csv
⚠️ isaaclab import → 프리앰블/AppLauncher 이후에만.
"""
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()



_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    # ★2026-08-10(F54) h5py 선로드 제거★ — 실측상 ★이 import가 Isaac 센서 DLL을 깨뜨린다★.
    #   h5py 3.16.0(HDF5 2.0.0)이 Isaac 번들 hdf5.dll(1.x)보다 먼저 로드되면
    #   generic_mo_io.dll이 WinError 127로 실패하고 GUI가 access violation으로 죽는다.
    #   (구 h5py에서는 반대로 작동해 워크어라운드였다 — 근거·측정 = Failure_Check F54)
    pass
except Exception:
    pass

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="손가락 콜라이더 벽 Y-sweep 실측.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401

from ur5e_binpick.tasks.binpick.mdp.scene import NUM_OBJECTS, OBJECT_SIZE, object_names

_TASK = "Isaac-BinPick-UR5e-Teacher-v0"


def main() -> None:
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env = gym.make(_TASK, cfg=env_cfg)
    uenv = env.unwrapped
    n = uenv.num_envs
    dev = uenv.device
    uenv._bp_pregrasp_frac = 1.0
    env.reset()

    robot = uenv.scene["robot"]
    cube = uenv.scene[object_names(NUM_OBJECTS)[0]]
    fj_ids, _ = robot.find_joints(["left_finger_joint", "right_finger_joint"])
    fbL, _ = robot.find_bodies("left_finger_link")
    fbR, _ = robot.find_bodies("right_finger_link")

    def _f(sname):
        s = uenv.scene.sensors[sname]
        fm = s.data.force_matrix_w
        return torch.full((n,), float("nan"), device=dev) if fm is None else fm.squeeze(1).norm(dim=-1).sum(dim=-1)

    # 손가락을 현행 닫힘(0.020)으로 고정
    with torch.inference_mode():
        fq = torch.full((n, len(fj_ids)), 0.020, device=dev)
        robot.write_joint_state_to_sim(fq, torch.zeros_like(fq), joint_ids=fj_ids)
        robot.set_joint_position_target(fq, joint_ids=fj_ids)
        pl = robot.data.body_state_w[:, fbL[0], :3]
        pr = robot.data.body_state_w[:, fbR[0], :3]
        mid = (pl + pr) / 2.0                          # world finger_mid(env0 기준 sweep)
    print(f"[map] 손가락 닫힘 q=0.020 고정. finger_mid(env0)=({mid[0,0]:.4f},{mid[0,1]:.4f},{mid[0,2]:.4f})")
    print(f"[map] 큐브 반폭={OBJECT_SIZE/2*1000:.0f}mm. Y로 쓸며 접촉 온셋 = 실제 콜라이더 벽.")
    # sweep 축 = 손가락을 잇는 방향(pl→pr) 단위벡터
    axis = (pr - pl)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    print(f"[map] 파지축(pl→pr) 단위벡터 env0=({axis[0,0]:.3f},{axis[0,1]:.3f},{axis[0,2]:.3f})")

    out = os.path.join(_REPO_ROOT, "Phase1", "Result", "diagnostics",
                       f"collider_map_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fcsv = open(out, "w", newline="", encoding="utf-8")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["dy_mm", "fL", "fR"])

    # ★두 축 sweep★: (a) Y(파지축, finger_mid에서) (b) 세계 -Z(아래로 — 패드가 finger_mid보다
    #   14cm 아래에 있다는 가설 검증). 각 sweep에서 접촉 온셋 위치 = 실제 패드 위치.
    down = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n, 3)
    sweeps = [("Y_at_mid", axis, torch.linspace(-0.05, 0.05, 21, device=dev)),
              ("Z_down", down, torch.linspace(0.0, -0.22, 23, device=dev))]
    try:
        for sname, svec, offs in sweeps:
            print(f"\n── sweep [{sname}] ──  (접촉 켜지는 지점 = 실제 패드)")
            print(f"{'d_mm':>7} {'fL':>8} {'fR':>8}")
            for d in offs:
                with torch.inference_mode():
                    target = mid + svec * d
                    pose = torch.zeros(n, 7, device=dev)
                    pose[:, :3] = target
                    pose[:, 3] = 1.0
                    for _ in range(4):
                        cube.write_root_pose_to_sim(pose)
                        cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
                        robot.set_joint_position_target(fq, joint_ids=fj_ids)
                        env.step(torch.zeros(n, 7, device=dev))
                    fL, fR = _f("left_finger_contact")[0].item(), _f("right_finger_contact")[0].item()
                wcsv.writerow([sname, f"{d.item()*1000:.1f}", f"{fL:.3f}", f"{fR:.3f}"])
                bar = "L" * min(20, int(fL)) + "R" * min(20, int(fR))
                print(f"{d.item()*1000:>7.1f} {fL:>8.2f} {fR:>8.2f}  {bar}")
    finally:
        fcsv.close()
        env.close()
    print(f"\n[map] ★산출★ {out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
