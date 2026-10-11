#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★파지 프레임 정밀 측정★ — 실제 finger '콜라이더 pad 중심'을 wrist_3_link 프레임(→tcp_offset.yaml)
   과 env-local world(→_PREGRASP_GRIP)로 변환해 정확한 값을 산출한다.

배경: tcp_offset.yaml/_PREGRASP_GRIP이 finger '바디원점(너클)'을 파지중심으로 써서 실제 pad보다
      ~3.5cm 벗어남 → reach가 헛점 유도 + pregrasp가 pad 밖에 큐브 배치. usd_collision_inspect로
      측정한 pad 중심(link frame): L=(-0.0003,0.0295,0.0197) R=(0.0225,0.0295,0.0197).
      이를 finger body world pose로 world 변환 → wrist_3 프레임/ env-local로 재변환.
실행: python diagnostics\\measure_grasp_frame.py --headless
산출: 콘솔 — 새 tcp_offset 병진(wrist_3 frame) + _PREGRASP_GRIP(env-local).
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

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_apply, quat_inv, subtract_frame_transforms

import ur5e_binpick.tasks  # noqa: F401

from ur5e_binpick.tasks.binpick.mdp.scene import NUM_OBJECTS, object_names

_TASK = "Isaac-BinPick-UR5e-Teacher-v0"


def main() -> None:
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env = gym.make(_TASK, cfg=env_cfg)
    uenv = env.unwrapped
    n = uenv.num_envs
    dev = uenv.device
    uenv._bp_pregrasp_frac = 0.0     # pregrasp 배치 없이 홈 포즈만 — 순수 기하 측정
    env.reset()
    # 홈 포즈 안정화
    for _ in range(3):
        with torch.inference_mode():
            env.step(torch.zeros(n, 7, device=dev))

    robot = uenv.scene["robot"]
    wids, _ = robot.find_bodies("wrist_3_link")
    wid = wids[0]
    fbL, _ = robot.find_bodies("left_finger_link")
    fbR, _ = robot.find_bodies("right_finger_link")

    off_L = torch.tensor([-0.0003, 0.0295, 0.0197], device=dev).expand(n, 3)
    off_R = torch.tensor([0.0225, 0.0295, 0.0197], device=dev).expand(n, 3)
    with torch.inference_mode():
        wp = robot.data.body_state_w[:, wid, :3]        # wrist_3 world pos
        wq = robot.data.body_state_w[:, wid, 3:7]       # wrist_3 world quat
        pL = robot.data.body_state_w[:, fbL[0], :3] + quat_apply(robot.data.body_state_w[:, fbL[0], 3:7], off_L)
        pR = robot.data.body_state_w[:, fbR[0], :3] + quat_apply(robot.data.body_state_w[:, fbR[0], 3:7], off_R)
        pad_c = (pL + pR) / 2.0                          # pad 중심 world
        env0 = uenv.scene.env_origins
        # ① wrist_3 프레임에서의 pad 중심 (→ tcp_offset 병진)
        off_in_wrist, _ = subtract_frame_transforms(wp, wq, pad_c)
        # ② env-local world (→ _PREGRASP_GRIP)
        pad_env_local = pad_c - env0

    print("=" * 66)
    print(f"[meas] wrist_3 world pos = ({wp[0,0]:.4f},{wp[0,1]:.4f},{wp[0,2]:.4f})")
    print(f"[meas] pad 중심 world    = ({pad_c[0,0]:.4f},{pad_c[0,1]:.4f},{pad_c[0,2]:.4f})")
    print(f"[meas] pad_L↔pad_R 거리  = {(pL-pR).norm():.4f} m  (홈 포즈 finger open state)")
    print("-" * 66)
    print(f"[meas] ★새 tcp_offset T_flange_tcp 병진(wrist_3 frame)★:")
    print(f"[meas]   x={off_in_wrist[0,0]:+.4f}  y={off_in_wrist[0,1]:+.4f}  z={off_in_wrist[0,2]:+.4f}")
    print(f"[meas] ★새 _PREGRASP_GRIP(env-local world)★:")
    print(f"[meas]   ({pad_env_local[0,0]:.4f}, {pad_env_local[0,1]:.4f}, {pad_env_local[0,2]:.4f})")
    print("=" * 66)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
