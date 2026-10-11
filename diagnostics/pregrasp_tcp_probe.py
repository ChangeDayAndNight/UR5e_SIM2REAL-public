#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★pregrasp 리셋 직후 ee_frame TCP vs 큐브 위치 직접 측정★ — reach 저조 원인 확정용.

목적: 학습 reach가 pregrasp env(큐브가 그립에 있어야 함)에서도 낮은 이유가
      (a) 큐브가 실제로 TCP에서 벗어나 배치되는가(ee_frame/pregrasp 불일치), 아니면
      (b) 리셋 땐 붙어있는데 정책이 즉시 흘리는가 를 판별한다.
      pregrasp_frac=1.0 강제, 리셋 직후 몇 스텝(zero-action)에서 env별 ee_frame TCP,
      object_0 위치, TCP↔cube 거리(=reach 입력), grasp 접촉을 출력한다.
실행: python diagnostics\\pregrasp_tcp_probe.py --headless
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
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401
from ur5e_binpick.tasks.binpick.mdp import observations as bp_obs
from ur5e_binpick.tasks.binpick.mdp.scene import NUM_OBJECTS

_TASK = "Isaac-BinPick-UR5e-Teacher-v0"


def main() -> None:
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 0
    env = gym.make(_TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    uenv._bp_pregrasp_frac = 1.0   # 전 env pregrasp
    env.reset()
    dev = uenv.device
    print(f"[probe] pregrasp_frac=1.0, {args_cli.num_envs} env — 리셋 직후 zero-action 스텝별 측정")
    print("  step | env | TCP(x,y,z) | cube(x,y,z) | TCP↔cube[cm] | reach")
    for step in range(6):
        with torch.inference_mode():
            tcp = bp_obs.tcp_pose_b(uenv)[:, :3]                       # ee_frame TCP (base)
            objs = bp_obs.object_positions_b(uenv).view(uenv.num_envs, NUM_OBJECTS, 3)
            d = torch.norm(objs - tcp.unsqueeze(1), dim=-1).min(dim=1)
            dist = d.values
            reach = 1.0 - torch.tanh(dist / 0.2)
            for e in range(min(args_cli.num_envs, 4)):
                oi = objs[e, d.indices[e]]
                t = tcp[e]
                print(f"   {step}  |  {e}  | ({t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}) | "
                      f"({oi[0]:+.3f},{oi[1]:+.3f},{oi[2]:+.3f}) | {100*dist[e]:5.1f} | {reach[e]:.3f}")
            env.step(torch.zeros(uenv.num_envs, 7, device=dev))
        print("  ---")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
