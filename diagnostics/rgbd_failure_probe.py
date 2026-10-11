"""RGB-D student 실패 진단 프로브 — 큐브 가시성·접근 거동·DR 암도 판별.

목적: eval 0.8% 실패의 원인 가설 판별(2026-07-18) —
  ①FOV 협각화(58→42.5°)로 큐브 시야 이탈 ②시각 DR(색/조명/지터) 난도 ③기타.
수집: env0 wrist RGB(x8)/depth 프레임 10장 + 에피소드별 TCP-큐브 최소거리 +
  큐브색-벨트색 유사도 + 프레임 밝기 분포.
실행: isaac51 python -u diagnostics/rgbd_failure_probe.py (headless, GPU 단독)
산출: Phase2/Result/diagnostics/rgbd_failure_probe/ (프레임 PNG + stats 콘솔)
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from scripts.training_env import preload_native_training_modules

preload_native_training_modules()

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless", "--enable_cameras"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import DistillationRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401
from scripts.rsl_rl import cli_args  # noqa: E402  (repo 프리앰블 경로)

TASK = "Isaac-BinPick-UR5e-Vision-Eval-v0"
CKPT = str(_ROOT / "logs/rsl_rl/ur5e_binpick_vision/2026-07-17_19-15-04/model_1999.pt")
OUT = _ROOT / "Phase2" / "Result" / "diagnostics" / "rgbd_failure_probe"
OUT.mkdir(parents=True, exist_ok=True)
N_ENVS = 16
STEPS = 340  # ~2 에피소드


def save_png(arr_u8, name):
    from PIL import Image

    Image.fromarray(arr_u8).save(str(OUT / name))


class _A:  # cli_args.parse_rsl_rl_cfg 최소 네임스페이스
    task = TASK
    seed = None
    resume = None
    load_run = None
    checkpoint = None
    experiment_name = None
    run_name = None
    logger = None
    log_project_name = None
    device = "cuda:0"


env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=N_ENVS)
agent_cfg = cli_args.parse_rsl_rl_cfg(TASK, _A())
env = gym.make(TASK, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
runner.load(CKPT)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(f"[probe] 정책 로드 OK: {CKPT}", flush=True)

uenv = env.unwrapped
sensor = uenv.scene.sensors["tiled_camera"]
obs = env.get_observations()

min_dist = torch.full((N_ENVS,), float("inf"), device=uenv.device)
ep_min_dists = []
bright_list = []
cube_belt_sim = []
frame_i = 0

for step in range(STEPS):
    with torch.inference_mode():
        act = policy(obs)
        obs, _, dones, _ = env.step(act)

    # TCP-큐브 거리 (최근접 객체)
    from ur5e_binpick.tasks.binpick.binpick_env_cfg import _nearest_object

    ee = uenv.scene["ee_frame"]
    tcp = ee.data.target_pos_source[:, 0, :]
    obj_pos, _ = _nearest_object(uenv)
    d = torch.norm(tcp - obj_pos, dim=1)
    min_dist = torch.minimum(min_dist, d)
    done_mask = dones.to(torch.bool)
    if done_mask.any():
        ep_min_dists += min_dist[done_mask].cpu().tolist()
        min_dist[done_mask] = float("inf")

    # 프레임 수집 (env0, 30스텝 간격)
    if step % 30 == 0:
        rgb = sensor.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        save_png(np.kron(rgb, np.ones((8, 8, 1), dtype=np.uint8)), f"f{frame_i:02d}_rgb.png")
        dpt = sensor.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        dv = np.where(np.isfinite(dpt) & (dpt > 0), dpt, np.nan)
        lo, hi = np.nanpercentile(dv, 5), np.nanpercentile(dv, 95)
        if np.isfinite(lo) and hi > lo:
            vis = np.nan_to_num(np.clip((dv - lo) / (hi - lo), 0, 1), nan=1.0)
            save_png(np.kron((vis * 255).astype(np.uint8), np.ones((8, 8), dtype=np.uint8)),
                     f"f{frame_i:02d}_depth.png")
        bright_list.append(float(rgb.mean()))
        frame_i += 1

print(f"[probe] ★에피소드별 TCP-큐브 최소거리★ n={len(ep_min_dists)}", flush=True)
if ep_min_dists:
    a = np.array(ep_min_dists)
    print(f"[probe]   med={np.median(a)*100:.1f}cm  p25={np.percentile(a,25)*100:.1f}  "
          f"p75={np.percentile(a,75)*100:.1f}  <6cm(파지권 진입) 비율={float((a<0.06).mean()):.2f}",
          flush=True)
print(f"[probe] 프레임 밝기(mean/255): min={min(bright_list):.1f} med={np.median(bright_list):.1f} "
      f"max={max(bright_list):.1f}  (극암<20이면 조명 DR 의심)", flush=True)
print(f"[probe] ★완료★ → {OUT}", flush=True)
env.close()
os._exit(0)
