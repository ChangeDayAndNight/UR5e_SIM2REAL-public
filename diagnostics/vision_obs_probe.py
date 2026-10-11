#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + repo 루트.
# ─────────────────────────────────────────────────────────────────────────────
"""vision 관측(depth) 건강 프로브 — student 정책이 받는 flat 관측의 각 슬라이스를 실측 검증.

목적: Phase2 증류가 정체할 때 "student 입력에 태스크 정보가 있는가"를 정량 판별한다.
      ① depth 12,544픽셀: far-포화/제로 비율, 공간 분산(큐브·로봇이 보이는가), 시간 분산
      ② proprio/target/last_action 슬라이스: 상수/NaN 여부
      (2026-07-13 최초 작성 — 증류 behavior loss 무진전 진단 중 depth-blind 가설 배제용.)
실행 (학습 PC, isaac51 — GPU 필요, 학습과 동시 실행 금지):
  python diagnostics\\vision_obs_probe.py --num_envs 4 --steps 40 --headless
산출: Phase2/Result/diagnostics/vision_obs_probe_<시각>/ — stats.txt + depth_env0.png + obs.npz
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

# ═══════════════════ AppLauncher (isaaclab import은 이 아래에서만) ═══════════════════
import argparse

from isaaclab.app import AppLauncher

_TASK = "Isaac-BinPick-UR5e-Vision-Eval-v0"  # 보조 off + TiledCamera (student와 동일 관측)

parser = argparse.ArgumentParser(description="vision 관측(depth) 건강 프로브.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = _TASK
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────── sim 기동 후 ───────────────────────────
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401  (gym.register)

from common.obs_preprocess import obs_layout  # 슬라이스 경계의 단일 소스(계약)
from common.config_io import build_preprocess_cfg, configs_dir, load_yaml


def main() -> None:
    pre_cfg = build_preprocess_cfg(load_yaml(configs_dir() / "common.yaml"))
    # obs_layout은 (이름, 크기) — 누적합으로 (이름, 시작, 끝) 슬라이스 경계를 만든다.
    layout, _off = [], 0
    for _nm, _sz in obs_layout(pre_cfg):
        layout.append((_nm, _off, _off + _sz))
        _off += _sz
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(_TASK, cfg=env_cfg)
    obs_dict, _ = env.reset()

    n = args_cli.num_envs
    frames = []
    with torch.inference_mode():
        for t in range(args_cli.steps):
            actions = torch.zeros(n, env.unwrapped.action_manager.total_action_dim,
                                  device=env.unwrapped.device)
            obs_dict, _, _, _, _ = env.step(actions)
            frames.append(obs_dict["policy"].detach().cpu().numpy().copy())
    env.close()

    X = np.stack(frames)  # [T, N, obs_dim] — 현행 RGB-D 계약은 obs_dim=50,206
    out_dir = os.path.join(_REPO_ROOT, "Phase2", "Result", "diagnostics",
                           "vision_obs_probe_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)

    lines = [f"vision obs 건강 프로브 — task={_TASK}, T={X.shape[0]}, N={X.shape[1]}, dim={X.shape[2]}"]
    verdicts = []
    for name, s, e in layout:
        sl = X[:, :, s:e]
        finite = np.isfinite(sl).all()
        lines.append(
            f"[{name:12s}] ({s:4d}:{e:4d}) min {np.nanmin(sl):+.4f} max {np.nanmax(sl):+.4f} "
            f"mean {np.nanmean(sl):+.4f} std {np.nanstd(sl):.5f} finite={finite}"
        )
        if name == "depth":
            far_frac = float((sl >= 0.999).mean())
            zero_frac = float((sl <= 0.001).mean())
            # 공간 분산: 프레임 내 픽셀 표준편차(평균) — 균일 이미지면 0
            spat_std = float(sl.reshape(sl.shape[0], sl.shape[1], -1).std(axis=2).mean())
            # 시간 분산: 픽셀별 시간 표준편차(평균) — 정지 화면이면 0
            temp_std = float(sl.std(axis=0).mean())
            lines.append(
                f"  depth 상세: far(=1.0) 비율 {far_frac:.1%} | zero 비율 {zero_frac:.1%} | "
                f"공간 std {spat_std:.5f} | 시간 std {temp_std:.5f}"
            )
            healthy = (far_frac < 0.98) and (spat_std > 0.005) and finite
            verdicts.append(("depth", healthy,
                             "구조 보임(공간 분산 있음)" if healthy else "★균일/포화 — student 장님 의심★"))
        else:
            healthy = finite and float(np.nanstd(sl)) > 1e-6 or name == "last_action"
            verdicts.append((name, bool(healthy), "" if healthy else "★상수/NaN 의심★"))

    lines.append("")
    lines.append("판정:")
    for name, ok, note in verdicts:
        lines.append(f"  {'✓' if ok else '✗'} {name} {note}")

    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("[probe]\n" + txt)

    # env0 depth 스냅샷 PNG (마지막 프레임) — 육안 확인용
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        d_s, d_e = layout[0][1], layout[0][2]
        img = X[-1, 0, d_s:d_e].reshape(pre_cfg.out_height, pre_cfg.out_width)
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(label="normalized depth")
        plt.title("policy obs depth (env0, last frame)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "depth_env0.png"), dpi=120)
        plt.close()
    except Exception as e:
        print(f"[probe][warn] PNG 저장 생략: {e}")

    np.savez_compressed(os.path.join(out_dir, "obs.npz"), obs=X[-5:])  # 마지막 5프레임만(용량)
    print(f"[probe] ★산출★ {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
