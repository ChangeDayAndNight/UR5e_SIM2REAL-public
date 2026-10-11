#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★그리퍼-큐브 파지 순간 렌더★ — 큐브를 pad 중심에 놓고 그리퍼를 닫으며 클로즈업 프레임을
   PNG로 저장한다. 수치 포렌식(접촉력·q)이 애매한 지점을 '눈으로' 확정하기 위한 시각 검증.

배경: pad 중심 배치 후에도 finger가 q=0.019까지 닫히며 접촉이 오히려 감소(0.017→0.019: 1N→0N).
      닫힘에서 콜라이더가 큐브를 벗어나는지(기하), 아니면 관통/이탈인지 육안 확인.
실행: python diagnostics\\render_grip.py --headless --enable_cameras
산출: <scratchpad>/grip_frame_XX.png (여러 스텝)
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

parser = argparse.ArgumentParser(description="그리퍼-큐브 파지 렌더.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401

from ur5e_binpick.tasks.binpick.mdp.scene import NUM_OBJECTS, OBJECT_SIZE, object_names
from ur5e_binpick.tasks.binpick.mdp.events_dr import _PREGRASP_GRIP

_TASK = "Isaac-BinPick-UR5e-Teacher-v0"


def _save_png(arr, path):
    try:
        from PIL import Image
        Image.fromarray(np.asarray(arr).astype(np.uint8)).save(path)
        return True
    except Exception as e:
        print(f"[render] PNG 저장 실패({e})")
        return False


def main() -> None:
    out_dir = args_cli.out or os.path.join(
        os.environ.get("TEMP", "/tmp"), "grip_render")
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    # ★파지축(Y)에 수직인 -X 시점★ — 두 손가락이 좌우로 벌어져 보이도록(파지축 따라 보면 겹침).
    # ★2026-08-03 수정★: 구버전은 구 홈 파지점(0.49,0.133,0.262) 기준 ★하드코딩★이었다.
    #   홈 포즈 교체(2026-07-28)와 배치 180° 회전으로 파지점이 (−0.298,−0.133,0.509)로 옮겨져
    #   카메라가 엉뚱한 곳을 봤다. ⇒ ★파지점 상수에서 파생★한다(홈이 바뀌면 자동 추종).
    #   프레이밍은 검증된 구 값 보존: 파지점에서 base 방향으로 0.272m 물러나 0.098m 위.
    import math as _math
    _g = _PREGRASP_GRIP
    _n = max(_math.hypot(_g[0], _g[1]), 1e-6)
    env_cfg.viewer.eye = (_g[0] - 0.272 * _g[0] / _n, _g[1] - 0.272 * _g[1] / _n, _g[2] + 0.098)
    env_cfg.viewer.lookat = (_g[0], _g[1], _g[2] + 0.073)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env = gym.make(_TASK, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped
    n = uenv.num_envs
    dev = uenv.device
    uenv._bp_pregrasp_frac = 1.0
    env.reset()

    robot = uenv.scene["robot"]
    cube = uenv.scene[object_names(NUM_OBJECTS)[0]]
    fj_ids, _ = robot.find_joints(["left_finger_joint", "right_finger_joint"])

    saved = []
    # ★수정된 pregrasp(박스 콜라이더+clamp)로 리셋된 상태 그대로★ — hold 후 lift.
    #   grip이 성립하면 cube가 pad 사이에 물린 채 상승한다.
    def _shot(name):
        fr = uenv.render()
        if fr is not None:
            p = os.path.join(out_dir, name)
            if _save_png(fr, p):
                saved.append(p)
                cz = cube.data.root_pos_w[0, 2] - uenv.scene.env_origins[0, 2]
                print(f"[render] {name}: cube_z={cz:.4f}")
    for _ in range(3):
        with torch.inference_mode():
            env.step(torch.zeros(n, 7, device=dev))
    _shot("g0_reset_grip.png")           # 리셋 직후(물린 상태)
    for step in range(18):               # hold 3 + lift 15
        actions = torch.zeros(n, 7, device=dev)
        actions[:, -1] = -1.0            # close 유지
        if step >= 3:
            actions[:, 2] = 0.1          # +z lift
        with torch.inference_mode():
            env.step(actions)
        if step in (2, 8, 17):
            _shot(f"g{step+1}_{'hold' if step < 3 else 'lift'}.png")
    env.close()
    print(f"[render] ★저장 {len(saved)}장★ {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
