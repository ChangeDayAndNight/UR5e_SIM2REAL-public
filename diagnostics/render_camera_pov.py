#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★캘리브 카메라 POV 렌더★ — hand_eye.yaml(실측)의 카메라 시점에서 씬을 렌더해 검증.

목적: 실측 hand-eye 반영 후 "카메라가 어디 붙어 무엇을 보는가"를 눈으로 확인.
  ① cam_pov      : 광학중심에서 광축 방향으로 본 화면(D435i가 보는 것)
  ② cam_side/back: 손목·카메라 장착부 클로즈업(외부에서 본 장착 상태)
실행: python diagnostics\\render_camera_pov.py --headless --enable_cameras
산출: Phase1/Result/diagnostics/camera_pov_<ts>/ PNG
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
import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
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
from common.calibration import load_hand_eye
from common.config_io import configs_dir

_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _save_png(frame, path) -> bool:
    try:
        from PIL import Image

        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            Image.fromarray(arr[:, :, :3].astype(np.uint8)).save(path)
            return True
    except Exception as e:  # noqa: BLE001
        print(f"[render] ⚠️ 저장 실패 {path}: {e}")
    return False


def main() -> None:
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=1)
    env_cfg.seed = 7
    env = gym.make(_TASK, cfg=env_cfg, render_mode="rgb_array")  # uenv.render()가 프레임 반환
    uenv = env.unwrapped
    env.reset()
    for _ in range(10):  # settle + RTX 워밍업
        with torch.inference_mode():
            env.step(torch.zeros(1, 7, device=uenv.device))
        uenv.render()

    robot = uenv.scene["robot"]
    w_idx = robot.body_names.index("wrist_3_link")
    wp = robot.data.body_pos_w[0, w_idx].cpu().numpy()
    wq = robot.data.body_quat_w[0, w_idx].cpu().numpy()
    Rw = _quat_to_mat(wq)
    T_fc = load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")
    cam_p = wp + Rw @ T_fc[:3, 3]           # 광학중심(world)
    cam_R = Rw @ T_fc[:3, :3]               # optical: z=전방
    look = cam_p + cam_R[:, 2] * 0.5
    print(f"[render] 카메라 광학중심(world) = {np.round(cam_p, 4).tolist()}")
    print(f"[render] 광축 방향(world)     = {np.round(cam_R[:, 2], 4).tolist()}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(_REPO_ROOT, "Phase1", "Result", "diagnostics", f"camera_pov_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    # cam_pov 시점: 광축 12cm 앞은 그리퍼 몸체 '내부'(카메라가 TCP 방향을 주시) → 이미지-상방향
    # (-y_cam)으로 6cm 비켜 메시 밖에서 같은 방향을 본다(평행이동 — 구도 거의 동일).
    pov_eye = cam_p + cam_R[:, 2] * 0.02 - cam_R[:, 1] * 0.06
    # ⚠️ 관측된 렌더러 버그: 뷰 전환마다 RTX 누적 상태가 토글돼 '짝수번째 뷰'가 전흑.
    #   → 희생 더미 뷰(_skip_)를 짝수 슬롯에 끼워 실제 뷰를 홀수 슬롯으로 민다.
    views = [
        ("_skip_a", (1.5, -1.2, 1.0), (0.45, 0.0, 0.1)),
        ("cam_pov", tuple(pov_eye), tuple(pov_eye + cam_R[:, 2] * 0.8)),   # ★D435i가 보는 화면★
        ("_skip_b", (1.5, -1.2, 1.0), (0.45, 0.0, 0.1)),
        ("mount_side", (cam_p[0] + 0.28, cam_p[1] - 0.30, cam_p[2] + 0.12), tuple(cam_p)),
        ("_skip_c", (1.5, -1.2, 1.0), (0.45, 0.0, 0.1)),
        ("mount_front", (cam_p[0] - 0.55, cam_p[1] + 0.45, cam_p[2] + 0.35), tuple(cam_p)),
        ("_skip_d", (1.5, -1.2, 1.0), (0.45, 0.0, 0.1)),
        ("mount_top", (cam_p[0] + 0.05, cam_p[1] + 0.02, cam_p[2] + 0.55), tuple(cam_p)),
    ]
    saved = 0
    for i, (name, eye, lookat) in enumerate(views):
        uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
        for _ in range(16 if i == 0 else 8):  # RTX 누적 워밍업(검정프레임 방지 — 넉넉히)
            with torch.inference_mode():
                env.step(torch.zeros(1, 7, device=uenv.device))
            uenv.render()
        # ★검정프레임 방지★: 더블버퍼 패리티로 프레임이 전흑일 수 있음 — 최대 6회 재캡처
        fr = uenv.render()
        for _retry in range(6):
            if fr is not None and np.asarray(fr)[:, :, :3].max() > 8:
                break
            with torch.inference_mode():
                env.step(torch.zeros(1, 7, device=uenv.device))
            uenv.render()
            fr = uenv.render()
        if name.startswith("_skip_"):
            continue
        if fr is not None and _save_png(fr, os.path.join(out_dir, f"{name}.png")):
            saved += 1
            print(f"[render] {name}.png 저장 (max={int(np.asarray(fr)[:, :, :3].max())})", flush=True)
    env.close()
    print(f"[render] ★완료: {saved}/{len(views)} → {out_dir}★")


if __name__ == "__main__":
    main()
    simulation_app.close()
