#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★로봇 전체 조망 렌더 + 체인 프림 감사★ — TCW1 체인 정정(2026-07-15) 수락 기준 증빙.

산출(수락 기준 대응):
  ① 뷰포트 PNG: 전체 조망 / 손목 클로즈업(TCW1·브라켓·D435i 본체) / 반대측 —
     "로봇 전체(팔+TCW1+그리퍼+카메라)가 정상적으로 보임" 증빙 + GUI 프레이밍 점검.
  ② Stage 프림 트리 덤프(wrist_3_link 하위): TCW1/브라켓/카메라 시각 프림 존재 확인
     (merge_fixed_joints=True라 별도 바디가 아닌 시각 프림으로 병합됨 — 표준).
  ③ 홈 포즈 수치: wrist_3/finger pad 월드 좌표 + NaN 검사(기구학 건강).
실행: python diagnostics\\render_robot_overview.py --headless --enable_cameras
산출: Phase1/Result/diagnostics/robot_overview_<ts>/ PNG + prims.txt
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
import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="로봇 전체 조망 렌더 + 체인 프림 감사.")
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

# eval cfg = 보조(pregrasp) 완전 off — 순수 홈 포즈 조망
_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"


def _save_png(arr, path: str) -> bool:
    try:
        from PIL import Image

        Image.fromarray(np.asarray(arr).astype(np.uint8)).save(path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[render] PNG 저장 실패({e})")
        return False


def main() -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args_cli.out or os.path.join(_REPO_ROOT, "Phase1", "Result", "diagnostics", f"robot_overview_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env = gym.make(_TASK, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped

    # 물리 settle + 렌더 워밍업
    env.reset()
    for _ in range(5):
        with torch.inference_mode():
            env.step(torch.zeros(uenv.num_envs, 7, device=uenv.device))

    # ── ② Stage 프림 감사: wrist_3_link 하위 트리(TCW1/브라켓/카메라 시각 병합 확인) ──
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    wrist = stage.GetPrimAtPath("/World/envs/env_0/Robot/wrist_3_link")
    lines = []
    if wrist.IsValid():
        from pxr import Usd

        for prim in Usd.PrimRange(wrist):
            depth = str(prim.GetPath()).count("/") - str(wrist.GetPath()).count("/")
            lines.append("  " * depth + f"{prim.GetName()} <{prim.GetTypeName()}>")
    else:
        lines.append("⚠️ wrist_3_link 프림 없음!")
    prim_txt = "\n".join(lines)
    with open(os.path.join(out_dir, "prims.txt"), "w", encoding="utf-8") as f:
        f.write(prim_txt + "\n")
    print(f"[audit] wrist_3_link 하위 프림 {len(lines)}개 → prims.txt")
    for key in ("tcw1", "bracket", "camera", "d435", "WristCam"):
        hit = [l for l in lines if key.lower() in l.lower()]
        print(f"[audit]   '{key}' 매칭 {len(hit)}건" + (f" — 예: {hit[0].strip()}" if hit else "  ⚠️ 없음"))

    # ── ③ 기구학·씬 수치: 링크 월드 좌표 + NaN 검사 + 씬 자산 위치 → audit.txt ──
    robot = uenv.scene["robot"]
    names = robot.body_names
    pos = robot.data.body_pos_w[0] - uenv.scene.env_origins[0]
    finite = bool(torch.isfinite(robot.data.body_pos_w).all() and torch.isfinite(robot.data.joint_pos).all())
    audit = [f"바디 {len(names)}개, 전체 finite={finite}", f"env_origin={tuple(float(v) for v in uenv.scene.env_origins[0])}"]
    for i, nm in enumerate(names):
        p = pos[i]
        audit.append(f"  {nm:24s} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    # 씬 주요 프림 world AABB (벨트/스탠드/바닥 — 프레이밍 차폐 원인 추적)
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    env0 = stage.GetPrimAtPath("/World/envs/env_0")
    targets = list(env0.GetChildren()) if env0.IsValid() else []
    world = stage.GetPrimAtPath("/World")
    targets += [p for p in (world.GetChildren() if world.IsValid() else []) if p.GetName() not in ("envs",)]
    for prim in targets:
        try:
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if r.IsEmpty():
                audit.append(f"  AABB {prim.GetPath()}: (빈 bound)")
                continue
            mn, mx = r.GetMin(), r.GetMax()
            audit.append(f"  AABB {prim.GetPath()}: min({mn[0]:+.3f},{mn[1]:+.3f},{mn[2]:+.3f}) max({mx[0]:+.3f},{mx[1]:+.3f},{mx[2]:+.3f})")
        except Exception as e:  # noqa: BLE001
            audit.append(f"  AABB {prim.GetPath()}: 계산 실패({e})")
    with open(os.path.join(out_dir, "audit.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(audit) + "\n")
    print("\n".join(f"[audit] {l}" for l in audit), flush=True)

    # ── ① 후보 시점 그리드 렌더(프레이밍 탐색 — 한 번의 부팅으로 전부) ──
    wrist_idx = names.index("wrist_3_link") if "wrist_3_link" in names else 0
    wrist_w = tuple(float(v) for v in robot.data.body_pos_w[0, wrist_idx])
    base_w = tuple(float(v) for v in robot.data.body_pos_w[0, 0])
    ctr = (base_w[0] + 0.35, base_w[1], base_w[2] + 0.15)  # 로봇 몸통 중심 근사
    views = [
        ("v1_front_high", (ctr[0] + 1.2, ctr[1] + 0.9, ctr[2] + 0.8), ctr),
        ("v2_front_low", (ctr[0] + 1.0, ctr[1] - 0.9, ctr[2] + 0.4), ctr),
        ("v3_side", (ctr[0] - 1.2, ctr[1] + 0.9, ctr[2] + 0.6), ctr),
        ("v4_back_high", (ctr[0] - 1.0, ctr[1] - 1.0, ctr[2] + 0.9), ctr),
        ("v5_top", (ctr[0] + 0.3, ctr[1] + 0.1, ctr[2] + 1.6), ctr),
        ("v6_wrist_a", (wrist_w[0] + 0.45, wrist_w[1] - 0.35, wrist_w[2] + 0.30), wrist_w),
        ("v7_wrist_b", (wrist_w[0] - 0.40, wrist_w[1] - 0.40, wrist_w[2] + 0.25), wrist_w),
        ("v8_wrist_cam_side", (wrist_w[0] + 0.05, wrist_w[1] - 0.55, wrist_w[2] - 0.05), wrist_w),
    ]
    # 브라켓+카메라 클러스터(공칭: wrist에서 tool -y 0.075·전방 0.12 — 홈 포즈 tool z=아래)
    cam_pt = (wrist_w[0], wrist_w[1], wrist_w[2] - 0.115)
    import math

    for k, az in enumerate((0, 90, 180, 270)):
        a = math.radians(az)
        eye = (cam_pt[0] + 0.34 * math.cos(a), cam_pt[1] + 0.34 * math.sin(a), cam_pt[2] + 0.10)
        views.append((f"orbit_{az:03d}", eye, cam_pt))

    saved = 0
    for i, (name, eye, lookat) in enumerate(views):
        uenv.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
        # ★첫 프레임 검정 방지★: RTX 누적 워밍업 — 뷰 이동 후 렌더 반복(첫 뷰는 더 길게)
        for _ in range(8 if i == 0 else 3):
            with torch.inference_mode():
                env.step(torch.zeros(uenv.num_envs, 7, device=uenv.device))
            uenv.render()
        fr = uenv.render()
        if fr is not None and _save_png(fr, os.path.join(out_dir, f"{name}.png")):
            saved += 1
            print(f"[render] {name}.png 저장 (eye={tuple(round(v,2) for v in eye)})", flush=True)
    env.close()
    print(f"[render] ★완료: PNG {saved}/{len(views)} + prims.txt + audit.txt → {out_dir}★", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
