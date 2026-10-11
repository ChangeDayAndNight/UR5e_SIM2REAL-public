#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""핑거 도달 깊이 실측 — ★TCP 기준으로 손가락이 얼마나 아래까지 내려가는가★ (진단 전용).

배경(F51, 2026-08-09): 하강 스윕에서 패드-벨트 접촉이 TCP z=0.0174에서 시작했는데,
URDF 콜리전 박스(0.0192×0.032×0.045)로 오프라인 계산하면 그보다 훨씬 위에서 멈춰야 한다.
또 정리 전 현장 진단 화면(Git history 보존)에서 핑거팁이 하우징 바깥으로 튀어나와 보였다.
⇒ 오프라인 추정을 버리고 ★런타임 실제 콜라이더 world AABB★를 직접 잰다.

산출: TCP world z · 좌우 핑거 콜라이더 world AABB · 벨트 상면 · "TCP − 핑거 최저점" 깊이.
      비주얼 메시 prim의 AABB도 함께 재서 ★비주얼 vs 콜리전 어긋남★을 정량화한다.

실행: python diagnostics\\finger_reach_probe.py --headless
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
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="핑거 도달 깊이 실측(진단 전용).")
parser.add_argument("--task", type=str, default="Isaac-BinPick-UR5e-Teacher-Eval-v0")
parser.add_argument("--out", type=str, default=None, help="결과 텍스트 저장 경로")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import ur5e_binpick.tasks  # noqa: F401,E402

from ur5e_binpick.tasks.binpick.mdp.observations import tcp_pose_b  # noqa: E402
from ur5e_binpick.tasks.binpick.mdp.scene import TABLE_HEIGHT, OBJECT_SIZE  # noqa: E402

lines: list[str] = []


def say(msg: str = "") -> None:
    lines.append(msg)
    print(msg)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.reset()
    for _ in range(5):  # 물리 안정화
        env.step(torch.zeros((1, env.unwrapped.action_space.shape[-1]), device=env.unwrapped.device))

    uenv = env.unwrapped
    stage = uenv.sim.stage
    tcp = tcp_pose_b(uenv)[0, :3].tolist()
    robot = uenv.scene["robot"]
    ids, names = robot.find_bodies(["left_finger_link", "right_finger_link"])
    body_pos = robot.data.body_pos_w[0, ids].tolist()
    origin = uenv.scene.env_origins[0].tolist()

    say("UR5e — ★핑거 도달 깊이 실측★ (런타임 콜라이더/비주얼 world AABB)")
    say("=" * 78)
    say(f"벨트 상면 TABLE_HEIGHT = {TABLE_HEIGHT:+.4f} m · 큐브 한 변 {OBJECT_SIZE:.3f} m "
        f"(안착 중심 z = {TABLE_HEIGHT + OBJECT_SIZE/2:+.4f})")
    say(f"env origin = {[round(v,4) for v in origin]}")
    say(f"TCP (base frame) = x{tcp[0]:+.4f} y{tcp[1]:+.4f} ★z{tcp[2]:+.4f}★")
    for nm, p in zip(names, body_pos):
        say(f"  {nm} body world = x{p[0]:+.4f} y{p[1]:+.4f} z{p[2]:+.4f}")
    say("-" * 78)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
                              UsdGeom.Tokens.guide])
    tcp_world_z = tcp[2] + origin[2]
    for finger in ("left_finger_link", "right_finger_link"):
        root = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{finger}")
        if not root or not root.IsValid():
            say(f"⚠️ prim 없음: {finger}")
            continue
        say(f"[{finger}]")
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Boundable):
                continue
            try:
                b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                if b.IsEmpty():
                    continue
                lo, hi = b.GetMin(), b.GetMax()
            except Exception:
                continue
            kind = "?"
            path = prim.GetPath().pathString
            if "collision" in path.lower():
                kind = "COLLISION"
            elif "visual" in path.lower():
                kind = "VISUAL"
            say(f"  {kind:<9} {prim.GetName():<22} "
                f"x[{lo[0]:+.4f},{hi[0]:+.4f}] y[{lo[1]:+.4f},{hi[1]:+.4f}] z[{lo[2]:+.4f},{hi[2]:+.4f}]"
                f"  ★TCP−최저 {(tcp_world_z - lo[2])*1000:6.1f} mm★")
    say("-" * 78)
    say("해석: 'TCP−최저'가 큐브 반변(25.0mm)보다 크면 ★큐브를 안착시킬 때 손가락이 먼저 바닥에 닿는다★.")
    say(f"      F51 하강 스윕 실측 접촉 개시 TCP z = 0.0174 → 그때 손가락 최저점 = 벨트 상면 {TABLE_HEIGHT:+.4f}")
    say(f"      ⇒ 역산한 도달 깊이 = {(0.0174 - TABLE_HEIGHT)*1000:.1f} mm")

    env.close()
    if args_cli.out:
        with open(args_cli.out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[probe] 저장: {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
