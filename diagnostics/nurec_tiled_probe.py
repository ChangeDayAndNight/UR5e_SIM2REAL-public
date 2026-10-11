"""NuRec × TiledCamera 호환 프로브 — RGB/depth annotator·변환·VRAM 실증.

검증 4항목:
  ① TiledCamera rgb annotator에 NuRec 가우시안이 렌더되는가 (최대 게이트)
  ② distance_to_image_plane annotator에 가우시안 depth가 잡히는가 (메시 큐브 대조)
  ③ 참조 루트 xformOp 이동/스케일이 볼륨 렌더에 반영되는가 (정합 실행 가능성)
  ④ 동일 usdz 2번째 참조 시 VRAM 증분 (인스턴싱 공유 여부)
실행: isaac51 python -u (headless, --enable_cameras 하드코딩)
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

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

parser = argparse.ArgumentParser()
parser.add_argument("--usdz", type=Path, default=ROOT / "assets" / "Scene" / "Workplace.usdz")
parser.add_argument("--out", type=Path, default=ROOT / "diagnostics" / "nurec_tiled_probe_out")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
if not args.usdz.is_file():
    raise SystemExit(f"[probe] optional USDZ 없음: {args.usdz}")
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Kit 기동 이후에만 무거운 import ----
import subprocess
import time

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.utils.stage import add_reference_to_stage

USDZ = str(args.usdz.resolve())
OUT = str(args.out.resolve())
os.makedirs(OUT, exist_ok=True)
T = (-1.0, -1.0, 1.0)  # 밀도 핵심 타겟 (nurec_render2 검증값)
CUBE = (T[0] + 1.5, T[1] + 1.5, T[2])


def vram(tag: str) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        print(f"[probe] VRAM({tag}) = {out} MiB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] VRAM({tag}) 측정 실패: {e}", flush=True)


def rgb_stats(rgb_np: np.ndarray, name: str) -> float:
    nonblack = float((rgb_np.max(axis=-1) > 10).mean())
    print(f"[probe] {name}: mean={rgb_np.mean():.1f} std={rgb_np.std():.1f} "
          f"비검정비율={nonblack:.3f}", flush=True)
    return nonblack


def save_png(arr_u8: np.ndarray, path: str) -> None:
    try:
        from PIL import Image
        Image.fromarray(arr_u8).save(path)
        print(f"[probe] 저장 {path}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[probe] PNG 저장 실패({path}): {e}", flush=True)


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
vram("Kit 기동")

# env 네임스페이스 2개 (TiledCamera 다중 뷰 = 학습 경로 그대로)
prim_utils.create_prim("/World/envs/env_0", "Xform")
prim_utils.create_prim("/World/envs/env_1", "Xform")

# 대조용 메시 큐브 (cam1 시야 중앙 — depth/occlusion 판별체)
cube_cfg = sim_utils.CuboidCfg(
    size=(0.5, 0.5, 0.5),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.1)),
)
cube_cfg.func("/World/RefCube", cube_cfg, translation=CUBE)

# NuRec 볼륨 1개 참조
t0 = time.time()
add_reference_to_stage(usd_path=USDZ, prim_path="/World/Workplace")
for _ in range(240):
    simulation_app.update()
print(f"[probe] NuRec 1볼륨 로드 {time.time()-t0:.0f}s", flush=True)
vram("NuRec 1볼륨")

# TiledCamera 2뷰 (rgb + depth 동시 — 학습 계약과 동일 annotator 쌍)
camera_cfg = TiledCameraCfg(
    prim_path="/World/envs/env_.*/Cam",
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 80.0)),
    width=160, height=160,
    data_types=["rgb", "distance_to_image_plane"],
    update_period=0.0,
    depth_clipping_behavior="none",
)
camera = TiledCamera(camera_cfg)
sim.reset()
print("[probe] sim.reset 완료 — TiledCamera 초기화됨", flush=True)

eyes = torch.tensor([
    [T[0] + 3.0, T[1] + 3.0, T[2] + 1.2],   # cam0: 밀도핵심 조망 (가우시안 전용 뷰)
    [CUBE[0] + 1.6, CUBE[1] + 1.6, CUBE[2] + 0.8],  # cam1: 큐브 중앙 + 가우시안 배경
], device=sim.device)
targets = torch.tensor([list(T), list(CUBE)], device=sim.device)
camera.set_world_poses_from_view(eyes, targets)

# RTX 워밍업 (첫 프레임 검정 함정)
for _ in range(90):
    sim.step()
    camera.update(dt=0.01)

rgb = camera.data.output["rgb"].cpu().numpy()          # [2,H,W,3|4] uint8
rgb = rgb[..., :3]
depth = camera.data.output["distance_to_image_plane"].cpu().numpy()  # [2,H,W,1]
depth = depth[..., 0]

print("=" * 60, flush=True)
print("[probe] ★검증① TiledCamera rgb에 NuRec 렌더 여부★", flush=True)
nb0 = rgb_stats(rgb[0], "cam0(가우시안 전용 뷰) rgb")
nb1 = rgb_stats(rgb[1], "cam1(큐브+가우시안 배경) rgb")
save_png(rgb[0], os.path.join(OUT, "cam0_rgb.png"))
save_png(rgb[1], os.path.join(OUT, "cam1_rgb.png"))
verdict1 = "렌더됨 ✓" if nb0 > 0.3 else ("부분/의심 ⚠️" if nb0 > 0.05 else "★렌더 안 됨 ✗★")
print(f"[probe] 판정①: cam0 비검정 {nb0:.3f} → {verdict1}", flush=True)

print("=" * 60, flush=True)
print("[probe] ★검증② depth annotator의 가우시안 기여★", flush=True)
for i, nm in enumerate(["cam0", "cam1"]):
    d = depth[i]
    finite = np.isfinite(d) & (d > 0)
    fr = float(finite.mean())
    if finite.any():
        vals = d[finite]
        print(f"[probe] {nm} depth: 유효비율={fr:.3f} min={vals.min():.2f} "
              f"med={np.median(vals):.2f} max={vals.max():.2f}", flush=True)
    else:
        print(f"[probe] {nm} depth: 유효 픽셀 0 (전부 inf/0)", flush=True)
    # depth 시각화 저장
    dv = np.where(np.isfinite(d), d, np.nan)
    lo, hi = np.nanpercentile(dv, 5), np.nanpercentile(dv, 95)
    if np.isfinite(lo) and hi > lo:
        vis = np.clip((dv - lo) / (hi - lo), 0, 1)
        vis = np.nan_to_num(vis, nan=1.0)
        save_png((vis * 255).astype(np.uint8), os.path.join(OUT, f"{nm}_depth.png"))
c = depth[1][70:90, 70:90]  # cam1 중앙 20x20 = 큐브 예상 (~2.4m)
cf = c[np.isfinite(c) & (c > 0)]
print(f"[probe] cam1 중앙 패치(큐브 예상): 유효 {cf.size}/400, "
      f"med={np.median(cf):.2f}m (기대 ~2.4m)" if cf.size else
      "[probe] cam1 중앙 패치: 유효 0 — 큐브 depth 미검출 ⚠️", flush=True)

print("=" * 60, flush=True)
print("[probe] ★검증③ xformOp 이동/스케일 반영 여부★", flush=True)
import omni.usd
from pxr import UsdGeom

stage = omni.usd.get_context().get_stage()
wp = stage.GetPrimAtPath("/World/Workplace")
api = UsdGeom.XformCommonAPI(wp)
api.SetTranslate((0.0, 0.0, 5.0))
for _ in range(60):
    sim.step()
    camera.update(dt=0.01)
rgb_t = camera.data.output["rgb"].cpu().numpy()[..., :3]
diff_t = float(np.abs(rgb_t[0].astype(np.int16) - rgb[0].astype(np.int16)).mean())
save_png(rgb_t[0], os.path.join(OUT, "cam0_rgb_translated.png"))
print(f"[probe] translate(0,0,+5) 후 cam0 평균 픽셀 변화 = {diff_t:.1f} "
      f"→ {'변환 반영 ✓' if diff_t > 5 else '★미반영(useProxyTransform 이슈) ✗★'}", flush=True)

api.SetScale((0.5, 0.5, 0.5))
for _ in range(60):
    sim.step()
    camera.update(dt=0.01)
rgb_s = camera.data.output["rgb"].cpu().numpy()[..., :3]
diff_s = float(np.abs(rgb_s[0].astype(np.int16) - rgb_t[0].astype(np.int16)).mean())
save_png(rgb_s[0], os.path.join(OUT, "cam0_rgb_scaled.png"))
print(f"[probe] scale(0.5) 후 cam0 평균 픽셀 변화 = {diff_s:.1f} "
      f"→ {'스케일 반영 ✓' if diff_s > 5 else '★미반영 ✗★'}", flush=True)
api.SetTranslate((0.0, 0.0, 0.0))
api.SetScale((1.0, 1.0, 1.0))

print("=" * 60, flush=True)
print("[probe] ★검증④ 2번째 볼륨 참조 VRAM 증분★", flush=True)
vram("2뷰 렌더 완료(1볼륨)")
add_reference_to_stage(usd_path=USDZ, prim_path="/World/Workplace2")
for _ in range(180):
    simulation_app.update()
vram("NuRec 2볼륨")

print("[probe] ★프로브 완료★ 산출물 → " + OUT, flush=True)
simulation_app.close()
