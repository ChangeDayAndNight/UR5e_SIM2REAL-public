"""학습 환경(Vision) 렌더 — 표준 scene 조망 + 정책 POV (RGB-D 기동 검증용).

목적: BinPickVisionEnvCfg를 실제로 부팅해
      ①조망 뷰(로봇·벨트·큐브)
      ②wrist 카메라 POV(정책이 먹는 RGB 112² + depth — 원본/×4 확대) 를 저장한다.
      부수 검증: RGB-D 관측(50,206D)·rgb DR 경로의 e2e 기동.
실행: isaac51 python -u diagnostics/render_vision_env.py  (headless, GPU 단독 사용 —
      GUI 등 다른 Kit 앱과 동시 실행 금지: RTX 검정 프레임 함정)
산출: Phase2/Result/diagnostics/vision_train_env_render/*.png + summary 콘솔([render] 태그)
"""
import sys

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
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless", "--enable_cameras"])
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Kit 기동 이후에만 무거운 import ----
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCameraCfg

import ur5e_binpick.tasks  # noqa: F401  (gym 등록 + vision_policy 주입)
from ur5e_binpick.tasks.binpick.binpick_env_cfg import BinPickVisionEnvCfg

OUT = _ROOT / "Phase2" / "Result" / "diagnostics" / "vision_train_env_render"
OUT.mkdir(parents=True, exist_ok=True)


def save_png(arr_u8: np.ndarray, name: str) -> None:
    from PIL import Image

    Image.fromarray(arr_u8).save(str(OUT / name))
    print(f"[render] 저장 {OUT / name}", flush=True)


print("[render] ★Vision env 부팅★", flush=True)
cfg = BinPickVisionEnvCfg()
cfg.scene.num_envs = 1  # 렌더 전용 — post_init 기본(4)을 축소 (VRAM·속도)
# 조망 카메라(렌더 전용 추가 — 학습 cfg에는 없음): 뷰별로 set_world_poses_from_view 재배치
cfg.scene.overview_cam = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/OverviewCam",
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 80.0)),
    width=800, height=600,
    data_types=["rgb"],
    update_period=0.0,
)
env = ManagerBasedRLEnv(cfg)
obs, _ = env.reset()
flat_dim = obs["policy"].shape[-1]
print(f"[render] env 기동 OK — policy obs = {tuple(obs['policy'].shape)} "
      f"({'★50,206D RGB-D 계약 일치★' if flat_dim == 50206 else '⚠️ 계약 불일치'})", flush=True)

act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for _ in range(25):  # settle + RTX/NuRec 워밍업 (첫 프레임 검정 함정)
    env.step(act)

# ── ① 정책 POV (wrist 112² RGB-D — 학습이 먹는 그대로) ──
wrist = env.scene.sensors["tiled_camera"]
rgb112 = wrist.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
d112 = wrist.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
save_png(rgb112, "wrist_rgb_112.png")
save_png(np.kron(rgb112, np.ones((4, 4, 1), dtype=np.uint8)), "wrist_rgb_x4.png")
dv = np.where(np.isfinite(d112) & (d112 > 0), d112, np.nan)
lo, hi = np.nanpercentile(dv, 5), np.nanpercentile(dv, 95)
if np.isfinite(lo) and hi > lo:
    vis = np.nan_to_num(np.clip((dv - lo) / (hi - lo), 0, 1), nan=1.0)
    save_png(np.kron((vis * 255).astype(np.uint8), np.ones((4, 4), dtype=np.uint8)),
             "wrist_depth_x4.png")
finite = np.isfinite(d112) & (d112 > 0)
print(f"[render] wrist depth: 유효 {finite.mean():.2f} med "
      f"{np.median(d112[finite]) if finite.any() else float('nan'):.3f}m | "
      f"rgb 비검정 {(rgb112.max(axis=-1) > 10).mean():.2f}", flush=True)

# ── ② 조망 뷰 (정합 확증: 고스트가 아닌 '진짜 학습 씬' + 스캔 배경 on/off 쌍) ──
# ⚠️ set_world_poses_from_view는 Fabric worldMatrix에 기록(xform_prim_view.py
#   L226-244)되어 RTX 렌더 프로덕트 카메라에 전파되지 않았다(실측: 5뷰 픽셀차
#   0.08 = 미이동). → ★USD xformOp 직접 저작★(이 프로젝트 검증 경로 — 배치 GUI·
#   카메라 DR과 동일 메커니즘)으로 대체.
import omni.usd
from pxr import Gf, UsdGeom

from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

ov = env.scene.sensors["overview_cam"]
stage = omni.usd.get_context().get_stage()
cam_prim = stage.GetPrimAtPath("/World/envs/env_0/OverviewCam")
nurec_prim = stage.GetPrimAtPath("/World/envs/env_0/NuRecBackground")


def _aim_overview(eye, tgt) -> None:
    """카메라 prim의 USD xformOp를 직접 저작 (env_0 원점=월드라 local==world)."""
    R = create_rotation_matrix_from_view(
        torch.tensor([list(eye)], dtype=torch.float32),
        torch.tensor([list(tgt)], dtype=torch.float32),
        "Z", device="cpu",
    )
    w, x, y, z = (float(v) for v in quat_from_matrix(R)[0])
    xf = UsdGeom.Xformable(cam_prim)
    ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
    top = ops.get("xformOp:translate") or xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    oop = ops.get("xformOp:orient") or xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    if top.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        top.Set(Gf.Vec3d(*[float(v) for v in eye]))
    else:
        top.Set(Gf.Vec3f(*[float(v) for v in eye]))
    if oop.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        oop.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        oop.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


views = [
    ("front_high",  (1.9, -1.7, 1.3), (0.45, 0.0, 0.10)),
    ("side_left",   (-1.2, 1.8, 1.0), (0.45, 0.0, 0.10)),
    ("belt_close",  (0.95, 0.75, 0.55), (0.45, 0.0, 0.05)),
    ("top_down",    (0.45, 0.05, 2.6), (0.45, 0.0, 0.0)),
    ("wrist_like",  (0.75, -0.45, 0.55), (0.45, 0.05, 0.0)),
]
for name, eye, tgt in views:
    _aim_overview(eye, tgt)
    for _ in range(10):  # 뷰 전환 후 RTX 누적 워밍업
        env.step(act)
    frame = ov.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    nb = float((frame.max(axis=-1) > 10).mean())
    save_png(frame, f"overview_{name}.png")
    # ★정합 확증 쌍★: NuRec 숨김 → 같은 뷰의 메시 전용 프레임(오버레이 대조)
    if nurec_prim.IsValid():
        UsdGeom.Imageable(nurec_prim).MakeInvisible()
        for _ in range(6):
            env.step(act)
        mesh_only = ov.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        save_png(mesh_only, f"overview_{name}_meshonly.png")
        UsdGeom.Imageable(nurec_prim).MakeVisible()
        diff = float(np.abs(frame.astype(np.int16) - mesh_only.astype(np.int16)).mean())
        print(f"[render] {name}: 비검정 {nb:.2f} | NuRec 기여(픽셀차) {diff:.1f}", flush=True)
    else:
        print(f"[render] {name}: 비검정 {nb:.2f} (NuRec prim 미발견 ⚠️)", flush=True)

print(f"[render] ★완료★ → {OUT}", flush=True)
env.close()
simulation_app.close()
os._exit(0)  # Kit close() 행 방지 (Grasp_PhysicsTest force-exit 패턴)
