"""★홈 포즈별 손목 카메라 시야 렌더 비교★ — 현행 / 사용자 새 홈 / 기하 최적 후보.

목적(2026-07-28 사용자 요청): "홈 포즈가 목표를 제대로 관측하지 못하는데 학습해도 되나?"에
      기하 계산이 아니라 ★실제 렌더 이미지★로 답한다. 세 자세를 동일 조건(같은 큐브 위치·
      같은 조명·같은 env)에서 부팅해 정책이 먹는 wrist RGB-D와 조망 뷰를 나란히 저장한다.

배경: 이 프로젝트는 '홈 자세 큐브 가시율 18.3% → student eval grasp 0.8%(접근 불능)' 실패
      이력이 있다(configs/common.yaml frame_mode 주석). 홈 가시율은 증류 성패의 구조적 변수다.

실행 (isaac51, headless — ⚠️GUI 등 다른 Kit 앱과 동시 실행 금지: RTX 검정 프레임 함정):
  python diagnostics/home_pose_camera_probe.py
산출: Phase2/Result/diagnostics/home_pose_probe/
        <pose>_wrist_rgb.png / _wrist_rgb_x4.png / _wrist_depth_x4.png / _overview.png
      + 콘솔 요약([probe] 태그): depth 유효율·중앙값, 큐브 픽셀 수(가시성 정량)

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import한다.
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
try:
    # ★2026-08-10(F54) h5py 선로드 제거★ — 실측상 ★이 import가 Isaac 센서 DLL을 깨뜨린다★.
    #   h5py 3.16.0(HDF5 2.0.0)이 Isaac 번들 hdf5.dll(1.x)보다 먼저 로드되면
    #   generic_mo_io.dll이 WinError 127로 실패하고 GUI가 access violation으로 죽는다.
    #   (구 h5py에서는 반대로 작동해 워크어라운드였다 — 근거·측정 = Failure_Check F54)
    pass
except Exception:  # noqa: BLE001
    pass

import argparse
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

# ── Kit 기동 이후에만 무거운 import ──
import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import TiledCameraCfg

import ur5e_binpick.tasks  # noqa: F401  (gym 등록 + 정책 클래스 주입)
from common.config_io import configs_dir, load_yaml
from common.robot_spec import UR5E_ARM_JOINT_NAMES
from ur5e_binpick.tasks.binpick.binpick_env_cfg import BinPickVisionEvalEnvCfg

OUT = _ROOT / "Phase2" / "Result" / "diagnostics" / "home_pose_probe"
OUT.mkdir(parents=True, exist_ok=True)

_SIM = load_yaml(configs_dir() / "sim.yaml")["scene"]
BELT_CX, BELT_CY = (float(v) for v in _SIM.get("belt_center_xy_m", [0.45, 0.0]))
OBJ = float(_SIM["object_size_m"])

# ── 비교 대상 3자세 [deg] (pan, lift, elbow, w1, w2, w3) ──
POSES = {
    "1_current": ("현행 홈 (검증된 자세)", (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)),
    "2_user_new": ("사용자 새 홈 (GUI 기록)", (0.0, -90.0, 0.0, -90.0, 0.0, 0.0)),
    "3_optimal": ("기하 최적 후보", (0.0, -90.0, 65.0, -45.0, -90.0, 0.0)),
}


def save_png(arr_u8: np.ndarray, name: str) -> None:
    from PIL import Image

    Image.fromarray(arr_u8).save(str(OUT / name))
    print(f"[probe] 저장 {OUT / name}", flush=True)


print("[probe] Vision-Eval env 부팅 (보조 off — 순수 관측 비교)", flush=True)
cfg = BinPickVisionEvalEnvCfg()
cfg.scene.num_envs = 1
# 조망 카메라(렌더 전용 — 학습 cfg에는 없음)
cfg.scene.overview_cam = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/OverviewCam",
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 80.0)),
    width=900, height=680, data_types=["rgb"], update_period=0.0,
)
env = ManagerBasedRLEnv(cfg)
obs, _ = env.reset()
print(f"[probe] env OK — policy obs {tuple(obs['policy'].shape)}", flush=True)

robot = env.scene["robot"]
wrist = env.scene.sensors["tiled_camera"]
ov = env.scene.sensors["overview_cam"]
act = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
arm_idx, arm_names = robot.find_joints(list(UR5E_ARM_JOINT_NAMES), preserve_order=True)
print(f"[probe] 팔 관절 인덱스 {arm_idx} ({arm_names})", flush=True)

# ── 조망 카메라 고정 배치(세 자세 동일 시점 — USD xformOp 직접 저작: 검증된 경로) ──
import omni.usd
from pxr import Gf, UsdGeom

from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

stage = omni.usd.get_context().get_stage()
cam_prim = stage.GetPrimAtPath("/World/envs/env_0/OverviewCam")


def aim_overview(eye, tgt) -> None:
    R = create_rotation_matrix_from_view(
        torch.tensor([list(eye)], dtype=torch.float32),
        torch.tensor([list(tgt)], dtype=torch.float32), "Z", device="cpu")
    w, x, y, z = (float(v) for v in quat_from_matrix(R)[0])
    xf = UsdGeom.Xformable(cam_prim)
    ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
    top = ops.get("xformOp:translate") or xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    oop = ops.get("xformOp:orient") or xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    top.Set(Gf.Vec3d(*[float(v) for v in eye]) if top.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
            else Gf.Vec3f(*[float(v) for v in eye]))
    oop.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)) if oop.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
            else Gf.Quatf(w, Gf.Vec3f(x, y, z)))


aim_overview((1.6, -1.5, 1.25), (0.40, 0.05, 0.20))


def place_cube_at_belt_center() -> None:
    """세 자세 비교의 공정성 — 큐브를 매번 같은 위치(벨트 중심)에 고정."""
    obj = env.scene["object_0"]
    st = obj.data.default_root_state.clone()
    st[:, :3] = env.scene.env_origins + torch.tensor(
        [BELT_CX, BELT_CY, OBJ / 2.0], device=env.device)
    st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    st[:, 7:] = 0.0
    obj.write_root_state_to_sim(st)


def set_pose(deg6) -> None:
    """팔 6관절을 지정 자세로 고정(상태 쓰기 + 목표 설정 — 액추에이터가 유지)."""
    q = robot.data.joint_pos.clone()
    qd = torch.zeros_like(q)
    q[:, arm_idx] = torch.tensor([math.radians(v) for v in deg6],
                                 device=env.device, dtype=q.dtype)
    robot.write_joint_state_to_sim(q, qd)
    robot.set_joint_position_target(q)


summary = []
for key, (label, deg6) in POSES.items():
    print(f"\n[probe] ── {key}: {label}  {deg6} ──", flush=True)
    set_pose(deg6)
    place_cube_at_belt_center()
    for _ in range(30):  # settle + RTX 누적 워밍업(첫 프레임 검정 함정)
        env.step(act)
        set_pose(deg6)              # IK 액션이 자세를 흘리지 않도록 매 스텝 고정
        place_cube_at_belt_center()  # 접촉으로 큐브가 밀리는 것 방지

    rgb = wrist.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    dep = wrist.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    save_png(rgb, f"{key}_wrist_rgb.png")
    save_png(np.kron(rgb, np.ones((4, 4, 1), dtype=np.uint8)), f"{key}_wrist_rgb_x4.png")
    dv = np.where(np.isfinite(dep) & (dep > 0), dep, np.nan)
    lo, hi = np.nanpercentile(dv, 5), np.nanpercentile(dv, 95)
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        vis = np.nan_to_num(np.clip((dv - lo) / (hi - lo), 0, 1), nan=1.0)
        save_png(np.kron((vis * 255).astype(np.uint8), np.ones((4, 4), dtype=np.uint8)),
                 f"{key}_wrist_depth_x4.png")

    frame = ov.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
    save_png(frame, f"{key}_overview.png")

    # ── 정량: 큐브 픽셀(파랑 계열: object_0 diffuse (0.2,0.4,0.8)) / 벨트 픽셀(초록) ──
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    cube_px = int(((b > r + 25) & (b > g + 15)).sum())
    belt_px = int(((g > r + 12) & (g > b + 6)).sum())
    fin = np.isfinite(dep) & (dep > 0)
    med = float(np.median(dep[fin])) if fin.any() else float("nan")
    tot = rgb.shape[0] * rgb.shape[1]
    summary.append((key, label, cube_px, 100.0 * belt_px / tot, 100.0 * fin.mean(), med))
    print(f"[probe]   큐브 픽셀 {cube_px} · 벨트 픽셀 {100.0*belt_px/tot:.1f}% · "
          f"depth 유효 {100.0*fin.mean():.1f}% · 중앙 {med:.3f} m", flush=True)

print("\n[probe] ═══════════ 요약 ═══════════", flush=True)
print("%-12s %-22s %-10s %-10s %-10s %s" % ("pose", "설명", "큐브px", "벨트%", "depth유효%", "중앙깊이"))
for key, label, cpx, bpct, fpct, med in summary:
    print("%-12s %-22s %-10d %-10.1f %-10.1f %.3f m" % (key, label, cpx, bpct, fpct, med), flush=True)
print(f"[probe] ★산출 위치★ {OUT}", flush=True)

env.close()
simulation_app.close()
