#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★D435i 위치 GUI 배치 도구★ — 사용자가 GUI에서 직접 카메라 위치를 잡으면 좌표를 기록한다.

사용법 ([사람] 실행):
  1) python scripts\\tools\\place_d435i_gui.py   (GUI로 뜸 — --headless 금지)
  2) 뷰포트에서 로봇이 홈 포즈로 서 있고, ★/World/D435i_Ghost★ 프림(초록 하이라이트
     가능한 별도 D435i 본체)이 현재 공칭 위치에 겹쳐 있다.
  3) Stage 트리에서 /World/D435i_Ghost 선택 → Move(W)/Rotate(E) 기즈모로 원하는
     위치·방향으로 이동. ★로봇에 구워진 본체는 움직이지 말 것(안 움직여짐)★ —
     고스트만 옮기면 된다.
  4) 콘솔에 1초마다 wrist_3(=flange) 기준 상대 좌표가 출력되고, 같은 내용이
     Phase1/logs/d435i_ghost_pose.yaml 에 계속 저장된다.
  5) 다 되면 창을 닫는다 → 마지막 포즈가 남는다. Claude에게 "배치 완료"라고 알리면
     그 값을 URDF(bracket_to_camera)+configs/calib/hand_eye.yaml에 굽고 재변환한다.

⚠️ 학습/프로브가 GPU를 쓰는 동안 실행 금지(VRAM 충돌 — 순차 실행).
⚠️ isaaclab import → 프리앰블/AppLauncher 이후에만.
"""
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()



_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

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

parser = argparse.ArgumentParser(description="D435i 위치 GUI 배치 도구.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.headless:
    raise SystemExit("[place] ⚠️ 이 도구는 GUI 전용 — --headless 없이 실행하세요.")
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401

from common.calibration import load_hand_eye, mat_to_quat
from common.config_io import configs_dir

_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"
_OUT = os.path.join(_REPO_ROOT, "Phase1", "logs", "d435i_ghost_pose.yaml")


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=1)
    env_cfg.seed = 42
    env_cfg.viewer.eye = (0.75, -0.45, 0.65)
    env_cfg.viewer.lookat = (0.49, 0.13, 0.35)
    # ★이 세션에선 런타임 D435i 본체를 아예 스폰하지 않는다★ — 고스트가 유일한 D435i
    #   (겹침 원천 차단, 사용자 요청). None이면 InteractiveScene이 스킵(검증됨).
    if hasattr(env_cfg.scene, "d435i_body"):
        env_cfg.scene.d435i_body = None
    env = gym.make(_TASK, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    for _ in range(5):
        with torch.inference_mode():
            env.step(torch.zeros(1, 7, device=uenv.device))

    robot = uenv.scene["robot"]
    w_idx = robot.body_names.index("wrist_3_link")

    # ── 고스트 스폰: 현재 공칭(hand_eye.yaml) 위치에서 시작 ──
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()

    # ── ★기존(로봇에 구워진) D435i 본체 숨김★ — 고스트와 겹쳐 혼동 방지(사용자 요청).
    #   merge_fixed_joints 변환은 실제 지오메트리를 ★wrist_3_link/visuals/ 밑★에 넣으므로
    #   경로 하나가 아니라 로봇 서브트리 전체에서 이름에 'd435'가 든 프림을 찾아 숨긴다
    #   (브라켓 'd435i_bracket*'은 배치 기준이라 제외). visibility는 서브트리에 상속.
    robot_root = stage.GetPrimAtPath("/World/envs/env_0/Robot")
    hidden = []
    if robot_root.IsValid():
        for prim in Usd.PrimRange(robot_root):
            nm = prim.GetName().lower()
            if "d435" in nm and "bracket" not in nm:
                try:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden.append(str(prim.GetPath()))
                except Exception as e:  # noqa: BLE001 — instance proxy 등은 건너뜀
                    print(f"[place] ⚠️ 숨김 실패 {prim.GetPath()}: {e}")
    if hidden:
        print(f"[place] 로봇에 구워진 D435i 프림 {len(hidden)}개 숨김 — 고스트만 표시됩니다:")
        for p in hidden:
            print(f"[place]   {p}")
    else:
        print("[place] ⚠️ 구워진 D435i 프림을 못 찾음 — 겹쳐 보이면 Stage에서 이름을 알려주세요.")

    ghost = stage.DefinePrim("/World/D435i_Ghost", "Xform")
    d435_usd = os.path.join(_REPO_ROOT, "assets", "d435i", "d435i.usd")
    ghost.GetReferences().AddReference(d435_usd.replace("\\", "/"))

    T_fc = load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")
    # hand_eye는 optical 관례 — 고스트(메시 프레임, 글라스=+y_mesh) 초기 포즈로 보정
    #   (scene.D435_MESH_IN_OPTICAL과 동일 상수. 기록 파일은 메시 프레임 그대로 저장
    #    — 굽기 시 Claude가 optical로 역변환).
    _MESH = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    wp = robot.data.body_pos_w[0, w_idx].cpu().numpy()
    wq = robot.data.body_quat_w[0, w_idx].cpu().numpy()
    Rw = _quat_to_mat(wq)
    g_pos = wp + Rw @ T_fc[:3, 3]
    g_rot = Rw @ T_fc[:3, :3] @ _MESH
    xf = UsdGeom.Xformable(ghost)
    xf.ClearXformOpOrder()
    m = Gf.Matrix4d(
        g_rot[0, 0], g_rot[1, 0], g_rot[2, 0], 0,
        g_rot[0, 1], g_rot[1, 1], g_rot[2, 1], 0,
        g_rot[0, 2], g_rot[1, 2], g_rot[2, 2], 0,
        float(g_pos[0]), float(g_pos[1]), float(g_pos[2]), 1,
    )
    xf.AddTransformOp().Set(m)
    print("[place] ★/World/D435i_Ghost 스폰(현 공칭 위치)★ — Stage에서 선택해 Move/Rotate로 옮기세요.")
    print(f"[place] 좌표 기록 파일: {_OUT} (1초마다 갱신, 창을 닫으면 마지막 값이 남음)")

    step = 0
    zero = torch.zeros(1, 7, device=uenv.device)
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(zero)
        step += 1
        if step % 60 != 0:
            continue
        # 고스트 world 포즈(USD 저작값 = 사용자가 드래그한 값)
        mw = UsdGeom.Xformable(ghost).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        Rg = np.array([[mw[0][0], mw[1][0], mw[2][0]],
                       [mw[0][1], mw[1][1], mw[2][1]],
                       [mw[0][2], mw[1][2], mw[2][2]]])
        pg = np.array([mw[3][0], mw[3][1], mw[3][2]])
        # wrist_3(물리) 기준 상대 = ★T_flange_cam 후보값★
        wp = robot.data.body_pos_w[0, w_idx].cpu().numpy()
        wq = robot.data.body_quat_w[0, w_idx].cpu().numpy()
        Rw = _quat_to_mat(wq)
        p_rel = Rw.T @ (pg - wp)
        R_rel = Rw.T @ Rg
        q_rel = mat_to_quat(R_rel)
        line = (f"[place] T_flange_cam 후보: pos=({p_rel[0]:+.4f},{p_rel[1]:+.4f},{p_rel[2]:+.4f}) "
                f"quat_wxyz=({q_rel[0]:+.4f},{q_rel[1]:+.4f},{q_rel[2]:+.4f},{q_rel[3]:+.4f})")
        print(line)
        rows = "\n".join("  - [" + ", ".join(f"{R_rel[i, j]:.10f}" for j in range(3)) + f", {p_rel[i]:.4f}]" for i in range(3))
        with open(_OUT, "w", encoding="utf-8") as f:
            f.write("# place_d435i_gui.py 기록 — wrist_3(flange) 기준 카메라 후보 포즈\n"
                    f"# (마지막 갱신 step={step})\n"
                    "T_flange_cam:\n" + rows + "\n  - [0.0, 0.0, 0.0, 1.0]\n")
    env.close()
    print(f"[place] 종료 — 마지막 포즈가 {_OUT} 에 저장됨. Claude에게 '배치 완료'라고 알려주세요.")


if __name__ == "__main__":
    main()
    simulation_app.close()
