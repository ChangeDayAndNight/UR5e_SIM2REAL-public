#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★브라켓 배치 GUI 도구★ — 캘리브된 D435i(고정) 옆에 브라켓 고스트를 사용자가 직접 맞춘다.

배경: hand_eye v6(실측, RMS 2.01mm)의 카메라 pose는 진실이라 ★고정★. 브라켓 메시(AI 생성)는
실물과 달라 자동 접촉 정합(v2)에도 잔차가 남음 → 사용자가 GUI로 브라켓만 직접 배치한다.

구성(고스트 전용 빈 스테이지 — 물리 없음, Play 금지):
  - /World/Robot_Ghost   : UR5e+TCW1+2FG7, ★홈 포즈★(검증된 FK) — 움직이지 말 것.
                           (로봇에 구워진 기존 브라켓은 자동 숨김 — 고스트가 유일한 브라켓)
  - /World/D435i_Fixed   : ★캘리브 pose의 D435i — 잠금★. 드래그해도 1초 내 원위치 스냅백.
  - /World/Bracket_Ghost : ★이걸 움직이세요★ — Move(W)/Rotate(E). 시작 위치 = 현 URDF 값.

사용법 ([사람] 실행 — GUI 필요, --headless 금지):
  1) python scripts\\tools\\place_bracket_gui.py
  2) Stage 트리에서 /World/Bracket_Ghost 선택 → 기즈모로 실물 장착과 같게 배치
     (카메라가 브라켓에 물리는 모습 기준 — 카메라는 고정이 진실).
  3) 콘솔·Phase1/logs/bracket_ghost_pose.yaml 에 1초마다 tool0(flange) 기준 브라켓 pose 저장.
  4) 창 닫고 Claude에게 "배치 완료" → URDF(tool0_to_bracket)에 굽고 bracket_to_camera는
     캘리브 카메라와의 차로 자동 산출(카메라 pose 불변) + USD 재변환 + 렌더 확인.

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
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
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

parser = argparse.ArgumentParser(description="브라켓 배치 GUI 도구 (카메라 고정·브라켓만 이동).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.headless:
    raise SystemExit("[place] ⚠️ 이 도구는 GUI 전용 — --headless 없이 실행하세요.")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import numpy as np

import urdf_fk  # 검증된 FK (tests/test_urdf_fk.py 골든)
from common.calibration import load_hand_eye
from common.config_io import configs_dir
from common.robot_spec import UR5E_HOME_JOINT_POS

_OUT = os.path.join(_REPO_ROOT, "Phase1", "logs", "bracket_ghost_pose.yaml")
_ROBOT_USD = os.path.join(_REPO_ROOT, "assets", "ur5e_2fg7", "ur5e_tcw1_2fg7.usd")
_URDF = os.path.join(_REPO_ROOT, "assets", "ur5e_2fg7", "ur5e_tcw1_2fg7.urdf")
_BRACKET_OBJ = os.path.join(_REPO_ROOT, "assets", "d435i", "D435i_Bracket.obj")
_D435_USD = os.path.join(_REPO_ROOT, "assets", "d435i", "d435i.usd")

# 현 URDF의 tool0_to_bracket (시작 위치) — 굽기 값과 동기 유지
_BRACKET_INIT_XYZ = (0.0, 0.0, -0.01233)
_BRACKET_INIT_YAW = 1.535890  # rad (88°)

# optical→mesh 보정 (scene.py D435_MESH_IN_OPTICAL)
_MESH = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

_MOVING_LINKS = (
    "shoulder_link", "upper_arm_link", "forearm_link",
    "wrist_1_link", "wrist_2_link", "wrist_3_link",
    "left_finger_link", "right_finger_link",
)


def _gf_to_np(m) -> np.ndarray:
    return np.array([[m[i][j] for j in range(4)] for i in range(4)]).T


def _np_to_gf(t: np.ndarray):
    from pxr import Gf

    return Gf.Matrix4d(
        t[0, 0], t[1, 0], t[2, 0], 0.0,
        t[0, 1], t[1, 1], t[2, 1], 0.0,
        t[0, 2], t[1, 2], t[2, 2], 0.0,
        t[0, 3], t[1, 3], t[2, 3], 1.0,
    )


def _set_link_pose(prim, t_np: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
    pos = t_np[:3, 3]
    w, x, y, z = urdf_fk.mat_to_quat_wxyz(t_np[:3, :3])
    top, oop = ops.get("xformOp:translate"), ops.get("xformOp:orient")
    if top is not None:
        top.Set(Gf.Vec3d(*map(float, pos)) if top.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
                else Gf.Vec3f(*map(float, pos)))
    if oop is not None:
        if oop.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            oop.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
        else:
            oop.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))


def _rpy_from_mat(R: np.ndarray):
    p = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])


def _load_obj_mesh(stage, path, obj_path, color=(0.75, 0.78, 0.82)):
    """OBJ를 UsdGeom.Mesh로 직접 생성(컨버터 불요 — 1125 정점 소형)."""
    from pxr import Gf, UsdGeom, Vt

    verts, counts, idx = [], [], []
    for line in open(obj_path, encoding="utf-8", errors="ignore"):
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            ids = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            counts.append(len(ids))
            idx.extend(ids)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def main() -> None:
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(3000.0)
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(600.0)

    # ── 로봇 고스트(홈 포즈) ──
    stage.DefinePrim("/World/Robot_Ghost", "Xform")
    arm = stage.DefinePrim("/World/Robot_Ghost/arm", "Xform")
    arm.GetReferences().AddReference(_ROBOT_USD.replace("\\", "/"))
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot_Ghost")):
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        except Exception:  # noqa: BLE001
            pass
    # 구워진 브라켓 숨김(고스트가 유일한 브라켓이 되도록)
    hidden = 0
    for prim in Usd.PrimRange(arm):
        if "bracket" in prim.GetName().lower():
            try:
                UsdGeom.Imageable(prim).MakeInvisible()
                hidden += 1
            except Exception:  # noqa: BLE001
                pass
    print(f"[place] 구워진 브라켓 프림 {hidden}개 숨김.")

    # FK 홈 포즈 저작 (검증: tests/test_urdf_fk.py 골든)
    joints = urdf_fk.load_joints(_URDF)
    poses = urdf_fk.fk(joints, dict(UR5E_HOME_JOINT_POS))
    base_prim = stage.GetPrimAtPath("/World/Robot_Ghost/arm/base_link")
    x_base = _gf_to_np(UsdGeom.Xformable(base_prim).GetLocalTransformation()) if base_prim.IsValid() else np.eye(4)
    for name in _MOVING_LINKS:
        p = stage.GetPrimAtPath(f"/World/Robot_Ghost/arm/{name}")
        if p.IsValid():
            _set_link_pose(p, x_base @ poses[name])
    T_w3 = x_base @ poses["wrist_3_link"]   # tool0 == wrist_3 (방향 항등 — 검증됨)
    print("[place] 로봇 고스트 = 홈 포즈. wrist_3(tool0) world z =", round(float(T_w3[2, 3]), 4))

    # ── D435i 고정(캘리브 pose — 잠금) ──
    T_fc = load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")
    T_cam = np.eye(4)
    T_cam[:3, :3] = T_w3[:3, :3] @ T_fc[:3, :3] @ _MESH
    T_cam[:3, 3] = T_w3[:3, 3] + T_w3[:3, :3] @ T_fc[:3, 3]
    cam = stage.DefinePrim("/World/D435i_Fixed", "Xform")
    cam.GetReferences().AddReference(_D435_USD.replace("\\", "/"))
    cxf = UsdGeom.Xformable(cam)
    cxf.ClearXformOpOrder()
    cam_op = cxf.AddTransformOp()
    cam_mat = _np_to_gf(T_cam)
    cam_op.Set(cam_mat)

    # ── 브라켓 고스트(이동 대상) — OBJ 직접 메시 ──
    ghost = stage.DefinePrim("/World/Bracket_Ghost", "Xform")
    _load_obj_mesh(stage, "/World/Bracket_Ghost/mesh", _BRACKET_OBJ)
    T_b0 = np.eye(4)
    cy, sy = math.cos(_BRACKET_INIT_YAW), math.sin(_BRACKET_INIT_YAW)
    T_b0[:3, :3] = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T_b0[:3, 3] = _BRACKET_INIT_XYZ
    T_bw = T_w3 @ T_b0
    gxf = UsdGeom.Xformable(ghost)
    gxf.ClearXformOpOrder()
    gxf.AddTransformOp().Set(_np_to_gf(T_bw))

    try:
        from isaacsim.core.utils.viewports import set_camera_view

        c = T_cam[:3, 3]
        set_camera_view(eye=[float(c[0]) + 0.35, float(c[1]) - 0.35, float(c[2]) + 0.2],
                        target=[float(c[0]), float(c[1]), float(c[2])])
    except Exception:  # noqa: BLE001
        pass

    print("[place] ★배치 대상 = /World/Bracket_Ghost★ (Move W / Rotate E).")
    print("[place]   D435i_Fixed는 캘리브 pose 잠금 — 옮겨도 1초 내 원위치 스냅백.")
    print(f"[place] 좌표 기록: {_OUT} (1초마다, tool0 기준)")
    print("[place] ─────────────────────────────────────────────")

    T_w3_inv = np.linalg.inv(T_w3)
    step = 0
    while simulation_app.is_running():
        simulation_app.update()
        step += 1
        if step % 60 != 0:
            continue
        # 카메라 스냅백(잠금 보장)
        cam_op.Set(cam_mat)
        # 브라켓 pose(tool0 기준) 기록
        mw = UsdGeom.Xformable(ghost).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        T_bw_now = _gf_to_np(mw)
        T_tb = T_w3_inv @ T_bw_now
        # 직교화(기즈모 스케일 오염 방지)
        U, _, Vt = np.linalg.svd(T_tb[:3, :3])
        T_tb[:3, :3] = U @ Vt
        r, p, y = _rpy_from_mat(T_tb[:3, :3])
        t = T_tb[:3, 3]
        print(f"[place] tool0→bracket: xyz=({t[0]:+.5f},{t[1]:+.5f},{t[2]:+.5f}) "
              f"rpy=({math.degrees(r):+.2f},{math.degrees(p):+.2f},{math.degrees(y):+.2f})°")
        with open(_OUT, "w", encoding="utf-8") as f:
            f.write(f"# place_bracket_gui.py 기록 — tool0(flange) 기준 브라켓 pose (step={step})\n")
            f.write("# Claude가 URDF tool0_to_bracket에 굽고, bracket_to_camera는 캘리브 카메라(불변)와의\n")
            f.write("# 차로 자동 산출한다.\n")
            f.write(f"xyz: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n")
            f.write(f"rpy_rad: [{r:.6f}, {p:.6f}, {y:.6f}]\n")
            f.write(f"rpy_deg: [{math.degrees(r):.3f}, {math.degrees(p):.3f}, {math.degrees(y):.3f}]\n")
    print(f"[place] 종료 — 마지막 pose가 {_OUT} 에 저장됨. Claude에게 '배치 완료'라고 알려주세요.")


if __name__ == "__main__":
    main()
    simulation_app.close()
