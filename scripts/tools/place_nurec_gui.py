#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★NuRec 정합 GUI 도구★ — 실셀 3DGS 스캔(Workplace.usdz)을 로봇 base frame에 배치·기록.

목적: NuRec 볼륨(/World/NuRec)을 ★기준 고스트★(원점 고정 로봇 + 벨트 프리미티브)에
겹치도록 Move/Rotate/Scale 하면, 그 similarity(T_base_nurec + scale)를
configs/calib/scene_registration.yaml 에 1초마다 라이브 기록한다(§4-5: 커밋 의무).

구성(물리 없음 — Play(▶) 누르지 말 것):
  - /World/Robot_Ghost : UR5e+TCW1+2FG7, ★원점 고정 = base frame 기준★ (움직이지 말 것)
  - /World/Belt_Ghost  : 파란 벨트(중심 x=0.45,y=0, 상면 z=0) + 빨간 +X 막대 (움직이지 말 것)
  - /World/NuRec       : Workplace.usdz 참조 — ★이것만★ 기즈모로 조작한다
  - "UR5e 관절" 패널    : 스캔 속 로봇 자세에 맞추는 FK 슬라이더(검증된 urdf_fk)

사용법 ([사람] 실행 — GUI 필요, --headless 금지):
  1) python scripts\\tools\\place_nurec_gui.py
  2) Stage 트리에서 ★/World/NuRec 루트★를 선택 → Move(W)/Rotate(E)/Scale(R)로
     스캔 속 로봇·벨트가 고스트에 겹치게 맞춘다. ⚠️ 하위 Volume 프림이 아니라
     ★루트★를 조작해야 xformOp가 볼륨 렌더에 반영된다(프로브 실증).
  3) 스캔 속 로봇 자세가 홈이 아니면 "UR5e 관절" 슬라이더로 고스트를 맞춘다.
  4) 콘솔·configs/calib/scene_registration.yaml 에 1초마다 저장(마지막 값 유지).
  5) 다 되면 창 닫고 Claude에게 "정합 완료" → (선택) register_nurec_umeyama.py 로
     수치 미세보정 → yaml 커밋.

⚠️ 학습/프로브가 GPU를 쓰는 동안 실행 금지(VRAM 충돌 — NuRec 3M 가우시안).
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

parser = argparse.ArgumentParser(description="NuRec 정합 GUI 도구 (고스트 전용 스테이지).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.headless:
    raise SystemExit("[place] ⚠️ 이 도구는 GUI 전용 — --headless 없이 실행하세요.")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
from datetime import date

import numpy as np
import yaml

import urdf_fk  # scripts/tools (프리앰블이 sys.path 보장) — FK는 pytest 골든으로 검증됨
from common.calibration import load_scene_registration, mat_to_quat, save_scene_registration
from common.config_io import configs_dir, load_yaml
from common.robot_spec import UR5E_HOME_JOINT_POS

_SIM = load_yaml(configs_dir() / "sim.yaml")
BELT_SIZE = tuple(float(v) for v in _SIM["scene"]["belt_size_m"])
TABLE_HEIGHT = float(_SIM["scene"]["table_height_m"])
_STAND_H = 0.80  # scene.py STAND_HEIGHT (시각 참고)

_BELT_CX, _BELT_CY = 0.45, 0.0  # scene.py BELT_CENTER 공칭 (기준 고스트 — 고정)
_DENSITY_CORE = (-1.0, -1.0, 1.0)  # usdz 밀도 핵심 (nurec_tiled_probe 검증값)

_OUT = os.path.join(_REPO_ROOT, "configs", "calib", "scene_registration.yaml")
# ★전 객체 스냅샷(2026-07-17 사용자 요청 "GUI 모든 객체 기록")★ — 감사/재현 기록.
#   ⚠️ 소비 금지: 학습이 읽는 원본은 sim.yaml(벨트)·scene_registration.yaml(NuRec)·
#   robot_spec(홈/원점) — 이 파일은 GUI 세션의 상태 대장(ledger)이다(이중 소스 방지).
_OUT_LAYOUT = os.path.join(_REPO_ROOT, "configs", "calib", "scene_layout.yaml")
_NUREC_USDZ = os.path.join(_REPO_ROOT, "assets", "Scene", "Workplace.usdz")
_ROBOT_USD = os.path.join(_REPO_ROOT, "assets", "ur5e_2fg7", "ur5e_tcw1_2fg7.usd")
_URDF = os.path.join(_REPO_ROOT, "assets", "ur5e_2fg7", "ur5e_tcw1_2fg7.urdf")

# USD(평면 계층)에 존재하는 가동 링크 — FK 결과를 저작할 대상
_MOVING_LINKS = (
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "left_finger_link",
    "right_finger_link",
)


def _add_box(stage, path, half, color, opacity, local_pos):
    from pxr import Gf, UsdGeom, Vt

    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*local_pos))
    xf.AddScaleOp().Set(Gf.Vec3f(*half))
    cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    cube.CreateDisplayOpacityAttr(Vt.FloatArray([opacity]))
    return cube


def _gf_to_np(m) -> np.ndarray:
    """Gf.Matrix4d(row-vector 관례) → np 4x4(column-vector 관례)."""
    t = np.array([[m[i][j] for j in range(4)] for i in range(4)])
    return t.T


def _set_link_pose(prim, t_np: np.ndarray) -> None:
    """링크 prim의 translate/orient op에 4x4 pose 저작(기존 op 순서·정밀도 유지)."""
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(prim)
    ops = {o.GetOpName(): o for o in xf.GetOrderedXformOps()}
    pos = t_np[:3, 3]
    w, x, y, z = urdf_fk.mat_to_quat_wxyz(t_np[:3, :3])
    top = ops.get("xformOp:translate")
    oop = ops.get("xformOp:orient")
    if top is not None:
        if top.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            top.Set(Gf.Vec3d(*[float(v) for v in pos]))
        else:
            top.Set(Gf.Vec3f(*[float(v) for v in pos]))
    if oop is not None:
        if oop.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            oop.Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
        else:
            oop.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))


def _decompose_similarity(m) -> tuple[np.ndarray, float, float, np.ndarray]:
    """local-to-world Gf.Matrix4d → (T_base_nurec 강체부 4x4, scale, 비균일도, scale 3벡터).

    column-vector 관례에서 M = T·R·S 이므로 각 열의 노름 = 축별 스케일.
    scale = 3축 평균(균일 가정), 비균일도 = max|s_i - mean| / mean.
    """
    M = _gf_to_np(m)
    Rs = M[:3, :3]
    s3 = np.array([np.linalg.norm(Rs[:, i]) for i in range(3)])
    s_mean = float(s3.mean()) or 1.0
    nonuni = float(np.abs(s3 - s_mean).max() / s_mean)
    R = Rs / np.where(s3 < 1e-12, 1.0, s3)[np.newaxis, :]
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = M[:3, 3]
    return T, s_mean, nonuni, s3


def main() -> None:
    import omni.ui as ui
    import omni.usd
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")

    # ── 조명 + 바닥(방향감) ──
    UsdLux.DistantLight.Define(stage, "/World/Sun").CreateIntensityAttr(3000.0)
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(600.0)
    _add_box(stage, "/World/ground", (3.0, 3.0, 0.01), (0.42, 0.44, 0.47), 1.0,
             (0.3, 0.0, TABLE_HEIGHT - _STAND_H - 0.01))

    # ── 기준 고스트 ①: 로봇(원점 고정 = base frame) — 움직이지 말 것 ──
    robot_g = stage.DefinePrim("/World/Robot_Ghost", "Xform")
    arm = stage.DefinePrim("/World/Robot_Ghost/arm", "Xform")
    arm.GetReferences().AddReference(_ROBOT_USD.replace("\\", "/"))
    rxf = UsdGeom.Xformable(robot_g)
    rxf.ClearXformOpOrder()
    rxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    # 안전: Play를 눌러도 낙하/폭발하지 않게 물리 비활성(표시 무영향 — 시뮬 안 함)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot_Ghost")):
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        except Exception:  # noqa: BLE001
            pass
    # 로봇 스탠드(자식 — 시각 참고)
    _add_box(stage, "/World/Robot_Ghost/stand", (0.11, 0.11, _STAND_H / 2), (0.62, 0.63, 0.66), 1.0,
             (0.0, 0.0, TABLE_HEIGHT - _STAND_H / 2))

    # ── FK 저작 준비: 링크 prim 맵 + base_link 기준변환 (place_workspace_gui 검증 패턴) ──
    joints = urdf_fk.load_joints(_URDF)
    link_prims = {}
    for name in _MOVING_LINKS:
        p = stage.GetPrimAtPath(f"/World/Robot_Ghost/arm/{name}")
        if not p.IsValid():  # 폴백: 서브트리에서 이름 검색
            for cand in Usd.PrimRange(arm):
                if cand.GetName() == name and cand.GetTypeName() == "Xform":
                    p = cand
                    break
        if p.IsValid():
            link_prims[name] = p
        else:
            print(f"[place] ⚠️ 링크 prim 미발견: {name} — 이 링크는 관절 저작에서 제외")
    base_prim = stage.GetPrimAtPath("/World/Robot_Ghost/arm/base_link")
    x_base = np.eye(4)
    if base_prim.IsValid():
        x_base = _gf_to_np(UsdGeom.Xformable(base_prim).GetLocalTransformation())

    # ── 관절 슬라이더 패널 (검증된 FK → 링크 재저작): 스캔 속 로봇 자세 맞추기용 ──
    arm_models = {}
    grip_model = None

    def _apply_joints(_m=None) -> None:
        q = {n: math.radians(m.get_value_as_float()) for n, m in arm_models.items()}
        if grip_model is not None:
            q["left_finger_joint"] = grip_model.get_value_as_float() / 1000.0  # mm→m (right=mimic)
        poses = urdf_fk.fk(joints, q)
        for name, prim in link_prims.items():
            _set_link_pose(prim, x_base @ poses[name])

    def _set_all(vals_deg: dict[str, float]) -> None:
        for n, m in arm_models.items():
            m.set_value(float(vals_deg.get(n, 0.0)))
        _apply_joints()

    _LIMITS_DEG = {j.name: (math.degrees(j.lower), math.degrees(j.upper))
                   for j in joints if j.name in urdf_fk.ARM_JOINT_NAMES}
    win = ui.Window("UR5e 관절 (Ghost)", width=400, height=340,
                    dockPreference=ui.DockPreference.RIGHT_BOTTOM)
    with win.frame:
        with ui.VStack(spacing=4, style={"margin": 4}):
            ui.Label("★스캔 속 로봇 자세에 고스트를 맞추는 슬라이더★", height=18)
            for jn in urdf_fk.ARM_JOINT_NAMES:
                lo, hi = _LIMITS_DEG.get(jn, (-360.0, 360.0))
                with ui.HStack(height=22, spacing=6):
                    ui.Label(jn.replace("_joint", ""), width=110)
                    m = ui.SimpleFloatModel(0.0)
                    ui.FloatSlider(model=m, min=lo, max=hi, step=1.0)
                    arm_models[jn] = m
                    m.add_value_changed_fn(_apply_joints)
            with ui.HStack(height=22, spacing=6):
                ui.Label("gripper close[mm]", width=110)
                grip_model = ui.SimpleFloatModel(0.0)
                ui.FloatSlider(model=grip_model, min=0.0, max=19.0, step=0.5)
                grip_model.add_value_changed_fn(_apply_joints)
            with ui.HStack(height=26, spacing=8):
                ui.Button("전부 0", clicked_fn=lambda: _set_all({}))
                ui.Button("홈 포즈", clicked_fn=lambda: _set_all(
                    {n: math.degrees(v) for n, v in UR5E_HOME_JOINT_POS.items()}))
    _set_all({n: math.degrees(v) for n, v in UR5E_HOME_JOINT_POS.items()})  # 초기 = 홈

    # ── 기준 고스트 ②: 벨트 프리미티브 (상면 z=0, 중심 0.45/0 — 고정, 움직이지 말 것) ──
    bx, by, bz = BELT_SIZE
    belt_g = stage.DefinePrim("/World/Belt_Ghost", "Xform")
    gxf = UsdGeom.Xformable(belt_g)
    gxf.ClearXformOpOrder()
    gxf.AddTranslateOp().Set(Gf.Vec3d(_BELT_CX, _BELT_CY, 0.0))
    _add_box(stage, "/World/Belt_Ghost/belt", (bx / 2, by / 2, bz / 2),
             (0.20, 0.45, 0.95), 0.5, (0.0, 0.0, TABLE_HEIGHT - bz / 2))
    _add_box(stage, "/World/Belt_Ghost/dir_x", (bx / 2 + 0.08, 0.012, 0.012),
             (0.95, 0.15, 0.15), 1.0, (bx / 2 + 0.05, 0.0, TABLE_HEIGHT + 0.03))

    # ── 정합 대상: /World/NuRec ← Workplace.usdz 참조 (기즈모 조작 대상) ──
    nurec = stage.DefinePrim("/World/NuRec", "Xform")
    add_reference_to_stage(usd_path=_NUREC_USDZ.replace("\\", "/"), prim_path="/World/NuRec")
    # 초기 xform = 기존 scene_registration.yaml (placeholder 항등 포함 — 이어서 수정 가능)
    T0, scale0 = np.eye(4), 1.0
    try:
        T0, scale0 = load_scene_registration(_OUT)
    except Exception as e:  # noqa: BLE001
        print(f"[place] ⚠️ 기존 scene_registration.yaml 로드 실패(항등에서 시작): {e}")
    q0 = mat_to_quat(T0[:3, :3])
    nxf = UsdGeom.Xformable(nurec)
    nxf.ClearXformOpOrder()
    nxf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in T0[:3, 3]]))
    nxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(float(q0[0]), Gf.Vec3d(float(q0[1]), float(q0[2]), float(q0[3]))))
    nxf.AddScaleOp().Set(Gf.Vec3f(float(scale0), float(scale0), float(scale0)))

    # ── 초기 카메라: 밀도 핵심 조망 (원점만 보면 스캔이 빈 것처럼 보이는 함정 방지) ──
    try:
        from isaacsim.core.utils.viewports import set_camera_view

        cx, cy, cz = _DENSITY_CORE
        set_camera_view(eye=[cx + 2.5, cy - 2.0, cz + 1.5], target=[cx, cy, cz])
    except Exception:  # noqa: BLE001
        pass

    print("[place] ★NuRec 정합 스테이지★ — 원점 고스트(로봇+벨트) vs /World/NuRec. Play(▶) 금지.")
    print("[place] ⚠️ 기즈모는 ★/World/NuRec 루트★에만 — 하위 Volume 프림을 직접 옮기면")
    print("[place]    기록되는 xform과 어긋난다(볼륨 변환은 참조 루트 xformOp로 반영 — 프로브 실증).")
    print("[place] ⚠️ Robot_Ghost/Belt_Ghost 는 base frame 기준이므로 움직이지 말 것.")
    print(f"[place] 프레이밍 힌트: usdz 밀도 핵심은 원점이 아니라 {_DENSITY_CORE} 부근이고")
    print("[place]    extent 는 ±수십 m(스케일 미정). 시작 카메라가 밀도 핵심을 보고 있다 —")
    print("[place]    스캔이 거대하게 보이면 Scale 을 먼저 크게 줄이고(예: 0.5→0.1 단위로),")
    print("[place]    Move 로 스캔 속 로봇 베이스를 월드 원점 고스트에 끌어온 뒤 미세 조정한다.")
    print(f"[place] 좌표 기록: {_OUT} (1초마다 갱신, 창 닫으면 마지막 값 유지 — §4-5 커밋 의무)")
    print("[place] ─────────────────────────────────────────────")

    # ── 전 객체 스냅샷 기록기(2026-07-17 사용자 요청) ──────────────────────────
    def _pose_of(prim) -> tuple[list[float], list[float], float]:
        """prim local-to-world → (pos[m], quat wxyz, 병진 노름[mm] — 원점 대비)."""
        mw = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        Tp, sp, _, _ = _decompose_similarity(mw)
        pos = [float(v) for v in Tp[:3, 3]]
        quat = [float(v) for v in mat_to_quat(Tp[:3, :3])]
        return pos, quat, sp

    def _write_scene_layout(T_n, s_n, moved_refs: list[str]) -> None:
        """GUI 화면의 모든 객체 좌표·회전·스케일을 scene_layout.yaml에 스냅샷.

        학습 소비 원본이 아니라 감사 기록(원본: sim.yaml/scene_registration.yaml/
        robot_spec). 기준 고스트(로봇·벨트)가 스폰 기준에서 이탈했으면 moved_refs에
        기록 — 학습에는 반영되지 않으므로 명시적 경고 대상.
        """
        r_pos, r_quat, r_s = _pose_of(robot_g)
        b_pos, b_quat, b_s = _pose_of(belt_g)
        joints_rad = {n: round(math.radians(m.get_value_as_float()), 6)
                      for n, m in arm_models.items()}
        grip_m = (grip_model.get_value_as_float() / 1000.0) if grip_model else 0.0
        data = {
            "_readme": ("GUI 전 객체 스냅샷 — 감사/재현 기록. ★학습 소비 금지★: "
                        "원본은 sim.yaml(벨트)·scene_registration.yaml(NuRec)·"
                        "robot_spec(홈·원점). 기준 고스트 이탈은 moved_reference_ghosts 참조."),
            "objects": {
                "robot_ghost": {
                    "pos_m": [round(v, 6) for v in r_pos],
                    "quat_wxyz": [round(v, 6) for v in r_quat],
                    "scale": round(r_s, 6),
                    "joints_rad": joints_rad,      # FK 슬라이더(스캔 속 로봇 자세 매칭값)
                    "gripper_close_m": round(grip_m, 6),
                    "reference": "base 원점 고정(기준 앵커) — 학습 로봇 base와 동일",
                },
                "belt_ghost": {
                    "pos_m": [round(v, 6) for v in b_pos],
                    "quat_wxyz": [round(v, 6) for v in b_quat],
                    "scale": round(b_s, 6),
                    "size_m": list(BELT_SIZE),
                    "reference": f"sim.yaml 기준({_BELT_CX},{_BELT_CY},0 상면) — 학습 벨트와 동일",
                },
                "nurec": {
                    "pos_m": [round(float(v), 6) for v in T_n[:3, 3]],
                    "quat_wxyz": [round(float(v), 6) for v in mat_to_quat(T_n[:3, :3])],
                    "scale": round(float(s_n), 6),
                    "reference": "scene_registration.yaml(T_base_nurec)이 소비 원본",
                },
            },
            "moved_reference_ghosts": moved_refs,  # 비어 있어야 정상
            "meta": {"date": date.today().isoformat(), "source": "place_nurec_gui.py"},
        }
        with open(_OUT_LAYOUT, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    def _reference_deviations() -> list[str]:
        """기준 고스트(로봇·벨트)의 스폰 기준 대비 이탈 목록(1mm 초과만)."""
        moved = []
        r_pos, _, _ = _pose_of(robot_g)
        if np.linalg.norm(r_pos) * 1000.0 > 1.0:
            moved.append(f"robot_ghost 원점 이탈 {np.linalg.norm(r_pos)*1000:.1f}mm")
        b_pos, _, _ = _pose_of(belt_g)
        db = np.linalg.norm(np.array(b_pos) - np.array([_BELT_CX, _BELT_CY, 0.0])) * 1000.0
        if db > 1.0:
            moved.append(f"belt_ghost 기준({_BELT_CX},{_BELT_CY},0) 이탈 {db:.1f}mm")
        return moved

    # ★변경 감지(dirty) 게이트 — 2026-07-17 적대 리뷰 수정★: 사용자가 실제로
    # 움직이기 '전'에는 기록하지 않는다. 구 동작(기동 1초 후 무조건 calibrated=true
    # 기록)은 GUI를 열어 구경만 해도 NuRec 파인튠 기동 거부 게이트를 무력화했다.
    # dirty 판정: 초기 로드값(T0, scale0) 대비 병진 >1mm 또는 회전 >0.05° 또는
    # 스케일 상대차 >1e-4 — 한 번이라도 넘으면 이후 계속 기록(미세 되돌림 허용).
    step = 0
    dirty = False
    while simulation_app.is_running():
        simulation_app.update()
        step += 1
        if step % 60 != 0:
            continue
        mw = UsdGeom.Xformable(nurec).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        T, s_mean, nonuni, s3 = _decompose_similarity(mw)
        if nonuni > 0.01:
            print(f"[place] ⚠️ 비균일 스케일 감지 s=({s3[0]:.4f},{s3[1]:.4f},{s3[2]:.4f}) — "
                  "similarity 는 균일 스케일 가정. 평균값으로 기록하니 축을 맞춰주세요.")
        if float(np.linalg.det(T[:3, :3])) < 0.0:
            print("[place] ⚠️ 반사(음수 스케일) 감지 — Scale 부호를 양수로 되돌리세요. 이번 기록 건너뜀.")
            continue
        if not dirty:
            dt_mm = float(np.linalg.norm(T[:3, 3] - T0[:3, 3])) * 1000.0
            cos_ang = (np.trace(T[:3, :3].T @ T0[:3, :3]) - 1.0) / 2.0
            dang_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ang))))
            ds_rel = abs(s_mean - scale0) / max(scale0, 1e-12)
            if dt_mm > 1.0 or dang_deg > 0.05 or ds_rel > 1e-4:
                dirty = True
                print("[place] ★변경 감지 — 이제부터 1초마다 yaml 기록★")
            else:
                continue  # 무조작 상태 — 기록 안 함(게이트 보호)
        tx, ty, tz = T[0, 3], T[1, 3], T[2, 3]
        yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        print(f"[place] T_base_nurec: pos=({tx:+.4f},{ty:+.4f},{tz:+.4f}) yaw≈{yaw:+.2f}° "
              f"scale={s_mean:.5f}")
        meta = {
            "calibrated": True,
            "method": "ghost_gui",
            "date": date.today().isoformat(),
            "residual_m": None,  # 육안 정합 — 잔차 미측정 (수치화는 register_nurec_umeyama.py)
            "notes": f"place_nurec_gui.py 라이브 기록 (step={step}) — 육안 검수 기반, "
                     "필요 시 register_nurec_umeyama.py 로 수치 미세보정",
        }
        try:
            save_scene_registration(_OUT, T, s_mean, meta)
        except Exception as e:  # noqa: BLE001
            print(f"[place] ⚠️ yaml 기록 실패: {e}")
        # ── 전 객체 스냅샷(사용자 요청) + 기준 고스트 이탈 감시 ──
        moved = _reference_deviations()
        if moved:
            print(f"[place] ⚠️ ★기준 고스트 이동 감지★: {'; '.join(moved)} — 기준(로봇 원점·"
                  "벨트 sim.yaml)은 학습에 반영되지 않습니다. NuRec 쪽을 움직여 맞추고 "
                  "기준은 되돌리세요 (이탈은 scene_layout.yaml에 기록됨).")
        try:
            _write_scene_layout(T, s_mean, moved)
        except Exception as e:  # noqa: BLE001
            print(f"[place] ⚠️ scene_layout 기록 실패: {e}")
    # 종료 스냅샷: NuRec 미변경 세션이어도 관절 슬라이더/기준 상태는 대장에 남긴다
    # (scene_registration.yaml은 dirty 게이트 그대로 — 스냅샷은 게이트와 무관한 감사 기록).
    try:
        mw = UsdGeom.Xformable(nurec).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        T_fin, s_fin, _, _ = _decompose_similarity(mw)
        _write_scene_layout(T_fin, s_fin, _reference_deviations())
        print(f"[place] 전 객체 스냅샷 저장 → {_OUT_LAYOUT}")
    except Exception as e:  # noqa: BLE001
        print(f"[place] ⚠️ 종료 스냅샷 실패: {e}")
    if dirty:
        print(f"[place] 종료 — 마지막 정합이 {_OUT} 에 저장됨. Claude에게 '정합 완료'라고 알려주세요.")
    else:
        print(f"[place] 종료 — ★변경 없음: {_OUT} 미기록★ (게이트 보호 — 기존 값 유지)")


if __name__ == "__main__":
    main()
    simulation_app.close()
