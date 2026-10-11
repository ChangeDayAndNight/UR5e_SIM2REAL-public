#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★로봇-벨트 배치 GUI 도구★ — 고스트 전용 스테이지에서 로봇(관절 포함)·워크스페이스 배치·기록.

★관절 조작(2026-07-16 확정)★: URDF→USD 변환본은 링크가 루트 아래 평면(sibling)이고 관절은
PhysX 전용 프림이라 ★기즈모로 조인트/링크를 회전해도 체인이 안 따라온다★(물리 없인 불가능).
→ 좌측 ★"UR5e 관절 (Ghost)" 패널의 슬라이더★로 관절을 움직인다 — 검증된 FK(urdf_fk.py,
tests/test_urdf_fk.py 골든: 홈 FK=실측 파지점 3mm 일치)로 모든 링크 pose를 재저작한다.

★화면 = 실셀 world 배치(2026-07-28 정정)★: sim.yaml belt_center_xy_m는 ★로봇 base frame★
좌표(−0.45 = 로봇 뒤쪽)인데, 구버전이 이를 world 좌표로 그대로 그려 ★배치해 둔 워크스페이스가
반대편으로 옮겨진 것처럼★ 보였다. 실제 셀은 벨트가 제자리(world +x)고 ★로봇이 180° 돌아 서 있다★
→ 로봇 ghost를 belt_yaw만큼 회전시키고 벨트 ghost는 Rz(belt_yaw)·(base 좌표) = 제자리에 둔다.
기록되는 base-frame 값(belt_center_x −0.45 · belt_yaw 180)은 sim.yaml과 그대로 일치한다.
⚠️학습 cfg는 base frame을 유지해야 한다 — observations.object_positions_b가
`root_pos_w − env_origins`(env 원점 = 로봇 base·무회전 전제)로 계산하므로, sim에서 로봇을
실제로 회전시키면 관측·보상이 전부 조용히 어긋난다. GUI만 world로 그리는 것이 맞다.

★2026-07-30 확장(사용자 요청) — "현재 학습 환경을 GUI로 직접 확인"★
  - /World/Cube_Ghost : ★실제 학습 크기(object_size_m) 큐브★ — 기즈모로 자유 이동.
  - /World/Belt_Ghost/pick_zone : 주황 반투명 판 = ★큐브가 실제로 스폰되는 구역★(pick_zone_m).
    물리 벨트(파란 판)와 다르다 — 벨트는 실측 크기고 픽존은 로봇이 닿는 범위다.
  - 그리퍼 슬라이더 범위 = ★CAD 스트로크★(GRIPPER_CLOSE_M, 도면 20mm/편측).
    [Open (q=0)] / [Close (grip)] / [Cube -> grip point] 버튼으로 즉시 상태 전환.
  - ★GRIP STATE (live) 패널★ — 그리퍼를 여닫거나 큐브를 옮기면 즉시 갱신:
      gap(안쪽 간격) · penetration(편측 침투) x stiffness = 접촉력[N] · pad 덮힘 비율 ·
      TCP↔큐브 중심 거리 · 종합 판정([O] GRIPPED / [!] WEAK GRIP / [X] TOO BIG / [ ] OPEN).
    판정 기준 23.4N = 구 계보(teacher placed 96.6%)에서 검증된 접촉력.
  ⚠️그리퍼 규격은 ur5e_assets.py에서 ★파싱★한다(단일 소스). 여기서 재정의하지 않는다.

구성(물리 없음 — Play(▶) 누르지 말 것):
  - /World/Robot_Ghost  : UR5e+TCW1+2FG7 (기즈모 Move/Rotate/Scale = base 배치, 초기 yaw=belt_yaw)
  - /World/Belt_Ghost   : 파란 벨트 + 초록 타겟존 + 빨간 +X 막대 (기즈모 Move/Rotate/Scale)
  - "UR5e Joints" 패널   : 팔 6관절[deg] + 그리퍼[mm] — ★슬라이더 + 수치 입력칸 병행★
    (2026-07-28 사용자 요청: 같은 모델을 공유해 어느 쪽을 바꿔도 즉시 반영·FK 갱신).
    [All zero]/[Home pose]/[Best candidate] 버튼.
  ⚠️UI 문자열은 ★영어·ASCII★ — Kit 폰트에 한글 글리프가 없어 '???'로 출력된다(2026-07-28).
  - ★"WristCam — 정책 시점(실시간)" 창★ (2026-07-28 추가): 손목에 붙은 카메라를 정책과
    ★같은 화각★(hand_eye.yaml 부착 + sim.yaml FOV)으로 렌더 — 관절을 움직이면 즉시 갱신.
  - ★실시간 가시율 표시★: 스폰 분포 중 카메라에 보이는 비율 + 3cm 큐브 화면크기[px].
    초록≥90% / 노랑≥50% / 빨강<50%. ★[이 자세 저장] 버튼으로 후보를 여러 개 기록★
    → Phase1/logs/home_pose_candidates.yaml (가시율·TCP·큐브크기 함께 저장).
    근거: 홈 가시율 18.3%에서 student grasp 0.8%로 붕괴한 이력(common.yaml frame_mode 주석).

사용법 ([사람] 실행 — GUI 필요, --headless 금지):
  1) python scripts\\tools\\place_workspace_gui.py
  2) 기즈모로 로봇 base·벨트를 실셀 배치(평행 등)에 맞추고, 슬라이더로 팔 자세를 재현한다.
  3) 콘솔·Phase1/logs/workspace_ghost_pose.yaml 에 1초마다 ★전 객체 Position/Rotation/Scale★ 저장
     (2026-07-27 확장 — 구 'yaw+xy'만 기록하던 한계 해소):
     ① 로봇 base 기준 벨트 pos(3)·quat·scale(3) ★학습용★ ② 로봇 world pose(quat 포함)·scale
     ③ 관절값[deg]·그리퍼[mm] ④ 자식 고스트(벨트판·타겟존·+X막대·로봇스탠드) world P/R/S
     ⑤ ★warnings★ — 축정렬/평면/원점 가정을 깨는 배치(roll·pitch·yaw·z·scale)를 즉시 경고.
  4) 창을 닫고: python scripts\\tools\\dump_scene_layout.py --from-gui
     → configs/calib/scene_layout.yaml(전 객체 레지스트리) 갱신 → 값 검토 후
       ★configs/sim.yaml scene.*(belt_center_xy_m·belt_size_m·stand_height_m)에 반영★해야 학습에 적용된다.

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

parser = argparse.ArgumentParser(description="Robot/belt layout GUI tool (ghost-only stage).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.headless:
    raise SystemExit("[place] [warn] GUI-only tool - run without --headless.")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
from pathlib import Path

import numpy as np
import yaml

import urdf_fk  # scripts/tools (프리앰블이 sys.path 보장) — FK는 pytest 골든으로 검증됨
from common.calibration import load_hand_eye  # T_flange_cam (요소4 단일 소스)
# ★2026-08-10 누락 import 복구★ — 165행에서 쓰는데 import가 없어 NameError로 기동 실패했다
#   (리팩터 때 이 도구가 누락된 것으로 보인다. tests가 GUI 도구를 import하지 않아 미검출).
from common.calibration import load_tcp_offset  # T_flange_tcp (요소4 단일 소스)
from common.calibration import mat_to_quat as _mat_to_quat  # 순수(numpy) — 상대 pose 산출
from common.calibration import quat_to_mat as _quat_to_mat
from common.config_io import configs_dir, load_yaml
from common.robot_spec import UR5E_HOME_JOINT_POS

_SIM = load_yaml(configs_dir() / "sim.yaml")
_COMMON = load_yaml(configs_dir() / "common.yaml")
BELT_SIZE = tuple(float(v) for v in _SIM["scene"]["belt_size_m"])
# ★픽존(2026-07-28)★ — 큐브가 실제로 스폰되는 구역. 물리 벨트(BELT_SIZE)는 시각용이라
#   가시율은 ★픽존 기준★으로 재야 한다(벨트 전역을 재면 스폰되지도 않는 곳을 세어 과소평가).
PICK_ZONE = tuple(float(v) for v in _SIM["scene"].get("pick_zone_m", BELT_SIZE[:2]))
TABLE_HEIGHT = float(_SIM["scene"]["table_height_m"])
_TGT = _COMMON["target"]["sample_range"]
# ★2026-07-27 yaml 승격 반영★ — 구 하드코딩(0.45/0.0/0.80) 대신 단일 소스에서 읽는다.
_STAND_H = float(_SIM["scene"].get("stand_height_m", 0.80))
_BELT_CX, _BELT_CY = (float(v) for v in _SIM["scene"].get("belt_center_xy_m", [0.45, 0.0]))
OBJECT_SIZE = float(_SIM["scene"]["object_size_m"])

# ★그리퍼 규격(2026-07-30 CAD 도면 2fg7.png 확정)★ — ur5e_assets.py에서 파싱한다.
#   직접 import하지 않는 이유: 그 모듈이 isaaclab을 끌어와 무거워지고, 여기선 숫자만 필요하다.
#   ⚠️단일 소스는 어디까지나 ur5e_assets.py다 — 여기서 값을 재정의하지 않는다.
def _grip_specs() -> dict:
    import re as _re
    _src = (Path(_REPO_ROOT) / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks"
            / "binpick" / "mdp" / "ur5e_assets.py").read_text(encoding="utf-8")
    def _f(pat, dflt):
        m = _re.search(pat, _src, _re.M | _re.S)
        return float(m.group(1)) if m else dflt
    return {
        "open_gap":  _f(r"GRIPPER_MAX_OPEN_GAP_M: float = ([0-9.]+)", 0.074),
        "close_q":   _f(r"^GRIPPER_CLOSE_M: float = ([0-9.]+)", 0.020),
        "open_q":    _f(r"^GRIPPER_OPEN_M: float = ([0-9.]+)", 0.0),
        "stiffness": _f(r'"gripper": ImplicitActuatorCfg\(.*?stiffness=([0-9.]+)', 3000.0),
        "pad_h":     0.045,   # 콜라이더 박스 z(높이) — URDF <box size="0.0192 0.032 0.045">
    }


_GS = _grip_specs()
# ★tcp_offset은 모듈 레벨에서 로드★ — 툴 체인 패널(UI 구성 시점)과 FK 계산(그 이후)이
#   같은 값을 써야 하고, 지역 변수로 두면 UI 구성이 먼저 실행돼 UnboundLocalError가 난다
#   (2026-07-30 실오류). 순수 yaml 로드라 isaaclab과 무관하다.
_T_FT = load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")

# ★툴 체인(2026-07-30 사용자 제공: 실기 5단 확인)★ — configs/calib/tool_chain.yaml
#   flange → tool flange → magbot TCW1 → ★OnRobot QC-RS v3★ → OnRobot 2FG7
#   sim URDF는 3단(QC 누락)이라 tcp_offset이 QC 높이만큼 짧다. GUI가 이를 표시한다.
try:
    _TOOL_CHAIN = load_yaml(configs_dir() / "calib" / "tool_chain.yaml")
except Exception:  # noqa: BLE001
    _TOOL_CHAIN = {"elements": [], "verification": {}}


def _chain_segments() -> list:
    """[(이름, 시작z, 높이, 색, 측정여부)] — flange(z=0)부터 위로 쌓는다. 단위 m."""
    segs, z = [], 0.0
    for e in _TOOL_CHAIN.get("elements", []):
        h = e.get("height_m")
        if h is None:                      # 미측정 → 추정치로 그리되 표시를 남긴다
            h = e.get("height_estimate_m") or 0.0
        if h is None:
            h = 0.0
        h = float(h)
        if e.get("height_to_grip_m") is not None:   # 2FG7: 파지중심까지
            h = float(e["height_to_grip_m"])
        if h <= 0.0:
            continue
        segs.append((str(e.get("name", "?")), z, h,
                     tuple(float(v) for v in e.get("color", [0.7, 0.7, 0.7])),
                     bool(e.get("measured", False))))
        z += h
    return segs

# ★고스트 world 배치(2026-07-28 사용자 지적 반영)★
#   [문제] sim.yaml belt_center_xy_m는 ★로봇 base frame★ 좌표다(−0.45 = 로봇 뒤쪽).
#     GUI가 이 값을 world 좌표로 그대로 써서, 사용자가 배치해 둔 워크스페이스가
#     ★반대편으로 옮겨진 것처럼 보였다★.
#   [실제 셀] 벨트는 제자리(world +x)에 있고 ★로봇이 180° 돌아 서 있다★.
#   [해법] 고스트를 그 관계 그대로 그린다 — 로봇 ghost를 belt_yaw만큼 회전시키고,
#     벨트 ghost의 world 위치를 Rz(belt_yaw)·(base 좌표)로 변환한다.
#     이러면 화면은 실셀과 같고, 기록되는 base-frame 값(belt_center_x=−0.45,
#     belt_yaw=180)은 sim.yaml과 그대로 일치한다.
#   ⚠️학습 cfg는 base frame(−0.45)을 유지해야 한다 — observations.object_positions_b가
#     `root_pos_w − env_origins`(= env 원점이 곧 로봇 base·무회전 전제)로 계산하므로,
#     sim에서 로봇을 실제로 회전시키면 관측·보상이 전부 조용히 어긋난다.
_GHOST_YAW_DEG = float(_SIM["scene"].get("belt_yaw_deg", 0.0))   # 로봇의 실셀 설치 방향
_c, _s = math.cos(math.radians(_GHOST_YAW_DEG)), math.sin(math.radians(_GHOST_YAW_DEG))
_BELT_WORLD_X = _c * _BELT_CX - _s * _BELT_CY   # Rz(yaw)·(base 좌표) = world 위치
_BELT_WORLD_Y = _s * _BELT_CX + _c * _BELT_CY

# ── ★손목 카메라 실시간 미리보기(2026-07-28 사용자 요청)★ ──
#   "관절을 움직이면서 정책이 보는 화면을 실시간으로 보고 싶다" → 고스트 손목에 정책과
#   ★동일한 FOV★의 USD 카메라를 붙이고 전용 뷰포트로 띄운다. FOV 계산은 mdp/scene.py
#   make_tiled_camera_cfg와 같은 식(초점거리 = aperture / (2·tan(fov/2)))을 쓴다.
_OBJ_SIZE = float(_SIM["scene"]["object_size_m"])
_CAM_W = int(_SIM["camera"]["width"])      # 정책 입력 해상도(구 하드코딩 112 제거)
_CAM_H = int(_SIM["camera"]["height"])
_CAM_VFOV = float(_SIM["camera"]["vertical_fov_deg"])
_CAM_HFOV = float(_SIM["camera"].get("horizontal_fov_deg", 0.0))
_CAM_APERTURE = 20.955  # scene.py와 동일한 USD 관례값
_PRE = _COMMON.get("preprocess", {})
_CROP = float(_PRE.get("crop_fraction", 1.0))
_FRAME_MODE = str(_PRE.get("frame_mode", "square"))
_DEPTH_NEAR = float(_PRE.get("depth_near_m", 0.28))
_DEPTH_FAR = float(_PRE.get("depth_far_m", 1.20))
_CAND_OUT = os.path.join(_REPO_ROOT, "Phase1", "logs", "home_pose_candidates.yaml")
_TGT_CX = (float(_TGT["x"][0]) + float(_TGT["x"][1])) / 2.0
_TGT_CY = (float(_TGT["y"][0]) + float(_TGT["y"][1])) / 2.0
_TGT_HALF_X = (float(_TGT["x"][1]) - float(_TGT["x"][0])) / 2.0
_TGT_HALF_Y = (float(_TGT["y"][1]) - float(_TGT["y"][0])) / 2.0

_OUT = os.path.join(_REPO_ROOT, "Phase1", "logs", "workspace_ghost_pose.yaml")
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


def _pose_full(prim) -> dict:
    """world 변환 → ★전체 P/R/S★ dict (2026-07-27 확장 — 구 yaw-only 기록의 한계 해소).

    반환: {pos(3), quat_wxyz(4), scale(3), rpy_deg(3)}.
    USD Gf.Matrix4d는 row-major·row-vector 관례(p' = p·M)라 상단 3x3의 '행'이 기저벡터의
    이미지다 → 열벡터 관례 회전행렬 R = Mᵀ(각 행을 스케일로 정규화한 뒤).
    ⚠️ 구현은 기존 yaw 계산(atan2(r0[1], r0[0]))과 동일한 규약을 유지한다.
    """
    from pxr import Usd, UsdGeom

    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rows = [(m[i][0], m[i][1], m[i][2]) for i in range(3)]
    scale = [math.sqrt(sum(v * v for v in r)) or 1.0 for r in rows]
    R = np.array([[rows[j][i] / scale[j] for j in range(3)] for i in range(3)])
    w, x, y, z = urdf_fk.mat_to_quat_wxyz(R)
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    return {
        "pos": (float(m[3][0]), float(m[3][1]), float(m[3][2])),
        "quat": (float(w), float(x), float(y), float(z)),
        "scale": tuple(float(s) for s in scale),
        "rpy_deg": (math.degrees(roll), math.degrees(pitch), math.degrees(yaw)),
    }


def _fmt3(v) -> str:
    return "[" + ", ".join(f"{float(c):.4f}" for c in v) + "]"


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


def main() -> None:
    import omni.ui as ui
    import omni.usd
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

    # ── 로봇 고스트: UR5e+TCW1+2FG7 USD 참조 (물리 없음 → 정지 표시) ──
    robot_g = stage.DefinePrim("/World/Robot_Ghost", "Xform")
    arm = stage.DefinePrim("/World/Robot_Ghost/arm", "Xform")
    arm.GetReferences().AddReference(_ROBOT_USD.replace("\\", "/"))
    rxf = UsdGeom.Xformable(robot_g)
    rxf.ClearXformOpOrder()
    rxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    rxf.AddRotateZOp().Set(_GHOST_YAW_DEG)   # ★실셀 설치 방향★ (벨트를 등지면 180°)
    rxf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    # 안전: Play를 눌러도 낙하/폭발하지 않게 물리 비활성(표시 무영향 — 시뮬 안 함)
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot_Ghost")):
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        except Exception:  # noqa: BLE001
            pass
    # 로봇 스탠드(자식 — 로봇과 함께 이동)
    _add_box(stage, "/World/Robot_Ghost/stand", (0.11, 0.11, _STAND_H / 2), (0.62, 0.63, 0.66), 1.0,
             (0.0, 0.0, TABLE_HEIGHT - _STAND_H / 2))

    # ── FK 저작 준비: 링크 prim 맵 + base_link 기준변환 ──
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
            print(f"[place] [warn] link prim not found: {name} - excluded from joint authoring")
    base_prim = stage.GetPrimAtPath("/World/Robot_Ghost/arm/base_link")
    x_base = np.eye(4)
    if base_prim.IsValid():
        x_base = _gf_to_np(UsdGeom.Xformable(base_prim).GetLocalTransformation())

    # ── 관절 슬라이더 패널 (검증된 FK → 링크 재저작) ──
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
    win = ui.Window("UR5e Joints (Ghost)", width=500, height=360,
                    dockPreference=ui.DockPreference.RIGHT_BOTTOM)
    with win.frame:
        with ui.VStack(spacing=4, style={"margin": 4}):
            ui.Label("[!] Use THESE sliders for joints "
                     "(gizmo rotation does not move the chain)", height=18)
            for jn in urdf_fk.ARM_JOINT_NAMES:
                lo, hi = _LIMITS_DEG.get(jn, (-360.0, 360.0))
                with ui.HStack(height=22, spacing=6):
                    ui.Label(jn.replace("_joint", ""), width=110)
                    m = ui.SimpleFloatModel(0.0)
                    ui.FloatSlider(model=m, min=lo, max=hi, step=1.0)
                    # ★수치 직접 입력(2026-07-28 사용자 요청)★ — 슬라이더와 ★같은 모델★을
                    #   공유하므로 어느 쪽을 바꿔도 서로 즉시 반영되고 FK도 함께 갱신된다
                    #   (정밀한 각도를 손으로 넣을 때 슬라이더 드래그보다 정확하다).
                    ui.FloatField(model=m, width=62)
                    ui.Label("deg", width=26)
                    arm_models[jn] = m
                    m.add_value_changed_fn(_apply_joints)
            with ui.HStack(height=22, spacing=6):
                # ★CAD 스트로크 반영(2026-07-30)★ 최대 = GRIPPER_CLOSE_M (도면 20mm/편측).
                #   q=0 -> 안쪽 간격 open_gap(74mm), q=max -> 34mm.
                ui.Label("gripper close[mm]", width=110)
                grip_model = ui.SimpleFloatModel(0.0)
                ui.FloatSlider(model=grip_model, min=0.0, max=_GS["close_q"] * 1000.0, step=0.5)
                ui.FloatField(model=grip_model, width=62)
                ui.Label("mm", width=26)
                grip_model.add_value_changed_fn(_apply_joints)
            with ui.HStack(height=26, spacing=8):
                ui.Button("Open (q=0)", clicked_fn=lambda: grip_model.set_value(0.0))
                ui.Button("Close (grip)",
                          clicked_fn=lambda: grip_model.set_value(_GS["close_q"] * 1000.0))
                ui.Button("Cube -> grip point", clicked_fn=lambda: _snap_cube())
            with ui.HStack(height=26, spacing=8):
                ui.Button("All zero", clicked_fn=lambda: _set_all({}))
                ui.Button("Home pose", clicked_fn=lambda: _set_all(
                    {n: math.degrees(v) for n, v in UR5E_HOME_JOINT_POS.items()}))
                # ★2026-07-28(2차) 배치 180° 반영★ — 구 후보(pan 0°)는 로봇이 벨트를 등지기
                #   전의 값이라 지금은 벨트 반대편을 본다. pan만 180°로 회전(나머지 불변).
                ui.Button("Best candidate", clicked_fn=lambda: _set_all(
                    {"shoulder_pan_joint": 180.0, "shoulder_lift_joint": -90.0, "elbow_joint": 65.0,
                     "wrist_1_joint": -45.0, "wrist_2_joint": -90.0, "wrist_3_joint": 0.0}))
            ui.Separator(height=6)
            # ★실시간 가시율★ — 관절을 움직이는 즉시 '정책이 큐브를 볼 수 있는 비율'이 갱신된다.
            vis_label = ui.Label("computing visibility...", height=22,
                                 style={"font_size": 16, "color": 0xFF66DD66})
            tcp_label = ui.Label("", height=20, style={"color": 0xFFBBBBBB})
            with ui.HStack(height=26, spacing=8):
                ui.Button("[*] Save this pose", clicked_fn=lambda: _save_candidate())
                ui.Button("Clear saved list", clicked_fn=lambda: _clear_candidates())
            save_label = ui.Label("", height=20, style={"color": 0xFFAACCFF})
            ui.Separator(height=6)
            # ★파지 상태 실시간 패널(2026-07-30 사용자 요청)★ — 그리퍼를 여닫거나 큐브를
            #   기즈모로 옮기면 즉시 갱신된다. 학습 환경의 파지 조건을 눈으로 검증하는 창.
            ui.Label("GRIP STATE (live)", height=18,
                     style={"font_size": 13, "color": 0xFF88AACC})
            gap_label = ui.Label("", height=20, style={"font_size": 15, "color": 0xFFDDDDDD})
            pen_label = ui.Label("", height=20, style={"color": 0xFFDDDDDD})
            fit_label = ui.Label("", height=20, style={"font_size": 15})
            algn_label = ui.Label("", height=20, style={"color": 0xFFBBBBBB})
            ui.Separator(height=6)
            # ★툴 체인 패널★ — 실기 5단 체인의 합계가 tcp_offset과 맞는지 즉시 보여준다.
            ui.Label("TOOL CHAIN (flange -> grip center)", height=18,
                     style={"font_size": 13, "color": 0xFF88AACC})
            for _nm, _z0, _h, _col, _meas in _chain_segments():
                ui.Label(f"  {_nm:18s} {_z0*1000:6.1f} -> {(_z0+_h)*1000:6.1f} mm"
                         f"   ({_h*1000:5.1f})" + ("" if _meas else "   [!] ESTIMATE"),
                         height=17,
                         style={"color": 0xFFCCCCCC if _meas else 0xFF00CCFF})
            _sum = sum(x[2] for x in _chain_segments())
            _tcpz = float(_T_FT[2, 3])
            _dz = (_sum - _tcpz) * 1000.0
            chain_label = ui.Label(
                f"  chain sum {_sum*1000:.1f}mm   vs tcp_offset {_tcpz*1000:.1f}mm"
                f"   -> diff {_dz:+.1f}mm", height=20,
                style={"font_size": 14,
                       "color": 0xFF00FF00 if abs(_dz) < 1.0 else 0xFF3333FF})
            if abs(_dz) >= 1.0:
                ui.Label("  [X] MISMATCH - QC-RS v3 not in URDF. Measure it and update"
                         " tool_chain.yaml + URDF + tcp_offset.", height=17,
                         style={"color": 0xFF3333FF})
    _apply_joints()  # 초기 = 전부 0

    # ══════════════════════════════════════════════════════════════════════
    # ★손목 카메라(정책 시점) — 고스트 손목에 부착 + 전용 뷰포트로 실시간 표시★
    #   FOV/초점거리는 mdp/scene.py make_tiled_camera_cfg와 동일 식으로 산출해
    #   '정책이 실제로 보는 화각'과 일치시킨다(해상도만 다름 — 뷰포트는 크게 보여주기용).
    #   ⚠️ USD 카메라는 opengl 관례(-Z 전방, +Y 위), hand_eye.yaml은 ros 관례(+Z 전방)라
    #      convert_camera_frame_orientation_convention(ros→opengl)로 변환해 저작한다.
    # ══════════════════════════════════════════════════════════════════════
    T_fc = load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")
    _eff_vfov = math.degrees(2.0 * math.atan(_CROP * math.tan(math.radians(_CAM_VFOV) / 2.0)))
    _focal = _CAM_APERTURE / (2.0 * math.tan(math.radians(_eff_vfov) / 2.0))
    if _FRAME_MODE == "full" and _CAM_HFOV > 0.0:
        _h_ap = 2.0 * _focal * math.tan(math.radians(_CAM_HFOV) / 2.0)
        _v_ap = _CAM_APERTURE
    else:
        _h_ap = _v_ap = _CAM_APERTURE

    cam_path = "/World/Robot_Ghost/arm/wrist_3_link/WristCam"

    def _attach_wrist_camera() -> None:
        """손목에 정책과 동일 화각의 USD 카메라를 붙이고 전용 뷰포트 창을 연다."""
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr().Set(float(_focal))
        cam.CreateHorizontalApertureAttr().Set(float(_h_ap))
        cam.CreateVerticalApertureAttr().Set(float(_v_ap))
        cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 30.0))
        # ros(+Z 전방, +Y 아래) → usd/opengl(-Z 전방, +Y 위) = ★X축 180° 후곱★
        #   (isaaclab convert_camera_frame_orientation_convention과 동일 정의. 그 함수는
        #    isaaclab.utils가 pxr을 끌어와 순수 검증이 불가해 직접 구현 — 등가성은
        #    tests/test_camera_convention.py 골든 7건으로 고정.)
        _R_usd = T_fc[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        _qu = urdf_fk.mat_to_quat_wxyz(_R_usd)
        cxf = UsdGeom.Xformable(cam)
        cxf.ClearXformOpOrder()
        # ⚠️ 정밀도 명시 필수 — AddOrientOp() 기본값은 ★PrecisionFloat★(GfQuatf)라 GfQuatd를
        #   넣으면 "Type mismatch ... expected 'GfQuatf', got 'GfQuatd'"로 죽는다.
        #   (AddTranslateOp() 기본값은 Double이라 orient만 어긋났다 — 2026-07-28 실오류.)
        cxf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*[float(v) for v in T_fc[:3, 3]]))
        cxf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(_qu[0]), Gf.Vec3d(*[float(v) for v in _qu[1:]])))
        print(f"[place] [*] wrist camera attached: {cam_path}  focal={_focal:.2f} "
              f"h_ap={_h_ap:.2f} v_ap={_v_ap:.2f} (FOV {_CAM_HFOV:.1f}deg x {_CAM_VFOV:.1f}deg)")

        from pxr import Sdf  # camera_path는 Sdf.Path 계약(utility/__init__.py:427)

        from omni.kit.viewport.utility import create_viewport_window

        create_viewport_window("WristCam - Policy View (live)", camera_path=Sdf.Path(cam_path),
                               width=560, height=560, position_x=40, position_y=80)
        print("[place] [*] 'WristCam - Policy View' window opened "
              "- updates live as you move joints")

    # ══════════════════════════════════════════════════════════════════════
    # ★툴 체인 고스트(2026-07-30 사용자 요청)★ — 5단을 각각 별개 프림으로 그린다.
    #   wrist_3_link 자식으로 붙이므로 관절을 움직이면 FK를 자동으로 따라온다.
    #   ★각 단을 클릭·기즈모로 선택/이동할 수 있다★(체인 길이 감각을 눈으로 확인).
    #   tcp_offset이 identity 회전 + z 이므로 wrist_3_link의 +z가 곧 툴 축이다.
    # ══════════════════════════════════════════════════════════════════════
    _chain_radius = {"magbot_TCW1": 0.045, "onrobot_QC_RS_v3": 0.032,
                     "onrobot_2FG7": 0.045}
    if link_prims.get("wrist_3_link") is not None:
        try:
            _w3 = "/World/Robot_Ghost/arm/wrist_3_link"
            stage.DefinePrim(f"{_w3}/ToolChain", "Xform")
            for _nm, _z0, _h, _col, _meas in _chain_segments():
                _r = _chain_radius.get(_nm, 0.038)
                # 미측정 단은 반투명하게 그려 눈에 띄게 한다
                _add_box(stage, f"{_w3}/ToolChain/{_nm}", (_r, _r, _h / 2.0),
                         _col, 0.85 if _meas else 0.40, (0.0, 0.0, _z0 + _h / 2.0))
            print(f"[place] [*] tool chain ghosts: {[x[0] for x in _chain_segments()]}")
        except Exception as e:  # noqa: BLE001
            print(f"[place][warn] tool chain ghost skipped ({type(e).__name__}: {e})")

    if link_prims.get("wrist_3_link") is not None:
        # ★미리보기는 부가 기능 — 실패해도 배치·가시율 도구는 계속 동작해야 한다★
        #   (2026-07-28: orient 정밀도 불일치로 GUI 전체가 죽어 실행을 한 번 날린 사고)
        try:
            _attach_wrist_camera()
        except Exception as e:  # noqa: BLE001
            print(f"[place][warn] wrist camera preview DISABLED ({type(e).__name__}: {e})")
            print(f"[place]        Layout/visibility tools still work. Manual fallback: "
                  f"Window > Viewport (new window), then pick camera {cam_path}.")
    else:
        print("[place][warn] wrist_3_link prim not found - wrist camera not attached")

    # ══════════════════════════════════════════════════════════════════════
    # ★실시간 가시율 계산★ — '정책이 에피소드 시작에 큐브를 볼 수 있는 비율'.
    #   근거: 이 프로젝트는 홈 가시율 18.3%에서 student grasp 0.8%로 붕괴한 이력이 있다
    #   (configs/common.yaml frame_mode 주석). 즉 홈 자세의 첫 프레임 가시성은
    #   증류 성패를 가르는 구조적 변수라 관절을 만질 때 즉시 보여준다.
    #   계산은 학습과 같은 base frame에서 한다(고스트를 월드에서 어디로 옮기든 무관).
    # ══════════════════════════════════════════════════════════════════════
    from common.calibration import load_tcp_offset

    _T_ft = _T_FT   # 모듈 레벨 단일 로드 재사용(위 주석 참조)
    _HF = math.radians(_CAM_HFOV if (_FRAME_MODE == "full" and _CAM_HFOV > 0) else _CAM_VFOV)
    _VF = math.radians(_CAM_VFOV)
    _mx = _OBJ_SIZE / 2.0 + 0.05                      # events_dr._SCATTER_MARGIN
    _hx = max(PICK_ZONE[0] / 2.0 - _mx, 0.01)   # ★픽존 기준★ (events_dr와 동일 식)
    _hy = max(PICK_ZONE[1] / 2.0 - _mx, 0.01)
    _GRID = [(_BELT_CX + ux, _BELT_CY + uy, TABLE_HEIGHT + _OBJ_SIZE / 2.0)
             for ux in np.linspace(-_hx, _hx, 13) for uy in np.linspace(-_hy, _hy, 13)]

    def _current_q_rad() -> dict:
        q = {n: math.radians(m.get_value_as_float()) for n, m in arm_models.items()}
        if grip_model is not None:
            q["left_finger_joint"] = grip_model.get_value_as_float() / 1000.0
        return q

    def _cube_prim():
        return stage.GetPrimAtPath("/World/Cube_Ghost")

    def _snap_cube() -> None:
        """큐브를 현재 자세의 파지중심(TCP)으로 옮긴다 — 파지 순간 기하를 눈으로 확인."""
        from pxr import Gf, UsdGeom

        P = urdf_fk.fk(joints, _current_q_rad())
        tcp_b = (P["wrist_3_link"] @ _T_ft)[:3, 3]          # base frame
        # base -> world: 로봇 ghost가 _GHOST_YAW_DEG만큼 돌아 있다
        c, s2 = math.cos(math.radians(_GHOST_YAW_DEG)), math.sin(math.radians(_GHOST_YAW_DEG))
        wx = c * tcp_b[0] - s2 * tcp_b[1]
        wy = s2 * tcp_b[0] + c * tcp_b[1]
        pr = _cube_prim()
        if not pr.IsValid():
            return
        xf = UsdGeom.Xformable(pr)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(float(wx), float(wy), float(tcp_b[2])))
                break

    def _grip_state():
        """(간격mm, 침투mm, 힘N, 덮힘%, TCP↔큐브 거리mm) — 슬라이더/기즈모 즉시 반영."""
        from pxr import Usd, UsdGeom

        q_mm = grip_model.get_value_as_float() if grip_model is not None else 0.0
        gap_mm = (_GS["open_gap"] - 2.0 * (q_mm / 1000.0)) * 1000.0
        cube_mm = OBJECT_SIZE * 1000.0
        pen_mm = (cube_mm - gap_mm) / 2.0                   # 음수 = 아직 여유
        force = _GS["stiffness"] * max(0.0, pen_mm) / 1000.0
        cov = min(_GS["pad_h"] * 1000.0, cube_mm) / cube_mm * 100.0
        # TCP(파지중심) ↔ 큐브 중심 거리 (world)
        d_mm = float("nan")
        pr = _cube_prim()
        if pr.IsValid():
            m = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            cw = np.array([m[3][0], m[3][1], m[3][2]])
            P = urdf_fk.fk(joints, _current_q_rad())
            tcp_b = (P["wrist_3_link"] @ _T_ft)[:3, 3]
            c, s2 = math.cos(math.radians(_GHOST_YAW_DEG)), math.sin(math.radians(_GHOST_YAW_DEG))
            tw = np.array([c * tcp_b[0] - s2 * tcp_b[1], s2 * tcp_b[0] + c * tcp_b[1], tcp_b[2]])
            d_mm = float(np.linalg.norm(cw - tw)) * 1000.0
        return gap_mm, pen_mm, force, cov, d_mm

    def _visibility():
        """(가시율[%], TCP(base), 큐브 화면크기[px], 평균거리[m]) — 순수 계산(렌더 무관)."""
        P = urdf_fk.fk(joints, _current_q_rad())
        Tw = P["wrist_3_link"]
        Tc = Tw @ T_fc
        p, R = Tc[:3, 3], Tc[:3, :3]
        tcp = (Tw @ _T_ft)[:3, 3]
        seen, dists = 0, []
        for pt in _GRID:
            v = R.T @ (np.array(pt) - p)
            if v[2] <= 0:
                continue
            if abs(math.atan2(v[0], v[2])) > _HF / 2 or abs(math.atan2(v[1], v[2])) > _VF / 2:
                continue
            d = float(np.linalg.norm(v))
            if _DEPTH_NEAR <= d <= _DEPTH_FAR:
                seen += 1
                dists.append(d)
        pct = 100.0 * seen / len(_GRID)
        md = float(np.mean(dists)) if dists else float("nan")
        # 화면 크기[px] = 3cm 큐브가 차지하는 픽셀 수 — 정책 해상도(sim.yaml camera)를 쓴다
        pw = (_OBJ_SIZE / (2 * md * math.tan(_HF / 2)) * _CAM_W) if dists else 0.0
        ph = (_OBJ_SIZE / (2 * md * math.tan(_VF / 2)) * _CAM_H) if dists else 0.0
        return pct, tcp, (pw, ph), md

    def _save_candidate() -> None:
        """현재 관절 자세를 후보 목록에 추가 (여러 각도를 비교하며 고를 수 있게)."""
        pct, tcp, px, md = _visibility()
        rec = {
            "joints_deg": {n: round(m.get_value_as_float(), 1) for n, m in arm_models.items()},
            "gripper_close_mm": round(grip_model.get_value_as_float(), 1) if grip_model else 0.0,
            "visibility_pct": round(pct, 1),
            "tcp_base_m": [round(float(v), 4) for v in tcp],
            "cube_px": [round(px[0], 1), round(px[1], 1)],
            "mean_dist_m": round(md, 3) if md == md else None,
        }
        old = []
        if os.path.isfile(_CAND_OUT):
            try:
                old = load_yaml(_CAND_OUT).get("candidates", []) or []
            except Exception:  # noqa: BLE001
                old = []
        old.append(rec)
        os.makedirs(os.path.dirname(_CAND_OUT), exist_ok=True)
        with open(_CAND_OUT, "w", encoding="utf-8") as f:
            f.write("# place_workspace_gui.py - saved home-pose candidates\n")
            f.write("# visibility_pct = fraction of the spawn distribution visible to the policy camera (higher is better)\n")
            yaml.safe_dump({"candidates": old}, f, allow_unicode=True, sort_keys=False)
        save_label.text = f"saved #{len(old)} - visibility {pct:.1f}% | cube {px[0]:.1f}x{px[1]:.1f}px"
        print(f"[place] [*] candidate saved #{len(old)} {rec['joints_deg']} "
              f"visibility {pct:.1f}% -> {_CAND_OUT}")

    def _clear_candidates() -> None:
        if os.path.isfile(_CAND_OUT):
            os.remove(_CAND_OUT)
        save_label.text = "saved list cleared"
        print(f"[place] candidate list deleted: {_CAND_OUT}")

    # ── 워크스페이스 고스트 ──
    bx, by, bz = BELT_SIZE
    ghost = stage.DefinePrim("/World/Belt_Ghost", "Xform")
    gxf = UsdGeom.Xformable(ghost)
    gxf.ClearXformOpOrder()
    # 벨트는 world 제자리 — 로봇이 도는 것이지 벨트가 옮겨지는 게 아니다.
    gxf.AddTranslateOp().Set(Gf.Vec3d(_BELT_WORLD_X, _BELT_WORLD_Y, 0.0))
    gxf.AddRotateZOp().Set(0.0)   # base yaw(180) + 로봇 yaw(180) = 360 ≡ 0 → world 축정렬
    gxf.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    _add_box(stage, "/World/Belt_Ghost/belt", (bx / 2, by / 2, bz / 2),
             (0.20, 0.45, 0.95), 0.5, (0.0, 0.0, TABLE_HEIGHT - bz / 2))
    _add_box(stage, "/World/Belt_Ghost/target_zone", (_TGT_HALF_X, _TGT_HALF_Y, 0.004),
             (0.15, 0.85, 0.25), 0.7, (_TGT_CX - _BELT_CX, _TGT_CY - _BELT_CY, TABLE_HEIGHT + 0.01))
    _add_box(stage, "/World/Belt_Ghost/dir_x", (bx / 2 + 0.08, 0.012, 0.012),
             (0.95, 0.15, 0.15), 1.0, (bx / 2 + 0.05, 0.0, TABLE_HEIGHT + 0.03))
    # ★픽존 표시(2026-07-30)★ — 큐브가 실제로 스폰되는 구역(물리 벨트와 다르다).
    _add_box(stage, "/World/Belt_Ghost/pick_zone", (PICK_ZONE[0] / 2, PICK_ZONE[1] / 2, 0.002),
             (0.95, 0.75, 0.20), 0.35, (0.0, 0.0, TABLE_HEIGHT + 0.004))

    # ★큐브 고스트(2026-07-30 사용자 요청)★ — 실제 학습 크기(object_size_m)로, 기즈모로 이동 가능.
    #   초기 위치 = 픽존 중앙, 벨트 상면에 안착(중심 z = 상면 + 반변).
    cube = stage.DefinePrim("/World/Cube_Ghost", "Xform")
    cxf2 = UsdGeom.Xformable(cube)
    cxf2.ClearXformOpOrder()
    cxf2.AddTranslateOp().Set(Gf.Vec3d(_BELT_WORLD_X, _BELT_WORLD_Y,
                                      TABLE_HEIGHT + OBJECT_SIZE / 2.0))
    cxf2.AddRotateZOp().Set(0.0)
    cxf2.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    _add_box(stage, "/World/Cube_Ghost/box",
             (OBJECT_SIZE / 2, OBJECT_SIZE / 2, OBJECT_SIZE / 2),
             (0.85, 0.45, 0.15), 0.9, (0.0, 0.0, 0.0))

    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=[2.0, -1.7, 1.5], target=[0.4, 0.0, 0.1])
    except Exception:  # noqa: BLE001
        pass

    print("[place] [*] GHOST-ONLY STAGE - 1 robot + belt. Do NOT press Play.")
    print("[place]   base layout = gizmo (Move/Rotate/Scale) | "
          "joints = 'UR5e Joints' panel sliders.")
    print(f"[place] pose log: {_OUT} (updated every 1s; last value kept on close)")
    print("[place] " + "-" * 60)

    # ★2026-07-27 전 객체 P/R/S 기록으로 확장(사용자 지시)★ — 구 기록은 로봇·벨트 2개의
    #   'yaw + xy'만 남겨 roll/pitch·z·3축 스케일이 유실됐다. 이제 world/base 양쪽의 전체
    #   pose(quaternion 포함)와 스케일, 자식 고스트의 로컬 배치까지 남긴다.
    _CHILD_GHOSTS = {
        "belt_plate": "/World/Belt_Ghost/belt",
        "target_zone": "/World/Belt_Ghost/target_zone",
        "dir_x_marker": "/World/Belt_Ghost/dir_x",
        "robot_stand": "/World/Robot_Ghost/stand",
        # ★2026-07-28 사각지대 수정★: 사용자가 부모(/World/Robot_Ghost)가 아니라 USD 참조를 담은
        #   자식(/World/Robot_Ghost/arm)을 기즈모로 잡는 경우가 있다 — 그때 robot_pose_world는
        #   항등 그대로라 '로봇을 옮겼는데 기록에 없다'가 된다. 이 프림을 명시 포함한다.
        "robot_arm_ref": "/World/Robot_Ghost/arm",
    }

    def _all_editable_prims():
        """/World 이하 깊이 3까지의 모든 Xform — 어떤 프림을 잡았든 놓치지 않는다.

        (깊이 4+ = 로봇 링크/메시 내부는 FK가 매 프레임 재저작하므로 사용자 편집 대상이 아니다.)
        """
        from pxr import Usd, UsdGeom

        out = {}
        wroot = stage.GetPrimAtPath("/World")
        if not wroot or not wroot.IsValid():
            return out
        for pr in Usd.PrimRange(wroot):
            path = str(pr.GetPath())
            if path == "/World":
                continue
            if path.count("/") > 3:  # /World/A/B = 3
                continue
            if not pr.IsA(UsdGeom.Xformable):
                continue
            out[path] = pr
        return out

    step = 0
    while simulation_app.is_running():
        simulation_app.update()
        step += 1
        # ★가시율/TCP는 자주 갱신(≈0.1초) — 슬라이더를 돌리는 즉시 숫자가 따라오도록★
        if step % 6 == 0:
            try:
                _pct, _tcp, _px, _md = _visibility()
                vis_label.text = (f"visibility {_pct:5.1f}%   cube {_px[0]:.1f} x {_px[1]:.1f} px"
                                  if _pct > 0 else "visibility 0.0%   [!] CUBE NOT IN VIEW")
                vis_label.style = {
                    "font_size": 16,
                    "color": (0xFF00FF00 if _pct >= 90 else 0xFF00FFFF if _pct >= 50 else 0xFF0000FF),
                }
                tcp_label.text = (f"TCP(base) ({_tcp[0]:+.3f}, {_tcp[1]:+.3f}, {_tcp[2]:+.3f})m"
                                  + (f" | mean dist {_md:.2f}m" if _md == _md else ""))
                # ── 파지 상태 ──
                _gap, _pen, _f, _cov, _d = _grip_state()
                _cube = OBJECT_SIZE * 1000.0
                gap_label.text = (f"gap {_gap:5.1f}mm   cube {_cube:.0f}mm   "
                                  f"(open {_GS['open_gap']*1000:.0f} / close "
                                  f"{(_GS['open_gap']-2*_GS['close_q'])*1000:.0f})")
                if _pen > 0:
                    pen_label.text = (f"penetration {_pen:4.1f}mm/side  x stiffness "
                                      f"{_GS['stiffness']:.0f}  =  {_f:5.1f}N   "
                                      f"| pad covers {_cov:.0f}% of cube")
                else:
                    pen_label.text = (f"clearance {-_pen:4.1f}mm/side (not touching)   "
                                      f"| pad covers {_cov:.0f}% of cube")
                if _cube >= _GS["open_gap"] * 1000.0:
                    fit_label.text = "[X] CUBE TOO BIG - does not fit when open"
                    fit_label.style = {"font_size": 15, "color": 0xFF3333FF}
                elif _pen <= 0:
                    fit_label.text = "[ ] OPEN - close the gripper to grip"
                    fit_label.style = {"font_size": 15, "color": 0xFF88CCFF}
                elif _f < 15.0:
                    fit_label.text = f"[!] WEAK GRIP {_f:.0f}N (validated baseline 23.4N)"
                    fit_label.style = {"font_size": 15, "color": 0xFF00CCFF}
                else:
                    fit_label.text = f"[O] GRIPPED - {_f:.0f}N, {_pen:.1f}mm/side"
                    fit_label.style = {"font_size": 15, "color": 0xFF00FF00}
                if _d == _d:
                    _ok = _d < OBJECT_SIZE * 1000.0 * 0.5
                    algn_label.text = (f"TCP <-> cube center {_d:5.1f}mm  "
                                       + ("(aligned)" if _ok else "(move cube or arm)"))
                    algn_label.style = {"color": 0xFF88DD88 if _ok else 0xFFAAAAAA}
            except Exception:  # noqa: BLE001
                pass
        if step % 60 != 0:
            continue
        rp = _pose_full(robot_g)
        bp = _pose_full(ghost)
        rx, ry, rz = rp["pos"]
        rr = math.radians(rp["rpy_deg"][2])
        rr_deg = rp["rpy_deg"][2]

        # ── 벨트를 ★로봇 base 프레임★으로 (학습 cfg가 쓰는 좌표계) ──
        R_r = _quat_to_mat(np.array(rp["quat"]))
        R_b = _quat_to_mat(np.array(bp["quat"]))
        d_w = np.array(bp["pos"]) - np.array(rp["pos"])
        belt_rel_pos = R_r.T @ d_w
        belt_rel_quat = _mat_to_quat(R_r.T @ R_b)
        belt_x, belt_y, belt_z = (float(v) for v in belt_rel_pos)
        belt_yaw = math.degrees(math.atan2((R_r.T @ R_b)[1, 0], (R_r.T @ R_b)[0, 0]))
        bsx, bsy, bsz = bp["scale"]
        jd = {n: round(m.get_value_as_float(), 1) for n, m in arm_models.items()}

        # ⚠️ 축정렬 전제(스폰/타겟 샘플링)를 깨는 배치는 즉시 경고 — 조용한 불일치 방지
        warns = []
        if abs(rp["rpy_deg"][0]) > 0.5 or abs(rp["rpy_deg"][1]) > 0.5:
            warns.append(f"robot roll/pitch={rp['rpy_deg'][0]:+.2f}/{rp['rpy_deg'][1]:+.2f}deg - "
                         "training cfg assumes robot base is level and at the origin")
        if abs(bp["rpy_deg"][0]) > 0.5 or abs(bp["rpy_deg"][1]) > 0.5:
            warns.append(f"belt roll/pitch={bp['rpy_deg'][0]:+.2f}/{bp['rpy_deg'][1]:+.2f}deg - "
                         "unsupported (flat-belt assumption)")
        # ★0 / ±180만 지원★ (2026-07-28(2차)): 180°는 (x,y)→(−x,−y)로 축정렬 박스를 보존하므로
        # 안전하다(로봇이 벨트를 등지는 실셀 배치). 그 외 각도만 경고 — scene.py 게이트와 동일 기준.
        if min(abs(belt_yaw), abs(abs(belt_yaw) - 180.0)) > 0.5:
            warns.append(f"belt yaw={belt_yaw:+.2f}deg - spawn/target sampling is axis-aligned; "
                         "only 0 / +-180 allowed (scene.py belt_yaw_deg gate refuses startup)")
        if abs(belt_z) > 0.005:
            warns.append(f"belt z={belt_z:+.4f}m - differs from sim.yaml table_height_m; "
                         "to change it move table_height_m AND common.yaml target.z together")
        if abs(bsx - 1.0) > 0.01 or abs(bsy - 1.0) > 0.01 or abs(bsz - 1.0) > 0.01:
            warns.append(f"belt scale=({bsx:.3f},{bsy:.3f},{bsz:.3f}) - size must go into "
                         "sim.yaml belt_size_m, not scale (real size = size x scale)")

        print(f"[place] belt(base): pos=({belt_x:+.4f},{belt_y:+.4f},{belt_z:+.4f}) yaw={belt_yaw:+.2f}deg "
              f"scale=({bsx:.3f},{bsy:.3f},{bsz:.3f}) | robot world=({rx:+.3f},{ry:+.3f},{rz:+.3f}) "
              f"yaw={rr_deg:+.2f}deg" + (f"  [warn] x{len(warns)}" if warns else ""))

        rec = {
            # ① 학습용(로봇 base 기준) — 구 키 유지(하위호환: dump_scene_layout --from-gui가 읽는다)
            "belt_center_x": round(belt_x, 4),
            "belt_center_y": round(belt_y, 4),
            "belt_yaw_deg": round(belt_yaw, 2),
            "belt_scale_x": round(bsx, 4),
            "belt_scale_y": round(bsy, 4),
            # ★확장: 전체 P/R/S★
            "belt_center_z": round(belt_z, 4),
            "belt_scale_z": round(bsz, 4),
            "belt_quat_wxyz_base": [round(float(v), 6) for v in belt_rel_quat],
            "belt_pose_world": {
                "pos": [round(v, 4) for v in bp["pos"]],
                "quat_wxyz": [round(v, 6) for v in bp["quat"]],
                "scale": [round(v, 4) for v in bp["scale"]],
                "rpy_deg": [round(v, 3) for v in bp["rpy_deg"]],
            },
            # ② 로봇 world pose·scale
            "robot_world_x": round(rx, 4),
            "robot_world_y": round(ry, 4),
            "robot_world_yaw_deg": round(rr_deg, 2),
            "robot_scale": round(rp["scale"][0], 4),
            "robot_pose_world": {
                "pos": [round(v, 4) for v in rp["pos"]],
                "quat_wxyz": [round(v, 6) for v in rp["quat"]],
                "scale": [round(v, 4) for v in rp["scale"]],
                "rpy_deg": [round(v, 3) for v in rp["rpy_deg"]],
            },
            # ③ 관절
            "ghost_joints_deg": jd,
            "ghost_gripper_close_mm": round(grip_model.get_value_as_float(), 1),
            # ④ 자식 고스트(로컬 배치 — 벨트/로봇을 따라 움직이는 것들)
            "child_ghosts_world": {},
            "warnings": warns,
            "meta": {
                "step": step,
                "tool": "scripts/tools/place_workspace_gui.py",
                "frame_note": "belt_* = robot base frame, *_world = USD world frame. quat=wxyz, units m/deg.",
                "next_step": "python scripts/tools/dump_scene_layout.py --from-gui -> updates "
                             "configs/calib/scene_layout.yaml; then reflect into configs/sim.yaml "
                             "scene.* (that is where training picks it up)",
            },
        }
        for _n, _p in _CHILD_GHOSTS.items():
            _prim = stage.GetPrimAtPath(_p)
            if _prim and _prim.IsValid():
                _cp = _pose_full(_prim)
                rec["child_ghosts_world"][_n] = {
                    "prim_path": _p,
                    "pos": [round(v, 4) for v in _cp["pos"]],
                    "quat_wxyz": [round(v, 6) for v in _cp["quat"]],
                    "scale": [round(v, 4) for v in _cp["scale"]],
                }
        # ★전 프림 스냅샷(사각지대 0)★ — 어떤 프림을 잡았든 여기에는 남는다.
        rec["all_prims_world"] = {}
        for _p, _prim in _all_editable_prims().items():
            _cp = _pose_full(_prim)
            rec["all_prims_world"][_p] = {
                "pos": [round(v, 4) for v in _cp["pos"]],
                "quat_wxyz": [round(v, 6) for v in _cp["quat"]],
                "scale": [round(v, 4) for v in _cp["scale"]],
            }
        # ★벨트 실치수는 '부모 scale'이 아니라 '벨트판 자식'이 진실★(2026-07-28 실사고):
        #   사용자가 부모 Belt_Ghost가 아니라 자식 belt 큐브를 직접 스케일하면 belt_scale_*는
        #   1.0 그대로라 변경이 학습용 필드에 안 잡힌다 → 자식 half-extent로 실치수를 병기한다.
        _bp = rec["child_ghosts_world"].get("belt_plate")
        if _bp:
            rec["belt_size_m_from_plate"] = [round(v * 2.0, 4) for v in _bp["scale"]]
            if any(abs(a - b) > 1e-3 for a, b in zip(rec["belt_size_m_from_plate"], BELT_SIZE)):
                rec["warnings"].append(
                    f"belt plate real size {rec['belt_size_m_from_plate']} != sim.yaml belt_size_m "
                    f"{list(BELT_SIZE)} - put it into configs/sim.yaml scene.belt_size_m to take effect "
                    "(also review pick_zone_m / target range)"
                )
        _tz = rec["child_ghosts_world"].get("target_zone")
        if _tz:
            rec["target_range_m_from_zone"] = {
                "x": [round(_tz["pos"][0] - _tz["scale"][0], 4), round(_tz["pos"][0] + _tz["scale"][0], 4)],
                "y": [round(_tz["pos"][1] - _tz["scale"][1], 4), round(_tz["pos"][1] + _tz["scale"][1], 4)],
            }
            _cur = [[float(_TGT["x"][0]), float(_TGT["x"][1])], [float(_TGT["y"][0]), float(_TGT["y"][1])]]
            _new = [rec["target_range_m_from_zone"]["x"], rec["target_range_m_from_zone"]["y"]]
            if any(abs(a - b) > 1e-3 for p, q in zip(_cur, _new) for a, b in zip(p, q)):
                rec["warnings"].append(
                    f"target zone range {rec['target_range_m_from_zone']} != common.yaml "
                    f"target.sample_range {{'x': {_cur[0]}, 'y': {_cur[1]}}} - "
                    "OBSERVATION CONTRACT CHANGE = requires retraining"
                )
        with open(_OUT, "w", encoding="utf-8") as f:
            f.write(f"# place_workspace_gui.py record - Position/Rotation/Scale of all objects (step={step})\n")
            f.write("# GUI session record. To apply: python scripts/tools/dump_scene_layout.py --from-gui\n")
            yaml.safe_dump(rec, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        if warns and step % 600 == 0:  # 10초마다 한 번만 상세 경고
            for w in warns:
                print(f"[place][warn] {w}")

    print(f"[place] exit - last P/R/S saved to {_OUT}")
    print("[place] [NEXT]: python scripts\\tools\\dump_scene_layout.py --from-gui")
    print("[place]   -> updates configs/calib/scene_layout.yaml (object registry), "
          "then reflect into configs/sim.yaml scene.*")


if __name__ == "__main__":
    main()
    simulation_app.close()
