#!/usr/bin/env python
"""★전 객체 배치 레지스트리(Position/Rotation/Scale) 생성기★ — configs → scene_layout.yaml.

목적(2026-07-27 사용자 지시): "모든 객체에 대해 언제든지 Position/Rotation/Scale을 기록해서
      앞으로의 환경 설정을 할 수 있도록" — 장면 객체의 배치가 sim.yaml·calib/*.yaml·코드 상수로
      흩어져 있어 GUI로 배치를 바꿔도 어디를 고쳐야 하는지 알 수 없던 문제를 해소한다.
      이 스크립트가 ★현재 설정이 만들어내는 실제 배치 수치★를 계산해 한 파일로 기록한다.

★역할 구분(중요)★
  - 이 레지스트리(configs/calib/scene_layout.yaml)는 ★기록·감사·GUI 왕복용 스냅샷★이다.
    학습 env가 직접 소비하지 않는다(소비 원본은 각 object의 source_of_truth 필드가 가리킨다).
  - 배치를 실제로 바꾸려면 source_of_truth(대부분 configs/sim.yaml scene.*)를 고치고
    이 스크립트를 다시 돌려 레지스트리를 갱신한다. 그래야 sim/real·학습/GUI가 갈라지지 않는다.

실행 (isaac 불필요 — numpy/yaml만):
  python scripts\\tools\\dump_scene_layout.py                       # configs 기준으로 재생성
  python scripts\\tools\\dump_scene_layout.py --from-gui            # GUI 실측(Phase1/logs) 반영해 재생성
  python scripts\\tools\\dump_scene_layout.py --check               # 갱신 없이 현재 파일과 차이만 보고(CI용)
산출: configs/calib/scene_layout.yaml  (§4-5 커밋 대상 — 어느 배치로 어느 실험을 했는지가 재현성)

⚠️ isaaclab을 import하지 않는다(순수 python) — mdp/scene.py의 배치 계산식을 '같은 입력'으로
   재현한다. 계산식이 갈라지지 않도록, scene.py는 하드코딩 대신 sim.yaml을 읽도록 승격돼 있고
   (2026-07-27), 아래 _MIRRORED 상수들만 코드에서 미러링한다(각각 출처 주석 명기).
"""
from __future__ import annotations

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

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

import argparse
import datetime as _dt
import math
from pathlib import Path

import numpy as np
import yaml

from common.calibration import (
    load_hand_eye,
    load_scene_registration,
    load_tcp_offset,
    mat_to_quat,
)
from common.config_io import configs_dir, load_yaml
from common.robot_spec import UR5E_ARM_JOINT_NAMES, UR5E_HOME_JOINT_POS

# ─────────────────────────────────────────────────────────────────────────────
# 코드에서만 정의되는 상수 미러링 — 각 출처를 명기(값 변경 시 양쪽 동기 필요)
# ─────────────────────────────────────────────────────────────────────────────
# mdp/scene.py:D435_MESH_IN_OPTICAL — d435.dae 메시축(렌즈=+y_mesh) → optical 관례 보정.
_MIRRORED_D435_MESH_IN_OPTICAL = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
)
# mdp/scene.py:_make_belt / _static_box 색 (기록용 — 배치와 함께 남겨두면 GUI 재현이 쉽다)
_MIRRORED_COLORS = {
    "belt": (0.08, 0.30, 0.18),
    "belt_stand": (0.62, 0.63, 0.66),
    "robot_stand": (0.62, 0.63, 0.66),
}
# mdp/events_dr.py:_SCATTER_MARGIN = OBJECT_SIZE/2 + 0.05 (벨트 가장자리 여유)
_MIRRORED_SCATTER_MARGIN_EXTRA = 0.05
# mdp/events_dr.py:_PREGRASP_GRIP — 홈 포즈 finger pad 중점(파지중심). 홈 포즈 변경 시 재실측 필요.
_MIRRORED_PREGRASP_GRIP = (-0.2975, -0.1328, 0.5091)  # 2026-07-30 툴 체인 5단 정정과 동기
# mdp/scene.py:make_nurec_belt_visual_cfg(sink_m=0.003)
_MIRRORED_NUREC_BELT_SINK_M = 0.003
# mdp/scene.py:BinPickSceneCfg.light — DomeLight
_MIRRORED_LIGHT = {"intensity": 3000.0, "color": (0.75, 0.75, 0.75)}

_IDENT_QUAT = [1.0, 0.0, 0.0, 0.0]


def _r(v, nd: int = 6):
    """재귀 반올림 — yaml에 긴 부동소수 꼬리를 남기지 않는다."""
    if isinstance(v, (list, tuple)):
        return [_r(x, nd) for x in v]
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return round(float(v), nd)


def _obj(
    prim_path: str,
    position_m,
    rotation_quat_wxyz=None,
    scale=None,
    size_m=None,
    source_of_truth: str = "",
    defined_in: str = "",
    editable_via: str = "",
    frame: str = "base",
    note: str = "",
    extra: dict | None = None,
) -> dict:
    d = {
        "prim_path": prim_path,
        "frame": frame,
        "position_m": _r(position_m) if position_m is not None else None,
        "rotation_quat_wxyz": _r(rotation_quat_wxyz if rotation_quat_wxyz is not None else _IDENT_QUAT),
        "scale": _r(scale if scale is not None else [1.0, 1.0, 1.0]),
        "size_m": _r(size_m) if size_m is not None else None,
        "source_of_truth": source_of_truth,
        "defined_in": defined_in,
        "editable_via": editable_via,
    }
    if note:
        d["note"] = note
    if extra:
        d.update(extra)
    return d


def build_registry(gui_yaml: dict | None = None) -> dict:
    """configs(+선택적 GUI 실측)에서 전 객체의 P/R/S를 계산해 레지스트리 dict 생성."""
    cdir = configs_dir()
    sim_y = load_yaml(cdir / "sim.yaml")
    common_y = load_yaml(cdir / "common.yaml")
    sc = sim_y["scene"]

    num_objects = int(sc["num_objects"])
    bx, by, bz = (float(v) for v in sc["belt_size_m"])
    obj_size = float(sc["object_size_m"])
    table_h = float(sc["table_height_m"])
    belt_cx, belt_cy = (float(v) for v in sc.get("belt_center_xy_m", [0.45, 0.0]))
    belt_yaw_deg = float(sc.get("belt_yaw_deg", 0.0))
    stand_h = float(sc.get("stand_height_m", 0.80))

    # ── GUI 실측 반영(선택) — place_workspace_gui.py 산출 ──
    gui_note = ""
    gui_warnings: list[str] = []
    gui_target_range: dict | None = None
    if gui_yaml:
        belt_cx = float(gui_yaml.get("belt_center_x", belt_cx))
        belt_cy = float(gui_yaml.get("belt_center_y", belt_cy))
        belt_yaw_deg = float(gui_yaml.get("belt_yaw_deg", belt_yaw_deg))
        gui_note = "GUI 실측(place_workspace_gui.py) 반영 — configs/sim.yaml에도 같은 값을 반영해야 학습에 적용된다"
        # GUI가 남긴 경고를 그대로 승계(조용한 불일치 방지)
        gui_warnings = list(gui_yaml.get("warnings") or [])
        # 벨트 크기: GUI는 '스케일 배율'로 조정되지만 학습 계약은 실치수(belt_size_m)다 → 환산해 알려준다
        gsx = float(gui_yaml.get("belt_scale_x", 1.0))
        gsy = float(gui_yaml.get("belt_scale_y", 1.0))
        gsz = float(gui_yaml.get("belt_scale_z", 1.0))
        if max(abs(gsx - 1.0), abs(gsy - 1.0), abs(gsz - 1.0)) > 0.01:
            gui_warnings.append(
                f"GUI 벨트 scale=({gsx:.3f},{gsy:.3f},{gsz:.3f}) → 실치수 환산 "
                f"belt_size_m ≈ [{bx * gsx:.4f}, {by * gsy:.4f}, {bz * gsz:.4f}] "
                "— ★scale이 아니라 sim.yaml scene.belt_size_m에 이 값을 넣어야 학습에 반영된다★"
            )
        # ★자식 프림 기반 실치수(2026-07-28 실사고 반영)★: 사용자가 부모 Belt_Ghost가 아니라
        #   자식 belt 큐브/타겟존을 직접 스케일하면 belt_scale_*는 1.0 그대로라 변경을 놓친다.
        #   → child_ghosts_world의 half-extent(×2)를 실치수의 진실로 삼는다.
        _ch = gui_yaml.get("child_ghosts_world") or {}
        _plate = _ch.get("belt_plate")
        if _plate and _plate.get("scale"):
            _new_size = [round(float(v) * 2.0, 4) for v in _plate["scale"]]
            if any(abs(a - b) > 1e-3 for a, b in zip(_new_size, (bx, by, bz))):
                gui_warnings.append(
                    f"★벨트 실치수 변경 감지★ {(_new_size)} ≠ sim.yaml belt_size_m {[bx, by, bz]} "
                    "— 레지스트리에는 GUI 값을 기록하되, ★학습에 적용하려면 configs/sim.yaml "
                    "scene.belt_size_m을 갱신하고 산란/타겟 범위·safety workspace를 함께 검토★"
                )
                bx, by, bz = _new_size
        _tz = _ch.get("target_zone")
        if _tz and _tz.get("scale") and _tz.get("pos"):
            gui_target_range = {
                "x": [round(_tz["pos"][0] - _tz["scale"][0], 4), round(_tz["pos"][0] + _tz["scale"][0], 4)],
                "y": [round(_tz["pos"][1] - _tz["scale"][1], 4), round(_tz["pos"][1] + _tz["scale"][1], 4)],
            }
            _cur_t = common_y["target"]["sample_range"]
            if (abs(gui_target_range["x"][0] - float(_cur_t["x"][0])) > 1e-3
                    or abs(gui_target_range["x"][1] - float(_cur_t["x"][1])) > 1e-3
                    or abs(gui_target_range["y"][0] - float(_cur_t["y"][0])) > 1e-3
                    or abs(gui_target_range["y"][1] - float(_cur_t["y"][1])) > 1e-3):
                gui_warnings.append(
                    f"★타겟존 범위 변경 감지★ {gui_target_range} ≠ common.yaml target.sample_range "
                    "— ★target_pos는 정책 관측 슬롯이므로 분포 변경 = 재학습 대상★"
                )
        # 벨트 z(상면=base z=0 가정) 이탈
        gz = float(gui_yaml.get("belt_center_z", 0.0))
        if abs(gz) > 0.005:
            gui_warnings.append(
                f"GUI 벨트 z={gz:+.4f}m — 벨트 상면=로봇 base z=0 가정 이탈. 반영하려면 "
                "sim.yaml scene.table_height_m ★과 common.yaml target.sample_range.z를 함께★ 이동할 것 "
                "(한쪽만 옮겨 placed가 영구 0이 된 과거 사고 있음)."
            )
        for _w in gui_warnings:
            print(f"[layout][warn] ⚠️ {_w}")

    belt_quat = [
        math.cos(math.radians(belt_yaw_deg) / 2.0), 0.0, 0.0,
        math.sin(math.radians(belt_yaw_deg) / 2.0),
    ]

    # ── 캘리브 변환 ──
    T_fc = load_hand_eye(cdir / "calib" / "hand_eye.yaml")           # flange → camera(optical)
    cam_pos = [float(v) for v in T_fc[:3, 3]]
    cam_quat = [float(v) for v in mat_to_quat(T_fc[:3, :3])]
    body_quat = [float(v) for v in mat_to_quat(T_fc[:3, :3] @ _MIRRORED_D435_MESH_IN_OPTICAL)]

    T_ft = load_tcp_offset(cdir / "calib" / "tcp_offset.yaml")       # flange → TCP(파지중심)
    tcp_pos = [float(v) for v in T_ft[:3, 3]]
    tcp_quat = [float(v) for v in mat_to_quat(T_ft[:3, :3])]

    nurec_path = cdir / "calib" / "scene_registration.yaml"
    nurec = None
    if nurec_path.is_file():
        try:
            T_bn, nurec_scale = load_scene_registration(nurec_path)
            nurec = {
                "pos": [float(v) for v in T_bn[:3, 3]],
                "quat": [float(v) for v in mat_to_quat(T_bn[:3, :3])],
                "scale": float(nurec_scale),
            }
        except Exception as e:  # noqa: BLE001
            print(f"[layout][warn] scene_registration 로드 실패 — nurec 항목 생략: {e}")

    objects: dict[str, dict] = {}

    # ① 바닥
    objects["ground_plane"] = _obj(
        "/World/GroundPlane",
        [0.0, 0.0, table_h - stand_h],
        source_of_truth="sim.yaml scene.table_height_m - scene.stand_height_m",
        defined_in="mdp/scene.py BinPickSceneCfg.plane",
        editable_via="configs/sim.yaml scene.stand_height_m",
        note="무한 평면(GroundPlaneCfg). 벨트 밖 낙하가 여기까지 떨어져 드롭 종료를 발화.",
    )

    # ② 로봇 (UR5e + TCW1 + 2FG7 결합 USD)
    objects["robot"] = _obj(
        "{ENV_REGEX_NS}/Robot",
        [0.0, 0.0, 0.0],
        source_of_truth="mdp/ur5e_assets.py UR5E_CFG.init_state.pos (= env origin 고정)",
        defined_in="mdp/ur5e_assets.py:UR5E_CFG",
        editable_via="★고정 권장★ — base = env origin이 요소4(hand_eye/tcp_offset) 전제. 옮기려면 캘리브 계약 재검토",
        note="USD = assets/ur5e_2fg7/ur5e_tcw1_2fg7.usd (UR5e + Magbot TCW1 + OnRobot 2FG7 + D435i 브라켓)",
        extra={
            "home_joints_rad": {j: _r(UR5E_HOME_JOINT_POS[j]) for j in UR5E_ARM_JOINT_NAMES},
            "home_joints_deg": {j: _r(math.degrees(UR5E_HOME_JOINT_POS[j]), 2) for j in UR5E_ARM_JOINT_NAMES},
            "gripper_joint_range_m": [0.0, 0.020],
        },
    )

    # ③ TCP 프레임(FrameTransformer offset — 물리 프림 아님)
    objects["tcp_frame"] = _obj(
        "{ENV_REGEX_NS}/Robot/wrist_3_link  →(offset)  TCP",
        tcp_pos,
        tcp_quat,
        frame="flange(wrist_3_link)",
        source_of_truth="configs/calib/tcp_offset.yaml T_flange_tcp",
        defined_in="mdp/scene.py BinPickSceneCfg.ee_frame",
        editable_via="deployment/Calib_TcpOffset.py(실측) 또는 configs/calib/tcp_offset.yaml",
        note="z=0.2262 = TCW1(86.5mm) 포함 파지중심. 프림이 아니라 센서 프레임이라 GUI 기즈모로 못 움직인다.",
    )

    # ④ 벨트(워크스페이스) — 물리 콜라이더 + 시각
    objects["belt"] = _obj(
        "{ENV_REGEX_NS}/Belt",
        [belt_cx, belt_cy, table_h - bz / 2.0],
        belt_quat,
        size_m=[bx, by, bz],
        source_of_truth="sim.yaml scene.belt_size_m + scene.belt_center_xy_m + scene.table_height_m",
        defined_in="mdp/scene.py _make_belt()",
        editable_via="configs/sim.yaml scene.belt_center_xy_m / belt_size_m",
        note=f"상면 = z={_r(table_h)} (base). 색 {_MIRRORED_COLORS['belt']} (실셀 초록 컨베이어 근사). "
        + (gui_note or "yaw는 0만 지원(스폰/타겟 샘플링이 축정렬)"),
    )

    # ⑤ 벨트 스탠드 / ⑥ 로봇 스탠드 (시각·낙하 스캐폴딩)
    objects["belt_stand"] = _obj(
        "{ENV_REGEX_NS}/BeltStand",
        [belt_cx, belt_cy, table_h - bz - (stand_h - bz) / 2.0],
        size_m=[bx * 0.7, by * 0.5, stand_h - bz],
        source_of_truth="mdp/scene.py (belt_size_m·stand_height_m 파생 — 비율 0.7/0.5는 코드 상수)",
        defined_in="mdp/scene.py BinPickSceneCfg.belt_stand",
        editable_via="configs/sim.yaml scene.stand_height_m (비율은 코드)",
        note="정적 큐보이드(콜라이더 있음, 물리 바디 없음). 학습 무영향 스캐폴딩.",
    )
    objects["robot_stand"] = _obj(
        "{ENV_REGEX_NS}/RobotStand",
        [0.0, 0.0, (table_h - stand_h) / 2.0],
        # ★2026-08-11 도면 정합★ 0.22 → 0.21 (도면: 마운트 플레이트·지지대 210×210).
        size_m=[0.21, 0.21, stand_h - table_h],
        source_of_truth="mdp/scene.py (0.21 각기둥은 코드 상수, 높이는 stand_height_m 파생)",
        defined_in="mdp/scene.py BinPickSceneCfg.robot_stand",
        editable_via="configs/sim.yaml scene.stand_height_m (단면 0.21은 코드)",
        note="상면 = 로봇 base(z=0)에 정확히 닿게 디커플됨.",
    )

    # ⑦ 조명
    objects["dome_light"] = _obj(
        "/World/light",
        None,
        source_of_truth="mdp/scene.py BinPickSceneCfg.light (intensity DR = configs/dr.yaml lighting)",
        defined_in="mdp/scene.py BinPickSceneCfg.light",
        editable_via="mdp/scene.py (세기 랜덤화 범위는 configs/dr.yaml)",
        frame="world",
        note="DomeLight — 위치 개념 없음(전역). 세기는 events_dr.randomize_light_intensity가 런타임 변경.",
        extra={"intensity": _MIRRORED_LIGHT["intensity"], "color": list(_MIRRORED_LIGHT["color"])},
    )

    # ⑧ 산란 큐브 — 스폰 기준 + 에피소드 랜덤 범위
    #   ★2026-07-28(3차)★ 산란은 물리 벨트가 아니라 ★픽존★에서 파생한다(events_dr와 동일 식).
    #   물리 벨트(100cm)가 UR5e 도달한계보다 커지면서 분리됐다 — sim.yaml scene.pick_zone_m.
    pzx, pzy = (float(v) for v in sc.get("pick_zone_m", [bx, by]))
    margin = obj_size / 2.0 + _MIRRORED_SCATTER_MARGIN_EXTRA
    half_x = max(pzx / 2.0 - margin, 0.01)
    half_y = max(pzy / 2.0 - margin, 0.01)
    for i in range(num_objects):
        spawn_z = table_h + obj_size / 2.0 + 0.04 + (obj_size + 0.02) * i
        objects[f"object_{i}"] = _obj(
            f"{{ENV_REGEX_NS}}/Object_{i}",
            [belt_cx, belt_cy, spawn_z],
            size_m=[obj_size, obj_size, obj_size],
            source_of_truth="sim.yaml scene.object_size_m + belt_center_xy_m (스폰 z는 scene.py 계단식)",
            defined_in="mdp/scene.py make_scatter_object_cfg()",
            editable_via="configs/sim.yaml scene.object_size_m / num_objects",
            note="★위 position은 '초기 스폰'일 뿐★ — 매 에피소드 events_dr.scatter_*가 아래 범위로 재배치한다.",
            extra={
                "mass_kg": 0.05,
                "friction_static_dynamic": [1.0, 1.0],
                "reset_scatter_range_base_m": {
                    "x": _r([belt_cx - half_x, belt_cx + half_x]),
                    "y": _r([belt_cy - half_y, belt_cy + half_y]),
                    "z": _r([spawn_z, spawn_z + 0.05]),
                    "yaw_rad": [-3.1416, 3.1416],
                },
            },
        )

    # ⑨ 손목 카메라(TiledCamera) + ⑩ D435i 본체(시각)
    objects["wrist_camera"] = _obj(
        "{ENV_REGEX_NS}/Robot/wrist_3_link/WristCam",
        cam_pos,
        cam_quat,
        frame="flange(wrist_3_link)",
        source_of_truth="configs/calib/hand_eye.yaml T_flange_cam (★sim/real 공유 단일 소스★)",
        defined_in="mdp/scene.py make_tiled_camera_cfg()",
        editable_via="deployment/Calib_HandEye.py(실측) — 손으로 고치지 말 것(요소4)",
        note="convention=ros(광축 +Z). 해상도/FOV는 sim.yaml camera. 실측-공칭 차가 dr.yaml camera_pose "
        "범위(±1cm/±1°)를 넘으면 재학습 판정(§4-5).",
        extra={
            "resolution_px": [int(sim_y["camera"]["width"]), int(sim_y["camera"]["height"])],
            "vertical_fov_deg": float(sim_y["camera"]["vertical_fov_deg"]),
            "horizontal_fov_deg": float(sim_y["camera"].get("horizontal_fov_deg", 0.0)),
            "clipping_near_m": float(sim_y["camera"]["clipping_near_m"]),
        },
    )
    objects["d435i_body_visual"] = _obj(
        "{ENV_REGEX_NS}/Robot/wrist_3_link/D435iBody",
        cam_pos,
        body_quat,
        frame="flange(wrist_3_link)",
        source_of_truth="configs/calib/hand_eye.yaml T_flange_cam @ D435_MESH_IN_OPTICAL(코드 상수)",
        defined_in="mdp/scene.py make_d435i_body_cfg()",
        editable_via="hand_eye.yaml (본체는 카메라를 자동 추종 — 따로 배치하지 않는다)",
        note="시각 전용(콜라이더·질량 없음). 회전만 메시축 보정이 들어가 카메라와 다르다.",
    )

    # ⑪ NuRec 실셀 배경(3DGS) + ⑫ NuRec depth 플레이트 — NuRec env 전용
    if nurec is not None:
        objects["nurec_background"] = _obj(
            "{ENV_REGEX_NS}/NuRecBackground",
            nurec["pos"],
            nurec["quat"],
            scale=[nurec["scale"]] * 3,
            source_of_truth="configs/calib/scene_registration.yaml T_base_nurec + scale",
            defined_in="mdp/scene.py make_nurec_background_cfg()",
            editable_via="scripts/tools/place_nurec_gui.py (GUI 배치 → 같은 yaml 갱신)",
            note="실셀 3DGS 스캔(assets/Scene/Workplace.usdz). ★NuRec 태스크에서만 부착★. rgb에만 렌더(depth 미기여).",
        )
        objects["belt_visual_nurec"] = _obj(
            "{ENV_REGEX_NS}/BeltVisualNuRec",
            [belt_cx, belt_cy, table_h - _MIRRORED_NUREC_BELT_SINK_M - bz / 2.0],
            belt_quat,
            size_m=[bx, by, bz],
            source_of_truth="sim.yaml scene.belt_* + sink 3mm(코드 상수)",
            defined_in="mdp/scene.py make_nurec_belt_visual_cfg()",
            editable_via="configs/sim.yaml scene.belt_* (sink는 코드)",
            note="NuRec env 전용 depth 대역 플레이트(무충돌). 물리 벨트는 비가시화되고 이 플레이트가 depth를 담당.",
        )

    # ⑬ 타겟존 — 프림이 아니라 커맨드 샘플 범위(관측에 들어가는 공간 계약)
    tgt = dict(common_y["target"]["sample_range"])
    if gui_target_range:  # GUI 실측이 있으면 그것을 기록(★반영은 common.yaml 갱신이 별도 필요★)
        tgt = {"x": gui_target_range["x"], "y": gui_target_range["y"], "z": tgt["z"]}
    objects["target_zone"] = _obj(
        "(프림 아님 — command: target_pos)",
        [
            (float(tgt["x"][0]) + float(tgt["x"][1])) / 2.0,
            (float(tgt["y"][0]) + float(tgt["y"][1])) / 2.0,
            (float(tgt["z"][0]) + float(tgt["z"][1])) / 2.0,
        ],
        size_m=[
            float(tgt["x"][1]) - float(tgt["x"][0]),
            float(tgt["y"][1]) - float(tgt["y"][0]),
            max(float(tgt["z"][1]) - float(tgt["z"][0]), 0.0),
        ],
        source_of_truth="configs/common.yaml target.sample_range (★sim/real 공유★)",
        defined_in="binpick_env_cfg.py CommandsCfg / common.yaml",
        editable_via="configs/common.yaml target.sample_range",
        note="place 목표의 균일 샘플 박스. 벨트를 옮기면 ★반드시 함께★ 옮겨야 한다(관측 계약).",
        extra={"range_m": {k: _r([float(v[0]), float(v[1])]) for k, v in tgt.items()}},
    )

    # ⑭ pre-grasp 보조 파지점 — 하드코딩 공간 상수(홈 포즈 종속)
    objects["pregrasp_assist_point"] = _obj(
        "(프림 아님 — reset 이벤트가 큐브를 여기 배치)",
        list(_MIRRORED_PREGRASP_GRIP),
        source_of_truth="mdp/events_dr.py _PREGRASP_GRIP (★하드코딩 — 홈 포즈 종속★)",
        defined_in="mdp/events_dr.py:_PREGRASP_GRIP",
        editable_via="diagnostics/measure_grasp_frame.py 재실측 (홈 포즈를 바꾸면 필수)",
        note="★홈 포즈 finger pad 중점★. 홈 포즈나 TCW1 체인을 바꾸면 이 값이 즉시 무효가 된다(과거 사고 지점). "
        "eval cfg에서는 pregrasp 보조가 꺼진다.",
    )

    return {
        "_readme": (
            "★전 객체 배치 레지스트리(Position/Rotation/Scale)★ — 생성기: scripts/tools/dump_scene_layout.py\n"
            "· 단위 m, 회전 quaternion = wxyz, frame 필드가 기준 좌표계(기본 base = UR5e base_link = env origin).\n"
            "· ★이 파일은 기록/감사/GUI 왕복용 스냅샷이다 — 학습 env가 직접 읽지 않는다.★\n"
            "  배치를 바꾸려면 각 객체의 source_of_truth(대개 configs/sim.yaml scene.*)를 고치고 생성기를 다시 실행.\n"
            "· GUI로 배치를 확정하는 절차: scripts/tools/place_workspace_gui.py 실행 → 기즈모/슬라이더로 배치 →\n"
            "  Phase1/logs/workspace_ghost_pose.yaml 기록 → `dump_scene_layout.py --from-gui` 로 이 파일 갱신 →\n"
            "  값 검토 후 configs/sim.yaml scene.* 에 반영(학습에 실제 적용되는 지점).\n"
            "· size_m은 큐보이드 실치수, scale은 USD 스케일 배율(3DGS 배경 등)."
        ),
        "schema_version": 2,
        "meta": {
            "date": _dt.date.today().isoformat(),
            "generated_by": "scripts/tools/dump_scene_layout.py",
            "generated_from": "configs/sim.yaml + common.yaml + calib/{hand_eye,tcp_offset,scene_registration}.yaml",
            "gui_source": "Phase1/logs/workspace_ghost_pose.yaml" if gui_yaml else None,
            "gui_warnings": gui_warnings or None,
            "frame_convention": "position [m], rotation quaternion wxyz, base frame = UR5e base_link = env origin",
            "num_objects": num_objects,
        },
        "derived_constants": {
            "table_height_m": _r(table_h),
            "stand_height_m": _r(stand_h),
            "belt_center_xy_m": _r([belt_cx, belt_cy]),
            "belt_yaw_deg": _r(belt_yaw_deg, 3),
            "pick_zone_m": [_r(v, 4) for v in (pzx, pzy)],  # 스폰/타겟 허용 구역(⊂ 벨트)
            "belt_size_m": _r([bx, by, bz]),
            "object_size_m": _r(obj_size),
        },
        "objects": objects,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="전 객체 배치 레지스트리 생성 (configs → scene_layout.yaml).")
    ap.add_argument("--from-gui", action="store_true",
                    help="Phase1/logs/workspace_ghost_pose.yaml(GUI 실측)을 반영해 생성")
    ap.add_argument("--gui-file", default=None, help="GUI 기록 파일 경로 재정의")
    ap.add_argument("--out", default=None, help="산출 경로 재정의 (기본 configs/calib/scene_layout.yaml)")
    ap.add_argument("--check", action="store_true", help="쓰지 않고 현재 파일과 차이만 보고(0=동일, 4=차이)")
    a = ap.parse_args()

    gui = None
    if a.from_gui or a.gui_file:
        gpath = Path(a.gui_file) if a.gui_file else Path(_REPO_ROOT) / "Phase1" / "logs" / "workspace_ghost_pose.yaml"
        if not gpath.is_file():
            print(f"[layout][error] GUI 기록 없음: {gpath}\n"
                  f"                먼저 scripts/tools/place_workspace_gui.py 를 실행해 배치를 기록하세요.")
            return 1
        gui = load_yaml(gpath)
        print(f"[layout] GUI 실측 반영: {gpath}")

    reg = build_registry(gui)
    out = Path(a.out) if a.out else configs_dir() / "calib" / "scene_layout.yaml"
    text = yaml.safe_dump(reg, allow_unicode=True, sort_keys=False, default_flow_style=False, width=120)

    if a.check:
        old = out.read_text(encoding="utf-8") if out.is_file() else ""
        if old.strip() == text.strip():
            print(f"[layout] ✅ 최신 상태 — {out}")
            return 0
        print(f"[layout] ⚠️ configs와 레지스트리가 다릅니다 — `python scripts/tools/dump_scene_layout.py`로 갱신하세요.\n"
              f"         대상: {out}")
        return 4

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    n = len(reg["objects"])
    print(f"[layout] ★기록 완료★ {out}  (객체 {n}개)")
    for name, o in reg["objects"].items():
        p = o.get("position_m")
        print(f"[layout]   - {name:22s} pos={p} scale={o.get('scale')} size={o.get('size_m')}")
    print("[layout] ※ 배치를 실제로 바꾸려면 각 항목의 source_of_truth(대개 configs/sim.yaml scene.*)를 수정할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
