"""configs/calib/scene_layout.yaml(전 객체 배치 레지스트리) 계약 테스트 — 순수(isaac 불필요).

목적(2026-07-27): 배치가 sim.yaml·calib/*.yaml·코드로 흩어져 있어 한쪽만 바뀌면 조용히
      어긋나던 문제를 테스트로 고정한다. 레지스트리는 scripts/tools/dump_scene_layout.py가
      생성하며, 이 테스트는 ①레지스트리가 현재 configs와 일치하는가(드리프트)
      ②배치 자체가 물리적으로 정합한가(스탠드 상면, 스폰/타겟이 벨트 안) 를 검증한다.

레지스트리가 낡으면: python scripts/tools/dump_scene_layout.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from common.calibration import load_hand_eye, load_tcp_offset
from common.config_io import configs_dir, load_yaml

_LAYOUT_PATH = configs_dir() / "calib" / "scene_layout.yaml"


@pytest.fixture(scope="module")
def layout() -> dict:
    if not _LAYOUT_PATH.is_file():
        pytest.skip(f"레지스트리 없음: {_LAYOUT_PATH} — dump_scene_layout.py 실행 필요")
    return load_yaml(_LAYOUT_PATH)


@pytest.fixture(scope="module")
def sim_scene() -> dict:
    return load_yaml(configs_dir() / "sim.yaml")["scene"]


def _pos(layout: dict, name: str) -> list[float]:
    return [float(v) for v in layout["objects"][name]["position_m"]]


def _size(layout: dict, name: str) -> list[float]:
    return [float(v) for v in layout["objects"][name]["size_m"]]


def test_registry_schema(layout: dict) -> None:
    """스키마 필수 필드 — 모든 객체가 P/R/S와 '어디를 고쳐야 하는지'를 갖는다."""
    assert layout.get("schema_version") == 2
    assert layout["objects"], "객체가 비어 있음"
    for name, o in layout["objects"].items():
        for key in ("prim_path", "frame", "rotation_quat_wxyz", "scale", "source_of_truth",
                    "defined_in", "editable_via"):
            assert key in o, f"{name}: '{key}' 누락"
        q = [float(v) for v in o["rotation_quat_wxyz"]]
        assert len(q) == 4
        assert abs(math.sqrt(sum(v * v for v in q)) - 1.0) < 1e-6, f"{name}: quat 비정규"
        assert len(o["scale"]) == 3


def test_derived_constants_match_sim_yaml(layout: dict, sim_scene: dict) -> None:
    """레지스트리 파생 상수 = configs/sim.yaml (드리프트 가드)."""
    d = layout["derived_constants"]
    assert d["table_height_m"] == pytest.approx(float(sim_scene["table_height_m"]))
    assert d["stand_height_m"] == pytest.approx(float(sim_scene.get("stand_height_m", 0.80)))
    assert d["object_size_m"] == pytest.approx(float(sim_scene["object_size_m"]))
    assert d["belt_size_m"] == pytest.approx([float(v) for v in sim_scene["belt_size_m"]])
    assert d["belt_center_xy_m"] == pytest.approx(
        [float(v) for v in sim_scene.get("belt_center_xy_m", [0.45, 0.0])]
    )


def test_belt_top_surface_is_table_height(layout: dict, sim_scene: dict) -> None:
    """★벨트 상면 = base z=table_height★ — 타겟 z(common.yaml)와 맞물린 핵심 계약.

    (과거 사고: 벨트만 내리고 target.z를 안 옮겨 placed가 영구 0이 됨 — sim.yaml 주석 참조.)
    """
    table_h = float(sim_scene["table_height_m"])
    bz = float(sim_scene["belt_size_m"][2])
    assert _pos(layout, "belt")[2] + bz / 2.0 == pytest.approx(table_h, abs=1e-9)


def test_stands_touch_their_surfaces(layout: dict, sim_scene: dict) -> None:
    """스탠드 상면 정합 — 로봇 받침대는 ★로봇 base(z=0)★, 벨트 받침대는 벨트 하면에 닿는다.

    ★두 기준은 별개다★ (2026-07-16 디커플, scene.py:550 주석): 로봇 base는 z=0 고정이고
    벨트 상면(table_height_m)은 실측에 따라 움직인다. 구 테스트는 로봇 받침대를 table_h와
    비교했는데, 당시 table_height_m=0.0이라 두 값이 우연히 같아 결함이 가려져 있었다
    (2026-07-28 벨트 −1cm 하강으로 노출). 로봇이 공중에 뜨거나 파묻히는 것을 잡는 것이 목적.
    """
    table_h = float(sim_scene["table_height_m"])
    bz = float(sim_scene["belt_size_m"][2])
    rs_z, rs_h = _pos(layout, "robot_stand")[2], _size(layout, "robot_stand")[2]
    assert rs_z + rs_h / 2.0 == pytest.approx(0.0, abs=1e-9), (
        "로봇 받침대 상면 ≠ 로봇 base(z=0) — 받침대는 벨트가 아니라 base 높이에 맞춰야 한다"
    )
    bs_z, bs_h = _pos(layout, "belt_stand")[2], _size(layout, "belt_stand")[2]
    assert bs_z + bs_h / 2.0 == pytest.approx(table_h - bz, abs=1e-9), "벨트 받침대 상면 ≠ 벨트 하면"


def test_ground_plane_below_stands(layout: dict, sim_scene: dict) -> None:
    """바닥은 스탠드 아래 — 벨트 밖 낙하가 바닥까지 떨어져 드롭 종료가 발화해야 한다."""
    stand_h = float(sim_scene.get("stand_height_m", 0.80))
    table_h = float(sim_scene["table_height_m"])
    assert _pos(layout, "ground_plane")[2] == pytest.approx(table_h - stand_h, abs=1e-9)
    assert _pos(layout, "ground_plane")[2] < _pos(layout, "belt")[2]


def test_scatter_range_inside_belt(layout: dict, sim_scene: dict) -> None:
    """★큐브 산란 범위 ⊂ 벨트 상면★ — 벗어나면 스폰 직후 굴러떨어져 학습이 죽는다."""
    bx, by, _ = (float(v) for v in sim_scene["belt_size_m"])
    cx, cy = (float(v) for v in sim_scene.get("belt_center_xy_m", [0.45, 0.0]))
    rng = layout["objects"]["object_0"]["reset_scatter_range_base_m"]
    assert rng["x"][0] >= cx - bx / 2.0 and rng["x"][1] <= cx + bx / 2.0, "산란 x가 벨트 밖"
    assert rng["y"][0] >= cy - by / 2.0 and rng["y"][1] <= cy + by / 2.0, "산란 y가 벨트 밖"


def test_target_zone_inside_belt_and_on_surface(layout: dict, sim_scene: dict) -> None:
    """★place 타겟 ⊂ 벨트 + 안착 높이 = 큐브 반높이★ — 관측 계약(target_pos)과 물리의 정합."""
    bx, by, _ = (float(v) for v in sim_scene["belt_size_m"])
    cx, cy = (float(v) for v in sim_scene.get("belt_center_xy_m", [0.45, 0.0]))
    obj = float(sim_scene["object_size_m"])
    table_h = float(sim_scene["table_height_m"])
    rng = layout["objects"]["target_zone"]["range_m"]
    assert rng["x"][0] >= cx - bx / 2.0 and rng["x"][1] <= cx + bx / 2.0, "타겟 x가 벨트 밖"
    assert rng["y"][0] >= cy - by / 2.0 and rng["y"][1] <= cy + by / 2.0, "타겟 y가 벨트 밖"
    # 안착 z = 벨트 상면 + 큐브 반높이 (큐브 크기를 바꾸면 함께 움직여야 함)
    assert rng["z"][0] == pytest.approx(table_h + obj / 2.0, abs=1e-6), (
        "타겟 z ≠ 벨트 상면 + 큐브 반높이 — common.yaml target.sample_range.z를 확인"
    )


def test_camera_and_tcp_match_calib_yaml(layout: dict) -> None:
    """손목 카메라·TCP 레지스트리 값 = calib yaml 원본(요소4 단일 소스 드리프트 가드)."""
    T_fc = load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml")
    assert _pos(layout, "wrist_camera") == pytest.approx([float(v) for v in T_fc[:3, 3]], abs=1e-6)
    T_ft = load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml")
    assert _pos(layout, "tcp_frame") == pytest.approx([float(v) for v in T_ft[:3, 3]], abs=1e-6)
    # 카메라와 D435i 본체는 같은 위치(회전만 메시축 보정 차이)
    assert _pos(layout, "d435i_body_visual") == pytest.approx(_pos(layout, "wrist_camera"), abs=1e-9)


def test_belt_yaw_is_axis_preserving(sim_scene: dict) -> None:
    """벨트 yaw는 0 또는 ±180만 지원 — 스폰/타겟 샘플링이 축정렬이기 때문.

    ★180°가 허용되는 이유★: z축 180° 회전은 (x,y)→(−x,−y)라 축정렬 박스를 자기 자신으로
    사상하고 모든 거리를 보존한다(로봇이 벨트를 등지는 실셀 배치, 2026-07-28 2차).
    그 외 각도는 큐브가 벨트 밖에 스폰되거나 타겟이 벨트를 벗어나는 조용한 불일치를 만든다.
    (scene.py가 기동을 거부하지만, 학습을 돌리기 전에 여기서 먼저 잡는다.)
    """
    yaw = float(sim_scene.get("belt_yaw_deg", 0.0))
    assert min(abs(yaw), abs(abs(yaw) - 180.0)) < 1e-6, (
        f"sim.yaml scene.belt_yaw_deg={yaw} — 허용값은 0 / ±180 뿐이다. 그 외 회전 배치를 쓰려면 "
        "events_dr 스폰 분포와 common.yaml target.sample_range도 함께 회전시켜야 한다"
    )


def test_pregrasp_point_above_belt(layout: dict, sim_scene: dict) -> None:
    """pre-grasp 보조 파지점은 벨트 위 공중(홈 포즈 pad 중점) — 홈 포즈 변경 시 재실측 대상."""
    p = _pos(layout, "pregrasp_assist_point")
    table_h = float(sim_scene["table_height_m"])
    assert p[2] > table_h + 0.05, "파지점이 벨트 표면에 붙어 있음 — measure_grasp_frame.py 재실측 필요"


# ═════════════════════════════════════════════════════════════════════════════
# ★2026-07-28(2차) 배치 180° 계약★ — 로봇이 벨트를 등지고 서는 실셀 배치.
#   벨트가 base frame −x로 갔고 홈 shoulder_pan이 0→π가 됐다. 이 변경은 6개 파일에
#   흩어져 있어(sim.yaml·common.yaml·safety.yaml·robot_spec·real.example.yaml·events_dr)
#   ★일부만 되돌려지면 조용히 어긋난다★ — 아래 테스트가 그 상태를 즉시 깬다.
# ═════════════════════════════════════════════════════════════════════════════

def test_belt_and_target_on_same_side(sim_scene: dict) -> None:
    """★핵심★ 벨트 중심과 타겟 샘플 범위가 base frame에서 ★같은 쪽★에 있어야 한다.

    한쪽만 부호를 뒤집으면 타겟이 로봇 반대편 허공에 생겨 정책이 영원히 도달하지 못한다.
    (부호 규약을 하드코딩하지 않고 '일치'만 강제 — 배치가 또 바뀌어도 이 테스트는 유효하다.)
    """
    from common.config_io import configs_dir, load_yaml

    cx = float(sim_scene["belt_center_xy_m"][0])
    rng = load_yaml(configs_dir() / "common.yaml")["target"]["sample_range"]["x"]
    lo, hi = float(rng[0]), float(rng[1])
    assert lo < hi, f"target.sample_range.x가 [lo,hi] 순서가 아니다: {rng}"
    assert lo <= cx <= hi, (
        f"벨트 중심 x={cx}가 타겟 범위 x[{lo},{hi}] 밖 — 배치 변경이 한쪽에만 반영됐다"
    )


def test_home_pose_faces_the_belt(sim_scene: dict) -> None:
    """홈 파지중심이 벨트와 같은 쪽에 있어야 한다 — 배치와 홈이 따로 놀면 안 된다.

    ★2026-08-30 정정★ 이전 구현은 `shoulder_pan≈±π`를 '−x 지향'의 대리 지표로 썼다.
    그 대리는 팔이 ★펴져 있을 때만★ 성립한다 — 새 홈(0/−76/−103/−96/+93/+4)처럼
    J2·J3·J4가 접히면 pan=0이어도 공구는 −x 쪽으로 넘어간다(실측 TCP x=−0.359).
    그래서 각도가 아니라 ★실제 파지중심 위치★를 FK로 구해 검사한다 — docstring이
    원래 말하던 그 성질을 직접 본다.
    """
    import importlib.util
    import sys

    import numpy as np
    import yaml

    from common.robot_spec import UR5E_HOME_JOINT_POS

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "urdf_fk_layout", root / "scripts" / "tools" / "urdf_fk.py"
    )
    urdf_fk = importlib.util.module_from_spec(spec)
    sys.modules["urdf_fk_layout"] = urdf_fk
    spec.loader.exec_module(urdf_fk)

    joints = urdf_fk.load_joints(root / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf")
    t_off = np.array(
        yaml.safe_load(
            (root / "configs" / "calib" / "tcp_offset.yaml").read_text(encoding="utf-8")
        )["T_flange_tcp"],
        dtype=float,
    )
    grip = (urdf_fk.fk(joints, dict(UR5E_HOME_JOINT_POS))["wrist_3_link"] @ t_off)[:3, 3]

    cx = float(sim_scene["belt_center_xy_m"][0])
    assert (float(grip[0]) < 0.0) == (cx < 0.0), (
        f"홈 파지중심 x={grip[0]:.4f}와 벨트 중심 x={cx}가 서로 반대쪽이다 "
        "— sim.yaml 배치와 robot_spec 홈 중 한쪽만 바뀌었다"
    )


def test_spawn_range_inside_safety_workspace(sim_scene: dict) -> None:
    """★신규 커플링★ 큐브 스폰 범위가 safety workspace AABB 안이어야 한다.

    벨트를 넓히면 스폰 분포도 함께 넓어진다(events_dr가 BELT_SIZE에서 파생). 스폰이 AABB
    밖으로 나가면 ★로봇이 그 큐브를 집으러 가는 순간 SafetyGuard가 trip★한다 — 학습은
    되는데 배포만 죽는 종류의 사고다. 벨트 65cm→99.2cm 확대(2026-07-28 2차) 때 실제로
    여유가 19mm까지 줄었다.
    """
    from common.config_io import build_safety_cfg, configs_dir, load_yaml

    # ★픽존 기준★ (2026-07-28 3차): 스폰은 물리 벨트가 아니라 pick_zone_m에서 파생한다.
    bx, by = (float(v) for v in sim_scene.get("pick_zone_m", sim_scene["belt_size_m"][:2]))
    cx, cy = (float(v) for v in sim_scene["belt_center_xy_m"])
    margin = float(sim_scene["object_size_m"]) / 2.0 + 0.05  # events_dr._SCATTER_MARGIN
    half_x = max(bx / 2.0 - margin, 0.01)
    half_y = max(by / 2.0 - margin, 0.01)

    safety = build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))
    lo, hi = safety.workspace_min, safety.workspace_max
    # ★경계 여유 ≥10mm★ — 홈 포즈에 이미 있는 규칙(test_home_pose_contract)을 스폰에도 적용한다.
    #   경계에 딱 붙으면 센서 노이즈·제어 오버슈트만으로 SafetyGuard가 trip한다.
    _MARGIN = 0.010
    for sx in (cx - half_x, cx + half_x):
        for sy in (cy - half_y, cy + half_y):
            assert lo[0] + _MARGIN <= sx <= hi[0] - _MARGIN, (
                f"스폰 x={sx:.4f}가 workspace x[{lo[0]},{hi[0]}] 밖 — 벨트 x 길이를 줄이거나 "
                "safety AABB를 실측 근거로 재산정할 것"
            )
            assert lo[1] + _MARGIN <= sy <= hi[1] - _MARGIN, (
                f"스폰 y={sy:.4f}가 workspace y[{lo[1]},{hi[1]}] 밖 — 벨트 y 너비를 줄이거나 "
                "safety AABB를 실측 근거로 재산정할 것"
            )


def test_spawn_and_targets_within_reach(sim_scene: dict) -> None:
    """최원 스폰·타겟이 UR5e 물리 도달 범위(상완 0.425 + 전완 0.3922) 안."""
    import math

    from common.config_io import configs_dir, load_yaml

    reach = 0.425 + 0.3922
    bx, by = (float(v) for v in sim_scene.get("pick_zone_m", sim_scene["belt_size_m"][:2]))
    cx, cy = (float(v) for v in sim_scene["belt_center_xy_m"])
    margin = float(sim_scene["object_size_m"]) / 2.0 + 0.05
    far_spawn = max(
        math.hypot(cx + sx * max(bx / 2.0 - margin, 0.01), cy + sy * max(by / 2.0 - margin, 0.01))
        for sx in (-1, 1) for sy in (-1, 1)
    )
    assert far_spawn < reach, f"최원 스폰 반경 {far_spawn:.3f}m ≥ 도달 한계 {reach:.3f}m"

    tgt = load_yaml(configs_dir() / "common.yaml")["target"]["sample_range"]
    far_tgt = max(math.hypot(float(x), float(y)) for x in tgt["x"] for y in tgt["y"])
    assert far_tgt < reach, f"최원 타겟 반경 {far_tgt:.3f}m ≥ 도달 한계 {reach:.3f}m"


def test_pick_zone_inside_physical_belt(sim_scene: dict) -> None:
    """★픽존 ⊂ 물리 벨트★ — 픽존이 벨트보다 크면 큐브가 허공/바닥에 스폰된다.

    (scene.py도 기동 시 같은 조건을 검사하지만, 학습을 띄우기 전에 여기서 먼저 잡는다.)
    """
    belt = [float(v) for v in sim_scene["belt_size_m"][:2]]
    pick = [float(v) for v in sim_scene.get("pick_zone_m", belt)]
    assert pick[0] <= belt[0] + 1e-9 and pick[1] <= belt[1] + 1e-9, (
        f"pick_zone_m {pick} > belt_size_m {belt} — 픽존은 벨트 안에 있어야 한다"
    )


def test_target_range_inside_pick_zone(sim_scene: dict) -> None:
    """★타겟 범위 ⊂ 픽존★ — 픽존 밖 타겟은 도달·안전 검증을 받지 않은 구역이다."""
    from common.config_io import configs_dir, load_yaml

    pick = [float(v) for v in sim_scene.get("pick_zone_m", sim_scene["belt_size_m"][:2])]
    cx, cy = (float(v) for v in sim_scene["belt_center_xy_m"])
    tgt = load_yaml(configs_dir() / "common.yaml")["target"]["sample_range"]
    assert float(tgt["x"][0]) >= cx - pick[0] / 2.0 - 1e-9, "타겟 x 하한이 픽존 밖"
    assert float(tgt["x"][1]) <= cx + pick[0] / 2.0 + 1e-9, "타겟 x 상한이 픽존 밖"
    assert float(tgt["y"][0]) >= cy - pick[1] / 2.0 - 1e-9, "타겟 y 하한이 픽존 밖"
    assert float(tgt["y"][1]) <= cy + pick[1] / 2.0 + 1e-9, "타겟 y 상한이 픽존 밖"
