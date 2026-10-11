"""★Student v2(2026-09-18, F89)★ 카메라 종횡비·depth_near 계약 회귀 고정 (Isaac 불필요)."""
from __future__ import annotations

import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_sim_render_grid_follows_real_color_fov_aspect():
    """Kit 렌더는 등방 픽셀이라 세로 FOV = 가로 FOV × height/width. 격자가 실기 69.4°×42.5°를 따라야 한다."""
    cam = _yaml("configs/sim.yaml")["camera"]
    pre = _yaml("configs/common.yaml")["preprocess"]
    assert pre["frame_mode"] == "full"
    expected_h = round(int(cam["width"]) * math.tan(math.radians(cam["vertical_fov_deg"]) / 2.0)
                       / math.tan(math.radians(cam["horizontal_fov_deg"]) / 2.0))
    assert int(cam["width"]) == int(pre["out_width"]) == 112
    assert int(cam["height"]) == expected_h == 63, (
        f"sim.yaml camera height {cam['height']} ≠ FOV 종횡비 기대값 {expected_h} — 112² 등방 렌더는 "
        "세로 69.4°가 되어 실기(42.5°)와 물체 크기가 1.78배 어긋난다(배포 v2 export 실측)"
    )
    rendered_vfov = math.degrees(2.0 * math.atan(63 / 112 * math.tan(math.radians(69.4) / 2.0)))
    assert abs(rendered_vfov - 42.5) < 0.2


def test_depth_near_matches_real_stream_min_z():
    """848×480 D435i Min-Z ≈ 0.17 m → 계약 0.20 m (구 0.28은 파지 단계 depth를 70% 결측시켰다)."""
    pre = _yaml("configs/common.yaml")["preprocess"]
    assert float(pre["depth_near_m"]) == 0.20
    src = (ROOT / "scripts/preflight_training.py").read_text(encoding="utf-8")
    assert "preprocess_cfg.depth_near_m == 0.20" in src, "preflight가 구 0.28 계약을 요구한다"
    assert "== 63" in src, "preflight가 렌더 격자 112×63 계약을 요구하지 않는다"


def test_scene_guard_uses_fov_aspect_not_square_equality():
    src = (ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/mdp/scene.py").read_text(encoding="utf-8")
    assert "_expected_h" in src and "tan(math.radians(effective_vfov) / 2.0)" in src
    assert "(CAM_WIDTH, CAM_HEIGHT) != (_pre.out_width, _pre.out_height)" not in src
