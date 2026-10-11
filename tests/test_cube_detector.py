"""큐브 인식기 + 배포 게이트 — 하드웨어 없이 계약을 검증한다."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from deployment.cube_detector import MIN_AREA_PX, detect_cube

ROOT = Path(__file__).resolve().parents[1]
_H, _W = 480, 848
_INTR = (604.4, 604.4, 424.0, 240.0)     # D435i color 848x480 (fx=912.271@1280 스케일)


def _scene(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """합성 장면 — 회색 벨트 위에 물체 하나."""
    rgb = np.full((_H, _W, 3), 120, np.uint8)
    depth = np.full((_H, _W), 0.60, np.float32)
    if kind == "red_cube":
        rgb[200:268, 400:468] = (204, 38, 38); depth[200:268, 400:468] = 0.45
    elif kind == "blue_cube":
        rgb[210:278, 380:448] = (38, 89, 217); depth[210:278, 380:448] = 0.45
    elif kind == "green_object":
        rgb[200:268, 400:468] = (40, 180, 60)
    elif kind == "tiny_red":
        rgb[230:245, 420:435] = (204, 38, 38)
    elif kind == "long_red_bar":
        rgb[220:250, 300:560] = (204, 38, 38)
    return rgb, depth


@pytest.mark.parametrize("kind", ["red_cube", "blue_cube"])
def test_trained_cube_colors_are_detected(kind):
    rgb, depth = _scene(kind)
    result = detect_cube(rgb, depth, intrinsics=_INTR)
    assert result.present, result.reason
    assert result.area_px >= MIN_AREA_PX
    assert result.color in ("red", "blue")


@pytest.mark.parametrize(
    ("kind", "expect_in_reason"),
    [
        ("green_object", "색 마스크"),      # 학습에 없던 색 → 애초에 안 잡힘
        ("tiny_red", "너무 작다"),          # 너무 멀거나 작은 것
        ("long_red_bar", "길쭉하다"),       # 같은 색이어도 큐브 형상이 아님
        ("empty", "색 마스크"),             # 빈 장면
    ],
)
def test_non_cube_scenes_are_rejected_with_a_reason(kind, expect_in_reason):
    """★3번 문제의 방어★ — 큐브가 아니면 거부하고 ★왜인지★ 남긴다(현장 디버깅용)."""
    rgb, depth = _scene(kind)
    result = detect_cube(rgb, depth, intrinsics=_INTR)
    assert not result.present
    assert expect_in_reason in result.reason, result.reason


def test_position_estimate_matches_the_synthetic_depth():
    """역투영이 맞는지 — 합성 depth 0.45m가 그대로 z로 나와야 한다."""
    rgb, depth = _scene("red_cube")
    result = detect_cube(rgb, depth, intrinsics=_INTR)
    assert result.position_cam is not None
    assert result.position_cam[2] == pytest.approx(0.45, abs=1e-3)
    # 큐브가 화면 중앙 근처이므로 x·y는 작아야 한다.
    assert abs(result.position_cam[0]) < 0.05 and abs(result.position_cam[1]) < 0.05


def test_base_frame_transform_is_applied_when_given():
    rgb, depth = _scene("red_cube")
    T = np.eye(4); T[:3, 3] = [1.0, 2.0, 3.0]      # 순수 평행이동
    result = detect_cube(rgb, depth, intrinsics=_INTR, T_base_cam=T)
    assert result.position_base is not None
    assert result.position_base[2] == pytest.approx(3.0 + 0.45, abs=1e-3)


def test_gate_blocks_policy_when_no_cube_and_passes_when_present():
    """미검출은 명시적 hold 요청, 검출되면 원래 정책으로 복귀한다."""
    spec = importlib.util.spec_from_file_location("run_task", ROOT / "Phase4" / "Run_Task.py")
    run_task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_task)

    class _Frame:
        def __init__(self, rgb): self.rgb = rgb

    class _Camera:
        def __init__(self, rgb): self._rgb = rgb
        def get_frame(self): return _Frame(self._rgb)

    class _Policy:
        def act(self, obs): return np.full((1, 7), 0.9)

    gate = run_task._CubeGate(None)
    policy = run_task._GatedPolicy(_Policy(), gate, 7)

    empty_cam = run_task._GatedCamera(_Camera(_scene("empty")[0]), gate)
    empty_cam.get_frame()
    assert not gate.present
    assert np.all(policy.act(None) == 0.0), "큐브가 없는데 정책 출력이 나갔다"
    assert policy.hold_requested, "zero action만으로는 그리퍼 유지가 보장되지 않는다"

    cube_cam = run_task._GatedCamera(_Camera(_scene("red_cube")[0]), gate)
    cube_cam.get_frame()
    assert gate.present
    assert np.all(policy.act(None) == 0.9), "큐브가 있는데 게이트가 막았다"
    assert not policy.hold_requested
    assert gate.blocked_steps == 1 and gate.total_steps == 2


def test_gate_holds_when_rgb_is_absent():
    """센서 부재로 게이트를 묵시 우회하지 않는다. depth-only는 명시적 설정 대상."""
    spec = importlib.util.spec_from_file_location("run_task2", ROOT / "Phase4" / "Run_Task.py")
    run_task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_task)

    class _DepthOnlyFrame:
        rgb = None

    gate = run_task._CubeGate(None)
    gate.judge(_DepthOnlyFrame())
    assert not gate.present and "RGB 없음" in gate.last_reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
