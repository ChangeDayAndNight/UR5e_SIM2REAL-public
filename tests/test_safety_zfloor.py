"""★감사 B1(2026-08-12) 회귀 고정★ — 실기 z-floor clamp가 sim 액션 클램프와 동치.

[사고] sim 액션 래퍼(delayed_actions.py)의 tcp_z_floor 클램프가 ★sim 전용★이었다.
  place 타겟 z(−0.0125)가 패드-벨트 접촉점(−0.0101)보다 아래라 정책은 '클램프가
  잡아 준다'를 전제로 하강을 명령하도록 학습되는데, 실기 경로에는 그 클램프가 없어
  매 정상 place 시도가 벨트 압착 경로를 밟았다. AABB 하한(−0.0375)은 벨트 상면이라
  위치 기반 안전층이 이 시나리오에서 구조적으로 무력했고, 리허설은 sim 액션텀을
  경유해 클램프가 있으므로 ★리허설로는 재현되지 않는★ 갭이었다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from common.config_io import build_safety_cfg, configs_dir, load_yaml
from common.safety import SafetyGuard

_ROOT = Path(__file__).resolve().parent.parent


def _guard() -> SafetyGuard:
    return SafetyGuard(build_safety_cfg(load_yaml(configs_dir() / "safety.yaml")))


def test_zfloor_clamps_descent_below_floor():
    """바닥 아래로 내려가는 dz는 (floor − z_cur)로 잘린다 — sim 의미론과 동일."""
    g = _guard()
    floor = g.cfg.tcp_z_floor_m
    assert floor is not None, "safety.yaml workspace.tcp_z_floor_m 누락 — B1 회귀"
    z_cur = floor + 0.010                      # 바닥 10mm 위
    delta = np.array([0.0, 0.0, -0.020, 0.0, 0.0, 0.0])   # 20mm 하강 명령
    out = g.clamp_task_delta(delta, dt=1.0, current_tcp_pose=[0.0, 0.0, z_cur])
    assert out[2] == np.float64(floor - z_cur)             # 정확히 바닥에서 멈춤
    assert z_cur + out[2] >= floor - 1e-12


def test_zfloor_pushes_up_when_already_below():
    """이미 바닥 아래면 상승을 강제한다 — sim `max(dz, floor−z_cur)`와 동일(복구 의미론)."""
    g = _guard()
    floor = g.cfg.tcp_z_floor_m
    z_cur = floor - 0.005                      # 바닥 5mm 아래(비정상 진입)
    delta = np.zeros(6)
    out = g.clamp_task_delta(delta, dt=1.0, current_tcp_pose=[0.0, 0.0, z_cur])
    assert out[2] > 0.0 and z_cur + out[2] >= floor - 1e-12


def test_zfloor_no_effect_above_floor_or_without_pose():
    """바닥 위 정상 이동은 무변화 / 현재 포즈 미전달이면 기존 동작(비활성)."""
    g = _guard()
    floor = g.cfg.tcp_z_floor_m
    small = np.array([0.0, 0.0, -0.001, 0.0, 0.0, 0.0])
    out = g.clamp_task_delta(small.copy(), dt=1.0, current_tcp_pose=[0.0, 0.0, floor + 0.100])
    np.testing.assert_allclose(out, small)
    out2 = g.clamp_task_delta(small.copy(), dt=1.0)        # 포즈 없음 — 구 시그니처 호환
    np.testing.assert_allclose(out2, small)


def test_zfloor_value_matches_sim_action_clamp():
    """★드리프트 가드★ — safety.yaml 값 = sim의 TABLE_HEIGHT + 여유(env cfg 소스에서 추출).

    sim 여유값(0.030)을 누가 바꾸면 이 테스트가 깨져 양쪽 동기를 강제한다
    (M11 유형의 이중 소스 드리프트를 테스트로 차단 — §9-5-2).
    """
    safety_floor = float(load_yaml(configs_dir() / "safety.yaml")["workspace"]["tcp_z_floor_m"])
    table_h = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["table_height_m"])
    src = (_ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/binpick_env_cfg.py").read_text(
        encoding="utf-8"
    )
    m = re.search(r"tcp_z_floor_m\s*=\s*TABLE_HEIGHT\s*\+\s*([0-9.]+)", src)
    assert m, "env cfg에서 tcp_z_floor 여유값을 찾지 못함 — 식 형태가 바뀌면 이 가드를 갱신할 것"
    margin = float(m.group(1))
    assert safety_floor == np.float64(table_h + margin), (
        f"z-floor 불일치: safety.yaml {safety_floor} != sim {table_h}+{margin} — "
        "sim 클램프에 기대 학습된 정책이 실기에서 다른 바닥을 만난다(B1 재발)"
    )
