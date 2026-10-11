"""pregrasp clamp 계약 — 그리퍼 기하 하드코딩 재발 방지 (sim 불필요, README 검증 계약).

배경(2026-07-29 실사고, F30): events_dr._pregrasp_reset의 clamp_q 식에 ★열림 폭 0.044가
      하드코딩★돼 있었다. 핑거팁 실측 반영으로 열림이 80mm가 되자 clamp_q가 0.0005(간격 79mm)로
      계산돼 ★pregrasp가 사실상 활짝 열린 채 시작★ → 큐브 즉시 낙하 → grasp 보상 영구 0.
      그런데 증상은 '파지가 안 된다'로 나타나서 마찰·강성·폐합속도·큐브크기를 한참 뒤졌다
      (수 시간 소모). 기하 상수는 반드시 단일 소스에서 파생시켜야 한다.

여기서 고정하는 계약(전부 순수 계산):
  ① clamp 식에 그리퍼 기하 숫자가 하드코딩돼 있지 않을 것
  ② 현재 큐브 크기에서 clamp_q가 '적당한 침투'(1~6mm/측)를 만들 것 — 너무 얕으면 즉시 낙하,
     너무 깊으면 depenetration으로 튕겨 나간다
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from common.config_io import configs_dir, load_yaml

_ROOT = Path(__file__).resolve().parent.parent
_EVENTS = (_ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
           / "mdp" / "events_dr.py")
_ASSETS = _EVENTS.parent / "ur5e_assets.py"


def _src(p: Path) -> str:
    if not p.is_file():
        pytest.skip("sim 확장(source/) 없음 — 배포 번들에서는 대조 불가")
    return p.read_text(encoding="utf-8")


def test_clamp_formula_has_no_hardcoded_geometry() -> None:
    """★핵심★ clamp_q 식이 그리퍼 열림 폭을 상수에서 가져와야 한다(하드코딩 금지)."""
    src = _src(_EVENTS)
    m = re.search(r"clamp_q\s*=\s*(.+)", src)
    assert m, "clamp_q 식을 찾지 못함"
    expr = m.group(1)
    assert "_GRIPPER_OPEN_GAP" in expr, (
        f"clamp_q 식이 열림 폭 상수를 쓰지 않는다: {expr}\n"
        "→ ur5e_assets.GRIPPER_MAX_OPEN_GAP_M에서 파생시킬 것(F30 재발 방지)"
    )
    # 0.0xx 형태의 기하 상수가 식에 직접 박혀 있으면 실패 (_pen 변수는 식 밖에 정의됨)
    bad = [t for t in re.findall(r"0\.0\d+", expr)]
    assert not bad, f"clamp_q 식에 기하 상수가 하드코딩돼 있다: {bad}"


def test_clamp_gives_reasonable_penetration() -> None:
    """현재 큐브 크기에서 clamp_q가 편측 침투 1~6mm를 만들어야 한다.

    얕으면(<1mm) t=0에 그립력이 없어 큐브가 즉시 떨어지고, 깊으면(>6mm) PhysX
    depenetration이 큐브를 튕겨낸다. 둘 다 pregrasp 보조가 무력화된다.
    """
    a = _src(_ASSETS)
    open_gap = float(re.search(r"GRIPPER_MAX_OPEN_GAP_M:\s*float\s*=\s*([0-9.]+)", a).group(1))
    close_q = float(re.search(r"GRIPPER_CLOSE_M:\s*float\s*=\s*([0-9.]+)", a).group(1))
    pen = float(re.search(r"_pen\s*=\s*([0-9.]+)", _src(_EVENTS)).group(1))
    cube = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["object_size_m"])

    clamp_q = max(0.0, min(close_q, (open_gap - cube + pen) / 2.0))
    gap = open_gap - 2.0 * clamp_q          # gap(q) = OPEN − 2q
    per_side_mm = (cube - gap) / 2.0 * 1000

    assert 1.0 <= per_side_mm <= 6.0, (
        f"pregrasp 편측 침투 {per_side_mm:.1f}mm (큐브 {cube*1000:.0f}mm, clamp_q {clamp_q:.4f}, "
        f"간격 {gap*1000:.1f}mm) — 1~6mm 밖이면 pregrasp 보조가 작동하지 않는다"
    )


def test_cube_is_grippable_at_full_close() -> None:
    """완전 폐합 시 큐브를 실제로 물어야 한다 — 침투가 양수여야 grip이 성립한다."""
    a = _src(_ASSETS)
    open_gap = float(re.search(r"GRIPPER_MAX_OPEN_GAP_M:\s*float\s*=\s*([0-9.]+)", a).group(1))
    close_q = float(re.search(r"GRIPPER_CLOSE_M:\s*float\s*=\s*([0-9.]+)", a).group(1))
    cube = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["object_size_m"])
    closed_gap = open_gap - 2.0 * close_q
    per_side_mm = (cube - closed_gap) / 2.0 * 1000
    assert per_side_mm > 0, (
        f"완전 폐합 간격 {closed_gap*1000:.1f}mm ≥ 큐브 {cube*1000:.0f}mm — 닫아도 물지 못한다"
    )
