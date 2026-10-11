"""파지 접촉력 계약 — 자산/큐브를 바꿀 때 강성 재보정을 강제한다 (sim 불필요, README 검증 계약).

배경(2026-07-29, F31): 그리퍼 실측 반영으로 폐합 간격이 6.6 → 42mm가 되자, 같은 강성에서
      ★파지력이 1/3로 떨어졌다★(편측 침투 11.7 → 4.0mm). 증상은 "파지를 학습하지 못한다"로
      나타나서 마찰·폐합속도·pad 크기·큐브 크기·커리큘럼을 반나절 뒤졌고, 정작 원인인
      ★접촉력★은 학습 환경에서 한 번도 재보지 않았다(물리 테스트에서만 쟀는데 그때는
      진단 도구가 학습과 다른 콜리전을 썼다).

계약: 접촉력 ≈ 강성 × 편측침투 가 ★검증된 기준값(23.4N)★ 근방이어야 한다.
      23.4N = 구 계보(3cm 큐브, 편측 11.7mm, 강성 2000)에서 teacher가 placed 96.6%를 달성한 조건.
      큐브 크기나 그리퍼 기하를 바꾸면 침투가 변하므로 ★강성을 함께 옮겨야★ 한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from common.config_io import configs_dir, load_yaml

_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = (_ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
           / "mdp" / "ur5e_assets.py")
_TARGET_N = 23.4          # 구 계보 검증 접촉력 [N]
_TOL = 0.35               # ±35% (스모크 봉우리가 4000~8000으로 완만해 이 정도가 실용 범위)


def _src() -> str:
    if not _ASSETS.is_file():
        pytest.skip("sim 확장(source/) 없음 — 배포 번들에서는 대조 불가")
    return _ASSETS.read_text(encoding="utf-8")


def _const(name: str) -> float:
    m = re.search(rf"^{name}:\s*float\s*=\s*([0-9.]+)", _src(), re.M)
    assert m, f"{name} 상수를 찾지 못함"
    return float(m.group(1))


def _gripper_stiffness() -> float:
    m = re.search(r'"gripper": ImplicitActuatorCfg\(.*?stiffness=([0-9.]+)', _src(), re.S)
    assert m, "gripper 액추에이터 stiffness를 찾지 못함"
    return float(m.group(1))


def test_grip_force_matches_validated_baseline() -> None:
    """★핵심★ 강성 × 편측침투가 검증 기준(23.4N) 근방이어야 한다.

    실패하면 강성을 `23.4 / 편측침투[m]` 로 재설정할 것. 이 테스트를 완화하지 말 것 —
    파지력이 부족하면 학습이 '이유 없이' 안 되고, 원인 추적에 반나절이 걸린다(F31).
    """
    open_gap = _const("GRIPPER_MAX_OPEN_GAP_M")
    close_q = _const("GRIPPER_CLOSE_M")
    cube = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["object_size_m"])
    stiffness = _gripper_stiffness()

    closed_gap = open_gap - 2.0 * close_q          # gap(q) = OPEN − 2q
    pen_m = (cube - closed_gap) / 2.0              # 편측 침투 [m]
    assert pen_m > 0, f"완전 폐합 간격 {closed_gap*1000:.1f}mm ≥ 큐브 {cube*1000:.0f}mm — 물지 못한다"

    force = stiffness * pen_m
    lo, hi = _TARGET_N * (1 - _TOL), _TARGET_N * (1 + _TOL)
    assert lo <= force <= hi, (
        f"파지 접촉력 {force:.1f}N (강성 {stiffness:.0f} × 침투 {pen_m*1000:.1f}mm)이 "
        f"검증 범위 [{lo:.1f}, {hi:.1f}]N 밖 — 권장 강성 = {_TARGET_N/pen_m:.0f}"
    )


def test_force_stays_within_hardware_limit() -> None:
    """sim 접촉력이 실물 2FG7 파지력 상한(140N)을 넘지 않아야 한다 — 넘으면 sim2real 역방향 갭."""
    open_gap = _const("GRIPPER_MAX_OPEN_GAP_M")
    close_q = _const("GRIPPER_CLOSE_M")
    cube = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["object_size_m"])
    force = _gripper_stiffness() * (cube - (open_gap - 2.0 * close_q)) / 2.0
    assert force <= 140.0, f"sim 접촉력 {force:.0f}N > 실물 2FG7 상한 140N"
