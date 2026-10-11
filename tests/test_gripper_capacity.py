"""그리퍼 파지 가능 폭 계약 — 못 집는 물체 크기를 학습 전에 차단 (sim 불필요, README 검증 계약).

배경(2026-07-29): 사용자가 "큐브는 5cm"라고 알려줬을 때, sim.yaml object_size_m만 0.05로
      바꿨다면 ★파지 성공률이 구조적으로 0%인 태스크★가 만들어질 뻔했다 —
      OnRobot 2FG7 핑거팁을 바깥 실장 위치로 보정해 inner face 최대 간격을 74mm로
      확보했다. 이 값이 다시 좁아져 50mm 큐브가 들어가지 않는 회귀를 계약으로 막는다.

⚠️ 실물 2FG7 데이터시트의 외측 파지 범위와 현행 URDF의 34~74mm는 핑거팁 바깥 위치
   모델이다. 더 큰 물체를 쓰려면
   핑거팁 실장 위치 확인 → URDF 오프셋 수정 → tcp_offset 재측정 → 홈 FK·_PREGRASP_GRIP
   재산출까지 한 묶음으로 가야 한다(값만 바꾸면 sim과 실물이 조용히 갈라진다).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from common.config_io import configs_dir, load_yaml

_ROOT = Path(__file__).resolve().parent.parent
_ASSETS_PY = (_ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
              / "mdp" / "ur5e_assets.py")
# 접근·정렬 오차를 흡수할 최소 여유(양측 합계) — 딱 맞으면 실기에서 매번 부딪힌다.
_MIN_CLEARANCE_M = 0.004


def _const(name: str) -> float:
    """isaaclab을 import하지 않고 소스에서 상수를 파싱한다(§6① 순수 테스트 유지)."""
    if not _ASSETS_PY.is_file():
        pytest.skip("sim 확장(source/) 없음 — 배포 번들에서는 대조 불가")
    src = _ASSETS_PY.read_text(encoding="utf-8")
    m = re.search(rf"^{name}:\s*float\s*=\s*([0-9.]+)", src, re.M)
    assert m, f"{name} 상수를 찾지 못함 — ur5e_assets.py 확인"
    return float(m.group(1))


def test_object_fits_in_open_gripper() -> None:
    """★핵심★ 큐브가 완전히 벌린 그리퍼 사이에 여유를 두고 들어가야 한다.

    실패하면 sim.yaml object_size_m을 줄이거나, 핑거팁을 바깥 위치로 옮기고
    URDF/STL·tcp_offset·홈 FK를 함께 갱신할 것. 이 테스트를 완화해서 통과시키지 말 것 —
    통과시켜 봐야 학습이 성공률 0%로 끝난다.
    """
    obj = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["object_size_m"])
    gap = _const("GRIPPER_MAX_OPEN_GAP_M")
    assert obj + _MIN_CLEARANCE_M <= gap, (
        f"큐브 {obj * 1000:.0f}mm + 여유 {_MIN_CLEARANCE_M * 1000:.0f}mm > "
        f"그리퍼 최대 열림 {gap * 1000:.0f}mm — ★이 크기는 원천적으로 파지 불가★"
    )


def test_close_target_actually_grips_the_object() -> None:
    """완전 폐합 시 큐브를 실제로 물어야 한다 — 닫힘 간격 < 큐브 크기.

    닫힘 간격이 큐브보다 크면 '닫아도 헛돈다'(접촉력 0). 2026-07-10에 close=0.011로
    얕게 잡았다가 접촉력 부족으로 성공률 0%였던 이력이 있다(ur5e_assets.py 주석).
    """
    obj = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["object_size_m"])
    open_gap = _const("GRIPPER_MAX_OPEN_GAP_M")
    q_close = _const("GRIPPER_CLOSE_M")
    # ★간격 모델: gap(q) = OPEN − 2q★ (손가락 2개가 각각 q만큼 안쪽으로 이동).
    #   구버전은 폐합 간격 0.006(구 자산값)을 하드코딩해 기울기를 잘못 잡았다 — F30과 같은 부류.
    closed_gap = open_gap - 2.0 * q_close
    assert closed_gap < obj, (
        f"완전 폐합 간격 {closed_gap * 1000:.1f}mm ≥ 큐브 {obj * 1000:.0f}mm — 닫아도 물지 못한다"
    )


def test_open_target_releases_the_object() -> None:
    """열림 목표에서는 큐브를 놓아야 한다 — release가 안 되면 place가 영원히 실패한다."""
    obj = float(load_yaml(configs_dir() / "sim.yaml")["scene"]["object_size_m"])
    open_gap = _const("GRIPPER_MAX_OPEN_GAP_M")
    q_open = _const("GRIPPER_OPEN_M")
    gap_at_open = open_gap - 2.0 * q_open
    assert gap_at_open > obj, (
        f"열림 간격 {gap_at_open * 1000:.1f}mm ≤ 큐브 {obj * 1000:.0f}mm — 놓을 수 없다"
    )


def test_joint_range_and_verified_gap_contract() -> None:
    """현행 URDF q/geometry와 관측 정규화 분모를 0.020·74→34mm로 고정한다."""
    from common.robot_spec import GRIPPER_JOINT_RANGE_M

    open_gap = _const("GRIPPER_MAX_OPEN_GAP_M")
    q_open = _const("GRIPPER_OPEN_M")
    q_close = _const("GRIPPER_CLOSE_M")
    assert q_open == 0.0
    assert q_close == GRIPPER_JOINT_RANGE_M == 0.020
    assert open_gap == pytest.approx(0.074)
    assert open_gap - 2.0 * q_close == pytest.approx(0.034)
