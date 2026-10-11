"""placed '정지' 임계 계약 — 지터 바닥보다 높아야 한다 (순수 정적 검사, sim 불필요).

배경(2026-07-31, F37): `placed`의 `max_speed`가 ★0.10 m/s★였는데, 이는 이 sim의
      ★접촉 지터 바닥보다 낮았다★. Grasp_PhysicsTest --release-test 실측
      (큐브를 벨트에 내려놓고 그리퍼 개방 후 1초, 120샘플):
        실제 이동 0.10cm / 0.99s = ★0.0010 m/s★  (사실상 정지)
        root_lin_vel_w 보고값   = 최소 0.105 / 중앙 0.154 / 최대 0.204  → ★153배 과대★
      즉 큐브는 멈춰 있는데 고주파 접촉 지터가 속도로 읽힌다. 임계 0.10은 지터 ★최소값
      (0.105)보다도 낮아★ placed가 구조적으로 거의 지급되지 않았고, 정책은 '놓아도 보상이
      없으니 쥐고 있는' 쪽을 학습했다(v11: hold_frac 0.891 / placed 6.8%).

계약:
  ①임계는 실측 지터 바닥(0.105)보다 충분히 높아야 한다 — 아니면 놓아도 성공 판정이 안 난다.
  ②임계는 ★단일 소스★여야 한다 — 학습과 평가가 다른 값을 쓰면 '학습에선 성공, 평가에선 실패'.
    (구버전은 binpick_env_cfg와 BinPick_Eval에 0.10이 ★따로★ 박혀 있었다 — F30과 같은 부류.)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CFG = (_ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
        / "binpick_env_cfg.py")
_EVAL = _ROOT / "Phase1" / "BinPick_Eval.py"

# ★2026-08-01 정정★ 기준값을 ★학습 환경(teacher env)★ 실측으로 교체.
#   구값(0.105~0.204)은 `Grasp_PhysicsTest`의 ★자체 큐브★에서 나온 것이라 비대표였다
#   (그 큐브는 RigidBodyPropertiesCfg(disable_gravity=False)만 설정 — solver 반복수 미적용).
#   학습 환경 큐브(solver_position_iteration_count=16 등) 실측: Pregrasp_Probe 40샘플,
#   벨트에 정지 시 ★전부 0.0010 m/s★ (100%가 문헌 표준 0.01 미만).
_JITTER_MAX = 0.0010          # 학습 환경 정지 큐브 실측 최대
_LITERATURE_STD = 0.01        # Isaac Lab pick&place 성공 기준의 정지 판정


def _cfg() -> str:
    if not _CFG.is_file():
        pytest.skip("sim 확장(source/) 없음")
    return _CFG.read_text(encoding="utf-8")


def _threshold() -> float:
    m = re.search(r"PLACED_MAX_SPEED:\s*float\s*=\s*([0-9.]+)", _cfg())
    assert m, "PLACED_MAX_SPEED 상수를 찾지 못했다 — 임계가 다시 하드코딩됐을 수 있다(F37)"
    return float(m.group(1))


def test_threshold_exceeds_measured_jitter_floor() -> None:
    """★핵심★ 임계가 ★학습 환경★ 실측 지터보다 충분히 높아야 놓기가 성공으로 인정된다.

    ⚠️기준값은 반드시 ★학습 환경★에서 재야 한다 — 진단 도구(Grasp_PhysicsTest)는 자체 큐브를
      쓰므로 물리 설정이 달라 값이 100배 이상 벌어진다(F37에서 실제로 오진했다).
    """
    thr = _threshold()
    assert thr > _JITTER_MAX * 10, (
        f"placed 정지 임계 {thr} m/s 가 학습 환경 실측 지터({_JITTER_MAX})의 10배 미만이다 — "
        "여유가 없으면 큐브가 멈춰 있어도 '정지'로 인정되지 않을 수 있다."
    )


def test_threshold_not_looser_than_literature_by_much() -> None:
    """★성공 기준을 부당하게 완화하지 말 것★ — 문헌 표준(0.01)의 20배 이내로 제한한다.

    F37에서 잘못된 지터 측정을 근거로 0.30(문헌의 30배)까지 올렸다가 정정한 이력이 있다.
    임계를 올리면 '아직 움직이는 큐브'도 안착으로 판정돼 성공률이 부풀려진다.
    """
    thr = _threshold()
    assert thr <= _LITERATURE_STD * 20, (
        f"placed 정지 임계 {thr} m/s 가 문헌 표준 {_LITERATURE_STD}의 20배를 넘는다 — "
        "성공 기준이 부당하게 느슨해져 실기 전이 시 과대평가로 이어진다."
    )


def test_threshold_is_not_absurdly_loose() -> None:
    """반대로 너무 느슨하면 '굴러가는 큐브'도 통과한다 — 물리적 상한을 둔다."""
    thr = _threshold()
    assert thr <= 0.5, (
        f"placed 정지 임계 {thr} m/s 는 과도하게 느슨하다 — 실제로 굴러가는 큐브가 "
        "'안착'으로 판정될 수 있다"
    )


def test_placed_term_consumes_the_constant() -> None:
    """placed RewTerm이 상수를 소비해야 한다(리터럴 재하드코딩 금지)."""
    src = _cfg()
    m = re.search(r"placed = RewTerm\(.*?params=\{(.*?)\}", src, re.S)
    assert m, "placed RewTerm을 찾지 못함"
    params = m.group(1)
    assert "PLACED_MAX_SPEED" in params, (
        f"placed가 상수를 쓰지 않는다: {params[:120]} — 값이 두 곳으로 갈라지면 F30/F37이 재발한다"
    )


def test_eval_uses_the_same_constant() -> None:
    """★평가도 같은 상수★ — 다르면 '학습에선 성공, 평가에선 실패'가 된다."""
    if not _EVAL.is_file():
        pytest.skip("BinPick_Eval.py 없음")
    src = _EVAL.read_text(encoding="utf-8")
    if "settled" not in src:
        pytest.skip("평가에 settled 판정이 없음(설계 변경)")
    assert "PLACED_MAX_SPEED" in src, (
        "BinPick_Eval이 placed 정지 임계를 자체 리터럴로 쓰고 있다 — "
        "binpick_env_cfg.PLACED_MAX_SPEED를 import해 쓸 것(F37)"
    )
    m = re.search(r"settled\s*=.*?<\s*([0-9.]+)", src)
    assert m is None, f"평가에 리터럴 임계가 남아 있다: {m.group(0)[:80]}"
