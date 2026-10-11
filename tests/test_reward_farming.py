"""보상 farming 경로 차단 계약 (순수 정적 검사, sim 불필요 — README 검증 계약).

배경(2026-07-30, F33): `release_bias_bootstrap`(w30)이 `_lifted_latch ∧ 타겟 근접`만 보고
      지급됐다. 그런데 `_lifted_latch`는 ★래치★라 pregrasp env가 리셋 즉시 공짜로 얻고,
      타겟은 커리큘럼상 물체 근처에 샘플링된다 → ★큐브를 떨어뜨리고 손만 벌린 채 두면★
      w30을 매 스텝 수확할 수 있었다. 잡고 유지하면 grasp(2)+lift(5)뿐이라 ★놓는 쪽이 압도★.
      teacher v8에서 frac=1.0인데도 grasp이 iter0부터 단조 감소한 근본원인.
      2x2 실측: 게이트 off → release_bias 0.4763 / 게이트 on → ★0.0015(300배 감소)★,
      같은 조건에서 grasp 0.0211 → 0.0304(+44%).

계약: ★place-side 보상(놓기를 유도하는 항)은 '공짜 래치'만으로 지급되면 안 된다★.
  각 항은 다음 중 하나 이상으로 방어돼야 한다:
    ① `_real_pick_mask` (pregrasp env 제외)  ② 현재 접촉 검사 `_any_finger_contact`
  ⚠️`_real_pick_mask` 단독 적용은 2026-07-16에 시도했다가 릴리스 학습이 죽었다(주석 참조) —
    release_bias는 ★접촉 게이트★로 방어하는 것이 현행 해법이다.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CFG = (Path(__file__).resolve().parent.parent / "source" / "ur5e_binpick" / "ur5e_binpick"
        / "tasks" / "binpick" / "binpick_env_cfg.py")

# 놓기(open)를 보상하거나 place 완성을 지급하는 항 — 전부 farming 방어가 필요하다
_PLACE_SIDE = ("release_bias_bootstrap", "placed_released", "place_fine")


def _src() -> str:
    if not _CFG.is_file():
        pytest.skip("sim 확장(source/) 없음 — 배포 번들에서는 대조 불가")
    return _CFG.read_text(encoding="utf-8")


def _body(name: str, src: str) -> str:
    m = re.search(rf"def {name}\(.*?(?=\ndef )", src, re.S)
    assert m, f"{name} 정의를 찾지 못함"
    return m.group(0)


@pytest.mark.parametrize("fn", _PLACE_SIDE)
def test_place_side_reward_is_guarded(fn: str) -> None:
    """★핵심★ place-side 보상은 mask 또는 접촉 게이트 중 하나로 방어돼야 한다."""
    src = _src()
    if f"def {fn}(" not in src:
        pytest.skip(f"{fn} 미정의(설계 변경) — 계약 재검토 필요")
    body = _body(fn, src)
    # 함수 본문 전체를 본다 - 구현이 여러 줄로 나뉘어도(pay 변수 경유 등) 가드를 놓치지 않는다.
    #   (구버전은 마지막 return 한 줄만 봐서 F39 리팩터 후 오탐이 났다.)
    guarded = any(g in body for g in ("_real_pick_mask", "_any_finger_contact", "_release_grace_mask"))
    assert guarded, (
        f"{fn}에 farming 가드가 없다\n"
        "→ 공짜 래치(pregrasp)만으로 지급되면 '떨어뜨리고 손 벌리기' farming이 열린다(F33).\n"
        "  _real_pick_mask 또는 _any_finger_contact로 방어할 것."
    )


def test_release_bias_uses_contact_based_gate() -> None:
    """release_bias는 ★접촉 기반★ 게이트로 방어한다 — mask 단독은 릴리스 학습을 죽인 전례가 있다.

    허용: `_any_finger_contact`(현재 접촉) 또는 `_release_grace_mask`(접촉 + 유한 유예).
    금지: `_real_pick_mask` 단독(2026-07-16에 릴리스 부트스트랩이 죽었다 — F33 주석 참조).
    """
    src = _src()
    body = _body("release_bias_bootstrap", src)
    assert ("_any_finger_contact" in body) or ("_release_grace_mask" in body), (
        "release_bias가 접촉 기반 게이트를 잃었다 — _real_pick_mask 단독으로 막았다가 "
        "릴리스 부트스트랩이 죽은 전례가 있다(F33 주석). 접촉 게이트 또는 유예 마스크가 현행 해법."
    )


def test_release_grace_window_is_finite() -> None:
    """★유예는 유한해야 한다★ — 무한 유예는 F33 farming을 그대로 되살린다.

    F34에서 '접촉 상실 후 유예'를 도입했다(낙하·정착 구간의 지급 골짜기를 메우려고).
    유예가 없거나 지나치게 길면 '오래 전에 떨어뜨린 큐브 옆에서 손 벌리기'가 다시 이득이 된다.
    물리 근거: 릴리스 높이 1~3cm 자유낙하 = 0.9~1.6 제어스텝(20Hz) + 정착 여유 → 한 자릿수 스텝.
    """
    src = _src()
    if "_release_grace_mask" not in src:
        pytest.skip("유예 마스크 미사용(현재 접촉 게이트만) — 이 계약은 해당 없음")
    body = _body("release_bias_bootstrap", src)
    m = re.search(r"_release_grace_mask\(env,\s*[\d.]+,\s*(\d+)\)", body)
    assert m, "유예 스텝 수를 리터럴로 명시할 것(감사 가능성)"
    grace = int(m.group(1))
    assert 1 <= grace <= 20, (
        f"유예 {grace}스텝이 물리 근거 범위를 벗어난다 — 낙하·정착은 한 자릿수 스텝이면 충분하고, "
        "길면 F33 farming이 되살아난다"
    )
    # ★F38★ 에피소드 예산 필수 — 유예만으로는 '열고-닫고 반복'으로 재수확된다
    mb = re.search(r"budget_steps\s*=\s*(\d+)", body)
    assert mb is not None, (
        "유예에 ★에피소드 예산★이 없다 — 열기(지급)→닫기(접촉 회복→유예 리셋)→열기 반복으로 "
        "무한 수확이 가능하다(F38 실사고: rel_bias 8.41 = 에피소드당 45스텝, grasp 1.67→1.02 붕괴)"
    )
    budget = int(mb.group(1))
    assert grace < budget <= 40, (
        f"예산 {budget}스텝이 부적절하다 — 유예({grace})보다 크되(1회는 완주 가능) "
        "정상 릴리스 몇 회분 이내여야 한다(큐브 1개 = 정상 1회 약 10스텝)"
    )


def test_release_budget_caps_reward_below_observed_farming() -> None:
    """★예산이 실사고 수준(rel_bias 8.41)을 구조적으로 불가능하게 만드는지 검산★.

    F38 실사고: 예산 없이 유예만 있을 때 rel_bias가 8.41까지 올라(에피소드당 45스텝 지급)
    grasp이 1.67 → 1.02로 붕괴했다. 예산 상한 = budget × weight / 에피소드길이.
    """
    src = _src()
    body = _body("release_bias_bootstrap", src)
    m = re.search(r"budget_steps\s*=\s*(\d+)", body)
    if not m:
        pytest.skip("예산 미사용")
    budget = int(m.group(1))
    w = re.search(r"release_bias\s*=\s*RewTerm\(.*?weight=([-\d.]+)", src, re.S)
    weight = float(w.group(1)) if w else 30.0
    # ★2026-08-06 정정★: 구버전은 `ep_len = 160` ★하드코딩★이었다. 에피소드를 8.0s → 5.0s로
    #   줄이자(F48 ③) 실제 100스텝인데 160으로 나눠 상한이 ★실제보다 작게★ 계산됐고,
    #   farming 검사가 헐거워졌다(무력화). ⇒ cfg 단일 소스에서 파생한다.
    m_ep = re.search(r"^EPISODE_LENGTH_S = ([\d.]+)", src, re.M)
    assert m_ep, "EPISODE_LENGTH_S 를 찾지 못함"
    ep_len = float(m_ep.group(1)) * 20.0   # control_hz = 20 (configs/common.yaml)
    cap = budget * weight / ep_len
    assert cap < 8.41, (
        f"rel_bias 상한 {cap:.2f} 가 실사고 수준(8.41) 미만이어야 한다 — "
        f"예산 {budget} × w{weight:.0f} / {ep_len} 이 너무 크다"
    )


def test_grasp_hold_outweighs_nothing_free() -> None:
    """놓기 보상(w)이 파지 유지 보상(w)보다 크면, 유지가 ★게이트로 보호★돼야 한다.

    가중치 자체는 설계 자유지만, '아무 기술 없이 받는 큰 보상'이 존재하면 안 된다는 것이 계약이다.
    """
    src = _src()
    def w(name: str) -> float:
        m = re.search(rf"{name}\s*=\s*RewTerm\(.*?weight=([-\d.]+)", src, re.S)
        return float(m.group(1)) if m else float("nan")
    w_rel, w_grasp = w("release_bias"), w("grasp")
    if w_rel > w_grasp:
        body = _body("release_bias_bootstrap", src)
        assert any(g in body for g in ("_any_finger_contact", "_real_pick_mask", "_release_grace_mask")), (
            f"release_bias(w{w_rel})가 grasp(w{w_grasp})보다 큰데 무방비다 — F33 재발"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ★F36(2026-07-31): carry 계열 보상항은 '최종 하강 예외'를 공유해야 한다★
#   goal_distance_carry는 2026-07-10 리뷰에서 (래치 ∧ 타겟 반경 내)면 z게이트를 면제하도록
#   고쳐졌는데 lift_grasped만 빠져 있었다. 그 결과 안착을 위해 하강하는 순간 lift(w5)만
#   사라져 스텝당 보상이 −12% 꺼지고, 정책이 ★타겟 위에 떠 있는 것★을 최적으로 학습했다
#   (v9 실측: hold_frac 0.876 · airborne_frac 0.908 · placed_frac 0.0000).
# ─────────────────────────────────────────────────────────────────────────────
_CARRY_SIDE = ("lift_grasped", "goal_distance_carry")


@pytest.mark.parametrize("fn", _CARRY_SIDE)
def test_carry_rewards_allow_final_descent(fn: str) -> None:
    """★carry 계열은 최종 하강 중에도 지급돼야 한다★ — 아니면 정책이 내려놓지 않는다.

    구현 형태: 파지 ∧ (z > minimal_height ∨ (배달 래치 ∧ 타겟 반경 내)).
    `latch_radius` 제한이 함께 있어야 '크로스벨트 드래그' farming이 막힌다(2026-07-10 리뷰).
    """
    src = _src()
    body = _body(fn, src)
    assert "_lifted_latch" in body, (
        f"{fn}에 배달 래치 분기가 없다 — 최종 하강(z<MIN_LIFT) 중 보상이 끊겨 "
        "정책이 타겟 위에 떠 있는 것을 최적으로 학습한다(F36)"
    )
    assert "latch_radius" in body, (
        f"{fn}의 래치 분기에 타겟 반경 제한이 없다 — 래치 후 바닥 끌기로 수확 가능(2026-07-10 리뷰)"
    )


def test_release_dominates_resting_at_target() -> None:
    """★놓기가 '잡은 채 얹어두기'보다 우월해야 한다★ — 아니면 새 farming이 열린다.

    F36으로 하강 중에도 lift가 지급되므로, '타겟에 얹어두고 계속 잡고 있기'가 새 후보가 된다.
    placed(w50)가 이를 압도하는지 가중치로 검산한다(순수 산술 — sim 불필요).
    """
    import math
    src = _src()
    def w(name: str) -> float:
        m = re.search(rf"\n    {name}\s*=\s*RewTerm\(.*?weight=([-\d.]+)", src, re.S)
        return float(m.group(1)) if m else 0.0
    tanh_term = lambda d, s: 1.0 - math.tanh(d / s)
    # 얹어두고 잡고 있기: reach + grasp + lift + goal + goal_fine (거리 ~0.5cm)
    resting = w("reach") + w("grasp") + w("lift") + w("goal") * tanh_term(0.005, 0.55) \
        + w("goal_fine") * tanh_term(0.005, 0.06)
    # 놓기(안착): reach + placed + goal_fine (파지 계열은 전부 0)
    released = w("reach") + w("placed") + w("goal_fine") * tanh_term(0.01, 0.06)
    assert released > resting * 1.5, (
        f"놓기 {released:.1f}/step 이 얹어두기 {resting:.1f}/step 을 1.5배 넘게 압도하지 못한다 "
        "— '타겟에 얹어두고 계속 잡고 있기' farming이 열린다(F36 도입 시 반드시 확인할 것)"
    )


def test_release_budget_consumed_only_on_payment() -> None:
    """★F39★ 예산은 '실제 지급' 시에만 차감돼야 한다 — 접촉만으로 소모되면 부트스트랩이 죽는다.

    실사고: 예산 차감을 `접촉 ∨ 유예`로 했더니 큐브를 ★쥐고만 있어도★ 매 스텝 깎여
      에피소드 초반 20스텝에 소진되고, 정작 놓을 때 지급이 0이 됐다
      (rel_bias 0.62 → ★0.027★, 23배 감소 = 릴리스 부트스트랩 사멸).
    올바른 형태: 지급액(pay)이 양수인 스텝에서만 차감.
    """
    src = _src()
    body = _body("release_bias_bootstrap", src)
    if "budget_steps" not in body:
        pytest.skip("예산 미사용")
    assert "grant_and_consume(pay, bud)" in body, (
        "예산 차감이 `grant_and_consume`(소모 = 지급액)을 거치지 않는다 — 접촉만으로 소모되면 "
        "릴리스 부트스트랩이 죽는다(F39)."
    )
    assert "a_open" in body, "지급식에 open 명령(a_open)이 반영돼야 한다"


# ═════════════════════════════════════════════════════════════════════════════
# ★F47(2026-08-03)★ placed 에피소드 예산 — '빨리 끝내는 이득' 제거
#
# 실측 근거: placed는 흡수 상태의 per-step 보상이라 1스텝 빨리 안착할수록 이득이 쌓인다.
#   그 결과 정책이 ★파지 신뢰성을 속도와 맞바꾼다★ — hold_frac이 학습 내내
#   72%(iter 6k, grasp 98.4%) → 2.8%(iter 21k, grasp 93.6%)로 단조 감소했고,
#   실패 1024ep 계측에서 실패는 스폰 위치와 ★무관★(반경별 3.5~5.9% 평평), 100%가 큐브에
#   ★접촉★하며(중앙 6스텝 — 성공과 동일) hold_frac은 ★0.00%★ = 잡아채고 재시도하지 않는다.
# ⇒ 에피소드당 상한을 두면 '언제 안착했는가'가 총 보상을 바꾸지 않아 서두를 이유가 사라진다.
# ═════════════════════════════════════════════════════════════════════════════


def test_placed_has_episode_budget() -> None:
    """placed가 무한 per-step 지급이면 '빨리 끝내기'가 파지 신뢰성을 잠식한다."""
    src = _src()
    body = _body("placed_released", src)
    assert "_bp_placed_budget" in body, (
        "placed에 에피소드 예산이 없습니다 — 서두름 유인이 파지 성공률을 갉아먹은 이력이 "
        "있습니다(F47: hold_frac 72% → 2.8%, grasp 98.4% → 93.6%)"
    )
    assert "PLACED_BUDGET_STEPS" in body, "예산 값은 모듈 상수 단일 소스에서 와야 합니다"


def test_placed_budget_consumed_only_on_payment() -> None:
    """예산은 ★지급이 발생한 스텝★에만 소모돼야 한다 (F39와 동일한 함정 방지).

    조건 미충족 스텝에서도 깎이면, 큐브를 쥐고만 있어도 예산이 말라 정작 안착 시 못 받는다.
    """
    body = _body("placed_released", _src())
    assert "paying = (ok > 0.0) & (env._bp_placed_budget > 0.0)" in body, (
        "예산 소모 조건이 '지급 발생'과 묶여 있지 않습니다"
    )
    assert "torch.where(\n        paying, env._bp_placed_budget - 1.0" in body or \
           "torch.where(paying, env._bp_placed_budget - 1.0" in body, (
        "예산 차감이 paying 마스크로 게이팅되지 않습니다"
    )


def test_placed_budget_resets_per_episode_only() -> None:
    """예산은 에피소드 리셋으로만 회복돼야 한다 — 그래야 반복 수확 경로가 없다."""
    body = _body("placed_released", _src())
    assert "episode_length_buf <= 1" in body, (
        "에피소드 시작 시 예산 복원 로직이 없습니다(또는 리셋 조건이 다릅니다)"
    )


def test_placed_budget_still_dominates_alternatives() -> None:
    """상한을 둬도 '안착해야 할 이유'는 남아야 한다.

    ★비교 틀(2026-08-05 정정)★: 이 프로젝트의 원 설계 근거는 ★스텝당★ 비교다
    (placed_released docstring: "최강 대안인 '타겟 위 호버-홀드'를 released가 2.2× 압도해야
    하강·릴리스가 학습된다"). 초판 테스트는 에피소드 총액을 비교하면서 대안 쪽에 shaping
    계수를 무시한 상한(전 가중치 합 × 전 에피소드)을 써서 ★대안을 과대평가★했다 —
    에피소드 길이를 줄이자 그 왜곡 때문에 실패했다.
    여기서는 ①스텝당 우위 ②예산이 에피소드의 유의미한 비율 두 가지를 본다.
    """
    import re

    src = _src()
    m = re.search(r"^PLACED_BUDGET_STEPS: int = (\d+)", src, re.M)
    assert m, "PLACED_BUDGET_STEPS 상수를 찾지 못함"
    budget = int(m.group(1))
    m = re.search(r"^\s+placed = RewTerm\((?:.|\n)*?weight=([\d.]+)", src, re.M)
    assert m, "placed weight 를 찾지 못함"
    w_placed = float(m.group(1))

    # 최강 대안 = 타겟 위에서 쥔 채 버티기: grasp + lift + goal + goal_fine 이 동시 지급되는 상태
    alt = 0.0
    for term in ("grasp", "lift", "goal", "goal_fine"):
        mm = re.search(rf"^\s+{term} = RewTerm\((?:.|\n)*?weight=([\d.]+)", src, re.M)
        assert mm, f"{term} weight 를 찾지 못함"
        alt += float(mm.group(1))
    assert w_placed > 1.5 * alt, (
        f"★스텝당★ placed({w_placed}) 가 대안 '쥔 채 버티기'({alt}) 를 1.5배 이상 압도하지 못한다 — "
        "하강·릴리스 대신 호버-홀드에 고착한 이력이 있다(F34)"
    )

    # 예산이 에피소드의 유의미한 비율이어야 '언젠가는 해야 한다'는 압력이 남는다
    m = re.search(r"^EPISODE_LENGTH_S = ([\d.]+)", src, re.M)
    assert m, "EPISODE_LENGTH_S 를 찾지 못함"
    ep_steps = float(m.group(1)) * 20.0        # control_hz = 20
    assert 0.3 <= budget / ep_steps <= 0.7, (
        f"예산 {budget} / 에피소드 {ep_steps:.0f}스텝 = {budget/ep_steps:.2f} — "
        "너무 작으면 안착 유인이 죽고, 너무 크면 서두름 유인이 남는다(0.3~0.7 권장)"
    )


# ═════════════════════════════════════════════════════════════════════════════
# ★F49(2026-08-06)★ 릴리스 shaping은 '닫힌 개체'에도 gradient를 줘야 한다
#
# 실측: 10k run에서 정지 93.9% 도달인데 placed 85.2% — ★래치∧릴리스 86.3%★로
#   13.7%가 잡은 뒤 끝까지 놓지 않았다. 그 개체들의 그리퍼 액션은 음수인데,
#   구현이 `a_open = clamp(action, 0, 1)`이라 ★값도 기울기도 정확히 0★이었다.
#   ⇒ 릴리스 절벽을 잇는 다리가, ★정작 그 다리가 필요한 개체에만 작동하지 않았다★.
#   (F46의 액션 포화와 같은 기전이 크기 페널티 도입 후 잔여 개체군에 남아 있던 것.)
# ═════════════════════════════════════════════════════════════════════════════


def test_release_payout_is_zero_while_gripper_closed() -> None:
    """★F57(2026-08-11) 회귀 고정 — F49 shaping 은퇴★: 닫힌 채(action ≤ 0) 지급은 ★정확히 0★.

    [폐기된 구 계약] F49는 "clamp면 action<0인 개체의 gradient가 0이라 못 배운다"며
      닫힘 구간에도 소액을 지급하는 sigmoid를 요구했다(구 테스트는 그것을 강제했다).
    [은퇴 근거] 그 전제(정책이 깊게 음수로 포화, 당시 −1.2~−9.9)는 F46 크기 페널티 이후
      성립하지 않는다 — 실측 액션 중앙은 ★−0.084★(2026-08-11 eval)로 개방 경계 바로 아래고,
      이 위치에서는 action > 0 이 매 스텝 충분히 표집된다. 반면 그 shaping은 farming
      ★4연속 재발(F52·F53·F56·F56수정)의 유일한 원인★이었다 — 이득은 사라지고 비용만 남았다.
    ⇒ 이제는 반대로 ★닫힘 구간 지급이 0인지★를 강제한다.

    수치 불변식 전수 검사는 `tests/test_release_gate.py`가 담당한다. 여기서는
    env cfg가 그 순수 모듈을 ★실제로 경유하는지★(우회 구현이 되살아나지 않았는지) 본다.
    """
    body = _body("release_bias_bootstrap", _src())
    assert "open_gain(" in body, (
        "지급 계수를 common/release_gate.open_gain 으로 산출하지 않는다 — "
        "env cfg 안에 별도 지급식이 다시 생기면 불변식 테스트가 무력해진다(F57)"
    )
    assert "torch.sigmoid" not in body, (
        "닫힘 구간에 지급이 생기는 sigmoid 램프가 되살아났다 (F52/F53/F56 기전 재발)"
    )
    assert "RELEASE_OPEN_TEMP" in body, "온도는 모듈 상수 단일 소스에서 올 것"


def test_release_has_no_separate_attempt_threshold() -> None:
    """★F57★ 지급과 예산 소모를 가르는 ★별도 임계 상수가 존재하지 않아야★ 한다.

    F52→F53→F56, 그리고 F56의 수정까지 네 번 모두 ★'지급식'과 '소모 판정'이 서로 다른 양을
    보는 구조★에서 나왔다. 임계를 어떻게 고르든 둘이 어긋나는 영역은 생길 수 있고,
    정책은 예외 없이 그 틈을 찾아냈다(액션 −0.249 / −0.301 / −0.084).
    ★상수를 없애는 것★이 유일하게 구조적인 해법이다.
    """
    src = _src()
    live = chr(10).join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "RELEASE_OPEN_ATTEMPT" not in live, (
        "예산 소모 임계 상수가 되살아났다 — 지급 영역과 소모 영역이 다시 어긋날 수 있다(F57)"
    )
    body = _body("release_bias_bootstrap", src)
    code = chr(10).join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    # ★F64★ 소모는 지급액 그 자체다 — 판정할 술어도, 비교할 상수도 존재하지 않는다.
    assert "grant, bud = grant_and_consume(pay, bud)" in code, (
        "예산 소모가 지급액에 묶여 있지 않다 — 별도 판정이 끼면 지급>0 ∧ 소모=0 영역이 "
        "다시 생긴다(F52/F53/F56이 네 번 연속 그렇게 실패했다)"
    )
    assert "bud - 1.0" not in code, (
        "★F39 재발★ 스텝당 1 차감은 닫힘 미세지급으로도 예산을 태워, 정작 놓을 때 지급이 0이 된다"
    )


def test_release_budget_cannot_burn_while_closed() -> None:
    """★F49의 원래 걱정 — 닫힌 채 예산이 타버리면 정작 놓을 때 지급이 0이 된다★.

    F49는 "smooth라 pay가 항상 > 0이므로 존에 들어서자마자 예산이 타버린다"를 막으려
    별도 임계를 뒀고, F57은 닫힘 지급 자체를 0으로 만들어 같은 목적을 달성했다.
    ★그런데 F57의 방식은 릴리스 학습을 죽였다★(F64: |a|rms 0.588→1.850, RB 7.99→0.05).

    ★F64의 해법★ — 닫힘 지급은 되살리되 소모를 ★지급액에 비례★시킨다. 그러면 닫힘 구간의
    미세 지급(0.0034/스텝)은 예산도 그만큼만 태우므로, 에피소드 전체를 쥐고 있어도
    예산은 거의 그대로 남는다. F49의 걱정과 F64의 복원력이 ★동시에★ 성립한다.
    """
    import torch as _t

    from common.release_gate import grant_and_consume, open_gain

    bud = _t.tensor([20.0])
    for _ in range(100):                       # 에피소드 전체를 '쥐고 버티기'
        pay = open_gain(_t.tensor([-1.7]), _t.tensor([0.4]), 0.3, 0.5)
        _, bud = grant_and_consume(pay, bud)
    spent = 20.0 - float(bud)
    assert spent < 1.0, (
        f"닫은 채 100스텝에 예산 {spent:.3f} 소모 — 놓을 때 쓸 예산이 남지 않는다(F39/F49)"
    )
