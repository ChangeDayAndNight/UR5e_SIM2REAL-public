"""릴리스 지급 계수의 ★수치 불변식★ — common/release_gate.py (sim 불필요, 순수).

★왜 이 파일이 따로 있는가 (F57, 2026-08-11)★
  '지급 > 0 인데 예산은 안 깎이는 영역'이 ★네 번 연속★ 재발했다(F52·F53·F56·F56수정).
  기존 회귀 테스트는 env cfg 소스를 ★문자열로 파싱해 상수를 검산★하는 방식이었는데,
  결정적으로 ★허용치를 검사 대상 상수 자신으로 표현★해 버려(`worst <= att`) 임계를
  올리면 테스트도 같이 느슨해졌다 — 그래서 F56 수정이 밴드를 넓혔는데도 통과했다.
  ⇒ 여기서는 ★실제 함수를 호출해 액션 전 구간을 훑는다★. 상수 값과 무관하게 성립해야 하는
    성질만 검사하므로, 파라미터를 어떻게 바꾸든 결함이 들어오면 반드시 깨진다.

★2026-08-13(F64) 이 파일의 절반이 틀린 것을 고정하고 있었다★
  구 `test_payout_is_exactly_zero_for_every_closed_action` / `..._would_earn_nothing`은
  "닫힘 구간 지급 = 0"을 ★요구★했다. 그런데 바로 그 성질이 릴리스 학습을 죽인 원인이었다
  (설정 동일·커밋 1개 A/B에서 `|a|`rms 0.588 → 1.850, release_bias 7.99 → 0.05).
  ⇒ 두 테스트를 삭제하고, ★farming을 막는 진짜 성질★로 교체했다:
    farming을 막는 것은 '닫힘 지급 0'이 아니라 ★소모 = 지급★(무한 스트림 영역 부재)이다.
    ★교훈★: 테스트는 '지금 구현이 하는 일'이 아니라 ★성립해야 하는 성질★을 적어야 한다.
    구현을 그대로 베낀 테스트는 그 구현이 틀렸을 때 ★틀림을 고정하는 장치★가 된다.
"""

from __future__ import annotations

import torch

from common.release_gate import grant_and_consume, open_gain, open_state_ratio

# 실제 사용 파라미터에 얽매이지 않도록 여러 조합을 훑는다.
_TEMPS = (0.1, 0.3, 0.5, 1.0)
_WEIGHTS = (0.0, 0.25, 0.5, 0.9)
# 완전개방 … 50mm 큐브 파지(q≈0.012) … 완전폐합
_STATES = (0.0, 0.2, 0.4, 0.6, 1.0)

_TAU = 0.3          # RELEASE_OPEN_TEMP (env cfg 실제값)
_W = 0.5            # RELEASE_STATE_WEIGHT (env cfg 실제값)


def _actions() -> torch.Tensor:
    """액션 −5.0 … +5.0 을 촘촘히 (경계 0 부근은 더 촘촘히)."""
    coarse = torch.arange(-500, 501, dtype=torch.float64) / 100.0
    fine = torch.arange(-200, 201, dtype=torch.float64) / 10000.0   # ±0.02 구간
    return torch.cat([coarse, fine]).sort().values


# ══════════════════════════════════════════════════════════════════════════════
# ★불변식 ① — 복원력: 깊게 닫힌 정책에게도 "덜 닫으면 더 받는다"가 성립해야 한다
# ══════════════════════════════════════════════════════════════════════════════
def test_closed_region_has_a_live_restoring_gradient():
    """★F64 핵심★ 닫힘 구간 어디서도 지급이 0이 아니고, 0에 가까울수록 커야 한다.

    이것이 없으면 `BinaryJointPositionAction`(부호만 보는 액션)에서 액션 크기에 복원력이
    사라져 정책이 −1.7 ~ −9.9로 밀려나고 개방을 ★표집조차 못 한다★(F46: 개방 0.6%,
    F57 이후: P(action>0) = 0.06%/스텝).
    """
    for temp in _TEMPS:
        for w in _WEIGHTS:
            for st in _STATES:
                if w == 1.0 and st == 0.0:      # 배수 0 = 지급 0 (해당 조합 제외)
                    continue
                a = torch.tensor([-5.0, -1.7, -1.0, -0.5, -0.1])
                g = open_gain(a, torch.full_like(a, st), temp, w)
                assert float(g.min()) > 0.0, (
                    f"닫힘 구간 지급이 0이다 (temp={temp}, W={w}, state={st}) — "
                    "복원력이 사라지면 액션이 음수로 밀려나 개방을 표집하지 못한다 (F64)"
                )
                d = g[1:] - g[:-1]
                assert float(d.min()) > 0.0, (
                    f"닫힘 구간에서 '덜 닫을수록 이득'이 성립하지 않는다 "
                    f"(temp={temp}, W={w}, state={st})"
                )


def test_restoring_force_is_steep_enough_to_matter_at_the_measured_saturation():
    """실측 포화점(−1.7)에서 한 단계 덜 닫으면(−1.0) 지급이 ★몇 배★ 뛰어야 한다.

    PPO는 표집된 액션들 사이의 리턴 ★차이★로 배운다. 차이가 미미하면 복원력은 이름뿐이다.
    τ=0.3에서 0.0034 → 0.0345 = 10배. 3배 미만이면 이 항은 사실상 평평하다.
    """
    st = torch.tensor([0.4])                       # 50mm 큐브 파지 시 실측 개도
    deep = float(open_gain(torch.tensor([-1.7]), st, _TAU, _W))
    less = float(open_gain(torch.tensor([-1.0]), st, _TAU, _W))
    assert less / deep >= 3.0, (
        f"−1.7 → −1.0 에서 지급이 {less / deep:.2f}배밖에 안 는다 — 복원력이 너무 평평하다"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ★불변식 ③ — 그래도 '쥐고 버티기'가 '놓기'보다 이득이면 안 된다
# ══════════════════════════════════════════════════════════════════════════════
def test_holding_closed_pays_negligibly_versus_actually_opening():
    """닫힘 지급을 되살리되, 개방 대비 ★무시 가능★해야 한다(F52 farming의 정량 차단).

    F52의 사고는 닫은 채 a_open≈0.20(개방의 20%)이 지급된 것이었다. 여기서는
    실측 포화점에서 개방의 1/200 이하여야 한다.
    """
    st_hold, st_open = torch.tensor([0.4]), torch.tensor([1.0])
    hold = float(open_gain(torch.tensor([-1.7]), st_hold, _TAU, _W))
    release = float(open_gain(torch.tensor([1.0]), st_open, _TAU, _W))
    assert release / hold >= 200.0, (
        f"쥐고 버티기 {hold:.5f} 대비 개방 {release:.5f} = {release / hold:.1f}배뿐 — "
        "닫힘 지급이 과하면 F52 farming이 재발한다"
    )


def test_payout_is_monotone_and_continuous_in_action():
    """액션에 대해 단조 증가하며 점프가 없다.

    단조성이 없으면 '적게 여는 것이 더 이득'인 구간이 생기고, 점프가 있으면 경계 근처에서
    표집 분산이 커질 때 보상이 튀어 학습이 불안정해진다.
    """
    a = _actions()
    for temp in _TEMPS:
        for w in _WEIGHTS:
            for st in _STATES:
                g = open_gain(a, torch.full_like(a, st), temp, w)
                d = g[1:] - g[:-1]
                assert float(d.min()) >= -1e-12, f"단조 증가 위반 (temp={temp}, W={w}, state={st})"
                assert float(d.max()) < 0.05, f"경계 도약 {float(d.max()):.4f} — 연속성 위반"


def test_actual_opening_pays_more_than_blocked_opening():
    """같은 개방 명령이라도 ★실제로 벌어진 쪽★이 더 받아야 한다.

    F50의 취지 — 명령은 완전 개방인데 접촉에 막혀 16.8mm에 정체한 상태가 만점을 받으면
    보상이 '놓으려는 시늉'과 '실제로 놓음'을 구분하지 못한다.
    """
    a = torch.tensor([1.0])
    for temp in _TEMPS:
        for w in (0.25, 0.5, 0.9):
            blocked = float(open_gain(a, torch.tensor([0.4]), temp, w))
            opened = float(open_gain(a, torch.tensor([1.0]), temp, w))
            assert opened > blocked, f"막힘 {blocked:.4f} ≥ 개방 {opened:.4f} (temp={temp}, W={w})"


def test_state_ratio_only_modulates_and_never_pays_on_its_own():
    """상태항은 배수 [1−W, 1]로 ★변조만★ 한다 — 스스로 지급을 만들지 않는다.

    F52의 발원지: 가산 구현(`(1−W)·a_cmd + W·a_state`)이라 닫은 채로도 a_state가 지급했다.
    """
    a = _actions()
    for temp in _TEMPS:
        for w in (0.0, 0.25, 0.5, 0.9):
            base = open_gain(a, torch.ones_like(a), temp, w)          # state=1 → 배수 1
            for st in _STATES:
                g = open_gain(a, torch.full_like(a, st), temp, w)
                ratio = g / base
                assert float(ratio.min()) >= (1.0 - w) - 1e-9, "배수가 1−W 아래로 내려갔다"
                assert float(ratio.max()) <= 1.0 + 1e-9, "배수가 1을 넘었다"


def test_state_ratio_maps_joint_position_to_opening():
    """개도 비율: 완전개방 → 1, 완전폐합 → 0, 범위 밖은 clamp."""
    open_m, close_m = 0.0, 0.020
    q = torch.tensor([[0.0, 0.0], [0.010, 0.010], [0.020, 0.020], [0.030, 0.030]])
    r = open_state_ratio(q, open_m, close_m)
    assert abs(float(r[0]) - 1.0) < 1e-9, "완전개방이 1이 아니다"
    assert abs(float(r[1]) - 0.5) < 1e-9, "중간 개도가 0.5가 아니다"
    assert abs(float(r[2]) - 0.0) < 1e-9, "완전폐합이 0이 아니다"
    assert float(r[3]) == 0.0, "폐합을 넘어선 값이 clamp되지 않았다"


# ══════════════════════════════════════════════════════════════════════════════
# ★불변식 ④⑤ — 예산: farming을 막는 진짜 장치
# ══════════════════════════════════════════════════════════════════════════════
def test_consumption_equals_grant_exactly_so_no_free_band_can_exist():
    """★F64 핵심★ 소모량 = 지급액. '지급 > 0 ∧ 소모 = 0'인 ★착취 가능한★ 액션이 없어야 한다.

    F52·F53·F56·F57이 네 번 연속 실패한 그 밴드를, 비교할 임계 상수를 없애 제거한다.

    ⚠️부동소수점 주의: 지급이 예산(20)의 eps보다 작으면 `20 − 1e-22 == 20`으로 ★흡수★되어
      소모가 0으로 보인다. 이는 설계 결함이 아니라 표현 한계이며, 그런 지급은 에피소드
      전체를 모아도 예산의 1e-6에도 못 미쳐 ★착취가 불가능하다★. 그래서 검사를
      "소모가 0인 지점은 전부 착취 불가능하다"로 세운다 — 이것이 실제로 성립해야 하는 성질이다.
    """
    a = _actions()
    EP_STEPS = 200                                   # 에피소드 예산(100스텝)의 2배로 여유
    for temp in _TEMPS:
        for w in _WEIGHTS:
            for st in _STATES:
                pay = open_gain(a, torch.full_like(a, st), temp, w)
                bud = torch.full_like(pay, 20.0)
                grant, new_bud = grant_and_consume(pay, bud)
                # 대수적 항등: 지급 = min(지급요청, 잔여예산)
                assert torch.allclose(grant, torch.minimum(pay, bud)), "지급이 산식과 다르다"
                spent = bud - new_bud
                absorbed = (grant > 0.0) & (spent <= 0.0)
                if not bool(absorbed.any()):
                    continue
                worst = float((grant[absorbed] * EP_STEPS).max())
                assert worst < 20.0 * 1e-6, (
                    f"소모 없이 모을 수 있는 총액 {worst:.3g} (temp={temp}, W={w}, state={st}) — "
                    "착취 가능한 무한 스트림 밴드다"
                )


def test_total_episode_payout_is_capped_by_the_budget():
    """★불변식 ⑤★ 에피소드 총 지급 ≤ 예산. 상한을 넘길 방법이 없어야 한다.

    실제 개방(≈1.0/스텝)을 200스텝 유지해도 총합이 예산 20을 넘지 않아야 한다.
    """
    bud = torch.tensor([20.0])
    total = 0.0
    for _ in range(200):
        pay = open_gain(torch.tensor([1.0]), torch.tensor([1.0]), _TAU, _W)
        grant, bud = grant_and_consume(pay, bud)
        total += float(grant)
    assert total <= 20.0 + 1e-6, f"총 지급 {total:.4f} 이 예산 20 을 넘었다"
    assert abs(total - 20.0) < 1e-4, f"예산을 다 쓰지 못했다 ({total:.4f}) — 상한 계산이 어긋난다"
    assert float(bud) <= 1e-6, "예산이 남아 있다"


def test_holding_closed_barely_drains_the_budget_over_a_full_episode():
    """★F39 재발 방지★ 쥐고만 있어도 예산이 타 버리면, 정작 놓을 때 지급이 0이 된다.

    F39 실측: 스텝당 1 차감이라 20스텝에 소진 → rel_bias 0.62 → 0.027(릴리스 사멸).
    비례 차감에서는 100스텝(에피소드 전체) 쥐고 있어도 예산의 5% 미만만 소모돼야 한다.
    """
    bud = torch.tensor([20.0])
    for _ in range(100):
        pay = open_gain(torch.tensor([-1.7]), torch.tensor([0.4]), _TAU, _W)
        _, bud = grant_and_consume(pay, bud)
    spent = 20.0 - float(bud)
    assert spent < 1.0, (
        f"쥐고 버티기 100스텝에 예산 {spent:.3f} 소모 — 놓을 때 쓸 예산이 남지 않는다 (F39)"
    )


def test_grant_never_exceeds_remaining_budget():
    """잔여 예산보다 많이 지급되지 않는다(음수 예산이 생기지 않는다)."""
    bud = torch.tensor([0.3, 0.0, -1.0, 5.0])
    pay = torch.tensor([1.0, 1.0, 1.0, 1.0])
    grant, new_bud = grant_and_consume(pay, bud)
    assert torch.allclose(grant, torch.tensor([0.3, 0.0, 0.0, 1.0])), f"지급 잘림 오류: {grant}"
    # 이미 음수인 예산(bud=-1)은 이 함수가 고칠 수 없다 — ★더 나빠지지 않는 것★이 불변식이다
    assert torch.all(new_bud >= torch.minimum(bud, torch.zeros_like(bud)) - 1e-9), (
        f"예산이 더 깎였다: {bud} → {new_bud}")
