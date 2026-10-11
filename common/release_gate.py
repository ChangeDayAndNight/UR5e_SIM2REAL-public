"""릴리스(그리퍼 개방) 보상 계수 — ★순수 모듈★ (§4-2: torch/stdlib만).

목적: `release_bias_bootstrap`의 지급 계수를 env cfg 밖으로 꺼내 ★수치 불변식을
      pytest로 직접 검증★할 수 있게 한다.

★왜 분리했는가 (2026-08-11, F57)★
  '지급은 되는데 예산은 안 깎이는 영역'이 생기는 결함이 ★네 번 연속★ 재발했다
  (F52 가산 baseline → F53 지급·소모 결합 단절 → F56 임계-액션 경계 불일치 →
   F56 수정이 밴드를 오히려 확대). 매번 사람이 상수를 검산하는 방식으로는 막지 못했고,
  정책은 예외 없이 그 틈을 찾아냈다. ⇒ ★상수 검산이 아니라 함수 형태로★ 막는다.

★★2026-08-13(F64) 지급식 재설계 — F57의 relu 램프가 릴리스 학습을 죽였다★★
  F57은 위 결함(무한 스트림)을 ★구조로★ 막았고 그 부분은 옳았다. 그러나 닫힘 구간 지급을
  통째로 없애면서 ★액션을 개방 경계로 끌어당기던 복원력★까지 함께 제거했다.
  설정이 바이트 단위로 동일하고 커밋이 하나(c4e2d24)만 다른 A/B가 이를 보여준다:

    | run | `|a_grip|`rms @iter2k | release_bias 최댓값 |
    |---|---:|---:|
    | 2026-08-11_07-21-47 (F57 ★이전★) | ★0.588★ | ★7.99★ |
    | 2026-08-11_09-44-57 (F57 ★이후★) | ★1.850★ | ★0.05★ |

  F57 이후 6개 run 전부 `|a|`rms 1.67~2.05 · release_bias ≤ 0.10 (자기 상한 6.0의 1.7%).
  P(action>0) = 0.06%/스텝 ⇒ ★개방을 표집조차 못 한다★.

  ★F57의 추론이 왜 틀렸는가 (교훈)★
    F57은 "실측 액션 중앙이 −0.084라 경계에 붙어 있으니 shaping이 불필요"라 판단했다.
    그러나 그 −0.084는 ★sigmoid shaping이 작동 중인 run★에서 측정한 값이다 —
    액션을 경계에 붙들어 두고 있던 것이 바로 그 shaping이었다.
    ⇒ ★치료 효과를 측정해 치료가 불필요하다고 결론지은 순환 추론★.
    제거하자 액션은 즉시 −1.7로 되밀렸고, F57이 스스로 적어 둔 트립와이어(≲−0.5)를
    3.5배 넘겼는데 ★6개 run 동안 아무도 확인하지 않았다★(주석은 게이트가 아니다 —
    그래서 이 항목을 `scripts/tools/train_watch.py` 감시 지표로 승격했다).

★불변식 (테스트로 고정 — tests/test_release_gate.py)★
  ① `open_gain`은 ★모든 유한 action에서 > 0★이고 action에 대해 ★순증가·연속★이다.
     ⇒ 깊게 닫힌 정책에게도 "덜 닫으면 더 받는다"는 ★복원력★이 있다(F49의 원래 논거).
  ② 상태항(a_state)은 지급액을 ★변조만★ 한다 — 배수 ∈ [1−W, 1], 스스로 지급하지 않는다.
  ③ ★닫힘 구간 지급은 개방 대비 무시 가능★ — 실측 포화점(−1.7)에서 개방(+1.0)의 1/200 이하.
     '쥐고 버티기'가 '놓기'보다 이득이 되는 일이 수치적으로 불가능하다.
  ④ ★예산 소모 = 지급액 (계수 1, `grant_and_consume`)★.
     '지급 > 0 인데 소모 = 0'인 영역이 ★정의상★ 존재할 수 없다 — F52·F53·F56·F57이
     매번 임계 상수로 맞추려다 네 번 연속 실패한 그 영역을, 비교할 상수를 없애 제거한다.
  ⑤ 에피소드 총 지급 ≤ 예산(20 단위) — 상한이 정확히 지켜진다(초과분은 잘린다).
"""

from __future__ import annotations

import torch

__all__ = ["open_gain", "open_state_ratio", "grant_and_consume"]


def open_state_ratio(
    finger_q: torch.Tensor, open_m: float, close_m: float
) -> torch.Tensor:
    """실제 손가락 관절 위치 → 개도 비율 [0,1] (1 = 완전 개방).

    ManiSkill3 `StackCube-v1`의 `ungrasp_reward = Σ q_finger / gripper_width` 대응.
    이 저장소는 open=0·close=GRIPPER_CLOSE_M이라 부호가 반대라 1에서 뺀다.
    """
    span = max(abs(close_m - open_m), 1.0e-6)
    return (1.0 - (finger_q - open_m).abs().mean(dim=-1) / span).clamp(0.0, 1.0)


def open_gain(
    action_last: torch.Tensor,
    state_ratio: torch.Tensor,
    temp: float,
    state_weight: float,
) -> torch.Tensor:
    """릴리스 지급 계수 (0,1). ★모든 action에서 > 0★이며 action에 대해 순증가한다.

    형태: `sigmoid(action / temp) · ((1−W) + W·state_ratio)`   (F49 원형 = F64 복원)

    ★왜 닫힘 구간에도 지급하는가 — 이것이 이 함수의 존재 이유다★
      `BinaryJointPositionAction`은 액션의 ★부호만★ 본다(−1이든 −10이든 효과가 같다).
      따라서 정책이 깊게 음수로 밀려나도 물리적 대가가 없어 ★크기에 복원력이 없고★,
      개방(action>0)을 표집할 확률이 0으로 수렴한다(F46: 중앙 −9.9에서 개방 0.6%).
      닫힘 구간에서 ★0에 가까울수록 더 받는★ 지급식은 그 복원력을 만든다:

        τ=0.3에서  action −1.7 → 0.0034 · action −1.0 → 0.0345 · action −0.5 → 0.159
        ⇒ 덜 닫으면 지급이 ★10배씩★ 뛴다. 이것이 액션을 개방 경계로 밀어 올린다.

      이 복원력이 있던 run들은 `|a|`rms 0.588 · release_bias 7.99를 냈고,
      없앤 run들(F57 이후 6개)은 1.67~2.05 · ≤0.10에 고착했다(모듈 docstring A/B 표).

    ★그런데 왜 예전엔 이게 farming을 일으켰는가 — 원인은 이 함수가 아니었다★
      farming은 "지급 > 0 인데 예산 소모 = 0"인 구간에서 생겼고, 그 구간은 소모를
      ★별도 임계 상수★로 판정했기 때문에 생겼다(F52·F53·F56이 그 상수를 맞추려다 실패).
      F64는 소모를 `grant_and_consume`으로 ★지급액 자체★에 묶어 그 구간을 없앤다.
      ⇒ 닫힘 지급을 되살려도 farming 경로는 열리지 않는다. 두 결함은 ★독립★이었고,
        F57은 하나를 고치면서 다른 하나를 부수는 방식으로 둘을 묶어 버렸다.

    Args:
        action_last: 정책 액션의 그리퍼 차원. ★부호 계약: <0 = close, >0 = open★
            (`BinaryJointPositionAction`).
        state_ratio: `open_state_ratio` 결과 — 실제 개도 [0,1].
        temp: 램프 온도 τ. 작을수록 닫힘 구간 지급이 가파르게 작아진다.
        state_weight: 상태항 변조 가중 W ∈ [0,1). 배수 범위 = [1−W, 1].
    """
    ramp = torch.sigmoid(action_last / max(temp, 1.0e-6))
    return ramp * ((1.0 - state_weight) + state_weight * state_ratio)


def grant_and_consume(
    pay: torch.Tensor, budget: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """지급액을 잔여 예산으로 잘라 주고, ★그만큼★ 예산을 깎는다.

    반환: `(grant, new_budget)` — `grant = min(pay, budget)`, `new_budget = budget − grant`.

    ★이 함수가 막는 것 (F52·F53·F56·F57이 네 번 연속 실패한 결함)★
      과거 구현은 지급은 연속량(`pay`)으로 하면서 소모는 ★임계 비교★로 판정했다
      (`a_open > 0.25`, 뒤에는 `a_cmd > 0.25`, 그 뒤엔 `pay > 0`에 스텝당 1). 그래서
      '지급 > 0 ∧ 소모 = 0'인 밴드가 매번 어딘가에 생겼고 ★정책은 예외 없이 그것을 찾아냈다★.
      여기서는 소모량이 지급액과 ★같은 수★다. 비교할 상수가 없으므로 어긋날 수도 없다.

    ★스텝당 1 차감이 아니라 비례 차감인 이유★
      스텝당 1로 깎으면 닫힘 구간의 미세 지급(0.0034)도 예산을 1씩 태워, 20스텝이면
      ★정작 놓아야 할 때 예산이 0★이 된다 — F39가 실측한 그 사고다
      (rel_bias 0.62 → 0.027, 릴리스 부트스트랩 사멸). 비례 차감에서는:
        · 쥐고 버티기(0.0034/스텝) → 100스텝에 0.34 소모 (예산 20의 1.7%)
        · 실제 개방(≈1.0/스텝)     → 20스텝에 소진 = 종전과 같은 상한
      ⇒ 복원력은 에피소드 내내 살아 있고, 총 지급 상한은 그대로다.

    ⚠️호출측은 `budget`을 에피소드 리셋에서 초기값으로 되돌려야 한다(이 함수는 상태를 갖지 않는다).
    """
    room = budget.clamp(min=0.0)
    grant = torch.minimum(pay.clamp(min=0.0), room)
    return grant, budget - grant
