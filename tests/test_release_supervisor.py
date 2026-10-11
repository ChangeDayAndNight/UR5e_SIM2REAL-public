"""릴리스 수퍼바이저 계약 회귀 (2026-08-16, F65 이후 신설).

이 항이 존재하는 이유는 프로브 실측이다(model_6000·256ep·성공종료 비활성·≥10스텝 배포 등급):

    정책이 스스로 개방                    0.0%
    강제 개방 + 팔 계속 이동             11.7%
    강제 개방 + 팔 정지 (큐브 pose 트리거) 63.7%   ← sim 전용 신호
    ★수퍼바이저(TCP 정지 트리거) + 팔 정지 67.6%★  ← 배포 가능

여기서 지키는 계약은 세 가지다.
  ① 판정은 ★배포 가능한 신호만★ 쓴다 — TCP pose·TCP 속도·그리퍼 명령 부호.
     큐브 pose를 끌어들이면 sim에서만 되는 정책이 된다(실기에 추정기가 없다).
  ② 릴리스는 ★한 방향으로 굳는다★ — 놓은 뒤 다시 닫으면 큐브를 건드린다.
  ③ 리셋에서 latch가 ★반드시 풀린다★ — 안 풀리면 다음 에피소드가 '이미 놓은' 상태로
     시작해 팔이 영구 정지한다(에피소드 전량 실패).
"""

from __future__ import annotations

import pytest
import torch

from common.release_supervisor import (
    ReleaseSupervisorCfg,
    carry_latch_update,
    edge_lockout_update,
    grasp_allowed,
    load_release_supervisor_cfg,
    release_decision,
    settle_counter_update,
)


def _cfg() -> ReleaseSupervisorCfg:
    return ReleaseSupervisorCfg(lift_rise_m=0.010, tcp_z_max_m=0.075, tcp_settle_mps=0.02,
                                grasp_z_max_m=0.040, tcp_settle_steps=3,
                                gripper_edge_lockout_steps=6)


def test_config_loads_from_common_yaml() -> None:
    """configs/common.yaml이 단일 소스다 — sim/real이 같은 상수를 읽는다."""
    cfg = load_release_supervisor_cfg()
    assert cfg.lift_rise_m > 0.0
    assert cfg.tcp_settle_mps > 0.0


def test_rejects_nonpositive_lift_rise() -> None:
    """폐합 뒤 실제로 올라와야 운반이다 — 상승량은 양수여야 한다."""
    with pytest.raises(ValueError):
        ReleaseSupervisorCfg(lift_rise_m=0.0, tcp_z_max_m=0.075, tcp_settle_mps=0.02,
                             grasp_z_max_m=0.040, tcp_settle_steps=3,
                             gripper_edge_lockout_steps=6)


def test_carry_latch_uses_relative_rise_not_absolute_height() -> None:
    """★F68-c★ 운반 판정은 폐합 시점 대비 상승량이다 — 절대 높이가 아니다.

    실측(채택 teacher 256ep): 들림 중 TCP z는 p50 0.0319 / p99 0.0618이고 구 임계 0.100을
    넘는 스텝은 0.01%뿐이었다. 절대 높이로는 정상 운반이 latch되지 않는다.
    """
    cfg = _cfg()
    carried = torch.zeros(4, dtype=torch.bool)
    closed = torch.tensor([True, True, False, True])
    rise = torch.tensor([0.02, 0.001, 0.02, 0.05])   # 2번째는 상승 부족, 3번째는 열림
    out = carry_latch_update(carried, closed, rise, cfg)
    assert out.tolist() == [True, False, False, True]

def test_carry_latch_is_sticky() -> None:
    """한 번 걸린 latch는 내려와도 유지된다 — 배치하려면 내려와야 하기 때문이다."""
    cfg = _cfg()
    carried = torch.tensor([True])
    out = carry_latch_update(carried, torch.tensor([False]), torch.tensor([0.0]), cfg)
    assert bool(out[0])


def test_release_requires_carried_low_and_settled() -> None:
    cfg = _cfg()
    carried = torch.tensor([True, True, True, False])
    tcp_z = torch.tensor([0.05, 0.05, 0.20, 0.05])       # 3번째는 아직 높다
    settle = torch.tensor([3, 1, 3, 3])                  # 2번째는 아직 연속 정지가 모자란다
    out = release_decision(carried, tcp_z, settle, cfg)
    assert out.tolist() == [True, False, False, False]


def test_settle_counter_requires_consecutive_slow_windows() -> None:
    """★F92★ 정지는 연속 N창이다 — 느린 창 하나로는 릴리스가 서지 않고, 빠른 창이 끼면 0으로 돌아간다."""
    cfg = _cfg()
    speeds = [0.001, 0.001, 0.30, 0.001, 0.001, 0.001]
    count = torch.tensor([0])
    seen = []
    for v in speeds:
        count = settle_counter_update(count, torch.tensor([v]), cfg)
        seen.append(int(count[0]))
    assert seen == [1, 2, 0, 1, 2, 3]
    assert not bool(release_decision(torch.tensor([True]), torch.tensor([0.05]), torch.tensor([2]), cfg)[0])
    assert bool(release_decision(torch.tensor([True]), torch.tensor([0.05]), torch.tensor([3]), cfg)[0])
    # 실기 스칼라 경로도 같은 산술이다.
    c = 0
    for v in speeds:
        c = int(settle_counter_update(c, v, cfg))
    assert c == 3


def test_settle_steps_span_more_than_one_control_window() -> None:
    """★F92★ 단일 창(50 ms)은 운반 중 방향 전환 한 번으로 비가역 릴리스를 굳힌다 — 2창 이상이어야 한다."""
    cfg = load_release_supervisor_cfg()
    assert cfg.tcp_settle_steps >= 2


def test_edge_lockout_ignores_reversal_inside_pulse_window() -> None:
    """★F92★ 전이 뒤 잠금 창 안의 반전은 무시되고(직전 상태 유지), 창이 끝나면 다시 따른다."""
    cfg = _cfg()
    prev, since = False, cfg.gripper_edge_lockout_steps          # 리셋 상태 = 잠금 해제
    trace = []
    wants = [True, False, True, False, False, False, False, False, False]
    for w in wants:
        now, since = edge_lockout_update(prev, w, since, cfg)
        trace.append((bool(now), int(since)))
        prev = bool(now)
    # 스텝0 폐합 전이(since→0), 스텝1~6 반전 무시(since 0~5 = 잠금), 스텝7에 개방 전이.
    #   전이 간 최소 간격 = lockout+1 = 7스텝(350 ms) > 펄스 0.30 s — 펄스 종료 직후 스텝은 보수적으로 잠근다.
    assert [t[0] for t in trace] == [True, True, True, True, True, True, True, False, False]
    assert [t[1] for t in trace] == [0, 1, 2, 3, 4, 5, 6, 0, 1]


def test_edge_lockout_tensor_and_scalar_agree() -> None:
    """sim(텐서)과 실기(스칼라)가 같은 잠금 궤적을 낸다."""
    cfg = _cfg()
    wants = [True, False, True, True, False, True, False, False]
    prev_t = torch.tensor([False]); since_t = torch.tensor([cfg.gripper_edge_lockout_steps])
    prev_s, since_s = False, cfg.gripper_edge_lockout_steps
    for w in wants:
        prev_t, since_t = edge_lockout_update(prev_t, torch.tensor([w]), since_t, cfg)
        prev_s, since_s = edge_lockout_update(prev_s, w, since_s, cfg)
        assert bool(prev_t[0]) == bool(prev_s) and int(since_t[0]) == int(since_s)


def test_edge_lockout_covers_real_do_pulse() -> None:
    """★F92★ 잠금 길이 × 제어주기 ≥ 실기 DO 펄스 길이 — 펄스 중 반전(e-stop)이 sim에서 불가능해야 한다."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    cfg = load_release_supervisor_cfg()
    common = yaml.safe_load((root / "configs/common.yaml").read_text(encoding="utf-8"))
    dt = 1.0 / float(common["control"]["control_hz"])
    for name in ("real.example.yaml", "real.local.yaml"):
        path = root / "configs" / name
        if not path.exists():
            continue
        pulse_s = float(yaml.safe_load(path.read_text(encoding="utf-8"))["gripper"]["pulse_s"])
        assert cfg.gripper_edge_lockout_steps * dt >= pulse_s - 1e-9, (
            f"{name}: pulse_s {pulse_s} > lockout {cfg.gripper_edge_lockout_steps}×{dt}"
        )


def test_release_uses_no_object_pose_signal() -> None:
    """★계약 ①★ — 판정 함수의 인자에 큐브/객체 신호가 없어야 한다.

    큐브 pose를 인자로 받기 시작하면 실기에서 그 값을 만들 방법이 없다(D435i RGB-D만
    있고 검증된 pose 추정기가 없다). 시그니처를 계약으로 고정한다.
    """
    import inspect

    params = set(inspect.signature(release_decision).parameters)
    assert params == {"carried", "tcp_z", "settle_count", "cfg"}
    params_latch = set(inspect.signature(carry_latch_update).parameters)
    assert params_latch == {"carried", "gripper_closed", "tcp_rise", "cfg"}
    for name in ("object", "cube", "obj_pos", "target"):
        assert not any(name in p for p in params | params_latch)


def test_release_is_monotonic_when_latched_by_caller() -> None:
    """★계약 ②★ — 호출측의 `released |= decision` 패턴이 단조임을 확인한다."""
    cfg = _cfg()
    released = torch.tensor([False])
    carried = torch.tensor([True])
    # 조건 성립 → 릴리스
    released = released | release_decision(carried, torch.tensor([0.05]),
                                           torch.tensor([3]), cfg)
    assert bool(released[0])
    # 이후 팔이 다시 움직여 조건이 깨져도 릴리스는 유지된다.
    released = released | release_decision(carried, torch.tensor([0.05]),
                                           torch.tensor([0]), cfg)
    assert bool(released[0])


def test_numpy_scalars_work_for_real_deployment() -> None:
    """실기 rollout은 텐서가 아니라 스칼라로 부른다 — 같은 함수가 양쪽을 받아야 한다."""
    cfg = _cfg()
    carried = carry_latch_update(False, True, 0.02, cfg)
    assert carried is True or bool(carried)
    assert bool(release_decision(True, 0.05, 3, cfg))
    assert not bool(release_decision(True, 0.05, 1, cfg))



def test_grasp_allowed_is_height_constraint_only() -> None:
    """★F68★ 폐합은 정책이 정하고 수퍼바이저는 높이만 제약한다.

    F67은 폐합 시점을 탈취했다가 teacher 학습을 F66의 20% 수준으로 망가뜨렸다
    (grasp 0-250 구간 0.031 vs F66 0.797). 그 근거였던 '닫으면 유지된다'는 전제는
    사실이 아니었다 — binary_joint_actions.py는 매 스텝 부호로 재계산한다(latch 없음).
    """
    import inspect
    cfg = _cfg()
    # 인자는 TCP 높이 하나뿐 — 정지 조건도, 객체 신호도 없다(배포 가능성 + 학습성 보존).
    assert set(inspect.signature(grasp_allowed).parameters) == {"tcp_z", "cfg"}
    assert bool(grasp_allowed(torch.tensor([0.01]), cfg)[0])      # 파지 높이 → 허용
    assert not bool(grasp_allowed(torch.tensor([0.30]), cfg)[0])  # 높은 곳 → 금지


def test_action_term_uses_policy_close_under_height_constraint() -> None:
    """액션 항이 '정책의 닫기 명령 ∧ 높이 허용'으로 폐합하는지 소스로 고정한다."""
    from pathlib import Path
    act = (Path(__file__).resolve().parent.parent / "source" / "ur5e_binpick" / "ur5e_binpick"
           / "tasks" / "binpick" / "mdp" / "delayed_actions.py").read_text(encoding="utf-8")
    assert "policy_close = " in act and "grasp_allowed(" in act, (
        "폐합이 '정책 명령 ∧ 높이 제약' 형태가 아닙니다 — F67의 시점 탈취로 되돌아갔습니다"
    )

def _carry_latch_call_sites() -> list:
    """`carry_latch_update(...)` 호출을 소스 전수에서 AST로 찾아 반환한다."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sites = []
    for path in root.rglob("*.py"):
        parts = set(path.parts)
        if parts & {".git", "build", "__pycache__", "tests"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "carry_latch_update":
                sites.append((path.relative_to(root).as_posix(), node))
    return sites


def test_every_carry_latch_call_site_passes_a_relative_rise() -> None:
    """★F71 회귀★ — 호출부가 ★절대 TCP z★가 아니라 ★차이★를 넘기는지 소스에서 강제한다.

    [왜 단위 테스트로 못 잡았나] `test_carry_latch_uses_relative_rise_not_absolute_height`는
    함수의 ★의미★만 검증한다. 실제 사고는 호출부가 계약을 어긴 것이었다 — delayed_actions와
    프로브 모두 `tcp[:, 2]`(절대 높이)를 넘겼고, `z_at_close`는 할당·리셋만 되고 기록도
    사용도 되지 않는 죽은 상태였다. 함수 단위 테스트는 전부 통과하는데 시스템은 구 의미로
    동작했다 = F68-c는 문서·설정·README에만 존재했다.

    [영향] lift_rise_m=0.010을 절대 높이로 읽으면 폐합 허용 높이(<0.040)에서 닫는 순간
    latch가 굳어, release_decision이 ★파지 직후 느려지는 순간★ 성립한다 — 집은 자리에
    그대로 놓는다.

    그래서 이 테스트는 값이 아니라 ★배선의 형태★를 본다: 세 번째 인자는 뺄셈이어야 한다.
    """
    import ast

    sites = _carry_latch_call_sites()
    assert sites, "carry_latch_update 호출부를 찾지 못했다 — 탐색 경로를 확인할 것"
    offenders = []
    for where, node in sites:
        if len(node.args) < 3:
            offenders.append(f"{where}: 위치 인자가 3개 미만")
            continue
        rise = node.args[2]
        if not (isinstance(rise, ast.BinOp) and isinstance(rise.op, ast.Sub)):
            offenders.append(f"{where}: 3번째 인자가 차이(뺄셈)가 아니다 — {ast.dump(rise)[:80]}")
    assert not offenders, (
        "carry_latch_update의 tcp_rise는 ★폐합 시점 대비 상승량★이다. "
        "절대 높이를 넘기면 파지 직후 릴리스가 터진다: " + "; ".join(offenders)
    )
