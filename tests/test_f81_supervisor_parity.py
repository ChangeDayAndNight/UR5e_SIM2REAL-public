"""F81 — 증류 리뷰(2026-09-03, 21건 생존)에서 나온 blocker/major 수정의 회귀 고정.

각 테스트 docstring이 '무엇이 사고였고 무엇이 계약인가'를 담는다. 값이 아니라
★배선의 형태★를 검사한다 — 이 파이프라인의 사고 대부분이 "함수는 옳은데 호출부가
계약을 어긴" 형태였다(F71·F79·F80).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_ENV_CFG = ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "binpick_env_cfg.py"
_OBS = ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "mdp" / "observations.py"
_EVENTS = ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "mdp" / "events_dr.py"
_DELAYED = ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "mdp" / "delayed_actions.py"
_ROLLOUT = ROOT / "common" / "rollout.py"
_RUN_TASK = ROOT / "Phase4" / "Run_Task.py"
_RUNNER_CFG = ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "agents" / "rsl_rl_ppo_cfg.py"
_VISION_POLICY = ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "agents" / "vision_policy.py"


def test_teacher_last_action_is_executable_normalized_not_raw():
    """★blocker #1★ teacher 관측의 last_action은 raw가 아니라 실행 가능 정규화 값이다.

    raw를 관측하면 ①무시 채널의 표류값(±7)이 자유 메모리로 되먹임되고 ②증류에서
    student 스케일과 갈라져 teacher가 분포 밖에서 질의된다(타깃 오염 33~86% 실측).
    """
    cfg_src = _ENV_CFG.read_text(encoding="utf-8")
    assert "func=bp_obs.executable_last_action" in cfg_src, (
        "PrivilegedObsCfg.last_action이 정규화 관측 함수를 쓰지 않는다"
    )
    # 특권 관측 절에 raw mdp.last_action ★배선★이 남아 있으면 안 된다(주석 언급은 무관).
    privileged_region = cfg_src.split("class PrivilegedObsCfg")[1].split("class ")[0]
    assert "ObsTerm(func=mdp.last_action)" not in privileged_region, (
        "특권 관측에 raw last_action 배선이 남아 있다"
    )

    obs_src = _OBS.read_text(encoding="utf-8")
    assert "def executable_last_action" in obs_src
    # Preserve the trained observation encoding, not the erroneous F80 claim
    # that observation clipping is equivalent to clipping executed IK commands.
    assert ".clamp(-1.0, 1.0)" in obs_src and "[:, 3:5] = 0.0" in obs_src
    vp_src = _VISION_POLICY.read_text(encoding="utf-8")
    assert "distillation_teacher_actions(super().evaluate(obs), _VERTICAL_GRASP_ONLY)" in vp_src
    assert "[..., 3:5] = 0.0" in vp_src


def test_curriculum_deadline_grants_eligibility_not_unbounded_drops():
    """★major #15★ deadline은 '자격'만 준다 — 갱신 주기는 항상 update_interval이다.

    구 조건 `(s - last >= update_interval or s >= deadline)`은 deadline 이후 호출마다
    max_drop을 깎아 0.9→0 급락을 만들었다(자기 계약 'Bounded updates' 위반).
    """
    src = _EVENTS.read_text(encoding="utf-8")
    assert "if eligible and s - last >= update_interval:" in src
    # 구 ★코드 라인★이 되살아나면 안 된다(주석의 이력 설명은 무관).
    assert "if eligible and (s - last >= update_interval or s >= deadline):" not in src, (
        "무제한 급락 분기가 되살아났다"
    )


def test_distill_env_uses_the_two_phase_carry_curriculum():
    """★F84 — 증류 커리큘럼은 2상 프로파일이다 (F81 단순 게이트 개방의 정정)★

    [사고 연쇄] F81이 min_grasp_ema=0.0으로 게이트만 열자 frac이 iter 600~2500에
    선형 소멸했고, carry 상태가 전체 스텝의 0.36%로 떨어져 student가 carry를 백지로
    남겼다(carry-probe 실측: cos(teacher,goal)=+0.54 vs cos(student,goal)=−0.15,
    latch 12.9% → zone 0.4% → placed 0). 성공 계보(08-21/08-27)는 deadline 버그의
    우연으로 iter 5000까지 0.9 고정 후 감쇠였다 — 그 2상 구조를 명시 재현한다.

    계약: ①EMA 게이트는 열려 있어야 하고(min_grasp_ema=0.0) ②0.9 고정 구간이
    iter 5000(=120k steps)까지 이어져야 하며 ③감쇠는 유계 자연 감쇠(43.2k steps)여야
    한다. 이 셋 중 하나라도 무너지면 carry 데이터 기근이 재발한다.
    """
    src = _ENV_CFG.read_text(encoding="utf-8")
    vision_region = src.split("class BinPickVisionEnvCfg")[1]
    assert "min_grasp_ema=0.0" in vision_region, (
        "증류 env가 pregrasp EMA 게이트를 열지 않는다 — student는 deadline까지 90% 보조에 갇힌다"
    )
    assert "warmup_steps=120_000" in vision_region, (
        "carry 마스터 구간(0.9 고정, iter 5000)이 사라졌다 — F84 carry 기근이 재발한다"
    )
    assert "decay_steps=43_200" in vision_region, (
        "감쇠 일정이 자연 감쇠 상한(max_drop 0.05×2400)과 어긋난다"
    )


def test_supervisor_state_is_never_seeded_at_reset():
    """★F83 — 시딩 금지 계약 (F81·F82의 철회를 고정한다)★

    [사고 연쇄] 리뷰가 'pregrasp 큐브가 1스텝에 강제 개방된다'를 결함으로 지목했고,
    F81이 closed_prev를, F82가 carried까지 물리 시딩해 '고쳤다'. 두 run 모두 placed가
    정확히 0이 됐다(0.0001/0.0004 vs 계보 0.804). 서명: time_out 98%·평균 99스텝·
    grasp/lift/goal은 계보보다 높음 = ★큐브를 든 채 배회하는 goal-farming 균형★.

    [진실] 성공한 모든 teacher는 시딩이 ★없어서★ may_close=False로 pregrasp 큐브가
    1스텝에 반드시 떨어졌고, 그 덕에 전 함대가 iter 0부터 '떨어진 큐브를 실제로 집는'
    연습을 했다(평균 에피소드 60스텝 = placed 종료). 강제 낙하가 커리큘럼 전체의
    하중을 받치고 있었다.

    시딩을 되살리려면 hold-farming 균형의 해소(호버 페널티/조기 종료 등)와 ★함께★
    설계하고 게이트 3000에서 placed·time_out·에피소드 길이를 대조해야 한다.
    그 전까지 이 테스트가 시딩의 재도입을 막는다.
    """
    src = _DELAYED.read_text(encoding="utf-8")
    assert "seed_pending" not in src, (
        "수퍼바이저 시딩이 재도입됐다 — F81/F82에서 placed를 0으로 만든 그 변경이다. "
        "docstring의 사고 연쇄를 읽고, hold-farming 해소 없이는 되살리지 말 것."
    )
    assert "find_joints" not in src, "리셋 시 물리 손가락 읽기(시딩 기전)가 재도입됐다"
    # 성공 동역학의 핵심(강제 낙하)을 만드는 게이트는 남아 있어야 한다.
    assert "grasp_allowed(tcp_pos[:, 2], _RELEASE_CFG) | st[\"closed_prev\"]" in src


def test_real_rollout_mirrors_the_sim_gripper_supervisor():
    """★blocker #10 — 실기에서 그리퍼가 큐브 근처도 아닌데 닫힌 사고의 수정★

    정책의 그리퍼 채널은 '항상 닫아라' 포화 신호다(중앙값 −1.6~−2.4). sim은 액션텀의
    수퍼바이저가 걸러 주지만 실기 루프에는 등가물이 없었다. rollout이 같은 함수·같은
    설정으로 같은 상태기계를 돌려야 한다.
    """
    src = _ROLLOUT.read_text(encoding="utf-8")
    for symbol in ("grasp_allowed", "carry_latch_update", "release_decision",
                   "load_release_supervisor_cfg"):
        assert symbol in src, f"rollout에 수퍼바이저 {symbol}가 없다"
    # 상태기계 순서 계약: 게이트 → 기준높이 기록 → latch → 릴리스 → 팔 정지.
    assert src.index("grasp_allowed(") < src.index("carry_latch_update(")
    assert src.index("carry_latch_update(") < src.index("release_decision(")
    # 릴리스 후 팔 정지(실측 63.7% vs 11.7%의 그 처방).
    assert "np.zeros_like" in src.split("release_decision(")[1][:2000]


def test_run_task_hands_policy_runner_the_unscaled_action_cfg():
    """★major #11★ --action-scale은 실행(mapper) 전용이다 — PolicyRunner에 축소 cfg를
    넘기면 obs_spec cfg_hash가 어긋나 정책 로드가 거부된다(실기 실증)."""
    tree = ast.parse(_RUN_TASK.read_text(encoding="utf-8"))
    runner_args, mapper_args = None, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", "")
            if name == "PolicyRunner":
                runner_args = [getattr(a, "id", "") for a in node.args]
            elif name == "ActionMapper":
                mapper_args = [getattr(a, "id", "") for a in node.args]
    assert runner_args is not None and "action_cfg" in runner_args, (
        f"PolicyRunner가 원본 action_cfg를 받지 않는다: {runner_args}"
    )
    assert "scaled_action_cfg" not in (runner_args or []), "축소 cfg가 해시 검증 경로에 들어간다"
    assert mapper_args == ["scaled_action_cfg"], (
        f"ActionMapper는 축소 cfg를 받아야 한다: {mapper_args}"
    )


def test_distillation_clips_gradients_like_the_teacher():
    """★major #7★ 증류에 grad clip이 없었고 손실은 12청크 sum 누적이다 — teacher와
    같은 1.0으로 클립한다."""
    src = _RUNNER_CFG.read_text(encoding="utf-8")
    distill_region = src.split("class BinPickDistillRunnerCfg")[1]
    assert "max_grad_norm=1.0" in distill_region


def test_aux_and_default_student_combo_fails_fast():
    """★minor #3/#8★ BINPICK_AUX=1 + 기본 CNN student는 '수 분 뒤 첫 update에서' 죽는
    조합이다 — 기동 시점에 명시적으로 거부해야 야간 무인 run을 잃지 않는다."""
    src = _RUNNER_CFG.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_require_valid_student_aux_combo"),
        None,
    )
    assert fn is not None, "aux/student 조합 가드 함수가 없다"
    # 함수만 떼어 실제로 실행 — 무효 조합이 RuntimeError를 내는지.
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<guard>", "exec"),
         {"os": __import__("os")}, namespace)
    guard = namespace["_require_valid_student_aux_combo"]
    import os as _os
    old = _os.environ.get("BINPICK_AUX")
    try:
        _os.environ["BINPICK_AUX"] = "1"
        with pytest.raises(RuntimeError, match="BINPICK_STUDENT=resnet"):
            guard("VisionStudentTeacher")
        assert guard("VisionStudentTeacherResNet") == "VisionStudentTeacherResNet"
        _os.environ["BINPICK_AUX"] = "0"
        assert guard("VisionStudentTeacher") == "VisionStudentTeacher"
    finally:
        if old is None:
            _os.environ.pop("BINPICK_AUX", None)
        else:
            _os.environ["BINPICK_AUX"] = old


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
