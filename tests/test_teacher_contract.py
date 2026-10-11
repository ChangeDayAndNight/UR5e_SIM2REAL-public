"""Static contracts for the scratch Teacher and strict placement definition."""

from pathlib import Path
import importlib.util

import torch
from common.teacher_contract import (
    TEACHER_ACTION_DIM,
    TEACHER_OBJECT_POS_SLICE,
    TEACHER_OBS_DIM,
    TEACHER_OBS_LAYOUT,
)


ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/binpick_env_cfg.py"
OBS = ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/mdp/observations.py"
ACT = ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/mdp/delayed_actions.py"
VISION = ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/agents/vision_policy.py"
PHASE2_TRAIN = ROOT / "Phase2/VisionPick_Train.py"


def test_teacher_actor_observes_only_deployable_signals():
    """★Teacher v2(2026-09-18)★ actor 관측은 실기에서 만들 수 있는 신호만 갖는다.

    실제 손가락 위치·물체 속도·접촉력은 critic 전용(TeacherCriticObsCfg)으로만 남는다.
    큐브 위치는 참값이 아니라 perception_model의 추정치이고, 그리퍼는 명령 상태다.
    """
    cfg = CFG.read_text(encoding="utf-8")
    obs = OBS.read_text(encoding="utf-8")
    actor = cfg[cfg.index("class PrivilegedObsCfg"):cfg.index("class TeacherCriticObsCfg")]
    assert "gripper_pos" not in actor and "contact_forces" not in actor and "object_vel" not in actor
    assert "object_pos = ObsTerm(func=bp_perc.object_pos_estimate)" in actor
    assert "held = ObsTerm(func=bp_perc.held_estimate)" in actor
    assert "gripper_command_state = ObsTerm(func=bp_perc.gripper_command_state_obs)" in actor
    critic = cfg[cfg.index("class TeacherCriticObsCfg"):cfg.index("class TeacherObservationsCfg")]
    assert "contact_forces = ObsTerm(func=bp_obs.finger_contact_forces)" in critic
    assert '"left_finger_contact", "right_finger_contact"' in obs
    assert 'SceneEntityCfg("finger_contacts")' not in obs


def test_teacher_flat_layout_is_34d_and_matches_cfg_order():
    assert TEACHER_OBS_DIM == 34
    assert TEACHER_ACTION_DIM == 7
    assert TEACHER_OBJECT_POS_SLICE == slice(23, 26)
    cfg = CFG.read_text(encoding="utf-8")
    body = cfg[cfg.index("class PrivilegedObsCfg"):cfg.index("class TeacherCriticObsCfg")]
    positions = [body.index(f"    {name} = ObsTerm") for name, _width in TEACHER_OBS_LAYOUT]
    assert positions == sorted(positions), "PrivilegedObsCfg 순서와 pure Teacher 계약이 다름"


def test_sim_perception_model_and_real_builder_share_one_rule_source():
    """sim 인식 모델과 실기 조립기가 같은 latch 상수·합성 규칙(단일 소스)을 쓴다."""
    perc = (ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/mdp/perception_model.py").read_text(encoding="utf-8")
    real = (ROOT / "common/teacher_obs.py").read_text(encoding="utf-8")
    for name in ("HELD_RADIUS_M", "HELD_OFFSET_FROM_TCP_M", "GRIPPER_CLOSED_THRESHOLD"):
        assert name in perc and name in real, f"{name}이 한쪽에서만 쓰인다 — 규칙이 갈라진다"
    cfg = CFG.read_text(encoding="utf-8")
    assert "self.events.perception_resample = bp_perc.make_perception_resample_event()" in cfg
    dr = (ROOT / "configs/dr.yaml").read_text(encoding="utf-8")
    assert "perception:" in dr and "blind_dist_m" in dr and "detect_prob" in dr


def test_sim_action_clamp_matches_deploy_safety_limits():
    """★Teacher v2★ sim 액션텀의 스텝당 상한 = safety.yaml limits × dt, 스케일 = 그 상한."""
    import yaml as _yaml

    cfg = CFG.read_text(encoding="utf-8")
    assert 'tcp_step_max_m=float(_SAFETY_LIMITS["tcp_linear_vel_max_m_s"]) / _CONTROL_HZ' in cfg
    assert 'tcp_rot_step_max_rad=float(_SAFETY_LIMITS["tcp_angular_vel_max_rad_s"]) / _CONTROL_HZ' in cfg
    common = _yaml.safe_load((ROOT / "configs/common.yaml").read_text(encoding="utf-8"))
    safety = _yaml.safe_load((ROOT / "configs/safety.yaml").read_text(encoding="utf-8"))
    dt = 1.0 / float(common["control"]["control_hz"])
    assert float(common["action"]["task_pos_scale_m"]) <= safety["limits"]["tcp_linear_vel_max_m_s"] * dt + 1e-12
    assert float(common["action"]["task_rot_scale_rad"]) <= safety["limits"]["tcp_angular_vel_max_rad_s"] * dt + 1e-12
    act = ACT.read_text(encoding="utf-8")
    assert "tcp_step_max_m" in act and "tcp_rot_step_max_rad" in act


def test_phase2_aux_uses_current_teacher_object_slice():
    src = VISION.read_text(encoding="utf-8")
    assert "self.aux_slice = TEACHER_OBJECT_POS_SLICE" in src
    assert "self.aux_slice = slice(22, 25)" not in src


def _synthetic_teacher_ckpt(path: Path, obs_dim: int) -> None:
    dims = [obs_dim, 256, 128, 64, 7]
    state = {}
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layer = 2 * index
        state[f"actor.{layer}.weight"] = torch.zeros(out_dim, in_dim)
        state[f"actor.{layer}.bias"] = torch.zeros(out_dim)
    torch.save({"model_state_dict": state}, path)


def test_phase2_teacher_preflight_accepts_42d_and_rejects_legacy_39d(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("vision_pick_train_contract", PHASE2_TRAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = tmp_path / "teacher_34.pt"
    legacy = tmp_path / "teacher_42.pt"
    _synthetic_teacher_ckpt(current, 34)
    _synthetic_teacher_ckpt(legacy, 42)
    assert module._verify_teacher_ckpt(str(current)) is True
    assert module._verify_teacher_ckpt(str(legacy)) is False


def test_placed_release_gate_is_contact_based_not_joint_tolerance():
    """★F50(2026-08-09) 회귀 고정★ — placed 성공 게이트에서 그리퍼 관절 허용대를 배제한다.

    2026-08-07 커밋 b6da816이 `_gripper_open_mask`(|q−0|≤3mm)를 성공 AND에 넣었고,
    타겟존에서 손가락이 접촉에 막혀 열리지 않아 20k run 내내 placed가 정확히 0이었다.
    문헌 표준(ManiSkill3 StackCube)은 릴리스 게이트를 ★접촉 기반 ¬is_grasping★으로 두고
    관절 개도는 dense reward로 준다 — 이 테스트가 그 구조로 되돌아가는 것을 막는다.
    """
    src = CFG.read_text(encoding="utf-8")
    body = src[src.index("def placed_success_mask("):src.index("def placed_released(")]
    assert "_any_finger_contact(env, contact_n)" in body, "릴리스 게이트(¬접촉)가 사라졌다"
    assert "_gripper_open_mask(env)" not in body, (
        "관절 허용대가 성공 게이트로 되돌아왔다 — F50 재발 (개도는 보상으로 줄 것)"
    )
    assert "streak >= PLACED_MIN_CONSECUTIVE_STEPS" in src


def test_release_reward_uses_actual_gripper_opening():
    """★사용자 지시(2026-08-09) + ManiSkill3 선례★ — 릴리스 보상이 실제 관절 개도를 반영한다.

    구 구현은 ★명령★(action[-1])만 봐서, 명령은 완전 개방인데 손가락이 접촉에 막혀
    실제로는 안 벌어진 상태를 만점으로 지급했다(F50 실측: q_tgt=0인데 q=16.8mm 정체).
    """
    src = CFG.read_text(encoding="utf-8")
    body = src[src.index("def release_bias_bootstrap("):src.index("def pregrasp_align(")]
    assert "joint_pos" in body, "릴리스 보상이 실제 관절 상태를 읽지 않는다"
    assert "RELEASE_STATE_WEIGHT" in body, "상태항 비중 상수가 적용되지 않았다"
    # 명령항은 반드시 살아 있어야 한다 — 상태 전용이면 '명령 없이 벌어진 상태'까지 지급된다.
    # ★F57(2026-08-11)★ 구 검사는 sigmoid 램프 문자열을 직접 요구했으나, 그 램프가
    #   ★닫힘 구간 지급(= farming 4연속 재발)의 원인★이라 은퇴시켰다. 이제 명령항은
    #   `open_gain(action, ...)`이 담당한다 — 액션이 인자로 들어가는지를 검사한다.
    assert "open_gain(" in body, "지급 계수가 common/release_gate.open_gain 을 경유하지 않는다"
    assert "env.action_manager.action[:, -1]" in body, "명령항(액션)이 지급식에 들어가지 않는다"
    weight = float(
        src.split("RELEASE_STATE_WEIGHT: float = ")[1].split("\n")[0].strip()
    )
    assert 0.0 < weight < 1.0, "상태 전용(1.0)은 폐합 개체의 gradient를 죽인다 — F46/F49 재발"


def test_action_delay_covers_arm_and_gripper():
    src = ACT.read_text(encoding="utf-8")
    assert "DelayedDifferentialInverseKinematicsAction" in src
    assert "DelayedBinaryJointPositionAction" in src
    assert "_bp_action_delay_lags" in src


def test_env_cfg_has_no_undefined_module_constants():
    """★F50 후속(2026-08-09) 회귀 가드★ — 보상 함수가 쓰는 모듈 상수가 실제로 정의돼 있는가.

    `release_bias_bootstrap`에 `GRIPPER_CLOSE_M`을 쓰면서 import를 빠뜨려, 문법·정적
    테스트는 전부 통과했는데 ★첫 보상 계산에서 NameError★로 죽었다. isaaclab을 import할 수
    없는 순수 pytest에서도 이 부류를 잡으려고, AST로 '함수가 참조하는 대문자 전역 이름'이
    모듈 최상위에 정의/임포트돼 있는지 대조한다.
    """
    import ast
    import builtins

    tree = ast.parse(CFG.read_text(encoding="utf-8"))
    defined: set[str] = set(dir(builtins))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)

    missing: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        local = {a.arg for a in node.args.args + node.args.kwonlyargs}
        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                local.update(t.id for t in sub.targets if isinstance(t, ast.Name))
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)) and isinstance(sub.target, ast.Name):
                local.add(sub.target.id)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                local.add(sub.id)
        for sub in ast.walk(node):
            # 대문자 상수만 본다 — 지역 변수·속성 접근은 이 검사의 대상이 아니다.
            if (
                isinstance(sub, ast.Name)
                and isinstance(sub.ctx, ast.Load)
                and sub.id.isupper()
                and sub.id not in local
                and sub.id not in defined
            ):
                missing.append(f"{node.name}() → {sub.id}")

    assert not missing, "정의되지 않은 모듈 상수 참조(런타임 NameError): " + ", ".join(sorted(set(missing)))

def test_release_state_term_cannot_farm_by_holding():
    """★F52(2026-08-10) 회귀 고정★ — 상태항이 '쥐고 버티기'를 보상하면 안 된다.

    50mm 큐브를 물면 손가락이 q≈0.012에서 버텨 개도 비율이 0.4쯤 된다. 상태항을 ★가산★으로
    넣었더니 그리퍼를 깊게 닫은 채로도 a_open≈0.20이 지급됐다(타겟존 open 명령 27.5%→2.13%).
    상태항은 명령항을 ★곱으로 변조★해야 하고, 스스로 지급해서는 안 된다.

    ★F64(2026-08-13) 판정 갱신★: "닫힘 지급 = 0"은 farming을 막는 조건이 ★아니었다★ —
    그 성질이 오히려 릴리스 학습을 죽였다(A/B: |a|rms 0.588→1.850, RB 7.99→0.05).
    farming을 막는 진짜 성질은 ★쥐고 버티기가 놓기보다 압도적으로 불리한 것★이다.
    상태항은 여전히 ★곱으로 변조만★ 하고 스스로 지급하지 않는다(F52의 가산 구현 금지).
    """
    import torch

    from common.release_gate import open_gain

    body = CFG.read_text(encoding="utf-8")
    body = body[body.index("def release_bias_bootstrap("):body.index("def pregrasp_align(")]
    assert "a_state" in body and "open_gain(" in body, "상태항이 지급 계수에 반영되지 않는다"

    # ① ★무엇이 farming을 막는가 (F64 정정)★
    #   구 테스트는 "닫힘 지급 = 0"을 요구했지만, ★placed ≥ 20을 낸 5개 run 전부★가
    #   닫힘 구간에 지급하는 sigmoid를 썼다. 즉 그 성질은 성공의 조건이 아니었고,
    #   오히려 그것을 강제한 F57 이후 6개 run이 전멸했다(RB ≤ 0.10).
    #   farming을 막는 것은 ★예산이 지급액만큼 깎이는 것★이다 — 버티며 번 만큼
    #   정작 놓을 때 쓸 예산이 사라지므로, '공짜 무한 스트림'이 성립하지 않는다.
    #   (구 구조에서는 버티기가 예산을 ★0★ 소모해 무한히 벌 수 있었다 = F52/F53/F56의 실체)
    from common.release_gate import grant_and_consume

    EP = 100
    def _collect(action: float, state: float) -> float:
        bud, tot = torch.tensor([20.0]), 0.0
        for _ in range(EP):
            g, bud = grant_and_consume(
                open_gain(torch.tensor([action]), torch.tensor([state]), 0.3, 0.5), bud)
            tot += float(g)
        return tot

    hover = _collect(-0.301, 0.4)        # F56이 실측한 farming 액션에서 버티기
    assert hover <= 20.0 + 1e-6, "예산 상한이 지켜지지 않는다 — 무한 스트림"
    # ★핵심★ 버틴 만큼 예산이 사라져, 그 뒤 실제 릴리스로 벌 수 있는 몫이 줄어든다
    bud_after = 20.0 - hover
    assert bud_after < 20.0 * 0.5, (
        f"버티기 100스텝 뒤에도 예산이 {bud_after:.1f} 남는다 — 버티기가 사실상 공짜다(F52 재발)"
    )
    # 그리고 실제 개방은 같은 예산을 ★훨씬 빨리★ 수확한다(버티기를 선택할 이유가 없다)
    steps_hold = 20.0 / float(open_gain(torch.tensor([-0.301]), torch.tensor([0.4]), 0.3, 0.5))
    steps_open = 20.0 / float(open_gain(torch.tensor([1.0]), torch.tensor([1.0]), 0.3, 0.5))
    assert steps_open * 5 < steps_hold, (
        f"개방 {steps_open:.0f}스텝 vs 버티기 {steps_hold:.0f}스텝 — 개방 우위가 5배 미만이다"
    )

    # ② 상태항은 변조만 한다 — 배수가 [1−W, 1]을 벗어나지 않는다
    a = torch.tensor([-1.0, 0.0, 1.0])
    base = open_gain(a, torch.ones_like(a), 0.3, 0.5)
    for st in (0.0, 0.4, 1.0):
        r = open_gain(a, torch.full_like(a, st), 0.3, 0.5) / base
        assert float(r.min()) >= 0.5 - 1e-9 and float(r.max()) <= 1.0 + 1e-9, (
            f"상태항이 변조 범위 [0.5, 1]을 벗어났다 (state={st}) — 가산 구현이면 F52 재발"
        )


def test_release_payout_and_budget_share_one_predicate():
    """★F53·F56(2026-08-10~11) 회귀 고정★ — '지급 > 0 인데 예산은 안 깎이는' 영역이 없어야 한다.

    이 결함은 ★네 번★ 재발했다(F52 → F53 → F56 → F56의 수정). 마지막이 특히 뼈아프다:
    F56에서 임계를 0.25 → 0.5로 올린 것은 문제의 영역을 (a_cmd<0.25)에서 ★(a_cmd<0.5)로
    넓힌★ 것이었고, 정책은 즉시 그 안(액션 −0.084)에 눌러앉아 또 open 명령 0.0%가 나왔다.

    ⚠️★구 테스트가 이것을 못 잡은 이유★(중요): 허용치를 ★검사 대상 상수 자신★으로
      표현했다(`worst <= att`). 임계를 올리면 제외 구간도 넓어지고 허용치도 같이 커져
      ★언제나 통과★하는 자기충족 검사였다. ⇒ 이제 상수와 무관한 ★구조★를 검사한다.
    """
    src = CFG.read_text(encoding="utf-8")
    live = chr(10).join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "RELEASE_OPEN_ATTEMPT" not in live, "별도 임계 상수 부활 — 어긋날 두 양이 다시 생겼다"

    body = src[src.index("def release_bias_bootstrap("):src.index("def pregrasp_align(")]
    code = chr(10).join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    # ★F64★ 소모량 = 지급액. 판정할 술어가 없으므로 어긋날 두 양도 없다.
    assert "grant, bud = grant_and_consume(pay, bud)" in code, (
        "예산 소모가 지급액에 묶여 있지 않다 — 지급 영역과 소모 영역이 어긋난다(F53/F56)"
    )
    # 수치로도 확인한다(소스 문자열만 믿지 않는다 — 구 테스트가 그래서 못 잡았다)
    import torch

    from common.release_gate import grant_and_consume, open_gain

    a = torch.arange(-500, 501, dtype=torch.float64) / 100.0
    pay = open_gain(a, torch.full_like(a, 0.4), 0.3, 0.5)
    bud = torch.full_like(pay, 20.0)
    grant, new_bud = grant_and_consume(pay, bud)
    spent = bud - new_bud
    exploitable = (grant > 0.0) & (spent <= 0.0) & (grant * 200 >= 20.0 * 1e-6)
    assert not bool(exploitable.any()), (
        f"소모 없이 착취 가능한 액션이 있다: {a[exploitable][:3].tolist()}"
    )


def test_training_placed_matches_the_five_successful_runs():
    """★F62(2026-08-12) 회귀 고정★ — 학습 보상용 placed 판정 = ★5/5 성공 run과 동일★.

    [증거] 학습측 `Episode_Reward/placed`(eval 프로토콜 무관) 18개 run 전수:
      max_speed 0.10 → 5개 run 전부 20.4~23.3 (성공) / 0.02 → 13개 run 전부 0.0000~0.0025.
      예외 0건. 이 상수를 다시 조이면 릴리스가 ★표집되지 않아★ 학습이 시작조차 못 한다
      (§9-5-1: 항을 고치기 전에 gradient가 흐르는지부터 재라).
    [분리 원칙] 배포 등급 엄격 판정은 STRICT_* 상수로 ★평가 보고 전용★이다 —
      '학습 신호로서 표집 가능한가'와 '성공으로 인정할 만한가'는 다른 축이며,
      한 상수가 둘을 겸하면 하나를 고칠 때 다른 하나가 조용히 망가진다(이번 사고의 본질).
    """
    src = CFG.read_text(encoding="utf-8")

    import re as _re

    # ⚠️★행 시작 앵커 필수★ — `STRICT_PLACED_MAX_SPEED: float = ` 는 `PLACED_MAX_SPEED: float = `를
    #   부분문자열로 포함한다. str.split로 찾으면 STRICT 쪽을 먼저 잡아 ★엉뚱한 값을 검사★한다
    #   (실제로 이 테스트가 그렇게 깨졌다 — 상수 이름에 접두를 붙일 때 흔한 함정).
    def _num(name: str) -> str:
        m = _re.search(rf"^{name}\s*:\s*\w+\s*=\s*([\d.]+)", src, _re.M)
        assert m, f"상수 {name} 정의를 찾지 못함(중복·오타 확인)"
        return m.group(1)

    def _c(name: str) -> float:
        return float(_num(name))

    def _i(name: str) -> int:
        return int(float(_num(name)))

    assert _c("PLACED_MAX_SPEED") >= 0.10, (
        "학습 보상용 속도 임계가 0.10 미만이다 — 13개 run이 전부 placed 0으로 끝난 구간이다"
    )
    assert _i("PLACED_MIN_CONSECUTIVE_STEPS") == 1, "성공 run은 전부 1스텝이었다"
    assert _c("PLACED_SURFACE_TOL_M") == 0.0, "표면 조건은 학습 마스크에서 비활성이어야 한다"

    # 배포 등급 상수는 ★보존★돼야 한다(보고의 정직성 — 95% 주장 시 기준 명시용)
    assert _c("STRICT_PLACED_MAX_SPEED") == 0.02
    assert _c("STRICT_PLACED_SURFACE_TOL_M") == 0.005
    assert _i("STRICT_PLACED_MIN_CONSECUTIVE_STEPS") == 10

    # release_land는 placed와 완전 중복이므로 RewTerm으로 등록돼 있으면 안 된다
    live = chr(10).join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "release_land = RewTerm(" not in live, (
        "release_land가 살아 있다 — placed와 같은 술어라 같은 사건에 100을 지급해 "
        "성공 run(50)과 다른 구성이 된다"
    )


def test_grasp_gated_terms_and_release_terms_are_mutually_exclusive():
    """★F61(2026-08-12) 회귀 고정★ — 파지 게이트 항과 릴리스 항은 ★한 스텝에서 동시 지급 불가★.

    [사고] `release_land` 가중치를 정당화한 주석이 "착지 시 release_land + ★goal 16★ + ..."로
      계산했는데, goal의 게이트는 `_grasped_mask(force_n)`이고 릴리스 술어는
      `~_any_finger_contact(contact_n)`다. contact_n ≤ force_n이면 두 술어는 ★배타★이므로
      없는 16을 더한 셈이었다. (F62에서 release_land는 제거됐고 `placed`가 그 역할을 한다.)

    [이 테스트가 지키는 것] 임계 상수의 ★대소 관계★. 이것이 깨지면 '파지 중'과 '손 뗌'이
      동시에 참인 구간이 생겨 파지 스트림과 릴리스 보상이 함께 지급되는 farming 경로가 열린다.
    """
    src = CFG.read_text(encoding="utf-8")

    def _param(term: str, key: str) -> float:
        seg = src.split(f"func={term},")[1][:400]
        return float(seg.split(f'"{key}":')[1].split(",")[0].strip())

    force_n = _param("goal_distance_carry", "force_n")            # 파지 판정 임계
    contact_n = _param("placed_released", "contact_n")            # 릴리스 판정 임계
    assert contact_n <= force_n, (
        f"contact_n({contact_n}) > force_n({force_n}) — '손 뗌'과 '파지 중'이 동시에 참인 구간이 "
        "생긴다. 파지 스트림(grasp/lift/goal)과 릴리스 보상이 함께 지급돼 farming 경로가 열린다."
    )
    # 폐기된 산술이 주석에 남아 다음 가중치 결정을 오염시키지 않도록 고정
    assert "release_land 20 + goal 16" not in src, "폐기된 산술(goal 16 동시 계상)이 남아 있다"


def test_env_cfg_has_no_orphan_name_expressions():
    """★F62(2026-08-12) 회귀 고정★ — 모듈 최상위에 ★이름만 있는 식★이 있으면 안 된다.

    [사고] 편집 중 `STRICT_# 주석...` 형태로 상수 접두가 주석 줄에 붙어 버렸다.
      이것은 ★구문상 완전히 유효★하다(이름 식 + 주석). `ast.parse`도 통과하고,
      이 파일의 다른 테스트들은 전부 ★텍스트 파싱★이라 역시 통과했다.
      그런데 실제 import에서는 `NameError: name 'STRICT_' is not defined`로 죽는다 —
      25k run이 기동 9초 만에 죽고서야 드러났다.
    [교훈] 이 저장소의 env cfg 테스트는 isaaclab 의존 때문에 import를 못 하므로
      ★'구문 OK'가 'import OK'를 뜻하지 않는다★. AST 수준에서 잡을 수 있는 것은 잡는다.
    """
    import ast

    tree = ast.parse(CFG.read_text(encoding="utf-8"))
    orphans = [
        (node.lineno, node.value.id)
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Name)
    ]
    assert not orphans, (
        "모듈 최상위에 이름만 있는 식이 있다(import 시 NameError) — "
        + ", ".join(f"line {ln}: {nm}" for ln, nm in orphans)
    )


def test_success_termination_bonus_exceeds_forfeited_return():
    """★F63(2026-08-13) 회귀 고정★ — 조기 종료로 포기하는 잔여 리턴 < 종단 보너스.

    [왜 필수인가] rsl_rl은 ★time_out에만 부트스트랩★한다(ppo.py: `rewards += gamma * values
      * time_outs`). `terminated`로 끝나면 잔여 가치가 0으로 잘린다 — 보너스가 부족하면
      ★성공하는 것이 손해★가 되어 정책이 성공을 적극적으로 회피한다(종료 도입의 전형적 함정).
    [수치] 호버 지속 가치 ≈ 1.30 reward/step(실측 Train/mean_reward ≈ 113/100스텝 기반,
      호버 국면은 그보다 높다) × 할인 지평 (1−γ^n)/(1−γ). n=75(래치∧tol 체류 실측 중앙).
    """
    import re as _re

    src = CFG.read_text(encoding="utf-8")
    agent = (
        ROOT / "source/ur5e_binpick/ur5e_binpick/tasks/binpick/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")

    gamma = float(_re.search(r"gamma\s*=\s*([\d.]+)", agent).group(1))
    w = float(src.split("func=placed_success_bonus,")[1].split("weight=")[1].split(",")[0])
    dt = 1.0 / float(
        __import__("yaml").safe_load(
            (ROOT / "configs/common.yaml").read_text(encoding="utf-8")
        )["control"]["control_hz"]
    )
    bonus = w * dt

    # ★F92(2026-09-20) 산식 정정(리뷰 RT-2)★ — time_out은 ★부트스트랩★되므로(위 docstring)
    #   호버 가치는 유한합이 아니라 r_hover/(1−γ)다. r_hover는 실측 평균(1.30)이 아니라 타겟 위
    #   저공 호버가 릴리스 없이 받는 ★가중치 합 상한★: reach 1 + grasp 2 + lift 5 + goal 16 +
    #   goal_fine 6 = 30/step × dt. (가중치를 올리면 이 상수도 함께 올릴 것.)
    hover_units_per_step = 30.0
    hover_per_step = hover_units_per_step * dt
    forfeited = hover_per_step / (1.0 - gamma)
    assert bonus > forfeited * 1.2, (
        f"종단 보너스 {bonus:.1f} 이 포기 리턴 {forfeited:.1f}의 1.2배 이하다 — "
        "성공이 손해가 되어 정책이 종료를 회피한다(F63)"
    )


def test_success_termination_requires_multi_step_confirmation():
    """★F63 회귀 고정★ — 종료·보너스는 ★연속 확정★을 요구해야 한다(blip farming 차단).

    운반 중 접촉이 한두 프레임 끊기는 지터만으로 에피소드가 끝나고 큰 보너스를 받으면,
    정책은 '타겟 근처에서 순간적으로 놓은 척하기'를 학습한다(F59가 그 지터를 실측했다).
    반대로 너무 길면 실측 창(릴리스 후 4~6스텝)을 넘겨 영영 성립하지 않는다.
    """
    src = CFG.read_text(encoding="utf-8")
    n = int(src.split("PLACED_DONE_CONSECUTIVE_STEPS: int = ")[1].splitlines()[0].strip())
    assert 2 <= n <= 5, f"종료 확정 스텝 {n} — 2~5 범위여야 한다(blip 차단 ↔ 실측 창 4~6스텝)"

    body = src[src.index("def _placed_done_mask("):src.index("def placed_success_done(")]
    assert "_real_pick_mask(env)" in body, "pregrasp env가 공짜 종료·보너스를 받으면 안 된다"
    assert "common_step_counter" in body, (
        "스텝당 1회 갱신 가드가 없다 — 종료 매니저와 보상 매니저가 같은 스텝에 둘 다 호출하므로 "
        "카운터가 2배로 뛴다(manager_based_rl_env: 종료 → 보상 순)"
    )
