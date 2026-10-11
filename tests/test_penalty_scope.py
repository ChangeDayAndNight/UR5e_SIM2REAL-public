"""★동작 페널티는 팔에만 걸린다 (F45, 2026-08-01)★ — 순수 정적 검사(README 검증 계약).

배경: '팔이 타겟 위에서 멈추지 않는다'(placed 병목)를 잡으려고 action_rate·joint_vel
      가중치를 업스트림 lift 태스크의 커리큘럼 종점 −1e-1로 올렸다. 정지 자체는
      완벽히 해결됐다 — 팔 관절속도 4.23 → ★0.043 rad/s★, 래치∧3cm→정지 전환율
      7.6% → ★79.5%★.

      그런데 ★릴리스가 죽었다★: 정지→placed 전환율 95.5% → ★3.3%★,
      래치∧릴리스 24.6% → 12.1%, placed 4.1% → 0.8%.

근본원인: 액션은 [팔 IK 6, ★그리퍼 1★] 레이아웃인데 업스트림 `action_rate_l2`는
      ★전 차원★의 (a_t − a_{t−1})²을 더한다 → 릴리스(그리퍼 값 반전)라는
      ★태스크 필수 행동에 직접 벌점★이 붙었다. `joint_vel_l2`도 asset_cfg가
      SceneEntityCfg("robot")이라 ★손가락 관절★을 포함했다.
      결과적으로 '큐브를 쥔 채 얼어붙기'가 최적해가 된다 — 잡고 있으면 grasp·lift·goal
      스트림이 계속 들어오고, 놓으면 벌점을 물면서 그 스트림까지 잃는다.

계약: ★동작 페널티는 팔 6축에만 걸린다★. 페널티의 목적은 팔의 부드러운 동작이지
      그리퍼 개폐 억제가 아니다(그리퍼는 이산 개폐라 저크 개념 자체가 부적절).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CFG = _ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "binpick_env_cfg.py"
_COMMON = _ROOT / "configs" / "common.yaml"


def _src() -> str:
    if not _CFG.is_file():
        pytest.skip("sim 확장(source/) 없음 — 배포 번들에서는 대조 불가")
    return _CFG.read_text(encoding="utf-8")


def test_action_rate_excludes_gripper_dim() -> None:
    """action_rate는 업스트림 전차원 판이 아니라 ★팔 한정★ 커스텀 항을 써야 한다."""
    src = _src()
    m = re.search(r"^\s+action_rate = RewTerm\(func=([\w.]+)", src, re.M)
    assert m, "action_rate RewTerm 정의를 찾지 못함"
    func = m.group(1)
    assert func != "mdp.action_rate_l2", (
        "action_rate가 업스트림 전차원 판을 쓰고 있습니다 — ★그리퍼 개폐(릴리스)에 벌점★이 붙어 "
        "정지→placed 전환율이 95.5%→3.3%로 붕괴한 이력이 있습니다(F45). 팔 한정 항을 쓸 것"
    )
    assert "arm" in func.lower(), f"팔 한정임이 이름에 드러나야 합니다: {func}"


def test_arm_action_rate_slices_only_arm_dims() -> None:
    """커스텀 항이 실제로 팔 차원만 슬라이싱하는가 (이름만 팔 한정이면 안 된다)."""
    src = _src()
    body = re.search(r"def arm_action_rate_l2\(env\)(?:.|\n)*?(?=\n@|\ndef |\Z)", src)
    assert body, "arm_action_rate_l2 정의를 찾지 못함"
    b = body.group(0)
    assert "_NUM_ARM_ACTIONS" in b and "[:, :_NUM_ARM_ACTIONS]" in b, (
        "팔 차원 슬라이싱이 없습니다 — 그리퍼 차원이 그대로 벌점에 포함됩니다"
    )
    # 업스트림과 동일한 커널(제곱합)이어야 한다 — 슬라이싱 외에는 바꾸지 않는다
    assert "torch.sum(torch.square(" in b, "업스트림 action_rate_l2와 다른 커널입니다"


def test_arm_action_count_matches_action_contract() -> None:
    """팔 액션 수가 관측/액션 계약(configs/common.yaml)과 일치하는가."""
    src = _src()
    assert "_NUM_ARM_ACTIONS: int = len(UR5E_ARM_JOINT_NAMES)" in src, (
        "팔 액션 수를 하드코딩하지 말 것 — UR5E_ARM_JOINT_NAMES에서 파생"
    )
    if not _COMMON.is_file():
        pytest.skip("configs/common.yaml 없음")
    m = re.search(r"^\s*num_arm_joints:\s*(\d+)", _COMMON.read_text(encoding="utf-8"), re.M)
    assert m, "common.yaml num_arm_joints 를 찾지 못함"
    from common.robot_spec import UR5E_ARM_JOINT_NAMES

    assert len(UR5E_ARM_JOINT_NAMES) == int(m.group(1)), (
        "팔 관절 수가 액션 계약과 어긋납니다 — 슬라이싱이 그리퍼를 잘못 자릅니다"
    )


def test_joint_vel_penalty_scoped_to_arm_joints() -> None:
    """joint_vel도 손가락 관절을 제외해야 한다 (파지/릴리스 동작에 벌점 금지)."""
    src = _src()
    m = re.search(r"^\s+joint_vel = RewTerm\((?:.|\n)*?\)\n", src, re.M)
    assert m, "joint_vel RewTerm 정의를 찾지 못함"
    body = m.group(0)
    assert "UR5E_ARM_JOINT_NAMES" in body, (
        'joint_vel의 asset_cfg가 팔로 한정되지 않았습니다 — SceneEntityCfg("robot")은 '
        "★손가락 관절까지★ 벌점 대상에 넣습니다(F45)"
    )


def test_gripper_joints_not_in_any_motion_penalty() -> None:
    """그리퍼 관절이 어떤 동작 페널티에도 등장하지 않는가 (역방향 확인)."""
    src = _src()
    for term in ("action_rate", "joint_vel"):
        m = re.search(rf"^\s+{term} = RewTerm\((?:.|\n)*?\)\n", src, re.M)
        assert m, f"{term} 정의를 찾지 못함"
        assert "GRIPPER_JOINT_NAMES" not in m.group(0), (
            f"{term} 페널티에 그리퍼 관절이 포함돼 있습니다"
        )


# ═════════════════════════════════════════════════════════════════════════════
# ★F46(2026-08-01)★ 그리퍼 액션 포화 잠김 방지
#
# 실측 결함: 타겟존에서 정책의 그리퍼 출력 중앙값이 ★−9.9★(p90 −8.5, 노이즈 std 0.37).
#   open(>0)을 명령한 에피소드가 ★0.6%★ — 표본추출조차 못 한다.
# 원인: BinaryJointPositionAction은 `actions < 0`으로 ★부호만★ 본다 → 크기에 복원력이 없다.
#   `ActionTermCfg.clip`은 raw 액션이 아니라 처리된 관절 명령에 걸려 이 드리프트를 못 막는다.
#   그리고 release_bias의 `a_open = clamp(action, 0, 1)`은 −10에서 값도 gradient도 0이라
#   ★한번 흘러가면 돌아올 길이 없다(비가역)★.
# ⇒ 릴리스가 학습되지 않은 진짜 이유. 기존 '릴리스'는 의도적 개방이 아니라 팔 요동으로
#   큐브를 놓친 것이었다(무접촉을 릴리스로 판정하므로).
# ═════════════════════════════════════════════════════════════════════════════


def test_gripper_dim_drift_is_structurally_prevented() -> None:
    """그리퍼 액션 표류(F46)가 ★구조적으로★ 막혀 있어야 한다.

    ★F67(2026-08-18) 계약 변경★
      원래 이 항은 "크기 페널티(gripper_action_l2)가 복원력을 준다"를 지켰다. F46 실측에서
      그리퍼 액션 중앙이 −9.9까지 흘러 개방을 ★표본추출조차 못 하는★ 사고가 있었기 때문이다.

      F67에서 폐합·개방 시점을 수퍼바이저가 정하면서 ★정책의 그리퍼 차원이 아예 사용되지
      않는다★. 사용되지 않는 차원은 아무리 흘러가도 환경에 영향이 없으므로 F46 사고는
      구조적으로 재발할 수 없고, 그 차원에 페널티를 유지하면 오히려 무의미한 gradient가 된다.

      그래서 이 테스트는 둘 중 ★하나★를 요구한다 — 원래 방어(음수 페널티)가 살아 있거나,
      그리퍼가 수퍼바이저 제어라 방어가 불필요하거나. 둘 다 없으면 F46이 재발한다.
    """
    src = _src()
    m = re.search(r"^\s+gripper_action = RewTerm\(func=gripper_action_l2, weight=([-\d.e]+)", src, re.M)
    assert m, "gripper_action RewTerm 배선이 없습니다"
    w = float(m.group(1))
    if w < 0:
        # 구 계약: 페널티가 복원력을 준다 — 크기가 파지를 깨지 않으면서 포화를 밀어내야 한다.
        dt = 1.0 / 20.0
        grasp_income = 2.0 * dt
        assert abs(w) * 1.5**2 * dt < 0.1 * grasp_income, f"가중치 {w}가 너무 큽니다"
        assert abs(w) * 10.0**2 * dt > grasp_income, f"가중치 {w}가 너무 작습니다"
        return
    # 신 계약: 페널티가 0이라면 그리퍼는 반드시 수퍼바이저가 제어해야 한다.
    sup = (_ROOT / "common" / "release_supervisor.py").read_text(encoding="utf-8")
    assert "def grasp_allowed(" in sup and "def release_decision(" in sup, (
        "그리퍼 페널티를 0으로 두려면 폐합·개방이 모두 수퍼바이저 primitive여야 합니다"
    )
    act = (_ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
           / "mdp" / "delayed_actions.py").read_text(encoding="utf-8")
    assert "grasp_allowed(" in act and "release_decision(" in act, (
        "액션 항이 수퍼바이저 판정을 사용하지 않습니다 — 그리퍼 차원이 방치됩니다(F46 재발)"
    )
    assert "closed_cmd = " in act and "policy_close" in act, (
        "그리퍼 명령이 수퍼바이저 상태기계에서 나오지 않습니다"
    )


def test_gripper_action_penalty_targets_only_gripper_dim() -> None:
    """팔 IK 델타는 크기가 의미를 가지므로 이 페널티에서 제외돼야 한다."""
    src = _src()
    body = re.search(r"def gripper_action_l2\(env\)(?:.|\n)*?(?=\n@|\ndef |\Z)", src)
    assert body, "gripper_action_l2 정의를 찾지 못함"
    b = body.group(0)
    assert "[:, _NUM_ARM_ACTIONS]" in b, (
        "그리퍼 차원만 인덱싱해야 합니다 — 팔 차원까지 크기를 벌주면 IK 이동이 억제됩니다"
    )
    assert "torch.square" in b


# ═════════════════════════════════════════════════════════════════════════════
# ★P7-2(2026-08-06)★ 능력 게이트 입력(자력 파지 EMA)은 보상 경로와 분리돼야 한다
#
# 감사에서 확인: `RewardManager.compute`는 `weight == 0.0`인 항을 ★건너뛴다★
#   (vendored reward_manager.py L145-148). 초판 구현은 EMA 갱신을 `grasped_gate`
#   (보상 함수) 안에 넣었으므로, grasp 가중치를 0으로 두는 실험을 하는 순간
#   EMA가 얼어붙어 ★능력 게이트가 영구 차단★된다 — 보상 설정이 커리큘럼을 조용히
#   망가뜨리는 숨은 결합이었다.
# ═════════════════════════════════════════════════════════════════════════════


def test_grasp_ema_not_updated_inside_reward_functions() -> None:
    """EMA 갱신이 보상 함수 안에 있으면 weight=0에서 조용히 죽는다."""
    src = _src()
    for fn in ("grasped_gate", "lift_grasped", "placed_released", "goal_distance_carry"):
        m = re.search(rf"def {fn}\((?:.|\n)*?(?=\ndef |\n@|\Z)", src)
        if not m:
            continue
        assert "_bp_grasp_ema" not in m.group(0), (
            f"{fn}(보상 함수) 안에서 _bp_grasp_ema를 갱신한다 — RewardManager는 weight==0인 항을 "
            "건너뛰므로(P7-2) 보상 설정 변경이 커리큘럼을 조용히 망가뜨린다. 전용 이벤트로 분리할 것."
        )


def test_grasp_ema_updated_by_dedicated_event() -> None:
    """전용 이벤트가 존재하고 EventCfg에 배선돼 있는가 (가중치와 무관하게 항상 실행)."""
    src = _src()
    assert "def _update_grasp_ema(" in src, "전용 EMA 갱신 함수가 없습니다"
    body = re.search(r"def _update_grasp_ema\((?:.|\n)*?(?=\ndef |\n@|\Z)", src)
    assert body and "_bp_grasp_ema" in body.group(0), "전용 함수가 EMA를 갱신하지 않습니다"
    assert '"interval"' in src, "EMA 이벤트가 interval 모드가 아닙니다(리셋 시에만 돌면 표본이 편향)"
    ev = re.search(r"class EventCfg:(?:.|\n)*?(?=\n@configclass|\n# ═)", src)
    assert ev and "grasp_ema" in ev.group(0), "EventCfg에 grasp_ema 항이 배선되지 않았습니다"


def test_grasp_ema_measures_self_grasp_only() -> None:
    """pregrasp 보조 env를 빼야 '자력' 파지 능력이 된다 (게이트의 의미)."""
    src = _src()
    body = re.search(r"def _update_grasp_ema\((?:.|\n)*?(?=\ndef |\n@|\Z)", src)
    assert body, "_update_grasp_ema 정의를 찾지 못함"
    assert "_real_pick_mask" in body.group(0), (
        "EMA가 pregrasp 보조 env를 포함한다 — '잡힌 채 시작'을 자력 파지로 세면 게이트가 "
        "실제 능력보다 일찍 열린다"
    )
