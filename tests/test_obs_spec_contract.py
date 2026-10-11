"""obs_spec 계약 ★부정 테스트★ — 불일치 시 '기동 거부'가 실제로 발화하는가 (순수, §6①).

배경: README 배포 계약은 "obs spec 불일치 시 기동 거부가 정상 동작"이라고 규정한다.
      이 부정 테스트가 실제 fail-closed 경로를 고정한다.

여기서 고정하는 계약:
  ① 각 항목(obs_dim/action_dim/layout/cfg_hash) 불일치 → ValueError
  ② ★action_scheme★ 불일치 → ValueError  (2026-07-28 추가 — 차원은 같고 '의미'만 다른 오역 차단.
     과거 감사 C-1에서 sim ik_rel을 배포 joint_absolute로 해석한 결함을 막았고,
     현행 sim/real은 모두 ik_rel이다.)
  ③ ★target_range★ 불일치 → ValueError  (2026-07-28 추가 — target_pos는 관측 슬롯인데
     sample_range가 해시에 없어 타겟 분포만 바꾸면 조용히 통과하던 사각지대.)
  ④ 정상 계약은 통과(거짓 양성 없음)
"""
from __future__ import annotations

import copy

import pytest

from common.config_io import build_action_cfg, build_preprocess_cfg, configs_dir, load_yaml
from common.policy_io import make_obs_spec, validate_obs_spec


@pytest.fixture(scope="module")
def cfgs():
    y = load_yaml(configs_dir() / "common.yaml")
    return build_preprocess_cfg(y), build_action_cfg(y)


@pytest.fixture(scope="module")
def good_spec(cfgs):
    """★배포측 계약과 정합하는★ 기준 spec.

    ⚠️2026-08-06(P7-1): `make_obs_spec`의 `action_scheme`이 ★필수 인자★가 됐다.
       구버전은 미지정 시 `act.mode`로 폴백했는데, 그 폴백 때문에 Export가 인자를
       빠뜨려도 배포 검증이 통과해 ★조용한 오역★이 가능했다. 여기서는 '정합 상태'를
       만들려는 것이므로 명시적으로 `act.mode`를 넣는다.
    """
    pre, act = cfgs
    return make_obs_spec(pre, act, action_scheme=act.mode)


def test_matching_spec_passes(cfgs, good_spec) -> None:
    """현행 configs로 만든 계약은 통과해야 한다 — 게이트가 과민하면 배포가 불가능해진다."""
    pre, act = cfgs
    validate_obs_spec(good_spec, pre, act)  # 예외 없으면 통과


def test_spec_contains_new_contract_fields(good_spec) -> None:
    """action_scheme·target_range가 계약서에 실제로 기록되는가(감사 지적 2건의 봉인 여부)."""
    assert "action_scheme" in good_spec and good_spec["action_scheme"]
    assert "target_range" in good_spec and good_spec["target_range"], (
        "target_range가 비었다 — configs/common.yaml target.sample_range를 읽지 못했다"
    )
    for axis in ("x", "y", "z"):
        assert axis in good_spec["target_range"]


@pytest.mark.parametrize("key,bad", [
    ("obs_dim", 12345),
    ("action_dim", 99),
    ("cfg_hash", "deadbeefdeadbeef"),
])
def test_scalar_mismatch_refuses(cfgs, good_spec, key, bad) -> None:
    """핵심 스칼라 항목이 다르면 기동 거부."""
    pre, act = cfgs
    spec = copy.deepcopy(good_spec)
    spec[key] = bad
    with pytest.raises(ValueError, match="obs_spec 불일치"):
        validate_obs_spec(spec, pre, act)


def test_layout_mismatch_refuses(cfgs, good_spec) -> None:
    """레이아웃(슬롯 순서·크기)이 다르면 기동 거부 — 관측이 조용히 섞이는 최악의 경우."""
    pre, act = cfgs
    spec = copy.deepcopy(good_spec)
    spec["layout"] = list(reversed(spec["layout"]))
    with pytest.raises(ValueError, match="obs_spec 불일치"):
        validate_obs_spec(spec, pre, act)


def test_action_scheme_mismatch_refuses(cfgs, good_spec) -> None:
    """★감사 C-1 직결★ — 차원은 같고 '의미'만 다른 경우를 잡아야 한다.

    실제 사고 형태: sim은 DiffIK(TCP pose delta)로 학습했는데 배포 ActionMapper는
    joint_absolute로 해석 → action[:6]이 관절 절대목표가 되어 위험한 명령이 나간다.
    두 스킴 모두 action_dim=7이라 기존 검증은 전부 통과했다.
    """
    pre, act = cfgs
    spec = copy.deepcopy(good_spec)
    spec["action_scheme"] = "ik_rel" if spec["action_scheme"] != "ik_rel" else "joint_absolute"
    with pytest.raises(ValueError, match="액션 스킴 불일치"):
        validate_obs_spec(spec, pre, act)


def test_target_range_mismatch_refuses(cfgs, good_spec) -> None:
    """★cfg_hash 사각지대 봉인 확인★ — 타겟 분포만 바꿔도 거부돼야 한다."""
    pre, act = cfgs
    spec = copy.deepcopy(good_spec)
    spec["target_range"] = {**spec["target_range"], "x": [0.0, 1.0]}
    with pytest.raises(ValueError, match="타겟 분포 불일치|obs_spec 불일치"):
        validate_obs_spec(spec, pre, act)


def test_target_range_is_in_cfg_hash(cfgs) -> None:
    """타겟 범위가 해시에 실제로 들어갔는가 — 들어가지 않으면 사각지대가 남는다."""
    pre, act = cfgs
    base = make_obs_spec(pre, act, action_scheme=act.mode)
    moved = make_obs_spec(pre, act, action_scheme=act.mode,
                          target_range={"x": [0.1, 0.2], "y": [-0.1, 0.1], "z": [0.0, 0.0]})
    assert base["cfg_hash"] != moved["cfg_hash"], (
        "target_range가 cfg_hash에 반영되지 않았다 — 관측 분포 변경이 조용히 통과한다"
    )


def test_action_scheme_is_required_not_defaulted(cfgs) -> None:
    """★2026-08-06(P7-1) 계약 반전★: 기본값 폴백을 ★금지★한다.

    구버전은 미지정 시 `ActionCfg.mode`(=배포측 계약값)로 폴백했다. 그런데 Export가
    인자를 빠뜨리자 기록값 == 배포 기댓값이 되어 ★검증이 통과★했고, sim이 `ik_rel`로
    학습했는데도 게이트가 열렸다. ★막으려던 사고를 기본값이 되레 허용한 것★이다.
    ⇒ 이제 빠뜨리면 export 시점에 TypeError로 즉시 실패한다.
    """
    pre, act = cfgs
    with pytest.raises(TypeError):
        make_obs_spec(pre, act)  # type: ignore[call-arg]
    assert make_obs_spec(pre, act, action_scheme=act.mode)["action_scheme"] == act.mode


def test_legacy_spec_without_new_fields_still_refused_by_hash(cfgs, good_spec) -> None:
    """구 계약서(새 필드 없음)는 cfg_hash 단계에서 먼저 걸린다 — 조용히 통과하면 안 된다."""
    pre, act = cfgs
    legacy = {k: v for k, v in good_spec.items() if k not in ("action_scheme", "target_range")}
    legacy["cfg_hash"] = "0123456789abcdef"  # 구 해시(타겟범위 미포함으로 계산됐던 값 대역)
    with pytest.raises(ValueError):
        validate_obs_spec(legacy, pre, act)
    # ★2026-08-06(P7-1)★ 해시가 우연히 맞더라도 ★action_scheme 부재만으로 거부★돼야 한다 —
    #   정책 출력의 의미를 모르는 계약서는 그 자체로 배포 금지다.
    legacy2 = {k: v for k, v in good_spec.items() if k != "action_scheme"}
    with pytest.raises(ValueError, match="action_scheme"):
        validate_obs_spec(legacy2, pre, act)
