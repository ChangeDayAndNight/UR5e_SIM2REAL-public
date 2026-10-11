"""★액션 스킴 계약 (감사 P7-1, 2026-08-06)★ — sim 없이 도는 순수 검사(README 검증 계약).

배경(실제 결함이었다): `obs_spec.json`의 `action_scheme`은 "정책이 내놓는 7차원이 무엇을
      뜻하는가"를 봉인하는 필드다. 차원(action_dim=7)이 같아도 의미가 다르면
      (sim = TCP pose delta / 배포 = 관절 절대목표) 다른 검증은 ★전부 통과★하면서
      실기에서 정책 출력이 ★조용히 오역★된다.

      그런데 과거 감사에서 확인된 결함:
        · `make_obs_spec(..., action_scheme=None)` 이 미지정 시 ★`action_cfg.mode`로 폴백★
        · `Phase2/Export_Policy.py` 가 이 인자를 ★넘기지 않고 있었다★
        ⇒ 기록값 = joint_absolute == 배포측 기댓값 → ★검증 통과★
        ⇒ sim은 `ik_rel`로 학습했는데 게이트가 열렸다.
      ★막으려던 사고를 기본값이 되레 허용하고, "검증했다"는 거짓 안심까지 줬다.★

계약:
  ① `make_obs_spec`은 `action_scheme`을 ★필수★로 받는다 (빠뜨리면 export 시점에 실패)
  ② `validate_obs_spec`은 `action_scheme` ★부재를 거부★한다 (의미 불명 = 배포 금지)
  ③ sim 스킴은 ★단일 소스★(`action_scheme.SIM_ACTION_SCHEME`)에서 오고, 실제 ActionsCfg
     배선과 일치한다
  ④ 현행 sim/real 계약은 둘 다 `ik_rel`이며, legacy `joint_absolute` 불일치 fixture는 거부된다
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from common.config_io import configs_dir, load_yaml
from common.policy_io import make_obs_spec, validate_obs_spec

_ROOT = Path(__file__).resolve().parent.parent
_EXT = _ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick"
_CFG = _EXT / "binpick_env_cfg.py"
_SCHEME = _EXT / "action_scheme.py"
_EXPORT = _ROOT / "Phase2" / "Export_Policy.py"


def _cfgs():
    """configs/common.yaml에서 전처리·액션 cfg를 만든다 (배포측 계약).

    빌더는 `common/config_io.py`의 것을 그대로 쓴다 — 배포 경로와 같은 함수여야
    "계약을 검사한다"는 말이 성립한다(§4-3 정신).
    """
    from common.config_io import build_action_cfg, build_preprocess_cfg

    c = load_yaml(configs_dir() / "common.yaml")
    return build_preprocess_cfg(c), build_action_cfg(c)


def _sim_scheme() -> str:
    """확장의 단일 소스 상수를 ★파일에서★ 읽는다(isaaclab import 없이)."""
    if not _SCHEME.is_file():
        pytest.skip("sim 확장 없음 — 배포 번들에서는 대조 불가")
    m = re.search(r'^SIM_ACTION_SCHEME:\s*str\s*=\s*"([^"]+)"', _SCHEME.read_text(encoding="utf-8"), re.M)
    assert m, "SIM_ACTION_SCHEME 상수를 찾지 못함"
    return m.group(1)


# ── ① make_obs_spec은 action_scheme을 필수로 받는다 ──────────────────────────
def test_make_obs_spec_requires_action_scheme() -> None:
    """빠뜨리면 ★export 시점에★ 실패해야 한다 — 배포 시점의 조용한 통과보다 안전하다."""
    sig = inspect.signature(make_obs_spec)
    p = sig.parameters["action_scheme"]
    assert p.default is inspect.Parameter.empty, (
        "action_scheme에 기본값이 있다 — 구버전은 배포측 계약값(action_cfg.mode)으로 폴백해 "
        "검증이 무의미해졌다(P7-1). 필수 인자여야 한다."
    )
    pre, act = _cfgs()
    with pytest.raises(TypeError):
        make_obs_spec(pre, act)          # type: ignore[call-arg]


def test_make_obs_spec_rejects_empty_scheme() -> None:
    pre, act = _cfgs()
    for bad in ("", None):
        with pytest.raises((ValueError, TypeError)):
            make_obs_spec(pre, act, action_scheme=bad)   # type: ignore[arg-type]


# ── ② validate_obs_spec은 부재/불일치를 거부한다 ────────────────────────────
def test_validate_rejects_missing_action_scheme() -> None:
    """구 계약 파일(필드 없음)이 조용히 통과하면 안 된다 — 의미 불명 = 배포 금지."""
    pre, act = _cfgs()
    spec = make_obs_spec(pre, act, action_scheme=act.mode)
    spec.pop("action_scheme")
    with pytest.raises(ValueError, match="action_scheme"):
        validate_obs_spec(spec, pre, act)


def test_validate_rejects_scheme_mismatch() -> None:
    """과거 사고 fixture: 현 ik_rel 배포에 legacy joint_absolute 정책을 섞으면 거부."""
    pre, act = _cfgs()
    legacy_scheme = "joint_absolute" if act.mode == "ik_rel" else "ik_rel"
    spec = make_obs_spec(pre, act, action_scheme=legacy_scheme)
    with pytest.raises(ValueError, match="액션 스킴 불일치"):
        validate_obs_spec(spec, pre, act)


# ── ③ sim 스킴은 단일 소스이며 실제 배선과 일치한다 ─────────────────────────
def test_sim_scheme_matches_actions_cfg_wiring() -> None:
    """상수가 ActionsCfg의 실제 액션텀과 일치하는가 (이름만 맞으면 안 된다)."""
    scheme = _sim_scheme()
    if not _CFG.is_file():
        pytest.skip("binpick_env_cfg.py 없음")
    src = _CFG.read_text(encoding="utf-8")
    m = re.search(r"class ActionsCfg:(?:.|\n)*?(?=\n@configclass)", src)
    assert m, "ActionsCfg 를 찾지 못함"
    body = m.group(0)
    if scheme == "ik_rel":
        assert "DifferentialInverseKinematicsActionCfg" in body, (
            "SIM_ACTION_SCHEME='ik_rel'인데 ActionsCfg가 IK 액션텀을 쓰지 않는다"
        )
        assert "use_relative_mode=True" in body, (
            "'ik_rel'은 ★상대★ 모드여야 한다 — use_relative_mode=True 가 없다"
        )
    elif scheme == "joint_absolute":
        assert "JointPositionActionCfg" in body, (
            "SIM_ACTION_SCHEME='joint_absolute'인데 관절 액션텀이 아니다"
        )
    else:
        pytest.fail(f"알 수 없는 스킴 {scheme!r} — 계약에 배선 검사를 추가할 것")


def test_export_passes_sim_scheme_explicitly() -> None:
    """Export_Policy가 ★단일 소스 상수★를 명시 전달하는가 (기본값 의존 금지)."""
    if not _EXPORT.is_file():
        pytest.skip("Phase2/Export_Policy.py 없음")
    src = _EXPORT.read_text(encoding="utf-8")
    m = re.search(r"make_obs_spec\((?:.|\n)*?\)", src)
    assert m, "Export_Policy의 make_obs_spec 호출을 찾지 못함"
    call = m.group(0)
    assert "action_scheme=" in call, (
        "Export가 action_scheme을 넘기지 않는다 — 기본값 폴백으로 검증이 무력화된다(P7-1 실사고)"
    )
    assert "SIM_ACTION_SCHEME" in call, (
        "리터럴이 아니라 ★단일 소스 상수★를 넘길 것(action_scheme.SIM_ACTION_SCHEME)"
    )


# ── ④ 현행 sim/real action 의미 정합 ──────────────────────────────────────────
def test_current_sim_and_deploy_scheme_match() -> None:
    """현행 sim/real은 둘 다 ik_rel이고 obs_spec 검증을 통과해야 한다."""
    pre, act = _cfgs()
    scheme = _sim_scheme()
    assert scheme == act.mode == "ik_rel"
    spec = make_obs_spec(pre, act, action_scheme=scheme)
    validate_obs_spec(spec, pre, act)
