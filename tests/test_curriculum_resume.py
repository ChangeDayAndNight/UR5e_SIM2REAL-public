"""★resume 시 커리큘럼 되감김 방지 계약 (F44, 2026-08-01)★ — sim 불필요(README 검증 계약).

배경: 스텝 기반 커리큘럼(pregrasp frac · target radius)은 `env.common_step_counter`를
      읽는데, 이 카운터는 ★env를 새로 만들 때마다 0에서 시작★한다. 즉 resume해도
      커리큘럼은 처음부터 다시 돈다.

      실제 사고: iter 9550 체크포인트에서 600 iter(=14400 스텝)만 resume했더니
      pregrasp warmup(20000 스텝)조차 못 넘겨 frac이 ★1.0으로 되돌아갔다★.
      그러면 전 env가 '잡힌 채 시작'이 되고 `_real_pick_mask`가 0이므로
      place 계열 보상이 ★구조적으로 정확히 0★이 된다:
          goal_fine 1.8691 → 0.0000 / placed 0.0725 → 0.0000
      베이스라인(frac=0)과 완전히 다른 조건이라 ★비교 자체가 성립하지 않는다★.
      이 함정으로 F42(속도 감쇠)·F43(페널티 강화) 검증 두 건이 무효가 됐다.

계약:
  ① 스텝 기반 커리큘럼은 `curriculum_step(env)`만 쓴다 — `common_step_counter` 직접 읽기 금지
  ② `curriculum_step`은 환경변수 BP_CURRICULUM_STEP_OFFSET을 더한다
  ③ 학습 래퍼는 `--resume` 시 체크포인트 iter × num_steps_per_env를 ★자동★ 계산해 주입한다
     (사람이 기억하는 방식은 실패했다 — 두 번 같은 함정에 빠졌다)
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_EVENTS = _ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "mdp" / "events_dr.py"
_TRAIN = _ROOT / "Phase1" / "BinPick_Train.py"
_PPO = _ROOT / "source" / "ur5e_binpick" / "ur5e_binpick" / "tasks" / "binpick" / "agents" / "rsl_rl_ppo_cfg.py"

_ENV_VAR = "BP_CURRICULUM_STEP_OFFSET"


def _events_src() -> str:
    if not _EVENTS.is_file():
        pytest.skip("sim 확장(source/) 없음 — 배포 번들에서는 대조 불가")
    return _EVENTS.read_text(encoding="utf-8")


def _load_train_module():
    """BinPick_Train은 stdlib만 import하는 얇은 래퍼라 테스트에서 직접 import 가능."""
    if not _TRAIN.is_file():
        pytest.skip("Phase1/BinPick_Train.py 없음")
    spec = importlib.util.spec_from_file_location("_bp_train_under_test", _TRAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_curriculum_step_applies_env_offset() -> None:
    """② curriculum_step이 오프셋 환경변수를 반영한다."""
    src = _events_src()
    assert "def curriculum_step(" in src, "curriculum_step 헬퍼가 없습니다 (F44 계약)"
    assert _ENV_VAR in src, f"{_ENV_VAR} 환경변수를 읽지 않습니다"
    body = re.search(r"def curriculum_step\(env\)(?:.|\n)*?(?=\ndef |\Z)", src)
    assert body and "_STEP_OFFSET" in body.group(0), "curriculum_step이 오프셋을 더하지 않습니다"


def test_no_schedule_reads_step_counter_directly() -> None:
    """① 스케줄 함수가 common_step_counter를 직접 읽으면 resume에서 되감긴다."""
    src = _events_src()
    offenders = []
    for m in re.finditer(r"def (_\w*schedule)\(env(?:.|\n)*?(?=\ndef |\Z)", src):
        if "common_step_counter" in m.group(0):
            offenders.append(m.group(1))
    assert not offenders, (
        f"스케줄 함수가 common_step_counter를 직접 읽습니다: {offenders} — "
        "curriculum_step(env)을 쓸 것 (resume에서 커리큘럼이 되감긴다)"
    )


def test_train_wrapper_computes_offset_from_checkpoint() -> None:
    """③ 래퍼가 --resume + --checkpoint에서 오프셋을 자동 산출한다."""
    mod = _load_train_module()
    spi = mod._steps_per_iter()
    assert spi > 0
    # 실제 사고 재현: iter 9550에서 resume
    args = ["--resume", "--load_run", "2026-08-01_08-00-05", "--checkpoint", "model_9550.pt"]
    assert mod._curriculum_offset(args) == 9550 * spi
    # resume이 아니면 0 (scratch 학습은 커리큘럼을 처음부터 도는 것이 정상)
    assert mod._curriculum_offset(["--max_iterations", "12000"]) == 0


def test_offset_would_have_prevented_the_f44_incident() -> None:
    """사고 조건에서 오프셋이 pregrasp warmup을 실제로 넘기는지 — 계약의 목적 검증.

    warmup 20000 스텝 / decay 120000 스텝이므로, iter 9550(=229200 스텝)이면
    frac은 이미 0(자력 파지)이어야 한다. 오프셋이 없으면 1.0(전 env 보조)이 된다.
    """
    mod = _load_train_module()
    src = _events_src()
    m = re.search(r"warmup_steps: int = (\d+), decay_steps: int = (\d+)", src)
    assert m, "pregrasp 커리큘럼 스텝 설정을 찾지 못함"
    warmup, decay = int(m.group(1)), int(m.group(2))
    off = mod._curriculum_offset(["--resume", "--checkpoint", "model_9550.pt"])
    assert off > warmup + decay, (
        f"오프셋 {off} 스텝이 커리큘럼 종료 {warmup + decay} 이전 — "
        "resume 측정이 여전히 베이스라인과 다른 조건이 됩니다"
    )
    # 오프셋 없이 짧게 resume하면 커리큘럼 ★종료 지점★에 못 미친다(= 사고 조건의 본질).
    #   ⚠️초판은 `< warmup` 으로 고정해 두었다가 커리큘럼 단축(2026-08-05) 때 깨졌다 —
    #     사고의 본질은 "짧은 resume이 커리큘럼을 되감는다"이지 특정 warmup 값이 아니다.
    assert 600 * mod._steps_per_iter() < warmup + decay


def test_steps_per_iter_matches_rsl_rl_cfg() -> None:
    """환산 상수는 rsl_rl cfg 단일 소스에서만 온다 (하드코딩 금지)."""
    mod = _load_train_module()
    if not _PPO.is_file():
        pytest.skip("rsl_rl cfg 없음")
    m = re.search(r"^\s*num_steps_per_env\s*=\s*(\d+)", _PPO.read_text(encoding="utf-8"), re.M)
    assert m and mod._steps_per_iter() == int(m.group(1))
