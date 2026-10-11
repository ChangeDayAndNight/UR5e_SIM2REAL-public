"""논문 산출물 생성기(analysis/) 단위테스트 — 순수 python(Isaac/sim 불필요).

eval_stats: Wilson CI 기지값·경계, 실패모드 분류(상호배타), analyze_one 집계.
paper_tables: DR 표가 dr.yaml 전 항목을 담는지, ablation/obs 표 생성, tex 이스케이프.
rollout_filmstrip: 스토리보드 수집 순서. (합성 렌더는 GPU 불필요하나 matplotlib만 확인.)

실행: python -m pytest tests/test_paper_outputs.py -q
산출: 없음 (pass/fail)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import eval_stats, exp_report, paper_tables  # noqa: E402


# ── eval_stats.wilson_ci ──────────────────────────────────────────────────
def test_wilson_ci_bounds_and_known():
    # n=0 → NaN
    lo, hi = eval_stats.wilson_ci(0, 0)
    assert np.isnan(lo) and np.isnan(hi)
    # 모든 성공/실패라도 구간이 [0,1] 안 (정규근사는 벗어남 — Wilson의 존재 이유)
    for k, n in [(512, 512), (0, 512), (1, 3), (509, 512)]:
        lo, hi = eval_stats.wilson_ci(k, n)
        assert 0.0 <= lo <= hi <= 1.0
        assert lo <= k / n <= hi + 1e-9
    # 기지값: k=50,n=100 → 95% CI ≈ [0.404, 0.596] (표준 Wilson)
    lo, hi = eval_stats.wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_wilson_ci_narrows_with_n():
    _, hi_small = eval_stats.wilson_ci(9, 10)
    _, hi_big = eval_stats.wilson_ci(900, 1000)
    lo_small, _ = eval_stats.wilson_ci(9, 10)
    lo_big, _ = eval_stats.wilson_ci(900, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)  # 표본 크면 구간 좁아짐


# ── eval_stats.classify_failure (상호배타·단계 게이트) ──────────────────────
@pytest.mark.parametrize("g,c,p,expect", [
    (0, 0, 0, "no_grasp"),
    (1, 0, 0, "grasp_no_carry"),
    (1, 1, 0, "carry_no_place"),
    (1, 1, 1, "placed"),
    (0, 1, 1, "placed"),   # placed면 상위 단계 무관하게 placed(최종 성공 우선)
])
def test_classify_failure(g, c, p, expect):
    row = pd.Series({"grasp_success": g, "carry_latch_success": c, "placed_success": p})
    assert eval_stats.classify_failure(row) == expect


def test_analyze_one_roundtrip(tmp_path):
    # 합성 CSV: 10 에피소드 (placed 7, carry만 2, no_grasp 1)
    rows = []
    for i in range(7):
        rows.append(dict(episode=i, grasp_success=1, carry_latch_success=1, placed_success=1,
                         min_goal_dist_m=0.02))
    for i in range(2):
        rows.append(dict(episode=7 + i, grasp_success=1, carry_latch_success=1, placed_success=0,
                         min_goal_dist_m=0.10))
    rows.append(dict(episode=9, grasp_success=0, carry_latch_success=0, placed_success=0,
                     min_goal_dist_m=0.30))
    csv = tmp_path / "episodes.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    st = eval_stats.analyze_one(csv)
    assert st["n_episodes"] == 10
    assert st["metrics"]["placed"]["k"] == 7
    assert st["metrics"]["grasp"]["k"] == 9
    assert st["failure"] == {"no_grasp": 1, "grasp_no_carry": 0,
                             "carry_no_place": 2, "placed": 7}
    # placed 성공 시 목표거리 = 0.02 (7개 전부)
    assert abs(st["dist"]["placed_min_goal_dist_m"]["mean"] - 0.02) < 1e-9


def test_eval_stats_writes_tables(tmp_path):
    df = pd.DataFrame([dict(episode=i, grasp_success=1, carry_latch_success=1,
                            placed_success=1, min_goal_dist_m=0.03) for i in range(20)])
    df.to_csv(tmp_path / "episodes.csv", index=False)
    eval_stats.write_result_table([eval_stats.analyze_one(tmp_path / "episodes.csv")],
                                  tmp_path, "t")
    text = (tmp_path / "results_table.txt").read_text(encoding="utf-8")
    tex = (tmp_path / "results_table.tex").read_text(encoding="utf-8")
    assert "grasp" in text and "placed" in text
    assert "\\toprule" in tex and "\\bottomrule" in tex and "\\%" in tex  # % 이스케이프


# ── paper_tables ────────────────────────────────────────────────────────────
def test_dr_table_covers_all_yaml_params(tmp_path):
    import yaml
    dr = yaml.safe_load((ROOT / "configs" / "dr.yaml").read_text(encoding="utf-8"))
    expected = sum(len(v) for v in dr.values() if isinstance(v, dict))
    paper_tables.build_dr_table(tmp_path)
    text = (tmp_path / "dr_table.txt").read_text(encoding="utf-8")
    # dr.yaml의 모든 파라미터명이 표에 등장(코드-config 어긋남 방지)
    for sec, params in dr.items():
        if isinstance(params, dict):
            for key in params:
                assert key in text, f"DR 표에 {key} 누락"
    # 데이터 행 수 = 파라미터 수
    body = [l for l in text.splitlines() if l.startswith("|") and "---" not in l]
    assert len(body) - 1 == expected  # 헤더 1행 제외


def test_ablation_and_obs_tables_build(tmp_path):
    paper_tables.build_ablation(tmp_path)
    paper_tables.build_obs_action(tmp_path)
    for f in ("ablation_student.txt", "ablation_teacher.tex",
              "obs_action_spec.txt", "obs_action_spec.tex"):
        assert (tmp_path / f).is_file() and (tmp_path / f).stat().st_size > 0
    # obs 스펙에 depth 해상도(common.yaml out_height×out_width) 반영 — 해상도 무관 검증
    #   (2026-07-19 112² 전환에서 "64" 하드코딩이 오탐 — yaml 값을 직접 대조)
    from common.config_io import build_preprocess_cfg, configs_dir, load_yaml

    _pre = build_preprocess_cfg(load_yaml(configs_dir() / "common.yaml"))
    _text = (tmp_path / "obs_action_spec.txt").read_text(encoding="utf-8")
    assert f"{_pre.out_height}×{_pre.out_width}" in _text


# ═════════════════════════════════════════════════════════════════════════════
# ★P7-3(2026-08-06)★ 보상값 비교 함정 — 산출물이 재현 정보를 반드시 담아야 한다
#
# Isaac Lab은 `Episode_Reward/*`를 `episode_sum / max_episode_length_s`로 나눠 기록한다
#   (vendored managers/reward_manager.py L120). 에피소드 길이를 8.0s → 5.0s로 바꾸면
#   ★같은 물리 거동이 1.6배 큰 값★으로 찍힌다.
# 실제로 이 프로젝트는 에피소드 길이를 바꿔가며 실험했으므로, 산출물에 길이·rollout 길이가
# 남아 있지 않으면 사후에 ★잘못된 비교★를 하게 된다.
# ═════════════════════════════════════════════════════════════════════════════


def test_summary_records_episode_length_and_rollout() -> None:
    """요약 생성기가 episode_length_s·num_steps_per_env를 기록하는가."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "analysis" / "exp_report.py").read_text(encoding="utf-8")
    assert "episode_length_s" in src, (
        "산출물에 episode_length_s가 없다 — 보상값이 이 길이로 정규화되므로 "
        "기록이 없으면 run 간 비교가 불가능해진다(P7-3)"
    )
    assert "num_steps_per_env" in src, "rollout 길이(num_steps_per_env)가 기록되지 않는다"


def test_exp_report_prefers_training_start_git_manifest(tmp_path: Path) -> None:
    """학습 중 HEAD가 바뀌어도 종료 리포트는 시작 commit을 기록해야 한다."""
    import json

    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "training_git_hash": "0123456789abcdef",
        "training_git_hash_short": "0123456",
        "training_started_at": "2026-08-08T00:24:45+09:00",
        "training_completed_at": "2026-08-09T01:00:00+09:00",
        "command": ["python", "Phase1/BinPick_Train.py"],
        "return_code": 0,
    }), encoding="utf-8")
    meta = exp_report.load_meta(tmp_path)
    assert meta["git_hash"] == "0123456"
    assert meta["training_git_hash_full"] == "0123456789abcdef"
    assert meta["report_git_hash"]  # 리포트 생성 시점 HEAD도 감사용으로 별도 보존


def test_summary_warns_about_reward_normalization() -> None:
    """요약에 ★비교 금지 경고★가 새겨져 있는가 — 사람이 기억하는 방식은 실패한다."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "analysis" / "exp_report.py").read_text(encoding="utf-8")
    assert "max_episode_length_s" in src and "비교" in src, (
        "보상 정규화 경고가 산출물에 없다 — 에피소드 길이가 다른 run의 보상값을 "
        "직접 비교하는 사고를 막을 장치가 필요하다(P7-3)"
    )


def test_exp_summary_uses_text_not_markdown(tmp_path: Path) -> None:
    exp_report.write_summary(
        tmp_path,
        {"run_name": "r", "git_hash": "abc", "hyperparams": {}},
        {"placed": 0.5},
        "teacher",
        None,
    )
    assert (tmp_path / "summary.txt").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert not list(tmp_path.glob("*.md"))
