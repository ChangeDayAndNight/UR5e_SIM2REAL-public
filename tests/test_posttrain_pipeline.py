"""Teacher 상위 5개 자동평가 오케스트레이터의 순수 로직 계약."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


_PATH = Path(__file__).parents[1] / "Phase1" / "BinPick_PostTrain.py"
_EVAL_PATH = Path(__file__).parents[1] / "Phase1" / "BinPick_Eval.py"
_SPEC = importlib.util.spec_from_file_location("binpick_posttrain", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def test_select_top_five_uses_reward_at_checkpoint(tmp_path: Path) -> None:
    checkpoints = []
    for iteration in range(0, 500, 50):
        checkpoint = tmp_path / f"model_{iteration}.pt"
        checkpoint.touch()
        checkpoints.append(checkpoint)
    steps = list(range(500))
    values = [float((step * 7) % 31) for step in steps]

    got = _MOD.select_top_checkpoints(checkpoints, steps, values, count=5)
    expected = sorted(
        checkpoints,
        key=lambda path: (values[_MOD.checkpoint_iteration(path)], _MOD.checkpoint_iteration(path)),
        reverse=True,
    )[:5]
    assert [Path(row["checkpoint"]).name for row in got] == [path.name for path in expected]
    assert len(got) == 5


def test_reward_lookup_uses_latest_prior_step() -> None:
    assert _MOD.reward_at_or_before([0, 50, 100], [1.0, 2.0, 3.0], 75) == 2.0
    with pytest.raises(ValueError):
        _MOD.reward_at_or_before([50], [2.0], 49)


def test_ranking_contract_is_placed_then_carry_then_grasp() -> None:
    rows = [
        {"iteration": 1, "placed_rate": 0.90, "carry_rate": 0.99, "grasp_rate": 1.00,
         "train_mean_reward": 20.0},
        {"iteration": 2, "placed_rate": 0.91, "carry_rate": 0.92, "grasp_rate": 0.93,
         "train_mean_reward": 10.0},
        {"iteration": 3, "placed_rate": 0.91, "carry_rate": 0.94, "grasp_rate": 0.92,
         "train_mean_reward": 5.0},
    ]
    assert [row["iteration"] for row in _MOD.rank_evaluations(rows)] == [3, 2, 1]


def test_checkpoint_name_contract() -> None:
    assert _MOD.checkpoint_iteration(Path("model_19999.pt")) == 19999
    with pytest.raises(ValueError):
        _MOD.checkpoint_iteration(Path("best.pt"))


def test_human_readable_reports_use_txt_not_markdown(tmp_path: Path) -> None:
    selection = [{
        "checkpoint": str(tmp_path / "model_10.pt"),
        "iteration": 10,
        "train_mean_reward": 1.25,
    }]
    ranking = [{
        **selection[0],
        "grasp_rate": 0.98,
        "carry_rate": 0.97,
        "placed_rate": 0.96,
        "placed_ci95": [0.94, 0.98],
        "n_episodes": 512,
    }]
    _MOD._write_selection(tmp_path, selection)
    _MOD._write_ranking(tmp_path, ranking)
    _MOD._write_analysis(tmp_path, _MOD.analyze_ranking(ranking))
    assert {path.name for path in tmp_path.glob("*.txt")} == {
        "top5_selection.txt", "ranking.txt", "performance_analysis.txt",
    }
    assert not list(tmp_path.glob("*.md"))


def test_performance_analysis_uses_95pct_and_f49_baseline() -> None:
    ranking = [{
        "iteration": 12350,
        "checkpoint": "model_12350.pt",
        "train_mean_reward": 140.0,
        "grasp_rate": 0.98,
        "carry_rate": 0.975,
        "placed_rate": 0.96,
        "placed_ci95": [0.94, 0.975],
        "n_episodes": 512,
        "failure_counts": {
            "no_grasp": 10,
            "grasp_no_carry": 3,
            "carry_no_place": 7,
            "placed": 492,
        },
        "distance_stats": {"placed_min_goal_dist_m": {"mean": 0.012}},
    }]
    got = _MOD.analyze_ranking(ranking)
    assert got["point_target_met"] is True
    assert got["wilson_lower_target_met"] is False
    assert got["delta_vs_f49_pp"] == pytest.approx(3.62)
    assert got["primary_bottleneck"] == "carry_to_placed_pp"
    assert got["winner_failure_rates"]["carry_no_place"] == pytest.approx(7 / 512)
    assert got["winner_distance_stats"]["placed_min_goal_dist_m"]["mean"] == 0.012
    assert got["top5_placed_span_pp"] == 0.0


def test_performance_analysis_blocks_phase2_on_point_target_miss() -> None:
    ranking = [{
        "iteration": 10000,
        "checkpoint": "model_10000.pt",
        "train_mean_reward": 130.0,
        "grasp_rate": 0.97,
        "carry_rate": 0.95,
        "placed_rate": 0.94,
        "placed_ci95": [0.91, 0.96],
    }]
    got = _MOD.analyze_ranking(ranking)
    assert got["point_target_met"] is False
    assert got["primary_bottleneck"] == "grasp_to_carry_pp"


def test_eval_caps_simultaneous_completions_to_requested_episode_count() -> None:
    source = _EVAL_PATH.read_text(encoding="utf-8")
    assert "remaining = max(0, args_cli.n_episodes - len(records))" in source
    assert "for i in done_idx[:remaining].tolist():" in source


def test_wait_for_pid_captures_wrapper_exit_code(tmp_path: Path) -> None:
    import psutil

    process = subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(0.2); raise SystemExit(7)",
    ])
    create_time = psutil.Process(process.pid).create_time()
    logger = _MOD.Logger(tmp_path / "pipeline.log")
    assert _MOD._wait_for_pid(process.pid, create_time, 1, logger) == 7
    assert "exit=7" in (tmp_path / "pipeline.log").read_text(encoding="utf-8")


def test_posttrain_never_mutates_schema2_training_receipt() -> None:
    source = _PATH.read_text(encoding="utf-8")
    pipeline = (Path(__file__).parents[1] / "scripts" / "adoption_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "_record_training_completion" not in source
    assert "schema-2 run_manifest.json changed during evaluation" in pipeline
    assert "sha256_file(run_manifest)" in pipeline


def test_episode_diagnostics_and_strict_stage_bottleneck(tmp_path: Path) -> None:
    csv_path = tmp_path / "episodes.csv"
    fieldnames = [
        "c_at_tol", "c_latch_tol", "c_latch_tol_settled", "c_released",
        "c_latch_released", "placed_success", "zone_steps", "zone_min_obj_spd",
        "zone_min_arm_qd", "zone_min_obj_z", "zone_mean_a_grip", "zone_max_a_grip",
    ]
    rows = [
        [1, 1, 1, 1, 1, 1, 4, 0.005, 0.02, 0.18, 0.1, 1.0],
        [1, 1, 1, 1, 1, 0, 3, 0.010, 0.05, 0.19, -0.2, 0.2],
        [1, 1, 0, 1, 0, 0, 2, 0.030, 0.10, 0.22, -1.0, -0.5],
        [1, 1, 0, 0, 0, 0, 0, "", "", "", "", ""],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    diagnostics = _MOD._episode_diagnostics(csv_path)
    assert diagnostics["record_count"] == 4
    assert diagnostics["condition_rates"]["c_latch_tol"] == 1.0
    assert diagnostics["condition_rates"]["c_latch_tol_settled"] == 0.5
    assert diagnostics["zone_medians"]["zone_min_obj_spd"] == 0.01
    assert diagnostics["zone_open_episode_rate"] == pytest.approx(2 / 3)

    strict = _MOD._strict_stage_analysis({
        "grasp_rate": 1.0,
        "carry_rate": 1.0,
        "placed_rate": 0.25,
        "episode_diagnostics": diagnostics,
    })
    assert strict["rates"]["settled_at_3cm"] == 0.5
    assert strict["primary_bottleneck"] == "latch_at_3cm_to_settled_at_3cm_pp"
    assert strict["losses_pp"]["latch_at_3cm_to_settled_at_3cm_pp"] == 50.0


def test_f85_error_scan_ignores_diagnostic_nan_lines() -> None:
    """★F85 — 라인 스캔은 진단 요약의 정당한 nan에 fatal을 선고하면 안 된다★

    [사고: 2026-09-07 채택 오탐] model_24999/seed43 평가가 receipt 봉인까지 정상
    완료했는데, BinPick_Eval의 무조건부 진단 요약(P13: 분모 0이면 의도적으로 nan을
    찍는다 — "체류 스텝 중앙 nan" 등)에 `\bnan\b`(IGNORECASE)가 걸려 채택 전체가
    중단됐다. 존 진입 0회 시드가 하나라도 나오는 저성공률 student에서 구조적으로
    재발한다. 진짜 NaN 병리의 정본 검증은 adoption.py의 finite 게이트
    (protocol.require_finite)가 receipt 실측값으로 수행한다 — 라인 스캔은
    '프로세스가 죽는' 부류만 담당해야 한다.
    """
    import re as _re

    spec = importlib.util.spec_from_file_location(
        "adoption_pipeline_f85", Path(__file__).parents[1] / "scripts" / "adoption_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scan = mod._ERROR_RE
    # 사고 당시 실제 출력 줄들 — fatal이 아니어야 한다.
    for line in (
        "  체류 스텝  중앙 nan / 최대 0",
        "  래치 후 최소 ★수평★거리 중앙 : nan m   (3D 최소 nan m)",
        "  steps_to_placed_after_forced_open : nan",
    ):
        assert scan.search(line) is None, f"진단 nan 줄이 fatal로 오판된다: {line!r}"
    # 죽는 부류는 계속 잡아야 한다.
    for line in (
        "RuntimeError: CUDA out of memory. Tried to allocate 512 MiB",
        "Traceback (most recent call last):",
        "FloatingPointError: overflow encountered",
    ):
        assert scan.search(line) is not None, f"치명 시그니처를 놓친다: {line!r}"


def test_f86_render_guard_covers_rgb_not_only_depth() -> None:
    """★F86 — 렌더 건전성 가드는 RGB-D 4채널을 전부 봐야 한다★

    [사고: 2026-09-07] 가드가 depth만 검사했는데 student는 RGB-D다. 같은 checkpoint·
    같은 seed로 3회 반복한 실측에서 depth 블록은 세 번 모두 완전히 동일했고
    (mean 0.9712 / 공간std 0.0744 / far 83.3%) RGB만 죽었다:
      run1 rgb std 0.2939 → latch 21.9% / placed 3.5%
      run2 rgb std 0.0011(검정 70.5%) → latch 0.2% / placed 0.0%   ★아티팩트★
      run3 rgb std 0.2939 → latch 19.1% / placed 2.3%
    depth 가드는 셋 다 통과시켰고, 채택 평가 21회 중 9회가 이 상태로 측정돼
    정책이 큐브를 못 잡는 것으로 오판됐다.

    계약: use_rgb 계약일 때 RGB 블록의 공간 std를 depth와 같은 임계(0.02)로 검사하고,
    미달이면 depth와 동일한 EVAL_RENDER_DEGENERATE 마커로 즉시 죽는다(채택
    파이프라인의 렌더 재시도 경로가 이 마커를 소비한다).
    """
    # ★F94★ 가드 본체는 common/render_guard.py로 옮겨 train.py와 공유한다 — 소스 검사 대상도 그쪽이다.
    guard_path = _EVAL_PATH.parent.parent / "common" / "render_guard.py"
    source = guard_path.read_text(encoding="utf-8")
    assert "from common.render_guard import check_vision_render_health" in _EVAL_PATH.read_text(encoding="utf-8"), (
        "BinPick_Eval이 공유 렌더 가드를 부르지 않는다(F94)"
    )
    assert "use_rgb" in source and "rgb 공간std" in source, (
        "렌더 가드가 RGB 블록을 검사하지 않는다 — depth만 살아도 RGB가 죽으면 "
        "성공률 0%가 정책 성능으로 오판된다"
    )
    assert source.count("RENDER_DEGENERATE_MARKER}") >= 2 and 'RENDER_DEGENERATE_MARKER = "EVAL_RENDER_DEGENERATE"' in source, (
        "RGB 열화가 depth와 같은 마커로 보고되지 않는다 — 채택 재시도가 동작하지 않는다"
    )
    guard_region = source.split('if s["use_rgb"]:')[1][:1200]
    assert 'if s["rgb_spatial_std"] < SPATIAL_STD_MIN' in guard_region and "SPATIAL_STD_MIN = 0.02" in source, (
        "RGB 임계가 depth 가드(0.02)와 다르다"
    )
    # 슬라이스 계약: depth 블록 다음 3채널이 RGB다(build_policy_obs의 concat 순서).
    assert "flat[:, dh:dh + 3 * dh]" in source, "RGB 블록 슬라이스가 레이아웃 계약과 어긋난다"
