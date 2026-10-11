#!/usr/bin/env python
"""Teacher top-5 multi-seed adoption entry point.

The active path delegates to :mod:`scripts.adoption_pipeline`: the top five
checkpoints by trailing-200-median ``Episode_Termination/placed_success`` (final
checkpoint always included, 500-iteration separation; F70) are evaluated at
seeds 42..45 with exactly 512 episodes each.  Instantaneous ``Train/mean_reward``
selection was rejected — early-terminating successes make it anti-correlate with
success rate (see README F70). Only immutable ``adoption.json`` can authorize the original
``logs/.../model_N.pt``; no weight copy is produced and the schema-2 training
receipt is never modified. Legacy pure report helpers below remain importable
for historical-result analysis and unit tests.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The README intentionally exposes this file as the canonical Teacher
# adoption entry point.  Keep every evaluation, TensorBoard read, and receipt
# verification on the exact isaac51 interpreter instead of inheriting an
# arbitrary shell Python.  This guard is stdlib-only and runs before importing
# the heavier adoption pipeline.
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

EVAL_SCRIPT = ROOT / "Phase1" / "BinPick_Eval.py"
RESULT_ROOT = ROOT / "Phase1" / "Result"
REWARD_TAG = "Train/mean_reward"
TARGET_PLACED_RATE = 0.95
F49_PLACED_RATE = 0.9238
_CKPT_RE = re.compile(r"model_(\d+)\.pt$")
_ERROR_RE = re.compile(
    r"CUDA out of memory|Traceback|Error executing job|FloatingPointError|\bnan\b",
    re.IGNORECASE,
)


def checkpoint_iteration(path: Path) -> int:
    """`model_<iter>.pt`의 iteration을 반환한다."""
    match = _CKPT_RE.search(path.name)
    if not match:
        raise ValueError(f"checkpoint 이름 형식 오류: {path}")
    return int(match.group(1))


def reward_at_or_before(steps: list[int], values: list[float], iteration: int) -> float:
    """checkpoint iteration 이하에서 가장 가까운 TensorBoard 보상을 반환한다."""
    index = bisect.bisect_right(steps, iteration) - 1
    if index < 0:
        raise ValueError(f"iter {iteration} 이전의 {REWARD_TAG} 값이 없습니다")
    return float(values[index])


def select_top_checkpoints(
    checkpoints: Iterable[Path], steps: list[int], values: list[float], count: int = 5,
) -> list[dict]:
    """저장 checkpoint에서 학습 보상 상위 `count`개를 결정적으로 선별한다.

    ⚠️★레거시★ — 활성 채택 경로(main → scripts.adoption_pipeline)는 이 함수를 쓰지
    않는다. 실제 선발은 adoption_selection.select_teacher_candidates(F70: placed_success
    trailing-median)다. 순간 mean_reward 선발은 성공의 조기 종료와 역상관해 기각됐다.
    """
    rows = []
    for checkpoint in checkpoints:
        iteration = checkpoint_iteration(checkpoint)
        reward = reward_at_or_before(steps, values, iteration)
        if not math.isfinite(reward):
            continue
        rows.append({
            "iteration": iteration,
            "checkpoint": str(checkpoint.resolve()),
            "train_mean_reward": reward,
        })
    # 동률이면 더 늦은 checkpoint를 우선한다(더 많은 학습을 거친 정책).
    rows.sort(key=lambda row: (row["train_mean_reward"], row["iteration"]), reverse=True)
    return rows[:count]


def rank_evaluations(rows: Iterable[dict]) -> list[dict]:
    """실험 문서 계약대로 placed → carry → grasp 순으로 정량평가 결과를 정렬한다."""
    ranked = list(rows)
    ranked.sort(
        key=lambda row: (
            row["placed_rate"], row["carry_rate"], row["grasp_rate"],
            row["train_mean_reward"], row["iteration"],
        ),
        reverse=True,
    )
    return ranked


def _episode_diagnostics(csv_path: Path) -> dict:
    """episodes.csv에서 strict placed 조건과 zone 계측을 순수 Python으로 집계한다."""
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"평가 episode CSV가 비어 있음: {csv_path}")

    binary_keys = (
        "c_at_tol", "c_latch_tol", "c_latch_tol_settled",
        "c_released", "c_latch_released", "placed_success",
    )
    condition_counts = {
        key: sum(int(row.get(key, "0") or 0) for row in rows) for key in binary_keys
    }
    condition_rates = {key: value / len(rows) for key, value in condition_counts.items()}

    zone_rows = [row for row in rows if int(row.get("zone_steps", "0") or 0) > 0]

    def numeric_values(key: str) -> list[float]:
        values = []
        for row in zone_rows:
            raw = row.get(key, "")
            if raw not in (None, ""):
                values.append(float(raw))
        return values

    zone_medians = {}
    for key in (
        "zone_steps", "zone_min_obj_spd", "zone_min_arm_qd", "zone_min_obj_z",
        "zone_mean_a_grip", "zone_max_a_grip",
    ):
        values = numeric_values(key)
        zone_medians[key] = statistics.median(values) if values else None
    zone_max_grip = numeric_values("zone_max_a_grip")
    return {
        "record_count": len(rows),
        "condition_counts": condition_counts,
        "condition_rates": condition_rates,
        "zone_episode_count": len(zone_rows),
        "zone_medians": zone_medians,
        "zone_open_episode_rate": (
            sum(value > 0.0 for value in zone_max_grip) / len(zone_max_grip)
            if zone_max_grip else None
        ),
    }


def _strict_stage_analysis(row: dict) -> dict:
    conditions = row.get("episode_diagnostics", {}).get("condition_rates", {})
    required = ("c_latch_tol", "c_latch_tol_settled")
    if any(key not in conditions for key in required):
        return {}
    rates = {
        "grasp": float(row["grasp_rate"]),
        "carry_8cm": float(row["carry_rate"]),
        "latch_at_3cm": float(conditions["c_latch_tol"]),
        "settled_at_3cm": float(conditions["c_latch_tol_settled"]),
        "placed": float(row["placed_rate"]),
    }
    order = list(rates)
    losses = {
        f"{left}_to_{right}_pp": 100.0 * (rates[left] - rates[right])
        for left, right in zip(order, order[1:])
    }
    return {
        "rates": rates,
        "losses_pp": losses,
        "primary_bottleneck": max(losses, key=losses.get),
    }


def analyze_ranking(ranking: list[dict]) -> dict:
    """평가 순위에서 목표 달성·F49 개선·단계 병목을 정량 판정한다."""
    if not ranking:
        raise ValueError("분석할 정량평가 결과가 없습니다")
    winner = ranking[0]
    grasp = float(winner["grasp_rate"])
    carry = float(winner["carry_rate"])
    placed = float(winner["placed_rate"])
    placed_lo, placed_hi = (float(value) for value in winner["placed_ci95"])
    stage_losses = {
        "grasp_to_carry_pp": 100.0 * (grasp - carry),
        "carry_to_placed_pp": 100.0 * (carry - placed),
    }
    bottleneck = max(stage_losses, key=stage_losses.get)
    n_episodes = int(winner.get("n_episodes", 0))
    failure_counts = {
        str(key): int(value) for key, value in winner.get("failure_counts", {}).items()
    }
    failure_rates = {
        key: value / n_episodes for key, value in failure_counts.items()
    } if n_episodes > 0 else {}
    placed_values = [float(row["placed_rate"]) for row in ranking]
    return {
        "winner_iteration": int(winner["iteration"]),
        "winner_checkpoint": winner["checkpoint"],
        "placed_rate": placed,
        "placed_ci95": [placed_lo, placed_hi],
        "point_target": TARGET_PLACED_RATE,
        "point_target_met": placed >= TARGET_PLACED_RATE,
        "wilson_lower_target_met": placed_lo >= TARGET_PLACED_RATE,
        "f49_placed_rate": F49_PLACED_RATE,
        "delta_vs_f49_pp": 100.0 * (placed - F49_PLACED_RATE),
        "stage_losses_pp": stage_losses,
        "primary_bottleneck": bottleneck,
        "n_episodes": n_episodes,
        "winner_failure_counts": failure_counts,
        "winner_failure_rates": failure_rates,
        "winner_distance_stats": winner.get("distance_stats", {}),
        "winner_episode_diagnostics": winner.get("episode_diagnostics", {}),
        "winner_strict_stage_analysis": winner.get(
            "strict_stage_analysis", _strict_stage_analysis(winner),
        ),
        "top5_placed_range": [min(placed_values), max(placed_values)],
        "top5_placed_span_pp": 100.0 * (max(placed_values) - min(placed_values)),
        "train_reward_winner_matches_eval_winner": (
            int(winner["iteration"])
            == int(max(ranking, key=lambda row: row["train_mean_reward"])["iteration"])
        ),
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


class Logger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def emit(self, message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def subprocess_line(self, line: str) -> None:
        print(line, end="", flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line)


def _wait_for_pid(
    pid: int, create_time: float | None, poll_seconds: int, logger: Logger,
) -> int | None:
    """학습 래퍼 종료를 기다리고 가능한 플랫폼에서는 실제 종료코드를 반환한다."""
    if pid <= 0:
        return None
    import psutil

    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        logger.emit(f"학습 래퍼 PID {pid}는 감시 시작 전에 이미 종료됨 — 산출물로 판정")
        return None
    if create_time is not None and abs(process.create_time() - create_time) > 2.0:
        raise RuntimeError(f"학습 래퍼 PID {pid} 생성시각 불일치 — PID 재사용 가능성")
    logger.emit(f"학습 래퍼 PID {pid} 종료 대기 시작")
    last_notice = 0.0
    while True:
        try:
            exit_code = process.wait(timeout=max(1, poll_seconds))
        except psutil.TimeoutExpired:
            now = time.monotonic()
            if now - last_notice >= 300.0:
                logger.emit(f"학습 래퍼 PID {pid} 실행 중 — 계속 대기")
                last_notice = now
            continue
        except psutil.NoSuchProcess:
            logger.emit(f"학습 래퍼 PID {pid} 종료 확인 — 종료코드 조회 불가")
            return None
        normalized = int(exit_code) if exit_code is not None else None
        logger.emit(
            f"학습 래퍼 PID {pid} 종료 확인 — exit="
            f"{normalized if normalized is not None else 'unknown'}"
        )
        return normalized


def _gpu_memory_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        if completed.returncode == 0:
            return max(int(line.strip()) for line in completed.stdout.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


def _wait_for_gpu(limit_mib: int, poll_seconds: int, logger: Logger) -> None:
    """학습/후처리 Kit이 GPU를 반납할 때까지 기다려 eval OOM을 막는다."""
    while True:
        used = _gpu_memory_used_mib()
        if used is None or used <= limit_mib:
            logger.emit(f"GPU 평가 게이트 통과: used={used if used is not None else 'unknown'} MiB")
            return
        logger.emit(f"GPU used={used} MiB > {limit_mib} MiB — 평가 시작 대기")
        time.sleep(max(5, poll_seconds))


def _load_reward_series(run_dir: Path) -> tuple[list[int], list[float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    if REWARD_TAG not in tags:
        raise RuntimeError(f"TensorBoard tag 없음: {REWARD_TAG}")
    events = accumulator.Scalars(REWARD_TAG)
    return [int(event.step) for event in events], [float(event.value) for event in events]


def _verify_complete(run_dir: Path, max_iterations: int, logger: Logger) -> tuple[list[int], list[float]]:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=checkpoint_iteration)
    if not checkpoints:
        raise RuntimeError(f"checkpoint 없음: {run_dir}")
    final_iteration = checkpoint_iteration(checkpoints[-1])
    expected_final = max_iterations - 1
    if final_iteration < expected_final:
        raise RuntimeError(
            f"20k 미완주: 최종 checkpoint model_{final_iteration}.pt < model_{expected_final}.pt"
        )
    steps, values = _load_reward_series(run_dir)
    if not steps or steps[-1] < expected_final:
        raise RuntimeError(f"TensorBoard 미완주: 마지막 step={steps[-1] if steps else None}")
    logger.emit(
        f"완주 검증 통과: {checkpoints[-1].name}, TB step={steps[-1]}, "
        f"checkpoints={len(checkpoints)}"
    )
    return steps, values


def _write_selection(post_dir: Path, selection: list[dict]) -> None:
    _atomic_json(post_dir / "top5_selection.json", selection)
    lines = [
        "# Teacher checkpoint 상위 5개 — 학습 보상 선별", "",
        f"선별 기준: 저장 시점의 `{REWARD_TAG}` 내림차순", "",
        "| 후보 | checkpoint | iteration | Train/mean_reward |", "|---:|---|---:|---:|",
    ]
    for index, row in enumerate(selection, start=1):
        lines.append(
            f"| {index} | `{Path(row['checkpoint']).name}` | {row['iteration']} | "
            f"{row['train_mean_reward']:.6f} |"
        )
    (post_dir / "top5_selection.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _new_eval_dir(before: set[Path]) -> Path | None:
    candidates = {path.resolve() for path in RESULT_ROOT.glob("eval_*") if path.is_dir()} - before
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _run_eval(checkpoint: Path, episodes: int, seed: int, logger: Logger) -> Path:
    before = {path.resolve() for path in RESULT_ROOT.glob("eval_*") if path.is_dir()}
    command = [
        sys.executable, "-u", str(EVAL_SCRIPT),
        "--num_envs", str(episodes), "--n_episodes", str(episodes),
        "--seed", str(seed), "--headless", "--checkpoint", str(checkpoint),
    ]
    logger.emit("평가 실행: " + " ".join(command))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    process = subprocess.Popen(
        command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.subprocess_line(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"평가 실패({checkpoint.name}): exit={return_code}")
    eval_dir = _new_eval_dir(before)
    if (
        eval_dir is None
        or not (eval_dir / "eval_stats.json").is_file()
        or not (eval_dir / "episodes.csv").is_file()
    ):
        raise RuntimeError(f"평가 산출물 탐색 실패: {checkpoint.name}")
    return eval_dir


def _evaluation_row(candidate: dict, eval_dir: Path) -> dict:
    payload = _read_json(eval_dir / "eval_stats.json", {})
    runs = payload.get("runs", []) if isinstance(payload, dict) else []
    if len(runs) != 1:
        raise RuntimeError(f"eval_stats 형식 오류: {eval_dir}")
    metrics = runs[0]["metrics"]
    failure = runs[0].get("failure", {})
    dist = runs[0].get("dist", {})
    diagnostics = _episode_diagnostics(eval_dir / "episodes.csv")
    n_episodes = int(runs[0]["n_episodes"])
    if diagnostics["record_count"] != n_episodes:
        raise RuntimeError(
            f"평가 표본수 불일치: CSV={diagnostics['record_count']} stats={n_episodes}"
        )
    row = dict(candidate)
    row.update({
        "eval_dir": str(eval_dir),
        "n_episodes": n_episodes,
        "grasp_rate": float(metrics["grasp"]["rate"]),
        "carry_rate": float(metrics["carry"]["rate"]),
        "placed_rate": float(metrics["placed"]["rate"]),
        "placed_ci95": [float(value) for value in metrics["placed"]["ci95"]],
        "failure_counts": {str(key): int(value) for key, value in failure.items()},
        "distance_stats": dist,
        "episode_diagnostics": diagnostics,
    })
    row["strict_stage_analysis"] = _strict_stage_analysis(row)
    return row


def _write_ranking(post_dir: Path, ranking: list[dict]) -> None:
    _atomic_json(post_dir / "ranking.json", ranking)
    lines = [
        "# Teacher 상위 5개 정량평가 순위", "",
        "프로토콜: 512 episodes · seed 42 · pregrasp off · DR 유지. "
        "순위 기준: placed → carry → grasp.", "",
        "| 순위 | checkpoint | train reward | grasp | carry | placed | placed 95% CI |", 
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(ranking, start=1):
        lo, hi = row["placed_ci95"]
        lines.append(
            f"| {index} | `{Path(row['checkpoint']).name}` | {row['train_mean_reward']:.3f} | "
            f"{100*row['grasp_rate']:.2f}% | {100*row['carry_rate']:.2f}% | "
            f"{100*row['placed_rate']:.2f}% | [{100*lo:.2f}, {100*hi:.2f}] |"
        )
    (post_dir / "ranking.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_analysis(post_dir: Path, analysis: dict) -> None:
    _atomic_json(post_dir / "performance_analysis.json", analysis)
    lo, hi = analysis["placed_ci95"]
    losses = analysis["stage_losses_pp"]
    failure_rates = analysis["winner_failure_rates"]
    strict = analysis["winner_strict_stage_analysis"]
    target_text = "달성" if analysis["point_target_met"] else "미달"
    lower_text = "달성" if analysis["wilson_lower_target_met"] else "미달"
    if analysis["point_target_met"]:
        decision = (
            "Teacher point target을 달성했다. Phase2 42D 계약검증과 짧은 Vision 증류 smoke를 "
            "통과한 뒤 student 본 학습 설정을 확정한다."
        )
    else:
        decision = (
            "Teacher 95% point target 미달이다. Phase2 본 학습을 자동 시작하지 않고, 상위 5개 "
            "실패모드와 strict placed 조건을 먼저 분석한다."
        )
    lines = [
        "# Teacher 20k 성능 분석", "",
        f"- 채택 checkpoint: `model_{analysis['winner_iteration']}.pt`",
        f"- placed: **{100*analysis['placed_rate']:.2f}%** "
        f"(Wilson 95% CI [{100*lo:.2f}, {100*hi:.2f}])",
        f"- 목표 95% point: **{target_text}** · Wilson 하한 95%: **{lower_text}**",
        f"- F49 10k placed 92.38% 대비: **{analysis['delta_vs_f49_pp']:+.2f}%p**",
        f"- grasp→carry 손실: {losses['grasp_to_carry_pp']:.2f}%p",
        f"- carry→placed 손실: {losses['carry_to_placed_pp']:.2f}%p",
        f"- 주 병목: `{analysis['primary_bottleneck']}`",
        f"- 상위 5개 placed 범위 폭: {analysis['top5_placed_span_pp']:.2f}%p",
        "- 학습 보상 1위와 eval 1위 일치: "
        f"{'예' if analysis['train_reward_winner_matches_eval_winner'] else '아니오'}",
        "", "## 승자 실패모드", "",
    ]
    if failure_rates:
        labels = {
            "no_grasp": "파지 실패",
            "grasp_no_carry": "파지 후 배달 실패",
            "carry_no_place": "배달 후 안착 실패",
            "placed": "안착 성공",
        }
        for key in ("no_grasp", "grasp_no_carry", "carry_no_place", "placed"):
            if key in failure_rates:
                count = analysis["winner_failure_counts"][key]
                lines.append(
                    f"- {labels[key]}: {count}/{analysis['n_episodes']} "
                    f"({100*failure_rates[key]:.2f}%)"
                )
    else:
        lines.append("- 실패모드 집계 없음")
    lines += ["", "## Strict placed 조건 사다리", ""]
    if strict:
        rates = strict["rates"]
        lines.append(
            "- grasp → carry(8cm) → latch@3cm → settled@3cm → placed: "
            + " → ".join(f"{100*rates[key]:.2f}%" for key in rates)
        )
        lines.append(f"- 주 조건 병목: `{strict['primary_bottleneck']}`")
        for key, value in strict["losses_pp"].items():
            lines.append(f"- `{key}`: {value:.2f}%p")
        diagnostics = analysis["winner_episode_diagnostics"]
        zone = diagnostics.get("zone_medians", {})
        if diagnostics.get("zone_episode_count", 0):
            lines.append(
                f"- 3cm zone 에피소드: {diagnostics['zone_episode_count']}/{analysis['n_episodes']} · "
                f"큐브 최저속도 중앙 {zone.get('zone_min_obj_spd')} m/s · "
                f"팔 속도 중앙 {zone.get('zone_min_arm_qd')} rad/s"
            )
            open_rate = diagnostics.get("zone_open_episode_rate")
            if open_rate is not None:
                lines.append(f"- zone 내 open 명령 에피소드: {100*open_rate:.2f}%")
    else:
        lines.append("- strict 조건 CSV 집계 없음")
    lines += ["", "## 자동 판정", "", decision]
    (post_dir / "performance_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _verify_phase2_teacher_contract(checkpoint: Path, logger: Logger) -> None:
    """최종 승자가 Phase2 42D distillation 사전 게이트를 통과하는지 CPU 검증."""
    module_path = ROOT / "Phase2" / "VisionPick_Train.py"
    spec = importlib.util.spec_from_file_location("vision_pick_train_postcheck", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Phase2 사전검사 모듈 로드 실패: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module._verify_teacher_ckpt(str(checkpoint)):
        raise RuntimeError(f"Phase2 Teacher 42D 계약 불일치: {checkpoint}")
    logger.emit(f"Phase2 Teacher 42D 계약검증 통과: {checkpoint.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Teacher top-5 x seeds 42..45 x 512ep fail-closed adoption"
    )
    parser.add_argument("--run", required=True, help="학습 run 디렉터리")
    parser.add_argument("--max_iterations", type=int, default=25000)
    parser.add_argument("--wait_pid", type=int, default=0, help="종료를 기다릴 학습 래퍼 PID")
    parser.add_argument("--wait_create_time", type=float, default=None, help="PID 재사용 방지용 create_time")
    parser.add_argument("--poll_seconds", type=int, default=30)
    args = parser.parse_args()
    from scripts.adoption_pipeline import run_pipeline

    return run_pipeline(
        role="teacher",
        run=args.run,
        max_iterations=args.max_iterations,
        wait_pid=args.wait_pid,
        wait_create_time=args.wait_create_time,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
