#!/usr/bin/env python
"""실험 결과 → ★논문(Isaac Sim/Robotics) 바로 사용 가능한 산출물★ 생성기.

목적: 모든 학습/평가 run에서 논문에 그대로 넣을 수 있는 산출물을 남긴다.
      (1) 학습 곡선 — 벡터 PDF(LaTeX용) + PNG(미리보기), 색맹 안전(Okabe-Ito) 팔레트,
          다중 seed면 mean±std 밴드. (2) 보상항 분해 곡선. (3) 재현 정보·하이퍼파라미터·
          핵심 지표 요약(summary.txt + summary.json). (4) ablation 비교 그림.
실행 (isaac51 또는 matplotlib/pandas/tensorboard 있는 아무 env — ★Isaac 불필요★):
  python analysis/exp_report.py                                  # 최신 teacher run
  python analysis/exp_report.py --run <run_dir>
  python analysis/exp_report.py --runs <d1> <d2> --labels seed0 seed1        # mean±std(같은 라벨)
  python analysis/exp_report.py --runs <d1> <d2> --labels "밀기허용" "수정본" --tag ablation  # 비교 그림
산출: ★Phase<N>/Result/<name>/★ (teacher→Phase1, vision→Phase2 — 사용자 결정 2026-07-10,
      루트 results/ 폐지)에 learning_curve.{pdf,png} / reward_terms.{pdf,png}
      + weights/model_<final>.pt(최종 가중치 사본)
      / metrics.csv / summary.{txt,json}

★논문 재현성★: summary에 git hash·seed·num_envs·sim.dt·decimation·하이퍼파라미터·
  재현 명령을 기록한다. PDF는 fonttype 42(편집가능 TrueType — IEEE/ACM 요건).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")  # headless 렌더
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # 프로젝트 루트
DEFAULT_EXPERIMENT = "ur5e_binpick_teacher"

# ★산출물 위치 = Phase<N>/Result/ (사용자 결정 2026-07-10)★ — 루트 results/ 폐지.
#   실험별 산출물(논문 그림·요약·진척영상·최종 가중치 사본)을 해당 Phase 폴더에 정리한다.
_EXPERIMENT_PHASE = {
    "ur5e_binpick_teacher": "Phase1",
    "ur5e_binpick_vision": "Phase2",
}


def phase_result_root(experiment: str) -> Path:
    """실험 → 해당 Phase의 Result 루트 (예: ur5e_binpick_teacher → Phase1/Result)."""
    return ROOT / _EXPERIMENT_PHASE.get(experiment, "Phase1") / "Result"

# ── TB 스칼라 태그 (rsl_rl) ──
TAG_REWARD = "Train/mean_reward"
TAG_EP_LEN = "Train/mean_episode_length"
REWARD_TERMS = ["reach", "grasp", "lift", "place", "placed"]
TAG_TERM = {t: f"Episode_Reward/{t}" for t in REWARD_TERMS}

# ── 색맹 안전 팔레트 (Okabe-Ito) — scripts/validate_palette.js 통과(ΔE≥17, light) ──
TERM_COLOR = {
    "reach": "#0072B2",   # blue
    "grasp": "#D55E00",   # vermillion
    "lift": "#009E73",    # bluish green
    "place": "#CC79A7",   # reddish purple
    "placed": "#E69F00",  # orange
}
ABLATION_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def _apply_pub_style() -> None:
    """논문 그림 스타일 — 단컬럼(3.6in), serif, 얇은 마크, 상/우 spine 제거, PDF 편집가능."""
    plt.rcParams.update({
        "figure.figsize": (3.6, 2.6),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "serif", "mathtext.fontset": "dejavuserif",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "lines.linewidth": 1.6,
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType 임베드(편집가능) — 논문 venue 요건
    })


# ═════════════════════════════════════════════════════════════════════════════
# 데이터 로드
# ═════════════════════════════════════════════════════════════════════════════
def load_scalars(run_dir: Path) -> pd.DataFrame:
    """run 폴더의 TB 이벤트 → DataFrame(index=iteration, columns=태그). 없으면 빈 DF."""
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return pd.DataFrame()
    data: dict[int, dict[str, float]] = {}
    for tag in tags:
        if tag.endswith("/time"):
            continue  # rsl_rl의 벽시계-indexed 중복(step=초) — index 오염 방지, 논문 불필요
        for ev in ea.Scalars(tag):
            data.setdefault(ev.step, {})[tag] = ev.value
    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df.index.name = "iteration"
    return df


def _git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def load_meta(run_dir: Path) -> dict:
    """params/agent.yaml·env.yaml에서 재현 정보·하이퍼파라미터 추출 (best-effort)."""
    # 리포트가 학습 종료 후 생성되는 동안 HEAD가 바뀔 수 있다. 현재 HEAD를 학습 코드로
    # 오기하지 않고, BinPick_Train이 시작 전에 캡처한 run_manifest를 최우선으로 쓴다.
    report_git_hash = _git_hash()
    meta: dict = {
        "run_dir": str(run_dir), "run_name": run_dir.name,
        "git_hash": report_git_hash, "report_git_hash": report_git_hash,
    }
    try:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        training_hash = str(manifest.get("training_git_hash_short") or
                            manifest.get("training_git_hash") or "unknown")
        meta["git_hash"] = training_hash
        meta["training_git_hash_full"] = manifest.get("training_git_hash")
        meta["training_started_at"] = manifest.get("training_started_at")
        meta["training_completed_at"] = manifest.get("training_completed_at")
        meta["training_command"] = manifest.get("command")
        meta["training_return_code"] = manifest.get("return_code")
    except Exception:
        meta["run_manifest_missing"] = True
    p = run_dir / "params"
    try:
        agent = yaml.safe_load((p / "agent.yaml").read_text(encoding="utf-8"))
        meta["seed"] = agent.get("seed")
        meta["experiment_name"] = agent.get("experiment_name")
        meta["max_iterations"] = agent.get("max_iterations")
        alg = agent.get("algorithm", {})
        pol = agent.get("policy", {})
        meta["hyperparams"] = {
            "learning_rate": alg.get("learning_rate"), "gamma": alg.get("gamma"),
            "lam": alg.get("lam"), "clip_param": alg.get("clip_param"),
            "entropy_coef": alg.get("entropy_coef"), "num_steps_per_env": agent.get("num_steps_per_env"),
            "actor_hidden_dims": pol.get("actor_hidden_dims"), "activation": pol.get("activation"),
        }
    except Exception as e:
        meta["agent_yaml_error"] = str(e)
    # env.yaml은 !!python/tuple·func 참조 태그가 있어 safe_load 불가 → regex로 추출(견고).
    try:
        import re
        txt = (p / "env.yaml").read_text(encoding="utf-8")

        def _grab(pat, cast):
            m = re.search(pat, txt)
            return cast(m.group(1)) if m else None

        meta["num_envs"] = _grab(r"\bnum_envs:\s*([0-9]+)", int)
        meta["decimation"] = _grab(r"\bdecimation:\s*([0-9]+)", int)
        meta["sim_dt"] = _grab(r"\n\s*dt:\s*([0-9.eE+-]+)", float)  # sim.dt (첫 dt)
        meta["episode_length_s"] = _grab(r"\bepisode_length_s:\s*([0-9.eE+-]+)", float)
    except Exception:
        pass
    return meta


# ═════════════════════════════════════════════════════════════════════════════
# 그림
# ═════════════════════════════════════════════════════════════════════════════
def _aligned_mean_std(dfs: list[pd.DataFrame], tag: str):
    """여러 seed의 같은 태그 → 공통 iteration에서 (index, mean, std)."""
    series = [d[tag].dropna() for d in dfs if tag in d.columns]
    if not series:
        return None
    mat = pd.concat(series, axis=1)
    return mat.index.to_numpy(), mat.mean(axis=1).to_numpy(), mat.std(axis=1, ddof=0).to_numpy()


def plot_learning_curve(groups: dict[str, list[pd.DataFrame]], out_base: Path, title: str) -> None:
    """Mean reward vs iteration. 라벨별 mean±std 밴드 + (라벨 2개 이상이면) 범례."""
    fig, ax = plt.subplots()
    for i, (label, dfs) in enumerate(groups.items()):
        res = _aligned_mean_std(dfs, TAG_REWARD)
        if res is None:
            continue
        x, mean, std = res
        c = ABLATION_COLORS[i % len(ABLATION_COLORS)]
        ax.plot(x, mean, color=c, label=label)
        if np.any(std > 0):
            ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18, linewidth=0)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Episode return (mean)")
    if title:
        ax.set_title(title)
    if len(groups) > 1:
        ax.legend(loc="best")
    _save(fig, out_base)


def plot_reward_terms(df: pd.DataFrame, out_base: Path, title: str) -> None:
    """보상항 분해(reach/grasp/lift/place/placed) — 색맹안전 색 + ★직접 끝라벨(저대비 relief)★ + 범례."""
    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    xmax = 1
    for term in REWARD_TERMS:
        tag = TAG_TERM[term]
        if tag not in df.columns:
            continue
        s = df[tag].dropna()
        if not len(s):
            continue
        ax.plot(s.index, s.to_numpy(), color=TERM_COLOR[term], label=term)
        xmax = max(xmax, int(s.index[-1]))
        # 직접 끝라벨은 저대비 색(place=#CC79A7·placed=#E69F00, 검증 WARN)에만 —
        # relief 요건 충족 + 0 근처 reach/grasp/lift 라벨 겹침 회피(나머지는 범례로 식별).
        if term in ("place", "placed"):
            ax.annotate(term, xy=(s.index[-1], s.iloc[-1]), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7,
                        color=TERM_COLOR[term])
    ax.set_xlim(right=xmax * 1.10)  # 끝라벨 공간
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Reward-term contribution")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)
    _save(fig, out_base)


def _save(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"))
    plt.close(fig)
    print(f"[report] 그림 저장: {out_base.name}.{{pdf,png}}")


# ═════════════════════════════════════════════════════════════════════════════
# 요약 (md + json)
# ═════════════════════════════════════════════════════════════════════════════
def _key_metrics(df: pd.DataFrame) -> dict:
    if TAG_REWARD not in df.columns or df.empty:
        return {}
    r = df[TAG_REWARD].dropna()
    m = {
        "iterations": int(df.index.max()) + 1 if len(df.index) else 0,
        "final_return": float(r.iloc[-1]), "best_return": float(r.max()),
        "best_iter": int(r.idxmax()), "last20_mean_return": float(r.iloc[-20:].mean()),
    }
    for term in REWARD_TERMS:
        tag = TAG_TERM[term]
        if tag in df.columns:
            m[f"final_{term}"] = float(df[tag].dropna().iloc[-1])
    return m


def write_summary(out_dir: Path, meta: dict, metrics: dict, task: str, video_dir: str | None) -> None:
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # ★실험 정체(BINPICK_* 게이트)는 기계 판독용 json에도 기록★ — ablation 자동 비교의 전제
    _env_flags = {k: v for k, v in sorted(os.environ.items()) if k.startswith("BINPICK_")}
    summary = {"generated_at": stamp, "task": task, **meta,
               "binpick_env_flags": _env_flags, "metrics": metrics}
    if video_dir:
        summary["behavior_video_dir"] = video_dir
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # UTF-8 text (논문 표/부록의 사람이 읽는 요약)
    hp = meta.get("hyperparams", {})
    lines = [
        f"# 실험 요약 — {task}", "",
        f"- 생성 시각: {stamp}",
        f"- run: `{meta.get('run_name')}`  · git `{meta.get('git_hash')}`  · seed `{meta.get('seed')}`",
        f"- num_envs `{meta.get('num_envs')}` · sim.dt `{meta.get('sim_dt')}` · decimation `{meta.get('decimation')}`"
        f" · episode `{meta.get('episode_length_s')}`s · num_steps_per_env `{hp.get('num_steps_per_env')}`",
        "",
        # ★2026-08-06(감사 P7-3)★ 보상값 비교 함정을 산출물 자체에 새겨 둔다.
        "> ⚠️ **보상값 비교 주의**: Isaac Lab은 `Episode_Reward/*`를 "
        "`episode_sum / max_episode_length_s`로 나눠 기록한다"
        "(`managers/reward_manager.py`). 따라서 **`episode_length_s`가 다른 run끼리 "
        "보상값을 직접 비교하면 안 된다** — 같은 물리 거동이 길이 비율만큼 다른 값으로 찍힌다. "
        "성능 비교는 반드시 `BinPick_Eval`의 **성공률**로 할 것.",
        "",
        "## 핵심 지표",
        "| 지표 | 값 |", "|---|---|",
    ]
    for k, v in metrics.items():
        lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    lines += ["", "## 하이퍼파라미터 (rsl_rl PPO)", "| 항목 | 값 |", "|---|---|"]
    for k, v in hp.items():
        lines.append(f"| {k} | {v} |")
    # ★2026-07-28 실험 정체(identity) 기록★ — 감사 지적 2건 해소:
    #   ① BINPICK_* 환경변수가 어디에도 기록되지 않아 ★어느 run이 어떤 카드였는지 증명 불가★였다
    #      (b6의 정체가 'BINPICK_AUG=1'인데 산출물만 보고는 구분할 수 없다).
    #   ② 재현 명령이 Phase 무관하게 Phase1\BinPick_Train.py로 하드코딩돼 vision run에 거짓이었다.
    env_flags = {k: v for k, v in sorted(os.environ.items()) if k.startswith("BINPICK_")}
    lines += ["", "## 실험 조건(BINPICK_* 게이트)", "| 변수 | 값 |", "|---|---|"]
    if env_flags:
        for k, v in env_flags.items():
            lines.append(f"| `{k}` | `{v}` |")
    else:
        lines.append("| (없음) | 기본 설정 — 게이트 미사용 |")

    _is_vision = "vision" in str(meta.get("run_dir", "")) or "vision" in str(out_dir)
    _script = "Phase2\\VisionPick_Train.py" if _is_vision else "Phase1\\BinPick_Train.py"
    _extra = " --headless --enable_cameras" if _is_vision else " --headless"
    _envline = ("  (환경변수: " + " ".join(f"{k}={v}" for k, v in env_flags.items()) + ")\n"
                if env_flags else "")
    lines += [
        "", "## 재현",
        f"```\n{_envline}python {_script} --num_envs {meta.get('num_envs')} "
        f"--max_iterations {meta.get('max_iterations')}{_extra}\n```",
        "",
        "⚠️ 위 환경변수가 이 run의 ★실험 정체★다 — 재현 시 반드시 함께 설정할 것.",
    ]
    if video_dir:
        lines += ["", f"## 거동 영상\n`{video_dir}` (재생: `python Phase1\\BinPick_Env.py`)"]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] 요약 저장: summary.txt / summary.json")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def find_latest_run(experiment: str) -> Path | None:
    store = ROOT / "logs" / "rsl_rl" / experiment
    if not store.is_dir():
        return None
    runs = [d for d in store.iterdir() if d.is_dir() and not d.name.startswith("_")]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def main() -> int:
    ap = argparse.ArgumentParser(description="실험 결과 → 논문급 산출물 생성")
    ap.add_argument("--run", type=str, help="단일 run 폴더 (기본: 최신 teacher run)")
    ap.add_argument("--runs", nargs="+", help="여러 run 폴더 (mean±std 또는 비교)")
    ap.add_argument("--labels", nargs="+", help="--runs 각각의 라벨 (같은 라벨=seed 집계)")
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--tag", default=None, help="출력 폴더 이름 접미 (예: ablation)")
    ap.add_argument("--title", default="", help="그림 제목(옵션)")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: Phase<N>/Result/<name>)")
    args = ap.parse_args()

    # 입력 run 목록 + 라벨 구성
    if args.runs:
        run_dirs = [Path(r) for r in args.runs]
        labels = args.labels or [Path(r).name for r in args.runs]
        if len(labels) != len(run_dirs):
            print("[report][error] --labels 개수가 --runs와 다릅니다.")
            return 1
        name = args.tag or "compare_" + "_".join(dict.fromkeys(labels))
    else:
        run = Path(args.run) if args.run else find_latest_run(args.experiment)
        if run is None or not run.is_dir():
            print(f"[report][error] run 폴더를 찾지 못했습니다: {args.run or '(최신 탐색 실패)'}")
            return 1
        run_dirs, labels, name = [run], [run.name], (args.tag or run.name)

    # 라벨별 그룹핑
    groups: dict[str, list[pd.DataFrame]] = {}
    per_run_df: list[tuple[Path, pd.DataFrame]] = []
    for rd, lb in zip(run_dirs, labels):
        df = load_scalars(rd)
        if df.empty:
            print(f"[report][warn] TB 스칼라 없음, 건너뜀: {rd}")
            continue
        groups.setdefault(lb, []).append(df)
        per_run_df.append((rd, df))
    if not groups:
        print("[report][error] 유효한 run이 없습니다.")
        return 1

    out_dir = Path(args.out) if args.out else phase_result_root(args.experiment) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _apply_pub_style()

    # ★모델 가중치 사본(사용자 결정 2026-07-10)★ — 최종 ckpt를 Result에 함께 보관
    #   (logs/는 작업 디렉토리·gitignore라 유실 위험, Result가 Phase별 보존 위치).
    try:
        import re as _re
        import shutil as _shutil
        ck = sorted(
            Path(run_dirs[0]).glob("model_*.pt"),
            key=lambda p: int(_re.findall(r"\d+", p.stem)[-1]),
        )
        if ck:
            wdir = out_dir / "weights"
            wdir.mkdir(exist_ok=True)
            _shutil.copy2(ck[-1], wdir / ck[-1].name)
            print(f"[report] 최종 가중치 사본 → {wdir / ck[-1].name}")
    except Exception as e:  # 가중치 복사는 보조 — 실패해도 리포트 진행
        print(f"[report][warn] 가중치 사본 실패: {e}")

    # 학습 곡선 (라벨별 mean±std / 비교)
    plot_learning_curve(groups, out_dir / "learning_curve", args.title)

    # 단일 run(또는 첫 run): 보상항 분해 + 요약 + metrics.csv
    rd0, df0 = per_run_df[0]
    plot_reward_terms(df0, out_dir / "reward_terms", args.title)
    df0.to_csv(out_dir / "metrics.csv", encoding="utf-8-sig")
    meta = load_meta(rd0)
    vids = ROOT / "Phase1" / "logs" / "videos"
    write_summary(out_dir, meta, _key_metrics(df0),
                  f"Isaac-BinPick-UR5e ({args.experiment})",
                  str(vids) if vids.is_dir() else None)

    print(f"[report] ★완료★ → {out_dir}")
    print(f"[report] 논문용: learning_curve.pdf(벡터) + reward_terms.pdf + summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
