#!/usr/bin/env python
"""eval episodes.csv → ★논문용 통계 산출물★ (성공률 신뢰구간·실패모드·결과표).

목적: BinPick_Eval이 남긴 에피소드별 CSV를 읽어 논문에 바로 넣을 통계를 만든다.
      (1) 성공률 + 95% 신뢰구간(Wilson score — 비율에 적합, 소표본 견고).
      (2) 실패모드 분해 — 각 에피소드가 어느 단계에서 멈췄는지(no-grasp / grasp만 /
          carry만 / placed) 분류해 병목을 정량화(막대그림 + 표).
      (3) 결과표 — UTF-8 text + LaTeX booktabs(논문 직접 삽입).
      (4) 다중 seed — eval 디렉토리 여러 개를 주면 pooled + seed별 mean±std.
실행 (Isaac 불필요 — pandas/numpy/matplotlib만):
  python analysis/eval_stats.py --eval-dir Phase1/Result/eval_<ts>
  python analysis/eval_stats.py --eval-dir <d_seed0> <d_seed1> <d_seed2> --label teacher
산출: 각 eval 디렉토리(또는 --out)에 results_table.txt / results_table.tex /
      failure_modes.{txt,png,pdf} / eval_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ── 색맹 안전(Okabe-Ito) — exp_report.py와 동일 팔레트 ──
_OKABE = {
    "no_grasp": "#999999",        # gray — 파지 자체 실패
    "grasp_no_carry": "#D55E00",  # vermillion — 잡았으나 운반 실패
    "carry_no_place": "#E69F00",  # orange — 운반했으나 안착 실패
    "placed": "#009E73",          # bluish green — 완전 성공
}
# 텍스트 표(사용자용)는 한국어, ★그림 텍스트는 영어★(논문 venue = 영문 + serif 폰트가
#   한글 미지원이라 □ 깨짐). 식별자·console은 규약대로 한국어 유지.
_STAGE_LABEL = {
    "no_grasp": "파지 실패",
    "grasp_no_carry": "파지 O·운반 X",
    "carry_no_place": "운반 O·안착 X",
    "placed": "완전 성공(placed)",
}
_STAGE_LABEL_EN = {
    "no_grasp": "grasp fail",
    "grasp_no_carry": "grasp, no carry",
    "carry_no_place": "carry, no place",
    "placed": "placed (success)",
}
# 공식 성공 지표(STAGE 게이트) — summary.txt와 동일 정의
_METRICS = [
    ("grasp", "grasp_success", "파지 성립(양손접촉∧정지)"),
    ("carry", "carry_latch_success", "배달(래치∧목표≤8cm)"),
    ("placed", "placed_success", "안착(릴리스·정지∧≤3cm)"),
]


def _apply_pub_style() -> None:
    plt.rcParams.update({
        "figure.figsize": (3.6, 2.6), "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "font.family": "serif", "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """이항 비율의 Wilson score 95% 신뢰구간 (n작아도 [0,1] 벗어나지 않음).

    정규근사(±z·√(p(1-p)/n))는 p가 0/1 근처나 소표본에서 구간이 경계를 벗어나
    논문 리뷰에서 지적받는다. Wilson은 그 문제를 구조적으로 피한다(표준 권장).
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def classify_failure(row: pd.Series) -> str:
    """에피소드를 도달한 최종 단계로 분류 (병목 정량화용, 상호배타)."""
    if int(row["placed_success"]) == 1:
        return "placed"
    if int(row["carry_latch_success"]) == 1:
        return "carry_no_place"
    if int(row["grasp_success"]) == 1:
        return "grasp_no_carry"
    return "no_grasp"


def analyze_one(csv_path: Path) -> dict:
    """단일 eval CSV → 지표별 성공수/총수 + 실패모드 카운트 + 거리 통계."""
    df = pd.read_csv(csv_path)
    n = len(df)
    out: dict = {"csv": str(csv_path), "n_episodes": n, "metrics": {}, "failure": {}, "dist": {}}
    for name, col, _desc in _METRICS:
        k = int(df[col].sum())
        lo, hi = wilson_ci(k, n)
        out["metrics"][name] = {"k": k, "n": n, "rate": k / n if n else float("nan"),
                                "ci95": [lo, hi]}
    cats = df.apply(classify_failure, axis=1)
    for cat in _OKABE:
        out["failure"][cat] = int((cats == cat).sum())
    # 파지시 목표거리(placed 성공 에피소드 기준 안착 정밀도)
    placed = df[df["placed_success"] == 1]["min_goal_dist_m"]
    if len(placed):
        out["dist"]["placed_min_goal_dist_m"] = {
            "mean": float(placed.mean()), "median": float(placed.median()),
            "p90": float(placed.quantile(0.9)),
        }
    return out


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}" if np.isfinite(x) else "—"


def write_result_table(stats_list: list[dict], out_dir: Path, label: str) -> None:
    """결과표 — UTF-8 text + LaTeX booktabs. 다중 seed면 mean±std(seed간)도 추가."""
    multi = len(stats_list) > 1
    # 지표별 pooled(전 에피소드 합산) + seed별 rate
    pooled_k = {m[0]: 0 for m in _METRICS}
    pooled_n = 0
    per_seed_rates: dict[str, list[float]] = {m[0]: [] for m in _METRICS}
    for st in stats_list:
        pooled_n_i = st["n_episodes"]
        pooled_n += pooled_n_i
        for name, _c, _d in _METRICS:
            pooled_k[name] += st["metrics"][name]["k"]
            per_seed_rates[name].append(st["metrics"][name]["rate"])

    md = [f"# 정량 평가 결과 — {label}", ""]
    md.append(f"- eval run 수: {len(stats_list)}" + (f" (seed간 mean±std 병기)" if multi else ""))
    md.append(f"- 총 에피소드: {pooled_n}" + (f" ({pooled_n // len(stats_list)}×{len(stats_list)})" if multi else ""))
    md.append("")
    hdr = "| 지표 | 성공률(%) | 95% CI | " + ("seed mean±std(%) | " if multi else "") + "N |"
    sep = "|---|---|---|" + ("---|" if multi else "") + "---|"
    md += [hdr, sep]
    tex_rows = []
    for name, col, desc in _METRICS:
        k, n = pooled_k[name], pooled_n
        lo, hi = wilson_ci(k, n)
        rate = k / n if n else float("nan")
        seed_cell = ""
        tex_seed = ""
        if multi:
            arr = np.array(per_seed_rates[name]) * 100
            seed_cell = f"{arr.mean():.1f}±{arr.std(ddof=1) if len(arr) > 1 else 0:.1f} | "
            tex_seed = f" & ${arr.mean():.1f}\\pm{arr.std(ddof=1) if len(arr) > 1 else 0:.1f}$"
        md.append(f"| {name} ({desc}) | {_fmt_pct(rate)} | [{_fmt_pct(lo)}, {_fmt_pct(hi)}] | {seed_cell}{n} |")
        tex_rows.append(f"{name} & {_fmt_pct(rate)} & $[{_fmt_pct(lo)},{_fmt_pct(hi)}]${tex_seed} & {n} \\\\")
    # 거리 정밀도(첫 run 기준 — 다중이면 pooled 근사)
    dists = [st["dist"].get("placed_min_goal_dist_m", {}).get("mean") for st in stats_list]
    dists = [d for d in dists if d is not None]
    if dists:
        md += ["", f"- 안착 정밀도(placed 성공 시 목표거리 평균): {100 * np.mean(dists):.1f} cm"]
    (out_dir / "results_table.txt").write_text("\n".join(md) + "\n", encoding="utf-8")

    # LaTeX (booktabs)
    seed_hdr = " & seed mean$\\pm$std" if multi else ""
    tex = [
        "% 자동 생성: analysis/eval_stats.py — 논문 본문에 \\input 또는 복붙",
        "\\begin{table}[t]", "\\centering",
        "\\caption{정량 평가 성공률 (Wilson 95\\% CI" + (", 다중 seed" if multi else "") + ").}",
        "\\label{tab:eval_success}", "\\begin{tabular}{lccc" + ("c" if multi else "") + "}",
        "\\toprule",
        "Metric & Success (\\%) & 95\\% CI" + seed_hdr + " & N \\\\",
        "\\midrule",
        *tex_rows,
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    (out_dir / "results_table.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"[stats] 결과표 저장: results_table.txt / results_table.tex")


def plot_failure_modes(stats_list: list[dict], out_dir: Path, label: str) -> None:
    """실패모드 분해 — 단계별 에피소드 비율 누적 막대 + 표."""
    agg = {cat: sum(st["failure"][cat] for st in stats_list) for cat in _OKABE}
    total = sum(agg.values())
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(3.6, 1.6))
    left = 0.0
    order = ["no_grasp", "grasp_no_carry", "carry_no_place", "placed"]
    for cat in order:
        frac = agg[cat] / total if total else 0
        ax.barh(0, frac, left=left, color=_OKABE[cat], edgecolor="white", linewidth=1.2,
                label=f"{_STAGE_LABEL_EN[cat]} ({100 * frac:.0f}%)")
        if frac > 0.06:
            ax.text(left + frac / 2, 0, f"{100 * frac:.0f}%", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
        left += frac
    ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_xlabel("episode fraction"); ax.set_title(f"stage reached / failure breakdown — {label}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.6), ncol=2, fontsize=6.5)
    for suf in (".pdf", ".png"):
        fig.savefig(out_dir / f"failure_modes{suf}")
    plt.close(fig)

    md = [f"# 실패모드 분해 — {label}", "", f"- 총 에피소드: {total}", "",
          "| 도달 단계 | 에피소드 | 비율(%) |", "|---|---|---|"]
    for cat in order:
        md.append(f"| {_STAGE_LABEL[cat]} | {agg[cat]} | {100 * agg[cat] / total if total else 0:.1f} |")
    md += ["", "해석: '파지 O·운반 X'가 크면 파지 유지/들어올림이 병목, '운반 O·안착 X'가",
           "크면 릴리스·안착 정밀도가 병목."]
    (out_dir / "failure_modes.txt").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[stats] 실패모드 저장: failure_modes.{{pdf,png,txt}}")


def main() -> int:
    ap = argparse.ArgumentParser(description="eval CSV → 논문 통계 산출물.")
    ap.add_argument("--eval-dir", nargs="+", required=True,
                    help="eval 디렉토리(들). 여러 개면 다중 seed로 집계.")
    ap.add_argument("--out", default=None, help="산출 위치(기본: 첫 eval-dir).")
    ap.add_argument("--label", default="teacher", help="표/그림 라벨.")
    a = ap.parse_args()

    stats_list = []
    for d in a.eval_dir:
        csv = Path(d) / "episodes.csv"
        if not csv.is_file():
            print(f"[stats][warn] episodes.csv 없음: {csv} — 건너뜀")
            continue
        stats_list.append(analyze_one(csv))
    if not stats_list:
        print("[stats] 유효한 eval CSV 없음"); return 1

    out_dir = Path(a.out) if a.out else Path(a.eval_dir[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_result_table(stats_list, out_dir, a.label)
    plot_failure_modes(stats_list, out_dir, a.label)
    (out_dir / "eval_stats.json").write_text(
        json.dumps({"label": a.label, "runs": stats_list}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"[stats] ★완료★ {out_dir} (results_table·failure_modes·eval_stats.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
