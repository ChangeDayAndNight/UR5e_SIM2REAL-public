#!/usr/bin/env python
"""Phase 3 게이트 5 — 관측 분포 패리티 검증. 순수 numpy — 아무 PC/env에서나 실행.

목적: sim 리허설 덤프(게이트 4 산출)와 실기 dry-run 덤프(Phase4 Deploy_DryRun,
      dry_run=True — 로봇 명령 미전송)의 '전처리 후 정책 관측'을 레이아웃 슬롯별로
      비교해, sim2real 갭을 실기 구동 '전에' 수치로 드러낸다.
실행: python deployment\\Verify_ObsParity.py --sim-dir Phase3\\logs\\obs_dump_sim --real-dir <실기 덤프>
산출: deployment/logs/obs_parity_<시각>.csv       (슬롯별 통계 + 괴리 판정)
      deployment/logs/obs_parity_hist_<시각>.csv  (슬롯별 히스토그램 — 그래프 없이 표로)

★원칙(절대 규칙): 분포 괴리가 나와도 common/obs_preprocess.py를 sim/real로 분기하지
  않는다 — 공유 전처리 단일 코드 경로가 요소1의 계약이며, 분기하는 순간 '학습 관측 =
  배포 관측'의 증명이 무너진다. 고치는 곳은 언제나 '입력측'뿐이다:
    - real 쪽 원인(센서): configs/real.local.yaml (camera.enable_filters 등) 조정
    - sim 쪽 원인(모사 부족): configs/dr.yaml (depth_noise/lighting/latency 범위)
      조정 → Phase2 재학습 → 게이트 4부터 재실행★

입력 형식: 각 디렉터리에서 step_*.npz를 '재귀'로 수집 (ep###/ 하위 구조 허용).
  npz 키 계약(common/rollout.py): obs [N, obs_dim] — N축·스텝축을 모두 표본으로 합친다.
판정 규칙(시작값 — CLI로 조정):
  - mean gap: |mean_sim − mean_real| / pooled_std ≤ --mean-gate (기본 0.5)
  - hist 교집합: Σ min(p_sim, p_real) ≥ --hist-gate (기본 0.5)
  두 조건 모두 만족해야 슬롯 PASS, 전 슬롯 PASS여야 게이트 통과(exit 0).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
# Canonical source와 build_bundle.py가 만든 hash bundle은 모두 deployment/의 부모를
# root로 사용한다. 생성 bundle의 사본 드리프트는 bundle_manifest.json 검증이 차단한다.
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config_io import build_preprocess_cfg, configs_dir, load_yaml  # noqa: E402
from common.obs_preprocess import obs_layout  # noqa: E402  (레이아웃 = 계약서 본문)
from deployment.contracts import load_real_config  # noqa: E402

QUANTILES = (5, 25, 50, 75, 95)


# =====================================================================
# 로드 / 슬라이스 (완전 구현)
# =====================================================================


def load_dump_dir(dump_dir: Path) -> np.ndarray:
    """step_*.npz 재귀 수집 → obs 행렬 [총 표본 수, obs_dim]."""
    files = sorted(dump_dir.rglob("step_*.npz"))
    if not files:
        raise FileNotFoundError(f"step_*.npz 없음: {dump_dir}")
    mats = []
    for fp in files:
        with np.load(fp) as z:
            if "obs" not in z:
                raise KeyError(f"{fp}: 'obs' 키 없음 (rollout 덤프 계약 위반)")
            obs = np.asarray(z["obs"], dtype=np.float64)
        if obs.ndim == 1:
            obs = obs[None, :]
        mats.append(obs)
    mat = np.concatenate(mats, axis=0)
    if mat.ndim != 2 or not np.isfinite(mat).all():
        raise ValueError(f"non-finite or invalid observation matrix: {dump_dir}")
    print(f"[parity] {dump_dir}: 파일 {len(files)}개 → 표본 {mat.shape[0]} × dim {mat.shape[1]}")
    return mat


def slot_slices(layout: list[tuple[str, int]]) -> list[tuple[str, slice]]:
    """(이름, 크기) 레이아웃 → (이름, flat 슬라이스) 목록."""
    out, offset = [], 0
    for name, size in layout:
        out.append((name, slice(offset, offset + size)))
        offset += size
    return out


# =====================================================================
# 통계 / 판정 (완전 구현)
# =====================================================================


def slot_stats(values: np.ndarray) -> dict:
    """슬롯 값(전 표본×전 요소 flatten) → 기본 통계 dict."""
    v = values.ravel()
    qs = np.percentile(v, QUANTILES)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        **{f"p{q}": float(x) for q, x in zip(QUANTILES, qs)},
    }


def histogram_pair(
    v_sim: np.ndarray, v_real: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """공유 범위 히스토그램 → (bin 경계, sim 비율, real 비율, 교집합[0..1])."""
    v_sim, v_real = v_sim.ravel(), v_real.ravel()
    lo = min(float(v_sim.min()), float(v_real.min()))
    hi = max(float(v_sim.max()), float(v_real.max()))
    if hi <= lo:  # 상수 슬롯 (예: 고정 target z) — 분포 비교 무의미, 완전 일치 취급
        edges = np.array([lo, lo + 1e-12])
        return edges, np.array([1.0]), np.array([1.0]), 1.0
    edges = np.linspace(lo, hi, bins + 1)
    f_sim = np.histogram(v_sim, bins=edges)[0].astype(np.float64)
    f_real = np.histogram(v_real, bins=edges)[0].astype(np.float64)
    f_sim /= f_sim.sum()
    f_real /= f_real.sum()
    return edges, f_sim, f_real, float(np.minimum(f_sim, f_real).sum())


def judge_slot(
    s_sim: dict, s_real: dict, hist_inter: float, mean_gate: float, hist_gate: float
) -> tuple[bool, float]:
    """슬롯 괴리 판정 → (pass 여부, mean gap [pooled σ 단위])."""
    pooled = max(np.sqrt((s_sim["std"] ** 2 + s_real["std"] ** 2) / 2.0), 1e-6)
    gap_sigma = abs(s_sim["mean"] - s_real["mean"]) / pooled
    return (gap_sigma <= mean_gate and hist_inter >= hist_gate), float(gap_sigma)


# =====================================================================
# 메인 게이트
# =====================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="sim↔real 관측 분포 패리티 게이트 (게이트 5)")
    parser.add_argument("--sim-dir", type=Path, required=True, help="sim 덤프 루트 (게이트 4 산출)")
    parser.add_argument("--real-dir", type=Path, required=True, help="real 덤프 루트 (Deploy_DryRun 산출)")
    parser.add_argument("--real-config", type=Path, required=True,
                        help="the reviewed real.local.yaml used to create the real dump")
    parser.add_argument("--out-dir", type=Path, default=HERE / "logs")
    parser.add_argument("--bins", type=int, default=20, help="히스토그램 bin 수")
    parser.add_argument("--mean-gate", type=float, default=0.5, help="평균 괴리 허용 [pooled σ 단위]")
    parser.add_argument("--hist-gate", type=float, default=0.5, help="히스토그램 교집합 하한 [0..1]")
    args = parser.parse_args()
    load_real_config(args.real_config)

    pre_cfg = build_preprocess_cfg(load_yaml(configs_dir() / "common.yaml"))
    layout = obs_layout(pre_cfg)
    expected_dim = sum(s for _, s in layout)

    try:
        mat_sim = load_dump_dir(args.sim_dir)
        mat_real = load_dump_dir(args.real_dir)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[parity] ⚠️ 덤프 로드 실패: {e}")
        return 2
    for name, mat in (("sim", mat_sim), ("real", mat_real)):
        if mat.shape[1] != expected_dim:
            print(f"[parity] ⚠️ {name} obs_dim {mat.shape[1]} ≠ 레이아웃 기대 {expected_dim} — "
                  f"configs/common.yaml이 덤프 시점과 다르거나 덤프가 손상됨. 중단.")
            return 2

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.out_dir / f"obs_parity_{ts}.csv"
    hist_path = args.out_dir / f"obs_parity_hist_{ts}.csv"
    report_path = args.out_dir / f"obs_parity_{ts}.json"

    stat_rows, hist_rows = [], []
    all_pass = True
    for name, sl in slot_slices(layout):
        v_sim, v_real = mat_sim[:, sl], mat_real[:, sl]
        s_sim, s_real = slot_stats(v_sim), slot_stats(v_real)
        edges, f_sim, f_real, inter = histogram_pair(v_sim, v_real, args.bins)
        ok, gap_sigma = judge_slot(s_sim, s_real, inter, args.mean_gate, args.hist_gate)
        all_pass &= ok

        row = {"slot": name, "verdict": "PASS" if ok else "DIVERGE",
               "mean_gap_sigma": round(gap_sigma, 4), "hist_intersection": round(inter, 4)}
        for src, s in (("sim", s_sim), ("real", s_real)):
            for k, v in s.items():
                row[f"{src}_{k}"] = round(v, 6) if isinstance(v, float) else v
        stat_rows.append(row)

        for b in range(len(f_sim)):
            hist_rows.append(
                {"slot": name, "bin_lo": round(float(edges[b]), 6),
                 "bin_hi": round(float(edges[b + 1]), 6),
                 "frac_sim": round(float(f_sim[b]), 6),
                 "frac_real": round(float(f_real[b]), 6)}
            )

        mark = "PASS  " if ok else "★괴리★"
        print(f"[parity] {mark} {name:<12} gap={gap_sigma:5.2f}σ hist∩={inter:.2f} "
              f"mean {s_sim['mean']:+.4f}/{s_real['mean']:+.4f} "
              f"std {s_sim['std']:.4f}/{s_real['std']:.4f} (sim/real)")

    with open(stats_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(stat_rows[0].keys()))
        w.writeheader()
        w.writerows(stat_rows)
    with open(hist_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)
    report = {
        "schema_version": 1,
        "kind": "observation_parity_raw",
        "passed": bool(all_pass),
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "sim_samples": int(mat_sim.shape[0]),
        "real_samples": int(mat_real.shape[0]),
        "observation_dim": int(expected_dim),
        "mean_gate_sigma": float(args.mean_gate),
        "histogram_intersection_gate": float(args.hist_gate),
        "slots": stat_rows,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[parity] 통계: {stats_path}")
    print(f"[parity] 히스토그램: {hist_path}")
    print(f"[parity] JSON report: {report_path}")
    if all_pass:
        print("[parity] 게이트 5: ★전 슬롯 PASS★ — 관측 parity만 충족; motion authorization은 계속 차단")
    else:
        print("[parity] 게이트 5: ⚠️ 괴리 슬롯 존재 — ★전처리(common/obs_preprocess.py) 분기 금지★. "
              "입력측을 고칠 것: real 원인→configs/real.local.yaml(필터 등), "
              "sim 원인→configs/dr.yaml(노이즈/지연 범위)→재학습→게이트 4부터 재실행.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
