#!/usr/bin/env python
"""★NuRec 수치 정합★ — 랜드마크 대응점(Umeyama) → scene_registration.yaml.

configs/calib/nurec_landmarks.yaml 의 대응점 쌍(p_nurec ↔ p_base)으로
common.calibration.umeyama_similarity 를 돌려 (T_base_nurec, scale, rms)를 산출한다.
순수 python(numpy/yaml) — Isaac 불필요, 어느 PC에서든 즉시 실행 가능.

실행:
  python scripts/tools/register_nurec_umeyama.py             # dry-run (표만 출력)
  python scripts/tools/register_nurec_umeyama.py --write     # yaml 기록 (measured=true 필수)

산출:
  - 콘솔: 점별 잔차 표 + T/scale/rms + 기존 scene_registration.yaml 대비 차이(검수용)
  - --write 시: configs/calib/scene_registration.yaml 갱신
    (method='umeyama_landmarks', residual_m=rms — §4-5: 생성물이지만 반드시 커밋)

★게이트★: rms > 0.01 m(벨트 배치 허용오차 1cm)이면 경고 — NuRec 파인튠 진입 금지
(scene_registration.yaml 헤더 계약과 동일 기준).
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.calibration import (  # noqa: E402
    load_scene_registration,
    save_scene_registration,
    umeyama_similarity,
)
from common.config_io import configs_dir  # noqa: E402

RMS_GATE_M = 0.01  # 벨트 배치 허용오차 1cm — scene_registration.yaml 헤더 게이트와 동일


def _load_landmarks(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, dict]:
    """nurec_landmarks.yaml → (names, src=p_nurec (N,3), dst=p_base (N,3), meta)."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pairs = data.get("pairs") or []
    if len(pairs) < 3:
        raise ValueError(f"{path}: 대응점 {len(pairs)}쌍 — Umeyama에는 비공선 3쌍 이상 필요")
    names, src, dst = [], [], []
    for i, p in enumerate(pairs):
        try:
            names.append(str(p.get("name", f"pair_{i}")))
            src.append([float(v) for v in p["p_nurec"]])
            dst.append([float(v) for v in p["p_base"]])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path}: pairs[{i}] 형식 오류 (p_nurec/p_base [x,y,z] 필요): {e}") from e
    return names, np.asarray(src), np.asarray(dst), data.get("meta") or {}


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """두 회전행렬 사이 각도[deg] = angle(Ra^T @ Rb)."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _print_matrix(T: np.ndarray) -> None:
    for row in T:
        print("[calib]   [" + ", ".join(f"{v:+.6f}" for v in row) + "]")


def main() -> int:
    parser = argparse.ArgumentParser(description="NuRec 랜드마크 Umeyama 정합 → scene_registration.yaml")
    parser.add_argument("--landmarks", type=Path, default=configs_dir() / "calib" / "nurec_landmarks.yaml",
                        help="대응점 yaml (기본: configs/calib/nurec_landmarks.yaml)")
    parser.add_argument("--registration", type=Path, default=configs_dir() / "calib" / "scene_registration.yaml",
                        help="출력/비교 대상 yaml (기본: configs/calib/scene_registration.yaml)")
    parser.add_argument("--write", action="store_true",
                        help="scene_registration.yaml 에 기록 (없으면 dry-run — 표만 출력)")
    args = parser.parse_args()

    names, src, dst, lm_meta = _load_landmarks(args.landmarks)
    T, scale, rms = umeyama_similarity(src, dst)
    R, t = T[:3, :3], T[:3, 3]

    # ── 점별 잔차 표 ──
    pred = scale * (src @ R.T) + t
    res = dst - pred
    print(f"[calib] ★NuRec Umeyama 정합★ — 대응점 {len(names)}쌍 ({args.landmarks})")
    print("[calib] ─────────────────────────────────────────────────────────────")
    print(f"[calib] {'name':<24} {'p_base 실측':<26} {'p_base 예측':<26} {'잔차[mm]':>8}")
    for nm, d, pr, r in zip(names, dst, pred, res):
        d_s = "(" + ", ".join(f"{v:+.4f}" for v in d) + ")"
        p_s = "(" + ", ".join(f"{v:+.4f}" for v in pr) + ")"
        print(f"[calib] {nm:<24} {d_s:<26} {p_s:<26} {np.linalg.norm(r) * 1000.0:8.2f}")
    print("[calib] ─────────────────────────────────────────────────────────────")
    print("[calib] T_base_nurec (강체부):")
    _print_matrix(T)
    print(f"[calib] scale = {scale:.6f} (NuRec 단위 → meter)")
    print(f"[calib] RMS 잔차 = {rms * 1000.0:.3f} mm")
    if rms > RMS_GATE_M:
        print(f"[calib] ⚠️★게이트 초과★ rms={rms:.4f} m > {RMS_GATE_M} m — NuRec 파인튠 진입 금지."
              " 랜드마크 재측정(모서리 정의·좌표계 확인) 후 재시도할 것.")
    else:
        print(f"[calib] 게이트 통과: rms ≤ {RMS_GATE_M} m (파인튠 진입 허용)")

    # ── 기존 scene_registration.yaml 대비 (검수용 — GUI 값과 교차 확인) ──
    if args.registration.is_file():
        try:
            T_old, scale_old = load_scene_registration(args.registration)
            dt = float(np.linalg.norm(t - T_old[:3, 3]))
            dr = _rot_angle_deg(T_old[:3, :3], R)
            print(f"[calib] 기존 {args.registration.name} 대비(검수용): "
                  f"병진 {dt * 1000.0:.1f} mm · 회전 {dr:.2f}° · scale {scale_old:.4f}→{scale:.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"[calib] ⚠️ 기존 yaml 비교 실패(형식 확인): {e}")

    # ── 기록 (--write 시에만) ──
    if not args.write:
        print("[calib] dry-run — 기록하려면 --write (단 landmarks meta.measured=true 필수)")
        return 0
    if not bool(lm_meta.get("measured", False)):
        print("[calib] ⚠️★기록 거부★ — nurec_landmarks.yaml 의 meta.measured=false"
              " (합성 예시 상태). 실측값으로 교체하고 measured: true 로 바꾼 뒤 --write.")
        return 2
    meta = {
        "calibrated": True,
        "method": "umeyama_landmarks",
        "date": date.today().isoformat(),
        "residual_m": round(float(rms), 6),
        "notes": f"register_nurec_umeyama.py — 대응점 {len(names)}쌍 ({args.landmarks.name})",
    }
    save_scene_registration(args.registration, T, scale, meta)
    print(f"[calib] ★기록 완료★ → {args.registration} (residual_m={rms:.6f})")
    print("[calib] §4-5: 이 yaml 을 커밋하고 커밋 메시지에 잔차를 남길 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
