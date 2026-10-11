#!/usr/bin/env python
"""학습 중 마일스톤 성능 점검 — ★GPU 불필요(TB 스칼라만)★. 학습 방해 없이 실행.

목적: 학습이 진행 중일 때(GPU 점유) eval을 못 돌리므로, TensorBoard 보상/보상항
      곡선을 읽어 성능을 분석한다. 보상항(reach/grasp/lift/goal/placed)은 태스크
      단계별 발현을 직접 반영하므로 teacher 성능의 강한 프록시다. ★이미 목표 성능
      (99%)에 도달한 참조 run과 같은 iter에서 대조★해 정상 궤도인지 판정한다.
실행 (Isaac 불필요):
  python analysis/milestone_check.py --iter 2500
  python analysis/milestone_check.py --iter 5000 --run <run_dir> --ref <ref_dir>
산출: 콘솔 표 + 판정. --out 주면 milestone_<iter>.txt 저장.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "logs" / "rsl_rl" / "ur5e_binpick_teacher"
# ★2026-07-28★ 구 참조 run(2026-07-12_19-33-03)은 ①산출물 전량 정리로 삭제됐고
#   ②설령 있어도 ★구 계약★(홈 -90/+90/-90, 타겟 x[0.35,0.55])이라 신 계약 run과 비교하면
#   오판한다(타겟 박스 면적 2.6배 → 같은 iter에 goal/placed가 정당하게 낮게 나온다).
#   → 기본 참조 없음. 신 계약 teacher 첫 완주 run을 --ref로 승격해 쓸 것.
REF_DEFAULT = None  # 참조 없음(구 계약 run 폐기) — --ref로 명시 지정

# 태스크 단계 프록시(순서 = 파이프라인) + 판정 가중
_TERMS = ["Train/mean_reward", "Episode_Reward/reach", "Episode_Reward/grasp",
          "Episode_Reward/lift", "Episode_Reward/goal", "Episode_Reward/placed"]
# 단계 발현의 '핵심' 항 — 이게 참조 대비 죽어 있으면 이상(파이프라인 붕괴)
_KEY = ["Episode_Reward/lift", "Episode_Reward/goal", "Episode_Reward/placed"]


def _load(run: Path) -> dict:
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    out = {}
    for t in _TERMS:
        if t in tags:
            ev = ea.Scalars(t)
            out[t] = ([e.step for e in ev], [e.value for e in ev])
    return out


def _at_iter(series, target: int):
    """target 이하의 최근 step 값 (없으면 첫 값). 반환 (step, value)."""
    steps, vals = series
    if not steps:
        return (None, None)
    best = 0
    for i, s in enumerate(steps):
        if s <= target:
            best = i
        else:
            break
    return (steps[best], vals[best])


def _latest_iter(data: dict) -> int:
    mr = data.get("Train/mean_reward")
    return mr[0][-1] if mr and mr[0] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="학습 마일스톤 TB 성능 점검.")
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--run", default=None, help="점검 대상 run(기본: 최신 teacher run).")
    ap.add_argument("--ref", default=(str(REF_DEFAULT) if REF_DEFAULT else None),
                    help="참조 run(같은 계약이어야 유효 — 기본 없음).")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.run:
        run_dir = Path(a.run)
    elif EXP_DIR.is_dir():  # ★산출물 정리 후 디렉토리 자체가 없을 수 있다(2026-07-28)★
        run_dir = max((p for p in EXP_DIR.iterdir() if p.is_dir()),
                      key=lambda p: p.name, default=None)
    else:
        run_dir = None
    if run_dir is None or not run_dir.is_dir():
        print(f"[mile] teacher run을 찾지 못함 (탐색 위치: {EXP_DIR}) — "
              "학습이 아직 시작되지 않았거나 --run 경로를 지정할 것")
        return 1
    ref_dir = Path(a.ref) if a.ref else None

    new = _load(run_dir)
    ref = _load(ref_dir) if (ref_dir is not None and ref_dir.is_dir()) else {}
    if a.ref and not ref:
        print(f"[mile] ⚠️ 참조 run을 찾지 못함: {a.ref} — 비교 없이 절대값만 보고한다")
    reached = _latest_iter(new)
    if reached < a.iter:
        print(f"[mile] ⚠️ 대상 run이 아직 iter {a.iter} 미도달(현재 {reached}) — 최신값으로 점검")

    lines = [f"# 마일스톤 점검 — iter {a.iter}",
             f"- 대상 run : {run_dir.name} (현재 iter {reached})",
             (f"- 참조 run : {ref_dir.name}" if ref
              else "- 참조 run : ★없음★ — 비교 판정 불가(절대값만 보고). "
                   "신 계약 run 완주 후 --ref로 승격할 것"), "",
             "| 항목 | 신 run | 참조 run | 비율(신/참조) | 판정 |",
             "|---|---|---|---|---|"]
    verdict_flags = []
    for t in _TERMS:
        if t not in new:
            continue
        _sn, vn = _at_iter(new[t], a.iter)
        _sr, vr = _at_iter(ref[t], a.iter) if t in ref else (None, None)
        ratio = (vn / vr) if (vr not in (None, 0)) else float("nan")
        # 판정: 핵심 단계 항이 참조 대비 <30%면 경고, 그 외 참고
        flag = "—"
        if t in _KEY and vr not in (None, 0):
            if ratio < 0.3:
                flag = "⚠️ 저조"; verdict_flags.append((t, ratio))
            elif ratio >= 0.7:
                flag = "✓ 정상"
            else:
                flag = "△ 추적중"
        name = t.split("/")[-1]
        vr_s = f"{vr:.3f}" if vr is not None else "—"
        ratio_s = f"{ratio:.2f}" if ratio == ratio else "—"
        lines.append(f"| {name} | {vn:.3f} | {vr_s} | {ratio_s} | {flag} |")

    # 종합 판정
    lines.append("")
    if verdict_flags:
        names = ", ".join(t.split("/")[-1] for t, _ in verdict_flags)
        lines.append(f"## 판정: ⚠️ 이상 신호 — 핵심 단계항 저조: {names}")
        lines.append("→ 원인 분석 필요(파이프라인 단계 미발현). 참조 궤적 대비 어느 단계서")
        lines.append("  끊기는지 확인 후 개입 판단(학습 중단→eval 확인 또는 계속 관망).")
    elif not ref:
        # ★2026-07-28 거짓 안심 수정★: 참조가 없으면 비교가 성립하지 않는데도 기존 코드는
        #   '✓ 정상 궤도 — 참조 run을 정상 추적 중'을 출력했다(verdict_flags가 빌 수밖에 없으므로).
        #   조기중단 게이트가 아무 근거 없이 '정상'을 보고하는 것은 안전망의 실패다.
        dead = [t.split("/")[-1] for t in _KEY
                if t in new and (_at_iter(new[t], a.iter)[1] or 0.0) <= 0.0]
        lines.append("## 판정: ⚠️ ★참조 없음 — 자동 판정 불가★ (절대값만 보고)")
        if dead:
            lines.append(f"→ ★미발현 단계항: {', '.join(dead)}★ — 해당 iter에 값이 0이다. "
                         "파이프라인이 그 단계에서 끊겼는지 확인할 것.")
        else:
            lines.append("→ 핵심 단계항(lift/goal/placed)은 모두 0보다 크다(발현은 확인). "
                         "다만 '충분한 수준'인지는 참조 없이 판단할 수 없다.")
        lines.append("→ 신 계약 teacher 첫 완주 run을 `--ref`로 지정하면 다음 재학습부터 비교 판정이 된다.")
    else:
        lines.append("## 판정: ✓ 정상 궤도 — 핵심 단계항이 참조 run을 정상 추적 중.")
    report = "\n".join(lines)
    print("\n" + report)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"milestone_{a.iter}.txt").write_text(report + "\n", encoding="utf-8")
        print(f"\n[mile] 저장: {a.out}/milestone_{a.iter}.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
