#!/usr/bin/env python
"""configs/ → ★논문용 설정 표★ (Domain Randomization · 관측/행동 스펙 · ablation 사다리).

목적: 논문의 "실험 설정"·"ablation" 절에 바로 넣을 표를 configs와 큐레이션 데이터에서
      자동 생성한다. 손으로 옮겨 적다 값이 코드와 어긋나는 사고를 막는다.
      (1) DR 표      — configs/dr.yaml을 파싱해 (항목·범위·단위·비고).
      (2) obs/action — configs/common.yaml + exported/obs_spec.json(있으면).
      (3) ablation   — student/teacher 사다리(큐레이션: 아래 _ABLATION 편집).
실행 (Isaac 불필요): python analysis/paper_tables.py [--out <dir>]
산출: <out>(기본 Phase1/Result/paper_tables/)에 dr_table.{txt,tex} /
      obs_action_spec.{txt,tex} / ablation_{student,teacher}.{txt,tex}
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

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CFG = ROOT / "configs"

# ── DR 항목 주석(단위·의미) — dr.yaml 값에 붙일 사람이 읽을 라벨 ──
_DR_ANNOT = {
    "object_mass_scale": ("×", "객체 질량 곱셈 스케일"),
    "object_friction": ("μ", "객체 정/동마찰(그립 유지 주 레버)"),
    "object_restitution": ("", "객체 반발계수"),
    "joint_damping_scale": ("×", "관절 감쇠 스케일"),
    "joint_stiffness_scale": ("×", "관절 강성 스케일"),
    "sigma_base_m": ("m", "depth 기저 노이즈 σ"),
    "sigma_scale_per_m2": ("m⁻¹", "depth 거리제곱 노이즈 계수"),
    "dropout_prob": ("", "depth 결측(hole) 비율"),
    "intensity_scale": ("×", "광원 세기 스케일"),
    "randomize_object_color": ("bool", "객체 색 랜덤화(depth 정책엔 무효, 제거됨)"),
    "pos_perturb_m": ("m", "카메라 위치 섭동(=캘리브 허용오차)"),
    "rot_perturb_deg": ("°", "카메라 자세 섭동(=캘리브 허용오차)"),
    "obs_delay_steps": ("step", "관측 지연(20Hz 기준 0~100ms)"),
    "action_delay_steps": ("step", "행동 지연(실측 후 활성)"),
}
_DR_SECTION = {"physics": "물리(Physics)", "depth_noise": "Depth 노이즈",
               "lighting": "조명(Lighting)", "camera_pose": "카메라 포즈", "latency": "지연(Latency)"}

# ═════════════════════════════════════════════════════════════════════════════
# ★ablation 사다리 (큐레이션 — 결과가 바뀌면 여기만 수정)★
#   출처: README의 실패/진행 이력. 각 행 = 한 설계 변경의 eval 성공률(전 assist off).
# ═════════════════════════════════════════════════════════════════════════════
_ABLATION_STUDENT = {
    "title": "Vision student 설계 사다리 (증류, eval 자력 파지)",
    "cols": ["설계", "grasp(%)", "carry(%)", "placed(%)", "비고"],
    "rows": [
        ["MLP-on-pixels", "0.0", "0.0", "0.0", "flat MLP — loss 수렴하나 태스크 능력 0(문헌 선례 없음)"],
        ["Nature-CNN, 고정캠", "1.6", "—", "0.2", "팔 가림 33%+큐브 2×2px"],
        ["CNN, eye-in-hand", "12.7", "10.7", "3.1", "손목 장착 — 자력 파지 최초 발현(8배)"],
        ["CNN-GRU, eye-in-hand", "0.6", "—", "—", "epochs=6×recurrent hidden 오염(후퇴)"],
        ["CNN eih + 연장 fine-tune", "21.6", "20.9", "15.0", "변경0 연장(grasp +70%·placed 5배)"],
    ],
    "note": "고정캠→eye-in-hand가 핵심 전환. 전 행 동일 teacher(특권 42D)·동일 obs 계약.",
}
_ABLATION_TEACHER = {
    "title": "Teacher(특권) 주요 마일스톤",
    "cols": ["구성", "grasp(%)", "carry(%)", "placed(%)", "비고"],
    "rows": [
        ["현행 F64 scratch 20000 iter", "—", "—", "—", "재학습·독립 eval 판정 대기"],
    ],
    "note": "구 99% 계보는 현행 자산·보상 계약의 배포 근거가 아니다. F64 run의 독립 eval 뒤에만 갱신한다.",
}


def _md_table(cols: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _tex_table(cols: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    def esc(s: str) -> str:
        return s.replace("%", "\\%").replace("×", "$\\times$").replace("μ", "$\\mu$").replace("±", "$\\pm$").replace("°", "$^\\circ$")
    spec = "l" + "c" * (len(cols) - 2) + "l" if len(cols) > 2 else "l" * len(cols)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    lines += [" & ".join(esc(str(c)) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def build_dr_table(out_dir: Path) -> None:
    dr = yaml.safe_load((CFG / "dr.yaml").read_text(encoding="utf-8"))
    rows = []
    for sec, params in dr.items():
        if not isinstance(params, dict):
            continue
        for key, val in params.items():
            unit, desc = _DR_ANNOT.get(key, ("", key))
            if isinstance(val, list):
                rng = f"[{val[0]}, {val[1]}]"
            else:
                rng = str(val)
            rows.append([_DR_SECTION.get(sec, sec), key, rng, unit, desc])
    cols = ["섹션", "파라미터", "범위/값", "단위", "설명"]
    (out_dir / "dr_table.txt").write_text(
        "# Domain Randomization (configs/dr.yaml 자동 추출)\n\n" + _md_table(cols, rows) + "\n",
        encoding="utf-8")
    (out_dir / "dr_table.tex").write_text(
        _tex_table(["Section", "Parameter", "Range", "Unit", "Description"],
                   [[r[0], r[1].replace("_", "\\_"), r[2], r[3], r[4]] for r in rows],
                   "Domain randomization ranges.", "tab:dr") + "\n", encoding="utf-8")
    print(f"[tables] DR 표: dr_table.{{txt,tex}} ({len(rows)}항)")


def build_obs_action(out_dir: Path) -> None:
    common = yaml.safe_load((CFG / "common.yaml").read_text(encoding="utf-8"))
    pre, act = common.get("preprocess", {}), common.get("action", {})
    spec_path = ROOT / "exported" / "obs_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.is_file() else {}
    rows = [
        ["Depth 해상도", f"{pre.get('out_height')}×{pre.get('out_width')}", "정책 입력 depth(정사각 crop+resize)"],
        ["Depth 범위(near/far)", f"{pre.get('depth_near_m')} / {pre.get('depth_far_m')} m", "near 미만=무효→far(D435i blind zone)"],
        ["Crop fraction", str(pre.get("crop_fraction")), "유효 FOV 크롭(eye-in-hand=1.0)"],
        ["행동 스킴(sim)", "task-space IK (ik_rel, scale 0.2)", "arm 6D delta + 그리퍼 binary"],
        ["행동 차원", str(act.get("num_arm_joints", 6) + 1), "arm 6 + gripper 1 = 7"],
        ["obs_dim(배포)", str(spec.get("obs_dim", "—")), "exported/obs_spec.json(없으면 —)"],
    ]
    cols = ["항목", "값", "설명"]
    (out_dir / "obs_action_spec.txt").write_text(
        "# 관측/행동 스펙 (configs/common.yaml + obs_spec.json)\n\n" + _md_table(cols, rows) + "\n",
        encoding="utf-8")
    (out_dir / "obs_action_spec.tex").write_text(
        _tex_table(["Item", "Value", "Description"],
                   [[r[0], r[1], r[2]] for r in rows],
                   "Observation and action space.", "tab:obsact") + "\n", encoding="utf-8")
    print(f"[tables] obs/action 스펙: obs_action_spec.{{txt,tex}}")


def build_ablation(out_dir: Path) -> None:
    for tag, data in (("student", _ABLATION_STUDENT), ("teacher", _ABLATION_TEACHER)):
        text = f"# {data['title']}\n\n" + _md_table(data["cols"], data["rows"]) + \
             f"\n\n> {data['note']}\n"
        (out_dir / f"ablation_{tag}.txt").write_text(text, encoding="utf-8")
        (out_dir / f"ablation_{tag}.tex").write_text(
            _tex_table([c.replace("(%)", " (\\%)").replace("설계", "Design").replace("구성", "Config").replace("비고", "Note")
                        for c in data["cols"]],
                       data["rows"], data["title"], f"tab:abl_{tag}") + "\n", encoding="utf-8")
    print(f"[tables] ablation 사다리: ablation_{{student,teacher}}.{{txt,tex}}")


def main() -> int:
    ap = argparse.ArgumentParser(description="configs → 논문 설정/ablation 표.")
    ap.add_argument("--out", default=str(ROOT / "Phase1" / "Result" / "paper_tables"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    build_dr_table(out)
    build_obs_action(out)
    build_ablation(out)
    print(f"[tables] ★완료★ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
