#!/usr/bin/env python
"""단계별 스토리보드 PNG → ★논문용 롤아웃 필름스트립★ (성공 pick-place 시퀀스 1장).

목적: diagnostics/render_pickplace.py(GPU)가 남긴 storyboard_*.png(SPAWN·PICK·LIFT·
      CARRY·GOAL)를 가로 필름스트립 1장으로 합성한다. 선택적으로 정책이 보는 depth
      관측(vision_obs_probe.py의 depth PNG)을 아래 행에 붙여 "무엇을 보고 무엇을
      했는가"를 한 그림에 담는다. ★GPU 불필요(순수 합성) — 렌더는 별도 GPU 단계★.
실행 (Isaac 불필요):
  python analysis/rollout_filmstrip.py --frames <storyboard_dir> --out <dir>
  python analysis/rollout_filmstrip.py --frames <dir> --depth <depth_png_dir> --out <dir>
산출: <out>/filmstrip.{pdf,png}
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# render_pickplace의 storyboard 파일명 접두(숫자)와 영문 캡션(논문 그림 = 영어)
_STAGES = [("1", "spawn"), ("2", "grasp"), ("3", "lift"), ("4", "carry"), ("5", "place")]


def _load(path: Path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def _find_frames(frame_dir: Path) -> list[tuple[str, Path]]:
    """storyboard_<n>*.png 를 단계 순서로 수집 (render_pickplace 명명 규칙)."""
    found = []
    for num, cap in _STAGES:
        cands = sorted(frame_dir.glob(f"storyboard_{num}*.png"))
        if cands:
            found.append((cap, cands[0]))
    if not found:  # fallback: 아무 png나 정렬 순
        pngs = sorted(frame_dir.glob("*.png"))
        found = [(f"t{i}", p) for i, p in enumerate(pngs[:5])]
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="스토리보드 PNG → 논문 필름스트립.")
    ap.add_argument("--frames", required=True, help="storyboard_*.png 디렉토리(render_pickplace 산출).")
    ap.add_argument("--depth", default=None, help="(선택) 정책 depth 관측 PNG 디렉토리.")
    ap.add_argument("--out", default=None, help="산출 위치(기본: --frames).")
    ap.add_argument("--title", default="Teacher pick-and-place rollout")
    a = ap.parse_args()

    fdir = Path(a.frames)
    frames = _find_frames(fdir)
    if not frames:
        print(f"[film] storyboard PNG 없음: {fdir}"); return 1
    out_dir = Path(a.out) if a.out else fdir
    out_dir.mkdir(parents=True, exist_ok=True)

    depth_imgs = []
    if a.depth:
        depth_imgs = sorted(Path(a.depth).glob("*.png"))[:len(frames)]
    nrow = 2 if depth_imgs else 1

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                         "savefig.dpi": 300, "savefig.bbox": "tight"})
    ncol = len(frames)
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.0 * nrow),
                            squeeze=False, gridspec_kw={"wspace": 0.04, "hspace": 0.12})
    for c, (cap, path) in enumerate(frames):
        axs[0][c].imshow(_load(path))
        axs[0][c].set_title(f"{c + 1}. {cap}", fontsize=10)
        axs[0][c].axis("off")
    if depth_imgs:
        for c in range(ncol):
            ax = axs[1][c]
            if c < len(depth_imgs):
                ax.imshow(_load(depth_imgs[c]), cmap="viridis")
            ax.axis("off")
            if c == 0:
                ax.text(-0.08, 0.5, "policy depth", rotation=90, va="center", ha="right",
                        transform=ax.transAxes, fontsize=9)
    fig.suptitle(a.title, fontsize=11, y=1.02)
    for suf in (".pdf", ".png"):
        fig.savefig(out_dir / f"filmstrip{suf}")
    plt.close(fig)
    print(f"[film] ★필름스트립★ {out_dir}/filmstrip.{{pdf,png}} ({ncol}단계"
          + (f" + depth {len(depth_imgs)}" if depth_imgs else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
