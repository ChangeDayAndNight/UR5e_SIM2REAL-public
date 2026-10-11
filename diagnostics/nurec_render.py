"""Workplace.usdz(NuRec 3DGS) headless 렌더 — Isaac Sim 5.1 검증 D."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

parser = argparse.ArgumentParser()
parser.add_argument("--usdz", type=Path, default=ROOT / "assets" / "Scene" / "Workplace.usdz")
parser.add_argument("--out", type=Path, default=ROOT / "diagnostics" / "nurec_render")
args = parser.parse_args()
if not args.usdz.is_file():
    raise SystemExit(f"[nurec] optional USDZ 없음: {args.usdz}")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import omni.usd
from pxr import Usd, UsdGeom

USDZ = str(args.usdz.resolve())
OUT = str(args.out.resolve())
os.makedirs(OUT, exist_ok=True)

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
stage.DefinePrim("/World", "Xform")
from isaacsim.core.utils.stage import add_reference_to_stage
add_reference_to_stage(usd_path=USDZ, prim_path="/World/Workplace")
print("[nurec] USDZ 레퍼런스 추가 — 로드 대기...")
for _ in range(180):
    app.update()

# ── 검증 D-1: NuRec 프림 존재 확인 ──
found = []
for prim in stage.Traverse():
    tn = prim.GetTypeName()
    if any(k in tn.lower() for k in ("nurec", "volume", "gauss")) or any(k in prim.GetName().lower() for k in ("nurec", "gauss")):
        found.append((str(prim.GetPath()), tn))
print(f"[nurec] NuRec/volume 후보 프림 {len(found)}개:")
for p, t in found[:8]:
    print(f"    {p}  <{t}>")

# ── 카메라 뷰 + 캡처 ──
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
vp = get_active_viewport()
# export 로그 extent: min(-56.75,-61.85,-79.27) max(93.43,43.88,53) → 중심·대각
cx, cy, cz = 18.3, -9.0, -13.1
views = [
    ("wide",  (cx + 60, cy + 60, cz + 40), (cx, cy, cz)),
    ("mid",   (cx + 20, cy + 20, cz + 12), (cx, cy, cz)),
    ("near",  (cx + 6,  cy + 6,  cz + 4),  (cx, cy, cz)),
    ("center",(cx + 1.5, cy + 1.5, cz + 1.0), (cx, cy, cz)),
]
for name, eye, tgt in views:
    set_camera_view(eye=list(eye), target=list(tgt))
    for _ in range(90):
        app.update()
    capture_viewport_to_file(vp, os.path.join(OUT, f"{name}.png"))
    for _ in range(30):
        app.update()
    print(f"[nurec] {name}.png 캡처")
print(f"[nurec] ★완료★ → {OUT}")
app.close()
