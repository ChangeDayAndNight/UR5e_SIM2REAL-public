"""Workplace.usdz NuRec 렌더 v2.1 — 캡처 완료를 파일 생성으로 확인."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

parser = argparse.ArgumentParser()
parser.add_argument("--usdz", type=Path, default=ROOT / "assets" / "Scene" / "Workplace.usdz")
parser.add_argument("--out", type=Path, default=ROOT / "diagnostics" / "nurec_render2")
args = parser.parse_args()
if not args.usdz.is_file():
    raise SystemExit(f"[nurec2] optional USDZ 없음: {args.usdz}")

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

USDZ = str(args.usdz.resolve())
OUT = str(args.out.resolve())
os.makedirs(OUT, exist_ok=True)
ctx = omni.usd.get_context(); ctx.new_stage()
ctx.get_stage().DefinePrim("/World", "Xform")
add_reference_to_stage(usd_path=USDZ, prim_path="/World/Workplace")
for _ in range(240): app.update()
vp = get_active_viewport()
print(f"[nurec2] viewport={vp}")

T = (-1.0, -1.0, 1.0)
views = []
for az in range(0, 360, 60):
    a = math.radians(az)
    views.append((f"orbit{az:03d}", (T[0]+7*math.cos(a), T[1]+7*math.sin(a), T[2]+2.5), T))
views += [("topdown", (T[0]+0.5, T[1]+0.5, T[2]+9), T),
          ("low", (T[0]+4, T[1]-4, T[2]+0.5), T),
          ("closeA", (T[0]+3, T[1]+3, T[2]+1.2), T),
          ("closeB", (T[0]-3, T[1]+3, T[2]+1.2), T)]
ok = 0
for name, eye, tgt in views:
    set_camera_view(eye=list(eye), target=list(tgt))
    for _ in range(80): app.update()
    path = os.path.join(OUT, f"{name}.png")
    cap = capture_viewport_to_file(vp, path)
    t0 = time.time()
    while not os.path.exists(path) and time.time() - t0 < 30:
        app.update()
    done = os.path.exists(path)
    ok += int(done)
    print(f"[nurec2] {name}: {'OK' if done else '★캡처 실패★'} ({os.path.getsize(path)//1024}KB)" if done else f"[nurec2] {name}: ★캡처 실패★")
print(f"[nurec2] ★완료 {ok}/{len(views)}★ → {OUT}")
for _ in range(30): app.update()
app.close()
