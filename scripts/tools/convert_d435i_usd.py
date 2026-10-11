#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + 경로.
# ─────────────────────────────────────────────────────────────────────────────
"""D435(i) 시각 메시(dae) → USD 변환 (시각 전용 카메라 본체).

목적: assets/d435i/meshes/d435.dae(Intel 공식, Apache-2.0)를 isaaclab MeshConverter로
      USD 변환 — ★시각 전용(콜라이더·질량 없음, 2026-07-15 사용자 승인)★.
      sim에서 wrist_3_link 하위 자식 프림으로 스폰되어(원본 로봇 USD 무수정)
      eye-in-hand 카메라 '본체'가 GUI에 보이게 한다. depth 센서(TiledCamera)는
      별개 프림으로 기존 계약 그대로(mdp/scene.py make_tiled_camera_cfg).
실행 (학습 PC, isaac51 — GPU 필요, 학습과 동시 실행 금지):
  python scripts\\tools\\convert_d435i_usd.py --headless
산출: assets/d435i/d435i.usd + Props/instanceable_meshes.usd.
      두 USD는 asset_manifest.json의 canonical Git 자산이다. 재변환했다면 manifest의
      크기/SHA-256과 Stage dependency를 갱신·검증한 뒤 함께 커밋한다.
"""
import os
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = str(_HERE.parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

# ═══════════════════ AppLauncher (isaaclab import은 이 아래에서만) ═══════════════════
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="D435(i) 시각 메시 → USD 변환 (시각 전용).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────── sim 기동 후 ───────────────────────────
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg


def main() -> None:
    src = os.path.join(_REPO_ROOT, "assets", "d435i", "meshes", "d435.dae")
    out_dir = os.path.join(_REPO_ROOT, "assets", "d435i")
    if not os.path.isfile(src):
        raise SystemExit(f"[convert] 메시 없음: {src}")
    cfg = MeshConverterCfg(
        asset_path=src,
        usd_dir=out_dir,
        usd_file_name="d435i.usd",
        force_usd_conversion=True,
        make_instanceable=True,     # 다수 env 복제 효율
        collision_props=None,       # ★시각 전용 — 콜라이더 없음(사용자 승인 2026-07-15)★
        mass_props=None,
        rigid_props=None,
    )
    converter = MeshConverter(cfg)
    print(f"[convert] ★완료★ {converter.usd_path}")
    print("[convert] 검증: GUI에서 열어 스케일(본체 폭 ~90mm)·원점을 확인할 것")


if __name__ == "__main__":
    main()
    simulation_app.close()
