"""Phase 1 — ★현재 학습 환경 정지화면 캡처★ (headless 오프스크린 렌더).

목적: 지금 학습에 쓰이는 씬(로봇·2FG7·벨트·작업대·큐브·타겟)을 여러 시점에서 PNG로 남긴다.
      GUI를 띄우지 않으므로 ★학습 run과 동시에 실행할 수 있다★(VRAM 여유만 확인할 것).

왜 필요한가: 지금까지 환경 확인이 GUI 관찰에만 의존해 [사람] 작업이었고, 배치 변경
      (작업대 도입·핑거팁 두께·벨트 규격)의 결과를 문서로 남길 수단이 없었다.
      이 스크립트의 산출물은 ★논문 그림·리뷰·회귀 비교★에 그대로 쓴다.

실행 (학습 PC, isaac51):
  python Phase1\\BinPick_Snapshot.py                      # 기본 6컷
  python Phase1\\BinPick_Snapshot.py --settle 120         # 물리 안정화 스텝 늘리기
  python Phase1\\BinPick_Snapshot.py --width 2560 --height 1440
  python Phase1\\BinPick_Snapshot.py --task Isaac-BinPick-UR5e-Teacher-v0   # 학습 cfg 그대로

산출: Phase1/Result/snapshot_<timestamp>/ → *.png + scene_info.txt(기하 실측·재현정보)
"""
from __future__ import annotations

import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()



_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
# ⚠️★h5py를 선로드하지 말 것★ — h5py 3.16.0(HDF5 2.0.0)이 Isaac 번들 hdf5.dll(1.x)보다 먼저
#   로드되면 generic_mo_io.dll이 WinError 127로 실패하고 렌더 확장이 죽는다(F54).

# ═══════════════════ AppLauncher (isaaclab import은 이 아래에서만 — §4-1 ④) ═══════════════════
import argparse

from isaaclab.app import AppLauncher

_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"

parser = argparse.ArgumentParser(description="현재 학습 환경을 headless로 렌더해 PNG로 저장.")
parser.add_argument("--task", type=str, default=_TASK,
                    help="캡처할 태스크 (기본 Eval cfg = 보조 off 순수 씬).")
parser.add_argument("--num_envs", type=int, default=1, help="병렬 env 수 — 캡처는 1이면 충분.")
parser.add_argument("--settle", type=int, default=90,
                    help="캡처 전 물리 스텝 수 — 큐브가 벨트에 안착할 시간을 준다.")
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--warm", type=int, default=60,
                    help="시점 전환 후 렌더 수렴 대기 프레임 — 적으면 노이즈가 남는다.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True          # ★오프스크린 렌더에 필수★
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 여기서부터 torch / isaaclab 계열 import 가능 ──
import datetime
import math
import subprocess as _sp
import time

import gymnasium as gym
import torch

import ur5e_binpick.tasks  # noqa: F401  — gym 태스크 자동 등록
from isaaclab_tasks.utils import parse_env_cfg
from ur5e_binpick.tasks.binpick.mdp.scene import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_SIZE,
    OBJECT_SIZE,
    TABLE_HEIGHT,
)


def _git_hash() -> str:
    try:
        return _sp.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(_REPO_ROOT, "Phase1", "Result", f"snapshot_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # ★뷰포트 해상도는 env cfg가 소유한다★ — render()가 이 값으로 rgb 프레임을 만든다.
    env_cfg.viewer.resolution = (args_cli.width, args_cli.height)
    # ★render_mode="rgb_array"★ — Isaac Lab이 영상 녹화(RecordVideo)에 쓰는 것과 ★같은 경로★다.
    #   [사고 2026-08-11] 앞서 두 방식이 모두 실패했다:
    #     ① `capture_viewport_to_file` → 6컷 전부 60초 타임아웃(뷰포트를 그리는 렌더러가 안 붙음)
    #     ② Replicator `orchestrator.step()` → ★1시간 20분 무응답★ + VRAM 7.8/8GB 포화
    #        (env가 이미 물리를 돌리는 상태에서 replicator 그래프를 따로 구동하려다 충돌)
    #   ⇒ 새 API를 시험하지 말고 ★이 저장소에서 이미 mp4를 만들고 있는 경로★를 쓴다.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    obs, _ = env.reset()

    # 큐브가 벨트에 안착할 때까지 ★무동작★으로 물리만 진행 — 캡처가 낙하 중간을 잡지 않게.
    uenv = env.unwrapped
    zero = torch.zeros((uenv.num_envs, uenv.action_manager.total_action_dim), device=uenv.device)
    for _ in range(max(0, args_cli.settle)):
        env.step(zero)

    import numpy as np
    from isaacsim.core.utils.viewports import set_camera_view
    from PIL import Image

    # 작업면 중심(벨트 중심 + 상면) 을 바라보는 시점들 — base frame 기준 좌표.
    cx, cy, cz = BELT_CENTER_X, BELT_CENTER_Y, TABLE_HEIGHT
    views = [
        ("01_overview",   (cx + 1.30, cy - 1.30, cz + 1.05), (cx, cy, cz)),
        # ★num_envs > 1이면 여러 env를 한 화면에★ — 큐브 색이 env마다 다른 것을 눈으로 확인하는 뷰.
        #   [필요성] 색은 env별로 팔레트에서 뽑히는데, 단일 env 뷰는 ★env_0의 색 하나★만 보여준다.
        #   실제로 env_0이 두 번 연속 파랑이 나와 "빨강이 안 나온다"로 오독할 뻔했다
        #   (로그는 16/16 적용·빨강 9 파랑 7이었다). ⇒ 분포를 보여주는 뷰를 따로 둔다.
        ("02_front",      (cx + 1.55, cy + 0.00, cz + 0.55), (cx, cy, cz + 0.05)),
        ("03_side",       (cx + 0.05, cy - 1.45, cz + 0.50), (cx, cy, cz + 0.05)),
        ("04_top",        (cx + 0.02, cy + 0.02, cz + 1.55), (cx, cy, cz)),
        ("05_robot",      (cx + 0.95, cy - 0.75, cz + 0.75), (0.0, 0.0, 0.10)),
        ("06_gripper",    (cx + 0.42, cy - 0.42, cz + 0.34), (cx + 0.02, cy, cz + 0.05)),
    ]
    if uenv.num_envs > 1:
        # env는 env_spacing 격자로 배치된다 — 대각 위에서 여러 셀을 한 프레임에 담는다.
        sp = float(getattr(env_cfg.scene, "env_spacing", 3.0))
        n_side = max(2, int(math.ceil(math.sqrt(uenv.num_envs))))
        span = sp * min(n_side, 4)          # 최대 4×4 구획만 담아 큐브가 너무 작아지지 않게
        mid = sp * (min(n_side, 4) - 1) / 2.0
        views.insert(0, ("00_farm", (mid + span * 0.75, mid - span * 0.75, span * 0.85),
                         (mid, mid, cz)))
    # ★★렌더러 워밍업 (2026-08-11)★★ — 첫 프레임은 수렴 전이라 ★검은 화면★이 나온다.
    #   [사고] 구현 초기에는 "프레임이 비어 있지 않으면 채택"이었는데, 첫 시점(01_overview)의
    #     첫 프레임이 ★전부 같은 값(검정)★인 유효 배열이라 그대로 저장됐다 —
    #     두 번의 실행에서 01_overview.png가 ★정확히 6,109 바이트로 동일★했고(나머지는 ~1MB)
    #     사용자가 검은 화면을 발견했다. 파일이 생겼으니 성공으로 보였던 것이 함정이다.
    #   [정정] ① 캡처 전 렌더러를 먼저 돌려 두고 ② 채택 기준을 '크기'가 아니라 ★내용★으로 바꾼다
    #     (아래 `_looks_rendered` — 픽셀 분산). 크기 검사는 검은 프레임을 통과시킨다.
    for _ in range(max(1, args_cli.warm)):
        env.render()
        simulation_app.update()

    def _looks_rendered(a) -> bool:
        """★내용 기반 판정★ — 렌더된 장면은 픽셀 분산이 크고, 미수렴 프레임은 단색이다."""
        return float(a[..., :3].std()) > 1.0

    saved = []
    for name, eye, tgt in views:
        set_camera_view(eye=list(eye), target=list(tgt))
        for _ in range(max(1, args_cli.warm)):
            simulation_app.update()
        path = os.path.join(out_dir, f"{name}.png")
        # 렌더가 새 시점으로 수렴하도록 몇 프레임 더 돌린 뒤 rgb 프레임을 직접 받는다.
        frame = None
        _t0 = time.time()
        while time.time() - _t0 < 30.0:
            f = env.render()
            if f is not None and getattr(f, "size", 0) > 0:
                a = np.asarray(f)
                if _looks_rendered(a):     # ★단색(검은) 프레임은 거부하고 더 돌린다★
                    frame = a
                    break
            simulation_app.update()
        if frame is None:
            print(f"[snapshot] ⚠️{name}.png ★캡처 실패★ — 30초 동안 단색 프레임만 나왔다")
            continue
        Image.fromarray(frame[..., :3].astype(np.uint8)).save(path)
        saved.append(name)
        print(f"[snapshot] {name}.png  ({os.path.getsize(path) // 1024} KB · {frame.shape[1]}x{frame.shape[0]})")

    # ── 기하 실측 기록 — 그림만 남기면 나중에 '무엇이 찍힌 것인지' 알 수 없다 ──
    obj = uenv.scene["object_0"].data.root_pos_w[0]
    robot_root = uenv.scene["robot"].data.root_pos_w[0]
    obj_b = (obj - robot_root).tolist()
    try:
        tgt_b = uenv.command_manager.get_command("target_pos")[0, :3].tolist()
    except Exception:
        tgt_b = None
    support_x = (BELT_CENTER_X - BELT_SIZE[0] / 2, BELT_CENTER_X + BELT_SIZE[0] / 2)
    support_y = (BELT_CENTER_Y - BELT_SIZE[1] / 2, BELT_CENTER_Y + BELT_SIZE[1] / 2)
    info = (
        f"UR5e bin-picking 학습 환경 스냅샷\n"
        f"{'=' * 60}\n"
        f"  task           : {args_cli.task}\n"
        f"  num_envs       : {args_cli.num_envs}   settle {args_cli.settle} steps\n"
        f"  해상도          : {args_cli.width}x{args_cli.height}\n"
        f"  git            : {_git_hash()}\n"
        f"  BINPICK_env    : "
        + (", ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith("BINPICK_")) or "(없음)")
        + "\n"
        f"{'-' * 60}\n"
        f"기하 (base frame, 단위 m)\n"
        f"  벨트 상면 z     : {TABLE_HEIGHT:+.4f}\n"
        f"  벨트 크기       : {BELT_SIZE[0]:.3f} x {BELT_SIZE[1]:.3f} x {BELT_SIZE[2]:.3f}\n"
        f"  벨트 범위       : x[{support_x[0]:+.4f}, {support_x[1]:+.4f}]  "
        f"y[{support_y[0]:+.4f}, {support_y[1]:+.4f}]\n"
        f"  큐브 한 변      : {OBJECT_SIZE:.3f}   (안착 중심 z = {TABLE_HEIGHT + OBJECT_SIZE / 2:+.4f})\n"
        f"  큐브 실측 위치   : x{obj_b[0]:+.4f} y{obj_b[1]:+.4f} z{obj_b[2]:+.4f}\n"
        + (f"  타겟 위치       : x{tgt_b[0]:+.4f} y{tgt_b[1]:+.4f} z{tgt_b[2]:+.4f}\n" if tgt_b else "")
        + f"{'-' * 60}\n"
        f"캡처 시점 ({len(saved)}컷)\n"
        + "".join(f"  {n}\n" for n in saved)
    )
    with open(os.path.join(out_dir, "scene_info.txt"), "w", encoding="utf-8") as f:
        f.write(info)
    print("\n" + info)
    print(f"[snapshot] ★완료★ {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
