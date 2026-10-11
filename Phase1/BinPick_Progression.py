#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + repo 루트.
# ─────────────────────────────────────────────────────────────────────────────
"""Phase 1 — ★학습 진척 영상★: 저장된 체크포인트를 iter 간격으로 순회하며 각 정책의
   '에피소드 전체'를 영상으로 렌더링 → 학습이 어떻게 좋아지는지 눈으로 본다.

왜 학습 중 --video가 아니라 이 방식인가:
  · 8GB VRAM에서 teacher 4096 env + 렌더 카메라 동시는 OOM 위험 → 학습은 카메라 없이 돌리고,
    끝난 뒤(또는 언제든) 체크포인트에서 소급 렌더(few env)하면 안전.
  · 카메라(클로즈업)·출력위치(Phase1/Result/)를 제어 가능. 같은 "N-iter 간격 에피소드 영상" 결과.
  · Isaac 1회 기동으로 여러 체크포인트를 순회(재기동 없이 weight만 교체) → 빠름.

실행 (학습 PC, isaac51):
  python Phase1\\BinPick_Progression.py                       # 최신 run, 200 iter 간격
  python Phase1\\BinPick_Progression.py --every 100           # 100 iter 간격(요청 기본)
  python Phase1\\BinPick_Progression.py --run logs\\rsl_rl\\ur5e_binpick_teacher\\<run>
산출: Phase1/Result/<run>/progression/iter_<N>.mp4 (+ contact sheet 프레임)
      pre-grasp 없는 eval 태스크로 렌더 → 순수 자력 거동.

⚠️ isaaclab을 import하므로 프리앰블/AppLauncher 이후에만 import한다.
"""
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import (
    configure_utf8_stdio,
    preload_native_training_modules,
    reexec_in_training_python,
)

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
try:
    # ★2026-08-10(F54) h5py 선로드 제거★ — 실측상 ★이 import가 Isaac 센서 DLL을 깨뜨린다★.
    #   h5py 3.16.0(HDF5 2.0.0)이 Isaac 번들 hdf5.dll(1.x)보다 먼저 로드되면
    #   generic_mo_io.dll이 WinError 127로 실패하고 GUI가 access violation으로 죽는다.
    #   (구 h5py에서는 반대로 작동해 워크어라운드였다 — 근거·측정 = Failure_Check F54)
    pass
except Exception:
    pass


# ═══════════════════ 드라이버 모드 (--driver: Isaac 미기동, 행 방어) ═══════════════════
# ★2026-07-13 행 방어★: 단일 프로세스에서 수천 render()를 돌리면 Kit 내부가 비결정적으로
#   데드락한다(2 run 연속 실측: 14클립/6클립 후 무한 행 — 파이썬 루프는 유한이므로 Kit 내부).
#   해법 = 체크포인트를 청크로 나눠 '자식 프로세스'로 렌더 + 청크별 타임아웃. 행에 걸린
#   청크만 죽이고 다음 청크로 진행 → 최악에도 클립 일부만 잃고 전체는 완주한다.
#   드라이버 자체는 isaaclab을 import하지 않는다(순수 파이썬 — 이 블록은 AppLauncher 이전).
def _driver_main() -> int:
    import re as _re
    import subprocess as _sp

    argv = sys.argv[1:]

    def _opt(name: str, default: str | None = None) -> str | None:
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else default

    run_dir = _opt("--run")
    if run_dir is None:  # 최신 run 자동 (main의 _pick_run과 동일 로직 — Isaac 불필요)
        root = os.path.join(_REPO_ROOT, "logs", "rsl_rl", "ur5e_binpick_teacher")
        subs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        run_dir = max(subs, key=os.path.getmtime)
    run_dir = os.path.abspath(run_dir)
    every = int(_opt("--every", "200"))
    chunk = int(_opt("--chunk-size", "5"))
    timeout_s = int(_opt("--chunk-timeout", "900"))  # 부팅 ~3분 + 5클립 × ~1.5분 → 15분 여유

    # 체크포인트 선별(아래 _checkpoints와 동일 규칙 — 결정적이라 자식과 일치)
    pairs = sorted(
        (int(m.group(1)), f) for f in os.listdir(run_dir)
        if (m := _re.fullmatch(r"model_(\d+)\.pt", f))
    )
    picked, last_it = [], -(10**9)
    for it, _f in pairs:
        if it - last_it >= every:
            picked.append(it); last_it = it
    if picked and pairs and picked[-1] != pairs[-1][0]:
        picked.append(pairs[-1][0])
    n_chunks = (len(picked) + chunk - 1) // chunk
    print(f"[prog][driver] {len(picked)}개 체크포인트를 {n_chunks}청크(각 {chunk}개, 타임아웃 {timeout_s}s)로 렌더")

    ok = skipped = 0
    base = [c for c in argv if c != "--driver"]
    if "--run" not in base:
        base += ["--run", run_dir]
    for ci in range(n_chunks):
        cmd = [sys.executable, "-u", os.path.abspath(__file__), *base,
               "--chunk-index", str(ci), "--chunk-size", str(chunk)]
        try:
            rc = _sp.run(cmd, timeout=timeout_s).returncode
            if rc == 0:
                ok += 1
            else:
                skipped += 1
                print(f"[prog][driver] ⚠️ 청크 {ci} 비정상 종료(rc={rc}) — 다음 청크 진행")
        except _sp.TimeoutExpired:
            skipped += 1
            print(f"[prog][driver] ⚠️ 청크 {ci} 타임아웃({timeout_s}s) — Kit 행으로 판단, 강제 종료 후 다음 청크")
    print(f"[prog][driver] ★완료★ 청크 {ok}/{n_chunks} 성공, {skipped} 스킵(행/오류)")
    return 0


if "--driver" in sys.argv:
    raise SystemExit(_driver_main())

preload_native_training_modules()

# ═══════════════════ AppLauncher (isaaclab import은 이 아래에서만) ═══════════════════
import argparse
import re

from isaaclab.app import AppLauncher

import cli_args  # scripts/rsl_rl/cli_args.py  # isort: skip

_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"  # pre-grasp 없는 eval 태스크(순수 거동)
_EXP = "ur5e_binpick_teacher"

parser = argparse.ArgumentParser(description="teacher 학습 진척 영상(체크포인트 순회 렌더).")
parser.add_argument("--run", type=str, default=None, help="run 폴더(기본: 최신).")
parser.add_argument("--every", type=int, default=200, help="영상 렌더 iter 간격(기본 200).")
parser.add_argument("--num_envs", type=int, default=1, help="렌더 env 수(기본 1 — 단일 로봇 클로즈업).")
parser.add_argument("--episode_steps", type=int, default=250, help="에피소드당 렌더 스텝(전체 에피소드 포괄).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--fps", type=int, default=20, help="출력 mp4 fps. ★기본 20 = control_hz라 "
                    "실시간★ (1 프레임=1 제어스텝). 30이면 1.5배 빠르게 재생됨.")
# 행 방어 청크 모드 (드라이버가 부여 — 사용자가 직접 쓸 일 없음)
parser.add_argument("--chunk-index", type=int, default=-1, help="(드라이버용) 렌더할 청크 번호")
parser.add_argument("--chunk-size", type=int, default=5, help="(드라이버용) 청크당 체크포인트 수")
parser.add_argument("--chunk-timeout", type=int, default=900, help="(드라이버 전용 — 자식에선 무시)")
parser.add_argument("--task", type=str, default=_TASK,
                    help="렌더 태스크(student: Isaac-BinPick-UR5e-Vision-Eval-v0 + --run).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_TASK = args_cli.task
args_cli.enable_cameras = True  # 렌더(rgb_array)에 필요
# ★★2026-08-11(F60) headless 강제 — 이 스크립트는 오프스크린 영상 녹화기다★★
#   [사고] headless를 지정하지 않아 AppLauncher가 ★GUI 계열 experience★로 기동했고
#     (isaaclab.python.rendering — 확장 285개), 그 프로필만 `isaacsim.sensors.rtx`와
#     `omni.sensors.nv.lidar`를 올린다. 이 확장들이 Isaac 번들 ★hdf5.dll(1.x)★을 먼저 로드하면
#     그 뒤 `isaaclab_tasks` → recorder_manager → `import h5py`(HDF5 2.0.0)가 DLL 심볼을 못 찾아
#     ImportError로 죽는다(F54 이후 h5py 선로드를 제거하면서 드러난 반대편 충돌).
#   [실측] 학습 종료 후 자동 호출되는 이 스크립트가 ★2026-08-11의 6개 run 전부에서 실패★해
#     `Phase1/Result/<run>/progression/`이 ★0 파일★이었다(08-10 run들은 2·10 파일 정상).
#     콘솔에 남지 않아 ★조용히 실패★하고 있었다.
#   [정정] headless 계열은 위 두 센서 확장을 ★아예 로드하지 않는다★(headless.rendering 153개 중 0개)
#     → h5py가 자기 DLL을 정상 로드한다. 오프스크린 렌더는 headless에서 그대로 되고
#     (`Phase1/BinPick_Snapshot.py`가 같은 조합으로 1920×1080 캡처를 산출한다),
#     영상 녹화기에 창은 애초에 필요 없다.
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────── sim 기동 후 ───────────────────────────
import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401  (gym.register)


def _runs_root() -> str:
    return os.path.abspath(os.path.join(_REPO_ROOT, "logs", "rsl_rl", _EXP))


def _pick_run() -> str:
    if args_cli.run:
        return os.path.abspath(args_cli.run)
    root = _runs_root()
    subs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not subs:
        raise RuntimeError(f"[prog] run 폴더 없음: {root}")
    return max(subs, key=os.path.getmtime)


def _checkpoints(run_dir: str, every: int) -> list[tuple[int, str]]:
    """iter 간격으로 체크포인트 선별 (+ 항상 마지막 것 포함)."""
    pairs = []
    for f in os.listdir(run_dir):
        m = re.fullmatch(r"model_(\d+)\.pt", f)
        if m:
            pairs.append((int(m.group(1)), os.path.join(run_dir, f)))
    pairs.sort()
    if not pairs:
        raise RuntimeError(f"[prog] 체크포인트(model_*.pt) 없음: {run_dir}")
    picked, last_it = [], -(10**9)
    for it, path in pairs:
        if it - last_it >= every:
            picked.append((it, path)); last_it = it
    if picked[-1][0] != pairs[-1][0]:
        picked.append(pairs[-1])  # 최종 정책은 항상 포함
    return picked


def _write_mp4(frames: list, path: str, fps: int) -> bool:
    try:
        import cv2
        if not frames:
            return False
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(np.asarray(fr), cv2.COLOR_RGB2BGR))
        vw.release()
        return True
    except Exception as e:
        print(f"[prog][warn] mp4 저장 실패({path}): {e}")
        return False


def main() -> None:
    run_dir = _pick_run()
    ckpts = _checkpoints(run_dir, args_cli.every)
    if args_cli.chunk_index >= 0:  # 드라이버 청크 모드 — 내 몫만 렌더 (선별 규칙은 드라이버와 동일=결정적)
        s = args_cli.chunk_index * args_cli.chunk_size
        ckpts = ckpts[s:s + args_cli.chunk_size]
        if not ckpts:
            print(f"[prog] 청크 {args_cli.chunk_index}: 렌더할 체크포인트 없음 — 종료")
            return
    # ★산출물 = Phase1/Result/<run>/progression (사용자 결정 2026-07-10 — 루트 results/ 폐지)★
    #   (vision → Phase2/Result — BinPick_Eval의 phase 매핑과 동일)
    _phase_dir = os.path.join(_REPO_ROOT, "Phase2") if "Vision" in _TASK else _HERE
    out_dir = os.path.join(_phase_dir, "Result", os.path.basename(os.path.normpath(run_dir)), "progression")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[prog] run={run_dir}")
    print(f"[prog] 체크포인트 {len(ckpts)}개 (간격 {args_cli.every} iter): {[it for it,_ in ckpts]}")
    print(f"[prog] 출력={out_dir}")

    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    agent_cfg = cli_args.parse_rsl_rl_cfg(_TASK, args_cli)
    env = gym.make(_TASK, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    _runner_cls = DistillationRunner if getattr(agent_cfg, "class_name", "OnPolicyRunner") == "DistillationRunner" else OnPolicyRunner
    runner = _runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    made = []
    for it, ckpt in ckpts:
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()  # rsl_rl 3.x(2.3.2 래퍼): TensorDict 단일 반환
        frames = []
        with torch.inference_mode():
            for _ in range(args_cli.episode_steps):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                fr = env.unwrapped.render()  # (H,W,3) rgb — 뷰어 클로즈업 cfg 사용
                if fr is not None:
                    frames.append(np.asarray(fr))
        path = os.path.join(out_dir, f"iter_{it:05d}.mp4")
        if _write_mp4(frames, path, args_cli.fps):
            made.append((it, path))
            print(f"[prog] ✓ iter {it}: {os.path.basename(path)} ({len(frames)} frames)")

    env.close()
    print(f"\n[prog] ★완료★ {len(made)}개 영상 → {out_dir}")
    print("[prog] 학습 진척: iter_00xxx.mp4 를 순서대로 보면 정책이 개선되는 과정이 보인다.")


if __name__ == "__main__":
    main()
    simulation_app.close()
