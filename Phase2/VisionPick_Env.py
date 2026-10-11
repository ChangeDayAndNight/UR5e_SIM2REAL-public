"""Phase 2 — vision student 정책 재생 래퍼 (공식 scripts/rsl_rl/play.py에 위임).

목적: 학습된 vision 정책(depth+proprio)을 GUI로 재생. 체크포인트는 play.py가
      logs/rsl_rl/ur5e_binpick_vision에서 자동 탐색(--checkpoint로 직접 지정 가능).
      카메라 렌더 비용이 크므로 기본 num_envs는 소수(4). play.py가 env 재실행·등록 담당.
실행 (학습 PC, isaac51):
  python Phase2\\VisionPick_Env.py
  python Phase2\\VisionPick_Env.py --num_envs 2 --headless
  python Phase2\\VisionPick_Env.py --checkpoint <path/model_*.pt>
"""
from __future__ import annotations

import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

try:  # cp949 콘솔/파이프에서도 한국어 print가 죽지 않도록
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TASK = "Isaac-BinPick-UR5e-Vision-v0"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLAY_SCRIPT = os.path.join(ROOT, "scripts", "rsl_rl", "play.py")
DEFAULT_NUM_ENVS = "4"  # 카메라 렌더 비용 — 재생은 소수 env로 충분


def main() -> int:
    if not os.path.isfile(PLAY_SCRIPT):
        print(f"[play] 재생 스크립트가 없습니다: {PLAY_SCRIPT}")
        return 1
    args = list(sys.argv[1:])
    if "--task" not in args:
        args += ["--task", TASK]
    if "--num_envs" not in args:
        args += ["--num_envs", DEFAULT_NUM_ENVS]
    if "--enable_cameras" not in args:
        # vision 태스크(TiledCamera)는 카메라 렌더 필수 — AppLauncher --enable_cameras 자동 부여
        args.append("--enable_cameras")
    cmd = [sys.executable, "-u", PLAY_SCRIPT, *args]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    print(f"[play] 실행: {' '.join(cmd)}  (cwd={ROOT})")
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
