"""Phase 2 — vision student 학습 성능을 TensorBoard로 확인.

목적: logs/rsl_rl/ur5e_binpick_vision(업스트림 train.py 산출 위치)의 모든 run을
      TensorBoard로 띄운다(PPO/증류 run 공용 보관소 — 타임스탬프로 구분).
실행 (프로젝트 루트에서):
  python Phase2\\VisionPick_Tensorboard.py                 # 기본 포트 6006, 브라우저 자동
  python Phase2\\VisionPick_Tensorboard.py --port 6007 --no-browser
산출: 없음 (뷰어 전용).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
import webbrowser

try:  # cp949 콘솔/파이프에서도 한국어 print가 죽지 않도록
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

EXPERIMENT = "ur5e_binpick_vision"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOGDIR = os.path.join(ROOT, "logs", "rsl_rl", EXPERIMENT)


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()



def _open_browser_later(url: str, delay: float = 4.0) -> None:
    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description="UR5e BinPick vision 학습 TensorBoard 뷰어")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--logdir", default=LOGDIR, help="기본: logs/rsl_rl/ur5e_binpick_vision")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        print(f"[tb] 로그 폴더가 없습니다 — 먼저 학습하세요(VisionPick_Train.py): {args.logdir}")
        return 1

    python_exe = sys.executable
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    cmd = [python_exe, "-m", "tensorboard.main", "--logdir", args.logdir, "--port", str(args.port)]
    url = f"http://localhost:{args.port}"
    print(f"[tb] logdir = {args.logdir}")
    print(f"[tb] 실행: {' '.join(cmd)}")
    print(f"[tb] 브라우저에서 {url}  (종료: Ctrl+C)")

    if not args.no_browser:
        threading.Thread(target=_open_browser_later, args=(url,), daemon=True).start()
    try:
        return subprocess.run(cmd, env=env).returncode
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
