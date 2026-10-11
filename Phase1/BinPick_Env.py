"""Phase 1 — teacher 정책 재생 래퍼 (공식 scripts/rsl_rl/play.py에 위임).

목적: 학습된 teacher 정책을 GUI로 재생해 거동을 눈으로 확인한다. 체크포인트는
      play.py가 logs/rsl_rl/ur5e_binpick_teacher에서 자동 탐색(--checkpoint로 직접 지정 가능).
      env 재실행 가드·태스크 등록은 play.py(공식 복사본)가 담당한다 — 이 래퍼는 순수 런처.

★기본 태스크 = Eval cfg(2026-07-12 변경)★: 학습 cfg(Teacher-v0)는 pregrasp 역커리큘럼이
켜져 있어 재생 시 env의 75%(frac 초기값)가 '큐브를 잡은 채' 시작한다 — 학습 부트스트랩용이지
정책의 실제 거동이 아니다(최종 정책은 BINPICK_STAGE=2, pregrasp off로 학습). 순수 자력
스폰→Pick→Carry→Place 거동을 보려면 보조가 전부 꺼진 Eval cfg가 맞으므로 기본으로 한다.
학습 env 그대로 보려면 `--task Isaac-BinPick-UR5e-Teacher-v0`를 명시.

실행 (학습 PC, isaac51):
  python Phase1\\BinPick_Env.py                       # 최신 ckpt GUI 재생(num_envs 8, 순수 거동)
  python Phase1\\BinPick_Env.py --num_envs 4 --video --headless  # 오프스크린 mp4 녹화(둘 다 필요)
  python Phase1\\BinPick_Env.py --checkpoint <path/model_*.pt>
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

TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"  # ★보조 off 순수 거동(위 docstring)★
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLAY_SCRIPT = os.path.join(ROOT, "scripts", "rsl_rl", "play.py")
DEFAULT_NUM_ENVS = "8"


def main() -> int:
    if not os.path.isfile(PLAY_SCRIPT):
        print(f"[play] 재생 스크립트가 없습니다: {PLAY_SCRIPT}")
        return 1
    args = list(sys.argv[1:])
    if "--task" not in args:
        args += ["--task", TASK]
    if "--num_envs" not in args:
        args += ["--num_envs", DEFAULT_NUM_ENVS]
    cmd = [sys.executable, "-u", PLAY_SCRIPT, *args]  # play.py가 env python으로 재실행
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    print(f"[play] 실행: {' '.join(cmd)}  (cwd={ROOT})")
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
