#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★수정된 teacher의 Pick&Place 거동 클로즈업 렌더★ — 큐브가 명확히 보이는 시점으로
   1에피소드(생성→접근→Pick→Lift→운반→목표) 렌더 + 단계별 라벨 스토리보드 저장.

실행: python diagnostics\\render_pickplace.py --headless
산출: <out>/pickplace_full.mp4 (20fps=실시간) + storyboard_*.png (단계별)
⚠️ isaaclab import → 프리앰블/AppLauncher 이후에만.
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
preload_native_training_modules()



_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
# scripts/rsl_rl(cli_args) + repo 루트 import 경로
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

import argparse

from isaaclab.app import AppLauncher

import cli_args  # scripts/rsl_rl/cli_args.py  # isort: skip

_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"  # pregrasp 없는 순수 거동
_EXP = "ur5e_binpick_teacher"

parser = argparse.ArgumentParser(description="teacher Pick&Place 클로즈업 렌더.")
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--seed", type=int, default=7, help="롤아웃 seed(성공 케이스 탐색용).")
parser.add_argument("--steps", type=int, default=170, help="렌더 스텝(1에피소드≈160).")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--tries", type=int, default=6,
                    help="안착 성공 롤아웃이 나올 때까지 최대 재시도 횟수(정성 그림용).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--task", type=str, default=_TASK,
                    help="렌더 태스크(student: Isaac-BinPick-UR5e-Vision-Eval-v0 + --checkpoint).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
_TASK = args_cli.task
# ★num_envs=1은 PhysX GPU 파이프라인이 배치와 다르게 동작해 파지 실패(2026-07-22 실측:
#   eval num_envs4 placed 95.8% vs render num_envs1 실패). env 0만 렌더하되 배치 8로 물리 정합.
args_cli.num_envs = 4
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401

from ur5e_binpick.tasks.binpick.binpick_env_cfg import MIN_LIFT
from ur5e_binpick.tasks.binpick.mdp.scene import (
    BELT_CENTER_X, BELT_CENTER_Y, NUM_OBJECTS, OBJECT_SIZE, TABLE_HEIGHT, object_names)


def _latest_ckpt() -> str:
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    root = os.path.join(_REPO_ROOT, "logs", "rsl_rl", _EXP)
    runs = sorted(glob.glob(os.path.join(root, "*")), key=os.path.getmtime)
    for run in reversed(runs):
        ck = sorted(glob.glob(os.path.join(run, "model_*.pt")),
                    key=lambda p: int(p.split("model_")[-1].split(".pt")[0]))
        if ck:
            return ck[-1]
    raise RuntimeError(f"체크포인트 없음: {root}")


def _write_mp4(frames, path, fps):
    try:
        import cv2
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            vw.write(cv2.cvtColor(np.asarray(fr).astype(np.uint8), cv2.COLOR_RGB2BGR))
        vw.release()
        return True
    except Exception as e:
        print(f"[pp][warn] mp4 실패: {e}")
        return False


def _label(img, text):
    """프레임 좌상단에 단계 라벨 (PIL)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(img).astype(np.uint8)).copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 12 + 9 * len(text), 34], fill=(0, 0, 0))
    d.text((8, 8), text, fill=(255, 230, 60))
    return im


def main() -> None:
    out_dir = args_cli.out or os.path.join(os.environ.get("TEMP", "/tmp"), "pickplace_render")
    os.makedirs(out_dir, exist_ok=True)
    ckpt = _latest_ckpt()
    print(f"[pp] checkpoint = {ckpt}")

    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=int(args_cli.num_envs),
                            use_fabric=not args_cli.disable_fabric)
    # ★env_cfg.seed는 --seed 명시 시에만 설정★ (2026-07-22 파지 실패 근본수정): seed를
    #   항상 고정하면 특정 실패 초기분포로 고정돼 파지 실패(eval은 seed 미설정=성공). 기본은
    #   미설정(랜덤)으로 두고, 재현이 필요할 때만 --seed로 지정.
    if args_cli.seed is not None and int(args_cli.seed) != 7:
        env_cfg.seed = int(args_cli.seed)
    # ★큐브가 보이는 시점(2026-07-22 각도 개선)★ — 픽존 앞-측면 3/4·1.15m로 로봇 팔+벨트+
    #   공중 운반이 프레임에, lookat을 픽·운반 중간(z 0.14)에 둬 파지→들어올림→운반이 명확.
    #   (더 가깝거나 정면이면 파지 순간 그리퍼가 카메라를 가림 — play.py 실측 후 이 거리 채택.)
    # ★2026-08-03 수정★: 구버전은 `eye=(1.14,0.72,0.64)`·`lookat=(0.44,0,0.14)`로 ★+x쪽 하드코딩★
    #   이었다. 2026-07-28 배치 180° 회전으로 작업 영역이 ★−x★로 옮겨졌는데 이 카메라만 갱신되지
    #   않아 ★빈 바닥만 렌더★됐다(필름스트립에 로봇·큐브가 안 보임).
    #   ⇒ 좌표를 다시 박지 않고 ★씬 상수에서 파생★한다 — 배치가 또 바뀌어도 따라간다.
    #   관심 영역은 '로봇 base(0,0)'와 '벨트 중심'을 잇는 구간 전체이므로 lookat을 그 중간
    #   (수평 0.72배)·벨트면 +30cm에 두고 1.5m 거리에서 3/4 부감으로 잡는다(팔 잘림 방지).
    _side = 1.0 if BELT_CENTER_X >= 0.0 else -1.0          # 작업 영역이 있는 쪽(base 기준)
    _lx, _ly, _lz = BELT_CENTER_X * 0.72, BELT_CENTER_Y, TABLE_HEIGHT + 0.30
    env_cfg.viewer.eye = (_lx + _side * 0.95, _ly + _side * 0.95, _lz + 0.62)
    env_cfg.viewer.lookat = (_lx, _ly, _lz)
    env_cfg.viewer.origin_type = "env"    # env 0 원점 기준 — num_envs>1에서도 안전
    env_cfg.viewer.env_index = 0
    print(f"[pp] 카메라 eye={env_cfg.viewer.eye} lookat={env_cfg.viewer.lookat} "
          f"(벨트중심 x={BELT_CENTER_X:+.2f}에서 파생)")
    agent_cfg = cli_args.parse_rsl_rl_cfg(_TASK, args_cli)

    env = gym.make(_TASK, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped
    dev = uenv.device
    cube = uenv.scene[object_names(NUM_OBJECTS)[0]]
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # student(증류) ckpt는 DistillationRunner로 로드 — BinPick_Eval의 러너 분기와 동일
    _runner_cls = DistillationRunner if getattr(agent_cfg, "class_name", "OnPolicyRunner") == "DistillationRunner" else OnPolicyRunner
    runner = _runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=dev)

    # ★렌더러 예열★: 첫 ~30프레임은 검정(파이프라인 초기화). 정책 정지(env 미전진) 상태에서
    #   충분히(45회) 렌더해 완전 예열 → 롤아웃 frame 0부터 유효(큐브 벨트 안착 장면 포착).
    with torch.inference_mode():
        for _ in range(45):
            uenv.render()
    # ★obs는 예열 '후'에 취득★ (2026-07-22 파지 실패 근본수정): 예열 전 get_observations()는
    #   gym.make 직후 리셋이 확정되지 않은 stale obs를 반환 → 첫 정책 액션이 큐브 위치를
    #   잘못 보고 파지 실패(eval은 예열이 없어 정상이었음). 예열 후 재취득해 유효 obs로 시작.
    obs = env.get_observations()  # rsl_rl 3.x(2.3.2 래퍼): TensorDict 단일 반환
    # ★2026-08-03 재시도 도입★: 정책 성공률이 100%가 아니므로(placed ≈ 92%) 단발 롤아웃은
    #   ★실패 장면을 논문 그림으로 남길 수 있다★ — 실제로 발생했다(5단계가 전부 frame 0으로 붕괴).
    #   ⇒ 안착에 성공한 에피소드가 나올 때까지 재시도한다. 정량 성능은 BinPick_Eval이 따로
    #     보고하므로 이 그림은 '태스크가 무엇인가'를 보이는 ★정성 산출물★이다.
    _REST_Z = TABLE_HEIGHT + OBJECT_SIZE / 2.0        # 벨트 위 안착 시 큐브 중심 z
    frames, czs = [], []
    for _try in range(1, int(args_cli.tries) + 1):
        frames, czs = [], []
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                fr = uenv.render()
                cz = float(cube.data.root_pos_w[0, 2] - uenv.scene.env_origins[0, 2])
                if fr is not None:
                    frames.append(np.asarray(fr)); czs.append(cz)
        _z = np.asarray(czs) if czs else np.zeros(1)
        # ★스폰 낙하 배제★: 큐브는 공중(≈0.08)에서 스폰돼 벨트로 떨어진다. 그 초기 높이를
        #   '들어올림'으로 오인하면 ★실패한 롤아웃이 성공으로 판정★된다(실제로 2회 발생).
        #   ⇒ 벨트에 처음 안착한 시점(t0) ★이후★ 구간에서만 최고점을 찾는다.
        _w = np.where(_z < _REST_Z + 0.010)[0]
        _t0 = int(_w[0]) if len(_w) else 0
        _post = _z[_t0:]
        _apex = float(_post.max()) if len(_post) else 0.0
        _ai = int(_post.argmax()) if len(_post) else 0
        _ok = bool(_apex > MIN_LIFT + 0.02 and _post[_ai:].min() < _REST_Z + 0.02)
        print(f"[pp] 롤아웃 시도 {_try}: 안착 t0={_t0} · 이후 최고 z={_apex:.3f} "
              f"(MIN_LIFT {MIN_LIFT:.3f}) → {'★픽앤플레이스 성공★' if _ok else '실패(재시도)'}")
        if _ok:
            break
        obs = env.get_observations()

    # ★검정(예열 잔여) 프레임 제거 — 2026-08-03 상대 기준으로 정정★
    #   구버전은 `mean(frame) > 12.0` ★절대 밝기★로 걸렀다. 카메라를 넓은 3/4 부감으로 옮기자
    #   장면 평균이 낮아져 ★유효 프레임 170장 중 11장만 남고★ 단계 판정이 붕괴했다.
    #   의도는 "앞쪽 예열 검정 구간만 버리기"이므로, 밝기 최대값 대비 상대 임계로 ★앞부분만★ 자른다.
    # ★2026-08-04(3차) 전역 필터로 정정★: 2차는 '앞쪽 예열 구간만 잘라내는' 방식이었는데,
    #   실측 결과 렌더러가 ★초반 이후에도 검정 프레임을 반환★한다(170장 중 유효 ~5~11장).
    #   그래서 중간·뒤쪽 검정이 그대로 스토리보드에 선택됐다(필름스트립 2~5번이 검정).
    #   ⇒ 밝기 상대 임계로 ★전역 필터★하고 czs를 함께 정렬한다. 유효 프레임이 적으면 경고한다.
    #   ⚠️근본 원인(렌더러가 프레임을 드롭하는 이유)은 미해결 — Future_Plan P6 참조.
    _b = np.array([float(np.mean(f)) for f in frames])
    _thr = 0.25 * float(_b.max()) if len(_b) else 0.0
    _keep = np.where(_b > _thr)[0]
    if len(_keep) == 0:
        print("[pp][warn] 유효 프레임 0 — 스토리보드 생략"); env.close(); return
    print(f"[pp] 프레임 {len(frames)}장 중 ★유효 {len(_keep)}장★ (밝기 max {_b.max():.1f} 임계 {_thr:.1f})")
    if len(_keep) < 30:
        print(f"[pp][warn] ⚠️유효 프레임이 {len(_keep)}장뿐 — 렌더러가 프레임을 드롭하고 있다. "
              f"필름스트립이 태스크 전 구간을 담지 못한다(Future_Plan P6).")
    frames = [frames[k] for k in _keep]; czs = [czs[k] for k in _keep]
    if not frames:
        print("[pp] 프레임 없음"); env.close(); return
    # mp4
    mp4 = os.path.join(out_dir, "pickplace_full.mp4")
    _write_mp4(frames, mp4, args_cli.fps)

    # ── 단계별 스토리보드 프레임 선정 ──
    # ★2026-08-03 재작성★: 구 로직은 `i_place = argmax(cube_z)`로 ★최고점을 place로 간주★했다.
    #   그건 목표가 '공중 hover'이던 구 태스크의 정의다. 현행 목표는 ★벨트 표면 안착★(target z)
    #   이라 최고점은 place가 아니라 ★운반 중★이며, 임계 0.05/0.12/0.24도 구 벨트 높이 기준이었다.
    #   ⇒ 현행 정의에 맞춰 ①상승 시작 ②MIN_LIFT 돌파 ③최고점(운반) ④하강 후 안착 으로 잡고,
    #     임계는 전부 ★씬 상수에서 파생★한다(하드코딩 금지).
    czs = np.array(czs)
    n = len(czs)
    # ★t0(첫 안착) 이후에서만 단계를 찾는다★ — 스폰 낙하 구간이 최고점으로 잡히면 전 단계가 붕괴한다
    _w0 = np.where(czs < _REST_Z + 0.010)[0]
    t0 = int(_w0[0]) if len(_w0) else 0
    i_apex = t0 + int(np.argmax(czs[t0:]))                     # 운반 중 최고점
    _rise = _REST_Z + 0.010                                    # 벨트를 떠나기 시작하는 높이
    w = np.where(czs[t0: i_apex + 1] > _rise)[0]
    i_pick = t0 + int(w[0]) if len(w) else max(t0, i_apex - 5)
    w = np.where(czs[t0: i_apex + 1] > MIN_LIFT)[0]
    i_lift = t0 + int(w[0]) if len(w) else max(i_pick, i_apex - 3)
    i_carry = i_apex
    # ★place = 최고점 이후 다시 안착 높이로 내려온 첫 프레임★ (없으면 마지막)
    w = np.where(czs[i_apex:] < _REST_Z + 0.020)[0]
    i_place = int(i_apex + w[0]) if len(w) else n - 1
    # spawn = 상승 시작 직전, 벨트에 놓여 있는 프레임
    w = np.where(czs[t0: max(t0 + 1, i_pick)] < _rise)[0]
    i_spawn = t0 + int(w[-1]) if len(w) else t0
    picks = [("1 SPAWN (cube on belt)", i_spawn),
             ("2 PICK (grasp)", i_pick),
             ("3 LIFT", i_lift),
             ("4 MOVE/CARRY", i_carry),
             ("5 PLACE (released on belt)", i_place)]
    from PIL import Image
    for tag, idx in picks:
        idx = int(np.clip(idx, 0, n - 1))
        Image.fromarray(_label_arr := np.asarray(_label(frames[idx], tag))).save(
            os.path.join(out_dir, f"storyboard_{tag.split()[0]}.png"))
        print(f"[pp] {tag}: frame {idx}  cube_z={czs[idx]:.3f}")
    env.close()
    print(f"[pp] ★영상★ {mp4}")
    print(f"[pp] ★스토리보드★ {out_dir}/storyboard_*.png")


if __name__ == "__main__":
    main()
    simulation_app.close()
