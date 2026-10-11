#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★pregrasp 보조 계측★ — 리셋 직후 '잡힌 채 시작'이 실제로 성립하는지 스텝별로 본다.

배경(2026-07-29, F30 후속): 그리퍼 자산을 실측으로 고친 뒤 학습에서 `Episode_Reward/grasp`가
      v3의 1/70로 떨어졌다. pregrasp_frac=0.75인데 iter0 lift가 0.0167 → ★리셋 직후 즉시
      큐브를 놓친다★는 뜻인데, Grasp_PhysicsTest(그리퍼 단독)는 39cm 들어올림에 성공한다.
      두 결과가 모순이라 ★실제 학습 env에서★ 무엇이 다른지 직접 계측해야 한다.

측정: 실제 teacher env를 소수 env로 띄우고 리셋 후 무행동(zero action)으로 N스텝 진행하며
      스텝마다 ①큐브 z(파지중심 대비) ②핑거 관절 q ③손가락 접촉력 ④grasp 게이트 성립 여부
      를 찍는다. 어느 스텝에서 무엇이 먼저 무너지는지가 곧 근본원인이다.

실행: python Phase0\\Pregrasp_Probe.py --num_envs 4 --steps 40 --headless
산출: 콘솔 표 + Phase0/logs/pregrasp_probe_<시각>.txt

⚠️ isaaclab은 AppLauncher 이후에만 import.
"""
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()



_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
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

parser = argparse.ArgumentParser(description="pregrasp 보조 계측 (리셋 직후 파지 유지 여부).")
parser.add_argument("--task", default="Isaac-BinPick-UR5e-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=40)
# ★2026-07-31 릴리스 계측 모드(F37 진단)★ — 잡은 상태에서 그리퍼를 열었을 때 큐브가
#   ★튕겨 나가는지★ 잰다. placed는 `속도 < max_speed(0.10 m/s)`(정지)를 요구하므로,
#   편측 침투 8mm x 강성 3000에 저장된 탄성에너지가 큐브를 밀어내면 안착이 구조적으로 실패한다.
parser.add_argument("--release-at", type=int, default=-1,
                    help="이 스텝부터 그리퍼 open 명령(기본 -1=사용 안 함). 큐브 속도를 함께 기록.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import datetime as _dt

import gymnasium as gym
import torch

import ur5e_binpick.tasks  # noqa: F401  (gym 등록)
from isaaclab_tasks.utils import parse_env_cfg

from ur5e_binpick.tasks.binpick.mdp.events_dr import _PREGRASP_GRIP
from ur5e_binpick.tasks.binpick.mdp.scene import OBJECT_SIZE


def main() -> int:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    # ★pregrasp를 전 env에 강제★ — 표본이 섞이면 원인 분리가 안 된다
    env._bp_pregrasp_frac = 1.0
    env.reset()

    robot = env.scene["robot"]
    obj = env.scene["object_0"]
    fids, _ = robot.find_joints(["left_finger_joint", "right_finger_joint"])
    grip = torch.tensor(_PREGRASP_GRIP, device=env.device)
    origins = env.scene.env_origins

    lines = []

    def emit(s: str) -> None:
        print(s)
        lines.append(s)

    emit(f"[probe] 큐브 {OBJECT_SIZE*1000:.0f}mm | pregrasp 파지중심(base) "
         f"({_PREGRASP_GRIP[0]:+.4f}, {_PREGRASP_GRIP[1]:+.4f}, {_PREGRASP_GRIP[2]:+.4f})")
    # ── pregrasp 배선 자체가 살아있는지 먼저 확인 (F30 후속 진단) ──
    from ur5e_binpick.tasks.binpick.mdp.ur5e_assets import (
        GRIPPER_CLOSE_M as _CQ, GRIPPER_MAX_OPEN_GAP_M as _OG)
    _pen = 0.007
    _clamp = max(0.0, min(_CQ, (_OG - OBJECT_SIZE + _pen) / 2.0))
    emit(f"[probe] 상수: OPEN {_OG:.3f} CLOSE {_CQ:.3f} _pen {_pen:.3f} -> clamp_q {_clamp:.4f} "
         f"(간격 {(_OG-2*_clamp)*1000:.1f}mm, 편측침투 {(OBJECT_SIZE-(_OG-2*_clamp))/2*1000:.1f}mm)")
    emit(f"[probe] env._bp_pregrasp_frac = {getattr(env, '_bp_pregrasp_frac', '★미설정★')}")
    _sn = list(env.scene.sensors.keys())
    emit(f"[probe] 등록된 센서: {_sn}")
    _fl = getattr(env, "_bp_pregrasp_env", None)
    emit(f"[probe] env._bp_pregrasp_env  = {'★미설정★' if _fl is None else f'{int(_fl.sum())}/{env.num_envs} env가 pregrasp'}")
    _ev = [n for n in dir(env.event_manager) if 'term' in n.lower()]
    try:
        _names = env.event_manager.active_terms.get("reset", [])
        emit(f"[probe] reset 이벤트 순서: {list(_names)}")
    except Exception as e:  # noqa: BLE001
        emit(f"[probe] 이벤트 목록 조회 실패: {e}")
    if args_cli.release_at >= 0:
        emit(f"[probe] ★릴리스 계측 모드★ step {args_cli.release_at}부터 open 명령 — "
             f"placed의 정지 임계는 0.10 m/s")
    emit(f"[probe] {'step':>4} {'큐브z':>8} {'파지중심대비Δz':>14} {'수평이탈':>9} "
         f"{'fingerL':>8} {'fingerR':>8} {'접촉력L':>8} {'접촉력R':>8} {'큐브속도':>8}")

    zero = torch.zeros((env.num_envs, env.action_space.shape[1]), device=env.device)
    for step in range(args_cli.steps):
        pos_b = obj.data.root_pos_w - origins            # (N,3) base
        dz = pos_b[:, 2] - grip[2]
        dxy = torch.norm(pos_b[:, :2] - grip[:2], dim=1)
        q = robot.data.joint_pos[:, fids]                # (N,2)

        # ★센서 이름은 left_finger_contact / right_finger_contact★ (binpick_env_cfg _grasped_mask와 동일).
        #   grasp 게이트는 net_forces가 아니라 ★force_matrix_w(객체별 filtered 접촉)★를 본다 —
        #   같은 것을 읽어야 게이트가 왜 안 열리는지 알 수 있다.
        fl = fr = float("nan")
        try:
            sl = env.scene.sensors.get("left_finger_contact")
            sr = env.scene.sensors.get("right_finger_contact")
            if sl is not None and sl.data.force_matrix_w is not None:
                fl = float(sl.data.force_matrix_w.squeeze(1).norm(dim=-1)[0].max().item())
            if sr is not None and sr.data.force_matrix_w is not None:
                fr = float(sr.data.force_matrix_w.squeeze(1).norm(dim=-1)[0].max().item())
        except Exception:  # noqa: BLE001
            pass

        spd = float(obj.data.root_lin_vel_w[0].norm().item())
        tag = ""
        if args_cli.release_at >= 0 and step >= args_cli.release_at:
            tag = "  <-open" + ("  ★튕김★" if spd > 0.10 else "")
        emit(f"[probe] {step:4d} {pos_b[0,2].item():8.4f} {dz[0].item():+14.4f} "
             f"{dxy[0].item():9.4f} {q[0,0].item():8.4f} {q[0,1].item():8.4f} "
             f"{fl:8.2f} {fr:8.2f} {spd:8.3f}{tag}")
        act = zero
        if args_cli.release_at >= 0 and step >= args_cli.release_at:
            act = zero.clone()
            act[:, -1] = 1.0          # ★open 명령★ (BinaryJointPositionAction: >0 = open)
        env.step(act)

    # 판정
    pos_b = obj.data.root_pos_w - origins
    held = (torch.abs(pos_b[:, 2] - grip[2]) < 0.03) & (torch.norm(pos_b[:, :2] - grip[:2], dim=1) < 0.03)
    emit(f"[probe] ★{args_cli.steps}스텝 후 파지중심 유지 env: {int(held.sum())}/{env.num_envs}★")
    emit("[probe] 판정: " + ("pregrasp 정상" if held.float().mean() > 0.5 else
                             "★pregrasp 실패 — 위 표에서 무너지는 스텝·양상 확인★"))

    outdir = os.path.join(_ROOT, "Phase0", "logs")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"pregrasp_probe_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[probe] 산출: {path}")
    env.close()
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)
