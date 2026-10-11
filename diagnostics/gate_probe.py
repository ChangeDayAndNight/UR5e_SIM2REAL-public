#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""파지 게이트 역학 직접 진단 — 정책 없이 스크립트 액션으로 게이트 발화 조건을 검증.

목적: close_bias 포화(1.93/2.0)에도 grasp 게이트 0 지속 — "그리퍼 열림" 가설 기각됨.
      pre-grasped 상태(frac=1.0 강제)에서 팔 정지(ik delta=0) + close 명령을 유지하며
      스텝별로 게이트 구성요소를 전부 기록해, 어디서 끊기는지 물리적으로 특정한다:
        (a) pregrasp 리셋이 실제로 큐브를 그리퍼에 넣는가 (cube_z ≈ 0.37?)
        (b) 양손 접촉력이 발생하는가 (fL/fR > 0.1N?)
        (c) 큐브-EE 상대속도가 게이트(0.2m/s) 안인가 (접촉 지터?)
        (d) 손가락 관절이 close(0.020)로 가는가
      25스텝 정지-유지 후 +z 상승 액션(들기)으로 파지 유지·게이트-이동 중 발화도 검증.
실행: python diagnostics\\gate_probe.py --headless   (GPU 필요, 학습과 동시 금지)
산출: 콘솔 스텝 테이블 + Phase1/Result/diagnostics/gate_probe_<ts>.csv

⚠️ isaaclab을 import하므로 프리앰블/AppLauncher 이후에만 import한다.
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
_REPO_ROOT = os.path.dirname(_HERE)

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
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

# ═══════════════════ AppLauncher ═══════════════════
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="파지 게이트 역학 진단 (스크립트 액션).")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--hold_steps", type=int, default=25, help="정지+close 유지 스텝.")
parser.add_argument("--lift_steps", type=int, default=25, help="이후 +z 상승 스텝.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────── sim 기동 후 ───────────────────────────
import csv
from datetime import datetime

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401

from ur5e_binpick.tasks.binpick.binpick_env_cfg import MIN_LIFT, _grasped_mask
from ur5e_binpick.tasks.binpick.mdp.scene import NUM_OBJECTS, object_names

_TASK = "Isaac-BinPick-UR5e-Teacher-v0"  # ★학습 태스크(pregrasp 포함) 그대로 진단★


def main() -> None:
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env = gym.make(_TASK, cfg=env_cfg)
    uenv = env.unwrapped
    n = uenv.num_envs
    dev = uenv.device

    # ★전 env pre-grasped 강제★ — 커리큘럼 대신 frac=1.0 (이벤트가 이 값을 읽음)
    uenv._bp_pregrasp_frac = 1.0
    obs, _ = env.reset()

    robot = uenv.scene["robot"]
    cube = uenv.scene[object_names(NUM_OBJECTS)[0]]
    ids, _ = robot.find_bodies("wrist_3_link")
    ee_id = ids[0]
    fj_ids, fj_names = robot.find_joints(["left_finger_joint", "right_finger_joint"])
    print(f"[gate] finger joints: {fj_names}")

    def _f(sname):
        s = uenv.scene.sensors.get(sname) if hasattr(uenv.scene.sensors, "get") else uenv.scene.sensors[sname]
        fm = s.data.force_matrix_w
        if fm is None:
            return torch.full((n,), float("nan"), device=dev)
        return fm.squeeze(1).norm(dim=-1).sum(dim=-1)

    # 손가락 패드 바디 인덱스 (그립점 실측용)
    fbL, _ = robot.find_bodies("left_finger_link")
    fbR, _ = robot.find_bodies("right_finger_link")

    # ★결정적 기하 검증(리셋 직후)★: 큐브가 실제로 두 손가락 '사이'에 놓였는가?
    #   pregrasp가 큐브를 _PREGRASP_GRIP(z=0.368)에 두는데, 이게 손가락 패드 사이가 아니라
    #   손목/TCP 지점이면 손가락은 큐브 위 빈 공간을 닫고 큐브는 낙하 → 관통처럼 보인다.
    with torch.inference_mode():
        c0 = cube.data.root_pos_w[0]
        pl0 = robot.data.body_state_w[0, fbL[0], :3]
        pr0 = robot.data.body_state_w[0, fbR[0], :3]
        wr0 = robot.data.body_state_w[0, ee_id, :3]
        pad_mid = (pl0 + pr0) / 2.0
        tcpw = uenv.scene["ee_frame"].data.target_pos_w[0, 0, :]
    print(f"[gate] ── 리셋 직후 world 위치(env0) ──")
    print(f"[gate]   cube          =({c0[0]:.4f},{c0[1]:.4f},{c0[2]:.4f})")
    print(f"[gate]   left_finger   =({pl0[0]:.4f},{pl0[1]:.4f},{pl0[2]:.4f})")
    print(f"[gate]   right_finger  =({pr0[0]:.4f},{pr0[1]:.4f},{pr0[2]:.4f})")
    print(f"[gate]   finger_mid    =({pad_mid[0]:.4f},{pad_mid[1]:.4f},{pad_mid[2]:.4f})")
    print(f"[gate]   wrist_3       =({wr0[0]:.4f},{wr0[1]:.4f},{wr0[2]:.4f})")
    print(f"[gate]   TCP(ee_frame) =({tcpw[0]:.4f},{tcpw[1]:.4f},{tcpw[2]:.4f})")
    print(f"[gate]   ★큐브↔finger_mid 거리 = {(c0-pad_mid).norm():.4f} m "
          f"(큐브가 손가락 사이면 ~0, 크면 파지불가)★")
    print(f"[gate]   큐브↔left={ (c0-pl0).norm():.4f}  큐브↔right={(c0-pr0).norm():.4f}  "
          f"손가락간격={(pl0-pr0).norm():.4f}")

    out = os.path.join(_REPO_ROOT, "Phase1", "Result", "diagnostics",
                       f"gate_probe_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    f = open(out, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["step", "phase", "env", "cube_z", "fL", "fR", "rel_speed",
                "fingerL_q", "fingerR_q", "grasped"])

    # ★워크플로 정정(5/5 에이전트)★: 이전 프로브는 그리퍼를 먼저 열고 무지지 큐브를 공중
    #   텔레포트+닫아서 자유낙하·관통이탈(아티팩트)했다. 실제 학습(_pregrasp_reset)처럼 큐브
    #   배치와 '동시에' 손가락을 contact_q(gap=큐브폭)로 straddle해야 진짜 파지·유지를 측정한다.
    #   → open phase 제거, close에서 straddle. 예측: q가 0.007에서 스톨·지속력 ~48N·큐브 유지.
    # ★클린 그립 테스트(핀 없음)★: pregrasp 리셋(frac=1.0)이 이미 큐브=파지중심, finger=contact_q
    #   (현행 50mm 큐브 접촉), 드라이브 target=close(0.020)로 그립을 세팅한다(events_dr._pregrasp_reset).
    #   여기선 인위적 핀을 걸지 않고 — 팔 정지(ik delta=0)+close 유지로 ①정적 유지(hold) ②상승(lift)에서
    #   큐브가 파지중심을 유지/따라오는지만 관측한다. ★핀 관통 아티팩트 제거 = 진짜 그립 성부 판정★.
    #   기대(솔버 반복수 수정 후): finger q가 큐브에 막혀 ~0.007에서 정지, fL·fR>0, cube_z 유지→상승.
    from ur5e_binpick.tasks.binpick.mdp.scene import OBJECT_SIZE
    from ur5e_binpick.tasks.binpick.mdp.ur5e_assets import GRIPPER_CLOSE_M, GRIPPER_MAX_OPEN_GAP_M
    contact_q = max(0.0, min(GRIPPER_CLOSE_M, (GRIPPER_MAX_OPEN_GAP_M - OBJECT_SIZE) / 2.0))
    # ★수정 검증(무override)★: pregrasp(_PREGRASP_GRIP·tcp_offset 정정 + 마찰재질 + 콜라이더 재변환)이
    #   큐브를 실제 pad 중심에 놓고 finger를 물린 상태로 리셋한다. 여기선 손대지 않고 그대로 hold+lift.
    print(f"[gate] ★수정 검증(무override) 그립 테스트★ contact_q={contact_q:.4f} pregrasp가 배치·물림 세팅")
    print(f"[gate] 기대: cube_z 유지(hold) + 상승(lift), fL·fR>0, grasp=1")
    z0 = (cube.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]).clone()
    print(f"[gate] 리셋 직후 cube_z: {[f'{v:.3f}' for v in z0.tolist()]}")
    phases = [("hold", 20), ("lift", 20)]
    print(f"{'st':>3} {'ph':<7} cube_z[env] maxF[env] q[env0] grasp[env]")
    step = 0
    try:
        for phase, count in phases:
            for _ in range(count):
                actions = torch.zeros(n, 7, device=dev)
                actions[:, -1] = -1.0                                        # 항상 close(<0)
                if phase == "lift":
                    actions[:, 2] = 0.05                                     # +z TCP delta(완만 — 실 정책 모션 근사)
                with torch.inference_mode():
                    env.step(actions)
                    cz = cube.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
                    fL, fR = _f("left_finger_contact"), _f("right_finger_contact")
                    ee_v = robot.data.body_lin_vel_w[:, ee_id, :]
                    rel = (cube.data.root_lin_vel_w - ee_v).norm(dim=-1)
                    fq = robot.data.joint_pos[:, fj_ids]
                    g = _grasped_mask(uenv, force_n=0.1, rel_speed_mps=0.2)
                for i in range(n):
                    w.writerow([step, phase, i, f"{cz[i]:.4f}", f"{fL[i]:.3f}", f"{fR[i]:.3f}",
                                f"{rel[i]:.3f}", f"{fq[i,0]:.4f}", f"{fq[i,1]:.4f}", int(bool(g[i]))])
                # 콘솔: 전 env 압축 표기 (dz 스윕 판독용)
                czs = " ".join(f"{cz[i]:.3f}" for i in range(n))
                fls = " ".join(f"{max(fL[i],fR[i]):.2f}" for i in range(n))
                gs = "".join(str(int(bool(g[i]))) for i in range(n))
                if n <= 8:
                    print(f"{step:>3} {phase:<5} cz[{czs}] f[{fls}] g[{gs}] qL={fq[0,0]:.4f}")
                step += 1
        # ★성공률 요약★: 마지막 lift 스텝에서 cube_z가 리셋높이 근처 이상(0.30) 유지 = 파지·상승 성공
        with torch.inference_mode():
            cz_final = cube.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
            g_final = _grasped_mask(uenv, force_n=0.1, rel_speed_mps=0.2)
        succ = int((cz_final > 0.30).sum().item())
        grasped_final = int(g_final.sum().item())
        # 실패-질량 상관: DR object_mass_scale(startup)로 env별 큐브 질량 고정 — 실패가 heavy/light?
        try:
            masses = cube.root_physx_view.get_masses().squeeze(-1).to(dev)
        except Exception:
            masses = torch.full((n,), float("nan"), device=dev)
        ok_m = masses[cz_final > 0.30]
        bad_m = masses[cz_final <= 0.30]
        print(f"\n[gate] ★파지·상승 성공률★: {succ}/{n} (cube_z>0.30 유지), grasped={grasped_final}/{n}")
        print(f"[gate]   성공 큐브질량 평균={ok_m.mean().item() if ok_m.numel() else float('nan'):.4f}kg "
              f"실패 평균={bad_m.mean().item() if bad_m.numel() else float('nan'):.4f}kg")
        print(f"[gate]   실패 env (cube_z, mass): "
              f"{[f'{i}:z{cz_final[i]:.3f}/m{masses[i]:.3f}' for i in range(n) if cz_final[i] <= 0.30]}")
    finally:
        f.close()
        env.close()
    print(f"[gate] ★산출★ {out}")
    print(f"[gate] MIN_LIFT={MIN_LIFT} — 성공 판독: lift 종료 시 cube_z>0.30 유지")


if __name__ == "__main__":
    main()
    simulation_app.close()
