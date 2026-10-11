#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9)
# ─────────────────────────────────────────────────────────────────────────────
"""★F91(2026-09-20)★ 2FG7 양 손가락 대칭 개폐 검증 프로브 — 액션 매니저를 거치지 않는 물리 직접 구동.

배경: 사용자가 학습 영상에서 "한쪽 손가락만 움직여 집는다"를 발견했다. 원인은 변환된 USD의
      right_finger_joint에 드라이브가 없고 PhysX mimic API(gearing −1, rotX 슬롯)만 남아
      수동 관절이 된 것(한계 −0.004 m에 걸린 채 큐브에 밀림). URDF의 mimic 태그를 제거하고
      독립 2관절로 재변환한 뒤, 이 프로브가 ★두 관절이 같은 목표로 같은 거리를 움직이는지★
      물리적으로 확인한다(실물 2FG7은 양측 대칭·2위치 그리퍼).
왜 액션 매니저를 안 쓰나: 릴리스 수퍼바이저(F68)가 TCP z > grasp_z_max(0.040)에서는 폐합을 막아
      홈 자세에서 close 명령이 항상 개방으로 덮인다(F83). 그리퍼 물리만 보려면 관절 목표를 직접 쓰고
      sim을 직접 스텝한다(팔은 홈 자세 유지 — 팔 목표는 현재값으로 고정).
검증: (A) pregrasp 리셋(frac=1.0)으로 5 cm 큐브가 손가락 사이에 있는 상태에서 close 목표 유지 40스텝:
         qL≈qR≈(OPEN_GAP−큐브)/2 ± 4 mm, |qL−qR| < tol, 양측 접촉력 > 0.1 N, 큐브 z 유지.
      (B) open 목표 40스텝: 둘 다 0 근처·대칭, 큐브 낙하(z 하강).
      (C) 빈 손 close 목표 40스텝: 둘 다 GRIPPER_CLOSE_M(0.020)에 수렴·대칭.
실행: python diagnostics\\gripper_symmetry_probe.py --headless   (GPU 필요, 학습과 동시 금지)
산출: 콘솔 판정(PASS/FAIL) + GRIPPER_SYMMETRY_OK/FAIL 센티널 + Phase1/Result/diagnostics/gripper_symmetry_<ts>.csv
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
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import argparse
import csv
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="2FG7 양 손가락 대칭 개폐 프로브")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--tol_mm", type=float, default=1.0, help="|qL−qR| 허용 [mm]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ur5e_binpick.tasks  # noqa: E402,F401  (gym 등록)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from ur5e_binpick.tasks.binpick.mdp.scene import NUM_OBJECTS, OBJECT_SIZE, object_names  # noqa: E402
from ur5e_binpick.tasks.binpick.mdp.ur5e_assets import (  # noqa: E402
    GRIPPER_CLOSE_M, GRIPPER_MAX_OPEN_GAP_M, GRIPPER_OPEN_M,
)

_TASK = "Isaac-BinPick-UR5e-Teacher-v0"


def _forces(uenv, name, n, dev):
    fm = uenv.scene.sensors[name].data.force_matrix_w
    if fm is None:
        return torch.zeros(n, device=dev)
    return fm.squeeze(1).norm(dim=-1).sum(dim=-1)


def main() -> int:
    tol = args_cli.tol_mm / 1000.0
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env = gym.make(_TASK, cfg=env_cfg)
    uenv = env.unwrapped
    n, dev = uenv.num_envs, uenv.device
    uenv._bp_pregrasp_frac = 1.0     # 커리큘럼이 리셋에서 덮어써도 warmup 구간 start=0.9라 대부분 pregrasp
    env.reset()
    robot = uenv.scene["robot"]
    cube = uenv.scene[object_names(NUM_OBJECTS)[0]]
    fj_ids, fj_names = robot.find_joints(["left_finger_joint", "right_finger_joint"])
    arm_ids = [i for i in range(robot.num_joints) if i not in fj_ids]
    dt = uenv.physics_dt
    sub = int(round(uenv.step_dt / dt))
    print(f"[sym] finger joints={fj_names} close={GRIPPER_CLOSE_M*1000:.1f}mm open gap={GRIPPER_MAX_OPEN_GAP_M*1000:.0f}mm "
          f"physics dt={dt:.4f}s x{sub}/policy step, envs={n}")

    out = os.path.join(_REPO_ROOT, "Phase1", "Result", "diagnostics",
                       f"gripper_symmetry_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fcsv = open(out, "w", newline="", encoding="utf-8")
    w = csv.writer(fcsv)
    w.writerow(["step", "phase", "env", "qL", "qR", "fL", "fR", "cube_z"])

    def drive(label, steps, finger_target, step0):
        """관절 목표를 직접 쓰고 sim을 policy-step 단위로 전진(액션 매니저·수퍼바이저 우회)."""
        last = None
        with torch.inference_mode():
            arm_hold = robot.data.joint_pos[:, arm_ids].clone()
            for k in range(steps):
                for _ in range(sub):
                    robot.set_joint_position_target(arm_hold, joint_ids=arm_ids)
                    robot.set_joint_position_target(
                        torch.full((n, len(fj_ids)), float(finger_target), device=dev), joint_ids=fj_ids)
                    uenv.scene.write_data_to_sim()
                    uenv.sim.step(render=False)
                    uenv.scene.update(dt)
                fq = robot.data.joint_pos[:, fj_ids].clone()
                fL = _forces(uenv, "left_finger_contact", n, dev)
                fR = _forces(uenv, "right_finger_contact", n, dev)
                cz = cube.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
                for i in range(n):
                    w.writerow([step0 + k, label, i, f"{fq[i,0]:.5f}", f"{fq[i,1]:.5f}", f"{fL[i]:.3f}", f"{fR[i]:.3f}", f"{cz[i]:.4f}"])
                if k % 10 == 0 or k == steps - 1:
                    print(f"  {label:<8} step {k:>2}  qL={fq[0,0]*1000:6.2f}mm qR={fq[0,1]*1000:6.2f}mm  "
                          f"|Δ|max={((fq[:,0]-fq[:,1]).abs().max()*1000):.2f}mm  fL={fL[0]:.1f}N fR={fR[0]:.1f}N  cz={cz[0]:.3f}")
                last = (fq, fL, fR, cz)
        return last

    verdicts = []
    with torch.inference_mode():
        fq0 = robot.data.joint_pos[:, fj_ids].clone()
        cz0 = cube.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
    in_hand = cz0 > 0.15
    print(f"[sym] 리셋 직후: qL={fq0[0,0]*1000:.2f} qR={fq0[0,1]*1000:.2f} mm, 손 안 큐브 env={int(in_hand.sum())}/{n}")
    expect_q = (GRIPPER_MAX_OPEN_GAP_M - OBJECT_SIZE) / 2.0

    # (A) 큐브를 쥔 채 close 유지
    fq, fL, fR, cz = drive("A_grip", 40, GRIPPER_CLOSE_M, 0)
    sel = in_hand
    dq = (fq[:, 0] - fq[:, 1]).abs()
    # 큐브를 쥔 상태의 좌우 차는 리셋 배치 오프셋(큐브 중심 vs 손가락 중점)이 만든다 — 드라이브 대칭은
    # 합계로 본다: qL+qR ≈ OPEN_GAP − 큐브 = 24 mm. 개별 차는 2·tol까지 허용(C 빈손 폐합이 tol로 대칭을 판정).
    qsum = fq[:, 0] + fq[:, 1]
    a1 = bool((dq[sel] < 2.0 * tol).all() and ((qsum[sel] - 2.0 * expect_q).abs() < 0.003).all()) if sel.any() else False
    a2 = bool(((fq[sel] - expect_q).abs() < 0.004).all()) if sel.any() else False
    a3 = bool((fL[sel] > 0.1).all() and (fR[sel] > 0.1).all()) if sel.any() else False
    a4 = bool((cz[sel] > 0.15).all()) if sel.any() else False
    verdicts += [("A1 파지 중 대칭 |qL−qR| < 2·tol ∧ qL+qR ≈ 24 mm", a1, f"max|Δ| {float(dq[sel].max()*1000) if sel.any() else float('nan'):.2f} mm, qL+qR={float(qsum[0]*1000):.2f} mm"),
                 ("A2 파지 q ≈ (74−50)/2 = 12 mm ±4", a2, f"qL={fq[0,0]*1000:.2f} qR={fq[0,1]*1000:.2f} mm"),
                 ("A3 양측 접촉력 > 0.1 N", a3, f"fL={fL[0]:.1f} fR={fR[0]:.1f} N"),
                 ("A4 큐브 유지(z > 0.15)", a4, f"cube_z={cz[0]:.3f}")]

    # (B) open — 둘 다 0으로, 큐브 낙하
    fq, fL, fR, cz = drive("B_open", 40, GRIPPER_OPEN_M, 40)
    dq = (fq[:, 0] - fq[:, 1]).abs()
    b1 = bool((fq.abs() < 0.0015).all()) and bool((dq < tol).all())
    b2 = bool((cz[sel] < 0.10).all()) if sel.any() else True
    verdicts += [("B1 개방: 양측 0 & 대칭", b1, f"qL={fq[0,0]*1000:.2f} qR={fq[0,1]*1000:.2f} mm, max|Δ| {float(dq.max()*1000):.2f} mm"),
                 ("B2 개방 후 큐브 낙하(z < 0.10)", b2, f"cube_z={cz[0]:.3f}")]

    # (C) 빈 손 close — 둘 다 20 mm
    fq, fL, fR, cz = drive("C_close", 40, GRIPPER_CLOSE_M, 80)
    dq = (fq[:, 0] - fq[:, 1]).abs()
    c1 = bool(((fq - GRIPPER_CLOSE_M).abs() < 0.0015).all()) and bool((dq < tol).all())
    verdicts += [("C1 빈 손 폐합: 양측 ≈20 mm & 대칭", c1, f"qL={fq[0,0]*1000:.2f} qR={fq[0,1]*1000:.2f} mm, max|Δ| {float(dq.max()*1000):.2f} mm")]

    fcsv.close()
    # ⚠️ env.close()가 Kit 앱을 내리면 이후 print가 실행되지 않는다 — 판정을 먼저 출력한다.
    print("\n[sym] ══════ 판정 ══════")
    ok_all = True
    for name, ok, detail in verdicts:
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  — {detail}")
    print(f"[sym] 산출: {out}")
    print("GRIPPER_SYMMETRY_OK" if ok_all else "GRIPPER_SYMMETRY_FAIL", flush=True)
    env.close()
    return 0 if ok_all else 1


if __name__ == "__main__":
    rc = main()
    simulation_app.close()
    raise SystemExit(rc)
