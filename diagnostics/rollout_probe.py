#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + repo 루트.
# ─────────────────────────────────────────────────────────────────────────────
"""STAGE 0 진단 프로브 — 정책 이상거동("빠름/부자연스러움")을 ★숫자로★ 확정한다.

목적 (README.md STAGE 0 — ★수정 금지, 진단 전용★):
  고정 seed 결정론 롤아웃 N에피소드에서 스텝별 지표를 CSV로 로깅:
    · 액션 벡터 norm
    · 관절별 |각속도| max/mean + 실기 스펙 한계(π rad/s) 대비 % ← "빠름"의 핵심 지표
      ⚠️ 2.1.0 implicit actuator는 cfg velocity_limit을 sim에 적용하지 않음(vendored
        actuator_pd.py L79-100 검증: "we continue to not use it" 후 None 처리) →
        sim 관절속도는 무클램프 — 실기 한계 초과 여부를 여기서 측정해야 함.
    · EE(TCP 바디) 선속도 [m/s]
    · 큐브 선속도 [m/s] ← 순간 스파이크 = 텔레포트/튕김/쳐냄 신호
    · 손가락↔큐브 접촉력(좌/우 개별) + 접촉 지속 프레임 ← 리워드 해킹(무파지 lift) 판별

검증된 API (vendored Isaac Lab 2.1.0, file:line):
  · env_cfg.seed = N → manager_based_env.py L84-88이 소비(replicator+torch+numpy 시딩)
  · robot.data.joint_vel (articulation_data.py L538), joint_vel_limits (L300)
  · robot.data.body_lin_vel_w (L794, (N,B,3)), robot.find_bodies (articulation.py L209),
    robot.find_joints (L224)
  · cube.data.root_lin_vel_w (rigid_object_data.py L263, (N,3))
  · ContactSensorData.force_matrix_w (contact_sensor_data.py L58-66, (N,B,M,3);
    filter_prim_paths_expr 필요, 비면 None) — ★filtered 접촉은 센서 prim_path가 env당
    단일 prim일 때만 보장(contact_sensor_cfg.py L47-51)★ → left/right 손가락 센서 분리.
  · articulation spawn에 activate_contact_sensors=True 필수(contact_sensor.py L266-268
    RuntimeError HINT; spawner_cfg.py L92) — 진단 env 인스턴스에서만 켠다(소스 무수정).

실행 (학습 PC, isaac51 — GPU 필요. 학습과 동시 실행 금지(8GB OOM)):
  python diagnostics\\rollout_probe.py --headless                     # 최신 ckpt
  python diagnostics\\rollout_probe.py --checkpoint <model_*.pt> --headless
산출: results/ur5e_binpick_teacher/probe_<ts>/ → probe_steps.csv + probe_episodes.csv
      + summary.txt (진단표 수치)

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
preload_native_training_modules()



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

# ═══════════════════ AppLauncher (isaaclab import은 이 아래에서만) ═══════════════════
import argparse

from isaaclab.app import AppLauncher

import cli_args  # scripts/rsl_rl/cli_args.py  # isort: skip

_TASK = "Isaac-BinPick-UR5e-Teacher-Eval-v0"  # pre-grasp 없는 eval 태스크(순수 거동)
_EXP = "ur5e_binpick_teacher"

parser = argparse.ArgumentParser(description="STAGE 0 진단 프로브 — 결정론 롤아웃 지표 CSV.")
parser.add_argument("--num_envs", type=int, default=8, help="병렬 env (기본 8 — 진단용 소규모).")
parser.add_argument("--n_episodes", type=int, default=20, help="집계할 완료 에피소드 수(기본 20).")
parser.add_argument("--seed", type=int, default=42, help="결정론 seed (env_cfg.seed로 주입).")
parser.add_argument("--max_steps", type=int, default=4000, help="안전 상한.")
parser.add_argument("--spike_mps", type=float, default=1.5, help="큐브 속도 스파이크 임계 [m/s].")
parser.add_argument("--contact_n", type=float, default=0.1, help="접촉 판정 힘 임계 [N].")
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.task = _TASK

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─────────────────────────── sim 기동 후 ───────────────────────────
import csv
import math
from datetime import datetime

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.sensors import ContactSensorCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import ur5e_binpick.tasks  # noqa: F401  (gym.register)

from common.robot_spec import UR5E_ARM_JOINT_NAMES
from ur5e_binpick.tasks.binpick.binpick_env_cfg import MIN_LIFT
from ur5e_binpick.tasks.binpick.mdp.scene import NUM_OBJECTS, object_names

# 실기 UR5e 관절속도 스펙(보수) — ur5e_assets.py cfg 의도값. sim은 무클램프이므로 이 값
# 대비 %가 "실기에서 불가능한 속도"의 직접 지표다.
_REAL_VEL_LIMIT_RAD_S = math.pi


def main() -> None:
    # ── env cfg + ★진단 전용 인스턴스 수정(소스 무수정 — STAGE 0 준수)★ ──
    env_cfg = parse_env_cfg(_TASK, device=args_cli.device, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)
    env_cfg.seed = int(args_cli.seed)  # 결정론(manager_based_env.py L84-88 소비)
    # 접촉 리포팅: 로봇+큐브 spawn에 contact reporter API 부여 (spawner_cfg.py L92)
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    for name in object_names(NUM_OBJECTS):
        getattr(env_cfg.scene, name).spawn.activate_contact_sensors = True
    # 손가락별 ContactSensor(단일 prim 매칭 — filtered 접촉 보장, cfg L47-51)
    _filter = [f"{{ENV_REGEX_NS}}/Object_{i}" for i in range(NUM_OBJECTS)]
    env_cfg.scene.left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_finger_link", filter_prim_paths_expr=_filter)
    env_cfg.scene.right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_finger_link", filter_prim_paths_expr=_filter)

    agent_cfg = cli_args.parse_rsl_rl_cfg(_TASK, args_cli)
    log_root = os.path.abspath(os.path.join(_REPO_ROOT, "logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[probe] 체크포인트: {resume_path}")
    print(f"[probe] seed={args_cli.seed} num_envs={args_cli.num_envs} n_episodes={args_cli.n_episodes}")

    env = gym.make(_TASK, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    dev = uenv.device
    n = uenv.num_envs
    robot = uenv.scene["robot"]
    cube = uenv.scene[object_names(NUM_OBJECTS)[0]]  # 진단은 object_0 기준(현재 1개)
    arm_ids, _ = robot.find_joints(list(UR5E_ARM_JOINT_NAMES))       # articulation.py L224
    ee_ids, _ = robot.find_bodies("wrist_3_link")                     # articulation.py L209
    ee_id = ee_ids[0]
    # sim측 한계(참고용 — PhysX/USD 값. cfg velocity_limit은 미적용임에 유의)
    sim_vel_limits = robot.data.joint_vel_limits[:, arm_ids]          # (N,6) AD L300

    # ContactSensor 핸들. ⚠️ prim 불일치/activate 미설정은 gym.make 단계에서 RuntimeError로
    # 이미 죽는다(contact_sensor.py L266-268) — 이 try는 scene.sensors 딕셔너리 키 부재만 잡는
    # 최후 방어이며, 여기 도달하면 접촉 지표는 n/a로 기록된다.
    try:
        s_left = uenv.scene.sensors["left_finger_contact"]
        s_right = uenv.scene.sensors["right_finger_contact"]
    except Exception as e:
        print(f"[probe] ⚠️ ContactSensor 미가용({e}) — 접촉 지표는 n/a로 기록")
        s_left = s_right = None
    contact_available = s_left is not None

    def _finger_force(sensor) -> torch.Tensor:
        """손가락↔큐브 접촉력 크기 (N,) — force_matrix_w (N,B=1,M,3) norm 합. None→NaN."""
        if sensor is None:
            return torch.full((n,), float("nan"), device=dev)
        fm = sensor.data.force_matrix_w  # (N,1,M,3) | None (filter 비면 — cfg상 항상 있음)
        if fm is None:
            return torch.full((n,), float("nan"), device=dev)
        return fm.squeeze(1).norm(dim=-1).sum(dim=-1)  # (N,)

    # ── 출력 준비 (증분 기록 — 중간 크래시에도 산출물 보존) ──
    out_dir = os.path.join(_REPO_ROOT, "Phase1", "Result", "diagnostics",
                           "probe_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)
    f_steps = open(os.path.join(out_dir, "probe_steps.csv"), "w", newline="", encoding="utf-8")
    w_steps = csv.writer(f_steps)
    w_steps.writerow([
        "env", "ep_id", "step_in_ep", "terminal", "action_norm",
        "arm_vel_max_rad_s", "arm_vel_mean_rad_s", "arm_vel_max_pct_real", "arm_vel_max_pct_sim",
        "ee_speed_mps", "cube_speed_mps", "cube_z_m", "rel_speed_cube_ee_mps",
        "force_left_n", "force_right_n", "contact_both",
    ])
    _EP_FIELDS = [
        "env", "ep_id", "steps", "term", "max_arm_vel_pct_real", "max_cube_speed_mps",
        "cube_spike_steps", "lift_steps", "lift_frac", "lift_with_contact_frac",
        "max_contact_run_steps",
    ]
    f_eps = open(os.path.join(out_dir, "probe_episodes.csv"), "w", newline="", encoding="utf-8")
    w_eps = csv.DictWriter(f_eps, fieldnames=_EP_FIELDS)
    w_eps.writeheader()

    # per-env 에피소드 누적기 (+ env별 에피소드 id — 스텝↔에피소드 CSV 조인 키)
    ep_id = torch.zeros(n, dtype=torch.long, device=dev)
    ep_steps = torch.zeros(n, dtype=torch.long, device=dev)
    ep_max_vel_pct = torch.zeros(n, device=dev)
    ep_max_cube_speed = torch.zeros(n, device=dev)
    ep_spike_steps = torch.zeros(n, dtype=torch.long, device=dev)       # 무접촉 고속 큐브
    ep_lift_steps = torch.zeros(n, dtype=torch.long, device=dev)
    ep_lift_contact_steps = torch.zeros(n, dtype=torch.long, device=dev)  # 들림∧양손접촉
    ep_contact_run = torch.zeros(n, dtype=torch.long, device=dev)       # 현재 연속 접촉
    ep_contact_maxrun = torch.zeros(n, dtype=torch.long, device=dev)    # 최장 연속 접촉
    episodes: list[dict] = []

    # ★리뷰 확정 사항★: ManagerBasedRLEnv.step()은 반환 '전에' done env를 리셋한다
    # (vendored manager_based_rl_env.py L214-242). 따라서 done 스텝에서 읽는 상태는
    # 리셋 후 값 → ① 종료 분류는 상태 재유도가 아니라 extras["time_outs"]로 판정
    # (RslRlVecEnvWrapper가 제공, vecenv_wrapper.py L184-185; dropped = done ∧ ¬time_out)
    # ② done env의 해당 스텝 상태 지표는 누적에서 제외하고 CSV에 terminal=1 + 상태 NaN 기록.
    obs = env.get_observations()  # rsl_rl 3.x(2.3.2 래퍼): TensorDict 단일 반환
    step = 0
    try:
        while len(episodes) < args_cli.n_episodes and step < args_cli.max_steps and simulation_app.is_running():
            with torch.inference_mode():
                actions = policy(obs)
                a_norm = actions.norm(dim=-1)                               # (N,)
                obs, _, dones, extras = env.step(actions)
                done_mask = dones.to(dev).bool()
                time_outs = extras.get("time_outs", torch.zeros(n, device=dev)).to(dev).bool()
                dropped_mask = done_mask & ~time_outs
                alive = ~done_mask  # 이번 스텝 상태 표본이 유효한 env

                jv = robot.data.joint_vel[:, arm_ids].abs()                  # (N,6) AD L538
                vel_max = jv.max(dim=1).values
                vel_mean = jv.mean(dim=1)
                pct_real = vel_max / _REAL_VEL_LIMIT_RAD_S * 100.0
                pct_sim = (jv / sim_vel_limits.clamp(min=1e-6)).max(dim=1).values * 100.0
                ee_v = robot.data.body_lin_vel_w[:, ee_id, :]                # (N,3) AD L794
                cube_v = cube.data.root_lin_vel_w                            # (N,3) ROD L263
                ee_speed = ee_v.norm(dim=-1)
                cube_speed = cube_v.norm(dim=-1)
                rel_speed = (cube_v - ee_v).norm(dim=-1)
                cube_z = cube.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]
                fL = _finger_force(s_left)
                fR = _finger_force(s_right)
                both = (fL > args_cli.contact_n) & (fR > args_cli.contact_n)

                # 스텝 CSV — done env는 terminal=1 + 상태 필드 NaN(리셋 후 값 오염 명시)
                for i in range(n):
                    term_flag = int(bool(done_mask[i]))
                    if term_flag:
                        state_cells = ["nan"] * 11
                    else:
                        state_cells = [
                            f"{vel_max[i]:.4f}", f"{vel_mean[i]:.4f}", f"{pct_real[i]:.1f}",
                            f"{pct_sim[i]:.1f}", f"{ee_speed[i]:.4f}", f"{cube_speed[i]:.4f}",
                            f"{cube_z[i]:.4f}", f"{rel_speed[i]:.4f}",
                            f"{fL[i]:.4f}", f"{fR[i]:.4f}", str(int(bool(both[i]))),
                        ]
                    w_steps.writerow([
                        i, int(ep_id[i]), int(ep_steps[i]) + 1, term_flag,
                        f"{a_norm[i]:.4f}", *state_cells,
                    ])

                # 에피소드 누적 — 상태 표본은 alive만(터미널 스텝 리셋값 오염 방지),
                # 에피소드 길이는 전 env(+1: 터미널 스텝도 에피소드의 실제 스텝).
                ep_steps += 1
                zerosf = torch.zeros_like(pct_real)
                ep_max_vel_pct = torch.maximum(ep_max_vel_pct, torch.where(alive, pct_real, zerosf))
                ep_max_cube_speed = torch.maximum(ep_max_cube_speed, torch.where(alive, cube_speed, zerosf))
                lifted = (cube_z > MIN_LIFT) & alive
                ep_lift_steps += lifted.long()
                contact_known = ~torch.isnan(fL)
                ep_lift_contact_steps += (lifted & both & contact_known).long()
                spike = (cube_speed > args_cli.spike_mps) & ~(both & contact_known) & alive
                ep_spike_steps += spike.long()
                ep_contact_run = torch.where(both & contact_known & alive, ep_contact_run + 1,
                                             torch.zeros_like(ep_contact_run))
                ep_contact_maxrun = torch.maximum(ep_contact_maxrun, ep_contact_run)

                done_idx = torch.nonzero(done_mask, as_tuple=False).flatten()
                for i in done_idx.tolist():
                    es = int(ep_steps[i])
                    if es <= 0:
                        continue
                    ls = int(ep_lift_steps[i])
                    # 들림 중 접촉비율은 '조건부' 지표 — 들림 0 또는 접촉 미가용이면 정의 안 됨("")
                    if ls > 0 and contact_available:
                        lwc = round(int(ep_lift_contact_steps[i]) / ls, 3)
                    else:
                        lwc = ""
                    rec = {
                        "env": i, "ep_id": int(ep_id[i]), "steps": es,
                        "term": "dropped" if bool(dropped_mask[i]) else "timeout",
                        "max_arm_vel_pct_real": round(float(ep_max_vel_pct[i]), 1),
                        "max_cube_speed_mps": round(float(ep_max_cube_speed[i]), 3),
                        "cube_spike_steps": int(ep_spike_steps[i]),
                        "lift_steps": ls,
                        "lift_frac": round(ls / es, 3),
                        "lift_with_contact_frac": lwc,
                        "max_contact_run_steps": int(ep_contact_maxrun[i]),
                    }
                    episodes.append(rec)
                    w_eps.writerow(rec)  # 증분 기록(크래시 내구성)
                    ep_id[i] += 1
                    for buf in (ep_steps, ep_lift_steps, ep_lift_contact_steps, ep_spike_steps,
                                ep_contact_run, ep_contact_maxrun):
                        buf[i] = 0
                    ep_max_vel_pct[i] = 0.0
                    ep_max_cube_speed[i] = 0.0
                if done_idx.numel():
                    f_eps.flush()
            step += 1
            if step % 200 == 0:
                f_steps.flush()
    finally:
        f_steps.close()
        f_eps.close()
        env.close()

    if not episodes:
        print("[probe] ⚠️ 완료 에피소드 0 — max_steps/num_envs 확인.")
        return

    # ── 진단 요약 (가설 판별 수치) ──
    m = len(episodes)
    def _avg(k): return sum(e[k] for e in episodes) / m
    def _mx(k): return max(e[k] for e in episodes)
    dropped_ratio = sum(1 for e in episodes if e["term"] == "dropped") / m
    # 들림 중 접촉비율 — 조건부 지표: 들림>0 ∧ 접촉 가용 에피소드만 모수(거짓 양성 방지)
    lwc_vals = [e["lift_with_contact_frac"] for e in episodes
                if isinstance(e["lift_with_contact_frac"], float)]
    if lwc_vals:
        lwc_line = (f"  들림 중 양손접촉 비율: 평균 {sum(lwc_vals)/len(lwc_vals)*100:.1f}%"
                    f" (모수 {len(lwc_vals)}/{m} ep — 낮음 = 무파지 lift = 해킹/스쿠핑 신호)\n")
    else:
        lwc_line = "  들림 중 양손접촉 비율: n/a (들림 에피소드 없음 또는 접촉센서 미가용)\n"
    summary = (
        f"STAGE 0 진단 프로브 — 결정론 롤아웃 (seed={args_cli.seed}, {m} ep)\n"
        f"{'='*64}\n"
        f"ckpt: {resume_path}\n"
        f"[가설(가) 폭주] 관절속도 (실기 스펙 π rad/s 대비 — sim은 무클램프):\n"
        f"  max % : 평균 {_avg('max_arm_vel_pct_real'):.1f}%  /  최악 {_mx('max_arm_vel_pct_real'):.1f}%"
        f"   (>100% = 실기에서 불가능한 속도)\n"
        f"[가설(다) 해킹/튕김] 큐브 속도·접촉:\n"
        f"  큐브 max 속도: 평균 {_avg('max_cube_speed_mps'):.2f} m/s / 최악 {_mx('max_cube_speed_mps'):.2f} m/s"
        f"   (스파이크 임계 {args_cli.spike_mps})\n"
        f"  무접촉 고속 스텝(스파이크): 평균 {_avg('cube_spike_steps'):.1f} / ep\n"
        f"{lwc_line}"
        f"  최장 연속 접촉: 평균 {_avg('max_contact_run_steps'):.1f} 스텝\n"
        f"[에피소드 구조]\n"
        f"  평균 길이 {_avg('steps'):.1f} 스텝 / dropped 종료 {dropped_ratio*100:.0f}%"
        f" / lift 시간비율 평균 {_avg('lift_frac')*100:.1f}%\n"
        f"{'-'*64}\n"
        f"산출: probe_steps.csv(스텝별, terminal=1 행은 상태 NaN=리셋후값) + probe_episodes.csv\n"
    )
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"[probe] ★산출★ {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
