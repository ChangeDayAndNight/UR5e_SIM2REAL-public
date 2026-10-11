#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# UR5e Sim2Real 프리앰블 (README.md §9) — isaac51 python 자기 재실행 + repo 루트.
# ─────────────────────────────────────────────────────────────────────────────
"""★큐브 가시율 프로브★ — 손목 카메라 시야에 큐브가 들어있는 스텝 비율을 잰다 (진단 전용).

왜 이 파일이 존재하는가 (2026-08-18)
────────────────────────────────────────────────────────────────────────────
F66 Teacher는 채택 평가 placed ★99.85%★를 냈는데, 그 teacher에서 증류한 RGB-D student는
무보조 7,058 iteration 동안 성공률 ★0.15~0.24%★로 평탄했고 behavior loss도 0.23에서
멈췄다(iter 12,072에서 중단). `vision_obs_probe`는 depth에 공간 구조가 있고(std 0.241,
zero 0%) rgb·tcp_pose·target_pos가 모두 정상임을 확인했다 — ★depth-blind는 아니다★.

남은 가설: ★부분관측★. Teacher 관측은 특권 상태(object_pos/vel, contact_forces)라 큐브
위치를 ★항상★ 알지만, student는 손목 카메라에 큐브가 들어올 때만 안다. 그리고 현행
student(`VisionStudentTeacherResNet`)는 ★메모리가 없어서★ 시야를 벗어나는 순간 정보가
사라진다. 그러면 student가 낼 수 있는 최선은 큐브 위치에 대한 조건부 평균이고, 그것이
비가역 behavior loss와 grasp 실패로 나타난다.

이 저장소에는 같은 실패의 전례가 있다 — `configs/common.yaml` frame_mode 주석:
  "42.5° 정사각에서 홈 자세 큐브 시야 내 비율 ★18.3%★ → 에피소드 82%가 큐브 불가시
   시작 → student 접근 불능(eval grasp 0.8%)". `frame_mode: full`로 48.6%까지 올려 고쳤다.

⇒ 현행 계약에서 그 비율이 여전히 낮으면 가설이 확정되고, 수정 방향은 ★순환 student★
  (시야 밖 기억) 또는 ★카메라 기하 변경★으로 갈린다.

왜 렌더링을 쓰지 않는가
────────────────────────────────────────────────────────────────────────────
`rgbd_failure_probe`는 `omni.syntheticdata`에서 기동 1.5초 만에 segfault했다. 가시성은
★순수 기하★로 계산할 수 있으므로(카메라 pose = FK × hand_eye T_flange_cam, 큐브 위치는
scene에서 직접) 렌더러를 태울 이유가 없다. Teacher env(카메라 없음)에서 4096 env를 그대로
돌려 표본을 크게 잡는다 — 빠르고 크래시 위험도 없다.

⚠️ 진단 전용 — 학습 코드·보상·cfg를 일절 수정하지 않는다.

실행:
  python diagnostics/cube_visibility_probe.py --checkpoint <teacher.pt> --headless
산출: 콘솔 요약 + Phase2/Result/diagnostics/cube_visibility_<시각>/visibility.json
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import (  # noqa: E402
    configure_utf8_stdio,
    preload_native_training_modules,
    reexec_in_training_python,
)

reexec_in_training_python(__file__)
configure_utf8_stdio()
preload_native_training_modules()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "scripts", "rsl_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="큐브 가시율 기하 프로브(진단 전용).")
parser.add_argument("--task", type=str, default="Isaac-BinPick-UR5e-Teacher-Eval-v0")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--n_episodes", type=int, default=512)
parser.add_argument("--max_steps", type=int, default=4000)
# --checkpoint 는 cli_args.add_rsl_rl_args 가 이미 제공한다(중복 선언 금지).
parser.add_argument("--seed", type=int, default=42)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

import ur5e_binpick.tasks  # noqa: F401,E402

from ur5e_binpick.tasks.binpick.binpick_env_cfg import (  # noqa: E402
    _lifted_latch,
    _nearest_object,
)
from ur5e_binpick.tasks.binpick.mdp.observations import tcp_pose_b as _tcp_b  # noqa: E402
from common.config_io import configs_dir, load_yaml  # noqa: E402

_HE = load_yaml(configs_dir() / "calib" / "hand_eye.yaml")
_SIM = load_yaml(configs_dir() / "sim.yaml")
_HFOV = float(_SIM["camera"]["horizontal_fov_deg"])
_VFOV = float(_SIM["camera"]["vertical_fov_deg"])
_NEAR = float(_SIM["camera"]["clipping_near_m"])
_OBJ = float(_SIM["scene"]["object_size_m"])


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    if hasattr(env_cfg, "seed"):
        env_cfg.seed = args_cli.seed
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    dev = uenv.device

    # flange(wrist_3) → camera(optical) 고정 변환 — 실기/sim 단일 소스.
    T = torch.tensor(_HE["T_flange_cam"], dtype=torch.float32, device=dev)
    R_fc, t_fc = T[:3, :3], T[:3, 3]
    tan_h = math.tan(math.radians(_HFOV) * 0.5)
    tan_v = math.tan(math.radians(_VFOV) * 0.5)
    body_ids, _ = uenv.scene["robot"].find_bodies(["wrist_3_link"])
    wid = body_ids[0]
    env_origin = uenv.scene.env_origins

    n = uenv.num_envs
    vis_steps = torch.zeros(n, dtype=torch.long, device=dev)
    tot_steps = torch.zeros(n, dtype=torch.long, device=dev)
    vis_pre = torch.zeros(n, dtype=torch.long, device=dev)   # 파지 전(래치 전) 구간
    tot_pre = torch.zeros(n, dtype=torch.long, device=dev)
    first_vis = torch.full((n,), -1, dtype=torch.long, device=dev)
    tcp_z_lat: list[float] = []   # ★들림(latch) 중 TCP z★ — carried latch 임계 실측용
    tcp_z_all: list[float] = []
    ep_steps = torch.zeros(n, dtype=torch.long, device=dev)
    recs: list[dict] = []

    obs = env.get_observations()   # rsl_rl 3.1.2: 단일 반환(PlacedProbe와 동일 관례)
    for _ in range(args_cli.max_steps):
        if len(recs) >= args_cli.n_episodes:
            break
        with torch.inference_mode():
            act = policy(obs)
        # ── 카메라 프레임에서의 큐브 위치 (world → flange → camera) ──
        rd = uenv.scene["robot"].data
        p_w = rd.body_pos_w[:, wid, :]                 # flange 위치(world)
        q_w = rd.body_quat_w[:, wid, :]                # flange 회전(world, wxyz)
        R_wf = matrix_from_quat(q_w)                   # (N,3,3)
        cube_w, _ = _nearest_object(uenv)              # base frame
        cube_world = cube_w + env_origin               # world frame으로 환원
        d_w = (cube_world - p_w).unsqueeze(-1)         # flange→cube (world)
        d_f = torch.bmm(R_wf.transpose(1, 2), d_w).squeeze(-1)   # flange frame
        d_c = (d_f - t_fc) @ R_fc                      # camera(optical) frame
        z = d_c[:, 2]
        # optical 관례: +z 전방, x 우, y 하. 큐브 반변만큼의 여유를 준다(중심이 살짝 밖이어도
        # 몸통 일부가 보이면 '보인다'로 센다 — 가시율을 과소평가하지 않기 위함).
        margin = _OBJ * 0.5
        visible = (
            (z > _NEAR)
            & (d_c[:, 0].abs() - margin < tan_h * z)
            & (d_c[:, 1].abs() - margin < tan_v * z)
        )
        latched = _lifted_latch(uenv)
        _tz = _tcp_b(uenv)[:, 2]
        tcp_z_all.extend(_tz.tolist())
        if bool(latched.any()):
            tcp_z_lat.extend(_tz[latched].tolist())
        vis_steps += visible.long()
        tot_steps += 1
        pre = ~latched
        vis_pre += (visible & pre).long()
        tot_pre += pre.long()
        first_vis = torch.where((first_vis < 0) & visible, ep_steps, first_vis)
        ep_steps += 1

        obs, _, dones, _ = env.step(act)
        did = torch.nonzero(dones.flatten()).flatten()
        for i in did.tolist():
            recs.append({
                "vis_frac": float(vis_steps[i]) / max(1, int(tot_steps[i])),
                "vis_frac_pregrasp": (float(vis_pre[i]) / int(tot_pre[i])) if int(tot_pre[i]) > 0 else float("nan"),
                "first_visible_step": int(first_vis[i]),
                "ep_steps": int(ep_steps[i]),
            })
        if did.numel() > 0:
            z0 = torch.zeros_like(ep_steps[did])
            vis_steps[did] = z0; tot_steps[did] = z0
            vis_pre[did] = z0; tot_pre[did] = z0
            ep_steps[did] = z0
            first_vis[did] = -1

    import statistics as st
    vf = [r["vis_frac"] for r in recs]
    vp = [r["vis_frac_pregrasp"] for r in recs if r["vis_frac_pregrasp"] == r["vis_frac_pregrasp"]]
    fv = [r["first_visible_step"] for r in recs]
    never = sum(1 for v in fv if v < 0)
    out = {
        "n_episodes": len(recs),
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "hfov_deg": _HFOV, "vfov_deg": _VFOV, "object_size_m": _OBJ,
        "vis_frac_mean": st.mean(vf) if vf else float("nan"),
        "vis_frac_median": st.median(vf) if vf else float("nan"),
        "vis_frac_pregrasp_mean": st.mean(vp) if vp else float("nan"),
        "vis_frac_pregrasp_median": st.median(vp) if vp else float("nan"),
        "never_visible_pct": 100.0 * never / max(1, len(recs)),
        "first_visible_step_median": st.median([v for v in fv if v >= 0]) if any(v >= 0 for v in fv) else float("nan"),
    }
    print("\n" + "=" * 62)
    import numpy as _np
    if tcp_z_lat:
        _q=[50,75,90,95,99]
        print("\n★TCP z 분포 [m] — carried latch 임계(lift_z_m) 실측★")
        print("  들림 중 : " + "  ".join("p%d=%.4f"%(a,_np.percentile(tcp_z_lat,a)) for a in _q)
              + "  max=%.4f"%max(tcp_z_lat))
        print("  전 구간 : " + "  ".join("p%d=%.4f"%(a,_np.percentile(tcp_z_all,a)) for a in _q)
              + "  max=%.4f"%max(tcp_z_all))
        for _t in (0.040,0.060,0.075,0.100):
            print("  들림 중 TCP z > %.3f 비율: %.2f%%"%(_t,
                  100.0*sum(1 for v in tcp_z_lat if v>_t)/len(tcp_z_lat)))
    else:
        print("\n[warn] 들림(latch) 표본이 없어 TCP z 분포를 낼 수 없다")
    print("★큐브 가시율 (손목 카메라 frustum, 순수 기하)★")
    print(f"  에피소드 {out['n_episodes']}  ·  FOV {_HFOV}°×{_VFOV}°  ·  큐브 {_OBJ*100:.0f}cm")
    print(f"  전체 스텝 가시율   평균 {100*out['vis_frac_mean']:.1f}%  중앙 {100*out['vis_frac_median']:.1f}%")
    print(f"  ★파지 전 가시율★  평균 {100*out['vis_frac_pregrasp_mean']:.1f}%  중앙 {100*out['vis_frac_pregrasp_median']:.1f}%")
    print(f"  한 번도 안 보인 에피소드 : {out['never_visible_pct']:.1f}%")
    print(f"  처음 보이기까지 스텝(중앙): {out['first_visible_step_median']}")
    print("-" * 62)
    print("  판정 기준(저장소 전례): 홈 가시율 18.3% -> student grasp 0.8%(접근 불능),")
    print("                         frame_mode=full 전환 후 48.6%로 개선.")
    print("=" * 62)
    d = os.path.join(_REPO_ROOT, "Phase2", "Result", "diagnostics",
                     f"cube_visibility_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "visibility.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[probe] 산출: {d}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
