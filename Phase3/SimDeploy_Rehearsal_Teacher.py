#!/usr/bin/env python
"""★미완성 — 실행 금지(2026-09-16)★ sim 폐루프 리허설

⚠️ 현재 이 스크립트는 ★환경 생성 직후 교착★한다. Kit 로그상 23초 지점에서
   `SimulationApp.close: Closing application`이 찍히는데 예외는 없고, 그 뒤 프로세스가
   종료되지 못한 채 매달린다(2026-09-15 4시간, 2026-09-16 25분 상한 종료 2회 실측).
   `PrivilegedBuilder`의 카메라 이중 읽기(observation_manager.compute)를 제거했는데도
   재현되므로 원인은 그쪽이 아니라 ★앱 수명 관리 구조★다.
   고치려면 검증된 `Phase3/SimDeploy_Rehearsal.py`의 기동/종료 구조(런타임 authority
   sentinel 발행 순서, main 반환 경로, app.close 시점)를 그대로 따라 다시 작성할 것.

   v1 번들의 핵심 검증은 이 스크립트 없이 이미 끝나 있다:
     · 정책 동치: 기록된 teacher 행동 12,800표본 max|Δ| 2.9e-6 (export 시 자동 검증)
     · 인식 정확도: 봉인 표본 3,072프레임 오차 p50 2.0~2.5cm (validate_teacher_perception_offline.py)
   이 리허설은 ★폐루프 성공률★이라는 추가 정보를 주는 것이지 배포 전제조건이 아니다.

★모듈형 배포 v1★ sim 폐루프 리허설 — 인식 모듈 + Teacher를 배포 코드 경로 그대로 sim에서. [학습 PC][Phase 3]

Phase3/SimDeploy_Rehearsal.py와 같은 구성(num_envs=1, SimRobot/SimCamera, common.rollout.run_rollout)
이지만 정책은 Teacher 번들이고 관측은 두 모드 중 하나다:
  --obs-mode privileged : env의 특권 42D(참 큐브 위치·접촉력)를 그대로 — 실행기·매퍼·수퍼바이저 체인 검증
                          (Teacher는 sim에서 100%였으므로 여기서 성공률이 낮으면 체인이 문제다)
  --obs-mode perception : common.teacher_obs.TeacherObsBuilder — depth 역투영 추정치로 구동
                          (= 실기와 같은 경로; 성공률 차이가 곧 인식·근사 항의 비용)
매 스텝 추정치를 참값과 대조해 오차 통계도 남긴다.

실행(GPU·Isaac 필요, 학습 루프와 동시 실행 금지):
  python Phase3/SimDeploy_Rehearsal_Teacher.py --policy-dir exported_experimental/teacher_<sha12> \\
      --obs-mode perception --episodes 50 --headless
산출: Phase3/logs/teacher_rehearsal_<시각>_<mode>/episodes.csv, summary.json
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "source/ur5e_binpick"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts.training_env import reexec_in_training_python  # noqa: E402
reexec_in_training_python(__file__)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--policy-dir", type=Path, required=True)
parser.add_argument("--obs-mode", choices=("privileged", "perception"), default="perception")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--max-steps", type=int, default=200)   # ★F93★ 학습 에피소드 10 s × 20 Hz
parser.add_argument("--seed", type=int, default=45)
parser.add_argument("--target-pos", default=None, help="고정 타겟 'x,y,z' (기본: common.yaml target.sample_range 무작위)")
parser.add_argument("--policy-device", default="cpu")
parser.add_argument("--gui", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = not args.gui
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from common.action_mapping import ActionMapper  # noqa: E402
from common.calibration import load_hand_eye, load_tcp_offset  # noqa: E402
from common.config_io import build_action_cfg, build_preprocess_cfg, build_safety_cfg, configs_dir, load_yaml  # noqa: E402
from common.kit_lifecycle import run_main_with_app_close  # noqa: E402
from common.rollout import run_rollout  # noqa: E402
from common.safety import SafetyGuard  # noqa: E402
from common.teacher_contract import teacher_obs_slice  # noqa: E402
from common.teacher_obs import TeacherObsBuilder, TeacherObsCfg, executable_last_action  # noqa: E402
from common.teacher_policy_io import TeacherPolicyRunner  # noqa: E402
from common.timing import RateLimiter  # noqa: E402
from sim.sim_camera import SimCamera  # noqa: E402
from sim.sim_robot import SimRobot  # noqa: E402
from ur5e_binpick.tasks.binpick.binpick_env_cfg import BinPickVisionEnvCfg  # noqa: E402
from ur5e_binpick.tasks.binpick.mdp import observations as bp_obs  # noqa: E402
from ur5e_binpick.tasks.binpick.mdp.observations import object_positions_b  # noqa: E402


def episode_success(env) -> bool | None:
    term = env.termination_manager.get_term("placed_success")
    if term is None:
        raise RuntimeError("placed_success termination is unavailable")
    return True if bool(term[0].item()) else None


def set_rollout_target(env, target: np.ndarray) -> None:
    command_term = env.command_manager.get_term("target_pos")
    pose = command_term.pose_command_b
    target_t = torch.as_tensor(target, dtype=pose.dtype, device=pose.device)
    pose[0, :3] = target_t
    observed = env.command_manager.get_command("target_pos")[0, :3]
    if not torch.allclose(observed, target_t, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("rollout target did not reach the environment command buffer")


def load_manifest(policy_dir: Path) -> dict:
    manifest = json.loads((policy_dir / "experimental_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "experimental_teacher_policy":
        raise SystemExit("[rehearsal-teacher] teacher 번들이 아니다")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((policy_dir / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"[rehearsal-teacher] {name} 해시 불일치")
    return manifest


class PrivilegedBuilder:
    """env 특권 42D(참값)를 ★항별 함수로 직접★ 조립한다.

    ⚠️ `observation_manager.compute()`를 쓰지 말 것 — 그 호출은 policy 그룹까지 재계산해
    TiledCamera annotator를 제어 스텝당 한 번 더 읽는다. SimCamera.get_frame()과 겹치면
    synthetic-data 파이프라인이 "OgnSdOnNewFrame: frames discarded"를 뿜으며 교착한다
    (2026-09-16 실측: 1에피소드 스모크가 4시간 진행 없이 CPU만 태움). 아래 항별 함수는
    전부 articulation/sensor 버퍼만 읽고 렌더를 건드리지 않는다.
    """

    hold_requested = False

    def __init__(self, env, vertical: bool, device: str):
        self.env, self.vertical, self.device = env, vertical, device

    def reset(self):
        pass

    def __call__(self, frame, state, target_pos, last_action):
        env = self.env
        obs = np.zeros(42, dtype=np.float32)
        def put(name, value):
            obs[teacher_obs_slice(name)] = value[0].detach().cpu().numpy()
        put("joint_pos", bp_obs.joint_pos_abs(env))
        put("joint_vel", bp_obs.joint_vel_abs(env))
        put("gripper_pos", bp_obs.gripper_pos_abs(env))
        put("tcp_pose", bp_obs.tcp_pose_b(env))
        put("object_pos", bp_obs.object_positions_b(env))
        put("object_vel", bp_obs.object_velocities(env))
        put("contact_forces", bp_obs.finger_contact_forces(env))
        obs[teacher_obs_slice("target_pos")] = np.asarray(target_pos, dtype=np.float32)
        obs[teacher_obs_slice("last_action")] = executable_last_action(last_action, vertical_grasp_only=self.vertical)
        if not np.isfinite(obs).all():
            raise ValueError("non-finite privileged observation")
        return torch.from_numpy(obs).unsqueeze(0).to(self.device)


class DiagnosedBuilder:
    """TeacherObsBuilder를 감싸 매 스텝 추정치를 참 큐브 위치와 대조한다."""

    def __init__(self, inner: TeacherObsBuilder, env):
        self.inner, self.env = inner, env
        self.errors: list[float] = []
        self.hold_requested = False

    def reset(self):
        self.inner.reset(); self.errors = []; self.hold_requested = False

    def stats(self):
        return self.inner.stats()

    @property
    def debug(self):
        return self.inner.debug

    def __call__(self, frame, state, target_pos, last_action):
        obs = self.inner(frame, state, target_pos, last_action)
        self.hold_requested = self.inner.hold_requested
        if self.inner.debug.detected:
            # 항별 함수 — observation_manager.compute()는 카메라 재읽기로 교착한다(위 주석).
            truth = object_positions_b(self.env)[0, :3].detach().cpu().numpy()
            self.errors.append(float(np.linalg.norm(self.inner.debug.object_pos - truth)))
        return obs


def main() -> int:
    print("[rehearsal-teacher] main start", flush=True)
    policy_dir = args.policy_dir.expanduser().resolve()
    manifest = load_manifest(policy_dir)
    common_yaml = load_yaml(configs_dir() / "common.yaml")
    pre_cfg = build_preprocess_cfg(common_yaml)
    action_cfg = build_action_cfg(common_yaml)
    safety_cfg = build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))
    control_hz = float(common_yaml["control"]["control_hz"])
    policy = TeacherPolicyRunner(policy_dir / "policy.pt", policy_dir / "obs_spec.json", action_cfg,
                                 device=args.policy_device, expected_policy_sha256=manifest["files"]["policy.pt"],
                                 expected_obs_spec_sha256=manifest["files"]["obs_spec.json"])
    obs_cfg = TeacherObsCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in manifest["teacher_obs_cfg"].items()})

    env_cfg = BinPickVisionEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = SimRobot(env)
    camera = SimCamera(env)
    camera.start()
    mapper = ActionMapper(action_cfg)
    rng = np.random.default_rng(args.seed)
    tr = common_yaml["target"]["sample_range"]
    fixed = None if args.target_pos is None else np.array([float(v) for v in args.target_pos.split(",")])

    if args.obs_mode == "privileged":
        builder = PrivilegedBuilder(env, bool(manifest["contract"]["vertical_grasp_only"]), args.policy_device)
    else:
        intr = camera.get_intrinsics()
        print(f"[rehearsal-teacher] sim intrinsics fx={intr.fx:.2f} fy={intr.fy:.2f} {intr.width}x{intr.height}")
        builder = DiagnosedBuilder(TeacherObsBuilder(
            intr, load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml"),
            load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml"), obs_cfg, dt_s=1.0 / control_hz,
            workspace_min=safety_cfg.workspace_min, workspace_max=safety_cfg.workspace_max,
            device=args.policy_device), env)

    out_dir = HERE / "logs" / f"teacher_rehearsal_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.obs_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"[rehearsal-teacher] mode={args.obs_mode} episodes={args.episodes} max_steps={args.max_steps} seed={args.seed}")
    try:
        for ep in range(args.episodes):
            env.reset()
            target = fixed if fixed is not None else np.array([rng.uniform(*tr["x"]), rng.uniform(*tr["y"]), rng.uniform(*tr["z"])])
            set_rollout_target(env, target)
            robot.begin_episode()
            builder.reset()
            guard = SafetyGuard(safety_cfg, estop_hook=lambda reason: robot.emergency_stop())
            rate = RateLimiter(control_hz)
            result = run_rollout(robot=robot, camera=camera, policy=policy, mapper=mapper, guard=guard, rate=rate,
                                 pre_cfg=pre_cfg, target_pos=target, max_steps=args.max_steps, dry_run=False,
                                 termination_check=lambda: episode_success(env), obs_builder=builder)
            succ = bool(result.success) if result.terminated else False
            row = {"episode": ep, "steps": result.steps, "success": int(succ), "tripped": int(result.tripped),
                   "trip_reason": result.trip_reason, "overruns": result.overruns,
                   "target_x": round(float(target[0]), 4), "target_y": round(float(target[1]), 4), "target_z": round(float(target[2]), 4)}
            if args.obs_mode == "perception":
                s = builder.stats(); e = builder.errors
                row.update({"detection_rate": round(s["detection_rate"], 3), "held_fraction": round(s["held_fraction"], 3),
                            "hold_fraction": round(s["hold_fraction"], 3), "jump_rejects": s["jump_rejects"],
                            "err_p50_cm": round(100 * float(np.median(e)), 2) if e else "",
                            "err_p90_cm": round(100 * float(np.percentile(e, 90)), 2) if e else ""})
            rows.append(row)
            print(f"  ep {ep:3d}: success={int(succ)} steps={result.steps} trip={int(result.tripped)}"
                  + (f" det={row['detection_rate']:.2f} held={row['held_fraction']:.2f} hold={row['hold_fraction']:.2f} err50={row['err_p50_cm']}cm"
                     if args.obs_mode == "perception" else ""))
    finally:
        with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        n = len(rows); ns = sum(r["success"] for r in rows); nt = sum(r["tripped"] for r in rows)
        summary = {"mode": args.obs_mode, "episodes": n, "success": ns, "success_rate": ns / max(1, n), "tripped": nt,
                   "policy_dir": str(policy_dir), "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                   "seed": args.seed, "max_steps": args.max_steps, "teacher_obs_cfg": manifest["teacher_obs_cfg"]}
        if args.obs_mode == "perception" and rows:
            det = [r["detection_rate"] for r in rows]; errs = [r["err_p50_cm"] for r in rows if r["err_p50_cm"] != ""]
            summary.update({"mean_detection_rate": float(np.mean(det)), "mean_hold_fraction": float(np.mean([r["hold_fraction"] for r in rows])),
                            "median_episode_err_p50_cm": float(np.median(errs)) if errs else None})
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[rehearsal-teacher] {args.obs_mode}: 성공 {ns}/{n} ({100*ns/max(1,n):.1f}%)  trip {nt}  → {out_dir}")
        camera.stop(); env.close()
    return 0


def _main_reporting() -> int:
    """Kit 종료 과정이 traceback을 삼키지 않도록 예외를 먼저 flush 출력한다."""
    try:
        return main()
    except BaseException as exc:  # noqa: BLE001
        import traceback
        print("[rehearsal-teacher] ★예외★ " + "".join(traceback.format_exception(exc)), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(run_main_with_app_close(_main_reporting, app))
