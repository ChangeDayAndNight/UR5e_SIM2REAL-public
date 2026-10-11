#!/usr/bin/env python
"""Phase 3 게이트 4 — Sim 배포 리허설. [Claude][이 PC] (isaac51).

★존재 이유: '배포와 문자 그대로 같은 코드 경로'를 sim 백엔드로 통합 검증★
  - 제어 루프 = common.rollout.run_rollout  (Phase4 Deploy_DryRun/Deploy_Real과 동일 함수)
  - 전처리   = common.obs_preprocess.build_policy_obs  (학습·배포와 동일 함수)
  - 정책     = common.policy_io.PolicyRunner  (exported/ 번들 — 배포 PC와 동일 파일)
  - 안전     = common.safety.SafetyGuard  (configs/safety.yaml — 실기와 같은 값)
  - 주기     = common.timing.RateLimiter  (control_hz — servoJ 주기와 동일)
sim 백엔드(sim.sim_robot/sim_camera)를 real 백엔드로 바꾸면 그대로 Phase4가 된다.
여기서 안 도는 코드는 실기에서도 안 돈다 — 실기 디버깅을 sim으로 선지불하는 게이트.

⚠️ isaaclab을 import하므로 AppLauncher 이후에만 import할 것 (이 파일 내부에서 준수).

실행: python Phase3\\SimDeploy_Rehearsal.py `
        --candidate-dir logs/export_candidates/<CHECKPOINT_SHA_PREFIX> `
        --episodes 100 --max-steps 200
      (GUI 관찰: --gui; 정책/spec은 검증된 candidate에서만 로드)
산출: Phase3/logs/simdeploy_<시각>/episodes.csv (에피소드별 성공/trip/오버런 리포트)
      Phase3/logs/obs_dump_sim/<candidate-id>/<capture-id>/capture_manifest.json
      logs/parity/<candidate-and-capture-id>.npy 및 provenance sidecar
"""
# --- 자기 재실행 가드: 검증된 env python + 클린 환경 보장 (GUI.py 패턴) ---
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from scripts.training_env import configure_utf8_stdio, reexec_in_training_python

reexec_in_training_python(__file__)
configure_utf8_stdio()

from common.runtime_authority import (  # noqa: E402
    authorize_entrypoint,
    emit_runtime_authority_sentinel,
    runtime_authority_execution_receipt,
)

if __name__ == "__main__":
    authorize_entrypoint(
        _REPO_ROOT,
        __file__,
        purpose="phase3-rehearsal",
        sentinel_stage="phase3",
    )

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
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
import csv
import datetime as _dt
import json
from pathlib import Path
import secrets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description="Sim 배포 리허설 — 배포 코드 경로 통합 검증")
parser.add_argument("--episodes", type=int, default=100, help="최소 성공 리허설 에피소드 수")
parser.add_argument("--max-steps", type=int, default=200, help="에피소드당 최대 정책 스텝 (20Hz 기준 10s = 학습 에피소드; 상한 200)")
parser.add_argument("--candidate-dir", type=Path, default=None,
                    help="Phase2 --candidate-only이 만든 exact candidate 디렉터리")
parser.add_argument("--capture-id", default=None,
                    help="고유 UTC capture id (기본: 자동 생성, 기존 id 재사용 금지)")
parser.add_argument("--parity-output", type=Path, default=None,
                    help="봉인할 logs/parity/*.npy (기본: candidate/capture id 기반)")
parser.add_argument("--policy-device", default="cpu",
                    help="정책 추론 디바이스 — 기본 cpu (배포 PC 조건 모사; --device는 sim용 예약)")
parser.add_argument("--target-pos", default=None,
                    help="'x,y,z' 고정 place 목표 [m, base]. 미지정 시 common.yaml 범위에서 에피소드별 샘플")
parser.add_argument("--seed", type=int, default=45,
                    help="봉인 계약 seed (45 고정; 다른 seed는 진단 capture로도 봉인하지 않음)")
parser.add_argument("--gui", action="store_true", help="GUI로 관찰 (기본 headless)")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.candidate_dir is None:
    parser.error("--candidate-dir is required")
args.enable_cameras = True  # SimCamera(TiledCamera) 필수
args.headless = not args.gui
app = AppLauncher(args).app

# --- 여기서부터 isaaclab/torch import 허용 (AppLauncher 이후) ---
import numpy as np  # noqa: E402
import torch  # noqa: E402, F401 — Kit 이후 로드 순서 고정

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from common.action_mapping import ActionMapper  # noqa: E402
from common.kit_lifecycle import run_main_with_app_close  # noqa: E402
from common.config_io import (  # noqa: E402
    build_action_cfg,
    build_preprocess_cfg,
    build_safety_cfg,
    configs_dir,
    load_yaml,
    validate_target_sample_range,
)
from common.policy_io import PolicyRunner  # noqa: E402
from common.obs_preprocess import obs_dim  # noqa: E402
from common.path_safety import reset_episode_dump  # noqa: E402
from common.rollout import run_rollout  # noqa: E402
from common.safety import SafetyGuard  # noqa: E402
from common.timing import RateLimiter, derive_decimation  # noqa: E402
from common.types import ARM_COMMAND_CHANNELS, RolloutSafetyStats  # noqa: E402
from deployment.contracts import ContractError  # noqa: E402
from deployment.release import (  # noqa: E402
    MIN_PARITY_SAMPLES,
    candidate_receipt_binding,
    create_parity_capture,
    seal_parity_capture,
)  # noqa: E402
from scripts.build_parity_dataset import build_dataset  # noqa: E402
from ur5e_binpick.tasks.binpick.binpick_env_cfg import BinPickVisionEnvCfg  # noqa: E402
from sim.sim_camera import SimCamera  # noqa: E402
from sim.sim_robot import SimRobot  # noqa: E402


def episode_success(env) -> bool | None:
    """Return True only for the configured ``placed_success`` termination."""
    term = env.termination_manager.get_term("placed_success")
    if term is None:
        raise RuntimeError("placed_success termination is unavailable")
    return True if bool(term[0].item()) else None


def set_rollout_target(env, target: np.ndarray) -> None:
    """Set and verify the exact command observed by the policy/reward/termination."""
    command_term = env.command_manager.get_term("target_pos")
    pose = command_term.pose_command_b
    target_t = torch.as_tensor(target, dtype=pose.dtype, device=pose.device)
    pose[0, :3] = target_t
    observed = env.command_manager.get_command("target_pos")[0, :3]
    if not torch.allclose(observed, target_t, atol=1.0e-6, rtol=0.0):
        raise RuntimeError("rollout target did not reach the environment command buffer")


def main() -> int:
    if args.episodes < 100:
        raise SystemExit("[rehearsal] sealed Phase 3 capture requires --episodes >= 100")
    if not 1 <= args.max_steps <= 200:
        raise SystemExit("[rehearsal] --max-steps must be in [1, 200] (10 s horizon, Teacher v2)")
    if args.seed != 45:
        raise SystemExit("[rehearsal] sealed Phase 3 capture requires --seed 45")
    common_yaml = load_yaml(configs_dir() / "common.yaml")
    pre_cfg = build_preprocess_cfg(common_yaml)
    action_cfg = build_action_cfg(common_yaml)
    safety_cfg = build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))
    required_runtime_configs = {
        "configs/common.yaml",
        "configs/dr.yaml",
        "configs/safety.yaml",
        "configs/sim.yaml",
    }
    emit_runtime_authority_sentinel(
        "phase3",
        required_configs=required_runtime_configs,
    )
    runtime_authority = runtime_authority_execution_receipt(
        "phase3",
        required_configs=required_runtime_configs,
    )
    control_hz = float(common_yaml["control"]["control_hz"])
    physics_hz = float(common_yaml["control"]["physics_hz"])
    decimation = derive_decimation(physics_hz, control_hz)  # 요소3: 정합 검증 (실패 시 즉사)

    capture_id = args.capture_id or (
        _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
        + secrets.token_hex(4)
    )
    try:
        capture_dir, candidate_manifest = create_parity_capture(
            candidate_dir=args.candidate_dir,
            project_root=ROOT,
            capture_id=capture_id,
        )
        # Preserve the lexical CLI path so every downstream verifier can still
        # reject symlink/junction aliases before resolving it.
        candidate_dir = Path(args.candidate_dir).expanduser().absolute()
        candidate_binding = candidate_receipt_binding(
            candidate_dir, candidate_manifest, project_root=ROOT
        )
    except ContractError as exc:
        raise SystemExit(f"[rehearsal] candidate/capture validation failed: {exc}") from exc
    expected_obs = int(candidate_manifest["observation_contract"]["observation_dim"])
    if expected_obs != obs_dim(pre_cfg):
        raise SystemExit("[rehearsal] candidate observation dimension differs from current config")

    # --- 정책 로드 (배포와 동일: obs_spec 검증 → 불일치 시 기동 거부) ---
    policy = PolicyRunner(
        candidate_dir / "policy.pt",
        candidate_dir / "obs_spec.json",
        pre_cfg,
        action_cfg,
        device=args.policy_device,
        expected_policy_sha256=candidate_manifest["files"]["policy.pt"],
        expected_obs_spec_sha256=candidate_manifest["files"]["obs_spec.json"],
    )

    # --- env 구성: 학습과 같은 Vision cfg, 단 num_envs=1 (배포 조건 모사) ---
    env_cfg = BinPickVisionEnvCfg()
    env_cfg.scene.num_envs = 1
    # 재현성: ManagerBasedRLEnvCfg.seed 기본값은 None이므로 조건 없이 항상 덮어쓴다
    # (None 검사로 가드하면 기본 상태에서 seed가 영영 적용되지 않는다).
    env_cfg.seed = args.seed
    env = ManagerBasedRLEnv(cfg=env_cfg)
    if getattr(env_cfg, "decimation", None) != decimation:
        # env cfg도 derive_decimation을 쓰므로 항상 일치해야 정상 — 어긋나면 계약 위반
        print(f"[rehearsal] ⚠️ decimation 불일치: env_cfg={getattr(env_cfg, 'decimation', None)} "
              f"vs derive={decimation} — configs/common.yaml 계약 위반, 중단")
        env.close()
        return 2

    # --- sim 백엔드 어댑터 (RobotInterface/CameraInterface 구현) ---
    # ⚠️ TODO(확인 필요): SimRobot/SimCamera 생성자 시그니처 — sim/ 작성자와 조율.
    #   가정: env를 통째로 받아 내부에서 articulation/tiled camera 핸들을 얻는다.
    #   또한 send_joint_target이 내부에서 physics를 decimation 스텝만큼 진행한다고
    #   가정한다 (rollout 루프에는 sim step 개념이 없음 — 어댑터 책임). 정합 확인 필수.
    robot = SimRobot(env)
    camera = SimCamera(env)

    mapper = ActionMapper(action_cfg)
    rng = np.random.default_rng(args.seed)
    tr = common_yaml["target"]["sample_range"]

    def sample_target() -> np.ndarray:
        return np.array(
            [rng.uniform(*tr["x"]), rng.uniform(*tr["y"]), rng.uniform(*tr["z"])]
        )

    fixed_target = None
    if args.target_pos is not None:
        try:
            fixed_target = np.asarray(
                validate_target_sample_range(
                    (float(x) for x in args.target_pos.split(",")),
                    common_yaml,
                    tag="rehearsal",
                ),
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"[rehearsal] invalid --target-pos: {exc}") from exc

    out_dir = HERE / "logs" / f"simdeploy_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # The receipt belongs to the immutable capture authority, not merely the
    # human-readable rehearsal log directory.  Capture sealing reads and hashes
    # these exact bytes through one handle and rejects every later drift.
    with (capture_dir / "runtime_authority.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(runtime_authority, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"[rehearsal] ★candidate 코드 경로 리허설/수집★ min_episodes={args.episodes}, "
          f"control={control_hz}Hz (decimation={decimation}), policy_device={args.policy_device}")
    print(f"[rehearsal] capture={capture_dir} (minimum rows={MIN_PARITY_SAMPLES})")

    rows = []
    episode_outcomes = []
    collected_rows = 0
    issued_command_steps = 0
    arm_clamp_counts = np.zeros(ARM_COMMAND_CHANNELS, dtype=np.int64)
    camera.start()
    try:
        ep = 0
        while len(rows) < args.episodes or collected_rows < MIN_PARITY_SAMPLES:
            env.reset()
            target = fixed_target if fixed_target is not None else sample_target()
            set_rollout_target(env, target)
            # Refresh cached termination terms after env.reset and after the
            # exact episode target has been installed, without advancing time.
            robot.begin_episode()

            # trip latch는 에피소드 단위로 리셋 — guard/rate를 매 에피소드 새로 생성
            guard = SafetyGuard(safety_cfg, estop_hook=lambda reason: robot.emergency_stop())
            rate = RateLimiter(control_hz)

            ep_dump = reset_episode_dump(capture_dir, ep, repo_root=ROOT)

            # ★핵심: 배포와 '문자 그대로 같은 함수' 호출 — dry_run=False (sim이므로 안전)★
            result = run_rollout(
                robot=robot,
                camera=camera,
                policy=policy,
                mapper=mapper,
                guard=guard,
                rate=rate,
                pre_cfg=pre_cfg,
                target_pos=target,
                max_steps=args.max_steps,
                dry_run=False,
                obs_dump_dir=ep_dump,
                termination_check=lambda: episode_success(env),
            )
            succ = result.success if result.terminated else False
            rows.append(
                {
                    "episode": ep,
                    "steps": result.steps,
                    "success": int(succ),
                    "terminated": int(result.terminated),
                    "tripped": int(result.tripped),
                    "trip_reason": result.trip_reason,
                    "overruns": result.overruns,
                    "issued_command_steps": result.rollout_safety.denominator,
                    "arm_clamp_counts": ",".join(
                        str(value) for value in result.rollout_safety.arm_clamp_counts
                    ),
                    "target_x": round(float(target[0]), 4),
                    "target_y": round(float(target[1]), 4),
                    "target_z": round(float(target[2]), 4),
                    "dump_dir": str(ep_dump),
                }
            )
            episode_outcomes.append(
                {
                    "episode": ep,
                    "steps": int(result.steps),
                    "placed_success": bool(succ),
                    "tripped": bool(result.tripped),
                    "trip_reason": str(result.trip_reason),
                    "target_pos": [float(value) for value in target],
                }
            )
            collected_rows += int(result.steps)
            issued_command_steps += int(result.rollout_safety.denominator)
            arm_clamp_counts += np.asarray(
                result.rollout_safety.arm_clamp_counts, dtype=np.int64
            )
            print(f"[rehearsal] ep {ep:2d}: steps={result.steps} tripped={result.tripped} "
                  f"overruns={result.overruns} issued={issued_command_steps}/"
                  f"{MIN_PARITY_SAMPLES}")
            ep += 1
            if result.tripped or not succ or result.steps <= 0:
                print("[rehearsal] failed episode leaves an unsealed, unusable capture; stopping")
                break
    finally:
        try:
            camera.stop()
            robot.close()
        except Exception:
            pass

    if not rows:
        print("[rehearsal] ⚠️ 에피소드 0회 — 리포트 생략 (--episodes 확인)")
        env.close()
        return 1

    csv_path = out_dir / "episodes.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_trip = sum(r["tripped"] for r in rows)
    n_success = sum(r["success"] for r in rows)
    gate_ok = len(rows) >= 100 and n_trip == 0 and n_success == len(rows)
    gate_ok = gate_ok and collected_rows >= MIN_PARITY_SAMPLES
    gate_ok = gate_ok and issued_command_steps >= MIN_PARITY_SAMPLES
    gate_ok = gate_ok and issued_command_steps == collected_rows
    rollout_safety = RolloutSafetyStats(
        denominator=issued_command_steps,
        arm_clamp_counts=tuple(int(value) for value in arm_clamp_counts),
    ).as_manifest()
    gate_ok = gate_ok and bool(rollout_safety["all_channels_below_limit"])
    print(f"[rehearsal] 요약: {len(rows)}ep, success {n_success}, trip {n_trip}건 → "
          f"게이트 4 {'★통과★' if gate_ok else '⚠️ 불합격 — trip 원인 해소 전 실기 진입 금지'}")
    print(f"[rehearsal] 리포트: {csv_path}")
    env.close()
    if not gate_ok:
        print(f"[rehearsal] unsealed capture (release input으로 사용 불가): {capture_dir}")
        return 1
    try:
        capture_manifest = seal_parity_capture(
            capture_dir,
            candidate_dir=candidate_dir,
            project_root=ROOT,
            expected_dim=expected_obs,
            rollout_safety=rollout_safety,
            seed=args.seed,
            episode_outcomes=episode_outcomes,
        )
        parity_output = args.parity_output or (
            ROOT / "logs/parity"
            / f"{candidate_binding['manifest_sha256'][:16]}-{capture_id}.npy"
        )
        dataset, sidecar = build_dataset(
            candidate_dir,
            capture_manifest,
            parity_output,
        )
    except ContractError as exc:
        print(f"[rehearsal] capture/dataset sealing failed: {exc}")
        return 2
    print(f"[rehearsal] sealed capture manifest: {capture_manifest}")
    print(f"[rehearsal] parity dataset: {dataset}")
    print(f"[rehearsal] parity sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_main_with_app_close(main, app))
