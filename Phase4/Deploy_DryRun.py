#!/usr/bin/env python
"""Phase 4 — future policy dry-run interface, hard-blocked in this release.

The parser and unreachable implementation document the intended receive-only
shape, but ``main`` exits before config, release, SDK, or socket access. The
only supported hardware path is the canonical receive-only no-motion
diagnostic, which cannot produce passing readiness evidence yet.

목적:
  로봇·카메라에 "연결"하되 명령은 전송하지 않고(run_rollout dry_run=True — 취득/
  전처리/추론/안전판정은 전부 실행) 다음 3가지를 실측한다:
    ① 실측 관측 dump 생성 → deployment/Verify_ObsParity.py의 real측 입력
    ② 제어주기 지터 히스토그램 CSV (RateLimiter 반환값 수집 — 배포 PC 타이머 품질)
    ③ 정책 출력 통계 리포트 (채널별 분포/포화율 — sim 리허설과 비교)

실행 (배포 PC, 프로젝트 루트에서):
  conda activate env_ur_real
  python Phase4/Deploy_DryRun.py --target-pos=-0.50,0.0,-0.0125 \
      --real-config configs/real.local.yaml --release-dir exported
  python Phase4/Deploy_DryRun.py --target-pos=-0.50,0.0,-0.0125 --max-steps 200 \
      --real-config configs/real.local.yaml --release-dir exported

산출:
  Phase4/logs/obs_dump_real/step_*.npz     (관측 dump — deployment/Verify_ObsParity.py 입력)
  Phase4/logs/jitter_dryrun_<시각>.csv      (스텝별 주기 실측 — 히스토그램 원자료)
  콘솔: 지터 요약(ASCII 히스토그램) + 정책 출력 통계

⚠️ 이 스크립트는 로봇을 움직이지 않지만 RealRobot의 수신 전용 RTDE
   연결은 발생한다. 현 릴리스의 명령/모션 경로는 증거와 무관하게 차단된다.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _check_runtime_env() -> None:
    """배포 PC 환경 점검 — Isaac 재실행 가드가 아니라 '단순 확인'만 한다.

    (배포 PC에는 Isaac이 없으므로 isaac51 자기 재실행 패턴을 쓰지 않는다.)
    """
    missing = [m for m in ("numpy", "torch", "yaml") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"[dryrun][error] ⚠️ 필수 패키지 누락: {', '.join(missing)}")
        print("[dryrun]   → conda activate env_ur_real 후:")
        print("[dryrun]     python -m pip install --no-index --find-links deployment/wheelhouse "
              "--require-hashes -r deployment/requirements-win11.lock.txt")
        raise SystemExit(1)
    if "env_ur_real" not in sys.prefix.replace("\\", "/"):
        print(f"[dryrun][warn] ⚠️ 현재 파이썬이 env_ur_real이 아닐 수 있음: {sys.executable}")


_check_runtime_env()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import _deploy_common as dc  # noqa: E402
from common.action_mapping import ActionMapper  # noqa: E402
from common.policy_io import PolicyRunner  # noqa: E402
from common.rollout import run_rollout  # noqa: E402
from common.safety import SafetyGuard  # noqa: E402
from common.timing import derive_decimation  # noqa: E402
from deployment.contracts import (  # noqa: E402
    ContractError,
    require_policy_dryrun_authorization,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase4 무동작 리허설 — 관측 dump + 지터 실측 + 정책 출력 통계 (모션 없음)"
    )
    parser.add_argument("--target-pos", required=True,
                        help="place 목표 'x,y,z' [m, base frame] — 필수 (에피소드 가변 목표 계약)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="루프 스텝 수 (기본/최대 200 = 10s @20Hz = Teacher v2 학습 에피소드)")
    parser.add_argument("--real-config", required=True,
                        help="로컬 실기 설정 YAML (예: configs/real.local.yaml)")
    parser.add_argument("--release-dir", required=True,
                        help="policy.pt, obs_spec.json, release_manifest.json 디렉터리")
    parser.add_argument("--device", default="cpu",
                        help="추론 디바이스 (배포 PC GPU 미확정 — 기본 cpu)")
    parser.add_argument("--obs-dump-dir", default=str(dc.OBS_DUMP_REAL_DIR),
                        help="관측 dump 디렉토리 (기본: Phase4/logs/obs_dump_real — deployment/Verify_ObsParity.py 계약 경로)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # Do not let this short policy loop become an alternate HIL readiness gate.
    # Preserve --help, then stop before config/release/SDK/socket access until
    # RTDE and RealSense share a proven physical acquisition time base.
    try:
        require_policy_dryrun_authorization()
    except ContractError as exc:
        print(f"[dryrun][blocked] {exc}")
        return 2
    if not 1 <= args.max_steps <= 200:
        print("[dryrun][error] --max-steps must be in [1,200] (10 second learned horizon, Teacher v2)")
        return 1
    try:
        target = dc.parse_vec3(args.target_pos, "--target-pos")
        # ── 설정 로드 + 정합 검증 (configs = 학습 PC에서 복사된 번들) ──
        bundle = dc.load_deploy_bundle(args.real_config)
        decim = derive_decimation(bundle.physics_hz, bundle.control_hz)  # ★요소3: 항상 파생★
        dc.validate_target_pos(target, bundle, tag="dryrun")
    except ValueError as e:
        print(f"[dryrun][error] ⚠️ {e}")
        return 1
    print(f"[dryrun] control {bundle.control_hz:g}Hz / physics {bundle.physics_hz:g}Hz "
          f"(decimation={decim} — 번들 configs 정합 검증용 파생값)")

    # ── 정책 로드 (로봇 연결 전에 — obs_spec 불일치면 여기서 기동 거부) ──
    release_dir = Path(args.release_dir).resolve()
    policy_path = release_dir / "policy.pt"
    obs_spec_path = release_dir / "obs_spec.json"
    try:
        from deployment.release import verify_release

        release_manifest = verify_release(
            release_dir,
            config_root=ROOT,
            benchmark=True,
            expected_action_dim=bundle.action_cfg.action_dim,
        )
    except Exception as exc:
        print(f"[dryrun][error] release verification failed: {exc}")
        return 1
    for p in (policy_path, obs_spec_path):
        if not Path(p).is_file():
            print(f"[dryrun][error] ⚠️ 파일 없음: {p} — 번들에 exported/가 포함됐는지 확인")
            return 1
    policy = PolicyRunner(
        policy_path,
        obs_spec_path,
        bundle.pre_cfg,
        bundle.action_cfg,
        device=args.device,
        expected_policy_sha256=release_manifest["files"]["policy.pt"],
        expected_obs_spec_sha256=release_manifest["files"]["obs_spec.json"],
    )
    mapper = ActionMapper(bundle.action_cfg)

    # dry-run e-stop hook: 모션이 없으므로 로그만 남긴다 (실기 배선은 Deploy_Real에서
    # robot.emergency_stop으로 — 여기서 실로봇을 servoStop 상태로 만들 이유가 없다).
    guard = SafetyGuard(
        bundle.safety_cfg,
        estop_hook=lambda reason: print(f"[dryrun][safety] (무동작) e-stop hook 호출 로그: {reason}"),
    )
    rate = dc.RecordingRateLimiter(bundle.control_hz)  # sleep() 반환값 수집

    requested_dump = Path(args.obs_dump_dir).expanduser().resolve()
    allowed_root = dc.OBS_DUMP_REAL_DIR.parent.resolve()
    if requested_dump != dc.OBS_DUMP_REAL_DIR.resolve() and allowed_root not in requested_dump.parents:
        print(f"[dryrun][error] dump directory must stay under {allowed_root}: {requested_dump}")
        return 1
    # Preserve every previous evidence run; arbitrary recursive deletion is forbidden.
    dump_dir = requested_dump / dc.timestamp()
    if dump_dir.exists():
        print(f"[dryrun][error] refusing to overwrite existing dump: {dump_dir}")
        return 1

    robot = camera = None
    result = None
    try:
        print(f"[dryrun] 로봇 연결 시도: {bundle.real_yaml['robot']['ip']} (명령 미전송)")
        robot = dc.build_real_robot(bundle.real_yaml, receive_only=True)
        # enable_rgb = 관측 계약(PreprocessCfg.use_rgb) 단일 소스 — §4-3
        camera = dc.build_real_camera(bundle.real_yaml, enable_rgb=bundle.pre_cfg.use_rgb)
        camera.start()
        print("[dryrun] ★모션 없음★ dry_run=True — 취득/전처리/추론/안전판정만 실행")

        result = run_rollout(
            robot=robot,
            camera=camera,
            policy=policy,
            mapper=mapper,
            guard=guard,
            rate=rate,
            pre_cfg=bundle.pre_cfg,
            target_pos=target,
            max_steps=args.max_steps,
            dry_run=True,
            obs_dump_dir=dump_dir,
            require_force_feedback=True,
            max_sensor_age_s=2.0 / bundle.control_hz,
            max_sensor_skew_s=1.0 / bundle.control_hz,
            max_consecutive_overruns=0,
        )
    except KeyboardInterrupt:
        print("[dryrun] 사용자 중단 (Ctrl+C) — 모션 없음, 수집분까지 리포트")
    finally:
        # 자원 해제는 항상 (연결 실패 시 None 가드)
        if camera is not None:
            try:
                camera.stop()
            except Exception as e:
                print(f"[dryrun][warn] 카메라 종료 실패: {e}")
        if robot is not None:
            try:
                robot.close()
            except Exception as e:
                print(f"[dryrun][warn] 로봇 연결 종료 실패: {e}")

    # ── ② 지터 히스토그램 CSV + 요약 ──
    jitter_csv = dc.LOGS_DIR / f"jitter_dryrun_{dc.timestamp()}.csv"
    dc.write_jitter_csv(jitter_csv, rate.samples_s, bundle.control_hz)
    dc.summarize_jitter(rate.samples_s, bundle.control_hz, tag="dryrun")
    print(f"[dryrun] 지터 CSV: {jitter_csv}")

    # ── ③ 정책 출력 통계 (dump 후처리) ──
    stats = dc.dump_stats(dump_dir, bundle.pre_cfg, bundle.action_cfg)
    if stats is None:
        print(f"[dryrun][warn] ⚠️ dump 없음 — 통계 생략: {dump_dir}")
    else:
        dc.print_dump_report(stats, tag="dryrun")
        print(f"[dryrun] 관측 dump: {dump_dir} (step_*.npz {stats['n_steps']}개)")
        print("[dryrun] ★다음 단계★ deployment/Verify_ObsParity.py에 이 dump를 real측 입력으로 전달")

    if result is not None and result.tripped:
        print(f"[dryrun][warn] ⚠️ 무동작인데 trip 발생: {result.trip_reason} — "
              f"실기 상태/센서/safety.yaml 임계값을 먼저 점검할 것 (Deploy_Real 진행 금지)")
        return 2
    print("[dryrun] 완료 — 현 릴리스의 운영 범위는 offline/no-motion·수신 전용까지이며 실제 모션은 금지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
