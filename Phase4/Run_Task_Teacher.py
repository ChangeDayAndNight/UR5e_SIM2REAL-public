#!/usr/bin/env python
"""★모듈형 배포 v2★ 실기 실행 — 인식 모듈 + Teacher v2(상태 정책, 34D). [배포 PC][Phase 4]

Phase4/Run_Task.py(student RGB-D 정책)와 ★같은 제어 루프(common.rollout.run_rollout)·같은 안전
장치(SafetyGuard)·같은 릴리스 수퍼바이저·같은 모션 인가★를 쓴다. 다른 것은 두 가지뿐이다:
  · 정책: exported_experimental/teacher_<sha12>/ 의 TorchScript Teacher v2 (34D 상태 → 7D ik_rel)
  · 관측: common.teacher_obs.TeacherObsBuilder — depth 역투영으로 큐브 중심을 추정하고
          관절/TCP(RTDE)·그리퍼 명령 상태와 함께 34D를 조립한다(rollout의 obs_builder 훅).

★F93(2026-09-20) 시작 절차 = sim 리셋과 같은 상태에서 시작한다★
  1) 홈 자세 게이트: 관절이 홈(robot.home_joints_rad)에서 0.05 rad 넘게 벗어나 있으면 거부한다
     (--home-first 를 주면 moveJ 0.1 배속으로 홈 이동). 학습은 항상 홈에서 시작한다.
  2) 명시적 개방: 실기 어댑터는 첫 폐합 전에 개방 펄스 + 1.20 s 안정화를 요구한다(그 전엔
     gripper_command_state=None → run_rollout이 0스텝에서 trip). 매 에피소드 개방 상태(0.0)로 맞춘다.
  3) DRY-RUN(receive-only)은 펄스를 못 보내므로 command state를 '개방(0.0)'으로 ★가정★한다 — 관측
     조립·인식 점검 전용이며 held latch는 걸리지 않는다.
큐브가 안 보이고 쥐고 있지도 않은 채 hold 임계(4 s)를 넘으면 조립기가 hold를 요청해 팔을 세운다.
마지막 추정치 바로 위(예상된 블라인드 콘)나 그리퍼 전이 중의 결측은 세지 않는다.

기본값은 학습 조건이다: --max-steps 200(10 s), --action-scale 1.0(SafetyGuard가 이미 12.5 mm/스텝 상한).
--action-scale < 1.0은 릴리스 수퍼바이저의 ★절대★ 정지 임계와 어긋나 릴리스 시점이 달라진다 — 권장하지 않는다.

실행:
  python Phase4/Run_Task_Teacher.py --policy-dir exported_experimental/teacher_<sha12> --dry-run --verbose-perception
  python Phase4/Run_Task_Teacher.py --policy-dir ... --yes-i-am-at-the-estop [--home-first]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training_env import configure_utf8_stdio  # noqa: E402
configure_utf8_stdio()

from common.action_mapping import ActionMapper  # noqa: E402
from common.calibration import load_hand_eye, load_tcp_offset  # noqa: E402
from common.config_io import (  # noqa: E402
    build_action_cfg, build_preprocess_cfg, build_safety_cfg, configs_dir, load_yaml,
    validate_target_sample_range,
)
from common.rollout import run_rollout  # noqa: E402
from common.safety import SafetyGuard  # noqa: E402
from common.teacher_obs import TeacherObsBuilder, TeacherObsCfg  # noqa: E402
from common.teacher_policy_io import TeacherPolicyRunner  # noqa: E402
from common.timing import RateLimiter  # noqa: E402

_DEFAULT_TARGET = "-0.50,0.15,-0.0125"
_HOME_TOL_RAD = 0.05          # ★F93★ 시작 자세 게이트(관절당) — 학습은 항상 정확히 홈에서 시작한다


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _home_delta_rad(robot, state) -> np.ndarray:
    home = np.asarray(robot.home_joints_rad(), dtype=np.float64).reshape(6)
    return _wrap_pi(np.asarray(state.joint_pos, dtype=np.float64).reshape(6) - home)


def _explicit_open(robot, control_hz: float, timeout_s: float = 6.0):
    """★F93(리뷰 D1)★ 개방 펄스 후 command state가 0.0(개방)이 될 때까지 기다린다.

    · state None(기동 직후): 펄스 1회 → 어댑터가 펄스 완료 + 1.20 s 뒤 0.0을 공표한다(추가 명령 금지).
    · state 알려짐(>0, 직전 에피소드가 닫힌 채 끝남): 20 Hz로 개방을 재명령해 24스텝 추적기를 0까지 내린다.
    """
    period = 1.0 / float(control_hz)
    try:
        robot.set_gripper(False)
    except RuntimeError as exc:
        raise SystemExit(f"[run-teacher] 개방 펄스 실패: {exc}") from exc
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        time.sleep(period)
        st = robot.get_state()
        gs = st.gripper_command_state
        if gs is not None:
            if gs <= 0.0:
                return st
            try:
                robot.set_gripper(False)
            except RuntimeError as exc:
                if "pending" not in str(exc):
                    raise SystemExit(f"[run-teacher] 개방 재명령 실패: {exc}") from exc
    raise SystemExit("[run-teacher] 명시적 개방 후에도 gripper_command_state가 개방(0.0)이 되지 않았다 — "
                     "그리퍼 DO 배선(DO5/DO6)·펄스·settle 설정(real.local.yaml gripper)을 확인할 것")


class _AssumedOpenGripper:
    """DRY-RUN 전용 프록시 — receive-only 어댑터는 펄스를 못 보내 command state가 None이다. 명령을 보내지
    않는 모드이므로 관측 조립용으로 '개방(0.0)'을 가정한다. 그 외 호출은 그대로 위임한다."""

    def __init__(self, inner):
        self._inner = inner

    def get_state(self):
        st = self._inner.get_state()
        if st.gripper_command_state is None:
            st = dataclasses.replace(st, gripper_command_state=0.0)
        return st

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _calib_flags() -> dict[str, bool]:
    out = {}
    for name in ("hand_eye.yaml", "tcp_offset.yaml"):
        y = load_yaml(configs_dir() / "calib" / name)
        out[name] = bool((y.get("meta") or {}).get("calibrated", False))
    return out


def _load_policy_dir(policy_dir: Path) -> dict:
    manifest_path = policy_dir / "experimental_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"[run-teacher] 매니페스트 없음: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "experimental_teacher_policy":
        raise SystemExit(f"[run-teacher] teacher 번들이 아니다: artifact_type={manifest.get('artifact_type')!r} "
                         f"(student 번들은 Phase4/Run_Task.py)")
    for name, expected in manifest.get("files", {}).items():
        target = policy_dir / name
        if not target.is_file():
            raise SystemExit(f"[run-teacher] 파일 없음: {target}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"[run-teacher] {name} 해시 불일치 — 폴더를 다시 복사할 것")
    return manifest


class _VerboseBuilder:
    """TeacherObsBuilder를 감싸 매 스텝 인식 결과를 출력한다(hold_requested·stats·debug 위임)."""

    def __init__(self, inner: TeacherObsBuilder):
        self.inner = inner
        self.hold_requested = False

    def reset(self) -> None:
        self.inner.reset(); self.hold_requested = False

    def stats(self) -> dict:
        return self.inner.stats()

    @property
    def debug(self):
        return self.inner.debug

    def __call__(self, frame, state, target_pos, last_action):
        obs = self.inner(frame, state, target_pos, last_action)
        self.hold_requested = self.inner.hold_requested
        d = self.inner.debug
        print(f"   [perception] det={int(d.detected)} held={int(d.held)} hold={int(self.hold_requested)} "
              f"src={d.object_source} obj={np.round(d.object_pos, 3).tolist()} {d.reason}")
        return obs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-dir", required=True, type=Path)
    parser.add_argument("--target-pos", default=_DEFAULT_TARGET, help=f"놓을 위치 'x,y,z' [m] (기본 {_DEFAULT_TARGET})")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="에피소드 최대 스텝 (기본 200 = 학습 에피소드 10 s × 20 Hz)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="학습 액션 크기 대비 배율 (기본 1.0 = 학습 조건; SafetyGuard가 12.5 mm/스텝을 이미 상한한다. "
                             "<1.0은 릴리스 수퍼바이저 정지 임계와 어긋난다)")
    parser.add_argument("--dry-run", action="store_true",
                        help="명령을 보내지 않는다 — 취득·인식·추론·안전판정만 (gripper command state는 개방으로 가정)")
    parser.add_argument("--yes-i-am-at-the-estop", dest="operator_ack", action="store_true")
    parser.add_argument("--home-first", action="store_true",
                        help="시작 자세가 홈이 아니면 moveJ(0.1 배속)로 홈 이동 후 시작 (기본: 거부)")
    parser.add_argument("--accept-uncalibrated", action="store_true",
                        help="hand_eye/tcp_offset meta.calibrated=false 여도 모션 허용 (기본: 거부)")
    parser.add_argument("--max-consecutive-overruns", type=int, default=2,
                        help="제어주기 연속 초과 허용 횟수 — 초과하면 trip (기본 2)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dump-dir", type=Path, default=None, help="스텝별 관측/행동 .npz 저장")
    parser.add_argument("--verbose-perception", action="store_true", help="매 스텝 인식 결과 출력")
    args = parser.parse_args(argv[1:])
    if not args.dry_run and not args.operator_ack:
        print("[run-teacher] 모션을 내려면 --yes-i-am-at-the-estop 이 필요하다. 먼저 --dry-run 을 권장한다.")
        return 2
    if not 0.0 < args.action_scale <= 1.0:
        print("[run-teacher] --action-scale 은 (0, 1] 범위여야 한다"); return 2

    policy_dir = args.policy_dir.expanduser().resolve()
    manifest = _load_policy_dir(policy_dir)
    common_yaml = load_yaml(configs_dir() / "common.yaml")
    pre_cfg = build_preprocess_cfg(common_yaml)
    action_cfg = build_action_cfg(common_yaml)
    safety_cfg = build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))
    real_yaml = load_yaml(configs_dir() / "real.local.yaml")
    control_hz = float(common_yaml["control"]["control_hz"])
    scaled_action_cfg = action_cfg
    if args.action_scale < 1.0:
        scaled_action_cfg = dataclasses.replace(
            action_cfg, task_pos_scale_m=action_cfg.task_pos_scale_m * args.action_scale,
            task_rot_scale_rad=action_cfg.task_rot_scale_rad * args.action_scale)
    target = np.array([float(v) for v in args.target_pos.split(",")], dtype=np.float64)
    lo = np.asarray(safety_cfg.workspace_min, dtype=np.float64); hi = np.asarray(safety_cfg.workspace_max, dtype=np.float64)
    if np.any(target < lo) or np.any(target > hi):
        print(f"[run-teacher] --target-pos {target.tolist()} 가 작업영역 {lo.tolist()}..{hi.tolist()} 밖 — 거부"); return 2
    try:
        validate_target_sample_range(target, common_yaml, tag="run-teacher")   # ★F93(리뷰 OBS-3)★ 학습 분포 밖 거부
    except ValueError as exc:
        print(f"[run-teacher] {exc}"); return 2
    if args.action_scale < 1.0:
        print(f"[run-teacher][warn] --action-scale {args.action_scale:.2f} < 1.0: 학습 조건(1.0)과 다르다. 릴리스 수퍼바이저의 "
              "정지 임계(0.02 m/s)는 절대값이라 느린 실행에서 릴리스가 더 일찍/자주 성립한다. 관찰 목적 외에는 1.0을 쓸 것.")
    calib = _calib_flags()
    if not all(calib.values()):
        missing = [k for k, v in calib.items() if not v]
        if args.dry_run:
            print(f"[run-teacher][warn] 캘리브 미완료(meta.calibrated=false): {missing} — dry-run은 진행하지만 인식 위치가 통째로 밀릴 수 있다")
        elif not args.accept_uncalibrated:
            print(f"[run-teacher] 캘리브 미완료(meta.calibrated=false): {missing} — deployment/Calib_HandEye.py·Calib_TcpOffset.py 후 "
                  "실행하거나 --accept-uncalibrated 로 명시적으로 감수할 것"); return 2
        else:
            print(f"[run-teacher][warn] 캘리브 미완료 상태로 모션을 진행한다(--accept-uncalibrated): {missing}")
    if not args.dry_run and not bool((real_yaml.get("robot") or {}).get("tcp_command_verified", False)):
        print("[run-teacher][warn] real.local.yaml robot.tcp_command_verified=false — 실기 어댑터가 servoL(ik_rel)을 거부한다. "
              "TCP 명령 검증(README §5)을 마친 뒤 true로 바꿔야 모션이 나간다.")

    obs_cfg = TeacherObsCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in manifest["teacher_obs_cfg"].items()})
    print("═" * 72)
    print(f" 정책 : {policy_dir.name}  (teacher iter {manifest['checkpoint']['iteration']}, 릴리스 아님)")
    print(f" 모드 : {'DRY-RUN (명령 미전송)' if args.dry_run else '★실제 모션★'}   액션 스케일 {args.action_scale:.2f}")
    print(f" 타겟 : {target.tolist()}   최대 {args.max_steps}스텝 × {args.episodes}회")
    print(f" 인식 : depth 높이 띠 {obs_cfg.z_band_m} m, 쥠 반경 {obs_cfg.held_radius_m} m, 작업면 z {obs_cfg.table_height_m} m")
    print("═" * 72)

    policy = TeacherPolicyRunner(
        policy_dir / "policy.pt", policy_dir / "obs_spec.json", action_cfg, device=args.device,
        expected_policy_sha256=manifest["files"]["policy.pt"],
        expected_obs_spec_sha256=manifest["files"]["obs_spec.json"])
    mapper = ActionMapper(scaled_action_cfg)
    tripped: list[str] = []

    def _estop(reason: str) -> None:
        tripped.append(reason); print(f"\n[run-teacher][SAFETY] ★trip★ {reason}")

    guard = SafetyGuard(safety_cfg, estop_hook=_estop)
    rate = RateLimiter(control_hz)
    from real.real_camera import RealCamera
    from real.real_robot import RealRobot
    robot = camera = None
    try:
        robot = RealRobot(real_yaml, configs_dir() / "calib" / "tcp_offset.yaml", receive_only=args.dry_run,
                          motion_scope="policy", operator_ack=args.operator_ack)
        if args.dry_run:
            robot = _AssumedOpenGripper(robot)   # ★F93★ receive-only: command state 개방 가정
        # ★F93(리뷰 OBS-8)★ depth→color 정렬(rs.align)은 student의 use_rgb 플래그와 무관하게 ★항상★ 켠다 —
        #   hand_eye v7은 color optical 프레임 기준이고 locator는 color intrinsics로 역투영한다.
        camera = RealCamera(real_yaml["camera"], enable_rgb=True)
        camera.start()
        intr = camera.get_intrinsics()
        builder = TeacherObsBuilder(
            intr, load_hand_eye(configs_dir() / "calib" / "hand_eye.yaml"),
            load_tcp_offset(configs_dir() / "calib" / "tcp_offset.yaml"), obs_cfg, dt_s=1.0 / control_hz,
            workspace_min=safety_cfg.workspace_min, workspace_max=safety_cfg.workspace_max, device=args.device)
        print(f"[run-teacher] 카메라 intrinsics fx={intr.fx:.1f} fy={intr.fy:.1f} {intr.width}x{intr.height}")
        if args.verbose_perception:
            builder = _VerboseBuilder(builder)
        for episode in range(1, args.episodes + 1):
            print(f"\n── 에피소드 {episode}/{args.episodes} ──")
            builder.reset()
            # ★F93(리뷰 OBS-2/D13)★ 시작 자세 게이트 — 학습은 항상 홈에서 시작한다.
            delta = _home_delta_rad(robot, robot.get_state())
            if float(np.abs(delta).max()) > _HOME_TOL_RAD:
                msg = f"시작 자세가 홈과 다르다: |Δq|max={np.degrees(float(np.abs(delta).max())):.1f}° (학습은 항상 홈에서 시작)"
                if args.dry_run:
                    print(f"[run-teacher][warn] {msg}")
                elif args.home_first:
                    print(f"[run-teacher] {msg} → 홈으로 이동(moveJ 0.1 배속)")
                    robot.move_to_joints(robot.home_joints_rad(), speed_scale=0.1)
                    delta = _home_delta_rad(robot, robot.get_state())
                    if float(np.abs(delta).max()) > _HOME_TOL_RAD:
                        _estop("홈 이동 후에도 시작 자세가 홈이 아니다"); break
                else:
                    _estop(msg + " — --home-first 로 홈 이동을 허용하거나 펜던트로 홈에 둘 것"); break
            if not args.dry_run:
                # ★F93(리뷰 D1)★ 명시적 개방 + command state 0.0 — sim 리셋(열린 그리퍼)과 같은 시작.
                _explicit_open(robot, control_hz)
            result = run_rollout(robot=robot, camera=camera, policy=policy, mapper=mapper, guard=guard, rate=rate,
                                 pre_cfg=pre_cfg, target_pos=target, max_steps=args.max_steps, dry_run=args.dry_run,
                                 obs_dump_dir=(args.dump_dir / f"ep{episode:02d}") if args.dump_dir else None,
                                 obs_builder=builder,
                                 # ★F93(리뷰 D8/AX-04)★ Deploy_DryRun과 같은 신선도·스큐·오버런 가드
                                 max_sensor_age_s=2.0 / control_hz, max_sensor_skew_s=1.0 / control_hz,
                                 max_consecutive_overruns=args.max_consecutive_overruns)
            s = builder.stats()
            print(f"   스텝 {result.steps}  trip {result.tripped}  overruns {result.overruns}")
            print(f"   인식: 검출 {s['detection_rate']*100:.0f}%  쥠 {s['held_fraction']*100:.0f}%  "
                  f"hold {s['hold_fraction']*100:.0f}%  급점프 거부 {s['jump_rejects']}  "
                  f"최대 블라인드 {s.get('max_blind_streak', 0)}프레임  마지막: {builder.debug.reason or 'ok'}")
            if s["detection_rate"] < 0.3:
                print("   ★큐브를 거의 못 봤다★ — 카메라·작업면 z·캘리브(hand_eye/tcp_offset)를 확인할 것")
            if result.tripped:
                break
    finally:
        try:
            if camera is not None:
                camera.stop()
        finally:
            if robot is not None:
                robot.close()
    return 1 if tripped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
