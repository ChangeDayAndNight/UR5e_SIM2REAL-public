#!/usr/bin/env python
"""★배포 v2★ 실기 실행 — 시각 student(이력 정책). [배포 PC][Phase 4]

Phase4/Run_Task.py와 ★같은 제어 루프·안전 장치·릴리스 수퍼바이저·모션 인가★를 쓴다.
다른 것은 둘이다:
  · 정책: 무상태 이력 actor + StudentHistoryPolicyRunner(16스텝 버퍼, 에피소드마다 reset)
  · 카메라: manifest의 fov_match가 켜져 있으면 FovMatchedCamera로 학습 화각에 맞춘다

큐브 게이트는 Run_Task.py와 동일하게 적용한다 — end-to-end 정책은 큐브를 인식하지 않으므로
빈 장면에서도 무언가를 출력한다.

실행:
  python Phase4/Run_Task_Student.py --policy-dir exported_experimental/student_<sha12>_fovmatch --dry-run
  python Phase4/Run_Task_Student.py --policy-dir ... --yes-i-am-at-the-estop
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training_env import configure_utf8_stdio  # noqa: E402
configure_utf8_stdio()

from common.action_mapping import ActionMapper  # noqa: E402
from common.config_io import (  # noqa: E402
    build_action_cfg, build_preprocess_cfg, build_safety_cfg, configs_dir, load_yaml,
)
from common.fov_match import FovMatchedCamera, FovSpec  # noqa: E402
from common.rollout import run_rollout  # noqa: E402
from common.safety import SafetyGuard  # noqa: E402
from common.student_policy_io import StudentHistoryPolicyRunner  # noqa: E402
from common.timing import RateLimiter  # noqa: E402

_DEFAULT_TARGET = "-0.50,0.15,-0.0125"


class _CubeGate:
    """큐브가 보이는지 판정한다(Phase4/Run_Task.py의 게이트와 같은 검출기·같은 계약)."""

    def __init__(self, min_area: int | None) -> None:
        self.min_area, self.present, self.last_reason = min_area, False, "(아직 판정 없음)"
        self.total_steps = self.blocked_steps = 0

    def judge(self, frame) -> None:
        from deployment.cube_detector import MIN_AREA_PX, detect_cube
        if frame.rgb is None:
            self.present, self.last_reason = False, "RGB 없음 — hold"
            return
        result = detect_cube(frame.rgb, min_area_px=self.min_area or MIN_AREA_PX)
        self.present, self.last_reason = result.present, result.reason


class _GatedCamera:
    def __init__(self, camera, gate: _CubeGate) -> None:
        self._camera, self._gate = camera, gate

    def __getattr__(self, name):
        return getattr(self._camera, name)

    def get_frame(self):
        frame = self._camera.get_frame()
        self._gate.judge(frame)
        return frame


class _GatedPolicy:
    """큐브 미검출이면 run_rollout에 hold를 요청한다(그리퍼 명령은 보존)."""

    def __init__(self, policy, gate: _CubeGate) -> None:
        self._policy, self._gate = policy, gate
        self.hold_requested = False

    def reset(self) -> None:
        self._policy.reset()

    def act(self, obs):
        self._gate.total_steps += 1
        self.hold_requested = not self._gate.present
        if self.hold_requested:
            self._gate.blocked_steps += 1
        return self._policy.act(obs)


def _load_policy_dir(policy_dir: Path) -> dict:
    manifest_path = policy_dir / "experimental_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"[run-student] 매니페스트 없음: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "experimental_student_policy":
        raise SystemExit(f"[run-student] student 번들이 아니다: {manifest.get('artifact_type')!r} "
                         f"(teacher 번들은 Phase4/Run_Task_Teacher.py)")
    for name, expected in manifest.get("files", {}).items():
        target = policy_dir / name
        if not target.is_file():
            raise SystemExit(f"[run-student] 파일 없음: {target}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"[run-student] {name} 해시 불일치 — 폴더를 다시 복사할 것")
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-dir", required=True, type=Path)
    parser.add_argument("--target-pos", default=_DEFAULT_TARGET)
    parser.add_argument("--max-steps", type=int, default=200,
                        help="에피소드 최대 스텝 (기본 200 = 학습 에피소드 10 s × 20 Hz)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="기본 1.0 = 학습 조건 (SafetyGuard 상한과 동일). <1.0은 릴리스 수퍼바이저 정지 임계와 어긋난다 — F93")
    parser.add_argument("--dry-run", action="store_true", help="명령 미전송 — 취득·전처리·추론·안전판정만")
    parser.add_argument("--yes-i-am-at-the-estop", dest="operator_ack", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dump-dir", type=Path, default=None)
    parser.add_argument("--no-cube-gate", action="store_true")
    parser.add_argument("--gate-min-area", type=int, default=None)
    args = parser.parse_args(argv[1:])
    if not args.dry_run and not args.operator_ack:
        print("[run-student] 모션을 내려면 --yes-i-am-at-the-estop 이 필요하다. 먼저 --dry-run 을 권장한다.")
        return 2
    if not 0.0 < args.action_scale <= 1.0:
        print("[run-student] --action-scale 은 (0, 1] 범위여야 한다"); return 2

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
        print(f"[run-student] --target-pos {target.tolist()} 가 작업영역 밖 — 거부"); return 2

    fov = manifest["fov_match"]
    hist = manifest["history"]
    print("═" * 72)
    print(f" 정책 : {policy_dir.name}  (student iter {manifest['checkpoint']['iteration']}, 릴리스 아님)")
    print(f" 모드 : {'DRY-RUN (명령 미전송)' if args.dry_run else '★실제 모션★'}   액션 스케일 {args.action_scale:.2f}")
    print(f" 이력 : {hist['steps']}단계 (에피소드마다 reset)")
    print(f" 화각 : 보정 {'★ON★' if fov['enabled'] else 'OFF'}"
          + (f"  실기 세로 {fov['vertical_content_fraction']*100:.0f}%를 학습 화각 가운데 배치" if fov["enabled"] else "  프레임 그대로"))
    print(f" 타겟 : {target.tolist()}   최대 {args.max_steps}스텝 × {args.episodes}회")
    print("═" * 72)

    policy = StudentHistoryPolicyRunner(
        policy_dir / "policy.pt", policy_dir / "obs_spec.json", pre_cfg, action_cfg,
        history_steps=int(hist["steps"]), feature_dim=int(hist["feature_dim"]), device=args.device,
        expected_policy_sha256=manifest["files"]["policy.pt"],
        expected_obs_spec_sha256=manifest["files"]["obs_spec.json"])
    mapper = ActionMapper(scaled_action_cfg)
    tripped: list[str] = []

    def _estop(reason: str) -> None:
        tripped.append(reason); print(f"\n[run-student][SAFETY] ★trip★ {reason}")

    guard = SafetyGuard(safety_cfg, estop_hook=_estop)
    rate = RateLimiter(control_hz)
    from real.real_camera import RealCamera
    from real.real_robot import RealRobot
    robot = camera = None
    try:
        robot = RealRobot(real_yaml, configs_dir() / "calib" / "tcp_offset.yaml", receive_only=args.dry_run,
                          motion_scope="policy", operator_ack=args.operator_ack)
        camera = RealCamera(real_yaml["camera"], enable_rgb=pre_cfg.use_rgb)
        camera.start()
        if fov["enabled"]:
            training = manifest["training_fov_deg"]
            real = manifest["real_fov_deg"]
            camera = FovMatchedCamera(
                camera, FovSpec(float(training["horizontal"]), float(training["vertical"])),
                source=FovSpec(float(real["horizontal"]), float(real["vertical"])))
            print(f"[run-student] 화각 보정 활성 — 실기 {real['horizontal']}x{real['vertical']}° "
                  f"→ 학습 {training['horizontal']}x{training['vertical']}°")
        gate = None
        runner = policy
        if not args.no_cube_gate:
            if not pre_cfg.use_rgb:
                print("[run-student][warn] use_rgb=false — 색 기반 게이트를 쓸 수 없어 해제한다")
            else:
                gate = _CubeGate(args.gate_min_area)
                camera = _GatedCamera(camera, gate)
                runner = _GatedPolicy(policy, gate)
                print("[run-student] 큐브 게이트 활성 — 미검출 시 정지(--no-cube-gate로 해제)")
        for episode in range(1, args.episodes + 1):
            print(f"\n── 에피소드 {episode}/{args.episodes} ──")
            policy.reset()          # ★이력 버퍼 초기화 — 에피소드가 이어지지 않게★
            result = run_rollout(robot=robot, camera=camera, policy=runner, mapper=mapper, guard=guard,
                                 rate=rate, pre_cfg=pre_cfg, target_pos=target, max_steps=args.max_steps,
                                 dry_run=args.dry_run,
                                 obs_dump_dir=(args.dump_dir / f"ep{episode:02d}") if args.dump_dir else None)
            print(f"   스텝 {result.steps}  trip {result.tripped}  overruns {result.overruns}  "
                  f"정책 스텝 {policy.steps}")
            if gate is not None and gate.total_steps:
                ratio = gate.blocked_steps / gate.total_steps * 100
                print(f"   큐브 게이트: {gate.blocked_steps}/{gate.total_steps} 스텝 정지 ({ratio:.0f}%)  "
                      f"마지막: {gate.last_reason}")
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
