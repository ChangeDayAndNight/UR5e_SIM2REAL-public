#!/usr/bin/env python
"""학습한 Task를 실기에서 ★한 명령으로★ 실행한다. [사람][배포 PC]

═══════════════════════════════════════════════════════════════════════════════
 로봇이 실제로 움직인다. e-stop에 손이 닿는 위치에 서 있을 것.
═══════════════════════════════════════════════════════════════════════════════

무엇을 걷어냈나 (마찰만 만들고 안전을 더하지 않던 것)
  · 인가 YAML 파일 · 만료시각 · 확인사항 3항  → `--yes-i-am-at-the-estop` 플래그 하나로
  · 캘리브레이션 완료 요구(meta.calibrated)   → 공칭값으로 진행(정확도는 떨어진다)
  · 릴리스 계약(채택·parity·번들 매니페스트)  → 실험 정책 폴더를 직접 읽는다

무엇을 남겼나 (이건 절차가 아니라 물리적 방어다)
  · SafetyGuard — 작업영역 AABB 이탈, TCP z 바닥(벨트 압착), 관절속도·힘 상한에서 즉시 trip
  · 액션 스케일 축소 기본값 0.3 — 학습 시 크기의 30%로 시작한다
  · max-steps 상한 — 한 에피소드가 끝없이 돌지 않는다
  · `--dry-run` — 명령만 빼고 전부 실행(첫 확인용)

실행:
  cd <Tomorrow 폴더>
  conda activate env_ur_real

  # ① 먼저 이것부터 — 로봇은 안 움직이고 추론만 한다
  python Phase4/Run_Task.py --policy-dir policy/d3d2ce73636f --dry-run

  # ② 실제 모션 (천천히)
  python Phase4/Run_Task.py --policy-dir policy/d3d2ce73636f --yes-i-am-at-the-estop

  # ③ 익숙해지면 스케일을 올린다
  python Phase4/Run_Task.py --policy-dir policy/d3d2ce73636f --yes-i-am-at-the-estop
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.action_mapping import ActionMapper
from common.config_io import (
    build_action_cfg,
    build_preprocess_cfg,
    build_safety_cfg,
    configs_dir,
    load_yaml,
)
from common.policy_io import PolicyRunner
from common.rollout import run_rollout
from common.safety import SafetyGuard
from common.timing import RateLimiter

# 학습 시 타겟 샘플 범위의 중심 — 지정하지 않으면 여기로 놓으려 한다.
_DEFAULT_TARGET = "-0.50,0.0,-0.0125"


class _CubeGate:
    """큐브 미검출은 팔 정지 요청이다. 그리퍼 개방 요청이 아니다.

    왜 이 게이트가 필요한가
      student 정책은 큐브를 인식하지 않는다(end-to-end 사상). 신경망은 어떤 입력에도
      출력을 내므로 큐브가 아닌 물체나 빈 장면에서도 액션이 나가 로봇이 움직인다.
      정책을 바꾸지 않고 ★정책 앞에★ 게이트를 두어 그 구멍을 막는다.

    왜 래퍼인가
      `run_rollout`은 `camera.get_frame()` → `policy.act(obs)` 순서로 돈다. 게이트는
      원본 RGB가 필요한데 정책은 전처리된 obs만 받는다. 그래서 카메라를 감싸 프레임을
      가로채 판정하고, 정책이 hold_requested로 공유 rollout에 정지를 요청한다.
      rollout은 팔 정지와 그리퍼 개방을 구분하고 supervisor의 거짓 릴리스를 막는다.
    """

    def __init__(self, min_area_px: int | None) -> None:
        self.min_area_px = min_area_px
        self.present = False
        self.last_reason = "아직 프레임 없음"
        self.blocked_steps = 0
        self.total_steps = 0

    def judge(self, frame) -> None:
        from deployment.cube_detector import MIN_AREA_PX, detect_cube

        rgb = getattr(frame, "rgb", None)
        if rgb is None:
            self.present, self.last_reason = False, "RGB 없음 — hold"
            return
        import numpy as _np

        result = detect_cube(
            _np.asarray(rgb),
            min_area_px=self.min_area_px if self.min_area_px is not None else MIN_AREA_PX,
        )
        self.present = result.present
        self.last_reason = result.reason or f"{result.color} {result.area_px}px²"


class _GatedCamera:
    """카메라 래퍼 — 프레임을 가로채 게이트를 갱신하고 그대로 넘긴다."""

    def __init__(self, camera, gate: _CubeGate) -> None:
        self._camera, self._gate = camera, gate

    def get_frame(self):
        frame = self._camera.get_frame()
        self._gate.judge(frame)
        return frame

    def __getattr__(self, name):     # start/stop/is_ok 등은 원본에 위임
        return getattr(self._camera, name)


class _GatedPolicy:
    """Request run_rollout's explicit hold; a zero gripper action means OPEN.

    The consumer must honor hold_requested, preserve the issued gripper command,
    and suspend release detection while the camera cannot see the cube.
    """

    def __init__(self, policy, gate: _CubeGate, action_dim: int) -> None:
        self._policy, self._gate = policy, gate
        self._zero = np.zeros(action_dim, dtype=np.float64)
        self.hold_requested = False

    def act(self, obs):
        self._gate.total_steps += 1
        self.hold_requested = not self._gate.present
        if self.hold_requested:
            self._gate.blocked_steps += 1
            return self._zero.reshape(1, -1)
        return self._policy.act(obs)

    def __getattr__(self, name):
        return getattr(self._policy, name)


def _load_policy_dir(policy_dir: Path) -> dict:
    """실험 정책 폴더를 읽고 파일 해시를 대조한다(전송 중 손상 차단)."""
    manifest_path = policy_dir / "experimental_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"[run] 매니페스트 없음: {manifest_path}\n"
                         f"      exported_experimental/<sha12> 폴더를 지정할 것")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib

    for name, expected in manifest.get("files", {}).items():
        target = policy_dir / name
        if not target.is_file():
            raise SystemExit(f"[run] 파일 없음: {target}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"[run] {name} 해시 불일치 — 폴더를 다시 복사할 것")
    return manifest


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-dir", required=True, type=Path)
    parser.add_argument("--target-pos", default=_DEFAULT_TARGET,
                        help=f"놓을 위치 'x,y,z' [m] (기본 {_DEFAULT_TARGET})")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="한 에피소드 최대 스텝 (기본 200 = 20Hz에서 10초 = 학습 에피소드)")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="학습 액션 크기 대비 배율 (기본 1.0 = 학습 조건; SafetyGuard가 스텝 상한을 이미 건다. "
                             "<1.0은 릴리스 수퍼바이저 정지 임계와 어긋난다 — F93)")
    parser.add_argument("--dry-run", action="store_true",
                        help="명령을 보내지 않는다 — 취득·전처리·추론·안전판정만 실행")
    parser.add_argument("--yes-i-am-at-the-estop", dest="operator_ack", action="store_true",
                        help="★실제 모션★ — e-stop 앞에 있음을 선언한다")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-cube-gate", action="store_true",
                        help="큐브 인식 게이트 해제 — 큐브가 안 보여도 정책을 그대로 실행")
    parser.add_argument("--gate-min-area", type=int, default=None,
                        help="큐브 판정 최소 면적[px²] (기본: 검출기 기본값 700)")
    args = parser.parse_args(argv[1:])

    if not args.dry_run and not args.operator_ack:
        print("[run] 모션을 내려면 --yes-i-am-at-the-estop 이 필요하다.")
        print("      먼저 --dry-run 으로 추론만 확인하는 것을 권장한다.")
        return 2
    if not 0.0 < args.action_scale <= 1.0:
        print("[run] --action-scale 은 (0, 1] 범위여야 한다")
        return 2

    policy_dir = args.policy_dir.expanduser().resolve()
    manifest = _load_policy_dir(policy_dir)

    # ── 설정: 번들 매니페스트를 거치지 않고 configs/ 에서 직접 만든다 ──────────
    common_yaml = load_yaml(configs_dir() / "common.yaml")
    pre_cfg = build_preprocess_cfg(common_yaml)
    action_cfg = build_action_cfg(common_yaml)
    safety_cfg = build_safety_cfg(load_yaml(configs_dir() / "safety.yaml"))
    real_yaml = load_yaml(configs_dir() / "real.local.yaml")
    control_hz = float(common_yaml["control"]["control_hz"])

    # ★액션 스케일 축소★ — 정책 출력은 그대로 두고 ★변환 단계★에서 줄인다.
    #   정책을 건드리지 않으므로 방향은 학습 그대로이고 크기만 작아진다.
    scaled_action_cfg = action_cfg
    if args.action_scale < 1.0:
        import dataclasses

        scaled_action_cfg = dataclasses.replace(
            action_cfg,
            task_pos_scale_m=action_cfg.task_pos_scale_m * args.action_scale,
            task_rot_scale_rad=action_cfg.task_rot_scale_rad * args.action_scale,
        )

    target = np.array([float(v) for v in args.target_pos.split(",")], dtype=np.float64)
    lo = np.asarray(safety_cfg.workspace_min, dtype=np.float64)
    hi = np.asarray(safety_cfg.workspace_max, dtype=np.float64)
    if np.any(target < lo) or np.any(target > hi):
        print(f"[run] --target-pos {target.tolist()} 가 작업영역 "
              f"{lo.tolist()}..{hi.tolist()} 밖 — 거부")
        return 2

    print("═" * 72)
    print(f" 정책 : {policy_dir.name}  (iter {manifest['checkpoint']['iteration']}, "
          f"릴리스 아님)")
    print(f" 모드 : {'DRY-RUN (명령 미전송)' if args.dry_run else '★실제 모션★'}"
          f"   액션 스케일 {args.action_scale:.2f}")
    print(f" 타겟 : {target.tolist()}   최대 {args.max_steps}스텝 × {args.episodes}회")
    print(f" 안전 : AABB {lo.tolist()}..{hi.tolist()}, TCP z 바닥 "
          f"{getattr(safety_cfg, 'tcp_z_floor_m', 'n/a')}")
    print("═" * 72)

    # ★F81 정정 — PolicyRunner에는 ★원본★ action_cfg를 넘긴다 (리뷰 major #11)★
    #   obs_spec의 cfg_hash는 export 시점의 action_cfg 전체를 해시한 값이다. 축소된
    #   cfg를 넘기면 해시가 어긋나 정책 로드가 ★거부★된다(실기에서 실제로 겪은 실패).
    #   속도 축소는 관측 계약과 무관한 '실행' 단계의 결정이므로 ActionMapper에만 적용한다.
    policy = PolicyRunner(
        policy_dir / "policy.pt",
        policy_dir / "obs_spec.json",
        pre_cfg,
        action_cfg,
        device=args.device,
        expected_policy_sha256=manifest["files"]["policy.pt"],
        expected_obs_spec_sha256=manifest["files"]["obs_spec.json"],
    )
    mapper = ActionMapper(scaled_action_cfg)

    tripped: list[str] = []

    def _estop(reason: str) -> None:
        tripped.append(reason)
        print(f"\n[run][SAFETY] ★trip★ {reason}")

    guard = SafetyGuard(safety_cfg, estop_hook=_estop)
    rate = RateLimiter(control_hz)

    from real.real_camera import RealCamera
    from real.real_robot import RealRobot

    robot = camera = None
    try:
        robot = RealRobot(
            real_yaml,
            configs_dir() / "calib" / "tcp_offset.yaml",
            receive_only=args.dry_run,
            motion_scope="policy",
            operator_ack=args.operator_ack,
        )
        camera = RealCamera(real_yaml["camera"], enable_rgb=pre_cfg.use_rgb)
        camera.start()

        # ★큐브 인식 게이트★ — 큐브가 안 보이면 정책 출력을 0으로 만들어 로봇을 세운다.
        gate = None
        if not args.no_cube_gate:
            if not pre_cfg.use_rgb:
                print("[run][warn] use_rgb=false — 색 기반 게이트를 쓸 수 없어 해제한다")
            else:
                gate = _CubeGate(args.gate_min_area)
                camera = _GatedCamera(camera, gate)
                policy = _GatedPolicy(policy, gate, scaled_action_cfg.action_dim)
                print("[run] 큐브 게이트 활성 — 큐브 미검출 시 정지(--no-cube-gate로 해제)")

        for episode in range(1, args.episodes + 1):
            print(f"\n── 에피소드 {episode}/{args.episodes} ──")
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
                dry_run=args.dry_run,
            )
            print(f"   스텝 {getattr(result, 'steps', '?')}  "
                  f"trip {getattr(result, 'tripped', bool(tripped))}")
            if gate is not None and gate.total_steps:
                ratio = gate.blocked_steps / gate.total_steps * 100
                print(f"   큐브 게이트: {gate.blocked_steps}/{gate.total_steps} 스텝 정지 "
                      f"({ratio:.0f}%)  마지막 판정: {gate.last_reason}")
                if gate.blocked_steps == gate.total_steps:
                    print("   ★큐브를 한 번도 못 봤다★ — 조명/색/거리를 확인하거나 "
                          "--gate-min-area 를 낮춰 볼 것")
            if tripped:
                print("   ★안전 정지가 걸렸다 — 원인을 확인하기 전에는 계속하지 말 것★")
                break
    except KeyboardInterrupt:
        print("\n[run] 사용자 중단 (Ctrl+C)")
    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[run][warn] 카메라 종료 실패: {exc}")
        if robot is not None:
            try:
                robot.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[run][warn] 로봇 종료 실패: {exc}")

    print("\n" + "═" * 72)
    if tripped:
        print(f" ⚠️  안전 정지 {len(tripped)}회: {tripped[:3]}")
        print("    액션 스케일을 낮추거나(--action-scale 0.15) 타겟을 조정할 것")
        return 1
    print(" ✅ 종료 — 안전 정지 없음")
    if args.dry_run:
        print("    다음: --yes-i-am-at-the-estop 을 붙여 실제 모션으로")
    print("═" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
