#!/usr/bin/env python
"""배포 PC 전수 점검 실행기 — 실기 구동까지 단계별로 확인하고 다음 할 일을 알려준다.

═══════════════════════════════════════════════════════════════════════════════
 이 스크립트는 ★로봇을 움직이지 않는다★. 점검과 안내만 한다.
 실제 모션은 단계 6이 통과한 뒤 Phase4 스크립트로 별도 실행한다.
═══════════════════════════════════════════════════════════════════════════════

왜 이 파일이 있는가
  실기 브링업은 "환경 → 산출물 → 설정 → 하드웨어 → 캘리브레이션 → 인가" 여섯 겹이
  모두 맞아야 한 걸음이 나간다. 어느 하나가 틀리면 증상은 대개 ★엉뚱한 곳★에서
  나타난다(예: 캘리브레이션 미완인데 "정책 로드 실패"로 보임). 그래서 각 겹을 순서대로
  독립 확인하고, 실패하면 ★그 겹에서 멈춰★ 무엇을 해야 하는지 말한다.

실행 (배포 PC, 프로젝트 루트에서):
  conda activate env_ur_real
  python deployment/DeployPC_Runbook.py --policy-dir exported_experimental/<sha12>
  python deployment/DeployPC_Runbook.py --policy-dir <...> --with-hardware   # 로봇/카메라 연결까지

옵션:
  --with-hardware   RTDE 수신 연결과 RealSense 열기를 시도한다(수신 전용, 모션 없음)
  --stage N         특정 단계만 실행
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training_env import configure_utf8_stdio  # noqa: E402

configure_utf8_stdio()   # ★F93(리뷰 D11)★ 이모지·박스문자가 cp949 콘솔에서 UnicodeEncodeError로 죽지 않게

_PASS, _FAIL, _SKIP, _WARN = "PASS", "FAIL", "SKIP", "WARN"
_MARK = {_PASS: "✅", _FAIL: "❌", _SKIP: "⏭️ ", _WARN: "⚠️ "}


class Stage:
    """한 겹의 점검 — 결과와 '다음에 할 일'을 함께 들고 다닌다."""

    def __init__(self, number: int, title: str) -> None:
        self.number = number
        self.title = title
        self.status = _SKIP
        self.lines: list[str] = []
        self.next_action = ""

    def ok(self, message: str) -> None:
        self.lines.append(f"    · {message}")

    def fail(self, message: str, next_action: str) -> None:
        self.status = _FAIL
        self.lines.append(f"    ✗ {message}")
        if not self.next_action:
            self.next_action = next_action

    def warn(self, message: str) -> None:
        if self.status != _FAIL:
            self.status = _WARN
        self.lines.append(f"    ⚠ {message}")

    def done(self) -> None:
        if self.status == _SKIP:
            self.status = _PASS

    def render(self) -> None:
        print(f"\n[{self.number}] {_MARK[self.status]} {self.title}")
        for line in self.lines:
            print(line)
        if self.next_action:
            print(f"    ▶ 할 일: {self.next_action}")


# ══════════════════════════════════════════════════════════════════════════════
def stage_environment() -> Stage:
    """① 파이썬 환경 — 배포 PC에만 있는 SDK가 실제로 import되는가."""
    stage = Stage(1, "실행 환경 (env_ur_real)")
    stage.ok(f"python {sys.version.split()[0]} — {sys.executable}")
    for module, why, hard in (
        ("torch", "정책 추론", True),
        ("numpy", "관측 전처리", True),
        ("yaml", "설정 로드", True),
        ("rtde_receive", "로봇 상태 수신(ur-rtde)", False),
        ("rtde_control", "로봇 명령(ur-rtde)", False),
        ("pyrealsense2", "카메라(D435i)", False),
    ):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "")
            stage.ok(f"{module} {version} — {why}")
        except ImportError:
            if hard:
                stage.fail(f"{module} 없음 ({why})",
                           "conda activate env_ur_real 후 재실행. 환경이 없으면 "
                           "deployment/requirements-win11.lock.txt로 설치")
            else:
                stage.warn(f"{module} 없음 ({why}) — 하드웨어 단계는 건너뛴다")
    stage.done()
    return stage


def stage_policy_artifact(policy_dir: Path) -> tuple[Stage, dict | None]:
    """② 정책 산출물 — 해시가 맞고 TorchScript가 단독으로 추론되는가."""
    stage = Stage(2, "정책 산출물 무결성")
    manifest_path = policy_dir / "experimental_manifest.json"
    if not manifest_path.is_file():
        stage.fail(f"매니페스트 없음: {manifest_path}",
                   "학습 PC에서 만든 exported_experimental/<sha12> 폴더를 통째로 복사할 것")
        return stage, None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage.ok(f"artifact_type={manifest.get('artifact_type')} "
             f"release={manifest.get('release')} adopted={manifest.get('adopted')}")
    if manifest.get("release") is not False:
        stage.warn("이 폴더가 릴리스로 표시돼 있다 — 실험 레인 산출물이 맞는지 확인할 것")
    stage.ok(f"checkpoint iter={manifest['checkpoint']['iteration']} "
             f"sha={manifest['checkpoint']['sha256'][:12]}")

    import hashlib

    for name, expected in manifest.get("files", {}).items():
        target = policy_dir / name
        if not target.is_file():
            stage.fail(f"파일 없음: {name}", "폴더를 다시 복사할 것(일부만 복사된 상태)")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            stage.fail(f"{name} 해시 불일치", "폴더를 다시 복사할 것(전송 중 손상)")
        else:
            stage.ok(f"{name} 해시 일치")

    try:
        import torch

        policy = torch.jit.load(str(policy_dir / "policy.pt"), map_location="cpu")
        policy.eval()
        obs_dim = int(manifest["contract"]["obs_dim"])
        with torch.no_grad():
            action = policy(torch.zeros(2, obs_dim))
        if action.shape != (2, int(manifest["contract"]["action_dim"])):
            stage.fail(f"행동 폭이 계약과 다르다: {tuple(action.shape)}",
                       "학습 PC에서 export를 다시 수행할 것")
        elif not bool(torch.isfinite(action).all()):
            stage.fail("정책 출력에 NaN/Inf", "이 정책은 실기에 올리지 말 것")
        else:
            stage.ok(f"TorchScript 단독 추론 OK ({obs_dim} → {action.shape[1]})")
    except ImportError:
        stage.fail("torch 없음", "단계 1을 먼저 통과시킬 것")
    except Exception as exc:  # noqa: BLE001
        stage.fail(f"정책 로드 실패: {exc}", "폴더 재복사 또는 학습 PC에서 재export")
    stage.done()
    return stage, manifest


def stage_configs(policy_dir: Path) -> Stage:
    """③ 설정 — 안전 한계가 실재하고 값이 상식적인가."""
    stage = Stage(3, "설정과 안전 한계")
    try:
        import yaml
    except ImportError:
        stage.fail("yaml 없음", "단계 1을 먼저 통과시킬 것")
        return stage

    safety_path = ROOT / "configs" / "safety.yaml"
    if not safety_path.is_file():
        stage.fail("configs/safety.yaml 없음", "저장소 configs/를 배포 PC로 동기화할 것")
        return stage
    safety = yaml.safe_load(safety_path.read_text(encoding="utf-8")) or {}
    workspace = safety.get("workspace", {})
    limits = safety.get("limits", {})
    if not workspace.get("min") or not workspace.get("max"):
        stage.fail("workspace AABB가 비어 있다", "configs/safety.yaml을 확인할 것")
    else:
        stage.ok(f"workspace AABB min={workspace['min']} max={workspace['max']}")
        stage.ok(f"TCP z 바닥 = {workspace.get('tcp_z_floor_m')} m")
    stage.ok(f"관절속도 상한 = {limits.get('joint_vel_max_rad_s')} rad/s, "
             f"TCP 힘 상한 = {limits.get('tcp_force_max_n')} N")

    real_path = ROOT / "configs" / "real.local.yaml"
    if not real_path.is_file():
        stage.fail("configs/real.local.yaml 없음",
                   "configs/real.example.yaml을 복사해 로봇 IP·카메라 시리얼을 채울 것")
    else:
        real = yaml.safe_load(real_path.read_text(encoding="utf-8")) or {}
        stage.ok(f"로봇 IP = {real.get('robot', {}).get('ip')}")
        serial = str(real.get("camera", {}).get("serial", "") or "")
        if not serial:
            stage.fail("camera.serial 이 비어 있다",
                       "D435i 시리얼을 configs/real.local.yaml의 camera.serial에 기입할 것 "
                       "(자동 선택은 계약상 금지)")
        else:
            stage.ok(f"카메라 시리얼 = {serial}")
    stage.done()
    return stage


def stage_calibration() -> Stage:
    """④ 캘리브레이션 — policy 모션의 선행 조건."""
    stage = Stage(4, "캘리브레이션 상태")
    try:
        import yaml
    except ImportError:
        stage.fail("yaml 없음", "단계 1을 먼저 통과시킬 것")
        return stage
    pending = []
    for name in ("tcp_offset.yaml", "hand_eye.yaml"):
        path = ROOT / "configs" / "calib" / name
        if not path.is_file():
            stage.fail(f"{name} 없음", "저장소 configs/calib/을 동기화할 것")
            continue
        meta = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("meta", {})
        calibrated = bool(meta.get("calibrated"))
        residual = meta.get("residual_m")
        stage.ok(f"{name}: calibrated={calibrated}"
                 + (f", residual={residual}" if residual is not None else ""))
        if not calibrated:
            pending.append(name)
    if pending:
        stage.warn(f"미완: {', '.join(pending)} — `policy` 범위 모션은 코드가 거부한다")
        stage.next_action = ("캘리브레이션을 먼저 수행하고 meta.calibrated: true 로 갱신할 것. "
                             "그 전까지는 policy_dryrun(추론만)까지 진행 가능")
    stage.done()
    return stage


def stage_authorization() -> Stage:
    """⑤ 모션 인가 — 없으면 모든 모션이 차단된다(기본값은 안전)."""
    stage = Stage(5, "모션 인가")
    from deployment.contracts import ContractError, load_motion_authorization

    try:
        payload = load_motion_authorization(ROOT)
    except ContractError as exc:
        stage.warn(f"인가 없음/무효: {exc}")
        stage.next_action = ("configs/motion_authorization.example.yaml 을 "
                             "configs/motion_authorization.yaml 로 복사해 채울 것 "
                             "(operator, expires_at ≤ 72h, scopes)")
        stage.done()
        return stage
    stage.ok(f"입회자 = {payload.get('operator')}")
    stage.ok(f"만료 = {payload.get('expires_at')}")
    stage.ok(f"허용 범위 = {payload.get('scopes')}")
    for scope in ("freedrive", "scripted_pose", "gripper", "policy_dryrun", "policy"):
        try:
            from deployment.contracts import require_motion_authorization

            require_motion_authorization(scope, project_root=ROOT)
            stage.ok(f"범위 {scope}: 통과")
        except ContractError as exc:
            stage.ok(f"범위 {scope}: 차단 ({str(exc)[:60]})")
    stage.done()
    return stage


def stage_hardware(enabled: bool) -> Stage:
    """⑥ 하드웨어 — 수신 전용 연결만 시도한다(명령 없음)."""
    stage = Stage(6, "하드웨어 연결 (수신 전용, 모션 없음)")
    if not enabled:
        stage.lines.append("    · --with-hardware 를 주면 실제 연결을 시도한다")
        return stage
    import yaml

    real = yaml.safe_load((ROOT / "configs" / "real.local.yaml").read_text(encoding="utf-8"))
    try:
        from real.real_robot import RealRobot

        robot = RealRobot(real, ROOT / "configs" / "calib" / "tcp_offset.yaml", receive_only=True)
        state = robot.get_state()   # ★F93(리뷰 D4)★ 실제 어댑터 API (read_state는 존재하지 않았다)
        stage.ok(f"RTDE 수신 OK — 관절 {[round(float(v), 3) for v in state.joint_pos[:6]]}")
        stage.ok(f"TCP pose = {[round(float(v), 4) for v in state.tcp_pose[:3]]}")
        close = getattr(robot, "close", None) or getattr(robot, "disconnect", None)
        if callable(close):
            close()
    except Exception as exc:  # noqa: BLE001
        stage.fail(f"RTDE 연결 실패: {exc}",
                   "① 로봇 전원/Remote Control ② IP·서브넷 ③ 방화벽 "
                   "④ 다른 RTDE 클라이언트 점유 여부를 점검할 것")
    try:
        from real.real_camera import RealCamera

        camera = RealCamera(real["camera"], enable_rgb=True)
        camera.start()              # ★F93(리뷰 D4)★ 실제 어댑터 API: start / get_frame / stop
        try:
            frame = camera.get_frame()
            stage.ok(f"RealSense OK — depth {getattr(frame, 'depth_m', None) is not None}, "
                     f"rgb {getattr(frame, 'rgb', None) is not None}")
        finally:
            camera.stop()
    except Exception as exc:  # noqa: BLE001
        stage.fail(f"카메라 열기 실패: {exc}",
                   "① USB3 연결 ② 시리얼 일치 ③ 다른 프로세스 점유 ④ 펌웨어를 점검할 것")
    stage.done()
    return stage


# ══════════════════════════════════════════════════════════════════════════════
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", required=True, type=Path)
    parser.add_argument("--with-hardware", action="store_true")
    parser.add_argument("--stage", type=int, default=None)
    args = parser.parse_args(argv[1:])

    policy_dir = args.policy_dir.expanduser().resolve()
    print("═" * 78)
    print(" 배포 PC 전수 점검 — 이 스크립트는 로봇을 움직이지 않는다")
    print(f" 정책 폴더: {policy_dir}")
    print("═" * 78)

    stages: list[Stage] = []
    manifest = None
    for number, runner in (
        (1, lambda: stage_environment()),
        (2, lambda: None),          # 아래에서 특별 처리(매니페스트 반환)
        (3, lambda: stage_configs(policy_dir)),
        (4, lambda: stage_calibration()),
        (5, lambda: stage_authorization()),
        (6, lambda: stage_hardware(args.with_hardware)),
    ):
        if args.stage is not None and args.stage != number:
            continue
        if number == 2:
            stage, manifest = stage_policy_artifact(policy_dir)
        else:
            stage = runner()
        stages.append(stage)
        stage.render()

    print("\n" + "═" * 78)
    failed = [s for s in stages if s.status == _FAIL]
    warned = [s for s in stages if s.status == _WARN]
    if failed:
        print(f" ❌ {len(failed)}개 단계 실패 — 위 '할 일'을 처리하고 재실행할 것")
        for stage in failed:
            print(f"    [{stage.number}] {stage.title}: {stage.next_action}")
        return 1

    print(" ✅ 실패한 단계 없음")
    if warned:
        print(f" ⚠️  {len(warned)}개 단계 주의 — 다음 단계 진행 조건을 확인할 것")
        for stage in warned:
            if stage.next_action:
                print(f"    [{stage.number}] {stage.next_action}")
    print("\n 다음 순서:")
    print("   ① 캘리브레이션 완료 → configs/calib/*.yaml 의 meta.calibrated: true")
    print("   ② 모션 인가에 scope 'policy_dryrun' 추가 → 정책 추론만 확인(명령 미전송)")
    print("   ③ 확인 후 scope 'policy' 추가 → 낮은 speed_scale 로 실제 모션")
    print("   ★ 작업 내내 e-stop 도달 거리 안에 사람이 있어야 한다 ★")
    print("═" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
