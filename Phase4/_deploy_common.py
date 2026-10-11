# Phase4 공용 헬퍼 — Deploy_DryRun / Deploy_Real이 함께 쓰는 순수 로직.
# ★배포 PC(env_ur_real) 전제: 이 파일 어디서도 isaaclab / isaacsim을 import하지
#   않는다. ur_rtde / pyrealsense2도 직접 만지지 않는다 — real/ 어댑터가 소유하고,
#   여기서는 build_real_robot/build_real_camera의 lazy import로만 접근한다.★
from __future__ import annotations

import csv
import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.action_mapping import ActionMapper
from common.config_io import (
    build_action_cfg,
    build_preprocess_cfg,
    build_safety_cfg,
    load_yaml_snapshot,
    validate_target_sample_range,
)
from common.file_snapshot import FileSnapshotError
from common.interfaces import CameraInterface, RobotInterface
from common.obs_preprocess import obs_dim
from common.timing import RateLimiter
from common.types import ActionCfg, PreprocessCfg, RobotState, SafetyCfg
from deployment.contracts import (
    ContractError,
    load_real_config,
    require_bundle_local_real_config,
    verify_evidence,
)
from deployment.release import verify_release
from deployment.verify_deployment import verify_bundle_manifest

# ---------- 경로 상수 (이 파일 위치 기준 — cwd 무관) ----------

PHASE4_DIR = Path(__file__).resolve().parent
ROOT_DIR = PHASE4_DIR.parent
LOGS_DIR = PHASE4_DIR / "logs"
OBS_DUMP_REAL_DIR = LOGS_DIR / "obs_dump_real"  # deployment/Verify_ObsParity.py의 real측 입력
TRIP_LOG_PATH = LOGS_DIR / "trips.log"

# Phase2/Export_Policy.py가 확정한 flat release trio:
# exported/policy.pt + exported/obs_spec.json + exported/release_manifest.json.
EXPORTED_DIR = ROOT_DIR / "exported"
DEFAULT_POLICY = EXPORTED_DIR / "policy.pt"
DEFAULT_OBS_SPEC = EXPORTED_DIR / "obs_spec.json"


# ---------- 설정 로딩 (configs/*.yaml → 타입 있는 번들) ----------


@dataclass(frozen=True)
class DeployBundle:
    """배포 스크립트가 쓰는 설정 일괄 — 전부 configs/에서 파생 (단일 소스)."""

    common_yaml: dict
    real_yaml: dict
    safety_yaml: dict
    pre_cfg: PreprocessCfg
    action_cfg: ActionCfg
    safety_cfg: SafetyCfg
    control_hz: float
    physics_hz: float
    real_config_path: Path
    bundle_manifest: dict
    bundle_manifest_sha256: str
    release_manifest_sha256: str


def _load_manifested_yaml(manifest: dict, relative: str, *, label: str) -> dict:
    expected = manifest.get("files", {}).get(relative)
    if expected is None:
        raise ContractError(f"bundle manifest does not bind {relative}")
    try:
        value, _ = load_yaml_snapshot(
            ROOT_DIR / relative,
            expected_sha256=expected,
            label=label,
        )
    except FileSnapshotError as exc:
        raise ContractError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a mapping")
    return value


def load_deploy_bundle(real_config: Path | str | None) -> DeployBundle:
    """Load shared configs plus an explicitly selected local real config.

    변환은 전부 common.config_io 경유 — sim측과 같은 함수로 같은 yaml을 읽는다
    (필드명 매핑이 갈라지는 것을 방지하는 프로젝트 계약).
    """
    manifest = verify_bundle_manifest(ROOT_DIR)
    manifest_sha256 = getattr(manifest, "_verified_manifest_sha256", None)
    release_manifest_sha256 = getattr(
        manifest, "_verified_release_manifest_sha256", None
    )
    if re.fullmatch(r"[0-9a-f]{64}", str(manifest_sha256 or "")) is None:
        raise ContractError("bundle manifest verified-byte hash is unavailable")
    if re.fullmatch(r"[0-9a-f]{64}", str(release_manifest_sha256 or "")) is None:
        raise ContractError("embedded release manifest verified-byte hash is unavailable")
    common_yaml = _load_manifested_yaml(
        manifest, "configs/common.yaml", label="bundled common config"
    )
    exact_config = require_bundle_local_real_config(ROOT_DIR, real_config)
    real_yaml, real_config_path = load_real_config(exact_config)
    safety_yaml = _load_manifested_yaml(
        manifest, "configs/safety.yaml", label="bundled safety config"
    )
    return DeployBundle(
        common_yaml=common_yaml,
        real_yaml=real_yaml,
        safety_yaml=safety_yaml,
        pre_cfg=build_preprocess_cfg(common_yaml),
        action_cfg=build_action_cfg(common_yaml),
        safety_cfg=build_safety_cfg(safety_yaml),
        control_hz=float(common_yaml["control"]["control_hz"]),
        physics_hz=float(common_yaml["control"]["physics_hz"]),
        real_config_path=real_config_path,
        bundle_manifest=manifest,
        bundle_manifest_sha256=manifest_sha256,
        release_manifest_sha256=release_manifest_sha256,
    )


def validate_real_readiness(
    bundle: DeployBundle,
    *,
    release_dir: Path | str,
) -> dict[str, dict]:
    """Verify release hashes, calibration facts, and immutable evidence documents."""
    problems: list[str] = []
    expected_release = (ROOT_DIR / "exported").resolve()
    if Path(release_dir).resolve() != expected_release:
        problems.append(
            f"release_dir must be generated bundle exported/: {expected_release}"
        )
    if bundle.action_cfg.mode != "ik_rel":
        problems.append(f"action.mode={bundle.action_cfg.mode!r} (expected 'ik_rel')")
    robot = bundle.real_yaml["robot"]
    if not bool(robot.get("tcp_command_verified", False)):
        problems.append("robot.tcp_command_verified=false")
    latency = bundle.real_yaml.get("latency", {})
    dr_yaml = _load_manifested_yaml(
        bundle.bundle_manifest, "configs/dr.yaml", label="bundled DR config"
    )
    dr_latency = dr_yaml["latency"]
    period_ms = 1000.0 / bundle.control_hz
    for key, dr_key in (
        ("measured_obs_delay_ms", "obs_delay_steps"),
        ("measured_action_delay_ms", "action_delay_steps"),
    ):
        measured = latency.get(key)
        if measured is None:
            problems.append(f"latency.{key}=null")
            continue
        lo_steps, hi_steps = (int(v) for v in dr_latency[dr_key])
        if not (lo_steps * period_ms <= float(measured) <= hi_steps * period_ms):
            problems.append(
                f"latency.{key}={measured}ms outside DR "
                f"[{lo_steps * period_ms:.1f}, {hi_steps * period_ms:.1f}]ms"
            )
    gripper = bundle.real_yaml.get("gripper", {})
    # Any future released policy consumes gripper semantics, so driver/width
    # evidence remains mandatory; the removed --no-gripper bypass cannot weaken it.
    if not bool(gripper.get("driver_verified", False)):
        problems.append("gripper.driver_verified=false")
    if not bool(gripper.get("width_feedback_verified", False)):
        problems.append("gripper.width_feedback_verified=false")
    for filename, max_residual in (
        ("tcp_offset.yaml", 0.002),
        ("hand_eye.yaml", 0.020),
        ("scene_registration.yaml", 0.020),
    ):
        calibration = _load_manifested_yaml(
            bundle.bundle_manifest,
            f"configs/calib/{filename}",
            label=f"bundled calibration {filename}",
        )
        meta = calibration.get("meta", {})
        if not bool(meta.get("calibrated", False)):
            problems.append(f"calib/{filename}: meta.calibrated=false")
        residual = meta.get("residual_m")
        if residual is None or float(residual) > max_residual:
            problems.append(
                f"calib/{filename}: residual_m must be present and <= {max_residual:g}"
            )
    if problems:
        raise RuntimeError(
            "실기 배포 readiness gate 실패:\n  - "
            + "\n  - ".join(problems)
            + "\n측정/하드웨어 확인 없이 값을 추정해 true로 바꾸지 말 것."
        )
    try:
        release = verify_release(
            release_dir,
            config_root=ROOT_DIR,
            expected_obs_dim=obs_dim(bundle.pre_cfg),
            expected_action_dim=bundle.action_cfg.action_dim,
            expected_manifest_sha256=bundle.release_manifest_sha256,
        )
        verified_release_sha256 = getattr(release, "_verified_manifest_sha256", None)
        if verified_release_sha256 != bundle.release_manifest_sha256:
            raise ContractError("release manifest differs from verified bundle authority")
        bundle_files = bundle.bundle_manifest["files"]
        if release.get("files") != {
            "policy.pt": bundle_files.get("exported/policy.pt"),
            "obs_spec.json": bundle_files.get("exported/obs_spec.json"),
        }:
            raise ContractError("release policy/spec differ from verified bundle authority")
        return verify_evidence(
            bundle.real_yaml,
            bundle.real_config_path,
            common_cfg=bundle.common_yaml,
            dr_cfg=dr_yaml,
            release=release,
            release_manifest_sha256=bundle.release_manifest_sha256,
            bundle_manifest_sha256=bundle.bundle_manifest_sha256,
            allowed_evidence_root=ROOT_DIR / "deployment" / "evidence",
        )
    except ContractError as exc:
        raise RuntimeError(f"real readiness evidence failed: {exc}") from exc


# ---------- 인자 파싱/검증 ----------


def parse_vec3(text: str, name: str) -> np.ndarray:
    """'x,y,z' 문자열 → (3,) float64. 형식 오류 시 ValueError."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"{name} 은 'x,y,z' 형식이어야 함 (입력: {text!r})")
    return np.array([float(p) for p in parts], dtype=np.float64)


def validate_target_pos(target: np.ndarray, bundle: DeployBundle, tag: str) -> None:
    """--target-pos 안전/분포 검증.

    - workspace AABB(safety.yaml) 밖 → ValueError (실행 거부: 정책이 거기로 가려다 trip)
    - 학습 샘플 범위(common.yaml target.sample_range) 밖 → ValueError
    """
    lo = np.asarray(bundle.safety_cfg.workspace_min, dtype=np.float64)
    hi = np.asarray(bundle.safety_cfg.workspace_max, dtype=np.float64)
    if np.any(target < lo) or np.any(target > hi):
        raise ValueError(
            f"--target-pos {target.tolist()} 가 workspace AABB "
            f"{lo.tolist()}..{hi.tolist()} 밖 — safety.yaml 기준 실행 거부"
        )
    validate_target_sample_range(target, bundle.common_yaml, tag=tag)


# ---------- 제어주기 지터 계측 (RateLimiter 반환값 수집) ----------


class RecordingRateLimiter(RateLimiter):
    """common.timing.RateLimiter + sleep() 반환값(실측 주기[s]) 기록.

    run_rollout은 rate.sleep()의 반환값을 버리므로, 루프를 건드리지 않고
    지터를 수집하려면 이 서브클래스를 주입하는 것이 유일한 계약 준수 경로다.
    """

    def __init__(self, hz: float):
        super().__init__(hz)
        self.samples_s: list[float] = []

    def sleep(self) -> float:
        elapsed = super().sleep()
        self.samples_s.append(elapsed)
        return elapsed


def write_jitter_csv(path: Path | str, samples_s: list[float], hz: float) -> None:
    """스텝별 실측 주기 CSV — [step, period_ms, deviation_ms]. 히스토그램 원자료."""
    period_ms = 1000.0 / hz
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["step", "period_ms", "deviation_ms"])
        for i, s in enumerate(samples_s):
            ms = s * 1000.0
            w.writerow([i, round(ms, 3), round(ms - period_ms, 3)])


def summarize_jitter(samples_s: list[float], hz: float, tag: str) -> None:
    """지터 요약 + ASCII 히스토그램 콘솔 출력 (배포 PC OS 타이머 품질 실측 판단용)."""
    if not samples_s:
        print(f"[{tag}][warn] ⚠️ 지터 샘플 없음 (스텝 0회?)")
        return
    period_ms = 1000.0 / hz
    dev = np.array(samples_s, dtype=np.float64) * 1000.0 - period_ms
    p50, p95, p99 = np.percentile(dev, [50, 95, 99])
    print(f"[{tag}] 제어주기 지터 (목표 {period_ms:.1f}ms @ {hz:g}Hz, n={dev.size}):")
    print(
        f"[{tag}]   편차[ms] 평균 {dev.mean():+.2f} / p50 {p50:+.2f} / "
        f"p95 {p95:+.2f} / p99 {p99:+.2f} / 최대 {dev.max():+.2f}"
    )
    # 고정 경계 히스토그램 — Windows 타이머(~15.6ms 조각)면 +10ms 이상 구간이 두꺼워진다
    edges = [-20.0, -10.0, -5.0, -2.0, -1.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    bins = [(-np.inf, edges[0])] + list(zip(edges[:-1], edges[1:])) + [(edges[-1], np.inf)]
    counts = [int(np.sum((dev >= lo) & (dev < hi))) for lo, hi in bins]
    peak = max(counts) or 1
    for (lo, hi), c in zip(bins, counts):
        label = f"{lo:>5.0f}..{hi:<5.0f}" if np.isfinite(lo) and np.isfinite(hi) else (
            f"  <{hi:<5.0f}   " if not np.isfinite(lo) else f" >={lo:<5.0f}   "
        )
        bar = "#" * max(1 if c else 0, round(c / peak * 40))
        print(f"[{tag}]   {label} ms | {c:5d} | {bar}")
    if p95 > period_ms * 0.25:
        print(
            f"[{tag}][warn] ⚠️ p95 편차가 주기의 25% 초과 — README의 "
            f"'타이머 지터' 절 참조 (절전모드/타이머 해상도/우선순위 점검)"
        )
    print(
        f"[{tag}] → configs/real.local.yaml latency.measured_obs_delay_ms 실측 기록 후보: "
        f"p95 {p95 + period_ms:.1f}ms (전처리+추론 포함 루프 실측은 obs dump timestamp 참조)"
    )


# ---------- 관측/정책 출력 dump 통계 (run_rollout의 .npz 산출 후처리) ----------


def dump_stats(dump_dir: Path | str, pre_cfg: PreprocessCfg, action_cfg: ActionCfg) -> dict | None:
    """obs_dump 디렉토리의 step_*.npz → 정책 출력/관측 건강 통계.

    반환 dict:
      n_steps, action_mean/std/min/max(채널별), saturation_frac(|a|>=0.99),
      gripper_close_frac, depth_far_frac(결측→far 대체 비율 — real 센서 품질 지표)
    """
    dump_dir = Path(dump_dir)
    files = sorted(dump_dir.glob("step_*.npz"))
    if not files:
        return None
    actions, far_fracs = [], []
    n_depth = pre_cfg.out_height * pre_cfg.out_width
    for fp in files:
        with np.load(fp) as z:
            actions.append(np.asarray(z["action"], dtype=np.float64).reshape(-1))
            depth_flat = np.asarray(z["obs"], dtype=np.float64).reshape(-1)[:n_depth]
            far_fracs.append(float(np.mean(depth_flat >= 0.999)))  # 정규화 후 far=1.0
    a = np.stack(actions)  # [T, action_dim]
    return {
        "n_steps": int(a.shape[0]),
        "action_mean": a.mean(axis=0),
        "action_std": a.std(axis=0),
        "action_min": a.min(axis=0),
        "action_max": a.max(axis=0),
        "saturation_frac": np.mean(np.abs(a) >= 0.99, axis=0),
        # ★2026-07-09 부호 관례(action<0=close, ActionMapper.map과 동일 strict <)★
        #   — 이전 `>` 는 open 비율을 close로 잘못 보고했다 (2026-07-11 검토에서 정정)
        "gripper_close_frac": float(
            np.mean(a[:, -1] < action_cfg.gripper_close_threshold)
        ),
        "depth_far_frac": float(np.mean(far_fracs)),
    }


def print_dump_report(stats: dict, tag: str) -> None:
    """정책 출력 통계 리포트 — sim 리허설(Phase3) 출력과 나란히 비교하는 용도."""
    print(f"[{tag}] 정책 출력 통계 (n={stats['n_steps']} steps):")
    print(f"[{tag}]   ch |   mean |    std |    min |    max | sat(|a|>=0.99)")
    names = [f"j{i}" for i in range(len(stats["action_mean"]) - 1)] + ["grip"]
    for i, name in enumerate(names):
        print(
            f"[{tag}]   {name:>4s} | {stats['action_mean'][i]:+.3f} | "
            f"{stats['action_std'][i]:.3f} | {stats['action_min'][i]:+.3f} | "
            f"{stats['action_max'][i]:+.3f} | {stats['saturation_frac'][i]:.2%}"
        )
    print(f"[{tag}]   그리퍼 close 비율: {stats['gripper_close_frac']:.2%}")
    print(
        f"[{tag}]   depth 결측(far 대체) 비율: {stats['depth_far_frac']:.2%} "
        f"(높으면 카메라 노출/거리/반사 점검 — sim 대비 크게 다르면 ObsParity 실패 예고)"
    )
    sat_max = float(np.max(stats["saturation_frac"][:-1]))
    if sat_max > 0.5:
        print(
            f"[{tag}][warn] ⚠️ 관절 채널 포화율 {sat_max:.0%} — 관측 분포가 학습과 "
            f"다를 가능성 (Verify_ObsParity 먼저 통과할 것)"
        )


# ---------- real 어댑터 빌더 (lazy import — env_ur_real에서만 성립) ----------


def build_real_robot(real_yaml: dict, *, receive_only: bool = False) -> RobotInterface:
    """real/real_robot.py의 RealRobot 생성 (ur_rtde 연결 발생).

    생성자 계약(real/real_robot.py __init__): RealRobot(real_cfg dict, tcp_offset.yaml 경로)
    — tcp_offset은 요소4: RobotState.tcp_pose 보정에 즉시 사용된다
    (deployment/rtde_connect_test.py와 동일 호출 형태).
    receive_only=True이면 RTDEControl/RTDEIO를 만들지 않고 receive interface만 연결한다.
    """
    from real.real_robot import RealRobot  # env_ur_real에서만 import 성립

    return RealRobot(
        real_yaml,
        configs_dir() / "calib" / "tcp_offset.yaml",
        receive_only=receive_only,
    )


def build_real_camera(real_yaml: dict, enable_rgb: bool = False) -> CameraInterface:
    """real/real_camera.py의 RealCamera 생성 (pyrealsense2).

    생성자 계약(real/real_camera.py __init__): RealCamera(camera 섹션 dict, enable_rgb)
    — real.local.yaml 전체가 아니라 camera 섹션만 받는다
    (deployment/realsense_capture_test.py와 동일 호출 형태).
    ★enable_rgb는 반드시 bundle.pre_cfg.use_rgb를 넘길 것★ (common.yaml
    preprocess.use_rgb 단일 소스 — real.local.yaml 별도 스위치 금지, README 관측 계약).
    """
    from real.real_camera import RealCamera

    # ★depth-only 실기 배포 게이트(2026-07-17 적대 리뷰 수정)★: enable_rgb=False면
    #   real은 rs.align 없이 depth 스트림 원본(depth optical frame, vfov ~58°)을 쓰는데
    #   sim은 color 기준 42.5°(sim.yaml)로 렌더·학습된다 — 화각·기준 프레임이 조용히
    #   어긋난 채 obs_spec(4125)은 양쪽 일치라 통과해 버린다. depth-only(use_rgb=false)는
    #   ★sim 전용 ablation★이며, 실기 배포하려면 58° depth 계약(sim.yaml vfov 복원 +
    #   재학습)을 먼저 되살려야 한다 — 그 전까지 기동 거부가 정상 동작.
    if not enable_rgb:
        raise RuntimeError(
            "[deploy] ★기동 거부★ — use_rgb=false(depth-only)는 sim 전용 ablation 계약. "
            "현행 sim 카메라(42.5°, color 기준)와 real depth 원본(58°, depth 기준)의 "
            "화각·프레임이 불일치한다. RGB-D 계약(use_rgb=true)으로 export/배포하거나, "
            "depth-only 배포가 필요하면 sim.yaml vfov 58° 복원 + 재학습 후 이 게이트를 갱신할 것."
        )
    return RealCamera(real_yaml["camera"], enable_rgb=enable_rgb)


# ---------- 저속 램프 매퍼 (--speed-scale, 실행측 스텝당 이동 상한) ----------


class SpeedLimitedMapper(ActionMapper):
    """실행측 저속 램프 — 현행 ik_rel 명령은 task-space 스텝 상한으로 clamp.

    구 joint_absolute 호환 fixture에서는 action_scale_rad를 줄이면 '감속'이
    아니라 목표 자세를 home 쪽으로 압축한다. 그 legacy branch는 원 목표를
    유지하고 스텝당 |q_target − q_cur| ≤ max_step_rad만 제한한다.
    max_step_rad = speed_scale × safety.joint_vel_max_rad_s × dt 로 호출측이 파생.
    SafetyGuard.clamp_joint_target(배율 1.0)는 run_rollout에서 그대로 한 번 더
    걸린다(안전 계약 불변). RecordingRateLimiter와 같은 '서브클래스 주입' 관례 —
    run_rollout 루프는 무수정.
    """

    def __init__(
        self,
        cfg: ActionCfg,
        max_step_rad: float,
        max_task_step_m: float | None = None,
        max_task_step_rad: float | None = None,
    ):
        super().__init__(cfg)
        if max_step_rad <= 0.0:
            raise ValueError(f"max_step_rad 는 양수여야 함: {max_step_rad}")
        self.max_step_rad = float(max_step_rad)
        self.max_task_step_m = float(
            max_step_rad if max_task_step_m is None else max_task_step_m
        )
        self.max_task_step_rad = float(
            max_step_rad if max_task_step_rad is None else max_task_step_rad
        )
        if self.max_task_step_m <= 0.0 or self.max_task_step_rad <= 0.0:
            raise ValueError("task-space step limits must be positive")

    def map(self, action: np.ndarray, q_cur: np.ndarray) -> tuple[np.ndarray, bool]:
        q_target, gripper_closed = super().map(action, q_cur)
        if self.cfg.mode == "ik_rel":
            delta = q_target.copy()
            for sl, limit in (
                (slice(0, 3), self.max_task_step_m),
                (slice(3, 6), self.max_task_step_rad),
            ):
                norm = float(np.linalg.norm(delta[sl]))
                if norm > limit:
                    delta[sl] *= limit / norm
            return delta, gripper_closed
        q_cur = np.asarray(q_cur, dtype=np.float64).reshape(-1)
        delta = np.clip(q_target - q_cur, -self.max_step_rad, self.max_step_rad)
        return q_cur + delta, gripper_closed


# ---------- trip 사유 로그 ----------


def append_trip_log(reason: str, context: str) -> None:
    """Phase4/logs/trips.log에 타임스탬프 부착 append (실기 사고 이력의 단일 파일)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(TRIP_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{stamp} | {context} | {reason}\n")


def timestamp() -> str:
    """파일명용 시각 문자열 (reference 프로젝트와 동일 포맷)."""
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
