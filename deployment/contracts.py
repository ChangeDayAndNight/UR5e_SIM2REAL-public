"""Fail-closed real-machine configuration and evidence contracts.

This module deliberately has no Isaac, RTDE, or RealSense imports.  Hardware
scripts load it before opening a socket so a missing, stale, or mismatched
piece of evidence cannot be bypassed by flipping a boolean in YAML.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping

from common.config_io import load_yaml_snapshot
from common.file_snapshot import (
    FileSnapshotError,
    RegularFileSnapshot,
    read_regular_file_snapshot,
)

EVIDENCE_SCHEMA_VERSION = 1
BASE_EVIDENCE_KINDS = (
    "tcp_command",
    "tcp_offset",
    "hand_eye",
    "scene_registration",
    "camera_stream",
    "joint_parity",
    "latency",
    "force_feedback",
    "emergency_stop",
    "workspace",
    "motion_safety",
    "no_motion",
    "firmware_inventory",
)
GRIPPER_EVIDENCE_KINDS = ("gripper_driver", "gripper_width")

# Release-time calibration approval thresholds.  The evidence validators and
# the tracked calibration-YAML gate intentionally share these exact constants
# so a hashed config cannot use a weaker interpretation than deployment
# readiness.
TCP_OFFSET_MAX_RESIDUAL_M = 0.002
HAND_EYE_MAX_MEDIAN_ERROR_M = 0.010
HAND_EYE_MAX_P95_ERROR_M = 0.020
SCENE_REGISTRATION_MAX_RESIDUAL_M = 0.020

_RELEASE_CALIBRATION_PATHS = {
    "tcp_offset": "configs/calib/tcp_offset.yaml",
    "hand_eye": "configs/calib/hand_eye.yaml",
    "scene_registration": "configs/calib/scene_registration.yaml",
    "nurec_landmarks": "configs/calib/nurec_landmarks.yaml",
    "tool_chain": "configs/calib/tool_chain.yaml",
}


class ContractError(RuntimeError):
    """A deployment contract is incomplete or inconsistent."""


def is_link_or_reparse(path: Path | str) -> bool:
    """Return true for POSIX links and Windows reparse points/junctions."""
    try:
        info = Path(path).lstat()
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return stat.S_ISLNK(info.st_mode) or bool(reparse and attributes & reparse)


def path_contains_link_or_reparse(path: Path | str) -> bool:
    raw = Path(path).expanduser().absolute()
    return any(is_link_or_reparse(item) for item in (raw, *raw.parents))


# ── 모션 인가 (F74) ──────────────────────────────────────────────────────────
#   [이전 상태] `require_motion_authorization()`이 조건 없이 예외를 던져 ★모든★ 모션이
#     막혀 있었다. 그 결과 위험도가 전혀 다른 세 가지가 하나의 스위치에 묶였다:
#       · freedrive     — 사람이 손으로 로봇을 민다 (명령 없음)
#       · scripted_pose — 사전 정의된 관절 자세로 저속 이동
#       · policy        — 정책이 실시간 생성한 목표를 servoJ로 스트리밍
#     가장 안전한 freedrive조차 못 켜서 ★캘리브레이션을 못 하고★, 캘리브레이션이 없으니
#     좌표가 안 맞아 모션이 더 위험해지는 교착이 생겼다.
#   [현행] 인가를 없애지 않는다. "구현 안 됨(영구 거부)"을 "범위·기한·대상이 명시된
#     인가"로 바꾼다. ★인가 파일이 없으면 이전과 똑같이 거부된다★ — 기본값은 안전이다.
#   [의도적 설계] 인가 파일은 git 추적 대상이 아니다(.gitignore). 현장의 1회성 결정이
#     저장소에 커밋되면 "승인된 커미셔닝 절차"로 오인된다 — 원본 계약이 경계한 함정이다.
MOTION_SCOPES = ("freedrive", "scripted_pose", "gripper", "policy_dryrun", "policy")

# 캘리브레이션 완료를 추가로 요구하는 범위 — 좌표 미검증 상태에서 정책 궤적 금지.
_SCOPES_REQUIRING_CALIBRATION = ("policy",)
_MAX_AUTHORIZATION_WINDOW_S = 72 * 3600
_REQUIRED_ACKNOWLEDGEMENTS = 3


def _authorization_path(project_root: Path | str | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    return root / "configs" / "motion_authorization.yaml"


def load_motion_authorization(project_root: Path | str | None = None) -> dict:
    """인가 파일을 읽어 구조를 검증한다. 없거나 깨졌으면 ContractError."""
    import datetime as _dt

    path = _authorization_path(project_root)
    if path_contains_link_or_reparse(path.absolute()):
        raise ContractError("motion authorization path traverses a symlink or reparse point")
    if not path.is_file():
        raise ContractError(
            f"motion authorization absent: {path.name} not found — motion stays blocked"
        )
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - 어떤 파싱 실패든 차단으로 강등
        raise ContractError(f"motion authorization is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ContractError("motion authorization has a wrong or missing schema_version")
    if not str(payload.get("operator", "")).strip():
        raise ContractError("motion authorization requires a named operator (입회자)")

    acknowledgements = payload.get("acknowledgements")
    if not isinstance(acknowledgements, list) or len(acknowledgements) < _REQUIRED_ACKNOWLEDGEMENTS:
        raise ContractError(
            f"motion authorization requires {_REQUIRED_ACKNOWLEDGEMENTS} explicit acknowledgements"
        )

    try:
        acknowledged = _dt.datetime.fromisoformat(str(payload["acknowledged_at"]))
        expires = _dt.datetime.fromisoformat(str(payload["expires_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("motion authorization timestamps are missing or unparsable") from exc
    if acknowledged.tzinfo is None or expires.tzinfo is None:
        raise ContractError("motion authorization timestamps must carry an explicit timezone")
    now = _dt.datetime.now(_dt.timezone.utc)
    if expires <= now:
        raise ContractError(f"motion authorization expired at {expires.isoformat()}")
    if (expires - acknowledged).total_seconds() > _MAX_AUTHORIZATION_WINDOW_S:
        raise ContractError("motion authorization window exceeds 72h — re-acknowledge on site")

    scopes = payload.get("scopes")
    if not isinstance(scopes, list) or not scopes:
        raise ContractError("motion authorization lists no scopes")
    unknown = [scope for scope in scopes if scope not in MOTION_SCOPES]
    if unknown:
        raise ContractError(f"motion authorization names unknown scopes: {unknown}")
    return payload


def require_motion_authorization(
    scope: str = "policy", *, project_root=None, operator_ack: bool = False
) -> None:
    """`scope`의 모션이 인가되었는지 확인한다 — 아니면 ContractError로 차단한다.

    호출측은 ★자기가 실제로 필요한 최소 범위★를 선언한다. 사람이 손으로 미는
    freedrive 캘리브레이션이 정책 servoJ와 같은 열쇠를 쓰지 않도록 하는 것이 요지다.

    ★operator_ack — 현장 입회 경로 (2026-08-30)★
      인가 ★파일★은 무인·원격 실행처럼 "그 자리에 사람이 없을 수 있는" 경우를 위한
      장치다. 사람이 로봇 앞에 서서 e-stop에 손이 닿는 상태로 직접 명령을 타이핑하는
      감독 하 실험에서는, 파일·만료·캘리브레이션 게이트가 ★안전을 더해 주지 않으면서
      마찰만 만든다★ — 실제로 사고를 막는 것은 SafetyGuard(작업영역 AABB·TCP z 바닥·
      관절속도 상한)와 사람의 손이다.
      그래서 호출측이 ★명시적 인자★로 입회를 선언하면 파일 검사를 건너뛴다. 이때도
      핵심 성질은 그대로다: ★모션은 결코 우연히 일어나지 않는다★ — 누군가 플래그를
      직접 타이핑해야만 이 경로가 열린다.
    """
    if scope not in MOTION_SCOPES:
        raise ContractError(f"unknown motion scope {scope!r}; expected one of {MOTION_SCOPES}")
    if operator_ack:
        print("=" * 72)
        print(f" ★모션 인가: 현장 입회 (scope={scope})★")
        print(" 인가 파일 대신 조작자가 직접 선언했다. SafetyGuard는 그대로 작동한다.")
        print(" e-stop에 손이 닿는 위치에 있어야 한다.")
        print("=" * 72)
        return
    payload = load_motion_authorization(project_root)
    if scope not in payload["scopes"]:
        raise ContractError(
            f"motion scope {scope!r} is not authorized (authorized: {payload['scopes']})"
        )

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    real_config = root / "configs" / "real.local.yaml"
    expected = payload.get("real_config_sha256")
    if expected is not None:
        if not real_config.is_file():
            raise ContractError("motion authorization pins real.local.yaml but the file is absent")
        if sha256_file(real_config) != expected:
            raise ContractError(
                "real.local.yaml changed since authorization — re-acknowledge before motion"
            )

    if scope in _SCOPES_REQUIRING_CALIBRATION:
        for name in ("tcp_offset.yaml", "hand_eye.yaml"):
            calib_path = root / "configs" / "calib" / name
            try:
                import yaml

                meta = (yaml.safe_load(calib_path.read_text(encoding="utf-8")) or {}).get("meta", {})
            except Exception as exc:  # noqa: BLE001
                raise ContractError(f"cannot read calibration {name}: {exc}") from exc
            if not bool(meta.get("calibrated")):
                raise ContractError(
                    f"scope {scope!r} requires calibrated {name} (meta.calibrated is not true) — "
                    "run the calibration gates first"
                )


def require_policy_dryrun_authorization(*, project_root=None) -> None:
    """정책 dry-run(명령 미전송) 인가 — 모션 인가와 같은 기전, 범위만 다르다."""
    require_motion_authorization("policy_dryrun", project_root=project_root)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def real_contract_sha256(real_cfg: dict) -> str:
    """Hash motion-affecting config while excluding evidence paths/hashes.

    Excluding only the top-level ``evidence`` mapping avoids a circular hash:
    each evidence document binds to this contract hash, while real.local.yaml
    can bind to the evidence document's full-file hash.
    """
    payload = copy.deepcopy(real_cfg)
    payload.pop("evidence", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def load_real_config_snapshot(
    path: Path | str | None,
    *,
    require_identity: bool = True,
) -> tuple[dict, Path, RegularFileSnapshot]:
    """Validate a real config and return the exact YAML bytes it consumed."""
    if path is None or not str(path).strip():
        raise ContractError("--real-config is required; no repository default is permitted")
    config_raw = Path(path).expanduser().absolute()
    if path_contains_link_or_reparse(config_raw):
        raise ContractError("real config path must not contain symlinks/reparse points")
    config_path = config_raw.resolve()
    if not config_path.is_file():
        raise ContractError(f"real config does not exist: {config_path}")
    if config_path.name == "real.example.yaml":
        raise ContractError("configs/real.example.yaml is a fail-closed template; copy it to real.local.yaml")
    try:
        cfg, snapshot = load_yaml_snapshot(
            config_path, label="real config"
        )
    except FileSnapshotError as exc:
        raise ContractError(str(exc)) from exc
    if not isinstance(cfg, dict):
        raise ContractError("real config must contain a YAML mapping")
    for section in ("robot", "gripper", "camera", "latency", "evidence"):
        if not isinstance(cfg.get(section), dict):
            raise ContractError(f"real config missing mapping: {section}")
    if require_identity:
        ip = str(cfg["robot"].get("ip", "")).strip()
        serial = str(cfg["camera"].get("serial", "")).strip()
        if not ip:
            raise ContractError("robot.ip is empty")
        try:
            parsed = ipaddress.ip_address(ip)
            if parsed.is_unspecified or parsed.is_loopback or parsed.is_multicast:
                raise ValueError
        except ValueError as exc:
            raise ContractError(f"robot.ip must be a non-loopback IP address: {ip!r}") from exc
        if not serial:
            raise ContractError("camera.serial is required; first-device auto-selection is forbidden")
        if not str(cfg["robot"].get("controller_version", "")).strip():
            raise ContractError("robot.controller_version is required")
        if not str(cfg["camera"].get("firmware_version", "")).strip():
            raise ContractError("camera.firmware_version is required")
        if cfg["camera"].get("global_time_enabled") is not True:
            raise ContractError("camera.global_time_enabled must be explicitly true")
        if cfg["camera"].get("frames_queue_size") != 1:
            raise ContractError("camera.frames_queue_size must be exactly 1")
    return cfg, config_path, snapshot


def load_real_config(path: Path | str | None, *, require_identity: bool = True) -> tuple[dict, Path]:
    """Backward-compatible real-config loader without discarding validation."""
    cfg, config_path, _ = load_real_config_snapshot(
        path, require_identity=require_identity
    )
    return cfg, config_path


def require_bundle_local_real_config(bundle_root: Path | str, path: Path | str | None) -> Path:
    """Require the bundle's sole mutable local config; reject links/alternate paths."""
    root_raw = Path(bundle_root).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw):
        raise ContractError("bundle root must not contain symlinks/reparse points")
    root = root_raw.resolve()
    expected_raw = root / "configs" / "real.local.yaml"
    if path is None or not str(path).strip():
        raise ContractError("--real-config must be bundle-root/configs/real.local.yaml")
    supplied = Path(path).expanduser().absolute()
    if path_contains_link_or_reparse(supplied) or path_contains_link_or_reparse(expected_raw):
        raise ContractError("configs/real.local.yaml must not contain a symlink/reparse point")
    if supplied.resolve() != expected_raw.resolve():
        raise ContractError(
            f"--real-config must resolve exactly to bundle config: {expected_raw.resolve()}"
        )
    return expected_raw.resolve()


def _require_evidence_path(path: Path, allowed_root: Path | str) -> Path:
    root_raw = Path(allowed_root).absolute()
    path_raw = Path(path).absolute()
    if path_contains_link_or_reparse(root_raw) or path_contains_link_or_reparse(path_raw):
        raise ContractError("evidence path must not contain a symlink/reparse point")
    root = root_raw.resolve()
    resolved = path_raw.resolve()
    if resolved.parent != root and root not in resolved.parents:
        raise ContractError(f"evidence path must stay under bundle evidence root: {root}")
    cursor = path_raw.parent
    while cursor != root_raw.parent:
        if is_link_or_reparse(cursor):
            raise ContractError(f"evidence path contains a symlink/reparse point: {cursor}")
        if cursor == root_raw:
            break
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return resolved


def _parse_time(value: Any, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"evidence {field} is required")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"invalid evidence {field}: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"evidence {field} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _number(m: dict, key: str) -> float:
    value = m.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"evidence measurement {key!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"evidence measurement {key!r} must be finite")
    return number


def _truth(m: dict, key: str) -> None:
    if m.get(key) is not True:
        raise ContractError(f"evidence measurement {key!r} must be true")


def _validate_tcp_command(m: dict, _: dict, __: dict) -> None:
    for key in ("controller_tcp_zero", "base_frame_roundtrip_passed", "servol_signature_verified"):
        _truth(m, key)


def _validate_tcp_offset(m: dict, _: dict, __: dict) -> None:
    _truth(m, "calibrated")
    residual = _number(m, "residual_m")
    if residual < 0 or residual > TCP_OFFSET_MAX_RESIDUAL_M:
        raise ContractError(
            "tcp_offset residual_m must be non-negative and no greater than "
            f"{TCP_OFFSET_MAX_RESIDUAL_M:.3f} m"
        )


def _validate_hand_eye(m: dict, _: dict, __: dict) -> None:
    _truth(m, "post_correction_verified")
    median = _number(m, "median_error_m")
    p95 = _number(m, "p95_error_m")
    if (
        median < 0
        or p95 < median
        or median > HAND_EYE_MAX_MEDIAN_ERROR_M
        or p95 > HAND_EYE_MAX_P95_ERROR_M
    ):
        raise ContractError(
            "hand_eye errors must satisfy 0 <= median <= 10 mm and median <= p95 <= 20 mm"
        )


def _validate_scene(m: dict, _: dict, __: dict) -> None:
    residual = _number(m, "residual_m")
    if residual < 0 or residual > SCENE_REGISTRATION_MAX_RESIDUAL_M:
        raise ContractError(
            "scene_registration residual_m must be non-negative and no greater than "
            f"{SCENE_REGISTRATION_MAX_RESIDUAL_M:.3f} m"
        )


def _validate_camera(m: dict, real: dict, _: dict) -> None:
    _truth(m, "rgb_depth_aligned")
    _truth(m, "timestamps_monotonic")
    if str(m.get("serial", "")) != str(real["camera"]["serial"]):
        raise ContractError("camera evidence serial does not match real config")
    if _number(m, "stale_frames") != 0 or _number(m, "disconnects") != 0:
        raise ContractError("camera evidence contains stale frames or disconnects")


def _validate_joint(m: dict, _: dict, __: dict) -> None:
    if _number(m, "max_slope_error_fraction") > 0.05:
        raise ContractError("joint parity slope error exceeds 5%")
    if _number(m, "max_offset_rad") > 0.01:
        raise ContractError("joint parity offset exceeds 0.01 rad")
    if _number(m, "max_cross_axis_rad") > 0.0174533:
        raise ContractError("joint parity cross-axis motion exceeds 1 degree")


def _validate_latency(m: dict, real: dict, ctx: dict) -> None:
    obs = _number(m, "obs_delay_ms")
    action = _number(m, "action_delay_ms")
    configured_obs = real["latency"].get("measured_obs_delay_ms")
    configured_action = real["latency"].get("measured_action_delay_ms")
    if not isinstance(configured_obs, (int, float)) or isinstance(configured_obs, bool):
        raise ContractError("latency.measured_obs_delay_ms must be numeric")
    if not isinstance(configured_action, (int, float)) or isinstance(configured_action, bool):
        raise ContractError("latency.measured_action_delay_ms must be numeric")
    if obs != float(configured_obs):
        raise ContractError("latency evidence obs_delay_ms differs from real config")
    if action != float(configured_action):
        raise ContractError("latency evidence action_delay_ms differs from real config")
    period_ms = 1000.0 / float(ctx["common"]["control"]["control_hz"])
    dr = ctx["dr"]["latency"]
    for value, key in ((obs, "obs_delay_steps"), (action, "action_delay_steps")):
        lo, hi = (float(v) * period_ms for v in dr[key])
        if not lo <= value <= hi:
            raise ContractError(f"{key} evidence {value} ms is outside DR [{lo}, {hi}] ms")
    if _number(m, "stale_samples") != 0 or _number(m, "max_consecutive_overruns") != 0:
        raise ContractError("latency evidence has stale samples or consecutive overruns")
    if _number(m, "overrun_fraction") > 0.01:
        raise ContractError("latency evidence overrun_fraction exceeds 1%")
    _validate_release_binding(m, ctx, "latency")


def _validate_force(m: dict, _: dict, __: dict) -> None:
    _truth(m, "available")
    _truth(m, "frame_verified")
    if _number(m, "read_failures") != 0:
        raise ContractError("force feedback evidence contains read failures")


def _validate_estop(m: dict, _: dict, ctx: dict) -> None:
    for key in ("latched", "no_post_stop_commands", "camera_disconnect_tested", "rtde_disconnect_tested"):
        _truth(m, key)
    if _number(m, "stop_time_s") <= 0 or _number(m, "stop_distance_m") < 0:
        raise ContractError("invalid emergency-stop time/distance evidence")
    _validate_release_binding(m, ctx, "emergency-stop")


def _validate_workspace(m: dict, _: dict, __: dict) -> None:
    for key in ("cell_measured", "home_path_collision_checked", "hardware_estop_accessible"):
        _truth(m, key)


def _validate_release_binding(m: dict, ctx: dict, label: str) -> None:
    release = ctx.get("release")
    bundle_hash = ctx.get("bundle_manifest_sha256")
    release_hash = ctx.get("release_manifest_sha256")
    if not isinstance(release, dict) or not bundle_hash or not release_hash:
        raise ContractError(f"{label} current release/bundle binding context is missing")
    expected = {
        "release_checkpoint_sha256": release["source"]["checkpoint_sha256"],
        "release_manifest_sha256": release_hash,
        "policy_sha256": release["files"]["policy.pt"],
        "obs_spec_sha256": release["files"]["obs_spec.json"],
        "bundle_manifest_sha256": bundle_hash,
    }
    for key, value in expected.items():
        if m.get(key) != value:
            raise ContractError(f"{label} {key} does not match the current bundle/release")


def _validate_motion_safety(m: dict, _: dict, ctx: dict) -> None:
    for key in (
        "joint_position_limits_verified",
        "singularity_manipulability_verified",
        "self_collision_verified",
        "cell_collision_verified",
        "tcp_accel_jerk_torque_verified",
        "command_ack_verified",
        "low_speed_ladder_verified",
        "stop_distance_verified",
        "fault_injection_verified",
        "no_post_fault_commands",
    ):
        _truth(m, key)
    if _number(m, "minimum_joint_limit_margin_rad") <= 0:
        raise ContractError("motion safety joint-limit margin must be positive")
    if _number(m, "minimum_manipulability") <= 0:
        raise ContractError("motion safety manipulability margin must be positive")
    for measured_key, limit_key in (
        ("max_tcp_accel_m_s2", "tcp_accel_limit_m_s2"),
        ("max_tcp_jerk_m_s3", "tcp_jerk_limit_m_s3"),
        ("max_stop_distance_m", "stop_distance_limit_m"),
        ("max_stop_time_s", "stop_time_limit_s"),
    ):
        measured = _number(m, measured_key)
        limit = _number(m, limit_key)
        if measured < 0 or limit <= 0 or measured > limit:
            raise ContractError(f"motion safety {measured_key} exceeds its positive approved limit")
    torque_ratio = _number(m, "max_joint_torque_ratio")
    if torque_ratio < 0 or torque_ratio > 1:
        raise ContractError("motion safety joint torque exceeded the approved limit")
    if _number(m, "command_ack_failures") != 0:
        raise ContractError("motion safety command acknowledgement failures must be zero")
    levels = _number(m, "low_speed_levels_tested")
    max_speed_scale = _number(m, "max_tested_speed_scale")
    if not levels.is_integer() or levels < 3 or max_speed_scale <= 0 or max_speed_scale > 0.25:
        raise ContractError("motion safety requires a >=3-level ladder capped at 0.25 speed scale")
    fault_cases = _number(m, "fault_cases_tested")
    if not fault_cases.is_integer() or fault_cases < 5 or _number(m, "post_fault_commands") != 0:
        raise ContractError("motion safety requires >=5 fault cases and zero post-fault commands")
    _validate_release_binding(m, ctx, "motion safety")


def _validate_no_motion(m: dict, real: dict, ctx: dict) -> None:
    for key in ("release_verified", "dependencies_verified", "offline_tests_passed"):
        _truth(m, key)
    for key in (
        "hardware_receive_only",
        "camera_sequence_monotonic",
        "source_timestamps_verified",
        "source_timestamp_monotonic",
        "source_cadence_verified",
    ):
        _truth(m, key)
    if m.get("command_methods_exposed") is not False:
        raise ContractError("no-motion adapter must not expose command methods")
    if _number(m, "duration_s") < 600 or _number(m, "samples") < 10_000:
        raise ContractError("no-motion evidence requires >=600 seconds and >=10,000 samples")
    for key in ("read_failures", "disconnects", "nonfinite_samples", "stale_samples"):
        if _number(m, key) != 0:
            raise ContractError(f"no-motion evidence {key} must be zero")
    if _number(m, "overrun_fraction") >= 0.01 or _number(m, "max_consecutive_overruns") > 0:
        raise ContractError("no-motion timing requires <1% overruns and zero consecutive overruns")
    if _number(m, "sensor_age_p99_s") > 0.05 or _number(m, "sensor_skew_p99_s") > 0.05:
        raise ContractError("no-motion sensor age/skew p99 exceeds 50 ms")
    if _number(m, "catch_up_bursts") != 0:
        raise ContractError("no-motion evidence contains catch-up bursts")
    skipped = _number(m, "skipped_periods")
    overruns = _number(m, "overruns")
    if skipped < 0 or not skipped.is_integer() or skipped > overruns:
        raise ContractError("no-motion skipped-period accounting is invalid")
    domains = m.get("source_timestamp_domains")
    if not isinstance(domains, dict) or domains.get("robot") != "ur_rtde_controller_clock":
        raise ContractError("no-motion robot timestamp must come from the RTDE controller clock")
    if domains.get("camera") not in {"realsense_global_time", "realsense_system_time"}:
        raise ContractError("no-motion camera timestamp must come from RealSense frame metadata")
    alignment = m.get("source_clock_alignment")
    if not isinstance(alignment, dict) or any(
        not isinstance(alignment.get(source), dict)
        or alignment[source].get("mapping") != "none_raw_clock_only"
        for source in ("robot", "camera")
    ):
        raise ContractError("no-motion source clock alignment provenance is missing")
    absolute = m.get("absolute_time_validation")
    if (
        not isinstance(absolute, dict)
        or absolute.get("method")
        != "absolute_camera_epoch_and_fresh_rtde_arrival_bounds"
        or absolute.get("robot_freshness")
        != "new_package_after_request_same_timestamp_bracket"
    ):
        raise ContractError("no-motion absolute age/skew validation provenance is missing")
    if str(m.get("camera_serial", "")) != str(real["camera"]["serial"]):
        raise ContractError("no-motion camera serial differs from real config")
    _validate_release_binding(m, ctx, "no-motion")
    # This release can collect diagnostics but cannot bind the UR controller's
    # boot-relative clock to the RealSense acquisition clock.  Keep readiness
    # closed even for a syntactically perfect hand-written document.
    raise ContractError(
        "no-motion evidence issuance is blocked: timestamp_sync_unverified"
    )


def _validate_firmware(m: dict, real: dict, __: dict) -> None:
    if str(m.get("controller_version", "")) != str(real["robot"]["controller_version"]):
        raise ContractError("controller firmware/version differs from real config")
    if str(m.get("camera_firmware_version", "")) != str(real["camera"]["firmware_version"]):
        raise ContractError("camera firmware differs from real config")
    if str(m.get("camera_serial", "")) != str(real["camera"]["serial"]):
        raise ContractError("firmware evidence camera serial differs from real config")


def _validate_gripper_driver(m: dict, _: dict, __: dict) -> None:
    for key in ("driver_path_verified", "pulse_cancel_verified", "safe_outputs_verified"):
        _truth(m, key)


def _validate_gripper_width(m: dict, _: dict, __: dict) -> None:
    _truth(m, "policy_observation_semantics_match")
    _truth(m, "empty_close_vs_grasp_distinguishable")


_VALIDATORS: dict[str, Callable[[dict, dict, dict], None]] = {
    "tcp_command": _validate_tcp_command,
    "tcp_offset": _validate_tcp_offset,
    "hand_eye": _validate_hand_eye,
    "scene_registration": _validate_scene,
    "camera_stream": _validate_camera,
    "joint_parity": _validate_joint,
    "latency": _validate_latency,
    "force_feedback": _validate_force,
    "emergency_stop": _validate_estop,
    "workspace": _validate_workspace,
    "motion_safety": _validate_motion_safety,
    "no_motion": _validate_no_motion,
    "firmware_inventory": _validate_firmware,
    "gripper_driver": _validate_gripper_driver,
    "gripper_width": _validate_gripper_width,
}


def _release_calibration_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"release calibration {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ContractError(f"release calibration {label} must be {qualifier}")
    return result


def _release_calibration_meta(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ContractError(f"release calibration {label}.meta must be a mapping")
    if value.get("calibrated") is not True:
        raise ContractError(f"release calibration {label} is not calibrated")
    for key in ("method", "date"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ContractError(f"release calibration {label}.meta.{key} is required")
    try:
        dt.date.fromisoformat(value["date"])
    except ValueError as exc:
        raise ContractError(
            f"release calibration {label}.meta.date must be ISO YYYY-MM-DD"
        ) from exc
    return value


def _release_rigid_transform(value: object, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4 \
            or any(not isinstance(row, list) or len(row) != 4 for row in value):
        raise ContractError(f"release calibration {label} must be a 4x4 matrix")
    matrix = [
        [_release_calibration_number(item, f"{label}[{i}][{j}]") for j, item in enumerate(row)]
        for i, row in enumerate(value)
    ]
    if any(abs(matrix[3][j] - expected) > 1.0e-9
           for j, expected in enumerate((0.0, 0.0, 0.0, 1.0))):
        raise ContractError(f"release calibration {label} is not homogeneous")
    rotation = [row[:3] for row in matrix[:3]]
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            expected = 1.0 if i == j else 0.0
            if abs(dot - expected) > 1.0e-6:
                raise ContractError(f"release calibration {label} rotation is not orthonormal")
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1.0e-6:
        raise ContractError(f"release calibration {label} rotation determinant is not +1")
    return matrix


def _release_landmark_vector(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ContractError(f"release calibration {label} must be a 3-vector")
    return tuple(
        _release_calibration_number(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _landmarks_are_noncollinear(points: list[tuple[float, float, float]]) -> bool:
    if len(points) < 3:
        return False
    origin = points[0]
    vectors = [tuple(point[i] - origin[i] for i in range(3)) for point in points[1:]]
    for i, left in enumerate(vectors):
        for right in vectors[i + 1:]:
            cross = (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )
            if sum(component * component for component in cross) > 1.0e-16:
                return True
    return False


def _load_release_calibration(
    root: Path, relative: str, *, expected_sha256: str | None = None
) -> tuple[dict, str]:
    raw = root / relative
    if path_contains_link_or_reparse(raw):
        raise ContractError(f"release calibration path contains a symlink/reparse point: {relative}")
    path = raw.resolve()
    if root not in path.parents or not path.is_file():
        raise ContractError(f"release calibration file is missing or escapes root: {relative}")
    try:
        value, snapshot = load_yaml_snapshot(
            path,
            expected_sha256=expected_sha256,
            label=f"release calibration {relative}",
        )
    except (FileSnapshotError, OSError, ValueError) as exc:
        raise ContractError(f"cannot load release calibration {relative}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"release calibration {relative} must contain a mapping")
    return value, snapshot.sha256


def validate_release_calibrations(
    config_root: Path | str,
    *,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict:
    """Require canonical, measured calibration semantics before final release.

    Artifact hashes bind bytes but cannot turn ``calibrated:false``, a null
    residual, or synthetic landmarks into approval.  Candidate export remains
    intentionally outside this gate; every final release creation/verification
    and therefore bundle/offline verification calls this function.
    """
    root_raw = Path(config_root).expanduser().absolute()
    if path_contains_link_or_reparse(root_raw):
        raise ContractError("release calibration root must not contain symlinks/reparse points")
    root = root_raw.resolve()
    if not root.is_dir():
        raise ContractError(f"release calibration root is not a directory: {root}")

    if expected_hashes is not None and not isinstance(expected_hashes, Mapping):
        raise ContractError("release calibration expected hashes must be a mapping")
    configs: dict[str, dict] = {}
    consumed_hashes: dict[str, str] = {}
    for name, relative in _RELEASE_CALIBRATION_PATHS.items():
        expected = expected_hashes.get(relative) if expected_hashes is not None else None
        if expected_hashes is not None and re.fullmatch(
            r"[0-9a-f]{64}", str(expected or "")
        ) is None:
            raise ContractError(
                f"release calibration expected hash is missing/invalid: {relative}"
            )
        value, digest = _load_release_calibration(
            root, relative, expected_sha256=expected
        )
        configs[name] = value
        consumed_hashes[relative] = digest

    tcp = configs["tcp_offset"]
    tcp_matrix = _release_rigid_transform(tcp.get("T_flange_tcp"), "tcp_offset.T_flange_tcp")
    tcp_meta = _release_calibration_meta(tcp.get("meta"), "tcp_offset")
    if tcp_meta.get("method") != "fixed_point_multi_orientation":
        raise ContractError("release calibration tcp_offset method is not the approved measured method")
    if tcp_meta.get("gripper_model") != "onrobot_2fg7":
        raise ContractError("release calibration tcp_offset gripper_model must be onrobot_2fg7")
    _validate_tcp_offset(tcp_meta, {}, {})

    hand_eye = configs["hand_eye"]
    _release_rigid_transform(hand_eye.get("T_flange_cam"), "hand_eye.T_flange_cam")
    hand_meta = _release_calibration_meta(hand_eye.get("meta"), "hand_eye")
    if hand_meta.get("mount") != "eye_in_hand" \
            or not str(hand_meta.get("method", "")).startswith("opencv_calibrateHandEye_"):
        raise ContractError("release calibration hand_eye mount/method is not approved")
    if hand_meta.get("dr_coverage_verified") is not True:
        raise ContractError("release calibration hand_eye DR coverage is not verified")
    _validate_hand_eye(hand_meta, {}, {})
    median = _release_calibration_number(hand_meta.get("median_error_m"), "hand_eye.median_error_m")
    p95 = _release_calibration_number(hand_meta.get("p95_error_m"), "hand_eye.p95_error_m")
    residual = _release_calibration_number(hand_meta.get("residual_m"), "hand_eye.residual_m")
    if p95 < median or abs(residual - median) > 1.0e-12:
        raise ContractError("release calibration hand_eye residual/median/p95 ordering is inconsistent")

    landmarks = configs["nurec_landmarks"]
    landmark_meta = landmarks.get("meta")
    if not isinstance(landmark_meta, dict) or landmark_meta.get("measured") is not True:
        raise ContractError("release calibration nurec landmarks are not measured")
    landmark_date = landmark_meta.get("date")
    if not isinstance(landmark_date, str) or not landmark_date.strip():
        raise ContractError("release calibration nurec landmark measurement date is required")
    try:
        dt.date.fromisoformat(landmark_date)
    except ValueError as exc:
        raise ContractError(
            "release calibration nurec landmark date must be ISO YYYY-MM-DD"
        ) from exc
    pairs = landmarks.get("pairs")
    if not isinstance(pairs, list) or len(pairs) < 3:
        raise ContractError("release calibration nurec landmarks require at least three pairs")
    names: set[str] = set()
    nurec_points: list[tuple[float, float, float]] = []
    base_points: list[tuple[float, float, float]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ContractError("release calibration nurec landmark pair must be a mapping")
        name = pair.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ContractError("release calibration nurec landmark names must be unique/non-empty")
        names.add(name)
        nurec_points.append(_release_landmark_vector(pair.get("p_nurec"), f"landmarks[{index}].p_nurec"))
        base_points.append(_release_landmark_vector(pair.get("p_base"), f"landmarks[{index}].p_base"))
    if not _landmarks_are_noncollinear(nurec_points) or not _landmarks_are_noncollinear(base_points):
        raise ContractError("release calibration nurec landmark geometry is collinear/degenerate")

    scene = configs["scene_registration"]
    _release_rigid_transform(scene.get("T_base_nurec"), "scene_registration.T_base_nurec")
    scene_scale = _release_calibration_number(
        scene.get("scale"), "scene_registration.scale", positive=True
    )
    scene_meta = _release_calibration_meta(scene.get("meta"), "scene_registration")
    if scene_meta.get("method") != "umeyama_landmarks":
        raise ContractError("release calibration scene_registration method is not measured landmarks")
    _validate_scene(scene_meta, {}, {})

    chain = configs["tool_chain"]
    elements = chain.get("elements")
    if not isinstance(elements, list) or not elements:
        raise ContractError("release calibration tool_chain elements are missing")
    required_names = {
        "tool_flange_plate", "magbot_TCW1", "onrobot_QC_RS_v3", "onrobot_2FG7",
    }
    chain_names: set[str] = set()
    chain_total = 0.0
    for element in elements:
        if not isinstance(element, dict) or element.get("measured") is not True:
            raise ContractError("release calibration tool_chain contains an unmeasured element")
        name = element.get("name")
        if not isinstance(name, str) or not name.strip() or name in chain_names:
            raise ContractError("release calibration tool_chain names must be unique/non-empty")
        chain_names.add(name)
        height = element.get("height_to_grip_m", element.get("height_m"))
        chain_total += _release_calibration_number(height, f"tool_chain.{name}.height", positive=True)
    if not required_names.issubset(chain_names):
        raise ContractError("release calibration tool_chain is missing a required physical element")
    verification = chain.get("verification")
    if not isinstance(verification, dict) or verification.get("qc_missing") is not False:
        raise ContractError("release calibration tool_chain QC-RS verification is incomplete")
    expected_tcp_z = _release_calibration_number(
        verification.get("expected_tcp_z_m"), "tool_chain.expected_tcp_z_m", positive=True
    )
    yaml_tcp_z = _release_calibration_number(
        verification.get("tcp_offset_yaml_z_m"), "tool_chain.tcp_offset_yaml_z_m", positive=True
    )
    if any(abs(value - chain_total) >= 1.0e-4 for value in (expected_tcp_z, yaml_tcp_z, tcp_matrix[2][3])):
        raise ContractError("release calibration tool_chain sum differs from approved TCP offset")

    return {
        "thresholds": {
            "tcp_offset_residual_m_max": TCP_OFFSET_MAX_RESIDUAL_M,
            "hand_eye_median_error_m_max": HAND_EYE_MAX_MEDIAN_ERROR_M,
            "hand_eye_p95_error_m_max": HAND_EYE_MAX_P95_ERROR_M,
            "scene_registration_residual_m_max": SCENE_REGISTRATION_MAX_RESIDUAL_M,
        },
        "tcp_offset_residual_m": float(tcp_meta["residual_m"]),
        "hand_eye_median_error_m": median,
        "hand_eye_p95_error_m": p95,
        "scene_registration_residual_m": float(scene_meta["residual_m"]),
        "scene_scale": scene_scale,
        "nurec_landmark_pairs": len(pairs),
        "tool_chain_elements": len(elements),
        "config_sha256": consumed_hashes,
    }


def verify_evidence(
    real_cfg: dict,
    real_config_path: Path | str,
    *,
    common_cfg: dict,
    dr_cfg: dict,
    now: dt.datetime | None = None,
    release: dict | None = None,
    release_manifest_sha256: str | None = None,
    bundle_manifest_sha256: str | None = None,
    allowed_evidence_root: Path | str | None = None,
) -> dict[str, dict]:
    """Validate all required evidence and return the decoded documents."""
    config_path = Path(real_config_path).resolve()
    contract_hash = real_contract_sha256(real_cfg)
    required = list(BASE_EVIDENCE_KINDS)
    # Gripper state is part of every released observation contract.  Disabling
    # actuation must never turn driver/width feedback evidence into an option.
    required.extend(GRIPPER_EVIDENCE_KINDS)
    current = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    decoded: dict[str, dict] = {}
    for kind in required:
        ref = real_cfg.get("evidence", {}).get(kind)
        if not isinstance(ref, dict):
            raise ContractError(f"missing evidence reference: {kind}")
        raw_path = str(ref.get("path", "")).strip()
        expected_hash = str(ref.get("sha256", "")).lower().strip()
        if not raw_path or len(expected_hash) != 64:
            raise ContractError(f"evidence {kind} requires path and 64-char sha256")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        if allowed_evidence_root is not None:
            path = _require_evidence_path(path, allowed_evidence_root)
        else:
            path = path.resolve()
        if not path.is_file():
            raise ContractError(f"evidence {kind} file not found: {path}")
        try:
            snapshot = read_regular_file_snapshot(
                path,
                label=f"evidence {kind}",
                expected_sha256=expected_hash,
            )
            doc = json.loads(snapshot.payload.decode("utf-8"))
        except FileSnapshotError as exc:
            if "sha256 mismatch" in str(exc):
                raise ContractError(f"evidence {kind} sha256 mismatch") from exc
            raise ContractError(f"cannot read evidence {kind}: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read evidence {kind}: {exc}") from exc
        if not isinstance(doc, dict):
            raise ContractError(f"evidence {kind} must contain a JSON object")
        if doc.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ContractError(f"evidence {kind} schema_version mismatch")
        if doc.get("kind") != kind or doc.get("passed") is not True:
            raise ContractError(f"evidence {kind} kind/pass mismatch")
        if doc.get("contract_sha256") != contract_hash:
            raise ContractError(f"evidence {kind} was collected for a different real config")
        recorded = _parse_time(doc.get("recorded_at"), "recorded_at")
        valid_until = _parse_time(doc.get("valid_until"), "valid_until")
        if recorded > current + dt.timedelta(minutes=5):
            raise ContractError(f"evidence {kind} recorded_at is in the future")
        if valid_until <= current:
            raise ContractError(f"evidence {kind} has expired")
        measurements = doc.get("measurements")
        if not isinstance(measurements, dict):
            raise ContractError(f"evidence {kind} measurements must be a mapping")
        _VALIDATORS[kind](measurements, real_cfg, {
            "common": common_cfg,
            "dr": dr_cfg,
            "release": release,
            "release_manifest_sha256": release_manifest_sha256,
            "bundle_manifest_sha256": bundle_manifest_sha256,
        })
        decoded[kind] = doc
    return decoded
