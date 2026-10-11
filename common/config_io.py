# configs/*.yaml → 타입 있는 cfg dataclass 변환의 단일 지점.
# sim(_binpick_env_cfg)과 real(Deploy_*)이 "같은 함수"로 같은 yaml을 읽는다 —
# 필드명 매핑이 두 군데로 갈라지는 것을 방지한다.
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import yaml

from common.file_snapshot import (
    FileSnapshotError,
    RegularFileSnapshot,
    read_regular_file_snapshot,
)
from common.gripper_command import validate_gripper_command_config
from common.robot_spec import (
    UR5E_ARM_JOINT_NAMES,
    UR5E_HOME_JOINT_POS,
    UR5E_MAX_REACH_M,
)
from common.types import ActionCfg, PreprocessCfg, SafetyCfg


def load_yaml_snapshot(
    path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    label: str | None = None,
) -> tuple[object, RegularFileSnapshot]:
    """Decode YAML from the same bytes used for optional authority checks."""
    source = Path(path)
    description = label or f"YAML file {source}"
    # A long-running authorized child must consume the exact YAML bytes frozen
    # by its parent, not reopen a pathname that could be swapped and restored.
    # The lazy import avoids a module cycle during the authority bootstrap.
    from common.runtime_authority import runtime_yaml_snapshot

    snapshot = runtime_yaml_snapshot(source)
    if snapshot is None:
        snapshot = read_regular_file_snapshot(
            source,
            label=description,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )
    else:
        if expected_sha256 is not None and snapshot.sha256 != expected_sha256:
            raise FileSnapshotError(f"{description} sha256 mismatch")
        if expected_size is not None and snapshot.size != expected_size:
            raise FileSnapshotError(f"{description} size mismatch")
    try:
        value = yaml.safe_load(snapshot.payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FileSnapshotError(f"cannot decode {description}: {exc}") from exc
    return value, snapshot


def load_yaml(path: Path | str) -> dict:
    value, _ = load_yaml_snapshot(path)
    return value


def build_action_cfg(common_yaml: dict) -> ActionCfg:
    a = common_yaml["action"]
    # ★절대 모드 offset(home) = sim default_joint_pos와 동일한 robot_spec 단일 소스★
    #   관절 순서는 UR5E_ARM_JOINT_NAMES 고정(sim ArticulationCfg·real q 순서와 일치).
    home = tuple(float(UR5E_HOME_JOINT_POS[j]) for j in UR5E_ARM_JOINT_NAMES)
    return ActionCfg(
        mode=a["mode"],
        num_arm_joints=int(a["num_arm_joints"]),
        action_scale_rad=float(a["action_scale_rad"]),
        gripper_close_threshold=float(a["gripper_close_threshold"]),
        task_pos_scale_m=float(a.get("task_pos_scale_m", 0.2)),
        task_rot_scale_rad=float(a.get("task_rot_scale_rad", 0.2)),
        home_joint_pos=home,
        vertical_grasp_only=bool(a.get("vertical_grasp_only", False)),
    )


def build_preprocess_cfg(common_yaml: dict) -> PreprocessCfg:
    p = common_yaml["preprocess"]
    pr = common_yaml["proprio"]
    action_cfg = build_action_cfg(common_yaml)
    transition_s = float(common_yaml["gripper_command_state"]["transition_s"])
    transition_steps = validate_gripper_command_config(
        float(common_yaml["control"]["control_hz"]), transition_s
    )
    return PreprocessCfg(
        depth_near_m=float(p["depth_near_m"]),
        depth_far_m=float(p["depth_far_m"]),
        out_height=int(p["out_height"]),
        out_width=int(p["out_width"]),
        joint_pos_scale=float(pr["joint_pos_scale"]),
        joint_vel_scale=float(pr["joint_vel_scale"]),
        tcp_pos_scale=float(pr["tcp_pos_scale"]),
        action_dim=action_cfg.action_dim,
        crop_fraction=float(p.get("crop_fraction", 1.0)),
        use_rgb=bool(p.get("use_rgb", False)),
        frame_mode=str(p.get("frame_mode", "square")),
        gripper_command_transition_s=transition_s,
        gripper_command_transition_steps=transition_steps,
    )


def build_safety_cfg(safety_yaml: dict) -> SafetyCfg:
    ws = safety_yaml["workspace"]
    lim = safety_yaml["limits"]
    return SafetyCfg(
        workspace_min=tuple(float(v) for v in ws["min"]),
        workspace_max=tuple(float(v) for v in ws["max"]),
        joint_vel_max_rad_s=float(lim["joint_vel_max_rad_s"]),
        tcp_force_max_n=float(lim["tcp_force_max_n"]),
        tcp_linear_vel_max_m_s=float(lim.get("tcp_linear_vel_max_m_s", 0.25)),
        tcp_angular_vel_max_rad_s=float(lim.get("tcp_angular_vel_max_rad_s", 0.5)),
        # ★감사 B1★ 없으면 None(비활성)이 아니라 ★있어야 정상★ — 드리프트 가드가 값을 검증한다.
        tcp_z_floor_m=(float(ws["tcp_z_floor_m"]) if "tcp_z_floor_m" in ws else None),
    )


def validate_target_sample_range(
    target: Iterable[float], common_yaml: dict, *, tag: str
) -> tuple[float, float, float]:
    """Return a finite XYZ target only when it is inside the trained range."""
    values = tuple(float(value) for value in target)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"[{tag}] target must contain exactly three finite values")
    ranges = common_yaml["target"]["sample_range"]
    for index, axis in enumerate(("x", "y", "z")):
        lower, upper = (float(value) for value in ranges[axis])
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError(f"[{tag}] invalid trained target range for {axis}")
        if not lower <= values[index] <= upper:
            raise ValueError(
                f"[{tag}] target {axis}={values[index]:.6g} is outside the trained "
                f"range [{lower}, {upper}]"
            )
    return values


def validate_phase3_target_geometry(
    target: Iterable[float],
    common_yaml: dict,
    sim_yaml: dict,
    safety_yaml: dict,
    *,
    tag: str = "phase3",
) -> tuple[float, float, float]:
    """Recompute target feasibility solely from canonical config geometry.

    The receipt caller supplies only XYZ.  Feasibility is derived here from
    the trained range, support surface, safety workspace and UR5e reach, so a
    hand-written ``target_infeasible: false`` can never open the release gate.
    """
    values = validate_target_sample_range(target, common_yaml, tag=tag)
    x, y, z = values
    scene = sim_yaml["scene"]
    belt_size = tuple(float(value) for value in scene["belt_size_m"])
    belt_center = tuple(float(value) for value in scene["belt_center_xy_m"])
    object_size = float(scene["object_size_m"])
    table_height = float(scene["table_height_m"])
    belt_yaw = float(scene.get("belt_yaw_deg", 0.0))
    if len(belt_size) != 3 or len(belt_center) != 2 \
            or not all(math.isfinite(value) and value > 0.0 for value in belt_size) \
            or not all(math.isfinite(value) for value in belt_center) \
            or not math.isfinite(object_size) or object_size <= 0.0 \
            or not math.isfinite(table_height) \
            or not any(abs(belt_yaw - allowed) <= 1.0e-9 for allowed in (0.0, 180.0, -180.0)):
        raise ValueError(f"[{tag}] canonical support geometry is invalid")
    half_object = object_size / 2.0
    support_x = (
        belt_center[0] - belt_size[0] / 2.0 + half_object,
        belt_center[0] + belt_size[0] / 2.0 - half_object,
    )
    support_y = (
        belt_center[1] - belt_size[1] / 2.0 + half_object,
        belt_center[1] + belt_size[1] / 2.0 - half_object,
    )
    expected_z = table_height + half_object
    if not support_x[0] <= x <= support_x[1] \
            or not support_y[0] <= y <= support_y[1] \
            or not math.isclose(z, expected_z, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"[{tag}] target is outside the canonical support geometry")
    workspace_min = tuple(float(value) for value in safety_yaml["workspace"]["min"])
    workspace_max = tuple(float(value) for value in safety_yaml["workspace"]["max"])
    if len(workspace_min) != 3 or len(workspace_max) != 3 \
            or any(not math.isfinite(value) for value in (*workspace_min, *workspace_max)) \
            or any(lower > upper for lower, upper in zip(workspace_min, workspace_max)) \
            or any(not lower <= value <= upper for value, lower, upper in zip(
                values, workspace_min, workspace_max
            )):
        raise ValueError(f"[{tag}] target is outside the canonical safety workspace")
    if math.hypot(x, y) > UR5E_MAX_REACH_M + 1.0e-9:
        raise ValueError(f"[{tag}] target exceeds canonical UR5e planar reach")
    return values


def configs_dir() -> Path:
    """프로젝트 루트의 configs/ 절대 경로 (이 파일 위치 기준 — cwd 무관)."""
    return Path(__file__).resolve().parent.parent / "configs"
