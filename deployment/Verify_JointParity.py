#!/usr/bin/env python
"""Verify UR5e joint order/sign/offset and base-frame parity.

``--run`` is retained only as a future interface and is unconditionally blocked
before configuration or socket access. ``--check-only`` is receive-only. Reports
are JSON/TXT only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.calibration import load_tcp_offset, mat_to_pose7, quat_to_mat  # noqa: E402
from common.robot_spec import UR5E_ARM_JOINT_NAMES, UR5E_HOME_JOINT_POS  # noqa: E402
from deployment.contracts import load_real_config, require_motion_authorization  # noqa: E402

_FK_PATH = ROOT / "scripts" / "tools" / "urdf_fk.py"
_URDF = ROOT / "assets" / "ur5e_2fg7" / "ur5e_tcw1_2fg7.urdf"
_SLOPE_TOL = 0.05
_OFFSET_TOL_RAD = 0.01
_CROSS_AXIS_TOL_RAD = math.radians(1.0)
_TCP_POS_TOL_M = 0.03
_TCP_ROT_TOL_RAD = math.radians(5.0)


def _load_fk():
    spec = importlib.util.spec_from_file_location("urdf_fk", _FK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load FK helper: {_FK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sim_home_pose(urdf_fk) -> np.ndarray:
    poses = urdf_fk.forward_kinematics(_URDF, UR5E_HOME_JOINT_POS)
    transform = poses["wrist_3_link"] @ load_tcp_offset(ROOT / "configs/calib/tcp_offset.yaml")
    return mat_to_pose7(transform)


def _fit_sign_offset(cmd: np.ndarray, real: np.ndarray) -> tuple[float, float, float]:
    slope, offset = np.polyfit(cmd, real, 1)
    residual = float(np.max(np.abs(real - (slope * cmd + offset))))
    return float(slope), float(offset), residual


def _verdict_joint(name: str, slope: float, offset: float, residual: float) -> tuple[bool, str]:
    if slope < 0:
        return False, f"{name}: sign inversion (slope={slope:+.4f})"
    if abs(slope - 1.0) > _SLOPE_TOL:
        return False, f"{name}: slope error exceeds {_SLOPE_TOL:.0%}"
    if abs(offset) > _OFFSET_TOL_RAD:
        return False, f"{name}: offset {offset:+.5f} rad exceeds {_OFFSET_TOL_RAD}"
    return True, f"{name}: slope={slope:.4f}, offset={offset:+.5f}, residual={residual:.5f}"


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    relative = quat_to_mat(a).T @ quat_to_mat(b)
    return float(math.acos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))


def _set_target_commandless_check(state) -> None:
    arrays = (state.joint_pos, state.joint_vel, state.tcp_pose, state.tcp_force)
    if state.tcp_force is None or not all(
        np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in arrays
    ):
        raise RuntimeError("finite robot state and TCP force are required")


def main() -> int:
    parser = argparse.ArgumentParser(description="joint/base parity verification")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="pure regression self-test")
    mode.add_argument("--check-only", action="store_true", help="receive-only base-frame check")
    mode.add_argument("--run", action="store_true", help="perform low-speed joint motion")
    parser.add_argument("--real-config", required=True, type=Path)
    parser.add_argument("--delta-deg", type=float, default=10.0)
    parser.add_argument("--speed-scale", type=float, default=0.05)
    args = parser.parse_args()
    if args.run:
        require_motion_authorization("scripted_pose")
    real_cfg, _ = load_real_config(args.real_config)
    if not 2.0 <= args.delta_deg <= 10.0:
        raise SystemExit("--delta-deg must be in [2, 10]")
    if not 0.01 <= args.speed_scale <= 0.05:
        raise SystemExit("--speed-scale must be in [0.01, 0.05]")

    fk = _load_fk()
    home = np.asarray([UR5E_HOME_JOINT_POS[name] for name in UR5E_ARM_JOINT_NAMES], dtype=float)
    sim_pose = _sim_home_pose(fk)
    if args.dry_run:
        cmd = np.radians(np.array([-10.0, -5.0, 0.0, 5.0, 10.0]))
        cases = ((cmd, True), (-cmd, False), (cmd + math.radians(2.0), False), (0.9 * cmd, False))
        passed = all(_verdict_joint("synthetic", *_fit_sign_offset(cmd, real))[0] == expected
                     for real, expected in cases)
        print("JOINT_PARITY_DRY_OK" if passed else "JOINT_PARITY_DRY_FAIL")
        return 0 if passed else 1

    from real.real_robot import RealRobot

    robot = RealRobot(
        real_cfg,
        ROOT / "configs/calib/tcp_offset.yaml",
        receive_only=args.check_only,
    )
    try:
        if not robot.is_ok():
            raise RuntimeError("robot health is not OK")
        state = robot.get_state()
        _set_target_commandless_check(state)
        current = np.asarray(state.joint_pos, dtype=float)
        near_home = float(np.max(np.abs(current - home))) <= 0.10
        pos_error = float(np.linalg.norm(np.asarray(state.tcp_pose[:3]) - sim_pose[:3]))
        rot_error = _quat_angle(np.asarray(state.tcp_pose[3:]), sim_pose[3:])
        base_ok = near_home and pos_error <= _TCP_POS_TOL_M and rot_error <= _TCP_ROT_TOL_RAD
        if args.check_only:
            print(
                f"JOINT_PARITY_CHECK {'OK' if base_ok else 'FAIL'} "
                f"position_error_m={pos_error:.6f} rotation_error_rad={rot_error:.6f}"
            )
            return 0 if base_ok else 2
        if float(np.max(np.abs(current - home))) > 0.5:
            raise RuntimeError("current pose is more than 0.5 rad from reviewed home")

        rows: list[dict] = []
        deltas = [-args.delta_deg, -args.delta_deg / 2.0, 0.0, args.delta_deg / 2.0, args.delta_deg]
        try:
            robot.move_to_joints(home, speed_scale=args.speed_scale)
            for joint_index, joint_name in enumerate(UR5E_ARM_JOINT_NAMES):
                for delta_deg in deltas:
                    command = home.copy()
                    command[joint_index] += math.radians(delta_deg)
                    robot.move_to_joints(command, speed_scale=args.speed_scale)
                    time.sleep(0.4)
                    measured_state = robot.get_state()
                    _set_target_commandless_check(measured_state)
                    measured = np.asarray(measured_state.joint_pos, dtype=float)
                    rows.append({
                        "joint_index": joint_index,
                        "joint_name": joint_name,
                        "delta_deg": delta_deg,
                        **{f"cmd_{idx}": float(command[idx]) for idx in range(6)},
                        **{f"real_{idx}": float(measured[idx]) for idx in range(6)},
                    })
                robot.move_to_joints(home, speed_scale=args.speed_scale)
        except BaseException:
            robot.emergency_stop()
            raise

        results: list[dict] = []
        max_slope_error = max_offset = max_cross = 0.0
        all_ok = bool(base_ok)
        for joint_index, joint_name in enumerate(UR5E_ARM_JOINT_NAMES):
            subset = [row for row in rows if row["joint_index"] == joint_index]
            command = np.asarray([row[f"cmd_{joint_index}"] for row in subset])
            measured = np.asarray([row[f"real_{joint_index}"] for row in subset])
            slope, offset, residual = _fit_sign_offset(command, measured)
            ok, reason = _verdict_joint(joint_name, slope, offset, residual)
            cross = max(
                float(np.max(np.abs(np.asarray([row[f"real_{other}"] for row in subset]) - home[other])))
                for other in range(6) if other != joint_index
            )
            ok = ok and cross <= _CROSS_AXIS_TOL_RAD
            all_ok = all_ok and ok
            max_slope_error = max(max_slope_error, abs(slope - 1.0))
            max_offset = max(max_offset, abs(offset))
            max_cross = max(max_cross, cross)
            results.append({
                "joint": joint_name,
                "slope": slope,
                "offset_rad": offset,
                "max_residual_rad": residual,
                "max_cross_axis_rad": cross,
                "passed": ok,
                "reason": reason,
            })
        robot.move_to_joints(home, speed_scale=args.speed_scale)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = HERE / "evidence" / f"joint_parity_{stamp}"
        output.mkdir(parents=True, exist_ok=True)
        with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "schema_version": 1,
            "kind": "joint_parity_raw",
            "passed": all_ok,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "measurements": {
                "base_position_error_m": pos_error,
                "base_rotation_error_rad": rot_error,
                "max_slope_error_fraction": max_slope_error,
                "max_offset_rad": max_offset,
                "max_cross_axis_rad": max_cross,
            },
            "joints": results,
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (output / "report.txt").write_text(
            "JOINT PARITY\n"
            f"passed: {all_ok}\n"
            f"position error [m]: {pos_error:.6f}\n"
            f"rotation error [rad]: {rot_error:.6f}\n"
            f"max slope error: {max_slope_error:.6f}\n"
            f"max offset [rad]: {max_offset:.6f}\n"
            f"max cross-axis [rad]: {max_cross:.6f}\n",
            encoding="utf-8",
        )
        print(f"JOINT_PARITY_{'OK' if all_ok else 'FAIL'} report={output / 'report.json'}")
        return 0 if all_ok else 4
    except BaseException:
        if args.run:
            robot.emergency_stop()
        raise
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
