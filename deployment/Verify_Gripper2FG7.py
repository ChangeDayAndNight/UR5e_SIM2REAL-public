#!/usr/bin/env python
"""Verify the configured 2FG7 digital-output path before deployment.

The command-capable mode is unconditionally blocked before configuration or
socket access in this receive-only release. ``--dry`` is receive-only. Reports
are JSON/TXT so running the verifier cannot recreate project Markdown.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.config_io import build_safety_cfg, load_yaml  # noqa: E402
from common.safety import SafetyGuard  # noqa: E402
from deployment.contracts import load_real_config, require_motion_authorization  # noqa: E402

_Q_TOL_RAD = 1.0e-3
_MAX_CALL_MS = 25.0
_MIN_CYCLES = 3


def _state_with_force(robot):
    state = robot.get_state()
    if state.tcp_force is None:
        raise RuntimeError("TCP force feedback is unavailable; gripper verification is blocked")
    arrays = (state.joint_pos, state.joint_vel, state.tcp_pose, state.tcp_force)
    if not all(np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in arrays):
        raise RuntimeError("non-finite robot state during gripper verification")
    return state


def _q(robot) -> np.ndarray:
    return np.asarray(_state_with_force(robot).joint_pos, dtype=np.float64)


def _gate_guard(safety_cfg, robot) -> tuple[SafetyGuard, bool, np.ndarray]:
    """Create the gripper-only guard while retaining velocity and force limits."""
    import dataclasses

    state = _state_with_force(robot)
    tcp = np.asarray(state.tcp_pose[:3], dtype=np.float64)
    lo = np.asarray(safety_cfg.workspace_min, dtype=np.float64)
    hi = np.asarray(safety_cfg.workspace_max, dtype=np.float64)
    inside = bool(np.all(tcp >= lo) and np.all(tcp <= hi))
    inf = float("inf")
    gate_cfg = dataclasses.replace(
        safety_cfg, workspace_min=(-inf, -inf, -inf), workspace_max=(inf, inf, inf)
    )
    return SafetyGuard(gate_cfg, estop_hook=lambda reason: robot.emergency_stop()), inside, tcp


def _one_action(
    robot,
    guard: SafetyGuard,
    closed: bool,
    settle_s: float,
    q_tol: float,
    records: list,
) -> None:
    label = "close(DO5)" if closed else "open(DO6)"
    before_state = _state_with_force(robot)
    verdict = guard.check(before_state)
    if not verdict.ok:
        guard.trip(verdict.reason)
        raise SystemExit(f"gripper safety check failed before {label}: {verdict.reason}")
    q_before = np.asarray(before_state.joint_pos, dtype=np.float64)
    t0 = time.perf_counter()
    robot.set_gripper(closed)
    call_ms = (time.perf_counter() - t0) * 1000.0
    if hasattr(robot, "wait_for_gripper"):
        robot.wait_for_gripper(timeout_s=max(settle_s, 1.0))
    time.sleep(settle_s)
    after_state = _state_with_force(robot)
    q_after = np.asarray(after_state.joint_pos, dtype=np.float64)
    dq = float(np.max(np.abs(q_after - q_before)))
    verdict_after = guard.check(after_state)
    record = {
        "action": label,
        "call_ms": call_ms,
        "dq": dq,
        "ok_q": dq < q_tol,
        "safe": verdict_after.ok,
        "reason": verdict_after.reason,
    }
    records.append(record)
    if call_ms >= _MAX_CALL_MS:
        guard.trip(f"blocking gripper call: {call_ms:.1f} ms")
        raise SystemExit(f"set_gripper blocked for {call_ms:.1f} ms (limit {_MAX_CALL_MS} ms)")
    if dq >= q_tol:
        guard.trip(f"gripper action moved arm joints: max dq={dq:.6f} rad")
        raise SystemExit(f"gripper action moved arm joints: {dq:.6f} rad >= {q_tol}")
    if not verdict_after.ok:
        guard.trip(verdict_after.reason)
        raise SystemExit(f"gripper safety check failed after {label}: {verdict_after.reason}")


def _write_report(
    out_dir: Path,
    cfg: dict,
    records: list,
    cycles: int,
    q_tol: float,
    settle_s: float,
    extra: dict | None = None,
) -> None:
    if not records:
        raise ValueError("cannot write a passing report without measurements")
    extra = extra or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    dqs = [float(r["dq"]) for r in records]
    calls = [float(r["call_ms"]) for r in records]
    passed = (
        len(records) == cycles * 2 + 1
        and max(dqs) < q_tol
        and max(calls) < _MAX_CALL_MS
        and all(bool(r["safe"]) and bool(r["ok_q"]) for r in records)
    )
    report = {
        "schema_version": 1,
        "kind": "gripper_driver_raw",
        "passed": passed,
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configuration": {
            "driver": cfg["gripper"]["driver"],
            "do_open": int(cfg["gripper"]["do_open"]),
            "do_close": int(cfg["gripper"]["do_close"]),
            "pulse_s": float(cfg["gripper"]["pulse_s"]),
            "cycles": cycles,
            "settle_s": settle_s,
            "q_tol_rad": q_tol,
        },
        "measurements": {
            "actions": len(records),
            "max_joint_delta_rad": max(dqs),
            "median_joint_delta_rad": statistics.median(dqs),
            "max_call_ms": max(calls),
            "median_call_ms": statistics.median(calls),
            "force_available_every_sample": True,
            "tcp_inside_workspace_at_start": bool(extra.get("tcp_inside", False)),
            "tcp_start_m": [float(v) for v in np.asarray(extra.get("tcp0", [])).reshape(-1)],
        },
        "records": records,
        "limitations": [
            "This test does not prove gripper width/occupancy feedback semantics.",
            "Do not set gripper_width evidence from this report.",
        ],
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    text = [
        "2FG7 DRIVER VERIFICATION",
        f"passed: {passed}",
        f"actions: {len(records)}",
        f"max joint delta [rad]: {max(dqs):.6f}",
        f"max non-blocking call [ms]: {max(calls):.3f}",
        "width feedback verified: false (separate evidence required)",
    ]
    (out_dir / "report.txt").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="2FG7 configured-driver verification")
    parser.add_argument("--real-config", required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=_MIN_CYCLES)
    parser.add_argument("--dry", action="store_true", help="receive-only connectivity check")
    args = parser.parse_args()
    if args.cycles < _MIN_CYCLES:
        raise SystemExit(f"--cycles must be >= {_MIN_CYCLES}")
    if not args.dry:
        require_motion_authorization("gripper")   # 그리퍼 개폐만 — 팔은 정지
    real_cfg, _ = load_real_config(args.real_config)
    if real_cfg["gripper"].get("driver") != "do_pulse":
        raise SystemExit("this verifier requires gripper.driver=do_pulse")
    safety_cfg = build_safety_cfg(load_yaml(ROOT / "configs" / "safety.yaml"))
    from real.real_robot import RealRobot

    robot = RealRobot(
        real_cfg, ROOT / "configs" / "calib" / "tcp_offset.yaml", receive_only=args.dry
    )
    try:
        if not robot.is_ok():
            raise RuntimeError("robot health is not OK")
        if args.dry:
            _state_with_force(robot)
            print("GRIPPER_2FG7_DRY_OK")
            return 0
        records: list[dict] = []
        guard, tcp_inside, tcp0 = _gate_guard(safety_cfg, robot)
        settle_s = float(real_cfg["gripper"].get("settle_s", 1.2))
        q0 = _q(robot)
        for _ in range(args.cycles):
            _one_action(robot, guard, False, settle_s, _Q_TOL_RAD, records)
            _one_action(robot, guard, True, settle_s, _Q_TOL_RAD, records)
        _one_action(robot, guard, False, settle_s, _Q_TOL_RAD, records)
        drift = float(np.max(np.abs(_q(robot) - q0)))
        if drift >= _Q_TOL_RAD:
            guard.trip(f"total joint drift {drift:.6f} rad")
            raise RuntimeError("total arm-joint drift exceeded limit")
        out = HERE / "evidence" / f"gripper_2fg7_{dt.datetime.now():%Y%m%d_%H%M%S}"
        _write_report(
            out,
            real_cfg,
            records,
            args.cycles,
            _Q_TOL_RAD,
            settle_s,
            {"tcp_inside": tcp_inside, "tcp0": tcp0},
        )
        print(f"GRIPPER_2FG7_OK report={out / 'report.json'}")
        return 0
    except BaseException:
        # Any SDK/state/force/worker/report exception after command capability
        # is opened must latch and attempt the full emergency-stop sequence.
        if not args.dry:
            robot.emergency_stop()
        raise
    finally:
        robot.close()


if __name__ == "__main__":
    raise SystemExit(main())
