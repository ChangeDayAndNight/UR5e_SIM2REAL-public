"""Shared deterministic gripper-command observation contract.

The Student cannot observe the physical 2FG7 opening on the current DO-pulse
wiring.  It therefore observes a *command state*, not a joint position or
width estimate.  Both simulation and the real adapter advance this state once
per accepted 20 Hz policy command using the exact same float32 step.
"""
from __future__ import annotations

import math

import numpy as np
import torch

GRIPPER_COMMAND_OPEN: float = 0.0
GRIPPER_COMMAND_CLOSED: float = 1.0
GRIPPER_COMMAND_CONTROL_HZ: float = 20.0
GRIPPER_COMMAND_TRANSITION_S: float = 1.20
GRIPPER_COMMAND_TRANSITION_STEPS: int = 24
GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S: float = 0.040
# Use the float32 value in both scalar(real) and tensor(sim) paths so their
# traces agree to substantially better than the 1e-7 parity requirement.
GRIPPER_COMMAND_STEP: float = float(
    np.float32(1.0 / GRIPPER_COMMAND_TRANSITION_STEPS)
)


def validate_gripper_command_config(control_hz: float, transition_s: float) -> int:
    """Validate the frozen 1.20 s / 20 Hz contract and return 24 steps."""
    hz = float(control_hz)
    duration = float(transition_s)
    if not math.isfinite(hz) or not math.isfinite(duration) or hz <= 0.0 or duration <= 0.0:
        raise ValueError("gripper command control_hz/transition_s must be finite and positive")
    product = hz * duration
    rounded = round(product)
    if not math.isclose(product, rounded, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("gripper command transition must contain an integer number of control steps")
    if (
        not math.isclose(hz, GRIPPER_COMMAND_CONTROL_HZ, rel_tol=0.0, abs_tol=1.0e-12)
        or not math.isclose(
            duration, GRIPPER_COMMAND_TRANSITION_S, rel_tol=0.0, abs_tol=1.0e-12
        )
        or rounded != GRIPPER_COMMAND_TRANSITION_STEPS
    ):
        raise ValueError(
            "gripper command contract is frozen at transition_s=1.20, "
            "control_hz=20.0, transition_steps=24"
        )
    return int(rounded)


def validate_gripper_command_state(value: object, *, name: str = "gripper_command_state") -> float:
    """Return a finite scalar in [0, 1]; unknown/out-of-range values fail closed."""
    if value is None:
        raise ValueError(f"{name} is unknown/invalid")
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar) or not GRIPPER_COMMAND_OPEN <= scalar <= GRIPPER_COMMAND_CLOSED:
        raise ValueError(f"{name} must be finite and within [0, 1], got {scalar!r}")
    return scalar


def advance_gripper_command_state(
    current: object,
    closed: bool,
) -> float:
    """Move a real/scalar state by exactly 1/24 after one accepted command.

    A physical motion adapter starts with no trustworthy command history and
    must remain invalid until an explicit-open settling interval has completed.
    Therefore this helper never initializes an unknown value; real startup is
    handled by the adapter's timed initialization gate.
    """
    if current is None:
        raise ValueError("gripper command state is unknown/invalid")
    value = np.float32(validate_gripper_command_state(current))
    target = np.float32(GRIPPER_COMMAND_CLOSED if bool(closed) else GRIPPER_COMMAND_OPEN)
    step = np.float32(GRIPPER_COMMAND_STEP)
    if value < target:
        value = np.minimum(np.float32(value + step), target)
    elif value > target:
        value = np.maximum(np.float32(value - step), target)
    return float(np.float32(value))


def advance_gripper_command_state_tensor(
    current: torch.Tensor, closed: torch.Tensor
) -> torch.Tensor:
    """Vectorized sim equivalent of :func:`advance_gripper_command_state`."""
    if not torch.is_floating_point(current):
        raise ValueError("gripper command state tensor must be floating point")
    if current.shape != closed.shape:
        raise ValueError(
            f"gripper command state/closed shape mismatch: {tuple(current.shape)} "
            f"vs {tuple(closed.shape)}"
        )
    if not torch.isfinite(current).all() or not ((current >= 0.0) & (current <= 1.0)).all():
        raise ValueError("gripper command state tensor must be finite and within [0, 1]")
    target = closed.to(dtype=current.dtype)
    step = torch.as_tensor(GRIPPER_COMMAND_STEP, dtype=current.dtype, device=current.device)
    return torch.minimum(torch.maximum(target, current - step), current + step).clamp_(0.0, 1.0)


def gripper_command_spec(*, transition_s: float, transition_steps: int) -> dict:
    """Serializable observation semantics embedded in every new obs spec."""
    validate_gripper_command_config(
        GRIPPER_COMMAND_CONTROL_HZ,
        transition_s,
    )
    if int(transition_steps) != GRIPPER_COMMAND_TRANSITION_STEPS:
        raise ValueError("gripper command transition_steps must be exactly 24")
    return {
        "name": "gripper_command_state",
        "semantics": "delayed_dispatched_binary_command_move_toward",
        "update_point": "after_delayed_or_successfully_dispatched_command",
        "visible_at": "next_observation",
        "range": [GRIPPER_COMMAND_OPEN, GRIPPER_COMMAND_CLOSED],
        "open": GRIPPER_COMMAND_OPEN,
        "closed": GRIPPER_COMMAND_CLOSED,
        "control_hz": GRIPPER_COMMAND_CONTROL_HZ,
        "transition_s": GRIPPER_COMMAND_TRANSITION_S,
        "transition_steps": GRIPPER_COMMAND_TRANSITION_STEPS,
        "sim_reset": "open",
        "real_startup": "unknown_until_explicit_open_settled",
        "real_dispatch_gate": "target_pin_high_sdk_success_bounded_ack",
        "real_sdk_success": "return_is_exactly_true",
        "real_dispatch_ack_timeout_s": GRIPPER_COMMAND_DISPATCH_ACK_TIMEOUT_S,
        "real_startup_settle_origin": "successful_pulse_completion",
        "rapid_reversal_while_pulse_active": "invalid_fail_closed",
        "late_high_after_sdk_entry": "possible_safe_low_attempt_only_motion_p0",
        "fault_or_estop": "invalid",
        "physical_width_feedback": False,
    }
