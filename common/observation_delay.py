"""Keep image latency separate from current robot/control-state observations."""
from __future__ import annotations

import torch


def delay_policy_observation(flat, delay_buffer, image_dim: int, *, delay_proprio: bool = False):
    """Delay RGB-D together, preserving latest proprio unless explicitly requested.

    delay_proprio=True reproduces the legacy whole-observation latency ablation.
    The caller must reset the buffer for each reset environment.
    """
    if flat.ndim != 2 or not 0 < image_dim < flat.shape[-1]:
        raise ValueError("expected [N, image + proprio] observations")
    if delay_buffer is None:
        return flat
    if delay_proprio:
        return delay_buffer.compute(flat)
    image = delay_buffer.compute(flat[:, :image_dim])
    return torch.cat((image, flat[:, image_dim:]), dim=-1)
