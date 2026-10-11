"""Image-noise parameter broadcasting and isolated per-episode sampling."""
from __future__ import annotations

import math

import torch


def image_parameter(value, image: torch.Tensor, name: str) -> torch.Tensor:
    """Accept a scalar or one value per image, never implicit spatial broadcasts."""
    param = torch.as_tensor(value, device=image.device, dtype=image.dtype)
    if not bool(torch.isfinite(param).all()):
        raise ValueError(f"{name} must be finite")
    if param.ndim == 0:
        return param
    if param.ndim != 1 or image.ndim < 3 or param.shape[0] != image.shape[0]:
        raise ValueError(f"{name} must be scalar or [N] for a batch of N images")
    return param.reshape((image.shape[0],) + (1,) * (image.ndim - 1))


def resample_episode_parameters(env, attribute: str, env_ids, ranges: dict) -> None:
    """Update only reset environments; initialize others to range midpoints.

    This isolates episode parameters, not the entire simulator's RNG stream.
    Empty/partial resets must not change ongoing episodes' noise strengths.
    """
    all_ids = torch.arange(env.num_envs, device=env.device)
    ids = (all_ids if env_ids is None else all_ids[env_ids] if isinstance(env_ids, slice)
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    for name, (lo, hi) in ranges.items():
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
            raise ValueError(f"invalid noise range {name}: {(lo, hi)}")
    params = getattr(env, attribute, None)
    if params is None:
        params = {}
        setattr(env, attribute, params)
    for name, (lo, hi) in ranges.items():
        values = params.get(name)
        if not torch.is_tensor(values) or values.shape != (env.num_envs,):
            values = torch.full((env.num_envs,), (lo + hi) / 2.0, device=env.device)
            params[name] = values
        values[ids] = torch.empty(ids.numel(), device=env.device).uniform_(lo, hi)
