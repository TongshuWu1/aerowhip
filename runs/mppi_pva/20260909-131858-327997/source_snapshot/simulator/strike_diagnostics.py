"""Replay hit diagnostics using the same contact and velocity gates as training."""

from __future__ import annotations

import numpy as np
import torch

from learning.point_force_env import _segment_enters_sphere, tip_velocity_strike_gate


def replay_strike_diagnostics(
    positions: np.ndarray, velocities: np.ndarray, marker_indices: list[int],
    target: np.ndarray, direction: np.ndarray, *, radius_m: float,
    minimum_speed_m_s: float, maximum_angle_deg: float,
    tip_must_enter_first: bool = True, use_float32: bool = False,
) -> dict[str, np.ndarray]:
    dtype = torch.float32 if use_float32 else torch.float64
    position = torch.as_tensor(positions, dtype=dtype)
    velocity = torch.as_tensor(velocities, dtype=dtype)
    center = torch.as_tensor(target, dtype=dtype).expand(len(position), -1)
    previous = torch.cat((position[:1], position[:-1]), dim=0)
    crossing = _segment_enters_sphere(previous[:, marker_indices], position[:, marker_indices], center, radius_m)
    crossing[0] = False  # The reset frame has no simulated transition.
    finite = torch.isfinite(position).flatten(1).all(1) & torch.isfinite(velocity).flatten(1).all(1)
    crossing &= finite[:, None]
    tip, other = crossing[:, -1], crossing[:, :-1].any(dim=-1)
    non_tip_entry = other & (~tip | tip_must_enter_first)
    disqualified = non_tip_entry.cumsum(0) > 0
    eligible = tip & ~other & ~disqualified
    velocity_ok = tip_velocity_strike_gate(velocity[:, -1], torch.as_tensor(direction, dtype=dtype),
        minimum_directed_speed_m_s=minimum_speed_m_s, maximum_direction_error_deg=maximum_angle_deg)
    hit = eligible & velocity_ok
    return {name: value.numpy() for name, value in {
        "tip_contact": tip, "other_contact": other, "disqualified": disqualified,
        "eligible": eligible, "hit": hit, "success_so_far": hit.cumsum(0) > 0,
    }.items()}
