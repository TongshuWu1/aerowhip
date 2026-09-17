"""Flight envelope checks on predicted attitude, separate from model fitting."""
import math
import torch


def attitude_valid(rotation_tracking_to_world, rotation_command_from_tracking, maximum_tilt_deg):
    """Limit command-frame z relative to world up, respecting tracking alignment."""
    command_to_world = rotation_tracking_to_world @ rotation_command_from_tracking.transpose(-1, -2)
    return torch.isfinite(command_to_world).all(dim=-1).all(dim=-1) & (
        command_to_world[..., 2, 2] >= math.cos(math.radians(maximum_tilt_deg)) - 1e-10)
