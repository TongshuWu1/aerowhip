"""Small shared helpers required by the production decomposed fitter."""

from __future__ import annotations

import numpy as np
import torch

from simulator.uav.state import ResidualHistoryState

from .config import FitConfiguration
from .dataset import ProcessedTake
from .initialization import causal_polynomial_derivative
from .windows import PredictionWindow


UAV_PARAMETER_NAMES = ("K_p", "K_v", "k_a", "K_R", "K_omega")


def euler_xyz_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.clamp(
        torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), min=1.0e-15
    )
    x, y, z, w = quaternion.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def measured_residual_history(
    take: ProcessedTake,
    window: PredictionWindow,
    config: FitConfiguration,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> ResidualHistoryState:
    """Build the pre-t0 FIFO using causal measured initialization data only."""

    arrays = take.arrays
    history_samples = int(config.uav_residual_ablation["history_samples"])
    derivative_frames = config.initialization_history_frames
    features = []
    for index in range(window.start_index - history_samples, window.start_index):
        derivative_slice = slice(index - derivative_frames + 1, index + 1)
        velocity = causal_polynomial_derivative(
            arrays["uav_position_m"][derivative_slice],
            arrays["time_s"][derivative_slice],
        )
        features.append(
            np.concatenate(
                (
                    arrays["command_position_m"][index]
                    - arrays["uav_position_m"][index],
                    arrays["command_velocity_mps"][index] - velocity,
                    arrays["command_acceleration_mps2"][index],
                )
            )
        )
    return ResidualHistoryState(
        torch.as_tensor(np.stack(features)[None], dtype=dtype, device=device)
    )
