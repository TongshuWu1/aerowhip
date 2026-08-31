from __future__ import annotations

import math
from pathlib import Path

import torch

from simulator.parameters import SimulatorSettings
from simulator.parameters import CableParameters, SimulatorParameters
from simulator.uav.state import FullStateCommandSequence, UAVCommandSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = SimulatorSettings.load(PROJECT_ROOT / "config" / "default.json")
STABLE_CPU_TEST_PARAMETERS = SimulatorParameters(
    cable=CableParameters(EI=1.0e-5, Cb=1.0e-6),
    uav=SETTINGS.parameters.uav,
)


def command_sequence(
    step_count: int,
    *,
    mode: str,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> UAVCommandSequence:
    times = (
        torch.arange(1, step_count + 1, dtype=torch.float64, device=device)
        * SETTINGS.dt_s
    )
    positions = torch.tensor(
        SETTINGS.initial_root_position_m,
        dtype=torch.float64,
        device=device,
    ).view(1, 1, 3).expand(step_count, batch_size, 3).clone()
    velocities = torch.zeros_like(positions)
    if mode == "sinusoidal":
        amplitude = SETTINGS.debug.sinusoidal_amplitude_m
        frequency = SETTINGS.debug.sinusoidal_frequency_hz
    elif mode == "aggressive":
        amplitude = SETTINGS.debug.aggressive_amplitude_m
        frequency = SETTINGS.debug.aggressive_frequency_hz
    elif mode == "static":
        amplitude = 0.0
        frequency = 0.0
    else:
        raise ValueError(f"Unknown debug mode: {mode}")
    if frequency:
        omega = 2.0 * math.pi * frequency
        positions[:, :, 0] += amplitude * torch.sin(omega * times)[:, None]
        velocities[:, :, 0] = amplitude * omega * torch.cos(omega * times)[:, None]
    return UAVCommandSequence(positions, velocities)


def fullstate_command_sequence(
    step_count: int,
    *,
    mode: str,
    batch_size: int = 1,
    device: torch.device | str = "cpu",
) -> FullStateCommandSequence:
    """Create kinematically consistent research/debug FullState commands."""

    times = (
        torch.arange(1, step_count + 1, dtype=torch.float64, device=device)
        * SETTINGS.dt_s
    )
    positions = torch.tensor(
        SETTINGS.initial_uav_position_m,
        dtype=torch.float64,
        device=device,
    ).view(1, 1, 3).expand(step_count, batch_size, 3).clone()
    velocities = torch.zeros_like(positions)
    accelerations = torch.zeros_like(positions)
    orientations = torch.zeros(
        (step_count, batch_size, 4), dtype=torch.float64, device=device
    )
    orientations[..., 3] = 1.0
    angular_velocities = torch.zeros_like(positions)
    if mode in ("sinusoidal", "aggressive", "coupled"):
        if mode == "sinusoidal":
            amplitude = SETTINGS.debug.sinusoidal_amplitude_m
            frequency = SETTINGS.debug.sinusoidal_frequency_hz
        else:
            amplitude = SETTINGS.debug.aggressive_amplitude_m
            frequency = SETTINGS.debug.aggressive_frequency_hz
        omega = 2.0 * math.pi * frequency
        phase = omega * times
        positions[:, :, 0] += amplitude * torch.sin(phase)[:, None]
        velocities[:, :, 0] = amplitude * omega * torch.cos(phase)[:, None]
        accelerations[:, :, 0] = -amplitude * omega**2 * torch.sin(phase)[:, None]
    elif mode not in ("hover", "attitude"):
        raise ValueError(f"Unknown FullState debug mode: {mode}")
    if mode in ("attitude", "coupled"):
        amplitude_rad = math.radians(SETTINGS.debug.attitude_amplitude_deg)
        angular_frequency = 2.0 * math.pi * SETTINGS.debug.attitude_frequency_hz
        angle = amplitude_rad * torch.sin(angular_frequency * times)
        angle_rate = (
            amplitude_rad * angular_frequency * torch.cos(angular_frequency * times)
        )
        orientations[..., 1] = torch.sin(0.5 * angle)[:, None]
        orientations[..., 3] = torch.cos(0.5 * angle)[:, None]
        angular_velocities[..., 1] = angle_rate[:, None]
    return FullStateCommandSequence(
        positions,
        velocities,
        accelerations,
        orientations,
        angular_velocities,
    )
