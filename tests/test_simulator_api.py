from __future__ import annotations

import pytest
import torch

from simulator.simulator import CoupledSimulator

from ._common import SETTINGS, STABLE_CPU_TEST_PARAMETERS, command_sequence


def test_reset_step_and_rollout_shapes_and_state_ownership() -> None:
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        STABLE_CPU_TEST_PARAMETERS,
        dt_s=SETTINGS.dt_s,
        device="cpu",
    )
    root = torch.tensor(SETTINGS.initial_root_position_m, dtype=torch.float64).repeat(2, 1)
    initial = simulator.reset(root)
    commands = command_sequence(5, mode="sinusoidal", batch_size=2)
    trajectory = simulator.rollout(initial, commands, create_graph=False)

    assert trajectory.uav_positions_m.shape == (6, 2, 3)
    assert trajectory.cable_positions_m.shape == (
        6,
        2,
        SETTINGS.cable_configuration.node_count,
        3,
    )
    assert simulator.state is initial
    next_state = simulator.step(commands.command_at(0))
    assert next_state.time_s == pytest.approx(SETTINGS.dt_s)
    assert simulator.state is next_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_batched_rollout_remains_available() -> None:
    device = torch.device("cuda")
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        SETTINGS.parameters,
        dt_s=SETTINGS.dt_s,
        device=device,
    )
    root = torch.tensor(
        SETTINGS.initial_root_position_m, dtype=torch.float64, device=device
    ).repeat(4, 1)
    initial = simulator.reset(root)
    trajectory = simulator.rollout(
        initial,
        command_sequence(5, mode="aggressive", batch_size=4, device=device),
        create_graph=False,
    )
    assert trajectory.cable_positions_m.is_cuda
    assert bool(torch.isfinite(trajectory.cable_positions_m).all())
