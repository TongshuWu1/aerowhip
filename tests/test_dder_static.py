from __future__ import annotations

import torch

from simulator.simulator import CoupledSimulator

from ._common import SETTINGS, STABLE_CPU_TEST_PARAMETERS, command_sequence


def test_static_hanging_cable_remains_finite_and_constrained() -> None:
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        STABLE_CPU_TEST_PARAMETERS,
        dt_s=SETTINGS.dt_s,
        device="cpu",
    )
    initial = simulator.reset(
        torch.tensor(SETTINGS.initial_root_position_m, dtype=torch.float64)
    )
    trajectory = simulator.rollout(
        initial,
        command_sequence(30, mode="static"),
        create_graph=False,
    )

    assert torch.isfinite(trajectory.cable_positions_m).all()
    assert torch.isfinite(trajectory.cable_velocities_m_s).all()
    torch.testing.assert_close(
        trajectory.cable_positions_m[:, :, 0],
        trajectory.uav_positions_m,
        atol=0.0,
        rtol=0.0,
    )
    error = simulator.cable_model.maximum_segment_error_m(
        trajectory.cable_positions_m.reshape(
            -1, SETTINGS.cable_configuration.node_count, 3
        )
    )
    assert float(torch.amax(error)) < 1.0e-10
    assert float(trajectory.tip_positions_m[-1, 0, 2]) < float(
        trajectory.uav_positions_m[-1, 0, 2]
    )
