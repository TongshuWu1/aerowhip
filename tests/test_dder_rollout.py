from __future__ import annotations

import torch

from simulator.simulator import CoupledSimulator

from ._common import SETTINGS, STABLE_CPU_TEST_PARAMETERS, command_sequence


def _run(mode: str, steps: int) -> tuple[CoupledSimulator, object]:
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        STABLE_CPU_TEST_PARAMETERS,
        dt_s=SETTINGS.dt_s,
        device="cpu",
    )
    initial = simulator.reset(
        torch.tensor(SETTINGS.initial_root_position_m, dtype=torch.float64)
    )
    return simulator, simulator.rollout(
        initial,
        command_sequence(steps, mode=mode),
        create_graph=False,
    )


def test_sinusoidal_root_response_is_dynamic_and_deterministic() -> None:
    first_simulator, first = _run("sinusoidal", 40)
    _second_simulator, second = _run("sinusoidal", 40)
    torch.testing.assert_close(
        first.cable_positions_m,
        second.cable_positions_m,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.isfinite(first.cable_positions_m).all()
    assert float(torch.amax(torch.abs(first.cable_velocities_m_s))) > 0.05
    relative_tip_x = first.tip_positions_m[..., 0] - first.uav_positions_m[..., 0]
    assert float(torch.amax(torch.abs(relative_tip_x))) > 1.0e-4
    assert float(
        torch.amax(first_simulator.cable_model.maximum_segment_error_m(
            first.cable_positions_m.reshape(
                -1, SETTINGS.cable_configuration.node_count, 3
            )
        ))
    ) < 1.0e-8


def test_aggressive_reversal_remains_stable_and_moves_free_tip() -> None:
    simulator, trajectory = _run("aggressive", 60)
    assert torch.isfinite(trajectory.cable_positions_m).all()
    assert torch.isfinite(trajectory.cable_velocities_m_s).all()
    assert float(torch.amax(torch.abs(trajectory.tip_positions_m[..., 0]))) > 0.01
    assert float(torch.amax(torch.linalg.vector_norm(
        trajectory.cable_velocities_m_s[..., -1, :], dim=-1
    ))) > 0.1
    assert float(
        torch.amax(simulator.cable_model.maximum_segment_error_m(
            trajectory.cable_positions_m.reshape(
                -1, SETTINGS.cable_configuration.node_count, 3
            )
        ))
    ) < 1.0e-7
