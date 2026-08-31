from __future__ import annotations

import torch

from simulator.parameters import CableParameters, SimulatorParameters
from simulator.simulator import CoupledSimulator

from ._common import SETTINGS, command_sequence


def test_dynamic_rollout_has_finite_nonzero_ei_and_cb_gradients() -> None:
    EI = torch.tensor(2.0e-6, dtype=torch.float64, requires_grad=True)
    Cb = torch.tensor(3.872983346207417e-8, dtype=torch.float64, requires_grad=True)
    parameters = SimulatorParameters(
        cable=CableParameters(EI=EI, Cb=Cb),
        uav=SETTINGS.parameters.uav,
    )
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        parameters,
        dt_s=SETTINGS.dt_s,
        device="cpu",
    )
    initial = simulator.reset(
        torch.tensor(SETTINGS.initial_root_position_m, dtype=torch.float64)
    )
    trajectory = simulator.rollout(
        initial,
        command_sequence(24, mode="aggressive"),
        parameters,
        create_graph=True,
    )
    loss = (
        trajectory.tip_positions_m[..., 0].square().mean()
        + 0.01 * trajectory.cable_velocities_m_s[..., 1:, :].square().mean()
    )
    loss.backward()

    assert EI.grad is not None and bool(torch.isfinite(EI.grad))
    assert Cb.grad is not None and bool(torch.isfinite(Cb.grad))
    assert float(torch.abs(EI.grad)) > 0.0
    assert float(torch.abs(Cb.grad)) > 0.0
