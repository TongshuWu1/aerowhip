from __future__ import annotations

import pytest
import torch

from simulator.parameters import (
    CableParameters,
    SimulatorParameters,
    UAVResponseParameters,
)
from simulator.simulator import CoupledSimulator
from simulator.uav.model import FullStateUAVModel

from ._common import SETTINGS, fullstate_command_sequence


def _simulator(parameters: SimulatorParameters = SETTINGS.parameters) -> CoupledSimulator:
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        parameters,
        dt_s=SETTINGS.dt_s,
        device="cuda",
        uav_model=FullStateUAVModel(),
        attachment_offset_body_m=SETTINGS.attachment_offset_body_m,
        attachment_tangent_body=SETTINGS.attachment_tangent_body,
    )
    simulator.reset(
        torch.tensor(SETTINGS.initial_uav_position_m, device="cuda"),
        uav_orientation_xyzw=torch.tensor(
            SETTINGS.initial_uav_orientation_xyzw, device="cuda"
        ),
    )
    return simulator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_clamped_cuda_uses_float32_production_runtime_and_exact_boundary() -> None:
    simulator = _simulator()
    state = simulator.step(
        fullstate_command_sequence(1, mode="coupled", device="cuda").command_at(0)
    )

    assert state.cable.positions_m.dtype == torch.float32
    assert simulator.last_dder_execution == "production_cuda_fused"
    boundary = simulator.root_boundary.evaluate(
        state.uav, SETTINGS.cable_configuration.rest_lengths_m[0]
    )
    torch.testing.assert_close(
        state.cable.positions_m[:, :2],
        boundary.prescribed_positions_m,
        atol=2.0e-7,
        rtol=0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fused_clamped_runtime_matches_pytorch_pcg32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused = _simulator()
    pytorch = _simulator()
    commands = fullstate_command_sequence(4, mode="coupled", device="cuda")
    for index in range(commands.step_count):
        monkeypatch.setenv("CABLE_TWIN_FUSED_FIXED_DAMPING", "0")
        monkeypatch.setenv("CABLE_TWIN_FUSED_FIXED_PROJECTION", "0")
        reference = pytorch.step(commands.command_at(index))
        monkeypatch.setenv("CABLE_TWIN_FUSED_FIXED_DAMPING", "1")
        monkeypatch.setenv("CABLE_TWIN_FUSED_FIXED_PROJECTION", "1")
        production = fused.step(commands.command_at(index))

    torch.testing.assert_close(
        production.cable.positions_m,
        reference.cable.positions_m,
        atol=1.0e-6,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        production.cable.velocities_m_s,
        reference.cable.velocities_m_s,
        atol=7.0e-5,
        rtol=2.0e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_differentiable_production_runtime_reaches_all_seven_parameters() -> None:
    names_and_values = (
        ("K_p", 16.0),
        ("K_v", 8.0),
        ("k_a", 1.0),
        ("K_R", 25.0),
        ("K_omega", 10.0),
        ("EI", 2.0e-6),
        ("Cb", 3.872983346207417e-8),
    )
    values = {
        name: torch.tensor(value, device="cuda", requires_grad=True)
        for name, value in names_and_values
    }
    parameters = SimulatorParameters(
        CableParameters(values["EI"], values["Cb"]),
        UAVResponseParameters(
            values["K_p"],
            values["K_v"],
            values["k_a"],
            values["K_R"],
            values["K_omega"],
        ),
    )
    simulator = _simulator(parameters)
    trajectory = simulator.rollout(
        simulator.state,
        fullstate_command_sequence(3, mode="coupled", device="cuda"),
        parameters,
        create_graph=True,
    )
    loss = (
        trajectory.tip_positions_m[..., 0].square().mean()
        + 0.01 * trajectory.cable_velocities_m_s[..., 2:, :].square().mean()
        + 0.1 * trajectory.uav_positions_m[..., 0].square().mean()
        + 0.1 * trajectory.uav_orientations_xyzw[..., 1].square().mean()
    )
    loss.backward()

    assert (
        simulator.last_dder_execution
        == "production_cuda_differentiable_cholesky"
    )
    for value in values.values():
        assert value.grad is not None
        assert bool(torch.isfinite(value.grad))
