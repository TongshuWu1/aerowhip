from __future__ import annotations

import json
from pathlib import Path

import torch

from simulator import ForceControlledPointCable


ROOT = Path(__file__).resolve().parents[2]


def _model(*, external_drag_s_inv=None) -> tuple[dict[str, object], ForceControlledPointCable]:
    payload = json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))
    if external_drag_s_inv is not None:
        payload['cable']['external_drag_s_inv'] = external_drag_s_inv
    return payload, ForceControlledPointCable.from_mapping(payload)


def test_point_controller_applies_force_only_at_shared_node_zero() -> None:
    _, model = _model()
    command = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float64)
    node_forces = model.controller.node_forces(command)
    assert node_forces.shape == (1, 12, 3)
    torch.testing.assert_close(node_forces[:, 0], command)
    torch.testing.assert_close(node_forces[:, 1:], torch.zeros_like(node_forces[:, 1:]))


def test_zero_command_produces_rigid_free_fall() -> None:
    payload, model = _model(external_drag_s_inv=0.)
    state = model.hanging_state(torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float64))
    dt = float(payload["simulation"]["dt_s"])
    for _ in range(20):
        result = model.step(state, torch.zeros((1, 3), dtype=torch.float64), dt)
        state = result.state
    expected_velocity = 20.0 * dt * torch.tensor(
        payload["cable"]["gravity_m_s2"], dtype=torch.float64
    )
    # Repeated constraint projection at twelve substeps accumulates roundoff.
    torch.testing.assert_close(state.velocities_m_s[0, 0], expected_velocity, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(
        result.forces.effective_cable_reaction_on_point_world_n,
        torch.zeros((1, 3), dtype=torch.float64),
        atol=1e-10,
        rtol=0.0,
    )


def test_total_system_weight_holds_a_hanging_cable_at_rest() -> None:
    payload, model = _model(external_drag_s_inv=0.)
    state = model.hanging_state(torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float64))
    initial_positions = state.positions_m.clone()
    hover = model.hover_force_world_n(dtype=torch.float64, device="cpu")[None]
    dt = float(payload["simulation"]["dt_s"])
    for _ in range(20):
        result = model.step(state, hover, dt)
        state = result.state
    torch.testing.assert_close(state.positions_m, initial_positions, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(state.velocities_m_s, torch.zeros_like(state.velocities_m_s), atol=1e-12, rtol=0.0)
    gravity = torch.tensor(payload["cable"]["gravity_m_s2"], dtype=torch.float64)
    expected_cable_reaction = model.cable_mass_kg * gravity
    torch.testing.assert_close(
        result.forces.effective_cable_reaction_on_point_world_n[0],
        expected_cable_reaction,
        atol=1e-11,
        rtol=0.0,
    )
    reconstructed_net = (
        result.forces.commanded_force_world_n
        + result.forces.gravity_force_on_point_world_n
        + result.forces.effective_cable_reaction_on_point_world_n
    )
    torch.testing.assert_close(
        reconstructed_net,
        result.forces.net_force_on_point_world_n,
        atol=1e-12,
        rtol=0.0,
    )


def test_horizontal_command_changes_only_total_system_momentum() -> None:
    payload, model = _model(external_drag_s_inv=0.)
    state = model.hanging_state(torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float64))
    command = model.hover_force_world_n(dtype=torch.float64, device="cpu")
    command[0] = 1.0
    dt = float(payload["simulation"]["dt_s"])
    state = model.step(state, command[None], dt).state
    masses = torch.tensor(
        model.cable_configuration.vertex_masses_kg, dtype=torch.float64
    )
    masses[0] += model.point_mass_kg
    center_of_mass_velocity = (
        state.velocities_m_s[0] * masses[:, None]
    ).sum(dim=0) / model.system_mass_kg
    expected = torch.tensor(
        [dt / model.system_mass_kg, 0.0, 0.0], dtype=torch.float64
    )
    torch.testing.assert_close(
        center_of_mass_velocity, expected, atol=1e-12, rtol=0.0
    )


def test_runtime_transition_matches_validated_transition_on_cpu() -> None:
    payload, model = _model()
    state = model.hanging_state(torch.tensor([[0.0, 0.0, 1.5]], dtype=torch.float64))
    command = model.hover_force_world_n(dtype=torch.float64, device="cpu")[None]
    command[:, 0] = 0.2
    dt = float(payload["simulation"]["dt_s"])
    validated = model.step(state, command, dt)
    runtime = model.step_runtime(state, command, dt)
    torch.testing.assert_close(
        runtime.state.positions_m, validated.state.positions_m, atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        runtime.state.velocities_m_s,
        validated.state.velocities_m_s,
        atol=1e-12,
        rtol=0.0,
    )
