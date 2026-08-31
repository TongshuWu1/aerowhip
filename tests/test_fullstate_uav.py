from __future__ import annotations

import math

import pytest
import torch

from simulator.coupling.attachment import rigid_attachment_state
from simulator.coupling.root_boundary import ClampedRootBoundary, PivotRootBoundary
from simulator.parameters import (
    CableParameters,
    SimulatorParameters,
    UAVResponseParameters,
)
from simulator.simulator import CoupledSimulator
from simulator.uav.model import FullStateUAVModel
from simulator.uav.state import FullStateCommand, UAVState

from ._common import SETTINGS, fullstate_command_sequence


def _fullstate_simulator(
    *,
    device: torch.device | str = "cpu",
    parameters: SimulatorParameters | None = None,
    offset: tuple[float, float, float] | None = None,
    tangent: tuple[float, float, float] | None = None,
) -> CoupledSimulator:
    return CoupledSimulator(
        SETTINGS.cable_configuration,
        SETTINGS.parameters if parameters is None else parameters,
        dt_s=SETTINGS.dt_s,
        device=device,
        uav_model=FullStateUAVModel(),
        attachment_offset_body_m=(
            SETTINGS.attachment_offset_body_m if offset is None else offset
        ),
        attachment_tangent_body=(
            SETTINGS.attachment_tangent_body if tangent is None else tangent
        ),
    )


def _initial(simulator: CoupledSimulator, batch_size: int = 1):
    return simulator.reset(
        torch.tensor(
            SETTINGS.initial_uav_position_m,
            dtype=torch.float64,
            device=simulator.device,
        ).repeat(batch_size, 1),
        uav_orientation_xyzw=torch.tensor(
            SETTINGS.initial_uav_orientation_xyzw,
            dtype=torch.float64,
            device=simulator.device,
        ).repeat(batch_size, 1),
    )


def test_fullstate_hover_is_stationary_and_uses_clamped_boundary() -> None:
    simulator = _fullstate_simulator()
    assert isinstance(simulator.root_boundary, ClampedRootBoundary)
    initial = _initial(simulator)
    trajectory = simulator.rollout(
        initial,
        fullstate_command_sequence(20, mode="hover"),
        create_graph=False,
    )
    torch.testing.assert_close(
        trajectory.uav_positions_m,
        trajectory.uav_positions_m[:1].expand_as(trajectory.uav_positions_m),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        trajectory.uav_velocities_m_s,
        torch.zeros_like(trajectory.uav_velocities_m_s),
        atol=0.0,
        rtol=0.0,
    )
    quaternion_norm = torch.linalg.vector_norm(
        trajectory.uav_orientations_xyzw, dim=-1
    )
    torch.testing.assert_close(
        quaternion_norm, torch.ones_like(quaternion_norm), atol=1.0e-14, rtol=0.0
    )


def test_uav_only_stage_a_path_matches_coupled_uav_trajectory_exactly() -> None:
    simulator = _fullstate_simulator()
    initial = _initial(simulator)
    commands = fullstate_command_sequence(20, mode="coupled")
    coupled = simulator.rollout(initial, commands, create_graph=False)
    uav_only = simulator.rollout_uav_only(initial.uav, commands)
    torch.testing.assert_close(uav_only.positions_m, coupled.uav_positions_m, atol=0.0, rtol=0.0)
    torch.testing.assert_close(uav_only.velocities_m_s, coupled.uav_velocities_m_s, atol=0.0, rtol=0.0)
    torch.testing.assert_close(uav_only.orientations_xyzw, coupled.uav_orientations_xyzw, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        uav_only.angular_velocities_world_rad_s,
        coupled.uav_angular_velocities_world_rad_s,
        atol=0.0,
        rtol=0.0,
    )


def test_translation_feedforward_constructs_roll_pitch_from_acceleration() -> None:
    model = FullStateUAVModel()
    state = UAVState(
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
    )
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    command = FullStateCommand(
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        identity,
        torch.zeros(1, 3, dtype=torch.float64),
    )
    parameters = UAVResponseParameters(16.0, 8.0, 1.0, 25.0, 10.0)
    next_state = model.step(state, command, 0.01, parameters)
    # Translation now responds through the semi-implicitly updated actual
    # attitude, so the first horizontal step is positive but smaller than the
    # direct-acceleration request while the vehicle is still tilting.
    assert 0.0 < float(next_state.velocity_m_s[0, 0]) < 0.01
    assert float(next_state.angular_velocity_world_rad_s[0, 1]) > 0.0
    assert float(next_state.orientation_xyzw[0, 1]) > 0.0
    torch.testing.assert_close(
        torch.linalg.vector_norm(next_state.orientation_xyzw, dim=-1),
        torch.ones(1, dtype=torch.float64),
        atol=1.0e-14,
        rtol=0.0,
    )


def test_command_quaternion_roll_pitch_is_not_tracked() -> None:
    model = FullStateUAVModel()
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    state = UAVState(torch.zeros(1, 3, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64))
    nonstock_pitch_quaternion = torch.tensor(
        [[0.0, math.sin(0.2), 0.0, math.cos(0.2)]], dtype=torch.float64
    )
    command = FullStateCommand(
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        nonstock_pitch_quaternion,
        torch.zeros(1, 3, dtype=torch.float64),
    )
    next_state = model.step(
        state, command, 0.01, UAVResponseParameters(16.0, 8.0, 1.0, 25.0, 10.0)
    )
    torch.testing.assert_close(next_state.orientation_xyzw, identity, atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        next_state.angular_velocity_world_rad_s,
        torch.zeros_like(next_state.angular_velocity_world_rad_s),
        atol=0.0,
        rtol=0.0,
    )


def test_body_frame_omega_command_is_rotated_for_world_state_kinematics() -> None:
    model = FullStateUAVModel()
    half_yaw = 0.25 * math.pi
    yaw_quaternion = torch.tensor(
        [[0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]], dtype=torch.float64
    )
    state = UAVState(
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        yaw_quaternion,
        torch.zeros(1, 3, dtype=torch.float64),
    )
    command = FullStateCommand(
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.zeros(1, 3, dtype=torch.float64),
        yaw_quaternion,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
    )
    next_state = model.step(
        state, command, 0.01, UAVResponseParameters(16.0, 8.0, 1.0, 1.0, 10.0)
    )
    assert abs(float(next_state.angular_velocity_world_rad_s[0, 0])) < 1.0e-3
    assert float(next_state.angular_velocity_world_rad_s[0, 1]) > 0.09


def test_rigid_attachment_and_clamped_reset_geometry() -> None:
    half_angle = 0.25 * math.pi
    orientation = torch.tensor(
        [[0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]],
        dtype=torch.float64,
    )
    state = UAVState(
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64),
        torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64),
        orientation,
        torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float64),
    )
    edge_length = SETTINGS.cable_configuration.rest_lengths_m[0]
    rigid = rigid_attachment_state(
        state,
        torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        edge_length,
    )
    torch.testing.assert_close(
        rigid.root.position_m,
        torch.tensor([[1.0, 2.1, 3.0]], dtype=torch.float64),
        atol=1.0e-14,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rigid.root.analytic_velocity_m_s,
        torch.tensor([[-0.1, 0.2, 0.3]], dtype=torch.float64),
        atol=1.0e-14,
        rtol=0.0,
    )
    simulator = _fullstate_simulator(
        offset=(0.1, 0.0, 0.0), tangent=(1.0, 0.0, 0.0)
    )
    initial = simulator.reset(
        state.position_m,
        state.velocity_m_s,
        state.orientation_xyzw,
        state.angular_velocity_world_rad_s,
    )
    torch.testing.assert_close(
        initial.cable.positions_m[:, :2],
        torch.stack((rigid.root.position_m, rigid.first_edge_position_m), dim=1),
        atol=1.0e-14,
        rtol=0.0,
    )


def test_attitude_motion_rotates_and_preserves_first_edge_constraint() -> None:
    simulator = _fullstate_simulator()
    initial = _initial(simulator)
    trajectory = simulator.rollout(
        initial,
        fullstate_command_sequence(45, mode="attitude"),
        create_graph=False,
    )
    assert trajectory.prescribed_root_tangents_world is not None
    edge = trajectory.cable_positions_m[..., 1, :] - trajectory.cable_positions_m[..., 0, :]
    length = torch.linalg.vector_norm(edge, dim=-1)
    tangent = edge / length[..., None]
    torch.testing.assert_close(
        length,
        torch.full_like(length, SETTINGS.cable_configuration.rest_lengths_m[0]),
        atol=3.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        tangent,
        trajectory.prescribed_root_tangents_world,
        atol=2.0e-14,
        rtol=0.0,
    )
    assert float(torch.amax(torch.abs(tangent[..., 0]))) > 1.0e-3
    assert float(torch.amax(torch.abs(trajectory.tip_positions_m[..., 0]))) > 1.0e-6


def test_combined_loss_has_gradients_for_all_seven_parameters() -> None:
    values = {
        "K_p": torch.tensor(16.0, dtype=torch.float64, requires_grad=True),
        "K_v": torch.tensor(8.0, dtype=torch.float64, requires_grad=True),
        "k_a": torch.tensor(1.0, dtype=torch.float64, requires_grad=True),
        "K_R": torch.tensor(25.0, dtype=torch.float64, requires_grad=True),
        "K_omega": torch.tensor(10.0, dtype=torch.float64, requires_grad=True),
        "EI": torch.tensor(2.0e-6, dtype=torch.float64, requires_grad=True),
        "Cb": torch.tensor(
            3.872983346207417e-8, dtype=torch.float64, requires_grad=True
        ),
    }
    parameters = SimulatorParameters(
        cable=CableParameters(values["EI"], values["Cb"]),
        uav=UAVResponseParameters(
            values["K_p"],
            values["K_v"],
            values["k_a"],
            values["K_R"],
            values["K_omega"],
        ),
    )
    simulator = _fullstate_simulator(parameters=parameters)
    trajectory = simulator.rollout(
        _initial(simulator),
        fullstate_command_sequence(20, mode="coupled"),
        parameters,
        create_graph=True,
    )
    loss = (
        trajectory.uav_positions_m[..., 0].square().mean()
        + trajectory.uav_velocities_m_s[..., 0].square().mean()
        + trajectory.uav_orientations_xyzw[..., 1].square().mean()
        + trajectory.uav_angular_velocities_world_rad_s[..., 1].square().mean()
        + trajectory.tip_positions_m[..., 0].square().mean()
        + 0.01 * trajectory.cable_velocities_m_s[..., 1:, :].square().mean()
    )
    loss.backward()
    for name, value in values.items():
        assert value.grad is not None, name
        assert bool(torch.isfinite(value.grad)), name
        assert float(torch.abs(value.grad)) > 0.0, name


def test_cable_only_loss_backpropagates_through_clamped_attitude_path() -> None:
    K_R = torch.tensor(25.0, dtype=torch.float64, requires_grad=True)
    K_omega = torch.tensor(10.0, dtype=torch.float64, requires_grad=True)
    parameters = SimulatorParameters(
        cable=SETTINGS.parameters.cable,
        uav=UAVResponseParameters(16.0, 8.0, 1.0, K_R, K_omega),
    )
    simulator = _fullstate_simulator(parameters=parameters)
    trajectory = simulator.rollout(
        _initial(simulator),
        fullstate_command_sequence(30, mode="attitude"),
        parameters,
        create_graph=True,
    )
    cable_only_loss = (
        trajectory.cable_positions_m[..., 2:, 0].square().mean()
        + 0.01 * trajectory.cable_velocities_m_s[..., 2:, :].square().mean()
    )
    cable_only_loss.backward()
    assert K_R.grad is not None and bool(torch.isfinite(K_R.grad))
    assert K_omega.grad is not None and bool(torch.isfinite(K_omega.grad))
    assert float(torch.abs(K_R.grad)) > 0.0
    assert float(torch.abs(K_omega.grad)) > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fullstate_clamped_cuda_batch_rollout() -> None:
    simulator = _fullstate_simulator(device="cuda")
    trajectory = simulator.rollout(
        _initial(simulator, batch_size=3),
        fullstate_command_sequence(5, mode="coupled", batch_size=3, device="cuda"),
        create_graph=False,
    )
    assert trajectory.cable_positions_m.shape == (6, 3, 12, 3)
    assert trajectory.cable_positions_m.is_cuda
    assert bool(torch.isfinite(trajectory.cable_positions_m).all())


def test_prescribed_model_defaults_to_legacy_pivot_boundary() -> None:
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        SETTINGS.parameters,
        dt_s=SETTINGS.dt_s,
        device="cpu",
    )
    assert isinstance(simulator.root_boundary, PivotRootBoundary)
