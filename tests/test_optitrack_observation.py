from __future__ import annotations

from dataclasses import fields

import torch

from simulator.observation import (
    OPTITRACK_CABLE_MARKER_NODE_INDICES,
    OptiTrackObservation,
    observe_simulator_state,
    observe_simulator_trajectory,
)
from simulator.simulator import CoupledSimulator
from simulator.uav.model import FullStateUAVModel

from ._common import SETTINGS, fullstate_command_sequence


def _simulator(batch_size: int = 1) -> tuple[CoupledSimulator, object]:
    simulator = CoupledSimulator(
        SETTINGS.cable_configuration,
        SETTINGS.parameters,
        dt_s=SETTINGS.dt_s,
        device="cpu",
        uav_model=FullStateUAVModel(),
        attachment_offset_body_m=SETTINGS.attachment_offset_body_m,
        attachment_tangent_body=SETTINGS.attachment_tangent_body,
    )
    state = simulator.reset(
        torch.tensor(SETTINGS.initial_uav_position_m, dtype=torch.float64).repeat(
            batch_size, 1
        ),
        uav_orientation_xyzw=torch.tensor(
            SETTINGS.initial_uav_orientation_xyzw, dtype=torch.float64
        ).repeat(batch_size, 1),
    )
    return simulator, state


def test_optitrack_state_observation_has_only_pose_and_ten_moving_markers() -> None:
    _sim, state = _simulator(batch_size=2)
    observation = observe_simulator_state(state)  # type: ignore[arg-type]
    assert tuple(item.name for item in fields(OptiTrackObservation)) == (
        "uav_position_m",
        "uav_orientation_xyzw",
        "cable_marker_positions_m",
    )
    assert observation.uav_position_m.shape == (2, 3)
    assert observation.uav_orientation_xyzw.shape == (2, 4)
    assert observation.cable_marker_positions_m.shape == (2, 10, 3)
    assert OPTITRACK_CABLE_MARKER_NODE_INDICES == tuple(range(2, 12))
    assert SETTINGS.cable_configuration.marker_node_indices[1:] == (
        OPTITRACK_CABLE_MARKER_NODE_INDICES
    )
    torch.testing.assert_close(observation.uav_position_m, state.uav.position_m)
    torch.testing.assert_close(
        observation.uav_orientation_xyzw,
        state.uav.orientation_xyzw,
    )
    torch.testing.assert_close(
        observation.cable_marker_positions_m,
        state.cable.positions_m[:, OPTITRACK_CABLE_MARKER_NODE_INDICES],
    )
    assert not hasattr(observation, "uav_velocity_m_s")
    assert not hasattr(observation, "uav_angular_velocity_world_rad_s")
    assert not hasattr(observation, "cable_velocities_m_s")


def test_observation_is_deterministic_and_does_not_alias_or_mutate_state() -> None:
    _sim, state = _simulator()
    original_position = state.uav.position_m.clone()
    original_markers = state.cable.positions_m[
        :, OPTITRACK_CABLE_MARKER_NODE_INDICES
    ].clone()
    first = observe_simulator_state(state)  # type: ignore[arg-type]
    second = observe_simulator_state(state)  # type: ignore[arg-type]
    torch.testing.assert_close(first.uav_position_m, second.uav_position_m)
    torch.testing.assert_close(
        first.cable_marker_positions_m,
        second.cable_marker_positions_m,
    )
    first.uav_position_m.add_(3.0)
    first.cable_marker_positions_m.zero_()
    torch.testing.assert_close(state.uav.position_m, original_position)
    torch.testing.assert_close(
        state.cable.positions_m[:, OPTITRACK_CABLE_MARKER_NODE_INDICES],
        original_markers,
    )


def test_batched_trajectory_observation_preserves_time_and_batch_axes() -> None:
    simulator, state = _simulator(batch_size=3)
    trajectory = simulator.rollout(
        state,  # type: ignore[arg-type]
        fullstate_command_sequence(
            4,
            mode="coupled",
            batch_size=3,
        ),
        create_graph=False,
    )
    observation = observe_simulator_trajectory(trajectory)
    assert observation.uav_position_m.shape == (5, 3, 3)
    assert observation.uav_orientation_xyzw.shape == (5, 3, 4)
    assert observation.cable_marker_positions_m.shape == (5, 3, 10, 3)
    torch.testing.assert_close(
        observation.cable_marker_positions_m,
        trajectory.cable_positions_m[
            ..., OPTITRACK_CABLE_MARKER_NODE_INDICES, :
        ],
    )
