from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from simulator.gui.snapshot import create_visualization_snapshot
from simulator.gui.viewer_3d import pose_transform_matrix_xyzw
from simulator.simulator import CoupledSimulator
from simulator.uav.model import FullStateUAVModel

from ._common import SETTINGS, fullstate_command_sequence


@pytest.mark.parametrize(
    ("axis", "expected"),
    (
        (
            0,
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
            ),
        ),
        (
            1,
            np.array(
                [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
            ),
        ),
        (
            2,
            np.array(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        ),
    ),
)
def test_viewer_pose_transform_matches_known_90_degree_rotations(
    axis: int,
    expected: np.ndarray,
) -> None:
    quaternion = np.zeros(4)
    quaternion[axis] = math.sin(0.25 * math.pi)
    quaternion[3] = math.cos(0.25 * math.pi)
    position = np.array((0.4, -0.2, 1.3))
    transform = pose_transform_matrix_xyzw(position, quaternion)
    np.testing.assert_allclose(transform[:3, :3], expected, atol=2.0e-15)
    np.testing.assert_allclose(transform[:3, 3], position, atol=0.0)
    np.testing.assert_allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=0.0)


def test_identity_viewer_pose_transform() -> None:
    transform = pose_transform_matrix_xyzw(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(transform, np.array(
        [[1.0, 0.0, 0.0, 1.0],
         [0.0, 1.0, 0.0, 2.0],
         [0.0, 0.0, 1.0, 3.0],
         [0.0, 0.0, 0.0, 1.0]]
    ), atol=0.0)


def test_visualization_snapshot_is_compact_immutable_and_physics_aligned() -> None:
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
        torch.tensor(SETTINGS.initial_uav_position_m, dtype=torch.float64),
        uav_orientation_xyzw=torch.tensor(
            SETTINGS.initial_uav_orientation_xyzw, dtype=torch.float64
        ),
    )
    command = fullstate_command_sequence(1, mode="coupled").command_at(0)
    state = simulator.step(command, create_graph=False)
    positions_before = state.cable.positions_m.clone()
    velocities_before = state.cable.velocities_m_s.clone()
    uav_position_before = state.uav.position_m.clone()
    uav_orientation_before = state.uav.orientation_xyzw.clone()
    snapshot = create_visualization_snapshot(simulator, state, command)
    assert snapshot.cable_positions_m.shape == (12, 3)
    assert not snapshot.cable_positions_m.flags.writeable
    assert not snapshot.uav_position_m.flags.writeable
    assert not snapshot.uav_orientation_xyzw.flags.writeable
    assert not snapshot.attachment_position_m.flags.writeable
    np.testing.assert_allclose(
        snapshot.attachment_position_m,
        state.cable.positions_m[0, 0].detach().cpu().numpy(),
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        snapshot.cable_positions_m[:2],
        state.cable.positions_m[0, :2].detach().cpu().numpy(),
        atol=0.0,
    )
    np.testing.assert_allclose(
        snapshot.commanded_uav_position_m,
        command.position_m[0].detach().cpu().numpy(),
        atol=0.0,
    )
    np.testing.assert_allclose(
        snapshot.commanded_uav_orientation_xyzw,
        command.orientation_xyzw[0].detach().cpu().numpy(),
        atol=0.0,
    )
    assert not hasattr(snapshot, "cable_velocities_m_s")
    assert not hasattr(snapshot, "uav_velocity_m_s")
    assert not hasattr(snapshot, "uav_angular_velocity_world_rad_s")
    torch.testing.assert_close(state.cable.positions_m, positions_before)
    torch.testing.assert_close(state.cable.velocities_m_s, velocities_before)
    torch.testing.assert_close(state.uav.position_m, uav_position_before)
    torch.testing.assert_close(state.uav.orientation_xyzw, uav_orientation_before)
