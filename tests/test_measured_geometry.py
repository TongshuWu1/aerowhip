from __future__ import annotations

import json
import math

import pytest
import torch

from simulator.coupling.attachment import rigid_attachment_state
from simulator.observation import OPTITRACK_CABLE_MARKER_NODE_INDICES
from simulator.simulator import CoupledSimulator
from simulator.uav.model import FullStateUAVModel
from simulator.uav.state import UAVState

from ._common import SETTINGS, PROJECT_ROOT, fullstate_command_sequence


EXPECTED_INTERVALS_M = (
    0.063,
    0.087,
    0.100,
    0.100,
    0.100,
    0.100,
    0.100,
    0.1025,
    0.100,
    0.100,
)


def _fullstate_simulator(*, device: str = "cpu") -> CoupledSimulator:
    return CoupledSimulator(
        SETTINGS.cable_configuration,
        SETTINGS.parameters,
        dt_s=SETTINGS.dt_s,
        device=device,
        uav_model=FullStateUAVModel(),
        attachment_offset_body_m=SETTINGS.attachment_offset_body_m,
        attachment_tangent_body=SETTINGS.attachment_tangent_body,
    )


def test_measured_geometry_is_the_only_length_source_of_truth() -> None:
    configuration = SETTINGS.cable_configuration
    assert configuration.marker_interval_lengths_m == EXPECTED_INTERVALS_M
    assert configuration.interval_subdivisions == (2, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    assert configuration.node_count == 12
    assert configuration.edge_count == 11
    assert configuration.length_m == pytest.approx(0.9525, rel=0.0, abs=1.0e-15)
    assert configuration.rest_lengths_m[:2] == (0.0315, 0.0315)
    assert configuration.rest_lengths_m == (
        0.0315,
        0.0315,
        0.087,
        0.100,
        0.100,
        0.100,
        0.100,
        0.100,
        0.1025,
        0.100,
        0.100,
    )
    payload = json.loads((PROJECT_ROOT / "config" / "default.json").read_text())
    assert "total_cable_length" not in payload
    assert "length_m" not in payload["cable"]


def test_identity_attachment_is_55mm_downward_and_c1_is_63mm() -> None:
    simulator = _fullstate_simulator()
    uav_position = torch.tensor((0.0, 0.0, 1.25), dtype=torch.float64)
    state = simulator.reset(
        uav_position,
        uav_orientation_xyzw=torch.tensor(
            (0.0, 0.0, 0.0, 1.0), dtype=torch.float64
        ),
    )
    connector_offset = state.cable.positions_m[0, 0] - state.uav.position_m[0]
    first_edge = state.cable.positions_m[0, 1] - state.cable.positions_m[0, 0]
    connector_to_c1 = (
        state.cable.positions_m[0, 2] - state.cable.positions_m[0, 0]
    )
    torch.testing.assert_close(
        connector_offset,
        torch.tensor((0.0, 0.0, -0.055), dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(connector_offset),
        torch.tensor(0.055, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(first_edge),
        torch.tensor(0.0315, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(connector_to_c1),
        torch.tensor(0.063, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        first_edge / torch.linalg.vector_norm(first_edge),
        torch.tensor(SETTINGS.attachment_tangent_body, dtype=torch.float64),
        atol=1.0e-15,
        rtol=0.0,
    )


def test_body_downward_offset_and_tangent_rotate_with_the_same_quaternion() -> None:
    half_angle = 0.25 * math.pi
    orientation = torch.tensor(
        ((math.sin(half_angle), 0.0, 0.0, math.cos(half_angle)),),
        dtype=torch.float64,
    )
    uav = UAVState(
        torch.tensor(((0.3, -0.2, 1.1),), dtype=torch.float64),
        torch.zeros((1, 3), dtype=torch.float64),
        orientation,
        torch.zeros((1, 3), dtype=torch.float64),
    )
    rigid = rigid_attachment_state(
        uav,
        SETTINGS.attachment_offset_body_m,
        SETTINGS.attachment_tangent_body,
        SETTINGS.cable_configuration.rest_lengths_m[0],
    )
    offset_world = rigid.root.position_m - uav.position_m
    torch.testing.assert_close(
        offset_world,
        torch.tensor(((0.0, 0.055, 0.0),), dtype=torch.float64),
        atol=2.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rigid.tangent_world,
        torch.tensor(((0.0, 1.0, 0.0),), dtype=torch.float64),
        atol=2.0e-15,
        rtol=0.0,
    )
    torch.testing.assert_close(
        rigid.first_edge_position_m - rigid.root.position_m,
        0.0315 * rigid.tangent_world,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_total_bare_and_marker_masses_are_preserved_and_redistributed() -> None:
    configuration = SETTINGS.cable_configuration
    masses = list(configuration.vertex_masses_kg)
    for node_index, marker_mass in zip(
        configuration.marker_node_indices[1:],
        configuration.moving_marker_masses_kg,
        strict=True,
    ):
        masses[node_index] -= marker_mass
    assert sum(masses) == pytest.approx(0.007, rel=0.0, abs=2.0e-18)
    assert sum(configuration.moving_marker_masses_kg) == pytest.approx(
        0.00909091, rel=0.0, abs=1.0e-15
    )
    assert sum(configuration.vertex_masses_kg) == pytest.approx(
        0.01609091, rel=0.0, abs=1.0e-15
    )
    assert configuration.bare_cable_mass_kg / configuration.length_m == pytest.approx(
        0.007349081364829396, rel=0.0, abs=1.0e-15
    )
    assert configuration.vertex_masses_kg[0] == pytest.approx(
        0.00011574803149606299, rel=0.0, abs=1.0e-15
    )
    assert configuration.vertex_masses_kg[1] == pytest.approx(
        0.00023149606299212598, rel=0.0, abs=1.0e-15
    )


def test_marker_mapping_and_cpu_rollout_health_remain_valid() -> None:
    assert OPTITRACK_CABLE_MARKER_NODE_INDICES == tuple(range(2, 12))
    assert SETTINGS.cable_configuration.marker_node_indices[1:] == tuple(range(2, 12))
    simulator = _fullstate_simulator()
    initial = simulator.reset(
        torch.tensor(SETTINGS.initial_uav_position_m, dtype=torch.float64),
        uav_orientation_xyzw=torch.tensor(
            SETTINGS.initial_uav_orientation_xyzw, dtype=torch.float64
        ),
    )
    trajectory = simulator.rollout(
        initial,
        fullstate_command_sequence(8, mode="coupled"),
        create_graph=False,
    )
    assert bool(torch.isfinite(trajectory.cable_positions_m).all())
    assert bool(torch.isfinite(trajectory.cable_velocities_m_s).all())
    maximum_error = simulator.cable_model.maximum_segment_error_m(
        trajectory.cable_positions_m.reshape(-1, 12, 3)
    )
    assert float(torch.amax(maximum_error)) < 1.0e-8
