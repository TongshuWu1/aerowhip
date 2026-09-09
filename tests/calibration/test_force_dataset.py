from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experimental_data.force_dataset import (
    DifferentiationSettings,
    build_force_dataset,
    estimate_point_force_world_n,
    local_polynomial_derivative,
    reconstruct_dder_nodes,
)
from simulator.cable import CableConfiguration


ROOT = Path(__file__).resolve().parents[2]


def _model() -> dict[str, object]:
    return json.loads((ROOT / "config/model.json").read_text(encoding="utf-8"))


def test_local_polynomial_derivative_recovers_quadratic_motion() -> None:
    time_s = np.arange(101, dtype=np.float64) * 0.01
    acceleration = np.array([1.2, -0.4, 0.8])
    velocity = np.array([-0.2, 0.7, 0.1])
    position = 0.5 * time_s[:, None] ** 2 * acceleration + time_s[:, None] * velocity
    recovered, valid = local_polynomial_derivative(
        position,
        np.ones(len(time_s), dtype=bool),
        dt_s=0.01,
        derivative_order=2,
        settings=DifferentiationSettings(window_samples=11, polynomial_order=3),
    )
    np.testing.assert_allclose(
        recovered[valid], np.broadcast_to(acceleration, (91, 3)), atol=1.0e-10
    )
    assert np.count_nonzero(valid) == 91


def test_node_reconstruction_matches_current_twelve_node_topology() -> None:
    model = _model()
    cable = CableConfiguration.from_mapping(model["cable"])
    root = np.zeros((2, 3), dtype=np.float64)
    markers = np.zeros((2, 10, 3), dtype=np.float64)
    markers[:, :, 0] = np.arange(1, 11, dtype=np.float64)
    nodes, valid = reconstruct_dder_nodes(
        root,
        np.ones(2, dtype=bool),
        markers,
        np.ones((2, 10), dtype=bool),
        cable,
    )
    assert nodes.shape == (2, 12, 3)
    assert np.all(valid)
    np.testing.assert_allclose(nodes[:, 1, 0], 0.5)
    np.testing.assert_allclose(
        nodes[:, 2:, 0], np.broadcast_to(np.arange(1, 11), (2, 10))
    )


def test_whole_system_force_balance_includes_point_and_cable_mass() -> None:
    model = _model()
    cable = CableConfiguration.from_mapping(model["cable"])
    acceleration = np.array([[0.5, -0.2, 1.0]])
    node_acceleration = np.broadcast_to(
        acceleration[:, None], (1, cable.node_count, 3)
    ).copy()
    force = estimate_point_force_world_n(
        acceleration,
        node_acceleration,
        np.array([True]),
        point_mass_kg=float(model["point_mass"]["mass_kg"]),
        cable_vertex_masses_kg=np.asarray(cable.vertex_masses_kg),
        gravity_world_m_s2=np.asarray(cable.gravity_m_s2),
    )
    total_mass = float(model["point_mass"]["mass_kg"]) + cable.total_dynamic_mass_kg
    expected = total_mass * (acceleration - np.asarray(cable.gravity_m_s2))
    np.testing.assert_allclose(force, expected)


def test_real_take_build_has_new_contract_and_honors_explicit_protected_split(tmp_path: Path) -> None:
    manifest = build_force_dataset(take_id="osc_001", output_root=tmp_path)
    assert manifest["summary"]["take_count"] == 1
    with np.load(tmp_path / "osc_001" / "take.npz", allow_pickle=False) as take:
        assert take["cable_node_position_world_m"].shape[1:] == (12, 3)
        assert take["estimated_point_force_world_n"].shape[1:] == (3,)
        assert "command_acceleration_mps2" not in take.files
        force = take["estimated_point_force_world_n"][take["force_estimate_valid"]]
        assert 1.6 < float(np.median(np.linalg.norm(force, axis=1))) < 1.9
    metadata = json.loads(
        (tmp_path / "osc_001" / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["force_estimation"]["status"] == "estimated_not_measured"
    # The user now admits all historical takes. Test the preserved protection
    # contract using an explicit historical split, without editing live roles.
    historical = json.loads((ROOT / 'data/dataset_manifest.json').read_text())
    historical['takes']['fig8vertical_002']['role'] = 'untouched_test'
    manifest_path = tmp_path / 'historical_manifest.json'
    manifest_path.write_text(json.dumps(historical))
    with pytest.raises(ValueError, match="protected_untouched_test"):
        build_force_dataset(take_id="fig8vertical_002", output_root=tmp_path, manifest_path=manifest_path)
