"""Physical property tests that do not require Isaac, CUDA or Newton."""

import json
from pathlib import Path
import numpy as np
import pytest
from isaac_simulation.rod_material import (
    rod_coefficients,
    rod_mass_properties,
    aerodynamic_loads,
)


def cable():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "config/isaac_physx/rig_153g_paracord.json"
        ).read_text()
    )["cable"]


def test_resolution_changes_joint_coefficients_not_material():
    c = cable()
    a = rod_coefficients(c)
    c["segments"] *= 2
    b = rod_coefficients(c)
    for key in a:
        assert b[key] == pytest.approx(2 * a[key])


def test_marker_com_and_inertia_preserve_global_mass_moments():
    c = cable()
    mass, com, inertia, _ = rod_mass_properties(c)
    length = c["length_m"] / c["segments"]
    centers = (np.arange(len(mass)) + 0.5) * length + com
    expected_first = c["bare_mass_kg"] * c["length_m"] / 2 + c["marker_mass_kg"] * sum(
        c["marker_distances_m"]
    )
    assert mass.sum() == pytest.approx(0.018)
    assert np.sum(mass * centers) == pytest.approx(expected_first)
    # Inertia about the attachment X axis from exact distributed bare cord + spheres.
    expected = c["bare_mass_kg"] * (
        c["length_m"] ** 2 / 3 + (c["diameter_m"] / 2) ** 2 / 4
    )
    expected += sum(
        c["marker_mass_kg"] * (d * d + 0.4 * c["marker_radius_m"] ** 2)
        for d in c["marker_distances_m"]
    )
    assert np.sum(inertia[:, 0] + mass * centers**2) == pytest.approx(expected)
    assert np.all(inertia > 0)


def test_aerodynamic_power_dissipates_with_rotating_offset_mass():
    c = cable()
    n = c["segments"]
    _, com, _, _ = rod_mass_properties(c)
    r = np.tile(np.eye(3), (n, 1, 1))
    p = np.zeros((n, 3))
    rng = np.random.default_rng(2)
    v = rng.normal(size=(n, 3))
    omega = rng.normal(size=(n, 3)) * 30
    f, t = aerodynamic_loads(p, r, v, omega, com, c)
    assert np.sum(f * v + t * omega) < 0
    f, t = aerodynamic_loads(p, r, np.zeros_like(v), np.zeros_like(omega), com, c)
    np.testing.assert_allclose(f, 0)
    np.testing.assert_allclose(t, 0)


def test_bending_and_torsion_are_independent():
    c = cable()
    a = rod_coefficients(c)
    c["rod_material"]["torsional_rigidity_n_m2"] *= 10
    b = rod_coefficients(c)
    assert b["twist_stiffness"] == pytest.approx(10 * a["twist_stiffness"])
    assert b["bend_stiffness"] == a["bend_stiffness"]
