"""CPU tests for the new plant's independent math; no Isaac installation needed."""

import json
from pathlib import Path
import numpy as np
import pytest
from isaac_simulation.dynamics import (
    FlightPath,
    QuadrotorController,
    Reference,
    cable_masses,
    cable_loads,
    tracked_kinematics,
    rotation_from_wxyz,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / "config/isaac_physx/rig_153g.json").read_text())


@pytest.mark.parametrize("kind", ["circle", "figure8", "vertical8"])
def test_reference_pva_and_closed_smooth_ends(kind):
    path = FlightPath(config()["trajectory"], kind)
    h = 1e-4
    for t in [path.hold + 0.4, path.hold + 3.0, path.end - 0.4]:
        left, mid, right = [path.sample(x) for x in [t - h, t, t + h]]
        np.testing.assert_allclose(
            (right.position - left.position) / (2 * h), mid.velocity, atol=1e-7
        )
        np.testing.assert_allclose(
            (right.velocity - left.velocity) / (2 * h), mid.acceleration, atol=1e-7
        )
    a, b = path.sample(0), path.sample(path.duration)
    np.testing.assert_allclose(a.position, b.position, atol=1e-12)
    for t in [path.hold, path.end]:
        np.testing.assert_allclose(path.sample(t).velocity, 0, atol=1e-12)
        np.testing.assert_allclose(path.sample(t).acceleration, 0, atol=1e-12)


def test_hover_actuation_supports_drone_plus_distributed_cable():
    c = config()
    controller = QuadrotorController(c)
    assert cable_masses(c).sum() == pytest.approx(0.018)
    reference = Reference(np.array([0, 0, 1.5]), np.zeros(3), np.zeros(3))
    controller.update(
        reference.position, np.zeros(3), np.eye(3), np.zeros(3), reference, 1 / 300
    )
    wrench = controller.actuator_step(1 / 600)
    np.testing.assert_allclose(wrench, [0.171 * 9.80665, 0, 0, 0], atol=1e-12)
    assert not controller.saturated
    controller.motor_target[:] = c["drone"]["maximum_motor_thrust_n"]
    before = controller.motor_thrust.copy()
    controller.actuator_step(1 / 600)
    assert np.all(controller.motor_thrust > before)
    assert np.all(controller.motor_thrust < controller.motor_target)


def test_internal_cable_torques_cancel_and_drag_dissipates_energy():
    c = config()["cable"]
    n = c["segments"]
    angles = np.linspace(0, 0.5, n)
    q = np.column_stack(
        (np.cos(angles / 2), np.sin(angles / 2), np.zeros(n), np.zeros(n))
    )
    r = rotation_from_wxyz(q)
    v = np.tile([1.0, 0.5, 0.3], (n, 1))
    f, t = cable_loads(r, v, c)
    np.testing.assert_allclose(t.sum(0), 0, atol=1e-14)
    assert np.sum(f * v) < 0
    # Straight motionless cable has no artificial restoring load at its free pivot.
    f, t = cable_loads(np.tile(np.eye(3), (n, 1, 1)), np.zeros((n, 3)), c)
    np.testing.assert_allclose(f, 0)
    np.testing.assert_allclose(t, 0)


def test_tracked_origin_velocity_includes_rotating_lever_arm():
    p, v = tracked_kinematics(
        np.zeros(3), np.zeros(3), np.eye(3), np.array([0, 1.0, 0]), [0, 0, 0.025]
    )
    np.testing.assert_allclose(p, [0, 0, 0.025])
    np.testing.assert_allclose(v, [0.025, 0, 0])
    c = config()["drone"]
    np.testing.assert_allclose(
        np.array(c["attachment_body_m"]) - c["tracked_origin_body_m"], [0, 0, -0.055]
    )


def test_configuration_rejects_inconsistent_rates_and_unflyable_mass():
    c = config()
    validate_config(c)
    c["physics"]["command_rate_hz"] = 20
    with pytest.raises(ValueError, match="30 Hz"):
        validate_config(c)
    c = config()
    c["drone"]["mass_kg"] = 1.0
    with pytest.raises(ValueError, match="thrust"):
        validate_config(c)


def test_bending_spring_matches_energy_gradient():
    c = config()["cable"]
    c["segments"] = 2
    angle = 0.2
    r = rotation_from_wxyz(
        np.array([[1, 0, 0, 0], [np.cos(angle / 2), np.sin(angle / 2), 0, 0]])
    )
    _, torque = cable_loads(r, np.zeros((2, 3)), c)
    assert torque[1, 0] < 0  # positive bend gives negative restoring torque
    energy = (
        lambda a: c["bending_stiffness_n_m2"]
        / (c["length_m"] / c["segments"])
        * (1 - np.cos(a))
    )
    h = 1e-5
    assert torque[1, 0] == pytest.approx(
        -(energy(angle + h) - energy(angle - h)) / (2 * h)
    )
