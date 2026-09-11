"""Resolution-scaled rod properties and exact lumped-marker mass geometry."""

import numpy as np


def rod_coefficients(cable):
    length = cable["length_m"] / cable["segments"]
    m = cable["rod_material"]
    result = {}
    for mode, key in [
        ("bend", "bending_rigidity_n_m2"),
        ("twist", "torsional_rigidity_n_m2"),
    ]:
        value = m[key]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        tau = m[mode + "_relaxation_time_s"]
        if not np.isfinite(tau) or tau < 0:
            raise ValueError("Material relaxation times must be nonnegative")
        result[mode + "_stiffness"] = value / length
        result[mode + "_damping"] = value * tau / length
    return result


def rod_mass_properties(cable):
    """COM/inertia in a segment frame with +Z pointing along the cable.

    Bare cord is a uniform cylinder; spherical marker masses are placed at
    their actual local distances, including parallel-axis contributions.
    """
    n = cable["segments"]
    length = cable["length_m"] / n
    radius = cable["diameter_m"] / 2
    bare = cable["bare_mass_kg"] / n
    masses = np.full(n, bare)
    first = np.zeros(n)
    second = np.zeros(n)
    marker_count = np.zeros(n)
    bindings = []
    for distance in cable["marker_distances_m"]:
        i = min(n - 1, int(distance / length))
        z = distance - (i + 0.5) * length
        mass = cable["marker_mass_kg"]
        masses[i] += mass
        first[i] += mass * z
        second[i] += mass * z * z
        marker_count[i] += 1
        bindings.append((i, z))
    com = first / masses
    sphere_inertia = (
        marker_count * (2 / 5) * cable["marker_mass_kg"] * cable["marker_radius_m"] ** 2
    )
    transverse = (
        bare * (length**2 / 12 + radius**2 / 4)
        + second
        - masses * com**2
        + sphere_inertia
    )
    axial = bare * radius**2 / 2 + sphere_inertia
    return masses, com, np.column_stack((transverse, transverse, axial)), bindings


def aerodynamic_loads(positions, rotations, velocities_com, omegas, com_z, cable):
    """Distributed crossflow + axial skin drag; marker sphere drag, with moments.

    Two-point Gauss quadrature includes the local velocity from link rotation.
    Every load is applied relative to the physical COM, preserving power sign.
    """
    n = len(positions)
    length = cable["length_m"] / n
    axis = rotations[:, :, 2]
    rho = cable["air_density_kg_m3"]
    wind = np.asarray(cable.get("wind_world_m_s", [0.0, 0.0, 0.0]))
    force = np.zeros((n, 3))
    torque = np.zeros_like(force)
    for z in [-length / (2 * np.sqrt(3)), length / (2 * np.sqrt(3))]:
        arm = axis * (z - com_z)[:, None]
        velocity = velocities_com + np.cross(omegas, arm) - wind
        axial = np.sum(velocity * axis, axis=1)[:, None] * axis
        normal = velocity - axial
        f = (
            -0.5
            * rho
            * cable["crossflow_drag_coefficient"]
            * cable["diameter_m"]
            * (length / 2)
            * np.linalg.norm(normal, axis=1)[:, None]
            * normal
        )
        f -= (
            0.5
            * rho
            * cable["axial_skin_drag_coefficient"]
            * np.pi
            * cable["diameter_m"]
            * (length / 2)
            * np.linalg.norm(axial, axis=1)[:, None]
            * axial
        )
        force += f
        torque += np.cross(arm, f)
    for distance in cable["marker_distances_m"]:
        i = min(n - 1, int(distance / length))
        z = distance - (i + 0.5) * length
        arm = axis[i] * (z - com_z[i])
        v = velocities_com[i] + np.cross(omegas[i], arm) - wind
        f = (
            -0.5
            * rho
            * cable["marker_drag_coefficient"]
            * np.pi
            * cable["marker_radius_m"] ** 2
            * np.linalg.norm(v)
            * v
        )
        force[i] += f
        torque[i] += np.cross(arm, f)
    return force, torque
