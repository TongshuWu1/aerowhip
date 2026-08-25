"""Physical configuration and continuum-to-PhysX parameter mapping.

This module deliberately has no Isaac Sim imports.  Its unit conversions and
mass accounting can therefore be tested with the project's ordinary Python
environment before launching Kit.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import json
import math
from pathlib import Path
from typing import Any

from drone_mpc.model import CableModelSnapshot, load_cable_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "optitrack_offline" / "models" / "cable_model.json"
DEFAULT_DRONE_CONFIG_PATH = Path(__file__).with_name("drone_config.json")


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _vec3(name: str, value: object, *, positive: bool = False) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three values.")
    result = tuple(float(component) for component in value)
    if any(not math.isfinite(component) for component in result):
        raise ValueError(f"{name} must be finite.")
    if positive and any(component <= 0.0 for component in result):
        raise ValueError(f"{name} must be positive.")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class DronePhysicalConfig:
    """Explicit 6-DoF drone assumptions used by the first Isaac plant.

    These values are intentionally stored separately from the identified cable
    artifact.  They are provisional until the flight vehicle is weighed and its
    inertia/thrust limits are measured.
    """

    mass_kg: float
    diagonal_inertia_kg_m2: tuple[float, float, float]
    body_size_m: tuple[float, float, float]
    arm_length_m: float
    attachment_offset_body_m: tuple[float, float, float]
    maximum_collective_thrust_n: float
    maximum_body_torque_n_m: tuple[float, float, float]
    initial_position_m: tuple[float, float, float]
    position_kp_n_m: tuple[float, float, float]
    position_kd_n_s_m: tuple[float, float, float]
    attitude_kp_n_m_rad: tuple[float, float, float]
    attitude_kd_n_m_s_rad: tuple[float, float, float]
    provisional: bool = True
    note: str = ""

    def __post_init__(self) -> None:
        _positive_finite("mass_kg", self.mass_kg)
        _vec3("diagonal_inertia_kg_m2", self.diagonal_inertia_kg_m2, positive=True)
        _vec3("body_size_m", self.body_size_m, positive=True)
        _positive_finite("arm_length_m", self.arm_length_m)
        _vec3("attachment_offset_body_m", self.attachment_offset_body_m)
        _positive_finite("maximum_collective_thrust_n", self.maximum_collective_thrust_n)
        _vec3("maximum_body_torque_n_m", self.maximum_body_torque_n_m, positive=True)
        _vec3("initial_position_m", self.initial_position_m)
        _vec3("position_kp_n_m", self.position_kp_n_m, positive=True)
        _vec3("position_kd_n_s_m", self.position_kd_n_s_m, positive=True)
        _vec3("attitude_kp_n_m_rad", self.attitude_kp_n_m_rad, positive=True)
        _vec3("attitude_kd_n_m_s_rad", self.attitude_kd_n_m_s_rad, positive=True)

    @property
    def maximum_total_mass_kg(self) -> float:
        return self.maximum_collective_thrust_n / 9.80665


def load_drone_config(path: str | Path = DEFAULT_DRONE_CONFIG_PATH) -> DronePhysicalConfig:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "isaac_whip_drone_physical_config_v2":
        raise ValueError(f"Unsupported drone configuration schema in {source}.")
    values = payload.get("drone")
    if not isinstance(values, dict):
        raise ValueError("Drone configuration is missing its 'drone' object.")
    return DronePhysicalConfig(
        mass_kg=_positive_finite("mass_kg", values["mass_kg"]),
        diagonal_inertia_kg_m2=_vec3(
            "diagonal_inertia_kg_m2", values["diagonal_inertia_kg_m2"], positive=True
        ),
        body_size_m=_vec3("body_size_m", values["body_size_m"], positive=True),
        arm_length_m=_positive_finite("arm_length_m", values["arm_length_m"]),
        attachment_offset_body_m=_vec3(
            "attachment_offset_body_m", values["attachment_offset_body_m"]
        ),
        maximum_collective_thrust_n=_positive_finite(
            "maximum_collective_thrust_n", values["maximum_collective_thrust_n"]
        ),
        maximum_body_torque_n_m=_vec3(
            "maximum_body_torque_n_m", values["maximum_body_torque_n_m"], positive=True
        ),
        initial_position_m=_vec3("initial_position_m", values["initial_position_m"]),
        position_kp_n_m=_vec3("position_kp_n_m", values["position_kp_n_m"], positive=True),
        position_kd_n_s_m=_vec3(
            "position_kd_n_s_m", values["position_kd_n_s_m"], positive=True
        ),
        attitude_kp_n_m_rad=_vec3(
            "attitude_kp_n_m_rad", values["attitude_kp_n_m_rad"], positive=True
        ),
        attitude_kd_n_m_s_rad=_vec3(
            "attitude_kd_n_m_s_rad", values["attitude_kd_n_m_s_rad"], positive=True
        ),
        provisional=bool(values.get("provisional", True)),
        note=str(values.get("note", "")),
    )


@dataclass(frozen=True, slots=True)
class CablePlantSpec:
    """A complete, SI-unit articulation discretization of one cable artifact."""

    source_path: Path
    source_sha256: str
    source_schema: str
    provenance_note: str
    provisional: bool
    link_count: int
    length_m: float
    diameter_m: float
    bending_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    material_coordinates_m: tuple[float, ...]
    rest_lengths_m: tuple[float, ...]
    vertex_masses_kg: tuple[float, ...]
    link_masses_kg: tuple[float, ...]
    link_com_x_m: tuple[float, ...]
    link_diagonal_inertia_kg_m2: tuple[tuple[float, float, float], ...]
    hinge_dual_lengths_m: tuple[float, ...]
    hinge_stiffness_n_m_rad: tuple[float, ...]
    hinge_damping_n_m_s_rad: tuple[float, ...]
    usd_hinge_stiffness_n_m_deg: tuple[float, ...]
    usd_hinge_damping_n_m_s_deg: tuple[float, ...]

    @property
    def node_count(self) -> int:
        return self.link_count + 1

    @property
    def total_mass_kg(self) -> float:
        return float(sum(self.link_masses_kg))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": "isaac_whip_cable_plant_manifest_v1",
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "source_schema": self.source_schema,
            "provenance_note": self.provenance_note,
            "provisional": self.provisional,
            "link_count": self.link_count,
            "node_count": self.node_count,
            "length_m": self.length_m,
            "diameter_m": self.diameter_m,
            "total_mass_kg": self.total_mass_kg,
            "bending_stiffness_n_m2": self.bending_stiffness_n_m2,
            "bending_damping_n_m2_s": self.bending_damping_n_m2_s,
            "material_coordinates_m": list(self.material_coordinates_m),
            "rest_lengths_m": list(self.rest_lengths_m),
            "vertex_masses_kg": list(self.vertex_masses_kg),
            "link_masses_kg": list(self.link_masses_kg),
            "hinge_dual_lengths_m": list(self.hinge_dual_lengths_m),
            "hinge_stiffness_n_m_rad": list(self.hinge_stiffness_n_m_rad),
            "hinge_damping_n_m_s_rad": list(self.hinge_damping_n_m_s_rad),
            "usd_hinge_stiffness_n_m_deg": list(self.usd_hinge_stiffness_n_m_deg),
            "usd_hinge_damping_n_m_s_deg": list(self.usd_hinge_damping_n_m_s_deg),
            "internal_twist_mode": "free; no stiffness, damping, friction, or limit",
            "root_boundary": "fixed material frame at drone attachment",
            "distal_boundary": "mechanically free",
        }


def _deposit_vertex_masses(
    source_coordinates: tuple[float, ...],
    source_masses: tuple[float, ...],
    target_coordinates: tuple[float, ...],
) -> tuple[float, ...]:
    """Linearly deposit fitted material-point masses onto another grid."""

    deposited = [0.0] * len(target_coordinates)
    for coordinate, mass in zip(source_coordinates, source_masses):
        upper = bisect.bisect_left(target_coordinates, coordinate)
        if upper <= 0:
            deposited[0] += mass
        elif upper >= len(target_coordinates):
            deposited[-1] += mass
        else:
            lower = upper - 1
            span = target_coordinates[upper] - target_coordinates[lower]
            fraction = (coordinate - target_coordinates[lower]) / span
            deposited[lower] += mass * (1.0 - fraction)
            deposited[upper] += mass * fraction
    return tuple(deposited)


def _target_grid(snapshot: CableModelSnapshot, link_count: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    source_coordinates = tuple(float(value) for value in snapshot.rod_material_coordinates_m)
    source_masses = tuple(float(value) for value in snapshot.model.parameters.vertex_masses_kg)
    if link_count == snapshot.node_count - 1:
        return source_coordinates, source_masses
    length = snapshot.cable_length_m
    coordinates = tuple(length * index / link_count for index in range(link_count + 1))
    masses = _deposit_vertex_masses(source_coordinates, source_masses, coordinates)
    return coordinates, masses


def _link_mass_properties(
    rest_lengths: tuple[float, ...],
    vertex_masses: tuple[float, ...],
    radius_m: float,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[tuple[float, float, float], ...],
]:
    link_masses: list[float] = []
    com_positions: list[float] = []
    inertias: list[tuple[float, float, float]] = []
    last = len(rest_lengths) - 1
    for index, length in enumerate(rest_lengths):
        left_mass = vertex_masses[index] if index == 0 else 0.5 * vertex_masses[index]
        right_mass = vertex_masses[index + 1] if index == last else 0.5 * vertex_masses[index + 1]
        mass = left_mass + right_mass
        if mass <= 0.0:
            raise ValueError(f"Link {index} received non-positive mass during regridding.")
        left_x = -0.5 * length
        right_x = 0.5 * length
        com_x = (left_mass * left_x + right_mass * right_x) / mass
        radial_axial = 0.5 * mass * radius_m * radius_m
        radial_transverse = 0.25 * mass * radius_m * radius_m
        transverse = (
            left_mass * (left_x - com_x) ** 2
            + right_mass * (right_x - com_x) ** 2
            + radial_transverse
        )
        link_masses.append(mass)
        com_positions.append(com_x)
        inertias.append(
            (
                max(radial_axial, 1.0e-12),
                max(transverse, 1.0e-12),
                max(transverse, 1.0e-12),
            )
        )
    return tuple(link_masses), tuple(com_positions), tuple(inertias)


def build_cable_spec(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    link_count: int = 20,
) -> tuple[CableModelSnapshot, CablePlantSpec]:
    """Load one fitted artifact and map its EI/Cb/mass to a rigid-link chain."""

    if not 4 <= int(link_count) <= 80:
        raise ValueError("link_count must be between 4 and 80.")
    snapshot = load_cable_model(model_path)
    parameters = snapshot.model.parameters
    coordinates, vertex_masses = _target_grid(snapshot, int(link_count))
    rest_lengths = tuple(
        coordinates[index + 1] - coordinates[index] for index in range(int(link_count))
    )
    if any(length <= 0.0 for length in rest_lengths):
        raise ValueError("Cable material coordinates must be strictly increasing.")
    radius = 0.5 * parameters.cable_diameter_m
    link_masses, com_x, inertia = _link_mass_properties(rest_lengths, vertex_masses, radius)
    dual_lengths = tuple(
        0.5 * (rest_lengths[index - 1] + rest_lengths[index])
        for index in range(1, int(link_count))
    )
    stiffness_rad = tuple(snapshot.bending_stiffness_n_m2 / value for value in dual_lengths)
    damping_rad = tuple(snapshot.bending_damping_n_m2_s / value for value in dual_lengths)
    radians_per_degree = math.pi / 180.0
    stiffness_usd = tuple(value * radians_per_degree for value in stiffness_rad)
    damping_usd = tuple(value * radians_per_degree for value in damping_rad)
    payload_schema = str(snapshot.payload.get("schema", "unknown"))
    spec = CablePlantSpec(
        source_path=snapshot.source_path,
        source_sha256=snapshot.sha256,
        source_schema=payload_schema,
        provenance_note=snapshot.provenance_note,
        provisional=snapshot.provisional,
        link_count=int(link_count),
        length_m=snapshot.cable_length_m,
        diameter_m=parameters.cable_diameter_m,
        bending_stiffness_n_m2=snapshot.bending_stiffness_n_m2,
        bending_damping_n_m2_s=snapshot.bending_damping_n_m2_s,
        material_coordinates_m=coordinates,
        rest_lengths_m=rest_lengths,
        vertex_masses_kg=vertex_masses,
        link_masses_kg=link_masses,
        link_com_x_m=com_x,
        link_diagonal_inertia_kg_m2=inertia,
        hinge_dual_lengths_m=dual_lengths,
        hinge_stiffness_n_m_rad=stiffness_rad,
        hinge_damping_n_m_s_rad=damping_rad,
        usd_hinge_stiffness_n_m_deg=stiffness_usd,
        usd_hinge_damping_n_m_s_deg=damping_usd,
    )
    if not math.isclose(spec.total_mass_kg, sum(vertex_masses), rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("Rigid-link conversion did not conserve cable mass.")
    if not math.isclose(spec.length_m, sum(rest_lengths), rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("Rigid-link conversion did not conserve cable length.")
    return snapshot, spec
