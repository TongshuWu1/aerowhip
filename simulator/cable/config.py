"""Fixed, measured cable geometry and mass distribution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .dder import DderParameters


@dataclass(frozen=True, slots=True)
class CableConfiguration:
    """Measured quantities that remain fixed during EI/Cb identification."""

    marker_interval_lengths_m: tuple[float, ...]
    bare_cable_mass_kg: float
    moving_marker_masses_kg: tuple[float, ...]
    diameter_m: float
    segments_per_marker_interval: int | tuple[int, ...]
    gravity_m_s2: tuple[float, float, float]
    substeps: int
    constraint_iterations: int
    external_drag_s_inv: float = 0.0

    def __post_init__(self) -> None:
        lengths = tuple(float(value) for value in self.marker_interval_lengths_m)
        marker_masses = tuple(float(value) for value in self.moving_marker_masses_kg)
        gravity = tuple(float(value) for value in self.gravity_m_s2)
        if len(lengths) != 10:
            raise ValueError("The current cable requires ten measured intervals.")
        if len(marker_masses) != len(lengths):
            raise ValueError("One moving marker mass is required for c1 through c10.")
        positive = lengths + marker_masses + (
            float(self.bare_cable_mass_kg),
            float(self.diameter_m),
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Cable dimensions and masses must be finite and positive.")
        if len(gravity) != 3 or not all(math.isfinite(value) for value in gravity):
            raise ValueError("Gravity must contain three finite SI values.")
        raw_subdivisions = self.segments_per_marker_interval
        subdivisions = (
            (int(raw_subdivisions),) * len(lengths)
            if isinstance(raw_subdivisions, int)
            else tuple(int(value) for value in raw_subdivisions)
        )
        if len(subdivisions) != len(lengths):
            raise ValueError(
                "segments_per_marker_interval must provide one subdivision "
                "count per measured interval."
            )
        if any(not 1 <= value <= 8 for value in subdivisions):
            raise ValueError("Interval subdivisions must be between 1 and 8.")
        if min(int(self.substeps), int(self.constraint_iterations)) < 1:
            raise ValueError("DDER solver counts must be positive.")
        if not math.isfinite(self.external_drag_s_inv) or self.external_drag_s_inv < 0:
            raise ValueError("External drag rate must be finite and non-negative.")
        object.__setattr__(self, "marker_interval_lengths_m", lengths)
        object.__setattr__(self, "moving_marker_masses_kg", marker_masses)
        object.__setattr__(self, "gravity_m_s2", gravity)
        object.__setattr__(self, "segments_per_marker_interval", subdivisions)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CableConfiguration":
        return cls(
            marker_interval_lengths_m=tuple(payload["marker_interval_lengths_m"]),
            bare_cable_mass_kg=float(payload["bare_cable_mass_kg"]),
            moving_marker_masses_kg=tuple(payload["moving_marker_masses_kg"]),
            diameter_m=float(payload["diameter_m"]),
            segments_per_marker_interval=_interval_subdivisions(
                payload["segments_per_marker_interval"]
            ),
            gravity_m_s2=tuple(payload["gravity_m_s2"]),
            substeps=int(payload["substeps"]),
            constraint_iterations=int(payload["constraint_iterations"]),
            external_drag_s_inv=float(payload.get("external_drag_s_inv", 0.0)),
        )

    @property
    def measured_site_count(self) -> int:
        return len(self.marker_interval_lengths_m) + 1

    @property
    def moving_marker_count(self) -> int:
        return self.measured_site_count - 1

    @property
    def node_count(self) -> int:
        return (
            sum(self.interval_subdivisions) + 1
        )

    @property
    def edge_count(self) -> int:
        return self.node_count - 1

    @property
    def length_m(self) -> float:
        return sum(self.marker_interval_lengths_m)

    @property
    def total_dynamic_mass_kg(self) -> float:
        return self.bare_cable_mass_kg + sum(self.moving_marker_masses_kg)

    @property
    def marker_node_indices(self) -> tuple[int, ...]:
        indices = [0]
        for subdivision in self.interval_subdivisions:
            indices.append(indices[-1] + subdivision)
        return tuple(indices)

    @property
    def rest_lengths_m(self) -> tuple[float, ...]:
        return tuple(
            length / subdivision
            for length, subdivision in zip(
                self.marker_interval_lengths_m,
                self.interval_subdivisions,
                strict=True,
            )
            for _ in range(subdivision)
        )

    @property
    def interval_subdivisions(self) -> tuple[int, ...]:
        value = self.segments_per_marker_interval
        assert isinstance(value, tuple)
        return value

    @property
    def vertex_masses_kg(self) -> tuple[float, ...]:
        density = self.bare_cable_mass_kg / self.length_m
        rest = self.rest_lengths_m
        masses = [0.0] * self.node_count
        masses[0] = 0.5 * density * rest[0]
        masses[-1] = 0.5 * density * rest[-1]
        for index in range(1, self.node_count - 1):
            masses[index] = 0.5 * density * (rest[index - 1] + rest[index])
        for index, marker_mass in zip(
            self.marker_node_indices[1:], self.moving_marker_masses_kg
        ):
            masses[index] += marker_mass
        return tuple(masses)

    def dder_parameters(self, *, EI: float, Cb: float) -> DderParameters:
        """Build the preserved DDER parameter object without extra physics."""

        return DderParameters(
            node_count=self.node_count,
            cable_length_m=self.length_m,
            cable_mass_kg=self.total_dynamic_mass_kg,
            cable_diameter_m=self.diameter_m,
            bending_stiffness_n_m2=float(EI),
            bending_damping_n_m2_s=float(Cb),
            gravity_world_m_s2=self.gravity_m_s2,
            torsional_stiffness_n_m2=0.0,
            external_drag_s_inv=self.external_drag_s_inv,
            rest_lengths_m=self.rest_lengths_m,
            vertex_masses_kg=self.vertex_masses_kg,
            substeps=self.substeps,
            constraint_iterations=self.constraint_iterations,
        )


def _interval_subdivisions(value: object) -> int | tuple[int, ...]:
    """Accept historical scalar configs and the production per-interval topology."""

    if isinstance(value, bool):
        raise ValueError("Cable interval subdivisions must be integer counts.")
    if isinstance(value, int):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(int(item) for item in value)
    raise ValueError(
        "segments_per_marker_interval must be an integer or an integer sequence."
    )
