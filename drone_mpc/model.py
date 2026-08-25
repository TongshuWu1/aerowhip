"""Strict loading of the identified one-attachment cable artifact."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from cable_twin.shared.dder import DderModel, DderParameters
from cable_twin.shared.observation_data import sha256_file
from optitrack_offline.fitting import MODEL_SCHEMA
from optitrack_offline.validation import load_fitted_optitrack_model


@dataclass(frozen=True, slots=True)
class CableModelSnapshot:
    """Immutable physical-model snapshot used by one MPC solve."""

    source_path: Path
    sha256: str
    model: DderModel
    marker_node_indices: tuple[int, ...]
    rod_material_coordinates_m: tuple[float, ...]
    bending_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    payload: dict[str, object]
    provisional: bool
    provenance_note: str

    @property
    def node_count(self) -> int:
        return self.model.parameters.node_count

    @property
    def cable_length_m(self) -> float:
        return self.model.parameters.cable_length_m


def load_cable_model(path: str | Path) -> CableModelSnapshot:
    """Load a free-tip artifact or an explicitly marked provisional old fit."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Fit the one-attachment cable model first: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema == "optitrack_twist_aware_rod_v5":
        return _provisional_two_holder_fit(source, payload)
    if schema != MODEL_SCHEMA:
        raise ValueError(
            "Drone MPC requires the one-attachment/free-tip EI/Cb artifact or "
            "the explicitly supported v5 two-holder fit for provisional testing."
        )
    measured = payload.get("measured")
    optimized = payload.get("optimized")
    boundary = payload.get("boundary_condition")
    if not all(isinstance(value, dict) for value in (measured, optimized, boundary)):
        raise ValueError("The fitted cable artifact is incomplete.")
    assert isinstance(measured, dict)
    assert isinstance(optimized, dict)
    assert isinstance(boundary, dict)
    if "torsional_stiffness_n_m2" in optimized:
        raise ValueError("The free-tip MPC model must not contain a fitted GJ.")
    prescribed = tuple(int(value) for value in boundary.get("prescribed_vertices", ()))
    if prescribed != (0,):
        raise ValueError("Drone MPC requires exactly one prescribed attachment vertex.")

    model = load_fitted_optitrack_model(source)
    marker_indices = tuple(int(value) for value in measured["marker_node_indices"])
    material_coordinates = tuple(
        float(value) for value in measured["rod_material_coordinates_m"]
    )
    if (
        int(measured["marker_count"]) != 11
        or len(marker_indices) != 11
        or len(material_coordinates) != model.parameters.node_count
        or marker_indices[0] != 0
        or marker_indices[-1] != model.parameters.node_count - 1
    ):
        raise ValueError("The fitted cable observation/discretization map is invalid.")
    return CableModelSnapshot(
        source_path=source,
        sha256=sha256_file(source),
        model=model,
        marker_node_indices=marker_indices,
        rod_material_coordinates_m=material_coordinates,
        bending_stiffness_n_m2=float(optimized["bending_stiffness_n_m2"]),
        bending_damping_n_m2_s=float(optimized["bending_damping_n_m2_s"]),
        payload=payload,
        provisional=False,
        provenance_note="identified from one-attachment/free-tip recordings",
    )


def _provisional_two_holder_fit(
    source: Path,
    payload: dict[str, object],
) -> CableModelSnapshot:
    """Transfer EI/Cb from the latest two-holder fit for pre-data simulation.

    This is deliberately narrow: only the final v5 artifact is accepted.  GJ
    and both old terminal-frame boundary conditions are not transferred.  The
    newly free distal marker is assigned the mean mass of the nine identical
    interior markers used in that experiment.
    """

    measured = payload.get("measured")
    optimized = payload.get("optimized")
    solver = payload.get("solver")
    if not all(isinstance(value, dict) for value in (measured, optimized, solver)):
        raise ValueError("The provisional two-holder artifact is incomplete.")
    assert isinstance(measured, dict)
    assert isinstance(optimized, dict)
    assert isinstance(solver, dict)
    marker_count = int(measured.get("marker_count", 0))
    node_count = int(measured.get("node_count", 0))
    marker_indices = tuple(int(value) for value in measured["marker_node_indices"])
    material_coordinates = tuple(
        float(value) for value in measured["rod_material_coordinates_m"]
    )
    rest_lengths = tuple(float(value) for value in measured["rest_lengths_m"])
    vertex_masses = [float(value) for value in measured["vertex_masses_kg"]]
    interior_marker_masses = tuple(
        float(value) for value in measured.get("interior_marker_masses_kg", ())
    )
    if (
        marker_count != 11
        or node_count != len(rest_lengths) + 1
        or len(vertex_masses) != node_count
        or len(marker_indices) != marker_count
        or marker_indices[0] != 0
        or marker_indices[-1] != node_count - 1
        or len(material_coordinates) != node_count
        or len(interior_marker_masses) != 9
        or any(value <= 0.0 for value in interior_marker_masses)
    ):
        raise ValueError("The v5 two-holder fit has an incompatible discretization.")
    free_tip_marker_mass = sum(interior_marker_masses) / len(interior_marker_masses)
    vertex_masses[-1] += free_tip_marker_mass
    gravity = tuple(float(value) for value in solver["gravity_m_s2"])
    ei = float(optimized["bending_stiffness_n_m2"])
    cb = float(optimized["bending_damping_n_m2_s"])
    model = DderModel(
        DderParameters(
            node_count=node_count,
            cable_length_m=float(measured["length_m"]),
            cable_mass_kg=sum(vertex_masses),
            cable_diameter_m=float(measured["diameter_m"]),
            bending_stiffness_n_m2=ei,
            bending_damping_n_m2_s=cb,
            torsional_stiffness_n_m2=0.0,
            external_drag_s_inv=float(optimized.get("external_drag_s_inv", 0.0)),
            gravity_camera_m_s2=gravity,  # type: ignore[arg-type]
            rest_lengths_m=rest_lengths,
            vertex_masses_kg=tuple(vertex_masses),
            substeps=int(solver["substeps"]),
            constraint_iterations=int(solver["constraint_iterations"]),
        )
    )
    note = (
        "PROVISIONAL transfer from the v5 two-holder fit: EI/Cb retained, GJ and "
        "terminal-frame constraints removed, one measured marker mass added at "
        "the free tip; refit with one-attachment data before reporting results"
    )
    return CableModelSnapshot(
        source_path=source,
        sha256=sha256_file(source),
        model=model,
        marker_node_indices=marker_indices,
        rod_material_coordinates_m=material_coordinates,
        bending_stiffness_n_m2=ei,
        bending_damping_n_m2_s=cb,
        payload=payload,
        provisional=True,
        provenance_note=note,
    )
