from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .dder import DderModel, DderParameters


PLANAR_SCHEMA = "planar_rgb_reference_dder_v2"
OPTITRACK_SCHEMA = "optitrack_direct_observation_rod_v3"
TWIST_AWARE_OPTITRACK_SCHEMA = "optitrack_twist_aware_rod_v2"
REFINED_TWIST_AWARE_OPTITRACK_SCHEMA = "optitrack_twist_aware_rod_v3"
DRAG_AWARE_OPTITRACK_SCHEMA = "optitrack_twist_aware_rod_v4"
CURRENT_OPTITRACK_SCHEMA = "optitrack_twist_aware_rod_v5"


@dataclass(frozen=True, slots=True)
class DderArtifact:
    path: Path
    schema: str
    cable_identity: int
    node_count: int
    cable_length_m: float
    cable_mass_kg: float
    bare_cable_mass_kg: float
    cable_diameter_m: float
    bending_stiffness_n_m2: float
    torsional_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    external_drag_s_inv: float
    substeps: int
    constraint_iterations: int
    rest_lengths_m: tuple[float, ...] | None
    vertex_masses_kg: tuple[float, ...] | None
    payload: dict[str, Any]

    def make_model(self, gravity_camera_m_s2: tuple[float, float, float]) -> DderModel:
        return DderModel(
            DderParameters(
                node_count=self.node_count,
                cable_length_m=self.cable_length_m,
                cable_mass_kg=self.cable_mass_kg,
                cable_diameter_m=self.cable_diameter_m,
                bending_stiffness_n_m2=self.bending_stiffness_n_m2,
                bending_damping_n_m2_s=self.bending_damping_n_m2_s,
                torsional_stiffness_n_m2=self.torsional_stiffness_n_m2,
                external_drag_s_inv=self.external_drag_s_inv,
                gravity_camera_m_s2=gravity_camera_m_s2,
                rest_lengths_m=self.rest_lengths_m,
                vertex_masses_kg=self.vertex_masses_kg,
                substeps=self.substeps,
                constraint_iterations=self.constraint_iterations,
            )
        )

    def make_rescaled_bare_model(
        self,
        gravity_camera_m_s2: tuple[float, float, float],
        *,
        cable_length_m: float,
        cable_diameter_m: float,
        node_count: int,
        substeps: int,
        constraint_iterations: int,
    ) -> DderModel:
        """Transfer homogeneous material parameters to a bare cable length."""

        linear_density = self.bare_cable_mass_kg / self.cable_length_m
        cable_mass = linear_density * float(cable_length_m)
        return DderModel(
            DderParameters(
                node_count=int(node_count),
                cable_length_m=float(cable_length_m),
                cable_mass_kg=cable_mass,
                cable_diameter_m=float(cable_diameter_m),
                bending_stiffness_n_m2=self.bending_stiffness_n_m2,
                bending_damping_n_m2_s=self.bending_damping_n_m2_s,
                # The online image-only path has no measured material frames;
                # retain learned EI/Cb but do not invent torsional boundaries.
                torsional_stiffness_n_m2=0.0,
                # The identified drag includes the OptiTrack marker hardware;
                # do not transfer it to a differently mass-loaded bare cable.
                external_drag_s_inv=0.0,
                gravity_camera_m_s2=gravity_camera_m_s2,
                substeps=int(substeps),
                constraint_iterations=int(constraint_iterations),
            )
        )


def load_dder_artifact(path: str | Path) -> DderArtifact:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    schema = str(payload.get("schema", ""))
    if schema not in {
        PLANAR_SCHEMA,
        OPTITRACK_SCHEMA,
        TWIST_AWARE_OPTITRACK_SCHEMA,
        REFINED_TWIST_AWARE_OPTITRACK_SCHEMA,
        DRAG_AWARE_OPTITRACK_SCHEMA,
        CURRENT_OPTITRACK_SCHEMA,
    }:
        raise ValueError(f"Unsupported DDER model schema in {source}.")
    measured = payload.get("measured")
    optimized = payload.get("optimized")
    solver = payload.get("solver")
    fit = payload.get("fit")
    if not all(isinstance(value, dict) for value in (measured, optimized, solver, fit)):
        raise ValueError(
            "DDER model is missing physical, solver, or fitting fields."
        )
    if schema == PLANAR_SCHEMA and fit.get("status") != "completed":
        raise ValueError("DDER parameter fitting did not complete.")
    if (
        schema in {
            TWIST_AWARE_OPTITRACK_SCHEMA,
            REFINED_TWIST_AWARE_OPTITRACK_SCHEMA,
            DRAG_AWARE_OPTITRACK_SCHEMA,
            CURRENT_OPTITRACK_SCHEMA,
        }
        and "torsional_stiffness_n_m2" not in optimized
    ):
        raise ValueError("Twist-aware OptiTrack model is missing fitted GJ.")
    if schema in {
        OPTITRACK_SCHEMA,
        TWIST_AWARE_OPTITRACK_SCHEMA,
            REFINED_TWIST_AWARE_OPTITRACK_SCHEMA,
            DRAG_AWARE_OPTITRACK_SCHEMA,
            CURRENT_OPTITRACK_SCHEMA,
    }:
        rest_lengths = tuple(float(value) for value in measured["rest_lengths_m"])
        vertex_masses = tuple(float(value) for value in measured["vertex_masses_kg"])
        mass_key = (
            "total_dynamic_mass_kg"
            if schema in {
                TWIST_AWARE_OPTITRACK_SCHEMA,
                REFINED_TWIST_AWARE_OPTITRACK_SCHEMA,
                DRAG_AWARE_OPTITRACK_SCHEMA,
                CURRENT_OPTITRACK_SCHEMA,
            }
            else "total_instrumented_mass_kg"
        )
        bare_mass = float(measured["bare_cable_mass_kg"])
    else:
        rest_lengths = None
        vertex_masses = None
        mass_key = "mass_kg"
        bare_mass = float(measured[mass_key])
    artifact = DderArtifact(
        path=source,
        schema=schema,
        cable_identity=int(payload.get("cable_identity", 1)),
        node_count=int(measured["node_count"]),
        cable_length_m=float(measured["length_m"]),
        cable_mass_kg=float(measured[mass_key]),
        bare_cable_mass_kg=bare_mass,
        cable_diameter_m=float(measured["diameter_m"]),
        bending_stiffness_n_m2=float(optimized["bending_stiffness_n_m2"]),
        torsional_stiffness_n_m2=float(
            optimized.get("torsional_stiffness_n_m2", 0.0)
        ),
        bending_damping_n_m2_s=float(optimized["bending_damping_n_m2_s"]),
        external_drag_s_inv=float(optimized.get("external_drag_s_inv", 0.0)),
        substeps=int(solver["substeps"]),
        constraint_iterations=int(solver["constraint_iterations"]),
        rest_lengths_m=rest_lengths,
        vertex_masses_kg=vertex_masses,
        payload=payload,
    )
    positive = (
        artifact.cable_length_m,
        artifact.cable_mass_kg,
        artifact.bare_cable_mass_kg,
        artifact.cable_diameter_m,
        artifact.bending_stiffness_n_m2,
        artifact.bending_damping_n_m2_s,
    )
    if artifact.node_count < 6 or min(artifact.substeps, artifact.constraint_iterations) < 1:
        raise ValueError("DDER model discretization is invalid.")
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("DDER model physical values must be finite and positive.")
    if (
        not math.isfinite(artifact.torsional_stiffness_n_m2)
        or artifact.torsional_stiffness_n_m2 < 0.0
    ):
        raise ValueError("DDER model torsional stiffness is invalid.")
    if not math.isfinite(artifact.external_drag_s_inv) or artifact.external_drag_s_inv < 0.0:
        raise ValueError("DDER model ambient drag is invalid.")
    if artifact.rest_lengths_m is not None:
        if len(artifact.rest_lengths_m) != artifact.node_count - 1:
            raise ValueError("OptiTrack model must contain one rest length per edge.")
        if artifact.vertex_masses_kg is None or len(artifact.vertex_masses_kg) != artifact.node_count:
            raise ValueError("OptiTrack model must contain one mass per vertex.")
    return artifact
