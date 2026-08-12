from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .dder import DderModel, DderParameters


SUPPORTED_SCHEMA = "planar_rgb_reference_dder_v2"


@dataclass(frozen=True, slots=True)
class DderArtifact:
    path: Path
    cable_identity: int
    node_count: int
    cable_length_m: float
    cable_mass_kg: float
    cable_diameter_m: float
    bending_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    substeps: int
    constraint_iterations: int
    student_t_degrees_of_freedom: float
    unresolved_curve_noise_m: float
    unresolved_endpoint_noise_m: float
    process_acceleration_sigma_m_s2: float
    endpoint_acceleration_sigma_m_s2: float
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
                gravity_camera_m_s2=gravity_camera_m_s2,
                substeps=self.substeps,
                constraint_iterations=self.constraint_iterations,
            )
        )


def load_dder_artifact(path: str | Path) -> DderArtifact:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != SUPPORTED_SCHEMA:
        raise ValueError(f"Unsupported DDER model schema in {source}.")
    measured = payload.get("measured")
    optimized = payload.get("optimized")
    solver = payload.get("solver")
    observation = payload.get("observation_model")
    process = payload.get("process_model")
    fit = payload.get("fit")
    if not all(
        isinstance(value, dict)
        for value in (
            measured,
            optimized,
            solver,
            observation,
            process,
            fit,
        )
    ):
        raise ValueError(
            "DDER model is missing physical, solver, observation, or process fields."
        )
    if fit.get("status") != "completed":
        raise ValueError("DDER parameter fitting did not complete.")
    artifact = DderArtifact(
        path=source,
        cable_identity=int(payload["cable_identity"]),
        node_count=int(measured["node_count"]),
        cable_length_m=float(measured["length_m"]),
        cable_mass_kg=float(measured["mass_kg"]),
        cable_diameter_m=float(measured["diameter_m"]),
        bending_stiffness_n_m2=float(optimized["bending_stiffness_n_m2"]),
        bending_damping_n_m2_s=float(optimized["bending_damping_n_m2_s"]),
        substeps=int(solver["substeps"]),
        constraint_iterations=int(solver["constraint_iterations"]),
        student_t_degrees_of_freedom=float(observation["degrees_of_freedom"]),
        unresolved_curve_noise_m=float(observation["unresolved_curve_scale_m"]),
        unresolved_endpoint_noise_m=float(
            observation["unresolved_endpoint_scale_m"]
        ),
        process_acceleration_sigma_m_s2=float(
            process["interior_acceleration_sigma_m_s2"]
        ),
        endpoint_acceleration_sigma_m_s2=float(
            process["unobserved_endpoint_acceleration_sigma_m_s2"]
        ),
        payload=payload,
    )
    positive = (
        artifact.cable_length_m,
        artifact.cable_mass_kg,
        artifact.cable_diameter_m,
        artifact.bending_stiffness_n_m2,
        artifact.bending_damping_n_m2_s,
        artifact.student_t_degrees_of_freedom,
        artifact.process_acceleration_sigma_m_s2,
        artifact.endpoint_acceleration_sigma_m_s2,
    )
    if artifact.node_count < 6 or min(artifact.substeps, artifact.constraint_iterations) < 1:
        raise ValueError("DDER model discretization is invalid.")
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("DDER model physical values must be finite and positive.")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (
            artifact.unresolved_curve_noise_m,
            artifact.unresolved_endpoint_noise_m,
        )
    ):
        raise ValueError("DDER model unresolved observation scales must be non-negative.")
    return artifact
