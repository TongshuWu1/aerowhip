"""Shared differentiable rod mechanics for offline DDER identification.

The cable has a circular, isotropic cross-section and no observable material
frame texture.  Its state is therefore an ordered centreline with no fitted
twist variable.  Bending uses the geometric discrete-curvature-binormal energy
of a discrete elastic rod, and inextensibility uses differentiable,
mass-weighted position constraints.  Offline identification and future online
prediction must use this same transition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as functional


DDER_MODEL_SCHEMA_VERSION = 1
DDER_MODEL_FAMILY = "differentiable_discrete_elastic_rod"
DDER_EQUATIONS = "twist_free_isotropic_dder_v1"
DDER_ASSUMPTIONS = {
    "cross_section": "circular_isotropic",
    "rest_curvature": "straight",
    "twist_state": "omitted_unobservable",
    "stretch": "hard_inextensible",
}


def _batch_parameter(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    parameter = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if parameter.ndim == 0:
        return parameter
    if parameter.shape != (reference.shape[0],):
        raise ValueError(f"{name} must be scalar or contain one value per batch item")
    return parameter


def twist_free_bending_energy(
    positions: torch.Tensor,
    bending_stiffness_n_m2: torch.Tensor | float,
    rest_segment_length_m: float,
) -> torch.Tensor:
    """Return geometric isotropic DER bending energy for each centreline.

    For unit edge tangents ``t`` the curvature binormal at an interior vertex is
    ``kb = 2 (t_prev x t_next) / (1 + t_prev . t_next)``.  With a straight
    rest shape and uniform Voronoi rest length ``h``, the energy is
    ``0.5 EI sum(||kb||^2 / h)``.
    """

    if positions.ndim != 3 or positions.shape[1] < 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3 with at least three nodes")
    if not positions.is_floating_point():
        raise ValueError("positions must use a floating-point dtype")
    if not math.isfinite(rest_segment_length_m) or rest_segment_length_m <= 0.0:
        raise ValueError("rest_segment_length_m must be finite and positive")
    stiffness = _batch_parameter(
        bending_stiffness_n_m2,
        positions,
        "bending_stiffness_n_m2",
    )
    edges = positions[:, 1:] - positions[:, :-1]
    epsilon = 64.0 * torch.finfo(positions.dtype).eps
    lengths = torch.linalg.vector_norm(edges, dim=-1, keepdim=True)
    tangents = edges / torch.clamp(lengths, min=epsilon * rest_segment_length_m)
    tangent_dot = torch.sum(tangents[:, :-1] * tangents[:, 1:], dim=-1)
    denominator = torch.clamp(1.0 + tangent_dot, min=epsilon)
    curvature_binormal = (
        2.0
        * torch.cross(tangents[:, :-1], tangents[:, 1:], dim=-1)
        / denominator[..., None]
    )
    unit_stiffness_energy = 0.5 * torch.sum(
        curvature_binormal * curvature_binormal,
        dim=(1, 2),
    ) / rest_segment_length_m
    return unit_stiffness_energy * stiffness


def twist_free_bending_force(
    positions: torch.Tensor,
    bending_stiffness_n_m2: torch.Tensor | float,
    rest_segment_length_m: float,
) -> torch.Tensor:
    """Return ``-dE_b/dq`` while preserving higher derivatives when needed."""

    parameter_requires_grad = bool(
        isinstance(bending_stiffness_n_m2, torch.Tensor)
        and bending_stiffness_n_m2.requires_grad
    )
    create_graph = torch.is_grad_enabled() and (
        positions.requires_grad or parameter_requires_grad
    )
    with torch.enable_grad():
        differentiable_positions = positions
        if not differentiable_positions.requires_grad:
            differentiable_positions = positions.detach().requires_grad_(True)
        energy = twist_free_bending_energy(
            differentiable_positions,
            bending_stiffness_n_m2,
            rest_segment_length_m,
        )
        force = -torch.autograd.grad(
            energy.sum(),
            differentiable_positions,
            create_graph=create_graph,
        )[0]
    return force if create_graph else force.detach()


def project_inextensible(
    positions: torch.Tensor,
    segment_length_m: float,
    endpoints: torch.Tensor | None = None,
    *,
    iterations: int,
    relaxation: float = 1.0,
) -> torch.Tensor:
    """Project a batched chain with a differentiable coupled SHAKE solve."""

    if positions.ndim != 3 or positions.shape[1] < 2 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3")
    if not positions.is_floating_point():
        raise ValueError("positions must use a floating-point dtype")
    if not math.isfinite(segment_length_m) or segment_length_m <= 0.0:
        raise ValueError("segment_length_m must be finite and positive")
    if iterations <= 0 or not 0.0 < relaxation <= 1.0:
        raise ValueError("projection settings are invalid")
    fixed_endpoints = positions[:, (0, -1)].clone() if endpoints is None else endpoints
    if fixed_endpoints.shape != (positions.shape[0], 2, 3):
        raise ValueError("endpoints must have shape Bx2x3")
    if (
        fixed_endpoints.device != positions.device
        or fixed_endpoints.dtype != positions.dtype
    ):
        raise ValueError("endpoints must share the positions device and dtype")

    def set_endpoints(value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (fixed_endpoints[:, :1], value[:, 1:-1], fixed_endpoints[:, 1:]),
            dim=1,
        )

    value = set_endpoints(positions)
    node_weights = torch.ones(
        (1, positions.shape[1]),
        dtype=value.dtype,
        device=value.device,
    )
    node_weights[:, (0, -1)] = 0.0
    left_weight = node_weights[:, :-1]
    right_weight = node_weights[:, 1:]
    rest = torch.as_tensor(segment_length_m, dtype=value.dtype, device=value.device)
    epsilon = torch.finfo(value.dtype).eps
    constraint_count = positions.shape[1] - 1
    identity = torch.eye(
        constraint_count,
        dtype=value.dtype,
        device=value.device,
    ).unsqueeze(0)
    for _ in range(iterations):
        delta = value[:, 1:] - value[:, :-1]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / torch.clamp(distance[..., None], min=epsilon)
        diagonal = (left_weight + right_weight).expand(value.shape[0], -1)
        shared_weight = node_weights[:, 1:-1]
        adjacent_dot = torch.sum(direction[:, :-1] * direction[:, 1:], dim=-1)
        off_diagonal = -shared_weight * adjacent_dot
        system = torch.diag_embed(diagonal)
        system = system + torch.diag_embed(off_diagonal, offset=1)
        system = system + torch.diag_embed(off_diagonal, offset=-1)
        system = system + 32.0 * epsilon * identity
        multiplier = torch.linalg.solve(
            system,
            -(distance - rest).unsqueeze(-1),
        ).squeeze(-1)
        segment_correction = relaxation * multiplier[..., None] * direction
        left = -left_weight[..., None] * segment_correction
        right = right_weight[..., None] * segment_correction
        correction = functional.pad(left, (0, 0, 0, 1)) + functional.pad(
            right,
            (0, 0, 1, 0),
        )
        value = set_endpoints(value + correction)
    return value


@dataclass(frozen=True, slots=True)
class DderModelParameters:
    """Physical and numerical parameters required by a deployed DDER model."""

    node_count: int
    cable_length_m: float
    cable_radius_m: float
    linear_density_kg_m: float
    bending_stiffness_n_m2: float
    velocity_damping_s_inv: float
    gravity_camera_m_s2: tuple[float, float, float]
    substeps: int
    constraint_iterations: int
    constraint_relaxation: float = 1.0

    def __post_init__(self) -> None:
        if self.node_count < 4:
            raise ValueError("DDER model requires at least four nodes")
        for name in (
            "cable_length_m",
            "cable_radius_m",
            "linear_density_kg_m",
            "bending_stiffness_n_m2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.velocity_damping_s_inv)
            or self.velocity_damping_s_inv < 0.0
        ):
            raise ValueError("velocity_damping_s_inv must be finite and nonnegative")
        if len(self.gravity_camera_m_s2) != 3 or not all(
            math.isfinite(value) for value in self.gravity_camera_m_s2
        ):
            raise ValueError("gravity_camera_m_s2 must contain three finite values")
        if self.substeps <= 0 or self.constraint_iterations <= 0:
            raise ValueError("DDER solver counts must be positive")
        if not 0.0 < self.constraint_relaxation <= 1.0:
            raise ValueError("constraint_relaxation must be in (0, 1]")

    @property
    def segment_length_m(self) -> float:
        return self.cable_length_m / (self.node_count - 1)


class BatchedDderModel:
    """Differentiable batched twist-free DDER transition on a torch device."""

    def __init__(self, parameters: DderModelParameters) -> None:
        if not isinstance(parameters, DderModelParameters):
            raise TypeError("parameters must be DderModelParameters")
        self.parameters = parameters
        self._constant_cache: dict[
            tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _validate_state(self, positions: torch.Tensor, velocities: torch.Tensor) -> None:
        expected_tail = (self.parameters.node_count, 3)
        if positions.ndim != 3 or tuple(positions.shape[1:]) != expected_tail:
            raise ValueError(f"positions must have shape Bx{expected_tail[0]}x3")
        if velocities.shape != positions.shape:
            raise ValueError("velocities must match positions")
        if positions.device != velocities.device or positions.dtype != velocities.dtype:
            raise ValueError("positions and velocities must share device and dtype")
        if not positions.is_floating_point():
            raise ValueError("DDER state must use floating-point tensors")

    def _mass_and_gravity(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (state.device, state.dtype)
        constants = self._constant_cache.get(key)
        if constants is None:
            mass_weights = torch.ones(
                (1, self.parameters.node_count, 1),
                dtype=state.dtype,
                device=state.device,
            )
            mass_weights[:, (0, -1)] = 0.5
            node_mass = (
                self.parameters.linear_density_kg_m
                * self.parameters.segment_length_m
                * mass_weights
            )
            gravity = torch.as_tensor(
                self.parameters.gravity_camera_m_s2,
                dtype=state.dtype,
                device=state.device,
            ).reshape(1, 1, 3)
            constants = (node_mass, gravity)
            self._constant_cache[key] = constants
        return constants

    def bending_energy(
        self,
        positions: torch.Tensor,
        *,
        bending_stiffness_n_m2: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        stiffness = (
            self.parameters.bending_stiffness_n_m2
            if bending_stiffness_n_m2 is None
            else bending_stiffness_n_m2
        )
        return twist_free_bending_energy(
            positions,
            stiffness,
            self.parameters.segment_length_m,
        )

    def project_inextensible(
        self,
        positions: torch.Tensor,
        endpoints: torch.Tensor | None = None,
        *,
        iterations: int | None = None,
    ) -> torch.Tensor:
        """Project every segment to the fixed rest length with clamped ends."""

        count = self.parameters.constraint_iterations if iterations is None else iterations
        return project_inextensible(
            positions,
            self.parameters.segment_length_m,
            endpoints,
            iterations=count,
            relaxation=self.parameters.constraint_relaxation,
        )

    def acceleration(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        *,
        bending_stiffness_n_m2: torch.Tensor | float | None = None,
        velocity_damping_s_inv: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Return gravity, geometric bending, and viscous acceleration."""

        self._validate_state(positions, velocities)
        stiffness = _batch_parameter(
            self.parameters.bending_stiffness_n_m2
            if bending_stiffness_n_m2 is None
            else bending_stiffness_n_m2,
            positions,
            "bending_stiffness_n_m2",
        )
        damping = _batch_parameter(
            self.parameters.velocity_damping_s_inv
            if velocity_damping_s_inv is None
            else velocity_damping_s_inv,
            positions,
            "velocity_damping_s_inv",
        )
        bending_force = twist_free_bending_force(
            positions,
            stiffness,
            self.parameters.segment_length_m,
        )
        node_mass, gravity = self._mass_and_gravity(positions)
        if damping.ndim == 1:
            damping = damping.reshape(-1, 1, 1)
        return bending_force / node_mass + gravity - damping * velocities

    def step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        endpoint_positions_next: torch.Tensor,
        dt_s: torch.Tensor | float,
        *,
        bending_stiffness_n_m2: torch.Tensor | float | None = None,
        velocity_damping_s_inv: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance one camera interval with measured endpoint boundary motion."""

        self._validate_state(positions, velocities)
        if endpoint_positions_next.shape != (positions.shape[0], 2, 3):
            raise ValueError("endpoint_positions_next must have shape Bx2x3")
        if (
            endpoint_positions_next.device != positions.device
            or endpoint_positions_next.dtype != positions.dtype
        ):
            raise ValueError("endpoint_positions_next must share state device and dtype")
        if not isinstance(dt_s, torch.Tensor):
            if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
                raise ValueError("dt_s must be finite and positive")
        dt = torch.as_tensor(dt_s, dtype=positions.dtype, device=positions.device)
        if dt.ndim == 0:
            dt = dt.expand(positions.shape[0])
        if dt.shape != (positions.shape[0],):
            raise ValueError("dt_s must be scalar or contain one value per batch item")
        if not dt.is_cuda and bool(torch.any(~torch.isfinite(dt) | (dt <= 0.0))):
            raise ValueError("dt_s must be finite and positive")

        start_endpoints = positions[:, (0, -1)]
        q, v = positions, velocities
        substeps = self.parameters.substeps
        h = (dt / substeps).reshape(-1, 1, 1)
        for substep in range(substeps):
            fraction = float(substep + 1) / substeps
            substep_endpoints = start_endpoints + fraction * (
                endpoint_positions_next - start_endpoints
            )
            acceleration = self.acceleration(
                q,
                v,
                bending_stiffness_n_m2=bending_stiffness_n_m2,
                velocity_damping_s_inv=velocity_damping_s_inv,
            )
            predicted_velocity = v + h * acceleration
            predicted_position = q + h * predicted_velocity
            projected_position = self.project_inextensible(
                predicted_position,
                substep_endpoints,
            )
            v = (projected_position - q) / h
            q = projected_position
        return q, v

    def maximum_segment_error_m(self, positions: torch.Tensor) -> torch.Tensor:
        lengths = torch.linalg.vector_norm(
            positions[:, 1:] - positions[:, :-1],
            dim=-1,
        )
        return torch.amax(torch.abs(lengths - self.parameters.segment_length_m), dim=1)


def save_dder_model(
    path: Path,
    parameters: DderModelParameters,
    *,
    identification: Mapping[str, Any],
) -> None:
    """Atomically save a human-readable DDER model and provenance."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    payload = {
        "schema_version": DDER_MODEL_SCHEMA_VERSION,
        "model_family": DDER_MODEL_FAMILY,
        "equations": DDER_EQUATIONS,
        "assumptions": dict(DDER_ASSUMPTIONS),
        "parameters": asdict(parameters),
        "identification": dict(identification),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def load_dder_model(path: Path) -> tuple[DderModelParameters, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != DDER_MODEL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported DDER model schema: {source}")
    if payload.get("model_family") != DDER_MODEL_FAMILY:
        raise ValueError(f"Unsupported model family: {payload.get('model_family')!r}")
    if payload.get("equations") != DDER_EQUATIONS:
        raise ValueError(f"Unsupported DDER equations: {payload.get('equations')!r}")
    if payload.get("assumptions") != DDER_ASSUMPTIONS:
        raise ValueError("DDER model assumptions do not match this implementation")
    values = payload.get("parameters")
    if not isinstance(values, dict):
        raise ValueError("DDER model has no parameter mapping")
    values = dict(values)
    values["gravity_camera_m_s2"] = tuple(values["gravity_camera_m_s2"])
    parameters = DderModelParameters(**values)
    identification = payload.get("identification")
    if not isinstance(identification, dict):
        raise ValueError("DDER model has no identification provenance")
    return parameters, identification


__all__ = [
    "BatchedDderModel",
    "DDER_ASSUMPTIONS",
    "DDER_EQUATIONS",
    "DDER_MODEL_FAMILY",
    "DDER_MODEL_SCHEMA_VERSION",
    "DderModelParameters",
    "load_dder_model",
    "project_inextensible",
    "save_dder_model",
    "twist_free_bending_energy",
    "twist_free_bending_force",
]
