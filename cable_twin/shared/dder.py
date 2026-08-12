"""Exact, torsion-free differentiable Discrete Elastic Rod dynamics.

The Phase 1 cable is naturally straight and has an approximately circular,
isotropic cross-section.  Under those stated assumptions the material twist
decouples from centreline bending, so the physical state contains only ordered
vertex positions and velocities.  Bending still uses the nonlinear DER
curvature binormal; this is not a small-curvature spring approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


EXPLICIT_STABILITY_LIMIT = 1.5
EXPLICIT_DAMPING_STABILITY_LIMIT = 1.8


def _as_batch_scalar(
    value: torch.Tensor | float,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim == 0:
        return result.expand(reference.shape[0])
    if result.shape != (reference.shape[0],):
        raise ValueError(f"{name} must be scalar or contain one value per batch item.")
    return result


def _normalize(vector: torch.Tensor, *, name: str) -> torch.Tensor:
    epsilon = 64.0 * torch.finfo(vector.dtype).eps
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if bool(torch.any(norm <= epsilon).detach().cpu()):
        raise ValueError(f"{name} contains a zero-length vector.")
    return vector / norm


def _normalize_unchecked(vector: torch.Tensor) -> torch.Tensor:
    epsilon = 64.0 * torch.finfo(vector.dtype).eps
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector / torch.clamp(norm, min=epsilon)


def _curvature_binormals_impl(
    positions: torch.Tensor,
    *,
    validate: bool,
) -> torch.Tensor:
    edges = positions[:, 1:] - positions[:, :-1]
    tangents = (
        _normalize(edges, name="rod edges")
        if validate
        else _normalize_unchecked(edges)
    )
    previous = tangents[:, :-1]
    following = tangents[:, 1:]
    denominator = 1.0 + torch.sum(previous * following, dim=-1)
    if validate:
        singular = denominator <= 128.0 * torch.finfo(positions.dtype).eps
        if bool(torch.any(singular).detach().cpu()):
            raise ValueError("DER curvature is singular at an exact 180-degree fold.")
    denominator = torch.clamp(
        denominator,
        min=128.0 * torch.finfo(positions.dtype).eps,
    )
    return (
        2.0
        * torch.linalg.cross(previous, following, dim=-1)
        / denominator[..., None]
    )


def curvature_binormals(positions: torch.Tensor) -> torch.Tensor:
    """Return the exact DER curvature binormal at every interior vertex."""

    if positions.ndim != 3 or positions.shape[1] < 4 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3 with N >= 4.")
    if not positions.is_floating_point():
        raise ValueError("positions must use a floating-point dtype.")
    return _curvature_binormals_impl(positions, validate=True)


def _curvature_binormal_rates_impl(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    *,
    validate: bool,
) -> torch.Tensor:
    """Material time derivative of the nonlinear DER curvature binormal."""

    edges = positions[:, 1:] - positions[:, :-1]
    edge_velocities = velocities[:, 1:] - velocities[:, :-1]
    epsilon = 64.0 * torch.finfo(positions.dtype).eps
    lengths = torch.linalg.vector_norm(edges, dim=-1, keepdim=True)
    if validate and bool(torch.any(lengths <= epsilon).detach().cpu()):
        raise ValueError("rod edges contains a zero-length vector.")
    lengths = torch.clamp(lengths, min=epsilon)
    tangents = edges / lengths
    tangent_rates = (
        edge_velocities
        - tangents * torch.sum(tangents * edge_velocities, dim=-1, keepdim=True)
    ) / lengths

    previous, following = tangents[:, :-1], tangents[:, 1:]
    previous_rate, following_rate = tangent_rates[:, :-1], tangent_rates[:, 1:]
    cross = torch.linalg.cross(previous, following, dim=-1)
    cross_rate = torch.linalg.cross(previous_rate, following, dim=-1)
    cross_rate = cross_rate + torch.linalg.cross(
        previous, following_rate, dim=-1
    )
    denominator = 1.0 + torch.sum(previous * following, dim=-1)
    denominator_rate = torch.sum(previous_rate * following, dim=-1)
    denominator_rate = denominator_rate + torch.sum(
        previous * following_rate, dim=-1
    )
    if validate:
        singular = denominator <= 128.0 * torch.finfo(positions.dtype).eps
        if bool(torch.any(singular).detach().cpu()):
            raise ValueError("DER curvature is singular at an exact 180-degree fold.")
    denominator = torch.clamp(
        denominator,
        min=128.0 * torch.finfo(positions.dtype).eps,
    )
    return 2.0 * (
        cross_rate / denominator[..., None]
        - cross * denominator_rate[..., None] / denominator.square()[..., None]
    )


def curvature_binormal_rates(
    positions: torch.Tensor,
    velocities: torch.Tensor,
) -> torch.Tensor:
    """Return curvature-binormal rates for a batched rod state."""

    if positions.ndim != 3 or positions.shape[1] < 4 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3 with N >= 4.")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions.")
    return _curvature_binormal_rates_impl(positions, velocities, validate=True)


def solve_symmetric_tridiagonal(
    diagonal: torch.Tensor,
    off_diagonal: torch.Tensor,
    right_hand_side: torch.Tensor,
) -> torch.Tensor:
    """Solve a batched symmetric tridiagonal system by Thomas elimination."""

    if diagonal.ndim != 2 or right_hand_side.shape != diagonal.shape:
        raise ValueError("Diagonal and right-hand side must have shape BxN.")
    if off_diagonal.shape != (diagonal.shape[0], diagonal.shape[1] - 1):
        raise ValueError("Off diagonal must have shape Bx(N-1).")
    epsilon = 64.0 * torch.finfo(diagonal.dtype).eps

    def safe(value: torch.Tensor) -> torch.Tensor:
        sign = torch.where(value < 0.0, -torch.ones_like(value), torch.ones_like(value))
        return torch.where(torch.abs(value) < epsilon, sign * epsilon, value)

    modified_upper: list[torch.Tensor] = []
    modified_rhs: list[torch.Tensor] = []
    denominator = safe(diagonal[:, 0])
    modified_rhs.append(right_hand_side[:, 0] / denominator)
    if diagonal.shape[1] > 1:
        modified_upper.append(off_diagonal[:, 0] / denominator)
    for index in range(1, diagonal.shape[1]):
        denominator = safe(
            diagonal[:, index]
            - off_diagonal[:, index - 1] * modified_upper[index - 1]
        )
        modified_rhs.append(
            (
                right_hand_side[:, index]
                - off_diagonal[:, index - 1] * modified_rhs[index - 1]
            )
            / denominator
        )
        if index < diagonal.shape[1] - 1:
            modified_upper.append(off_diagonal[:, index] / denominator)
    solution: list[torch.Tensor] = [modified_rhs[-1]]
    for index in range(diagonal.shape[1] - 2, -1, -1):
        solution.append(
            modified_rhs[index] - modified_upper[index] * solution[-1]
        )
    return torch.stack(solution[::-1], dim=1)


def momentum_project_lengths(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    boundary_positions: torch.Tensor | None,
    *,
    iterations: int,
) -> torch.Tensor:
    """Coupled inverse-mass edge projection with optional pinned endpoints."""

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3.")
    batch, node_count, _ = positions.shape
    if rest_lengths.shape != (node_count - 1,):
        raise ValueError("rest_lengths must contain one value per edge.")
    if masses.shape != (node_count,):
        raise ValueError("masses must contain one value per vertex.")
    if boundary_positions is not None and boundary_positions.shape != (batch, 2, 3):
        raise ValueError("boundary_positions must have shape Bx2x3.")
    if iterations < 1:
        raise ValueError("Projection iterations must be positive.")

    inverse_mass = torch.reciprocal(masses).clone()
    if boundary_positions is not None:
        inverse_mass = torch.cat(
            (
                torch.zeros_like(inverse_mass[:1]),
                inverse_mass[1:-1],
                torch.zeros_like(inverse_mass[-1:]),
            )
        )

    def clamp_boundary(value: torch.Tensor) -> torch.Tensor:
        if boundary_positions is None:
            return value
        return torch.cat(
            (boundary_positions[:, :1], value[:, 1:-1], boundary_positions[:, 1:]),
            dim=1,
        )

    value = clamp_boundary(positions)
    epsilon = 64.0 * torch.finfo(value.dtype).eps
    for _ in range(iterations):
        delta = value[:, 1:] - value[:, :-1]
        distance = torch.linalg.vector_norm(delta, dim=-1)
        direction = delta / torch.clamp(distance[..., None], min=epsilon)
        diagonal = (inverse_mass[:-1] + inverse_mass[1:])[None].expand(batch, -1)
        adjacent = torch.sum(direction[:, :-1] * direction[:, 1:], dim=-1)
        off_diagonal = -inverse_mass[1:-1][None] * adjacent
        solver_regularization = 8.0 * torch.finfo(value.dtype).eps
        regularization = solver_regularization * torch.amax(
            diagonal, dim=1, keepdim=True
        )
        multiplier = solve_symmetric_tridiagonal(
            diagonal + regularization,
            off_diagonal,
            -(distance - rest_lengths[None]),
        )
        edge_correction = multiplier[..., None] * direction
        left = -inverse_mass[:-1][None, :, None] * edge_correction
        right = inverse_mass[1:][None, :, None] * edge_correction
        node_correction = torch.nn.functional.pad(left, (0, 0, 0, 1))
        node_correction = node_correction + torch.nn.functional.pad(
            right, (0, 0, 1, 0)
        )
        value = clamp_boundary(value + node_correction)
    return value


def momentum_project_velocities(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    masses: torch.Tensor,
    boundary_velocities: torch.Tensor,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Mass-weighted projection onto the inextensible velocity constraints."""

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3.")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions.")
    batch, node_count, _ = positions.shape
    if masses.shape != (node_count,):
        raise ValueError("masses must contain one value per vertex.")
    if boundary_velocities.shape != (batch, 2, 3):
        raise ValueError("boundary_velocities must have shape Bx2x3.")

    inverse_mass = torch.reciprocal(masses).clone()
    inverse_mass = torch.cat(
        (
            torch.zeros_like(inverse_mass[:1]),
            inverse_mass[1:-1],
            torch.zeros_like(inverse_mass[-1:]),
        )
    )
    value = torch.cat(
        (boundary_velocities[:, :1], velocities[:, 1:-1], boundary_velocities[:, 1:]),
        dim=1,
    )
    edges = positions[:, 1:] - positions[:, :-1]
    direction = (
        _normalize(edges, name="rod edges")
        if validate
        else _normalize_unchecked(edges)
    )
    diagonal = (inverse_mass[:-1] + inverse_mass[1:])[None].expand(batch, -1)
    adjacent = torch.sum(direction[:, :-1] * direction[:, 1:], dim=-1)
    off_diagonal = -inverse_mass[1:-1][None] * adjacent
    constraint_velocity = torch.sum(
        direction * (value[:, 1:] - value[:, :-1]),
        dim=-1,
    )
    solver_regularization = 8.0 * torch.finfo(value.dtype).eps
    regularization = solver_regularization * torch.amax(
        diagonal, dim=1, keepdim=True
    )
    multiplier = solve_symmetric_tridiagonal(
        diagonal + regularization,
        off_diagonal,
        -constraint_velocity,
    )
    edge_correction = multiplier[..., None] * direction
    left = -inverse_mass[:-1][None, :, None] * edge_correction
    right = inverse_mass[1:][None, :, None] * edge_correction
    node_correction = torch.nn.functional.pad(left, (0, 0, 0, 1))
    node_correction = node_correction + torch.nn.functional.pad(
        right,
        (0, 0, 1, 0),
    )
    corrected = value + node_correction
    return torch.cat(
        (
            boundary_velocities[:, :1],
            corrected[:, 1:-1],
            boundary_velocities[:, 1:],
        ),
        dim=1,
    )


@dataclass(frozen=True, slots=True)
class DderParameters:
    node_count: int
    cable_length_m: float
    cable_mass_kg: float
    cable_diameter_m: float
    bending_stiffness_n_m2: float
    bending_damping_n_m2_s: float
    gravity_camera_m_s2: tuple[float, float, float]
    substeps: int = 2
    constraint_iterations: int = 8

    def __post_init__(self) -> None:
        if self.node_count < 6:
            raise ValueError("Pinned-endpoint DDER requires at least six vertices.")
        positive = (
            self.cable_length_m,
            self.cable_mass_kg,
            self.cable_diameter_m,
            self.bending_stiffness_n_m2,
            self.bending_damping_n_m2_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("DDER physical parameters must be finite and positive.")
        if len(self.gravity_camera_m_s2) != 3 or not all(
            math.isfinite(value) for value in self.gravity_camera_m_s2
        ):
            raise ValueError("Gravity must contain three finite camera-frame values.")
        if min(self.substeps, self.constraint_iterations) < 1:
            raise ValueError("DDER solver counts must be positive.")

    @property
    def segment_length_m(self) -> float:
        return self.cable_length_m / (self.node_count - 1)


@dataclass(frozen=True, slots=True)
class DderState:
    positions_m: torch.Tensor
    velocities_m_s: torch.Tensor


@dataclass(frozen=True, slots=True)
class DderRuntimeConstants:
    rest_lengths_m: torch.Tensor
    masses_kg: torch.Tensor
    dual_lengths_m: torch.Tensor
    gravity_m_s2: torch.Tensor
    bending_stiffness_n_m2: torch.Tensor
    bending_damping_n_m2_s: torch.Tensor


class DderModel:
    """Batched CUDA transition for a straight, isotropic, inextensible cable."""

    def __init__(self, parameters: DderParameters) -> None:
        self.parameters = parameters
        self._rest_lengths = torch.full(
            (parameters.node_count - 1,),
            parameters.segment_length_m,
            dtype=torch.float64,
        )
        # Largest generalized eigenvalue of the straight-rod, unit-EI
        # small-deflection bending operator with fixed endpoint translations.
        # The runtime pins only terminal translations and leaves endpoint
        # rotation free, matching this stability operator.
        node_count = parameters.node_count
        second_difference = torch.zeros(
            (node_count - 2, node_count),
            dtype=torch.float64,
        )
        rows = torch.arange(node_count - 2)
        second_difference[rows, rows] = 1.0
        second_difference[rows, rows + 1] = -2.0
        second_difference[rows, rows + 2] = 1.0
        unit_stiffness = (
            second_difference.T @ second_difference / parameters.segment_length_m**3
        )
        rest_lengths, masses = self._constants(self._rest_lengths)
        del rest_lengths
        free_mass = masses[1:-1]
        free_stiffness = unit_stiffness[1:-1, 1:-1]
        mass_scaled = free_stiffness / torch.sqrt(
            free_mass[:, None] * free_mass[None, :]
        )
        self._maximum_bending_eigenvalue_per_ei = float(
            torch.linalg.eigvalsh(mass_scaled).max()
        )

    def maximum_stable_bending_stiffness(
        self,
        maximum_dt_s: float,
    ) -> float:
        """Conservative EI limit for the present explicit bending step."""

        dt = float(maximum_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("maximum_dt_s must be finite and positive.")
        substep = dt / self.parameters.substeps
        return (
            (EXPLICIT_STABILITY_LIMIT / substep) ** 2
            / self._maximum_bending_eigenvalue_per_ei
        )

    def maximum_stable_bending_damping(self, maximum_dt_s: float) -> float:
        """Conservative curvature-rate damping limit for the explicit step."""

        dt = float(maximum_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("maximum_dt_s must be finite and positive.")
        substep = dt / self.parameters.substeps
        return EXPLICIT_DAMPING_STABILITY_LIMIT / (
            substep * self._maximum_bending_eigenvalue_per_ei
        )

    def _constants(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rest_lengths = self._rest_lengths.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        masses = torch.empty(
            self.parameters.node_count,
            dtype=reference.dtype,
            device=reference.device,
        )
        linear_density = self.parameters.cable_mass_kg / self.parameters.cable_length_m
        masses[0] = 0.5 * rest_lengths[0]
        masses[-1] = 0.5 * rest_lengths[-1]
        masses[1:-1] = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
        masses = masses * linear_density
        return rest_lengths, masses

    def runtime_constants(self, reference: torch.Tensor) -> DderRuntimeConstants:
        """Materialize immutable inference constants on the state's device."""

        rest_lengths, masses = self._constants(reference)
        return DderRuntimeConstants(
            rest_lengths_m=rest_lengths,
            masses_kg=masses,
            dual_lengths_m=0.5 * (rest_lengths[:-1] + rest_lengths[1:]),
            gravity_m_s2=torch.tensor(
                self.parameters.gravity_camera_m_s2,
                dtype=reference.dtype,
                device=reference.device,
            )[None, None],
            bending_stiffness_n_m2=torch.full(
                (reference.shape[0],),
                self.parameters.bending_stiffness_n_m2,
                dtype=reference.dtype,
                device=reference.device,
            ),
            bending_damping_n_m2_s=torch.full(
                (reference.shape[0],),
                self.parameters.bending_damping_n_m2_s,
                dtype=reference.dtype,
                device=reference.device,
            ),
        )

    @staticmethod
    def _runtime_internal_force(
        positions: torch.Tensor,
        constants: DderRuntimeConstants,
    ) -> torch.Tensor:
        with torch.enable_grad():
            q = positions.detach().requires_grad_(True)
            curvature = _curvature_binormals_impl(q, validate=False)
            energy = 0.5 * constants.bending_stiffness_n_m2 * torch.sum(
                torch.sum(curvature.square(), dim=-1)
                / constants.dual_lengths_m[None],
                dim=1,
            )
            return -torch.autograd.grad(energy.sum(), q, create_graph=False)[0]

    @staticmethod
    def _runtime_damping_force(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        constants: DderRuntimeConstants,
    ) -> torch.Tensor:
        with torch.enable_grad():
            q = positions.detach()
            v = velocities.detach().requires_grad_(True)
            rate = _curvature_binormal_rates_impl(q, v, validate=False)
            dissipation = 0.5 * constants.bending_damping_n_m2_s * torch.sum(
                torch.sum(rate.square(), dim=-1)
                / constants.dual_lengths_m[None],
                dim=1,
            )
            return -torch.autograd.grad(
                dissipation.sum(), v, create_graph=False
            )[0]

    def bending_energy(
        self,
        positions: torch.Tensor,
        bending_stiffness_n_m2: torch.Tensor | float | None = None,
        *,
        _validate: bool = True,
    ) -> torch.Tensor:
        """Nonlinear DER bending energy for zero intrinsic curvature."""

        curvature = _curvature_binormals_impl(positions, validate=_validate)
        rest_lengths, _masses = self._constants(positions)
        dual_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
        stiffness = _as_batch_scalar(
            self.parameters.bending_stiffness_n_m2
            if bending_stiffness_n_m2 is None
            else bending_stiffness_n_m2,
            positions,
            "bending_stiffness_n_m2",
        )
        return 0.5 * stiffness * torch.sum(
            torch.sum(curvature.square(), dim=-1) / dual_lengths[None],
            dim=1,
        )

    def internal_force(
        self,
        positions: torch.Tensor,
        *,
        create_graph: bool,
        bending_stiffness_n_m2: torch.Tensor | float | None = None,
        _validate: bool = True,
    ) -> torch.Tensor:
        with torch.enable_grad():
            q = (
                positions
                if positions.requires_grad
                else positions.detach().requires_grad_(True)
            )
            energy = self.bending_energy(
                q,
                bending_stiffness_n_m2,
                _validate=_validate,
            )
            return -torch.autograd.grad(
                energy.sum(),
                q,
                create_graph=create_graph,
            )[0]

    def damping_force(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        *,
        create_graph: bool,
        bending_damping_n_m2_s: torch.Tensor | float | None = None,
        _validate: bool = True,
    ) -> torch.Tensor:
        """Kelvin-Voigt bending force from the curvature-rate dissipation."""

        with torch.enable_grad():
            velocity = (
                velocities
                if velocities.requires_grad
                else velocities.detach().requires_grad_(True)
            )
            rate = _curvature_binormal_rates_impl(
                positions,
                velocity,
                validate=_validate,
            )
            rest_lengths, _masses = self._constants(positions)
            dual_lengths = 0.5 * (rest_lengths[:-1] + rest_lengths[1:])
            damping = _as_batch_scalar(
                self.parameters.bending_damping_n_m2_s
                if bending_damping_n_m2_s is None
                else bending_damping_n_m2_s,
                positions,
                "bending_damping_n_m2_s",
            )
            dissipation = 0.5 * damping * torch.sum(
                torch.sum(rate.square(), dim=-1) / dual_lengths[None],
                dim=1,
            )
            return -torch.autograd.grad(
                dissipation.sum(),
                velocity,
                create_graph=create_graph,
            )[0]

    def initial_state(
        self,
        positions_m: torch.Tensor,
        velocities_m_s: torch.Tensor | None = None,
    ) -> DderState:
        if positions_m.shape[1:] != (self.parameters.node_count, 3):
            raise ValueError(
                f"positions_m must have shape Bx{self.parameters.node_count}x3."
            )
        velocities = (
            torch.zeros_like(positions_m)
            if velocities_m_s is None
            else velocities_m_s
        )
        if velocities.shape != positions_m.shape:
            raise ValueError("velocities_m_s must match positions_m.")
        return DderState(positions_m, velocities)

    def project_lengths(
        self,
        positions_m: torch.Tensor,
        boundary_positions_m: torch.Tensor,
    ) -> torch.Tensor:
        rest_lengths, masses = self._constants(positions_m)
        return momentum_project_lengths(
            positions_m,
            rest_lengths,
            masses,
            boundary_positions_m,
            iterations=self.parameters.constraint_iterations,
        )

    def project_free_lengths(self, positions_m: torch.Tensor) -> torch.Tensor:
        """Enforce length while allowing image evidence to locate every node."""

        rest_lengths, masses = self._constants(positions_m)
        return momentum_project_lengths(
            positions_m,
            rest_lengths,
            masses,
            None,
            iterations=self.parameters.constraint_iterations,
        )

    def project_velocities(
        self,
        positions_m: torch.Tensor,
        velocities_m_s: torch.Tensor,
        boundary_velocities_m_s: torch.Tensor,
    ) -> torch.Tensor:
        _rest_lengths, masses = self._constants(positions_m)
        return momentum_project_velocities(
            positions_m,
            velocities_m_s,
            masses,
            boundary_velocities_m_s,
            validate=True,
        )

    def step(
        self,
        state: DderState,
        boundary_positions_next_m: torch.Tensor,
        dt_s: torch.Tensor | float,
        *,
        create_graph: bool = False,
        bending_stiffness_n_m2: torch.Tensor | float | None = None,
        bending_damping_n_m2_s: torch.Tensor | float | None = None,
        _validate: bool = True,
    ) -> DderState:
        q = state.positions_m
        v = state.velocities_m_s
        if q.shape[1:] != (self.parameters.node_count, 3) or v.shape != q.shape:
            raise ValueError("DDER state has invalid position/velocity shape.")
        if boundary_positions_next_m.shape != (q.shape[0], 2, 3):
            raise ValueError("boundary_positions_next_m must have shape Bx2x3.")
        dt = _as_batch_scalar(dt_s, q, "dt_s")
        if _validate and bool(
            torch.any(~torch.isfinite(dt) | (dt <= 0.0)).detach().cpu()
        ):
            raise ValueError("dt_s must be finite and positive.")
        damping = _as_batch_scalar(
            self.parameters.bending_damping_n_m2_s
            if bending_damping_n_m2_s is None
            else bending_damping_n_m2_s,
            q,
            "bending_damping_n_m2_s",
        )
        if _validate and bool(
            torch.any(~torch.isfinite(damping) | (damping <= 0.0)).detach().cpu()
        ):
            raise ValueError("bending_damping_n_m2_s must be finite and positive.")
        stiffness = _as_batch_scalar(
            self.parameters.bending_stiffness_n_m2
            if bending_stiffness_n_m2 is None
            else bending_stiffness_n_m2,
            q,
            "bending_stiffness_n_m2",
        )
        if _validate and bool(
            torch.any(~torch.isfinite(stiffness) | (stiffness <= 0.0)).detach().cpu()
        ):
            raise ValueError("bending_stiffness_n_m2 must be finite and positive.")
        if _validate:
            stability_number = (dt / self.parameters.substeps) * torch.sqrt(
                stiffness * self._maximum_bending_eigenvalue_per_ei
            )
            if bool(
                torch.any(stability_number > EXPLICIT_STABILITY_LIMIT).detach().cpu()
            ):
                raise ValueError(
                    "Explicit DDER bending step is outside its mass/grid/time-step "
                    "stability limit. Increase substeps or use an implicit bending step."
                )
            damping_stability_number = (
                (dt / self.parameters.substeps)
                * damping
                * self._maximum_bending_eigenvalue_per_ei
            )
            if bool(
                torch.any(
                    damping_stability_number > EXPLICIT_DAMPING_STABILITY_LIMIT
                ).detach().cpu()
            ):
                raise ValueError(
                    "Explicit DDER curvature damping is outside its stability limit. "
                    "Increase substeps or reduce the maximum fitted Cb."
                )

        rest_lengths, masses = self._constants(q)
        gravity = torch.as_tensor(
            self.parameters.gravity_camera_m_s2,
            dtype=q.dtype,
            device=q.device,
        )[None, None]
        # Slice concatenation remains entirely on-device during CUDA graph
        # capture; tuple indexing would materialize a CPU index tensor.
        start_boundary = torch.cat((q[:, :1], q[:, -1:]), dim=1)
        boundary_velocity = (
            boundary_positions_next_m - start_boundary
        ) / dt[:, None, None]
        substep_dt = (dt / self.parameters.substeps)[:, None, None]

        for substep in range(self.parameters.substeps):
            fraction = float(substep + 1) / self.parameters.substeps
            boundary = start_boundary + fraction * (
                boundary_positions_next_m - start_boundary
            )
            force = self.internal_force(
                q,
                create_graph=create_graph,
                bending_stiffness_n_m2=stiffness,
                _validate=_validate,
            )
            force = force + self.damping_force(
                q,
                v,
                create_graph=create_graph,
                bending_damping_n_m2_s=damping,
                _validate=_validate,
            )
            acceleration = force / masses[None, :, None] + gravity
            predicted_v = v + substep_dt * acceleration
            predicted_q = q + substep_dt * predicted_v
            predicted_q = torch.cat(
                (boundary[:, :1], predicted_q[:, 1:-1], boundary[:, 1:]),
                dim=1,
            )
            next_q = momentum_project_lengths(
                predicted_q,
                rest_lengths,
                masses,
                boundary,
                iterations=self.parameters.constraint_iterations,
            )
            provisional_v = (next_q - q) / substep_dt
            v = momentum_project_velocities(
                next_q,
                provisional_v,
                masses,
                boundary_velocity,
                validate=_validate,
            )
            q = next_q
        return DderState(q, v)

    def step_unchecked(
        self,
        state: DderState,
        boundary_positions_next_m: torch.Tensor,
        dt_s: torch.Tensor | float,
    ) -> DderState:
        """Validated-model inference step without host synchronizations.

        Callers must validate finite state, positive time step, positive fitted
        parameters, and the explicit stability bound before entering a fixed
        CUDA replay path.  The numerical DDER update is otherwise identical to
        :meth:`step`.
        """

        return self.step(
            state,
            boundary_positions_next_m,
            dt_s,
            create_graph=False,
            _validate=False,
        )

    def step_runtime(
        self,
        state: DderState,
        boundary_positions_next_m: torch.Tensor,
        dt_s: torch.Tensor,
        constants: DderRuntimeConstants,
    ) -> DderState:
        """Fixed-shape inference update suitable for CUDA graph capture."""

        q, v = state.positions_m, state.velocities_m_s
        dt = dt_s
        start_boundary = torch.cat((q[:, :1], q[:, -1:]), dim=1)
        boundary_velocity = (
            boundary_positions_next_m - start_boundary
        ) / dt[:, None, None]
        substep_dt = (dt / self.parameters.substeps)[:, None, None]
        for substep in range(self.parameters.substeps):
            fraction = float(substep + 1) / self.parameters.substeps
            boundary = start_boundary + fraction * (
                boundary_positions_next_m - start_boundary
            )
            force = self._runtime_internal_force(q, constants)
            force = force + self._runtime_damping_force(q, v, constants)
            acceleration = (
                force / constants.masses_kg[None, :, None]
                + constants.gravity_m_s2
            )
            predicted_v = v + substep_dt * acceleration
            predicted_q = q + substep_dt * predicted_v
            predicted_q = torch.cat(
                (boundary[:, :1], predicted_q[:, 1:-1], boundary[:, 1:]),
                dim=1,
            )
            next_q = momentum_project_lengths(
                predicted_q,
                constants.rest_lengths_m,
                constants.masses_kg,
                boundary,
                iterations=self.parameters.constraint_iterations,
            )
            provisional_v = (next_q - q) / substep_dt
            v = momentum_project_velocities(
                next_q,
                provisional_v,
                constants.masses_kg,
                boundary_velocity,
                validate=False,
            )
            q = next_q
        return DderState(q, v)

    def maximum_segment_error_m(self, positions: torch.Tensor) -> torch.Tensor:
        rest_lengths = self._rest_lengths.to(
            device=positions.device,
            dtype=positions.dtype,
        )
        lengths = torch.linalg.vector_norm(
            positions[:, 1:] - positions[:, :-1],
            dim=-1,
        )
        return torch.amax(torch.abs(lengths - rest_lengths[None]), dim=1)
