"""Differentiable Discrete Elastic Rod dynamics.

Bending uses the nonlinear DER curvature binormal.  When endpoint material
frames are supplied, torsion is the quasi-static minimum-energy twist of a
straight, homogeneous isotropic rod.  The material director is parallel
transported along the centreline and the terminal frame supplies the total
twist; this is the analytic homogeneous solution of the DER twist minimization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import torch


EXPLICIT_STABILITY_LIMIT = 1.5

PinnedEndpointMask = tuple[bool, bool]
TWO_PINNED_ENDPOINTS: PinnedEndpointMask = (True, True)
START_PINNED_FREE_END: PinnedEndpointMask = (True, False)


def _validate_pinned_endpoints(
    pinned_endpoints: PinnedEndpointMask,
    *,
    supported_only: bool = False,
) -> PinnedEndpointMask:
    if (
        not isinstance(pinned_endpoints, tuple)
        or len(pinned_endpoints) != 2
        or any(type(value) is not bool for value in pinned_endpoints)
    ):
        raise ValueError("pinned_endpoints must be a tuple of two booleans.")
    if supported_only and pinned_endpoints not in (
        TWO_PINNED_ENDPOINTS,
        START_PINNED_FREE_END,
    ):
        raise ValueError(
            "DDER dynamics supports either two pinned endpoints or a pinned "
            "start with a dynamically free end."
        )
    return pinned_endpoints


def _pinned_values(
    values: torch.Tensor,
    pinned_endpoints: PinnedEndpointMask,
) -> torch.Tensor:
    parts = []
    if pinned_endpoints[0]:
        parts.append(values[:, :1])
    if pinned_endpoints[1]:
        parts.append(values[:, -1:])
    if not parts:
        return values[:, :0]
    return torch.cat(parts, dim=1)


def _combine_boundary_and_free_values(
    free_values: torch.Tensor,
    boundary_values: torch.Tensor,
    pinned_endpoints: PinnedEndpointMask,
) -> torch.Tensor:
    if pinned_endpoints == TWO_PINNED_ENDPOINTS:
        return torch.cat(
            (boundary_values[:, :1], free_values, boundary_values[:, 1:]),
            dim=1,
        )
    if pinned_endpoints == START_PINNED_FREE_END:
        return torch.cat((boundary_values[:, :1], free_values), dim=1)
    if pinned_endpoints == (False, True):
        return torch.cat((free_values, boundary_values[:, :1]), dim=1)
    return free_values


def _replace_pinned_values(
    values: torch.Tensor,
    boundary_values: torch.Tensor,
    pinned_endpoints: PinnedEndpointMask,
) -> torch.Tensor:
    free_start = int(pinned_endpoints[0])
    free_stop = values.shape[1] - int(pinned_endpoints[1])
    return _combine_boundary_and_free_values(
        values[:, free_start:free_stop],
        boundary_values,
        pinned_endpoints,
    )


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


def _terminal_curvature_binormals_impl(
    positions: torch.Tensor,
    endpoint_orientations: torch.Tensor,
    *,
    validate: bool,
) -> torch.Tensor:
    """DER curvature against two prescribed holder tangent ghost edges."""

    if endpoint_orientations.shape != (positions.shape[0], 2, 3, 3):
        raise ValueError("endpoint_orientations must have shape Bx2x3x3.")
    edges = positions[:, 1:] - positions[:, :-1]
    tangents = (
        _normalize(edges, name="rod edges")
        if validate
        else _normalize_unchecked(edges)
    )
    holder = (
        _normalize(
            endpoint_orientations[..., :, 0],
            name="holder local +X tangent",
        )
        if validate
        else _normalize_unchecked(endpoint_orientations[..., :, 0])
    )
    previous = torch.stack((holder[:, 0], tangents[:, -1]), dim=1)
    following = torch.stack((tangents[:, 0], holder[:, 1]), dim=1)
    denominator = 1.0 + torch.sum(previous * following, dim=-1)
    epsilon = 128.0 * torch.finfo(positions.dtype).eps
    if validate and bool(torch.any(denominator <= epsilon).detach().cpu()):
        raise ValueError("Endpoint DER curvature is singular at a 180-degree fold.")
    return 2.0 * torch.linalg.cross(previous, following, dim=-1) / torch.clamp(
        denominator[..., None],
        min=epsilon,
    )


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


def _terminal_curvature_binormal_rates_impl(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    endpoint_orientations: torch.Tensor,
    endpoint_orientation_rates: torch.Tensor,
    *,
    validate: bool,
) -> torch.Tensor:
    """Rate of holder-to-edge curvature at both terminal ghost edges."""

    if endpoint_orientations.shape != (positions.shape[0], 2, 3, 3):
        raise ValueError("endpoint_orientations must have shape Bx2x3x3.")
    if endpoint_orientation_rates.shape != endpoint_orientations.shape:
        raise ValueError("endpoint_orientation_rates must match endpoint_orientations.")
    epsilon = 64.0 * torch.finfo(positions.dtype).eps
    edges = positions[:, 1:] - positions[:, :-1]
    edge_velocities = velocities[:, 1:] - velocities[:, :-1]
    lengths = torch.linalg.vector_norm(edges, dim=-1, keepdim=True)
    if validate and bool(torch.any(lengths <= epsilon).detach().cpu()):
        raise ValueError("rod edges contains a zero-length vector.")
    lengths = torch.clamp(lengths, min=epsilon)
    tangents = edges / lengths
    tangent_rates = (
        edge_velocities
        - tangents * torch.sum(tangents * edge_velocities, dim=-1, keepdim=True)
    ) / lengths

    holder_raw = endpoint_orientations[..., :, 0]
    holder_raw_rate = endpoint_orientation_rates[..., :, 0]
    holder_length = torch.linalg.vector_norm(holder_raw, dim=-1, keepdim=True)
    if validate and bool(torch.any(holder_length <= epsilon).detach().cpu()):
        raise ValueError("holder local +X tangent has zero length.")
    holder_length = torch.clamp(holder_length, min=epsilon)
    holder = holder_raw / holder_length
    holder_rate = (
        holder_raw_rate
        - holder * torch.sum(holder * holder_raw_rate, dim=-1, keepdim=True)
    ) / holder_length

    previous = torch.stack((holder[:, 0], tangents[:, -1]), dim=1)
    following = torch.stack((tangents[:, 0], holder[:, 1]), dim=1)
    previous_rate = torch.stack((holder_rate[:, 0], tangent_rates[:, -1]), dim=1)
    following_rate = torch.stack((tangent_rates[:, 0], holder_rate[:, 1]), dim=1)
    cross = torch.linalg.cross(previous, following, dim=-1)
    cross_rate = torch.linalg.cross(previous_rate, following, dim=-1)
    cross_rate = cross_rate + torch.linalg.cross(previous, following_rate, dim=-1)
    denominator = 1.0 + torch.sum(previous * following, dim=-1)
    denominator_rate = torch.sum(previous_rate * following, dim=-1)
    denominator_rate = denominator_rate + torch.sum(
        previous * following_rate, dim=-1
    )
    if validate and bool(torch.any(denominator <= 128.0 * epsilon).detach().cpu()):
        raise ValueError("Endpoint DER curvature is singular at a 180-degree fold.")
    denominator = torch.clamp(denominator, min=128.0 * epsilon)
    return 2.0 * (
        cross_rate / denominator[..., None]
        - cross * denominator_rate[..., None] / denominator.square()[..., None]
    )


def _skew_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Return matrices whose product with ``x`` is ``vector cross x``."""

    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (zero, -z, y, z, zero, -x, -y, x, zero),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def _curvature_rate_jacobian_impl(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    endpoint_orientations: torch.Tensor | None,
    endpoint_orientation_rates: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``J, r, w`` for ``0.5 * Cb * ||J v + r||_w^2``."""

    batch, node_count, _ = positions.shape
    edges = positions[:, 1:] - positions[:, :-1]
    epsilon = 64.0 * torch.finfo(positions.dtype).eps
    lengths = torch.clamp(torch.linalg.vector_norm(edges, dim=-1), min=epsilon)
    tangents = edges / lengths[..., None]
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device)
    tangent_rate_blocks = (
        identity[None, None]
        - tangents[..., :, None] * tangents[..., None, :]
    ) / lengths[..., None, None]

    def curvature_derivatives(
        previous: torch.Tensor,
        following: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cross = torch.linalg.cross(previous, following, dim=-1)
        denominator = torch.clamp(
            1.0 + torch.sum(previous * following, dim=-1),
            min=128.0 * torch.finfo(positions.dtype).eps,
        )
        derivative_previous = 2.0 * (
            -_skew_matrix(following) / denominator[..., None, None]
            - cross[..., :, None]
            * following[..., None, :]
            / denominator.square()[..., None, None]
        )
        derivative_following = 2.0 * (
            _skew_matrix(previous) / denominator[..., None, None]
            - cross[..., :, None]
            * previous[..., None, :]
            / denominator.square()[..., None, None]
        )
        return derivative_previous, derivative_following

    def objective_curvature_derivatives(
        previous: torch.Tensor,
        following: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Derivative of curvature in the minimally rotating local frame.

        The ordinary world-frame derivative of the curvature binormal is not
        objective: a rigidly rotating curved rod would dissipate energy.  The
        two adjacent tangents determine the local angular velocity through
        ``t_dot = omega cross t``.  Removing that rigid spin gives the
        corotational Kelvin--Voigt bending strain rate.
        """

        derivative_previous, derivative_following = curvature_derivatives(
            previous, following
        )
        curvature = 2.0 * torch.linalg.cross(previous, following, dim=-1) / torch.clamp(
            1.0 + torch.sum(previous * following, dim=-1),
            min=128.0 * torch.finfo(positions.dtype).eps,
        )[..., None]
        identity_batch = identity[None].expand(previous.shape[0], -1, -1)
        cosine = torch.sum(previous * following, dim=-1)
        tangent_sum = previous + following
        tangent_difference = previous - following
        normal = torch.linalg.cross(previous, following, dim=-1)

        def projector(vector: torch.Tensor) -> torch.Tensor:
            norm_squared = torch.sum(vector.square(), dim=-1)
            return (
                vector[..., :, None] * vector[..., None, :]
                / torch.clamp(norm_squared[..., None, None], min=epsilon)
            )

        # Eigenvectors of 2I-aa^T-bb^T are a+b, a-b, and a cross b,
        # with eigenvalues 1-c, 1+c, and 2.  This closed form is the same
        # Moore-Penrose inverse as the former batched SVD, but avoids one tiny
        # decomposition per curvature site and time step.  At a nearly
        # straight joint, rotation about the common tangent is unobservable;
        # the exact limiting pseudoinverse is 0.5(I-aa^T).
        general_inverse = (
            projector(tangent_sum)
            / torch.clamp((1.0 - cosine)[..., None, None], min=epsilon)
            + projector(tangent_difference)
            / torch.clamp((1.0 + cosine)[..., None, None], min=epsilon)
            + 0.5 * projector(normal)
        )
        straight_inverse = 0.5 * (
            identity_batch
            - previous[..., :, None] * previous[..., None, :]
        )
        angular_inverse = torch.where(
            ((1.0 - cosine) > 1.0e-7)[..., None, None],
            general_inverse,
            straight_inverse,
        )
        spin_previous = angular_inverse @ _skew_matrix(previous)
        spin_following = angular_inverse @ _skew_matrix(following)
        corotation = _skew_matrix(curvature)
        return (
            derivative_previous + corotation @ spin_previous,
            derivative_following + corotation @ spin_following,
        )

    zero_block = torch.zeros(
        (batch, 3, 3), dtype=positions.dtype, device=positions.device
    )
    rows: list[torch.Tensor] = []
    affine_rows: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for interior in range(node_count - 2):
        derivative_previous, derivative_following = objective_curvature_derivatives(
            tangents[:, interior], tangents[:, interior + 1]
        )
        previous_block = derivative_previous @ tangent_rate_blocks[:, interior]
        following_block = derivative_following @ tangent_rate_blocks[:, interior + 1]
        blocks = [zero_block] * node_count
        blocks[interior] = -previous_block
        blocks[interior + 1] = previous_block - following_block
        blocks[interior + 2] = following_block
        rows.append(torch.cat(blocks, dim=-1))
        affine_rows.append(torch.zeros_like(positions[:, 0]))
        weights.append(
            torch.ones_like(lengths[:, interior])
            * 2.0
            / (rest_lengths[interior] + rest_lengths[interior + 1])
        )

    if endpoint_orientations is not None:
        if endpoint_orientation_rates is None:
            raise ValueError(
                "Endpoint orientation rates are required for terminal damping."
            )
        holder_raw = endpoint_orientations[..., :, 0]
        holder_raw_rate = endpoint_orientation_rates[..., :, 0]
        holder_length = torch.clamp(
            torch.linalg.vector_norm(holder_raw, dim=-1, keepdim=True), min=epsilon
        )
        holder = holder_raw / holder_length
        holder_rate = (
            holder_raw_rate
            - holder * torch.sum(holder * holder_raw_rate, dim=-1, keepdim=True)
        ) / holder_length

        derivative_holder, derivative_edge = objective_curvature_derivatives(
            holder[:, 0], tangents[:, 0]
        )
        edge_block = derivative_edge @ tangent_rate_blocks[:, 0]
        blocks = [zero_block] * node_count
        blocks[0], blocks[1] = -edge_block, edge_block
        rows.append(torch.cat(blocks, dim=-1))
        affine_rows.append((derivative_holder @ holder_rate[:, 0, :, None])[..., 0])
        weights.append(
            torch.ones_like(lengths[:, 0]) / rest_lengths[0]
        )

        derivative_edge, derivative_holder = objective_curvature_derivatives(
            tangents[:, -1], holder[:, 1]
        )
        edge_block = derivative_edge @ tangent_rate_blocks[:, -1]
        blocks = [zero_block] * node_count
        blocks[-2], blocks[-1] = -edge_block, edge_block
        rows.append(torch.cat(blocks, dim=-1))
        affine_rows.append((derivative_holder @ holder_rate[:, 1, :, None])[..., 0])
        weights.append(
            torch.ones_like(lengths[:, -1]) / rest_lengths[-1]
        )

    jacobian = torch.cat(rows, dim=1)
    affine = torch.stack(affine_rows, dim=1).reshape(batch, -1)
    weight = torch.stack(weights, dim=1).repeat_interleave(3, dim=1)
    return jacobian, affine, weight


def _implicit_bending_damping_velocity(
    positions: torch.Tensor,
    undamped_velocity: torch.Tensor,
    boundary_velocity: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    substep_dt: torch.Tensor,
    damping: torch.Tensor,
    endpoint_orientations: torch.Tensor | None,
    endpoint_orientation_rates: torch.Tensor | None,
    *,
    conjugate_gradient_iterations: int | None = None,
    damping_backend: str = "pcg60_reference",
    pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
) -> torch.Tensor:
    """Backward-Euler solve for exact Kelvin--Voigt curvature damping.

    Parameter fitting uses a direct Cholesky solve. The fixed-shape online
    runtime may use conjugate gradient for the same SPD system because MAGMA's
    batched Cholesky solve cannot be captured by a CUDA graph. With one
    iteration per free scalar degree of freedom, CG is exact in exact
    arithmetic and closely matches the direct solve in float32.
    """

    pinned_endpoints = _validate_pinned_endpoints(pinned_endpoints)
    from .cuda_fixed_pcg import (
        REFERENCE_DAMPING_BACKEND,
        validate_damping_backend,
    )

    selected_backend = validate_damping_backend(damping_backend)
    batch, node_count, _ = positions.shape
    boundary_count = int(pinned_endpoints[0]) + int(pinned_endpoints[1])
    if boundary_velocity.shape != (batch, boundary_count, 3):
        raise ValueError(
            f"boundary_velocity must have shape Bx{boundary_count}x3 for "
            "the selected pinned endpoints."
        )
    fixed_runtime_damping = (
        positions.is_cuda
        and positions.dtype == torch.float32
        and positions.shape[1] in (11, 21, 31)
        and pinned_endpoints == START_PINNED_FREE_END
        and endpoint_orientations is None
        and endpoint_orientation_rates is None
        and conjugate_gradient_iterations == 6 * (positions.shape[1] - 1)
        and (
            selected_backend == REFERENCE_DAMPING_BACKEND
            or positions.shape[1] == 11
        )
        and os.environ.get("CABLE_TWIN_FUSED_FIXED_DAMPING", "1") != "0"
    )
    if fixed_runtime_damping:
        from .cuda_fixed_pcg import fixed_damping_supported_nodes

        return fixed_damping_supported_nodes(
            positions,
            undamped_velocity,
            boundary_velocity,
            rest_lengths,
            masses,
            substep_dt,
            damping,
            backend=selected_backend,
        )
    if selected_backend != REFERENCE_DAMPING_BACKEND:
        if selected_backend == "pcg32_experimental":
            conjugate_gradient_iterations = 32
        elif selected_backend == "pcg24_experimental":
            conjugate_gradient_iterations = 24
        elif selected_backend == "pcg16_experimental":
            conjugate_gradient_iterations = 16
        elif selected_backend == "block_banded_direct_experimental":
            conjugate_gradient_iterations = None
    jacobian, affine, weight = _curvature_rate_jacobian_impl(
        positions, rest_lengths, endpoint_orientations, endpoint_orientation_rates
    )
    free_start = int(pinned_endpoints[0])
    free_stop = node_count - int(pinned_endpoints[1])
    free_jacobian = jacobian[..., 3 * free_start : 3 * free_stop]
    free_zeros = torch.zeros_like(undamped_velocity[:, free_start:free_stop])
    boundary_only = _combine_boundary_and_free_values(
        free_zeros,
        boundary_velocity,
        pinned_endpoints,
    )
    prescribed_rate = (
        jacobian @ boundary_only.reshape(batch, node_count * 3, 1)
    )[..., 0] + affine
    unit_damping = free_jacobian.transpose(-1, -2) @ (
        weight[..., None] * free_jacobian
    )
    free_mass = masses[free_start:free_stop].repeat_interleave(3)
    step = substep_dt.reshape(batch, 1)
    coefficient = step * damping[:, None]
    system = torch.diag_embed(free_mass[None].expand(batch, -1))
    system = system + coefficient[..., None] * unit_damping
    damping_from_prescribed = -damping[:, None] * (
        free_jacobian.transpose(-1, -2)
        @ (weight * prescribed_rate)[..., None]
    )[..., 0]
    right_hand_side = (
        free_mass[None]
        * undamped_velocity[:, free_start:free_stop].reshape(batch, -1)
        + step * damping_from_prescribed
    )
    if conjugate_gradient_iterations is None:
        factor, _info = torch.linalg.cholesky_ex(system)
        solution = torch.cholesky_solve(
            right_hand_side[..., None], factor
        )[..., 0]
    else:
        if conjugate_gradient_iterations < 1:
            raise ValueError("Conjugate-gradient iteration count must be positive.")
        # The online state is float32, but these small ill-conditioned systems
        # need float64 Krylov arithmetic to match the direct SPD solve closely.
        iterative_system = system.to(dtype=torch.float64)
        iterative_rhs = right_hand_side.to(dtype=torch.float64)
        fixed_runtime_pcg = (
            iterative_system.is_cuda
            and iterative_system.shape[1:] == (30, 30)
            and conjugate_gradient_iterations == 60
            and pinned_endpoints == START_PINNED_FREE_END
            and os.environ.get("CABLE_TWIN_FUSED_FIXED_PCG", "1") != "0"
        )
        if fixed_runtime_pcg:
            # Exact same assembled A/rhs and fixed float64 PCG recurrence, but
            # one cooperative CUDA kernel retains all vectors for 60 steps.
            from .cuda_fixed_pcg import fixed_pcg_30x60

            solution = fixed_pcg_30x60(
                iterative_system.contiguous(), iterative_rhs.contiguous()
            )
        else:
            solution = torch.zeros_like(iterative_rhs)
            residual = iterative_rhs.clone()
            diagonal = torch.diagonal(iterative_system, dim1=-2, dim2=-1)
            preconditioned = residual / diagonal
            direction = preconditioned.clone()
            residual_product = torch.sum(
                residual * preconditioned, dim=-1, keepdim=True
            )
            tiny = torch.finfo(iterative_system.dtype).tiny
            for _ in range(conjugate_gradient_iterations):
                applied = (iterative_system @ direction[..., None])[..., 0]
                denominator = torch.sum(
                    direction * applied, dim=-1, keepdim=True
                )
                alpha = torch.where(
                    torch.abs(denominator) > tiny,
                    residual_product / denominator,
                    torch.zeros_like(denominator),
                )
                solution = solution + alpha * direction
                residual = residual - alpha * applied
                preconditioned = residual / diagonal
                next_product = torch.sum(
                    residual * preconditioned, dim=-1, keepdim=True
                )
                beta = torch.where(
                    torch.abs(residual_product) > tiny,
                    next_product / residual_product,
                    torch.zeros_like(residual_product),
                )
                direction = preconditioned + beta * direction
                residual_product = next_product
        solution = solution.to(dtype=right_hand_side.dtype)
    free_velocity = solution.reshape(batch, free_stop - free_start, 3)
    return _combine_boundary_and_free_values(
        free_velocity,
        boundary_velocity,
        pinned_endpoints,
    )


def _project_director(
    director: torch.Tensor,
    tangent: torch.Tensor,
    *,
    validate: bool,
) -> torch.Tensor:
    projected = director - tangent * torch.sum(director * tangent, dim=-1, keepdim=True)
    return (
        _normalize(projected, name="endpoint material director")
        if validate
        else _normalize_unchecked(projected)
    )


def _parallel_transport_director(
    director: torch.Tensor,
    previous_tangent: torch.Tensor,
    next_tangent: torch.Tensor,
    *,
    validate: bool,
) -> torch.Tensor:
    """Minimal-rotation (Bishop) transport between adjacent DER edges."""

    cross = torch.linalg.cross(previous_tangent, next_tangent, dim=-1)
    cosine = torch.sum(previous_tangent * next_tangent, dim=-1, keepdim=True)
    denominator = 1.0 + cosine
    epsilon = 128.0 * torch.finfo(director.dtype).eps
    if validate and bool(torch.any(denominator <= epsilon).detach().cpu()):
        raise ValueError("Bishop transport is singular at an exact 180-degree fold.")
    transported = director + torch.linalg.cross(cross, director, dim=-1)
    transported = transported + torch.linalg.cross(
        cross,
        torch.linalg.cross(cross, director, dim=-1),
        dim=-1,
    ) / torch.clamp(denominator, min=epsilon)
    return _project_director(transported, next_tangent, validate=validate)


def _endpoint_twist_angle_impl(
    positions: torch.Tensor,
    endpoint_orientations: torch.Tensor,
    *,
    validate: bool,
) -> torch.Tensor:
    """Principal terminal material twist after Bishop transport."""

    if endpoint_orientations.shape != (positions.shape[0], 2, 3, 3):
        raise ValueError("endpoint_orientations must have shape Bx2x3x3.")
    edges = positions[:, 1:] - positions[:, :-1]
    tangents = (
        _normalize(edges, name="rod edges")
        if validate
        else _normalize_unchecked(edges)
    )
    holder_tangents = (
        _normalize(
            endpoint_orientations[..., :, 0],
            name="holder local +X tangent",
        )
        if validate
        else _normalize_unchecked(endpoint_orientations[..., :, 0])
    )
    # Motive rigid-body local +X is the cable tangent in the relaxed fixture;
    # local +Y is the material director.  Transport through both terminal
    # holder-to-rod ghost bends; direct projection is not the DER minimal
    # rotation when a terminal rod edge differs from its holder tangent.
    director = _project_director(
        endpoint_orientations[:, 0, :, 1],
        holder_tangents[:, 0],
        validate=validate,
    )
    director = _parallel_transport_director(
        director,
        holder_tangents[:, 0],
        tangents[:, 0],
        validate=validate,
    )
    for edge in range(1, tangents.shape[1]):
        director = _parallel_transport_director(
            director,
            tangents[:, edge - 1],
            tangents[:, edge],
            validate=validate,
        )
    terminal = _project_director(
        endpoint_orientations[:, 1, :, 1],
        holder_tangents[:, 1],
        validate=validate,
    )
    terminal = _parallel_transport_director(
        terminal,
        holder_tangents[:, 1],
        tangents[:, -1],
        validate=validate,
    )
    sine = torch.sum(
        tangents[:, -1]
        * torch.linalg.cross(director, terminal, dim=-1),
        dim=-1,
    )
    cosine = torch.sum(director * terminal, dim=-1)
    return torch.atan2(sine, cosine)


def endpoint_twist_angle(
    positions: torch.Tensor,
    endpoint_orientations: torch.Tensor,
) -> torch.Tensor:
    """Return the principal homogeneous DER twist imposed by two holders."""

    if positions.ndim != 3 or positions.shape[1] < 4 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3 with N >= 4.")
    return _endpoint_twist_angle_impl(
        positions,
        endpoint_orientations,
        validate=True,
    )


def unwrap_angles(
    principal_angles: torch.Tensor,
    *,
    dim: int = 0,
) -> torch.Tensor:
    """Unwrap a sampled angle sequence while retaining its tensor device/dtype.

    Consecutive samples must differ by less than pi.  This is the same
    observability requirement as any sampled rotation measurement: turns that
    occur inside an unobserved interval cannot be reconstructed afterward.
    """

    if principal_angles.ndim == 0:
        raise ValueError("principal_angles must contain a sampled dimension.")
    if not -principal_angles.ndim <= dim < principal_angles.ndim:
        raise ValueError("unwrap dimension is outside principal_angles.")
    moved = torch.movedim(principal_angles, dim, 0)
    if moved.shape[0] < 2:
        return principal_angles.clone()
    difference = moved[1:] - moved[:-1]
    two_pi = 2.0 * math.pi
    wrapped = torch.remainder(difference + math.pi, two_pi) - math.pi
    # Match the conventional unwrap choice at the exactly ambiguous pi case.
    wrapped = torch.where(
        (wrapped == -math.pi) & (difference > 0.0),
        torch.full_like(wrapped, math.pi),
        wrapped,
    )
    unwrapped = torch.cat(
        (moved[:1], moved[:1] + torch.cumsum(wrapped, dim=0)),
        dim=0,
    )
    return torch.movedim(unwrapped, 0, dim)


def continuous_endpoint_twist_angles(
    positions: torch.Tensor,
    endpoint_orientations: torch.Tensor,
) -> torch.Tensor:
    """Return the continuous terminal twist of one time-ordered trajectory.

    The leading dimension is time.  The sequence must begin on the intended
    calibrated twist branch and remain observed continuously.
    """

    principal = endpoint_twist_angle(positions, endpoint_orientations)
    return unwrap_angles(principal, dim=0)


def _angle_nearest_reference(
    principal_angle: torch.Tensor,
    reference_angle: torch.Tensor | None,
) -> torch.Tensor:
    """Choose the 2*pi branch nearest a previously observed continuous angle."""

    if reference_angle is None:
        return principal_angle
    if reference_angle.shape != principal_angle.shape:
        raise ValueError("twist reference must have shape B.")
    turns = torch.round(
        (reference_angle.detach() - principal_angle.detach()) / (2.0 * math.pi)
    )
    return principal_angle + (2.0 * math.pi) * turns


def _relative_rotation_generator(
    start: torch.Tensor,
    finish: torch.Tensor,
) -> torch.Tensor:
    """Return the principal SO(3) logarithm of ``start.T @ finish``."""

    relative = start.transpose(-1, -2) @ finish
    skew = 0.5 * (relative - relative.transpose(-1, -2))
    vee = torch.stack((skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]), dim=-1)
    sine = torch.linalg.vector_norm(vee, dim=-1)
    cosine = torch.clamp(
        0.5 * (torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0),
        min=-1.0,
        max=1.0,
    )
    angle = torch.atan2(sine, cosine)
    epsilon = 64.0 * torch.finfo(start.dtype).eps
    scale = torch.where(
        sine > epsilon,
        angle / torch.clamp(sine, min=epsilon),
        1.0 + angle.square() / 6.0,
    )
    return scale[..., None, None] * skew


def _interpolate_rotations(
    start: torch.Tensor,
    finish: torch.Tensor,
    fraction: float,
    dt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant-angular-velocity interpolation on SO(3)."""

    generator = _relative_rotation_generator(start, finish)
    # Closed-form Rodrigues exponential avoids a CPU workspace allocation made
    # by torch.matrix_exp during CUDA graph capture on current PyTorch builds.
    angle = torch.sqrt(
        torch.clamp(
            0.5 * torch.sum(generator.square(), dim=(-2, -1)),
            min=0.0,
        )
    )
    scaled_angle = float(fraction) * angle
    epsilon = 64.0 * torch.finfo(start.dtype).eps
    safe_angle = torch.clamp(angle, min=epsilon)
    first = torch.sin(scaled_angle) / safe_angle
    second = (1.0 - torch.cos(scaled_angle)) / safe_angle.square()
    first = torch.where(
        angle > epsilon,
        first,
        float(fraction) - (float(fraction) ** 3) * angle.square() / 6.0,
    )
    second = torch.where(
        angle > epsilon,
        second,
        0.5 * (float(fraction) ** 2)
        - (float(fraction) ** 4) * angle.square() / 24.0,
    )
    identity = torch.eye(3, dtype=start.dtype, device=start.device).expand_as(
        generator
    )
    increment = (
        identity
        + first[..., None, None] * generator
        + second[..., None, None] * (generator @ generator)
    )
    orientation = start @ increment
    orientation_rate = orientation @ generator / dt[:, None, None, None]
    return orientation, orientation_rate


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


def solve_symmetric_tridiagonal_dense(
    diagonal: torch.Tensor,
    off_diagonal: torch.Tensor,
    right_hand_side: torch.Tensor,
) -> torch.Tensor:
    """Solve the same batched system with a dense CUDA-parallel factorization.

    The rod systems are only one equation per edge (ten equations for the
    current OptiTrack cable).  During offline differentiation, the additional
    storage is small and ``torch.linalg.solve`` exposes substantially more
    parallel work to CUDA than the sequential Thomas recurrence.  Runtime
    filtering keeps the linear-memory tridiagonal solver above.
    """

    if diagonal.ndim != 2 or right_hand_side.shape != diagonal.shape:
        raise ValueError("Diagonal and right-hand side must have shape BxN.")
    if off_diagonal.shape != (diagonal.shape[0], diagonal.shape[1] - 1):
        raise ValueError("Off diagonal must have shape Bx(N-1).")
    system = torch.diag_embed(diagonal)
    system = system + torch.diag_embed(off_diagonal, offset=1)
    system = system + torch.diag_embed(off_diagonal, offset=-1)
    return torch.linalg.solve(system, right_hand_side.unsqueeze(-1)).squeeze(-1)


def momentum_project_lengths(
    positions: torch.Tensor,
    rest_lengths: torch.Tensor,
    masses: torch.Tensor,
    boundary_positions: torch.Tensor | None,
    *,
    iterations: int,
    dense_solve: bool = False,
    pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
) -> torch.Tensor:
    """Coupled inverse-mass edge projection with optional pinned endpoints."""

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3.")
    batch, node_count, _ = positions.shape
    if rest_lengths.shape != (node_count - 1,):
        raise ValueError("rest_lengths must contain one value per edge.")
    if masses.shape != (node_count,):
        raise ValueError("masses must contain one value per vertex.")
    pinned_endpoints = _validate_pinned_endpoints(pinned_endpoints)
    effective_pins = (
        (False, False) if boundary_positions is None else pinned_endpoints
    )
    boundary_count = int(effective_pins[0]) + int(effective_pins[1])
    if boundary_positions is not None and boundary_positions.shape != (
        batch,
        boundary_count,
        3,
    ):
        raise ValueError(
            f"boundary_positions must have shape Bx{boundary_count}x3 for "
            "the selected pinned endpoints."
        )
    if iterations < 1:
        raise ValueError("Projection iterations must be positive.")

    inverse_mass = torch.reciprocal(masses).clone()
    if effective_pins[0]:
        inverse_mass = torch.cat(
            (torch.zeros_like(inverse_mass[:1]), inverse_mass[1:])
        )
    if effective_pins[1]:
        inverse_mass = torch.cat(
            (inverse_mass[:-1], torch.zeros_like(inverse_mass[-1:]))
        )

    def clamp_boundary(value: torch.Tensor) -> torch.Tensor:
        if boundary_positions is None:
            return value
        return _replace_pinned_values(value, boundary_positions, effective_pins)

    solve = (
        solve_symmetric_tridiagonal_dense
        if dense_solve
        else solve_symmetric_tridiagonal
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
        multiplier = solve(
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
    dense_solve: bool = False,
    pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
) -> torch.Tensor:
    """Mass-weighted projection onto the inextensible velocity constraints."""

    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("positions must have shape BxNx3.")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions.")
    batch, node_count, _ = positions.shape
    if masses.shape != (node_count,):
        raise ValueError("masses must contain one value per vertex.")
    pinned_endpoints = _validate_pinned_endpoints(pinned_endpoints)
    boundary_count = int(pinned_endpoints[0]) + int(pinned_endpoints[1])
    if boundary_velocities.shape != (batch, boundary_count, 3):
        raise ValueError(
            f"boundary_velocities must have shape Bx{boundary_count}x3 for "
            "the selected pinned endpoints."
        )

    inverse_mass = torch.reciprocal(masses).clone()
    if pinned_endpoints[0]:
        inverse_mass = torch.cat(
            (torch.zeros_like(inverse_mass[:1]), inverse_mass[1:])
        )
    if pinned_endpoints[1]:
        inverse_mass = torch.cat(
            (inverse_mass[:-1], torch.zeros_like(inverse_mass[-1:]))
        )
    value = _replace_pinned_values(
        velocities,
        boundary_velocities,
        pinned_endpoints,
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
    solve = (
        solve_symmetric_tridiagonal_dense
        if dense_solve
        else solve_symmetric_tridiagonal
    )
    multiplier = solve(
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
    return _replace_pinned_values(
        value + node_correction,
        boundary_velocities,
        pinned_endpoints,
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
    torsional_stiffness_n_m2: float = 0.0
    external_drag_s_inv: float = 0.0
    rest_lengths_m: tuple[float, ...] | None = None
    vertex_masses_kg: tuple[float, ...] | None = None
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
        if (
            not math.isfinite(self.torsional_stiffness_n_m2)
            or self.torsional_stiffness_n_m2 < 0.0
        ):
            raise ValueError("DDER torsional stiffness must be finite and non-negative.")
        if not math.isfinite(self.external_drag_s_inv) or self.external_drag_s_inv < 0.0:
            raise ValueError("DDER external drag must be finite and non-negative.")
        if len(self.gravity_camera_m_s2) != 3 or not all(
            math.isfinite(value) for value in self.gravity_camera_m_s2
        ):
            raise ValueError("Gravity must contain three finite camera-frame values.")
        if min(self.substeps, self.constraint_iterations) < 1:
            raise ValueError("DDER solver counts must be positive.")
        if self.rest_lengths_m is not None:
            rest_lengths = tuple(float(value) for value in self.rest_lengths_m)
            if len(rest_lengths) != self.node_count - 1:
                raise ValueError("rest_lengths_m must contain one value per edge.")
            if any(not math.isfinite(value) or value <= 0.0 for value in rest_lengths):
                raise ValueError("Every DDER rest length must be finite and positive.")
            if not math.isclose(
                sum(rest_lengths),
                self.cable_length_m,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                raise ValueError("DDER rest lengths must sum to cable_length_m.")
            object.__setattr__(self, "rest_lengths_m", rest_lengths)
        if self.vertex_masses_kg is not None:
            vertex_masses = tuple(float(value) for value in self.vertex_masses_kg)
            if len(vertex_masses) != self.node_count:
                raise ValueError("vertex_masses_kg must contain one value per vertex.")
            if any(not math.isfinite(value) or value <= 0.0 for value in vertex_masses):
                raise ValueError("Every DDER vertex mass must be finite and positive.")
            if not math.isclose(
                sum(vertex_masses),
                self.cable_mass_kg,
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            ):
                raise ValueError("DDER vertex masses must sum to cable_mass_kg.")
            object.__setattr__(self, "vertex_masses_kg", vertex_masses)

    @property
    def segment_length_m(self) -> float:
        return self.cable_length_m / (self.node_count - 1)


@dataclass(frozen=True, slots=True)
class DderState:
    positions_m: torch.Tensor
    velocities_m_s: torch.Tensor
    endpoint_orientations: torch.Tensor | None = None
    endpoint_twist_rad: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class DderRuntimeConstants:
    rest_lengths_m: torch.Tensor
    masses_kg: torch.Tensor
    dual_lengths_m: torch.Tensor
    gravity_m_s2: torch.Tensor
    bending_stiffness_n_m2: torch.Tensor
    bending_damping_n_m2_s: torch.Tensor
    torsional_stiffness_n_m2: torch.Tensor
    external_drag_s_inv: torch.Tensor


class DderModel:
    """Batched CUDA transition for a straight, isotropic, inextensible cable."""

    def __init__(self, parameters: DderParameters) -> None:
        self.parameters = parameters
        self._rest_lengths = torch.as_tensor(
            parameters.rest_lengths_m
            if parameters.rest_lengths_m is not None
            else (parameters.segment_length_m,) * (parameters.node_count - 1),
            dtype=torch.float64,
        )
        if parameters.vertex_masses_kg is None:
            masses = torch.empty(parameters.node_count, dtype=torch.float64)
            linear_density = parameters.cable_mass_kg / parameters.cable_length_m
            masses[0] = 0.5 * self._rest_lengths[0]
            masses[-1] = 0.5 * self._rest_lengths[-1]
            masses[1:-1] = 0.5 * (
                self._rest_lengths[:-1] + self._rest_lengths[1:]
            )
            self._vertex_masses = masses * linear_density
        else:
            self._vertex_masses = torch.as_tensor(
                parameters.vertex_masses_kg,
                dtype=torch.float64,
            )
        # Largest generalized eigenvalue of the straight-rod, unit-EI
        # small-deflection bending operator with fixed endpoint translations.
        # The runtime pins only terminal translations and leaves endpoint
        # rotation free, matching this stability operator.
        node_count = parameters.node_count
        tangent_difference = torch.zeros(
            (node_count - 2, node_count),
            dtype=torch.float64,
        )
        rows = torch.arange(node_count - 2)
        tangent_difference[rows, rows] = 1.0 / self._rest_lengths[:-1]
        tangent_difference[rows, rows + 1] = -(
            1.0 / self._rest_lengths[:-1] + 1.0 / self._rest_lengths[1:]
        )
        tangent_difference[rows, rows + 2] = 1.0 / self._rest_lengths[1:]
        dual_lengths = 0.5 * (self._rest_lengths[:-1] + self._rest_lengths[1:])
        interior_unit_stiffness = (
            tangent_difference.T
            @ (tangent_difference / dual_lengths[:, None])
        )
        # Conservative addition for the two prescribed holder-tangent ghost
        # edges used by the twist-aware offline model.
        terminal_difference = torch.zeros((2, node_count), dtype=torch.float64)
        terminal_difference[0, :2] = torch.tensor(
            (-1.0, 1.0), dtype=torch.float64
        ) / self._rest_lengths[0]
        terminal_difference[1, -2:] = torch.tensor(
            (-1.0, 1.0), dtype=torch.float64
        ) / self._rest_lengths[-1]
        terminal_dual = torch.stack((self._rest_lengths[0], self._rest_lengths[-1]))
        two_holder_unit_stiffness = (
            interior_unit_stiffness
            + terminal_difference.T
            @ (terminal_difference / terminal_dual[:, None])
        )
        masses = self._vertex_masses

        def maximum_free_eigenvalue(
            unit_stiffness: torch.Tensor,
            pinned_endpoints: PinnedEndpointMask,
        ) -> float:
            free_start = int(pinned_endpoints[0])
            free_stop = node_count - int(pinned_endpoints[1])
            free_mass = masses[free_start:free_stop]
            free_stiffness = unit_stiffness[
                free_start:free_stop,
                free_start:free_stop,
            ]
            mass_scaled = free_stiffness / torch.sqrt(
                free_mass[:, None] * free_mass[None, :]
            )
            return float(torch.linalg.eigvalsh(mass_scaled).max())

        self._maximum_bending_eigenvalue_per_ei = maximum_free_eigenvalue(
            two_holder_unit_stiffness,
            TWO_PINNED_ENDPOINTS,
        )
        self._maximum_bending_eigenvalue_per_ei_one_attached = (
            maximum_free_eigenvalue(
                interior_unit_stiffness,
                START_PINNED_FREE_END,
            )
        )

    def maximum_stable_bending_stiffness(
        self,
        maximum_dt_s: float,
        *,
        pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
    ) -> float:
        """Conservative EI limit for the present explicit bending step."""

        pinned_endpoints = _validate_pinned_endpoints(
            pinned_endpoints,
            supported_only=True,
        )
        dt = float(maximum_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("maximum_dt_s must be finite and positive.")
        substep = dt / self.parameters.substeps
        maximum_eigenvalue = (
            self._maximum_bending_eigenvalue_per_ei
            if pinned_endpoints == TWO_PINNED_ENDPOINTS
            else self._maximum_bending_eigenvalue_per_ei_one_attached
        )
        return (EXPLICIT_STABILITY_LIMIT / substep) ** 2 / maximum_eigenvalue

    def maximum_stable_bending_damping(self, maximum_dt_s: float) -> float:
        """Return infinity because curvature damping is integrated implicitly."""

        dt = float(maximum_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("maximum_dt_s must be finite and positive.")
        return math.inf

    def _constants(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rest_lengths = self._rest_lengths.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        masses = self._vertex_masses.to(
            device=reference.device,
            dtype=reference.dtype,
        )
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
            torsional_stiffness_n_m2=torch.full(
                (reference.shape[0],),
                self.parameters.torsional_stiffness_n_m2,
                dtype=reference.dtype,
                device=reference.device,
            ),
            external_drag_s_inv=torch.full(
                (reference.shape[0],),
                self.parameters.external_drag_s_inv,
                dtype=reference.dtype,
                device=reference.device,
            ),
        )

    @staticmethod
    def _runtime_internal_force(
        positions: torch.Tensor,
        constants: DderRuntimeConstants,
        endpoint_orientations: torch.Tensor | None = None,
        endpoint_twist_reference_rad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.enable_grad():
            q = positions.detach().requires_grad_(True)
            curvature = _curvature_binormals_impl(q, validate=False)
            energy = 0.5 * constants.bending_stiffness_n_m2 * torch.sum(
                torch.sum(curvature.square(), dim=-1)
                / constants.dual_lengths_m[None],
                dim=1,
            )
            if endpoint_orientations is not None:
                terminal_curvature = _terminal_curvature_binormals_impl(
                    q,
                    endpoint_orientations.detach(),
                    validate=False,
                )
                terminal_dual = torch.stack(
                    (constants.rest_lengths_m[0], constants.rest_lengths_m[-1])
                )
                energy = energy + 0.5 * constants.bending_stiffness_n_m2 * torch.sum(
                    torch.sum(terminal_curvature.square(), dim=-1)
                    / terminal_dual[None],
                    dim=1,
                )
                twist = _endpoint_twist_angle_impl(
                    q,
                    endpoint_orientations.detach(),
                    validate=False,
                )
                twist = _angle_nearest_reference(
                    twist,
                    endpoint_twist_reference_rad,
                )
                energy = energy + 0.5 * constants.torsional_stiffness_n_m2 * (
                    twist.square() / torch.sum(constants.rest_lengths_m)
                )
            return -torch.autograd.grad(energy.sum(), q, create_graph=False)[0]

    def bending_energy(
        self,
        positions: torch.Tensor,
        bending_stiffness_n_m2: torch.Tensor | float | None = None,
        endpoint_orientations: torch.Tensor | None = None,
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
        energy = 0.5 * stiffness * torch.sum(
            torch.sum(curvature.square(), dim=-1) / dual_lengths[None],
            dim=1,
        )
        if endpoint_orientations is not None:
            terminal = _terminal_curvature_binormals_impl(
                positions,
                endpoint_orientations,
                validate=_validate,
            )
            terminal_dual = torch.stack((rest_lengths[0], rest_lengths[-1]))
            energy = energy + 0.5 * stiffness * torch.sum(
                torch.sum(terminal.square(), dim=-1) / terminal_dual[None],
                dim=1,
            )
        return energy

    def twist_energy(
        self,
        positions: torch.Tensor,
        endpoint_orientations: torch.Tensor,
        torsional_stiffness_n_m2: torch.Tensor | float | None = None,
        endpoint_twist_reference_rad: torch.Tensor | None = None,
        *,
        _validate: bool = True,
    ) -> torch.Tensor:
        """Minimum twist energy of a homogeneous isotropic DER.

        The internal twist angles are eliminated analytically.  For total
        imposed twist ``phi`` and length ``L``, the minimum is
        ``0.5 * GJ * phi**2 / L``.
        """

        stiffness = _as_batch_scalar(
            self.parameters.torsional_stiffness_n_m2
            if torsional_stiffness_n_m2 is None
            else torsional_stiffness_n_m2,
            positions,
            "torsional_stiffness_n_m2",
        )
        angle = _endpoint_twist_angle_impl(
            positions,
            endpoint_orientations,
            validate=_validate,
        )
        angle = _angle_nearest_reference(angle, endpoint_twist_reference_rad)
        return 0.5 * stiffness * angle.square() / self.parameters.cable_length_m

    def internal_force(
        self,
        positions: torch.Tensor,
        *,
        create_graph: bool,
        bending_stiffness_n_m2: torch.Tensor | float | None = None,
        endpoint_orientations: torch.Tensor | None = None,
        torsional_stiffness_n_m2: torch.Tensor | float | None = None,
        endpoint_twist_reference_rad: torch.Tensor | None = None,
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
                endpoint_orientations,
                _validate=_validate,
            )
            if endpoint_orientations is not None:
                energy = energy + self.twist_energy(
                    q,
                    endpoint_orientations,
                    torsional_stiffness_n_m2,
                    endpoint_twist_reference_rad,
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
        endpoint_orientations: torch.Tensor | None = None,
        endpoint_orientation_rates: torch.Tensor | None = None,
        _validate: bool = True,
    ) -> torch.Tensor:
        """Kelvin-Voigt bending force from the curvature-rate dissipation."""

        with torch.enable_grad():
            velocity = (
                velocities
                if velocities.requires_grad
                else velocities.detach().requires_grad_(True)
            )
            rest_lengths, _masses = self._constants(positions)
            if endpoint_orientations is not None and endpoint_orientation_rates is None:
                raise ValueError(
                    "Endpoint orientation rates are required for terminal damping."
                )
            jacobian, affine, weight = _curvature_rate_jacobian_impl(
                positions,
                rest_lengths,
                endpoint_orientations,
                endpoint_orientation_rates,
            )
            rate = (
                jacobian @ velocity.reshape(velocity.shape[0], -1, 1)
            )[..., 0] + affine
            damping = _as_batch_scalar(
                self.parameters.bending_damping_n_m2_s
                if bending_damping_n_m2_s is None
                else bending_damping_n_m2_s,
                positions,
                "bending_damping_n_m2_s",
            )
            dissipation = 0.5 * damping * torch.sum(weight * rate.square(), dim=1)
            return -torch.autograd.grad(
                dissipation.sum(),
                velocity,
                create_graph=create_graph,
            )[0]

    def initial_state(
        self,
        positions_m: torch.Tensor,
        velocities_m_s: torch.Tensor | None = None,
        endpoint_orientations: torch.Tensor | None = None,
        endpoint_twist_rad: torch.Tensor | None = None,
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
        if endpoint_orientations is not None and endpoint_orientations.shape != (
            positions_m.shape[0],
            2,
            3,
            3,
        ):
            raise ValueError("endpoint_orientations must have shape Bx2x3x3.")
        if endpoint_twist_rad is not None:
            if endpoint_twist_rad.shape != (positions_m.shape[0],):
                raise ValueError("endpoint_twist_rad must have shape B.")
            if endpoint_orientations is None:
                raise ValueError("endpoint_twist_rad requires endpoint orientations.")
        elif endpoint_orientations is not None:
            endpoint_twist_rad = endpoint_twist_angle(
                positions_m,
                endpoint_orientations,
            ).detach()
        return DderState(
            positions_m,
            velocities,
            endpoint_orientations,
            endpoint_twist_rad,
        )

    def project_lengths(
        self,
        positions_m: torch.Tensor,
        boundary_positions_m: torch.Tensor,
        *,
        pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
    ) -> torch.Tensor:
        rest_lengths, masses = self._constants(positions_m)
        return momentum_project_lengths(
            positions_m,
            rest_lengths,
            masses,
            boundary_positions_m,
            iterations=self.parameters.constraint_iterations,
            pinned_endpoints=pinned_endpoints,
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
        *,
        pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
    ) -> torch.Tensor:
        _rest_lengths, masses = self._constants(positions_m)
        return momentum_project_velocities(
            positions_m,
            velocities_m_s,
            masses,
            boundary_velocities_m_s,
            validate=True,
            pinned_endpoints=pinned_endpoints,
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
        boundary_orientations_next: torch.Tensor | None = None,
        torsional_stiffness_n_m2: torch.Tensor | float | None = None,
        external_drag_s_inv: torch.Tensor | float | None = None,
        _validate: bool = True,
        _dense_constraint_solve: bool = False,
        pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
    ) -> DderState:
        q = state.positions_m
        v = state.velocities_m_s
        if q.shape[1:] != (self.parameters.node_count, 3) or v.shape != q.shape:
            raise ValueError("DDER state has invalid position/velocity shape.")
        pinned_endpoints = _validate_pinned_endpoints(
            pinned_endpoints,
            supported_only=True,
        )
        boundary_count = int(pinned_endpoints[0]) + int(pinned_endpoints[1])
        if boundary_positions_next_m.shape != (q.shape[0], boundary_count, 3):
            raise ValueError(
                f"boundary_positions_next_m must have shape Bx{boundary_count}x3 "
                "for the selected pinned endpoints."
            )
        one_attached = pinned_endpoints == START_PINNED_FREE_END
        if one_attached and (
            boundary_orientations_next is not None
            or state.endpoint_orientations is not None
            or state.endpoint_twist_rad is not None
        ):
            raise ValueError(
                "The position-only one-attached model has no terminal material "
                "frames or torsional state."
            )
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
            maximum_eigenvalue = (
                self._maximum_bending_eigenvalue_per_ei
                if pinned_endpoints == TWO_PINNED_ENDPOINTS
                else self._maximum_bending_eigenvalue_per_ei_one_attached
            )
            stability_number = (dt / self.parameters.substeps) * torch.sqrt(
                stiffness * maximum_eigenvalue
            )
            if bool(
                torch.any(stability_number > EXPLICIT_STABILITY_LIMIT).detach().cpu()
            ):
                raise ValueError(
                    "Explicit DDER bending step is outside its mass/grid/time-step "
                    "stability limit. Increase substeps or use an implicit bending step."
                )
        torsional_stiffness = _as_batch_scalar(
            self.parameters.torsional_stiffness_n_m2
            if torsional_stiffness_n_m2 is None
            else torsional_stiffness_n_m2,
            q,
            "torsional_stiffness_n_m2",
        )
        if _validate and bool(
            torch.any(~torch.isfinite(torsional_stiffness) | (torsional_stiffness < 0.0))
            .detach()
            .cpu()
        ):
            raise ValueError("torsional_stiffness_n_m2 must be finite and non-negative.")
        if one_attached and _validate and bool(
            torch.any(torsional_stiffness != 0.0).detach().cpu()
        ):
            raise ValueError(
                "The position-only one-attached model must have zero torsional "
                "stiffness."
            )
        external_drag = _as_batch_scalar(
            self.parameters.external_drag_s_inv
            if external_drag_s_inv is None
            else external_drag_s_inv,
            q,
            "external_drag_s_inv",
        )
        if _validate and bool(
            torch.any(~torch.isfinite(external_drag) | (external_drag < 0.0))
            .detach()
            .cpu()
        ):
            raise ValueError("external_drag_s_inv must be finite and non-negative.")
        if boundary_orientations_next is not None and boundary_orientations_next.shape != (
            q.shape[0],
            2,
            3,
            3,
        ):
            raise ValueError("boundary_orientations_next must have shape Bx2x3x3.")
        if (
            not one_attached
            and boundary_orientations_next is None
            and (
                torsional_stiffness_n_m2 is not None
                or self.parameters.torsional_stiffness_n_m2 > 0.0
            )
        ):
            raise ValueError("Twist-aware DDER requires both endpoint orientations.")
        start_orientations = state.endpoint_orientations
        if start_orientations is None:
            start_orientations = boundary_orientations_next
        continuous_twist = state.endpoint_twist_rad
        if continuous_twist is not None:
            if continuous_twist.shape != (q.shape[0],):
                raise ValueError("DDER state endpoint_twist_rad must have shape B.")
            if _validate and bool(
                torch.any(~torch.isfinite(continuous_twist)).detach().cpu()
            ):
                raise ValueError("DDER state endpoint twist must be finite.")
        elif start_orientations is not None:
            continuous_twist = _endpoint_twist_angle_impl(
                q,
                start_orientations,
                validate=_validate,
            ).detach()
        rest_lengths, masses = self._constants(q)
        gravity = torch.as_tensor(
            self.parameters.gravity_camera_m_s2,
            dtype=q.dtype,
            device=q.device,
        )[None, None]
        # Slice concatenation remains entirely on-device during CUDA graph
        # capture; tuple indexing would materialize a CPU index tensor.
        start_boundary = _pinned_values(q, pinned_endpoints)
        boundary_velocity = (
            boundary_positions_next_m - start_boundary
        ) / dt[:, None, None]
        substep_dt = (dt / self.parameters.substeps)[:, None, None]

        for substep in range(self.parameters.substeps):
            fraction = float(substep + 1) / self.parameters.substeps
            boundary = start_boundary + fraction * (
                boundary_positions_next_m - start_boundary
            )
            orientations = None
            orientation_rate = None
            if boundary_orientations_next is not None:
                assert start_orientations is not None
                orientations, orientation_rate = _interpolate_rotations(
                    start_orientations,
                    boundary_orientations_next,
                    fraction,
                    dt,
                )
            force = self.internal_force(
                q,
                create_graph=create_graph,
                bending_stiffness_n_m2=stiffness,
                endpoint_orientations=orientations,
                torsional_stiffness_n_m2=torsional_stiffness,
                endpoint_twist_reference_rad=continuous_twist,
                _validate=_validate,
            )
            acceleration = force / masses[None, :, None] + gravity
            undamped_v = (v + substep_dt * acceleration) * torch.exp(
                -external_drag[:, None, None] * substep_dt
            )
            predicted_v = _implicit_bending_damping_velocity(
                q,
                undamped_v,
                boundary_velocity,
                rest_lengths,
                masses,
                substep_dt[:, 0, 0],
                damping,
                orientations,
                orientation_rate,
                pinned_endpoints=pinned_endpoints,
            )
            predicted_q = q + substep_dt * predicted_v
            predicted_q = _replace_pinned_values(
                predicted_q,
                boundary,
                pinned_endpoints,
            )
            next_q = momentum_project_lengths(
                predicted_q,
                rest_lengths,
                masses,
                boundary,
                iterations=self.parameters.constraint_iterations,
                dense_solve=_dense_constraint_solve,
                pinned_endpoints=pinned_endpoints,
            )
            provisional_v = (next_q - q) / substep_dt
            v = momentum_project_velocities(
                next_q,
                provisional_v,
                masses,
                boundary_velocity,
                validate=_validate,
                dense_solve=_dense_constraint_solve,
                pinned_endpoints=pinned_endpoints,
            )
            q = next_q
            if orientations is not None:
                principal_twist = _endpoint_twist_angle_impl(
                    q,
                    orientations,
                    validate=False,
                )
                continuous_twist = _angle_nearest_reference(
                    principal_twist,
                    continuous_twist,
                ).detach()
        return DderState(
            q,
            v,
            boundary_orientations_next,
            continuous_twist,
        )

    def step_unchecked(
        self,
        state: DderState,
        boundary_positions_next_m: torch.Tensor,
        dt_s: torch.Tensor | float,
        *,
        pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
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
            pinned_endpoints=pinned_endpoints,
        )

    def step_runtime(
        self,
        state: DderState,
        boundary_positions_next_m: torch.Tensor,
        dt_s: torch.Tensor,
        constants: DderRuntimeConstants,
        boundary_orientations_next: torch.Tensor | None = None,
        *,
        iterative_damping: bool = False,
        damping_backend: str = "pcg60_reference",
        pinned_endpoints: PinnedEndpointMask = TWO_PINNED_ENDPOINTS,
    ) -> DderState:
        """Fixed-shape inference update.

        ``iterative_damping`` selects the capture-safe CG solution used by the
        online particle batch. Offline validation retains the direct solve.
        """

        pinned_endpoints = _validate_pinned_endpoints(
            pinned_endpoints,
            supported_only=True,
        )
        one_attached = pinned_endpoints == START_PINNED_FREE_END
        if one_attached and (
            boundary_orientations_next is not None
            or state.endpoint_orientations is not None
            or state.endpoint_twist_rad is not None
        ):
            raise ValueError(
                "The position-only one-attached model has no terminal material "
                "frames or torsional state."
            )
        q, v = state.positions_m, state.velocities_m_s
        dt = dt_s
        boundary_count = int(pinned_endpoints[0]) + int(pinned_endpoints[1])
        if boundary_positions_next_m.shape != (q.shape[0], boundary_count, 3):
            raise ValueError(
                f"boundary_positions_next_m must have shape Bx{boundary_count}x3 "
                "for the selected pinned endpoints."
            )
        start_boundary = _pinned_values(q, pinned_endpoints)
        boundary_velocity = (
            boundary_positions_next_m - start_boundary
        ) / dt[:, None, None]
        substep_dt = (dt / self.parameters.substeps)[:, None, None]
        start_orientations = state.endpoint_orientations
        if start_orientations is None:
            start_orientations = boundary_orientations_next
        continuous_twist = state.endpoint_twist_rad
        if continuous_twist is None and start_orientations is not None:
            continuous_twist = _endpoint_twist_angle_impl(
                q,
                start_orientations,
                validate=False,
            ).detach()
        for substep in range(self.parameters.substeps):
            fraction = float(substep + 1) / self.parameters.substeps
            boundary = start_boundary + fraction * (
                boundary_positions_next_m - start_boundary
            )
            orientations = None
            orientation_rate = None
            if boundary_orientations_next is not None:
                assert start_orientations is not None
                orientations, orientation_rate = _interpolate_rotations(
                    start_orientations,
                    boundary_orientations_next,
                    fraction,
                    dt,
                )
            force = self._runtime_internal_force(
                q,
                constants,
                orientations,
                continuous_twist,
            )
            acceleration = (
                force / constants.masses_kg[None, :, None]
                + constants.gravity_m_s2
            )
            undamped_v = (v + substep_dt * acceleration) * torch.exp(
                -constants.external_drag_s_inv[:, None, None] * substep_dt
            )
            predicted_v = _implicit_bending_damping_velocity(
                q,
                undamped_v,
                boundary_velocity,
                constants.rest_lengths_m,
                constants.masses_kg,
                substep_dt[:, 0, 0],
                constants.bending_damping_n_m2_s,
                orientations,
                orientation_rate,
                conjugate_gradient_iterations=(
                    6 * (q.shape[1] - boundary_count)
                    if iterative_damping
                    else None
                ),
                damping_backend=damping_backend,
                pinned_endpoints=pinned_endpoints,
            )
            predicted_q = q + substep_dt * predicted_v
            predicted_q = _replace_pinned_values(
                predicted_q,
                boundary,
                pinned_endpoints,
            )
            fixed_runtime_projection = (
                predicted_q.is_cuda
                and predicted_q.dtype == torch.float32
                and predicted_q.shape[1] in (11, 21, 31)
                and pinned_endpoints == START_PINNED_FREE_END
                and self.parameters.constraint_iterations == 4
                and os.environ.get("CABLE_TWIN_FUSED_FIXED_PROJECTION", "1") != "0"
            )
            if fixed_runtime_projection:
                from .cuda_fixed_pcg import fixed_projection_supported_nodes

                next_q, v = fixed_projection_supported_nodes(
                    predicted_q,
                    q,
                    constants.rest_lengths_m,
                    constants.masses_kg,
                    boundary,
                    boundary_velocity,
                    substep_dt[:, 0, 0],
                )
            else:
                next_q = momentum_project_lengths(
                    predicted_q,
                    constants.rest_lengths_m,
                    constants.masses_kg,
                    boundary,
                    iterations=self.parameters.constraint_iterations,
                    pinned_endpoints=pinned_endpoints,
                )
                provisional_v = (next_q - q) / substep_dt
                v = momentum_project_velocities(
                    next_q,
                    provisional_v,
                    constants.masses_kg,
                    boundary_velocity,
                    validate=False,
                    pinned_endpoints=pinned_endpoints,
                )
            q = next_q
            if orientations is not None:
                principal_twist = _endpoint_twist_angle_impl(
                    q,
                    orientations,
                    validate=False,
                )
                continuous_twist = _angle_nearest_reference(
                    principal_twist,
                    continuous_twist,
                ).detach()
        return DderState(
            q,
            v,
            boundary_orientations_next,
            continuous_twist,
        )

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
