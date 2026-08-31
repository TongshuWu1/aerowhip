"""Low-dimensional, time-ordered codecs for complete one-shot maneuvers.

The production action remains the authoritative normalized 49-D vector.  These
codecs only provide a regression coordinate system; every decoded action is
canonicalized by the same coordinate and radial bounds as the production
decoder before it is executed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .policy_action import POLICY_ACTION_DIM


def open_uniform_bspline_basis(
    sample_count: int,
    control_point_count: int,
    *,
    degree: int = 3,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return a clamped open-uniform B-spline design matrix.

    The samples include both endpoints.  A direct Cox-de Boor construction is
    used so the permanent codec has no SciPy runtime dependency.
    """

    if sample_count < 2:
        raise ValueError("At least two trajectory samples are required.")
    if control_point_count < 2:
        raise ValueError("At least two spline control points are required.")
    effective_degree = min(int(degree), control_point_count - 1)
    interior_count = control_point_count - effective_degree - 1
    interior = (
        torch.arange(1, interior_count + 1, dtype=dtype) / (interior_count + 1)
        if interior_count > 0
        else torch.empty((0,), dtype=dtype)
    )
    knots = torch.cat(
        (
            torch.zeros((effective_degree + 1,), dtype=dtype),
            interior,
            torch.ones((effective_degree + 1,), dtype=dtype),
        )
    )
    samples = torch.linspace(0.0, 1.0, sample_count, dtype=dtype)
    basis = torch.zeros((sample_count, control_point_count), dtype=dtype)
    for index in range(control_point_count):
        basis[:, index] = ((samples >= knots[index]) & (samples < knots[index + 1])).to(dtype)
    basis[-1].zero_()
    basis[-1, -1] = 1.0

    for order in range(1, effective_degree + 1):
        updated = torch.zeros_like(basis)
        for index in range(control_point_count):
            left_width = knots[index + order] - knots[index]
            if float(left_width) > 0.0:
                updated[:, index] += (samples - knots[index]) / left_width * basis[:, index]
            if index + 1 < control_point_count:
                right_width = knots[index + order + 1] - knots[index + 1]
                if float(right_width) > 0.0:
                    updated[:, index] += (
                        (knots[index + order + 1] - samples) / right_width
                    ) * basis[:, index + 1]
        basis = updated
        basis[-1].zero_()
        basis[-1, -1] = 1.0

    if not torch.allclose(
        basis.sum(dim=1), torch.ones((sample_count,), dtype=dtype), atol=1.0e-12, rtol=0.0
    ):
        raise RuntimeError("B-spline basis does not form a partition of unity.")
    return basis


def canonicalize_normalized_action(action: torch.Tensor) -> torch.Tensor:
    """Apply the production action bounds without changing its 49-D schema."""

    value = torch.as_tensor(action)
    one_row = value.ndim == 1
    if one_row:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[1] != POLICY_ACTION_DIM:
        raise ValueError("Complete normalized actions must have shape 49 or Bx49.")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Complete normalized actions must be finite.")
    bounded = value.clamp(-1.0, 1.0)
    knots = bounded[:, :48].reshape(-1, 16, 3)
    norm = torch.linalg.vector_norm(knots, dim=-1, keepdim=True)
    knots = knots * torch.clamp(1.0 / torch.clamp(norm, min=1.0e-12), max=1.0)
    result = torch.cat((knots.reshape(-1, 48), bounded[:, 48:49]), dim=-1)
    return result[0] if one_row else result


@dataclass(frozen=True, slots=True)
class FixedCubicSplineActionCodec:
    """Fit/reconstruct the 16 ordered acceleration knots with fixed splines."""

    control_point_count: int

    @property
    def latent_dimension(self) -> int:
        return 3 * int(self.control_point_count) + 1

    @property
    def basis(self) -> torch.Tensor:
        return open_uniform_bspline_basis(16, self.control_point_count)

    def encode(self, normalized_action: torch.Tensor) -> torch.Tensor:
        value = canonicalize_normalized_action(normalized_action)
        one_row = value.ndim == 1
        if one_row:
            value = value.unsqueeze(0)
        basis = self.basis.to(device=value.device, dtype=value.dtype)
        pseudo_inverse = torch.linalg.pinv(basis)
        knots = value[:, :48].reshape(-1, 16, 3)
        controls = torch.einsum("ci,bia->bca", pseudo_inverse, knots)
        latent = torch.cat((controls.reshape(value.shape[0], -1), value[:, 48:49]), dim=-1)
        return latent[0] if one_row else latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        value = torch.as_tensor(latent)
        one_row = value.ndim == 1
        if one_row:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[1] != self.latent_dimension:
            raise ValueError(
                f"Spline latent must have shape {self.latent_dimension} or Bx{self.latent_dimension}."
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Spline latent must be finite.")
        basis = self.basis.to(device=value.device, dtype=value.dtype)
        controls = value[:, :-1].reshape(-1, self.control_point_count, 3)
        knots = torch.einsum("ic,bca->bia", basis, controls)
        action = torch.cat((knots.reshape(value.shape[0], -1), value[:, -1:]), dim=-1)
        result = canonicalize_normalized_action(action)
        return result[0] if one_row else result


@dataclass(frozen=True, slots=True)
class CompleteActionCodec:
    """Identity regression coordinates for the full production action."""

    @property
    def latent_dimension(self) -> int:
        return POLICY_ACTION_DIM

    def encode(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return canonicalize_normalized_action(normalized_action)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return canonicalize_normalized_action(latent)
