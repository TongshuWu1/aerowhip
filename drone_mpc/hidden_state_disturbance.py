"""Evaluator-only hidden cable-state disturbances for observer experiments.

This module is deliberately separate from the observer and controller.  A
disturbance changes plant cable velocity while leaving root/tip position and
instantaneous root/tip velocity unchanged.  The online algorithms receive no
disturbance label, coefficient, direction, or injection time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from cable_twin.shared.dder import DderState

from .simulator import DroneCableState


REFERENCE_AMPLITUDE_M_S = 0.45
RANDOM_SMOOTH_SEED = 20260826


@dataclass(frozen=True, slots=True)
class HiddenVelocityDisturbance:
    """A normalized sine-series interior velocity disturbance."""

    name: str
    coefficients: tuple[float, ...]
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Hidden disturbance requires a name.")
        if not self.coefficients or not all(
            math.isfinite(value) for value in self.coefficients
        ):
            raise ValueError("Disturbance coefficients must be finite.")
        direction = np.asarray(self.direction, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if direction.shape != (3,) or not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("Disturbance direction must be a finite nonzero vector.")
        object.__setattr__(self, "direction", tuple((direction / norm).tolist()))

    def with_direction(
        self, direction: tuple[float, float, float], *, name: str | None = None
    ) -> "HiddenVelocityDisturbance":
        return replace(self, name=self.name if name is None else name, direction=direction)

    def raw_profile(self, node_count: int) -> np.ndarray:
        if node_count < 3:
            raise ValueError("Hidden disturbance requires at least three nodes.")
        coordinate = np.linspace(0.0, 1.0, node_count, dtype=np.float64)
        profile = np.zeros(node_count, dtype=np.float64)
        for mode, coefficient in enumerate(self.coefficients, start=1):
            profile += float(coefficient) * np.sin(mode * math.pi * coordinate)
        # Enforce the information boundary exactly rather than relying on
        # floating-point sin(k*pi) cancellation.
        profile[0] = 0.0
        profile[-1] = 0.0
        return profile

    def normalized_velocity_m_s(self, node_count: int) -> np.ndarray:
        profile = self.raw_profile(node_count)
        raw_rms = float(np.sqrt(np.mean(np.square(profile[1:]))))
        target_rms = reference_velocity_rms_m_s(node_count)
        if raw_rms <= np.finfo(np.float64).eps:
            if np.allclose(profile, 0.0):
                return np.zeros((node_count, 3), dtype=np.float64)
            raise ValueError("Hidden disturbance has numerically zero RMS.")
        scalar = target_rms * profile / raw_rms
        return scalar[:, None] * np.asarray(self.direction, dtype=np.float64)[None]


def reference_velocity_rms_m_s(node_count: int) -> float:
    """RMS over dynamic nodes of 0.45*sin(pi*s), including the free tip."""

    coordinate = np.linspace(0.0, 1.0, node_count, dtype=np.float64)
    profile = REFERENCE_AMPLITUDE_M_S * np.sin(math.pi * coordinate)
    profile[0] = 0.0
    profile[-1] = 0.0
    return float(np.sqrt(np.mean(np.square(profile[1:]))))


def apply_hidden_velocity_disturbance(
    state: DroneCableState,
    disturbance: HiddenVelocityDisturbance,
) -> DroneCableState:
    """Return plant truth with only the distributed cable velocity changed."""

    delta = torch.as_tensor(
        disturbance.normalized_velocity_m_s(state.cable.positions_m.shape[1]),
        dtype=state.cable.velocities_m_s.dtype,
        device=state.cable.velocities_m_s.device,
    )
    return DroneCableState(
        state.drone_position_m,
        state.drone_velocity_m_s,
        DderState(
            state.cable.positions_m,
            state.cable.velocities_m_s + delta[None],
        ),
    )


def disturbance_diagnostics(
    disturbance: HiddenVelocityDisturbance,
    node_count: int,
) -> dict[str, float]:
    delta = disturbance.normalized_velocity_m_s(node_count)
    speed = np.linalg.norm(delta, axis=1)
    return {
        "rms_velocity_m_s": float(np.sqrt(np.mean(np.square(speed[1:])))),
        "maximum_velocity_m_s": float(np.max(speed)),
        "squared_velocity_sum_m2_s2": float(np.sum(np.square(delta))),
    }


def basis_representability(
    disturbance: HiddenVelocityDisturbance,
    basis: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Evaluator-only least-squares projection into the observer basis."""

    matrix = np.asarray(basis, dtype=np.float64)
    delta = disturbance.normalized_velocity_m_s(matrix.shape[0])
    projection = np.empty_like(delta)
    coefficients = np.empty((matrix.shape[1], 3), dtype=np.float64)
    for axis in range(3):
        coefficient, *_ = np.linalg.lstsq(matrix, delta[:, axis], rcond=None)
        coefficients[:, axis] = coefficient
        projection[:, axis] = matrix @ coefficient
    denominator = float(np.linalg.norm(delta))
    residual = float(np.linalg.norm(delta - projection))
    relative = 0.0 if denominator == 0.0 else residual / denominator
    return {
        "relative_residual": relative,
        "explained_fraction": 1.0 - relative * relative,
        "projection_m_s": projection,
        "coefficients": coefficients,
    }


def primary_hidden_velocity_disturbances(
    *, random_seed: int = RANDOM_SMOOTH_SEED
) -> tuple[HiddenVelocityDisturbance, ...]:
    rng = np.random.default_rng(random_seed)
    # Decaying coefficients produce a smooth deterministic mixture while still
    # including modes outside the frozen four-function observer basis.
    random_coefficients = tuple(
        float(value) for value in rng.normal(size=6) / np.arange(1.0, 7.0)
    )
    return (
        HiddenVelocityDisturbance("clean", (0.0,)),
        HiddenVelocityDisturbance("sin1", (1.0,)),
        HiddenVelocityDisturbance("sin2", (0.0, 1.0)),
        HiddenVelocityDisturbance("sin3", (0.0, 0.0, 1.0)),
        HiddenVelocityDisturbance("sin4", (0.0, 0.0, 0.0, 1.0)),
        HiddenVelocityDisturbance("sin5", (0.0, 0.0, 0.0, 0.0, 1.0)),
        HiddenVelocityDisturbance("represented_mixture", (0.6, -0.4, 0.3)),
        HiddenVelocityDisturbance(
            "mixed_span", (0.55, -0.35, 0.0, 0.30, -0.20)
        ),
        HiddenVelocityDisturbance("random_smooth", random_coefficients),
    )
