"""Iterative residual outcome prediction for dynamic cable maneuvers.

The model follows the useful part of Iterative Residual Policy (IRP): after
executing a maneuver, it predicts how a proposed *small action correction*
will change the observed physical outcome.  It is not an action-regression
policy and it never calls the simulator or CEM while ranking corrections.

The authoritative maneuver remains the production normalized 49-D action.
Only the local correction search is compact: candidates are sampled in a
low-rank basis fitted from existing CEM-family residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from .policy_action import POLICY_ACTION_DIM, canonicalize_normalized_action
from .policy_context import POLICY_CONTEXT_DIM


CONTINUOUS_OUTCOME_NAMES = (
    "log_tip_distance",
    "directed_tip_speed",
    "direction_cosine",
    "log1p_uav_displacement",
    "log1p_uav_speed",
    "log1p_command_acceleration",
)
BINARY_OUTCOME_NAMES = ("tip_first", "feasible", "scientific_success")
CONTINUOUS_OUTCOME_DIM = len(CONTINUOUS_OUTCOME_NAMES)
BINARY_OUTCOME_DIM = len(BINARY_OUTCOME_NAMES)
OBSERVED_OUTCOME_DIM = CONTINUOUS_OUTCOME_DIM + BINARY_OUTCOME_DIM


def outcome_arrays_from_archive(archive) -> tuple[np.ndarray, np.ndarray]:
    """Extract stable physical targets from a robustness-outcome archive."""

    distance = np.asarray(archive["best_event_tip_distance_m"], dtype=np.float32)
    angle = np.asarray(archive["best_event_direction_angle_deg"], dtype=np.float32)
    continuous = np.stack(
        (
            np.log(np.maximum(distance, 0.0) + 1.0e-4),
            np.asarray(archive["best_event_directed_speed_m_s"], dtype=np.float32),
            np.cos(np.deg2rad(angle)),
            np.log1p(np.maximum(np.asarray(archive["maximum_uav_displacement_m"], dtype=np.float32), 0.0)),
            np.log1p(np.maximum(np.asarray(archive["maximum_uav_speed_m_s"], dtype=np.float32), 0.0)),
            np.log1p(np.maximum(np.asarray(archive["maximum_command_acceleration_m_s2"], dtype=np.float32), 0.0)),
        ),
        axis=-1,
    ).astype(np.float32)
    binary = np.stack(
        (
            np.asarray(archive["first_entry_marker"] == 10, dtype=np.float32),
            np.asarray(archive["feasible"], dtype=np.float32),
            np.asarray(archive["success"], dtype=np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    if not np.isfinite(continuous).all() or not np.isfinite(binary).all():
        raise ValueError("Outcome archive contains non-finite model targets.")
    return continuous, binary


def outcome_arrays_from_rows(
    rows: list[list[dict[str, object]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert authoritative result rows to the same learned-outcome schema."""

    continuous_rows: list[list[float]] = []
    binary_rows: list[list[float]] = []
    for context_rows in rows:
        if len(context_rows) != 1:
            raise ValueError("Observed outcome conversion expects one executed action/context.")
        metric = context_rows[0]
        angle = float(metric["best_event_direction_angle_deg"])
        distance = float(metric["best_event_tip_distance_m"])
        continuous_rows.append(
            [
                math.log(max(distance, 0.0) + 1.0e-4),
                float(metric["best_event_directed_speed_m_s"]),
                math.cos(math.radians(angle)),
                math.log1p(max(float(metric["maximum_uav_displacement_m"]), 0.0)),
                math.log1p(max(float(metric["maximum_uav_speed_m_s"]), 0.0)),
                math.log1p(max(float(metric["maximum_command_acceleration_m_s2"]), 0.0)),
            ]
        )
        marker = metric.get("first_entry_marker")
        binary_rows.append(
            [
                float(marker is not None and int(marker) == 10),
                float(bool(metric["feasible"])),
                float(bool(metric["success"])),
            ]
        )
    continuous = np.asarray(continuous_rows, dtype=np.float32)
    binary = np.asarray(binary_rows, dtype=np.float32)
    if not np.isfinite(continuous).all() or not np.isfinite(binary).all():
        raise ValueError("Authoritative outcome rows contain non-finite values.")
    return continuous, binary


@dataclass(frozen=True, slots=True)
class OutcomeNormalizer:
    mean: torch.Tensor
    standard_deviation: torch.Tensor

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float32).reshape(-1)
        scale = torch.as_tensor(self.standard_deviation, dtype=torch.float32).reshape(-1)
        if mean.shape != (CONTINUOUS_OUTCOME_DIM,) or scale.shape != mean.shape:
            raise ValueError("Outcome normalizer has the wrong shape.")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(scale).all()):
            raise ValueError("Outcome normalizer must be finite.")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", torch.clamp(scale, min=1.0e-4))

    @classmethod
    def fit(cls, values: torch.Tensor) -> "OutcomeNormalizer":
        tensor = torch.as_tensor(values, dtype=torch.float32).reshape(-1, CONTINUOUS_OUTCOME_DIM)
        return cls(tensor.mean(dim=0), tensor.std(dim=0, unbiased=False))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(values)
        return (tensor - self.mean.to(tensor.device)) / self.standard_deviation.to(tensor.device)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(values)
        return tensor * self.standard_deviation.to(tensor.device) + self.mean.to(tensor.device)

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "iterative_residual_outcome_normalizer_v1",
            "continuous_names": list(CONTINUOUS_OUTCOME_NAMES),
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
        }


class IterativeResidualOutcomeModel(nn.Module):
    """Predict the next physical outcome from an observed rollout and delta."""

    def __init__(self, hidden_dimension: int = 384) -> None:
        super().__init__()
        self.hidden_dimension = int(hidden_dimension)
        if self.hidden_dimension < 32:
            raise ValueError("Residual outcome model is unexpectedly small.")
        input_dimension = (
            POLICY_CONTEXT_DIM
            + POLICY_ACTION_DIM
            + OBSERVED_OUTCOME_DIM
            + POLICY_ACTION_DIM
        )
        self.trunk = nn.Sequential(
            nn.Linear(input_dimension, self.hidden_dimension),
            nn.SiLU(),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.SiLU(),
            nn.Linear(self.hidden_dimension, self.hidden_dimension),
            nn.SiLU(),
        )
        self.continuous_head = nn.Linear(self.hidden_dimension, CONTINUOUS_OUTCOME_DIM)
        self.binary_head = nn.Linear(self.hidden_dimension, BINARY_OUTCOME_DIM)

    def forward(
        self,
        normalized_context: torch.Tensor,
        current_action: torch.Tensor,
        normalized_current_outcome: torch.Tensor,
        action_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.as_tensor(normalized_context)
        action = torch.as_tensor(current_action)
        outcome = torch.as_tensor(normalized_current_outcome)
        delta = torch.as_tensor(action_delta)
        if context.shape[-1] != POLICY_CONTEXT_DIM:
            raise ValueError("Context must end in 83 dimensions.")
        if action.shape[-1] != POLICY_ACTION_DIM or delta.shape[-1] != POLICY_ACTION_DIM:
            raise ValueError("Current action and residual must end in 49 dimensions.")
        if outcome.shape[-1] != OBSERVED_OUTCOME_DIM:
            raise ValueError("Observed outcome has the wrong dimension.")
        hidden = self.trunk(torch.cat((context, action, outcome, delta), dim=-1))
        return self.continuous_head(hidden), self.binary_head(hidden)


def fit_compact_residual_basis(
    initial_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    *,
    rank: int = 13,
) -> torch.Tensor:
    """Fit an orthonormal local correction basis without decoding teachers."""

    initial = torch.as_tensor(initial_actions, dtype=torch.float64)
    teacher = torch.as_tensor(teacher_actions, dtype=torch.float64)
    if initial.shape != teacher.shape or initial.ndim != 2 or initial.shape[1] != POLICY_ACTION_DIM:
        raise ValueError("Residual-basis actions must have identical Nx49 shape.")
    if rank < 1 or rank > min(initial.shape[0], POLICY_ACTION_DIM):
        raise ValueError("Residual-basis rank is invalid.")
    residual = teacher - initial
    _u, _s, right = torch.linalg.svd(residual, full_matrices=False)
    basis = right[:rank]
    if not torch.allclose(
        basis @ basis.T,
        torch.eye(rank, dtype=basis.dtype),
        atol=1.0e-10,
        rtol=0.0,
    ):
        raise RuntimeError("Compact residual basis is not orthonormal.")
    return basis.float()


def deterministic_compact_candidates(
    current_action: torch.Tensor,
    basis: torch.Tensor,
    *,
    candidate_count: int,
    coordinate_rms_scales: tuple[float, ...],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct one deterministic, trust-region-bounded candidate bank."""

    current = torch.as_tensor(current_action, dtype=torch.float32).reshape(POLICY_ACTION_DIM)
    directions = torch.as_tensor(basis, dtype=torch.float32)
    if directions.ndim != 2 or directions.shape[1] != POLICY_ACTION_DIM:
        raise ValueError("Compact basis must have shape rank x 49.")
    if candidate_count < 2 or not coordinate_rms_scales:
        raise ValueError("Candidate bank needs at least two rows and one scale.")
    if any(scale <= 0.0 for scale in coordinate_rms_scales):
        raise ValueError("Candidate scales must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rank = int(directions.shape[0])
    deltas = [torch.zeros(POLICY_ACTION_DIM, dtype=torch.float32)]
    # Deterministic signed axes make every compact coordinate observable even
    # for small candidate counts.
    for direction in directions:
        for sign in (-1.0, 1.0):
            if len(deltas) >= candidate_count:
                break
            deltas.append(sign * float(coordinate_rms_scales[-1]) * math.sqrt(POLICY_ACTION_DIM) * direction)
    remaining = candidate_count - len(deltas)
    if remaining > 0:
        coefficients = torch.randn((remaining, rank), generator=generator)
        levels = torch.tensor(
            [coordinate_rms_scales[index % len(coordinate_rms_scales)] for index in range(remaining)],
            dtype=torch.float32,
        )
        coefficient_scale = levels * math.sqrt(POLICY_ACTION_DIM / rank)
        random_delta = (coefficients * coefficient_scale[:, None]) @ directions
        deltas.extend(random_delta)
    requested = torch.stack(deltas[:candidate_count])
    proposed = canonicalize_normalized_action(current[None] + requested)
    actual_delta = proposed - current[None]
    return proposed, actual_delta


def predicted_candidate_score(
    normalized_continuous: torch.Tensor,
    binary_logits: torch.Tensor,
    normalizer: OutcomeNormalizer,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank candidates by predicted scientific gates, without a simulator."""

    physical = normalizer.denormalize(normalized_continuous)
    probability = torch.sigmoid(binary_logits)
    tip_distance = torch.exp(physical[:, 0]) - 1.0e-4
    directed_speed = physical[:, 1]
    direction_cosine = physical[:, 2]
    displacement = torch.expm1(physical[:, 3])
    uav_speed = torch.expm1(physical[:, 4])
    command_acceleration = torch.expm1(physical[:, 5])
    margins = torch.stack(
        (
            (0.050 - tip_distance) / 0.050,
            (directed_speed - 4.0) / 4.0,
            (direction_cosine - math.cos(math.radians(30.0)))
            / (1.0 - math.cos(math.radians(30.0))),
            (0.50 - displacement) / 0.50,
            (3.0 - uav_speed) / 3.0,
            (20.0 - command_acceleration) / 20.0,
            (probability[:, 0] - 0.5) / 0.5,
            (probability[:, 1] - 0.5) / 0.5,
        ),
        dim=-1,
    )
    minimum_margin = margins.min(dim=-1).values
    score = (
        probability[:, 2]
        + 0.25 * probability[:, 1]
        + 0.10 * torch.tanh(minimum_margin)
        - 0.01 * torch.clamp(tip_distance, min=0.0)
    )
    return score, {
        "tip_distance_m": tip_distance,
        "directed_speed_m_s": directed_speed,
        "direction_cosine": direction_cosine,
        "uav_displacement_m": displacement,
        "uav_speed_m_s": uav_speed,
        "command_acceleration_m_s2": command_acceleration,
        "tip_first_probability": probability[:, 0],
        "feasible_probability": probability[:, 1],
        "success_probability": probability[:, 2],
        "minimum_gate_margin": minimum_margin,
    }
