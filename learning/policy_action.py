"""Normalized one-shot policy action and physical maneuver decoding."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from planning.cem_task import VariableDurationWhipTask
from planning.variable_duration import project_decisions


POLICY_ACTION_DIM = 49


def canonicalize_normalized_action(action: torch.Tensor) -> torch.Tensor:
    """Apply the production coordinate and per-knot radial action bounds.

    This is the representation-preserving projection used before decoding a
    normalized complete maneuver.  It lives beside the production action
    codec because robustness diagnostics also need the exact same contract.
    """

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
class DecodedPolicyAction:
    """A complete maneuver expressed in the query-time policy-local frame."""

    normalized_action: torch.Tensor
    acceleration_knots_local_m_s2: torch.Tensor
    duration_s: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.normalized_action.shape[0])

    def physical_decision_tensor(self) -> torch.Tensor:
        return torch.cat(
            (self.acceleration_knots_local_m_s2.reshape(self.batch_size, -1), self.duration_s[:, None]),
            dim=-1,
        )


def decode_policy_action(
    normalized_action: torch.Tensor,
    task: VariableDurationWhipTask,
    *,
    duration_max_s: float | None = None,
) -> DecodedPolicyAction:
    """Map one normalized 49D policy output into one complete physical maneuver."""

    values = torch.as_tensor(normalized_action)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[1] != POLICY_ACTION_DIM:
        raise ValueError(f"Normalized policy action must have shape 49 or Bx{POLICY_ACTION_DIM}.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Normalized policy actions must be finite.")
    bounded = values.clamp(-1.0, 1.0)
    raw_knots = (
        bounded[:, :48].reshape(-1, task.cem.knot_count, 3)
        * task.maximum_command_acceleration_m_s2
    )
    duration_max = (
        task.cem.duration_max_initial_s if duration_max_s is None else float(duration_max_s)
    )
    if duration_max <= task.cem.duration_min_s:
        raise ValueError("Action duration maximum must exceed the minimum.")
    duration = task.cem.duration_min_s + 0.5 * (bounded[:, -1] + 1.0) * (
        duration_max - task.cem.duration_min_s
    )
    physical = project_decisions(
        torch.cat((raw_knots.reshape(values.shape[0], -1), duration[:, None]), dim=-1),
        task,
        duration_max,
    )
    physical_knots = physical[:, :-1].reshape(-1, task.cem.knot_count, 3)
    effective_duration = physical[:, -1]
    effective_normalized_duration = 2.0 * (
        (effective_duration - task.cem.duration_min_s)
        / (duration_max - task.cem.duration_min_s)
    ) - 1.0
    effective_normalized = torch.cat(
        (
            (physical_knots / task.maximum_command_acceleration_m_s2).reshape(values.shape[0], -1),
            effective_normalized_duration[:, None],
        ),
        dim=-1,
    )
    return DecodedPolicyAction(
        normalized_action=effective_normalized,
        acceleration_knots_local_m_s2=physical_knots,
        duration_s=effective_duration,
    )


def encode_physical_action(
    acceleration_knots_local_m_s2: torch.Tensor,
    duration_s: torch.Tensor | float,
    task: VariableDurationWhipTask,
    *,
    duration_max_s: float | None = None,
) -> torch.Tensor:
    """Encode an already-valid physical maneuver for deterministic reference tests."""

    knots = torch.as_tensor(acceleration_knots_local_m_s2)
    if knots.ndim == 2:
        knots = knots.unsqueeze(0)
    if knots.ndim != 3 or knots.shape[1:] != (task.cem.knot_count, 3):
        raise ValueError("Physical knots must have shape 16x3 or Bx16x3.")
    duration = torch.as_tensor(duration_s, dtype=knots.dtype, device=knots.device).reshape(-1)
    if duration.numel() == 1 and knots.shape[0] > 1:
        duration = duration.expand(knots.shape[0])
    if duration.shape != (knots.shape[0],):
        raise ValueError("Duration must be scalar or contain one value per action row.")
    norms = torch.linalg.vector_norm(knots, dim=-1)
    if not bool(torch.isfinite(knots).all() and torch.isfinite(duration).all()):
        raise ValueError("Physical policy actions must be finite.")
    if not bool((norms <= task.maximum_command_acceleration_m_s2 + 1.0e-5).all()):
        raise ValueError("Physical acceleration knot exceeds the vector-norm limit.")
    duration_max = (
        task.cem.duration_max_initial_s if duration_max_s is None else float(duration_max_s)
    )
    if duration_max <= task.cem.duration_min_s:
        raise ValueError("Action duration maximum must exceed the minimum.")
    if not bool(
        (
            (duration >= task.cem.duration_min_s)
            & (duration <= duration_max)
        ).all()
    ):
        raise ValueError("Physical maneuver duration lies outside policy bounds.")
    normalized_duration = 2.0 * (
        (duration - task.cem.duration_min_s)
        / (duration_max - task.cem.duration_min_s)
    ) - 1.0
    normalized = torch.cat(
        (
            (knots / task.maximum_command_acceleration_m_s2).reshape(knots.shape[0], -1),
            normalized_duration[:, None],
        ),
        dim=-1,
    )
    if normalized.shape != (knots.shape[0], POLICY_ACTION_DIM):
        raise RuntimeError("Encoded policy action has the wrong dimension.")
    return normalized
