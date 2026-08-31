"""Learned physical-outcome scorer and deterministic scientific selector."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .amortized_cem_data import (
    BINARY_OUTCOME_NAMES,
    CONTINUOUS_OUTCOME_NAMES,
)
from .policy_action import POLICY_ACTION_DIM
from .policy_context import POLICY_CONTEXT_DIM


class ManeuverOutcomeScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(POLICY_CONTEXT_DIM + POLICY_ACTION_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.continuous_head = nn.Linear(256, len(CONTINUOUS_OUTCOME_NAMES))
        self.binary_head = nn.Linear(256, len(BINARY_OUTCOME_NAMES))

    def forward(
        self, normalized_context: torch.Tensor, normalized_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.as_tensor(normalized_context)
        action = torch.as_tensor(
            normalized_action, dtype=context.dtype, device=context.device
        )
        if context.ndim != 2 or context.shape[1] != POLICY_CONTEXT_DIM:
            raise ValueError("Scorer context must have shape Bx83.")
        if action.shape != (context.shape[0], POLICY_ACTION_DIM):
            raise ValueError("Scorer action must have shape Bx49.")
        hidden = self.trunk(torch.cat((context, action), dim=-1))
        return self.continuous_head(hidden), self.binary_head(hidden)


@dataclass(frozen=True, slots=True)
class OutcomeTargetNormalizer:
    mean: torch.Tensor
    standard_deviation: torch.Tensor
    sample_count: int
    standard_deviation_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(self.standard_deviation, dtype=torch.float32).reshape(-1)
        if mean.shape != (len(CONTINUOUS_OUTCOME_NAMES),) or std.shape != mean.shape:
            raise ValueError("Scorer target normalizer has the wrong shape.")
        safe = torch.where(std >= self.standard_deviation_floor, std, torch.ones_like(std))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", safe)
        if self.sample_count < 1:
            raise ValueError("Scorer target normalizer needs training rows.")

    @classmethod
    def fit(cls, training_targets: torch.Tensor) -> "OutcomeTargetNormalizer":
        value = torch.as_tensor(training_targets, dtype=torch.float32)
        if value.ndim != 2 or value.shape[1] != len(CONTINUOUS_OUTCOME_NAMES):
            raise ValueError("Continuous scorer targets must have shape Nx7.")
        return cls(value.mean(0), value.std(0, unbiased=False), int(value.shape[0]))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        return (tensor - self.mean.to(tensor.device)) / self.standard_deviation.to(tensor.device)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        return tensor * self.standard_deviation.to(tensor.device) + self.mean.to(tensor.device)

    def save(self, path: str | Path) -> None:
        payload = {
            "schema": "amortized_cem_scorer_target_normalization_v1",
            "names": list(CONTINUOUS_OUTCOME_NAMES),
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
            "sample_count": self.sample_count,
            "standard_deviation_floor": self.standard_deviation_floor,
            "training_split_only": True,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "OutcomeTargetNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "amortized_cem_scorer_target_normalization_v1":
            raise ValueError("Unsupported scorer normalizer schema.")
        return cls(
            torch.tensor(payload["mean"]),
            torch.tensor(payload["standard_deviation"]),
            int(payload["sample_count"]),
            float(payload["standard_deviation_floor"]),
        )


def binary_positive_weights(training_binary: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor(training_binary, dtype=torch.float32)
    positive = target.sum(dim=0)
    negative = target.shape[0] - positive
    return torch.clamp(negative / torch.clamp(positive, min=1.0), min=1.0, max=20.0)


@dataclass(frozen=True, slots=True)
class ScorerLoss:
    total: torch.Tensor
    continuous: torch.Tensor
    binary: torch.Tensor


def scorer_loss(
    predicted_continuous: torch.Tensor,
    predicted_binary_logits: torch.Tensor,
    target_continuous: torch.Tensor,
    target_binary: torch.Tensor,
    *,
    target_normalizer: OutcomeTargetNormalizer,
    positive_weights: torch.Tensor,
) -> ScorerLoss:
    standardized = target_normalizer.normalize(target_continuous)
    continuous_parts = F.smooth_l1_loss(
        predicted_continuous, standardized, reduction="none", beta=1.0
    ).mean(dim=0)
    binary_parts = F.binary_cross_entropy_with_logits(
        predicted_binary_logits,
        target_binary,
        reduction="none",
        pos_weight=positive_weights.to(predicted_binary_logits.device),
    ).mean(dim=0)
    continuous = continuous_parts.mean()
    binary = binary_parts.mean()
    return ScorerLoss(continuous + binary, continuous, binary)


class OutcomeStratifiedSampler:
    """Deterministic balanced sampling over success/feasible/unsafe outcome strata."""

    def __init__(self, continuous: np.ndarray, binary: np.ndarray, *, seed: int) -> None:
        cont = np.asarray(continuous, dtype=np.float32)
        bits = np.asarray(binary, dtype=np.float32)
        success = bits[:, 2] > 0.5
        finite = bits[:, 1] > 0.5
        displacement = cont[:, 4] <= 0.50
        speed = cont[:, 5] <= 3.0
        acceleration = cont[:, 6] <= 20.0
        feasible = finite & displacement & speed & acceleration
        self.strata = [
            np.flatnonzero(success),
            np.flatnonzero(feasible & ~success),
            np.flatnonzero(~feasible),
        ]
        self.all = np.arange(cont.shape[0], dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def sample(self, count: int) -> np.ndarray:
        quota = count // len(self.strata)
        selected: list[np.ndarray] = []
        for indices in self.strata:
            source = self.all if not indices.size else indices
            selected.append(self.rng.choice(source, size=quota, replace=True))
        remaining = count - quota * len(self.strata)
        if remaining:
            selected.append(self.rng.choice(self.all, size=remaining, replace=True))
        value = np.concatenate(selected)
        self.rng.shuffle(value)
        return value


@dataclass(frozen=True, slots=True)
class PredictedPhysicalOutcomes:
    continuous: torch.Tensor
    binary_probability: torch.Tensor

    @property
    def reward(self) -> torch.Tensor:
        return self.continuous[:, 0]

    @property
    def tip_distance_m(self) -> torch.Tensor:
        return torch.exp(self.continuous[:, 1]) - 1.0e-4

    @property
    def directed_speed_m_s(self) -> torch.Tensor:
        return self.continuous[:, 2]

    @property
    def direction_cosine(self) -> torch.Tensor:
        return self.continuous[:, 3]


@torch.no_grad()
def predict_physical_outcomes(
    scorer: ManeuverOutcomeScorer,
    normalized_context: torch.Tensor,
    normalized_action: torch.Tensor,
    target_normalizer: OutcomeTargetNormalizer,
) -> PredictedPhysicalOutcomes:
    continuous_standard, binary_logits = scorer(normalized_context, normalized_action)
    return PredictedPhysicalOutcomes(
        target_normalizer.denormalize(continuous_standard),
        torch.sigmoid(binary_logits),
    )


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    selected_index: int
    predicted_gate_pass: torch.Tensor
    minimum_margin: torch.Tensor
    predicted_outcomes: PredictedPhysicalOutcomes


def select_candidate_from_predictions(
    outcomes: PredictedPhysicalOutcomes,
) -> CandidateSelection:
    continuous = outcomes.continuous
    probability = outcomes.binary_probability
    cosine_gate = math.cos(math.radians(30.0))
    margins = torch.stack(
        (
            (0.050 - outcomes.tip_distance_m) / 0.050,
            (outcomes.directed_speed_m_s - 4.0) / 4.0,
            (outcomes.direction_cosine - cosine_gate) / max(1.0 - cosine_gate, 1.0e-6),
            (0.50 - continuous[:, 4]) / 0.50,
            (3.0 - continuous[:, 5]) / 3.0,
            (20.0 - continuous[:, 6]) / 20.0,
            (probability[:, 0] - 0.5) / 0.5,
            (probability[:, 1] - 0.5) / 0.5,
        ),
        dim=-1,
    )
    minimum = margins.min(dim=-1).values
    passed = (margins >= 0.0).all(dim=-1)
    if bool(passed.any()):
        indices = torch.nonzero(passed, as_tuple=False).flatten()
        local = torch.argmax(outcomes.reward[indices])
        selected = int(indices[local])
    else:
        best_margin = torch.max(minimum)
        indices = torch.nonzero(minimum == best_margin, as_tuple=False).flatten()
        local = torch.argmax(outcomes.reward[indices])
        selected = int(indices[local])
    return CandidateSelection(selected, passed, minimum, outcomes)


def _binary_curve_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(score, dtype=np.float64)
    positive, negative = int(y.sum()), int(y.size - y.sum())
    order = np.argsort(-p, kind="stable")
    sorted_y = y[order]
    true_positive = np.cumsum(sorted_y)
    false_positive = np.cumsum(1.0 - sorted_y)
    auroc = None
    if positive and negative:
        tpr = np.concatenate(([0.0], true_positive / positive, [1.0]))
        fpr = np.concatenate(([0.0], false_positive / negative, [1.0]))
        auroc = float(np.trapezoid(tpr, fpr))
    auprc = None
    if positive:
        precision = true_positive / np.arange(1, y.size + 1)
        recall = true_positive / positive
        previous = np.concatenate(([0.0], recall[:-1]))
        auprc = float(np.sum((recall - previous) * precision))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(np.mean((p - y) ** 2)),
        "positive_count": positive,
    }


def scorer_validation_metrics(
    true_continuous: np.ndarray,
    predicted_continuous: np.ndarray,
    true_binary: np.ndarray,
    predicted_binary_probability: np.ndarray,
) -> dict[str, object]:
    truth = np.asarray(true_continuous, dtype=np.float64)
    prediction = np.asarray(predicted_continuous, dtype=np.float64)
    continuous: dict[str, object] = {}
    for index, name in enumerate(CONTINUOUS_OUTCOME_NAMES):
        correlation = None
        if np.std(truth[:, index]) > 0.0 and np.std(prediction[:, index]) > 0.0:
            correlation = float(np.corrcoef(truth[:, index], prediction[:, index])[0, 1])
        error = prediction[:, index] - truth[:, index]
        continuous[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "correlation": correlation,
        }
    binary = {
        name: _binary_curve_metrics(true_binary[:, index], predicted_binary_probability[:, index])
        for index, name in enumerate(BINARY_OUTCOME_NAMES)
    }
    return {"continuous": continuous, "binary": binary}
