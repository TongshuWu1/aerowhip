"""Fixed training-context normalization for off-policy one-shot learning."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from .policy_context import POLICY_CONTEXT_DIM


@dataclass(frozen=True, slots=True)
class FixedContextNormalizer:
    mean: torch.Tensor
    standard_deviation: torch.Tensor
    sample_count: int
    standard_deviation_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        mean = torch.as_tensor(self.mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(self.standard_deviation, dtype=torch.float32).reshape(-1)
        if mean.shape != (POLICY_CONTEXT_DIM,) or std.shape != (POLICY_CONTEXT_DIM,):
            raise ValueError("Context normalizer must contain 83 means and scales.")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError("Context normalizer must be finite.")
        safe = torch.where(std >= self.standard_deviation_floor, std, torch.ones_like(std))
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", safe)
        if self.sample_count < 1:
            raise ValueError("Context normalizer requires at least one sample.")

    @classmethod
    def fit(cls, samples: torch.Tensor, *, floor: float = 1.0e-6) -> "FixedContextNormalizer":
        values = torch.as_tensor(samples, dtype=torch.float32)
        if values.ndim != 2 or values.shape[1] != POLICY_CONTEXT_DIM:
            raise ValueError("Normalizer samples must have shape Nx83.")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Normalizer samples must be finite.")
        return cls(
            values.mean(dim=0),
            values.std(dim=0, unbiased=False),
            int(values.shape[0]),
            float(floor),
        )

    def normalize(self, context: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(context)
        if values.shape[-1] != POLICY_CONTEXT_DIM:
            raise ValueError("Context tensor has the wrong final dimension.")
        return (values - self.mean.to(values.device)) / self.standard_deviation.to(values.device)

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "fixed_policy_context_normalizer_v1",
            "sample_count": self.sample_count,
            "dimension": POLICY_CONTEXT_DIM,
            "standard_deviation_floor": self.standard_deviation_floor,
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
            "nominal_only_theta": True,
            "validation_data_used": False,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FixedContextNormalizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "fixed_policy_context_normalizer_v1":
            raise ValueError("Unsupported context normalizer artifact.")
        return cls(
            torch.tensor(payload["mean"], dtype=torch.float32),
            torch.tensor(payload["standard_deviation"], dtype=torch.float32),
            int(payload["sample_count"]),
            float(payload["standard_deviation_floor"]),
        )
