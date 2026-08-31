"""A small deterministic residual policy around a target-conditioned CEM seed."""

from __future__ import annotations

import torch
from torch import nn

from .policy_context import POLICY_CONTEXT_DIM


class DeterministicCemResidualPolicy(nn.Module):
    """Map one normalized context to a residual in maneuver coordinates."""

    def __init__(self, latent_dimension: int, hidden_dimension: int = 256) -> None:
        super().__init__()
        if latent_dimension < 1 or hidden_dimension < 1:
            raise ValueError("Policy dimensions must be positive.")
        self.latent_dimension = int(latent_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.network = nn.Sequential(
            nn.Linear(POLICY_CONTEXT_DIM, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, latent_dimension),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def residual(self, normalized_context: torch.Tensor) -> torch.Tensor:
        context = torch.as_tensor(normalized_context)
        if context.shape[-1] != POLICY_CONTEXT_DIM:
            raise ValueError("Normalized policy context must end in 83 dimensions.")
        return self.network(context)

    def forward(self, normalized_context: torch.Tensor, center_latent: torch.Tensor) -> torch.Tensor:
        center = torch.as_tensor(
            center_latent,
            dtype=normalized_context.dtype,
            device=normalized_context.device,
        )
        if center.shape[-1] != self.latent_dimension:
            raise ValueError("Center latent has the wrong final dimension.")
        return center + self.residual(normalized_context)
