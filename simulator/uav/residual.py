"""Small causal translational residual for the effective UAV model."""

from __future__ import annotations

import torch
from torch import nn


RESIDUAL_FEATURE_NAMES = (
    "position_error_x",
    "position_error_y",
    "position_error_z",
    "velocity_error_x",
    "velocity_error_y",
    "velocity_error_z",
    "command_acceleration_x",
    "command_acceleration_y",
    "command_acceleration_z",
)


class CausalTranslationalResidual(nn.Module):
    """Fixed 90 -> 32 -> 32 -> 3 causal MLP.

    Inputs have shape ``B x 10 x 9``.  Normalization statistics are training-
    only buffers and therefore serialize with the learned state.  The output
    is an acceleration in m/s^2.
    """

    history_samples = 10
    feature_dimension = 9
    hidden_dimensions = (32, 32)
    output_dimension = 3

    def __init__(
        self,
        feature_mean: torch.Tensor | None = None,
        feature_std: torch.Tensor | None = None,
        *,
        seed: int = 42,
    ) -> None:
        super().__init__()
        mean = (
            torch.zeros(self.feature_dimension, dtype=torch.float64)
            if feature_mean is None
            else torch.as_tensor(feature_mean).reshape(self.feature_dimension)
        )
        std = (
            torch.ones(self.feature_dimension, dtype=mean.dtype, device=mean.device)
            if feature_std is None
            else torch.as_tensor(
                feature_std, dtype=mean.dtype, device=mean.device
            ).reshape(self.feature_dimension)
        )
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("Residual normalization must be finite.")
        if bool((std <= 0.0).any()):
            raise ValueError("Residual normalization standard deviations must be positive.")
        self.register_buffer("feature_mean", mean.detach().clone())
        self.register_buffer("feature_std", std.detach().clone())

        # Isolate deterministic initialization from the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.network = nn.Sequential(
                nn.Linear(90, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
                nn.Linear(32, 3),
            )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, history_features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(history_features)
        expected = (
            features.ndim == 3
            and features.shape[1] == self.history_samples
            and features.shape[2] == self.feature_dimension
        )
        if not expected:
            raise ValueError("Residual history must have shape Bx10x9.")
        mean = self.feature_mean.to(features)
        std = self.feature_std.to(features)
        normalized = (features - mean) / std
        return self.network(normalized.reshape(features.shape[0], -1))


def residual_feature(
    command_position_m: torch.Tensor,
    command_velocity_m_s: torch.Tensor,
    command_acceleration_m_s2: torch.Tensor,
    simulated_position_m: torch.Tensor,
    simulated_velocity_m_s: torch.Tensor,
) -> torch.Tensor:
    """Construct exactly [e_p, e_v, a_cmd] with no measured-state input."""

    return torch.cat(
        (
            command_position_m - simulated_position_m,
            command_velocity_m_s - simulated_velocity_m_s,
            command_acceleration_m_s2,
        ),
        dim=-1,
    )
