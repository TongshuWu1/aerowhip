"""Conditional 49-D DDPM/DDIM generator for production-normalized maneuvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn

from .policy_action import POLICY_ACTION_DIM
from .policy_context import POLICY_CONTEXT_DIM


DIFFUSION_TRAINING_STEPS = 100
TIMESTEP_EMBEDDING_DIM = 64
# The cosine schedule ends in an isolated near-zero-alpha cliff.  Epsilon
# prediction error at that point is amplified by ~2,000 when converted to x0.
# Starting DDIM at t=95 retains 25 deterministic steps and a near-standard-
# normal initial marginal without entering that numerically ill-conditioned
# terminal cliff.  See Milestone 7A.2 diagnostics.
DDIM_TERMINAL_CLIFF_STEPS = 4


def cosine_alpha_bar_schedule(
    steps: int = DIFFUSION_TRAINING_STEPS, *, s: float = 0.008
) -> torch.Tensor:
    if steps < 2:
        raise ValueError("Diffusion needs at least two timesteps.")
    time = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos(((time + s) / (1.0 + s)) * math.pi / 2.0).square()
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    betas = betas.clamp(1.0e-6, 0.999)
    return torch.cumprod(1.0 - betas, dim=0).to(torch.float32)


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, dimension: int = TIMESTEP_EMBEDDING_DIM
) -> torch.Tensor:
    if dimension % 2:
        raise ValueError("Sinusoidal timestep embedding dimension must be even.")
    time = torch.as_tensor(timesteps, dtype=torch.float32).reshape(-1)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(dimension // 2, device=time.device, dtype=time.dtype)
        / max(dimension // 2 - 1, 1)
    )
    angle = time[:, None] * frequencies[None]
    return torch.cat((torch.sin(angle), torch.cos(angle)), dim=-1)


class ResidualMlpBlock(nn.Module):
    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 512),
            nn.SiLU(),
            nn.Linear(512, width),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.network(value)


class ConditionalActionDiffusion(nn.Module):
    """Exact Milestone-7A compact conditional residual MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(POLICY_CONTEXT_DIM, 256), nn.SiLU(), nn.Linear(256, 256)
        )
        self.timestep_encoder = nn.Sequential(
            nn.Linear(TIMESTEP_EMBEDDING_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.action_projection = nn.Linear(POLICY_ACTION_DIM, 256)
        self.blocks = nn.ModuleList(ResidualMlpBlock() for _ in range(4))
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, POLICY_ACTION_DIM))

    def forward(
        self,
        noisy_action: torch.Tensor,
        normalized_context: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        action = torch.as_tensor(noisy_action)
        context = torch.as_tensor(
            normalized_context, dtype=action.dtype, device=action.device
        )
        time = torch.as_tensor(timesteps, device=action.device).reshape(-1)
        if action.ndim != 2 or action.shape[1] != POLICY_ACTION_DIM:
            raise ValueError("Noisy diffusion action must have shape Bx49.")
        if context.shape != (action.shape[0], POLICY_CONTEXT_DIM):
            raise ValueError("Diffusion context must have shape Bx83.")
        if time.shape != (action.shape[0],):
            raise ValueError("Diffusion timestep must have shape B.")
        embedded = sinusoidal_timestep_embedding(time).to(action.dtype)
        hidden = (
            self.action_projection(action)
            + self.context_encoder(context)
            + self.timestep_encoder(embedded)
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class FiLMResidualMlpBlock(nn.Module):
    """Residual MLP block modulated by the condition at every depth."""

    def __init__(self, width: int = 256) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(width)
        self.modulation = nn.Linear(width, 2 * width)
        self.network = nn.Sequential(
            nn.Linear(width, 512),
            nn.SiLU(),
            nn.Linear(512, width),
        )

    def forward(
        self, value: torch.Tensor, condition_and_time: torch.Tensor
    ) -> torch.Tensor:
        gamma, beta = self.modulation(condition_and_time).chunk(2, dim=-1)
        normalized = self.normalization(value)
        modulated = normalized * (1.0 + torch.tanh(gamma)) + beta
        return value + self.network(modulated)


class StructuredConditionEncoder(nn.Module):
    """Order-aware encoder for the unchanged normalized 83-D context.

    Cable positions and velocities are joined per physical node, enriched with
    learned node identity, and processed along the c1..c10 chain. The public
    context contract remains exactly the production 83-vector.
    """

    def __init__(self) -> None:
        super().__init__()
        self.uav_encoder = nn.Sequential(
            nn.Linear(10, 64), nn.SiLU(), nn.Linear(64, 64)
        )
        self.cable_node_encoder = nn.Sequential(
            nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 64)
        )
        self.cable_node_embedding = nn.Parameter(torch.empty(10, 64))
        nn.init.normal_(self.cable_node_embedding, std=0.02)
        self.cable_chain_encoder = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.cable_projection = nn.Sequential(
            nn.Linear(4 * 64, 128), nn.SiLU(), nn.Linear(128, 128)
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(6, 64), nn.SiLU(), nn.Linear(64, 64)
        )
        self.theta_encoder = nn.Sequential(
            nn.Linear(7, 32), nn.SiLU(), nn.Linear(32, 32)
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 128 + 64 + 32, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def forward(self, normalized_context: torch.Tensor) -> torch.Tensor:
        context = torch.as_tensor(normalized_context)
        if context.ndim != 2 or context.shape[1] != POLICY_CONTEXT_DIM:
            raise ValueError("Structured diffusion context must have shape Bx83.")
        uav = self.uav_encoder(context[:, :10])
        positions = context[:, 10:40].reshape(-1, 10, 3)
        velocities = context[:, 40:70].reshape(-1, 10, 3)
        cable_nodes = torch.cat((positions, velocities), dim=-1)
        cable_nodes = self.cable_node_encoder(cable_nodes)
        cable_nodes = cable_nodes + self.cable_node_embedding[None].to(
            dtype=context.dtype, device=context.device
        )
        cable_chain = self.cable_chain_encoder(cable_nodes.transpose(1, 2))
        cable_summary = torch.cat(
            (
                cable_chain.mean(dim=-1),
                cable_chain.amax(dim=-1),
                cable_chain[:, :, 0],
                cable_chain[:, :, -1],
            ),
            dim=-1,
        )
        cable = self.cable_projection(cable_summary)
        goal = self.goal_encoder(context[:, 70:76])
        theta = self.theta_encoder(context[:, 76:83])
        return self.fusion(torch.cat((uav, cable, goal, theta), dim=-1))


class StructuredConditionalActionDiffusion(nn.Module):
    """Permanent structured-FiLM replacement for the flat conditioner."""

    def __init__(self) -> None:
        super().__init__()
        self.context_encoder = StructuredConditionEncoder()
        self.timestep_encoder = nn.Sequential(
            nn.Linear(TIMESTEP_EMBEDDING_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.action_projection = nn.Linear(POLICY_ACTION_DIM, 256)
        self.blocks = nn.ModuleList(FiLMResidualMlpBlock() for _ in range(4))
        self.output = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, POLICY_ACTION_DIM))

    def forward(
        self,
        noisy_action: torch.Tensor,
        normalized_context: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        action = torch.as_tensor(noisy_action)
        context = torch.as_tensor(
            normalized_context, dtype=action.dtype, device=action.device
        )
        time = torch.as_tensor(timesteps, device=action.device).reshape(-1)
        if action.ndim != 2 or action.shape[1] != POLICY_ACTION_DIM:
            raise ValueError("Noisy diffusion action must have shape Bx49.")
        if context.shape != (action.shape[0], POLICY_CONTEXT_DIM):
            raise ValueError("Diffusion context must have shape Bx83.")
        if time.shape != (action.shape[0],):
            raise ValueError("Diffusion timestep must have shape B.")
        embedded = sinusoidal_timestep_embedding(time).to(action.dtype)
        condition = self.context_encoder(context)
        timestep = self.timestep_encoder(embedded)
        condition_and_time = condition + timestep
        hidden = self.action_projection(action) + condition_and_time
        for block in self.blocks:
            hidden = block(hidden, condition_and_time)
        return self.output(hidden)


@dataclass(frozen=True, slots=True)
class DiffusionLoss:
    loss: torch.Tensor
    noisy_action: torch.Tensor
    noise: torch.Tensor
    predicted_noise: torch.Tensor
    timesteps: torch.Tensor


def diffusion_epsilon_loss(
    model: nn.Module,
    normalized_context: torch.Tensor,
    clean_action: torch.Tensor,
    *,
    alpha_bar: torch.Tensor,
    generator: torch.Generator | None = None,
) -> DiffusionLoss:
    action = torch.as_tensor(clean_action)
    context = torch.as_tensor(
        normalized_context, dtype=action.dtype, device=action.device
    )
    if action.ndim != 2 or action.shape[1] != POLICY_ACTION_DIM:
        raise ValueError("Clean diffusion actions must have shape Bx49.")
    steps = int(alpha_bar.numel())
    timesteps = torch.randint(
        steps,
        (action.shape[0],),
        generator=generator,
        device=action.device,
    )
    noise = torch.randn(
        action.shape,
        generator=generator,
        device=action.device,
        dtype=action.dtype,
    )
    selected = alpha_bar.to(action.device)[timesteps].to(action.dtype)[:, None]
    noisy = torch.sqrt(selected) * action + torch.sqrt(1.0 - selected) * noise
    prediction = model(noisy, context, timesteps)
    loss = torch.mean((prediction - noise) ** 2)
    return DiffusionLoss(loss, noisy, noise, prediction, timesteps)


def ddim_timestep_schedule(
    training_steps: int = DIFFUSION_TRAINING_STEPS,
    sampling_steps: int = 25,
    *,
    maximum_timestep: int | None = None,
) -> torch.Tensor:
    if sampling_steps < 2 or sampling_steps > training_steps:
        raise ValueError("Invalid DDIM sampling step count.")
    maximum = (
        training_steps - 1 - DDIM_TERMINAL_CLIFF_STEPS
        if maximum_timestep is None
        else int(maximum_timestep)
    )
    if maximum < sampling_steps - 1 or maximum >= training_steps:
        raise ValueError("Invalid DDIM maximum timestep.")
    schedule = torch.linspace(maximum, 0, sampling_steps).round().long()
    if int(torch.unique(schedule).numel()) != sampling_steps:
        raise RuntimeError("DDIM schedule contains duplicated timesteps.")
    return schedule


@dataclass(frozen=True, slots=True)
class DdimSample:
    raw_action: torch.Tensor
    bounded_action: torch.Tensor
    clamped_coordinate_fraction: float


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    normalized_context: torch.Tensor,
    initial_noise: torch.Tensor,
    *,
    alpha_bar: torch.Tensor,
    timestep_schedule: torch.Tensor | None = None,
) -> DdimSample:
    model.eval()
    context = torch.as_tensor(normalized_context)
    noise = torch.as_tensor(
        initial_noise, dtype=context.dtype, device=context.device
    )
    if noise.ndim != 2 or noise.shape[1] != POLICY_ACTION_DIM:
        raise ValueError("DDIM initial noise must have shape Nx49.")
    if context.ndim != 2 or context.shape[1] != POLICY_CONTEXT_DIM:
        raise ValueError("DDIM context must have shape Bx83.")
    if context.shape[0] == 1 and noise.shape[0] > 1:
        context = context.expand(noise.shape[0], -1)
    if context.shape[0] != noise.shape[0]:
        raise ValueError("DDIM context/noise batch mismatch.")
    schedule = (
        ddim_timestep_schedule(int(alpha_bar.numel()), 25)
        if timestep_schedule is None
        else torch.as_tensor(timestep_schedule, dtype=torch.int64)
    ).to(context.device)
    cumulative = alpha_bar.to(device=context.device, dtype=context.dtype)
    value = noise.clone()
    raw_x0 = value
    for index, timestep in enumerate(schedule.tolist()):
        time = torch.full((value.shape[0],), int(timestep), device=value.device, dtype=torch.long)
        predicted_noise = model(value, context, time)
        current = cumulative[int(timestep)]
        raw_x0 = (value - torch.sqrt(1.0 - current) * predicted_noise) / torch.sqrt(current)
        if index + 1 < schedule.numel():
            next_alpha = cumulative[int(schedule[index + 1])]
            value = torch.sqrt(next_alpha) * raw_x0 + torch.sqrt(1.0 - next_alpha) * predicted_noise
        else:
            value = raw_x0
    bounded = raw_x0.clamp(-1.0, 1.0)
    fraction = float((raw_x0.abs() > 1.0).to(torch.float32).mean().cpu())
    return DdimSample(raw_x0, bounded, fraction)


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, *, decay: float = 0.999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must lie in (0,1).")
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            source = value.detach()
            if self.shadow[name].device != source.device:
                self.shadow[name] = self.shadow[name].to(source.device)
            if torch.is_floating_point(source):
                self.shadow[name].mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(source)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {
            name: torch.as_tensor(value).detach().clone()
            for name, value in dict(state["shadow"]).items()
        }


def clone_model_with_state(
    state: dict[str, torch.Tensor], *, device: torch.device | str
) -> ConditionalActionDiffusion:
    model = ConditionalActionDiffusion().to(device)
    model.load_state_dict(state)
    model.eval()
    return model
