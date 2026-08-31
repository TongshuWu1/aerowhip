"""Simulator-free deployment API for the amortized-CEM diffusion policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .action_diffusion import (
    ConditionalActionDiffusion,
    cosine_alpha_bar_schedule,
    ddim_timestep_schedule,
    sample_ddim,
)
from .normalization import FixedContextNormalizer
from .outcome_scorer import (
    ManeuverOutcomeScorer,
    OutcomeTargetNormalizer,
    predict_physical_outcomes,
    select_candidate_from_predictions,
)
from .policy_action import POLICY_ACTION_DIM
from .policy_context import POLICY_CONTEXT_DIM


@dataclass(frozen=True, slots=True)
class DeploymentInitialState:
    uav_velocity_world_m_s: torch.Tensor
    uav_orientation_world_xyzw: torch.Tensor
    uav_angular_velocity_world_rad_s: torch.Tensor
    cable_c1_to_c10_positions_world_m: torch.Tensor
    cable_c1_to_c10_velocities_world_m_s: torch.Tensor
    physical_root_position_world_m: torch.Tensor


@dataclass(frozen=True, slots=True)
class DeploymentTheta:
    K_p: float
    K_v: float
    k_a: float
    K_R: float
    K_omega: float
    EI: float
    Cb: float

    def learning_tensor(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        value = torch.tensor(
            (
                self.K_p,
                self.K_v,
                self.k_a,
                self.K_R,
                self.K_omega,
                math.log(self.EI),
                math.log(self.Cb),
            ),
            device=device,
            dtype=dtype,
        )
        if not bool(torch.isfinite(value).all()) or self.EI <= 0.0 or self.Cb <= 0.0:
            raise ValueError("Deployment theta must be finite and physically positive.")
        return value


def _quaternion_multiply_xyzw(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lx, ly, lz, lw = left.unbind(-1)
    rx, ry, rz, rw = right.unbind(-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


def _canonical_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    value = quaternion / torch.linalg.vector_norm(quaternion)
    index = int(torch.argmax(torch.abs(value)))
    return value if value[index] >= 0.0 else -value


def _yaw_rotation(yaw: torch.Tensor) -> torch.Tensor:
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack(
        (
            torch.stack((cosine, -sine, zero)),
            torch.stack((sine, cosine, zero)),
            torch.stack((zero, zero, one)),
        )
    )


def build_deployment_context_tensor(
    initial_state: DeploymentInitialState,
    target_position_world_m: torch.Tensor,
    desired_direction_world: torch.Tensor,
    theta: DeploymentTheta,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    selected = torch.device(device)
    quaternion = torch.as_tensor(
        initial_state.uav_orientation_world_xyzw, dtype=torch.float32, device=selected
    ).reshape(4)
    quaternion = quaternion / torch.linalg.vector_norm(quaternion)
    x, y, z, w = quaternion
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    rotation = _yaw_rotation(yaw)

    def vector_local(value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=selected)
        return torch.einsum("ij,...j->...i", rotation.T, tensor)

    yaw_conjugate = torch.stack(
        (torch.zeros_like(yaw), torch.zeros_like(yaw), -torch.sin(0.5 * yaw), torch.cos(0.5 * yaw))
    )
    local_quaternion = _canonical_quaternion(
        _quaternion_multiply_xyzw(yaw_conjugate, quaternion)
    )
    root = torch.as_tensor(
        initial_state.physical_root_position_world_m, dtype=torch.float32, device=selected
    ).reshape(3)
    cable_positions = torch.as_tensor(
        initial_state.cable_c1_to_c10_positions_world_m,
        dtype=torch.float32,
        device=selected,
    ).reshape(10, 3)
    cable_velocities = torch.as_tensor(
        initial_state.cable_c1_to_c10_velocities_world_m_s,
        dtype=torch.float32,
        device=selected,
    ).reshape(10, 3)
    target = torch.as_tensor(target_position_world_m, dtype=torch.float32, device=selected).reshape(3)
    direction = torch.as_tensor(desired_direction_world, dtype=torch.float32, device=selected).reshape(3)
    direction = direction / torch.linalg.vector_norm(direction)
    context = torch.cat(
        (
            vector_local(initial_state.uav_velocity_world_m_s).reshape(3),
            local_quaternion,
            vector_local(initial_state.uav_angular_velocity_world_rad_s).reshape(3),
            vector_local(cable_positions - root).reshape(30),
            vector_local(cable_velocities).reshape(30),
            vector_local(target - root).reshape(3),
            vector_local(direction).reshape(3),
            theta.learning_tensor(device=selected, dtype=torch.float32),
        )
    ).reshape(1, POLICY_CONTEXT_DIM)
    if not bool(torch.isfinite(context).all()):
        raise ValueError("Deployment context contains NaN/Inf.")
    return context


@dataclass(frozen=True, slots=True)
class ManeuverInferenceResult:
    selected_normalized_action: torch.Tensor
    candidate_normalized_actions: torch.Tensor
    selected_index: int
    clamp_fraction: float
    predicted_gate_pass: torch.Tensor
    predicted_minimum_margin: torch.Tensor
    predicted_continuous_outcomes: torch.Tensor
    predicted_binary_probabilities: torch.Tensor

    def diagnostics(self) -> dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "clamp_fraction": self.clamp_fraction,
            "predicted_gate_pass_count": int(self.predicted_gate_pass.sum()),
            "selected_minimum_margin": float(self.predicted_minimum_margin[self.selected_index]),
        }


class AmortizedCemDiffusionPolicy:
    """One context query to 32 neural candidates, one score, and one action."""

    def __init__(
        self,
        diffusion: ConditionalActionDiffusion,
        scorer: ManeuverOutcomeScorer,
        context_normalizer: FixedContextNormalizer,
        scorer_target_normalizer: OutcomeTargetNormalizer,
        fixed_noise_bank: torch.Tensor,
    ) -> None:
        self.diffusion = diffusion.eval()
        self.scorer = scorer.eval()
        self.context_normalizer = context_normalizer
        self.scorer_target_normalizer = scorer_target_normalizer
        device = next(diffusion.parameters()).device
        self.fixed_noise_bank = torch.as_tensor(
            fixed_noise_bank, dtype=torch.float32, device=device
        )
        if self.fixed_noise_bank.shape != (32, POLICY_ACTION_DIM):
            raise ValueError("Frozen diffusion noise bank must have shape 32x49.")
        self.context_mean = context_normalizer.mean.to(device=device, dtype=torch.float32)
        self.context_standard_deviation = context_normalizer.standard_deviation.to(
            device=device, dtype=torch.float32
        )
        self.alpha_bar = cosine_alpha_bar_schedule().to(device)
        self.ddim_schedule = ddim_timestep_schedule(100, 25).to(device)

    @property
    def device(self) -> torch.device:
        return next(self.diffusion.parameters()).device

    @torch.no_grad()
    def infer_from_context_tensor(self, context_tensor: torch.Tensor) -> ManeuverInferenceResult:
        raw = torch.as_tensor(context_tensor, dtype=torch.float32, device=self.device).reshape(1, 83)
        normalized = (raw - self.context_mean) / self.context_standard_deviation
        sample = sample_ddim(
            self.diffusion,
            normalized,
            self.fixed_noise_bank,
            alpha_bar=self.alpha_bar,
            timestep_schedule=self.ddim_schedule,
        )
        repeated = normalized.expand(32, -1)
        outcomes = predict_physical_outcomes(
            self.scorer,
            repeated,
            sample.bounded_action,
            self.scorer_target_normalizer,
        )
        selection = select_candidate_from_predictions(outcomes)
        return ManeuverInferenceResult(
            sample.bounded_action[selection.selected_index].detach().clone(),
            sample.bounded_action.detach().clone(),
            selection.selected_index,
            sample.clamped_coordinate_fraction,
            selection.predicted_gate_pass.detach().clone(),
            selection.minimum_margin.detach().clone(),
            outcomes.continuous.detach().clone(),
            outcomes.binary_probability.detach().clone(),
        )

    @torch.no_grad()
    def infer_maneuver(
        self,
        initial_state: DeploymentInitialState,
        target: torch.Tensor,
        direction: torch.Tensor,
        theta: DeploymentTheta,
    ) -> ManeuverInferenceResult:
        context = build_deployment_context_tensor(
            initial_state, target, direction, theta, device=self.device
        )
        return self.infer_from_context_tensor(context)
