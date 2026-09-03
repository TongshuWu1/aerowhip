"""Variable-horizon sequential whip environment shared by SAC/PPO baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch

from planning.rollout import clone_state_batch
from planning.task import CanonicalWhipTask
from simulator.cable.dder import DderState
from simulator.simulator import CoupledSimulator
from simulator.parameters import SimulatorParameters
from simulator.state import SimulatorState
from simulator.uav.state import (
    FullStateCommand,
    FullStateCommandSequence,
    ResidualHistoryState,
    UAVState,
)

from .normalization import FixedContextNormalizer
from .policy_context import POLICY_CONTEXT_DIM, build_policy_context


SEQUENTIAL_WHIP_OBSERVATION_DIM = POLICY_CONTEXT_DIM + 1
# Compatibility for the retired SAC runner and its historical checkpoints.
SEQUENTIAL_SAC_OBSERVATION_DIM = SEQUENTIAL_WHIP_OBSERVATION_DIM

FULL_6D_ACTION_MODE = "full_6d"
TARGET_ALIGNED_SAGITTAL_ACTION_MODE = "target_aligned_sagittal_3d"
WORLD_TIP_SPEED_SHAPING = "world_tip"
ATTACHMENT_RELATIVE_SPEED_SHAPING = "attachment_relative"
WORLD_TIP_PROGRESS_SHAPING = "world_tip"
ATTACHMENT_COMPENSATED_PROGRESS_SHAPING = "attachment_compensated_tip"
BLENDED_PROGRESS_SHAPING = "blended_world_attachment"


def sequential_action_dimension(action_mode: str) -> int:
    """Return the policy output width for one supported action contract."""

    dimensions = {
        FULL_6D_ACTION_MODE: 6,
        TARGET_ALIGNED_SAGITTAL_ACTION_MODE: 3,
    }
    try:
        return dimensions[str(action_mode)]
    except KeyError as error:
        raise ValueError(f"Unsupported sequential-whip action mode: {action_mode}") from error


def decode_target_aligned_sagittal_action(
    normalized_action: torch.Tensor,
    desired_direction_world: torch.Tensor,
    *,
    maximum_acceleration_m_s2: float,
    maximum_body_rate_rad_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode [along-target acceleration, vertical acceleration, pitch rate]."""

    if normalized_action.ndim != 2 or normalized_action.shape[-1] != 3:
        raise ValueError("Target-aligned sagittal action must have shape Bx3.")
    direction = torch.as_tensor(
        desired_direction_world,
        dtype=normalized_action.dtype,
        device=normalized_action.device,
    )
    if direction.shape != (normalized_action.shape[0], 3):
        raise ValueError("Desired direction must have shape Bx3.")
    forward = direction.clone()
    forward[:, 2] = 0.0
    forward_norm = torch.linalg.vector_norm(forward, dim=-1, keepdim=True)
    if not bool((forward_norm > torch.finfo(forward.dtype).eps).all()):
        raise ValueError("Target-aligned sagittal control requires a horizontal direction.")
    forward = forward / forward_norm

    sagittal_acceleration = normalized_action[:, :2]
    acceleration_norm = torch.linalg.vector_norm(
        sagittal_acceleration, dim=-1, keepdim=True
    )
    sagittal_acceleration = sagittal_acceleration / torch.maximum(
        acceleration_norm, torch.ones_like(acceleration_norm)
    )
    vertical = torch.tensor(
        (0.0, 0.0, 1.0),
        dtype=normalized_action.dtype,
        device=normalized_action.device,
    ).view(1, 3)
    acceleration_world = float(maximum_acceleration_m_s2) * (
        sagittal_acceleration[:, :1] * forward
        + sagittal_acceleration[:, 1:2] * vertical
    )
    body_rate = torch.zeros_like(acceleration_world)
    body_rate[:, 1] = float(maximum_body_rate_rad_s) * normalized_action[:, 2]
    return acceleration_world, body_rate


def shaping_tip_velocity(
    tip_velocity_world_m_s: torch.Tensor,
    attachment_velocity_world_m_s: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Select the velocity used only by dense reward shaping."""

    if mode == WORLD_TIP_SPEED_SHAPING:
        return tip_velocity_world_m_s
    if mode == ATTACHMENT_RELATIVE_SPEED_SHAPING:
        return tip_velocity_world_m_s - attachment_velocity_world_m_s
    raise ValueError(f"Unsupported directed-speed shaping reference: {mode}")


def progress_shaping_distance(
    tip_position_world_m: torch.Tensor,
    attachment_position_world_m: torch.Tensor,
    target_position_world_m: torch.Tensor,
    initial_attachment_position_world_m: torch.Tensor,
    *,
    mode: str,
    attachment_compensation_fraction: float = 0.5,
) -> torch.Tensor:
    """Distance used by reward progress without changing world-frame success."""

    fraction = float(attachment_compensation_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Attachment-compensation fraction must lie in [0,1].")
    world_distance = torch.linalg.vector_norm(
        tip_position_world_m - target_position_world_m, dim=-1
    )
    cable_span = tip_position_world_m - attachment_position_world_m
    desired_span = target_position_world_m - initial_attachment_position_world_m
    attachment_distance = torch.linalg.vector_norm(
        cable_span - desired_span, dim=-1
    )
    if mode == WORLD_TIP_PROGRESS_SHAPING:
        return world_distance
    if mode == ATTACHMENT_COMPENSATED_PROGRESS_SHAPING:
        return attachment_distance
    if mode == BLENDED_PROGRESS_SHAPING:
        return (
            (1.0 - fraction) * world_distance
            + fraction * attachment_distance
        )
    raise ValueError(f"Unsupported progress shaping reference: {mode}")


@dataclass(frozen=True, slots=True)
class SimpleRewardWeights:
    progress: float = 5.0
    directed_speed_near_target: float = 0.5
    direction_near_target: float = 0.5
    uav_displacement: float = 0.1
    success_bonus: float = 100.0
    numerical_failure: float = 100.0
    proximity_scale_m: float = 0.15
    non_tip_first: float = 0.0
    strike_quality_improvement: float = 0.0
    maximum_displacement: float = 0.0
    terminal_displacement: float = 0.0
    terminal_displacement_success_only: bool = False
    displacement_integral: float = 0.0
    success_forward_return_bonus: float = 0.0
    success_release_bonus: float = 0.0
    return_release_improvement: float = 0.0
    success_release_at_strike: bool = False
    forward_excursion_scale_m: float = 0.35
    uav_backward_speed_scale_m_s: float = 1.0
    relative_tip_forward_speed_scale_m_s: float = 4.0
    uav_speed_integral: float = 0.0
    acceleration_effort: float = 0.0
    body_rate_effort: float = 0.0
    action_smoothness: float = 0.0
    time_to_success: float = 0.0
    directed_speed_reward_cap_m_s: float = math.inf
    success_compactness_bonus: float = 0.0
    success_compactness_scale_m: float = 0.5
    displacement_cost_scale_m: float = 0.0


@dataclass(frozen=True, slots=True)
class ControlStepResult:
    next_observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    include_transition: torch.Tensor
    newly_successful: torch.Tensor
    final_tip_distance_m: torch.Tensor
    minimum_tip_distance_m: torch.Tensor
    maximum_uav_displacement_m: torch.Tensor
    numerical_failure: torch.Tensor


def simple_endpoint_success(
    tip_position_m: torch.Tensor,
    tip_velocity_m_s: torch.Tensor,
    target_position_m: torch.Tensor,
    desired_direction: torch.Tensor,
    *,
    radius_m: float = 0.05,
    minimum_directed_speed_m_s: float = 4.0,
    maximum_direction_error_deg: float = 30.0,
) -> torch.Tensor:
    """The reported success: endpoint position, speed, and direction only."""

    delta = tip_position_m - target_position_m
    distance = torch.linalg.vector_norm(delta, dim=-1)
    speed = torch.linalg.vector_norm(tip_velocity_m_s, dim=-1)
    direction = desired_direction / torch.linalg.vector_norm(
        desired_direction, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(desired_direction.dtype).eps)
    directed_speed = (tip_velocity_m_s * direction).sum(dim=-1)
    cosine = directed_speed / speed.clamp_min(torch.finfo(speed.dtype).eps)
    return (
        (distance <= radius_m)
        & (directed_speed >= minimum_directed_speed_m_s)
        & (cosine >= math.cos(math.radians(maximum_direction_error_deg)))
        & torch.isfinite(distance)
        & torch.isfinite(speed)
    )


def diagnostic_scientific_whip_success(
    *,
    endpoint_event_found: torch.Tensor,
    first_entry_marker: torch.Tensor,
    maximum_uav_displacement_m: torch.Tensor,
    maximum_uav_speed_m_s: torch.Tensor,
    maximum_command_acceleration_m_s2: torch.Tensor,
    finite: torch.Tensor,
    maximum_uav_displacement_limit_m: float = 0.50,
    maximum_uav_speed_limit_m_s: float = 3.0,
    maximum_command_acceleration_limit_m_s2: float = 20.0,
) -> torch.Tensor:
    """Legacy numerical-gate diagnostic; never used as shaped-PPO success."""

    return (
        endpoint_event_found
        & (first_entry_marker == 10)
        & (maximum_uav_displacement_m <= maximum_uav_displacement_limit_m)
        & (maximum_uav_speed_m_s <= maximum_uav_speed_limit_m_s)
        & (
            maximum_command_acceleration_m_s2
            <= maximum_command_acceleration_limit_m_s2
        )
        & finite
    )


def task_whip_success(
    *,
    endpoint_event_found: torch.Tensor,
    first_entry_marker: torch.Tensor,
    finite: torch.Tensor,
) -> torch.Tensor:
    """Single-attempt tip-first whip success without time or UAV safety gates."""

    return (
        endpoint_event_found
        & (first_entry_marker == 10)
        & finite
    )


def strike_quality(
    distance_m: torch.Tensor,
    directed_speed_m_s: torch.Tensor,
    direction_cosine: torch.Tensor,
    *,
    proximity_scale_m: float,
    target_directed_speed_m_s: float,
    maximum_direction_error_deg: float = 30.0,
    speed_reward_cap_m_s: float = math.inf,
) -> torch.Tensor:
    """Bounded soft conjunction of proximity, speed, and strike alignment."""

    proximity = torch.exp(-0.5 * torch.square(distance_m / proximity_scale_m))
    speed_width = max(0.25 * target_directed_speed_m_s, 1.0e-6)
    speed = torch.sigmoid(
        (directed_speed_m_s - target_directed_speed_m_s) / speed_width
    )
    if math.isfinite(speed_reward_cap_m_s):
        baseline = torch.sigmoid(
            torch.as_tensor(
                -target_directed_speed_m_s / speed_width,
                dtype=speed.dtype,
                device=speed.device,
            )
        )
        at_cap = torch.sigmoid(
            torch.as_tensor(
                (speed_reward_cap_m_s - target_directed_speed_m_s)
                / speed_width,
                dtype=speed.dtype,
                device=speed.device,
            )
        )
        speed = ((speed - baseline) / (at_cap - baseline).clamp_min(
            torch.finfo(speed.dtype).eps
        )).clamp(0.0, 1.0)
    direction = direction_gate_quality(
        direction_cosine,
        maximum_direction_error_deg=maximum_direction_error_deg,
    )
    return proximity * speed * direction


def direction_gate_quality(
    direction_cosine: torch.Tensor,
    *,
    maximum_direction_error_deg: float = 30.0,
    width: float = 0.15,
) -> torch.Tensor:
    """Smooth alignment quality centered on the actual scientific angle gate.

    The historical ``(cosine + 1) / 2`` score assigned roughly 60% credit to
    the observed 78-degree sideways strike.  This normalized sigmoid retains a
    gradient outside the gate while making that behavior low reward.
    """

    direction_threshold = math.cos(math.radians(maximum_direction_error_deg))
    direction = torch.sigmoid(
        (direction_cosine - direction_threshold) / max(width, 1.0e-6)
    )
    perfect_alignment = torch.sigmoid(
        torch.as_tensor(
            (1.0 - direction_threshold) / max(width, 1.0e-6),
            dtype=direction.dtype,
            device=direction.device,
        )
    )
    return (direction / perfect_alignment).clamp(0.0, 1.0)


def soft_near_target_strike_components(
    distance_m: torch.Tensor,
    tip_speed_m_s: torch.Tensor,
    directed_speed_m_s: torch.Tensor,
    direction_cosine: torch.Tensor,
    *,
    proximity_scale_m: float,
    target_directed_speed_m_s: float,
    speed_reward_cap_m_s: float = math.inf,
    maximum_direction_error_deg: float = 30.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded speed/alignment potentials with pre-success gradients."""

    proximity = torch.exp(
        -0.5 * torch.square(distance_m / max(proximity_scale_m, 1.0e-6))
    )
    speed_width = max(0.25 * target_directed_speed_m_s, 1.0e-6)
    speed_midpoint = 0.25 * target_directed_speed_m_s
    speed_baseline = torch.sigmoid(
        torch.as_tensor(
            -speed_midpoint / speed_width,
            dtype=directed_speed_m_s.dtype,
            device=directed_speed_m_s.device,
        )
    )
    speed_score = (
        (
            torch.sigmoid(
                (directed_speed_m_s - speed_midpoint) / speed_width
            )
            - speed_baseline
        )
        / (1.0 - speed_baseline).clamp_min(
            torch.finfo(speed_baseline.dtype).eps
        )
    ).clamp(0.0, 1.0)
    if math.isfinite(speed_reward_cap_m_s):
        cap_score = (
            torch.sigmoid(
                torch.as_tensor(
                    (speed_reward_cap_m_s - speed_midpoint) / speed_width,
                    dtype=directed_speed_m_s.dtype,
                    device=directed_speed_m_s.device,
                )
            )
            - speed_baseline
        ) / (1.0 - speed_baseline).clamp_min(
            torch.finfo(speed_baseline.dtype).eps
        )
        speed_score = (speed_score / cap_score.clamp_min(
            torch.finfo(speed_score.dtype).eps
        )).clamp(0.0, 1.0)

    motion_width = max(0.125 * target_directed_speed_m_s, 1.0e-6)
    motion_midpoint = 0.25 * target_directed_speed_m_s
    motion_baseline = torch.sigmoid(
        torch.as_tensor(
            -motion_midpoint / motion_width,
            dtype=tip_speed_m_s.dtype,
            device=tip_speed_m_s.device,
        )
    )
    motion_score = (
        (
            torch.sigmoid((tip_speed_m_s - motion_midpoint) / motion_width)
            - motion_baseline
        )
        / (1.0 - motion_baseline).clamp_min(
            torch.finfo(motion_baseline.dtype).eps
        )
    ).clamp(0.0, 1.0)
    alignment_score = direction_gate_quality(
        direction_cosine,
        maximum_direction_error_deg=maximum_direction_error_deg,
    )
    return (
        proximity * speed_score,
        proximity * motion_score * alignment_score,
    )


def smooth_success_compactness(
    maximum_displacement_m: torch.Tensor, *, scale_m: float
) -> torch.Tensor:
    """Smooth [0,1] preference for compact successful maneuvers."""

    return torch.exp(
        -0.5 * torch.square(maximum_displacement_m / max(scale_m, 1.0e-6))
    )


def smooth_displacement_cost(
    displacement_m: torch.Tensor, *, scale_m: float
) -> torch.Tensor:
    """Smooth non-negative displacement cost with zero cost at the origin."""

    return torch.log1p(
        torch.square(displacement_m / max(scale_m, 1.0e-6))
    )


def terminal_displacement_charge_mask(
    terminal_now: torch.Tensor,
    newly_successful: torch.Tensor,
    *,
    success_only: bool,
) -> torch.Tensor:
    """Choose whether terminal compactness is charged on all endings or successes."""

    if terminal_now.dtype != torch.bool or newly_successful.dtype != torch.bool:
        raise TypeError("Terminal displacement masks must be boolean tensors.")
    if terminal_now.shape != newly_successful.shape:
        raise ValueError("Terminal and success masks must have identical shapes.")
    return newly_successful if success_only else terminal_now


def forward_return_quality(
    peak_forward_displacement_m: torch.Tensor,
    current_forward_displacement_m: torch.Tensor,
    *,
    excursion_scale_m: float,
) -> torch.Tensor:
    """Bounded quality for genuine forward loading followed by return."""

    scale = max(float(excursion_scale_m), 1.0e-6)
    peak = peak_forward_displacement_m.clamp_min(0.0)
    returned = (peak - current_forward_displacement_m).clamp_min(0.0)
    return_fraction = (returned / peak.clamp_min(1.0e-6)).clamp(max=1.0)
    forward_activation = 1.0 - torch.exp(-peak / scale)
    return forward_activation * return_fraction


def forward_release_quality(
    peak_forward_displacement_m: torch.Tensor,
    uav_forward_speed_m_s: torch.Tensor,
    relative_tip_forward_speed_m_s: torch.Tensor,
    *,
    excursion_scale_m: float,
    backward_speed_scale_m_s: float,
    tip_speed_scale_m_s: float,
) -> torch.Tensor:
    """Bounded evidence of a retreating UAV releasing cable motion forward."""

    forward_activation = 1.0 - torch.exp(
        -peak_forward_displacement_m.clamp_min(0.0)
        / max(float(excursion_scale_m), 1.0e-6)
    )
    backward_quality = 1.0 - torch.exp(
        -(-uav_forward_speed_m_s).clamp_min(0.0)
        / max(float(backward_speed_scale_m_s), 1.0e-6)
    )
    tip_quality = 1.0 - torch.exp(
        -relative_tip_forward_speed_m_s.clamp_min(0.0)
        / max(float(tip_speed_scale_m_s), 1.0e-6)
    )
    return forward_activation * backward_quality * tip_quality


def return_release_strike_quality(
    peak_forward_displacement_m: torch.Tensor,
    current_forward_displacement_m: torch.Tensor,
    uav_forward_speed_m_s: torch.Tensor,
    relative_tip_forward_speed_m_s: torch.Tensor,
    current_strike_quality: torch.Tensor,
    *,
    excursion_scale_m: float,
    backward_speed_scale_m_s: float,
    tip_speed_scale_m_s: float,
) -> torch.Tensor:
    """Dense quality for loading, returning, and releasing toward a viable strike."""

    return_quality = forward_return_quality(
        peak_forward_displacement_m,
        current_forward_displacement_m,
        excursion_scale_m=excursion_scale_m,
    )
    backward_quality = 1.0 - torch.exp(
        -(-uav_forward_speed_m_s).clamp_min(0.0)
        / max(float(backward_speed_scale_m_s), 1.0e-6)
    )
    tip_quality = 1.0 - torch.exp(
        -relative_tip_forward_speed_m_s.clamp_min(0.0)
        / max(float(tip_speed_scale_m_s), 1.0e-6)
    )
    return (
        return_quality
        * backward_quality
        * tip_quality
        * current_strike_quality.clamp(0.0, 1.0)
    )


def simple_dense_reward(
    previous_distance_m: torch.Tensor,
    current_distance_m: torch.Tensor,
    best_proximity_directed_speed: torch.Tensor,
    best_proximity_direction_cosine: torch.Tensor,
    maximum_uav_displacement_m: torch.Tensor,
    newly_successful: torch.Tensor,
    numerical_failure: torch.Tensor,
    weights: SimpleRewardWeights,
) -> torch.Tensor:
    """Transparent task reward with only the four user-requested concepts."""

    return (
        weights.progress * (previous_distance_m - current_distance_m)
        + weights.directed_speed_near_target * best_proximity_directed_speed
        + weights.direction_near_target * best_proximity_direction_cosine
        - weights.uav_displacement * maximum_uav_displacement_m
        + weights.success_bonus * newly_successful.to(previous_distance_m.dtype)
        - weights.numerical_failure * numerical_failure.to(previous_distance_m.dtype)
    )


def _row_finite(state: SimulatorState) -> torch.Tensor:
    tensors: list[torch.Tensor] = [
        state.uav.position_m,
        state.uav.velocity_m_s,
        state.uav.orientation_xyzw,
        state.uav.angular_velocity_world_rad_s,
        state.cable.positions_m,
        state.cable.velocities_m_s,
    ]
    if state.uav.residual_history is not None:
        tensors.append(state.uav.residual_history.features)
    if state.uav.residual_acceleration_m_s2 is not None:
        tensors.append(state.uav.residual_acceleration_m_s2)
    if state.cable.endpoint_orientations is not None:
        tensors.append(state.cable.endpoint_orientations)
    if state.cable.endpoint_twist_rad is not None:
        tensors.append(state.cable.endpoint_twist_rad)
    finite = torch.ones(state.uav.batch_size, dtype=torch.bool, device=state.uav.position_m.device)
    for tensor in tensors:
        finite &= torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1)
    return finite


def _replace_rows(state: SimulatorState, source: SimulatorState, replace: torch.Tensor) -> SimulatorState:
    """Replace failed rows with a finite reset row so other batched rows continue."""

    def merged(value: torch.Tensor | None, fallback: torch.Tensor | None) -> torch.Tensor | None:
        if value is None or fallback is None:
            return value
        mask = replace.reshape((replace.shape[0],) + (1,) * (value.ndim - 1))
        return torch.where(mask, fallback, value)

    history = None
    if state.uav.residual_history is not None and source.uav.residual_history is not None:
        history = ResidualHistoryState(
            merged(state.uav.residual_history.features, source.uav.residual_history.features)
        )
    uav = UAVState(
        merged(state.uav.position_m, source.uav.position_m),
        merged(state.uav.velocity_m_s, source.uav.velocity_m_s),
        merged(state.uav.orientation_xyzw, source.uav.orientation_xyzw),
        merged(
            state.uav.angular_velocity_world_rad_s,
            source.uav.angular_velocity_world_rad_s,
        ),
        history,
        merged(
            state.uav.residual_acceleration_m_s2,
            source.uav.residual_acceleration_m_s2,
        ),
    )
    cable = DderState(
        merged(state.cable.positions_m, source.cable.positions_m),
        merged(state.cable.velocities_m_s, source.cable.velocities_m_s),
        merged(state.cable.endpoint_orientations, source.cable.endpoint_orientations),
        merged(state.cable.endpoint_twist_rad, source.cable.endpoint_twist_rad),
    )
    return SimulatorState(state.time_s, uav, cable)


class SequentialWhipEnvironment:
    """Synchronous batched MDP using the unmodified coupled production physics."""

    def __init__(
        self,
        simulator: CoupledSimulator,
        task: CanonicalWhipTask,
        initial_state: SimulatorState,
        normalizer: FixedContextNormalizer,
        *,
        batch_size: int,
        episode_duration_s: float = 10.0,
        control_dt_s: float = 0.1,
        maximum_acceleration_m_s2: float = 10.0,
        maximum_body_rate_rad_s: float = 4.0,
        observation_clip: float = 10.0,
        reward_weights: SimpleRewardWeights = SimpleRewardWeights(),
        action_mode: str = FULL_6D_ACTION_MODE,
        progress_shaping_reference: str = WORLD_TIP_PROGRESS_SHAPING,
        progress_attachment_compensation_fraction: float = 0.5,
        directed_speed_shaping_reference: str = WORLD_TIP_SPEED_SHAPING,
        success_mode: str = "simple_endpoint",
        reward_mode: str = "legacy_dense",
        terminate_on_success: bool = False,
        rollout_parameters: SimulatorParameters | None = None,
        record_fullstate_commands: bool = False,
        record_state_trajectory: bool = False,
        state_postprocessor: Callable[[int, SimulatorState], SimulatorState] | None = None,
    ) -> None:
        ratio = control_dt_s / simulator.dt_s
        if abs(ratio - round(ratio)) > 1.0e-9:
            raise ValueError("control_dt_s must be an integer multiple of physics dt.")
        control_steps = episode_duration_s / control_dt_s
        if abs(control_steps - round(control_steps)) > 1.0e-9:
            raise ValueError("episode_duration_s must be an integer multiple of control dt.")
        self.simulator = simulator
        self.task = task
        self.normalizer = normalizer
        self.batch_size = int(batch_size)
        self.physics_steps_per_control = int(round(ratio))
        self.control_step_count = int(round(control_steps))
        self.control_dt_s = float(control_dt_s)
        self.maximum_acceleration_m_s2 = float(maximum_acceleration_m_s2)
        self.maximum_body_rate_rad_s = float(maximum_body_rate_rad_s)
        self.observation_clip = float(observation_clip)
        self.reward_weights = reward_weights
        self.action_mode = str(action_mode)
        self.action_dimension = sequential_action_dimension(self.action_mode)
        if progress_shaping_reference not in {
            WORLD_TIP_PROGRESS_SHAPING,
            ATTACHMENT_COMPENSATED_PROGRESS_SHAPING,
            BLENDED_PROGRESS_SHAPING,
        }:
            raise ValueError(
                "Unsupported progress shaping reference: "
                f"{progress_shaping_reference}"
            )
        self.progress_shaping_reference = str(progress_shaping_reference)
        if not 0.0 <= float(progress_attachment_compensation_fraction) <= 1.0:
            raise ValueError(
                "Progress attachment-compensation fraction must lie in [0,1]."
            )
        self.progress_attachment_compensation_fraction = float(
            progress_attachment_compensation_fraction
        )
        if directed_speed_shaping_reference not in {
            WORLD_TIP_SPEED_SHAPING,
            ATTACHMENT_RELATIVE_SPEED_SHAPING,
        }:
            raise ValueError(
                "Unsupported directed-speed shaping reference: "
                f"{directed_speed_shaping_reference}"
            )
        self.directed_speed_shaping_reference = str(
            directed_speed_shaping_reference
        )
        self.body_rate_effort_axis_count = (
            3 if self.action_mode == FULL_6D_ACTION_MODE else 1
        )
        self.displacement_cost_scale_m = (
            float(reward_weights.displacement_cost_scale_m)
            if reward_weights.displacement_cost_scale_m > 0.0
            else float(task.maximum_uav_displacement_m)
        )
        # These options are diagnostic execution controls.  Defaults preserve
        # the training path exactly.  In particular, build_policy_context still
        # reads the nominal simulator parameters even when propagation uses a
        # mismatch override, so the frozen PPO is not given theta' implicitly.
        self.rollout_parameters = (
            simulator.parameters if rollout_parameters is None else rollout_parameters
        )
        self.record_fullstate_commands = bool(record_fullstate_commands)
        self.record_state_trajectory = bool(record_state_trajectory)
        self.state_postprocessor = state_postprocessor
        if success_mode not in {"simple_endpoint", "task_whip_once"}:
            raise ValueError(f"Unsupported sequential-whip success mode: {success_mode}")
        if reward_mode not in {"legacy_dense", "whip_potential"}:
            raise ValueError(f"Unsupported sequential-whip reward mode: {reward_mode}")
        self.success_mode = success_mode
        self.reward_mode = reward_mode
        self.terminate_on_success = bool(terminate_on_success)
        self._initial_state = clone_state_batch(initial_state, batch_size)
        self._initial_yaw_command = torch.full(
            (batch_size,),
            task.initial_yaw_rad,
            dtype=simulator.dtype,
            device=simulator.device,
        )
        self._target = torch.tensor(
            task.target_position_m, dtype=simulator.dtype, device=simulator.device
        ).view(1, 3).expand(batch_size, -1)
        self._direction = torch.tensor(
            task.desired_direction,
            dtype=simulator.dtype,
            device=simulator.device,
        ).view(1, 3).expand(batch_size, -1)
        self._initial_attachment_position_m = (
            self.simulator.root_boundary.evaluate(
                self._initial_state.uav,
                self.simulator.cable_configuration.rest_lengths_m[0],
            ).attachment_position_m
        )
        self.state = self._initial_state
        self.control_index = 0
        self.yaw_command = self._initial_yaw_command.clone()
        self.failed = torch.zeros(batch_size, dtype=torch.bool, device=simulator.device)
        self.episode_success = torch.zeros_like(self.failed)
        self.episode_endpoint_success = torch.zeros_like(self.failed)
        self.episode_scientific_success = torch.zeros_like(self.failed)
        self.episode_scientific_event = torch.zeros_like(self.failed)
        self.episode_first_entry_marker = torch.zeros(
            batch_size, dtype=torch.int64, device=simulator.device
        )
        self.episode_first_entry_physics_step = torch.zeros(
            batch_size, dtype=torch.int64, device=simulator.device
        )
        self.episode_first_entry_time_s = torch.full(
            (batch_size,), torch.nan, dtype=simulator.dtype, device=simulator.device
        )
        self.episode_first_entry_tip_distance = torch.full_like(
            self.episode_first_entry_time_s, torch.nan
        )
        self.episode_first_entry_tip_speed = torch.full_like(
            self.episode_first_entry_time_s, torch.nan
        )
        self.episode_first_entry_directed_speed = torch.full_like(
            self.episode_first_entry_time_s, torch.nan
        )
        self.episode_first_entry_direction_error_deg = torch.full_like(
            self.episode_first_entry_time_s, torch.nan
        )
        self.episode_reward = torch.zeros(batch_size, dtype=simulator.dtype, device=simulator.device)
        self.episode_maximum_displacement = torch.zeros_like(self.episode_reward)
        self.episode_terminal_displacement = torch.full_like(
            self.episode_reward, torch.nan
        )
        self.episode_peak_forward_displacement = torch.zeros_like(
            self.episode_reward
        )
        self.episode_success_return_distance = torch.full_like(
            self.episode_reward, torch.nan
        )
        self.episode_success_return_quality = torch.full_like(
            self.episode_reward, torch.nan
        )
        self.episode_best_release_quality = torch.zeros_like(self.episode_reward)
        self.episode_best_return_release_quality = torch.zeros_like(
            self.episode_reward
        )
        self.episode_success_release_quality = torch.full_like(
            self.episode_reward, torch.nan
        )
        self.episode_success_forward_displacement = torch.full_like(
            self.episode_reward, torch.nan
        )
        self.episode_success_uav_forward_speed = torch.full_like(
            self.episode_reward, torch.nan
        )
        self.episode_success_relative_tip_forward_speed = torch.full_like(
            self.episode_reward, torch.nan
        )
        self.episode_maximum_uav_speed = torch.zeros_like(self.episode_reward)
        self.episode_maximum_command_acceleration = torch.zeros_like(self.episode_reward)
        self.episode_minimum_tip_distance = torch.full_like(self.episode_reward, torch.inf)
        self.episode_minimum_progress_distance = torch.full_like(
            self.episode_reward, torch.inf
        )
        self.episode_best_strike_quality = torch.zeros_like(self.episode_reward)
        self.episode_best_directed_speed_quality = torch.zeros_like(
            self.episode_reward
        )
        self.episode_best_direction_quality = torch.zeros_like(self.episode_reward)
        self.previous_normalized_action = torch.zeros(
            (batch_size, self.action_dimension),
            dtype=simulator.dtype,
            device=simulator.device,
        )
        self.episode_success_tip_distance = torch.full_like(self.episode_reward, torch.nan)
        self.episode_success_tip_speed = torch.full_like(self.episode_reward, torch.nan)
        self.episode_success_directed_speed = torch.full_like(self.episode_reward, torch.nan)
        self.episode_success_direction_error_deg = torch.full_like(
            self.episode_reward, torch.nan
        )
        self._command_recording: dict[str, torch.Tensor] | None = None
        self._state_recording: dict[str, torch.Tensor] | None = None
        self._allocate_recordings()

    @property
    def physics_step_count(self) -> int:
        return self.control_step_count * self.physics_steps_per_control

    def _allocate_recordings(self) -> None:
        if self.record_fullstate_commands:
            vector_shape = (self.physics_step_count, self.batch_size, 3)
            self._command_recording = {
                "position_m": torch.empty(
                    vector_shape, dtype=self.simulator.dtype, device=self.simulator.device
                ),
                "velocity_m_s": torch.empty(
                    vector_shape, dtype=self.simulator.dtype, device=self.simulator.device
                ),
                "acceleration_m_s2": torch.empty(
                    vector_shape, dtype=self.simulator.dtype, device=self.simulator.device
                ),
                "orientation_xyzw": torch.empty(
                    (self.physics_step_count, self.batch_size, 4),
                    dtype=self.simulator.dtype,
                    device=self.simulator.device,
                ),
                "angular_velocity_body_rad_s": torch.empty(
                    vector_shape, dtype=self.simulator.dtype, device=self.simulator.device
                ),
            }
        if self.record_state_trajectory:
            step_shape = (self.physics_step_count + 1, self.batch_size)
            self._state_recording = {
                "uav_position_m": torch.empty(
                    (*step_shape, 3), dtype=self.simulator.dtype, device=self.simulator.device
                ),
                "uav_velocity_m_s": torch.empty(
                    (*step_shape, 3), dtype=self.simulator.dtype, device=self.simulator.device
                ),
                "uav_orientation_xyzw": torch.empty(
                    (*step_shape, 4), dtype=self.simulator.dtype, device=self.simulator.device
                ),
                "uav_angular_velocity_world_rad_s": torch.empty(
                    (*step_shape, 3), dtype=self.simulator.dtype, device=self.simulator.device
                ),
                "residual_acceleration_m_s2": torch.empty(
                    (*step_shape, 3), dtype=self.simulator.dtype, device=self.simulator.device
                ),
                "cable_positions_m": torch.empty(
                    (*step_shape, 12, 3),
                    dtype=self.simulator.dtype,
                    device=self.simulator.device,
                ),
                "cable_velocities_m_s": torch.empty(
                    (*step_shape, 12, 3),
                    dtype=self.simulator.dtype,
                    device=self.simulator.device,
                ),
            }

    def _record_state(self, index: int) -> None:
        if self._state_recording is None:
            return
        self._state_recording["uav_position_m"][index].copy_(
            self.state.uav.position_m
        )
        self._state_recording["uav_velocity_m_s"][index].copy_(
            self.state.uav.velocity_m_s
        )
        self._state_recording["uav_orientation_xyzw"][index].copy_(
            self.state.uav.orientation_xyzw
        )
        self._state_recording["uav_angular_velocity_world_rad_s"][index].copy_(
            self.state.uav.angular_velocity_world_rad_s
        )
        residual_acceleration = self.state.uav.residual_acceleration_m_s2
        if residual_acceleration is None:
            self._state_recording["residual_acceleration_m_s2"][index].zero_()
        else:
            self._state_recording["residual_acceleration_m_s2"][index].copy_(
                residual_acceleration
            )
        self._state_recording["cable_positions_m"][index].copy_(
            self.state.cable.positions_m
        )
        self._state_recording["cable_velocities_m_s"][index].copy_(
            self.state.cable.velocities_m_s
        )

    def recorded_command_sequence(self, *, clone: bool = True) -> FullStateCommandSequence:
        if self._command_recording is None:
            raise RuntimeError("FullState command recording was not enabled.")
        if self.control_index != self.control_step_count:
            raise RuntimeError("The PPO episode is incomplete; no compiled trajectory is ready.")

        def value(name: str) -> torch.Tensor:
            tensor = self._command_recording[name]
            return tensor.clone() if clone else tensor

        return FullStateCommandSequence(
            value("position_m"),
            value("velocity_m_s"),
            value("acceleration_m_s2"),
            value("orientation_xyzw"),
            value("angular_velocity_body_rad_s"),
        )

    def recorded_state_trajectory(self, *, clone: bool = True) -> dict[str, torch.Tensor]:
        if self._state_recording is None:
            raise RuntimeError("State-trajectory recording was not enabled.")
        if self.control_index != self.control_step_count:
            raise RuntimeError("The PPO episode is incomplete; no state trajectory is ready.")
        return {
            name: tensor.clone() if clone else tensor
            for name, tensor in self._state_recording.items()
        }

    def _observation(self) -> tuple[torch.Tensor, object]:
        context = build_policy_context(
            self.simulator,
            self.state,
            target_position_world_m=self._target,
            desired_direction_world=self._direction,
            command_yaw_world_rad=self.yaw_command,
        )
        normalized = self.normalizer.normalize(context.to_tensor().float())
        normalized = normalized.clamp(-self.observation_clip, self.observation_clip)
        remaining = torch.full(
            (self.batch_size, 1),
            1.0 - self.control_index / self.control_step_count,
            dtype=normalized.dtype,
            device=normalized.device,
        )
        return torch.cat((normalized, remaining), dim=-1), context

    def set_episode_initial_state(
        self,
        initial_state: SimulatorState,
        *,
        command_yaw_world_rad: torch.Tensor | float | None = None,
    ) -> None:
        """Install one complete causal batch for the next episode reset."""

        if initial_state.uav.batch_size != self.batch_size:
            raise ValueError(
                "Episode initial-state batch must match the environment batch size."
            )
        self._initial_state = clone_state_batch(initial_state, self.batch_size)
        self._initial_attachment_position_m = (
            self.simulator.root_boundary.evaluate(
                self._initial_state.uav,
                self.simulator.cable_configuration.rest_lengths_m[0],
            ).attachment_position_m
        )
        yaw = torch.as_tensor(
            self.task.initial_yaw_rad
            if command_yaw_world_rad is None
            else command_yaw_world_rad,
            dtype=self.simulator.dtype,
            device=self.simulator.device,
        )
        if yaw.ndim == 0:
            yaw = yaw.expand(self.batch_size)
        if yaw.shape != (self.batch_size,) or not bool(torch.isfinite(yaw).all()):
            raise ValueError(
                "Episode initial command yaw must be finite with one value per row."
            )
        self._initial_yaw_command = yaw.clone()

    def reset(self) -> torch.Tensor:
        self.state = clone_state_batch(self._initial_state, self.batch_size)
        self.control_index = 0
        self.yaw_command.copy_(self._initial_yaw_command)
        self.failed.zero_()
        self.episode_success.zero_()
        self.episode_endpoint_success.zero_()
        self.episode_scientific_success.zero_()
        self.episode_scientific_event.zero_()
        self.episode_first_entry_marker.zero_()
        self.episode_first_entry_physics_step.zero_()
        self.episode_first_entry_time_s.fill_(torch.nan)
        self.episode_first_entry_tip_distance.fill_(torch.nan)
        self.episode_first_entry_tip_speed.fill_(torch.nan)
        self.episode_first_entry_directed_speed.fill_(torch.nan)
        self.episode_first_entry_direction_error_deg.fill_(torch.nan)
        self.episode_reward.zero_()
        self.episode_maximum_displacement.zero_()
        self.episode_terminal_displacement.fill_(torch.nan)
        self.episode_peak_forward_displacement.zero_()
        self.episode_success_return_distance.fill_(torch.nan)
        self.episode_success_return_quality.fill_(torch.nan)
        self.episode_best_release_quality.zero_()
        self.episode_best_return_release_quality.zero_()
        self.episode_success_release_quality.fill_(torch.nan)
        self.episode_success_forward_displacement.fill_(torch.nan)
        self.episode_success_uav_forward_speed.fill_(torch.nan)
        self.episode_success_relative_tip_forward_speed.fill_(torch.nan)
        self.episode_maximum_uav_speed.zero_()
        self.episode_maximum_command_acceleration.zero_()
        initial_distance = torch.linalg.vector_norm(
            self.state.cable.positions_m[:, -1] - self._target, dim=-1
        )
        self.episode_minimum_tip_distance.copy_(initial_distance)
        initial_attachment = self.simulator.root_boundary.evaluate(
            self.state.uav,
            self.simulator.cable_configuration.rest_lengths_m[0],
        ).attachment_position_m
        self.episode_minimum_progress_distance.copy_(
            progress_shaping_distance(
                self.state.cable.positions_m[:, -1],
                initial_attachment,
                self._target,
                self._initial_attachment_position_m,
                mode=self.progress_shaping_reference,
                attachment_compensation_fraction=(
                    self.progress_attachment_compensation_fraction
                ),
            )
        )
        self.episode_best_strike_quality.zero_()
        self.episode_best_directed_speed_quality.zero_()
        self.episode_best_direction_quality.zero_()
        self.previous_normalized_action.zero_()
        self.episode_success_tip_distance.fill_(torch.nan)
        self.episode_success_tip_speed.fill_(torch.nan)
        self.episode_success_directed_speed.fill_(torch.nan)
        self.episode_success_direction_error_deg.fill_(torch.nan)
        self._record_state(0)
        observation, _ = self._observation()
        return observation

    def _decode_action(self, normalized_action: torch.Tensor, frame: object) -> tuple[torch.Tensor, torch.Tensor]:
        action = normalized_action.to(dtype=self.simulator.dtype, device=self.simulator.device)
        if self.action_mode == TARGET_ALIGNED_SAGITTAL_ACTION_MODE:
            acceleration_world, body_rate = decode_target_aligned_sagittal_action(
                action,
                self._direction,
                maximum_acceleration_m_s2=self.maximum_acceleration_m_s2,
                maximum_body_rate_rad_s=self.maximum_body_rate_rad_s,
            )
        else:
            local_acceleration = action[:, :3]
            norm = torch.linalg.vector_norm(local_acceleration, dim=-1, keepdim=True)
            local_acceleration = local_acceleration / torch.maximum(
                norm, torch.ones_like(norm)
            )
            local_acceleration = self.maximum_acceleration_m_s2 * local_acceleration
            acceleration_world = frame.vectors_to_world(local_acceleration)
            body_rate = self.maximum_body_rate_rad_s * action[:, 3:6]
        inactive = self.failed | (
            self.episode_success
            if self.terminate_on_success
            else torch.zeros_like(self.episode_success)
        )
        acceleration_world = torch.where(
            inactive[:, None], torch.zeros_like(acceleration_world), acceleration_world
        )
        body_rate = torch.where(
            inactive[:, None], torch.zeros_like(body_rate), body_rate
        )
        return acceleration_world, body_rate

    @torch.no_grad()
    def step(self, normalized_action: torch.Tensor) -> ControlStepResult:
        expected_shape = (self.batch_size, self.action_dimension)
        if normalized_action.shape != expected_shape:
            raise ValueError(
                f"Sequential whip action must have shape {expected_shape}."
            )
        observation_before, context = self._observation()
        del observation_before
        failed_before = self.failed.clone()
        success_before = self.episode_success.clone()
        inactive_before = failed_before | (
            success_before
            if self.terminate_on_success
            else torch.zeros_like(success_before)
        )
        acceleration_world, body_rate = self._decode_action(normalized_action, context.frame)
        normalized_action = normalized_action.to(
            dtype=self.simulator.dtype, device=self.simulator.device
        )
        previous_episode_minimum_progress_distance = (
            self.episode_minimum_progress_distance.clone()
        )
        previous_episode_best_strike = self.episode_best_strike_quality.clone()
        previous_episode_best_directed_speed = (
            self.episode_best_directed_speed_quality.clone()
        )
        previous_episode_best_direction = self.episode_best_direction_quality.clone()
        previous_episode_best_return_release = (
            self.episode_best_return_release_quality.clone()
        )
        previous_episode_maximum_displacement = (
            self.episode_maximum_displacement.clone()
        )
        self.yaw_command = torch.atan2(
            torch.sin(self.yaw_command + body_rate[:, 2] * self.control_dt_s),
            torch.cos(self.yaw_command + body_rate[:, 2] * self.control_dt_s),
        )
        orientation_command = torch.zeros(
            (self.batch_size, 4), dtype=self.simulator.dtype, device=self.simulator.device
        )
        orientation_command[:, 2] = torch.sin(0.5 * self.yaw_command)
        orientation_command[:, 3] = torch.cos(0.5 * self.yaw_command)

        initial_position = self._initial_state.uav.position_m
        previous_distance = torch.linalg.vector_norm(
            self.state.cable.positions_m[:, -1] - self._target, dim=-1
        )
        minimum_distance = previous_distance.clone()
        initial_attachment = self.simulator.root_boundary.evaluate(
            self.state.uav,
            self.simulator.cable_configuration.rest_lengths_m[0],
        ).attachment_position_m
        minimum_progress_distance = progress_shaping_distance(
            self.state.cable.positions_m[:, -1],
            initial_attachment,
            self._target,
            self._initial_attachment_position_m,
            mode=self.progress_shaping_reference,
            attachment_compensation_fraction=(
                self.progress_attachment_compensation_fraction
            ),
        )
        interval_maximum_displacement = torch.linalg.vector_norm(
            self.state.uav.position_m - initial_position, dim=-1
        )
        interval_maximum_uav_speed = torch.linalg.vector_norm(
            self.state.uav.velocity_m_s, dim=-1
        )
        interval_maximum_command_acceleration = torch.linalg.vector_norm(
            acceleration_world, dim=-1
        )
        interval_displacement_integral = torch.zeros_like(previous_distance)
        interval_uav_speed_integral = torch.zeros_like(previous_distance)
        best_proximity_speed = torch.full_like(previous_distance, -torch.inf)
        best_proximity_direction = torch.full_like(previous_distance, -torch.inf)
        interval_best_strike_quality = torch.zeros_like(previous_distance)
        interval_best_return_release_quality = torch.zeros_like(previous_distance)
        interval_best_directed_speed_quality = torch.zeros_like(previous_distance)
        interval_best_direction_quality = torch.zeros_like(previous_distance)
        interval_endpoint_success = torch.zeros_like(self.failed)
        interval_scientific_event = torch.zeros_like(self.failed)
        interval_non_tip_first = torch.zeros_like(self.failed)
        new_numerical_failure = torch.zeros_like(self.failed)
        interval_terminated = (
            success_before.clone()
            if self.terminate_on_success
            else torch.zeros_like(success_before)
        )

        for physics_substep in range(self.physics_steps_per_control):
            # Re-anchor p/v at the current measured state.  The action is direct
            # acceleration plus body rates, not position-trajectory tracking.
            active = ~(self.failed | interval_terminated)
            command = FullStateCommand(
                self.state.uav.position_m,
                self.state.uav.velocity_m_s,
                torch.where(
                    active[:, None], acceleration_world, torch.zeros_like(acceleration_world)
                ),
                orientation_command,
                torch.where(active[:, None], body_rate, torch.zeros_like(body_rate)),
            )
            physics_step = (
                self.control_index * self.physics_steps_per_control
                + physics_substep
                + 1
            )
            if self._command_recording is not None:
                record_index = physics_step - 1
                self._command_recording["position_m"][record_index].copy_(
                    command.position_m
                )
                self._command_recording["velocity_m_s"][record_index].copy_(
                    command.velocity_m_s
                )
                self._command_recording["acceleration_m_s2"][record_index].copy_(
                    command.acceleration_m_s2
                )
                self._command_recording["orientation_xyzw"][record_index].copy_(
                    command.orientation_xyzw
                )
                self._command_recording["angular_velocity_body_rad_s"][record_index].copy_(
                    command.angular_velocity_body_rad_s
                )
            state_before_propagation = self.state
            propagated = self.simulator._propagate(
                self.state, command, self.rollout_parameters, create_graph=False
            )
            if self.state_postprocessor is not None:
                propagated = self.state_postprocessor(physics_step, propagated)
            finite = _row_finite(propagated)
            newly_invalid = (~finite) & active
            new_numerical_failure |= newly_invalid
            self.failed |= newly_invalid
            propagated = _replace_rows(
                propagated, state_before_propagation, interval_terminated
            )
            self.state = _replace_rows(propagated, self._initial_state, self.failed)
            self._record_state(physics_step)

            tip_position = self.state.cable.positions_m[:, -1]
            tip_velocity = self.state.cable.velocities_m_s[:, -1]
            distance = torch.linalg.vector_norm(tip_position - self._target, dim=-1)
            speed = torch.linalg.vector_norm(tip_velocity, dim=-1)
            directed_speed = (tip_velocity * self._direction).sum(dim=-1)
            direction_cosine = directed_speed / speed.clamp_min(
                torch.finfo(speed.dtype).eps
            )
            physical_attachment = self.simulator.root_boundary.evaluate(
                self.state.uav,
                self.simulator.cable_configuration.rest_lengths_m[0],
            )
            physical_attachment_velocity = (
                physical_attachment.attachment_velocity_analytic_m_s
            )
            progress_distance = progress_shaping_distance(
                tip_position,
                physical_attachment.attachment_position_m,
                self._target,
                self._initial_attachment_position_m,
                mode=self.progress_shaping_reference,
                attachment_compensation_fraction=(
                    self.progress_attachment_compensation_fraction
                ),
            )
            attachment_velocity = (
                physical_attachment_velocity
                if self.directed_speed_shaping_reference
                == ATTACHMENT_RELATIVE_SPEED_SHAPING
                else torch.zeros_like(tip_velocity)
            )
            reward_tip_velocity = shaping_tip_velocity(
                tip_velocity,
                attachment_velocity,
                mode=self.directed_speed_shaping_reference,
            )
            reward_tip_speed = torch.linalg.vector_norm(
                reward_tip_velocity, dim=-1
            )
            reward_directed_speed = (
                reward_tip_velocity * self._direction
            ).sum(dim=-1)
            reward_direction_cosine = reward_directed_speed / reward_tip_speed.clamp_min(
                torch.finfo(reward_tip_speed.dtype).eps
            )
            direction_error = torch.rad2deg(
                torch.acos(direction_cosine.clamp(-1.0, 1.0))
            )
            proximity = torch.exp(-torch.square(distance / self.reward_weights.proximity_scale_m))
            eligible = ~(self.failed | interval_terminated)
            best_proximity_speed = torch.where(
                eligible,
                torch.maximum(
                    best_proximity_speed, proximity * reward_directed_speed
                ),
                best_proximity_speed,
            )
            best_proximity_direction = torch.where(
                eligible,
                torch.maximum(
                    best_proximity_direction,
                    proximity * reward_direction_cosine,
                ),
                best_proximity_direction,
            )
            minimum_distance = torch.where(
                eligible, torch.minimum(minimum_distance, distance), minimum_distance
            )
            minimum_progress_distance = torch.where(
                eligible,
                torch.minimum(minimum_progress_distance, progress_distance),
                minimum_progress_distance,
            )
            displacement = torch.linalg.vector_norm(
                self.state.uav.position_m - initial_position, dim=-1
            )
            forward_displacement = (
                (self.state.uav.position_m - initial_position) * self._direction
            ).sum(dim=-1)
            self.episode_peak_forward_displacement = torch.where(
                eligible,
                torch.maximum(
                    self.episode_peak_forward_displacement,
                    forward_displacement,
                ),
                self.episode_peak_forward_displacement,
            )
            uav_forward_speed = (
                self.state.uav.velocity_m_s * self._direction
            ).sum(dim=-1)
            relative_tip_forward_speed = (
                (tip_velocity - physical_attachment_velocity) * self._direction
            ).sum(dim=-1)
            release_quality = forward_release_quality(
                self.episode_peak_forward_displacement,
                uav_forward_speed,
                relative_tip_forward_speed,
                excursion_scale_m=self.reward_weights.forward_excursion_scale_m,
                backward_speed_scale_m_s=(
                    self.reward_weights.uav_backward_speed_scale_m_s
                ),
                tip_speed_scale_m_s=(
                    self.reward_weights.relative_tip_forward_speed_scale_m_s
                ),
            )
            self.episode_best_release_quality = torch.where(
                eligible,
                torch.maximum(
                    self.episode_best_release_quality, release_quality
                ),
                self.episode_best_release_quality,
            )
            uav_speed = torch.linalg.vector_norm(self.state.uav.velocity_m_s, dim=-1)
            interval_maximum_displacement = torch.where(
                eligible,
                torch.maximum(interval_maximum_displacement, displacement),
                interval_maximum_displacement,
            )
            interval_maximum_uav_speed = torch.where(
                eligible,
                torch.maximum(interval_maximum_uav_speed, uav_speed),
                interval_maximum_uav_speed,
            )
            interval_displacement_integral += torch.where(
                eligible,
                smooth_displacement_cost(
                    displacement,
                    scale_m=self.displacement_cost_scale_m,
                )
                * self.simulator.dt_s,
                torch.zeros_like(displacement),
            )
            interval_uav_speed_integral += torch.where(
                eligible,
                torch.log1p(
                    torch.square(
                        uav_speed / max(self.task.maximum_uav_speed_m_s, 1.0e-6)
                    )
                )
                * self.simulator.dt_s,
                torch.zeros_like(uav_speed),
            )
            current_strike_quality = strike_quality(
                distance,
                reward_directed_speed,
                reward_direction_cosine,
                proximity_scale_m=self.reward_weights.proximity_scale_m,
                target_directed_speed_m_s=(
                    self.task.minimum_directed_speed_m_s
                ),
                maximum_direction_error_deg=(
                    self.task.maximum_direction_error_deg
                ),
                speed_reward_cap_m_s=(
                    self.reward_weights.directed_speed_reward_cap_m_s
                ),
            )
            interval_best_strike_quality = torch.where(
                eligible,
                torch.maximum(
                    interval_best_strike_quality,
                    current_strike_quality,
                ),
                interval_best_strike_quality,
            )
            current_return_release_quality = return_release_strike_quality(
                self.episode_peak_forward_displacement,
                forward_displacement,
                uav_forward_speed,
                relative_tip_forward_speed,
                current_strike_quality,
                excursion_scale_m=self.reward_weights.forward_excursion_scale_m,
                backward_speed_scale_m_s=(
                    self.reward_weights.uav_backward_speed_scale_m_s
                ),
                tip_speed_scale_m_s=(
                    self.reward_weights.relative_tip_forward_speed_scale_m_s
                ),
            )
            interval_best_return_release_quality = torch.where(
                eligible,
                torch.maximum(
                    interval_best_return_release_quality,
                    current_return_release_quality,
                ),
                interval_best_return_release_quality,
            )
            soft_speed_quality, soft_direction_quality = (
                soft_near_target_strike_components(
                    distance,
                    reward_tip_speed,
                    reward_directed_speed,
                    reward_direction_cosine,
                    proximity_scale_m=self.reward_weights.proximity_scale_m,
                    target_directed_speed_m_s=(
                        self.task.minimum_directed_speed_m_s
                    ),
                    speed_reward_cap_m_s=(
                        self.reward_weights.directed_speed_reward_cap_m_s
                    ),
                    maximum_direction_error_deg=(
                        self.task.maximum_direction_error_deg
                    ),
                )
            )
            interval_best_directed_speed_quality = torch.where(
                eligible,
                torch.maximum(
                    interval_best_directed_speed_quality, soft_speed_quality
                ),
                interval_best_directed_speed_quality,
            )
            interval_best_direction_quality = torch.where(
                eligible,
                torch.maximum(interval_best_direction_quality, soft_direction_quality),
                interval_best_direction_quality,
            )
            point_endpoint_success = eligible & simple_endpoint_success(
                tip_position,
                tip_velocity,
                self._target,
                self._direction,
                radius_m=self.task.success_radius_m,
                minimum_directed_speed_m_s=self.task.minimum_directed_speed_m_s,
                maximum_direction_error_deg=self.task.maximum_direction_error_deg,
            )
            marker_positions = self.state.cable.positions_m[:, 2:12]
            marker_distance = torch.linalg.vector_norm(
                marker_positions - self._target[:, None, :], dim=-1
            )
            marker_ids = torch.arange(
                1, 11, dtype=torch.int64, device=self.simulator.device
            )[None, :]
            inside = marker_distance <= self.task.success_radius_m
            candidate_marker = torch.where(
                inside, marker_ids, torch.full_like(marker_ids, 11)
            ).amin(dim=1)
            new_entry = (
                eligible
                & (self.episode_first_entry_marker == 0)
                & (candidate_marker <= 10)
            )
            entry_step = torch.full_like(
                self.episode_first_entry_physics_step, physics_step
            )
            self.episode_first_entry_marker = torch.where(
                new_entry, candidate_marker, self.episode_first_entry_marker
            )
            self.episode_first_entry_physics_step = torch.where(
                new_entry, entry_step, self.episode_first_entry_physics_step
            )
            entry_time = entry_step.to(self.simulator.dtype) * self.simulator.dt_s
            self.episode_first_entry_time_s = torch.where(
                new_entry, entry_time, self.episode_first_entry_time_s
            )
            self.episode_first_entry_tip_distance = torch.where(
                new_entry, distance, self.episode_first_entry_tip_distance
            )
            self.episode_first_entry_tip_speed = torch.where(
                new_entry, speed, self.episode_first_entry_tip_speed
            )
            self.episode_first_entry_directed_speed = torch.where(
                new_entry, directed_speed, self.episode_first_entry_directed_speed
            )
            self.episode_first_entry_direction_error_deg = torch.where(
                new_entry,
                direction_error,
                self.episode_first_entry_direction_error_deg,
            )
            interval_non_tip_first |= new_entry & (candidate_marker != 10)
            point_task_event = (
                point_endpoint_success
                & new_entry
                & (candidate_marker == 10)
            )
            first_reported_at_point = (
                point_endpoint_success
                & (~self.episode_endpoint_success)
                & (~interval_endpoint_success)
            )
            if self.success_mode == "task_whip_once":
                first_reported_at_point = point_task_event
            self.episode_success_tip_distance = torch.where(
                first_reported_at_point, distance, self.episode_success_tip_distance
            )
            self.episode_success_tip_speed = torch.where(
                first_reported_at_point, speed, self.episode_success_tip_speed
            )
            self.episode_success_directed_speed = torch.where(
                first_reported_at_point,
                directed_speed,
                self.episode_success_directed_speed,
            )
            self.episode_success_direction_error_deg = torch.where(
                first_reported_at_point,
                direction_error,
                self.episode_success_direction_error_deg,
            )
            interval_endpoint_success |= point_endpoint_success
            interval_scientific_event |= point_task_event
            if self.terminate_on_success:
                terminal_event = (
                    point_task_event
                    if self.success_mode == "task_whip_once"
                    else point_endpoint_success
                )
                interval_terminated |= terminal_event

        current_distance = torch.linalg.vector_norm(
            self.state.cable.positions_m[:, -1] - self._target, dim=-1
        )
        best_proximity_speed = torch.where(
            torch.isfinite(best_proximity_speed), best_proximity_speed, torch.zeros_like(best_proximity_speed)
        )
        best_proximity_direction = torch.where(
            torch.isfinite(best_proximity_direction),
            best_proximity_direction,
            torch.zeros_like(best_proximity_direction),
        )
        self.episode_endpoint_success |= interval_endpoint_success
        self.episode_scientific_event |= interval_scientific_event
        self.episode_maximum_displacement = torch.maximum(
            self.episode_maximum_displacement, interval_maximum_displacement
        )
        self.episode_maximum_uav_speed = torch.maximum(
            self.episode_maximum_uav_speed, interval_maximum_uav_speed
        )
        self.episode_maximum_command_acceleration = torch.maximum(
            self.episode_maximum_command_acceleration,
            interval_maximum_command_acceleration,
        )
        self.episode_best_strike_quality = torch.maximum(
            self.episode_best_strike_quality, interval_best_strike_quality
        )
        self.episode_best_return_release_quality = torch.maximum(
            self.episode_best_return_release_quality,
            interval_best_return_release_quality,
        )
        self.episode_best_directed_speed_quality = torch.maximum(
            self.episode_best_directed_speed_quality,
            interval_best_directed_speed_quality,
        )
        self.episode_best_direction_quality = torch.maximum(
            self.episode_best_direction_quality, interval_best_direction_quality
        )
        self.episode_minimum_tip_distance = torch.minimum(
            self.episode_minimum_tip_distance, minimum_distance
        )
        self.episode_minimum_progress_distance = torch.minimum(
            self.episode_minimum_progress_distance,
            minimum_progress_distance,
        )
        horizon_done = self.control_index + 1 >= self.control_step_count
        pre_success_time_increment = (
            (~self.episode_success).to(previous_distance.dtype) * self.control_dt_s
        )
        if self.success_mode == "task_whip_once":
            task_success = task_whip_success(
                endpoint_event_found=self.episode_scientific_event,
                first_entry_marker=self.episode_first_entry_marker,
                finite=~self.failed,
            )
            newly_successful = task_success & (~self.episode_success)
            self.episode_success |= task_success
            self.episode_success &= ~self.failed
            scientific_success = diagnostic_scientific_whip_success(
                endpoint_event_found=task_success,
                first_entry_marker=self.episode_first_entry_marker,
                maximum_uav_displacement_m=self.episode_maximum_displacement,
                maximum_uav_speed_m_s=self.episode_maximum_uav_speed,
                maximum_command_acceleration_m_s2=self.episode_maximum_command_acceleration,
                finite=~self.failed,
                maximum_uav_displacement_limit_m=self.task.maximum_uav_displacement_m,
                maximum_uav_speed_limit_m_s=self.task.maximum_uav_speed_m_s,
                maximum_command_acceleration_limit_m_s2=(
                    self.task.maximum_command_acceleration_m_s2
                ),
            )
            record_scientific_result = torch.full_like(
                scientific_success, horizon_done
            )
            if self.terminate_on_success:
                record_scientific_result |= newly_successful
            self.episode_scientific_success = torch.where(
                record_scientific_result,
                scientific_success,
                self.episode_scientific_success,
            )
        else:
            newly_successful = interval_endpoint_success & (~self.episode_success)
            self.episode_success |= interval_endpoint_success
        if self.reward_mode == "legacy_dense":
            reward = simple_dense_reward(
                previous_distance,
                current_distance,
                best_proximity_speed,
                best_proximity_direction,
                interval_maximum_displacement,
                newly_successful,
                new_numerical_failure,
                self.reward_weights,
            )
        else:
            initial_distance = torch.linalg.vector_norm(
                self._initial_state.cable.positions_m[:, -1] - self._target, dim=-1
            ).clamp_min(1.0e-6)
            progress_improvement = (
                previous_episode_minimum_progress_distance
                - self.episode_minimum_progress_distance
            ).clamp_min(0.0) / initial_distance
            strike_improvement = (
                self.episode_best_strike_quality - previous_episode_best_strike
            ).clamp_min(0.0)
            directed_speed_improvement = (
                self.episode_best_directed_speed_quality
                - previous_episode_best_directed_speed
            ).clamp_min(0.0)
            direction_improvement = (
                self.episode_best_direction_quality - previous_episode_best_direction
            ).clamp_min(0.0)
            return_release_improvement = (
                self.episode_best_return_release_quality
                - previous_episode_best_return_release
            ).clamp_min(0.0)
            maximum_displacement_increase = (
                smooth_displacement_cost(
                    self.episode_maximum_displacement,
                    scale_m=self.displacement_cost_scale_m,
                )
                - smooth_displacement_cost(
                    previous_episode_maximum_displacement,
                    scale_m=self.displacement_cost_scale_m,
                )
            ).clamp_min(0.0)
            terminal_now = torch.full_like(newly_successful, horizon_done)
            if self.terminate_on_success:
                terminal_now |= newly_successful
            terminal_displacement = torch.linalg.vector_norm(
                self.state.uav.position_m - initial_position, dim=-1
            )
            terminal_forward_displacement = (
                (self.state.uav.position_m - initial_position) * self._direction
            ).sum(dim=-1)
            terminal_return_distance = (
                self.episode_peak_forward_displacement
                - terminal_forward_displacement
            ).clamp_min(0.0)
            terminal_return_quality = forward_return_quality(
                self.episode_peak_forward_displacement,
                terminal_forward_displacement,
                excursion_scale_m=self.reward_weights.forward_excursion_scale_m,
            )
            terminal_release_quality = forward_release_quality(
                self.episode_peak_forward_displacement,
                uav_forward_speed,
                relative_tip_forward_speed,
                excursion_scale_m=self.reward_weights.forward_excursion_scale_m,
                backward_speed_scale_m_s=(
                    self.reward_weights.uav_backward_speed_scale_m_s
                ),
                tip_speed_scale_m_s=(
                    self.reward_weights.relative_tip_forward_speed_scale_m_s
                ),
            )
            credited_release_quality = (
                terminal_release_quality
                if self.reward_weights.success_release_at_strike
                else self.episode_best_release_quality
            )
            self.episode_success_return_distance = torch.where(
                newly_successful,
                terminal_return_distance,
                self.episode_success_return_distance,
            )
            self.episode_success_return_quality = torch.where(
                newly_successful,
                terminal_return_quality,
                self.episode_success_return_quality,
            )
            self.episode_success_release_quality = torch.where(
                newly_successful,
                credited_release_quality,
                self.episode_success_release_quality,
            )
            self.episode_success_forward_displacement = torch.where(
                newly_successful,
                terminal_forward_displacement,
                self.episode_success_forward_displacement,
            )
            self.episode_success_uav_forward_speed = torch.where(
                newly_successful,
                uav_forward_speed,
                self.episode_success_uav_forward_speed,
            )
            self.episode_success_relative_tip_forward_speed = torch.where(
                newly_successful,
                relative_tip_forward_speed,
                self.episode_success_relative_tip_forward_speed,
            )
            terminal_displacement_cost = smooth_displacement_cost(
                terminal_displacement,
                scale_m=self.displacement_cost_scale_m,
            ) * terminal_displacement_charge_mask(
                terminal_now,
                newly_successful,
                success_only=(
                    self.reward_weights.terminal_displacement_success_only
                ),
            ).to(previous_distance.dtype)
            first_terminal = (
                terminal_now
                & torch.isnan(self.episode_terminal_displacement)
                & (~inactive_before)
            )
            self.episode_terminal_displacement = torch.where(
                first_terminal,
                terminal_displacement,
                self.episode_terminal_displacement,
            )
            acceleration_effort = (
                torch.linalg.vector_norm(acceleration_world, dim=-1)
                / max(self.maximum_acceleration_m_s2, 1.0e-6)
            ).square() * self.control_dt_s
            body_rate_effort = (
                torch.linalg.vector_norm(body_rate, dim=-1)
                / max(
                    self.maximum_body_rate_rad_s
                    * math.sqrt(float(self.body_rate_effort_axis_count)),
                    1.0e-6,
                )
            ).square() * self.control_dt_s
            smoothness = torch.mean(
                (normalized_action - self.previous_normalized_action).square(), dim=-1
            )
            reward = (
                self.reward_weights.progress * progress_improvement
                + self.reward_weights.directed_speed_near_target
                * directed_speed_improvement
                + self.reward_weights.direction_near_target * direction_improvement
                + self.reward_weights.strike_quality_improvement * strike_improvement
                + self.reward_weights.return_release_improvement
                * return_release_improvement
                + self.reward_weights.success_bonus
                * newly_successful.to(previous_distance.dtype)
                + self.reward_weights.success_compactness_bonus
                * smooth_success_compactness(
                    self.episode_maximum_displacement,
                    scale_m=self.reward_weights.success_compactness_scale_m,
                )
                * newly_successful.to(previous_distance.dtype)
                + self.reward_weights.success_forward_return_bonus
                * terminal_return_quality
                * newly_successful.to(previous_distance.dtype)
                + self.reward_weights.success_release_bonus
                * credited_release_quality
                * newly_successful.to(previous_distance.dtype)
                - self.reward_weights.non_tip_first
                * interval_non_tip_first.to(previous_distance.dtype)
                - self.reward_weights.maximum_displacement
                * maximum_displacement_increase
                - self.reward_weights.terminal_displacement
                * terminal_displacement_cost
                - self.reward_weights.displacement_integral
                * interval_displacement_integral
                - self.reward_weights.uav_speed_integral
                * interval_uav_speed_integral
                - self.reward_weights.acceleration_effort * acceleration_effort
                - self.reward_weights.body_rate_effort * body_rate_effort
                - self.reward_weights.action_smoothness * smoothness
                - self.reward_weights.time_to_success * pre_success_time_increment
                - self.reward_weights.numerical_failure
                * new_numerical_failure.to(previous_distance.dtype)
            )
        reward = torch.where(inactive_before, torch.zeros_like(reward), reward)
        self.episode_reward += reward
        self.previous_normalized_action.copy_(normalized_action)
        self.control_index += 1
        horizon_done = self.control_index >= self.control_step_count
        success_terminal = (
            self.episode_success
            if self.terminate_on_success
            else torch.zeros_like(self.episode_success)
        )
        done = (
            new_numerical_failure
            | success_terminal
            | torch.full_like(self.failed, horizon_done)
        )
        next_observation, _ = self._observation()
        return ControlStepResult(
            next_observation=next_observation,
            reward=reward[:, None].float(),
            done=done[:, None].float(),
            include_transition=~inactive_before,
            newly_successful=newly_successful,
            final_tip_distance_m=current_distance,
            minimum_tip_distance_m=minimum_distance,
            maximum_uav_displacement_m=interval_maximum_displacement,
            numerical_failure=new_numerical_failure,
        )
