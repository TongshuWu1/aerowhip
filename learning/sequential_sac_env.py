"""Ten-second sequential whip environment shared by the SAC/PPO baselines."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from planning.rollout import clone_state_batch
from planning.task import CanonicalWhipTask
from simulator.cable.dder import DderState
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.state import FullStateCommand, ResidualHistoryState, UAVState

from .normalization import FixedContextNormalizer
from .policy_context import POLICY_CONTEXT_DIM, build_policy_context


SEQUENTIAL_WHIP_OBSERVATION_DIM = POLICY_CONTEXT_DIM + 1
# Compatibility for the retired SAC runner and its historical checkpoints.
SEQUENTIAL_SAC_OBSERVATION_DIM = SEQUENTIAL_WHIP_OBSERVATION_DIM


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
    displacement_integral: float = 0.0
    uav_speed_integral: float = 0.0
    acceleration_effort: float = 0.0
    body_rate_effort: float = 0.0
    action_smoothness: float = 0.0


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
) -> torch.Tensor:
    """Bounded soft conjunction of proximity, speed, and strike alignment."""

    proximity = torch.exp(-0.5 * torch.square(distance_m / proximity_scale_m))
    speed = torch.sigmoid(
        (directed_speed_m_s - target_directed_speed_m_s)
        / max(0.25 * target_directed_speed_m_s, 1.0e-6)
    )
    # The former (cosine + 1) / 2 map still awarded 64.6% direction credit at
    # 73 degrees.  Centering a smooth score on the scientific direction limit
    # retains gradients but makes a fast sideways target entry unattractive.
    direction_threshold = math.cos(math.radians(maximum_direction_error_deg))
    direction_width = 0.15
    direction = torch.sigmoid(
        (direction_cosine - direction_threshold) / direction_width
    )
    perfect_alignment = torch.sigmoid(
        torch.as_tensor(
            (1.0 - direction_threshold) / direction_width,
            dtype=direction.dtype,
            device=direction.device,
        )
    )
    direction = (direction / perfect_alignment).clamp(0.0, 1.0)
    return proximity * speed * direction


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
        success_mode: str = "simple_endpoint",
        reward_mode: str = "legacy_dense",
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
        if success_mode not in {"simple_endpoint", "task_whip_once"}:
            raise ValueError(f"Unsupported sequential-whip success mode: {success_mode}")
        if reward_mode not in {"legacy_dense", "whip_potential"}:
            raise ValueError(f"Unsupported sequential-whip reward mode: {reward_mode}")
        self.success_mode = success_mode
        self.reward_mode = reward_mode
        self._initial_state = clone_state_batch(initial_state, batch_size)
        self._target = torch.tensor(
            task.target_position_m, dtype=simulator.dtype, device=simulator.device
        ).view(1, 3).expand(batch_size, -1)
        self._direction = torch.tensor(
            task.desired_direction,
            dtype=simulator.dtype,
            device=simulator.device,
        ).view(1, 3).expand(batch_size, -1)
        self.state = self._initial_state
        self.control_index = 0
        self.yaw_command = torch.full(
            (batch_size,), task.initial_yaw_rad, dtype=simulator.dtype, device=simulator.device
        )
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
        self.episode_maximum_uav_speed = torch.zeros_like(self.episode_reward)
        self.episode_maximum_command_acceleration = torch.zeros_like(self.episode_reward)
        self.episode_minimum_tip_distance = torch.full_like(self.episode_reward, torch.inf)
        self.episode_best_strike_quality = torch.zeros_like(self.episode_reward)
        self.previous_normalized_action = torch.zeros(
            (batch_size, 6), dtype=simulator.dtype, device=simulator.device
        )
        self.episode_success_tip_distance = torch.full_like(self.episode_reward, torch.nan)
        self.episode_success_tip_speed = torch.full_like(self.episode_reward, torch.nan)
        self.episode_success_directed_speed = torch.full_like(self.episode_reward, torch.nan)
        self.episode_success_direction_error_deg = torch.full_like(
            self.episode_reward, torch.nan
        )

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

    def reset(self) -> torch.Tensor:
        self.state = clone_state_batch(self._initial_state, self.batch_size)
        self.control_index = 0
        self.yaw_command.fill_(self.task.initial_yaw_rad)
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
        self.episode_maximum_uav_speed.zero_()
        self.episode_maximum_command_acceleration.zero_()
        initial_distance = torch.linalg.vector_norm(
            self.state.cable.positions_m[:, -1] - self._target, dim=-1
        )
        self.episode_minimum_tip_distance.copy_(initial_distance)
        self.episode_best_strike_quality.zero_()
        self.previous_normalized_action.zero_()
        self.episode_success_tip_distance.fill_(torch.nan)
        self.episode_success_tip_speed.fill_(torch.nan)
        self.episode_success_directed_speed.fill_(torch.nan)
        self.episode_success_direction_error_deg.fill_(torch.nan)
        observation, _ = self._observation()
        return observation

    def _decode_action(self, normalized_action: torch.Tensor, frame: object) -> tuple[torch.Tensor, torch.Tensor]:
        action = normalized_action.to(dtype=self.simulator.dtype, device=self.simulator.device)
        local_acceleration = action[:, :3]
        norm = torch.linalg.vector_norm(local_acceleration, dim=-1, keepdim=True)
        local_acceleration = local_acceleration / torch.maximum(norm, torch.ones_like(norm))
        local_acceleration = self.maximum_acceleration_m_s2 * local_acceleration
        acceleration_world = frame.vectors_to_world(local_acceleration)
        body_rate = self.maximum_body_rate_rad_s * action[:, 3:6]
        acceleration_world = torch.where(self.failed[:, None], torch.zeros_like(acceleration_world), acceleration_world)
        body_rate = torch.where(self.failed[:, None], torch.zeros_like(body_rate), body_rate)
        return acceleration_world, body_rate

    @torch.no_grad()
    def step(self, normalized_action: torch.Tensor) -> ControlStepResult:
        if normalized_action.shape != (self.batch_size, 6):
            raise ValueError("Sequential SAC action must have shape Bx6.")
        observation_before, context = self._observation()
        del observation_before
        failed_before = self.failed.clone()
        acceleration_world, body_rate = self._decode_action(normalized_action, context.frame)
        normalized_action = normalized_action.to(
            dtype=self.simulator.dtype, device=self.simulator.device
        )
        previous_episode_minimum_distance = self.episode_minimum_tip_distance.clone()
        previous_episode_best_strike = self.episode_best_strike_quality.clone()
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
        interval_endpoint_success = torch.zeros_like(self.failed)
        interval_scientific_event = torch.zeros_like(self.failed)
        interval_non_tip_first = torch.zeros_like(self.failed)
        new_numerical_failure = torch.zeros_like(self.failed)

        for physics_substep in range(self.physics_steps_per_control):
            # Re-anchor p/v at the current measured state.  The action is direct
            # acceleration plus body rates, not position-trajectory tracking.
            command = FullStateCommand(
                self.state.uav.position_m,
                self.state.uav.velocity_m_s,
                acceleration_world,
                orientation_command,
                body_rate,
            )
            propagated = self.simulator._propagate(
                self.state, command, self.simulator.parameters, create_graph=False
            )
            finite = _row_finite(propagated)
            newly_invalid = (~finite) & (~self.failed)
            new_numerical_failure |= newly_invalid
            self.failed |= ~finite
            self.state = _replace_rows(propagated, self._initial_state, self.failed)

            tip_position = self.state.cable.positions_m[:, -1]
            tip_velocity = self.state.cable.velocities_m_s[:, -1]
            distance = torch.linalg.vector_norm(tip_position - self._target, dim=-1)
            speed = torch.linalg.vector_norm(tip_velocity, dim=-1)
            directed_speed = (tip_velocity * self._direction).sum(dim=-1)
            direction_cosine = directed_speed / speed.clamp_min(
                torch.finfo(speed.dtype).eps
            )
            direction_error = torch.rad2deg(
                torch.acos(direction_cosine.clamp(-1.0, 1.0))
            )
            proximity = torch.exp(-torch.square(distance / self.reward_weights.proximity_scale_m))
            eligible = ~self.failed
            best_proximity_speed = torch.where(
                eligible,
                torch.maximum(best_proximity_speed, proximity * directed_speed),
                best_proximity_speed,
            )
            best_proximity_direction = torch.where(
                eligible,
                torch.maximum(best_proximity_direction, proximity * direction_cosine),
                best_proximity_direction,
            )
            minimum_distance = torch.where(
                eligible, torch.minimum(minimum_distance, distance), minimum_distance
            )
            displacement = torch.linalg.vector_norm(
                self.state.uav.position_m - initial_position, dim=-1
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
                torch.log1p(
                    torch.square(
                        displacement
                        / max(self.task.maximum_uav_displacement_m, 1.0e-6)
                    )
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
            interval_best_strike_quality = torch.where(
                eligible,
                torch.maximum(
                    interval_best_strike_quality,
                    strike_quality(
                        distance,
                        directed_speed,
                        direction_cosine,
                        proximity_scale_m=self.reward_weights.proximity_scale_m,
                        target_directed_speed_m_s=(
                            self.task.minimum_directed_speed_m_s
                        ),
                        maximum_direction_error_deg=(
                            self.task.maximum_direction_error_deg
                        ),
                    ),
                ),
                interval_best_strike_quality,
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
            physics_step = (
                self.control_index * self.physics_steps_per_control
                + physics_substep
                + 1
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
        self.episode_minimum_tip_distance = torch.minimum(
            self.episode_minimum_tip_distance, minimum_distance
        )
        horizon_done = self.control_index + 1 >= self.control_step_count
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
            self.episode_scientific_success = torch.where(
                torch.full_like(scientific_success, horizon_done),
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
                previous_episode_minimum_distance
                - self.episode_minimum_tip_distance
            ).clamp_min(0.0) / initial_distance
            strike_improvement = (
                self.episode_best_strike_quality - previous_episode_best_strike
            ).clamp_min(0.0)
            maximum_displacement_increase = (
                torch.log1p(
                    torch.square(
                        self.episode_maximum_displacement
                        / max(self.task.maximum_uav_displacement_m, 1.0e-6)
                    )
                )
                - torch.log1p(
                    torch.square(
                        previous_episode_maximum_displacement
                        / max(self.task.maximum_uav_displacement_m, 1.0e-6)
                    )
                )
            ).clamp_min(0.0)
            acceleration_effort = (
                torch.linalg.vector_norm(acceleration_world, dim=-1)
                / max(self.maximum_acceleration_m_s2, 1.0e-6)
            ).square() * self.control_dt_s
            body_rate_effort = (
                torch.linalg.vector_norm(body_rate, dim=-1)
                / max(self.maximum_body_rate_rad_s * math.sqrt(3.0), 1.0e-6)
            ).square() * self.control_dt_s
            smoothness = torch.mean(
                (normalized_action - self.previous_normalized_action).square(), dim=-1
            )
            reward = (
                self.reward_weights.progress * progress_improvement
                + self.reward_weights.strike_quality_improvement * strike_improvement
                + self.reward_weights.success_bonus
                * newly_successful.to(previous_distance.dtype)
                - self.reward_weights.non_tip_first
                * interval_non_tip_first.to(previous_distance.dtype)
                - self.reward_weights.maximum_displacement
                * maximum_displacement_increase
                - self.reward_weights.displacement_integral
                * interval_displacement_integral
                - self.reward_weights.uav_speed_integral
                * interval_uav_speed_integral
                - self.reward_weights.acceleration_effort * acceleration_effort
                - self.reward_weights.body_rate_effort * body_rate_effort
                - self.reward_weights.action_smoothness * smoothness
                - self.reward_weights.numerical_failure
                * new_numerical_failure.to(previous_distance.dtype)
            )
        reward = torch.where(failed_before, torch.zeros_like(reward), reward)
        self.episode_reward += reward
        self.previous_normalized_action.copy_(normalized_action)
        self.control_index += 1
        horizon_done = self.control_index >= self.control_step_count
        done = new_numerical_failure | torch.full_like(self.failed, horizon_done)
        next_observation, _ = self._observation()
        return ControlStepResult(
            next_observation=next_observation,
            reward=reward[:, None].float(),
            done=done[:, None].float(),
            include_transition=~failed_before,
            newly_successful=newly_successful,
            final_tip_distance_m=current_distance,
            minimum_tip_distance_m=minimum_distance,
            maximum_uav_displacement_m=interval_maximum_displacement,
            numerical_failure=new_numerical_failure,
        )
