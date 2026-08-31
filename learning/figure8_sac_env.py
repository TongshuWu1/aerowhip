"""Sequential Figure-8 tracking environment for the minimal SAC baseline.

The environment is deliberately separate from the whip environment.  It uses
the same six-dimensional direct acceleration/body-rate action and the same
production simulator, but ports the geometric objective from the legacy
online Figure-8 controller into an additive MDP reward.
"""

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


FIGURE8_SAC_OBSERVATION_DIM = POLICY_CONTEXT_DIM + 1


def build_analytic_figure8_normalizer(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    reference: "Figure8Reference",
    *,
    command_yaw_world_rad: float,
) -> FixedContextNormalizer:
    """Deterministic task-local scales; no rollout/test data are fitted.

    The previous whip normalizer compresses a tip-height Figure-8 goal to the
    observation clip because its target-z support was centered near the whip
    strike plane.  Centering on the settled Figure-8 state avoids that unrelated
    distribution mismatch while retaining the exact 83-D context schema.
    """

    batch = initial_state.uav.batch_size
    progress = torch.zeros(batch, dtype=simulator.dtype, device=simulator.device)
    context = build_policy_context(
        simulator,
        initial_state,
        target_position_world_m=reference.point(progress),
        desired_direction_world=reference.tangent(progress),
        command_yaw_world_rad=command_yaw_world_rad,
    )
    mean = context.to_tensor()[0].detach().float().cpu()
    scale = torch.ones(POLICY_CONTEXT_DIM, dtype=torch.float32)
    scale[0:3] = 1.0       # UAV velocity, m/s
    scale[3:7] = 0.5       # local quaternion deviation
    scale[7:10] = 2.0      # UAV angular velocity, rad/s
    scale[10:40] = 0.5     # ordered cable positions, m
    scale[40:70] = 2.0     # ordered cable velocities, m/s
    scale[70:73] = 0.7     # local Figure-8 reference point, m
    scale[73:76] = 1.0     # unit tangent
    scale[76:83] = 1.0     # nominal theta is centered and constant
    return FixedContextNormalizer(mean, scale, sample_count=1)


@dataclass(frozen=True, slots=True)
class Figure8Reference:
    """Horizontal Gerono Figure-8 centered on the settled cable tip."""

    center_position_m: tuple[float, float, float]
    amplitude_x_m: float = 0.7
    amplitude_y_m: float = 0.5

    def __post_init__(self) -> None:
        if len(self.center_position_m) != 3:
            raise ValueError("Figure-8 center must have three coordinates.")
        values = (*self.center_position_m, self.amplitude_x_m, self.amplitude_y_m)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("Figure-8 reference values must be finite.")
        if self.amplitude_x_m <= 0.0 or self.amplitude_y_m <= 0.0:
            raise ValueError("Figure-8 amplitudes must be positive.")

    def point(self, progress_cycles: torch.Tensor) -> torch.Tensor:
        center = torch.as_tensor(
            self.center_position_m,
            dtype=progress_cycles.dtype,
            device=progress_cycles.device,
        )
        phase = 2.0 * math.pi * progress_cycles
        return torch.stack(
            (
                center[0] + self.amplitude_x_m * torch.sin(phase),
                center[1] + self.amplitude_y_m * torch.sin(2.0 * phase),
                torch.full_like(progress_cycles, center[2]),
            ),
            dim=-1,
        )

    def tangent(self, progress_cycles: torch.Tensor) -> torch.Tensor:
        phase = 2.0 * math.pi * progress_cycles
        derivative = torch.stack(
            (
                2.0 * math.pi * self.amplitude_x_m * torch.cos(phase),
                4.0 * math.pi * self.amplitude_y_m * torch.cos(2.0 * phase),
                torch.zeros_like(progress_cycles),
            ),
            dim=-1,
        )
        return derivative / torch.linalg.vector_norm(
            derivative, dim=-1, keepdim=True
        ).clamp_min(1.0e-9)

    @property
    def approximate_length_m(self) -> float:
        progress = torch.linspace(0.0, 1.0, 1441, dtype=torch.float64)
        points = self.point(progress)
        return float(torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum())


@dataclass(frozen=True, slots=True)
class LegacyFigure8RewardWeights:
    """Exact active coefficients from the legacy online Figure-8 objective."""

    tracking: float = 5000.0
    path_progress: float = 5.0
    control_effort: float = 2.0e-4
    control_smoothness: float = 5.0e-2
    tip_motion_smoothness: float = 5.0e-1
    speed_limit: float = 80.0
    ground_safety: float = 100.0
    ground_clearance_m: float = 0.02
    numerical_failure: float = 1000.0


@dataclass(frozen=True, slots=True)
class Figure8ControlStepResult:
    next_observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    include_transition: torch.Tensor
    final_tracking_error_m: torch.Tensor
    path_progress_cycles: torch.Tensor
    maximum_uav_displacement_m: torch.Tensor
    numerical_failure: torch.Tensor


def figure8_interval_reward(
    tracking_mean_squared_error_m2: torch.Tensor,
    progress_delta_m: torch.Tensor,
    tip_acceleration_mean_squared_m_s4: torch.Tensor,
    acceleration_effort_m2_s4: torch.Tensor,
    acceleration_change_m2_s4: torch.Tensor,
    speed_violation_squared: torch.Tensor,
    ground_violation_squared: torch.Tensor,
    numerical_failure: torch.Tensor,
    *,
    acceleration_scale_m_s2: float,
    weights: LegacyFigure8RewardWeights,
) -> torch.Tensor:
    """Negative legacy cost, expressed as an additive SAC reward."""

    cost = (
        weights.tracking * tracking_mean_squared_error_m2
        - weights.path_progress * progress_delta_m
        + weights.tip_motion_smoothness
        * tip_acceleration_mean_squared_m_s4
        / (acceleration_scale_m_s2**2)
        + weights.control_effort * acceleration_effort_m2_s4
        + weights.control_smoothness * acceleration_change_m2_s4
        + weights.speed_limit * speed_violation_squared
        + weights.ground_safety * ground_violation_squared
        + weights.numerical_failure * numerical_failure.to(tracking_mean_squared_error_m2.dtype)
    )
    return -cost


def figure8_episode_success(
    progress_cycles: torch.Tensor,
    tracking_rmse_m: torch.Tensor,
    observed_maximum_uav_speed_m_s: torch.Tensor,
    observed_minimum_scene_height_m: torch.Tensor,
    numerical_failure: torch.Tensor,
    *,
    required_cycles: float = 1.0,
    maximum_tracking_rmse_m: float = 0.10,
    maximum_uav_speed_m_s: float = 3.0,
    minimum_scene_height_m: float = 0.02,
) -> torch.Tensor:
    """Transparent reporting gate; it does not add a success bonus to reward."""

    return (
        (progress_cycles >= required_cycles)
        & (tracking_rmse_m <= maximum_tracking_rmse_m)
        & (observed_maximum_uav_speed_m_s <= maximum_uav_speed_m_s)
        & (observed_minimum_scene_height_m >= minimum_scene_height_m)
        & (~numerical_failure)
        & torch.isfinite(tracking_rmse_m)
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
    finite = torch.ones(
        state.uav.batch_size, dtype=torch.bool, device=state.uav.position_m.device
    )
    for tensor in tensors:
        finite &= torch.isfinite(tensor).reshape(tensor.shape[0], -1).all(dim=1)
    return finite


def _replace_rows(
    state: SimulatorState, source: SimulatorState, replace: torch.Tensor
) -> SimulatorState:
    """Quarantine a numerically failed row while the rest of the batch continues."""

    def merged(
        value: torch.Tensor | None, fallback: torch.Tensor | None
    ) -> torch.Tensor | None:
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


class SequentialFigure8Environment:
    """Batched 10-second Figure-8 MDP using unmodified production physics."""

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
        amplitude_x_m: float = 0.7,
        amplitude_y_m: float = 0.5,
        maximum_acceleration_m_s2: float = 10.0,
        maximum_body_rate_rad_s: float = 4.0,
        observation_clip: float = 10.0,
        reward_weights: LegacyFigure8RewardWeights = LegacyFigure8RewardWeights(),
        success_required_cycles: float = 1.0,
        success_maximum_tracking_rmse_m: float = 0.10,
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
        self.success_required_cycles = float(success_required_cycles)
        self.success_maximum_tracking_rmse_m = float(
            success_maximum_tracking_rmse_m
        )
        self._initial_state = clone_state_batch(initial_state, batch_size)
        center = tuple(
            float(value)
            for value in initial_state.cable.positions_m[0, -1].detach().cpu().tolist()
        )
        self.reference = Figure8Reference(center, amplitude_x_m, amplitude_y_m)
        self.path_length_m = self.reference.approximate_length_m
        self.state = self._initial_state
        self.control_index = 0
        self.yaw_command = torch.full(
            (batch_size,),
            task.initial_yaw_rad,
            dtype=simulator.dtype,
            device=simulator.device,
        )
        self.failed = torch.zeros(batch_size, dtype=torch.bool, device=simulator.device)
        self.episode_success = torch.zeros_like(self.failed)
        self.episode_reward = torch.zeros(
            batch_size, dtype=simulator.dtype, device=simulator.device
        )
        self.path_progress_cycles = torch.zeros_like(self.episode_reward)
        self.episode_error_squared_sum = torch.zeros_like(self.episode_reward)
        self.episode_error_samples = 0
        self.episode_maximum_displacement = torch.zeros_like(self.episode_reward)
        self.episode_maximum_speed = torch.zeros_like(self.episode_reward)
        self.episode_minimum_scene_height = torch.full_like(
            self.episode_reward, torch.inf
        )
        self.previous_acceleration_world = torch.zeros(
            (batch_size, 3), dtype=simulator.dtype, device=simulator.device
        )

    @property
    def episode_tracking_rmse_m(self) -> torch.Tensor:
        return torch.sqrt(
            self.episode_error_squared_sum / max(self.episode_error_samples, 1)
        )

    def _reference_condition(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.reference.point(self.path_progress_cycles),
            self.reference.tangent(self.path_progress_cycles),
        )

    def _observation(self) -> tuple[torch.Tensor, object]:
        target, direction = self._reference_condition()
        context = build_policy_context(
            self.simulator,
            self.state,
            target_position_world_m=target,
            desired_direction_world=direction,
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
        self.episode_reward.zero_()
        self.path_progress_cycles.zero_()
        self.episode_error_squared_sum.zero_()
        self.episode_error_samples = 0
        self.episode_maximum_displacement.zero_()
        self.episode_maximum_speed.zero_()
        self.episode_minimum_scene_height.fill_(torch.inf)
        self.previous_acceleration_world.zero_()
        observation, _ = self._observation()
        return observation

    def _decode_action(
        self, normalized_action: torch.Tensor, frame: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = normalized_action.to(
            dtype=self.simulator.dtype, device=self.simulator.device
        )
        local_acceleration = action[:, :3]
        norm = torch.linalg.vector_norm(local_acceleration, dim=-1, keepdim=True)
        local_acceleration = local_acceleration / torch.maximum(norm, torch.ones_like(norm))
        local_acceleration = self.maximum_acceleration_m_s2 * local_acceleration
        acceleration_world = frame.vectors_to_world(local_acceleration)
        body_rate = self.maximum_body_rate_rad_s * action[:, 3:6]
        acceleration_world = torch.where(
            self.failed[:, None], torch.zeros_like(acceleration_world), acceleration_world
        )
        body_rate = torch.where(
            self.failed[:, None], torch.zeros_like(body_rate), body_rate
        )
        return acceleration_world, body_rate

    def _project_tip_forward(
        self, tip_position_m: torch.Tensor, previous_tip_position_m: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        travel = torch.linalg.vector_norm(
            tip_position_m - previous_tip_position_m, dim=-1
        )
        center = self.path_progress_cycles + torch.clamp(
            travel / self.path_length_m, min=0.0, max=0.05
        )
        lower = self.path_progress_cycles - 0.03
        upper = self.path_progress_cycles + 0.08
        selected = center
        for _ in range(5):
            phase = 2.0 * math.pi * selected
            point = self.reference.point(selected)
            derivative = torch.stack(
                (
                    2.0
                    * math.pi
                    * self.reference.amplitude_x_m
                    * torch.cos(phase),
                    4.0
                    * math.pi
                    * self.reference.amplitude_y_m
                    * torch.cos(2.0 * phase),
                    torch.zeros_like(selected),
                ),
                dim=-1,
            )
            residual = point - tip_position_m
            step = (residual * derivative).sum(dim=-1) / derivative.square().sum(
                dim=-1
            ).clamp_min(1.0e-9)
            selected = torch.minimum(torch.maximum(selected - step, lower), upper)
        selected = torch.maximum(self.path_progress_cycles, selected)
        reference_position = self.reference.point(selected)
        error = torch.linalg.vector_norm(reference_position - tip_position_m, dim=-1)
        return selected, error

    @torch.no_grad()
    def step(self, normalized_action: torch.Tensor) -> Figure8ControlStepResult:
        if normalized_action.shape != (self.batch_size, 6):
            raise ValueError("Sequential Figure-8 SAC action must have shape Bx6.")
        _, context = self._observation()
        failed_before = self.failed.clone()
        acceleration_world, body_rate = self._decode_action(normalized_action, context.frame)
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
        previous_tip_position = self.state.cable.positions_m[:, -1]
        previous_tip_velocity = self.state.cable.velocities_m_s[:, -1]
        start_progress = self.path_progress_cycles.clone()
        error_squared_sum = torch.zeros_like(self.path_progress_cycles)
        tip_acceleration_squared_sum = torch.zeros_like(self.path_progress_cycles)
        maximum_speed = torch.zeros_like(self.path_progress_cycles)
        minimum_height = torch.full_like(self.path_progress_cycles, torch.inf)
        maximum_displacement = torch.linalg.vector_norm(
            self.state.uav.position_m - initial_position, dim=-1
        )
        new_numerical_failure = torch.zeros_like(self.failed)

        for _ in range(self.physics_steps_per_control):
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
            selected, error = self._project_tip_forward(tip_position, previous_tip_position)
            eligible = ~self.failed
            self.path_progress_cycles = torch.where(
                eligible, selected, self.path_progress_cycles
            )
            error_squared_sum += torch.where(eligible, error.square(), torch.zeros_like(error))
            tip_acceleration = (tip_velocity - previous_tip_velocity) / self.simulator.dt_s
            tip_acceleration_squared_sum += torch.where(
                eligible,
                tip_acceleration.square().sum(dim=-1),
                torch.zeros_like(error),
            )
            speed = torch.linalg.vector_norm(self.state.uav.velocity_m_s, dim=-1)
            maximum_speed = torch.where(eligible, torch.maximum(maximum_speed, speed), maximum_speed)
            scene_height = torch.minimum(
                self.state.uav.position_m[:, 2],
                torch.amin(self.state.cable.positions_m[:, :, 2], dim=1),
            )
            minimum_height = torch.where(
                eligible, torch.minimum(minimum_height, scene_height), minimum_height
            )
            displacement = torch.linalg.vector_norm(
                self.state.uav.position_m - initial_position, dim=-1
            )
            maximum_displacement = torch.where(
                eligible,
                torch.maximum(maximum_displacement, displacement),
                maximum_displacement,
            )
            previous_tip_position = tip_position
            previous_tip_velocity = tip_velocity

        mean_error_squared = error_squared_sum / self.physics_steps_per_control
        mean_tip_acceleration_squared = (
            tip_acceleration_squared_sum / self.physics_steps_per_control
        )
        progress_delta_m = (self.path_progress_cycles - start_progress) * self.path_length_m
        acceleration_effort = acceleration_world.square().sum(dim=-1)
        acceleration_change = (
            acceleration_world - self.previous_acceleration_world
        ).square().sum(dim=-1)
        speed_violation = torch.relu(
            maximum_speed / self.task.maximum_uav_speed_m_s - 1.0
        ).square()
        ground_violation = torch.relu(
            (self.reward_weights.ground_clearance_m - minimum_height)
            / self.reward_weights.ground_clearance_m
        ).square()
        reward = figure8_interval_reward(
            mean_error_squared,
            progress_delta_m,
            mean_tip_acceleration_squared,
            acceleration_effort,
            acceleration_change,
            speed_violation,
            ground_violation,
            new_numerical_failure,
            acceleration_scale_m_s2=self.maximum_acceleration_m_s2,
            weights=self.reward_weights,
        )
        reward = torch.where(failed_before, torch.zeros_like(reward), reward)
        self.previous_acceleration_world = torch.where(
            self.failed[:, None], self.previous_acceleration_world, acceleration_world
        )
        self.episode_reward += reward
        self.episode_error_squared_sum += error_squared_sum
        self.episode_error_samples += self.physics_steps_per_control
        self.episode_maximum_displacement = torch.maximum(
            self.episode_maximum_displacement, maximum_displacement
        )
        self.episode_maximum_speed = torch.maximum(
            self.episode_maximum_speed, maximum_speed
        )
        self.episode_minimum_scene_height = torch.minimum(
            self.episode_minimum_scene_height, minimum_height
        )
        self.control_index += 1
        horizon_done = self.control_index >= self.control_step_count
        if horizon_done:
            self.episode_success = figure8_episode_success(
                self.path_progress_cycles,
                self.episode_tracking_rmse_m,
                self.episode_maximum_speed,
                self.episode_minimum_scene_height,
                self.failed,
                required_cycles=self.success_required_cycles,
                maximum_tracking_rmse_m=self.success_maximum_tracking_rmse_m,
                maximum_uav_speed_m_s=self.task.maximum_uav_speed_m_s,
                minimum_scene_height_m=self.reward_weights.ground_clearance_m,
            )
        done = new_numerical_failure | torch.full_like(self.failed, horizon_done)
        next_observation, _ = self._observation()
        final_error = torch.linalg.vector_norm(
            self.reference.point(self.path_progress_cycles)
            - self.state.cable.positions_m[:, -1],
            dim=-1,
        )
        return Figure8ControlStepResult(
            next_observation=next_observation,
            reward=reward[:, None].float(),
            done=done[:, None].float(),
            include_transition=~failed_before,
            final_tracking_error_m=final_error,
            path_progress_cycles=self.path_progress_cycles.clone(),
            maximum_uav_displacement_m=maximum_displacement,
            numerical_failure=new_numerical_failure,
        )
