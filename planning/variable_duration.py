"""Variable-duration FullState commands and masked whip rollouts."""

from __future__ import annotations

import math

import torch

from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState

from .cem_task import VariableDurationWhipTask
from .command_parameterization import FullStateCommandTrajectory
from .metrics import PopulationRolloutResult, feasibility_first_cost
from .rl_reward import (
    RLWhipRewardConfig,
    compose_rl_terminal_reward,
    rl_event_reward,
)
from .rollout import DeterministicReplay, clone_state_batch


TERMINAL_SETTLE_DURATION_S = 0.30


def project_decisions(
    decisions: torch.Tensor,
    task: VariableDurationWhipTask,
    duration_max_s: float,
) -> torch.Tensor:
    """Project 48 acceleration variables and duration onto their hard bounds."""

    values = torch.as_tensor(decisions)
    one_row = values.ndim == 1
    if one_row:
        values = values.unsqueeze(0)
    expected = task.cem.knot_count * 3 + 1
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"CEM decisions must have shape Bx{expected}.")
    knots = values[:, :-1].reshape(-1, task.cem.knot_count, 3)
    norms = torch.linalg.vector_norm(knots, dim=-1, keepdim=True)
    knots = knots * torch.clamp(
        task.maximum_command_acceleration_m_s2 / torch.clamp(norms, min=1e-12),
        max=1.0,
    )
    duration = values[:, -1:].clamp(task.cem.duration_min_s, duration_max_s)
    result = torch.cat((knots.reshape(values.shape[0], -1), duration), dim=1)
    return result[0] if one_row else result


def variable_duration_fullstate(
    knots_m_s2: torch.Tensor,
    durations_s: torch.Tensor,
    *,
    initial_position_m: torch.Tensor,
    initial_velocity_m_s: torch.Tensor,
    yaw_rad: float | torch.Tensor,
    maximum_time_s: float,
    dt_s: float,
    settle_duration_s: float = TERMINAL_SETTLE_DURATION_S,
) -> FullStateCommandTrajectory:
    """Build an exact active command, smooth terminal settle, then hold.

    Acceleration knots live in normalized candidate time.  The active portion
    is the analytic integral of the piecewise-linear acceleration curve.  At
    the continuous candidate duration (which need not lie on the physics grid)
    a fixed cubic-Hermite velocity settle preserves position, velocity, and
    acceleration continuity.  It reaches zero velocity and acceleration after
    ``settle_duration_s`` and then holds the analytically determined position.
    """

    knots = torch.as_tensor(knots_m_s2)
    if knots.ndim == 2:
        knots = knots.unsqueeze(0)
    durations = torch.as_tensor(durations_s, dtype=knots.dtype, device=knots.device).reshape(-1)
    if knots.ndim != 3 or knots.shape[-1] != 3 or durations.shape[0] != knots.shape[0]:
        raise ValueError("Knots must be BxKx3 and durations must contain B values.")
    if settle_duration_s <= 0.0:
        raise ValueError("Terminal settle duration must be positive.")
    raw_steps = maximum_time_s / dt_s
    if abs(raw_steps - round(raw_steps)) > 1e-8:
        raise ValueError("Maximum time must contain an integer number of physics steps.")
    step_count = int(round(raw_steps))
    batch, knot_count, _ = knots.shape
    device, dtype = knots.device, knots.dtype
    times = torch.arange(step_count + 1, device=device, dtype=dtype) * dt_s
    normalized = (times[:, None] / durations[None]).clamp(0.0, 1.0)
    coordinate = normalized * (knot_count - 1)
    lower = torch.floor(coordinate).to(torch.int64).clamp(0, knot_count - 2)
    fraction = (coordinate - lower.to(dtype)).clamp(0.0, 1.0)
    batch_ids = torch.arange(batch, device=device)[None].expand(step_count + 1, batch)
    lower_values = knots[batch_ids, lower]
    upper_values = knots[batch_ids, lower + 1]
    delta_values = upper_values - lower_values
    active_acceleration = lower_values + delta_values * fraction[..., None]

    def batched(value: torch.Tensor, name: str) -> torch.Tensor:
        item = torch.as_tensor(value, dtype=dtype, device=device)
        if item.ndim == 1:
            item = item.unsqueeze(0)
        if item.shape == (1, 3) and batch > 1:
            item = item.expand(batch, 3)
        if item.shape != (batch, 3):
            raise ValueError(f"{name} must have shape 3, 1x3, or Bx3.")
        return item

    initial_position = batched(initial_position_m, "initial_position_m")
    initial_velocity = batched(initial_velocity_m_s, "initial_velocity_m_s")

    # Analytic primitives of the piecewise-linear acceleration curve.  This is
    # exact even when a knot or the candidate duration lies between dt samples.
    segment_duration = durations / float(knot_count - 1)
    segment_delta = knots[:, 1:] - knots[:, :-1]
    segment_dv = 0.5 * (knots[:, :-1] + knots[:, 1:]) * segment_duration[:, None, None]
    cumulative_dv = torch.cat(
        (torch.zeros((batch, 1, 3), dtype=dtype, device=device), torch.cumsum(segment_dv, dim=1)),
        dim=1,
    )
    knot_velocity = initial_velocity[:, None] + cumulative_dv
    segment_dp = (
        knot_velocity[:, :-1] * segment_duration[:, None, None]
        + 0.5 * knots[:, :-1] * segment_duration[:, None, None].square()
        + (1.0 / 6.0) * segment_delta * segment_duration[:, None, None].square()
    )
    cumulative_dp = torch.cat(
        (torch.zeros((batch, 1, 3), dtype=dtype, device=device), torch.cumsum(segment_dp, dim=1)),
        dim=1,
    )
    knot_position = initial_position[:, None] + cumulative_dp
    lower_velocity = knot_velocity[batch_ids, lower]
    lower_position = knot_position[batch_ids, lower]
    local_time = segment_duration[None] * fraction
    active_velocity = (
        lower_velocity
        + lower_values * local_time[..., None]
        + 0.5
        * delta_values
        * segment_duration[None, :, None]
        * fraction[..., None].square()
    )
    active_position = (
        lower_position
        + lower_velocity * local_time[..., None]
        + 0.5 * lower_values * local_time[..., None].square()
        + (1.0 / 6.0)
        * delta_values
        * segment_duration[None, :, None].square()
        * fraction[..., None].pow(3)
    )

    p_terminal = knot_position[:, -1]
    v_terminal = knot_velocity[:, -1]
    a_terminal = knots[:, -1]
    settle_duration = torch.as_tensor(settle_duration_s, dtype=dtype, device=device)
    tau = (times[:, None] - durations[None]).clamp(0.0, settle_duration_s)
    s = tau / settle_duration
    h00 = 2.0 * s.pow(3) - 3.0 * s.square() + 1.0
    h10 = s.pow(3) - 2.0 * s.square() + s
    settle_velocity = (
        h00[..., None] * v_terminal[None]
        + h10[..., None] * settle_duration * a_terminal[None]
    )
    dh00 = 6.0 * s.square() - 6.0 * s
    dh10 = 3.0 * s.square() - 4.0 * s + 1.0
    settle_acceleration = (
        dh00[..., None] * v_terminal[None] / settle_duration
        + dh10[..., None] * a_terminal[None]
    )
    integral_h00 = 0.5 * s.pow(4) - s.pow(3) + s
    integral_h10 = 0.25 * s.pow(4) - (2.0 / 3.0) * s.pow(3) + 0.5 * s.square()
    settle_position = (
        p_terminal[None]
        + settle_duration * integral_h00[..., None] * v_terminal[None]
        + settle_duration.square() * integral_h10[..., None] * a_terminal[None]
    )
    hold_position = (
        p_terminal
        + 0.5 * settle_duration * v_terminal
        + (settle_duration.square() / 12.0) * a_terminal
    )

    epsilon = 8.0 * torch.finfo(dtype).eps
    active = times[:, None] <= durations[None] + epsilon
    # Snap the analytically zero settle endpoint to the exact hold values; this
    # avoids a few float32 ulps of residual acceleration at s=1.
    settling = (~active) & (
        times[:, None] < durations[None] + settle_duration - epsilon
    )
    position = torch.where(
        active[..., None],
        active_position,
        torch.where(settling[..., None], settle_position, hold_position[None]),
    )
    velocity = torch.where(
        active[..., None],
        active_velocity,
        torch.where(settling[..., None], settle_velocity, torch.zeros_like(settle_velocity)),
    )
    acceleration = torch.where(
        active[..., None],
        active_acceleration,
        torch.where(
            settling[..., None], settle_acceleration, torch.zeros_like(settle_acceleration)
        ),
    )

    orientation = torch.zeros((step_count + 1, batch, 4), dtype=dtype, device=device)
    yaw = torch.as_tensor(yaw_rad, dtype=dtype, device=device)
    if yaw.ndim == 0:
        yaw = yaw.expand(batch)
    if yaw.shape != (batch,):
        raise ValueError("yaw_rad must be scalar or contain one yaw per batch row.")
    orientation[..., 2] = torch.sin(0.5 * yaw)[None]
    orientation[..., 3] = torch.cos(0.5 * yaw)[None]
    return FullStateCommandTrajectory(
        times,
        position,
        velocity,
        acceleration,
        orientation,
        torch.zeros_like(position),
    )


class VariableWhipAccumulator:
    """Per-row duration masking and first-valid-strike termination."""

    def __init__(
        self,
        task: VariableDurationWhipTask,
        durations_s: torch.Tensor,
        knots_m_s2: torch.Tensor,
        *,
        evaluation_durations_s: torch.Tensor | None = None,
        target_positions_m: torch.Tensor | None = None,
        desired_directions: torch.Tensor | None = None,
        initial_uav_positions_m: torch.Tensor | None = None,
        rl_reward_config: RLWhipRewardConfig | None = None,
    ) -> None:
        self.task = task
        self.rl_reward_config = rl_reward_config
        self.maneuver_durations = durations_s
        self.durations = (
            durations_s
            if evaluation_durations_s is None
            else torch.as_tensor(
                evaluation_durations_s,
                dtype=durations_s.dtype,
                device=durations_s.device,
            ).reshape(-1)
        )
        if self.durations.shape != durations_s.shape:
            raise ValueError("Evaluation durations must match maneuver-duration rows.")
        if not bool((self.durations + 1.0e-7 >= durations_s).all()):
            raise ValueError("Evaluation time cannot end before the active maneuver.")
        self.knots = knots_m_s2
        batch = durations_s.shape[0]
        device, dtype = durations_s.device, durations_s.dtype
        inf = torch.full((batch,), float("inf"), dtype=dtype, device=device)
        nan = torch.full((batch,), float("nan"), dtype=dtype, device=device)
        zeros = torch.zeros((batch,), dtype=dtype, device=device)
        def batch_vectors(
            value: torch.Tensor | tuple[float, float, float], name: str
        ) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=dtype, device=device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape == (1, 3) and batch > 1:
                tensor = tensor.expand(batch, -1)
            if tensor.shape != (batch, 3) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must be finite with shape 3, 1x3, or Bx3.")
            return tensor

        self.target = batch_vectors(
            task.target_position_m if target_positions_m is None else target_positions_m,
            "target_positions_m",
        )
        self.direction = batch_vectors(
            task.desired_direction if desired_directions is None else desired_directions,
            "desired_directions",
        )
        direction_norm = torch.linalg.vector_norm(self.direction, dim=-1, keepdim=True)
        if not bool((direction_norm > torch.finfo(dtype).eps).all()):
            raise ValueError("Every desired strike direction must be nonzero.")
        self.direction = self.direction / direction_norm
        self.initial_uav = batch_vectors(
            task.initial_uav_position_m
            if initial_uav_positions_m is None
            else initial_uav_positions_m,
            "initial_uav_positions_m",
        )
        self.cos_limit = math.cos(math.radians(task.maximum_direction_error_deg))
        legacy = task.legacy_run_online_objective
        self.cos_direction_shaping_limit = math.cos(
            math.radians(
                task.maximum_direction_error_deg
                if legacy is None
                else legacy.direction_shaping_error_deg
            )
        )
        self.best_event_cost = inf.clone()
        self.best_rl_event_reward = torch.full_like(inf, -float("inf"))
        self.initial_tip_distance = inf.clone()
        self.progress_minimum_tip_distance = inf.clone()
        self.minimum_tip_distance = inf.clone()
        self.best_event_time = nan.clone()
        self.best_event_distance = inf.clone()
        self.best_event_tip_speed = zeros.clone()
        self.maximum_tip_speed = zeros.clone()
        self.best_event_directed_speed = zeros.clone()
        self.best_event_direction_angle = torch.full_like(zeros, 180.0)
        self.best_event_displacement = zeros.clone()
        self.first_entry_marker = torch.zeros(batch, dtype=torch.int64, device=device)
        self.first_entry_time = nan.clone()
        self.first_entry_tip_distance = inf.clone()
        self.first_entry_tip_speed = zeros.clone()
        self.first_entry_directed_speed = zeros.clone()
        self.first_entry_direction_angle = torch.full_like(zeros, 180.0)
        self.uav_speed_at_entry = zeros.clone()
        self.uav_displacement_at_entry = zeros.clone()
        self.maximum_uav_displacement = zeros.clone()
        self.maximum_uav_speed = zeros.clone()
        self.minimum_non_tip_distance = inf.clone()
        self.finite = torch.ones(batch, dtype=torch.bool, device=device)
        self.success = torch.zeros(batch, dtype=torch.bool, device=device)
        self.completion_uav_position = torch.zeros(batch, 3, dtype=dtype, device=device)
        self.completion_uav_velocity = torch.zeros_like(self.completion_uav_position)
        self.completion_c10_position = torch.zeros_like(self.completion_uav_position)

    def observe(
        self,
        time_s: float,
        *,
        uav_position_m: torch.Tensor,
        uav_velocity_m_s: torch.Tensor,
        cable_positions_m: torch.Tensor,
        cable_velocities_m_s: torch.Tensor,
    ) -> None:
        active = (time_s <= self.durations + 1e-7) & ~self.success
        markers = cable_positions_m[:, 2:12]
        marker_velocity = cable_velocities_m_s[:, 2:12]
        tip_position, tip_velocity = markers[:, -1], marker_velocity[:, -1]
        tip_distance = torch.linalg.vector_norm(tip_position - self.target, dim=-1)
        tip_speed = torch.linalg.vector_norm(tip_velocity, dim=-1)
        self.maximum_tip_speed = torch.where(
            active, torch.maximum(self.maximum_tip_speed, tip_speed), self.maximum_tip_speed
        )
        directed = torch.sum(tip_velocity * self.direction, dim=-1)
        cosine = torch.clamp(directed / torch.clamp(tip_speed, min=1e-9), -1.0, 1.0)
        angle = torch.rad2deg(torch.acos(cosine))
        displacement = torch.linalg.vector_norm(uav_position_m - self.initial_uav, dim=-1)
        uav_speed = torch.linalg.vector_norm(uav_velocity_m_s, dim=-1)
        at_initial_sample = active & (abs(time_s) <= 1.0e-12)
        self.initial_tip_distance = torch.where(
            at_initial_sample, tip_distance, self.initial_tip_distance
        )
        self.progress_minimum_tip_distance = torch.where(
            active,
            torch.minimum(self.progress_minimum_tip_distance, tip_distance),
            self.progress_minimum_tip_distance,
        )
        self.maximum_uav_displacement = torch.where(
            active, torch.maximum(self.maximum_uav_displacement, displacement), self.maximum_uav_displacement
        )
        self.maximum_uav_speed = torch.where(
            active, torch.maximum(self.maximum_uav_speed, uav_speed), self.maximum_uav_speed
        )
        finite_now = (
            torch.isfinite(uav_position_m).all(dim=-1)
            & torch.isfinite(uav_velocity_m_s).all(dim=-1)
            & torch.isfinite(cable_positions_m).all(dim=(-2, -1))
            & torch.isfinite(cable_velocities_m_s).all(dim=(-2, -1))
        )
        self.finite = torch.where(active, self.finite & finite_now, self.finite)
        self.completion_uav_position = torch.where(active[:, None], uav_position_m, self.completion_uav_position)
        self.completion_uav_velocity = torch.where(active[:, None], uav_velocity_m_s, self.completion_uav_velocity)
        self.completion_c10_position = torch.where(active[:, None], tip_position, self.completion_c10_position)

        marker_distance = torch.linalg.vector_norm(
            markers - self.target[:, None, :], dim=-1
        )
        marker_ids = torch.arange(1, 11, device=markers.device, dtype=torch.int64)[None]
        inside = marker_distance <= self.task.success_radius_m
        candidate_marker = torch.where(inside, marker_ids, torch.full_like(marker_ids, 11)).amin(dim=1)
        new_entry = active & (self.first_entry_marker == 0) & (candidate_marker <= 10)
        now = torch.full_like(self.first_entry_time, time_s)
        self.first_entry_marker = torch.where(new_entry, candidate_marker, self.first_entry_marker)
        self.first_entry_time = torch.where(new_entry, now, self.first_entry_time)
        self.first_entry_tip_distance = torch.where(new_entry, tip_distance, self.first_entry_tip_distance)
        self.first_entry_tip_speed = torch.where(new_entry, tip_speed, self.first_entry_tip_speed)
        self.first_entry_directed_speed = torch.where(new_entry, directed, self.first_entry_directed_speed)
        self.first_entry_direction_angle = torch.where(new_entry, angle, self.first_entry_direction_angle)
        self.uav_speed_at_entry = torch.where(new_entry, uav_speed, self.uav_speed_at_entry)
        self.uav_displacement_at_entry = torch.where(new_entry, displacement, self.uav_displacement_at_entry)

        non_tip = marker_distance[:, :-1].amin(dim=1)
        self.minimum_non_tip_distance = torch.where(
            active,
            torch.minimum(self.minimum_non_tip_distance, non_tip),
            self.minimum_non_tip_distance,
        )
        legacy = self.task.legacy_run_online_objective
        if legacy is None:
            e_pos = tip_distance / self.task.success_radius_m
            e_speed = torch.relu(
                (self.task.minimum_directed_speed_m_s - directed)
                / self.task.minimum_directed_speed_m_s
            )
            e_dir = torch.relu((self.cos_limit - cosine) / (1.0 - self.cos_limit))
            e_non_tip = torch.relu(
                (self.task.non_tip_clearance_m - non_tip)
                / self.task.non_tip_clearance_m
            )
            weights = self.task.weights
            event = (
                weights.event_position * e_pos.square()
                + weights.event_speed_deficiency * e_speed.square()
                + weights.event_direction_deficiency * e_dir.square()
                + weights.event_non_tip_proximity * e_non_tip.square()
            )
        else:
            # Port of the effective legacy run_online full strike objective.
            # The legacy GUI used a bounded position term and proximity-gated
            # speed/direction penalties.  Its displayed predictive-speed
            # weight was 10 for the corrected runtime profile.
            distance_squared = tip_distance.square()
            proximity = torch.exp(
                -distance_squared / (2.0 * legacy.velocity_gate_sigma_m**2)
            )
            predictive_proximity = torch.exp(
                -distance_squared
                / (2.0 * legacy.predictive_velocity_gate_sigma_m**2)
            )
            speed_deficit = torch.relu(
                legacy.directed_speed_shaping_m_s - directed
            )
            predictive_speed_deficit = torch.relu(
                legacy.predictive_speed_ratio
                * legacy.directed_speed_shaping_m_s
                - directed
            )
            # The hard success gate remains ``cos_limit``.  A tuned reward may
            # use a stricter shaping angle so valid candidates retain margin
            # instead of sitting numerically on the scientific boundary.
            direction_deficit = torch.relu(
                self.cos_direction_shaping_limit - cosine
            )
            event = (
                legacy.position_weight
                * distance_squared
                / (distance_squared + legacy.position_sigma_m**2)
                + legacy.speed_weight * proximity * speed_deficit.square()
                + legacy.predictive_speed_weight
                * predictive_proximity
                * predictive_speed_deficit.square()
                + legacy.direction_weight
                * proximity
                * direction_deficit.square()
                + legacy.drone_displacement_weight * displacement.square()
            )
        better = active & (time_s > 0.0) & (event < self.best_event_cost)
        if self.rl_reward_config is not None:
            rl_event = rl_event_reward(
                tip_distance,
                directed,
                tip_speed,
                cosine,
                self.rl_reward_config,
            )
            better_rl = active & (time_s > 0.0) & (
                rl_event > self.best_rl_event_reward
            )
            self.best_rl_event_reward = torch.where(
                better_rl, rl_event, self.best_rl_event_reward
            )
        self.best_event_cost = torch.where(better, event, self.best_event_cost)
        self.minimum_tip_distance = torch.where(
            active & (time_s > 0.0), torch.minimum(self.minimum_tip_distance, tip_distance), self.minimum_tip_distance
        )
        self.best_event_time = torch.where(better, now, self.best_event_time)
        self.best_event_distance = torch.where(better, tip_distance, self.best_event_distance)
        self.best_event_tip_speed = torch.where(better, tip_speed, self.best_event_tip_speed)
        self.best_event_directed_speed = torch.where(better, directed, self.best_event_directed_speed)
        self.best_event_direction_angle = torch.where(better, angle, self.best_event_direction_angle)
        self.best_event_displacement = torch.where(
            better, displacement, self.best_event_displacement
        )

        feasible_now = (
            (self.maximum_uav_displacement <= self.task.maximum_uav_displacement_m)
            & (self.maximum_uav_speed <= self.task.maximum_uav_speed_m_s)
            & self.finite
        )
        valid_strike = (
            active
            & (time_s > 0.0)
            & (tip_distance <= self.task.success_radius_m)
            & (directed >= self.task.minimum_directed_speed_m_s)
            & (angle <= self.task.maximum_direction_error_deg)
            & (self.first_entry_marker == 10)
            & feasible_now
        )
        # A first valid hit is the scientific terminal state.  Store strike
        # values (rather than values at mere geometric entry) for reporting.
        newly_successful = valid_strike & ~self.success
        self.success = self.success | newly_successful
        self.first_entry_time = torch.where(newly_successful, now, self.first_entry_time)
        self.first_entry_tip_distance = torch.where(newly_successful, tip_distance, self.first_entry_tip_distance)
        self.first_entry_tip_speed = torch.where(newly_successful, tip_speed, self.first_entry_tip_speed)
        self.first_entry_directed_speed = torch.where(newly_successful, directed, self.first_entry_directed_speed)
        self.first_entry_direction_angle = torch.where(newly_successful, angle, self.first_entry_direction_angle)
        self.uav_speed_at_entry = torch.where(newly_successful, uav_speed, self.uav_speed_at_entry)
        self.uav_displacement_at_entry = torch.where(newly_successful, displacement, self.uav_displacement_at_entry)

    def finalize(self, command: FullStateCommandTrajectory) -> PopulationRolloutResult:
        task = self.task
        command_norm = torch.linalg.vector_norm(command.accelerations_m_s2, dim=-1)
        # The hard command-acceleration gate covers the deterministic settle as
        # well as the optimized knots.  A terminal state that cannot settle
        # within the physical acceleration limit is therefore infeasible; no
        # settle acceleration is clipped.
        maximum_command = command_norm.amax(dim=0)
        active = command.times_s[:, None] <= self.maneuver_durations[None] + 1e-7
        active_count = active.sum(dim=0).clamp(min=1).to(command_norm.dtype)
        effort = torch.sum(
            torch.where(active, (command_norm / task.maximum_command_acceleration_m_s2).square(), 0.0),
            dim=0,
        ) / active_count
        knot_delta = self.knots[:, 1:] - self.knots[:, :-1]
        smoothness = torch.mean(
            (torch.linalg.vector_norm(knot_delta, dim=-1) / task.maximum_command_acceleration_m_s2).square(),
            dim=1,
        )
        displacement_violation = torch.relu(
            (self.maximum_uav_displacement - task.maximum_uav_displacement_m)
            / task.maximum_uav_displacement_m
        ).square()
        speed_violation = torch.relu(
            (self.maximum_uav_speed - task.maximum_uav_speed_m_s)
            / task.maximum_uav_speed_m_s
        ).square()
        acceleration_violation = torch.relu(
            (maximum_command - task.maximum_command_acceleration_m_s2)
            / task.maximum_command_acceleration_m_s2
        ).square()
        feasible = (
            (self.maximum_uav_displacement <= task.maximum_uav_displacement_m)
            & (self.maximum_uav_speed <= task.maximum_uav_speed_m_s)
            & (maximum_command <= task.maximum_command_acceleration_m_s2 + 1e-5)
            & self.finite
        )
        success = self.success & feasible
        legacy = task.legacy_run_online_objective
        if legacy is None:
            task_cost = (
                self.best_event_cost
                + task.weights.command_effort * effort
                + task.weights.command_smoothness * smoothness
            )
        else:
            # Retain feasibility-first ordering as the hard safety contract,
            # while reproducing the legacy safety shaping inside each class.
            # Terms that can activate under the current task are normalized in
            # the same squared form as legacy run_online.
            non_tip_violation = torch.relu(
                (task.success_radius_m - self.minimum_non_tip_distance)
                / task.success_radius_m
            ).square()
            first_non_tip = (self.first_entry_marker > 0) & (
                self.first_entry_marker < 10
            )
            non_tip_violation = torch.where(
                first_non_tip,
                torch.maximum(non_tip_violation, torch.ones_like(non_tip_violation)),
                non_tip_violation,
            )
            safety_violation = (
                speed_violation + acceleration_violation + non_tip_violation
            )
            success_bonus = torch.where(
                success,
                torch.full_like(self.best_event_cost, -legacy.success_cost),
                torch.zeros_like(self.best_event_cost),
            )
            task_cost = (
                self.best_event_cost
                + legacy.safety_weight * safety_violation
                + legacy.control_effort_weight * effort
                + legacy.control_smoothness_weight * smoothness
                + success_bonus
            )
        violation = displacement_violation + speed_violation + acceleration_violation
        cost = feasibility_first_cost(
            task_cost,
            feasible=feasible,
            violation=violation,
            finite=self.finite,
            invalid_cost=task.weights.invalid_cost,
        )
        final_speed = torch.linalg.vector_norm(self.completion_uav_velocity, dim=-1)
        rl_components = None
        if self.rl_reward_config is not None:
            rl_components = compose_rl_terminal_reward(
                strike_reward=self.best_rl_event_reward,
                success=success,
                first_entry_marker=self.first_entry_marker,
                first_hit_time_s=self.first_entry_time,
                maximum_uav_displacement_m=self.maximum_uav_displacement,
                maximum_uav_speed_m_s=self.maximum_uav_speed,
                maximum_command_acceleration_m_s2=maximum_command,
                effort_cost=effort,
                smoothness_cost=smoothness,
                displacement_limit_m=task.maximum_uav_displacement_m,
                speed_limit_m_s=task.maximum_uav_speed_m_s,
                acceleration_limit_m_s2=task.maximum_command_acceleration_m_s2,
                config=self.rl_reward_config,
                initial_tip_distance_m=self.initial_tip_distance,
                minimum_tip_distance_m=self.progress_minimum_tip_distance,
            )
        return PopulationRolloutResult(
            cost=cost,
            task_cost=task_cost,
            feasible=feasible,
            feasibility_violation=violation,
            success=success,
            finite=self.finite & torch.isfinite(task_cost),
            minimum_tip_target_distance_m=self.minimum_tip_distance,
            best_event_time_s=self.best_event_time,
            best_event_tip_distance_m=self.best_event_distance,
            best_event_tip_speed_m_s=self.best_event_tip_speed,
            maximum_tip_speed_m_s=self.maximum_tip_speed,
            best_event_directed_speed_m_s=self.best_event_directed_speed,
            best_event_direction_angle_deg=self.best_event_direction_angle,
            first_entry_marker=self.first_entry_marker,
            first_entry_time_s=self.first_entry_time,
            first_entry_tip_distance_m=self.first_entry_tip_distance,
            first_entry_tip_speed_m_s=self.first_entry_tip_speed,
            first_entry_directed_speed_m_s=self.first_entry_directed_speed,
            first_entry_direction_angle_deg=self.first_entry_direction_angle,
            maximum_uav_displacement_m=self.maximum_uav_displacement,
            maximum_uav_speed_m_s=self.maximum_uav_speed,
            maximum_command_acceleration_m_s2=maximum_command,
            final_uav_position_m=self.completion_uav_position,
            final_c10_position_m=self.completion_c10_position,
            uav_speed_at_entry_m_s=self.uav_speed_at_entry,
            uav_displacement_at_entry_m=self.uav_displacement_at_entry,
            event_cost=self.best_event_cost,
            effort_cost=effort,
            smoothness_cost=smoothness,
            final_uav_speed_cost=(final_speed / task.maximum_uav_speed_m_s).square(),
            displacement_violation_cost=displacement_violation,
            speed_violation_cost=speed_violation,
            command_acceleration_violation_cost=acceleration_violation,
            rl_reward_components=rl_components,
        )


def run_variable_population_rollout(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    knots_m_s2: torch.Tensor,
    durations_s: torch.Tensor,
    task: VariableDurationWhipTask,
    *,
    maximum_time_s: float,
    evaluation_time_s: float | None = None,
    command_initial_positions_m: torch.Tensor | None = None,
    command_initial_velocities_m_s: torch.Tensor | None = None,
    command_yaws_rad: torch.Tensor | float | None = None,
    target_positions_m: torch.Tensor | None = None,
    desired_directions: torch.Tensor | None = None,
    initial_uav_positions_m: torch.Tensor | None = None,
    rl_reward_config: RLWhipRewardConfig | None = None,
) -> PopulationRolloutResult:
    knots = torch.as_tensor(knots_m_s2, dtype=simulator.dtype, device=simulator.device)
    durations = torch.as_tensor(durations_s, dtype=simulator.dtype, device=simulator.device).reshape(-1)
    command = variable_duration_fullstate(
        knots,
        durations,
        initial_position_m=(
            torch.tensor(task.initial_uav_position_m, device=simulator.device, dtype=simulator.dtype)
            if command_initial_positions_m is None
            else command_initial_positions_m
        ),
        initial_velocity_m_s=(
            torch.tensor(task.initial_uav_velocity_m_s, device=simulator.device, dtype=simulator.dtype)
            if command_initial_velocities_m_s is None
            else command_initial_velocities_m_s
        ),
        yaw_rad=task.initial_yaw_rad if command_yaws_rad is None else command_yaws_rad,
        maximum_time_s=maximum_time_s,
        dt_s=simulator.dt_s,
    )
    state = clone_state_batch(initial_state, knots.shape[0])
    evaluation_durations = None
    if evaluation_time_s is not None:
        if evaluation_time_s + 1.0e-9 < float(durations.max().detach().cpu()):
            raise ValueError("Evaluation time cannot precede a maneuver duration.")
        evaluation_durations = torch.full_like(durations, float(evaluation_time_s))
    accumulator = VariableWhipAccumulator(
        task,
        durations,
        knots,
        evaluation_durations_s=evaluation_durations,
        target_positions_m=target_positions_m,
        desired_directions=desired_directions,
        initial_uav_positions_m=initial_uav_positions_m,
        rl_reward_config=rl_reward_config,
    )
    accumulator.observe(
        0.0,
        uav_position_m=state.uav.position_m,
        uav_velocity_m_s=state.uav.velocity_m_s,
        cable_positions_m=state.cable.positions_m,
        cable_velocities_m_s=state.cable.velocities_m_s,
    )
    sequence = command.simulator_sequence()
    with torch.no_grad():
        for index in range(sequence.step_count):
            state = simulator._propagate(  # noqa: SLF001
                state, sequence.command_at(index), simulator.parameters, create_graph=False
            )
            accumulator.observe(
                (index + 1) * simulator.dt_s,
                uav_position_m=state.uav.position_m,
                uav_velocity_m_s=state.uav.velocity_m_s,
                cable_positions_m=state.cable.positions_m,
                cable_velocities_m_s=state.cable.velocities_m_s,
            )
    return accumulator.finalize(command)


def replay_variable_trajectory(
    simulator: CoupledSimulator,
    initial_state: SimulatorState,
    knots_m_s2: torch.Tensor,
    duration_s: float,
    task: VariableDurationWhipTask,
    *,
    command_initial_position_m: torch.Tensor | None = None,
    command_initial_velocity_m_s: torch.Tensor | None = None,
    command_yaw_rad: torch.Tensor | float | None = None,
    target_position_m: torch.Tensor | None = None,
    desired_direction: torch.Tensor | None = None,
    initial_uav_position_m: torch.Tensor | None = None,
    evaluation_time_s: float | None = None,
) -> DeterministicReplay:
    """Replay one maneuver, optionally observing a post-maneuver hold window."""

    # The production physics grid defines observable event times.  Replay only
    # through the last physical state sample at or before the continuous T.
    requested_evaluation = duration_s if evaluation_time_s is None else float(evaluation_time_s)
    if requested_evaluation + 1.0e-9 < duration_s:
        raise ValueError("Evaluation time cannot precede the active maneuver duration.")
    maximum_time = math.floor(requested_evaluation / simulator.dt_s + 1e-9) * simulator.dt_s
    knots = torch.as_tensor(knots_m_s2, dtype=simulator.dtype, device=simulator.device)
    if knots.ndim == 2:
        knots = knots.unsqueeze(0)
    durations = torch.tensor([duration_s], dtype=simulator.dtype, device=simulator.device)
    command = variable_duration_fullstate(
        knots,
        durations,
        initial_position_m=(
            torch.tensor(task.initial_uav_position_m, device=simulator.device, dtype=simulator.dtype)
            if command_initial_position_m is None
            else command_initial_position_m
        ),
        initial_velocity_m_s=(
            torch.tensor(task.initial_uav_velocity_m_s, device=simulator.device, dtype=simulator.dtype)
            if command_initial_velocity_m_s is None
            else command_initial_velocity_m_s
        ),
        yaw_rad=task.initial_yaw_rad if command_yaw_rad is None else command_yaw_rad,
        maximum_time_s=maximum_time,
        dt_s=simulator.dt_s,
    )
    state = clone_state_batch(initial_state, 1)
    states = [state]
    evaluation_durations = None
    if evaluation_time_s is not None:
        evaluation_durations = torch.tensor(
            [maximum_time], dtype=simulator.dtype, device=simulator.device
        )
    accumulator = VariableWhipAccumulator(
        task,
        durations,
        knots,
        evaluation_durations_s=evaluation_durations,
        target_positions_m=target_position_m,
        desired_directions=desired_direction,
        initial_uav_positions_m=initial_uav_position_m,
    )
    accumulator.observe(
        0.0,
        uav_position_m=state.uav.position_m,
        uav_velocity_m_s=state.uav.velocity_m_s,
        cable_positions_m=state.cable.positions_m,
        cable_velocities_m_s=state.cable.velocities_m_s,
    )
    sequence = command.simulator_sequence()
    with torch.no_grad():
        for index in range(sequence.step_count):
            state = simulator._propagate(  # noqa: SLF001
                state, sequence.command_at(index), simulator.parameters, create_graph=False
            )
            states.append(state)
            accumulator.observe(
                (index + 1) * simulator.dt_s,
                uav_position_m=state.uav.position_m,
                uav_velocity_m_s=state.uav.velocity_m_s,
                cable_positions_m=state.cable.positions_m,
                cable_velocities_m_s=state.cable.velocities_m_s,
            )
    result = accumulator.finalize(command)
    residual = torch.stack([
        torch.zeros_like(item.uav.position_m)
        if item.uav.residual_acceleration_m_s2 is None
        else item.uav.residual_acceleration_m_s2
        for item in states
    ])
    return DeterministicReplay(
        command=command,
        metrics=result,
        times_s=command.times_s,
        uav_positions_m=torch.stack([item.uav.position_m for item in states]),
        uav_velocities_m_s=torch.stack([item.uav.velocity_m_s for item in states]),
        uav_orientations_xyzw=torch.stack([item.uav.orientation_xyzw for item in states]),
        uav_angular_velocities_world_rad_s=torch.stack([item.uav.angular_velocity_world_rad_s for item in states]),
        cable_positions_m=torch.stack([item.cable.positions_m for item in states]),
        cable_velocities_m_s=torch.stack([item.cable.velocities_m_s for item in states]),
        residual_accelerations_m_s2=residual,
    )
