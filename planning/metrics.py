"""GPU-resident whip task cost and hard success evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .command_parameterization import FullStateCommandTrajectory
from .rl_reward import RLWhipRewardComponents
from .task import CanonicalWhipTask


def feasibility_first_cost(
    task_cost: torch.Tensor,
    *,
    feasible: torch.Tensor,
    violation: torch.Tensor,
    finite: torch.Tensor,
    invalid_cost: float,
) -> torch.Tensor:
    """Rank every finite feasible candidate before every infeasible one.

    The offset is derived from the current population's task-cost span.  It
    therefore provides a lexicographic feasible-first ordering without an
    arbitrary giant constant, while the nonnegative violation magnitude and
    original task cost retain a continuous ordering among infeasible rows.
    """

    if not (
        task_cost.shape == feasible.shape == violation.shape == finite.shape
    ):
        raise ValueError("Feasibility-first inputs must have identical shapes.")
    finite_task = finite & torch.isfinite(task_cost)
    finite_min = torch.amin(
        torch.where(
            finite_task,
            task_cost,
            torch.full_like(task_cost, float("inf")),
        )
    )
    finite_max = torch.amax(
        torch.where(
            finite_task,
            task_cost,
            torch.full_like(task_cost, -float("inf")),
        )
    )
    finite_any = torch.any(finite_task)
    population_span = torch.where(
        finite_any,
        torch.clamp(finite_max - finite_min, min=1.0),
        torch.ones_like(finite_min),
    )
    ranked = task_cost + (~feasible).to(task_cost.dtype) * (
        population_span + 1.0 + population_span * torch.clamp(violation, min=0.0)
    )
    valid_rank = finite_task & torch.isfinite(ranked)
    return torch.where(
        valid_rank,
        ranked,
        torch.full_like(ranked, invalid_cost),
    )


@dataclass(frozen=True, slots=True)
class PopulationRolloutResult:
    cost: torch.Tensor
    task_cost: torch.Tensor
    feasible: torch.Tensor
    feasibility_violation: torch.Tensor
    success: torch.Tensor
    finite: torch.Tensor
    minimum_tip_target_distance_m: torch.Tensor
    best_event_time_s: torch.Tensor
    best_event_tip_distance_m: torch.Tensor
    best_event_tip_speed_m_s: torch.Tensor
    maximum_tip_speed_m_s: torch.Tensor
    best_event_directed_speed_m_s: torch.Tensor
    best_event_direction_angle_deg: torch.Tensor
    first_entry_marker: torch.Tensor
    first_entry_time_s: torch.Tensor
    first_entry_tip_distance_m: torch.Tensor
    first_entry_tip_speed_m_s: torch.Tensor
    first_entry_directed_speed_m_s: torch.Tensor
    first_entry_direction_angle_deg: torch.Tensor
    maximum_uav_displacement_m: torch.Tensor
    maximum_uav_speed_m_s: torch.Tensor
    maximum_command_acceleration_m_s2: torch.Tensor
    final_uav_position_m: torch.Tensor
    final_c10_position_m: torch.Tensor
    uav_speed_at_entry_m_s: torch.Tensor
    uav_displacement_at_entry_m: torch.Tensor
    event_cost: torch.Tensor
    effort_cost: torch.Tensor
    smoothness_cost: torch.Tensor
    final_uav_speed_cost: torch.Tensor
    displacement_violation_cost: torch.Tensor
    speed_violation_cost: torch.Tensor
    command_acceleration_violation_cost: torch.Tensor
    rl_reward_components: RLWhipRewardComponents | None = None

    def row(self, index: int) -> dict[str, object]:
        """Materialize one candidate's compact diagnostics on the host."""

        def scalar(tensor: torch.Tensor) -> float:
            return float(tensor[index].detach().cpu())

        marker = int(self.first_entry_marker[index].detach().cpu())
        entry_time = scalar(self.first_entry_time_s)
        result = {
            "cost": scalar(self.cost),
            "task_cost": scalar(self.task_cost),
            "feasible": bool(self.feasible[index].detach().cpu()),
            "feasibility_violation": scalar(self.feasibility_violation),
            "success": bool(self.success[index].detach().cpu()),
            "finite": bool(self.finite[index].detach().cpu()),
            "minimum_tip_target_distance_m": scalar(self.minimum_tip_target_distance_m),
            "best_event_time_s": scalar(self.best_event_time_s),
            "best_event_tip_distance_m": scalar(self.best_event_tip_distance_m),
            "best_event_tip_speed_m_s": scalar(self.best_event_tip_speed_m_s),
            "maximum_tip_speed_m_s": scalar(self.maximum_tip_speed_m_s),
            "best_event_directed_speed_m_s": scalar(self.best_event_directed_speed_m_s),
            "best_event_direction_angle_deg": scalar(self.best_event_direction_angle_deg),
            "first_entry_marker": None if marker == 0 else marker,
            "first_entry_time_s": None if not math.isfinite(entry_time) else entry_time,
            "first_entry_tip_distance_m": scalar(self.first_entry_tip_distance_m),
            "first_entry_tip_speed_m_s": scalar(self.first_entry_tip_speed_m_s),
            "first_entry_directed_speed_m_s": scalar(self.first_entry_directed_speed_m_s),
            "first_entry_direction_angle_deg": scalar(self.first_entry_direction_angle_deg),
            "maximum_uav_displacement_m": scalar(self.maximum_uav_displacement_m),
            "maximum_uav_speed_m_s": scalar(self.maximum_uav_speed_m_s),
            "maximum_command_acceleration_m_s2": scalar(self.maximum_command_acceleration_m_s2),
            "final_uav_position_m": (
                self.final_uav_position_m[index].detach().cpu().tolist()
            ),
            "final_c10_position_m": (
                self.final_c10_position_m[index].detach().cpu().tolist()
            ),
            "uav_speed_at_entry_m_s": scalar(self.uav_speed_at_entry_m_s),
            "uav_displacement_at_entry_m": scalar(self.uav_displacement_at_entry_m),
            "cost_components": {
                "event": scalar(self.event_cost),
                "effort": scalar(self.effort_cost),
                "smoothness": scalar(self.smoothness_cost),
                "final_uav_speed": scalar(self.final_uav_speed_cost),
                "displacement_violation": scalar(self.displacement_violation_cost),
                "speed_violation": scalar(self.speed_violation_cost),
                "command_acceleration_violation": scalar(
                    self.command_acceleration_violation_cost
                ),
            },
        }
        if self.rl_reward_components is not None:
            components = self.rl_reward_components
            result["rl_reward"] = scalar(components.total)
            result["rl_reward_components"] = {
                "progress": scalar(components.progress),
                "strike": scalar(components.strike),
                "success": scalar(components.success),
                "safety": scalar(components.safety),
                "non_tip": scalar(components.non_tip),
                "control": scalar(components.control),
                "time": scalar(components.time),
                "displacement_violation": scalar(components.displacement_violation),
                "speed_violation": scalar(components.speed_violation),
                "acceleration_violation": scalar(components.acceleration_violation),
                "normalized_progress": scalar(components.normalized_progress),
                "initial_tip_distance_m": scalar(components.initial_tip_distance_m),
                "minimum_tip_distance_m": scalar(components.minimum_tip_distance_m),
            }
        return result


class WhipMetricAccumulator:
    """Accumulate population costs without retaining population trajectories."""

    def __init__(
        self,
        task: CanonicalWhipTask,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.task = task
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        inf = torch.full((batch_size,), float("inf"), dtype=dtype, device=device)
        nan = torch.full((batch_size,), float("nan"), dtype=dtype, device=device)
        zeros = torch.zeros((batch_size,), dtype=dtype, device=device)
        self.target = torch.tensor(task.target_position_m, dtype=dtype, device=device)
        self.direction = torch.tensor(task.desired_direction, dtype=dtype, device=device)
        self.initial_uav = torch.tensor(
            task.initial_uav_position_m, dtype=dtype, device=device
        )
        self.cos_limit = math.cos(math.radians(task.maximum_direction_error_deg))
        self.best_event_cost = inf.clone()
        self.minimum_tip_distance = inf.clone()
        self.best_event_time = nan.clone()
        self.best_event_distance = inf.clone()
        self.best_event_tip_speed = zeros.clone()
        self.maximum_tip_speed = zeros.clone()
        self.best_event_directed_speed = zeros.clone()
        self.best_event_direction_angle = torch.full_like(zeros, 90.0)
        self.first_entry_marker = torch.zeros(
            (batch_size,), dtype=torch.int64, device=device
        )
        self.first_entry_time = nan.clone()
        self.first_entry_tip_distance = inf.clone()
        self.first_entry_tip_speed = zeros.clone()
        self.first_entry_directed_speed = zeros.clone()
        self.first_entry_direction_angle = torch.full_like(zeros, 90.0)
        self.uav_speed_at_entry = zeros.clone()
        self.uav_displacement_at_entry = zeros.clone()
        self.maximum_uav_displacement = zeros.clone()
        self.maximum_uav_speed = zeros.clone()
        self.finite = torch.ones((batch_size,), dtype=torch.bool, device=device)

    def observe(
        self,
        time_s: float,
        *,
        uav_position_m: torch.Tensor,
        uav_velocity_m_s: torch.Tensor,
        cable_positions_m: torch.Tensor,
        cable_velocities_m_s: torch.Tensor,
    ) -> None:
        marker_positions = cable_positions_m[:, 2:12]
        marker_velocities = cable_velocities_m_s[:, 2:12]
        tip_position = marker_positions[:, -1]
        tip_velocity = marker_velocities[:, -1]
        tip_offset = tip_position - self.target
        tip_distance = torch.linalg.vector_norm(tip_offset, dim=-1)
        tip_speed = torch.linalg.vector_norm(tip_velocity, dim=-1)
        self.maximum_tip_speed = torch.maximum(self.maximum_tip_speed, tip_speed)
        directed_speed = torch.sum(tip_velocity * self.direction, dim=-1)
        direction_cosine = torch.clamp(
            directed_speed / torch.clamp(tip_speed, min=1.0e-9), -1.0, 1.0
        )
        direction_angle = torch.rad2deg(torch.acos(direction_cosine))
        uav_displacement = torch.linalg.vector_norm(
            uav_position_m - self.initial_uav, dim=-1
        )
        uav_speed = torch.linalg.vector_norm(uav_velocity_m_s, dim=-1)
        self.maximum_uav_displacement = torch.maximum(
            self.maximum_uav_displacement, uav_displacement
        )
        self.maximum_uav_speed = torch.maximum(self.maximum_uav_speed, uav_speed)
        self.finite = self.finite & (
            torch.isfinite(uav_position_m).all(dim=-1)
            & torch.isfinite(uav_velocity_m_s).all(dim=-1)
            & torch.isfinite(cable_positions_m).all(dim=(-2, -1))
            & torch.isfinite(cable_velocities_m_s).all(dim=(-2, -1))
        )

        marker_distance = torch.linalg.vector_norm(
            marker_positions - self.target, dim=-1
        )
        marker_ids = torch.arange(1, 11, dtype=torch.int64, device=self.device)[None]
        inside = marker_distance <= self.task.success_radius_m
        candidate_marker = torch.where(
            inside, marker_ids, torch.full_like(marker_ids, 11)
        ).amin(dim=1)
        new_entry = (self.first_entry_marker == 0) & (candidate_marker <= 10)
        time = torch.full_like(self.first_entry_time, time_s)
        self.first_entry_marker = torch.where(
            new_entry, candidate_marker, self.first_entry_marker
        )
        self.first_entry_time = torch.where(new_entry, time, self.first_entry_time)
        self.first_entry_tip_distance = torch.where(
            new_entry, tip_distance, self.first_entry_tip_distance
        )
        self.first_entry_tip_speed = torch.where(
            new_entry, tip_speed, self.first_entry_tip_speed
        )
        self.first_entry_directed_speed = torch.where(
            new_entry, directed_speed, self.first_entry_directed_speed
        )
        self.first_entry_direction_angle = torch.where(
            new_entry, direction_angle, self.first_entry_direction_angle
        )
        self.uav_speed_at_entry = torch.where(new_entry, uav_speed, self.uav_speed_at_entry)
        self.uav_displacement_at_entry = torch.where(
            new_entry, uav_displacement, self.uav_displacement_at_entry
        )

        if self.task.impact_window_s[0] - 1.0e-9 <= time_s <= self.task.impact_window_s[1] + 1.0e-9:
            non_tip_distance = marker_distance[:, :-1].amin(dim=1)
            e_position = tip_distance / self.task.success_radius_m
            e_speed = torch.relu(
                (self.task.minimum_directed_speed_m_s - directed_speed)
                / self.task.minimum_directed_speed_m_s
            )
            e_direction = torch.relu(
                (self.cos_limit - direction_cosine) / (1.0 - self.cos_limit)
            )
            e_non_tip = torch.relu(
                (self.task.non_tip_clearance_m - non_tip_distance)
                / self.task.non_tip_clearance_m
            )
            weights = self.task.weights
            event = (
                weights.event_position * e_position.square()
                + weights.event_speed_deficiency * e_speed.square()
                + weights.event_direction_deficiency * e_direction.square()
                + weights.event_non_tip_proximity * e_non_tip.square()
            )
            better = event < self.best_event_cost
            self.best_event_cost = torch.minimum(self.best_event_cost, event)
            self.minimum_tip_distance = torch.minimum(
                self.minimum_tip_distance, tip_distance
            )
            self.best_event_time = torch.where(better, time, self.best_event_time)
            self.best_event_distance = torch.where(
                better, tip_distance, self.best_event_distance
            )
            self.best_event_tip_speed = torch.where(
                better, tip_speed, self.best_event_tip_speed
            )
            self.best_event_directed_speed = torch.where(
                better, directed_speed, self.best_event_directed_speed
            )
            self.best_event_direction_angle = torch.where(
                better, direction_angle, self.best_event_direction_angle
            )

    def finalize(
        self,
        *,
        final_uav_position_m: torch.Tensor,
        final_uav_velocity_m_s: torch.Tensor,
        final_cable_positions_m: torch.Tensor,
        command: FullStateCommandTrajectory,
        knots_m_s2: torch.Tensor,
    ) -> PopulationRolloutResult:
        task, weights = self.task, self.task.weights
        command_norm = torch.linalg.vector_norm(command.accelerations_m_s2, dim=-1)
        maximum_command = command_norm.amax(dim=0)
        effort = torch.mean(
            (command_norm / task.maximum_command_acceleration_m_s2).square(), dim=0
        )
        knot_delta = knots_m_s2[:, 1:] - knots_m_s2[:, :-1]
        smoothness = torch.mean(
            (
                torch.linalg.vector_norm(knot_delta, dim=-1)
                / task.maximum_command_acceleration_m_s2
            ).square(),
            dim=1,
        )
        final_uav_speed = torch.linalg.vector_norm(final_uav_velocity_m_s, dim=-1)
        final_speed_cost = (final_uav_speed / task.maximum_uav_speed_m_s).square()
        displacement_violation = torch.relu(
            (self.maximum_uav_displacement - task.maximum_uav_displacement_m)
            / task.maximum_uav_displacement_m
        ).square()
        speed_violation = torch.relu(
            (self.maximum_uav_speed - task.maximum_uav_speed_m_s)
            / task.maximum_uav_speed_m_s
        ).square()
        command_acceleration_violation = torch.relu(
            (maximum_command - task.maximum_command_acceleration_m_s2)
            / task.maximum_command_acceleration_m_s2
        ).square()
        entry_in_window = (
            (self.first_entry_time >= task.impact_window_s[0] - 1.0e-9)
            & (self.first_entry_time <= task.impact_window_s[1] + 1.0e-9)
        )
        local_success = (
            (self.first_entry_marker == 10)
            & entry_in_window
            & (self.first_entry_tip_distance <= task.success_radius_m)
            & (self.first_entry_directed_speed >= task.minimum_directed_speed_m_s)
            & (self.first_entry_direction_angle <= task.maximum_direction_error_deg)
        )
        feasible = (
            (self.maximum_uav_displacement <= task.maximum_uav_displacement_m)
            & (self.maximum_uav_speed <= task.maximum_uav_speed_m_s)
            & (maximum_command <= task.maximum_command_acceleration_m_s2 + 1.0e-5)
            & self.finite
        )
        success = local_success & feasible
        # The scientific task cost remains the existing event objective with
        # only small command effort and knot-smoothness regularization.  UAV
        # feasibility is handled separately below so an illegal close hit can
        # never outrank a legal near-miss merely through a hand-tuned penalty.
        task_cost = (
            self.best_event_cost
            + weights.command_effort * effort
            + weights.command_smoothness * smoothness
            - weights.success_bonus * success.to(self.dtype)
        )
        feasibility_violation = (
            displacement_violation
            + speed_violation
            + command_acceleration_violation
        )
        cost = feasibility_first_cost(
            task_cost,
            feasible=feasible,
            violation=feasibility_violation,
            finite=self.finite,
            invalid_cost=weights.invalid_cost,
        )
        return PopulationRolloutResult(
            cost=cost,
            task_cost=task_cost,
            feasible=feasible,
            feasibility_violation=feasibility_violation,
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
            final_uav_position_m=final_uav_position_m,
            final_c10_position_m=final_cable_positions_m[:, 11],
            uav_speed_at_entry_m_s=self.uav_speed_at_entry,
            uav_displacement_at_entry_m=self.uav_displacement_at_entry,
            event_cost=self.best_event_cost,
            effort_cost=effort,
            smoothness_cost=smoothness,
            final_uav_speed_cost=final_speed_cost,
            displacement_violation_cost=displacement_violation,
            speed_violation_cost=speed_violation,
            command_acceleration_violation_cost=command_acceleration_violation,
        )


def synthetic_hard_success(
    task: CanonicalWhipTask,
    *,
    tip_position_m: torch.Tensor,
    tip_velocity_m_s: torch.Tensor,
    first_entry_marker: int,
    maximum_uav_displacement_m: float,
    maximum_uav_speed_m_s: float,
    maximum_command_acceleration_m_s2: float,
) -> bool:
    """Small host-side oracle used only by task-logic unit tests."""

    target = torch.tensor(task.target_position_m, dtype=torch.float64)
    direction = torch.tensor(task.desired_direction, dtype=torch.float64)
    position = torch.as_tensor(tip_position_m, dtype=torch.float64)
    velocity = torch.as_tensor(tip_velocity_m_s, dtype=torch.float64)
    distance = float(torch.linalg.vector_norm(position - target))
    speed = float(torch.linalg.vector_norm(velocity))
    directed = float(torch.dot(velocity, direction))
    cosine = directed / max(speed, 1.0e-12)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return bool(
        distance <= task.success_radius_m
        and directed >= task.minimum_directed_speed_m_s
        and angle <= task.maximum_direction_error_deg
        and first_entry_marker == 10
        and maximum_uav_displacement_m <= task.maximum_uav_displacement_m
        and maximum_uav_speed_m_s <= task.maximum_uav_speed_m_s
        and maximum_command_acceleration_m_s2
        <= task.maximum_command_acceleration_m_s2
    )
