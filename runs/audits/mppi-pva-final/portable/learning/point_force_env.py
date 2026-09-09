"""Batched PPO environment for direct 3D force control of the DDER cable."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

import torch

from simulator.cable import DderState
from simulator.point_mass import ForceControlledPointCable


POINT_FORCE_OBSERVATION_DIM = 79


@dataclass(frozen=True, slots=True)
class PointForceRewardWeights:
    progress: float
    strike_quality_improvement: float
    point_displacement_integral: float
    maximum_displacement: float
    success: float
    success_forward_return: float
    success_release: float
    non_tip_first: float
    invalid_tip_entry: float
    timeout: float
    terminal_displacement: float
    time_per_s: float
    numerical_failure: float
    proximity_scale_m: float
    directed_speed_cap_m_s: float
    forward_excursion_scale_m: float
    point_backward_speed_scale_m_s: float
    relative_tip_forward_speed_scale_m_s: float
    displacement_cost_scale_m: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PointForceRewardWeights":
        result = cls(
            progress=float(value["progress_weight"]),
            maximum_displacement=float(value.get('maximum_displacement_weight',0.)),
            strike_quality_improvement=float(
                value["strike_quality_improvement_weight"]
            ),
            point_displacement_integral=float(
                value["point_displacement_integral_weight"]
            ),
            success=float(value["success_bonus"]),
            success_forward_return=float(
                value["success_forward_return_bonus_weight"]
            ),
            success_release=float(value["success_release_bonus_weight"]),
            non_tip_first=float(value["non_tip_first_penalty"]),
            invalid_tip_entry=float(value["invalid_tip_entry_penalty"]),
            timeout=float(value["timeout_penalty"]),
            terminal_displacement=float(value["terminal_displacement_weight"]),
            time_per_s=float(value["time_to_success_weight_per_s"]),
            numerical_failure=float(value["numerical_failure_penalty"]),
            proximity_scale_m=float(value["proximity_scale_m"]),
            directed_speed_cap_m_s=float(value["directed_speed_reward_cap_m_s"]),
            forward_excursion_scale_m=float(value["forward_excursion_scale_m"]),
            point_backward_speed_scale_m_s=float(
                value["point_backward_speed_scale_m_s"]
            ),
            relative_tip_forward_speed_scale_m_s=float(
                value["relative_tip_forward_speed_scale_m_s"]
            ),
            displacement_cost_scale_m=float(value["displacement_cost_scale_m"]),
        )
        if any(
            not math.isfinite(number) or number < 0.0
            for number in (
                result.progress,
                result.maximum_displacement,
                result.strike_quality_improvement,
                result.point_displacement_integral,
                result.success,
                result.success_forward_return,
                result.success_release,
                result.non_tip_first,
                result.invalid_tip_entry,
                result.timeout,
                result.terminal_displacement,
                result.time_per_s,
                result.numerical_failure,
            )
        ):
            raise ValueError("Reward weights must be finite and non-negative.")
        if min(
            result.proximity_scale_m,
            result.directed_speed_cap_m_s,
            result.forward_excursion_scale_m,
            result.point_backward_speed_scale_m_s,
            result.relative_tip_forward_speed_scale_m_s,
            result.displacement_cost_scale_m,
        ) <= 0.0:
            raise ValueError("Reward scales must be positive.")
        return result


@dataclass(frozen=True, slots=True)
class PointForceRewardComponents:
    progress: torch.Tensor
    strike_quality: torch.Tensor
    point_displacement_integral: torch.Tensor
    maximum_displacement: torch.Tensor
    success: torch.Tensor
    forward_return: torch.Tensor
    release: torch.Tensor
    non_tip_first: torch.Tensor
    invalid_tip_entry: torch.Tensor
    timeout: torch.Tensor
    terminal_displacement: torch.Tensor
    time: torch.Tensor
    numerical_failure: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return (
            self.progress
            + self.strike_quality
            + self.point_displacement_integral
            + self.maximum_displacement
            + self.success
            + self.forward_return
            + self.release
            + self.non_tip_first
            + self.invalid_tip_entry
            + self.timeout
            + self.terminal_displacement
            + self.time
            + self.numerical_failure
        )


@dataclass(frozen=True, slots=True)
class PointForceEnvironmentStep:
    next_observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    include_transition: torch.Tensor
    newly_successful: torch.Tensor
    newly_non_tip_first: torch.Tensor
    newly_invalid_tip_entry: torch.Tensor
    newly_timed_out: torch.Tensor
    numerical_failure: torch.Tensor
    commanded_force_world_n: torch.Tensor
    components: PointForceRewardComponents


def near_target_speed_quality(
    distance_m: torch.Tensor,
    directed_speed_m_s: torch.Tensor,
    *,
    proximity_scale_m: float,
    directed_speed_cap_m_s: float,
) -> torch.Tensor:
    """Bounded strike potential with no angle term."""

    proximity = torch.exp(
        -0.5 * torch.square(distance_m / max(proximity_scale_m, 1.0e-6))
    )
    width = max(0.25 * directed_speed_cap_m_s, 1.0e-6)
    speed = torch.sigmoid(
        (directed_speed_m_s - directed_speed_cap_m_s) / width
    )
    baseline = torch.sigmoid(
        torch.as_tensor(
            -directed_speed_cap_m_s / width,
            dtype=speed.dtype,
            device=speed.device,
        )
    )
    at_cap = torch.sigmoid(torch.zeros((), dtype=speed.dtype, device=speed.device))
    normalized_speed = ((speed - baseline) / (at_cap - baseline)).clamp(0.0, 1.0)
    return proximity * normalized_speed


def near_target_world_relative_quality(distance_m, world_speed_m_s, relative_speed_m_s,
                                       *, proximity_scale_m, world_cap_m_s, relative_cap_m_s):
    """Credit requires both forward world motion and motion relative to the root.

    Retreating under a stationary tip cannot earn this quality; translating a
    motionless cable cannot either. This encourages dynamic strikes, but is not
    a classification of a travelling whip wave versus a pendulum swing.
    """
    world = near_target_speed_quality(distance_m, world_speed_m_s,
        proximity_scale_m=proximity_scale_m, directed_speed_cap_m_s=world_cap_m_s)
    relative = near_target_speed_quality(torch.zeros_like(distance_m), relative_speed_m_s,
        proximity_scale_m=proximity_scale_m, directed_speed_cap_m_s=relative_cap_m_s)
    return world * relative


def required_reach_allowance(initial_root_m, target_m, cable_length_m, target_radius_m, margin_m):
    """Triangle-inequality lower bound plus explicit preparation margin, per row.

    The bound assumes a fully extended cable; it is not a dynamic reachability
    guarantee. The margin is a reward choice, not a fitted physical parameter.
    """
    reach = torch.linalg.vector_norm(target_m - initial_root_m, dim=-1)
    return (reach - cable_length_m - target_radius_m).clamp_min(0.) + margin_m


def excursion_cost(displacement_m, allowance_m, scale_m):
    return torch.log1p(((displacement_m - allowance_m).clamp_min(0.) / scale_m).square())


def forward_return_quality(
    peak_forward_displacement_m: torch.Tensor,
    current_forward_displacement_m: torch.Tensor,
    *,
    excursion_scale_m: float,
) -> torch.Tensor:
    peak = peak_forward_displacement_m.clamp_min(0.0)
    returned = (peak - current_forward_displacement_m).clamp_min(0.0)
    return_fraction = (returned / peak.clamp_min(1.0e-6)).clamp(max=1.0)
    activation = 1.0 - torch.exp(
        -peak / max(float(excursion_scale_m), 1.0e-6)
    )
    return activation * return_fraction


def release_quality(
    peak_forward_displacement_m: torch.Tensor,
    point_forward_speed_m_s: torch.Tensor,
    relative_tip_forward_speed_m_s: torch.Tensor,
    *,
    excursion_scale_m: float,
    backward_speed_scale_m_s: float,
    tip_speed_scale_m_s: float,
) -> torch.Tensor:
    activation = 1.0 - torch.exp(
        -peak_forward_displacement_m.clamp_min(0.0)
        / max(float(excursion_scale_m), 1.0e-6)
    )
    backward = 1.0 - torch.exp(
        -(-point_forward_speed_m_s).clamp_min(0.0)
        / max(float(backward_speed_scale_m_s), 1.0e-6)
    )
    relative_tip = 1.0 - torch.exp(
        -relative_tip_forward_speed_m_s.clamp_min(0.0)
        / max(float(tip_speed_scale_m_s), 1.0e-6)
    )
    return activation * backward * relative_tip


def tip_velocity_strike_gate(
    tip_velocity_world_m_s: torch.Tensor,
    desired_strike_direction_world: torch.Tensor,
    *,
    minimum_directed_speed_m_s: float,
    maximum_direction_error_deg: float,
) -> torch.Tensor:
    """Gate actual world tip velocity against the desired strike direction."""

    velocity = torch.as_tensor(tip_velocity_world_m_s)
    desired = torch.as_tensor(
        desired_strike_direction_world,
        dtype=velocity.dtype,
        device=velocity.device,
    )
    desired_unit = desired / torch.linalg.vector_norm(
        desired, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(velocity.dtype).eps)
    directed_speed = (velocity * desired_unit).sum(dim=-1)
    direction_cosine = directed_speed / torch.linalg.vector_norm(
        velocity, dim=-1
    ).clamp_min(torch.finfo(velocity.dtype).eps)
    cosine_threshold = math.cos(math.radians(maximum_direction_error_deg))
    cosine_tolerance = 8.0 * torch.finfo(velocity.dtype).eps
    return (directed_speed >= minimum_directed_speed_m_s) & (
        direction_cosine >= cosine_threshold - cosine_tolerance
    )


def _segment_enters_sphere(
    previous_m: torch.Tensor,
    current_m: torch.Tensor,
    center_m: torch.Tensor,
    radius_m: float,
) -> torch.Tensor:
    displacement = current_m - previous_m
    center_offset = center_m[:, None] - previous_m
    denominator = displacement.square().sum(dim=-1).clamp_min(
        torch.finfo(previous_m.dtype).eps
    )
    fraction = (center_offset * displacement).sum(dim=-1) / denominator
    closest = previous_m + fraction.clamp(0.0, 1.0)[..., None] * displacement
    return torch.linalg.vector_norm(closest - center_m[:, None], dim=-1) <= radius_m


def _replace_state_rows(
    candidate: DderState, previous: DderState, accept: torch.Tensor
) -> DderState:
    mask = accept[:, None, None]

    def choose(
        value: torch.Tensor | None, fallback: torch.Tensor | None
    ) -> torch.Tensor | None:
        if value is None or fallback is None:
            return fallback if value is None else value
        value_mask = accept.reshape(
            (accept.shape[0],) + (1,) * (value.ndim - 1)
        )
        return torch.where(value_mask, value, fallback)

    return DderState(
        positions_m=torch.where(mask, candidate.positions_m, previous.positions_m),
        velocities_m_s=torch.where(mask, candidate.velocities_m_s, previous.velocities_m_s),
        endpoint_orientations=choose(
            candidate.endpoint_orientations, previous.endpoint_orientations
        ),
        endpoint_twist_rad=choose(
            candidate.endpoint_twist_rad, previous.endpoint_twist_rad
        ),
    )


class PointForceWhipEnvironment:
    """Synchronous batched one-attempt whip task for PPO collection."""

    observation_dim = POINT_FORCE_OBSERVATION_DIM
    action_dim = 3

    def __init__(
        self,
        model_config: Mapping[str, Any],
        task_config: Mapping[str, Any],
        ppo_config: Mapping[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.model = ForceControlledPointCable.from_mapping(model_config)
        if model_config.get('fullstate_execution',{}).get('schema')=='tracked_pose_execution_v1':
            from simulator.research_physics import install_runtime
            install_runtime(self.model,graph=bool(ppo_config.get('cuda_graph_physics',False)))
        elif device.type=='cuda' and ppo_config.get('cuda_graph_physics', False):
            from simulator.cuda_graph_physics import install_graph_runtime
            install_graph_runtime(self.model)
        self.model_config = model_config
        self.task_config = task_config
        self.ppo_config = ppo_config
        self.batch_size = int(batch_size)
        self.device = device
        physics_dtype = ppo_config.get('physics_dtype', 'float32' if device.type == 'cuda' else 'float64')
        if physics_dtype not in ('float32','float64'):
            raise ValueError('Physics dtype must be float32 or float64.')
        self.dtype = torch.float64 if physics_dtype == 'float64' else torch.float32
        self.physics_dt_s = float(model_config["simulation"]["dt_s"])
        self.control_dt_s = float(task_config["control_dt_s"])
        ratio = self.control_dt_s / self.physics_dt_s
        self.physics_steps_per_control = int(round(ratio))
        if not math.isclose(
            ratio, self.physics_steps_per_control, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError("control_dt_s must be an integer multiple of physics dt.")
        duration_ratio = float(task_config["episode_duration_s"]) / self.control_dt_s
        self.control_step_count = int(round(duration_ratio))
        if not math.isclose(
            duration_ratio, self.control_step_count, rel_tol=0.0, abs_tol=1.0e-10
        ):
            raise ValueError("Episode duration must align with control_dt_s.")

        self.target = self._batch_vector(task_config["target_position_m"])
        direction = self._batch_vector(task_config["desired_strike_direction_world"])
        self.desired_direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(self.dtype).eps)
        success = task_config["success"]
        self.target_radius_m = float(success["tip_target_distance_m"])
        self.minimum_directed_speed_m_s = float(
            success["minimum_directed_tip_speed_m_s"]
        )
        self.maximum_tip_velocity_direction_error_deg = float(
            success["maximum_tip_velocity_to_desired_direction_error_deg"]
        )
        self.tip_must_enter_first = bool(success["tip_must_enter_before_other_markers"])
        self.first_contact_only = bool(success.get('first_contact_only', False))
        self.marker_indices = torch.tensor(
            self.model.cable_configuration.marker_node_indices[1:],
            dtype=torch.long,
            device=device,
        )

        action = ppo_config["action"]
        self.action_delta_scale_n = self._batch_vector(action["delta_force_scale_n"])
        self.minimum_vertical_force_n = float(action["minimum_vertical_force_n"])
        self.maximum_force_norm_n = float(action["maximum_force_norm_n"])
        self.hover_force_world_n = self.model.hover_force_world_n(
            dtype=self.dtype, device=self.device
        )[None]
        self.reward_weights = PointForceRewardWeights.from_mapping(ppo_config["reward"])
        self.speed_shaping_reference=ppo_config['reward'].get('directed_speed_shaping_reference','attachment_relative')
        if self.speed_shaping_reference not in ('world','attachment_relative','world_and_attachment_relative'):
            raise ValueError('Unknown speed shaping reference.')
        reward = ppo_config['reward']
        self.relative_speed_cap_m_s = float(reward.get('relative_directed_speed_reward_cap_m_s', 6.))
        self.displacement_allowance_mode = reward.get('displacement_allowance_mode', 'none')
        self.displacement_allowance_margin_m = float(reward.get('displacement_allowance_margin_m', 0.))
        if not math.isfinite(self.relative_speed_cap_m_s) or self.relative_speed_cap_m_s <= 0:
            raise ValueError('Relative speed cap must be finite and positive.')
        if self.displacement_allowance_mode not in ('none', 'required_reach_plus_margin'):
            raise ValueError('Unknown displacement allowance mode.')
        if not math.isfinite(self.displacement_allowance_margin_m) or self.displacement_allowance_margin_m < 0:
            raise ValueError('Preparation margin must be finite and nonnegative.')
        numerical = ppo_config["numerical_limits"]
        self.numerical_position_limit_m = float(numerical["absolute_position_m"])
        self.numerical_speed_limit_m_s = float(numerical["node_speed_m_s"])
        observation = ppo_config["observation"]
        self.position_scale_m = float(observation["position_scale_m"])
        self.velocity_scale_m_s = float(observation["velocity_scale_m_s"])
        self.observation_clip = float(observation["clip"])
        self.reset()

    def _batch_vector(self, values: Any) -> torch.Tensor:
        vector = torch.as_tensor(values, dtype=self.dtype, device=self.device)
        if vector.shape != (3,):
            raise ValueError("Expected one three-dimensional vector.")
        return vector[None].expand(self.batch_size, -1).clone()

    def set_success_condition(
        self,
        *,
        target_radius_m: float,
        minimum_directed_speed_m_s: float,
        maximum_tip_velocity_direction_error_deg: float,
    ) -> None:
        radius = float(target_radius_m)
        speed = float(minimum_directed_speed_m_s)
        angle = float(maximum_tip_velocity_direction_error_deg)
        if radius <= 0.0 or speed < 0.0 or not 0.0 <= angle <= 180.0:
            raise ValueError("Invalid curriculum success condition.")
        self.target_radius_m = radius
        self.minimum_directed_speed_m_s = speed
        self.maximum_tip_velocity_direction_error_deg = angle

    def reset(self, state: DderState | None = None) -> torch.Tensor:
        root = self._batch_vector(self.task_config["initial_root_position_m"])
        root_velocity = self._batch_vector(
            self.task_config["initial_root_velocity_m_s"]
        )
        if state is None:
            self.state = self.model.hanging_state(root, root_velocity)
        else:
            shape = (self.batch_size, self.model.cable_configuration.node_count, 3)
            if state.positions_m.shape != shape or state.velocities_m_s.shape != shape:
                raise ValueError("Initial cable state has an incompatible shape.")
            if not bool(torch.isfinite(state.positions_m).all() & torch.isfinite(state.velocities_m_s).all()):
                raise ValueError("Initial cable state must be finite.")
            self.state = DderState(
                state.positions_m.to(dtype=self.dtype, device=self.device).detach().clone(),
                state.velocities_m_s.to(dtype=self.dtype, device=self.device).detach().clone(),
            )
            root = self.state.positions_m[:, 0]
        self.initial_root_position = root.clone()
        self.displacement_allowance_m = torch.zeros(self.batch_size, dtype=self.dtype, device=self.device)
        if self.displacement_allowance_mode == 'required_reach_plus_margin':
            self.displacement_allowance_m = required_reach_allowance(root, self.target,
                self.model.cable_configuration.length_m,
                float(self.task_config['success']['tip_target_distance_m']),
                self.displacement_allowance_margin_m)
        self.initial_tip_distance = torch.linalg.vector_norm(
            self.state.positions_m[:, -1] - self.target, dim=-1
        )
        self.best_progress = torch.zeros(
            self.batch_size, dtype=self.dtype, device=self.device
        )
        self.best_strike_quality = torch.zeros_like(self.best_progress)
        self.peak_forward_displacement = torch.zeros_like(self.best_progress)
        self.active = torch.ones(
            self.batch_size, dtype=torch.bool, device=self.device
        )
        self.episode_success = torch.zeros_like(self.active)
        self.episode_non_tip_first = torch.zeros_like(self.active)
        self.episode_invalid_tip_entry = torch.zeros_like(self.active)
        self.episode_timed_out = torch.zeros_like(self.active)
        self.failed = torch.zeros_like(self.active)
        self.episode_nonfinite = torch.zeros_like(self.active)
        self.episode_position_limit = torch.zeros_like(self.active)
        self.episode_speed_limit = torch.zeros_like(self.active)
        self.episode_reward = torch.zeros_like(self.best_progress)
        self.episode_component_sums = {name: torch.zeros_like(self.best_progress)
            for name in PointForceRewardComponents.__dataclass_fields__}
        self.episode_impact_speed = torch.full_like(self.best_progress,torch.nan)
        self.episode_hit_tip_directed_speed = torch.full_like(self.best_progress, torch.nan)
        self.episode_hit_relative_tip_directed_speed = torch.full_like(self.best_progress, torch.nan)
        self.episode_hit_attachment_directed_speed = torch.full_like(self.best_progress, torch.nan)
        self.episode_first_tip_directed_speed = torch.full_like(self.best_progress,torch.nan)
        self.episode_first_tip_angle = torch.full_like(self.best_progress,torch.nan)
        self.episode_point_displacement_integral_m_s = torch.zeros_like(
            self.best_progress
        )
        self.episode_point_displacement_cost_integral_s = torch.zeros_like(
            self.best_progress
        )
        self.episode_minimum_tip_distance = self.initial_tip_distance.clone()
        self.episode_maximum_point_displacement = torch.zeros_like(self.best_progress)
        self.episode_terminal_point_displacement = torch.full_like(
            self.best_progress, torch.nan
        )
        self.episode_hit_time_s = torch.full_like(self.best_progress, torch.nan)
        self.control_step_index = 0
        self.physics_step_index = 0
        return self.observe()

    def observe(self) -> torch.Tensor:
        relative_positions = (
            self.state.positions_m - self.initial_root_position[:, None]
        ) / self.position_scale_m
        velocities = self.state.velocities_m_s / self.velocity_scale_m_s
        relative_target = (
            self.target - self.initial_root_position
        ) / self.position_scale_m
        remaining = torch.full(
            (self.batch_size, 1),
            max(0.0, 1.0 - self.control_step_index / self.control_step_count),
            dtype=self.dtype,
            device=self.device,
        )
        observation = torch.cat(
            (
                relative_positions.reshape(self.batch_size, -1),
                velocities.reshape(self.batch_size, -1),
                relative_target,
                self.desired_direction,
                remaining,
            ),
            dim=-1,
        )
        if observation.shape[1] != self.observation_dim:
            raise RuntimeError("Point-force observation schema changed unexpectedly.")
        return observation.clamp(-self.observation_clip, self.observation_clip).float()

    def physical_force(self, normalized_action: torch.Tensor) -> torch.Tensor:
        action = torch.as_tensor(
            normalized_action, dtype=self.dtype, device=self.device
        ).detach()
        if action.shape != (self.batch_size, 3):
            raise ValueError("PPO action must have shape Bx3.")
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
            -1.0, 1.0
        )
        force = self.hover_force_world_n + action * self.action_delta_scale_n
        force[:, 2].clamp_(min=self.minimum_vertical_force_n)
        norm = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
        force = force * torch.clamp(
            self.maximum_force_norm_n / norm.clamp_min(torch.finfo(self.dtype).eps),
            max=1.0,
        )
        return force

    def _empty_components(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.zeros(
                self.batch_size, dtype=self.dtype, device=self.device
            )
            for name in (
                "progress",
                "strike_quality",
                "point_displacement_integral",
                "maximum_displacement",
                "success",
                "forward_return",
                "release",
                "non_tip_first",
                "invalid_tip_entry",
                "timeout",
                "terminal_displacement",
                "time",
                "numerical_failure",
            )
        }

    def step(
        self,
        normalized_action: torch.Tensor,
        *,
        stop_when_all_done: bool = False,
        physics_trace_callback: Callable[
            [DderState, torch.Tensor, torch.Tensor], None
        ]
        | None = None,
    ) -> PointForceEnvironmentStep:
        if self.control_step_index >= self.control_step_count:
            raise RuntimeError("Episode is complete; call reset before another step.")
        include_transition = self.active.clone()
        action_is_finite = torch.isfinite(normalized_action).all(dim=-1)
        force = self.physical_force(normalized_action)
        components = self._empty_components()
        done_this_control = torch.zeros_like(self.active)
        success_this_control = torch.zeros_like(self.active)
        non_tip_this_control = torch.zeros_like(self.active)
        invalid_tip_this_control = torch.zeros_like(self.active)
        timeout_this_control = torch.zeros_like(self.active)
        failure_this_control = self.active & ~action_is_finite
        components["numerical_failure"] -= (
            self.reward_weights.numerical_failure
            * failure_this_control.to(self.dtype)
        )
        self.failed |= failure_this_control
        self.active &= ~failure_this_control
        done_this_control |= failure_this_control

        for _ in range(self.physics_steps_per_control):
            from .training_control import check_training_stop
            check_training_stop()
            active_before = self.active.clone()
            previous = self.state
            applied_force = torch.where(
                active_before[:, None],
                force,
                self.hover_force_world_n,
            )
            transition = self.model.step_runtime(
                previous, applied_force, self.physics_dt_s
            )
            candidate = transition.state
            finite = (
                torch.isfinite(candidate.positions_m).reshape(self.batch_size, -1).all(dim=1)
                & torch.isfinite(candidate.velocities_m_s)
                .reshape(self.batch_size, -1)
                .all(dim=1)
            )
            bounded = (
                candidate.positions_m.abs().reshape(self.batch_size, -1).max(dim=1).values
                <= self.numerical_position_limit_m
            ) & (
                torch.linalg.vector_norm(candidate.velocities_m_s, dim=-1)
                .max(dim=1)
                .values
                <= self.numerical_speed_limit_m_s
            )
            failed_now = active_before & ~(finite & bounded)
            self.episode_nonfinite |= active_before & ~finite
            self.episode_position_limit |= active_before & finite & (candidate.positions_m.abs().flatten(1).amax(-1) > self.numerical_position_limit_m)
            self.episode_speed_limit |= active_before & finite & (candidate.velocities_m_s.norm(dim=-1).amax(-1) > self.numerical_speed_limit_m_s)
            accepted = active_before & finite & bounded
            self.state = _replace_state_rows(candidate, previous, accepted)
            components["numerical_failure"] -= (
                self.reward_weights.numerical_failure * failed_now.to(self.dtype)
            )
            self.failed |= failed_now
            self.active &= ~failed_now
            done_this_control |= failed_now
            failure_this_control |= failed_now

            # Keep control flow on the GPU. Inactive rows are frozen by
            # _replace_state_rows and all state/reward updates use tensor masks.
            positions = self.state.positions_m
            velocities = self.state.velocities_m_s
            tip_distance = torch.linalg.vector_norm(
                positions[:, -1] - self.target, dim=-1
            )
            self.episode_minimum_tip_distance = torch.minimum(
                self.episode_minimum_tip_distance, tip_distance
            )
            normalized_progress = (
                (self.initial_tip_distance - tip_distance)
                / self.initial_tip_distance.clamp_min(torch.finfo(self.dtype).eps)
            ).clamp(0.0, 1.0)
            improved_progress = (
                torch.maximum(self.best_progress, normalized_progress)
                - self.best_progress
            ) * accepted.to(self.dtype)
            self.best_progress = torch.maximum(
                self.best_progress, normalized_progress * accepted.to(self.dtype)
            )
            components["progress"] += (
                self.reward_weights.progress * improved_progress
            )

            tip_velocity = velocities[:, -1]
            relative_tip_velocity = tip_velocity - velocities[:, 0]
            relative_tip_directed_speed = (
                relative_tip_velocity * self.desired_direction
            ).sum(dim=-1)
            strike_quality = near_target_speed_quality(
                tip_distance,
                (tip_velocity*self.desired_direction).sum(-1) if self.speed_shaping_reference=='world' else relative_tip_directed_speed,
                proximity_scale_m=self.reward_weights.proximity_scale_m,
                directed_speed_cap_m_s=(
                    self.reward_weights.directed_speed_cap_m_s
                ),
            )
            if self.speed_shaping_reference == 'world_and_attachment_relative':
                strike_quality = near_target_world_relative_quality(tip_distance,
                    (tip_velocity * self.desired_direction).sum(-1), relative_tip_directed_speed,
                    proximity_scale_m=self.reward_weights.proximity_scale_m,
                    world_cap_m_s=self.reward_weights.directed_speed_cap_m_s,
                    relative_cap_m_s=self.relative_speed_cap_m_s)
            improved_strike = (
                torch.maximum(self.best_strike_quality, strike_quality)
                - self.best_strike_quality
            ) * accepted.to(self.dtype)
            self.best_strike_quality = torch.maximum(
                self.best_strike_quality,
                strike_quality * accepted.to(self.dtype),
            )
            components["strike_quality"] += (
                self.reward_weights.strike_quality_improvement
                * improved_strike
            )

            root_displacement = positions[:, 0] - self.initial_root_position
            root_displacement_norm = torch.linalg.vector_norm(
                root_displacement, dim=-1
            )
            displacement_cost = excursion_cost(root_displacement_norm,
                self.displacement_allowance_m, self.reward_weights.displacement_cost_scale_m)
            active_dt = self.physics_dt_s * accepted.to(self.dtype)
            self.episode_point_displacement_integral_m_s += (
                root_displacement_norm * active_dt
            )
            displacement_cost_integral_increment = displacement_cost * active_dt
            self.episode_point_displacement_cost_integral_s += (
                displacement_cost_integral_increment
            )
            components["point_displacement_integral"] -= (
                self.reward_weights.point_displacement_integral
                * displacement_cost_integral_increment
            )
            previous_maximum_cost = excursion_cost(self.episode_maximum_point_displacement,
                self.displacement_allowance_m, self.reward_weights.displacement_cost_scale_m)
            self.episode_maximum_point_displacement = torch.maximum(
                self.episode_maximum_point_displacement,
                root_displacement_norm * accepted.to(self.dtype),
            )
            maximum_cost = excursion_cost(self.episode_maximum_point_displacement,
                self.displacement_allowance_m, self.reward_weights.displacement_cost_scale_m)
            components['maximum_displacement'] -= self.reward_weights.maximum_displacement * (maximum_cost-previous_maximum_cost)
            forward_displacement = (
                root_displacement * self.desired_direction
            ).sum(dim=-1)
            self.peak_forward_displacement = torch.maximum(
                self.peak_forward_displacement,
                forward_displacement * accepted.to(self.dtype),
            )

            marker_previous = previous.positions_m[:, self.marker_indices]
            marker_current = positions[:, self.marker_indices]
            crossing = _segment_enters_sphere(
                marker_previous,
                marker_current,
                self.target,
                self.target_radius_m,
            ) & accepted[:, None]
            tip_entry = crossing[:, -1]
            other_entry = crossing[:, :-1].any(dim=-1)
            non_tip_entry = other_entry & (
                ~tip_entry | torch.full_like(tip_entry, self.tip_must_enter_first)
            )
            endpoint_entry = tip_entry & ~other_entry
            first_entry=endpoint_entry & ~torch.isfinite(self.episode_first_tip_directed_speed)
            directed=(tip_velocity*self.desired_direction).sum(-1)
            angle=torch.rad2deg(torch.acos((directed/tip_velocity.norm(dim=-1).clamp_min(1e-12)).clamp(-1,1)))
            self.episode_first_tip_directed_speed=torch.where(first_entry,directed,self.episode_first_tip_directed_speed)
            self.episode_first_tip_angle=torch.where(first_entry,angle,self.episode_first_tip_angle)

            success_now = (
                endpoint_entry
                & ~self.episode_non_tip_first
                & ~self.episode_success
                & ~(self.episode_invalid_tip_entry & self.first_contact_only)
                & tip_velocity_strike_gate(
                    tip_velocity,
                    self.desired_direction,
                    minimum_directed_speed_m_s=(
                        self.minimum_directed_speed_m_s
                    ),
                    maximum_direction_error_deg=(
                        self.maximum_tip_velocity_direction_error_deg
                    ),
                )
            )
            non_tip_now = non_tip_entry & ~self.episode_non_tip_first
            invalid_tip_now = (
                endpoint_entry & ~success_now & ~self.episode_invalid_tip_entry
            )

            components["non_tip_first"] -= (
                self.reward_weights.non_tip_first
                * non_tip_now.to(self.dtype)
            )
            components["invalid_tip_entry"] -= (
                self.reward_weights.invalid_tip_entry
                * invalid_tip_now.to(self.dtype)
            )
            point_forward_speed = (
                velocities[:, 0] * self.desired_direction
            ).sum(dim=-1)
            relative_tip_forward_speed = (
                relative_tip_velocity * self.desired_direction
            ).sum(dim=-1)
            forward_return = forward_return_quality(
                self.peak_forward_displacement,
                forward_displacement,
                excursion_scale_m=(
                    self.reward_weights.forward_excursion_scale_m
                ),
            )
            release = release_quality(
                self.peak_forward_displacement,
                point_forward_speed,
                relative_tip_forward_speed,
                excursion_scale_m=(
                    self.reward_weights.forward_excursion_scale_m
                ),
                backward_speed_scale_m_s=(
                    self.reward_weights.point_backward_speed_scale_m_s
                ),
                tip_speed_scale_m_s=(
                    self.reward_weights.relative_tip_forward_speed_scale_m_s
                ),
            )
            components["success"] += (
                self.reward_weights.success * success_now.to(self.dtype)
            )
            components["forward_return"] += (
                self.reward_weights.success_forward_return
                * forward_return
                * success_now.to(self.dtype)
            )
            components["release"] += (
                self.reward_weights.success_release
                * release
                * success_now.to(self.dtype)
            )
            components["terminal_displacement"] -= (
                self.reward_weights.terminal_displacement
                * displacement_cost
                * success_now.to(self.dtype)
            )
            self.episode_terminal_point_displacement = torch.where(
                success_now,
                root_displacement_norm,
                self.episode_terminal_point_displacement,
            )
            hit_time = (self.physics_step_index + 1) * self.physics_dt_s
            self.episode_hit_time_s = torch.where(
                success_now,
                torch.full_like(self.episode_hit_time_s, hit_time),
                self.episode_hit_time_s,
            )
            self.episode_success |= success_now
            self.episode_impact_speed = torch.where(success_now,
                tip_velocity.norm(dim=-1),self.episode_impact_speed)
            self.episode_hit_tip_directed_speed = torch.where(success_now, directed,
                self.episode_hit_tip_directed_speed)
            self.episode_hit_relative_tip_directed_speed = torch.where(success_now, relative_tip_directed_speed,
                self.episode_hit_relative_tip_directed_speed)
            self.episode_hit_attachment_directed_speed = torch.where(success_now, point_forward_speed,
                self.episode_hit_attachment_directed_speed)
            self.episode_non_tip_first |= non_tip_now
            self.episode_invalid_tip_entry |= invalid_tip_now
            success_this_control |= success_now
            non_tip_this_control |= non_tip_now
            invalid_tip_this_control |= invalid_tip_now
            if not getattr(self, 'ignore_contact_termination', False):
                done_this_control |= success_now
                self.active &= ~success_now
            if self.first_contact_only and not getattr(self, 'ignore_contact_termination', False) and getattr(self, 'terminate_on_invalid_contact', True):
                invalid_contact = non_tip_now | invalid_tip_now
                done_this_control |= invalid_contact
                self.active &= ~invalid_contact

            if physics_trace_callback is not None:
                reaction = torch.where(
                    accepted[:, None],
                    transition.forces.effective_cable_reaction_on_point_world_n,
                    torch.zeros_like(
                        transition.forces.effective_cable_reaction_on_point_world_n
                    ),
                )
                physics_trace_callback(self.state, applied_force, reaction)

            components["time"] -= (
                self.reward_weights.time_per_s
                * self.physics_dt_s
                * active_before.to(self.dtype)
            )
            self.physics_step_index += 1
            # Replay can terminate at the exact hit step. Training keeps its
            # fixed-size, synchronization-free rollout by leaving this off.
            if stop_when_all_done and not bool(self.active.any()):
                break

        self.control_step_index += 1
        if self.control_step_index >= self.control_step_count:
            timeout_this_control = self.active.clone()
            components["timeout"] -= (
                self.reward_weights.timeout
                * timeout_this_control.to(self.dtype)
            )
            self.episode_timed_out |= timeout_this_control
            done_this_control |= timeout_this_control
            self.active[:] = False

        component_object = PointForceRewardComponents(
            **{name: value[:, None] for name, value in components.items()}
        )
        reward = component_object.total
        self.episode_reward += reward[:, 0]
        for name,value in components.items():
            self.episode_component_sums[name] += value
        return PointForceEnvironmentStep(
            next_observation=self.observe(),
            reward=reward,
            done=done_this_control[:, None].to(self.dtype),
            include_transition=include_transition,
            newly_successful=success_this_control,
            newly_non_tip_first=non_tip_this_control,
            newly_invalid_tip_entry=invalid_tip_this_control,
            newly_timed_out=timeout_this_control,
            numerical_failure=failure_this_control,
            commanded_force_world_n=force,
            components=component_object,
        )
