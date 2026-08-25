"""CUDA-vectorized goal-conditioned drone-whip learning environment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from cable_twin.shared.dder import DderState

from .simulator import DroneCableState, TensorRollout, WhipSimulator


REFERENCE_CONTROL_INTERVAL_S = 0.10


@dataclass(frozen=True, slots=True)
class TaskDistribution:
    """Three-dimensional cable-tip impact task."""

    horizontal_distance_min_m: float = 0.55
    horizontal_distance_max_m: float = 1.20
    target_azimuth_min_deg: float = -180.0
    target_azimuth_max_deg: float = 180.0
    target_height_offset_min_m: float = -0.20
    target_height_offset_max_m: float = 0.00
    desired_impact_azimuth_offset_min_deg: float = 0.0
    desired_impact_azimuth_offset_max_deg: float = 0.0
    desired_impact_elevation_deg: float = 0.0
    desired_impact_elevation_min_deg: float | None = None
    desired_impact_elevation_max_deg: float | None = None
    minimum_impact_speed_min_m_s: float = 1.5
    minimum_impact_speed_max_m_s: float = 1.5
    hit_tolerance_m: float = 0.05
    impact_angle_deg: float = 35.0
    # Retained in artifacts for historical replay.  The new goal is defined by
    # target contact and impact velocity, so cable extension is not a success
    # condition.
    minimum_extension_ratio: float = 0.0
    drone_keepout_radius_m: float = 0.25
    strike_accuracy_decay_m_inv: float = 2.5
    # Deprecated near-miss term retained for old checkpoint compatibility.
    strike_event_reward: float = 0.0
    progress_reward: float = 5.0
    success_reward: float = 100.0
    safety_penalty: float = 10.0
    safety_margin_penalty: float = 0.25
    time_penalty: float = 0.01
    action_effort_penalty: float = 0.002
    action_change_penalty: float = 0.0
    impact_drone_displacement_penalty: float = 2.0

    def __post_init__(self) -> None:
        finite = tuple(
            float(value)
            for value in (
                self.horizontal_distance_min_m,
                self.horizontal_distance_max_m,
                self.target_azimuth_min_deg,
                self.target_azimuth_max_deg,
                self.target_height_offset_min_m,
                self.target_height_offset_max_m,
                self.desired_impact_azimuth_offset_min_deg,
                self.desired_impact_azimuth_offset_max_deg,
                self.desired_impact_elevation_deg,
                self.minimum_impact_speed_min_m_s,
                self.minimum_impact_speed_max_m_s,
                self.hit_tolerance_m,
                self.impact_angle_deg,
                self.minimum_extension_ratio,
                self.drone_keepout_radius_m,
                self.strike_accuracy_decay_m_inv,
                self.strike_event_reward,
                self.progress_reward,
                self.success_reward,
                self.safety_penalty,
                self.safety_margin_penalty,
                self.time_penalty,
                self.action_effort_penalty,
                self.action_change_penalty,
                self.impact_drone_displacement_penalty,
            )
        )
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("SAC task and reward settings must be finite.")
        if not 0.0 < self.horizontal_distance_min_m <= self.horizontal_distance_max_m:
            raise ValueError("Horizontal target distance range is invalid.")
        if not (
            -180.0
            <= self.target_azimuth_min_deg
            <= self.target_azimuth_max_deg
            <= 180.0
        ):
            raise ValueError("Target azimuths must be an ordered range in [-180, 180].")
        if not self.target_height_offset_min_m <= self.target_height_offset_max_m:
            raise ValueError("Target-height offsets must be an ordered range.")
        if not (
            -180.0
            <= self.desired_impact_azimuth_offset_min_deg
            <= self.desired_impact_azimuth_offset_max_deg
            <= 180.0
        ):
            raise ValueError(
                "Impact-direction azimuth offsets must be ordered in [-180, 180]."
            )
        if not (
            0.0
            < self.minimum_impact_speed_min_m_s
            <= self.minimum_impact_speed_max_m_s
        ):
            raise ValueError("Impact-speed range is invalid.")
        if not 0.0 < self.impact_angle_deg < 90.0:
            raise ValueError("Impact angle must be between zero and 90 degrees.")
        if not -89.0 <= self.desired_impact_elevation_deg <= 89.0:
            raise ValueError("Desired impact elevation must be between -89 and 89 degrees.")
        elevation_min = (
            self.desired_impact_elevation_deg
            if self.desired_impact_elevation_min_deg is None
            else float(self.desired_impact_elevation_min_deg)
        )
        elevation_max = (
            self.desired_impact_elevation_deg
            if self.desired_impact_elevation_max_deg is None
            else float(self.desired_impact_elevation_max_deg)
        )
        if not (
            math.isfinite(elevation_min)
            and math.isfinite(elevation_max)
            and -89.0 <= elevation_min <= elevation_max <= 89.0
        ):
            raise ValueError("Impact elevations must be ordered in [-89, 89].")
        if self.hit_tolerance_m <= 0.0:
            raise ValueError("Hit tolerance must be positive.")
        if not 0.0 <= self.minimum_extension_ratio <= 1.0:
            raise ValueError("Minimum cable extension ratio must be in [0, 1].")
        if self.drone_keepout_radius_m <= 0.0:
            raise ValueError("Drone keepout radius must be positive.")
        if self.strike_accuracy_decay_m_inv <= 0.0:
            raise ValueError("Strike-accuracy decay must be positive.")
        reward_magnitudes = (
            self.strike_event_reward,
            self.progress_reward,
            self.success_reward,
            self.safety_penalty,
            self.safety_margin_penalty,
            self.time_penalty,
            self.action_effort_penalty,
            self.action_change_penalty,
            self.impact_drone_displacement_penalty,
        )
        if any(value < 0.0 for value in reward_magnitudes):
            raise ValueError("Reward magnitudes must be non-negative.")
        if self.success_reward <= self.strike_event_reward:
            raise ValueError(
                "Valid-hit reward must exceed the maximum near-miss event reward."
            )

    @property
    def impact_elevation_range_deg(self) -> tuple[float, float]:
        lower = (
            self.desired_impact_elevation_deg
            if self.desired_impact_elevation_min_deg is None
            else float(self.desired_impact_elevation_min_deg)
        )
        upper = (
            self.desired_impact_elevation_deg
            if self.desired_impact_elevation_max_deg is None
            else float(self.desired_impact_elevation_max_deg)
        )
        return lower, upper


@dataclass(frozen=True, slots=True)
class EnvironmentStep:
    transition_observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    success: torch.Tensor
    strike_attempt: torch.Tensor
    unsafe: torch.Tensor
    minimum_tip_error_m: torch.Tensor
    directional_tip_speed_m_s: torch.Tensor
    relative_cable_kinetic_energy_j: torch.Tensor
    impact_drone_displacement_m: torch.Tensor


class VectorWhipEnvironment:
    """Batched episodic environment with goal-conditioned observations.

    The policy acts in an episode-fixed target frame. Its acceleration is held
    for one control interval while DDER advances at the physics rate. Target
    contact is tested at every physics frame, so a fast cable crossing cannot
    disappear between policy actions.
    """

    def __init__(
        self,
        simulator: WhipSimulator,
        environment_count: int,
        episode_horizon_s: float,
        task: TaskDistribution = TaskDistribution(),
        *,
        initial_drone_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: int = 42,
    ) -> None:
        if environment_count < 1:
            raise ValueError("SAC requires at least one parallel environment.")
        steps = episode_horizon_s / simulator.settings.control_interval_s
        if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Episode horizon must be an integer number of actions.")
        if seed < 0:
            raise ValueError("Environment seed must be non-negative.")
        self.simulator = simulator
        self.environment_count = int(environment_count)
        self.maximum_steps = int(round(steps))
        control_interval_s = simulator.settings.control_interval_s
        # The task weights were originally specified at 10 Hz.  Preserve their
        # physical-time meaning when the policy rate changes.  State costs are
        # integrated in time; the squared command-difference cost is scaled as
        # a discrete approximation to the integral of squared command rate.
        self.per_step_reward_scale = (
            control_interval_s / REFERENCE_CONTROL_INTERVAL_S
        )
        self.action_change_reward_scale = (
            REFERENCE_CONTROL_INTERVAL_S / control_interval_s
        )
        self.task = task
        self.device = simulator.device
        self.dtype = simulator.dtype
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        initial = simulator.initial_state(initial_drone_position_m)
        self._initial = WhipSimulator._repeat_state(initial, self.environment_count)
        self.state = self._clone_state(self._initial)
        self.start_drone_position_m = self.state.drone_position_m.clone()
        self.target_position_m = torch.zeros_like(self.state.drone_position_m)
        self.target_frame = torch.eye(
            3, dtype=self.dtype, device=self.device
        )[None].repeat(self.environment_count, 1, 1)
        self.minimum_impact_speed_m_s = torch.zeros(
            self.environment_count, dtype=self.dtype, device=self.device
        )
        self.hit_tolerance_m = torch.zeros_like(self.minimum_impact_speed_m_s)
        self.impact_angle_deg = torch.zeros_like(self.minimum_impact_speed_m_s)
        self.desired_impact_direction_frame = torch.zeros_like(
            self.state.drone_position_m
        )
        self.step_index = torch.zeros(
            self.environment_count, dtype=torch.long, device=self.device
        )
        self.previous_action = torch.zeros(
            (self.environment_count, 3), dtype=self.dtype, device=self.device
        )
        self.peak_relative_cable_energy_j = torch.zeros(
            self.environment_count, dtype=self.dtype, device=self.device
        )
        masses = self.simulator.snapshot.model.parameters.vertex_masses_kg
        if masses is None:
            raise ValueError("Nominal SAC requires explicit DER vertex masses.")
        self.vertex_masses_kg = torch.tensor(
            masses, dtype=self.dtype, device=self.device
        )
        self.reset_done(
            torch.ones(self.environment_count, dtype=torch.bool, device=self.device)
        )

    @property
    def observation_size(self) -> int:
        # Full 3-D state plus explicit tip target/impact vectors, time, and
        # action.  The explicit tip error is redundant in principle but is
        # the task-space quantity on which the strike decision depends.
        return 6 * self.simulator.snapshot.node_count + 22

    @property
    def action_size(self) -> int:
        # Acceleration in the target-aligned forward, lateral, and vertical axes.
        return 3

    @staticmethod
    def _clone_state(state: DroneCableState) -> DroneCableState:
        return DroneCableState(
            state.drone_position_m.clone(),
            state.drone_velocity_m_s.clone(),
            DderState(
                state.cable.positions_m.clone(),
                state.cable.velocities_m_s.clone(),
            ),
        )

    def _uniform(self, count: int, lower: float, upper: float) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(
            count,
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )

    def reset_done(self, mask: torch.Tensor) -> torch.Tensor:
        active = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if active.shape != (self.environment_count,):
            raise ValueError("Environment reset mask has the wrong shape.")
        count = int(torch.count_nonzero(active).detach().cpu())
        if count == 0:
            return self.observation()
        selector = active[:, None]
        cable_selector = active[:, None, None]
        self.state = DroneCableState(
            torch.where(selector, self._initial.drone_position_m, self.state.drone_position_m),
            torch.where(selector, self._initial.drone_velocity_m_s, self.state.drone_velocity_m_s),
            DderState(
                torch.where(
                    cable_selector,
                    self._initial.cable.positions_m,
                    self.state.cable.positions_m,
                ),
                torch.where(
                    cable_selector,
                    self._initial.cable.velocities_m_s,
                    self.state.cable.velocities_m_s,
                ),
            ),
        )
        start = self._initial.drone_position_m[active]
        angle = torch.deg2rad(
            self._uniform(
                count,
                self.task.target_azimuth_min_deg,
                self.task.target_azimuth_max_deg,
            )
        )
        radius = self._uniform(
            count,
            self.task.horizontal_distance_min_m,
            self.task.horizontal_distance_max_m,
        )
        height = self._uniform(
            count,
            self.task.target_height_offset_min_m,
            self.task.target_height_offset_max_m,
        )
        forward = torch.stack((torch.cos(angle), torch.sin(angle), torch.zeros_like(angle)), dim=1)
        up = torch.zeros_like(forward)
        up[:, 2] = 1.0
        lateral = torch.linalg.cross(up, forward, dim=1)
        frame = torch.stack((forward, lateral, up), dim=1)
        target = start + radius[:, None] * forward
        target[:, 2] += height
        self.start_drone_position_m[active] = start
        self.target_position_m[active] = target
        self.target_frame[active] = frame
        self.minimum_impact_speed_m_s[active] = self._uniform(
            count,
            self.task.minimum_impact_speed_min_m_s,
            self.task.minimum_impact_speed_max_m_s,
        )
        self.hit_tolerance_m[active] = self.task.hit_tolerance_m
        self.impact_angle_deg[active] = self.task.impact_angle_deg
        impact_yaw = torch.deg2rad(
            self._uniform(
                count,
                self.task.desired_impact_azimuth_offset_min_deg,
                self.task.desired_impact_azimuth_offset_max_deg,
            )
        )
        elevation_min, elevation_max = self.task.impact_elevation_range_deg
        impact_elevation = torch.deg2rad(
            self._uniform(count, elevation_min, elevation_max)
        )
        cos_elevation = torch.cos(impact_elevation)
        self.desired_impact_direction_frame[active] = torch.stack(
            (
                cos_elevation * torch.cos(impact_yaw),
                cos_elevation * torch.sin(impact_yaw),
                torch.sin(impact_elevation),
            ),
            dim=1,
        )
        self.step_index[active] = 0
        self.previous_action[active] = 0.0
        self.peak_relative_cable_energy_j[active] = 0.0
        return self.observation()

    def _to_frame(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.ndim == 2:
            return torch.einsum("bij,bj->bi", self.target_frame, vectors)
        if vectors.ndim == 3:
            return torch.einsum("bij,bnj->bni", self.target_frame, vectors)
        raise ValueError("Target-frame conversion expects Bx3 or BxNx3.")

    def _to_world(self, vectors: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bij,bi->bj", self.target_frame, vectors)

    def target_beyond_initial_cable_reach(self) -> torch.Tensor:
        """Return targets outside the straight cable sphere at episode start.

        This is an evaluation stratum, not a feasibility gate.  The drone may
        translate and dynamically whip the cable to reach these targets.
        """

        initial_attachment = self._initial.cable.positions_m[:, 0]
        distance = torch.linalg.vector_norm(
            self.target_position_m - initial_attachment, dim=1
        )
        return distance > self.simulator.snapshot.cable_length_m

    def observation(self) -> torch.Tensor:
        cable = self.state.cable
        relative_cable_position = self._to_frame(
            cable.positions_m - cable.positions_m[:, :1]
        )
        relative_cable_velocity = self._to_frame(
            cable.velocities_m_s - self.state.drone_velocity_m_s[:, None]
        )
        drone_displacement = self._to_frame(
            self.state.drone_position_m - self.start_drone_position_m
        )
        drone_velocity = self._to_frame(self.state.drone_velocity_m_s)
        target_relative = self._to_frame(
            self.target_position_m - self.state.drone_position_m
        )
        tip_to_target = self._to_frame(
            self.target_position_m - cable.positions_m[:, -1]
        )
        length = self.simulator.snapshot.cable_length_m
        speed = self.simulator.settings.maximum_speed_m_s
        remaining = 1.0 - self.step_index.to(self.dtype) / self.maximum_steps
        observation = torch.cat(
            (
                relative_cable_position.reshape(self.environment_count, -1) / length,
                relative_cable_velocity.reshape(self.environment_count, -1) / speed,
                drone_displacement / length,
                drone_velocity / speed,
                target_relative / length,
                tip_to_target / length,
                self.desired_impact_direction_frame,
                (self.minimum_impact_speed_m_s / speed)[:, None],
                (self.hit_tolerance_m / length)[:, None],
                (self.impact_angle_deg / 90.0)[:, None],
                remaining[:, None],
                self.previous_action,
            ),
            dim=1,
        )
        if observation.shape != (self.environment_count, self.observation_size):
            raise RuntimeError("Goal-conditioned SAC observation has the wrong shape.")
        return observation

    def _observation_slices(self) -> dict[str, slice]:
        node_values = 3 * self.simulator.snapshot.node_count
        offset = 0
        layout: dict[str, slice] = {}
        for name, width in (
            ("cable_position", node_values),
            ("cable_velocity", node_values),
            ("drone_displacement", 3),
            ("drone_velocity", 3),
            ("target_relative", 3),
            ("tip_to_target", 3),
            ("impact_direction", 3),
            ("impact_speed", 1),
            ("hit_tolerance", 1),
            ("impact_angle", 1),
            ("remaining_time", 1),
            ("previous_action", 3),
        ):
            layout[name] = slice(offset, offset + width)
            offset += width
        if offset != self.observation_size:
            raise RuntimeError("SAC observation layout is internally inconsistent.")
        return layout

    def _achieved_tip_goal(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tip position from episode start and velocity in episode frame."""

        layout = self._observation_slices()
        values = torch.as_tensor(observation, dtype=self.dtype, device=self.device)
        cable_length = self.simulator.snapshot.cable_length_m
        speed_scale = self.simulator.settings.maximum_speed_m_s
        node_count = self.simulator.snapshot.node_count
        cable_position = values[:, layout["cable_position"]].reshape(
            -1, node_count, 3
        )
        cable_velocity = values[:, layout["cable_velocity"]].reshape(
            -1, node_count, 3
        )
        drone_displacement = values[:, layout["drone_displacement"]]
        drone_velocity = values[:, layout["drone_velocity"]]
        attachment_drop = torch.tensor(
            (0.0, 0.0, -self.simulator.settings.attachment_drop_m),
            dtype=self.dtype,
            device=self.device,
        )
        tip_position = (
            drone_displacement * cable_length
            + attachment_drop[None]
            + cable_position[:, -1] * cable_length
        )
        tip_velocity = (
            drone_velocity + cable_velocity[:, -1]
        ) * speed_scale
        return tip_position, tip_velocity

    def _with_goal(
        self,
        observation: torch.Tensor,
        target_from_start_m: torch.Tensor,
        impact_velocity_m_s: torch.Tensor,
    ) -> torch.Tensor:
        """Relabel observations without changing their episode coordinate frame."""

        layout = self._observation_slices()
        relabeled = observation.clone()
        cable_length = self.simulator.snapshot.cable_length_m
        speed_scale = self.simulator.settings.maximum_speed_m_s
        tip_position, _ = self._achieved_tip_goal(observation)
        drone_displacement = (
            observation[:, layout["drone_displacement"]] * cable_length
        )
        impact_speed = torch.linalg.vector_norm(
            impact_velocity_m_s, dim=1, keepdim=True
        )
        impact_direction = impact_velocity_m_s / torch.clamp(
            impact_speed, min=1.0e-8
        )
        relabeled[:, layout["target_relative"]] = (
            target_from_start_m - drone_displacement
        ) / cable_length
        relabeled[:, layout["tip_to_target"]] = (
            target_from_start_m - tip_position
        ) / cable_length
        relabeled[:, layout["impact_direction"]] = impact_direction
        # A relabeled strike asks for essentially the achieved speed.  The
        # small margin avoids turning floating-point interpolation into a
        # false failure at the terminal transition.
        relabeled[:, layout["impact_speed"]] = (
            0.95 * impact_speed / speed_scale
        )
        return relabeled

    def _score_observation_transitions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        unsafe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score relabeled transitions using the same physical hit contract.

        The original episode's target-dependent unsafe flag is deliberately
        not reused after HER changes the goal. Safety is reconstructed from
        the stored endpoint states; the argument remains in the signature so
        existing episode-buffer callers stay source-compatible.
        """

        layout = self._observation_slices()
        cable_length = self.simulator.snapshot.cable_length_m
        speed_scale = self.simulator.settings.maximum_speed_m_s
        tip_before, tip_velocity_before = self._achieved_tip_goal(observation)
        tip_after, tip_velocity_after = self._achieved_tip_goal(next_observation)
        tip_to_target_before = observation[:, layout["tip_to_target"]] * cable_length
        tip_to_target_after = next_observation[:, layout["tip_to_target"]] * cable_length
        target = tip_after + tip_to_target_after
        error_before = torch.linalg.vector_norm(tip_to_target_before, dim=1)
        error_after = torch.linalg.vector_norm(tip_to_target_after, dim=1)
        tolerance = (
            next_observation[:, layout["hit_tolerance"]].squeeze(1)
            * cable_length
        )
        direction = next_observation[:, layout["impact_direction"]]
        required_speed = (
            next_observation[:, layout["impact_speed"]].squeeze(1)
            * speed_scale
        )
        tip_delta = tip_after - tip_before
        tip_delta_norm_squared = torch.sum(torch.square(tip_delta), dim=1)
        contact_fraction = torch.clamp(
            torch.sum((target - tip_before) * tip_delta, dim=1)
            / torch.clamp(tip_delta_norm_squared, min=1.0e-12),
            min=0.0,
            max=1.0,
        )
        contact_position = tip_before + contact_fraction[:, None] * tip_delta
        contact_error = torch.linalg.vector_norm(contact_position - target, dim=1)
        contact_velocity = tip_velocity_before + contact_fraction[:, None] * (
            tip_velocity_after - tip_velocity_before
        )
        directed_speed = torch.sum(contact_velocity * direction, dim=1)
        lateral_speed = torch.linalg.vector_norm(
            contact_velocity - directed_speed[:, None] * direction, dim=1
        )
        angle = (
            next_observation[:, layout["impact_angle"]].squeeze(1) * 90.0
        )
        drone_before = observation[:, layout["drone_displacement"]] * cable_length
        drone_after = next_observation[:, layout["drone_displacement"]] * cable_length
        drone_delta = drone_after - drone_before
        drone_delta_norm_squared = torch.sum(torch.square(drone_delta), dim=1)
        closest_drone_fraction = torch.clamp(
            torch.sum((target - drone_before) * drone_delta, dim=1)
            / torch.clamp(drone_delta_norm_squared, min=1.0e-12),
            min=0.0,
            max=1.0,
        )
        closest_drone = drone_before + closest_drone_fraction[:, None] * drone_delta
        minimum_clearance = torch.linalg.vector_norm(closest_drone - target, dim=1)
        drone_velocity_before = observation[:, layout["drone_velocity"]] * speed_scale
        drone_velocity_after = next_observation[:, layout["drone_velocity"]] * speed_scale
        maximum_drone_speed = torch.maximum(
            torch.linalg.vector_norm(drone_velocity_before, dim=1),
            torch.linalg.vector_norm(drone_velocity_after, dim=1),
        )
        relabeled_unsafe = (
            (minimum_clearance < self.task.drone_keepout_radius_m)
            | (maximum_drone_speed > self.simulator.settings.maximum_speed_m_s)
        )
        attempt = contact_error <= tolerance
        success = (
            attempt
            & ~relabeled_unsafe
            & (directed_speed >= required_speed)
            & (
                lateral_speed
                <= torch.tan(torch.deg2rad(angle))
                * torch.relu(directed_speed)
            )
        )

        def potential(distance: torch.Tensor) -> torch.Tensor:
            return torch.clamp(
                1.0 - distance / (2.0 * cable_length), min=0.0, max=1.0
            )

        progress = self.task.progress_reward * (
            potential(error_after) - potential(error_before)
        )
        effort = torch.sum(torch.square(action), dim=1)
        previous_action = observation[:, layout["previous_action"]]
        change = torch.sum(torch.square(action - previous_action), dim=1)
        clearance_margin = torch.relu(
            (1.5 * self.task.drone_keepout_radius_m - minimum_clearance)
            / max(0.5 * self.task.drone_keepout_radius_m, 1.0e-9)
        )
        speed_ratio = maximum_drone_speed / max(
            self.simulator.settings.maximum_speed_m_s, 1.0e-9
        )
        safety_margin = torch.square(clearance_margin) + 0.25 * torch.pow(
            speed_ratio, 4
        )
        drone_displacement = torch.linalg.vector_norm(
            next_observation[:, layout["drone_displacement"]], dim=1
        )
        reward = (
            progress
            + self.task.success_reward * success.to(self.dtype)
            - self.task.safety_penalty * relabeled_unsafe.to(self.dtype)
            - self.task.safety_margin_penalty
            * self.per_step_reward_scale
            * safety_margin
            - self.task.time_penalty * self.per_step_reward_scale
            - self.task.action_effort_penalty
            * self.per_step_reward_scale
            * effort
            - self.task.action_change_penalty
            * self.action_change_reward_scale
            * change
            - self.task.impact_drone_displacement_penalty
            * success.to(self.dtype)
            * torch.square(drone_displacement)
        )
        return reward, attempt | relabeled_unsafe, success

    def hindsight_relabel_episode(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        unsafe: torch.Tensor,
        *,
        generator: torch.Generator,
        goals_per_episode: int,
        minimum_achieved_speed_m_s: float,
    ) -> tuple[torch.Tensor, ...] | None:
        """Create coherent future-goal HER prefixes from one completed episode."""

        if goals_per_episode < 1:
            return None
        if observations.ndim != 2 or len(observations) != len(actions) + 1:
            raise ValueError("HER episode tensors have inconsistent lengths.")
        _, achieved_velocity = self._achieved_tip_goal(observations[1:])
        achieved_speed = torch.linalg.vector_norm(achieved_velocity, dim=1)
        candidates = torch.nonzero(
            achieved_speed >= minimum_achieved_speed_m_s, as_tuple=False
        ).flatten()
        if len(candidates) == 0:
            return None
        relabeled_observations: list[torch.Tensor] = []
        relabeled_actions: list[torch.Tensor] = []
        relabeled_rewards: list[torch.Tensor] = []
        relabeled_next_observations: list[torch.Tensor] = []
        relabeled_done: list[torch.Tensor] = []
        sample_index = torch.randint(
            len(candidates),
            (goals_per_episode,),
            device=self.device,
            generator=generator,
        )
        for selected in candidates[sample_index]:
            terminal = int(selected.detach().cpu()) + 1
            state_sequence = observations[: terminal + 1]
            target, velocity = self._achieved_tip_goal(
                state_sequence[terminal : terminal + 1]
            )
            target = target.repeat(terminal, 1)
            velocity = velocity.repeat(terminal, 1)
            relabeled = self._with_goal(
                state_sequence[:-1], target, velocity
            )
            relabeled_next = self._with_goal(
                state_sequence[1:], target, velocity
            )
            reward, done, success = self._score_observation_transitions(
                relabeled,
                actions[:terminal],
                relabeled_next,
                unsafe[:terminal],
            )
            # No transition after the chosen achieved goal belongs to this
            # relabeled task, even when numerical tolerances make contact occur
            # one frame earlier.
            first_done = torch.nonzero(done, as_tuple=False).flatten()
            keep = (
                int(first_done[0].detach().cpu()) + 1
                if len(first_done)
                else terminal
            )
            # A high velocity observed while the tip is already inside the
            # relabeled target is not a valid impact example: the physical
            # target would have been contacted earlier.  Keep only achieved
            # goals whose first contact also satisfies the requested vector.
            if len(first_done) and not bool(success[keep - 1].detach().cpu()):
                continue
            done = done[:keep].clone()
            done[-1] = True
            relabeled_observations.append(relabeled[:keep])
            relabeled_actions.append(actions[:keep])
            relabeled_rewards.append(reward[:keep])
            relabeled_next_observations.append(relabeled_next[:keep])
            relabeled_done.append(done)
        if not relabeled_observations:
            return None
        return (
            torch.cat(relabeled_observations),
            torch.cat(relabeled_actions),
            torch.cat(relabeled_rewards),
            torch.cat(relabeled_next_observations),
            torch.cat(relabeled_done),
        )

    def step(self, normalized_action: torch.Tensor) -> EnvironmentStep:
        action = torch.as_tensor(
            normalized_action, dtype=self.dtype, device=self.device
        )
        if action.shape != (self.environment_count, self.action_size):
            raise ValueError("SAC action must have shape environments x 3.")
        if not bool(torch.all(torch.isfinite(action)).detach().cpu()):
            raise ValueError("SAC action must be finite.")
        action_norm = torch.linalg.vector_norm(action, dim=1, keepdim=True)
        action = action * torch.clamp(
            1.0 / torch.clamp(action_norm, min=1.0e-12), max=1.0
        )
        world_action = self._to_world(action)
        acceleration = world_action * self.simulator.settings.maximum_acceleration_m_s2
        with torch.no_grad():
            rollout = self.simulator.rollout(
                self.state,
                acceleration[:, None],
                create_graph=False,
            )
        step = self._score_rollout(rollout, action)
        self.state = rollout.final_state()
        self.step_index += 1
        self.previous_action = action
        self.peak_relative_cable_energy_j = torch.maximum(
            self.peak_relative_cable_energy_j, step[8]
        )
        terminal_observation = self.observation()
        return EnvironmentStep(
            transition_observation=terminal_observation,
            reward=step[0],
            done=step[1],
            success=step[2],
            strike_attempt=step[10].to(torch.bool),
            unsafe=step[3],
            minimum_tip_error_m=step[4],
            directional_tip_speed_m_s=step[5],
            relative_cable_kinetic_energy_j=step[6],
            impact_drone_displacement_m=step[7],
        )

    def _score_rollout(
        self,
        rollout: TensorRollout,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        direction = self._to_world(self.desired_impact_direction_frame)
        tip_frames = torch.cat(
            (self.state.cable.positions_m[:, None, -1], rollout.cable_positions_m[:, :, -1]),
            dim=1,
        )
        tip_velocity_frames = torch.cat(
            (self.state.cable.velocities_m_s[:, None, -1], rollout.cable_velocities_m_s[:, :, -1]),
            dim=1,
        )
        target = self.target_position_m[:, None]
        segment_start = tip_frames[:, :-1]
        segment_delta = tip_frames[:, 1:] - segment_start
        segment_norm_squared = torch.sum(torch.square(segment_delta), dim=2)
        contact_fraction = torch.clamp(
            torch.sum((target - segment_start) * segment_delta, dim=2)
            / torch.clamp(segment_norm_squared, min=1.0e-12),
            min=0.0,
            max=1.0,
        )
        contact_position = segment_start + contact_fraction[:, :, None] * segment_delta
        contact_error = torch.linalg.vector_norm(contact_position - target, dim=2)
        contact = contact_error <= self.hit_tolerance_m[:, None]
        contact_velocity = tip_velocity_frames[:, :-1] + contact_fraction[:, :, None] * (
            tip_velocity_frames[:, 1:] - tip_velocity_frames[:, :-1]
        )
        contact_directed_speed = torch.sum(
            contact_velocity * direction[:, None], dim=2
        )
        contact_lateral_speed = torch.linalg.vector_norm(
            contact_velocity
            - contact_directed_speed[:, :, None] * direction[:, None],
            dim=2,
        )
        attempt = torch.any(contact, dim=1)
        first_contact_frame = torch.argmax(contact.to(torch.int64), dim=1)
        batch_index = torch.arange(self.environment_count, device=self.device)
        event_directed_speed = contact_directed_speed[
            batch_index, first_contact_frame
        ]
        event_lateral_speed = contact_lateral_speed[
            batch_index, first_contact_frame
        ]
        tip = rollout.cable_positions_m[:, :, -1]
        tip_velocity = rollout.cable_velocities_m_s[:, :, -1]
        error = torch.linalg.vector_norm(tip - target, dim=2)
        directed_speed = torch.sum(tip_velocity * direction[:, None], dim=2)
        drone = rollout.drone_positions_m
        drone_speed = torch.linalg.vector_norm(rollout.drone_velocities_m_s, dim=2)
        clearance = torch.linalg.vector_norm(drone - target, dim=2)
        unsafe_by_frame = (
            (clearance < self.task.drone_keepout_radius_m)
            | (drone_speed > self.simulator.settings.maximum_speed_m_s)
        )
        frame_number = torch.arange(drone.shape[1], device=self.device)[None]
        active_until_event = frame_number <= first_contact_frame[:, None]
        active_until_event = torch.where(
            attempt[:, None], active_until_event, torch.ones_like(active_until_event)
        )
        unsafe = torch.any(unsafe_by_frame & active_until_event, dim=1)
        valid_attempt = attempt & ~unsafe
        angle_limit = torch.tan(torch.deg2rad(self.impact_angle_deg))
        success = (
            valid_attempt
            & (event_directed_speed >= self.minimum_impact_speed_m_s)
            & (
                event_lateral_speed
                <= angle_limit * torch.relu(event_directed_speed)
            )
        )

        drone_frames = torch.cat(
            (self.state.drone_position_m[:, None], rollout.drone_positions_m), dim=1
        )
        drone_at_event = drone_frames[:, :-1] + contact_fraction[:, :, None] * (
            drone_frames[:, 1:] - drone_frames[:, :-1]
        )
        drone_at_hit = drone_at_event[batch_index, first_contact_frame]
        hit_drone_displacement = torch.linalg.vector_norm(
            drone_at_hit - self.start_drone_position_m, dim=1
        )
        hit_drone_displacement = torch.where(
            success,
            hit_drone_displacement,
            torch.full_like(hit_drone_displacement, torch.nan),
        )
        normalized_hit_displacement = torch.nan_to_num(
            hit_drone_displacement
            / max(self.simulator.snapshot.cable_length_m, 1.0e-9),
            nan=0.0,
        )
        minimum_error = torch.amin(contact_error, dim=1)
        required_speed = torch.clamp(self.minimum_impact_speed_m_s, min=0.1)
        speed_quality = torch.clamp(
            torch.relu(event_directed_speed) / required_speed, max=1.0
        )
        cone_capacity = torch.tan(torch.deg2rad(self.impact_angle_deg))
        cone_capacity = cone_capacity * torch.clamp(
            torch.relu(event_directed_speed), min=0.25 * required_speed
        )
        vector_quality = torch.exp(
            -math.log(2.0)
            * torch.square(
                event_lateral_speed / torch.clamp(cone_capacity, min=1.0e-6)
            )
        )
        strike_quality = speed_quality * vector_quality
        effort = torch.sum(torch.square(action), dim=1)
        change = torch.sum(torch.square(action - self.previous_action), dim=1)
        relative_velocity = (
            rollout.cable_velocities_m_s - rollout.drone_velocities_m_s[:, :, None]
        )
        relative_energy = 0.5 * torch.sum(
            self.vertex_masses_kg[None, None, :, None]
            * torch.square(relative_velocity),
            dim=(2, 3),
        )
        final_relative_energy = relative_energy[:, -1]
        peak_relative_energy = torch.amax(relative_energy, dim=1)
        clearance_margin = torch.relu(
            (
                1.5 * self.task.drone_keepout_radius_m
                - torch.amin(clearance, dim=1)
            )
            / max(0.5 * self.task.drone_keepout_radius_m, 1.0e-9)
        )
        speed_ratio = torch.amax(drone_speed, dim=1) / max(
            self.simulator.settings.maximum_speed_m_s, 1.0e-9
        )
        safety_margin = (
            torch.square(clearance_margin)
            + 0.25 * torch.pow(speed_ratio, 4)
        )
        cable_length = max(self.simulator.snapshot.cable_length_m, 1.0e-9)
        error_before = torch.linalg.vector_norm(tip_frames[:, 0] - target[:, 0], dim=1)
        error_after = torch.linalg.vector_norm(tip_frames[:, -1] - target[:, 0], dim=1)

        def potential(distance: torch.Tensor) -> torch.Tensor:
            return torch.clamp(1.0 - distance / (2.0 * cable_length), min=0.0, max=1.0)

        progress = self.task.progress_reward * (
            potential(error_after) - potential(error_before)
        )
        reward = (
            progress
            + self.task.strike_event_reward
            * strike_quality
            * valid_attempt.to(self.dtype)
            + self.task.success_reward * success.to(self.dtype)
            - self.task.safety_penalty * unsafe.to(self.dtype)
            - self.task.safety_margin_penalty
            * self.per_step_reward_scale
            * safety_margin
            - self.task.time_penalty * self.per_step_reward_scale
            - self.task.action_effort_penalty
            * self.per_step_reward_scale
            * effort
            - self.task.action_change_penalty
            * self.action_change_reward_scale
            * change
            - self.task.impact_drone_displacement_penalty
            * torch.square(normalized_hit_displacement)
        )
        timeout = self.step_index + 1 >= self.maximum_steps
        done = attempt | unsafe | timeout
        selected_speed = torch.amax(
            torch.where(
                error <= 3.0 * self.hit_tolerance_m[:, None],
                directed_speed,
                torch.full_like(directed_speed, -torch.inf),
            ),
            dim=1,
        )
        selected_speed = torch.where(
            attempt,
            event_directed_speed,
            torch.where(
                torch.isfinite(selected_speed),
                selected_speed,
                torch.amax(directed_speed, dim=1),
            ),
        )
        return (
            reward,
            done,
            success,
            unsafe,
            minimum_error,
            selected_speed,
            final_relative_energy,
            hit_drone_displacement,
            peak_relative_energy,
            strike_quality,
            attempt,
        )
