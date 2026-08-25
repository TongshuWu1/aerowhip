"""Compact goal-conditioned Soft Actor-Critic implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from cable_twin.shared.observation_data import sha256_file

from .model import CableModelSnapshot
from .oracle import ORACLE_SCHEMA
from .reduced import stable_controller_model
from .rl_env import TaskDistribution, VectorWhipEnvironment
from .simulator import SimulationSettings, WhipSimulator


SAC_CHECKPOINT_SCHEMA = "drone_whip_goal_conditioned_her_sac_3d_v10"
LEGACY_SAC_CHECKPOINT_SCHEMAS = frozenset(
    {
        "drone_whip_goal_conditioned_strike_event_sac_3d_v9",
        "drone_whip_goal_conditioned_two_phase_vector_sac_3d_v8",
        "drone_whip_goal_conditioned_two_phase_sac_3d_v7",
        "drone_whip_goal_conditioned_sac_3d_v6",
        "drone_whip_goal_conditioned_prior_replay_sac_3d_v5",
    }
)
TASK_DISTRIBUTION_LABEL = "goal_conditioned_target_and_impact_velocity_distribution"
ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class SacSettings:
    total_transitions: int = 2_500_000
    environment_count: int = 128
    replay_capacity: int = 250_000
    warmup_transitions: int = 100_000
    batch_size: int = 512
    hidden_size: int = 256
    actor_learning_rate: float = 1.0e-4
    critic_learning_rate: float = 3.0e-4
    entropy_learning_rate: float = 1.0e-4
    initial_entropy_coefficient: float = 1.0
    # 0.998 at 50 Hz has essentially the same discount per physical second as
    # the former 0.99 at 10 Hz: 0.998^50 ~= 0.99^10.
    discount: float = 0.998
    target_update_rate: float = 0.005
    gradient_steps_per_iteration: int = 1
    gradient_clip_norm: float = 10.0
    controller_node_count: int = 15
    episode_horizon_s: float = 4.0
    physics_dt_s: float = 0.01
    control_interval_s: float = 0.02
    attachment_drop_m: float = 0.10
    maximum_acceleration_m_s2: float = 20.0
    maximum_speed_m_s: float = 3.0
    log_interval_transitions: int = 50_000
    evaluation_episodes: int = 256
    observation_clip: float = 10.0
    degradation_patience_evaluations: int = 20
    hindsight_goals_per_episode: int = 2
    hindsight_minimum_achieved_speed_m_s: float = 0.25
    seed: int = 42

    def __post_init__(self) -> None:
        integers = (
            self.total_transitions,
            self.environment_count,
            self.replay_capacity,
            self.batch_size,
            self.hidden_size,
            self.gradient_steps_per_iteration,
            self.controller_node_count,
            self.log_interval_transitions,
            self.evaluation_episodes,
            self.degradation_patience_evaluations,
            self.hindsight_goals_per_episode,
        )
        if any(value < 1 for value in integers):
            raise ValueError("SAC counts and network sizes must be positive.")
        if self.replay_capacity < self.batch_size:
            raise ValueError("Replay capacity must be at least one SAC batch.")
        if self.environment_count > self.replay_capacity:
            raise ValueError("Replay capacity must hold one environment step.")
        if self.warmup_transitions < 0 or self.warmup_transitions > self.total_transitions:
            raise ValueError("SAC warmup cannot exceed total transitions.")
        if not 3 <= self.controller_node_count:
            raise ValueError("SAC controller requires at least three cable nodes.")
        finite = (
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.entropy_learning_rate,
            self.initial_entropy_coefficient,
            self.discount,
            self.target_update_rate,
            self.gradient_clip_norm,
            self.episode_horizon_s,
            self.physics_dt_s,
            self.control_interval_s,
            self.attachment_drop_m,
            self.maximum_acceleration_m_s2,
            self.maximum_speed_m_s,
            self.observation_clip,
            self.hindsight_minimum_achieved_speed_m_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite):
            raise ValueError("SAC rates, times, and limits must be finite and positive.")
        if not 0.0 < self.discount < 1.0:
            raise ValueError("SAC discount must be between zero and one.")
        if not 0.0 < self.target_update_rate <= 1.0:
            raise ValueError("SAC target-update rate must be in (0,1].")
        if self.seed < 0:
            raise ValueError("SAC seed must be non-negative.")


def is_supported_sac_checkpoint_schema(value: object) -> bool:
    return value == SAC_CHECKPOINT_SCHEMA or value in LEGACY_SAC_CHECKPOINT_SCHEMAS


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    transitions: int
    completed_episodes: int
    successful_episodes: int
    training_success_rate: float
    training_mean_episode_reward: float
    evaluation_success_rate: float
    evaluation_mean_episode_reward: float
    evaluation_mean_minimum_error_m: float
    evaluation_mean_directional_speed_m_s: float
    evaluation_mean_peak_relative_cable_energy_j: float
    evaluation_mean_hit_drone_displacement_m: float
    evaluation_unsafe_rate: float
    evaluation_within_initial_reach_success_rate: float
    evaluation_beyond_initial_reach_success_rate: float
    best_transitions: int
    stopped_early: bool
    policy_path: Path
    final_policy_path: Path
    log_path: Path
    run_status: str
    latest_policy_path: Path
    held_out_evaluated: bool


@dataclass(frozen=True, slots=True)
class SacTrainingUpdate:
    """One durable, UI-safe SAC training update.

    Missing measurements use ``None`` rather than NaN so the same object can be
    emitted as strict JSON by the command-line runner.
    """

    run_status: str
    transitions: int
    transition_limit: int | None
    endless: bool
    elapsed_s: float
    transitions_per_s: float
    completed_episodes: int
    recent_training_success_rate: float | None
    recent_training_mean_episode_reward: float | None
    checkpoint_evaluation_episodes: int
    validation_success_rate: float | None
    validation_mean_episode_reward: float | None
    validation_mean_minimum_error_m: float | None
    validation_mean_directional_speed_m_s: float | None
    validation_mean_peak_relative_cable_energy_j: float | None
    validation_mean_hit_drone_displacement_m: float | None
    validation_unsafe_rate: float | None
    validation_within_initial_reach_success_rate: float | None
    validation_beyond_initial_reach_success_rate: float | None
    actor_loss: float | None
    critic_loss: float | None
    alpha: float
    best_validation_success_rate: float | None
    best_transitions: int
    best_checkpoint_updated: bool
    best_policy_path: str | None
    latest_policy_path: str


MetricsProgressCallback = Callable[[SacTrainingUpdate], None]


@dataclass(frozen=True, slots=True)
class SacValidationPreview:
    """One deterministic checkpoint-validation episode for UI playback."""

    checkpoint_transitions: int
    completed_training_episodes: int
    validation_seed: int
    time_s: list[float]
    drone_positions_m: list[list[float]]
    cable_positions_m: list[list[list[float]]]
    target_position_m: list[float]
    desired_impact_direction: list[float]
    success: bool
    unsafe: bool
    minimum_tip_error_m: float
    event_type: str = "validation_preview"


ValidationPreviewCallback = Callable[[SacValidationPreview], None]


class SquashedGaussianActor(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, observation_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)
        # Begin with small, zero-mean commands.  Full-scale accelerations remain
        # available, but an untrained actor should not terminate almost every
        # episode by immediately leaving the drone workspace.
        nn.init.uniform_(self.mean.weight, -1.0e-3, 1.0e-3)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -2.0)

    def distribution_parameters(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.backbone(observation)
        mean = self.mean(feature)
        log_std = torch.clamp(self.log_std(feature), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        raw = normal.rsample()
        action, log_determinant = self._unit_ball_squash(raw)
        log_probability = torch.sum(normal.log_prob(raw), dim=1, keepdim=True) - log_determinant
        return action, log_probability

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_parameters(observation)
        return self._unit_ball_squash(mean)[0]

    @staticmethod
    def _unit_ball_squash(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Radial tanh bijection R^3 -> open unit ball and its log Jacobian."""

        radius = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
        squashed_radius = torch.tanh(radius)
        scale = squashed_radius / torch.clamp(radius, min=1.0e-8)
        action = raw * scale
        dimension = raw.shape[1]
        radial_log_jacobian = torch.log(
            torch.clamp(1.0 - squashed_radius.square(), min=1.0e-8)
        )
        tangential_log_jacobian = (dimension - 1) * (
            torch.log(torch.clamp(squashed_radius, min=1.0e-8))
            - torch.log(torch.clamp(radius, min=1.0e-8))
        )
        return action, radial_log_jacobian + tangential_log_jacobian


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size + action_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((observation, action), dim=1))


class ReplayBuffer:
    """Fixed-size device replay buffer; no Python objects per transition."""

    def __init__(
        self,
        capacity: int,
        observation_size: int,
        action_size: int,
        *,
        device: torch.device,
        seed: int,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.observation = torch.empty(
            (capacity, observation_size), dtype=torch.float32, device=device
        )
        self.action = torch.empty((capacity, action_size), dtype=torch.float32, device=device)
        self.reward = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.next_observation = torch.empty_like(self.observation)
        self.done = torch.empty((capacity, 1), dtype=torch.float32, device=device)
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        count = observation.shape[0]
        if count > self.capacity:
            raise ValueError("One SAC transition batch exceeds replay capacity.")
        index = (
            torch.arange(count, device=self.device, dtype=torch.long) + self.position
        ) % self.capacity
        self.observation[index] = observation.detach()
        self.action[index] = action.detach()
        self.reward[index, 0] = reward.detach()
        self.next_observation[index] = next_observation.detach()
        self.done[index, 0] = done.to(torch.float32)
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("Replay buffer does not yet contain one training batch.")
        index = torch.randint(
            self.size,
            (batch_size,),
            device=self.device,
            generator=self.generator,
        )
        return (
            self.observation[index],
            self.action[index],
            self.reward[index],
            self.next_observation[index],
            self.done[index],
        )


class HindsightEpisodeBuffer:
    """GPU episode recorder that emits coherent future-goal HER prefixes."""

    def __init__(
        self,
        environment: VectorWhipEnvironment,
        observation_size: int,
        action_size: int,
        *,
        goals_per_episode: int,
        minimum_achieved_speed_m_s: float,
        seed: int,
    ) -> None:
        self.environment = environment
        self.environment_count = environment.environment_count
        self.maximum_steps = environment.maximum_steps
        self.goals_per_episode = int(goals_per_episode)
        self.minimum_achieved_speed_m_s = float(minimum_achieved_speed_m_s)
        self.device = environment.device
        self.observations = torch.empty(
            (self.environment_count, self.maximum_steps + 1, observation_size),
            dtype=environment.dtype,
            device=self.device,
        )
        self.actions = torch.empty(
            (self.environment_count, self.maximum_steps, action_size),
            dtype=environment.dtype,
            device=self.device,
        )
        self.unsafe = torch.empty(
            (self.environment_count, self.maximum_steps),
            dtype=torch.bool,
            device=self.device,
        )
        self.length = torch.zeros(
            self.environment_count, dtype=torch.long, device=self.device
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
        unsafe: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        batch = torch.arange(self.environment_count, device=self.device)
        if bool(torch.any(self.length >= self.maximum_steps).detach().cpu()):
            raise RuntimeError("HER episode exceeded the configured horizon.")
        self.observations[batch, self.length] = observation.detach()
        self.actions[batch, self.length] = action.detach()
        self.unsafe[batch, self.length] = unsafe.detach()
        self.length += 1
        self.observations[batch, self.length] = next_observation.detach()

        completed = torch.nonzero(done, as_tuple=False).flatten()
        if len(completed) == 0:
            return None
        batches: list[tuple[torch.Tensor, ...]] = []
        for environment_index in completed.tolist():
            episode_length = int(self.length[environment_index].detach().cpu())
            relabeled = self.environment.hindsight_relabel_episode(
                self.observations[environment_index, : episode_length + 1],
                self.actions[environment_index, :episode_length],
                self.unsafe[environment_index, :episode_length],
                generator=self.generator,
                goals_per_episode=self.goals_per_episode,
                minimum_achieved_speed_m_s=self.minimum_achieved_speed_m_s,
            )
            if relabeled is not None:
                batches.append(relabeled)
            self.length[environment_index] = 0
        if not batches:
            return None
        return tuple(
            torch.cat([batch_values[index] for batch_values in batches], dim=0)
            for index in range(5)
        )


@dataclass(frozen=True, slots=True)
class DemonstrationTransitions:
    """Transitions re-simulated under the exact SAC environment contract."""

    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor
    provenance: tuple[dict[str, object], ...]

    @property
    def count(self) -> int:
        return int(self.observation.shape[0])


class DemonstrationReplay:
    """Immutable sample-only replay for verified prior trajectories."""

    def __init__(self, transitions: DemonstrationTransitions, *, seed: int) -> None:
        if transitions.count < 1:
            raise ValueError("Demonstration replay cannot be empty.")
        self.observation = transitions.observation.detach().clone().to(torch.float32)
        self.action = transitions.action.detach().clone().to(torch.float32)
        self.reward = transitions.reward.detach().clone().to(torch.float32)[:, None]
        self.next_observation = (
            transitions.next_observation.detach().clone().to(torch.float32)
        )
        self.done = transitions.done.detach().clone().to(torch.float32)[:, None]
        self.size = transitions.count
        self.device = self.observation.device
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        if batch_size < 1:
            raise ValueError("Demonstration replay sample size must be positive.")
        index = torch.randint(
            self.size,
            (batch_size,),
            device=self.device,
            generator=self.generator,
        )
        return (
            self.observation[index],
            self.action[index],
            self.reward[index],
            self.next_observation[index],
            self.done[index],
        )


def _require_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(
            f"MPC demonstration {name}={actual:.9g} does not match "
            f"SAC {name}={expected:.9g}."
        )


def load_verified_demonstrations(
    paths: tuple[str | Path, ...] | list[str | Path],
    simulator: WhipSimulator,
    settings: SacSettings,
    task: TaskDistribution,
    device: torch.device,
) -> DemonstrationTransitions:
    """Strictly validate and re-simulate feasible MPC trajectories.

    Saved states are never imported.  World-frame acceleration knots are run
    through the exact controller discretization and SAC event detector, which
    creates observations, rewards, and terminal flags in the learning MDP.
    """

    requested = tuple(Path(path).expanduser().resolve() for path in paths)
    if not requested:
        raise ValueError("At least one MPC demonstration path is required.")
    expanded: list[Path] = []
    for source in requested:
        if source.is_dir():
            artifacts = sorted(source.glob("*.npz"))
            if not artifacts:
                raise ValueError(
                    f"MPC demonstration directory contains no NPZ artifacts: {source}"
                )
            expanded.extend(artifacts)
        else:
            expanded.append(source)
    sources = tuple(expanded)
    expected_settings = {
        "horizon_s": settings.episode_horizon_s,
        "physics_dt_s": settings.physics_dt_s,
        "control_interval_s": settings.control_interval_s,
        "attachment_drop_m": settings.attachment_drop_m,
        "maximum_acceleration_m_s2": settings.maximum_acceleration_m_s2,
        "maximum_speed_m_s": settings.maximum_speed_m_s,
    }
    expected_problem = {
        "maximum_tip_error_m": task.hit_tolerance_m,
        "maximum_impact_angle_deg": task.impact_angle_deg,
        "drone_keepout_radius_m": task.drone_keepout_radius_m,
    }

    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    next_observations: list[torch.Tensor] = []
    terminals: list[torch.Tensor] = []
    provenance: list[dict[str, object]] = []
    seen_hashes: set[str] = set()
    expected_control_count = int(
        round(settings.episode_horizon_s / settings.control_interval_s)
    )

    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"MPC demonstration does not exist: {source}")
        artifact_hash = sha256_file(source)
        if artifact_hash in seen_hashes:
            raise ValueError(f"Duplicate MPC demonstration: {source}")
        seen_hashes.add(artifact_hash)
        with np.load(source, allow_pickle=False) as archive:
            schema = str(np.asarray(archive["schema"]).item())
            if schema != ORACLE_SCHEMA:
                raise ValueError(f"Unsupported MPC demonstration schema: {schema}")
            if not bool(np.asarray(archive["feasible"]).item()):
                raise ValueError(f"MPC demonstration is not marked feasible: {source.name}")
            source_hash = str(np.asarray(archive["source_model_sha256"]).item())
            source_nodes = int(np.asarray(archive["source_node_count"]).item())
            controls = np.asarray(archive["controls_m_s2"], dtype=np.float64).copy()
            drone_positions = np.asarray(
                archive["drone_positions_m"], dtype=np.float64
            ).copy()
            target = np.asarray(archive["target_position_m"], dtype=np.float64).copy()
            direction = np.asarray(archive["impact_direction"], dtype=np.float64).copy()
            oracle_settings = json.loads(str(np.asarray(archive["settings_json"]).item()))
            oracle_problem = json.loads(str(np.asarray(archive["problem_json"]).item()))
        if source_hash != simulator.snapshot.sha256:
            raise ValueError(
                f"MPC demonstration {source.name} uses a different controller model."
            )
        if source_nodes != simulator.snapshot.node_count:
            raise ValueError(
                f"MPC demonstration {source.name} has {source_nodes} nodes; "
                f"SAC uses {simulator.snapshot.node_count}."
            )
        if controls.shape != (expected_control_count, 3) or not np.all(
            np.isfinite(controls)
        ):
            raise ValueError("MPC demonstration controls have the wrong shape or values.")
        if (
            drone_positions.ndim != 2
            or drone_positions.shape[1:] != (3,)
            or len(drone_positions) < 1
            or not np.all(np.isfinite(drone_positions))
            or target.shape != (3,)
            or direction.shape != (3,)
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(direction))
        ):
            raise ValueError("MPC demonstration geometry is missing or non-finite.")
        initial_drone = drone_positions[0]
        relative_target = target - initial_drone
        radius = float(np.linalg.norm(relative_target[:2]))
        height = float(relative_target[2])
        azimuth_deg = math.degrees(
            math.atan2(float(relative_target[1]), float(relative_target[0]))
        )
        tolerance = 1.0e-8
        if not (
            task.horizontal_distance_min_m - tolerance
            <= radius
            <= task.horizontal_distance_max_m + tolerance
            and task.target_height_offset_min_m - tolerance
            <= height
            <= task.target_height_offset_max_m + tolerance
            and task.target_azimuth_min_deg - tolerance
            <= azimuth_deg
            <= task.target_azimuth_max_deg + tolerance
        ):
            raise ValueError(
                f"MPC demonstration target is outside the SAC target distribution: "
                f"radius={radius:.3f}m height={height:.3f}m "
                f"azimuth={azimuth_deg:.1f}deg."
            )
        demonstration_speed = float(oracle_problem["minimum_impact_speed_m_s"])
        if not (
            task.minimum_impact_speed_min_m_s - tolerance
            <= demonstration_speed
            <= task.minimum_impact_speed_max_m_s + tolerance
        ):
            raise ValueError(
                "MPC demonstration impact speed is outside the SAC task distribution."
            )
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(task.desired_impact_elevation_deg)
        expected_direction = np.asarray(
            (
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ),
            dtype=np.float64,
        )
        if not np.allclose(direction, expected_direction, rtol=0.0, atol=1.0e-8):
            raise ValueError(
                "MPC demonstration impact direction is not radial for its target."
            )
        for name, expected in expected_settings.items():
            _require_close(name, float(oracle_settings[name]), float(expected))
        for name, expected in expected_problem.items():
            _require_close(name, float(oracle_problem[name]), float(expected))

        demonstration_task = replace(
            task,
            horizontal_distance_min_m=radius,
            horizontal_distance_max_m=radius,
            target_azimuth_min_deg=azimuth_deg,
            target_azimuth_max_deg=azimuth_deg,
            target_height_offset_min_m=height,
            target_height_offset_max_m=height,
            minimum_impact_speed_min_m_s=demonstration_speed,
            minimum_impact_speed_max_m_s=demonstration_speed,
        )

        environment = VectorWhipEnvironment(
            simulator,
            1,
            settings.episode_horizon_s,
            demonstration_task,
            initial_drone_position_m=tuple(float(value) for value in initial_drone),
            seed=settings.seed + 50_000 + len(provenance),
        )
        observation = environment.observation()
        verified = False
        verified_error = math.nan
        verified_speed = math.nan
        transition_count = 0
        for control in controls:
            world_acceleration = torch.as_tensor(
                control[None], dtype=simulator.dtype, device=device
            )
            target_frame_acceleration = torch.einsum(
                "bij,bj->bi", environment.target_frame, world_acceleration
            )
            action = target_frame_acceleration / settings.maximum_acceleration_m_s2
            action_norm = torch.linalg.vector_norm(action, dim=1, keepdim=True)
            if float(action_norm.max().detach().cpu()) > 1.0 + 1.0e-5:
                raise ValueError("MPC demonstration exceeds the SAC acceleration limit.")
            action = action * torch.clamp(
                1.0 / torch.clamp(action_norm, min=1.0e-12), max=1.0
            )
            step = environment.step(action)
            observations.append(observation.detach().clone())
            actions.append(action.detach().clone())
            rewards.append(step.reward.detach().clone())
            next_observations.append(step.transition_observation.detach().clone())
            terminals.append(step.done.detach().clone())
            transition_count += 1
            observation = step.transition_observation
            if bool(step.done[0].detach().cpu()):
                if not bool(step.success[0].detach().cpu()) or bool(
                    step.unsafe[0].detach().cpu()
                ):
                    raise ValueError(
                        f"MPC demonstration {source.name} fails exact SAC re-simulation."
                    )
                verified = True
                verified_error = float(step.minimum_tip_error_m[0].detach().cpu())
                verified_speed = float(
                    step.directional_tip_speed_m_s[0].detach().cpu()
                )
                break
        if not verified:
            raise ValueError(
                f"MPC demonstration {source.name} never reaches a valid SAC impact."
            )
        provenance.append(
            {
                "path": str(source),
                "sha256": artifact_hash,
                "schema": schema,
                "controller_model_sha256": source_hash,
                "controller_node_count": source_nodes,
                "target_position_m": [float(value) for value in target],
                "target_radius_m": radius,
                "target_height_offset_m": height,
                "target_azimuth_deg": azimuth_deg,
                "minimum_impact_speed_m_s": demonstration_speed,
                "verified_transition_count": transition_count,
                "verified_minimum_tip_error_m": verified_error,
                "verified_directional_tip_speed_m_s": verified_speed,
            }
        )

    return DemonstrationTransitions(
        observation=torch.cat(observations, dim=0),
        action=torch.cat(actions, dim=0),
        reward=torch.cat(rewards, dim=0),
        next_observation=torch.cat(next_observations, dim=0),
        done=torch.cat(terminals, dim=0),
        provenance=tuple(provenance),
    )


def _mixed_replay_batch(
    online: ReplayBuffer,
    prior: DemonstrationReplay | None,
    batch_size: int,
    prior_fraction: float,
) -> tuple[torch.Tensor, ...]:
    """Sample an exact prior/online ratio without changing either buffer."""

    if prior is None:
        return online.sample(batch_size)
    prior_count = int(round(batch_size * prior_fraction))
    prior_count = min(max(prior_count, 1), batch_size - 1)
    online_count = batch_size - prior_count
    online_batch = online.sample(online_count)
    prior_batch = prior.sample(prior_count)
    combined = tuple(
        torch.cat((online_tensor, prior_tensor), dim=0)
        for online_tensor, prior_tensor in zip(
            online_batch, prior_batch, strict=True
        )
    )
    permutation = torch.randperm(
        batch_size, device=online.device, generator=online.generator
    )
    return tuple(tensor[permutation] for tensor in combined)


class RunningMeanVariance:
    """Numerically stable device statistics for vector observations or returns."""

    def __init__(self, shape: tuple[int, ...], *, device: torch.device) -> None:
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.variance = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1.0e-4, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, samples: torch.Tensor) -> None:
        values = torch.as_tensor(samples, dtype=torch.float32, device=self.mean.device)
        if values.ndim != self.mean.ndim + 1 or values.shape[1:] != self.mean.shape:
            raise ValueError("Running statistics received samples with the wrong shape.")
        batch_count = values.shape[0]
        if batch_count == 0:
            return
        batch_mean = torch.mean(values, dim=0)
        batch_variance = torch.var(values, dim=0, unbiased=False)
        batch_count_tensor = torch.tensor(
            float(batch_count), dtype=torch.float64, device=self.mean.device
        )
        total = self.count + batch_count_tensor
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count_tensor / total).to(torch.float32)
        first_moment = self.variance * self.count.to(torch.float32)
        second_moment = batch_variance * float(batch_count)
        correction = delta.square() * (
            self.count * batch_count_tensor / total
        ).to(torch.float32)
        self.mean.copy_(new_mean)
        self.variance.copy_((first_moment + second_moment + correction) / total)
        self.count.copy_(total)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "mean": self.mean.detach().cpu(),
            "variance": self.variance.detach().cpu(),
            "count": self.count.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.variance.copy_(state["variance"].to(self.variance.device))
        self.count.copy_(state["count"].to(self.count.device))


class SacAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        settings: SacSettings,
        *,
        device: torch.device,
    ) -> None:
        torch.manual_seed(settings.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(settings.seed)
        self.device = device
        self.settings = settings
        hidden = settings.hidden_size
        self.actor = SquashedGaussianActor(observation_size, action_size, hidden).to(device)
        self.q1 = QNetwork(observation_size, action_size, hidden).to(device)
        self.q2 = QNetwork(observation_size, action_size, hidden).to(device)
        self.target_q1 = QNetwork(observation_size, action_size, hidden).to(device)
        self.target_q2 = QNetwork(observation_size, action_size, hidden).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)
        self.observation_statistics = RunningMeanVariance(
            (observation_size,), device=device
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=settings.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            tuple(self.q1.parameters()) + tuple(self.q2.parameters()),
            lr=settings.critic_learning_rate,
        )
        self.log_alpha = torch.tensor(
            math.log(settings.initial_entropy_coefficient),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            (self.log_alpha,), lr=settings.entropy_learning_rate
        )
        self.target_entropy = -float(action_size)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha)

    @torch.no_grad()
    def observe(self, observation: torch.Tensor) -> None:
        self.observation_statistics.update(observation)

    def _normalize_observation(self, observation: torch.Tensor) -> torch.Tensor:
        standard_deviation = torch.sqrt(
            torch.clamp(self.observation_statistics.variance, min=1.0e-6)
        )
        return torch.clamp(
            (observation - self.observation_statistics.mean) / standard_deviation,
            -self.settings.observation_clip,
            self.settings.observation_clip,
        )

    def act(self, observation: torch.Tensor, *, deterministic: bool) -> torch.Tensor:
        with torch.no_grad():
            normalized = self._normalize_observation(observation)
            if deterministic:
                return self.actor.deterministic(normalized)
            return self.actor.sample(normalized)[0]

    def update(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        observation, action, reward, next_observation, done = batch
        observation = self._normalize_observation(observation)
        next_observation = self._normalize_observation(next_observation)
        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(next_observation)
            next_q = torch.minimum(
                self.target_q1(next_observation, next_action),
                self.target_q2(next_observation, next_action),
            ) - self.alpha.detach() * next_log_probability
            target = reward + self.settings.discount * (1.0 - done) * next_q
        q1 = self.q1(observation, action)
        q2 = self.q2(observation, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            tuple(self.q1.parameters()) + tuple(self.q2.parameters()),
            self.settings.gradient_clip_norm,
        )
        self.critic_optimizer.step()

        sampled_action, log_probability = self.actor.sample(observation)
        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        sampled_q = torch.minimum(
            self.q1(observation, sampled_action),
            self.q2(observation, sampled_action),
        )
        actor_loss = torch.mean(self.alpha.detach() * log_probability - sampled_q)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.settings.gradient_clip_norm)
        self.actor_optimizer.step()
        self.q1.requires_grad_(True)
        self.q2.requires_grad_(True)

        alpha_loss = -torch.mean(
            self.log_alpha * (log_probability.detach() + self.target_entropy)
        )
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            tau = self.settings.target_update_rate
            for target_parameter, parameter in zip(
                self.target_q1.parameters(), self.q1.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, tau)
            for target_parameter, parameter in zip(
                self.target_q2.parameters(), self.q2.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
        }


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_policy(
    path: str | Path,
    agent: SacAgent,
    source_snapshot: CableModelSnapshot,
    controller_snapshot: CableModelSnapshot,
    settings: SacSettings,
    task: TaskDistribution,
    observation_size: int,
    action_size: int,
    transitions: int,
    *,
    checkpoint_role: str,
    evaluation: dict[str, float] | None = None,
    prior_replay: dict[str, object] | None = None,
    hindsight_replay: dict[str, object] | None = None,
) -> Path:
    output = Path(path).resolve()
    payload: dict[str, object] = {
        "schema": SAC_CHECKPOINT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_model_sha256": source_snapshot.sha256,
        "controller_model_sha256": controller_snapshot.sha256,
        "source_model_provisional": source_snapshot.provisional,
        "observation_size": observation_size,
        "action_size": action_size,
        "controller_node_count": controller_snapshot.node_count,
        "transitions": transitions,
        "checkpoint_role": checkpoint_role,
        "evaluation": evaluation,
        "algorithm": (
            "goal_conditioned_sac_with_future_her"
            if hindsight_replay is not None
            and bool(hindsight_replay.get("enabled"))
            else (
                "sac_with_symmetric_prior_replay"
                if prior_replay is not None and bool(prior_replay.get("enabled"))
                else "soft_actor_critic"
            )
        ),
        "task_distribution": TASK_DISTRIBUTION_LABEL,
        "prior_replay": prior_replay,
        "hindsight_replay": hindsight_replay,
        "settings": asdict(settings),
        "task": asdict(task),
        "actor_state_dict": {
            name: value.detach().cpu() for name, value in agent.actor.state_dict().items()
        },
        "q1_state_dict": {
            name: value.detach().cpu() for name, value in agent.q1.state_dict().items()
        },
        "q2_state_dict": {
            name: value.detach().cpu() for name, value in agent.q2.state_dict().items()
        },
        "log_alpha": agent.log_alpha.detach().cpu(),
        "observation_statistics": agent.observation_statistics.state_dict(),
    }
    _atomic_torch_save(payload, output)
    return output


def load_policy(
    path: str | Path,
    settings: SacSettings,
    task: TaskDistribution,
    observation_size: int,
    action_size: int,
    source_snapshot: CableModelSnapshot,
    controller_snapshot: CableModelSnapshot,
    *,
    device: torch.device,
) -> SacAgent:
    payload = torch.load(Path(path).resolve(), map_location=device, weights_only=False)
    if not is_supported_sac_checkpoint_schema(payload.get("schema")):
        raise ValueError("Selected file is not a goal-conditioned SAC policy.")
    if payload.get("source_model_sha256") != source_snapshot.sha256:
        raise ValueError("SAC policy was trained with a different cable model.")
    if payload.get("controller_model_sha256") != controller_snapshot.sha256:
        raise ValueError("SAC policy was trained with a different controller model.")
    if int(payload.get("controller_node_count", -1)) != controller_snapshot.node_count:
        raise ValueError("SAC policy controller resolution does not match.")
    saved_settings = payload.get("settings")
    if not isinstance(saved_settings, dict):
        raise ValueError("SAC policy is missing its training settings.")
    saved_settings = dict(saved_settings)
    # The v5 prior-replay checkpoint stored this inactive algorithm option in
    # the policy settings.  It has no effect on inference.
    saved_settings.pop("prior_replay_fraction", None)
    saved_settings.setdefault(
        "hindsight_goals_per_episode", settings.hindsight_goals_per_episode
    )
    saved_settings.setdefault(
        "hindsight_minimum_achieved_speed_m_s",
        settings.hindsight_minimum_achieved_speed_m_s,
    )
    if saved_settings != asdict(settings):
        raise ValueError("SAC policy settings do not match the active controller.")
    saved_task = payload.get("task")
    if not isinstance(saved_task, dict):
        raise ValueError("SAC policy is missing its impact-task settings.")
    saved_task = dict(saved_task)
    # Policies produced before the extension constraint retain their original
    # position/velocity-only hit semantics when loaded for historical replay.
    saved_task.setdefault("minimum_extension_ratio", 0.0)
    saved_task.setdefault("desired_impact_azimuth_offset_min_deg", 0.0)
    saved_task.setdefault("desired_impact_azimuth_offset_max_deg", 0.0)
    saved_task.setdefault("desired_impact_elevation_min_deg", None)
    saved_task.setdefault("desired_impact_elevation_max_deg", None)
    saved_task.setdefault("progress_reward", task.progress_reward)
    if saved_task != asdict(task):
        raise ValueError("SAC policy task does not match the active impact task.")
    if int(payload.get("observation_size", -1)) != observation_size:
        raise ValueError("SAC policy observation size does not match the environment.")
    if int(payload.get("action_size", -1)) != action_size:
        raise ValueError("SAC policy action size does not match the environment.")
    agent = SacAgent(observation_size, action_size, settings, device=device)
    agent.actor.load_state_dict(payload["actor_state_dict"])
    agent.q1.load_state_dict(payload["q1_state_dict"])
    agent.q2.load_state_dict(payload["q2_state_dict"])
    agent.target_q1.load_state_dict(agent.q1.state_dict())
    agent.target_q2.load_state_dict(agent.q2.state_dict())
    agent.log_alpha.data.copy_(payload["log_alpha"].to(device))
    agent.observation_statistics.load_state_dict(payload["observation_statistics"])
    return agent


def _random_actions(
    count: int,
    device: torch.device,
    generator: torch.Generator,
    maximum_radius: float,
    action_size: int = 3,
) -> torch.Tensor:
    direction = torch.randn((count, action_size), device=device, generator=generator)
    direction /= torch.clamp(
        torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1.0e-12
    )
    radius = torch.rand((count, 1), device=device, generator=generator).pow(1.0 / 3.0)
    return direction * (maximum_radius * radius)


def evaluate_policy(
    agent: SacAgent,
    simulator: WhipSimulator,
    settings: SacSettings,
    task: TaskDistribution,
    *,
    episodes: int,
    seed: int,
    cancelled: CancellationCallback | None = None,
    preview_progress: ValidationPreviewCallback | None = None,
    preview_checkpoint_transitions: int = 0,
    preview_completed_training_episodes: int = 0,
) -> dict[str, float] | None:
    """Evaluate deterministic tasks, or return ``None`` after a requested stop."""

    completed = 0
    successes = 0
    strike_attempts = 0
    unsafe_episodes = 0
    within_reach_episodes = 0
    within_reach_successes = 0
    beyond_reach_episodes = 0
    beyond_reach_successes = 0
    errors: list[torch.Tensor] = []
    speeds: list[torch.Tensor] = []
    episode_rewards: list[torch.Tensor] = []
    peak_relative_energies: list[torch.Tensor] = []
    hit_drone_displacements: list[torch.Tensor] = []
    batch_index = 0
    preview_emitted = False
    while completed < episodes:
        if cancelled is not None and cancelled():
            return None
        count = min(settings.environment_count, episodes - completed)
        environment = VectorWhipEnvironment(
            simulator,
            count,
            settings.episode_horizon_s,
            task,
            seed=seed + batch_index,
        )
        preview_drone_frames: list[torch.Tensor] | None = None
        preview_cable_frames: list[torch.Tensor] | None = None
        preview_target: torch.Tensor | None = None
        if preview_progress is not None and not preview_emitted and batch_index == 0:
            preview_drone_frames = [environment.state.drone_position_m[0].clone()]
            preview_cable_frames = [environment.state.cable.positions_m[0].clone()]
            preview_target = environment.target_position_m[0].clone()
        observation = environment.observation()
        finished = torch.zeros(count, dtype=torch.bool, device=simulator.device)
        episode_return = torch.zeros(
            count, dtype=simulator.dtype, device=simulator.device
        )
        for step_index in range(environment.maximum_steps):
            if cancelled is not None and cancelled():
                return None
            action = agent.act(observation, deterministic=True)
            action[finished] = 0.0
            step = environment.step(action)
            episode_return += torch.where(
                finished, torch.zeros_like(step.reward), step.reward
            )
            if preview_drone_frames is not None and not preview_emitted:
                preview_drone_frames.append(
                    environment.state.drone_position_m[0].clone()
                )
                assert preview_cable_frames is not None
                preview_cable_frames.append(
                    environment.state.cable.positions_m[0].clone()
                )
            newly_finished = step.done & ~finished
            completed_indices = torch.nonzero(
                newly_finished, as_tuple=False
            ).flatten()
            if len(completed_indices):
                completed_success = step.success[completed_indices]
                completed_attempt = step.strike_attempt[completed_indices]
                successes += int(
                    torch.count_nonzero(completed_success).detach().cpu()
                )
                strike_attempts += int(
                    torch.count_nonzero(completed_attempt).detach().cpu()
                )
                unsafe_episodes += int(
                    torch.count_nonzero(step.unsafe[completed_indices]).detach().cpu()
                )
                beyond_reach = environment.target_beyond_initial_cable_reach()[
                    completed_indices
                ]
                within_reach = ~beyond_reach
                within_reach_episodes += int(
                    torch.count_nonzero(within_reach).detach().cpu()
                )
                beyond_reach_episodes += int(
                    torch.count_nonzero(beyond_reach).detach().cpu()
                )
                within_reach_successes += int(
                    torch.count_nonzero(
                        completed_success & within_reach
                    ).detach().cpu()
                )
                beyond_reach_successes += int(
                    torch.count_nonzero(
                        completed_success & beyond_reach
                    ).detach().cpu()
                )
                errors.append(
                    step.minimum_tip_error_m[completed_indices].detach().cpu()
                )
                speeds.append(
                    step.directional_tip_speed_m_s[completed_indices].detach().cpu()
                )
                episode_rewards.append(
                    episode_return[completed_indices].detach().cpu()
                )
                peak_relative_energies.append(
                    environment.peak_relative_cable_energy_j[completed_indices]
                    .detach()
                    .cpu()
                )
                successful_indices = completed_indices[
                    step.success[completed_indices]
                ]
                if len(successful_indices):
                    hit_drone_displacements.append(
                        step.impact_drone_displacement_m[successful_indices]
                        .detach()
                        .cpu()
                    )
                if (
                    preview_progress is not None
                    and not preview_emitted
                    and preview_drone_frames is not None
                    and preview_cable_frames is not None
                    and preview_target is not None
                    and bool(
                        torch.any(completed_indices == 0).detach().cpu()
                    )
                ):
                    drone_preview = torch.stack(preview_drone_frames)
                    cable_preview = torch.stack(preview_cable_frames)
                    preview_error = torch.amin(
                        torch.linalg.vector_norm(
                            cable_preview[:, -1] - preview_target[None],
                            dim=1,
                        )
                    )
                    preview_progress(
                        SacValidationPreview(
                            checkpoint_transitions=int(
                                preview_checkpoint_transitions
                            ),
                            completed_training_episodes=int(
                                preview_completed_training_episodes
                            ),
                            validation_seed=int(seed),
                            time_s=[
                                float(index * settings.control_interval_s)
                                for index in range(step_index + 2)
                            ],
                            drone_positions_m=(
                                drone_preview.detach().cpu().tolist()
                            ),
                            cable_positions_m=(
                                cable_preview.detach().cpu().tolist()
                            ),
                            target_position_m=(
                                preview_target.detach().cpu().tolist()
                            ),
                            desired_impact_direction=(
                                environment._to_world(
                                    environment.desired_impact_direction_frame
                                )[0]
                                .detach()
                                .cpu()
                                .tolist()
                            ),
                            success=bool(step.success[0].detach().cpu()),
                            unsafe=bool(step.unsafe[0].detach().cpu()),
                            minimum_tip_error_m=float(
                                preview_error.detach().cpu()
                            ),
                        )
                    )
                    preview_emitted = True
            finished |= step.done
            observation = step.transition_observation
            if bool(torch.all(finished).detach().cpu()):
                break
        if not bool(torch.all(finished).detach().cpu()):
            raise RuntimeError("Deterministic SAC evaluation did not terminate every task.")
        completed += count
        batch_index += 1
    error = torch.cat(errors) if errors else torch.full((1,), math.nan)
    speed = torch.cat(speeds) if speeds else torch.full((1,), math.nan)
    episode_reward = (
        torch.cat(episode_rewards)
        if episode_rewards
        else torch.full((1,), math.nan)
    )
    peak_relative_energy = (
        torch.cat(peak_relative_energies)
        if peak_relative_energies
        else torch.full((1,), math.nan)
    )
    hit_drone_displacement = (
        torch.cat(hit_drone_displacements)
        if hit_drone_displacements
        else torch.full((1,), math.nan)
    )
    return {
        "episodes": float(episodes),
        "success_rate": successes / episodes,
        "strike_attempt_rate": strike_attempts / episodes,
        "mean_episode_reward": float(torch.mean(episode_reward)),
        "unsafe_rate": unsafe_episodes / episodes,
        "mean_minimum_error_m": float(torch.mean(error)),
        "mean_directional_speed_m_s": float(torch.mean(speed)),
        "mean_peak_relative_cable_energy_j": float(
            torch.mean(peak_relative_energy)
        ),
        "mean_hit_drone_displacement_m": float(
            torch.mean(hit_drone_displacement)
        ),
        "within_initial_reach_episodes": float(within_reach_episodes),
        "within_initial_reach_success_rate": (
            within_reach_successes / within_reach_episodes
            if within_reach_episodes
            else math.nan
        ),
        "beyond_initial_reach_episodes": float(beyond_reach_episodes),
        "beyond_initial_reach_success_rate": (
            beyond_reach_successes / beyond_reach_episodes
            if beyond_reach_episodes
            else math.nan
        ),
    }


def _evaluation_rank(evaluation: dict[str, float]) -> tuple[float, ...]:
    """Select a policy that succeeds in both geometric target strata."""

    displacement = evaluation["mean_hit_drone_displacement_m"]
    if not math.isfinite(displacement):
        displacement = math.inf
    within = evaluation.get("within_initial_reach_success_rate", math.nan)
    beyond = evaluation.get("beyond_initial_reach_success_rate", math.nan)
    if math.isfinite(within) and math.isfinite(beyond):
        balanced_success = 0.5 * (within + beyond)
    else:
        balanced_success = evaluation["success_rate"]
    return (
        balanced_success,
        evaluation["success_rate"],
        -evaluation["mean_minimum_error_m"],
        evaluation["mean_directional_speed_m_s"],
        -displacement,
    )


def _diagnostic_policy_path(path: str | Path) -> Path:
    output = Path(path).resolve()
    return output.with_name(f"{output.stem}.final{output.suffix}")


def _latest_policy_path(path: str | Path) -> Path:
    output = Path(path).resolve()
    return output.with_name(f"{output.stem}.latest{output.suffix}")


def _finite_or_none(value: float) -> float | None:
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _atomic_json_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _resolved_evaluation_seeds(
    settings: SacSettings,
    checkpoint_validation_seed: int | None,
    final_test_seed: int | None,
) -> tuple[int, int]:
    validation_seed = (
        settings.seed + 1_000
        if checkpoint_validation_seed is None
        else int(checkpoint_validation_seed)
    )
    test_seed = (
        settings.seed + 10_000
        if final_test_seed is None
        else int(final_test_seed)
    )
    if validation_seed < 0 or test_seed < 0:
        raise ValueError("SAC evaluation seeds must be non-negative.")
    return validation_seed, test_seed


def train_sac(
    source_snapshot: CableModelSnapshot,
    output_path: str | Path,
    settings: SacSettings = SacSettings(),
    task: TaskDistribution = TaskDistribution(),
    *,
    device: str = "cuda",
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
    metrics_progress: MetricsProgressCallback | None = None,
    validation_preview_progress: ValidationPreviewCallback | None = None,
    endless: bool = False,
    checkpoint_evaluation_episodes: int | None = None,
    checkpoint_validation_seed: int | None = None,
    final_test_seed: int | None = None,
) -> TrainingSummary:
    report = progress if progress is not None else (lambda _message: None)
    publish_metrics = (
        metrics_progress if metrics_progress is not None else (lambda _update: None)
    )
    checkpoint_episodes = (
        min(64, settings.evaluation_episodes)
        if checkpoint_evaluation_episodes is None
        else int(checkpoint_evaluation_episodes)
    )
    if checkpoint_episodes < 1:
        raise ValueError("SAC checkpoint evaluation episodes must be positive.")
    validation_seed, test_seed = _resolved_evaluation_seeds(
        settings,
        checkpoint_validation_seed,
        final_test_seed,
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    started_at = perf_counter()
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Canonical SAC training requires CUDA.")
    simulation = SimulationSettings(
        horizon_s=settings.control_interval_s,
        simulation_dt_s=settings.physics_dt_s,
        control_interval_s=settings.control_interval_s,
        attachment_drop_m=settings.attachment_drop_m,
        maximum_acceleration_m_s2=settings.maximum_acceleration_m_s2,
        maximum_speed_m_s=settings.maximum_speed_m_s,
    )
    controller_snapshot = stable_controller_model(
        source_snapshot,
        simulation_dt_s=settings.physics_dt_s,
        node_count=settings.controller_node_count,
        constraint_iterations=4,
    )
    simulator = WhipSimulator(controller_snapshot, simulation, device=torch_device)
    environment = VectorWhipEnvironment(
        simulator,
        settings.environment_count,
        settings.episode_horizon_s,
        task,
        seed=settings.seed,
    )
    environment.reset_done(
        torch.ones(
            settings.environment_count, dtype=torch.bool, device=torch_device
        )
    )
    agent = SacAgent(
        environment.observation_size,
        environment.action_size,
        settings,
        device=torch_device,
    )
    replay = ReplayBuffer(
        settings.replay_capacity,
        environment.observation_size,
        environment.action_size,
        device=torch_device,
        seed=settings.seed + 1,
    )
    hindsight = HindsightEpisodeBuffer(
        environment,
        environment.observation_size,
        environment.action_size,
        goals_per_episode=settings.hindsight_goals_per_episode,
        minimum_achieved_speed_m_s=(
            settings.hindsight_minimum_achieved_speed_m_s
        ),
        seed=settings.seed + 3,
    )
    prior_replay_metadata: dict[str, object] = {
        "enabled": False,
        "method": None,
        "prior_fraction": 0.0,
        "prior_samples_per_batch": 0,
        "online_samples_per_batch": settings.batch_size,
        "verified_transition_count": 0,
        "reward_semantics": "raw_current_environment_reward",
        "action_convention": (
            "world_acceleration_to_target_frame_then_divide_max_acceleration"
        ),
        "demonstrations": [],
    }
    hindsight_replay_metadata: dict[str, object] = {
        "enabled": True,
        "method": "future_achieved_tip_position_and_velocity",
        "goals_per_completed_episode": settings.hindsight_goals_per_episode,
        "minimum_achieved_tip_speed_m_s": (
            settings.hindsight_minimum_achieved_speed_m_s
        ),
        "generated_transition_count": 0,
        "reward_semantics": "recomputed_current_environment_reward",
        "demonstrations": False,
        "curriculum": False,
    }
    action_generator = torch.Generator(device=torch_device)
    action_generator.manual_seed(settings.seed + 2)
    observation = environment.observation()
    agent.observe(observation)
    transitions = 0
    completed = 0
    successes = 0
    recent_completed = 0
    recent_successes = 0
    total_episode_reward = 0.0
    recent_episode_reward = 0.0
    episode_return = torch.zeros(
        settings.environment_count, dtype=simulator.dtype, device=torch_device
    )
    metrics: dict[str, float] = {
        "critic_loss": math.nan,
        "actor_loss": math.nan,
        "alpha": settings.initial_entropy_coefficient,
    }
    history: list[dict[str, object]] = []
    best_evaluation: dict[str, float] | None = None
    last_validation_evaluation: dict[str, float] | None = None
    best_transitions = 0
    degradation_streak = 0
    stopped_early = False
    user_stopped = False
    deployment_path = Path(output_path).resolve()
    final_policy_path = _diagnostic_policy_path(deployment_path)
    latest_policy_path = _latest_policy_path(deployment_path)
    log_path = deployment_path.with_suffix(".json")
    next_log = settings.log_interval_transitions

    def json_evaluation(
        evaluation: dict[str, float] | None,
    ) -> dict[str, float | None] | None:
        if evaluation is None:
            return None
        return {name: _finite_or_none(value) for name, value in evaluation.items()}

    def training_update(
        run_status: str,
        recent_success_rate: float | None,
        recent_mean_episode_reward: float | None,
        validation: dict[str, float] | None,
        *,
        best_checkpoint_updated: bool,
    ) -> SacTrainingUpdate:
        elapsed_s = max(perf_counter() - started_at, 0.0)

        def value(name: str) -> float | None:
            if validation is None:
                return None
            return _finite_or_none(validation[name])

        return SacTrainingUpdate(
            run_status=run_status,
            transitions=transitions,
            transition_limit=None if endless else settings.total_transitions,
            endless=bool(endless),
            elapsed_s=elapsed_s,
            transitions_per_s=(transitions / elapsed_s if elapsed_s > 0.0 else 0.0),
            completed_episodes=completed,
            recent_training_success_rate=recent_success_rate,
            recent_training_mean_episode_reward=recent_mean_episode_reward,
            checkpoint_evaluation_episodes=checkpoint_episodes,
            validation_success_rate=value("success_rate"),
            validation_mean_episode_reward=value("mean_episode_reward"),
            validation_mean_minimum_error_m=value("mean_minimum_error_m"),
            validation_mean_directional_speed_m_s=value(
                "mean_directional_speed_m_s"
            ),
            validation_mean_peak_relative_cable_energy_j=value(
                "mean_peak_relative_cable_energy_j"
            ),
            validation_mean_hit_drone_displacement_m=value(
                "mean_hit_drone_displacement_m"
            ),
            validation_unsafe_rate=value("unsafe_rate"),
            validation_within_initial_reach_success_rate=value(
                "within_initial_reach_success_rate"
            ),
            validation_beyond_initial_reach_success_rate=value(
                "beyond_initial_reach_success_rate"
            ),
            actor_loss=_finite_or_none(metrics["actor_loss"]),
            critic_loss=_finite_or_none(metrics["critic_loss"]),
            alpha=float(metrics["alpha"]),
            best_validation_success_rate=(
                _finite_or_none(best_evaluation["success_rate"])
                if best_evaluation is not None
                else None
            ),
            best_transitions=best_transitions,
            best_checkpoint_updated=best_checkpoint_updated,
            best_policy_path=(
                str(deployment_path) if best_evaluation is not None else None
            ),
            latest_policy_path=str(latest_policy_path),
        )

    def log_payload(
        run_status: str,
        *,
        final_training_evaluation: dict[str, float] | None = None,
        held_out_evaluation: dict[str, float] | None = None,
        held_out_evaluated: bool = False,
    ) -> dict[str, object]:
        return {
            "schema": SAC_CHECKPOINT_SCHEMA,
            "created_utc": created_utc,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "run_status": run_status,
            "held_out_evaluated": held_out_evaluated,
            "transitions": transitions,
            "completed_episodes": completed,
            "successful_episodes": successes,
            "source_model_sha256": source_snapshot.sha256,
            "controller_model_sha256": controller_snapshot.sha256,
            "evaluation_model_sha256": controller_snapshot.sha256,
            "source_model_provisional": source_snapshot.provisional,
            "settings": asdict(settings),
            "task": asdict(task),
            "run_control": {
                "endless": bool(endless),
                "transition_limit": None if endless else settings.total_transitions,
                "checkpoint_evaluation_episodes": checkpoint_episodes,
                "hindsight_replay_transitions": int(
                    hindsight_replay_metadata["generated_transition_count"]
                ),
            },
            "history": history,
            "algorithm": (
                "goal_conditioned_sac_with_future_her"
            ),
            "task_distribution": TASK_DISTRIBUTION_LABEL,
            "checkpoint_validation_seed": validation_seed,
            "final_test_seed": test_seed,
            "prior_replay": prior_replay_metadata,
            "hindsight_replay": hindsight_replay_metadata,
            "best_transitions": best_transitions,
            "best_policy_path": (
                str(deployment_path) if best_evaluation is not None else None
            ),
            "latest_policy_path": str(latest_policy_path),
            "stopped_early": stopped_early,
            "best_checkpoint_evaluation_during_training": json_evaluation(
                best_evaluation
            ),
            "final_training_state_evaluation": json_evaluation(
                final_training_evaluation
            ),
            "evaluation": json_evaluation(held_out_evaluation),
        }

    while endless or transitions < settings.total_transitions:
        if cancelled is not None and cancelled():
            user_stopped = True
            break
        if transitions < settings.warmup_transitions:
            action = _random_actions(
                settings.environment_count,
                torch_device,
                action_generator,
                1.0,
                environment.action_size,
            )
        else:
            action = agent.act(observation, deterministic=False)
        step = environment.step(action)
        episode_return += step.reward
        agent.observe(step.transition_observation)
        replay.add(
            observation,
            action,
            step.reward,
            step.transition_observation,
            step.done,
        )
        hindsight_batch = hindsight.add(
            observation,
            action,
            step.transition_observation,
            step.unsafe,
            step.done,
        )
        if hindsight_batch is not None:
            replay.add(*hindsight_batch)
            generated = int(hindsight_batch[0].shape[0])
            hindsight_replay_metadata["generated_transition_count"] = (
                int(hindsight_replay_metadata["generated_transition_count"])
                + generated
            )
            agent.observe(hindsight_batch[0])
            agent.observe(hindsight_batch[3])
        transitions += settings.environment_count
        finished = int(torch.count_nonzero(step.done).detach().cpu())
        succeeded = int(torch.count_nonzero(step.success).detach().cpu())
        completed += finished
        successes += succeeded
        recent_completed += finished
        recent_successes += succeeded
        if finished:
            finished_reward = float(
                torch.sum(episode_return[step.done]).detach().cpu()
            )
            total_episode_reward += finished_reward
            recent_episode_reward += finished_reward
            episode_return[step.done] = 0.0
        environment.reset_done(step.done)
        observation = environment.observation()
        if replay.size >= settings.batch_size and transitions >= settings.warmup_transitions:
            for _ in range(settings.gradient_steps_per_iteration):
                metrics = agent.update(replay.sample(settings.batch_size))
        finite_limit_reached = (
            not endless and transitions >= settings.total_transitions
        )
        if transitions >= next_log or finite_limit_reached:
            success_rate = (
                recent_successes / recent_completed if recent_completed else None
            )
            mean_episode_reward = (
                recent_episode_reward / recent_completed
                if recent_completed
                else None
            )
            validation_evaluation: dict[str, float] | None = None
            best_checkpoint_updated = False
            save_policy(
                latest_policy_path,
                agent,
                source_snapshot,
                controller_snapshot,
                settings,
                task,
                environment.observation_size,
                environment.action_size,
                transitions,
                checkpoint_role="latest_training_state",
                evaluation=None,
                prior_replay=prior_replay_metadata,
                hindsight_replay=hindsight_replay_metadata,
            )
            if transitions >= settings.warmup_transitions:
                validation_evaluation = evaluate_policy(
                    agent,
                    simulator,
                    settings,
                    task,
                    episodes=checkpoint_episodes,
                    seed=validation_seed,
                    cancelled=cancelled,
                    preview_progress=validation_preview_progress,
                    preview_checkpoint_transitions=transitions,
                    preview_completed_training_episodes=completed,
                )
                if validation_evaluation is None:
                    user_stopped = True
                    break
                last_validation_evaluation = dict(validation_evaluation)
                if (
                    best_evaluation is None
                    or _evaluation_rank(validation_evaluation)
                    > _evaluation_rank(best_evaluation)
                ):
                    best_evaluation = dict(validation_evaluation)
                    best_transitions = transitions
                    degradation_streak = 0
                    best_checkpoint_updated = True
                    save_policy(
                        deployment_path,
                        agent,
                        source_snapshot,
                        controller_snapshot,
                        settings,
                        task,
                        environment.observation_size,
                        environment.action_size,
                        transitions,
                        checkpoint_role="best_deterministic_evaluation",
                        evaluation=best_evaluation,
                        prior_replay=prior_replay_metadata,
                        hindsight_replay=hindsight_replay_metadata,
                    )
                elif (
                    best_evaluation["success_rate"] > 0.0
                    and validation_evaluation["success_rate"]
                    < best_evaluation["success_rate"]
                ):
                    degradation_streak += 1
                else:
                    degradation_streak = 0
                save_policy(
                    latest_policy_path,
                    agent,
                    source_snapshot,
                    controller_snapshot,
                    settings,
                    task,
                    environment.observation_size,
                    environment.action_size,
                    transitions,
                    checkpoint_role="latest_training_state",
                    evaluation=validation_evaluation,
                    prior_replay=prior_replay_metadata,
                    hindsight_replay=hindsight_replay_metadata,
                )
            entry = {
                "transitions": transitions,
                "episodes": completed,
                "success_rate": success_rate,
                "mean_episode_reward": mean_episode_reward,
                "deterministic_policy_success_rate": (
                    _finite_or_none(validation_evaluation["success_rate"])
                    if validation_evaluation is not None
                    else None
                ),
                "deterministic_mean_episode_reward": (
                    _finite_or_none(validation_evaluation["mean_episode_reward"])
                    if validation_evaluation is not None
                    else None
                ),
                "deterministic_within_initial_reach_success_rate": (
                    _finite_or_none(
                        validation_evaluation[
                            "within_initial_reach_success_rate"
                        ]
                    )
                    if validation_evaluation is not None
                    else None
                ),
                "deterministic_beyond_initial_reach_success_rate": (
                    _finite_or_none(
                        validation_evaluation[
                            "beyond_initial_reach_success_rate"
                        ]
                    )
                    if validation_evaluation is not None
                    else None
                ),
                "deterministic_mean_minimum_error_m": (
                    _finite_or_none(validation_evaluation["mean_minimum_error_m"])
                    if validation_evaluation is not None
                    else None
                ),
                "deterministic_mean_directional_speed_m_s": (
                    _finite_or_none(
                        validation_evaluation["mean_directional_speed_m_s"]
                    )
                    if validation_evaluation is not None
                    else None
                ),
                "deterministic_mean_hit_drone_displacement_m": (
                    _finite_or_none(
                        validation_evaluation[
                            "mean_hit_drone_displacement_m"
                        ]
                    )
                    if validation_evaluation is not None
                    else None
                ),
                "deterministic_unsafe_rate": (
                    _finite_or_none(validation_evaluation["unsafe_rate"])
                    if validation_evaluation is not None
                    else None
                ),
                "best_deterministic_success_rate": (
                    _finite_or_none(best_evaluation["success_rate"])
                    if best_evaluation is not None
                    else None
                ),
                "best_transitions": best_transitions,
                "degradation_streak": degradation_streak,
                "checkpoint_evaluation_episodes": checkpoint_episodes,
                "hindsight_replay_transitions": int(
                    hindsight_replay_metadata["generated_transition_count"]
                ),
                "critic_loss": _finite_or_none(metrics["critic_loss"]),
                "actor_loss": _finite_or_none(metrics["actor_loss"]),
                "alpha": float(metrics["alpha"]),
            }
            update = training_update(
                "running",
                success_rate,
                mean_episode_reward,
                validation_evaluation,
                best_checkpoint_updated=best_checkpoint_updated,
            )
            entry["elapsed_s"] = update.elapsed_s
            entry["transitions_per_s"] = update.transitions_per_s
            history.append(entry)
            _atomic_json_save(log_payload("running"), log_path)
            publish_metrics(update)
            policy_success_rate = entry["deterministic_policy_success_rate"]
            policy_far_success_rate = entry[
                "deterministic_beyond_initial_reach_success_rate"
            ]
            policy_text = (
                "n/a"
                if policy_success_rate is None
                else f"{100.0 * float(policy_success_rate):.1f}%"
            )
            far_text = (
                "n/a"
                if policy_far_success_rate is None
                else f"{100.0 * float(policy_far_success_rate):.1f}%"
            )
            target_text = "endless" if endless else str(settings.total_transitions)
            report(
                f"SAC transitions={transitions}/{target_text} "
                f"episodes={completed} success="
                f"{'n/a' if success_rate is None else f'{100.0 * success_rate:.1f}%'} "
                f"reward/episode="
                f"{'n/a' if mean_episode_reward is None else f'{mean_episode_reward:.3f}'} "
                f"policy={policy_text} far={far_text} "
                f"HER={int(hindsight_replay_metadata['generated_transition_count'])} "
                f"actor={metrics['actor_loss']:.4g} critic={metrics['critic_loss']:.4g} "
                f"alpha={metrics['alpha']:.3f} best@{best_transitions}"
            )
            recent_completed = 0
            recent_successes = 0
            recent_episode_reward = 0.0
            while next_log <= transitions:
                next_log += settings.log_interval_transitions
            if (
                not endless
                and
                degradation_streak
                >= settings.degradation_patience_evaluations
            ):
                stopped_early = True
                report(
                    "Early stop: deterministic success remained below the best "
                    f"checkpoint for {degradation_streak} evaluations."
                )
                break

    latest_evaluation = (
        last_validation_evaluation
        if history and int(history[-1]["transitions"]) == transitions
        else None
    )
    save_policy(
        latest_policy_path,
        agent,
        source_snapshot,
        controller_snapshot,
        settings,
        task,
        environment.observation_size,
        environment.action_size,
        transitions,
        checkpoint_role="latest_training_state",
        evaluation=latest_evaluation,
        prior_replay=prior_replay_metadata,
        hindsight_replay=hindsight_replay_metadata,
    )

    final_evaluation: dict[str, float] | None = None
    evaluation: dict[str, float] | None = None
    held_out_evaluated = False
    run_status = "cancelled" if user_stopped else (
        "early_stopped" if stopped_early else "completed"
    )
    if not user_stopped:
        final_evaluation = evaluate_policy(
            agent,
            simulator,
            settings,
            task,
            episodes=settings.evaluation_episodes,
            seed=settings.seed + 9_000,
            cancelled=cancelled,
        )
        if final_evaluation is None:
            user_stopped = True
            run_status = "cancelled"

    if final_evaluation is not None:
        save_policy(
            final_policy_path,
            agent,
            source_snapshot,
            controller_snapshot,
            settings,
            task,
            environment.observation_size,
            environment.action_size,
            transitions,
            checkpoint_role="final_training_state",
            evaluation=final_evaluation,
            prior_replay=prior_replay_metadata,
            hindsight_replay=hindsight_replay_metadata,
        )
        if best_evaluation is None:
            best_evaluation = dict(final_evaluation)
            best_transitions = transitions
            save_policy(
                deployment_path,
                agent,
                source_snapshot,
                controller_snapshot,
                settings,
                task,
                environment.observation_size,
                environment.action_size,
                transitions,
                checkpoint_role="best_deterministic_evaluation",
                evaluation=best_evaluation,
                prior_replay=prior_replay_metadata,
                hindsight_replay=hindsight_replay_metadata,
            )

    if not user_stopped:
        # Standard SAC evaluation uses the same goal distribution and observation
        # dimension as training.  A full-plant robustness study requires an explicit
        # full-state-to-controller-state estimator and is a separate experiment.
        evaluation_simulator = WhipSimulator(
            controller_snapshot, simulation, device=torch_device
        )
        deployment_agent = load_policy(
            deployment_path,
            settings,
            task,
            environment.observation_size,
            environment.action_size,
            source_snapshot,
            controller_snapshot,
            device=torch_device,
        )
        evaluation = evaluate_policy(
            deployment_agent,
            evaluation_simulator,
            settings,
            task,
            episodes=settings.evaluation_episodes,
            seed=test_seed,
            cancelled=cancelled,
        )
        if evaluation is None:
            user_stopped = True
            run_status = "cancelled"
        else:
            held_out_evaluated = True

    _atomic_json_save(
        log_payload(
            run_status,
            final_training_evaluation=final_evaluation,
            held_out_evaluation=evaluation,
            held_out_evaluated=held_out_evaluated,
        ),
        log_path,
    )
    terminal_recent_success_rate = (
        recent_successes / recent_completed if recent_completed else None
    )
    terminal_recent_mean_episode_reward = (
        recent_episode_reward / recent_completed if recent_completed else None
    )
    publish_metrics(
        training_update(
            run_status,
            terminal_recent_success_rate,
            terminal_recent_mean_episode_reward,
            last_validation_evaluation,
            best_checkpoint_updated=False,
        )
    )
    if user_stopped:
        report(f"SAC training stopped cleanly; latest checkpoint: {latest_policy_path}")

    summary_evaluation = evaluation or {}
    summary_policy_path = (
        deployment_path if best_evaluation is not None else latest_policy_path
    )
    summary_final_path = (
        final_policy_path if final_evaluation is not None else latest_policy_path
    )
    return TrainingSummary(
        transitions=transitions,
        completed_episodes=completed,
        successful_episodes=successes,
        training_success_rate=successes / max(completed, 1),
        training_mean_episode_reward=total_episode_reward / max(completed, 1),
        evaluation_success_rate=float(
            summary_evaluation.get("success_rate", math.nan)
        ),
        evaluation_mean_episode_reward=float(
            summary_evaluation.get("mean_episode_reward", math.nan)
        ),
        evaluation_mean_minimum_error_m=float(
            summary_evaluation.get("mean_minimum_error_m", math.nan)
        ),
        evaluation_mean_directional_speed_m_s=float(
            summary_evaluation.get("mean_directional_speed_m_s", math.nan)
        ),
        evaluation_mean_peak_relative_cable_energy_j=float(
            summary_evaluation.get("mean_peak_relative_cable_energy_j", math.nan)
        ),
        evaluation_mean_hit_drone_displacement_m=float(
            summary_evaluation.get("mean_hit_drone_displacement_m", math.nan)
        ),
        evaluation_unsafe_rate=float(
            summary_evaluation.get("unsafe_rate", math.nan)
        ),
        evaluation_within_initial_reach_success_rate=float(
            summary_evaluation.get("within_initial_reach_success_rate", math.nan)
        ),
        evaluation_beyond_initial_reach_success_rate=float(
            summary_evaluation.get("beyond_initial_reach_success_rate", math.nan)
        ),
        best_transitions=best_transitions,
        stopped_early=stopped_early,
        policy_path=summary_policy_path,
        final_policy_path=summary_final_path,
        log_path=log_path,
        run_status=run_status,
        latest_policy_path=latest_policy_path,
        held_out_evaluated=held_out_evaluated,
    )
