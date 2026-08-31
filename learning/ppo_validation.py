"""Fixed mild-state validation and plotting for the simple PPO experiment."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learning.normalization import FixedContextNormalizer
from learning.sequential_sac_env import SequentialWhipEnvironment, SimpleRewardWeights
from learning.state_bank import InitialStateBank, initial_state_bank_from_state
from planning.rollout import hover_preroll
from planning.task import load_canonical_whip_task
from simulator.parameters import SimulatorSettings
from simulator.production import build_production_simulator


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES = 10_240


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _finite_summary(values: torch.Tensor) -> dict[str, float]:
    array = values[torch.isfinite(values)].detach().cpu().double().numpy()
    if array.size == 0:
        return {}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _state_distances(bank: InitialStateBank, canonical: InitialStateBank) -> torch.Tensor:
    pieces = (
        (bank.uav_position_m - canonical.uav_position_m[0]) / 0.10,
        (bank.uav_velocity_m_s - canonical.uav_velocity_m_s[0]) / 1.0,
        (bank.uav_orientation_xyzw - canonical.uav_orientation_xyzw[0]) / 0.25,
        (
            bank.uav_angular_velocity_world_rad_s
            - canonical.uav_angular_velocity_world_rad_s[0]
        )
        / 1.0,
        (bank.cable_positions_m - canonical.cable_positions_m[0]).reshape(len(bank), -1)
        / 0.10,
        (bank.cable_velocities_m_s - canonical.cable_velocities_m_s[0]).reshape(
            len(bank), -1
        )
        / 1.0,
    )
    features = torch.cat(
        tuple(piece.reshape(len(bank), -1) for piece in pieces), dim=1
    )
    return torch.sqrt(torch.mean(features.square(), dim=1))


class FixedMildStateValidationPanel:
    """Ten reproducible, physically propagated mild initial-state variations."""

    def __init__(self, config: dict[str, Any]) -> None:
        validation = config["validation"]
        count = int(validation["episodes"])
        seed = int(validation["state_selection_seed"])
        settings = SimulatorSettings.load(ROOT / config["simulator_config"])
        task = load_canonical_whip_task(ROOT / config["task_config"])
        simulator = build_production_simulator(settings)
        simulator.uav_model.set_fixed_evaluation_batch_size(count)
        base = hover_preroll(simulator, task)
        canonical = initial_state_bank_from_state(
            base,
            command_position_world_m=base.uav.position_m,
            command_velocity_world_m_s=base.uav.velocity_m_s,
            command_yaw_world_rad=task.initial_yaw_rad,
        )
        state_bank_root = ROOT / validation["state_bank_root"]
        bank = InitialStateBank.load(
            state_bank_root / "validation_state_bank.npz",
            state_bank_root / "validation_state_bank_manifest.json",
        )
        distances = _state_distances(bank, canonical)
        threshold = torch.quantile(distances, float(validation["maximum_distance_quantile"]))
        candidates = torch.nonzero(distances <= threshold).reshape(-1)
        if int(candidates.numel()) < count:
            raise RuntimeError("The mild-state validation bank has too few eligible rows.")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(int(candidates.numel()), generator=generator)
        self.indices = candidates[order[:count]]
        selected = bank.select(self.indices, device=simulator.device)
        reward = config["reward"]
        self.environment = SequentialWhipEnvironment(
            simulator,
            task,
            selected.state,
            FixedContextNormalizer.load(ROOT / config["context_normalizer"]),
            batch_size=count,
            episode_duration_s=float(config["episode_duration_s"]),
            control_dt_s=float(config["control_dt_s"]),
            maximum_acceleration_m_s2=float(
                config["action"]["maximum_acceleration_norm_m_s2"]
            ),
            maximum_body_rate_rad_s=float(config["action"]["maximum_body_rate_rad_s"]),
            observation_clip=float(config["observation"]["normalized_clip"]),
            reward_weights=SimpleRewardWeights(
                progress=float(reward["progress_weight"]),
                directed_speed_near_target=float(
                    reward["directed_speed_near_target_weight"]
                ),
                direction_near_target=float(reward["direction_near_target_weight"]),
                uav_displacement=float(reward["uav_displacement_weight"]),
                success_bonus=float(reward["success_bonus"]),
                numerical_failure=float(reward["numerical_failure_penalty"]),
                proximity_scale_m=float(reward["proximity_scale_m"]),
                non_tip_first=float(reward.get("non_tip_first_penalty", 0.0)),
                strike_quality_improvement=float(
                    reward.get("strike_quality_improvement_weight", 0.0)
                ),
                maximum_displacement=float(
                    reward.get("maximum_displacement_weight", 0.0)
                ),
                displacement_integral=float(
                    reward.get("displacement_integral_weight", 0.0)
                ),
                uav_speed_integral=float(
                    reward.get("uav_speed_integral_weight", 0.0)
                ),
                acceleration_effort=float(
                    reward.get("acceleration_effort_weight", 0.0)
                ),
                body_rate_effort=float(
                    reward.get("body_rate_effort_weight", 0.0)
                ),
                action_smoothness=float(
                    reward.get("action_smoothness_weight", 0.0)
                ),
            ),
            success_mode=str(config.get("success_mode", "simple_endpoint")),
            reward_mode=str(config.get("reward_mode", "legacy_dense")),
        )
        self.manifest = {
            "schema": "simple_ppo_fixed_mild_state_validation_v1",
            "state_bank_root": str(state_bank_root),
            "selection": (
                "seeded sample from states at or below the configured state-distance quantile"
            ),
            "selection_seed": seed,
            "state_indices": self.indices.tolist(),
            "state_distances": distances[self.indices].tolist(),
            "state_distance_quantile": float(validation["maximum_distance_quantile"]),
            "state_distance_threshold": float(threshold),
            "validation_episodes": count,
            "target": "canonical fixed target",
            "policy": "deterministic tanh(mean)",
            "model": "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI",
            "physics": "production UAV + causal residual + 12-node DDER",
        }

    @torch.no_grad()
    def evaluate(self, agent: Any, *, checkpoint_episodes: int) -> dict[str, Any]:
        was_training = bool(agent.policy.training)
        agent.policy.eval()
        observation = self.environment.reset()
        for _ in range(self.environment.control_step_count):
            action = agent.deterministic_action(observation)
            observation = self.environment.step(action).next_observation
        agent.policy.train(was_training)
        success = self.environment.episode_success
        endpoint_success = self.environment.episode_endpoint_success
        scientific_success = self.environment.episode_scientific_success
        entered = self.environment.episode_first_entry_marker > 0
        return {
            "checkpoint_episodes": int(checkpoint_episodes),
            "validation_episodes": int(success.numel()),
            "validation_successes": int(success.sum().cpu()),
            "validation_success_rate": float(success.float().mean().cpu()),
            "validation_endpoint_successes": int(endpoint_success.sum().cpu()),
            "validation_endpoint_success_rate": float(
                endpoint_success.float().mean().cpu()
            ),
            "validation_legacy_scientific_successes": int(
                scientific_success.sum().cpu()
            ),
            "validation_legacy_scientific_success_rate": float(
                scientific_success.float().mean().cpu()
            ),
            "median_minimum_tip_distance_m": float(
                self.environment.episode_minimum_tip_distance.median().cpu()
            ),
            "mean_maximum_uav_displacement_m": float(
                self.environment.episode_maximum_displacement.mean().cpu()
            ),
            "numerical_failures": int(self.environment.failed.sum().cpu()),
            "tip_first_count": int(
                (self.environment.episode_first_entry_marker == 10).sum().cpu()
            ),
            "mean_maximum_uav_speed_m_s": float(
                self.environment.episode_maximum_uav_speed.mean().cpu()
            ),
            "mean_maximum_command_acceleration_m_s2": float(
                self.environment.episode_maximum_command_acceleration.mean().cpu()
            ),
            "successful_first_entry_physics_step": _finite_summary(
                self.environment.episode_first_entry_physics_step[success].float()
            ),
            "successful_first_entry_time_s": _finite_summary(
                self.environment.episode_first_entry_time_s[success]
            ),
            "first_entry_tip_distance_m": _finite_summary(
                self.environment.episode_first_entry_tip_distance[entered]
            ),
            "first_entry_tip_speed_m_s": _finite_summary(
                self.environment.episode_first_entry_tip_speed[entered]
            ),
            "first_entry_directed_speed_m_s": _finite_summary(
                self.environment.episode_first_entry_directed_speed[entered]
            ),
            "first_entry_direction_error_deg": _finite_summary(
                self.environment.episode_first_entry_direction_error_deg[entered]
            ),
            "first_entry_speed_gate_passes": int(
                (
                    entered
                    & (
                        self.environment.episode_first_entry_directed_speed
                        >= self.environment.task.minimum_directed_speed_m_s
                    )
                ).sum().cpu()
            ),
            "first_entry_direction_gate_passes": int(
                (
                    entered
                    & (
                        self.environment.episode_first_entry_direction_error_deg
                        <= self.environment.task.maximum_direction_error_deg
                    )
                ).sum().cpu()
            ),
            "successful_tip_distance_m": _finite_summary(
                self.environment.episode_success_tip_distance[success]
            ),
            "successful_directed_speed_m_s": _finite_summary(
                self.environment.episode_success_directed_speed[success]
            ),
            "successful_direction_error_deg": _finite_summary(
                self.environment.episode_success_direction_error_deg[success]
            ),
        }


VALIDATION_FIELDS = (
    "checkpoint_episodes",
    "validation_episodes",
    "validation_successes",
    "validation_success_rate",
    "validation_endpoint_successes",
    "validation_endpoint_success_rate",
    "validation_legacy_scientific_successes",
    "validation_legacy_scientific_success_rate",
    "median_minimum_tip_distance_m",
    "mean_maximum_uav_displacement_m",
    "numerical_failures",
    "tip_first_count",
    "mean_maximum_uav_speed_m_s",
    "mean_maximum_command_acceleration_m_s2",
)


def append_validation_result(artifact: Path, result: dict[str, Any]) -> None:
    history = artifact / "validation_history.csv"
    exists = history.exists()
    with history.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=VALIDATION_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({name: result[name] for name in VALIDATION_FIELDS})
    _atomic_json(artifact / "validation_latest.json", result)


def rolling_success_rate(
    episodes: np.ndarray,
    cumulative_successes: np.ndarray,
    *,
    window_episodes: int = TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES,
) -> np.ndarray:
    """Return an episode-weighted trailing success rate from cumulative counts."""

    episodes = np.asarray(episodes, dtype=np.float64)
    successes = np.asarray(cumulative_successes, dtype=np.float64)
    if episodes.ndim != 1 or successes.shape != episodes.shape:
        raise ValueError("episodes and cumulative_successes must be matching vectors")
    if window_episodes <= 0:
        raise ValueError("window_episodes must be positive")
    rolling = np.zeros_like(episodes)
    for index, episode in enumerate(episodes):
        threshold = episode - float(window_episodes)
        baseline_index = int(np.searchsorted(episodes, threshold, side="right") - 1)
        baseline_episodes = episodes[baseline_index] if baseline_index >= 0 else 0.0
        baseline_successes = successes[baseline_index] if baseline_index >= 0 else 0.0
        denominator = max(episode - baseline_episodes, 1.0)
        rolling[index] = (successes[index] - baseline_successes) / denominator
    return rolling


def write_training_plots(artifact: Path) -> None:
    """Write separate and combined training/validation success plots."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training_path = artifact / "training_log.csv"
    if not training_path.exists():
        return
    with training_path.open(newline="", encoding="utf-8") as stream:
        training = list(csv.DictReader(stream))
    if not training:
        return
    episodes = np.asarray([float(row["episodes"]) for row in training])
    successes = np.asarray([float(row["successes"]) for row in training])
    total = 100.0 * np.asarray([float(row["total_success_rate"]) for row in training])
    rolling = 100.0 * rolling_success_rate(episodes, successes)
    endpoint_total = None
    endpoint_batch = None
    if "total_endpoint_success_rate" in training[0]:
        endpoint_total = 100.0 * np.asarray(
            [float(row["total_endpoint_success_rate"]) for row in training]
        )
        endpoint_batch = 100.0 * np.asarray(
            [float(row["batch_endpoint_success_rate"]) for row in training]
        )
    legacy_total = None
    if "total_legacy_scientific_success_rate" in training[0]:
        legacy_total = 100.0 * np.asarray(
            [
                float(row["total_legacy_scientific_success_rate"])
                for row in training
            ]
        )

    def save(figure: Any, filename: str) -> None:
        temporary = artifact / (filename + ".tmp.png")
        figure.savefig(temporary, dpi=160)
        plt.close(figure)
        os.replace(temporary, artifact / filename)

    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(episodes, total, label="cumulative", linewidth=2.0)
    axis.plot(
        episodes,
        rolling,
        label="rolling 10,240 episodes",
        linewidth=1.8,
    )
    if endpoint_total is not None and endpoint_batch is not None:
        axis.plot(
            episodes,
            endpoint_total,
            label="endpoint-only cumulative",
            linewidth=1.5,
            linestyle="--",
        )
    if legacy_total is not None:
        axis.plot(
            episodes,
            legacy_total,
            label="legacy numerical-gate diagnostic",
            linewidth=1.2,
            linestyle=":",
        )
    axis.set(xlabel="Training episodes", ylabel="Training success rate (%)", ylim=(-2, 102))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    save(figure, "training_success_vs_episodes.png")

    validation_path = artifact / "validation_history.csv"
    if not validation_path.exists():
        return
    with validation_path.open(newline="", encoding="utf-8") as stream:
        validation = list(csv.DictReader(stream))
    if not validation:
        return
    validation_episodes = np.asarray(
        [float(row["checkpoint_episodes"]) for row in validation]
    )
    validation_rate = 100.0 * np.asarray(
        [float(row["validation_success_rate"]) for row in validation]
    )
    validation_endpoint_rate = None
    if "validation_endpoint_success_rate" in validation[0]:
        validation_endpoint_rate = 100.0 * np.asarray(
            [float(row["validation_endpoint_success_rate"]) for row in validation]
        )
    validation_legacy_rate = None
    if "validation_legacy_scientific_success_rate" in validation[0]:
        validation_legacy_rate = 100.0 * np.asarray(
            [
                float(row["validation_legacy_scientific_success_rate"])
                for row in validation
            ]
        )
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(validation_episodes, validation_rate, "o-", linewidth=2.0)
    if validation_endpoint_rate is not None:
        axis.plot(
            validation_episodes,
            validation_endpoint_rate,
            "o--",
            linewidth=1.5,
            label="endpoint-only",
        )
    if validation_legacy_rate is not None:
        axis.plot(
            validation_episodes,
            validation_legacy_rate,
            "o:",
            linewidth=1.5,
            label="legacy numerical-gate diagnostic",
        )
    if validation_endpoint_rate is not None or validation_legacy_rate is not None:
        axis.legend(loc="best")
    axis.set(
        xlabel="Training episodes at validation checkpoint",
        ylabel="Validation success rate (%) (10 rollouts)",
        ylim=(-2, 102),
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    save(figure, "validation_success_vs_episodes.png")

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].plot(episodes, total, label="cumulative training success", linewidth=2.0)
    axes[0].plot(
        episodes,
        rolling,
        label="rolling 10,240-episode training success",
        linewidth=1.8,
    )
    if endpoint_total is not None:
        axes[0].plot(
            episodes,
            endpoint_total,
            label="endpoint-only cumulative",
            linewidth=1.5,
            linestyle="--",
        )
    if legacy_total is not None:
        axes[0].plot(
            episodes,
            legacy_total,
            label="legacy numerical-gate diagnostic",
            linewidth=1.2,
            linestyle=":",
        )
    axes[0].set(ylabel="Training success rate (%)", ylim=(-2, 102))
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(validation_episodes, validation_rate, "o-", linewidth=2.0)
    if validation_endpoint_rate is not None:
        axes[1].plot(
            validation_episodes,
            validation_endpoint_rate,
            "o--",
            linewidth=1.5,
            label="endpoint-only",
        )
        axes[1].legend(loc="best")
    axes[1].set(
        xlabel="Training episodes at validation checkpoint",
        ylabel="Validation success rate (%)",
        ylim=(-2, 102),
    )
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    save(figure, "success_rate_vs_episodes.png")
