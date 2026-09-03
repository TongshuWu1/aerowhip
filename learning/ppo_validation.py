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
TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES = 5_000


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
    """Reproducible, physically propagated mild initial-state variations."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        count_override: int | None = None,
        fixed_evaluation_batch_size: int | None = None,
        record_state_trajectory: bool = False,
    ) -> None:
        validation = config["validation"]
        count = (
            int(validation["episodes"])
            if count_override is None
            else int(count_override)
        )
        if count < 1:
            raise ValueError("Validation episode count must be positive.")
        seed = int(validation["state_selection_seed"])
        settings = SimulatorSettings.load(ROOT / config["simulator_config"])
        task = load_canonical_whip_task(ROOT / config["task_config"])
        simulator = build_production_simulator(settings)
        numerical_batch = (
            int(validation.get("fixed_numerical_batch_size", 2048))
            if fixed_evaluation_batch_size is None
            else int(fixed_evaluation_batch_size)
        )
        if numerical_batch < count:
            raise ValueError(
                "The fixed numerical validation batch cannot be smaller than "
                "the logical validation count."
            )
        simulator.uav_model.set_fixed_evaluation_batch_size(numerical_batch)
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
                terminal_displacement=float(
                    reward.get("terminal_displacement_weight", 0.0)
                ),
                terminal_displacement_success_only=bool(
                    reward.get("terminal_displacement_success_only", False)
                ),
                displacement_integral=float(
                    reward.get("displacement_integral_weight", 0.0)
                ),
                success_forward_return_bonus=float(
                    reward.get("success_forward_return_bonus_weight", 0.0)
                ),
                success_release_bonus=float(
                    reward.get("success_release_bonus_weight", 0.0)
                ),
                forward_excursion_scale_m=float(
                    reward.get("forward_excursion_scale_m", 0.35)
                ),
                uav_backward_speed_scale_m_s=float(
                    reward.get("uav_backward_speed_scale_m_s", 1.0)
                ),
                relative_tip_forward_speed_scale_m_s=float(
                    reward.get("relative_tip_forward_speed_scale_m_s", 4.0)
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
                time_to_success=float(
                    reward.get("time_to_success_weight_per_s", 0.0)
                ),
                directed_speed_reward_cap_m_s=float(
                    reward.get("directed_speed_reward_cap_m_s", float("inf"))
                ),
                success_compactness_bonus=float(
                    reward.get("success_compactness_bonus", 0.0)
                ),
                success_compactness_scale_m=float(
                    reward.get("success_compactness_scale_m", 0.5)
                ),
                displacement_cost_scale_m=float(
                    reward.get("displacement_cost_scale_m", 0.0)
                ),
            ),
            action_mode=str(config["action"].get("mode", "full_6d")),
            progress_shaping_reference=str(
                reward.get("progress_shaping_reference", "world_tip")
            ),
            progress_attachment_compensation_fraction=float(
                reward.get("progress_attachment_compensation_fraction", 0.5)
            ),
            directed_speed_shaping_reference=str(
                reward.get("directed_speed_shaping_reference", "world_tip")
            ),
            success_mode=str(config.get("success_mode", "simple_endpoint")),
            reward_mode=str(config.get("reward_mode", "legacy_dense")),
            terminate_on_success=not bool(
                config.get("reported_success", {}).get(
                    "episode_continues_after_success", True
                )
            ),
            record_state_trajectory=record_state_trajectory,
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
            "fixed_uav_residual_evaluation_batch_size": numerical_batch,
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
            "mean_terminal_uav_displacement_m": float(
                torch.nanmean(
                    self.environment.episode_terminal_displacement
                ).cpu()
            ),
            "median_terminal_uav_displacement_m": float(
                torch.nanmedian(
                    self.environment.episode_terminal_displacement
                ).cpu()
            ),
            "mean_successful_peak_forward_displacement_m": float(
                self.environment.episode_peak_forward_displacement[success]
                .mean()
                .cpu()
            ) if bool(success.any()) else float("nan"),
            "mean_successful_return_distance_m": float(
                torch.nanmean(
                    self.environment.episode_success_return_distance[success]
                ).cpu()
            ) if bool(success.any()) else float("nan"),
            "mean_successful_return_quality": float(
                torch.nanmean(
                    self.environment.episode_success_return_quality[success]
                ).cpu()
            ) if bool(success.any()) else float("nan"),
            "mean_successful_release_quality": float(
                torch.nanmean(
                    self.environment.episode_success_release_quality[success]
                ).cpu()
            ) if bool(success.any()) else float("nan"),
            "mean_best_return_release_quality": float(
                self.environment.episode_best_return_release_quality.mean().cpu()
            ),
            "mean_successful_forward_displacement_at_hit_m": float(
                torch.nanmean(
                    self.environment.episode_success_forward_displacement[success]
                ).cpu()
            ) if bool(success.any()) else float("nan"),
            "mean_successful_uav_forward_speed_at_hit_m_s": float(
                torch.nanmean(
                    self.environment.episode_success_uav_forward_speed[success]
                ).cpu()
            ) if bool(success.any()) else float("nan"),
            "mean_successful_relative_tip_forward_speed_at_hit_m_s": float(
                torch.nanmean(
                    self.environment.episode_success_relative_tip_forward_speed[success]
                ).cpu()
            ) if bool(success.any()) else float("nan"),
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
    "mean_terminal_uav_displacement_m",
    "median_terminal_uav_displacement_m",
    "mean_successful_peak_forward_displacement_m",
    "mean_successful_return_distance_m",
    "mean_successful_return_quality",
    "mean_successful_release_quality",
    "mean_successful_forward_displacement_at_hit_m",
    "mean_successful_uav_forward_speed_at_hit_m_s",
    "mean_successful_relative_tip_forward_speed_at_hit_m_s",
    "numerical_failures",
    "tip_first_count",
    "mean_maximum_uav_speed_m_s",
    "mean_maximum_command_acceleration_m_s2",
    "successful_first_entry_time_s_median",
    "successful_tip_distance_m_median",
    "successful_directed_speed_m_s_median",
    "successful_direction_error_deg_median",
)

MANUAL_VALIDATION_FIELDS = (
    "request_id",
    "completed_utc",
    "checkpoint_path",
    *VALIDATION_FIELDS,
)


def _flatten_validation_result(result: dict[str, Any]) -> dict[str, Any]:
    flattened = dict(result)
    for summary_name in (
        "successful_first_entry_time_s",
        "successful_tip_distance_m",
        "successful_directed_speed_m_s",
        "successful_direction_error_deg",
    ):
        summary = result.get(summary_name, {})
        flattened[f"{summary_name}_median"] = summary.get("median")
    return flattened


def append_validation_result(artifact: Path, result: dict[str, Any]) -> None:
    history = artifact / "validation_history.csv"
    exists = history.exists()
    flattened = _flatten_validation_result(result)
    fields = VALIDATION_FIELDS
    if exists:
        with history.open(newline="", encoding="utf-8") as stream:
            existing_fields = next(csv.reader(stream), [])
        if existing_fields:
            # Resuming an older artifact must preserve its existing CSV schema.
            fields = tuple(existing_fields)
    with history.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({name: flattened.get(name) for name in fields})
    _atomic_json(artifact / "validation_latest.json", result)


def append_manual_validation_result(artifact: Path, result: dict[str, Any]) -> None:
    """Persist one user-requested current-policy validation without touching scheduled history."""

    history = artifact / "manual_validation_history.csv"
    exists = history.exists()
    flattened = _flatten_validation_result(result)
    fields = MANUAL_VALIDATION_FIELDS
    if exists:
        with history.open(newline="", encoding="utf-8") as stream:
            existing_fields = next(csv.reader(stream), [])
        if existing_fields:
            fields = tuple(existing_fields)
    with history.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({name: flattened.get(name) for name in fields})
    _atomic_json(artifact / "manual_validation_latest.json", result)


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


def rolling_episode_mean(
    episodes: np.ndarray,
    batch_means: np.ndarray,
    *,
    window_episodes: int = TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES,
) -> np.ndarray:
    """Return a trailing episode-weighted mean from per-batch metric means.

    Each logged value summarizes the episodes since the preceding cumulative
    episode count. Boundary batches are weighted by their overlap with the
    requested trailing window, so changing PPO collection size does not change
    the interpretation of the curve.
    """

    episodes = np.asarray(episodes, dtype=np.float64)
    means = np.asarray(batch_means, dtype=np.float64)
    if episodes.ndim != 1 or means.shape != episodes.shape:
        raise ValueError("episodes and batch_means must be matching vectors")
    if window_episodes <= 0:
        raise ValueError("window_episodes must be positive")
    if episodes.size == 0:
        return np.asarray([], dtype=np.float64)
    if np.any(np.diff(episodes) <= 0.0) or episodes[0] <= 0.0:
        raise ValueError("episodes must be strictly increasing and positive")

    batch_starts = np.concatenate(([0.0], episodes[:-1]))
    rolling = np.empty_like(means)
    for index, episode_end in enumerate(episodes):
        window_start = max(0.0, episode_end - float(window_episodes))
        overlap = np.maximum(
            0.0,
            np.minimum(episodes[: index + 1], episode_end)
            - np.maximum(batch_starts[: index + 1], window_start),
        )
        denominator = float(np.sum(overlap))
        rolling[index] = float(
            np.sum(overlap * means[: index + 1]) / max(denominator, 1.0)
        )
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
    if "rolling_success_rate" in training[0]:
        rolling = 100.0 * np.asarray(
            [float(row["rolling_success_rate"]) for row in training]
        )
        rolling_window_episodes = int(float(training[-1]["rolling_window_episodes"]))
    elif "rolling_5000_success_rate" in training[0]:
        rolling = 100.0 * np.asarray(
            [float(row["rolling_5000_success_rate"]) for row in training]
        )
        rolling_window_episodes = 5_000
    else:
        rolling = 100.0 * rolling_success_rate(episodes, successes)
        rolling_window_episodes = TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES
    endpoint_total = None
    endpoint_batch = None
    if "total_endpoint_success_rate" in training[0]:
        endpoint_total = 100.0 * np.asarray(
            [float(row["total_endpoint_success_rate"]) for row in training]
        )
        endpoint_batch = 100.0 * np.asarray(
            [float(row["batch_endpoint_success_rate"]) for row in training]
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
        label=f"rolling {rolling_window_episodes:,} episodes",
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
    axis.set(xlabel="Training episodes", ylabel="Training success rate (%)", ylim=(-2, 102))
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    save(figure, "training_success_vs_episodes.png")

    if "mean_episode_reward" in training[0] and all(
        row.get("mean_episode_reward", "") != "" for row in training
    ):
        batch_reward = np.asarray(
            [float(row["mean_episode_reward"]) for row in training]
        )
        rolling_reward = rolling_episode_mean(
            episodes,
            batch_reward,
            window_episodes=rolling_window_episodes,
        )
        figure, axis = plt.subplots(figsize=(8.5, 4.8))
        axis.plot(
            episodes,
            batch_reward,
            color="#94a3b8",
            linewidth=1.0,
            alpha=0.55,
            label="batch mean episodic reward",
        )
        axis.plot(
            episodes,
            rolling_reward,
            color="#d97706",
            linewidth=2.0,
            label=f"rolling {rolling_window_episodes:,} episodes",
        )
        axis.set(
            xlabel="Training episodes",
            ylabel="Mean episodic reward",
        )
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        save(figure, "training_reward_vs_episodes.png")

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
    manual_validation: list[dict[str, str]] = []
    manual_path = artifact / "manual_validation_history.csv"
    if manual_path.exists():
        with manual_path.open(newline="", encoding="utf-8") as stream:
            manual_validation = list(csv.DictReader(stream))
    manual_episodes = np.asarray([], dtype=np.float64)
    manual_rate = np.asarray([], dtype=np.float64)
    manual_yerr: np.ndarray | None = None
    if manual_validation:
        manual_episodes = np.asarray(
            [float(row["checkpoint_episodes"]) for row in manual_validation]
        )
        manual_successes = np.asarray(
            [float(row["validation_successes"]) for row in manual_validation]
        )
        manual_counts = np.asarray(
            [float(row["validation_episodes"]) for row in manual_validation]
        )
        proportion = manual_successes / np.maximum(manual_counts, 1.0)
        z = 1.959963984540054
        denominator = 1.0 + z * z / manual_counts
        center = (proportion + z * z / (2.0 * manual_counts)) / denominator
        radius = (
            z
            * np.sqrt(
                proportion * (1.0 - proportion) / manual_counts
                + z * z / (4.0 * manual_counts * manual_counts)
            )
            / denominator
        )
        manual_rate = 100.0 * proportion
        low = 100.0 * np.maximum(0.0, center - radius)
        high = 100.0 * np.minimum(1.0, center + radius)
        manual_yerr = np.maximum(
            0.0,
            np.vstack((manual_rate - low, high - manual_rate)),
        )
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(
        validation_episodes,
        validation_rate,
        "o-",
        linewidth=2.0,
        label="scheduled fixed-state validation",
    )
    if validation_endpoint_rate is not None:
        axis.plot(
            validation_episodes,
            validation_endpoint_rate,
            "o--",
            linewidth=1.5,
            label="endpoint-only",
        )
    if manual_validation and manual_yerr is not None:
        axis.errorbar(
            manual_episodes,
            manual_rate,
            yerr=manual_yerr,
            fmt="s",
            color="#f59e0b",
            capsize=3,
            label="manual current-policy check (95% Wilson CI)",
        )
    axis.legend(loc="best")
    axis.set(
        xlabel="Training episodes at validation checkpoint",
        ylabel="Validation task success rate (%)",
        ylim=(-2, 102),
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    save(figure, "validation_success_vs_episodes.png")

    mechanism_fields = (
        "mean_successful_peak_forward_displacement_m",
        "mean_successful_return_distance_m",
        "mean_successful_forward_displacement_at_hit_m",
        "mean_successful_uav_forward_speed_at_hit_m_s",
        "mean_successful_relative_tip_forward_speed_at_hit_m_s",
        "mean_successful_return_quality",
        "mean_successful_release_quality",
    )
    if all(
        field in validation[0]
        and all(row.get(field, "") != "" for row in validation)
        for field in mechanism_fields
    ):
        mechanism = {
            field: np.asarray([float(row[field]) for row in validation])
            for field in mechanism_fields
        }
        figure, axes = plt.subplots(3, 1, figsize=(9.0, 9.5), sharex=True)
        axes[0].plot(
            validation_episodes,
            mechanism["mean_successful_peak_forward_displacement_m"],
            "o-",
            label="peak forward",
        )
        axes[0].plot(
            validation_episodes,
            mechanism["mean_successful_forward_displacement_at_hit_m"],
            "o-",
            label="forward position at hit",
        )
        axes[0].plot(
            validation_episodes,
            mechanism["mean_successful_return_distance_m"],
            "o-",
            label="returned before hit",
        )
        axes[0].set_ylabel("Target-axis distance (m)")
        axes[0].legend(loc="best")
        axes[1].plot(
            validation_episodes,
            mechanism["mean_successful_uav_forward_speed_at_hit_m_s"],
            "o-",
            label="UAV velocity at hit",
        )
        axes[1].plot(
            validation_episodes,
            mechanism["mean_successful_relative_tip_forward_speed_at_hit_m_s"],
            "o-",
            label="tip relative velocity at hit",
        )
        axes[1].axhline(0.0, color="#64748b", linewidth=0.8)
        axes[1].set_ylabel("Target-axis velocity (m/s)")
        axes[1].legend(loc="best")
        axes[2].plot(
            validation_episodes,
            mechanism["mean_successful_return_quality"],
            "o-",
            label="return quality",
        )
        axes[2].plot(
            validation_episodes,
            mechanism["mean_successful_release_quality"],
            "o-",
            label="release quality",
        )
        axes[2].set_ylabel("Bounded quality")
        axes[2].set_xlabel("Training episodes at validation checkpoint")
        axes[2].legend(loc="best")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        figure.tight_layout()
        save(figure, "validation_whip_mechanism_vs_episodes.png")

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
    if manual_validation and manual_yerr is not None:
        axes[1].errorbar(
            manual_episodes,
            manual_rate,
            yerr=manual_yerr,
            fmt="s",
            color="#f59e0b",
            capsize=3,
            label="manual current-policy check (95% Wilson CI)",
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
