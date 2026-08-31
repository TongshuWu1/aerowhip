"""Canonical SAC discovery loop for the full-rank spectral policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import torch

from planning.cem_task import VariableDurationWhipTask
from planning.rl_reward import RLWhipRewardConfig
from simulator.simulator import CoupledSimulator

from .canonical_pilot import (
    behavior_level,
    critic_diagnostic,
    repeated_canonical_specification,
)
from .context_sampling import ContextSpecification, build_context_from_specification
from .normalization import FixedContextNormalizer
from .one_shot_env import evaluate_open_loop_batch
from .replay import TerminalReplayBuffer
from .sac import TerminalSacAgent
from .spectral_sac import (
    alpha_explore_schedule,
    dominant_horizontal_sign_changes,
    frequency_energy_fractions,
    temporal_roughness,
)
from .state_bank import InitialStateBank
from .training import load_training_checkpoint, save_training_checkpoint


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    tensor = value.detach().float()
    return {
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.50)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "maximum": float(tensor.max()),
        "minimum": float(tensor.min()),
    }


def _fraction(mask: torch.Tensor) -> float:
    return float(mask.float().mean())


def _spectral_group_statistics(sample) -> dict[str, Any]:
    mean = sample.raw_mean.detach().reshape(-1, 3, 16)
    std = sample.log_std.detach().exp().reshape(-1, 3, 16)
    fractions = frequency_energy_fractions(sample.spectral_coefficients.detach())
    groups = {"low": slice(0, 4), "mid": slice(4, 8), "high": slice(8, 16)}
    return {
        name: {
            "mean_absolute_coefficient_mean": float(mean[:, :, indices].abs().mean()),
            "mean_actor_standard_deviation": float(std[:, :, indices].mean()),
            "sampled_energy_fraction": float(fractions[name].mean()),
        }
        for name, indices in groups.items()
    }


def _trajectory_structure(knots: torch.Tensor) -> dict[str, float]:
    norms = torch.linalg.vector_norm(knots, dim=-1)
    roughness = temporal_roughness(knots)
    sign_changes = dominant_horizontal_sign_changes(knots)
    return {
        "temporal_roughness_mean": float(roughness.mean()),
        "temporal_roughness_normalized_mean": float((roughness / 400.0).mean()),
        "maximum_knot_acceleration_m_s2": float(norms.max()),
        "mean_knot_acceleration_norm_m_s2": float(norms.mean()),
        "dominant_horizontal_sign_changes_mean": float(sign_changes.float().mean()),
    }


def _behavioral_diversity(result) -> dict[str, Any]:
    components = result.reward_components
    if components is None:
        raise RuntimeError("Spectral SAC requires rl_whip_reward_v2 components.")
    descriptor = torch.stack(
        (
            components.normalized_progress,
            result.tip_min_distance_m / 1.0,
            result.max_tip_speed_m_s / 10.0,
            result.directed_tip_speed_m_s / 4.0,
            result.max_uav_displacement_m / 0.5,
        ),
        dim=1,
    )
    centered = descriptor - descriptor.mean(dim=0, keepdim=True)
    return {
        "descriptor_scales": [1.0, 1.0, 10.0, 4.0, 0.5],
        "descriptor_names": [
            "progress",
            "minimum_tip_distance",
            "maximum_tip_speed",
            "directed_speed_at_reported_event",
            "maximum_uav_displacement",
        ],
        "per_dimension_standard_deviation": descriptor.std(dim=0).detach().cpu().tolist(),
        "mean_distance_from_descriptor_mean": float(
            torch.linalg.vector_norm(centered, dim=1).mean()
        ),
    }


def collect_spectral_batch(
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    replay: TerminalReplayBuffer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    reward_config: RLWhipRewardConfig,
    canonical_bank: InitialStateBank,
    canonical_specification: ContextSpecification,
    *,
    batch_size: int,
    episodes_before_collection: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    specification = repeated_canonical_specification(
        canonical_specification, batch_size, split="spectral_canonical_training"
    )
    context = build_context_from_specification(simulator, canonical_bank, specification)
    normalized = normalizer.normalize(context.to_tensor())
    if not torch.equal(normalized, normalized[:1].expand_as(normalized)):
        raise RuntimeError("Every spectral SAC collection context must be identical.")
    with torch.no_grad():
        sample = agent.actor(normalized)
        action = sample.normalized_action.detach()
        result = evaluate_open_loop_batch(
            simulator, context, action, task, rl_reward_config=reward_config
        )
    replay_start = replay.total_inserted
    replay.add(
        context=normalized,
        action=action,
        scaled_reward=result.reward,
        raw_reward=result.reward,
        success=result.task_success,
        feasible=result.feasible,
        tip_distance_m=result.tip_min_distance_m,
        directed_speed_m_s=result.directed_tip_speed_m_s,
        direction_error_deg=result.direction_error_deg,
    )
    components = result.reward_components
    if components is None:
        raise RuntimeError("Spectral SAC requires rl_whip_reward_v2 components.")
    progress = components.normalized_progress
    distance = result.tip_min_distance_m
    feasible = result.feasible
    success = result.task_success
    knots = result.decoded_action.acceleration_knots_local_m_s2
    metrics = {
        "reward": _quantiles(result.reward),
        "progress": _quantiles(progress),
        "minimum_tip_distance_m": _quantiles(distance),
        "maximum_tip_speed_m_s": _quantiles(result.max_tip_speed_m_s),
        "directed_tip_speed_m_s": _quantiles(result.directed_tip_speed_m_s),
        "maximum_uav_displacement_m": _quantiles(result.max_uav_displacement_m),
        "maximum_uav_speed_m_s": _quantiles(result.max_uav_speed_m_s),
        "success_rate": _fraction(success),
        "feasible_rate": _fraction(feasible),
        "finite_rate": _fraction(result.rollout_finite),
        "tip_first_rate": _fraction(result.first_entry_marker == 10),
        "fractions": {
            "progress_ge_0_25": _fraction(progress >= 0.25),
            "progress_ge_0_50": _fraction(progress >= 0.50),
            "progress_ge_0_75": _fraction(progress >= 0.75),
            "progress_ge_0_90": _fraction(progress >= 0.90),
            "d_min_le_0_50_m": _fraction(distance <= 0.50),
            "d_min_le_0_20_m": _fraction(distance <= 0.20),
            "d_min_le_0_10_m": _fraction(distance <= 0.10),
            "d_min_le_0_05_m": _fraction(distance <= 0.05),
            "feasible_and_progress_ge_0_50": _fraction(feasible & (progress >= 0.50)),
            "feasible_and_progress_ge_0_75": _fraction(feasible & (progress >= 0.75)),
            "feasible_and_d_min_le_0_20_m": _fraction(feasible & (distance <= 0.20)),
            "feasible_and_d_min_le_0_10_m": _fraction(feasible & (distance <= 0.10)),
        },
        "policy_entropy": float((-sample.log_prob).mean()),
        "mean_log_pi": float(sample.log_prob.mean()),
        "fixed_duration_s": float(result.decoded_action.duration_s[0]),
        "trajectory_structure": _trajectory_structure(knots),
    }
    spectral = _spectral_group_statistics(sample)
    diversity = _behavioral_diversity(result)
    success_metadata = None
    if bool(success.any()):
        candidates = torch.nonzero(success, as_tuple=False)[:, 0]
        best_local = int(candidates[result.reward[candidates].argmax()].detach().cpu())
        success_metadata = {
            "episode": episodes_before_collection + best_local + 1,
            "replay_insertion_index": replay_start + best_local,
            "collection_local_index": best_local,
            **result.row(best_local),
        }
    return metrics, spectral, diversity, success_metadata


def evaluate_spectral_policy(
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    reward_config: RLWhipRewardConfig,
    canonical_bank: InitialStateBank,
    canonical_specification: ContextSpecification,
    *,
    batch_size: int,
    level4_minimum_directed_speed_m_s: float,
) -> dict[str, Any]:
    specification = repeated_canonical_specification(
        canonical_specification, batch_size, split="spectral_canonical_evaluation"
    )
    context = build_context_from_specification(simulator, canonical_bank, specification)
    normalized = normalizer.normalize(context.to_tensor())
    with torch.no_grad():
        sample = agent.actor(normalized, deterministic=True)
        result = evaluate_open_loop_batch(
            simulator,
            context,
            sample.deterministic_mean_action,
            task,
            rl_reward_config=reward_config,
        )
    row = result.row(0)
    components = row["reward_components"]
    level = behavior_level(
        success=bool(row["task_success"]),
        progress=float(components["normalized_progress"]),
        minimum_tip_distance_m=float(row["tip_min_distance_m"]),
        directed_tip_speed_m_s=float(row["directed_tip_speed_m_s"]),
        level4_minimum_directed_speed_m_s=level4_minimum_directed_speed_m_s,
    )
    return {
        **row,
        "behavior_level": level,
        "progress": float(components["normalized_progress"]),
        "maximum_tip_speed_m_s": float(row["max_tip_speed_m_s"]),
        "trajectory_structure": _trajectory_structure(
            result.decoded_action.acceleration_knots_local_m_s2[:1]
        ),
        "spectral_statistics": _spectral_group_statistics(sample),
        "identical_context_rows": torch.equal(normalized, normalized[:1].expand_as(normalized)),
    }


def _selection_key(evaluation: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(bool(evaluation["task_success"])),
        float(evaluation["behavior_level"]),
        float(bool(evaluation["feasible"])),
        float(evaluation["progress"]),
        -float(evaluation["tip_min_distance_m"]),
        float(evaluation["directed_tip_speed_m_s"]),
        -float(evaluation["direction_error_deg"]),
    )


@dataclass(frozen=True, slots=True)
class SpectralPilotOutcome:
    classification: str
    episodes: int
    gradient_updates: int
    runtime_s: float
    highest_deterministic_behavior_level: int
    first_stochastic_success_episode: int | None
    total_stochastic_successes: int
    best_stochastic_success: dict[str, Any] | None
    selected_evaluation: dict[str, Any]
    best_checkpoint: Path
    latest_checkpoint: Path
    training_history: tuple[dict[str, Any], ...]
    evaluation_history: tuple[dict[str, Any], ...]
    spectral_statistics_history: tuple[dict[str, Any], ...]
    behavioral_diversity_history: tuple[dict[str, Any], ...]


def run_spectral_sac_pilot(
    *,
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    replay: TerminalReplayBuffer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    reward_config: RLWhipRewardConfig,
    canonical_bank: InitialStateBank,
    canonical_specification: ContextSpecification,
    config: dict[str, Any],
    artifact_directory: Path,
) -> SpectralPilotOutcome:
    sac, validation, budget = config["sac"], config["validation"], config["budget"]
    batch_size = int(sac["collection_batch_size"])
    checkpoint_directory = artifact_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    best = checkpoint_directory / "best_behavior.pt"
    latest = checkpoint_directory / "latest.pt"
    context_generator = torch.Generator().manual_seed(int(config["seed"]) + 1000)
    replay_generator = torch.Generator().manual_seed(int(config["seed"]) + 2000)
    level4_speed = float(validation["level4_minimum_directed_speed_m_s"])
    baseline = evaluate_spectral_policy(
        agent,
        normalizer,
        simulator,
        task,
        reward_config,
        canonical_bank,
        canonical_specification,
        batch_size=batch_size,
        level4_minimum_directed_speed_m_s=level4_speed,
    )
    evaluation_history: list[dict[str, Any]] = [
        {
            "episodes": 0,
            "gradient_updates": 0,
            "alpha_sac": float(agent.alpha.detach()),
            "alpha_explore": alpha_explore_schedule(0),
            "alpha_effective": float(agent.alpha.detach()) + alpha_explore_schedule(0),
            "canonical": baseline,
            "elapsed_s": 0.0,
        }
    ]
    training_history: list[dict[str, Any]] = []
    spectral_history: list[dict[str, Any]] = []
    diversity_history: list[dict[str, Any]] = []
    episodes = updates = total_successes = 0
    first_success: int | None = None
    best_stochastic: dict[str, Any] | None = None
    highest_level = int(baseline["behavior_level"])
    best_key = _selection_key(baseline)
    start = time.perf_counter()
    deadline = start + float(budget["maximum_wall_clock_s"])
    next_evaluation = int(validation["evaluation_interval_episodes"])
    save_training_checkpoint(
        best,
        agent=agent,
        episodes=0,
        gradient_updates=0,
        training_history=training_history,
        evaluation_history=evaluation_history,
        context_generator=context_generator,
        replay_metadata=replay.metadata(),
    )

    while episodes < int(budget["maximum_collected_episodes"]) and time.perf_counter() < deadline:
        collection_start = time.perf_counter()
        collection, spectral, diversity, success_metadata = collect_spectral_batch(
            agent,
            normalizer,
            replay,
            simulator,
            task,
            reward_config,
            canonical_bank,
            canonical_specification,
            batch_size=batch_size,
            episodes_before_collection=episodes,
        )
        successes_in_batch = int(round(collection["success_rate"] * batch_size))
        total_successes += successes_in_batch
        if success_metadata is not None:
            if first_success is None:
                first_success = int(success_metadata["episode"])
            if best_stochastic is None or float(success_metadata["reward"]) > float(
                best_stochastic["reward"]
            ):
                best_stochastic = success_metadata
        episodes += batch_size
        collection_runtime = time.perf_counter() - collection_start
        bonus = alpha_explore_schedule(episodes)
        updates_this_batch: list[dict[str, float]] = []
        update_start = time.perf_counter()
        if episodes >= int(sac["initial_exploration_episodes"]) and len(replay) >= int(
            sac["minibatch_size"]
        ):
            for _ in range(int(sac["updates_per_collection_batch"])):
                sample = replay.sample(
                    int(sac["minibatch_size"]),
                    device=simulator.device,
                    generator=replay_generator,
                )
                update = agent.update(
                    sample.context,
                    sample.action,
                    sample.scaled_reward,
                    actor_entropy_bonus=bonus,
                )
                if not all(torch.isfinite(torch.tensor(value)) for value in update.values()):
                    raise RuntimeError("Non-finite spectral SAC optimizer diagnostic.")
                updates_this_batch.append(update)
                updates += 1
        update_runtime = time.perf_counter() - update_start
        averaged = (
            {
                key: sum(row[key] for row in updates_this_batch) / len(updates_this_batch)
                for key in updates_this_batch[0]
            }
            if updates_this_batch
            else {}
        )
        training_history.append(
            {
                "episodes": episodes,
                "gradient_updates": updates,
                **collection,
                **averaged,
                "alpha_explore": bonus,
                "alpha_sac": float(agent.alpha.detach()),
                "alpha_effective": float(agent.alpha.detach()) + bonus,
                "replay_size": len(replay),
                "collection_runtime_s": collection_runtime,
                "update_runtime_s": update_runtime,
                "episodes_per_s": batch_size / max(collection_runtime, 1.0e-9),
                "elapsed_s": time.perf_counter() - start,
                "stochastic_successes_this_batch": successes_in_batch,
                "total_stochastic_successes": total_successes,
            }
        )
        spectral_history.append({"episodes": episodes, **spectral})
        diversity_history.append({"episodes": episodes, **diversity})

        if episodes < next_evaluation and episodes < int(budget["maximum_collected_episodes"]):
            continue
        canonical = evaluate_spectral_policy(
            agent,
            normalizer,
            simulator,
            task,
            reward_config,
            canonical_bank,
            canonical_specification,
            batch_size=batch_size,
            level4_minimum_directed_speed_m_s=level4_speed,
        )
        highest_level = max(highest_level, int(canonical["behavior_level"]))
        diagnostic = critic_diagnostic(
            agent,
            replay,
            count=int(validation["diagnostic_replay_rows"]),
            device=simulator.device,
        )
        evaluation = {
            "episodes": episodes,
            "gradient_updates": updates,
            "alpha_sac": float(agent.alpha.detach()),
            "alpha_explore": alpha_explore_schedule(episodes),
            "alpha_effective": float(agent.alpha.detach()) + alpha_explore_schedule(episodes),
            "canonical": canonical,
            "critic_diagnostic": diagnostic,
            "total_stochastic_successes": total_successes,
            "elapsed_s": time.perf_counter() - start,
        }
        evaluation_history.append(evaluation)
        next_evaluation += int(validation["evaluation_interval_episodes"])
        save_training_checkpoint(
            latest,
            agent=agent,
            episodes=episodes,
            gradient_updates=updates,
            training_history=training_history,
            evaluation_history=evaluation_history,
            context_generator=context_generator,
            replay_metadata=replay.metadata(),
        )
        key = _selection_key(canonical)
        if key > best_key:
            best_key = key
            save_training_checkpoint(
                best,
                agent=agent,
                episodes=episodes,
                gradient_updates=updates,
                training_history=training_history,
                evaluation_history=evaluation_history,
                context_generator=context_generator,
                replay_metadata=replay.metadata(),
            )
        print(
            "SPECTRAL_SAC_EVAL "
            + json.dumps(
                {
                    "episodes": episodes,
                    "level": canonical["behavior_level"],
                    "success": canonical["task_success"],
                    "stochastic_successes": total_successes,
                    "progress": canonical["progress"],
                    "d_min_m": canonical["tip_min_distance_m"],
                    "directed_m_s": canonical["directed_tip_speed_m_s"],
                    "feasible": canonical["feasible"],
                    "alpha_sac": float(agent.alpha.detach()),
                    "alpha_explore": alpha_explore_schedule(episodes),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if bool(canonical["task_success"]):
            break

    if not latest.is_file():
        save_training_checkpoint(
            latest,
            agent=agent,
            episodes=episodes,
            gradient_updates=updates,
            training_history=training_history,
            evaluation_history=evaluation_history,
            context_generator=context_generator,
            replay_metadata=replay.metadata(),
        )
    load_training_checkpoint(
        best, agent=agent, context_generator=torch.Generator().manual_seed(0)
    )
    selected = evaluate_spectral_policy(
        agent,
        normalizer,
        simulator,
        task,
        reward_config,
        canonical_bank,
        canonical_specification,
        batch_size=batch_size,
        level4_minimum_directed_speed_m_s=level4_speed,
    )
    if bool(selected["task_success"]):
        classification = "SAC_STRUCTURED_EXPLORATION_PASS"
    elif int(selected["behavior_level"]) >= 4 or total_successes > 0:
        classification = "SAC_STRUCTURED_EXPLORATION_PROMISING"
    else:
        classification = "SAC_STRUCTURED_EXPLORATION_LIMITED"
    return SpectralPilotOutcome(
        classification=classification,
        episodes=episodes,
        gradient_updates=updates,
        runtime_s=time.perf_counter() - start,
        highest_deterministic_behavior_level=highest_level,
        first_stochastic_success_episode=first_success,
        total_stochastic_successes=total_successes,
        best_stochastic_success=best_stochastic,
        selected_evaluation=selected,
        best_checkpoint=best,
        latest_checkpoint=latest,
        training_history=tuple(training_history),
        evaluation_history=tuple(evaluation_history),
        spectral_statistics_history=tuple(spectral_history),
        behavioral_diversity_history=tuple(diversity_history),
    )
