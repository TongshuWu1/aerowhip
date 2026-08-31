"""Focused near-canonical SAC pilot for the bounded RL whip reward."""

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

from .context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    pad_context_specification,
    target_direction_from_local_target,
)
from .normalization import FixedContextNormalizer
from .replay import TerminalReplayBuffer
from .sac import TerminalSacAgent
from .state_bank import InitialStateBank
from .training import load_training_checkpoint, save_training_checkpoint


def subset_state_bank(
    bank: InitialStateBank,
    indices: torch.Tensor,
    *,
    label: str,
) -> InitialStateBank:
    index = torch.as_tensor(indices, dtype=torch.int64, device="cpu").reshape(-1)
    fields = {
        name: getattr(bank, name)[index].clone()
        for name in (
            "uav_position_m",
            "uav_velocity_m_s",
            "uav_orientation_xyzw",
            "uav_angular_velocity_world_rad_s",
            "residual_history",
            "residual_acceleration_m_s2",
            "cable_positions_m",
            "cable_velocities_m_s",
            "command_position_world_m",
            "command_velocity_world_m_s",
            "command_yaw_world_rad",
        )
    }
    return InitialStateBank(
        **fields,
        seed=bank.seed,
        generation={
            "method": "nearest_to_canonical_existing_bank_subset",
            "label": label,
            "source_generation": bank.generation,
            "source_indices": index.tolist(),
            "generated_new_states": False,
        },
    )


def concatenate_state_banks(
    first: InitialStateBank,
    second: InitialStateBank,
    *,
    label: str,
) -> InitialStateBank:
    fields = {
        name: torch.cat((getattr(first, name), getattr(second, name)), dim=0)
        for name in (
            "uav_position_m",
            "uav_velocity_m_s",
            "uav_orientation_xyzw",
            "uav_angular_velocity_world_rad_s",
            "residual_history",
            "residual_acceleration_m_s2",
            "cable_positions_m",
            "cable_velocities_m_s",
            "command_position_world_m",
            "command_velocity_world_m_s",
            "command_yaw_world_rad",
        )
    }
    return InitialStateBank(
        **fields,
        seed=first.seed,
        generation={
            "method": "concatenated_existing_banks",
            "label": label,
            "first_count": len(first),
            "second_count": len(second),
            "generated_new_states": False,
        },
    )


def nearest_state_indices(
    simulator: CoupledSimulator,
    bank: InitialStateBank,
    canonical_bank: InitialStateBank,
    canonical_target_local_m: torch.Tensor,
    canonical_direction_local: torch.Tensor,
    normalizer: FixedContextNormalizer,
    *,
    count: int,
    batch_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    canonical_spec = ContextSpecification(
        torch.tensor([0]),
        canonical_target_local_m.reshape(1, 3),
        canonical_direction_local.reshape(1, 3),
        "canonical_distance_reference",
    )
    canonical_context = build_context_from_specification(
        simulator, canonical_bank, canonical_spec
    )
    canonical_features = normalizer.normalize(canonical_context.to_tensor())[0, :70]
    distances: list[torch.Tensor] = []
    for start in range(0, len(bank), batch_size):
        stop = min(start + batch_size, len(bank))
        rows = stop - start
        specification = ContextSpecification(
            torch.arange(start, stop),
            canonical_target_local_m.reshape(1, 3).repeat(rows, 1),
            canonical_direction_local.reshape(1, 3).repeat(rows, 1),
            "nearest_state_search",
        )
        context = build_context_from_specification(simulator, bank, specification)
        features = normalizer.normalize(context.to_tensor())[:, :70]
        distances.append(
            torch.sqrt(torch.mean((features - canonical_features).square(), dim=-1))
            .detach()
            .cpu()
        )
    all_distances = torch.cat(distances)
    order = torch.argsort(all_distances)
    return order[:count], all_distances[order[:count]]


def sample_pilot_context_specification(
    bank: InitialStateBank,
    *,
    count: int,
    generator: torch.Generator,
    canonical_target_local_m: torch.Tensor,
    canonical_fraction: float,
    split: str,
) -> ContextSpecification:
    indices = torch.randint(len(bank), (count,), generator=generator)
    random = torch.rand((count, 3), generator=generator)
    target = torch.empty((count, 3), dtype=torch.float32)
    target[:, 0] = 0.95 + 0.10 * random[:, 0]
    target[:, 1] = -0.05 + 0.10 * random[:, 1]
    target[:, 2] = float(canonical_target_local_m[2]) - 0.03 + 0.06 * random[:, 2]
    canonical = torch.rand((count,), generator=generator) < canonical_fraction
    target = torch.where(
        canonical[:, None],
        canonical_target_local_m.reshape(1, 3).repeat(count, 1),
        target,
    )
    return ContextSpecification(
        indices,
        target,
        target_direction_from_local_target(target),
        split,
    )


def evaluate_pilot_policy(
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    reward_config: RLWhipRewardConfig,
    bank: InitialStateBank,
    specification: ContextSpecification,
    *,
    canonical_batch_size: int = 2048,
) -> dict[str, Any]:
    padded = pad_context_specification(specification, canonical_batch_size)
    context = build_context_from_specification(simulator, bank, padded)
    with torch.no_grad():
        action = agent.actor(
            normalizer.normalize(context.to_tensor()), deterministic=True
        ).deterministic_mean_action
        result = __import__(
            "learning.one_shot_env", fromlist=["evaluate_open_loop_batch"]
        ).evaluate_open_loop_batch(
            simulator,
            context,
            action,
            task,
            rl_reward_config=reward_config,
        )
    count = specification.count
    success = result.task_success[:count]
    feasible = result.feasible[:count]
    finite = result.rollout_finite[:count]
    marker = result.first_entry_marker[:count]
    tip = result.tip_min_distance_m[:count]
    directed = result.directed_tip_speed_m_s[:count]
    direction = result.direction_error_deg[:count]
    duration = result.decoded_action.duration_s[:count]
    reward = result.reward[:count]
    return {
        "split": specification.split,
        "count": count,
        "hard_success_rate": float(success.float().mean()),
        "feasible_rate": float(feasible.float().mean()),
        "finite_rate": float(finite.float().mean()),
        "tip_first_rate": float((marker == 10).float().mean()),
        "mean_tip_error_m": float(tip.mean()),
        "median_tip_error_m": float(tip.median()),
        "mean_directed_tip_speed_m_s": float(directed.mean()),
        "median_directed_tip_speed_m_s": float(directed.median()),
        "mean_direction_error_deg": float(direction.mean()),
        "median_direction_error_deg": float(direction.median()),
        "mean_maneuver_duration_s": float(duration.mean()),
        "mean_reward": float(reward.mean()),
        "minimum_reward": float(reward.min()),
        "maximum_reward": float(reward.max()),
        "first_row": result.row(0),
    }


def collect_pilot_batch(
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    replay: TerminalReplayBuffer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    reward_config: RLWhipRewardConfig,
    bank: InitialStateBank,
    *,
    count: int,
    generator: torch.Generator,
    canonical_target_local_m: torch.Tensor,
) -> dict[str, float]:
    specification = sample_pilot_context_specification(
        bank,
        count=count,
        generator=generator,
        canonical_target_local_m=canonical_target_local_m,
        canonical_fraction=0.50,
        split="pilot_training",
    )
    context = build_context_from_specification(simulator, bank, specification)
    normalized = normalizer.normalize(context.to_tensor())
    with torch.no_grad():
        sample = agent.actor(normalized)
        action = sample.normalized_action.detach()
        result = __import__(
            "learning.one_shot_env", fromlist=["evaluate_open_loop_batch"]
        ).evaluate_open_loop_batch(
            simulator,
            context,
            action,
            task,
            rl_reward_config=reward_config,
        )
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
    return {
        "mean_reward": float(result.reward.mean()),
        "minimum_reward": float(result.reward.min()),
        "maximum_reward": float(result.reward.max()),
        "success_rate": float(result.task_success.float().mean()),
        "feasible_rate": float(result.feasible.float().mean()),
        "finite_rate": float(result.rollout_finite.float().mean()),
        "mean_tip_distance_m": float(result.tip_min_distance_m.mean()),
        "mean_directed_tip_speed_m_s": float(result.directed_tip_speed_m_s.mean()),
        "mean_direction_error_deg": float(result.direction_error_deg.mean()),
        "mean_duration_s": float(result.decoded_action.duration_s.mean()),
        "acceleration_normalized_norm_mean": float(
            torch.linalg.vector_norm(action[:, :48].reshape(-1, 16, 3), dim=-1).mean()
        ),
        "policy_entropy": float((-sample.log_prob).mean()),
    }


@dataclass(frozen=True, slots=True)
class PilotTrainingOutcome:
    classification: str
    episodes: int
    gradient_updates: int
    runtime_s: float
    best_checkpoint: Path
    training_history: tuple[dict[str, Any], ...]
    evaluation_history: tuple[dict[str, Any], ...]


def run_focused_sac_pilot(
    *,
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    replay: TerminalReplayBuffer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    reward_config: RLWhipRewardConfig,
    training_bank: InitialStateBank,
    canonical_bank: InitialStateBank,
    validation_bank: InitialStateBank,
    canonical_specification: ContextSpecification,
    heldout_specification: ContextSpecification,
    canonical_target_local_m: torch.Tensor,
    config: dict[str, Any],
    artifact_directory: Path,
) -> PilotTrainingOutcome:
    sac = config["sac"]
    budget = config["budget"]
    evaluation_interval = int(config["validation"]["evaluation_interval_episodes"])
    checkpoint_directory = artifact_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_directory / "latest.pt"
    best = checkpoint_directory / "best_nearcanonical.pt"
    context_generator = torch.Generator().manual_seed(int(config["seed"]) + 1000)
    replay_generator = torch.Generator().manual_seed(int(config["seed"]) + 2000)
    training_history: list[dict[str, Any]] = []
    baseline = {
        "episodes": 0,
        "canonical": evaluate_pilot_policy(
            agent, normalizer, simulator, task, reward_config, canonical_bank, canonical_specification
        ),
        "nearcanonical": evaluate_pilot_policy(
            agent, normalizer, simulator, task, reward_config, validation_bank, heldout_specification
        ),
    }
    evaluation_history: list[dict[str, Any]] = [baseline]
    episodes = 0
    updates = 0
    start = time.perf_counter()
    deadline = start + float(budget["maximum_wall_clock_s"])
    maximum_episodes = int(budget["maximum_collected_episodes"])
    next_evaluation = evaluation_interval
    best_key = (-1.0, float("-inf"), float("-inf"), float("-inf"))
    classification = "SAC_STILL_EXPLORATION_LIMITED"

    while episodes < maximum_episodes and time.perf_counter() < deadline:
        collection_start = time.perf_counter()
        collection = collect_pilot_batch(
            agent,
            normalizer,
            replay,
            simulator,
            task,
            reward_config,
            training_bank,
            count=int(sac["collection_batch_size"]),
            generator=context_generator,
            canonical_target_local_m=canonical_target_local_m,
        )
        episodes += int(sac["collection_batch_size"])
        collection_runtime = time.perf_counter() - collection_start
        update_rows: list[dict[str, float]] = []
        update_start = time.perf_counter()
        if episodes >= int(sac["initial_exploration_episodes"]) and len(replay) >= int(
            sac["minibatch_size"]
        ):
            for _ in range(int(sac["updates_per_collection_batch"])):
                batch = replay.sample(
                    int(sac["minibatch_size"]),
                    device=simulator.device,
                    generator=replay_generator,
                )
                update_rows.append(
                    agent.update(batch.context, batch.action, batch.scaled_reward)
                )
                updates += 1
        update_runtime = time.perf_counter() - update_start
        averaged = (
            {
                name: sum(row[name] for row in update_rows) / len(update_rows)
                for name in update_rows[0]
            }
            if update_rows
            else {}
        )
        training_history.append(
            {
                "episodes": episodes,
                "gradient_updates": updates,
                **collection,
                **averaged,
                "replay_size": len(replay),
                "collection_runtime_s": collection_runtime,
                "update_runtime_s": update_runtime,
                "episodes_per_s": int(sac["collection_batch_size"])
                / max(collection_runtime, 1.0e-9),
                "elapsed_s": time.perf_counter() - start,
            }
        )
        if episodes < next_evaluation and episodes < maximum_episodes:
            continue
        evaluation = {
            "episodes": episodes,
            "gradient_updates": updates,
            "canonical": evaluate_pilot_policy(
                agent,
                normalizer,
                simulator,
                task,
                reward_config,
                canonical_bank,
                canonical_specification,
            ),
            "nearcanonical": evaluate_pilot_policy(
                agent,
                normalizer,
                simulator,
                task,
                reward_config,
                validation_bank,
                heldout_specification,
            ),
            "elapsed_s": time.perf_counter() - start,
        }
        evaluation_history.append(evaluation)
        next_evaluation += evaluation_interval
        held = evaluation["nearcanonical"]
        key = (
            held["hard_success_rate"],
            -held["median_tip_error_m"],
            held["median_directed_tip_speed_m_s"],
            held["feasible_rate"],
        )
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
            "SAC_PILOT_EVAL "
            + json.dumps(
                {
                    "episodes": episodes,
                    "canonical_success": evaluation["canonical"]["hard_success_rate"],
                    "nearcanonical_success": held["hard_success_rate"],
                    "nearcanonical_feasible": held["feasible_rate"],
                    "median_tip_m": held["median_tip_error_m"],
                    "mean_reward": held["mean_reward"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if evaluation["canonical"]["hard_success_rate"] == 1.0 and held[
            "hard_success_rate"
        ] >= float(config["pilot_classification"]["strong_heldout_success_rate"]):
            classification = "PASS"
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
    if not best.is_file():
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

    restore_generator = torch.Generator().manual_seed(0)
    load_training_checkpoint(best, agent=agent, context_generator=restore_generator)
    final_canonical = evaluate_pilot_policy(
        agent, normalizer, simulator, task, reward_config, canonical_bank, canonical_specification
    )
    final_held = evaluate_pilot_policy(
        agent, normalizer, simulator, task, reward_config, validation_bank, heldout_specification
    )
    if classification != "PASS":
        criteria = config["pilot_classification"]
        baseline_held = baseline["nearcanonical"]
        promising = (
            final_held["median_tip_error_m"]
            <= baseline_held["median_tip_error_m"]
            * (1.0 - float(criteria["promising_tip_error_reduction_fraction"]))
            and final_held["median_directed_tip_speed_m_s"]
            >= baseline_held["median_directed_tip_speed_m_s"]
            + float(criteria["promising_directed_speed_increase_m_s"])
            and final_held["median_direction_error_deg"]
            <= baseline_held["median_direction_error_deg"]
            - float(criteria["promising_direction_improvement_deg"])
            and final_held["feasible_rate"]
            >= float(criteria["promising_minimum_feasible_rate"])
        )
        classification = "PROMISING" if promising else "SAC_STILL_EXPLORATION_LIMITED"
    return PilotTrainingOutcome(
        classification,
        episodes,
        updates,
        time.perf_counter() - start,
        best,
        tuple(training_history),
        tuple(evaluation_history),
    )
