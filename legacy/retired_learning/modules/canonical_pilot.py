"""Single-context SAC discovery pilot for the progress-shaped whip reward."""

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

from .context_sampling import ContextSpecification, build_context_from_specification
from .normalization import FixedContextNormalizer
from .replay import TerminalReplayBuffer
from .sac import TerminalSacAgent
from .state_bank import InitialStateBank
from .training import load_training_checkpoint, save_training_checkpoint


def repeated_canonical_specification(
    canonical: ContextSpecification,
    count: int,
    *,
    split: str,
) -> ContextSpecification:
    if canonical.count != 1 or count < 1:
        raise ValueError("A single canonical row and positive repeat count are required.")
    return ContextSpecification(
        torch.zeros(count, dtype=torch.int64),
        canonical.target_position_local_m.repeat(count, 1),
        canonical.desired_direction_local.repeat(count, 1),
        split,
    )


def behavior_level(
    *,
    success: bool,
    progress: float,
    minimum_tip_distance_m: float,
    directed_tip_speed_m_s: float,
    level4_minimum_directed_speed_m_s: float,
) -> int:
    if success:
        return 5
    if (
        minimum_tip_distance_m <= 0.10
        and directed_tip_speed_m_s >= level4_minimum_directed_speed_m_s
    ):
        return 4
    if minimum_tip_distance_m <= 0.20:
        return 3
    if progress >= 0.50:
        return 2
    if progress >= 0.25:
        return 1
    return 0


def evaluate_canonical_policy(
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    reward_config: RLWhipRewardConfig,
    canonical_bank: InitialStateBank,
    canonical_specification: ContextSpecification,
    *,
    evaluation_batch_size: int,
    level4_minimum_directed_speed_m_s: float,
) -> dict[str, Any]:
    specification = repeated_canonical_specification(
        canonical_specification, evaluation_batch_size, split="canonical_evaluation"
    )
    context = build_context_from_specification(simulator, canonical_bank, specification)
    normalized = normalizer.normalize(context.to_tensor())
    if not torch.equal(normalized, normalized[:1].expand_as(normalized)):
        raise RuntimeError("Canonical evaluation context rows are not identical.")
    with torch.no_grad():
        sample = agent.actor(normalized, deterministic=True)
        result = __import__(
            "learning.one_shot_env", fromlist=["evaluate_open_loop_batch"]
        ).evaluate_open_loop_batch(
            simulator,
            context,
            sample.deterministic_mean_action,
            task,
            rl_reward_config=reward_config,
        )
    row = result.row(0)
    components = row["reward_components"]
    knot_norms = torch.linalg.vector_norm(
        sample.deterministic_mean_action[0, :48].reshape(16, 3), dim=-1
    )
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
        "initial_tip_distance_m": float(components["initial_tip_distance_m"]),
        "minimum_tip_distance_m": float(components["minimum_tip_distance_m"]),
        "deterministic_acceleration_knot_normalized_norms": knot_norms.detach()
        .cpu()
        .tolist(),
        "identical_context_rows": True,
        "evaluation_batch_size": evaluation_batch_size,
    }


def collect_canonical_batch(
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
) -> dict[str, float | bool]:
    specification = repeated_canonical_specification(
        canonical_specification, batch_size, split="canonical_training"
    )
    context = build_context_from_specification(simulator, canonical_bank, specification)
    normalized = normalizer.normalize(context.to_tensor())
    identical_context = torch.equal(normalized, normalized[:1].expand_as(normalized))
    if not identical_context:
        raise RuntimeError("Every canonical SAC collection row must have the same context.")
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
    if float(action.std(dim=0).mean()) <= 0.0:
        raise RuntimeError("Stochastic actions did not vary across canonical rows.")
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
        raise RuntimeError("Progress-shaped collection requires RL reward components.")
    duration = result.decoded_action.duration_s
    knot_norm = torch.linalg.vector_norm(action[:, :48].reshape(-1, 16, 3), dim=-1)
    return {
        "mean_reward": float(result.reward.mean()),
        "minimum_reward": float(result.reward.min()),
        "maximum_reward": float(result.reward.max()),
        "success_rate": float(result.task_success.float().mean()),
        "feasible_rate": float(result.feasible.float().mean()),
        "finite_rate": float(result.rollout_finite.float().mean()),
        "mean_progress": float(components.normalized_progress.mean()),
        "maximum_progress": float(components.normalized_progress.max()),
        "mean_tip_distance_m": float(result.tip_min_distance_m.mean()),
        "minimum_tip_distance_m": float(result.tip_min_distance_m.min()),
        "mean_directed_tip_speed_m_s": float(result.directed_tip_speed_m_s.mean()),
        "maximum_directed_tip_speed_m_s": float(result.directed_tip_speed_m_s.max()),
        "mean_direction_error_deg": float(result.direction_error_deg.mean()),
        "mean_duration_s": float(duration.mean()),
        "minimum_duration_s": float(duration.min()),
        "maximum_duration_s": float(duration.max()),
        "acceleration_normalized_norm_mean": float(knot_norm.mean()),
        "acceleration_normalized_norm_p95": float(torch.quantile(knot_norm, 0.95)),
        "policy_entropy": float((-sample.log_prob).mean()),
        "identical_context_rows": identical_context,
        "action_std_across_rows": float(action.std(dim=0).mean()),
    }


def critic_diagnostic(
    agent: TerminalSacAgent,
    replay: TerminalReplayBuffer,
    *,
    count: int,
    device: torch.device,
) -> dict[str, float | int | None]:
    rows = min(len(replay), count)
    if rows < 2:
        return {"count": rows, "reward_q_correlation": None}
    context = replay.context[:rows].to(device)
    action = replay.action[:rows].to(device)
    reward = replay.scaled_reward[:rows, 0].to(device)
    with torch.no_grad():
        q1 = agent.critic1(context, action)[:, 0]
        q2 = agent.critic2(context, action)[:, 0]
        q = torch.minimum(q1, q2)
    centered_reward = reward - reward.mean()
    centered_q = q - q.mean()
    denominator = torch.sqrt(
        centered_reward.square().sum() * centered_q.square().sum()
    )
    correlation = (
        float((centered_reward * centered_q).sum() / denominator)
        if float(denominator) > 0.0
        else None
    )
    return {
        "count": rows,
        "reward_q_correlation": correlation,
        "true_reward_minimum": float(reward.min()),
        "true_reward_maximum": float(reward.max()),
        "q1_minimum": float(q1.min()),
        "q1_maximum": float(q1.max()),
        "q2_minimum": float(q2.min()),
        "q2_maximum": float(q2.max()),
    }


def _checkpoint_key(evaluation: dict[str, Any]) -> tuple[float, ...]:
    if evaluation["task_success"]:
        return (
            2.0,
            float(evaluation["reward"]),
            -float(evaluation["tip_min_distance_m"]),
            float(evaluation["directed_tip_speed_m_s"]),
            -float(evaluation["direction_error_deg"]),
            float(evaluation["feasible"]),
        )
    return (
        1.0,
        float(evaluation["behavior_level"]),
        float(evaluation["feasible"]),
        float(evaluation["progress"]),
        -float(evaluation["tip_min_distance_m"]),
        float(evaluation["directed_tip_speed_m_s"]),
    )


@dataclass(frozen=True, slots=True)
class CanonicalPilotOutcome:
    classification: str
    episodes: int
    gradient_updates: int
    runtime_s: float
    best_checkpoint: Path
    latest_checkpoint: Path
    selected_evaluation: dict[str, Any]
    highest_behavior_level: int
    training_history: tuple[dict[str, Any], ...]
    evaluation_history: tuple[dict[str, Any], ...]


def run_canonical_sac_pilot(
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
) -> CanonicalPilotOutcome:
    sac = config["sac"]
    validation = config["validation"]
    budget = config["budget"]
    batch_size = int(sac["collection_batch_size"])
    checkpoint_directory = artifact_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_directory / "latest.pt"
    best = checkpoint_directory / "best_behavior.pt"
    context_generator = torch.Generator().manual_seed(int(config["seed"]) + 1000)
    replay_generator = torch.Generator().manual_seed(int(config["seed"]) + 2000)
    evaluation_batch_size = batch_size
    level4_speed = float(validation["level4_minimum_directed_speed_m_s"])

    baseline = evaluate_canonical_policy(
        agent,
        normalizer,
        simulator,
        task,
        reward_config,
        canonical_bank,
        canonical_specification,
        evaluation_batch_size=evaluation_batch_size,
        level4_minimum_directed_speed_m_s=level4_speed,
    )
    evaluation_history: list[dict[str, Any]] = [
        {"episodes": 0, "gradient_updates": 0, "canonical": baseline, "elapsed_s": 0.0}
    ]
    training_history: list[dict[str, Any]] = []
    best_key = _checkpoint_key(baseline)
    selected = baseline
    episodes = 0
    updates = 0
    highest_level = int(baseline["behavior_level"])
    start = time.perf_counter()
    deadline = start + float(budget["maximum_wall_clock_s"])
    maximum_episodes = int(budget["maximum_collected_episodes"])
    next_evaluation = int(validation["evaluation_interval_episodes"])
    classification = "SAC_CANONICAL_EXPLORATION_LIMITED"

    # Preserve the initial actor as a valid diagnostic checkpoint.
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

    while episodes < maximum_episodes and time.perf_counter() < deadline:
        collection_start = time.perf_counter()
        collection = collect_canonical_batch(
            agent,
            normalizer,
            replay,
            simulator,
            task,
            reward_config,
            canonical_bank,
            canonical_specification,
            batch_size=batch_size,
        )
        episodes += batch_size
        collection_runtime = time.perf_counter() - collection_start

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
                update = agent.update(sample.context, sample.action, sample.scaled_reward)
                if not all(torch.isfinite(torch.tensor(value)) for value in update.values()):
                    raise RuntimeError("Non-finite SAC optimizer diagnostic encountered.")
                updates_this_batch.append(update)
                updates += 1
        update_runtime = time.perf_counter() - update_start
        averaged = (
            {
                name: sum(row[name] for row in updates_this_batch) / len(updates_this_batch)
                for name in updates_this_batch[0]
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
                "replay_size": len(replay),
                "collection_runtime_s": collection_runtime,
                "update_runtime_s": update_runtime,
                "episodes_per_s": batch_size / max(collection_runtime, 1.0e-9),
                "elapsed_s": time.perf_counter() - start,
            }
        )

        if episodes < next_evaluation and episodes < maximum_episodes:
            continue
        canonical = evaluate_canonical_policy(
            agent,
            normalizer,
            simulator,
            task,
            reward_config,
            canonical_bank,
            canonical_specification,
            evaluation_batch_size=evaluation_batch_size,
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
            "canonical": canonical,
            "critic_diagnostic": diagnostic,
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
        key = _checkpoint_key(canonical)
        if key > best_key:
            best_key = key
            selected = canonical
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
            "SAC_CANONICAL_EVAL "
            + json.dumps(
                {
                    "episodes": episodes,
                    "success": canonical["task_success"],
                    "feasible": canonical["feasible"],
                    "level": canonical["behavior_level"],
                    "progress": canonical["progress"],
                    "tip_m": canonical["tip_min_distance_m"],
                    "directed_m_s": canonical["directed_tip_speed_m_s"],
                    "duration_s": canonical["duration_s"],
                    "reward": canonical["reward"],
                    "reward_q_correlation": diagnostic.get("reward_q_correlation"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if canonical["task_success"]:
            classification = "SAC_CANONICAL_PASS"
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

    restore_generator = torch.Generator().manual_seed(0)
    load_training_checkpoint(best, agent=agent, context_generator=restore_generator)
    selected = evaluate_canonical_policy(
        agent,
        normalizer,
        simulator,
        task,
        reward_config,
        canonical_bank,
        canonical_specification,
        evaluation_batch_size=evaluation_batch_size,
        level4_minimum_directed_speed_m_s=level4_speed,
    )
    if selected["task_success"]:
        classification = "SAC_CANONICAL_PASS"
    elif int(selected["behavior_level"]) >= 4:
        classification = "SAC_CANONICAL_PROMISING"
    else:
        classification = "SAC_CANONICAL_EXPLORATION_LIMITED"
    return CanonicalPilotOutcome(
        classification=classification,
        episodes=episodes,
        gradient_updates=updates,
        runtime_s=time.perf_counter() - start,
        best_checkpoint=best,
        latest_checkpoint=latest,
        selected_evaluation=selected,
        highest_behavior_level=highest_level,
        training_history=tuple(training_history),
        evaluation_history=tuple(evaluation_history),
    )
