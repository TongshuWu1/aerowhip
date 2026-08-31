"""Collection, validation, checkpointing, and training for nominal one-shot SAC."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from planning.cem_task import VariableDurationWhipTask
from simulator.simulator import CoupledSimulator

from .context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    pad_context_specification,
    sample_context_specification,
)
from .normalization import FixedContextNormalizer
from .replay import TerminalReplayBuffer
from .sac import TerminalSacAgent
from .state_bank import InitialStateBank


@dataclass(frozen=True, slots=True)
class ValidationSuite:
    canonical: ContextSpecification
    held_out_iid: ContextSpecification
    edge_domain: ContextSpecification

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            canonical_state_indices=self.canonical.state_indices.numpy(),
            canonical_target_position_local_m=self.canonical.target_position_local_m.numpy(),
            canonical_desired_direction_local=self.canonical.desired_direction_local.numpy(),
            iid_state_indices=self.held_out_iid.state_indices.numpy(),
            iid_target_position_local_m=self.held_out_iid.target_position_local_m.numpy(),
            iid_desired_direction_local=self.held_out_iid.desired_direction_local.numpy(),
            edge_state_indices=self.edge_domain.state_indices.numpy(),
            edge_target_position_local_m=self.edge_domain.target_position_local_m.numpy(),
            edge_desired_direction_local=self.edge_domain.desired_direction_local.numpy(),
            excluded_from_replay=np.asarray(True),
            generated_from_physical_takes=np.asarray(False),
        )


def estimate_fixed_context_normalizer(
    simulator: CoupledSimulator,
    bank: InitialStateBank,
    *,
    sample_count: int,
    sample_batch_size: int,
    seed: int,
    standard_deviation_floor: float,
    canonical_fraction: float,
) -> FixedContextNormalizer:
    """Fit once from training-state/target contexts; validation is never read."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows: list[torch.Tensor] = []
    remaining = sample_count
    while remaining:
        batch = min(sample_batch_size, remaining)
        specification = sample_context_specification(
            bank,
            count=batch,
            generator=generator,
            split="training",
            canonical_fraction=canonical_fraction,
        )
        context = build_context_from_specification(simulator, bank, specification)
        rows.append(context.to_tensor().detach().cpu())
        remaining -= batch
    return FixedContextNormalizer.fit(torch.cat(rows, dim=0), floor=standard_deviation_floor)


def _summary(values: torch.Tensor) -> tuple[float, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.median())


def evaluate_deterministic_policy(
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    bank: InitialStateBank,
    specification: ContextSpecification,
    *,
    canonical_evaluation_batch_size: int = 2048,
) -> dict[str, Any]:
    """Evaluate a frozen validation set with the deterministic transformed mean."""

    padded = pad_context_specification(specification, canonical_evaluation_batch_size)
    context = build_context_from_specification(simulator, bank, padded)
    features = normalizer.normalize(context.to_tensor())
    with torch.no_grad():
        action = agent.actor(features, deterministic=True).deterministic_mean_action
        result = __import__("learning.one_shot_env", fromlist=["evaluate_open_loop_batch"]).evaluate_open_loop_batch(
            simulator, context, action, task
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
    displacement = result.max_uav_displacement_m[:count]
    speed = result.max_uav_speed_m_s[:count]
    raw_reward = result.reward[:count]
    mean_tip, median_tip = _summary(tip)
    mean_directed, median_directed = _summary(directed)
    mean_direction, median_direction = _summary(direction)
    return {
        "split": specification.split,
        "count": count,
        "hard_success_rate": float(success.float().mean()),
        "feasible_rate": float(feasible.float().mean()),
        "finite_rate": float(finite.float().mean()),
        "tip_first_rate": float((marker == 10).float().mean()),
        "mean_tip_error_m": mean_tip,
        "median_tip_error_m": median_tip,
        "mean_directed_tip_speed_m_s": mean_directed,
        "median_directed_tip_speed_m_s": median_directed,
        "mean_direction_error_deg": mean_direction,
        "median_direction_error_deg": median_direction,
        "mean_maneuver_duration_s": float(duration.mean()),
        "uav_displacement_violation_rate": float(
            (displacement > task.maximum_uav_displacement_m).float().mean()
        ),
        "uav_speed_violation_rate": float((speed > task.maximum_uav_speed_m_s).float().mean()),
        "mean_raw_reward": float(raw_reward.mean()),
        "first_row": result.row(0),
    }


def collect_training_batch(
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    replay: TerminalReplayBuffer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    training_bank: InitialStateBank,
    *,
    batch_size: int,
    context_generator: torch.Generator,
    canonical_fraction: float,
    reward_scale_divisor: float,
) -> dict[str, float]:
    """One policy query per context followed by detached black-box rollouts."""

    specification = sample_context_specification(
        training_bank,
        count=batch_size,
        generator=context_generator,
        split="training",
        canonical_fraction=canonical_fraction,
    )
    context = build_context_from_specification(simulator, training_bank, specification)
    normalized_context = normalizer.normalize(context.to_tensor())
    with torch.no_grad():
        sample = agent.actor(normalized_context)
        detached_action = sample.normalized_action.detach()
        result = __import__("learning.one_shot_env", fromlist=["evaluate_open_loop_batch"]).evaluate_open_loop_batch(
            simulator, context, detached_action, task
        )
    scaled_reward = result.reward / reward_scale_divisor
    replay.add(
        context=normalized_context,
        action=detached_action,
        scaled_reward=scaled_reward,
        raw_reward=result.reward,
        success=result.task_success,
        feasible=result.feasible,
        tip_distance_m=result.tip_min_distance_m,
        directed_speed_m_s=result.directed_tip_speed_m_s,
        direction_error_deg=result.direction_error_deg,
    )
    return {
        "mean_raw_reward": float(result.reward.mean()),
        "mean_scaled_reward": float(scaled_reward.mean()),
        "success_rate": float(result.task_success.float().mean()),
        "feasible_rate": float(result.feasible.float().mean()),
        "finite_rate": float(result.rollout_finite.float().mean()),
        "mean_tip_distance_m": float(result.tip_min_distance_m.mean()),
        "mean_duration_s": float(result.decoded_action.duration_s.mean()),
        "acceleration_normalized_norm_mean": float(
            torch.linalg.vector_norm(detached_action[:, :48].reshape(-1, 16, 3), dim=-1).mean()
        ),
        "policy_entropy": float((-sample.log_prob).mean()),
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    agent: TerminalSacAgent,
    episodes: int,
    gradient_updates: int,
    training_history: list[dict[str, Any]],
    evaluation_history: list[dict[str, Any]],
    context_generator: torch.Generator,
    replay_metadata: dict[str, Any],
) -> None:
    payload = {
        "schema": "oneshot_sac_nominal_training_checkpoint_v1",
        "agent": agent.checkpoint(),
        "episodes": int(episodes),
        "gradient_updates": int(gradient_updates),
        "training_history": training_history,
        "evaluation_history": evaluation_history,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "context_generator_state": context_generator.get_state(),
        "replay_metadata": replay_metadata,
        "model_physics": "NOMINAL_ONLY",
        "cem_training_data": "NOT USED",
    }
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_training_checkpoint(
    path: str | Path,
    *,
    agent: TerminalSacAgent,
    context_generator: torch.Generator,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=agent.log_alpha.device, weights_only=False)
    if payload.get("schema") != "oneshot_sac_nominal_training_checkpoint_v1":
        raise ValueError("Unsupported one-shot SAC training checkpoint.")
    agent.load_checkpoint(payload["agent"])
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    if torch.cuda.is_available() and payload["torch_cuda_rng_states"]:
        torch.cuda.set_rng_state_all(
            [state.detach().cpu() for state in payload["torch_cuda_rng_states"]]
        )
    context_generator.set_state(payload["context_generator_state"].cpu())
    return payload


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    classification: str
    episodes: int
    gradient_updates: int
    runtime_s: float
    best_checkpoint: Path
    latest_checkpoint: Path
    training_history: tuple[dict[str, Any], ...]
    evaluation_history: tuple[dict[str, Any], ...]


def run_terminal_sac_training(
    *,
    agent: TerminalSacAgent,
    normalizer: FixedContextNormalizer,
    replay: TerminalReplayBuffer,
    simulator: CoupledSimulator,
    task: VariableDurationWhipTask,
    training_bank: InitialStateBank,
    canonical_bank: InitialStateBank,
    validation_bank: InitialStateBank,
    validation: ValidationSuite,
    config: dict[str, Any],
    artifact_directory: Path,
) -> TrainingOutcome:
    sac = config["sac"]
    budget = config["budget"]
    early = config["early_stop"]
    target = config["target_domain_local_m"]
    validation_config = config["validation"]
    checkpoint_directory = artifact_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_directory / "latest.pt"
    best = checkpoint_directory / "best_held_out_iid.pt"
    replay_path = checkpoint_directory / "replay_latest.pt"
    context_generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 1000)
    replay_generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 2000)
    training_history: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    episodes = 0
    updates = 0
    best_key = (-1.0, float("-inf"), float("-inf"), float("-inf"))
    start = time.perf_counter()
    deadline = start + float(budget["maximum_wall_clock_s"])
    next_evaluation = int(validation_config["evaluation_interval_episodes"])
    initial_exploration = int(sac["initial_exploration_episodes"])
    batch_size = int(sac["collection_batch_size"])
    maximum_episodes = int(budget["maximum_collected_episodes"])
    classification = "SAC_TRAINING_FAILED"

    while episodes < maximum_episodes and time.perf_counter() < deadline:
        collection_start = time.perf_counter()
        collection = collect_training_batch(
            agent,
            normalizer,
            replay,
            simulator,
            task,
            training_bank,
            batch_size=batch_size,
            context_generator=context_generator,
            canonical_fraction=float(target["canonical_training_fraction"]),
            reward_scale_divisor=float(config["reward"]["sac_scale_divisor"]),
        )
        episodes += batch_size
        collection_runtime = time.perf_counter() - collection_start
        update_rows: list[dict[str, float]] = []
        update_start = time.perf_counter()
        if episodes >= initial_exploration and len(replay) >= int(sac["minibatch_size"]):
            for _ in range(int(sac["updates_per_collection_batch"])):
                batch = replay.sample(
                    int(sac["minibatch_size"]),
                    device=simulator.device,
                    generator=replay_generator,
                )
                update_rows.append(agent.update(batch.context, batch.action, batch.scaled_reward))
                updates += 1
        update_runtime = time.perf_counter() - update_start
        averaged_update = {
            name: sum(row[name] for row in update_rows) / len(update_rows)
            for name in update_rows[0]
        } if update_rows else {}
        training_history.append(
            {
                "episodes": episodes,
                "gradient_updates": updates,
                **collection,
                **averaged_update,
                "replay_size": len(replay),
                "collection_runtime_s": collection_runtime,
                "update_runtime_s": update_runtime,
                "episodes_per_s": batch_size / max(collection_runtime, 1.0e-9),
                "elapsed_s": time.perf_counter() - start,
            }
        )
        if episodes < next_evaluation and episodes < maximum_episodes:
            continue

        evaluation = {
            "episodes": episodes,
            "gradient_updates": updates,
            "canonical": evaluate_deterministic_policy(
                agent, normalizer, simulator, task, canonical_bank, validation.canonical
            ),
            "held_out_iid": evaluate_deterministic_policy(
                agent, normalizer, simulator, task, validation_bank, validation.held_out_iid
            ),
            "edge_domain": evaluate_deterministic_policy(
                agent, normalizer, simulator, task, validation_bank, validation.edge_domain
            ),
            "elapsed_s": time.perf_counter() - start,
        }
        evaluation_history.append(evaluation)
        next_evaluation += int(validation_config["evaluation_interval_episodes"])
        held = evaluation["held_out_iid"]
        key = (
            held["hard_success_rate"],
            -held["median_tip_error_m"],
            held["mean_directed_tip_speed_m_s"],
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
        # Preserve enough replay for an exact continuation.  One file is
        # replaced rather than accumulating enormous historical copies.
        replay.save(replay_path)
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
        stable = False
        checkpoints = int(early["stability_checkpoints"])
        if len(evaluation_history) >= checkpoints:
            recent = [item["held_out_iid"]["hard_success_rate"] for item in evaluation_history[-checkpoints:]]
            stable = max(recent) - min(recent) <= float(early["maximum_success_improvement_fraction"])
        if (
            evaluation["canonical"]["hard_success_rate"] >= 1.0
            and held["hard_success_rate"] >= float(early["held_out_iid_success_rate"])
            and held["feasible_rate"] >= float(early["held_out_iid_feasible_rate"])
            and stable
        ):
            classification = "TRAINED"
            break

    if classification != "TRAINED":
        canonical_successes = [row["canonical"]["hard_success_rate"] for row in evaluation_history]
        iid_successes = [row["held_out_iid"]["hard_success_rate"] for row in evaluation_history]
        if not canonical_successes or max(canonical_successes) < 1.0 or (iid_successes and max(iid_successes) < 0.10):
            classification = "SAC_EXPLORATION_LIMITED"
        else:
            classification = "SAC_TRAINING_FAILED"
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
        latest.replace(best)
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
    return TrainingOutcome(
        classification,
        episodes,
        updates,
        time.perf_counter() - start,
        best,
        latest,
        tuple(training_history),
        tuple(evaluation_history),
    )
