"""Run the isolated one-million-episode PPO comparison on the simple whip MDP."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from learning.sequential_sac_env import SEQUENTIAL_WHIP_OBSERVATION_DIM
from learning.ppo_validation import (
    FixedMildStateValidationPanel,
    append_validation_result,
    write_training_plots,
)
from learning.simple_ppo import (
    SIMPLE_PPO_ACTION_DIM,
    PPORollout,
    SimplePPOAgent,
)
from run_simple_sac import _build_environment


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "simple_sequential_ppo_10s_v1.json"


def _utc_stamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H%M%S.") + f"{now.microsecond:06d}Z"


def _safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=_safe) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "simple_sequential_ppo_10s_v1":
        raise ValueError("Unsupported simple-PPO configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Simple PPO requires the pinned production model freeze.")
    if config.get("comparison_environment") not in {
        "simple_sequential_sac_10s_v1",
        "task_whip_once_10s_v1",
    }:
        raise ValueError("Unsupported PPO comparison environment.")
    if int(config["requested_episodes"]) != 1_000_000:
        raise ValueError("The PPO pilot requests exactly one million episodes.")
    if not bool(config["batch_aligned_overshoot_allowed"]):
        raise ValueError("The fixed collection batch requires aligned overshoot.")
    if float(config["episode_duration_s"]) != 10.0:
        raise ValueError("The PPO comparison horizon is exactly 10 seconds.")
    if int(config["action"]["dimensions"]) != SIMPLE_PPO_ACTION_DIM:
        raise ValueError("Simple PPO requires the unchanged six-dimensional action.")
    success_mode = str(config.get("success_mode", "simple_endpoint"))
    if success_mode not in {"simple_endpoint", "task_whip_once"}:
        raise ValueError("Unsupported PPO success mode.")
    if success_mode == "task_whip_once":
        if str(config.get("reward_mode")) != "whip_potential":
            raise ValueError("Task-whip PPO requires bounded potential reward shaping.")
    validation = config.get("validation")
    if validation is not None and bool(validation.get("enabled", False)):
        if int(validation["episodes"]) != 10:
            raise ValueError("Validated PPO requires exactly ten validation rollouts.")
        interval = int(validation["every_episodes"])
        if interval <= 0 or interval % int(config["collection_batch"]) != 0:
            raise ValueError("Validation cadence must align with the collection batch.")
        if not 0.0 < float(validation["maximum_distance_quantile"]) <= 1.0:
            raise ValueError("Validation state-distance quantile must lie in (0,1].")
    return config


def _build_agent(config: dict[str, Any], device: torch.device) -> SimplePPOAgent:
    source = config["ppo"]
    return SimplePPOAgent(
        SEQUENTIAL_WHIP_OBSERVATION_DIM,
        SIMPLE_PPO_ACTION_DIM,
        device=device,
        hidden_dim=int(source["hidden_dim"]),
        learning_rate=float(source["learning_rate"]),
        gamma=float(source["gamma"]),
        gae_lambda=float(source["gae_lambda"]),
        clip_ratio=float(source["clip_ratio"]),
        value_coefficient=float(source["value_coefficient"]),
        entropy_coefficient=float(source["entropy_coefficient"]),
        maximum_gradient_norm=float(source["maximum_gradient_norm"]),
        target_kl=float(source["target_kl"]),
    )


def preflight(config: dict[str, Any]) -> None:
    """Run one complete production episode batch and one PPO update."""

    torch.manual_seed(int(config["seed"]))
    environment, device = _build_environment(config, batch_size=4)
    agent = _build_agent(config, device)
    observation = environment.reset()
    rollout = PPORollout.allocate(
        environment.control_step_count,
        4,
        SEQUENTIAL_WHIP_OBSERVATION_DIM,
        SIMPLE_PPO_ACTION_DIM,
        device=device,
    )
    result = None
    for step in range(environment.control_step_count):
        action, log_probability, value = agent.act(observation)
        result = environment.step(action)
        rollout.observations[step].copy_(observation)
        rollout.actions[step].copy_(action)
        rollout.rewards[step].copy_(result.reward)
        rollout.dones[step].copy_(result.done)
        rollout.masks[step].copy_(result.include_transition[:, None].float())
        rollout.log_probabilities[step].copy_(log_probability)
        rollout.values[step].copy_(value)
        observation = result.next_observation
    assert result is not None
    metrics = agent.update(
        rollout,
        minibatch_size=128,
        epochs=2,
        generator=torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 1),
    )
    print(
        json.dumps(
            {
                "preflight": "PASS",
                "device": str(device),
                "observation_shape": list(observation.shape),
                "action_shape": list(action.shape),
                "simulated_episode_duration_s": config["episode_duration_s"],
                "finite_rollout": bool(torch.isfinite(rollout.rewards).all()),
                "numerical_failures": int(environment.failed.sum().cpu()),
                "ppo_update": asdict(metrics),
            },
            indent=2,
        ),
        flush=True,
    )


def _save_checkpoint(
    artifact: Path,
    agent: SimplePPOAgent,
    *,
    episodes: int,
    successes: int,
    endpoint_successes: int,
    legacy_scientific_successes: int,
    elapsed_s: float,
    update_generator: torch.Generator,
) -> None:
    payload = agent.checkpoint()
    payload.update(
        {
            "episodes": int(episodes),
            "successes": int(successes),
            "endpoint_successes": int(endpoint_successes),
            "legacy_scientific_successes": int(legacy_scientific_successes),
            "elapsed_s": float(elapsed_s),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "update_generator_state": update_generator.get_state(),
        }
    )
    checkpoint_directory = artifact / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    temporary = checkpoint_directory / "latest.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, checkpoint_directory / "latest.pt")


def _consider_best_validation(
    artifact: Path,
    agent: SimplePPOAgent,
    result: dict[str, Any],
) -> bool:
    """Persist the best deterministic policy by task success, then compactness."""

    metadata_path = artifact / "best_validation.json"
    previous = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    rank = (
        float(result["validation_success_rate"]),
        -float(result["mean_maximum_uav_displacement_m"]),
        float(result["validation_legacy_scientific_success_rate"]),
    )
    previous_rank = (
        (
            float(previous["validation_success_rate"]),
            -float(previous["mean_maximum_uav_displacement_m"]),
            float(previous["validation_legacy_scientific_success_rate"]),
        )
        if previous is not None
        else None
    )
    if previous_rank is not None and rank <= previous_rank:
        return False
    temporary = artifact / "checkpoints" / "best_validation.pt.tmp"
    torch.save(agent.checkpoint(), temporary)
    os.replace(temporary, artifact / "checkpoints" / "best_validation.pt")
    _write_json(metadata_path, result)
    return True


def train(
    config: dict[str, Any],
    config_path: Path,
    artifact: Path,
    *,
    resume: bool = False,
) -> None:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    if resume:
        if not artifact.is_dir():
            raise FileNotFoundError(f"Cannot resume missing artifact directory: {artifact}")
        if not (artifact / "checkpoints" / "latest.pt").is_file():
            raise FileNotFoundError("Cannot resume without checkpoints/latest.pt.")
        (artifact / "STOP_REQUESTED").unlink(missing_ok=True)
    else:
        artifact.mkdir(parents=True, exist_ok=False)
        (artifact / "checkpoints").mkdir()
        _write_json(artifact / "config.json", config)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "learning" / "simple_ppo.py",
        ROOT / "learning" / "sequential_sac_env.py",
        ROOT / "learning" / "ppo_validation.py",
        ROOT / "run_simple_sac.py",
        config_path,
        ROOT / config["simulator_config"],
        ROOT / config["task_config"],
        ROOT / config["context_normalizer"],
        ROOT / "config" / "active_model.json",
    ]
    initialization = config.get("initialization", {})
    initialization_checkpoint = initialization.get("policy_checkpoint")
    if initialization_checkpoint:
        source_paths.append(ROOT / str(initialization_checkpoint))
    if not resume:
        _write_json(
            artifact / "source_hash_manifest.json",
            {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
        )
        full_mode = config.get("success_mode") == "task_whip_once"
        contract_title = (
            "# Task-whip PPO reward study — 10-second episode\n\n"
            if full_mode
            else "# Validated simple sequential PPO — 10-second whip\n\n"
        )
        success_contract = (
            "success requires the single first target entry to be tip-first with speed and direction; entry timestep is diagnostic only and numerical UAV limits are smooth costs. "
            if full_mode
            else "endpoint success gate is unchanged. "
        )
        (artifact / "RUN_CONTRACT.md").write_text(
            contract_title
            + "Training uses the canonical settled initial state. Every 51,200 episodes, "
            "the deterministic policy is evaluated on the same ten mildly varied, "
            "physically propagated initial states. Training and validation both use "
            "the production UAV + causal residual + 12-node DDER simulator. The policy "
            "action is 3-D acceleration plus roll/pitch/yaw body rates. The reported "
            + success_contract
            + "No CEM, protected data, or hardware "
            "is used. A STOP_REQUESTED file causes a checkpointed cooperative stop.\n",
            encoding="utf-8",
        )

    collection_batch = int(config["collection_batch"])
    environment, device = _build_environment(config, batch_size=collection_batch)
    if device.type != "cuda":
        raise RuntimeError("The one-million-episode PPO comparison requires CUDA.")
    agent = _build_agent(config, device)
    if not resume and initialization_checkpoint:
        source_checkpoint = torch.load(
            ROOT / str(initialization_checkpoint),
            map_location=device,
            weights_only=False,
        )
        agent.policy.load_state_dict(source_checkpoint["policy"])
    ppo = config["ppo"]
    rollout = PPORollout.allocate(
        environment.control_step_count,
        collection_batch,
        SEQUENTIAL_WHIP_OBSERVATION_DIM,
        SIMPLE_PPO_ACTION_DIM,
        device=device,
    )
    update_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    validation_config = config.get("validation", {})
    validation_enabled = bool(validation_config.get("enabled", False))
    validation_panel = None
    if validation_enabled:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        validation_panel = FixedMildStateValidationPanel(config)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        _write_json(artifact / "validation_state_manifest.json", validation_panel.manifest)
    requested_episodes = int(config["requested_episodes"])
    checkpoint_interval = int(config["logging"]["checkpoint_every_episodes"])
    validation_interval = int(
        validation_config.get("every_episodes", checkpoint_interval)
    )
    episodes = 0
    successes = 0
    endpoint_successes = 0
    legacy_scientific_successes = 0
    elapsed_before_resume = 0.0
    if resume:
        checkpoint = torch.load(
            artifact / "checkpoints" / "latest.pt",
            map_location=device,
            weights_only=False,
        )
        agent.policy.load_state_dict(checkpoint["policy"])
        agent.value.load_state_dict(checkpoint["value"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.gradient_updates = int(checkpoint["gradient_updates"])
        episodes = int(checkpoint["episodes"])
        successes = int(checkpoint["successes"])
        endpoint_successes = int(checkpoint.get("endpoint_successes", successes))
        legacy_scientific_successes = int(
            checkpoint.get("legacy_scientific_successes", 0)
        )
        elapsed_before_resume = float(checkpoint.get("elapsed_s", 0.0))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "update_generator_state" in checkpoint:
            update_generator.set_state(checkpoint["update_generator_state"])
    started = time.perf_counter() - elapsed_before_resume
    next_checkpoint = (episodes // checkpoint_interval + 1) * checkpoint_interval
    next_validation = (episodes // validation_interval + 1) * validation_interval
    log_path = artifact / "training_log.csv"
    fields = [
        "episodes",
        "requested_episodes",
        "successes",
        "total_success_rate",
        "batch_success_rate",
        "endpoint_successes",
        "total_endpoint_success_rate",
        "batch_endpoint_success_rate",
        "legacy_scientific_successes",
        "total_legacy_scientific_success_rate",
        "batch_legacy_scientific_success_rate",
        "episodes_per_second",
        "elapsed_s",
        "mean_episode_reward",
        "median_minimum_tip_distance_m",
        "mean_maximum_uav_displacement_m",
        "numerical_failure_rate",
        "gradient_updates",
        "valid_transitions",
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
        "gradient_norm",
        "epochs_completed",
    ]
    _write_json(
        artifact / "status.json",
        {
            "status": "RESUMING" if resume else "STARTING",
            "process_id": os.getpid(),
            "episodes": episodes,
            "requested_episodes": requested_episodes,
            "success_rate": 0.0 if episodes == 0 else successes / episodes,
            "endpoint_success_rate": (
                0.0 if episodes == 0 else endpoint_successes / episodes
            ),
            "episodes_per_second": 0.0,
        },
    )

    try:
        log_mode = "a" if resume else "w"
        with log_path.open(log_mode, newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not resume:
                writer.writeheader()
            stream.flush()
            if validation_panel is not None and not resume:
                initial_validation = validation_panel.evaluate(
                    agent, checkpoint_episodes=0
                )
                append_validation_result(artifact, initial_validation)
                _consider_best_validation(artifact, agent, initial_validation)
                write_training_plots(artifact)
            while episodes < requested_episodes:
                observation = environment.reset()
                for step in range(environment.control_step_count):
                    action, log_probability, value = agent.act(observation)
                    result = environment.step(action)
                    rollout.observations[step].copy_(observation)
                    rollout.actions[step].copy_(action)
                    rollout.rewards[step].copy_(result.reward)
                    rollout.dones[step].copy_(result.done)
                    rollout.masks[step].copy_(
                        result.include_transition[:, None].float()
                    )
                    rollout.log_probabilities[step].copy_(log_probability)
                    rollout.values[step].copy_(value)
                    observation = result.next_observation

                metrics = agent.update(
                    rollout,
                    minibatch_size=int(ppo["minibatch_transitions"]),
                    epochs=int(ppo["update_epochs"]),
                    generator=update_generator,
                )
                batch_successes = int(environment.episode_success.sum().cpu())
                batch_endpoint_successes = int(
                    environment.episode_endpoint_success.sum().cpu()
                )
                batch_legacy_scientific_successes = int(
                    environment.episode_scientific_success.sum().cpu()
                )
                successes += batch_successes
                endpoint_successes += batch_endpoint_successes
                legacy_scientific_successes += batch_legacy_scientific_successes
                episodes += collection_batch
                elapsed = time.perf_counter() - started
                row = {
                    "episodes": episodes,
                    "requested_episodes": requested_episodes,
                    "successes": successes,
                    "total_success_rate": successes / episodes,
                    "batch_success_rate": batch_successes / collection_batch,
                    "endpoint_successes": endpoint_successes,
                    "total_endpoint_success_rate": endpoint_successes / episodes,
                    "batch_endpoint_success_rate": (
                        batch_endpoint_successes / collection_batch
                    ),
                    "legacy_scientific_successes": legacy_scientific_successes,
                    "total_legacy_scientific_success_rate": (
                        legacy_scientific_successes / episodes
                    ),
                    "batch_legacy_scientific_success_rate": (
                        batch_legacy_scientific_successes / collection_batch
                    ),
                    "episodes_per_second": episodes / elapsed,
                    "elapsed_s": elapsed,
                    "mean_episode_reward": float(environment.episode_reward.mean().cpu()),
                    "median_minimum_tip_distance_m": float(
                        environment.episode_minimum_tip_distance.median().cpu()
                    ),
                    "mean_maximum_uav_displacement_m": float(
                        environment.episode_maximum_displacement.mean().cpu()
                    ),
                    "numerical_failure_rate": float(environment.failed.float().mean().cpu()),
                    "gradient_updates": agent.gradient_updates,
                    "valid_transitions": metrics.valid_transitions,
                    **asdict(metrics),
                }
                writer.writerow(row)
                stream.flush()
                _write_json(
                    artifact / "status.json",
                    {
                        "status": "RUNNING",
                        "process_id": os.getpid(),
                        "episodes": episodes,
                        "requested_episodes": requested_episodes,
                        "successes": successes,
                        "success_rate": successes / episodes,
                        "batch_success_rate": batch_successes / collection_batch,
                        "endpoint_success_rate": endpoint_successes / episodes,
                        "batch_endpoint_success_rate": (
                            batch_endpoint_successes / collection_batch
                        ),
                        "legacy_scientific_success_rate": (
                            legacy_scientific_successes / episodes
                        ),
                        "batch_legacy_scientific_success_rate": (
                            batch_legacy_scientific_successes / collection_batch
                        ),
                        "episodes_per_second": episodes / elapsed,
                        "elapsed_s": elapsed,
                        "mean_episode_reward": row["mean_episode_reward"],
                        "median_minimum_tip_distance_m": row[
                            "median_minimum_tip_distance_m"
                        ],
                    },
                )
                print(
                    f"episodes={episodes}/{requested_episodes} "
                    f"batch_success={100.0 * batch_successes / collection_batch:.3f}% "
                    f"total_success={100.0 * successes / episodes:.3f}% "
                    f"endpoint_success={100.0 * endpoint_successes / episodes:.3f}% "
                    f"episodes_per_second={episodes / elapsed:.2f}",
                    flush=True,
                )
                if episodes >= next_checkpoint:
                    _save_checkpoint(
                        artifact,
                        agent,
                        episodes=episodes,
                        successes=successes,
                        endpoint_successes=endpoint_successes,
                        legacy_scientific_successes=legacy_scientific_successes,
                        elapsed_s=elapsed,
                        update_generator=update_generator,
                    )
                    while next_checkpoint <= episodes:
                        next_checkpoint += checkpoint_interval
                if (
                    validation_panel is not None
                    and episodes >= next_validation
                ):
                    validation_result = validation_panel.evaluate(
                        agent, checkpoint_episodes=episodes
                    )
                    append_validation_result(artifact, validation_result)
                    _consider_best_validation(artifact, agent, validation_result)
                    write_training_plots(artifact)
                    print(
                        f"validation_episodes={episodes} "
                        f"success={validation_result['validation_successes']}/"
                        f"{validation_result['validation_episodes']}",
                        flush=True,
                    )
                    while next_validation <= episodes:
                        next_validation += validation_interval
                if (artifact / "STOP_REQUESTED").exists():
                    elapsed = time.perf_counter() - started
                    _save_checkpoint(
                        artifact,
                        agent,
                        episodes=episodes,
                        successes=successes,
                        endpoint_successes=endpoint_successes,
                        legacy_scientific_successes=legacy_scientific_successes,
                        elapsed_s=elapsed,
                        update_generator=update_generator,
                    )
                    write_training_plots(artifact)
                    summary = {
                        "status": "STOPPED_BY_USER",
                        "episodes": episodes,
                        "requested_episodes": requested_episodes,
                        "successes": successes,
                        "endpoint_successes": endpoint_successes,
                        "legacy_scientific_successes": legacy_scientific_successes,
                        "success_rate": successes / episodes,
                        "episodes_per_second": episodes / elapsed,
                        "elapsed_s": elapsed,
                        "latest_durable_checkpoint_episodes": episodes,
                    }
                    _write_json(artifact / "status.json", summary)
                    _write_json(artifact / "stopped_run_summary.json", summary)
                    print(json.dumps(summary, indent=2), flush=True)
                    return
    except BaseException as error:
        elapsed = time.perf_counter() - started
        _write_json(
            artifact / "status.json",
            {
                "status": "FAILED",
                "episodes": episodes,
                "requested_episodes": requested_episodes,
                "successes": successes,
                "success_rate": 0.0 if episodes == 0 else successes / episodes,
                "episodes_per_second": 0.0 if elapsed <= 0 else episodes / elapsed,
                "elapsed_s": elapsed,
                "error": repr(error),
            },
        )
        if episodes > 0:
            _save_checkpoint(
                artifact,
                agent,
                episodes=episodes,
                successes=successes,
                endpoint_successes=endpoint_successes,
                legacy_scientific_successes=legacy_scientific_successes,
                elapsed_s=elapsed,
                update_generator=update_generator,
            )
        write_training_plots(artifact)
        raise

    elapsed = time.perf_counter() - started
    _save_checkpoint(
        artifact,
        agent,
        episodes=episodes,
        successes=successes,
        endpoint_successes=endpoint_successes,
        legacy_scientific_successes=legacy_scientific_successes,
        elapsed_s=elapsed,
        update_generator=update_generator,
    )
    write_training_plots(artifact)
    summary = {
        "status": "COMPLETE",
        "episodes": episodes,
        "requested_episodes": requested_episodes,
        "batch_aligned_overshoot": episodes - requested_episodes,
        "successes": successes,
        "endpoint_successes": endpoint_successes,
        "legacy_scientific_successes": legacy_scientific_successes,
        "success_rate": successes / episodes,
        "episodes_per_second": episodes / elapsed,
        "elapsed_s": elapsed,
    }
    _write_json(artifact / "status.json", summary)
    _write_json(artifact / "final_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-directory", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--train", action="store_true")
    group.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    if arguments.resume:
        if arguments.artifact_directory is None:
            parser.error("--resume requires --artifact-directory")
        config_path = arguments.artifact_directory.resolve() / "config.json"
    else:
        config_path = arguments.config.resolve()
    config = _load_config(config_path)
    if arguments.preflight:
        preflight(config)
        return
    artifact_parent = ROOT / "data" / "policy_training" / str(config["experiment_id"])
    artifact = (
        artifact_parent / _utc_stamp()
        if arguments.artifact_directory is None
        else arguments.artifact_directory.resolve()
    )
    train(config, config_path, artifact, resume=bool(arguments.resume))


if __name__ == "__main__":
    main()
