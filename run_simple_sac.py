"""Launch the single authorized 10-second, three-million-episode SAC baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from learning.normalization import FixedContextNormalizer
from learning.sequential_sac_env import (
    SEQUENTIAL_SAC_OBSERVATION_DIM,
    SequentialWhipEnvironment,
    SimpleRewardWeights,
)
from learning.simple_sac import (
    SIMPLE_SAC_ACTION_DIM,
    SimpleSACAgent,
    TransitionReplayBuffer,
)
from planning.rollout import hover_preroll
from planning.task import load_canonical_whip_task
from simulator.parameters import SimulatorSettings
from simulator.production import build_production_simulator


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "simple_sequential_sac_10s_v1.json"
ARTIFACT_PARENT = ROOT / "data" / "policy_training" / "simple_sequential_sac_10s_v1"


def _utc_stamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H%M%S.") + f"{now.microsecond:06d}Z"


def _safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=_safe) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "simple_sequential_sac_10s_v1":
        raise ValueError("Unsupported simple-SAC configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Simple SAC must use the pinned production model freeze.")
    if int(config["total_episodes"]) != 3_000_000:
        raise ValueError("The authorized run must contain exactly 3,000,000 episodes.")
    if float(config["episode_duration_s"]) != 10.0:
        raise ValueError("The authorized episode horizon is exactly 10 seconds.")
    return config


def _build_environment(
    config: dict[str, Any],
    *,
    batch_size: int,
    record_fullstate_commands: bool = False,
    record_state_trajectory: bool = False,
) -> tuple[SequentialWhipEnvironment, torch.device]:
    settings = SimulatorSettings.load(ROOT / config["simulator_config"])
    task = load_canonical_whip_task(ROOT / config["task_config"])
    if abs(settings.dt_s - float(config["physics_dt_s"])) > 1.0e-12:
        raise ValueError("Configured SAC physics dt differs from production dt.")
    simulator = build_production_simulator(settings)
    if simulator.device.type != "cuda":
        raise RuntimeError("The three-million-episode baseline requires CUDA.")
    initial_state = hover_preroll(simulator, task)
    # The logical collection batch is itself fixed for all ordinary batches;
    # only the final partial batch is padded by the frozen UAV residual path.
    simulator.uav_model.set_fixed_evaluation_batch_size(int(config["collection_batch"]))
    normalizer = FixedContextNormalizer.load(ROOT / config["context_normalizer"])
    action_config = config["action"]
    reward_config = config["reward"]
    environment = SequentialWhipEnvironment(
        simulator,
        task,
        initial_state,
        normalizer,
        batch_size=batch_size,
        episode_duration_s=float(config["episode_duration_s"]),
        control_dt_s=float(config["control_dt_s"]),
        maximum_acceleration_m_s2=float(action_config["maximum_acceleration_norm_m_s2"]),
        maximum_body_rate_rad_s=float(action_config["maximum_body_rate_rad_s"]),
        observation_clip=float(config["observation"]["normalized_clip"]),
        reward_weights=SimpleRewardWeights(
            progress=float(reward_config["progress_weight"]),
            directed_speed_near_target=float(
                reward_config["directed_speed_near_target_weight"]
            ),
            direction_near_target=float(reward_config["direction_near_target_weight"]),
            uav_displacement=float(reward_config["uav_displacement_weight"]),
            success_bonus=float(reward_config["success_bonus"]),
            numerical_failure=float(reward_config["numerical_failure_penalty"]),
            proximity_scale_m=float(reward_config["proximity_scale_m"]),
            non_tip_first=float(reward_config.get("non_tip_first_penalty", 0.0)),
            strike_quality_improvement=float(
                reward_config.get("strike_quality_improvement_weight", 0.0)
            ),
            maximum_displacement=float(
                reward_config.get("maximum_displacement_weight", 0.0)
            ),
            terminal_displacement=float(
                reward_config.get("terminal_displacement_weight", 0.0)
            ),
            terminal_displacement_success_only=bool(
                reward_config.get("terminal_displacement_success_only", False)
            ),
            displacement_integral=float(
                reward_config.get("displacement_integral_weight", 0.0)
            ),
            uav_speed_integral=float(
                reward_config.get("uav_speed_integral_weight", 0.0)
            ),
            acceleration_effort=float(
                reward_config.get("acceleration_effort_weight", 0.0)
            ),
            body_rate_effort=float(
                reward_config.get("body_rate_effort_weight", 0.0)
            ),
            action_smoothness=float(
                reward_config.get("action_smoothness_weight", 0.0)
            ),
            time_to_success=float(
                reward_config.get("time_to_success_weight_per_s", 0.0)
            ),
            directed_speed_reward_cap_m_s=float(
                reward_config.get("directed_speed_reward_cap_m_s", float("inf"))
            ),
            success_compactness_bonus=float(
                reward_config.get("success_compactness_bonus", 0.0)
            ),
            success_compactness_scale_m=float(
                reward_config.get("success_compactness_scale_m", 0.5)
            ),
            displacement_cost_scale_m=float(
                reward_config.get("displacement_cost_scale_m", 0.0)
            ),
        ),
        action_mode=str(action_config.get("mode", "full_6d")),
        directed_speed_shaping_reference=str(
            reward_config.get("directed_speed_shaping_reference", "world_tip")
        ),
        success_mode=str(config.get("success_mode", "simple_endpoint")),
        reward_mode=str(config.get("reward_mode", "legacy_dense")),
        terminate_on_success=not bool(
            config.get("reported_success", {}).get(
                "episode_continues_after_success", True
            )
        ),
        record_fullstate_commands=record_fullstate_commands,
        record_state_trajectory=record_state_trajectory,
    )
    return environment, simulator.device


def _build_agent(config: dict[str, Any], device: torch.device) -> SimpleSACAgent:
    sac = config["sac"]
    return SimpleSACAgent(
        SEQUENTIAL_SAC_OBSERVATION_DIM,
        SIMPLE_SAC_ACTION_DIM,
        device=device,
        hidden_dim=int(sac["hidden_dim"]),
        actor_lr=float(sac["actor_lr"]),
        critic_lr=float(sac["critic_lr"]),
        alpha_lr=float(sac["alpha_lr"]),
        gamma=float(sac["gamma"]),
        tau=float(sac["tau"]),
        initial_alpha=float(sac["initial_alpha"]),
        target_entropy=float(sac["target_entropy"]),
    )


def preflight(config: dict[str, Any]) -> None:
    """One complete 10-second production episode plus one SAC update."""

    torch.manual_seed(int(config["seed"]))
    environment, device = _build_environment(config, batch_size=4)
    observation = environment.reset()
    replay = TransitionReplayBuffer(512, SEQUENTIAL_SAC_OBSERVATION_DIM, SIMPLE_SAC_ACTION_DIM)
    result = None
    for _ in range(environment.control_step_count):
        action = torch.empty((4, SIMPLE_SAC_ACTION_DIM), device=device).uniform_(-1.0, 1.0)
        result = environment.step(action)
        replay.add(
            observation,
            action,
            result.reward,
            result.next_observation,
            result.done,
            include=result.include_transition,
        )
        observation = result.next_observation
    assert result is not None
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    agent = _build_agent(config, device)
    metrics = agent.update(replay.sample(min(32, len(replay)), device=device, generator=generator))
    payload = {
        "preflight": "PASS",
        "device": str(device),
        "observation_shape": list(observation.shape),
        "action_shape": list(action.shape),
        "physics_backend": environment.simulator.forward_backend_audit(),
        "simulated_episode_duration_s": config["episode_duration_s"],
        "replay_transitions": len(replay),
        "finite_reward": bool(torch.isfinite(result.reward).all()),
        "numerical_failures": int(environment.failed.sum()),
        "sac_update": asdict(metrics),
    }
    print(json.dumps(payload, indent=2), flush=True)


def _save_checkpoint(
    artifact: Path,
    agent: SimpleSACAgent,
    *,
    episodes: int,
    successes: int,
    elapsed_s: float,
    replay: TransitionReplayBuffer,
) -> None:
    payload = agent.checkpoint()
    payload.update(
        {
            "episodes": episodes,
            "successes": successes,
            "elapsed_s": elapsed_s,
            "replay_metadata": replay.metadata(),
            "note": "Replay tensor contents are intentionally not duplicated in checkpoints.",
        }
    )
    checkpoints = artifact / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    temporary = checkpoints / "latest.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, checkpoints / "latest.pt")


def train(config: dict[str, Any], config_path: Path, artifact: Path) -> None:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    artifact.mkdir(parents=True, exist_ok=False)
    (artifact / "checkpoints").mkdir()
    _write_json(artifact / "config.json", config)
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "learning" / "simple_sac.py",
        ROOT / "learning" / "sequential_sac_env.py",
        config_path,
        ROOT / config["simulator_config"],
        ROOT / config["task_config"],
        ROOT / config["context_normalizer"],
        ROOT / "config" / "active_model.json",
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
    )
    (artifact / "RUN_CONTRACT.md").write_text(
        "# Simple sequential SAC 10-second baseline\n\n"
        "This is one conventional sequential SAC run. The 6-D action is 3-D "
        "yaw-local acceleration plus body roll/pitch/yaw rates. The reward has "
        "only endpoint progress, near-target speed/direction, UAV displacement, "
        "and a first-success bonus. An episode lasts 10.0 seconds unless a row "
        "becomes numerically non-finite. `training_log.csv` reports episode "
        "throughput and endpoint success rate. No CEM, demonstrations, protected "
        "data, or hardware are used.\n",
        encoding="utf-8",
    )

    collection_batch = int(config["collection_batch"])
    environment, device = _build_environment(config, batch_size=collection_batch)
    agent = _build_agent(config, device)
    sac = config["sac"]
    replay = TransitionReplayBuffer(
        int(sac["replay_capacity_transitions"]),
        SEQUENTIAL_SAC_OBSERVATION_DIM,
        SIMPLE_SAC_ACTION_DIM,
    )
    replay_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    total_episodes = int(config["total_episodes"])
    initial_random = int(sac["initial_random_episodes"])
    minibatch = int(sac["replay_minibatch"])
    updates_per_collection = int(sac["updates_per_collection"])
    checkpoint_interval = int(config["logging"]["checkpoint_every_episodes"])
    next_checkpoint = checkpoint_interval
    episodes = 0
    successes = 0
    start = time.perf_counter()
    log_path = artifact / "training_log.csv"
    fields = [
        "episodes",
        "successes",
        "total_success_rate",
        "batch_success_rate",
        "episodes_per_second",
        "elapsed_s",
        "mean_episode_reward",
        "median_minimum_tip_distance_m",
        "mean_maximum_uav_displacement_m",
        "numerical_failure_rate",
        "replay_size",
        "gradient_updates",
        "alpha",
        "critic1_loss",
        "critic2_loss",
        "actor_loss",
        "mean_q",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        handle.flush()

        _write_json(
            artifact / "status.json",
            {
                "status": "RUNNING",
                "pid": os.getpid(),
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "episodes": 0,
                "target_episodes": total_episodes,
            },
        )
        while episodes < total_episodes:
            logical_batch = min(collection_batch, total_episodes - episodes)
            if logical_batch != environment.batch_size:
                environment, _ = _build_environment(config, batch_size=logical_batch)
            observation = environment.reset()
            for _ in range(environment.control_step_count):
                if episodes < initial_random:
                    action = torch.empty(
                        (logical_batch, SIMPLE_SAC_ACTION_DIM), device=device
                    ).uniform_(-1.0, 1.0)
                else:
                    action = agent.act(observation)
                result = environment.step(action)
                replay.add(
                    observation,
                    action,
                    result.reward,
                    result.next_observation,
                    result.done,
                    include=result.include_transition,
                )
                observation = result.next_observation

            batch_successes = int(environment.episode_success.sum().detach().cpu())
            batch_failures = int(environment.failed.sum().detach().cpu())
            successes += batch_successes
            episodes += logical_batch
            update_rows = []
            if episodes >= initial_random and len(replay) >= minibatch:
                for _ in range(updates_per_collection):
                    update_rows.append(
                        agent.update(
                            replay.sample(
                                minibatch, device=device, generator=replay_generator
                            ),
                            gradient_clip=float(sac["gradient_clip"]),
                        )
                    )
            elapsed = time.perf_counter() - start

            def update_mean(name: str) -> float:
                if not update_rows:
                    return float("nan")
                return float(np.mean([getattr(row, name) for row in update_rows]))

            row = {
                "episodes": episodes,
                "successes": successes,
                "total_success_rate": successes / episodes,
                "batch_success_rate": batch_successes / logical_batch,
                "episodes_per_second": episodes / elapsed,
                "elapsed_s": elapsed,
                "mean_episode_reward": float(environment.episode_reward.mean().cpu()),
                "median_minimum_tip_distance_m": float(
                    environment.episode_minimum_tip_distance.median().cpu()
                ),
                "mean_maximum_uav_displacement_m": float(
                    environment.episode_maximum_displacement.mean().cpu()
                ),
                "numerical_failure_rate": batch_failures / logical_batch,
                "replay_size": len(replay),
                "gradient_updates": len(update_rows),
                "alpha": float(agent.alpha.detach().cpu()),
                "critic1_loss": update_mean("critic1_loss"),
                "critic2_loss": update_mean("critic2_loss"),
                "actor_loss": update_mean("actor_loss"),
                "mean_q": update_mean("mean_q"),
            }
            writer.writerow({key: _safe(value) for key, value in row.items()})
            handle.flush()
            _write_json(
                artifact / "status.json",
                {
                    "status": "RUNNING",
                    "pid": os.getpid(),
                    "episodes": episodes,
                    "target_episodes": total_episodes,
                    "successes": successes,
                    "success_rate": successes / episodes,
                    "episodes_per_second": episodes / elapsed,
                    "elapsed_s": elapsed,
                    "last_batch_success_rate": batch_successes / logical_batch,
                    "numerical_failure_rate": batch_failures / logical_batch,
                },
            )
            print(
                f"episodes={episodes}/{total_episodes} "
                f"successes={successes} "
                f"batch_success_rate={100.0 * batch_successes / logical_batch:.3f}% "
                f"total_success_rate={100.0 * successes / episodes:.3f}% "
                f"episodes_per_second={episodes / elapsed:.2f}",
                flush=True,
            )
            if episodes >= next_checkpoint or episodes == total_episodes:
                _save_checkpoint(
                    artifact,
                    agent,
                    episodes=episodes,
                    successes=successes,
                    elapsed_s=elapsed,
                    replay=replay,
                )
                while next_checkpoint <= episodes:
                    next_checkpoint += checkpoint_interval

    elapsed = time.perf_counter() - start
    final_checkpoint = agent.checkpoint()
    final_checkpoint.update({"episodes": episodes, "successes": successes, "elapsed_s": elapsed})
    torch.save(final_checkpoint, artifact / "final.pt")
    _write_json(
        artifact / "status.json",
        {
            "status": "COMPLETE",
            "pid": os.getpid(),
            "episodes": episodes,
            "target_episodes": total_episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "episodes_per_second": episodes / elapsed,
            "elapsed_s": elapsed,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    if args.preflight_only:
        preflight(config)
        return
    artifact = (
        args.artifact.expanduser().resolve()
        if args.artifact is not None
        else ARTIFACT_PARENT / _utc_stamp()
    )
    train(config, config_path, artifact)


if __name__ == "__main__":
    main()
