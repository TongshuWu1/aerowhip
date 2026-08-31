"""Launch the separate one-million-episode minimal Figure-8 SAC experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from learning.figure8_sac_env import (
    FIGURE8_SAC_OBSERVATION_DIM,
    Figure8Reference,
    LegacyFigure8RewardWeights,
    SequentialFigure8Environment,
    build_analytic_figure8_normalizer,
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
DEFAULT_CONFIG = ROOT / "config" / "learning" / "simple_figure8_sac_10s_v1.json"
ARTIFACT_PARENT = ROOT / "data" / "policy_training" / "simple_figure8_sac_10s_v1"


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
    temporary.write_text(
        json.dumps(payload, indent=2, default=_safe) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") != "simple_figure8_sac_10s_v1":
        raise ValueError("Unsupported Figure-8 SAC configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Figure-8 SAC must use the pinned production model freeze.")
    if int(config["total_episodes"]) != 1_000_000:
        raise ValueError("The authorized run must contain exactly 1,000,000 episodes.")
    if float(config["episode_duration_s"]) != 10.0:
        raise ValueError("The Figure-8 episode horizon is exactly 10 seconds.")
    if config["reward"].get("profile") != "legacy_online_figure8_geometric_tracking_v1":
        raise ValueError("The legacy online Figure-8 reward profile must remain selected.")
    return config


def _build_environment(
    config: dict[str, Any], *, batch_size: int
) -> tuple[SequentialFigure8Environment, torch.device]:
    settings = SimulatorSettings.load(ROOT / config["simulator_config"])
    task = load_canonical_whip_task(ROOT / config["initial_state_task_config"])
    if abs(settings.dt_s - float(config["physics_dt_s"])) > 1.0e-12:
        raise ValueError("Configured SAC physics dt differs from production dt.")
    simulator = build_production_simulator(settings)
    if simulator.device.type != "cuda":
        raise RuntimeError("The one-million-episode Figure-8 baseline requires CUDA.")
    initial_state = hover_preroll(simulator, task)
    simulator.uav_model.set_fixed_evaluation_batch_size(int(config["collection_batch"]))
    action = config["action"]
    reward = config["reward"]
    figure8 = config["figure8"]
    success = config["reported_success"]
    center = tuple(
        float(value)
        for value in initial_state.cable.positions_m[0, -1].detach().cpu().tolist()
    )
    reference = Figure8Reference(
        center,
        float(figure8["amplitude_x_m"]),
        float(figure8["amplitude_y_m"]),
    )
    normalizer = build_analytic_figure8_normalizer(
        simulator,
        initial_state,
        reference,
        command_yaw_world_rad=task.initial_yaw_rad,
    )
    environment = SequentialFigure8Environment(
        simulator,
        task,
        initial_state,
        normalizer,
        batch_size=batch_size,
        episode_duration_s=float(config["episode_duration_s"]),
        control_dt_s=float(config["control_dt_s"]),
        amplitude_x_m=float(figure8["amplitude_x_m"]),
        amplitude_y_m=float(figure8["amplitude_y_m"]),
        maximum_acceleration_m_s2=float(action["maximum_acceleration_norm_m_s2"]),
        maximum_body_rate_rad_s=float(action["maximum_body_rate_rad_s"]),
        observation_clip=float(config["observation"]["normalized_clip"]),
        reward_weights=LegacyFigure8RewardWeights(
            tracking=float(reward["tracking_weight"]),
            path_progress=float(reward["path_progress_reward_weight"]),
            control_effort=float(reward["control_effort_weight"]),
            control_smoothness=float(reward["control_smoothness_weight"]),
            tip_motion_smoothness=float(reward["tip_motion_smoothness_weight"]),
            speed_limit=float(reward["speed_limit_weight"]),
            ground_safety=float(reward["ground_safety_weight"]),
            ground_clearance_m=float(reward["ground_clearance_m"]),
            numerical_failure=float(reward["numerical_failure_penalty"]),
        ),
        success_required_cycles=float(success["minimum_completed_cycles"]),
        success_maximum_tracking_rmse_m=float(success["maximum_tracking_rmse_m"]),
    )
    return environment, simulator.device


def _build_agent(config: dict[str, Any], device: torch.device) -> SimpleSACAgent:
    sac = config["sac"]
    return SimpleSACAgent(
        FIGURE8_SAC_OBSERVATION_DIM,
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
    """One complete Figure-8 episode plus one standard SAC update."""

    torch.manual_seed(int(config["seed"]))
    environment, device = _build_environment(config, batch_size=4)
    observation = environment.reset()
    replay = TransitionReplayBuffer(
        512, FIGURE8_SAC_OBSERVATION_DIM, SIMPLE_SAC_ACTION_DIM
    )
    result = None
    for _ in range(environment.control_step_count):
        action = torch.empty((4, SIMPLE_SAC_ACTION_DIM), device=device).uniform_(-1.0, 1.0)
        result = environment.step(action)
        replay.add(
            observation,
            action,
            result.reward / float(config["sac"]["reward_scale_divisor"]),
            result.next_observation,
            result.done,
            include=result.include_transition,
        )
        observation = result.next_observation
    assert result is not None
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    agent = _build_agent(config, device)
    metrics = agent.update(
        replay.sample(min(32, len(replay)), device=device, generator=generator)
    )
    print(
        json.dumps(
            {
                "preflight": "PASS",
                "device": str(device),
                "observation_shape": list(observation.shape),
                "action_shape": list(action.shape),
                "physics_backend": environment.simulator.forward_backend_audit(),
                "reference_center_m": environment.reference.center_position_m,
                "reference_length_m": environment.path_length_m,
                "reward_profile": config["reward"]["profile"],
                "finite_reward": bool(torch.isfinite(result.reward).all()),
                "maximum_absolute_normalized_observation": float(
                    observation[:, :-1].abs().max().cpu()
                ),
                "tracking_rmse_m": float(environment.episode_tracking_rmse_m.mean().cpu()),
                "mean_progress_cycles": float(environment.path_progress_cycles.mean().cpu()),
                "numerical_failures": int(environment.failed.sum().cpu()),
                "sac_update": asdict(metrics),
            },
            indent=2,
        ),
        flush=True,
    )


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
            "experiment": "simple_figure8_sac_10s_v1",
            "episodes": episodes,
            "successes": successes,
            "elapsed_s": elapsed_s,
            "replay_metadata": replay.metadata(),
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
        ROOT / "learning" / "figure8_sac_env.py",
        config_path,
        ROOT / config["simulator_config"],
        ROOT / config["initial_state_task_config"],
        ROOT / "config" / "active_model.json",
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
    )
    (artifact / "RUN_CONTRACT.md").write_text(
        "# Simple sequential Figure-8 SAC\n\n"
        "This experiment is separate from the whip SAC baseline. It uses the "
        "same production physics and direct six-dimensional acceleration/body-rate "
        "action, but tracks a 0.7 x 0.5 m horizontal Figure-8 centered on the "
        "settled cable tip. The reward is the negative legacy online geometric "
        "tracking objective with its original coefficients. SAC stores that "
        "reward divided by 10,000 solely for critic numerical conditioning; "
        "this positive global scale does not change reward ordering. Each episode lasts "
        "10 seconds. Reported success means at least one forward cycle with <=100 "
        "mm tracking RMSE, UAV speed <=3 m/s, ground clearance >=20 mm, and finite "
        "physics. The reporting gate adds no reward bonus. No CEM, demonstrations, "
        "protected data, hardware, or whip artifact is used or modified.\n",
        encoding="utf-8",
    )

    collection_batch = int(config["collection_batch"])
    environment, device = _build_environment(config, batch_size=collection_batch)
    _write_json(artifact / "context_normalizer.json", environment.normalizer.to_json())
    agent = _build_agent(config, device)
    sac = config["sac"]
    replay = TransitionReplayBuffer(
        int(sac["replay_capacity_transitions"]),
        FIGURE8_SAC_OBSERVATION_DIM,
        SIMPLE_SAC_ACTION_DIM,
    )
    replay_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    total_episodes = int(config["total_episodes"])
    initial_random = int(sac["initial_random_episodes"])
    minibatch = int(sac["replay_minibatch"])
    updates_per_collection = int(sac["updates_per_collection"])
    reward_scale_divisor = float(sac["reward_scale_divisor"])
    if not np.isfinite(reward_scale_divisor) or reward_scale_divisor <= 0.0:
        raise ValueError("SAC reward scale divisor must be finite and positive.")
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
        "median_tracking_rmse_m",
        "mean_path_progress_cycles",
        "mean_maximum_uav_displacement_m",
        "mean_maximum_uav_speed_m_s",
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
                "task": "FIGURE8_TRACKING",
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
                    result.reward / reward_scale_divisor,
                    result.next_observation,
                    result.done,
                    include=result.include_transition,
                )
                observation = result.next_observation

            batch_successes = int(environment.episode_success.sum().cpu())
            batch_failures = int(environment.failed.sum().cpu())
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
                "median_tracking_rmse_m": float(
                    environment.episode_tracking_rmse_m.median().cpu()
                ),
                "mean_path_progress_cycles": float(
                    environment.path_progress_cycles.mean().cpu()
                ),
                "mean_maximum_uav_displacement_m": float(
                    environment.episode_maximum_displacement.mean().cpu()
                ),
                "mean_maximum_uav_speed_m_s": float(
                    environment.episode_maximum_speed.mean().cpu()
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
                    "task": "FIGURE8_TRACKING",
                    "episodes": episodes,
                    "target_episodes": total_episodes,
                    "successes": successes,
                    "success_rate": successes / episodes,
                    "episodes_per_second": episodes / elapsed,
                    "elapsed_s": elapsed,
                    "last_batch_success_rate": batch_successes / logical_batch,
                    "median_tracking_rmse_m": row["median_tracking_rmse_m"],
                    "mean_path_progress_cycles": row["mean_path_progress_cycles"],
                    "numerical_failure_rate": batch_failures / logical_batch,
                },
            )
            print(
                f"episodes={episodes}/{total_episodes} successes={successes} "
                f"batch_success_rate={100.0 * batch_successes / logical_batch:.3f}% "
                f"total_success_rate={100.0 * successes / episodes:.3f}% "
                f"rmse={1000.0 * row['median_tracking_rmse_m']:.1f}mm "
                f"cycles={row['mean_path_progress_cycles']:.3f} "
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
    final_checkpoint.update(
        {
            "experiment": "simple_figure8_sac_10s_v1",
            "episodes": episodes,
            "successes": successes,
            "elapsed_s": elapsed,
        }
    )
    torch.save(final_checkpoint, artifact / "final.pt")
    _write_json(
        artifact / "status.json",
        {
            "status": "COMPLETE",
            "pid": os.getpid(),
            "task": "FIGURE8_TRACKING",
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
