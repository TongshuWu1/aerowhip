"""Preflight, train, and evaluate PPO for the 3D point-force cable model."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import torch

from learning import (
    POINT_FORCE_OBSERVATION_DIM,
    PointForceWhipEnvironment,
    SimplePPOAgent,
)
from learning.checkpoint_library import write_active_run
from learning.simple_ppo import PPORollout
from learning.training_control import stoppable_training, TrainingStopped
from learning.reward_plateau import RewardPlateau
from simulator.artifact_io import replace_with_retry
from simulator.rollout import load_json, resolve_device


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "config" / "model.json"
TASK_PATH = ROOT / "config" / "task.json"
PPO_PATH = ROOT / "config" / "ppo.json"


def configure_accelerator(device: torch.device) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")


def _utc_stamp() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H%M%S.") + f"{now.microsecond:06d}Z"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.name == 'validation_latest.json':
        from learning.experiment_records import save_evaluation
        save_evaluation(path.parent, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    replace_with_retry(temporary, path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    replace_with_retry(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_configs(config_directory=None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = Path(config_directory) if config_directory else ROOT / 'config'
    return tuple(load_json(directory / name) for name in ('model.json', 'task.json', 'ppo.json'))


def validate_contract(
    model: dict[str, Any], task: dict[str, Any], config: dict[str, Any]
) -> None:
    from simulator.research_config import validate_research_contract
    validate_research_contract(model,task,config)
    if model["schema"] != "point_force_dder_model_v1":
        raise ValueError("PPO requires the point-force DDER model.")
    if task["schema"] != "force_whip_task_v1":
        raise ValueError("PPO requires the force-whip task.")
    if config["schema"] != "point_force_ppo_training_v1":
        raise ValueError("Unsupported PPO configuration schema.")
    if int(config["action"]["dimensions"]) != 3:
        raise ValueError("The point-force policy must have exactly three actions.")
    if int(config["observation"]["dimensions"]) != POINT_FORCE_OBSERVATION_DIM:
        raise ValueError("Configured PPO observation dimension is stale.")
    deployment = config.get("deployment", {})
    if deployment.get("enabled", False):
        if config["ppo"]["gamma"] != 1.0 or config["ppo"]["gae_lambda"] != 1.0:
            raise ValueError("Open-loop plan credit requires gamma=gae_lambda=1.")
        if not 0 <= deployment["nominal_fraction"] <= 1:
            raise ValueError("Nominal deployment fraction must lie in [0, 1].")
        for key in ("initial_position_m", "initial_velocity_m_s", "initial_cable_tilt_deg",
                    "initial_angular_velocity_rad_s", "state_position_error_m",
                    "state_velocity_error_m_s", "state_cable_tilt_error_deg",
                    "force_lag_max_s", "recovery_failure_penalty"):
            if not math.isfinite(deployment[key]) or deployment[key] < 0:
                raise ValueError(f"Invalid deployment tolerance: {key}")
        for key in ("stiffness_fraction", "damping_fraction", "force_gain_fraction"):
            if not 0 <= deployment[key] < 1:
                raise ValueError(f"Invalid deployment tolerance: {key}")
        if deployment["recovery_duration_s"] < .5:
            raise ValueError("Deployment evaluation must include PID settling time.")
        for key in ('initial_cable_bend_deg','state_cable_bend_error_deg','launch_position_drift_m','launch_velocity_drift_m_s',
                    'initial_position_radius_m','target_position_radius_m'):
            if key in deployment and (not math.isfinite(deployment[key]) or deployment[key] < 0):
                raise ValueError(f'Invalid deployment tolerance: {key}')
    elif any(float(deployment.get(key, 0.)) > 0 for key in ('initial_position_radius_m', 'target_position_radius_m')):
        raise ValueError('Position radius randomization requires open-loop deployment training.')
    stochastic_indices = [
        int(index) for index in config["ppo"].get("stochastic_action_indices", [0, 1, 2])
    ]
    if not stochastic_indices or len(stochastic_indices) != len(set(stochastic_indices)):
        raise ValueError("Stochastic PPO action indices must be unique and non-empty.")
    if any(index < 0 or index >= 3 for index in stochastic_indices):
        raise ValueError("A stochastic PPO action index is out of range.")
    if 1 not in stochastic_indices:
        if any(float(deployment.get(key, 0.)) > 0 for key in ('initial_position_radius_m', 'target_position_radius_m')):
            raise ValueError('3D position randomization requires stochastic Fy exploration.')
        lateral_target_offset = float(task["target_position_m"][1]) - float(
            task["initial_root_position_m"][1]
        )
        lateral_strike = float(task["desired_strike_direction_world"][1])
        if abs(lateral_target_offset) > 1.0e-9 or abs(lateral_strike) > 1.0e-9:
            raise ValueError("Deterministic Fy exploration requires a symmetric task.")
    if float(config["reward"]["angle_shaping_weight"]) != 0.0:
        raise ValueError("Impact angle must not receive shaping reward.")
    if not bool(config["reward"]["angle_is_binary_success_gate_only"]):
        raise ValueError("Impact angle must remain a binary success gate.")
    if float(task["reward"]["angle_shaping_weight"]) != 0.0:
        raise ValueError("Task configuration unexpectedly rewards impact angle.")
    guard = config.get("update_guard", {})
    if bool(guard.get("enabled", False)):
        if not bool(config["validation"]["enabled"]):
            raise ValueError("The PPO update guard requires validation.")
        if not 0.0 <= float(guard["minimum_final_success_rate"]) <= 1.0:
            raise ValueError("Guard success rate must lie in [0, 1].")
        if float(guard["maximum_reward_regression"]) < 0.0:
            raise ValueError("Guard reward regression must be non-negative.")
        if float(guard["maximum_point_cost_regression_s"]) < 0.0:
            raise ValueError("Guard point-cost regression must be non-negative.")
        decay = float(guard["learning_rate_decay_factor"])
        if not 0.0 < decay <= 1.0:
            raise ValueError("Guard learning-rate decay must lie in (0, 1].")
        if int(guard["cost_rejections_before_decay"]) < 1:
            raise ValueError("Guard cost-rejection patience must be positive.")
    bootstrap = config.get("bootstrap")
    if bootstrap and bool(bootstrap.get("enabled", False)):
        expected_steps = round(
            float(task["episode_duration_s"]) / float(task["control_dt_s"])
        )
        if int(bootstrap["total_control_steps"]) != expected_steps:
            raise ValueError("The action prior does not match the task horizon.")
    curriculum = config.get("curriculum")
    if curriculum and bool(curriculum.get("enabled", False)):
        stages = list(curriculum.get("stages", []))
        if not stages:
            raise ValueError("The enabled success curriculum has no stages.")
        previous_fraction = 0.0
        for stage in stages:
            fraction = float(stage["until_fraction"])
            if not previous_fraction < fraction <= 1.0:
                raise ValueError("Curriculum stage fractions must increase to 1.")
            previous_fraction = fraction
        final = stages[-1]
        final_success = task["success"]
        expected = (
            float(final_success["tip_target_distance_m"]),
            float(final_success["minimum_directed_tip_speed_m_s"]),
            float(final_success["maximum_tip_velocity_to_desired_direction_error_deg"]),
        )
        actual = (
            float(final["target_radius_m"]),
            float(final["minimum_directed_speed_m_s"]),
            float(final["maximum_tip_velocity_direction_error_deg"]),
        )
        if previous_fraction != 1.0 or any(
            not math.isclose(left, right) for left, right in zip(actual, expected)
        ):
            raise ValueError("The last curriculum stage must equal the final task.")
    if not math.isclose(
        float(task["control_dt_s"]) / float(model["simulation"]["dt_s"]),
        round(float(task["control_dt_s"]) / float(model["simulation"]["dt_s"])),
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise ValueError("Control and physics time steps do not align.")


def build_agent(config: dict[str, Any], device: torch.device) -> SimplePPOAgent:
    source = config["ppo"]
    bootstrap = config.get("bootstrap")
    action_prior = (
        bootstrap
        if bootstrap is not None and bool(bootstrap.get("enabled", False))
        else None
    )
    return SimplePPOAgent(
        POINT_FORCE_OBSERVATION_DIM,
        3,
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
        action_prior=action_prior,
        stochastic_action_indices=source.get("stochastic_action_indices"),
        initial_log_std=source.get("initial_log_std", -0.5),
        minimum_log_std=float(source.get("minimum_log_std", -10.0)),
    )


def training_success_condition(
    config: dict[str, Any],
    task: dict[str, Any],
    *,
    episodes: int,
    episodes_target: int,
) -> dict[str, Any]:
    final = task["success"]
    curriculum = config.get("curriculum")
    if curriculum is None or not bool(curriculum.get("enabled", False)):
        return {
            "name": "final",
            "index": 0,
            "target_radius_m": float(final["tip_target_distance_m"]),
            "minimum_directed_speed_m_s": float(
                final["minimum_directed_tip_speed_m_s"]
            ),
            "maximum_tip_velocity_direction_error_deg": float(
                final["maximum_tip_velocity_to_desired_direction_error_deg"]
            ),
        }
    fraction = episodes / max(episodes_target, 1)
    stages = list(curriculum["stages"])
    for index, stage in enumerate(stages):
        if fraction < float(stage["until_fraction"]) or index == len(stages) - 1:
            return {
                "name": str(stage["name"]),
                "index": index + 1,
                "target_radius_m": float(stage["target_radius_m"]),
                "minimum_directed_speed_m_s": float(
                    stage["minimum_directed_speed_m_s"]
                ),
                "maximum_tip_velocity_direction_error_deg": float(
                    stage["maximum_tip_velocity_direction_error_deg"]
                ),
            }
    raise RuntimeError("Curriculum selection failed.")


def guarded_update_is_acceptable(
    candidate: dict[str, Any],
    accepted: dict[str, Any],
    guard: dict[str, Any],
) -> bool:
    """Preserve the hit contract, then optimize the user's displacement cost."""

    deployment_ok = all(
        float(candidate.get(key, 1.0)) + float(guard.get("maximum_deployment_success_regression", 0.0))
        >= float(accepted.get(key, 1.0))
        for key in ("plan_success_rate", "success_rate", "hit_and_recovery_rate",
                    "nominal_success_rate", "nominal_hit_and_recovery_rate")
        if candidate.get(key) is not None and accepted.get(key) is not None
    ) if "hit_and_recovery_rate" in candidate else True
    return deployment_ok and (
        float(candidate["success_rate"])
        >= float(guard["minimum_final_success_rate"])
        and float(candidate["mean_point_displacement_cost_integral_s"])
        <= float(accepted["mean_point_displacement_cost_integral_s"])
        + float(guard["maximum_point_cost_regression_s"])
        and float(candidate["mean_episode_reward"])
        + float(guard["maximum_reward_regression"])
        >= float(accepted["mean_episode_reward"])
    )


def validation_rank(result: dict[str, Any]) -> tuple[float, float, float]:
    """Rank baseline policies by success, scalar reward, then compactness."""

    whip_only = result.get("evaluation_mode") == "30hz_frozen_reference_tracked_pose_and_cable_whip_only"
    success = result["success_rate"] if whip_only else result.get("hit_and_recovery_rate", result["success_rate"])
    return (
        float(success),
        float(result["mean_episode_reward"]),
        -float(result["mean_point_displacement_cost_integral_s"]),
    )


def guarded_update_is_unsafe(
    candidate: dict[str, Any],
    accepted: dict[str, Any],
    guard: dict[str, Any],
) -> bool:
    """Identify rejections that justify reducing the step size."""

    deployment_regression = any(
        float(candidate.get(key, 1.0)) + float(guard.get("maximum_deployment_success_regression", 0.0))
        < float(accepted.get(key, 1.0))
        for key in ("plan_success_rate", "success_rate", "hit_and_recovery_rate",
                    "nominal_success_rate", "nominal_hit_and_recovery_rate")
        if candidate.get(key) is not None and accepted.get(key) is not None
    ) if "hit_and_recovery_rate" in candidate else False
    return deployment_regression or (
        float(candidate["success_rate"])
        < float(guard["minimum_final_success_rate"])
        or float(candidate["mean_episode_reward"])
        + float(guard["maximum_reward_regression"])
        < float(accepted["mean_episode_reward"])
    )


def collect_rollout(
    environment: PointForceWhipEnvironment,
    agent: SimplePPOAgent,
    rollout: PPORollout,
    *, progress=None,
) -> None:
    if hasattr(environment,'collect_training_rollout'):
        environment.collect_training_rollout(agent,rollout,progress=progress)
        return
    if environment.ppo_config.get("deployment", {}).get("enabled", False):
        from learning.deployment_rollout import collect_deployment_rollout
        collect_deployment_rollout(environment, agent, rollout, progress=progress)
        return
    observation = environment.reset()
    for step_index in range(environment.control_step_count):
        action, log_probability, value = agent.act(observation)
        result = environment.step(action)
        rollout.observations[step_index].copy_(observation)
        rollout.actions[step_index].copy_(action)
        rollout.rewards[step_index].copy_(result.reward)
        rollout.dones[step_index].copy_(result.done)
        rollout.masks[step_index].copy_(
            result.include_transition[:, None].to(rollout.masks.dtype)
        )
        rollout.log_probabilities[step_index].copy_(log_probability)
        rollout.values[step_index].copy_(value)
        observation = result.next_observation
        if progress is not None:
            progress('Simulating training batch', step_index + 1, environment.control_step_count)


@torch.no_grad()
def evaluate(
    model: dict[str, Any],
    task: dict[str, Any],
    config: dict[str, Any],
    agent: SimplePPOAgent,
    *,
    episodes: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    if config.get("deployment", {}).get("enabled", False):
        from learning.deployment_rollout import evaluate_deployment
        return evaluate_deployment(model, task, config, agent, episodes=episodes,
                                   batch_size=batch_size, device=device)
    successes = 0
    non_tip = 0
    invalid_tip = 0
    timeouts = 0
    failures = 0
    rewards: list[torch.Tensor] = []
    minimum_distances: list[torch.Tensor] = []
    maximum_displacements: list[torch.Tensor] = []
    displacement_integrals: list[torch.Tensor] = []
    displacement_cost_integrals: list[torch.Tensor] = []
    hit_times: list[torch.Tensor] = []
    completed = 0
    while completed < episodes:
        current_batch = min(batch_size, episodes - completed)
        environment = PointForceWhipEnvironment(
            model, task, config, batch_size=current_batch, device=device
        )
        observation = environment.reset()
        for _ in range(environment.control_step_count):
            action = agent.deterministic_action(observation)
            observation = environment.step(action).next_observation
        successes += int(environment.episode_success.sum().detach().cpu())
        non_tip += int(environment.episode_non_tip_first.sum().detach().cpu())
        invalid_tip += int(
            environment.episode_invalid_tip_entry.sum().detach().cpu()
        )
        timeouts += int(environment.episode_timed_out.sum().detach().cpu())
        failures += int(environment.failed.sum().detach().cpu())
        rewards.append(environment.episode_reward.detach().cpu())
        minimum_distances.append(
            environment.episode_minimum_tip_distance.detach().cpu()
        )
        maximum_displacements.append(
            environment.episode_maximum_point_displacement.detach().cpu()
        )
        displacement_integrals.append(
            environment.episode_point_displacement_integral_m_s.detach().cpu()
        )
        displacement_cost_integrals.append(
            environment.episode_point_displacement_cost_integral_s.detach().cpu()
        )
        finite_hits = environment.episode_hit_time_s[
            torch.isfinite(environment.episode_hit_time_s)
        ]
        if finite_hits.numel():
            hit_times.append(finite_hits.detach().cpu())
        completed += current_batch
    reward = torch.cat(rewards)
    minimum_distance = torch.cat(minimum_distances)
    displacement = torch.cat(maximum_displacements)
    displacement_integral = torch.cat(displacement_integrals)
    displacement_cost_integral = torch.cat(displacement_cost_integrals)
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "non_tip_first_rate": non_tip / episodes,
        "invalid_tip_entry_rate": invalid_tip / episodes,
        "timeout_rate": timeouts / episodes,
        "numerical_failure_rate": failures / episodes,
        "mean_episode_reward": float(reward.mean()),
        "median_minimum_tip_distance_m": float(minimum_distance.median()),
        "mean_maximum_point_displacement_m": float(displacement.mean()),
        "mean_point_displacement_integral_m_s": float(displacement_integral.mean()),
        "mean_point_displacement_cost_integral_s": float(
            displacement_cost_integral.mean()
        ),
        "mean_hit_time_s": (
            float(torch.cat(hit_times).mean()) if hit_times else None
        ),
    }


def _checkpoint_payload(
    agent: SimplePPOAgent,
    *,
    episodes: int,
    successes: int,
    elapsed_s: float,
) -> dict[str, Any]:
    return {
        **agent.checkpoint(),
        "training_execution_mode": getattr(agent, "training_execution_mode", "feedback"),
        "episodes": episodes,
        "successes": successes,
        "elapsed_s": elapsed_s,
        "observation_dim": POINT_FORCE_OBSERVATION_DIM,
        "action_dim": 3,
    }


def _load_checkpoint(
    agent: SimplePPOAgent, path: Path, *, load_optimizer: bool
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=agent.device, weights_only=False)
    _restore_agent(agent, checkpoint, load_optimizer=load_optimizer)
    return checkpoint


def _restore_agent(
    agent: SimplePPOAgent,
    checkpoint: dict[str, Any],
    *,
    load_optimizer: bool,
) -> None:
    if load_optimizer and not {
        "policy_optimizer",
        "value_optimizer",
    }.issubset(checkpoint):
        raise ValueError(
            "This checkpoint predates guarded PPO with separate optimizers; "
            "start a fresh run instead of continuing it."
        )
    agent.policy.set_action_prior(checkpoint.get("action_prior"))
    agent.policy.load_state_dict(checkpoint["policy"])
    agent.value.load_state_dict(checkpoint["value"])
    if load_optimizer and "policy_optimizer" in checkpoint:
        agent.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
    if load_optimizer and "value_optimizer" in checkpoint:
        agent.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
    agent.gradient_updates = int(checkpoint.get("gradient_updates", 0))


def _set_learning_rate(agent: SimplePPOAgent, learning_rate: float) -> None:
    for optimizer in (agent.policy_optimizer, agent.value_optimizer):
        for group in optimizer.param_groups:
            group["lr"] = float(learning_rate)


def _learning_rate(agent: SimplePPOAgent) -> float:
    return float(agent.policy_optimizer.param_groups[0]["lr"])


def preflight(device_name: str) -> dict[str, Any]:
    model, task, config = load_configs()
    validate_contract(model, task, config)
    device = resolve_device(device_name)
    configure_accelerator(device)
    torch.manual_seed(int(config["seed"]))
    batch_size = 4
    environment = PointForceWhipEnvironment(
        model, task, config, batch_size=batch_size, device=device
    )
    agent = build_agent(config, device)
    rollout = PPORollout.allocate(
        environment.control_step_count,
        batch_size,
        POINT_FORCE_OBSERVATION_DIM,
        3,
        device=device,
    )
    started = time.perf_counter()
    collect_rollout(environment, agent, rollout)
    metrics = agent.update(
        rollout,
        minibatch_size=min(128, int(rollout.masks.sum().detach().cpu())),
        epochs=int(config["ppo"]["update_epochs"]),
        generator=torch.Generator(device=device).manual_seed(int(config["seed"]) + 1),
    )
    result = {
        "preflight": "PASS",
        "device": str(device),
        "batch_size": batch_size,
        "control_steps": environment.control_step_count,
        "physics_steps_per_episode": (
            environment.control_step_count * environment.physics_steps_per_control
        ),
        "observation_dim": POINT_FORCE_OBSERVATION_DIM,
        "action_dim": 3,
        "finite_rewards": bool(torch.isfinite(rollout.rewards).all().detach().cpu()),
        "successes": int(environment.episode_success.sum().detach().cpu()),
        "numerical_failures": int(environment.failed.sum().detach().cpu()),
        "elapsed_s": time.perf_counter() - started,
        "ppo_update": asdict(metrics),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


@stoppable_training(lambda: ROOT / 'runs/ppo' / _utc_stamp())
def train(
    *,
    device_name: str,
    requested_episodes: int | None,
    batch_size: int | None,
    artifact: Path | None,
    resume_checkpoint: Path | None,
    config_directory: Path | None = None,
    environment_factory=None,
) -> Path:
    model, task, config = load_configs(config_directory)
    environment_factory=environment_factory or PointForceWhipEnvironment
    validate_contract(model, task, config)
    device = resolve_device(device_name)
    configure_accelerator(device)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    configured_training = config["training"]
    episodes_target = int(
        configured_training["requested_episodes"]
        if requested_episodes is None
        else requested_episodes
    )
    collection_batch = int(
        configured_training["collection_batch"]
        if batch_size is None
        else batch_size
    )
    if episodes_target < 1 or collection_batch < 1:
        raise ValueError("Training episodes and collection batch must be positive.")
    artifact = (
        ROOT / "runs" / "ppo" / _utc_stamp() if artifact is None else artifact.resolve()
    )
    if (artifact / 'checkpoints/latest.pt').exists():
        raise ValueError('Use a new run directory; existing checkpoints are preserved.')
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "checkpoints").mkdir(exist_ok=True)
    run_metadata_path = artifact / "run.json"
    try:
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        run_metadata = {
            "schema": "point_force_training_run_v1",
            "display_name": artifact.name,
            "created_at_utc": _utc_stamp(),
            "resumed_from": (
                str(resume_checkpoint.resolve())
                if resume_checkpoint is not None
                else None
            ),
        }
    run_metadata["target_episodes"] = episodes_target
    run_metadata["collection_batch"] = collection_batch
    _atomic_json(run_metadata_path, run_metadata)
    write_active_run(ROOT, artifact)
    for name, value in zip(('model.json', 'task.json', 'ppo.json'), (model, task, config)):
        _atomic_json(artifact / name, value)
    _atomic_json(
        artifact / "source_hashes.json",
        {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in (
                ROOT / "learning" / "point_force_env.py",
                ROOT / "learning" / "simple_ppo.py",
                ROOT / "learning" / "training_control.py",
                ROOT / "learning" / "reward_plateau.py",
                ROOT / "learning" / "live_scene.py",
                ROOT / "simulator" / "artifact_io.py",
                ROOT / "experimental_data" / "io.py",
                ROOT / "learning" / "deployment_rollout.py",
                ROOT / "learning" / "fullstate_rollout.py",
                ROOT / "simulator" / "fullstate_execution.py",
                ROOT / "simulator" / "drone_tracking.py",
                ROOT / "simulator" / "research_pose.py",
                ROOT / "simulator" / "research_physics.py",
                ROOT / "simulator" / "research_reference.py",
                ROOT / "simulator" / "drone_pose_residual.py",
                ROOT / "learning" / "research_rollout.py",
                ROOT / "simulator" / "live_flight.py",
                ROOT / "run_ppo.py",
                ROOT / "simulator" / "point_mass.py",
                ROOT / "simulator" / "cuda_graph_physics.py",
                ROOT / "simulator" / "cable" / "dder.py",
            )
        },
    )
    _atomic_json(
        artifact / "status.json",
        {
            "status": "STARTING",
            "pid": os.getpid(),
            "device": str(device),
            "episodes": 0,
            "successes": 0,
            "target_episodes": episodes_target,
            "collection_batch": collection_batch,
            "update_guard_enabled": bool(
                config.get("update_guard", {}).get("enabled", False)
            ),
            "artifact": str(artifact),
        },
    )

    environment = environment_factory(
        model, task, config, batch_size=collection_batch, device=device
    )
    agent = build_agent(config, device)
    episodes = 0
    successes = 0
    elapsed_offset = 0.0
    if resume_checkpoint is not None:
        # A change from feedback rewards to whole-plan returns should not carry
        # the old Adam moments or override the configured continuation step size.
        checkpoint = _load_checkpoint(
            agent, resume_checkpoint.resolve(),
            load_optimizer=not bool(config.get("deployment", {}).get("reset_optimizer_on_resume", False)),
        )
        episodes = int(checkpoint.get("episodes", 0))
        successes = int(checkpoint.get("successes", 0))
        elapsed_offset = float(checkpoint.get("elapsed_s", 0.0))
        run_metadata["parent_training_episodes"] = episodes
        run_metadata["optimizer_state_restored"] = not bool(config.get("deployment", {}).get("reset_optimizer_on_resume", False))
        _atomic_json(run_metadata_path, run_metadata)

    agent.training_execution_mode = (
        "initial_state_only_open_loop_once_with_pid_recovery"
        if config.get("deployment", {}).get("enabled", False) else "feedback"
    )
    if config.get('deployment',{}).get('termination')=='execution_success_or_timeout':
        agent.training_execution_mode='initial_state_only_open_loop_predicted_hit_or_timeout'

    rollout = PPORollout.allocate(
        environment.control_step_count,
        collection_batch,
        POINT_FORCE_OBSERVATION_DIM,
        3,
        device=device,
    )
    update_generator = torch.Generator(device=device).manual_seed(seed + 1)
    training_log = artifact / "training_log.csv"
    fieldnames = (
        "episodes",
        "success_rate",
        "batch_success_rate",
        "mean_episode_reward",
        "median_minimum_tip_distance_m",
        "mean_maximum_point_displacement_m",
        "mean_point_displacement_cost_integral_s",
        "numerical_failure_rate",
        "nonfinite_rate", "position_limit_rate", "speed_limit_rate",
        "non_tip_first_rate",
        "invalid_tip_entry_rate",
        "timeout_rate",
        "curriculum_stage",
        "curriculum_target_radius_m",
        "curriculum_minimum_speed_m_s",
        "curriculum_maximum_angle_deg",
        "episodes_per_second",
        "elapsed_s",
        "policy_loss",
        "value_loss",
        "entropy",
        "approximate_kl",
        "clip_fraction",
        "gradient_norm",
        "gradient_updates",
        "valid_transitions",
        "update_accepted",
        "update_guard_enabled",
        "guard_candidate_success_rate",
        "guard_candidate_reward",
        "guard_candidate_point_cost_s",
        "accepted_validation_reward",
        "accepted_point_cost_s",
        "policy_learning_rate",
        "rejected_updates",
        "consecutive_cost_rejections",
    )
    stream = training_log.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    if training_log.stat().st_size == 0:
        writer.writeheader()
    best_rank = (-1.0, -math.inf, -math.inf)
    accepted_validation: dict[str, Any] | None = None
    rejected_updates = 0
    consecutive_cost_rejections = 0
    initial_checkpoint = _checkpoint_payload(
        agent,
        episodes=episodes,
        successes=successes,
        elapsed_s=elapsed_offset,
    )
    _atomic_checkpoint(artifact / "checkpoints" / "latest.pt", initial_checkpoint)
    validation = config["validation"]
    plateau = RewardPlateau.from_history(config.get('early_stopping'), start_episodes=episodes,
                                        history=config.get('reward_plateau_resume'))
    if plateau.enabled and not bool(validation['enabled']):
        raise ValueError('Reward plateau stopping requires validation.')
    best_reward = -math.inf

    def observe_reward(result, checkpoint):
        nonlocal best_reward
        if not plateau.enabled:
            return False
        reward = float(result['mean_episode_reward'])
        if math.isfinite(reward) and reward > best_reward:
            best_reward = reward
            _atomic_checkpoint(artifact / 'checkpoints/best_reward.pt', checkpoint)
            _atomic_json(artifact / 'best_reward.json', dict(result))
        record = dict(result)
        if hasattr(result, 'evaluation_id'):
            record['evaluation_id'] = result.evaluation_id
        decision = plateau.observe(record)
        _atomic_json(artifact / 'reward_convergence.json', decision)
        return decision['should_stop']

    if bool(validation["enabled"]):
        initial_validation = evaluate(
            model,
            task,
            config,
            agent,
            episodes=int(validation["episodes"]),
            batch_size=int(validation["episodes"]),
            device=device,
        )
        initial_validation["training_episodes"] = episodes
        accepted_validation = initial_validation
        _atomic_json(artifact / "validation_latest.json", initial_validation)
        best_rank = validation_rank(initial_validation)
        _atomic_checkpoint(
            artifact / "checkpoints" / "best_validation.pt", initial_checkpoint
        )
        _atomic_json(artifact / "best_validation.json", initial_validation)
        observe_reward(initial_validation, initial_checkpoint)
    started = time.perf_counter()
    _atomic_json(artifact / "status.json", {
        "status": "RUNNING", "stage": "Collecting training batch",
        "pid": os.getpid(), "device": str(device), "episodes": episodes,
        "successes": successes, "target_episodes": episodes_target,
        "collection_batch": collection_batch, "artifact": str(artifact),
    })
    last_row: dict[str, Any] = {}
    last_progress_time = 0.0
    last_progress_stage = None
    scene_batch_index = 0
    def report_progress(stage, step=0, total=0):
        nonlocal last_progress_time, last_progress_stage
        now = time.perf_counter()
        if stage == last_progress_stage and now - last_progress_time < 2 and step != total:
            return
        last_progress_time, last_progress_stage = now, stage
        _atomic_json(artifact / 'status.json', {
            **last_row, 'status': 'RUNNING', 'stage': stage,
            'pid': os.getpid(), 'device': str(device), 'episodes': episodes,
            'successes': successes, 'target_episodes': episodes_target,
            'collection_batch': collection_batch, 'artifact': str(artifact),
            'phase_step': step, 'phase_total': total,
            'progress_updated_at': time.time(),
        })
    try:
        while episodes < episodes_target:
            if (artifact / "STOP_REQUESTED").exists():
                break
            # A resumed run may have fewer attempts left than one GPU batch.
            # Collect only that remainder so the recorded budget stays exact.
            remaining = episodes_target - episodes
            if remaining < collection_batch:
                collection_batch = remaining
                environment = environment_factory(
                    model, task, config, batch_size=collection_batch, device=device
                )
                rollout = PPORollout.allocate(
                    environment.control_step_count, collection_batch,
                    POINT_FORCE_OBSERVATION_DIM, 3, device=device,
                )
            curriculum_condition = training_success_condition(
                config,
                task,
                episodes=episodes,
                episodes_target=episodes_target,
            )
            environment.set_success_condition(
                target_radius_m=curriculum_condition["target_radius_m"],
                minimum_directed_speed_m_s=(
                    curriculum_condition["minimum_directed_speed_m_s"]
                ),
                maximum_tip_velocity_direction_error_deg=(
                    curriculum_condition["maximum_tip_velocity_direction_error_deg"]
                ),
            )
            report_progress('Preparing training batch')
            if config.get('live_scene', {}).get('enabled', False):
                if model.get('fullstate_execution', {}).get('schema') != 'tracked_pose_execution_v1':
                    raise ValueError('Live Isaac scenes require the native 30 Hz tracked-pose model.')
                environment._live_scene_context = dict(artifact=str(artifact), episodes_before=episodes,
                    batch_index=scene_batch_index)
            collect_rollout(environment, agent, rollout, progress=report_progress)
            scene_batch_index += 1
            report_progress('Updating PPO')
            valid_transitions = int(rollout.masks.sum().detach().cpu())
            pre_update_checkpoint = copy.deepcopy(agent.checkpoint())
            metrics = agent.update(
                rollout,
                minibatch_size=min(
                    int(config["ppo"]["minibatch_transitions"]), valid_transitions
                ),
                epochs=int(config["ppo"]["update_epochs"]),
                generator=update_generator,
            )
            guard_config = config.get("update_guard", {})
            guard_enabled = bool(guard_config.get("enabled", False))
            update_accepted = True
            guard_candidate: dict[str, Any] | None = None
            if guard_enabled:
                if accepted_validation is None:
                    raise RuntimeError("The update guard has no accepted baseline.")
                report_progress('Validating proposed PPO update')
                guard_candidate = evaluate(
                    model,
                    task,
                    config,
                    agent,
                    episodes=int(guard_config["validation_episodes"]),
                    batch_size=int(guard_config["validation_episodes"]),
                    device=device,
                )
                guard_candidate["training_episodes"] = episodes + collection_batch
                update_accepted = guarded_update_is_acceptable(
                    guard_candidate,
                    accepted_validation,
                    guard_config,
                )
                _atomic_json(artifact / "guard_candidate_latest.json", guard_candidate)
                if update_accepted:
                    accepted_validation = guard_candidate
                    consecutive_cost_rejections = 0
                else:
                    _restore_agent(
                        agent, pre_update_checkpoint, load_optimizer=True
                    )
                    rejected_updates += 1
                    unsafe_rejection = guarded_update_is_unsafe(
                        guard_candidate,
                        accepted_validation,
                        guard_config,
                    )
                    if unsafe_rejection:
                        consecutive_cost_rejections = 0
                    else:
                        consecutive_cost_rejections += 1
                    decay_for_cost_plateau = (
                        consecutive_cost_rejections
                        >= int(guard_config["cost_rejections_before_decay"])
                    )
                    if unsafe_rejection or decay_for_cost_plateau:
                        next_learning_rate = max(
                            float(guard_config["minimum_learning_rate"]),
                            _learning_rate(agent)
                            * float(guard_config["learning_rate_decay_factor"]),
                        )
                        _set_learning_rate(agent, next_learning_rate)
                        consecutive_cost_rejections = 0
            batch_successes = int(
                environment.episode_success.sum().detach().cpu()
            )
            batch_failures = int(environment.failed.sum().detach().cpu())
            batch_non_tip = int(
                environment.episode_non_tip_first.sum().detach().cpu()
            )
            batch_invalid_tip = int(
                environment.episode_invalid_tip_entry.sum().detach().cpu()
            )
            batch_timeouts = int(
                environment.episode_timed_out.sum().detach().cpu()
            )
            episodes += collection_batch
            successes += batch_successes
            elapsed = elapsed_offset + time.perf_counter() - started
            row = {
                "episodes": episodes,
                "success_rate": successes / episodes,
                "batch_success_rate": batch_successes / collection_batch,
                "mean_episode_reward": float(
                    environment.episode_reward.mean().detach().cpu()
                ),
                "median_minimum_tip_distance_m": float(
                    environment.episode_minimum_tip_distance.median().detach().cpu()
                ),
                "mean_maximum_point_displacement_m": float(
                    environment.episode_maximum_point_displacement.mean().detach().cpu()
                ),
                "mean_point_displacement_cost_integral_s": float(
                    environment.episode_point_displacement_cost_integral_s.mean()
                    .detach()
                    .cpu()
                ),
                "numerical_failure_rate": batch_failures / collection_batch,
                "nonfinite_rate": float(environment.episode_nonfinite.float().mean()),
                "position_limit_rate": float(environment.episode_position_limit.float().mean()),
                "speed_limit_rate": float(environment.episode_speed_limit.float().mean()),
                "non_tip_first_rate": batch_non_tip / collection_batch,
                "invalid_tip_entry_rate": batch_invalid_tip / collection_batch,
                "timeout_rate": batch_timeouts / collection_batch,
                "curriculum_stage": curriculum_condition["name"],
                "curriculum_target_radius_m": curriculum_condition[
                    "target_radius_m"
                ],
                "curriculum_minimum_speed_m_s": curriculum_condition[
                    "minimum_directed_speed_m_s"
                ],
                "curriculum_maximum_angle_deg": curriculum_condition[
                    "maximum_tip_velocity_direction_error_deg"
                ],
                "episodes_per_second": episodes / max(elapsed, 1.0e-9),
                "elapsed_s": elapsed,
                **asdict(metrics),
                "gradient_updates": agent.gradient_updates,
                "update_accepted": update_accepted,
                "update_guard_enabled": guard_enabled,
                "guard_candidate_success_rate": (
                    ""
                    if guard_candidate is None
                    else float(guard_candidate["success_rate"])
                ),
                "guard_candidate_reward": (
                    ""
                    if guard_candidate is None
                    else float(guard_candidate["mean_episode_reward"])
                ),
                "guard_candidate_point_cost_s": (
                    ""
                    if guard_candidate is None
                    else float(
                        guard_candidate["mean_point_displacement_cost_integral_s"]
                    )
                ),
                "accepted_validation_reward": (
                    ""
                    if accepted_validation is None
                    else float(accepted_validation["mean_episode_reward"])
                ),
                "accepted_point_cost_s": (
                    ""
                    if accepted_validation is None
                    else float(
                        accepted_validation[
                            "mean_point_displacement_cost_integral_s"
                        ]
                    )
                ),
                "policy_learning_rate": _learning_rate(agent),
                "rejected_updates": rejected_updates,
                "consecutive_cost_rejections": consecutive_cost_rejections,
            }
            if hasattr(environment,'deployment'):
                row['reward_components']={key:float(value.mean()) for key,value in environment.episode_component_sums.items()}
                row['median_planning_minimum_tip_distance_m']=float(environment.deployment['planning_minimum_tip_distance_m'].median())
                row['mean_planning_maximum_drone_displacement_m']=float(environment.deployment['planning_maximum_drone_displacement_m'].mean())
            last_row = row
            writer.writerow({name: row[name] for name in fieldnames})
            stream.flush()
            checkpoint = _checkpoint_payload(
                agent,
                episodes=episodes,
                successes=successes,
                elapsed_s=elapsed,
            )
            _atomic_checkpoint(artifact / "checkpoints" / "latest.pt", checkpoint)
            if config.get('logging',{}).get('per_attempt',False):
                from learning.attempt_records import save_attempts
                save_attempts(artifact,environment,episodes,elapsed)
            if guard_enabled and accepted_validation is not None:
                current_validation = copy.deepcopy(accepted_validation)
                current_validation["training_episodes"] = episodes
                _atomic_json(
                    artifact / "validation_latest.json", current_validation
                )
                rank = validation_rank(current_validation)
                if rank > best_rank:
                    best_rank = rank
                    _atomic_checkpoint(
                        artifact / "checkpoints" / "best_validation.pt", checkpoint
                    )
                    _atomic_json(
                        artifact / "best_validation.json", current_validation
                    )
            status = {
                "status": "RUNNING",
                "pid": os.getpid(),
                "device": str(device),
                "target_episodes": episodes_target,
                "collection_batch": collection_batch,
                "update_guard_enabled": bool(
                    config.get("update_guard", {}).get("enabled", False)
                ),
                "artifact": str(artifact),
                **row,
            }
            _atomic_json(artifact / "status.json", status)
            print(json.dumps(status), flush=True)

            cadence = int(validation["every_episodes"])
            previous_episodes = episodes - collection_batch
            if bool(validation["enabled"]) and episodes // cadence > previous_episodes // cadence:
                report_progress('Validating updated policy')
                validation_result = (
                    copy.deepcopy(accepted_validation)
                    if guard_enabled and accepted_validation is not None
                    else evaluate(
                        model,
                        task,
                        config,
                        agent,
                        episodes=int(validation["episodes"]),
                        batch_size=int(validation["episodes"]),
                        device=device,
                    )
                )
                validation_result["training_episodes"] = episodes
                _atomic_json(artifact / "validation_latest.json", validation_result)
                rank = validation_rank(validation_result)
                if rank > best_rank:
                    best_rank = rank
                    _atomic_checkpoint(
                        artifact / "checkpoints" / "best_validation.pt", checkpoint
                    )
                    _atomic_json(
                        artifact / "best_validation.json", validation_result
                    )
                if observe_reward(validation_result, checkpoint):
                    break
    except TrainingStopped:
        raise
    except BaseException as error:
        _atomic_json(
            artifact / "status.json",
            {
                "status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "pid": os.getpid(),
                "episodes": episodes,
                "successes": successes,
                "target_episodes": episodes_target,
                "collection_batch": collection_batch,
                "artifact": str(artifact),
            },
        )
        raise
    finally:
        stream.close()

    elapsed = elapsed_offset + time.perf_counter() - started
    final_checkpoint = _checkpoint_payload(
        agent,
        episodes=episodes,
        successes=successes,
        elapsed_s=elapsed,
    )
    _atomic_checkpoint(artifact / "checkpoints" / "terminal.pt", final_checkpoint)
    if config.get("deployment", {}).get("enabled", False) and bool(validation["enabled"]):
        # Pilot runs can reserve final-test scenarios until the protocol is locked.
        run_holdout = bool(config['deployment'].get('evaluate_final_holdout', True))
    else:
        run_holdout = False
    if run_holdout:
        # A separate, unused seed measures the selected checkpoint after training.
        selected = build_agent(config, device)
        selected_checkpoint = _load_checkpoint(
            selected, artifact / "checkpoints" / "best_validation.pt", load_optimizer=False)
        holdout_config = copy.deepcopy(config)
        holdout_config["deployment"]["validation_seed"] = int(config["deployment"]["holdout_seed"])
        holdout = evaluate(model, task, holdout_config, selected,
                           episodes=int(config["deployment"]["holdout_episodes"]),
                           batch_size=int(validation["episodes"]), device=device)
        holdout["training_episodes"] = int(selected_checkpoint["episodes"])
        holdout["checkpoint"] = str(artifact / "checkpoints" / "best_validation.pt")
        _atomic_json(artifact / "deployment_holdout.json", holdout)
    final = {
        **last_row,
        "status": (
            "STOPPED" if episodes < episodes_target else "COMPLETED"
        ),
        "pid": os.getpid(),
        "device": str(device),
        "episodes": episodes,
        "successes": successes,
        "target_episodes": episodes_target,
        "collection_batch": collection_batch,
        "update_guard_enabled": bool(
            config.get("update_guard", {}).get("enabled", False)
        ),
        "success_rate": successes / max(episodes, 1),
        "elapsed_s": elapsed,
        "artifact": str(artifact),
    }
    if plateau.enabled:
        final['reward_convergence'] = dict(plateau.decision)
        final['stop_reason'] = ('Validation reward plateau' if plateau.decision['should_stop']
                                else 'User request' if episodes < episodes_target
                                else 'Attempt budget reached; reward plateau not established')
        final['stage'] = final['stop_reason']
    _atomic_json(artifact / "status.json", final)
    print(json.dumps(final, indent=2), flush=True)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--evaluate", type=Path, metavar="CHECKPOINT")
    parser.add_argument("--device", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument('--config-directory', type=Path)
    arguments = parser.parse_args()

    if arguments.preflight:
        preflight(arguments.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return 0
    if arguments.train:
        _, _, config = load_configs(arguments.config_directory)
        train(
            device_name=arguments.device or str(config["training"]["device"]),
            requested_episodes=arguments.episodes,
            batch_size=arguments.batch_size,
            artifact=arguments.artifact_directory,
            resume_checkpoint=arguments.resume_checkpoint,
            config_directory=arguments.config_directory,
        )
        return 0

    assert arguments.evaluate is not None
    model, task, config = load_configs()
    validate_contract(model, task, config)
    device = resolve_device(
        arguments.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    configure_accelerator(device)
    agent = build_agent(config, device)
    _load_checkpoint(agent, arguments.evaluate.resolve(), load_optimizer=False)
    result = evaluate(
        model,
        task,
        config,
        agent,
        episodes=arguments.episodes or int(config["validation"]["episodes"]),
        batch_size=arguments.batch_size or int(config["validation"]["episodes"]),
        device=device,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
