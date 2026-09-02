"""Run the selected pure-PPO whip controller training configuration."""

from __future__ import annotations

import argparse
from collections import deque
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

from learning.sequential_sac_env import (
    SEQUENTIAL_WHIP_OBSERVATION_DIM,
    sequential_action_dimension,
)
from learning.ppo_validation import (
    FixedMildStateValidationPanel,
    TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES,
    append_manual_validation_result,
    append_validation_result,
    write_training_plots,
    rolling_success_rate,
)
from learning.simple_ppo import PPORollout, SimplePPOAgent
from learning.ppo_publication import export_ppo_publication_figures
from learning.ppo_initial_states import MixedPPOInitialStateSampler
from learning.sequential_sac_env import SequentialWhipEnvironment
from run_simple_sac import _build_environment


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "learning"
    / "whip_ppo_state_bank_generalization_v1.json"
)


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


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _service_manual_validation_request(
    artifact: Path,
    config: dict[str, Any],
    agent: SimplePPOAgent,
    *,
    checkpoint_episodes: int,
) -> None:
    """Evaluate a UI request between PPO batches in the training process."""

    request_path = artifact / "manual_validation_request.json"
    request = _read_json(request_path)
    if request is None:
        return
    request_id = str(request.get("request_id", "manual")).replace("/", "_").replace(
        "\\", "_"
    )
    count = int(request.get("validation_episodes", 0))
    status_path = artifact / "manual_validation_status.json"
    if not 1 <= count <= 128:
        _write_json(
            status_path,
            {
                "status": "FAILED",
                "request_id": request_id,
                "error": "Manual validation count must be in [1, 128].",
            },
        )
        request_path.unlink(missing_ok=True)
        return
    _write_json(
        status_path,
        {
            "status": "RUNNING",
            "request_id": request_id,
            "validation_episodes": count,
            "checkpoint_episodes": checkpoint_episodes,
            "execution": "IN_TRAINING_PROCESS_BETWEEN_BATCHES",
        },
    )
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    try:
        panel = FixedMildStateValidationPanel(
            config,
            count_override=count,
            fixed_evaluation_batch_size=2048,
        )
        result = panel.evaluate(agent, checkpoint_episodes=checkpoint_episodes)
        result.update(
            {
                "schema": "ppo_manual_validation_v1",
                "request_id": request_id,
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_path": "IN_MEMORY_POLICY",
                "fixed_uav_residual_evaluation_batch_size": 2048,
                "new_training": False,
                "protected_test_evaluated": False,
                "hardware_executed": False,
            }
        )
        append_manual_validation_result(artifact, result)
        _write_json(
            artifact / "manual_validation_runs" / f"{request_id}.json",
            {"result": result, "state_manifest": panel.manifest},
        )
        _write_json(status_path, {"status": "COMPLETE", **result})
        write_training_plots(artifact)
    except BaseException as error:
        _write_json(
            status_path,
            {
                "status": "FAILED",
                "request_id": request_id,
                "error": f"{type(error).__name__}: {error}",
            },
        )
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        request_path.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema") not in {
        "simple_sequential_ppo_10s_v1",
        "simple_sequential_ppo_v1",
    }:
        raise ValueError("Unsupported simple-PPO configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Simple PPO requires the pinned production model freeze.")
    if config.get("comparison_environment") not in {
        "simple_sequential_sac_10s_v1",
        "task_whip_once_10s_v1",
        "task_whip_once_v1",
    }:
        raise ValueError("Unsupported PPO comparison environment.")
    requested_episodes = int(config["requested_episodes"])
    if requested_episodes < int(config["collection_batch"]):
        raise ValueError("PPO training must request at least one collection batch.")
    if not bool(config["batch_aligned_overshoot_allowed"]):
        raise ValueError("The fixed collection batch requires aligned overshoot.")
    if float(config["episode_duration_s"]) <= 0.0:
        raise ValueError("The PPO episode duration must be positive.")
    action_mode = str(config["action"].get("mode", "full_6d"))
    expected_action_dimension = sequential_action_dimension(action_mode)
    if int(config["action"]["dimensions"]) != expected_action_dimension:
        raise ValueError(
            f"Action mode {action_mode} requires {expected_action_dimension} dimensions."
        )
    success_mode = str(config.get("success_mode", "simple_endpoint"))
    if success_mode not in {"simple_endpoint", "task_whip_once"}:
        raise ValueError("Unsupported PPO success mode.")
    if success_mode == "task_whip_once":
        if str(config.get("reward_mode")) != "whip_potential":
            raise ValueError("Task-whip PPO requires bounded potential reward shaping.")
        speed_cap = float(config["reward"]["directed_speed_reward_cap_m_s"])
        if not math.isfinite(speed_cap) or speed_cap <= 0.0:
            raise ValueError("Directed-speed reward cap must be positive and finite.")
        compactness_bonus = float(
            config["reward"].get("success_compactness_bonus", 0.0)
        )
        compactness_scale = float(
            config["reward"].get("success_compactness_scale_m", 0.5)
        )
        if compactness_bonus < 0.0:
            raise ValueError("Success compactness bonus cannot be negative.")
        if not math.isfinite(compactness_scale) or compactness_scale <= 0.0:
            raise ValueError("Success compactness scale must be positive and finite.")
        displacement_cost_scale = float(
            config["reward"].get("displacement_cost_scale_m", 0.0)
        )
        if not math.isfinite(displacement_cost_scale) or displacement_cost_scale < 0.0:
            raise ValueError("Displacement cost scale must be finite and non-negative.")
        terminal_displacement_weight = float(
            config["reward"].get("terminal_displacement_weight", 0.0)
        )
        if (
            not math.isfinite(terminal_displacement_weight)
            or terminal_displacement_weight < 0.0
        ):
            raise ValueError(
                "Terminal displacement weight must be finite and non-negative."
            )
        terminal_success_only = config["reward"].get(
            "terminal_displacement_success_only", False
        )
        if not isinstance(terminal_success_only, bool):
            raise ValueError(
                "terminal_displacement_success_only must be a JSON boolean."
            )
        shaping_reference = str(
            config["reward"].get("directed_speed_shaping_reference", "world_tip")
        )
        if shaping_reference not in {"world_tip", "attachment_relative"}:
            raise ValueError("Unsupported directed-speed shaping reference.")
    validation = config.get("validation")
    if validation is not None and bool(validation.get("enabled", False)):
        if int(validation["episodes"]) < 10:
            raise ValueError("Validated PPO requires at least ten validation rollouts.")
        interval = int(validation["every_episodes"])
        if interval <= 0 or interval % int(config["collection_batch"]) != 0:
            raise ValueError("Validation cadence must align with the collection batch.")
        if not 0.0 < float(validation["maximum_distance_quantile"]) <= 1.0:
            raise ValueError("Validation state-distance quantile must lie in (0,1].")
        if int(validation.get("fixed_numerical_batch_size", 2048)) != 2048:
            raise ValueError(
                "PPO validation must use the fixed production numerical batch of 2048."
            )
    rolling_window = int(
        config.get("logging", {}).get(
            "training_success_rolling_window_episodes", 5_000
        )
    )
    if rolling_window <= 0:
        raise ValueError("Training-success rolling window must be positive.")
    early_stopping = config.get("early_stopping", {})
    if bool(early_stopping.get("enabled", False)):
        if int(early_stopping.get("minimum_episodes", 0)) < 0:
            raise ValueError("Early-stopping minimum episodes cannot be negative.")
        if int(early_stopping.get("validation_patience_evaluations", 0)) < 1:
            raise ValueError(
                "Validation early stopping requires positive evaluation patience."
            )
        selection_metric = str(
            early_stopping.get(
                "selection_metric",
                "validation_success_rate_then_lower_mean_uav_displacement",
            )
        )
        if selection_metric not in {
            "validation_success_rate_then_lower_mean_uav_displacement",
            "lower_mean_uav_displacement_subject_to_success_floor",
        }:
            raise ValueError("Unsupported validation checkpoint-selection metric.")
        if selection_metric == "lower_mean_uav_displacement_subject_to_success_floor":
            success_floor = float(
                early_stopping.get("minimum_validation_success_rate", 0.0)
            )
            if not 0.0 <= success_floor <= 1.0:
                raise ValueError("Validation success floor must lie in [0,1].")
        degradation_floor = early_stopping.get(
            "abort_below_validation_success_rate"
        )
        if degradation_floor is not None:
            degradation_floor = float(degradation_floor)
            if not 0.0 <= degradation_floor <= 1.0:
                raise ValueError(
                    "Validation degradation-abort floor must lie in [0,1]."
                )
            if int(
                early_stopping.get(
                    "abort_below_patience_evaluations", 1
                )
            ) <= 0:
                raise ValueError(
                    "Validation degradation-abort patience must be positive."
                )
    initial_states = config.get("training_initial_states", {"mode": "canonical"})
    initial_state_mode = str(initial_states.get("mode", "canonical"))
    if initial_state_mode not in {"canonical", "mixed_state_bank"}:
        raise ValueError("Unsupported PPO training initial-state mode.")
    if initial_state_mode == "mixed_state_bank":
        required = (
            "training_bank",
            "training_bank_manifest",
            "validation_bank",
            "validation_bank_manifest",
            "canonical_fraction",
            "seed",
        )
        missing = [name for name in required if name not in initial_states]
        if missing:
            raise ValueError(
                f"Mixed PPO initial-state configuration is missing: {missing}"
            )
        canonical_fraction = float(initial_states["canonical_fraction"])
        if not 0.0 <= canonical_fraction <= 1.0:
            raise ValueError("Canonical initial-state fraction must lie in [0,1].")
        if Path(str(initial_states["training_bank"])) == Path(
            str(initial_states["validation_bank"])
        ):
            raise ValueError("Training and validation state-bank paths must differ.")
    return config


def _build_initial_state_sampler(
    config: dict[str, Any],
    environment: SequentialWhipEnvironment,
) -> MixedPPOInitialStateSampler | None:
    source = config.get("training_initial_states", {"mode": "canonical"})
    if str(source.get("mode", "canonical")) == "canonical":
        return None
    return MixedPPOInitialStateSampler.create(
        training_bank_path=ROOT / str(source["training_bank"]),
        training_manifest_path=ROOT / str(source["training_bank_manifest"]),
        validation_bank_path=ROOT / str(source["validation_bank"]),
        validation_manifest_path=ROOT / str(source["validation_bank_manifest"]),
        canonical_state=environment.state,
        command_yaw_world_rad=environment.task.initial_yaw_rad,
        batch_size=environment.batch_size,
        canonical_fraction=float(source["canonical_fraction"]),
        seed=int(source["seed"]),
    )


def _reset_training_environment(
    environment: SequentialWhipEnvironment,
    sampler: MixedPPOInitialStateSampler | None,
) -> torch.Tensor:
    if sampler is not None:
        selected = sampler.sample(device=environment.simulator.device)
        environment.set_episode_initial_state(
            selected.state,
            command_yaw_world_rad=selected.command_yaw_world_rad,
        )
    return environment.reset()


def _build_agent(config: dict[str, Any], device: torch.device) -> SimplePPOAgent:
    source = config["ppo"]
    return SimplePPOAgent(
        SEQUENTIAL_WHIP_OBSERVATION_DIM,
        int(config["action"]["dimensions"]),
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
    initial_state_sampler = _build_initial_state_sampler(config, environment)
    agent = _build_agent(config, device)
    initialization_checkpoint = config.get("initialization", {}).get(
        "policy_checkpoint"
    )
    if initialization_checkpoint:
        source_checkpoint = torch.load(
            ROOT / str(initialization_checkpoint),
            map_location=device,
            weights_only=False,
        )
        agent.policy.load_state_dict(source_checkpoint["policy"])
        if bool(
            config.get("initialization", {}).get("load_value_network", False)
        ):
            agent.value.load_state_dict(source_checkpoint["value"])
    observation = _reset_training_environment(environment, initial_state_sampler)
    rollout = PPORollout.allocate(
        environment.control_step_count,
        4,
        SEQUENTIAL_WHIP_OBSERVATION_DIM,
        int(config["action"]["dimensions"]),
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
    rolling_success_window: deque[int],
    initial_state_sampler: MixedPPOInitialStateSampler | None,
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
            "initial_state_generator_state": (
                None
                if initial_state_sampler is None
                else initial_state_sampler.get_state()
            ),
            "rolling_success_window": torch.tensor(
                list(rolling_success_window), dtype=torch.uint8
            ),
            "rolling_success_window_episodes": int(
                rolling_success_window.maxlen or len(rolling_success_window)
            ),
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
    config: dict[str, Any],
) -> bool:
    """Persist using the configured validation success/compactness contract."""

    metadata_path = artifact / "best_validation.json"
    previous = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    early_stopping = config.get("early_stopping", {})
    selection_metric = str(
        early_stopping.get(
            "selection_metric",
            "validation_success_rate_then_lower_mean_uav_displacement",
        )
    )

    def validation_rank(row: dict[str, Any]) -> tuple[float, float, float]:
        success = float(row["validation_success_rate"])
        displacement = float(row["mean_maximum_uav_displacement_m"])
        if selection_metric == "lower_mean_uav_displacement_subject_to_success_floor":
            floor = float(early_stopping["minimum_validation_success_rate"])
            return (
                float(success >= floor),
                -displacement if success >= floor else success,
                success if success >= floor else -displacement,
            )
        return (
            success,
            -displacement,
            float(row["validation_legacy_scientific_success_rate"]),
        )

    rank = validation_rank(result)
    previous_rank = (
        validation_rank(previous)
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
        ROOT / "learning" / "ppo_initial_states.py",
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
    initial_state_config = config.get(
        "training_initial_states", {"mode": "canonical"}
    )
    if str(initial_state_config.get("mode", "canonical")) == "mixed_state_bank":
        source_paths.extend(
            ROOT / str(initial_state_config[name])
            for name in (
                "training_bank",
                "training_bank_manifest",
                "validation_bank",
                "validation_bank_manifest",
            )
        )
    if not resume:
        _write_json(
            artifact / "source_hash_manifest.json",
            {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
        )
        full_mode = config.get("success_mode") == "task_whip_once"
        episode_duration_s = float(config["episode_duration_s"])
        validation_every = int(config.get("validation", {}).get("every_episodes", 0))
        validation_episodes = int(config.get("validation", {}).get("episodes", 0))
        contract_title = (
            f"# Task-whip PPO reward study — {episode_duration_s:g}-second episode\n\n"
            if full_mode
            else f"# Validated simple sequential PPO — {episode_duration_s:g}-second whip\n\n"
        )
        success_contract = (
            "success requires the single first target entry to be tip-first with speed and direction; entry timestep is diagnostic only and numerical UAV limits are smooth costs. "
            if full_mode
            else "endpoint success gate is unchanged. "
        )
        speed_reward_cap = float(
            config["reward"].get("directed_speed_reward_cap_m_s", math.inf)
        )
        action_mode = str(config["action"].get("mode", "full_6d"))
        action_description = (
            "target-aligned forward/backward and vertical acceleration plus pitch rate"
            if action_mode == "target_aligned_sagittal_3d"
            else "3-D acceleration plus roll/pitch/yaw body rates"
        )
        shaping_reference = str(
            config["reward"].get("directed_speed_shaping_reference", "world_tip")
        )
        initial_state_description = (
            "Training uses the canonical settled initial state. "
            if str(initial_state_config.get("mode", "canonical")) == "canonical"
            else (
                "Every training batch is sampled from the existing physically "
                "propagated TRAIN state bank with "
                f"{100.0 * float(initial_state_config['canonical_fraction']):g}% "
                "canonical rows; the validation state bank is disjoint and never "
                "used for gradients. "
            )
        )
        (artifact / "RUN_CONTRACT.md").write_text(
            contract_title
            + initial_state_description
            + "Every "
            + f"{validation_every:,} episodes, "
            f"the deterministic policy is evaluated on the same {validation_episodes} held-out, physically varied, "
            "physically propagated initial states. Training and validation both use "
            "the production UAV + causal residual + 12-node DDER simulator. The policy "
            + f"action is {action_description}. The reported "
            + success_contract
            + f"Speed-shaping credit is capped at {speed_reward_cap:g} m/s. "
            + f"The dense speed-shaping reference is {shaping_reference}; hard success "
            "still uses true world-frame tip velocity. "
            + "No CEM, protected data, or hardware "
            "is used. A STOP_REQUESTED file causes a checkpointed cooperative stop.\n",
            encoding="utf-8",
        )

    collection_batch = int(config["collection_batch"])
    environment, device = _build_environment(config, batch_size=collection_batch)
    if device.type != "cuda":
        raise RuntimeError("PPO training requires CUDA.")
    initial_state_sampler = _build_initial_state_sampler(config, environment)
    if initial_state_sampler is not None and not resume:
        _write_json(
            artifact / "training_initial_state_manifest.json",
            initial_state_sampler.manifest,
        )
    agent = _build_agent(config, device)
    if not resume and initialization_checkpoint:
        source_checkpoint = torch.load(
            ROOT / str(initialization_checkpoint),
            map_location=device,
            weights_only=False,
        )
        agent.policy.load_state_dict(source_checkpoint["policy"])
        if bool(initialization.get("load_value_network", False)):
            agent.value.load_state_dict(source_checkpoint["value"])
    ppo = config["ppo"]
    rollout = PPORollout.allocate(
        environment.control_step_count,
        collection_batch,
        SEQUENTIAL_WHIP_OBSERVATION_DIM,
        int(config["action"]["dimensions"]),
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
    rolling_window_episodes = int(
        config.get("logging", {}).get(
            "training_success_rolling_window_episodes",
            TRAINING_SUCCESS_ROLLING_WINDOW_EPISODES,
        )
    )
    rolling_success_window: deque[int] = deque(maxlen=rolling_window_episodes)
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
        saved_rolling_window = checkpoint.get("rolling_success_window")
        if saved_rolling_window is not None:
            rolling_success_window.extend(
                int(value) for value in saved_rolling_window.cpu().tolist()
            )
        # ``map_location=device`` is required for the model/optimizer state, but
        # RNG APIs require their serialized byte-state tensors on CPU.
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "update_generator_state" in checkpoint:
            update_generator.set_state(checkpoint["update_generator_state"].cpu())
        saved_initial_state_generator = checkpoint.get(
            "initial_state_generator_state"
        )
        if initial_state_sampler is not None and saved_initial_state_generator is not None:
            initial_state_sampler.set_state(saved_initial_state_generator)
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
        "rolling_success_rate",
        "rolling_window_episodes",
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
        "mean_terminal_uav_displacement_m",
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
    if resume and log_path.is_file():
        with log_path.open(newline="", encoding="utf-8") as stream:
            existing_rows = list(csv.DictReader(stream))
        if existing_rows and "rolling_success_rate" not in existing_rows[0]:
            historical_episodes = np.asarray(
                [float(row["episodes"]) for row in existing_rows]
            )
            historical_successes = np.asarray(
                [float(row["successes"]) for row in existing_rows]
            )
            historical_rolling = rolling_success_rate(
                historical_episodes,
                historical_successes,
                window_episodes=rolling_window_episodes,
            )
            for row, value in zip(existing_rows, historical_rolling, strict=True):
                row["rolling_success_rate"] = float(value)
                row["rolling_window_episodes"] = rolling_window_episodes
            temporary_log = log_path.with_suffix(".csv.tmp")
            with temporary_log.open("w", newline="", encoding="utf-8") as stream:
                migration_writer = csv.DictWriter(stream, fieldnames=fields)
                migration_writer.writeheader()
                migration_writer.writerows(
                    {name: row.get(name) for name in fields}
                    for row in existing_rows
                )
            os.replace(temporary_log, log_path)
    _write_json(
        artifact / "status.json",
        {
            "status": "RESUMING" if resume else "STARTING",
            "process_id": os.getpid(),
            "episodes": episodes,
            "requested_episodes": requested_episodes,
            "success_rate": 0.0 if episodes == 0 else successes / episodes,
            "rolling_window_episodes": rolling_window_episodes,
            "endpoint_success_rate": (
                0.0 if episodes == 0 else endpoint_successes / episodes
            ),
            "episodes_per_second": 0.0,
        },
    )
    early_stopping_triggered = False
    early_stopping_reason: str | None = None
    consecutive_degraded_validations = 0

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
                _consider_best_validation(
                    artifact, agent, initial_validation, config
                )
                write_training_plots(artifact)
            while episodes < requested_episodes:
                observation = _reset_training_environment(
                    environment, initial_state_sampler
                )
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
                rolling_success_window.extend(
                    int(value)
                    for value in environment.episode_success.detach().cpu().tolist()
                )
                current_rolling_success_rate = (
                    sum(rolling_success_window) / len(rolling_success_window)
                )
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
                    "rolling_success_rate": current_rolling_success_rate,
                    "rolling_window_episodes": rolling_window_episodes,
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
                    "mean_terminal_uav_displacement_m": float(
                        torch.nanmean(
                            environment.episode_terminal_displacement
                        ).cpu()
                    ),
                    "numerical_failure_rate": float(environment.failed.float().mean().cpu()),
                    "gradient_updates": agent.gradient_updates,
                    "valid_transitions": metrics.valid_transitions,
                    **asdict(metrics),
                }
                writer.writerow({name: row.get(name) for name in fields})
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
                        "rolling_success_rate": current_rolling_success_rate,
                        "rolling_window_episodes": rolling_window_episodes,
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
                        "mean_maximum_uav_displacement_m": row[
                            "mean_maximum_uav_displacement_m"
                        ],
                        "mean_terminal_uav_displacement_m": row[
                            "mean_terminal_uav_displacement_m"
                        ],
                    },
                )
                print(
                    f"episodes={episodes}/{requested_episodes} "
                    f"batch_success={100.0 * batch_successes / collection_batch:.3f}% "
                    f"rolling_{rolling_window_episodes}="
                    f"{100.0 * current_rolling_success_rate:.3f}% "
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
                        rolling_success_window=rolling_success_window,
                        initial_state_sampler=initial_state_sampler,
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
                    validation_improved = _consider_best_validation(
                        artifact, agent, validation_result, config
                    )
                    write_training_plots(artifact)
                    print(
                        f"validation_episodes={episodes} "
                        f"success={validation_result['validation_successes']}/"
                        f"{validation_result['validation_episodes']}",
                        flush=True,
                    )
                    while next_validation <= episodes:
                        next_validation += validation_interval
                    early_stopping = config.get("early_stopping", {})
                    degradation_floor = early_stopping.get(
                        "abort_below_validation_success_rate"
                    )
                    if degradation_floor is not None:
                        if float(validation_result["validation_success_rate"]) < float(
                            degradation_floor
                        ):
                            consecutive_degraded_validations += 1
                        else:
                            consecutive_degraded_validations = 0
                        degradation_patience = int(
                            early_stopping.get(
                                "abort_below_patience_evaluations", 1
                            )
                        )
                        if (
                            consecutive_degraded_validations
                            >= degradation_patience
                        ):
                            early_stopping_triggered = True
                            early_stopping_reason = (
                                "Validation safety stop: success remained below "
                                f"{100.0 * float(degradation_floor):.1f}% for "
                                f"{consecutive_degraded_validations} consecutive "
                                "scheduled evaluations."
                            )
                    if (
                        bool(early_stopping.get("enabled", False))
                        and episodes
                        >= int(early_stopping.get("minimum_episodes", 0))
                        and not validation_improved
                    ):
                        best = _read_json(artifact / "best_validation.json") or {}
                        best_episodes = int(best.get("checkpoint_episodes", 0))
                        patience = int(
                            early_stopping["validation_patience_evaluations"]
                        )
                        if (
                            not early_stopping_triggered
                            and episodes - best_episodes
                            >= patience * validation_interval
                        ):
                            early_stopping_triggered = True
                            early_stopping_reason = (
                                f"No validation improvement for {patience} scheduled "
                                f"evaluations after best checkpoint at {best_episodes:,} episodes."
                            )
                _service_manual_validation_request(
                    artifact,
                    config,
                    agent,
                    checkpoint_episodes=episodes,
                )
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
                        rolling_success_window=rolling_success_window,
                        initial_state_sampler=initial_state_sampler,
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
                        "rolling_success_rate": (
                            sum(rolling_success_window) / len(rolling_success_window)
                        ),
                        "rolling_window_episodes": rolling_window_episodes,
                        "episodes_per_second": episodes / elapsed,
                        "elapsed_s": elapsed,
                        "latest_durable_checkpoint_episodes": episodes,
                    }
                    _write_json(artifact / "status.json", summary)
                    _write_json(artifact / "stopped_run_summary.json", summary)
                    print(json.dumps(summary, indent=2), flush=True)
                    return
                if early_stopping_triggered:
                    break
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
                "rolling_window_episodes": rolling_window_episodes,
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
                rolling_success_window=rolling_success_window,
                initial_state_sampler=initial_state_sampler,
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
        rolling_success_window=rolling_success_window,
        initial_state_sampler=initial_state_sampler,
    )
    write_training_plots(artifact)
    summary = {
        "status": (
            "EARLY_STOPPED_VALIDATION" if early_stopping_triggered else "COMPLETE"
        ),
        "episodes": episodes,
        "requested_episodes": requested_episodes,
        "batch_aligned_overshoot": max(0, episodes - requested_episodes),
        "episodes_remaining": max(0, requested_episodes - episodes),
        "successes": successes,
        "endpoint_successes": endpoint_successes,
        "legacy_scientific_successes": legacy_scientific_successes,
        "success_rate": successes / episodes,
        "rolling_success_rate": (
            sum(rolling_success_window) / len(rolling_success_window)
        ),
        "rolling_window_episodes": rolling_window_episodes,
        "episodes_per_second": episodes / elapsed,
        "elapsed_s": elapsed,
        "early_stopping_reason": early_stopping_reason,
    }
    _write_json(artifact / "status.json", summary)
    _write_json(artifact / "final_summary.json", summary)
    try:
        export_ppo_publication_figures(artifact)
    except (FileNotFoundError, ValueError, OSError) as error:
        _write_json(
            artifact / "publication_figure_export_error.json",
            {"error": repr(error)},
        )
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
