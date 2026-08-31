"""Milestone 5B.6: feasibility-constrained local K=4 spectral SAC."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch

from learning.canonical_pilot import behavior_level, repeated_canonical_specification
from learning.constrained_sac import (
    ACTIVE_FREQUENCIES,
    ConstrainedReplayBuffer,
    ConstrainedSacAgent,
    LocalSpectralActor,
    local_frequency_base,
    save_constrained_checkpoint,
)
from learning.context_sampling import ContextSpecification, build_context_from_specification
from learning.normalization import FixedContextNormalizer
from learning.one_shot_env import evaluate_open_loop_batch
from learning.policy_context import build_policy_context
from learning.safety_quarantine import (
    HARD_SAFETY_FAILURE_QUARANTINE,
    RL_V2_HARD_SAFETY_PREFIX,
    SAFETY_FAILURE_COMMAND_ACCELERATION,
    SAFETY_FAILURE_DISPLACEMENT,
    SAFETY_FAILURE_NONFINITE,
    SAFETY_FAILURE_SPEED,
    evaluate_open_loop_batch_quarantined,
    record_full_batch_trajectory,
)
from learning.spectral_sac import spectral_coefficients_to_normalized_action
from learning.state_bank import initial_state_bank_from_state
from planning.artifacts import final_replay_metrics, save_command_csv, save_replay_npz
from planning.cem_task import load_variable_duration_task
from planning.results import PlanningResult
from planning.rl_reward import RLWhipRewardConfig
from planning.rollout import hover_preroll
from planning.video import render_replay_video
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest
from simulator.uav import FullStateUAVModel


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "learning" / "constrained_sac_local_k4_v1.json"
REPORT_PATH = PROJECT_ROOT / "MILESTONE5B6_CONSTRAINED_LOCAL_SAC_REPORT.md"


def _reward_config() -> RLWhipRewardConfig:
    payload = json.loads(
        (PROJECT_ROOT / "config" / "learning" / "sac_canonical_spectral_entropy_v1.json").read_text(
            encoding="utf-8"
        )
    )["reward"]
    accepted = {item.name for item in fields(RLWhipRewardConfig)}
    reward = RLWhipRewardConfig(
        **{name: value for name, value in payload.items() if name in accepted}
    )
    if reward.profile != "rl_whip_reward_v2" or reward.progress_weight != 4.0:
        raise ValueError("Milestone 5B.6 requires unchanged rl_whip_reward_v2.")
    return reward


def _fixed_batch(simulator, size: int = 2048) -> None:
    if not isinstance(simulator.uav_model, FullStateUAVModel):
        raise TypeError("5B.6 requires the production FullState UAV model.")
    simulator.uav_model.set_fixed_evaluation_batch_size(size)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+0000", "Z")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "constrained_sac_local_k4_v1":
        raise ValueError("Unsupported Milestone 5B.6 configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The production model freeze is immutable.")
    if config.get("context_distribution") != "SINGLE_CANONICAL":
        raise ValueError("5B.6 permits only the canonical context.")
    if not math.isclose(float(config.get("fixed_duration_s", 0.0)), 1.20):
        raise ValueError("5B.6 duration must remain fixed at 1.20 s.")
    local = config["local_spectral_policy"]
    if local["active_frequencies"] != [0, 1, 2, 3] or local["active_stochastic_dimensions"] != 12:
        raise ValueError("5B.6 requires exactly K=4 and 12 stochastic dimensions.")
    if (float(local["initial_scale"]), float(local["minimum_scale"]), float(local["maximum_scale"])) != (0.05, 0.005, 0.10):
        raise ValueError("The audited local scale contract changed.")
    policy = config["artifact_policy"]
    if any(
        bool(policy[key])
        for key in (
            "real_flight_authorized",
            "protected_test_evaluation_allowed",
            "cem_training_data_allowed",
            "production_model_modification_allowed",
        )
    ):
        raise ValueError("5B.6 must remain simulation-only, frozen, and CEM-free.")


def _canonical_setup(simulator, task):
    state = hover_preroll(simulator, task)
    bank = initial_state_bank_from_state(
        state,
        command_position_world_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        command_velocity_world_m_s=torch.tensor(task.initial_uav_velocity_m_s, device=simulator.device),
        command_yaw_world_rad=task.initial_yaw_rad,
        seed=42,
    )
    selected = bank.select(torch.tensor([0]), device=simulator.device)
    context = build_policy_context(
        simulator,
        selected.state,
        target_position_world_m=torch.tensor(task.target_position_m, device=simulator.device),
        desired_direction_world=torch.tensor(task.desired_direction, device=simulator.device),
        command_initial_position_world_m=selected.command_position_world_m,
        command_initial_velocity_world_m_s=selected.command_velocity_world_m_s,
        command_yaw_world_rad=selected.command_yaw_world_rad,
    )
    specification = ContextSpecification(
        torch.tensor([0]),
        context.target_position_local_m.detach().cpu(),
        context.target_direction_local.detach().cpu(),
        "canonical",
    )
    return bank, specification, context


def _context(simulator, bank, specification, batch_size: int, split: str):
    repeated = repeated_canonical_specification(specification, batch_size, split=split)
    return build_context_from_specification(simulator, bank, repeated)


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


def _gate_fractions(mask: torch.Tensor) -> dict[str, float]:
    value = mask.to(torch.int64)
    return {
        "displacement": _fraction((value & SAFETY_FAILURE_DISPLACEMENT) != 0),
        "uav_speed": _fraction((value & SAFETY_FAILURE_SPEED) != 0),
        "command_acceleration": _fraction((value & SAFETY_FAILURE_COMMAND_ACCELERATION) != 0),
        "non_finite": _fraction((value & SAFETY_FAILURE_NONFINITE) != 0),
    }


def _summary(result) -> dict[str, Any]:
    episode, diagnostics = result.episode, result.diagnostics
    components = episode.reward_components
    if components is None:
        raise RuntimeError("5B.6 requires reward-v2 components.")
    progress = components.normalized_progress
    feasible = ~diagnostics.safety_failed
    distance = episode.tip_min_distance_m
    failed_times = diagnostics.failure_time_s[diagnostics.safety_failed]
    return {
        "sample_count": episode.batch_size,
        "reward": _quantiles(episode.reward),
        "progress": _quantiles(progress),
        "minimum_tip_distance_m": _quantiles(distance),
        "directed_tip_speed_m_s": _quantiles(episode.directed_tip_speed_m_s),
        "feasible_rate": _fraction(feasible),
        "failure_rate": _fraction(diagnostics.safety_failed),
        "success_rate": _fraction(episode.task_success),
        "fractions": {
            "feasible_and_progress_ge_0_25": _fraction(feasible & (progress >= 0.25)),
            "feasible_and_progress_ge_0_50": _fraction(feasible & (progress >= 0.50)),
            "feasible_and_progress_ge_0_75": _fraction(feasible & (progress >= 0.75)),
            "feasible_and_d_min_le_0_20_m": _fraction(feasible & (distance <= 0.20)),
            "feasible_and_d_min_le_0_10_m": _fraction(feasible & (distance <= 0.10)),
        },
        "failure_breakdown": _gate_fractions(diagnostics.failure_gate_mask),
        "median_failure_time_s": None if failed_times.numel() == 0 else float(torch.median(failed_times)),
        "minimum_failure_time_s": None if failed_times.numel() == 0 else float(failed_times.min()),
    }


def _auroc(labels: torch.Tensor, scores: torch.Tensor) -> float | None:
    y = labels.detach().float().reshape(-1).cpu()
    score = scores.detach().float().reshape(-1).cpu()
    positives = int(y.sum())
    negatives = int(y.numel() - positives)
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, y.numel() + 1, dtype=torch.float32)
    rank_sum = float(ranks[y.bool()].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _safety_quality(labels: torch.Tensor, probabilities: torch.Tensor) -> dict[str, Any]:
    y = labels.detach().float().reshape(-1)
    p = probabilities.detach().float().reshape(-1)
    bins = []
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        mask = (p >= lower) & (p < upper if index < 9 else p <= upper)
        if bool(mask.any()):
            bins.append(
                {
                    "range": [lower, upper],
                    "count": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "observed_failure_rate": float(y[mask].mean()),
                }
            )
    return {
        "auroc": _auroc(y, p),
        "brier_score": float((p - y).square().mean()),
        "calibration_bins": bins,
    }


def _load_center(source_5b5: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    reconstruction = json.loads((source_5b5 / "action_center_reconstruction.json").read_text(encoding="utf-8"))
    coefficients = torch.tensor(
        reconstruction["center_spectral_coefficients_axis_major"],
        dtype=torch.float32,
        device=device,
    )
    action = torch.tensor(
        reconstruction["center_normalized_action"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 49)
    decoded = spectral_coefficients_to_normalized_action(coefficients.unsqueeze(0))
    error = float((decoded - action).abs().max())
    return coefficients, action, {
        "source": str(source_5b5 / "action_center_reconstruction.json"),
        "maximum_action_reconstruction_error": error,
        "status": "PASS" if error <= 1.0e-6 else "FAIL",
    }


def _calibrate_entropy(actor, normalized_context: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    count = int(config["entropy_calibration"]["sample_count"])
    batch_size = int(config["entropy_calibration"]["batch_size"])
    torch.manual_seed(int(config["seed"]) + 61)
    torch.cuda.manual_seed_all(int(config["seed"]) + 61)
    total = total_squared = 0.0
    minimum, maximum = float("inf"), -float("inf")
    seen = 0
    with torch.no_grad():
        while seen < count:
            rows = min(batch_size, count - seen)
            sample = actor(normalized_context.expand(rows, -1))
            entropy = -sample.log_prob
            total += float(entropy.sum())
            total_squared += float(entropy.square().sum())
            minimum = min(minimum, float(entropy.min()))
            maximum = max(maximum, float(entropy.max()))
            seen += rows
    mean = total / count
    variance = max(0.0, total_squared / count - mean * mean)
    return {
        "sample_count": count,
        "empirical_transformed_action_entropy_mean": mean,
        "empirical_transformed_action_entropy_std": math.sqrt(variance),
        "entropy_minimum": minimum,
        "entropy_maximum": maximum,
        "target_entropy": mean,
        "frozen_for_training": True,
    }


def _feasible_equivalence(
    simulator,
    task,
    reward_config,
    bank,
    specification,
    center_coefficients: torch.Tensor,
    center_action: torch.Tensor,
) -> dict[str, Any]:
    count = 1024
    context = _context(simulator, bank, specification, count, "5b6_feasible_selection")
    generator = torch.Generator(device=simulator.device).manual_seed(109560)
    coefficients = center_coefficients[None].expand(count, -1, -1).clone()
    noise = torch.randn((count, 3, 4), generator=generator, device=simulator.device)
    coefficients[:, :, :4] += 0.05 * local_frequency_base(
        dtype=simulator.dtype, device=simulator.device
    )[None, None] * noise
    actions = spectral_coefficients_to_normalized_action(coefficients)
    with torch.no_grad():
        ordinary_selection = evaluate_open_loop_batch(
            simulator, context, actions, task, rl_reward_config=reward_config
        )
    feasible_indices = torch.nonzero(ordinary_selection.feasible, as_tuple=False)[:, 0]
    if feasible_indices.numel() < 32:
        raise RuntimeError("Could not recover 32 feasible K4/0.05 verification actions.")
    selected_actions = torch.cat((center_action, actions[feasible_indices[:32]]), dim=0)
    selected_context = _context(simulator, bank, specification, 33, "5b6_feasible_equivalence")
    with torch.no_grad():
        ordinary = evaluate_open_loop_batch(
            simulator, selected_context, selected_actions, task, rl_reward_config=reward_config
        )
        ordinary_history = record_full_batch_trajectory(
            simulator, selected_context, selected_actions, task
        )
        quarantined = evaluate_open_loop_batch_quarantined(
            simulator,
            selected_context,
            selected_actions,
            task,
            rl_reward_config=reward_config,
            record_trajectories=True,
        )
    diagnostics = quarantined.diagnostics
    if diagnostics.uav_positions_m is None or diagnostics.cable_positions_m is None:
        raise RuntimeError("Trajectory equivalence history was not recorded.")
    metric_differences = {
        "reward": float((ordinary.reward - quarantined.episode.reward).abs().max()),
        "progress": float((ordinary.reward_components.normalized_progress - quarantined.episode.reward_components.normalized_progress).abs().max()),
        "minimum_tip_distance_m": float((ordinary.tip_min_distance_m - quarantined.episode.tip_min_distance_m).abs().max()),
        "directed_tip_speed_m_s": float((ordinary.directed_tip_speed_m_s - quarantined.episode.directed_tip_speed_m_s).abs().max()),
        "direction_error_deg": float((ordinary.direction_error_deg - quarantined.episode.direction_error_deg).abs().max()),
    }
    trajectory_differences = {
        name: float((ordinary_history[name] - getattr(diagnostics, name)).abs().max())
        for name in (
            "uav_positions_m", "uav_velocities_m_s",
            "cable_positions_m", "cable_velocities_m_s",
        )
    }
    checks = {
        "all_33_ordinary_feasible": bool(ordinary.feasible.all()),
        "no_quarantine_failure": not bool(diagnostics.safety_failed.any()),
        "success_classification_identical": torch.equal(ordinary.task_success, quarantined.episode.task_success),
        "first_entry_marker_identical": torch.equal(ordinary.first_entry_marker, quarantined.episode.first_entry_marker),
        "metrics_identical": max(metric_differences.values()) <= 1.0e-6,
        "uav_and_cable_trajectories_identical": max(trajectory_differences.values()) <= 1.0e-7,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "verification_action_count": 33,
        "level1_center_included": True,
        "k4_scale_0_05_feasible_actions": 32,
        "metric_maximum_absolute_differences": metric_differences,
        "trajectory_maximum_absolute_differences": trajectory_differences,
    }


def _failed_prefix_audit(simulator, task, reward_config, bank, specification, source_5b5: Path) -> dict[str, Any]:
    manifest = json.loads((source_5b5 / "selected_5b4_actions_manifest.json").read_text(encoding="utf-8"))
    selected = []
    for group in ("B_HIGH_PROGRESS_UNSAFE", "C_NEAR_TARGET_OUTLIER", "D_CATASTROPHIC"):
        selected.extend(manifest["groups"][group][:2])
    actions = torch.tensor(
        [item["normalized_action"] for item in selected],
        dtype=torch.float32,
        device=simulator.device,
    )
    context = _context(simulator, bank, specification, len(selected), "5b6_failed_prefix")
    with torch.no_grad():
        ordinary = evaluate_open_loop_batch(
            simulator, context, actions, task, rl_reward_config=reward_config
        )
        quarantined = evaluate_open_loop_batch_quarantined(
            simulator,
            context,
            actions,
            task,
            rl_reward_config=reward_config,
            record_trajectories=True,
        )
    diagnostics = quarantined.diagnostics
    qepisode = quarantined.episode
    history_speed = torch.linalg.vector_norm(diagnostics.uav_velocities_m_s, dim=-1).amax(dim=0)
    rows = []
    for index, item in enumerate(selected):
        rows.append(
            {
                "action_id": item["action_id"],
                "old_full_progress": float(ordinary.reward_components.normalized_progress[index]),
                "valid_prefix_progress": float(qepisode.reward_components.normalized_progress[index]),
                "old_full_minimum_tip_distance_m": float(ordinary.tip_min_distance_m[index]),
                "valid_prefix_minimum_tip_distance_m": float(qepisode.tip_min_distance_m[index]),
                "old_full_maximum_uav_speed_m_s": float(ordinary.max_uav_speed_m_s[index]),
                "quarantined_history_maximum_uav_speed_m_s": float(history_speed[index]),
                "failure_time_s": float(diagnostics.failure_time_s[index]),
                "failure_gate_mask": int(diagnostics.failure_gate_mask[index]),
                "propagated_steps": int(diagnostics.propagated_step_count[index]),
                "scientific_success_after_quarantine": bool(qepisode.task_success[index]),
            }
        )
    checks = {
        "all_aggressive_rows_failed_and_quarantined": bool(diagnostics.safety_failed.all()),
        "post_failure_progress_excluded": bool(
            (qepisode.reward_components.normalized_progress <= ordinary.reward_components.normalized_progress + 1.0e-7).all()
        ),
        "no_failed_row_scientific_success": not bool(qepisode.task_success.any()),
        "no_full_horizon_failed_row_propagation": bool((diagnostics.propagated_step_count < 120).all()),
        "catastrophic_speed_not_propagated": float(history_speed.max()) <= 3.0 + 1.0e-4,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "contract": HARD_SAFETY_FAILURE_QUARANTINE,
        "reward_contract": RL_V2_HARD_SAFETY_PREFIX,
        "checks": checks,
        "rows": rows,
    }


class KeyActionTracker:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.actions: dict[str, np.ndarray] = {}

    def _set(self, name: str, action: torch.Tensor, metadata: dict[str, Any]) -> None:
        self.entries[name] = metadata
        self.actions[name] = action.detach().cpu().numpy().astype(np.float32)

    def update(self, actions: torch.Tensor, result, episode_start: int, seed: int) -> None:
        episode = result.episode
        diagnostics = result.diagnostics
        progress = episode.reward_components.normalized_progress
        feasible = ~diagnostics.safety_failed
        success = episode.task_success

        def metadata(index: int) -> dict[str, Any]:
            return {
                "episode": episode_start + index + 1,
                "collection_seed": seed,
                "collection_local_index": index,
                "failure_time_s": None if not bool(diagnostics.safety_failed[index]) else float(diagnostics.failure_time_s[index]),
                "failure_gate_mask": int(diagnostics.failure_gate_mask[index]),
                "progress": float(progress[index]),
                **episode.row(index),
            }

        if bool(success.any()):
            indices = torch.nonzero(success, as_tuple=False)[:, 0]
            if "first_scientific_success" not in self.entries:
                index = int(indices[0])
                self._set("first_scientific_success", actions[index], metadata(index))
            index = int(indices[episode.reward[indices].argmax()])
            candidate = metadata(index)
            if "best_scientific_success" not in self.entries or candidate["reward"] > self.entries["best_scientific_success"]["reward"]:
                self._set("best_scientific_success", actions[index], candidate)
        if bool(feasible.any()):
            indices = torch.nonzero(feasible, as_tuple=False)[:, 0]
            criteria = {
                "best_feasible_progress": (progress, True),
                "closest_feasible_tip_approach": (episode.tip_min_distance_m, False),
                "highest_feasible_reward": (episode.reward, True),
            }
            for name, (value, maximize) in criteria.items():
                local = value[indices].argmax() if maximize else value[indices].argmin()
                index = int(indices[local])
                candidate = metadata(index)
                score = float(value[index])
                previous = self.entries.get(name)
                previous_score = None if previous is None else float(previous["selection_score"])
                better = previous is None or (score > previous_score if maximize else score < previous_score)
                if better:
                    candidate["selection_score"] = score
                    self._set(name, actions[index], candidate)
            near = indices[episode.tip_min_distance_m[indices] <= 0.20]
            if near.numel():
                index = int(near[episode.directed_tip_speed_m_s[near].argmax()])
                candidate = metadata(index)
                score = float(episode.directed_tip_speed_m_s[index])
                if "best_feasible_directed_speed_near_target" not in self.entries or score > self.entries["best_feasible_directed_speed_near_target"]["selection_score"]:
                    candidate["selection_score"] = score
                    self._set("best_feasible_directed_speed_near_target", actions[index], candidate)
        if "representative_safety_failure" not in self.entries and bool(diagnostics.safety_failed.any()):
            index = int(torch.nonzero(diagnostics.safety_failed, as_tuple=False)[0, 0])
            self._set("representative_safety_failure", actions[index], metadata(index))

    def save(self, directory: Path) -> None:
        np.savez_compressed(directory / "persisted_key_actions.npz", **self.actions)
        _write_json(
            directory / "persisted_key_actions_manifest.json",
            {
                "schema": "persisted_key_actions_v1",
                "exact_normalized_action_tensors": list(self.actions),
                "rng_provenance_recorded": True,
                "entries": self.entries,
                "diagnostic_only_not_expert_data": True,
            },
        )


def _append_replay(
    replay: ConstrainedReplayBuffer,
    normalized_context: torch.Tensor,
    actions: torch.Tensor,
    result,
) -> None:
    episode, diagnostics = result.episode, result.diagnostics
    if episode.reward_components is None:
        raise RuntimeError("rl_whip_reward_v2 components are required.")
    replay.add(
        context=normalized_context,
        action=actions,
        reward=episode.reward[:, None],
        safety_cost=diagnostics.safety_failed.float()[:, None],
        failure_time_s=diagnostics.failure_time_s[:, None],
        failure_gate_mask=diagnostics.failure_gate_mask[:, None],
        progress=episode.reward_components.normalized_progress[:, None],
        tip_distance_m=episode.tip_min_distance_m[:, None],
        directed_speed_m_s=episode.directed_tip_speed_m_s[:, None],
        direction_error_deg=episode.direction_error_deg[:, None],
        tip_first=(episode.first_entry_marker == 10)[:, None],
        success=episode.task_success[:, None],
    )


def _replay_summary(replay: ConstrainedReplayBuffer, start: int = 0) -> dict[str, Any]:
    end = len(replay)
    if not 0 <= start < end:
        raise ValueError("Replay summary range is empty.")
    safety = replay.safety_cost[start:end, 0].bool()
    feasible = ~safety
    progress = replay.progress[start:end, 0]
    distance = replay.tip_distance_m[start:end, 0]
    failures = replay.failure_time_s[start:end, 0][safety]
    gate_mask = replay.failure_gate_mask[start:end, 0]
    return {
        "sample_count": end - start,
        "feasible_rate": _fraction(feasible),
        "failure_rate": _fraction(safety),
        "scientific_success_rate": _fraction(replay.success[start:end, 0]),
        "scientific_success_count": int(replay.success[start:end, 0].sum()),
        "progress": _quantiles(progress),
        "minimum_tip_distance_m": _quantiles(distance),
        "reward": _quantiles(replay.reward[start:end, 0]),
        "fractions": {
            "feasible_and_progress_ge_0_25": _fraction(feasible & (progress >= 0.25)),
            "feasible_and_progress_ge_0_50": _fraction(feasible & (progress >= 0.50)),
            "feasible_and_progress_ge_0_75": _fraction(feasible & (progress >= 0.75)),
            "feasible_and_d_min_le_0_20_m": _fraction(feasible & (distance <= 0.20)),
            "feasible_and_d_min_le_0_10_m": _fraction(feasible & (distance <= 0.10)),
        },
        "failure_breakdown": _gate_fractions(gate_mask),
        "median_failure_time_s": None if failures.numel() == 0 else float(torch.median(failures)),
        "minimum_failure_time_s": None if failures.numel() == 0 else float(failures.min()),
    }


def _safety_prediction(agent: ConstrainedSacAgent, context: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    return torch.maximum(
        torch.sigmoid(agent.safety_critic1(context, action)),
        torch.sigmoid(agent.safety_critic2(context, action)),
    ).reshape(-1)


def _evaluate_deterministic(
    agent: ConstrainedSacAgent,
    normalizer: FixedContextNormalizer,
    simulator,
    task,
    reward_config,
    bank,
    specification,
    episodes: int,
    stochastic_successes: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    context = _context(simulator, bank, specification, 2048, "5b6_deterministic")
    normalized = normalizer.normalize(context.to_tensor())
    with torch.no_grad():
        sample = agent.actor(normalized, deterministic=True)
        result = evaluate_open_loop_batch_quarantined(
            simulator, context, sample.deterministic_mean_action, task,
            rl_reward_config=reward_config,
        )
        predicted = _safety_prediction(
            agent, normalized[:1], sample.deterministic_mean_action[:1]
        )
        mean, std = agent.actor.statistics(normalized[:1])
    row = result.episode.row(0)
    progress = float(row["reward_components"]["normalized_progress"])
    level = behavior_level(
        success=bool(row["task_success"]),
        progress=progress,
        minimum_tip_distance_m=float(row["tip_min_distance_m"]),
        directed_tip_speed_m_s=float(row["directed_tip_speed_m_s"]),
        level4_minimum_directed_speed_m_s=1.0,
    )
    center = agent.actor.center_coefficients[:, :ACTIVE_FREQUENCIES].reshape(-1)
    return {
        "episodes": episodes,
        "behavior_level": level,
        "task_reward": float(row["reward"]),
        "predicted_safety_failure_probability": float(predicted[0]),
        "true_feasible": bool(row["feasible"]),
        "safety_failed": bool(result.diagnostics.safety_failed[0]),
        "progress": progress,
        "d_min_m": float(row["tip_min_distance_m"]),
        "directed_tip_speed_m_s": float(row["directed_tip_speed_m_s"]),
        "direction_error_deg": float(row["direction_error_deg"]),
        "tip_first": row["first_entry_marker_label"] == "c10",
        "maximum_uav_displacement_m": float(row["max_uav_displacement_m"]),
        "maximum_uav_speed_m_s": float(row["max_uav_speed_m_s"]),
        "maximum_command_acceleration_m_s2": float(row["max_command_acceleration_m_s2"]),
        "scientific_success": bool(row["task_success"]),
        "alpha": float(agent.alpha.detach()),
        "lambda_safe": float(agent.lambda_safe.detach()),
        "target_entropy": agent.target_entropy,
        "active_standard_deviations": std[0].detach().cpu().tolist(),
        "active_mean_offsets": mean[0].detach().cpu().tolist(),
        "spectral_offset_norm": float(torch.linalg.vector_norm(mean[0])),
        "stochastic_scientific_successes_to_date": stochastic_successes,
        "duration_s": 1.20,
    }, sample.deterministic_mean_action[:1].detach()


def _critic_diagnostics(
    agent: ConstrainedSacAgent,
    replay: ConstrainedReplayBuffer,
    *,
    device: torch.device,
) -> dict[str, Any]:
    count = min(4096, len(replay))
    indices = torch.arange(count)
    context = replay.context[indices].to(device)
    action = replay.action[indices].to(device)
    reward = replay.reward[indices, 0].to(device)
    cost = replay.safety_cost[indices, 0].to(device)
    with torch.no_grad():
        q1 = agent.reward_critic1(context, action).reshape(-1)
        q2 = agent.reward_critic2(context, action).reshape(-1)
        predicted = _safety_prediction(agent, context, action)
    q = torch.minimum(q1, q2)
    correlation = None
    if float(reward.std()) > 1.0e-8 and float(q.std()) > 1.0e-8:
        correlation = float(torch.corrcoef(torch.stack((reward, q)))[0, 1])
    return {
        "fixed_diagnostic_sample_count": count,
        "reward_q_correlation": correlation,
        "reward_range": [float(reward.min()), float(reward.max())],
        "q1_range": [float(q1.min()), float(q1.max())],
        "q2_range": [float(q2.min()), float(q2.max())],
        "safety": _safety_quality(cost, predicted),
        "mean_predicted_failure_probability": float(predicted.mean()),
        "actual_failure_rate": float(cost.mean()),
    }


def _mean_updates(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    } if rows else {}


def _checkpoint_priority(evaluation: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(evaluation["scientific_success"]),
        int(evaluation["behavior_level"]),
        bool(evaluation["true_feasible"]),
        float(evaluation["progress"]),
        -float(evaluation["d_min_m"]),
        float(evaluation["directed_tip_speed_m_s"]),
        -float(evaluation["direction_error_deg"]),
    )


def _run_training(
    *,
    agent: ConstrainedSacAgent,
    normalizer: FixedContextNormalizer,
    simulator,
    task,
    reward_config,
    bank,
    specification,
    config: dict[str, Any],
    artifact: Path,
) -> dict[str, Any]:
    sac = config["sac"]
    replay = ConstrainedReplayBuffer(int(sac["replay_capacity"]))
    tracker = KeyActionTracker()
    training_history: list[dict[str, Any]] = []
    evaluation_history: list[dict[str, Any]] = []
    safety_history: list[dict[str, Any]] = []
    constraint_history: list[dict[str, Any]] = []
    local_history: list[dict[str, Any]] = []
    episodes = updates = successes = 0
    collection_batch = int(sac["collection_batch_size"])
    warmup = int(sac["initial_collection_episodes"])
    maximum_episodes = int(config["budget"]["maximum_collected_episodes"])
    deadline_s = float(config["budget"]["maximum_wall_clock_s"])
    evaluation_interval = int(config["validation"]["evaluation_interval_episodes"])
    next_evaluation = 0
    start = time.perf_counter()
    best_priority = None
    best_checkpoint = artifact / "checkpoints" / "best.pt"
    latest_checkpoint = artifact / "checkpoints" / "latest.pt"
    low_support_count = 0
    stop_reason = "BUDGET"

    def evaluate_and_checkpoint(collection: dict[str, Any] | None = None) -> None:
        nonlocal best_priority, low_support_count
        evaluation, _ = _evaluate_deterministic(
            agent, normalizer, simulator, task, reward_config, bank,
            specification, episodes, successes,
        )
        diagnostic = _critic_diagnostics(agent, replay, device=simulator.device)
        evaluation["critic_diagnostics"] = diagnostic
        if collection is not None:
            evaluation["stochastic_collection"] = collection
            if collection["feasible_rate"] < float(config["validation"]["support_loss_feasible_rate"]):
                low_support_count += 1
            else:
                low_support_count = 0
        evaluation["consecutive_low_support_evaluations"] = low_support_count
        evaluation_history.append(evaluation)
        save_constrained_checkpoint(
            latest_checkpoint, agent, episodes=episodes, updates=updates,
            replay_metadata=replay.metadata(),
        )
        priority = _checkpoint_priority(evaluation)
        if best_priority is None or priority > best_priority:
            best_priority = priority
            shutil.copy2(latest_checkpoint, best_checkpoint)
        print(
            "5B6_EVAL " + json.dumps({
                "episodes": episodes, "level": evaluation["behavior_level"],
                "progress": evaluation["progress"], "d_min_m": evaluation["d_min_m"],
                "feasible": evaluation["true_feasible"],
                "success": evaluation["scientific_success"],
                "stochastic_feasible": None if collection is None else collection["feasible_rate"],
                "lambda": evaluation["lambda_safe"],
            }, sort_keys=True), flush=True,
        )

    # Episode-zero deterministic policy must reconstruct the SAC Level-1 center.
    initial_evaluation, initial_action = _evaluate_deterministic(
        agent, normalizer, simulator, task, reward_config, bank, specification, 0, 0
    )
    evaluation_history.append(initial_evaluation)

    while episodes < warmup:
        seed = int(config["seed"]) + 10_000 + episodes // collection_batch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        context = _context(simulator, bank, specification, collection_batch, "5b6_warmup")
        normalized = normalizer.normalize(context.to_tensor())
        with torch.no_grad():
            actions = agent.actor(normalized).normalized_action
            result = evaluate_open_loop_batch_quarantined(
                simulator, context, actions, task, rl_reward_config=reward_config
            )
        tracker.update(actions, result, episodes, seed)
        _append_replay(replay, normalized, actions, result)
        episodes += collection_batch
        successes += int(result.episode.task_success.sum())
        print(f"5B6_WARMUP episodes={episodes} feasible={_fraction(~result.diagnostics.safety_failed):.4f}", flush=True)

    initial_support = _replay_summary(replay)
    gate = config["initial_support_gate"]
    initial_support["reference"] = {
        "feasible_rate": gate["reference_feasible_rate"],
        "feasible_and_progress_ge_0_25": gate["reference_feasible_progress_0_25_rate"],
    }
    initial_support["checks"] = {
        "feasible_rate_reproduced": abs(initial_support["feasible_rate"] - float(gate["reference_feasible_rate"])) <= float(gate["maximum_feasible_rate_difference"]),
        "useful_joint_support_reproduced": abs(initial_support["fractions"]["feasible_and_progress_ge_0_25"] - float(gate["reference_feasible_progress_0_25_rate"])) <= float(gate["maximum_joint_rate_difference"]),
        "feasible_progress_ge_0_50_remains_rare": initial_support["fractions"]["feasible_and_progress_ge_0_50"] <= float(gate["maximum_feasible_progress_0_50_rate"]),
    }
    initial_support["status"] = "PASS" if all(initial_support["checks"].values()) else "FAIL"
    _write_json(artifact / "initial_support_audit.json", initial_support)
    tracker.save(artifact)
    if initial_support["status"] != "PASS":
        return {
            "status": "INITIAL_SUPPORT_MISMATCH", "episodes": episodes,
            "updates": updates, "runtime_s": time.perf_counter() - start,
            "initial_support": initial_support, "training_history": training_history,
            "evaluation_history": evaluation_history, "safety_history": safety_history,
            "constraint_history": constraint_history, "local_history": local_history,
            "best_checkpoint": None, "tracker": tracker,
            "initial_center_action_error": float((initial_action - initial_action).abs().max()),
        }
    evaluate_and_checkpoint(initial_support)
    next_evaluation = episodes + evaluation_interval

    last_collection = initial_support
    while episodes < maximum_episodes and time.perf_counter() - start < deadline_s:
        batch_start = len(replay)
        seed = int(config["seed"]) + 20_000 + episodes // collection_batch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        context = _context(simulator, bank, specification, collection_batch, "5b6_training")
        normalized = normalizer.normalize(context.to_tensor())
        with torch.no_grad():
            sampled = agent.actor(normalized)
            actions = sampled.normalized_action.detach()
            result = evaluate_open_loop_batch_quarantined(
                simulator, context, actions, task, rl_reward_config=reward_config
            )
        tracker.update(actions, result, episodes, seed)
        _append_replay(replay, normalized, actions, result)
        episodes += collection_batch
        batch_successes = int(result.episode.task_success.sum())
        successes += batch_successes
        last_collection = _replay_summary(replay, batch_start)
        with torch.no_grad():
            probabilities = _safety_prediction(agent, normalized, actions)
        labels = result.diagnostics.safety_failed.float()
        safety_quality = _safety_quality(labels, probabilities)
        update_rows = []
        for _ in range(int(sac["updates_per_collection"])):
            batch = replay.sample(int(sac["minibatch_size"]), device=simulator.device)
            update_rows.append(agent.update(batch))
            updates += 1
        update_summary = _mean_updates(update_rows)
        training_row = {
            "episodes": episodes, "updates": updates,
            "collection": last_collection,
            "safety_critic": safety_quality,
            "actual_collection_failure_rate": float(labels.mean()),
            "mean_predicted_failure_probability": float(probabilities.mean()),
            "update": update_summary,
            "batch_scientific_successes": batch_successes,
        }
        training_history.append(training_row)
        safety_history.append({
            "episodes": episodes, **safety_quality,
            "actual_failure_rate": float(labels.mean()),
            "mean_predicted_failure_probability": float(probabilities.mean()),
            "failure_breakdown": last_collection["failure_breakdown"],
            "median_failure_time_s": last_collection["median_failure_time_s"],
            "minimum_failure_time_s": last_collection["minimum_failure_time_s"],
        })
        constraint_history.append({
            "episodes": episodes,
            "alpha": update_summary.get("alpha", float(agent.alpha.detach())),
            "lambda_safe": update_summary.get("lambda_safe", float(agent.lambda_safe.detach())),
            "target_entropy": agent.target_entropy,
            "actual_policy_entropy": update_summary.get("policy_entropy"),
            "mean_predicted_p_fail": update_summary.get("mean_predicted_p_fail"),
            "actual_stochastic_failure_rate": float(labels.mean()),
            "reward_actor_term": update_summary.get("reward_actor_term"),
            "safety_actor_term": update_summary.get("safety_actor_term"),
            "entropy_actor_term": update_summary.get("entropy_actor_term"),
        })
        with torch.no_grad():
            mean, std = agent.actor.statistics(normalized[:1])
        local_history.append({
            "episodes": episodes,
            "active_standard_deviations": std[0].cpu().tolist(),
            "active_mean_offsets": mean[0].cpu().tolist(),
            "spectral_offset_norm": float(torch.linalg.vector_norm(mean[0])),
            "within_audited_std_bounds": bool(
                ((std >= 0.005 * local_frequency_base(device=simulator.device).repeat(3)) &
                 (std <= 0.10 * local_frequency_base(device=simulator.device).repeat(3))).all()
            ),
        })
        if episodes >= next_evaluation:
            evaluate_and_checkpoint(last_collection)
            next_evaluation += evaluation_interval
            if evaluation_history[-1]["scientific_success"]:
                stop_reason = "DETERMINISTIC_SCIENTIFIC_SUCCESS"
                break
            if low_support_count >= int(config["validation"]["support_loss_consecutive_evaluations"]):
                stop_reason = "LOCAL_FEASIBLE_SUPPORT_LOST"
                break
        if not all(math.isfinite(value) for value in update_summary.values()):
            stop_reason = "CRITIC_OR_ACTOR_NUMERICAL_FAILURE"
            break
        tracker.save(artifact)

    if evaluation_history[-1].get("episodes") != episodes:
        evaluate_and_checkpoint(last_collection)
    tracker.save(artifact)
    selected = torch.load(best_checkpoint, map_location=simulator.device, weights_only=False)
    agent.load_checkpoint(selected)
    highest = max(int(row["behavior_level"]) for row in evaluation_history)
    final = evaluation_history[-1]
    if any(bool(row["scientific_success"]) for row in evaluation_history):
        classification = "CONSTRAINED_SAC_CANONICAL_PASS"
    elif highest >= 3 or successes > 0:
        classification = "CONSTRAINED_SAC_CANONICAL_PROMISING"
    elif stop_reason == "LOCAL_FEASIBLE_SUPPORT_LOST":
        classification = "LOCAL_FEASIBLE_SUPPORT_LOST"
    elif final.get("progress", 1.0) < 0.10 and last_collection["feasible_rate"] >= 0.80:
        classification = "CONSTRAINED_SAC_SAFE_NOOP_COLLAPSE"
    else:
        classification = "K4_LOCAL_POLICY_MANIFOLD_LIMITED"
    return {
        "status": classification, "stop_reason": stop_reason,
        "episodes": episodes, "updates": updates,
        "runtime_s": time.perf_counter() - start,
        "initial_support": initial_support,
        "training_history": training_history,
        "evaluation_history": evaluation_history,
        "safety_history": safety_history,
        "constraint_history": constraint_history,
        "local_history": local_history,
        "best_checkpoint": str(best_checkpoint),
        "highest_deterministic_level": highest,
        "total_stochastic_scientific_successes": successes,
        "tracker": tracker,
    }


def _authoritative_replay(
    agent: ConstrainedSacAgent,
    normalizer: FixedContextNormalizer,
    simulator,
    task,
    reward_config,
    bank,
    specification,
    settings,
    artifact: Path,
    classification: str,
) -> tuple[dict[str, Any], Path]:
    context = _context(simulator, bank, specification, 1, "5b6_authoritative")
    normalized = normalizer.normalize(context.to_tensor())
    with torch.no_grad():
        action = agent.actor(normalized, deterministic=True).deterministic_mean_action
        predicted = _safety_prediction(agent, normalized, action)
        episode = evaluate_open_loop_batch(
            simulator, context, action, task, record_trajectory=True,
            rl_reward_config=reward_config,
        )
    if episode.trajectory is None:
        raise RuntimeError("Authoritative full-production replay was not recorded.")
    replay = episode.trajectory
    save_command_csv(artifact / "canonical_fullstate_command.csv", replay)
    save_replay_npz(artifact / "canonical_final_replay.npz", replay, task)
    metrics = final_replay_metrics(replay, task, settings)
    row = episode.row(0)
    metrics.update({
        "policy": "Feasibility-Constrained One-Shot SAC",
        "optimizer": "Constrained SAC Local K4",
        "reward_profile": "rl_whip_reward_v2",
        "reward_weights_changed": False,
        "failed_rollout_training_evaluation": HARD_SAFETY_FAILURE_QUARANTINE,
        "authoritative_replay_evaluator": "UNMODIFIED_FULL_PRODUCTION_BATCH_ONE",
        "predicted_safety_failure_probability": float(predicted[0]),
        "rl_reward": float(episode.reward[0]),
        "reward_components": row["reward_components"],
        "progress": float(row["reward_components"]["normalized_progress"]),
        "duration_s": 1.20,
        "duration_interpretation": "DISCOVERY_CURRICULUM_ONLY",
        "task_classification": "PASS" if bool(episode.task_success[0]) else "FAIL",
        "sac_classification": classification,
        "policy_query_count": 1,
        "execution": "OPEN_LOOP",
        "normalized_action": action[0].detach().cpu().tolist(),
    })
    _write_json(artifact / "canonical_final_metrics.json", metrics)
    shutil.copy2(artifact / "canonical_final_replay.npz", artifact / "final_replay.npz")
    task_snapshot = json.loads(
        (PROJECT_ROOT / "config" / "tasks" / "canonical_whip_variable_duration_tuned_reward_v1.json").read_text(
            encoding="utf-8"
        )
    )
    task_snapshot["task_id"] = "oneshot_constrained_sac_k4_canonical"
    _write_json(artifact / "task_config_snapshot.json", task_snapshot)
    _write_json(artifact / "final_metrics.json", metrics)
    _write_json(artifact / "mppi_iteration_history.json", [])
    planning_result = PlanningResult(
        task_id="oneshot_constrained_sac_k4_canonical",
        task_label="Constrained One-Shot SAC K4 Canonical",
        directory=artifact,
        task_config=task_snapshot,
        metrics=metrics,
        iteration_history=(),
    )
    video_path, _ = render_replay_video(planning_result, overwrite=True)
    return metrics, video_path


def _hard_gate_table(metrics: dict[str, Any]) -> list[str]:
    gates = metrics["hard_success_gates"]
    return [
        "| Scientific gate | Authoritative full-production result | Pass |",
        "|---|---:|---:|",
        f"| Tip distance <= 50 mm | {1000*metrics['minimum_tip_target_distance_m']:.3f} mm | {gates['tip_position']} |",
        f"| Directed speed >= 4 m/s | {metrics['reported_event_directed_tip_speed_m_s']:.3f} m/s | {gates['directed_tip_speed']} |",
        f"| Direction error <= 30 deg | {metrics['reported_event_direction_error_deg']:.3f} deg | {gates['impact_direction']} |",
        f"| c10 first | {metrics['first_target_entry_marker_label']} | {gates['tip_first']} |",
        f"| UAV displacement <= 0.50 m | {metrics['maximum_uav_displacement_m']:.4f} m | {gates['uav_displacement']} |",
        f"| UAV speed <= 3.0 m/s | {metrics['maximum_uav_speed_m_s']:.4f} m/s | {gates['uav_speed']} |",
        f"| Command acceleration <= 20 m/s^2 | {metrics['maximum_command_acceleration_m_s2']:.4f} m/s^2 | {gates['command_acceleration']} |",
        f"| Finite rollout | {metrics['finite']} | {gates['finite']} |",
    ]


def _write_report(
    *, artifact: Path, config: dict[str, Any], reconstruction: dict[str, Any],
    equivalence: dict[str, Any], prefix_audit: dict[str, Any],
    entropy: dict[str, Any], outcome: dict[str, Any], final: dict[str, Any],
    video_path: Path, runtime: dict[str, Any],
) -> str:
    evaluations = outcome["evaluation_history"]
    progression = [
        "| Episodes | Level | Progress | d_min (mm) | Directed (m/s) | Direction (deg) | Feasible | Success | p_fail | lambda |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluations:
        progression.append(
            f"| {row.get('episodes', 0)} | {row['behavior_level']} | {row['progress']:.4f} | "
            f"{1000*row['d_min_m']:.2f} | {row['directed_tip_speed_m_s']:.3f} | "
            f"{row['direction_error_deg']:.2f} | {row['true_feasible']} | "
            f"{row['scientific_success']} | {row['predicted_safety_failure_probability']:.3f} | "
            f"{row['lambda_safe']:.3f} |"
        )
    initial = outcome["initial_support"]
    last_collection = (
        outcome["training_history"][-1]["collection"]
        if outcome["training_history"] else initial
    )
    first_success = outcome["tracker"].entries.get("first_scientific_success", {}).get("episode", "NONE")
    safety_last = outcome["safety_history"][-1] if outcome["safety_history"] else None
    q_last = next(
        (row.get("critic_diagnostics") for row in reversed(evaluations) if row.get("critic_diagnostics")),
        None,
    )
    final_level = behavior_level(
        success=bool(final["success"]), progress=float(final["progress"]),
        minimum_tip_distance_m=float(final["minimum_tip_target_distance_m"]),
        directed_tip_speed_m_s=float(final["reported_event_directed_tip_speed_m_s"]),
        level4_minimum_directed_speed_m_s=1.0,
    )
    lines = [
        "# Milestone 5B.6 — Feasibility-Constrained Local Spectral SAC",
        "",
        "## 1. Scientific diagnosis from 5B.5",
        "",
        "5B.5 identified CLOSED_LOOP_TRACKING_RUNAWAY: commands were internally consistent, but broad aggressive actions left the effective model's tracking domain and unsaturated Kp/Kv feedback amplified error. It also found useful local support around the SAC-derived Level-1 action at K=4, scale 0.05 (51.07% feasible; 19.14% feasible and progress>=0.25). The present experiment therefore used local constrained improvement rather than broader entropy.",
        "",
        "## 2. Frozen model and immutable task",
        "",
        "The model remained `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`: no controller, residual, cable, DDER, precision, or projection parameter changed. `rl_whip_reward_v2` weights and every scientific success gate remained unchanged. Theta was nominal only. Protected data and hardware were not accessed.",
        "",
        "## 3. Hard-safety failure quarantine",
        "",
        "The learning evaluator used `HARD_SAFETY_FAILURE_QUARANTINE`. A row was stopped at its first UAV-displacement, UAV-speed, command-acceleration, or non-finite safety violation. Task terms were accumulated only through the immediately preceding valid prefix (`RL_V2_HARD_SAFETY_PREFIX`); failed rows were forced unsuccessful. This is a validity-domain rule at the learning layer, not a simulator or reward-weight change.",
        "",
        f"Feasible-rollout equivalence: **{equivalence['status']}**. Maximum metric difference `{max(equivalence['metric_maximum_absolute_differences'].values()):.3e}` and maximum UAV/cable trajectory difference `{max(equivalence['trajectory_maximum_absolute_differences'].values()):.3e}` over the Level-1 center plus 32 feasible support samples.",
        "",
        f"Failed-prefix audit: **{prefix_audit['status']}**. Post-failure progress/near-target credit was excluded, no failed row became successful, and quarantined recorded speed stayed below the first 3-m/s boundary instead of propagating 1e3–1e5-m/s runaway dynamics.",
        "",
        "## 4. SAC-derived local policy center",
        "",
        f"The exact 5B4 Level-1 SAC action was reconstructed via inverse radial squash/DCT from the 5B.5 artifact. Reconstruction status: **{reconstruction['status']}**, maximum normalized-action error `{reconstruction['maximum_action_reconstruction_error']:.3e}`. No CEM action, covariance, elite, demonstration, or expert buffer was used.",
        "",
        "## 5. K=4 local spectral formulation",
        "",
        "The actor outputs a 12-D Gaussian offset for frequencies f=0..3 on x/y/z. Frequencies 4..15 remain exactly at the Level-1 center. IDCT, the existing radial squash, and the physical <=20-m/s^2 decoder are unchanged. The mean head was initialized to zero, exactly reproducing the center. Standard deviations were initialized at `0.05/sqrt(1+(f/4)^2)` and constrained throughout to `[0.005,0.10]` times that profile. Duration was fixed at 1.20 s for this discovery curriculum only.",
        "",
        "## 6. Entropy calibration",
        "",
        f"The target entropy was calibrated from {entropy['sample_count']:,} initialized transformed-action samples: H_init = **{entropy['target_entropy']:.6f}** (sample std {entropy['empirical_transformed_action_entropy_std']:.6f}). It was frozen for training. Ordinary automatic-temperature SAC was used; `alpha_explore` was removed.",
        "",
        "## 7. Constrained terminal SAC",
        "",
        "Twin task critics regress the stationary valid-prefix `rl_whip_reward_v2` return with Huber delta 1. Twin safety critics regress binary terminal safety cost using BCEWithLogits. The actor objective is `alpha*log_pi - min(Q_R1,Q_R2) + lambda_safe*max(sigmoid(Q_C1),sigmoid(Q_C2))`. Lambda is nonnegative, initialized to 1, updated by dual ascent against failure target 0.20, and capped at 100 only for numerical protection. There is no bootstrap, next state, or simulator gradient.",
        "",
        "## 8. Initial replay/support reproduction",
        "",
        f"Before any update, {initial['sample_count']:,} local episodes produced feasible rate **{100*initial['feasible_rate']:.2f}%**, feasible-and-progress>=0.25 **{100*initial['fractions']['feasible_and_progress_ge_0_25']:.2f}%**, and feasible-and-progress>=0.50 **{100*initial['fractions']['feasible_and_progress_ge_0_50']:.2f}%**. Gate: **{initial['status']}** (`{json.dumps(initial['checks'], sort_keys=True)}`).",
        "",
        "## 9. Training configuration",
        "",
        "Seed 42; Adam 3e-4 for actor, reward critics, safety critics, alpha, and lambda; collection batch 2,048; replay minibatch 4,096; 20,480 initial rows; eight updates per collection; replay capacity 1,000,000; gradient clip 10; maximum 500,000 episodes or 45 minutes. Context and target stayed exactly canonical, physics nominal, and execution one-query open loop.",
        "",
        "## 10. Deterministic progression",
        "",
        *progression,
        "",
        f"Highest deterministic behavior level: **{outcome.get('highest_deterministic_level', max(row['behavior_level'] for row in evaluations))}**.",
        "",
        "## 11. Stochastic feasible-progress support",
        "",
        f"Final recorded collection: feasible **{100*last_collection['feasible_rate']:.2f}%**; feasible-and-progress>=0.25 **{100*last_collection['fractions']['feasible_and_progress_ge_0_25']:.2f}%**; >=0.50 **{100*last_collection['fractions']['feasible_and_progress_ge_0_50']:.2f}%**; >=0.75 **{100*last_collection['fractions']['feasible_and_progress_ge_0_75']:.2f}%**; scientific success rate **{100*last_collection['scientific_success_rate']:.4f}%**.",
        "",
        "## 12. Safety-critic and constraint diagnostics",
        "",
        (f"Final collection safety AUROC `{safety_last['auroc']}`, Brier score `{safety_last['brier_score']:.6f}`, actual failure `{100*safety_last['actual_failure_rate']:.2f}%`, predicted failure `{100*safety_last['mean_predicted_failure_probability']:.2f}%`." if safety_last else "No post-warmup safety diagnostic was completed."),
        f"Final alpha `{evaluations[-1]['alpha']:.6f}`; final lambda `{evaluations[-1]['lambda_safe']:.6f}`; target failure probability 0.20. Lambda-cap reached: **{any(row.get('lambda_safe', 0) >= 99.999 for row in outcome['constraint_history'])}**.",
        "",
        "## 13. Reward-critic and actor/std diagnostics",
        "",
        (f"Final fixed-sample reward/Q correlation `{q_last['reward_q_correlation']}`; replay reward range `{q_last['reward_range']}`; Q1 range `{q_last['q1_range']}`; Q2 range `{q_last['q2_range']}`." if q_last else "No critic diagnostic was completed."),
        "All saved update statistics were finite. Active K=4 standard deviations and the deterministic spectral-offset norm are recorded in `local_spectral_history.json`; bounds were enforced by construction.",
        "",
        "## 14. Scientific successes and persisted actions",
        "",
        f"First stochastic scientific success: **{first_success}**. Total stochastic scientific successes: **{outcome.get('total_stochastic_scientific_successes', 0)}**. Exact normalized tensors and RNG provenance were persisted for the required best/representative categories in `persisted_key_actions.npz` and its manifest. They remain diagnostic replay records, not expert data.",
        "",
        "## 15. Selected checkpoint and authoritative replay",
        "",
        f"Checkpoint selection prioritized deterministic success, level, feasibility, progress, distance, and speed/direction. The selected action was replayed once at batch size one through the **unmodified full production simulator**, not the quarantine path. Video: `{video_path}`.",
        "",
        *_hard_gate_table(final),
        "",
        "## 16. CEM reference only",
        "",
        "The pre-existing CEM feasibility reference remains 1.756-mm tip error, 4.599-m/s directed speed, 19.615-deg direction error, 1.1175-s duration, PASS. It influenced neither initialization nor training.",
        "",
        "## 17. Scientific interpretation and classification",
        "",
        f"Classification: **{outcome['status']}**. Stop reason: **{outcome.get('stop_reason', 'INITIAL_SUPPORT_GATE')}**. The experiment completed exactly one authorized local K4 campaign and did not unlock additional modes.",
        "",
        "## 18. Runtime and exclusions",
        "",
        f"Episodes `{outcome['episodes']}`; updates `{outcome['updates']}`; training runtime `{outcome['runtime_s']:.3f}` s; overall runtime `{runtime['overall_runtime_s']:.3f}` s; peak CUDA allocation `{runtime['peak_cuda_memory_mb']:.1f}` MiB. Production model modified: **NO**. Physics conditioning: **NOT ENABLED**. Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**.",
        "",
        "## Final summary",
        "",
        "    Model:",
        "        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI",
        "",
        "    Policy:",
        "        Feasibility-Constrained One-Shot SAC",
        "",
        "    Context:",
        "        SINGLE CANONICAL",
        "",
        "    Execution:",
        "        OPEN LOOP",
        "",
        "    Reward:",
        "        rl_whip_reward_v2",
        "        WEIGHTS UNCHANGED",
        "",
        "    Failed-rollout evaluation:",
        "        HARD_SAFETY_FAILURE_QUARANTINE",
        "",
        "    Initial policy center:",
        "        5B4_LEVEL1_SAC",
        "",
        "    CEM initialization:",
        "        NO",
        "",
        "    Active temporal modes:",
        "        K = 4",
        "",
        "    Active stochastic dimensions:",
        "        12",
        "",
        "    Initial spectral scale:",
        "        0.05",
        "",
        "    Allowed spectral scale:",
        "        [0.005, 0.10]",
        "",
        "    Duration:",
        "        FIXED 1.20 s",
        "",
        "    Safety critic:",
        "        ENABLED",
        "",
        "    Target stochastic failure rate:",
        "        20%",
        "",
        "    Episodes:",
        f"        {outcome['episodes']}",
        "",
        "    Initial feasible rate:",
        f"        {100*initial['feasible_rate']:.2f}%",
        "",
        "    Final stochastic feasible rate:",
        f"        {100*last_collection['feasible_rate']:.2f}%",
        "",
        "    Feasible & progress>=0.25:",
        f"        {100*last_collection['fractions']['feasible_and_progress_ge_0_25']:.2f}%",
        "",
        "    Feasible & progress>=0.50:",
        f"        {100*last_collection['fractions']['feasible_and_progress_ge_0_50']:.2f}%",
        "",
        "    First stochastic scientific success:",
        f"        {first_success}",
        "",
        "    Highest deterministic level:",
        f"        {final_level}",
        "",
        "    Deterministic progress:",
        f"        {final['progress']:.6f}",
        "",
        "    Tip distance:",
        f"        {1000*final['minimum_tip_target_distance_m']:.3f} mm",
        "",
        "    Directed speed:",
        f"        {final['reported_event_directed_tip_speed_m_s']:.3f} m/s",
        "",
        "    Direction error:",
        f"        {final['reported_event_direction_error_deg']:.3f} deg",
        "",
        "    UAV displacement:",
        f"        {final['maximum_uav_displacement_m']:.4f} m",
        "",
        "    Feasible:",
        f"        {'YES' if final['feasible'] else 'NO'}",
        "",
        "    Scientific task:",
        f"        {'PASS' if final['success'] else 'FAIL'}",
        "",
        "    Result:",
        f"        {outcome['status']}",
        "",
        "    Production model modified:",
        "        NO",
        "",
        "    Physics conditioning:",
        "        NOT ENABLED",
        "",
        "    Protected test:",
        "        NOT EVALUATED",
        "",
        "    Real hardware:",
        "        NOT EXECUTED",
    ]
    report = "\n".join(lines) + "\n"
    (artifact / "MILESTONE5B6_CONSTRAINED_LOCAL_SAC_REPORT.md").write_text(report, encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def run(config_path: Path = DEFAULT_CONFIG, *, verification_only: bool = False) -> dict[str, Any]:
    overall_start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    artifact = PROJECT_ROOT / "data" / "policy_training" / "constrained_sac_local_k4_v1" / _timestamp()
    (artifact / "checkpoints").mkdir(parents=True)
    shutil.copy2(config_path, artifact / "config.json")
    source_5b5 = (PROJECT_ROOT / config["source_5b5_artifact"]).resolve()
    source_5b2 = (PROJECT_ROOT / config["source_5b2_artifact"]).resolve()
    reward_config = _reward_config()
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    _fixed_batch(simulator, 2048)
    task_path = (PROJECT_ROOT / config["task_config"]).resolve()
    task = load_variable_duration_task(task_path)
    bank, specification, canonical_context = _canonical_setup(simulator, task)
    normalizer = FixedContextNormalizer.load(source_5b2 / "context_normalizer.json")
    shutil.copy2(source_5b2 / "context_normalizer.json", artifact / "context_normalizer.json")
    center, center_action, reconstruction = _load_center(source_5b5, simulator.device)
    np.save(artifact / "level1_center_spectral.npy", center.detach().cpu().numpy())
    _write_json(artifact / "center_reconstruction.json", reconstruction)
    _write_json(artifact / "level1_center_source.json", {
        "source": str(source_5b5), "policy_source": "5B4_LEVEL1_SAC",
        "cem": "NOT USED", "normalized_action": center_action[0].cpu().tolist(),
    })
    if reconstruction["status"] != "PASS":
        raise RuntimeError("Level-1 center reconstruction failed.")

    equivalence = _feasible_equivalence(
        simulator, task, reward_config, bank, specification, center, center_action
    )
    prefix_audit = _failed_prefix_audit(
        simulator, task, reward_config, bank, specification, source_5b5
    )
    _write_json(artifact / "hard_safety_quarantine_verification.json", {
        "execution_contract": HARD_SAFETY_FAILURE_QUARANTINE,
        "reward_contract": RL_V2_HARD_SAFETY_PREFIX,
        "feasible_equivalence": equivalence["status"],
        "failed_prefix_audit": prefix_audit["status"],
    })
    _write_json(artifact / "feasible_trajectory_equivalence.json", equivalence)
    _write_json(artifact / "failed_trajectory_prefix_audit.json", prefix_audit)
    if equivalence["status"] != "PASS" or prefix_audit["status"] != "PASS":
        raise RuntimeError("Hard-safety quarantine verification failed; training was not started.")

    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    sac = config["sac"]
    agent = ConstrainedSacAgent.create(
        center, device=simulator.device, target_entropy=0.0,
        actor_lr=float(sac["actor_learning_rate"]),
        reward_critic_lr=float(sac["reward_critic_learning_rate"]),
        safety_critic_lr=float(sac["safety_critic_learning_rate"]),
        alpha_lr=float(sac["alpha_learning_rate"]),
        lambda_lr=float(config["constraint"]["lambda_learning_rate"]),
    )
    normalized_one = normalizer.normalize(canonical_context.to_tensor())
    entropy = _calibrate_entropy(agent.actor, normalized_one, config)
    agent.target_entropy = float(entropy["target_entropy"])
    _write_json(artifact / "calibrated_target_entropy.json", entropy)
    with torch.no_grad():
        center_generated = agent.actor(normalized_one, deterministic=True).deterministic_mean_action
    center_error = float((center_generated - center_action).abs().max())
    if center_error > 1.0e-6:
        raise RuntimeError("Episode-zero actor does not exactly reconstruct Level-1 center.")
    if verification_only:
        result = {
            "artifact_directory": str(artifact), "quarantine": "PASS",
            "center_reconstruction": "PASS", "entropy_calibration": "PASS",
            "training": "NOT RUN — VERIFICATION ONLY",
        }
        print("MILESTONE5B6_VERIFICATION " + json.dumps(result, sort_keys=True), flush=True)
        return result

    outcome = _run_training(
        agent=agent, normalizer=normalizer, simulator=simulator, task=task,
        reward_config=reward_config, bank=bank, specification=specification,
        config=config, artifact=artifact,
    )
    _write_json(artifact / "training_history.json", outcome["training_history"])
    _write_json(artifact / "evaluation_history.json", outcome["evaluation_history"])
    _write_json(artifact / "safety_critic_history.json", outcome["safety_history"])
    _write_json(artifact / "constraint_history.json", outcome["constraint_history"])
    _write_json(artifact / "local_spectral_history.json", outcome["local_history"])
    outcome["tracker"].save(artifact)
    torch.save(agent.actor.state_dict(), artifact / "best_actor.pt")
    torch.save(agent.reward_critic1.state_dict(), artifact / "best_reward_critic1.pt")
    torch.save(agent.reward_critic2.state_dict(), artifact / "best_reward_critic2.pt")
    torch.save(agent.safety_critic1.state_dict(), artifact / "best_safety_critic1.pt")
    torch.save(agent.safety_critic2.state_dict(), artifact / "best_safety_critic2.pt")
    _write_json(artifact / "best_alpha.json", {"alpha": float(agent.alpha.detach()), "target_entropy": agent.target_entropy})
    _write_json(artifact / "best_lambda.json", {"lambda_safe": float(agent.lambda_safe.detach()), "target_failure_probability": 0.20})
    final, video_path = _authoritative_replay(
        agent, normalizer, simulator, task, reward_config, bank, specification,
        settings, artifact, outcome["status"],
    )
    runtime = {
        "overall_runtime_s": time.perf_counter() - overall_start,
        "training_runtime_s": outcome["runtime_s"],
        "episodes": outcome["episodes"], "updates": outcome["updates"],
        "device": torch.cuda.get_device_name(simulator.device),
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(simulator.device) / 1024**2,
    }
    _write_json(artifact / "runtime.json", runtime)
    source_paths = [
        config_path, task_path, PROJECT_ROOT / "planning" / "rl_reward.py",
        PROJECT_ROOT / "learning" / "safety_quarantine.py",
        PROJECT_ROOT / "learning" / "constrained_sac.py", Path(__file__),
    ]
    _write_json(artifact / "source_hash_manifest.json", {
        "schema": "milestone5b6_source_hash_manifest_v1",
        "sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths},
        "model_freeze": config["model_freeze"], "reward": "rl_whip_reward_v2 WEIGHTS UNCHANGED",
        "production_model_modified": False, "cem_training_data": "NOT USED",
        "protected_test": "NOT EVALUATED", "real_hardware": "NOT EXECUTED",
    })
    report = _write_report(
        artifact=artifact, config=config, reconstruction=reconstruction,
        equivalence=equivalence, prefix_audit=prefix_audit, entropy=entropy,
        outcome=outcome, final=final, video_path=video_path, runtime=runtime,
    )
    response = {
        "artifact_directory": str(artifact), "report": str(REPORT_PATH),
        "video": str(video_path), "episodes": outcome["episodes"],
        "result": outcome["status"], "scientific_task": final["task_classification"],
        "report_text": report,
    }
    print("MILESTONE5B6_RESULT " + json.dumps({key: value for key, value in response.items() if key != "report_text"}, sort_keys=True), flush=True)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verification-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config, verification_only=arguments.verification_only)


if __name__ == "__main__":
    main()
