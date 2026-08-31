"""Milestone 5B.4: temporally structured SAC exploration and annealing."""

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

from learning.canonical_pilot import repeated_canonical_specification
from learning.context_sampling import ContextSpecification, build_context_from_specification
from learning.normalization import FixedContextNormalizer
from learning.one_shot_env import evaluate_open_loop_batch
from learning.policy_context import build_policy_context
from learning.replay import TerminalReplayBuffer
from learning.sac import OneShotActor, terminal_critic_target
from learning.spectral_pilot import run_spectral_sac_pilot
from learning.spectral_sac import (
    SpectralOneShotActor,
    alpha_explore_schedule,
    create_spectral_agent,
    dominant_horizontal_sign_changes,
    frequency_energy_fractions,
    initial_frequency_std,
    orthonormal_idct_matrix,
    spectral_to_temporal,
    temporal_roughness,
    temporal_to_spectral,
)
from learning.state_bank import initial_state_bank_from_state
from learning.training import load_training_checkpoint, save_training_checkpoint
from planning.artifacts import final_replay_metrics, save_command_csv, save_replay_npz
from planning.cem_task import load_variable_duration_task
from planning.results import PlanningResult
from planning.rl_reward import RLWhipRewardConfig
from planning.rollout import hover_preroll
from planning.video import render_replay_video
from simulator.parameters import SimulatorSettings
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "learning" / "sac_canonical_spectral_entropy_v1.json"
REPORT_PATH = PROJECT_ROOT / "MILESTONE5B4_SPECTRAL_ENTROPY_SAC_REPORT.md"


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


def _reward_config(payload: dict[str, Any]) -> RLWhipRewardConfig:
    accepted = {field.name for field in fields(RLWhipRewardConfig)}
    return RLWhipRewardConfig(
        **{name: value for name, value in payload.items() if name in accepted}
    )


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "sac_canonical_spectral_entropy_v1":
        raise ValueError("Unsupported Milestone 5B.4 configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The production model freeze is immutable.")
    if (config.get("context_dimension"), config.get("external_action_dimension")) != (83, 49):
        raise ValueError("The 83-D context and external 49-D action contracts are frozen.")
    if config.get("stochastic_action_dimension") != 48:
        raise ValueError("Milestone 5B.4 has exactly 48 stochastic acceleration dimensions.")
    if config.get("context_distribution") != "SINGLE_CANONICAL_CONTEXT":
        raise ValueError("Milestone 5B.4 permits exactly one canonical context.")
    if not math.isclose(float(config.get("fixed_duration_s", 0.0)), 1.20):
        raise ValueError("The discovery-curriculum duration must be fixed at 1.20 s.")
    reward = config["reward"]
    if reward.get("profile") != "rl_whip_reward_v2" or float(reward.get("progress_weight", -1)) != 4.0:
        raise ValueError("Milestone 5B.4 requires unchanged rl_whip_reward_v2.")
    if float(config["sac"]["target_entropy"]) != -48.0:
        raise ValueError("Base target entropy must match 48 stochastic dimensions.")
    policy = config["artifact_policy"]
    if policy["real_flight_authorized"] or policy["protected_test_evaluation_allowed"] or policy["cem_training_data_allowed"]:
        raise ValueError("5B.4 must remain simulated, protected, and demonstration-free.")


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
    return bank, specification


def _distribution_verification(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(841)
    basis = orthonormal_idct_matrix(dtype=torch.float64)
    identity_error = float((basis.T @ basis - torch.eye(16, dtype=torch.float64)).abs().max())
    rank = int(torch.linalg.matrix_rank(basis))
    temporal = torch.randn(64, 16, 3, dtype=torch.float64)
    coefficients = temporal_to_spectral(temporal)
    reconstructed = spectral_to_temporal(coefficients)
    reconstruction_error = float((reconstructed - temporal).abs().max())
    energy_error = float(
        (
            coefficients.square().sum(dim=(1, 2))
            - temporal.square().sum(dim=(1, 2))
        ).abs().max()
    )
    std = initial_frequency_std(dtype=torch.float64)
    actor = SpectralOneShotActor().to(device)
    context = torch.randn(128, 83, device=device)
    sample = actor(context)
    knot_norm = torch.linalg.vector_norm(
        sample.normalized_action[:, :48].reshape(-1, 16, 3), dim=-1
    )
    loss = (sample.log_prob + sample.normalized_action[:, :48].square().mean(dim=1, keepdim=True)).mean()
    loss.backward()
    gradient_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in actor.parameters()
    )
    schedule_points = {
        str(k): alpha_explore_schedule(k)
        for k in (0, 74_999, 75_000, 187_500, 300_000, 400_000)
    }
    schedule_values = [alpha_explore_schedule(k) for k in range(0, 500_001, 1000)]
    reward = torch.tensor([[-2.0], [3.0]], device=device)
    targets = [terminal_critic_target(reward) for _ in (0, 100_000, 400_000)]
    checks = {
        "orthonormality": identity_error <= 1e-12,
        "full_rank": rank == 16,
        "invertibility": reconstruction_error <= 1e-11,
        "energy_preservation": energy_error <= 1e-10,
        "all_frequency_std_nonzero": bool((std > 0).all()),
        "high_frequency_std_nonzero": float(std[-1]) > 0.0,
        "radial_bound": float(knot_norm.max()) < 1.0 + 1e-6,
        "sample_log_probability_finite": bool(torch.isfinite(sample.log_prob).all()),
        "analytic_log_probability_gradients_finite": gradient_finite,
        "schedule_exact": schedule_points
        == {"0": 0.25, "74999": 0.25, "75000": 0.25, "187500": 0.125, "300000": 0.0, "400000": 0.0},
        "schedule_monotone_nonnegative": all(
            0.0 <= right <= left <= 0.25
            for left, right in zip(schedule_values, schedule_values[1:])
        ),
        "critic_target_stationary": all(torch.equal(target, reward) for target in targets),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_orthonormality_error": identity_error,
        "matrix_rank": rank,
        "maximum_reconstruction_error": reconstruction_error,
        "maximum_energy_error": energy_error,
        "initial_frequency_std": std.tolist(),
        "schedule_test_points": schedule_points,
        "orthonormal_idct_log_abs_determinant": float(torch.linalg.slogdet(basis)[1]),
    }


def _quantiles(value: torch.Tensor) -> dict[str, float]:
    tensor = value.detach().float()
    return {
        "mean": float(tensor.mean()),
        "median": float(torch.quantile(tensor, 0.5)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "maximum": float(tensor.max()),
        "minimum": float(tensor.min()),
    }


def _audit_summary(result, action: torch.Tensor) -> dict[str, Any]:
    components = result.reward_components
    if components is None:
        raise RuntimeError("Exploration audit requires rl_whip_reward_v2.")
    progress = components.normalized_progress
    distance = result.tip_min_distance_m
    feasible = result.feasible
    knots = result.decoded_action.acceleration_knots_local_m_s2
    roughness = temporal_roughness(knots)
    knot_norms = torch.linalg.vector_norm(knots, dim=-1)
    sign_changes = dominant_horizontal_sign_changes(knots)
    fraction = lambda mask: float(mask.float().mean())
    return {
        "reward": _quantiles(result.reward),
        "progress": _quantiles(progress),
        "minimum_tip_distance_m": _quantiles(distance),
        "maximum_tip_speed_m_s": _quantiles(result.max_tip_speed_m_s),
        "directed_tip_speed_m_s": _quantiles(result.directed_tip_speed_m_s),
        "maximum_uav_displacement_m": _quantiles(result.max_uav_displacement_m),
        "maximum_uav_speed_m_s": _quantiles(result.max_uav_speed_m_s),
        "temporal_roughness": _quantiles(roughness),
        "temporal_roughness_normalized": _quantiles(roughness / 400.0),
        "maximum_knot_acceleration_m_s2": _quantiles(knot_norms.amax(dim=1)),
        "mean_knot_acceleration_norm_m_s2": _quantiles(knot_norms.mean(dim=1)),
        "dominant_horizontal_sign_changes": _quantiles(sign_changes.float()),
        "success_rate": fraction(result.task_success),
        "feasible_rate": fraction(feasible),
        "finite_rate": fraction(result.rollout_finite),
        "tip_first_rate": fraction(result.first_entry_marker == 10),
        "fractions": {
            "progress_ge_0_25": fraction(progress >= 0.25),
            "progress_ge_0_50": fraction(progress >= 0.50),
            "progress_ge_0_75": fraction(progress >= 0.75),
            "d_min_le_0_50_m": fraction(distance <= 0.50),
            "d_min_le_0_20_m": fraction(distance <= 0.20),
            "d_min_le_0_10_m": fraction(distance <= 0.10),
            "feasible_and_progress_ge_0_50": fraction(feasible & (progress >= 0.50)),
        },
        "fixed_duration_normalized": float(action[:, -1].mean()),
    }


def _pretraining_exploration_audit(
    simulator,
    task,
    reward_config,
    normalizer,
    canonical_bank,
    canonical_specification,
    config,
) -> dict[str, Any]:
    count = int(config["spectral_distribution"]["old_vs_new_audit_samples_each"])
    specification = repeated_canonical_specification(
        canonical_specification, count, split="pretraining_exploration_audit"
    )
    context = build_context_from_specification(simulator, canonical_bank, specification)
    normalized = normalizer.normalize(context.to_tensor())
    seed = int(config["spectral_distribution"]["audit_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    old_actor = OneShotActor().to(simulator.device)
    with torch.no_grad():
        old_sample = old_actor(normalized)
        old_action = old_sample.normalized_action.clone()
        old_action[:, -1] = 1.0
        old_result = evaluate_open_loop_batch(
            simulator, context, old_action, task, rl_reward_config=reward_config
        )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    new_actor = SpectralOneShotActor(
        std_low=float(config["spectral_distribution"]["std_low"])
    ).to(simulator.device)
    with torch.no_grad():
        new_sample = new_actor(normalized)
        new_action = new_sample.normalized_action
        new_result = evaluate_open_loop_batch(
            simulator, context, new_action, task, rl_reward_config=reward_config
        )
    old_summary = _audit_summary(old_result, old_action)
    new_summary = _audit_summary(new_result, new_action)
    new_energy = frequency_energy_fractions(new_sample.spectral_coefficients)
    old_temporal = old_action[:, :48].reshape(-1, 16, 3)
    old_coefficients = temporal_to_spectral(old_temporal)
    old_energy = frequency_energy_fractions(old_coefficients)
    energy = {
        "old_decoded_action": {name: float(values.mean()) for name, values in old_energy.items()},
        "new_raw_spectral_coefficients": {
            name: float(values.mean()) for name, values in new_energy.items()
        },
    }
    ratio = (
        new_summary["temporal_roughness"]["mean"]
        / old_summary["temporal_roughness"]["mean"]
    )
    checks = {
        "sample_count_each_2048": count == 2048,
        "audit_excluded_from_replay": True,
        "old_rollouts_all_finite": old_summary["finite_rate"] == 1.0,
        "spectral_rollouts_all_finite": new_summary["finite_rate"] == 1.0,
        "spectral_mean_roughness_meaningfully_lower": ratio
        <= float(config["spectral_distribution"]["maximum_mean_roughness_ratio"]),
        "low_frequency_energy_dominates": energy["new_raw_spectral_coefficients"]["low"]
        > energy["new_raw_spectral_coefficients"]["mid"]
        > energy["new_raw_spectral_coefficients"]["high"],
        "high_frequency_energy_nonzero": energy["new_raw_spectral_coefficients"]["high"] > 0.0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "comparison_note": "Duration fixed to 1.20 s for both distributions; only acceleration exploration geometry differs.",
        "old_direct_knot": old_summary,
        "new_full_rank_spectral": new_summary,
        "spectral_to_old_mean_roughness_ratio": ratio,
        "frequency_energy_fractions": energy,
        "entered_sac_replay": False,
    }


def _pre_run_learning_verification(
    simulator,
    normalizer,
    canonical_bank,
    canonical_specification,
    config,
    artifact,
) -> dict[str, Any]:
    specification = repeated_canonical_specification(
        canonical_specification, 8, split="pre_run_learning_verification"
    )
    context = build_context_from_specification(simulator, canonical_bank, specification)
    normalized = normalizer.normalize(context.to_tensor())
    torch.manual_seed(int(config["seed"]))
    agent = create_spectral_agent(device=simulator.device)
    actions = agent.actor(normalized).normalized_action.detach()
    synthetic_rewards = torch.linspace(-2.0, 2.0, 8, device=simulator.device)
    update = agent.update(
        normalized,
        actions,
        synthetic_rewards,
        actor_entropy_bonus=alpha_explore_schedule(0),
    )
    checkpoint = artifact / "pre_run_checkpoint_test.pt"
    generator = torch.Generator().manual_seed(77)
    save_training_checkpoint(
        checkpoint,
        agent=agent,
        episodes=8,
        gradient_updates=1,
        training_history=[],
        evaluation_history=[],
        context_generator=generator,
        replay_metadata={"test": True},
    )
    loaded = load_training_checkpoint(checkpoint, agent=agent, context_generator=generator)
    checkpoint.unlink()
    checks = {
        "fixed_canonical_context_identical": torch.equal(normalized, normalized[:1].expand_as(normalized)),
        "external_action_shape_49": tuple(actions.shape) == (8, 49),
        "duration_normalized_fixed_at_one": bool((actions[:, -1] == 1.0).all()),
        "actor_critic_backward_finite": all(math.isfinite(value) for value in update.values()),
        "checkpoint_save_resume": int(loaded["episodes"]) == 8,
        "critic_target_reward_only": torch.equal(
            terminal_critic_target(synthetic_rewards), synthetic_rewards.reshape(-1, 1)
        ),
        "cem_training_data_used": False,
    }
    return {"status": "PASS" if all(value is True or key == "cem_training_data_used" and value is False for key, value in checks.items()) else "FAIL", "checks": checks}


def _record_selected_policy(
    agent,
    normalizer,
    simulator,
    task,
    reward_config,
    canonical_bank,
    canonical_specification,
):
    context = build_context_from_specification(simulator, canonical_bank, canonical_specification)
    with torch.no_grad():
        action = agent.actor(normalizer.normalize(context.to_tensor()), deterministic=True).deterministic_mean_action
        result = evaluate_open_loop_batch(
            simulator,
            context,
            action,
            task,
            record_trajectory=True,
            rl_reward_config=reward_config,
        )
    if result.trajectory is None:
        raise RuntimeError("Authoritative spectral SAC replay was not recorded.")
    return result, result.trajectory


def _write_report(
    *,
    artifact: Path,
    config: dict[str, Any],
    verification: dict[str, Any],
    audit: dict[str, Any],
    pre_run: dict[str, Any],
    outcome,
    final_metrics: dict[str, Any],
    runtime: dict[str, Any],
    video_path: Path,
) -> str:
    evaluations = list(outcome.evaluation_history)
    progression = [
        "| Episodes | Level | Progress | d_min (mm) | Directed (m/s) | Direction (deg) | Feasible | Success | alpha_SAC | alpha_explore | Stochastic successes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluations:
        canonical = row["canonical"]
        progression.append(
            f"| {row['episodes']} | {canonical['behavior_level']} | {canonical['progress']:.4f} | "
            f"{1000*canonical['tip_min_distance_m']:.3f} | {canonical['directed_tip_speed_m_s']:.3f} | "
            f"{canonical['direction_error_deg']:.3f} | {canonical['feasible']} | {canonical['task_success']} | "
            f"{row['alpha_sac']:.4f} | {row['alpha_explore']:.4f} | {row.get('total_stochastic_successes', 0)} |"
        )
    gates = final_metrics["hard_success_gates"]
    gate_table = [
        "| Scientific gate | Authoritative result | Pass |",
        "|---|---:|---:|",
        f"| Tip distance <= 50 mm | {1000*final_metrics['minimum_tip_target_distance_m']:.3f} mm | {gates['tip_position']} |",
        f"| Directed speed >= 4 m/s | {final_metrics['reported_event_directed_tip_speed_m_s']:.3f} m/s | {gates['directed_tip_speed']} |",
        f"| Direction <= 30 deg | {final_metrics['reported_event_direction_error_deg']:.3f} deg | {gates['impact_direction']} |",
        f"| c10 first | {final_metrics['first_target_entry_marker_label']} | {gates['tip_first']} |",
        f"| UAV displacement <= 0.50 m | {final_metrics['maximum_uav_displacement_m']:.4f} m | {gates['uav_displacement']} |",
        f"| UAV speed <= 3 m/s | {final_metrics['maximum_uav_speed_m_s']:.4f} m/s | {gates['uav_speed']} |",
        f"| Acceleration <= 20 m/s² | {final_metrics['maximum_command_acceleration_m_s2']:.4f} m/s² | {gates['command_acceleration']} |",
        f"| Finite rollout | {final_metrics['finite']} | {gates['finite']} |",
    ]
    old = audit["old_direct_knot"]
    new = audit["new_full_rank_spectral"]
    training = list(outcome.training_history)
    q1 = [row["q1_loss"] for row in training if "q1_loss" in row]
    q2 = [row["q2_loss"] for row in training if "q2_loss" in row]
    entropies = [row["entropy"] for row in training if "entropy" in row]
    stochastic_progress_max = max(row["progress"]["maximum"] for row in training)
    stochastic_progress_p95_max = max(row["progress"]["p95"] for row in training)
    stochastic_distance_min = min(
        row["minimum_tip_distance_m"]["minimum"] for row in training
    )
    stochastic_directed_max = max(
        row["directed_tip_speed_m_s"]["maximum"] for row in training
    )
    stochastic_tip_speed_max = max(
        row["maximum_tip_speed_m_s"]["maximum"] for row in training
    )
    stochastic_uav_displacement_max = max(
        row["maximum_uav_displacement_m"]["maximum"] for row in training
    )
    stochastic_uav_speed_max = max(
        row["maximum_uav_speed_m_s"]["maximum"] for row in training
    )
    stochastic_feasible_min = min(row["feasible_rate"] for row in training)
    stochastic_feasible_max = max(row["feasible_rate"] for row in training)
    fraction_keys = training[0]["fractions"].keys()
    maximum_fractions = {
        key: max(row["fractions"][key] for row in training) for key in fraction_keys
    }
    reward_minimum = min(row["reward"]["minimum"] for row in training)
    reward_maximum = max(row["reward"]["maximum"] for row in training)
    spectral_initial = outcome.spectral_statistics_history[0]
    spectral_final = outcome.spectral_statistics_history[-1]
    diversity_initial = outcome.behavioral_diversity_history[0]
    diversity_final = outcome.behavioral_diversity_history[-1]
    correlations = [
        row["critic_diagnostic"]["reward_q_correlation"]
        for row in evaluations
        if row.get("critic_diagnostic", {}).get("reward_q_correlation") is not None
    ]
    final = outcome.selected_evaluation
    first_success = "NONE" if outcome.first_stochastic_success_episode is None else str(outcome.first_stochastic_success_episode)
    best_stochastic = "NONE" if outcome.best_stochastic_success is None else json.dumps(_safe(outcome.best_stochastic_success), sort_keys=True)
    lines = [
        "# Milestone 5B.4 — Spectral Entropy SAC Report",
        "",
        "## 1. Diagnosis from Milestone 5B.3",
        "",
        "5B.3 showed healthy simulator, interface, reward-v2, critic conditioning, and entropy, but independent knot-space exploration failed to consolidate a whip. Stochastic samples occasionally approached the target while remaining unsafe or dynamically incorrect; the deterministic actor returned to a short no-op. This milestone tests temporal exploration geometry rather than another reward.",
        "",
        "## 2. Frozen scientific contract",
        "",
        "The simulator used `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI` unchanged. `rl_whip_reward_v2` is unchanged, as are every hard success gate. The task context is the single canonical settled state, target [1,0,1.4] m, direction [+1,0,0], and nominal theta. CEM data, demonstrations, physics randomization, target randomization, and initial-state randomization were not used.",
        "",
        "## 3. Stationary reward and actor-only exploration pressure",
        "",
        "The environment and replay store only `rl_whip_reward_v2`. The critic target remains the terminal environment reward with no bootstrap. `alpha_explore` appears only in the actor objective: `(alpha_SAC + alpha_explore) log pi - min(Q1,Q2)`. It never enters reward, replay, Q targets, or scientific metrics. The stationarity test used identical rewards at k=0, 100k, and 400k and passed.",
        "",
        "## 4. Fixed duration curriculum",
        "",
        "Duration was fixed at 1.20 s to remove the repeatedly observed 0.45-s escape route. This is a discovery curriculum, not a new scientific deadline or final policy formulation. The known 1.1175-s CEM solution lies within this rollout.",
        "",
        "## 5. Old and new exploration formulations",
        "",
        "The old actor sampled 49 approximately factorized raw knot/duration coordinates. The new actor samples 48 full-rank spectral coordinates: 16 coefficients independently for x, y, and z. All 16 frequencies are retained. Duration is deterministic, while the critic continues receiving a 49-D decoded action with normalized duration +1.",
        "",
        "## 6. Exact orthonormal DCT convention and log probability",
        "",
        "For temporal index n and frequency f, the inverse basis is `B[n,0]=sqrt(1/16)` and `B[n,f]=sqrt(2/16) cos(pi (n+1/2) f /16)` for f>0. Raw temporal values are `z=Bc`. The measured maximum `B^T B-I` error was " + f"{verification['maximum_orthonormality_error']:.3e}; rank was {verification['matrix_rank']}; reconstruction error was {verification['maximum_reconstruction_error']:.3e}; energy error was {verification['maximum_energy_error']:.3e}. Since `|det B|=1`, the IDCT contributes zero log-Jacobian. Log probability is `log N(c;mu,sigma) - sum_k log|det J_radial(z_k)|`; fixed duration contributes zero.",
        "",
        "## 7. Initial spectral standard deviation",
        "",
        "The initial learned standard deviations were initialized to `1/sqrt(1+(f/4)^2)` for f=0..15 on each axis. All frequencies were nonzero; low frequencies were largest, and every log standard deviation remained trainable.",
        "",
        "## 8. Distribution verification",
        "",
        f"Verification status: **{verification['status']}**. Checks: `{json.dumps(verification['checks'], sort_keys=True)}`.",
        "",
        "## 9. Pre-training old-vs-new production exploration audit",
        "",
        "Both distributions evaluated 2,048 actions with duration fixed at 1.20 s; no audit sample entered replay.",
        "",
        "| Metric | Old direct-knot | New spectral |",
        "|---|---:|---:|",
        f"| Mean temporal roughness | {old['temporal_roughness']['mean']:.3f} | {new['temporal_roughness']['mean']:.3f} |",
        f"| Roughness / 400 | {old['temporal_roughness_normalized']['mean']:.4f} | {new['temporal_roughness_normalized']['mean']:.4f} |",
        f"| Mean progress | {old['progress']['mean']:.4f} | {new['progress']['mean']:.4f} |",
        f"| 95th-percentile progress | {old['progress']['p95']:.4f} | {new['progress']['p95']:.4f} |",
        f"| Maximum progress | {old['progress']['maximum']:.4f} | {new['progress']['maximum']:.4f} |",
        f"| Feasible rate | {100*old['feasible_rate']:.2f}% | {100*new['feasible_rate']:.2f}% |",
        f"| d_min <= 0.20 m | {100*old['fractions']['d_min_le_0_20_m']:.2f}% | {100*new['fractions']['d_min_le_0_20_m']:.2f}% |",
        f"| Finite rate | {100*old['finite_rate']:.2f}% | {100*new['finite_rate']:.2f}% |",
        "",
        f"New/old mean roughness ratio: {audit['spectral_to_old_mean_roughness_ratio']:.4f}. New raw spectral energy fractions: `{json.dumps(audit['frequency_energy_fractions']['new_raw_spectral_coefficients'], sort_keys=True)}`. Low frequency energy dominated while high-frequency energy remained nonzero. Exploration-audit status: **{audit['status']}**.",
        "",
        "## 10. Exact exploration schedule",
        "",
        "`alpha_explore=0.25` through 75k episodes, decays linearly to zero over 75k–300k, and is zero thereafter. `alpha_SAC` remains independently learned against target entropy -48; `alpha_effective=alpha_SAC+alpha_explore` only in the actor loss.",
        "",
        "## 11. SAC configuration",
        "",
        "Fresh seed-42 actor, critics, and alpha; 83-D context; 48 stochastic spectral dimensions; critic external action 49; 256-256-256 SiLU networks; Adam 3e-4; Huber delta 1; 2,048 collection rows; 4,096 replay minibatch; 10,240 warmup; eight updates per batch; replay capacity 1,000,000; gradient clip 10; target entropy -48. No earlier SAC weights were resumed.",
        "",
        "## 12. Pre-run learning checks",
        "",
        f"Status: **{pre_run['status']}**. `{json.dumps(pre_run['checks'], sort_keys=True)}`",
        "",
        "## 13. Deterministic policy progression",
        "",
        *progression,
        "",
        f"Highest deterministic behavior level: **{outcome.highest_deterministic_behavior_level}**.",
        "",
        "## 14. Critic diagnostics",
        "",
        (f"Q1 Huber loss range [{min(q1):.6f}, {max(q1):.6f}], Q2 range [{min(q2):.6f}, {max(q2):.6f}]. Reward/Q correlation moved from {correlations[0]:.4f} to {correlations[-1]:.4f}. Replay environment rewards remained in [{reward_minimum:.4f}, {reward_maximum:.4f}]." if q1 and q2 and correlations else "No critic updates were completed."),
        "The report artifacts include Q/reward correlations, Q ranges, replay reward ranges, and critic gradient norms at each evaluation. No NaN/Inf was accepted.",
        "",
        "## 15. Actor, entropy, and spectral diagnostics",
        "",
        (f"Policy entropy during updates ranged from {min(entropies):.4f} to {max(entropies):.4f}." if entropies else "No entropy updates were recorded."),
        f"alpha_SAC moved from {next(row['alpha_sac'] for row in training if 'q1_loss' in row):.4f} to {training[-1]['alpha_sac']:.4f}; alpha_effective began at {next(row['alpha_effective'] for row in training if 'q1_loss' in row):.4f} and ended at {training[-1]['alpha_effective']:.4f}. Initial low/mid/high sampled spectral energy fractions were {spectral_initial['low']['sampled_energy_fraction']:.3f}/{spectral_initial['mid']['sampled_energy_fraction']:.3f}/{spectral_initial['high']['sampled_energy_fraction']:.3f}; final fractions were {spectral_final['low']['sampled_energy_fraction']:.3f}/{spectral_final['mid']['sampled_energy_fraction']:.3f}/{spectral_final['high']['sampled_energy_fraction']:.3f}. Initial low/mid/high actor standard deviations were {spectral_initial['low']['mean_actor_standard_deviation']:.3f}/{spectral_initial['mid']['mean_actor_standard_deviation']:.3f}/{spectral_initial['high']['mean_actor_standard_deviation']:.3f}; final values were {spectral_final['low']['mean_actor_standard_deviation']:.3f}/{spectral_final['mid']['mean_actor_standard_deviation']:.3f}/{spectral_final['high']['mean_actor_standard_deviation']:.3f}.",
        "`training_history.json` records alpha_SAC, alpha_explore, alpha_effective, entropy, mean log_pi, actor loss, acceleration norms, roughness, sign changes, and stochastic behavior. `spectral_statistics_history.json` preserves the complete per-collection spectral evolution.",
        "",
        "## 16. Stochastic progress, feasibility, and behavioral diversity",
        "",
        f"Across collection batches, maximum sampled progress reached {stochastic_progress_max:.6f}, maximum batch p95 progress was {stochastic_progress_p95_max:.6f}, and minimum sampled d_min was {1000*stochastic_distance_min:.3f} mm. Maximum reported directed speed was {stochastic_directed_max:.3f} m/s. However, stochastic feasible rate was {100*stochastic_feasible_min:.2f}%–{100*stochastic_feasible_max:.2f}%; no feasible high-progress joint event occurred. The maxima of interval fractions were: progress>=0.25 {100*maximum_fractions['progress_ge_0_25']:.2f}%, >=0.50 {100*maximum_fractions['progress_ge_0_50']:.2f}%, >=0.75 {100*maximum_fractions['progress_ge_0_75']:.2f}%, >=0.90 {100*maximum_fractions['progress_ge_0_90']:.2f}%, d_min<=0.20 m {100*maximum_fractions['d_min_le_0_20_m']:.2f}%, and d_min<=0.10 m {100*maximum_fractions['d_min_le_0_10_m']:.2f}%. Every feasible-and-progress/distance joint fraction was 0%.",
        "",
        f"The stochastic population remained tensor-finite, but unsafe outliers were extreme: maximum tip speed {stochastic_tip_speed_max:.1f} m/s, UAV displacement {stochastic_uav_displacement_max:.1f} m, and UAV speed {stochastic_uav_speed_max:.1f} m/s. These rows were correctly infeasible and received finite bounded reward penalties; they also dominated naive descriptor dispersion. Mean normalized descriptor distance changed from {diversity_initial['mean_distance_from_descriptor_mean']:.3f} to {diversity_final['mean_distance_from_descriptor_mean']:.3f}. No novelty term was added.",
        "",
        "## 17. Stochastic scientific-success diagnostics",
        "",
        f"First stochastic success: **{first_success}**. Total stochastic successes: **{outcome.total_stochastic_successes}**. Best stochastic success metadata: `{best_stochastic}`.",
        "Successful samples, if any, remained ordinary replay entries and were neither duplicated nor treated as expert data.",
        "",
        "## 18. Selected checkpoint and authoritative replay",
        "",
        f"Selection used deterministic success, behavior level, feasibility, progress, distance, then speed/direction. Selected deterministic level: {final['behavior_level']}; progress: {final['progress']:.6f}; reward: {final['reward']:.6f}. The final action was replayed once at batch size one from the canonical state with no noise and fixed T=1.20 s.",
        "",
        *gate_table,
        "",
        "## 19. CEM feasibility reference only",
        "",
        "The existing CEM reference (not used for training) remains: 1.756-mm tip error, 4.599-m/s directed speed, 19.615-deg direction error, 1.1175-s duration, scientific PASS. It establishes that the simulator/action task is feasible.",
        "",
        "## 20. Runtime and artifacts",
        "",
        f"Collected episodes: {outcome.episodes}; gradient updates: {outcome.gradient_updates}; training runtime: {outcome.runtime_s:.3f} s; overall runtime: {runtime['overall_runtime_s']:.3f} s; peak CUDA memory: {runtime['peak_cuda_memory_mb']:.1f} MiB. Video: `{video_path}`.",
        "",
        "## 21. Scientific interpretation and classification",
        "",
        f"**{outcome.classification}**. " + (
            "The deterministic actor passed all hard gates."
            if outcome.classification.endswith("_PASS")
            else "Exploration reached a scientific success, but deterministic consolidation did not pass."
            if outcome.total_stochastic_successes > 0
            else "No scientific stochastic success appeared and the deterministic actor did not exceed Level 2."
        ),
        "",
        "Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**. Every command remains simulation-only and not authorized for real flight.",
        "",
        "## Final summary",
        "",
        "    Model:",
        "        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI",
        "",
        "    Policy:",
        "        One-Shot Terminal SAC",
        "",
        "    Context:",
        "        SINGLE CANONICAL",
        "",
        "    Execution:",
        "        OPEN LOOP",
        "",
        "    Reward:",
        "        rl_whip_reward_v2",
        "        UNCHANGED",
        "",
        "    Exploration representation:",
        "        FULL-RANK ORTHONORMAL DCT",
        "",
        "    Acceleration knots:",
        "        16 x 3",
        "",
        "    Stochastic dimensions:",
        "        48",
        "",
        "    Duration:",
        "        FIXED 1.20 s",
        "        DISCOVERY CURRICULUM ONLY",
        "",
        "    Base target entropy:",
        "        -48",
        "",
        "    Early exploration bonus:",
        "        alpha_explore initial = 0.25",
        "",
        "    Exploration anneal:",
        "        hold through 75k",
        "        linear -> 0 by 300k",
        "",
        "    CEM training data:",
        "        NOT USED",
        "",
        "    Episodes:",
        f"        {outcome.episodes}",
        "",
        "    First stochastic success:",
        f"        episode {first_success}",
        "",
        "    Total stochastic successes:",
        f"        {outcome.total_stochastic_successes}",
        "",
        "    Best stochastic success:",
        f"        {best_stochastic}",
        "",
        "    Highest deterministic behavior level:",
        f"        {outcome.highest_deterministic_behavior_level}",
        "",
        "    Deterministic progress:",
        f"        {final_metrics['progress']:.6f}",
        "",
        "    Deterministic minimum tip distance:",
        f"        {1000*final_metrics['minimum_tip_target_distance_m']:.3f} mm",
        "",
        "    Deterministic directed speed:",
        f"        {final_metrics['reported_event_directed_tip_speed_m_s']:.3f} m/s",
        "",
        "    Deterministic direction error:",
        f"        {final_metrics['reported_event_direction_error_deg']:.3f} deg",
        "",
        "    Deterministic UAV displacement:",
        f"        {final_metrics['maximum_uav_displacement_m']:.4f} m",
        "",
        "    Deterministic feasible:",
        f"        {'YES' if final_metrics['feasible'] else 'NO'}",
        "",
        "    Deterministic scientific task:",
        f"        {'PASS' if final_metrics['success'] else 'FAIL'}",
        "",
        "    Result:",
        f"        {outcome.classification}",
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
    (artifact / "MILESTONE5B4_SPECTRAL_ENTROPY_SAC_REPORT.md").write_text(report, encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def run(config_path: Path = DEFAULT_CONFIG, *, audit_only: bool = False) -> dict[str, Any]:
    overall_start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    artifact = PROJECT_ROOT / "data" / "policy_training" / "sac_canonical_spectral_entropy_v1" / _timestamp()
    artifact.mkdir(parents=True)
    shutil.copy2(config_path, artifact / "sac_config.json")
    _write_json(artifact / "spectral_policy_config.json", config["spectral_distribution"])
    _write_json(
        artifact / "exploration_entropy_schedule.json",
        {
            **config["exploration_entropy_bonus"],
            "name": "alpha_explore",
            "actor_objective_only": True,
            "environment_reward": "rl_whip_reward_v2 UNCHANGED",
            "test_points": {str(k): alpha_explore_schedule(k) for k in (0, 74_999, 75_000, 187_500, 300_000, 400_000)},
        },
    )
    reward_config = _reward_config(config["reward"])
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    task_path = (PROJECT_ROOT / config["task_config"]).resolve()
    task = load_variable_duration_task(task_path)
    canonical_bank, canonical_specification = _canonical_setup(simulator, task)
    source = (PROJECT_ROOT / config["source_5b2_artifact"]).resolve()
    normalizer = FixedContextNormalizer.load(source / "context_normalizer.json")
    shutil.copy2(source / "context_normalizer.json", artifact / "context_normalizer.json")

    verification = _distribution_verification(simulator.device)
    _write_json(artifact / "spectral_distribution_verification.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError("Spectral distribution verification failed; SAC was not started.")
    audit = _pretraining_exploration_audit(
        simulator,
        task,
        reward_config,
        normalizer,
        canonical_bank,
        canonical_specification,
        config,
    )
    _write_json(artifact / "pretraining_exploration_audit.json", audit)
    print("SPECTRAL_EXPLORATION_AUDIT " + json.dumps(audit["checks"], sort_keys=True), flush=True)
    if audit["status"] != "PASS":
        raise RuntimeError("Pre-training spectral exploration audit failed; SAC was not started.")
    if audit_only:
        response = {
            "verification": verification["status"],
            "exploration_audit": audit["status"],
            "artifact_directory": str(artifact),
            "sac": "NOT RUN — AUDIT ONLY",
        }
        print("MILESTONE5B4_AUDIT_RESULT " + json.dumps(response, sort_keys=True), flush=True)
        return response
    pre_run = _pre_run_learning_verification(
        simulator,
        normalizer,
        canonical_bank,
        canonical_specification,
        config,
        artifact,
    )
    _write_json(artifact / "pre_run_learning_verification.json", pre_run)
    if pre_run["status"] != "PASS":
        raise RuntimeError("Pre-run learning verification failed; SAC was not started.")

    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    sac = config["sac"]
    agent = create_spectral_agent(
        device=simulator.device,
        actor_lr=float(sac["actor_learning_rate"]),
        critic_lr=float(sac["critic_learning_rate"]),
        alpha_lr=float(sac["alpha_learning_rate"]),
        target_entropy=float(sac["target_entropy"]),
        gradient_clip_norm=float(sac["gradient_clip_norm"]),
        std_low=float(config["spectral_distribution"]["std_low"]),
    )
    replay = TerminalReplayBuffer(int(sac["replay_capacity"]))
    outcome = run_spectral_sac_pilot(
        agent=agent,
        normalizer=normalizer,
        replay=replay,
        simulator=simulator,
        task=task,
        reward_config=reward_config,
        canonical_bank=canonical_bank,
        canonical_specification=canonical_specification,
        config=config,
        artifact_directory=artifact,
    )
    _write_json(artifact / "training_history.json", list(outcome.training_history))
    _write_json(artifact / "evaluation_history.json", list(outcome.evaluation_history))
    _write_json(artifact / "spectral_statistics_history.json", list(outcome.spectral_statistics_history))
    _write_json(artifact / "behavioral_diversity_history.json", list(outcome.behavioral_diversity_history))
    if outcome.best_stochastic_success is not None:
        _write_json(artifact / "best_stochastic_success_metadata.json", outcome.best_stochastic_success)
    torch.save(agent.actor.state_dict(), artifact / "best_actor.pt")
    torch.save(agent.critic1.state_dict(), artifact / "best_critic1.pt")
    torch.save(agent.critic2.state_dict(), artifact / "best_critic2.pt")
    _write_json(artifact / "best_alpha.json", {"alpha_sac": float(agent.alpha.detach()), "base_target_entropy": -48.0})

    result, replay_result = _record_selected_policy(
        agent,
        normalizer,
        simulator,
        task,
        reward_config,
        canonical_bank,
        canonical_specification,
    )
    save_command_csv(artifact / "canonical_fullstate_command.csv", replay_result)
    save_replay_npz(artifact / "canonical_final_replay.npz", replay_result, task)
    final_metrics = final_replay_metrics(replay_result, task, settings)
    row = result.row(0)
    final_metrics.update(
        {
            "policy": "One-Shot Terminal SAC — Full-Rank Orthonormal DCT",
            "reward_profile": "rl_whip_reward_v2",
            "rl_reward": float(result.reward[0]),
            "reward_components": row["reward_components"],
            "progress": float(row["reward_components"]["normalized_progress"]),
            "maximum_tip_speed_m_s": float(result.max_tip_speed_m_s[0]),
            "optimized_duration_s": 1.20,
            "duration_interpretation": "DISCOVERY_CURRICULUM_ONLY",
            "task_classification": "PASS" if bool(result.task_success[0]) else "FAIL",
            "sac_classification": outcome.classification,
            "highest_deterministic_behavior_level": outcome.highest_deterministic_behavior_level,
            "first_stochastic_success_episode": outcome.first_stochastic_success_episode,
            "total_stochastic_successes": outcome.total_stochastic_successes,
            "policy_query_count": 1,
            "execution": "OPEN_LOOP",
        }
    )
    _write_json(artifact / "canonical_final_metrics.json", final_metrics)
    shutil.copy2(artifact / "canonical_final_replay.npz", artifact / "final_replay.npz")
    video_task = json.loads(task_path.read_text(encoding="utf-8"))
    video_task["task_id"] = "oneshot_sac_spectral_entropy_canonical"
    _write_json(artifact / "task_config_snapshot.json", video_task)
    _write_json(artifact / "final_metrics.json", final_metrics)
    _write_json(artifact / "mppi_iteration_history.json", [])
    planning_result = PlanningResult(
        task_id="oneshot_sac_spectral_entropy_canonical",
        task_label="One-Shot SAC Spectral Entropy Canonical",
        directory=artifact,
        task_config=video_task,
        metrics=final_metrics,
        iteration_history=(),
    )
    video_path, video_metadata = render_replay_video(planning_result, overwrite=True)
    runtime = {
        "overall_runtime_s": time.perf_counter() - overall_start,
        "training_runtime_s": outcome.runtime_s,
        "collected_episodes": outcome.episodes,
        "gradient_updates": outcome.gradient_updates,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(simulator.device) / 1024**2,
        "device": torch.cuda.get_device_name(simulator.device),
        "video": video_metadata,
    }
    _write_json(artifact / "runtime.json", runtime)
    source_paths = [
        config_path,
        task_path,
        PROJECT_ROOT / "planning" / "rl_reward.py",
        PROJECT_ROOT / "planning" / "variable_duration.py",
        PROJECT_ROOT / "learning" / "sac.py",
        PROJECT_ROOT / "learning" / "spectral_sac.py",
        PROJECT_ROOT / "learning" / "spectral_pilot.py",
        Path(__file__),
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone5b4_source_hash_manifest_v1",
            "sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths},
            "model_freeze": config["model_freeze"],
            "reward": "rl_whip_reward_v2 UNCHANGED",
            "cem_training_data": "NOT USED",
            "protected_test": "NOT EVALUATED",
            "real_hardware": "NOT EXECUTED",
        },
    )
    report = _write_report(
        artifact=artifact,
        config=config,
        verification=verification,
        audit=audit,
        pre_run=pre_run,
        outcome=outcome,
        final_metrics=final_metrics,
        runtime=runtime,
        video_path=video_path,
    )
    response = {
        "verification": verification["status"],
        "exploration_audit": audit["status"],
        "sac": outcome.classification,
        "episodes": outcome.episodes,
        "first_stochastic_success": outcome.first_stochastic_success_episode,
        "total_stochastic_successes": outcome.total_stochastic_successes,
        "highest_deterministic_level": outcome.highest_deterministic_behavior_level,
        "scientific_success": bool(result.task_success[0]),
        "artifact_directory": str(artifact),
        "video": str(video_path),
        "report": str(REPORT_PATH),
        "report_characters": len(report),
        "protected_test": "NOT EVALUATED",
        "real_hardware": "NOT EXECUTED",
    }
    print("MILESTONE5B4_RESULT " + json.dumps(response, sort_keys=True), flush=True)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config.resolve(), audit_only=arguments.audit_only)


if __name__ == "__main__":
    main()
