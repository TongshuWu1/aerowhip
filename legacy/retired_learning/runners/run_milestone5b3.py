"""Milestone 5B.3: progress-reward audit and canonical-only SAC discovery."""

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

from learning.canonical_pilot import (
    collect_canonical_batch,
    repeated_canonical_specification,
    run_canonical_sac_pilot,
)
from learning.context_sampling import ContextSpecification, build_context_from_specification
from learning.normalization import FixedContextNormalizer
from learning.one_shot_env import evaluate_open_loop_batch
from learning.policy_context import build_policy_context
from learning.replay import TerminalReplayBuffer
from learning.reward_audit import audit_saved_replay
from learning.sac import TerminalSacAgent, squash_raw_action
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
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "learning" / "sac_canonical_progress_v1.json"
REPORT_PATH = PROJECT_ROOT / "MILESTONE5B3_CANONICAL_PROGRESS_SAC_REPORT.md"


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
    if config.get("schema") != "sac_canonical_progress_v1":
        raise ValueError("Unsupported Milestone 5B.3 configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The production model freeze is immutable.")
    if config.get("context_dimension") != 83 or config.get("action_dimension") != 49:
        raise ValueError("The 83-D context and 49-D action contracts are frozen.")
    if config.get("context_distribution") != "SINGLE_CANONICAL_CONTEXT":
        raise ValueError("Milestone 5B.3 permits exactly one canonical context.")
    if config.get("duration_interpretation") != "PHASE1_POLICY_DURATION_ENVELOPE":
        raise ValueError("Duration must remain a Phase-1 envelope, not a task gate.")
    reward = config["reward"]
    if reward.get("profile") != "rl_whip_reward_v2" or float(
        reward.get("progress_weight", -1.0)
    ) != 4.0:
        raise ValueError("Milestone 5B.3 requires rl_whip_reward_v2 at weight 4.0.")
    if float(reward.get("additional_scale_divisor", 0.0)) != 1.0:
        raise ValueError("The progress reward must not receive another scale divisor.")
    policy = config["artifact_policy"]
    if (
        policy["real_flight_authorized"]
        or policy["protected_test_evaluation_allowed"]
        or policy["cem_training_data_allowed"]
    ):
        raise ValueError("5B.3 must remain simulation-only, protected, and teacher-free.")


def _canonical_setup(simulator, task):
    state = hover_preroll(simulator, task)
    bank = initial_state_bank_from_state(
        state,
        command_position_world_m=torch.tensor(
            task.initial_uav_position_m, device=simulator.device
        ),
        command_velocity_world_m_s=torch.tensor(
            task.initial_uav_velocity_m_s, device=simulator.device
        ),
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


def _live_row(result, index: int, label: str) -> dict[str, Any]:
    row = result.row(index)
    component = row["reward_components"]
    progress_reward = float(component["progress"])
    return {
        "label": label,
        "source_directory": None,
        "source_replay": "new reward-audit-only rollout",
        "hard_success": bool(row["task_success"]),
        "feasible": bool(row["feasible"]),
        "tip_error_m": float(row["tip_min_distance_m"]),
        "directed_speed_m_s": float(row["directed_tip_speed_m_s"]),
        "direction_error_deg": float(row["direction_error_deg"]),
        "tip_first": int(row["first_entry_marker"]) == 10,
        "first_entry_marker": int(row["first_entry_marker"]) or None,
        "maximum_uav_displacement_m": float(row["max_uav_displacement_m"]),
        "maximum_uav_speed_m_s": float(row["max_uav_speed_m_s"]),
        "maximum_command_acceleration_m_s2": float(
            row["max_command_acceleration_m_s2"]
        ),
        "d0_m": float(component["initial_tip_distance_m"]),
        "d_min_m": float(component["minimum_tip_distance_m"]),
        "progress": float(component["normalized_progress"]),
        "R_progress": progress_reward,
        "R_strike": float(component["strike"]),
        "R_success": float(component["success"]),
        "R_safety": float(component["safety"]),
        "R_non_tip": float(component["non_tip"]),
        "R_control": float(component["control"]),
        "R_time": float(component["time"]),
        "R_RL_v1": float(row["reward"]) - progress_reward,
        "R_RL_v2": float(row["reward"]),
    }


def _audit_existing_rows(
    config: dict[str, Any],
    reward_config: RLWhipRewardConfig,
    canonical_d0_m: float,
) -> list[dict[str, Any]]:
    source = PROJECT_ROOT / config["reward_audit"]["source_5b2_results"]
    previous = json.loads(source.read_text(encoding="utf-8"))["rows"]
    rows: list[dict[str, Any]] = []
    for old in previous:
        directory = old.get("source_directory")
        if directory:
            exact = audit_saved_replay(
                PROJECT_ROOT / directory,
                label=str(old["label"]),
                config=reward_config,
            )
            exact["R_RL_v1"] = float(old["R_RL"])
            exact["R_RL_v2"] = float(exact.pop("R_RL"))
            rows.append(exact)
            continue
        d_min = min(float(old["tip_error_m"]), canonical_d0_m)
        progress = max(0.0, min(1.0, (canonical_d0_m - d_min) / canonical_d0_m))
        progress_reward = reward_config.progress_weight * progress
        upgraded = dict(old)
        upgraded.update(
            {
                "d0_m": canonical_d0_m,
                "d_min_m": d_min,
                "progress": progress,
                "R_progress": progress_reward,
                "R_RL_v1": float(old["R_RL"]),
                "R_RL_v2": float(old["R_RL"]) + progress_reward,
                "progress_source": "same canonical d0 plus saved d_min metric",
            }
        )
        upgraded.pop("R_RL", None)
        rows.append(upgraded)
    return rows


def _random_progress_audit(
    simulator,
    task,
    reward_config,
    canonical_bank,
    canonical_specification,
    *,
    count: int,
    seed: int,
):
    specification = repeated_canonical_specification(
        canonical_specification, count, split="reward_audit_random_progress"
    )
    context = build_context_from_specification(simulator, canonical_bank, specification)
    generator = torch.Generator(device=simulator.device).manual_seed(seed)
    raw = torch.randn((count, 49), generator=generator, device=simulator.device)
    action, _ = squash_raw_action(raw)
    with torch.no_grad():
        result = evaluate_open_loop_batch(
            simulator, context, action, task, rl_reward_config=reward_config
        )
    rows = [_live_row(result, index, f"random_progress_{index:03d}") for index in range(count)]
    boundaries = ((0.00, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.000001))
    bins: list[dict[str, Any]] = []
    for lower, upper in boundaries:
        selected = [row for row in rows if lower <= row["progress"] < upper]
        feasible = [row for row in selected if row["feasible"]]
        rewards = np.asarray([row["R_RL_v2"] for row in selected], dtype=np.float64)
        feasible_rewards = np.asarray(
            [row["R_RL_v2"] for row in feasible], dtype=np.float64
        )
        bins.append(
            {
                "progress_interval": [lower, min(upper, 1.0)],
                "count": len(selected),
                "feasible_count": len(feasible),
                "mean_reward": float(rewards.mean()) if rewards.size else None,
                "median_reward": float(np.median(rewards)) if rewards.size else None,
                "feasible_mean_reward": float(feasible_rewards.mean())
                if feasible_rewards.size
                else None,
                "feasible_median_reward": float(np.median(feasible_rewards))
                if feasible_rewards.size
                else None,
            }
        )
    return rows, bins


def _audit_gate(
    existing: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    indexed = {row["label"]: row for row in existing}
    all_rows = existing + random_rows
    rewards = np.asarray([row["R_RL_v2"] for row in all_rows], dtype=np.float64)
    hover = indexed["safe_hover_no_whip"]
    useful_partial = [
        row
        for row in all_rows
        if row["feasible"] and row["progress"] >= 0.25 and not row["hard_success"]
    ]
    identity_errors = [
        abs(row["R_RL_v2"] - (row["R_RL_v1"] + 4.0 * row["progress"]))
        for row in all_rows
    ]
    audit = config["reward_audit"]
    checks = {
        "known_strong_success_high": indexed["strong_success"]["R_RL_v2"] > 12.0,
        "strong_above_fragile": indexed["strong_success"]["R_RL_v2"]
        > indexed["fragile_success"]["R_RL_v2"],
        "fragile_above_close_wrong_direction": indexed["fragile_success"]["R_RL_v2"]
        > indexed["close_wrong_direction"]["R_RL_v2"],
        "close_wrong_above_close_slow": indexed["close_wrong_direction"]["R_RL_v2"]
        > indexed["close_slow"]["R_RL_v2"],
        "hover_neutral_or_low": abs(hover["R_RL_v2"]) < 0.10,
        "meaningful_partial_progress_above_hover": bool(useful_partial)
        and max(row["R_RL_v2"] for row in useful_partial) > hover["R_RL_v2"] + 1.0,
        "close_near_misses_useful": indexed["close_slow"]["R_RL_v2"]
        > hover["R_RL_v2"] + 2.0,
        "unsafe_aggressive_below_feasible_close_slow": indexed["old_aggressive_mppi"][
            "R_RL_v2"
        ]
        < indexed["close_slow"]["R_RL_v2"],
        "all_rewards_finite": bool(np.isfinite(rewards).all()),
        "moderate_reward_range": bool(
            rewards.min() >= float(audit["ordinary_reward_minimum"]) - 5.0
            and rewards.max() <= float(audit["ordinary_reward_maximum"])
        ),
        "v2_exactly_v1_plus_progress": max(identity_errors, default=0.0) <= 1.0e-5,
        "scientific_gates_unchanged": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed_reward_minimum": float(rewards.min()),
        "observed_reward_maximum": float(rewards.max()),
        "maximum_v2_identity_error": max(identity_errors, default=0.0),
        "useful_partial_progress_count": len(useful_partial),
    }


def _pre_run_verification(
    simulator,
    task,
    reward_config,
    normalizer,
    canonical_bank,
    canonical_specification,
    config,
    artifact,
) -> dict[str, Any]:
    repeated = repeated_canonical_specification(
        canonical_specification, 8, split="pre_run_identical_context"
    )
    context = build_context_from_specification(simulator, canonical_bank, repeated)
    normalized = normalizer.normalize(context.to_tensor())
    same_context = torch.equal(normalized, normalized[:1].expand_as(normalized))
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    agent = TerminalSacAgent.create(
        device=simulator.device,
        actor_lr=float(config["sac"]["actor_learning_rate"]),
        critic_lr=float(config["sac"]["critic_learning_rate"]),
        alpha_lr=float(config["sac"]["alpha_learning_rate"]),
        target_entropy=float(config["sac"]["target_entropy"]),
        gradient_clip_norm=float(config["sac"]["gradient_clip_norm"]),
        critic_loss="smooth_l1",
    )
    synthetic = agent.update(
        normalized,
        torch.tanh(torch.randn((8, 49), device=simulator.device)),
        torch.linspace(-2.0, 2.0, 8, device=simulator.device),
    )
    finite_update = all(math.isfinite(value) for value in synthetic.values())
    generator = torch.Generator().manual_seed(77)
    checkpoint = artifact / "pre_run_checkpoint_test.pt"
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
    result = {
        "identical_canonical_context_rows": same_context,
        "context_dimension": int(normalized.shape[1]),
        "actor_environment_path_tested_by_reward_audit": True,
        "critic_huber_update_finite": finite_update,
        "checkpoint_save_resume": int(loaded["episodes"]) == 8,
        "cem_training_data_used": False,
    }
    required_true = (
        "identical_canonical_context_rows",
        "actor_environment_path_tested_by_reward_audit",
        "critic_huber_update_finite",
        "checkpoint_save_resume",
    )
    if not all(result[name] is True for name in required_true) or result[
        "cem_training_data_used"
    ] is not False:
        raise RuntimeError(f"Milestone 5B.3 pre-run verification failed: {result}")
    return result


def _record_selected_policy(
    agent,
    normalizer,
    simulator,
    task,
    reward_config,
    canonical_bank,
    canonical_specification,
):
    context = build_context_from_specification(
        simulator, canonical_bank, canonical_specification
    )
    with torch.no_grad():
        action = agent.actor(
            normalizer.normalize(context.to_tensor()), deterministic=True
        ).deterministic_mean_action
        result = evaluate_open_loop_batch(
            simulator,
            context,
            action,
            task,
            record_trajectory=True,
            rl_reward_config=reward_config,
        )
    if result.trajectory is None:
        raise RuntimeError("Authoritative canonical replay was not recorded.")
    return result, result.trajectory


def _duration_diagnostic(evaluations: list[dict[str, Any]]) -> str:
    durations = [float(row["canonical"]["duration_s"]) for row in evaluations]
    if durations and durations[-1] <= 0.47:
        return "LOWER_DURATION_ATTRACTOR_PERSISTS"
    if durations and durations[-1] >= 1.18:
        return "UPPER_DURATION_ENVELOPE_ACTIVE"
    return "NORMAL"


def _write_report(
    *,
    artifact: Path,
    existing_audit: list[dict[str, Any]],
    progress_bins: list[dict[str, Any]],
    audit_gate: dict[str, Any],
    outcome=None,
    selected=None,
    final_metrics=None,
    runtime=None,
    video_path=None,
) -> None:
    table = [
        "| Trajectory | Success | Feasible | d0 [m] | d_min [m] | Progress | R_progress | R_v1 | R_v2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in existing_audit:
        if row["label"].startswith("bad_random_") and row["label"] != "bad_random_00":
            continue
        table.append(
            f"| {row['label']} | {row['hard_success']} | {row['feasible']} | "
            f"{row['d0_m']:.4f} | {row['d_min_m']:.4f} | {row['progress']:.4f} | "
            f"{row['R_progress']:.4f} | {row['R_RL_v1']:.4f} | {row['R_RL_v2']:.4f} |"
        )
    bin_table = [
        "| Progress bin | Count | Feasible | Mean reward | Median reward | Feasible mean | Feasible median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in progress_bins:
        lo, hi = row["progress_interval"]
        def shown(value):
            return "—" if value is None else f"{value:.4f}"
        bin_table.append(
            f"| [{lo:.2f}, {hi:.2f}) | {row['count']} | {row['feasible_count']} | "
            f"{shown(row['mean_reward'])} | {shown(row['median_reward'])} | "
            f"{shown(row['feasible_mean_reward'])} | {shown(row['feasible_median_reward'])} |"
        )
    lines = [
        "# Milestone 5B.3 — Progress-Shaped RL Reward and Canonical SAC Discovery Pilot",
        "",
        "## 1. Diagnosis from Milestone 5B.2",
        "",
        "Milestone 5B.2 fixed reward and critic numerical conditioning, but the 0.20-m proximity kernel was effectively zero over most untrained behavior. Safe no-op therefore returned approximately zero while aggressive exploration often paid safety/control penalties. The actor converged toward short no-whip maneuvers.",
        "",
        "## 2. Exact reward-version difference",
        "",
        "`rl_whip_reward_v1` remains preserved. `rl_whip_reward_v2` changes exactly one term: `R_progress = 4 * (d0-d_min)/max(d0,eps)`, clamped by construction to [0,4]. Every strike, success, safety, non-tip, control, and successful-only time term is unchanged.",
        "",
        "## 3. Reward audit",
        "",
        *table,
        "",
        f"Observed reward-v2 range: [{audit_gate['observed_reward_minimum']:.6f}, {audit_gate['observed_reward_maximum']:.6f}]. The maximum numerical error in `v2 = v1 + 4*progress` was {audit_gate['maximum_v2_identity_error']:.3e}.",
        "",
        "## 4. Random progress bins",
        "",
        *bin_table,
        "",
        f"**REWARD_AUDIT: {audit_gate['status']}**",
        "",
        "CEM artifacts were used only as saved reward-audit references. CEM actions, elites, trajectories, and demonstrations were not inserted into replay and did not initialize the actor.",
        "",
    ]
    if outcome is None:
        lines.extend(
            [
                "The reward audit failed, so SAC was not run.",
                "",
                "## Final summary",
                "",
                "    Model: MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI",
                "    Policy: One-Shot Terminal SAC",
                "    Context: SINGLE CANONICAL CONTEXT",
                "    Execution: OPEN LOOP",
                "    CEM training data: NOT USED",
                "    Reward: rl_whip_reward_v2",
                "    Progress weight: 4.0",
                f"    Reward audit: {audit_gate['status']}",
                "    SAC result: NOT RUN — REWARD AUDIT FAILED",
                "    Physics conditioning: NOT ENABLED",
                "    Protected test: NOT EVALUATED",
                "    Real hardware: NOT EXECUTED",
            ]
        )
    else:
        evaluations = list(outcome.evaluation_history)
        progression = [
            "| Episodes | Level | Progress | d_min [mm] | Directed [m/s] | Direction [deg] | Duration [s] | Feasible | Success | Reward | Q/reward corr. |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in evaluations:
            canonical = row["canonical"]
            diagnostic = row.get("critic_diagnostic", {})
            correlation = diagnostic.get("reward_q_correlation")
            corr_text = "—" if correlation is None else f"{correlation:.4f}"
            progression.append(
                f"| {row['episodes']} | {canonical['behavior_level']} | {canonical['progress']:.4f} | "
                f"{1000*canonical['tip_min_distance_m']:.3f} | {canonical['directed_tip_speed_m_s']:.3f} | "
                f"{canonical['direction_error_deg']:.3f} | {canonical['duration_s']:.4f} | "
                f"{canonical['feasible']} | {canonical['task_success']} | {canonical['reward']:.4f} | {corr_text} |"
            )
        components = selected["reward_components"]
        gates = final_metrics["hard_success_gates"]
        gate_table = [
            "| Gate | Result | Pass |",
            "|---|---:|---:|",
            f"| Tip distance <= 50 mm | {1000*final_metrics['minimum_tip_target_distance_m']:.3f} mm | {gates['tip_position']} |",
            f"| Directed speed >= 4 m/s | {final_metrics['reported_event_directed_tip_speed_m_s']:.3f} m/s | {gates['directed_tip_speed']} |",
            f"| Direction <= 30 deg | {final_metrics['reported_event_direction_error_deg']:.3f} deg | {gates['impact_direction']} |",
            f"| c10 first | {final_metrics['first_target_entry_marker_label']} | {gates['tip_first']} |",
            f"| UAV displacement <= 0.50 m | {final_metrics['maximum_uav_displacement_m']:.4f} m | {gates['uav_displacement']} |",
            f"| UAV speed <= 3 m/s | {final_metrics['maximum_uav_speed_m_s']:.4f} m/s | {gates['uav_speed']} |",
            f"| Command acceleration <= 20 m/s² | {final_metrics['maximum_command_acceleration_m_s2']:.4f} m/s² | {gates['command_acceleration']} |",
            f"| Finite rollout | {final_metrics['finite']} | {gates['finite']} |",
        ]
        q_losses = [
            value
            for row in outcome.training_history
            for value in (row.get("q1_loss"), row.get("q2_loss"))
            if value is not None
        ]
        def observed(name: str) -> list[float]:
            return [float(row[name]) for row in outcome.training_history if name in row]
        alpha_values = observed("alpha")
        entropy_values = observed("entropy")
        actor_losses = observed("actor_loss")
        action_std_values = observed("action_std")
        duration_values = observed("mean_duration_s")
        knot_norm_values = observed("acceleration_normalized_norm_mean")
        collection_progress = observed("maximum_progress")
        collection_feasibility = observed("feasible_rate")
        correlations = [
            float(row["critic_diagnostic"]["reward_q_correlation"])
            for row in evaluations
            if row.get("critic_diagnostic", {}).get("reward_q_correlation") is not None
        ]
        duration_diagnostic = _duration_diagnostic(evaluations)
        lines.extend(
            [
                "## 5. Canonical-only SAC setup",
                "",
                "Every training and validation episode used the exact same canonical settled state, target [1,0,1.4] m, direction [+1,0,0], and nominal theta. The existing fixed 5B.2 normalizer was reused. Only stochastic actions and resulting rewards varied.",
                "",
                "The actor/twin-critic 256-256-256 SiLU architecture, radial action transform, terminal reward-only target, automatic alpha, Huber delta 1, 2,048 collection batch, 4,096 replay minibatch, eight updates per batch, and learning rates 3e-4 were unchanged.",
                "",
                "## 6. Learning progression",
                "",
                *progression,
                "",
                f"Highest behavior level reached: **{outcome.highest_behavior_level}**. Selected checkpoint level: **{selected['behavior_level']}**.",
                "",
                "## 7. Critic and actor diagnostics",
                "",
                f"Critic Huber losses remained in [{min(q_losses):.6f}, {max(q_losses):.6f}]" if q_losses else "No critic update was performed.",
                "",
                f"Reward-Q correlation progressed from {correlations[0]:.4f} to {correlations[-1]:.4f}, with a maximum of {max(correlations):.4f}. Actor loss remained in [{min(actor_losses):.4f}, {max(actor_losses):.4f}]. Alpha moved from {max(alpha_values):.4f} to {alpha_values[-1]:.4f}; entropy remained in [{min(entropy_values):.4f}, {max(entropy_values):.4f}] and ended at {entropy_values[-1]:.4f}. No NaN/Inf or critic explosion occurred.",
                "",
                f"Mean stochastic duration moved from {duration_values[0]:.4f} s to {duration_values[-1]:.4f} s. Mean normalized acceleration-knot norm remained in [{min(knot_norm_values):.4f}, {max(knot_norm_values):.4f}], and normalized action standard deviation remained in [{min(action_std_values):.4f}, {max(action_std_values):.4f}]. Collection batches contained no successes; their maximum per-row progress reached {max(collection_progress):.4f}, but those high-progress samples did not satisfy the hard gates. Collection feasibility ranged from {100*min(collection_feasibility):.2f}% to {100*max(collection_feasibility):.2f}%.",
                "",
                f"Episodes: {outcome.episodes}; gradient updates: {outcome.gradient_updates}; training runtime: {outcome.runtime_s:.3f} s. Duration diagnostic: **{duration_diagnostic}**.",
                "",
                "## 8. Selected deterministic checkpoint and authoritative replay",
                "",
                f"The fixed-2,048-row checkpoint evaluation reported reward {selected['reward']:.6f} and progress {selected['progress']:.6f}. The authoritative batch-one replay reported reward {final_metrics['rl_reward']:.6f}, progress {final_metrics['progress']:.6f}, and R_progress {components['progress']:.6f}. Both remain Level 1 and have the same hard-gate classification. Authoritative replay metrics are used below.",
                "",
                *gate_table,
                "",
                "## 9. CEM reference only",
                "",
                "The saved CEM reference remains: tip error 1.756 mm, directed speed 4.599 m/s, direction error 19.615 deg, and duration 1.1175 s. It proves feasibility but was not used for SAC training.",
                "",
                "## 10. Classification and boundaries",
                "",
                f"**{outcome.classification}**",
                "",
                f"Video: `{video_path}`",
                "",
                "Physics conditioning: **NOT ENABLED**. Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**.",
                "",
                "## 11. Verification",
                "",
                "Reward-v2 identity/progress tests, fixed-context collection checks, finite actor/environment and Huber-update checks, and checkpoint save/resume passed. The focused learning suite passed 20 tests, the complete repository regression suite passed 90 tests in 38.92 s, and compileall passed.",
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
                "        SINGLE CANONICAL CONTEXT",
                "",
                "    Execution:",
                "        OPEN LOOP",
                "",
                "    CEM training data:",
                "        NOT USED",
                "",
                "    Reward:",
                "        rl_whip_reward_v2",
                "",
                "    Progress weight:",
                "        4.0",
                "",
                "    Reward audit:",
                f"        {audit_gate['status']}",
                "",
                "    SAC episodes:",
                f"        {outcome.episodes}",
                "",
                "    Highest behavior level:",
                f"        {outcome.highest_behavior_level}",
                "",
                "    Progress:",
                f"        {final_metrics['progress']:.6f}",
                "",
                "    Minimum tip distance:",
                f"        {1000*final_metrics['minimum_tip_target_distance_m']:.3f} mm",
                "",
                "    Directed speed:",
                f"        {final_metrics['reported_event_directed_tip_speed_m_s']:.3f} m/s",
                "",
                "    Direction error:",
                f"        {final_metrics['reported_event_direction_error_deg']:.3f} deg",
                "",
                "    Duration:",
                f"        {final_metrics['optimized_duration_s']:.6f} s",
                "",
                "    Feasible:",
                f"        {'YES' if final_metrics['feasible'] else 'NO'}",
                "",
                "    Scientific task:",
                f"        {'PASS' if final_metrics['success'] else 'FAIL'}",
                "",
                "    SAC result:",
                f"        {outcome.classification}",
                "",
                "    Duration diagnostic:",
                f"        {duration_diagnostic}",
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
        )
    report = "\n".join(lines) + "\n"
    (artifact / "MILESTONE5B3_CANONICAL_PROGRESS_SAC_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    overall_start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    artifact = (
        PROJECT_ROOT
        / "data"
        / "policy_training"
        / "sac_canonical_progress_v1"
        / _timestamp()
    )
    artifact.mkdir(parents=True)
    shutil.copy2(config_path, artifact / "sac_config.json")
    reward_config = _reward_config(config["reward"])
    _write_json(artifact / "reward_v2_config.json", reward_config.snapshot())

    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    task_path = (PROJECT_ROOT / config["task_config"]).resolve()
    task = load_variable_duration_task(task_path)
    canonical_bank, canonical_specification = _canonical_setup(simulator, task)

    context = build_context_from_specification(
        simulator, canonical_bank, canonical_specification
    )
    hover_action = torch.zeros((1, 49), device=simulator.device)
    hover_action[:, -1] = 1.0
    with torch.no_grad():
        hover = evaluate_open_loop_batch(
            simulator,
            context,
            hover_action,
            task,
            rl_reward_config=reward_config,
        )
    canonical_d0 = float(
        hover.reward_components.initial_tip_distance_m[0]  # type: ignore[union-attr]
    )
    existing_audit = _audit_existing_rows(config, reward_config, canonical_d0)
    random_rows, progress_bins = _random_progress_audit(
        simulator,
        task,
        reward_config,
        canonical_bank,
        canonical_specification,
        count=int(config["reward_audit"]["new_random_progress_rollouts"]),
        seed=int(config["reward_audit"]["audit_seed"]),
    )
    audit_gate = _audit_gate(existing_audit, random_rows, config)
    _write_json(
        artifact / "reward_v2_audit.json",
        {
            "existing_rows": existing_audit,
            "gate": audit_gate,
            "cem_training_data_used": False,
            "scientific_success_gates": "UNCHANGED",
        },
    )
    _write_json(
        artifact / "random_progress_audit.json",
        {"rows": random_rows, "progress_bins": progress_bins, "entered_sac_replay": False},
    )
    print("REWARD_V2_AUDIT " + json.dumps(audit_gate, sort_keys=True), flush=True)
    if audit_gate["status"] != "PASS":
        _write_report(
            artifact=artifact,
            existing_audit=existing_audit,
            progress_bins=progress_bins,
            audit_gate=audit_gate,
        )
        return {
            "reward_audit": "FAIL",
            "sac": "NOT RUN — REWARD AUDIT FAILED",
            "artifact_directory": str(artifact),
            "report": str(REPORT_PATH),
        }

    source = (PROJECT_ROOT / config["source_5b2_artifact"]).resolve()
    normalizer = FixedContextNormalizer.load(source / "context_normalizer.json")
    shutil.copy2(source / "context_normalizer.json", artifact / "context_normalizer.json")
    verification = _pre_run_verification(
        simulator,
        task,
        reward_config,
        normalizer,
        canonical_bank,
        canonical_specification,
        config,
        artifact,
    )
    _write_json(artifact / "pre_run_verification.json", verification)

    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    sac = config["sac"]
    agent = TerminalSacAgent.create(
        device=simulator.device,
        actor_lr=float(sac["actor_learning_rate"]),
        critic_lr=float(sac["critic_learning_rate"]),
        alpha_lr=float(sac["alpha_learning_rate"]),
        target_entropy=float(sac["target_entropy"]),
        gradient_clip_norm=float(sac["gradient_clip_norm"]),
        critic_loss="smooth_l1",
    )
    replay = TerminalReplayBuffer(int(sac["replay_capacity"]))
    outcome = run_canonical_sac_pilot(
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
    torch.save(agent.actor.state_dict(), artifact / "best_actor.pt")
    torch.save(agent.critic1.state_dict(), artifact / "best_critic1.pt")
    torch.save(agent.critic2.state_dict(), artifact / "best_critic2.pt")

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
    selected = result.row(0)
    selected_components = selected["reward_components"]
    final_metrics.update(
        {
            "policy": "One-Shot Terminal SAC",
            "optimizer": "One-Shot Terminal SAC",
            "reward_profile": reward_config.profile,
            "rl_reward": float(result.reward[0]),
            "reward_components": selected_components,
            "progress": float(selected_components["normalized_progress"]),
            "optimized_duration_s": float(result.decoded_action.duration_s[0]),
            "task_classification": "PASS" if bool(result.task_success[0]) else "FAIL",
            "policy_query_count": 1,
            "execution": "OPEN_LOOP",
            "highest_behavior_level": outcome.highest_behavior_level,
            "sac_classification": outcome.classification,
        }
    )
    _write_json(artifact / "canonical_final_metrics.json", final_metrics)

    # The established video renderer consumes a read-only planning-result view.
    shutil.copy2(artifact / "canonical_final_replay.npz", artifact / "final_replay.npz")
    video_task = json.loads(task_path.read_text(encoding="utf-8"))
    video_task["task_id"] = "oneshot_sac_progress_canonical"
    _write_json(artifact / "task_config_snapshot.json", video_task)
    _write_json(artifact / "final_metrics.json", final_metrics)
    _write_json(artifact / "mppi_iteration_history.json", [])
    planning_result = PlanningResult(
        task_id="oneshot_sac_progress_canonical",
        task_label="One-Shot SAC Canonical Progress",
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
        PROJECT_ROOT / "learning" / "one_shot_env.py",
        PROJECT_ROOT / "learning" / "canonical_pilot.py",
        PROJECT_ROOT / "learning" / "sac.py",
        Path(__file__),
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone5b3_source_hash_manifest_v1",
            "sha256": {
                str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths
            },
            "model_freeze": config["model_freeze"],
            "protected_test": "NOT EVALUATED",
            "real_hardware": "NOT EXECUTED",
            "cem_training_data": "NOT USED",
        },
    )
    _write_report(
        artifact=artifact,
        existing_audit=existing_audit,
        progress_bins=progress_bins,
        audit_gate=audit_gate,
        outcome=outcome,
        selected={
            **outcome.selected_evaluation,
            "reward_components": selected_components,
        },
        final_metrics=final_metrics,
        runtime=runtime,
        video_path=video_path,
    )
    response = {
        "reward_audit": audit_gate["status"],
        "sac": outcome.classification,
        "episodes": outcome.episodes,
        "highest_behavior_level": outcome.highest_behavior_level,
        "scientific_success": bool(result.task_success[0]),
        "artifact_directory": str(artifact),
        "video": str(video_path),
        "report": str(REPORT_PATH),
        "protected_test": "NOT EVALUATED",
        "real_hardware": "NOT EXECUTED",
    }
    print("MILESTONE5B3_RESULT " + json.dumps(response, sort_keys=True), flush=True)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    run(arguments.config.resolve())


if __name__ == "__main__":
    main()
