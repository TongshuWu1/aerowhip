"""Milestone 5B.2: audit a bounded RL reward, then run one gated SAC pilot."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
)
from learning.normalization import FixedContextNormalizer
from learning.pilot_training import (
    concatenate_state_banks,
    evaluate_pilot_policy,
    nearest_state_indices,
    run_focused_sac_pilot,
    sample_pilot_context_specification,
    subset_state_bank,
)
from learning.policy_context import build_policy_context
from learning.replay import TerminalReplayBuffer
from learning.reward_audit import analytic_reward_slices, audit_saved_replay
from learning.sac import TerminalSacAgent, squash_raw_action
from learning.state_bank import InitialStateBank, initial_state_bank_from_state
from learning.training import load_training_checkpoint
from planning.artifacts import final_replay_metrics, save_command_csv, save_replay_npz
from planning.cem_task import load_variable_duration_task
from planning.rl_reward import RLWhipRewardConfig
from planning.rollout import hover_preroll
from simulator.parameters import SimulatorSettings
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "learning" / "rl_reward_audit_sac_pilot_v1.json"
REPORT_PATH = PROJECT_ROOT / "MILESTONE5B2_RL_REWARD_AND_SAC_PILOT_REPORT.md"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+0000", "Z")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _safe(value.item())
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
    if config.get("schema") != "rl_reward_audit_sac_pilot_v1":
        raise ValueError("Unsupported Milestone 5B.2 configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The production model freeze is immutable.")
    if config.get("context_dimension") != 83 or config.get("action_dimension") != 49:
        raise ValueError("The 83-D context and 49-D action contracts are frozen.")
    if config.get("duration_interpretation") != "PHASE1_POLICY_DURATION_ENVELOPE":
        raise ValueError("The duration range must be labeled as a Phase-1 envelope.")
    policy = config["artifact_policy"]
    if (
        policy["real_flight_authorized"]
        or policy["protected_test_evaluation_allowed"]
        or policy["cem_training_data_allowed"]
    ):
        raise ValueError("5B.2 must remain simulated, protected, and teacher-free.")
    if float(config["reward"]["additional_scale_divisor"]) != 1.0:
        raise ValueError("rl_whip_reward_v1 must not receive another scale divisor.")


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


def _live_audit_rows(simulator, task, reward_config, canonical_bank, canonical_spec, count, seed):
    context = build_context_from_specification(simulator, canonical_bank, canonical_spec)
    hover_action = torch.zeros((1, 49), device=simulator.device)
    hover_action[:, -1] = 1.0
    env = __import__("learning.one_shot_env", fromlist=["evaluate_open_loop_batch"])
    with torch.no_grad():
        hover = env.evaluate_open_loop_batch(
            simulator,
            context,
            hover_action,
            task,
            rl_reward_config=reward_config,
        )

    def row_from_result(result, index, label):
        row = result.row(index)
        components = row["reward_components"]
        return {
            "label": label,
            "source_directory": None,
            "source_replay": "new audit-only rollout",
            "hard_success": row["task_success"],
            "feasible": row["feasible"],
            "tip_error_m": row["tip_min_distance_m"],
            "directed_speed_m_s": row["directed_tip_speed_m_s"],
            "direction_error_deg": row["direction_error_deg"],
            "tip_first": row["first_entry_marker"] == 10,
            "first_entry_marker": row["first_entry_marker"] or None,
            "maximum_uav_displacement_m": row["max_uav_displacement_m"],
            "maximum_uav_speed_m_s": row["max_uav_speed_m_s"],
            "maximum_command_acceleration_m_s2": row[
                "max_command_acceleration_m_s2"
            ],
            "legacy_reference_reward": -row["task_cost"],
            "R_strike": components["strike"],
            "R_success": components["success"],
            "R_safety": components["safety"],
            "R_non_tip": components["non_tip"],
            "R_control": components["control"],
            "R_time": components["time"],
            "R_RL": row["reward"],
        }

    rows = [row_from_result(hover, 0, "safe_hover_no_whip")]
    if count:
        repeated = ContextSpecification(
            torch.zeros(count, dtype=torch.int64),
            canonical_spec.target_position_local_m.repeat(count, 1),
            canonical_spec.desired_direction_local.repeat(count, 1),
            "audit_random",
        )
        random_context = build_context_from_specification(
            simulator, canonical_bank, repeated
        )
        generator = torch.Generator(device=simulator.device).manual_seed(seed)
        raw = torch.randn((count, 49), generator=generator, device=simulator.device)
        random_action, _ = squash_raw_action(raw)
        with torch.no_grad():
            random_result = env.evaluate_open_loop_batch(
                simulator,
                random_context,
                random_action,
                task,
                rl_reward_config=reward_config,
            )
        rows.extend(
            row_from_result(random_result, index, f"bad_random_{index:02d}")
            for index in range(count)
        )
    return rows


def _plot_slices(slices: dict[str, np.ndarray], directory: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(slices["distance_m"], slices["distance_score"], linewidth=2)
    plt.xlabel("Tip-target distance [m]")
    plt.ylabel("Event reward")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(directory / "reward_distance_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for score, distance in zip(slices["speed_scores"], (0.02, 0.20, 0.80)):
        plt.plot(slices["directed_speed_m_s"], score, label=f"d={distance:.2f} m")
    plt.xlabel("Directed tip speed [m/s]")
    plt.ylabel("Event reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(directory / "reward_speed_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    for score, distance in zip(slices["direction_scores"], (0.02, 0.20, 0.80)):
        plt.plot(slices["direction_angle_deg"], score, label=f"d={distance:.2f} m")
    plt.xlabel("Impact-direction error [deg]")
    plt.ylabel("Event reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(directory / "reward_direction_curves.png", dpi=160)
    plt.close()

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(slices["uav_displacement_m"], slices["displacement_penalty"])
    axes[0].set_xlabel("Maximum UAV displacement [m]")
    axes[0].set_ylabel("RL safety component")
    axes[1].plot(slices["uav_speed_m_s"], slices["speed_penalty"])
    axes[1].set_xlabel("Maximum UAV speed [m/s]")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(directory / "reward_safety_curves.png", dpi=160)
    plt.close(figure)


def _audit_gate(rows: list[dict[str, Any]], slices: dict[str, np.ndarray]) -> dict[str, Any]:
    indexed = {row["label"]: row for row in rows}
    random_rewards = [
        row["R_RL"] for row in rows if row["label"].startswith("bad_random_")
    ]
    rewards = np.asarray([row["R_RL"] for row in rows], dtype=np.float64)
    checks = {
        "strong_above_fragile": indexed["strong_success"]["R_RL"]
        > indexed["fragile_success"]["R_RL"],
        "fragile_above_wrong_direction": indexed["fragile_success"]["R_RL"]
        > indexed["close_wrong_direction"]["R_RL"],
        "wrong_direction_above_close_slow": indexed["close_wrong_direction"]["R_RL"]
        > indexed["close_slow"]["R_RL"],
        "close_slow_above_hover": indexed["close_slow"]["R_RL"]
        > indexed["safe_hover_no_whip"]["R_RL"],
        "hover_above_median_random": indexed["safe_hover_no_whip"]["R_RL"]
        > float(np.median(random_rewards)),
        "unsafe_aggressive_below_close_slow": indexed["old_aggressive_mppi"]["R_RL"]
        < indexed["close_slow"]["R_RL"],
        "failed_sac_below_hover": indexed["failed_sac"]["R_RL"]
        < indexed["safe_hover_no_whip"]["R_RL"],
        "strong_reward_clearly_high": indexed["strong_success"]["R_RL"] > 8.0,
        "all_audit_rewards_finite": bool(np.isfinite(rewards).all()),
        "moderate_observed_range": bool(rewards.min() >= -30.0 and rewards.max() <= 12.0),
        "distance_monotone": bool(np.all(np.diff(slices["distance_score"]) <= 1.0e-10)),
        "speed_curves_monotone": bool(
            np.all(np.diff(slices["speed_scores"], axis=1) >= -1.0e-9)
        ),
        "direction_curves_monotone": bool(
            np.all(np.diff(slices["direction_scores"], axis=1) <= 1.0e-9)
        ),
        "analytic_slices_finite": bool(
            all(np.isfinite(value).all() for value in slices.values())
        ),
        "no_large_analytic_step": bool(
            max(
                np.max(np.abs(np.diff(slices["distance_score"]))),
                np.max(np.abs(np.diff(slices["speed_scores"], axis=1))),
                np.max(np.abs(np.diff(slices["direction_scores"], axis=1))),
            )
            < 0.75
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed_reward_minimum": float(rewards.min()),
        "observed_reward_maximum": float(rewards.max()),
        "random_reward_median": float(np.median(random_rewards)),
    }


def _record_policy(
    agent,
    normalizer,
    simulator,
    task,
    reward_config,
    bank,
    specification,
):
    context = build_context_from_specification(simulator, bank, specification)
    with torch.no_grad():
        action = agent.actor(
            normalizer.normalize(context.to_tensor()), deterministic=True
        ).deterministic_mean_action
        result = __import__(
            "learning.one_shot_env", fromlist=["evaluate_open_loop_batch"]
        ).evaluate_open_loop_batch(
            simulator,
            context,
            action,
            task,
            record_trajectory=True,
            rl_reward_config=reward_config,
        )
    if result.trajectory is None:
        raise RuntimeError("Requested final deterministic replay was not recorded.")
    return result, result.trajectory


def _write_report(
    *,
    artifact: Path,
    config: dict[str, Any],
    audit_rows: list[dict[str, Any]],
    audit_gate: dict[str, Any],
    outcome=None,
    final_validation=None,
    canonical_metrics=None,
    runtime=None,
) -> None:
    table = [
        "| Trajectory | Success | Feasible | Tip [mm] | Directed [m/s] | Direction [deg] | R_strike | R_success | R_safety | R_non_tip | R_control | R_time | R_RL | Legacy reference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit_rows:
        if row["label"].startswith("bad_random_") and row["label"] != "bad_random_00":
            continue
        table.append(
            "| {label} | {success} | {feasible} | {tip:.3f} | {speed:.3f} | {direction:.3f} | {strike:.4f} | {bonus:.4f} | {safety:.4f} | {non_tip:.4f} | {control:.4f} | {time:.4f} | {total:.4f} | {legacy:.4f} |".format(
                label=row["label"],
                success=row["hard_success"],
                feasible=row["feasible"],
                tip=1000.0 * row["tip_error_m"],
                speed=row["directed_speed_m_s"],
                direction=row["direction_error_deg"],
                strike=row["R_strike"],
                bonus=row["R_success"],
                safety=row["R_safety"],
                non_tip=row["R_non_tip"],
                control=row["R_control"],
                time=row["R_time"],
                total=row["R_RL"],
                legacy=row["legacy_reference_reward"],
            )
        )
    reward = config["reward"]
    lines = [
        "# Milestone 5B.2 — RL Reward Audit and Focused One-Shot SAC Pilot",
        "",
        "## 1. CEM reward versus RL reward",
        "",
        "The legacy `legacy_run_online_strike_margin_tuned_v4` reward remains unchanged as the optimizer/reference objective. SAC uses the separate `rl_whip_reward_v1`; no CEM action, elite, or trajectory entered replay or policy training.",
        "",
        "## 2. Frozen scientific contract",
        "",
        "The model remained `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. The one-query, fully open-loop 83-D context and 49-D maneuver interface remained unchanged. Scientific success gates, hard feasibility, nominal theta, DDER, geometry, and solver settings were unchanged.",
        "",
        "## 3. Exact RL reward",
        "",
        f"`P=exp(-d^2/(2*{reward['sigma_position_m']}^2))`, `V=tanh(v_dir/{reward['directed_velocity_scale_m_s']})`, and direction alignment is smoothly suppressed at zero speed then converges to `0.5*(1+alignment)`. `R_strike=max_t(2P+2PV+PD)`. Total reward is `R_strike + 5*success - 2*(clipped normalized safety violations) - 1*non_tip_first - 0.05*effort - 0.05*smoothness - 0.10*t_hit` for successful episodes. No `/100`, batch normalization, or global clipping is applied.",
        "",
        "Each normalized safety violation is independently clipped at 4 only for RL conditioning. Hard feasibility remains unclipped and unchanged.",
        "",
        "## 4. Existing-trajectory reward audit",
        "",
        *table,
        "",
        f"All {len(audit_rows)} audit rows, including every random rollout, are stored in `reward_audit_results.json`.",
        "",
        "## 5. Analytic reward slices",
        "",
        "Distance, directed-speed, direction, displacement, and speed slices were finite and were checked for the expected monotonic behavior. The saved plots are `reward_distance_curve.png`, `reward_speed_curves.png`, `reward_direction_curves.png`, and `reward_safety_curves.png`.",
        "",
        "## 6. Reward-audit gate",
        "",
        f"**REWARD_AUDIT: {audit_gate['status']}**",
        "",
        f"Observed audit reward range: [{audit_gate['observed_reward_minimum']:.6f}, {audit_gate['observed_reward_maximum']:.6f}].",
        "",
    ]
    if outcome is None:
        lines.extend(
            [
                "The gate failed, so SAC was not run.",
                "",
                "## Final summary",
                "",
                "    Model: MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI",
                "    Policy: One-Shot Terminal SAC",
                "    Execution: OPEN LOOP",
                "    Scientific success gates: UNCHANGED",
                "    CEM training data: NOT USED",
                "    Optimizer/reference reward: legacy_run_online_strike_margin_tuned_v4",
                "    SAC reward: rl_whip_reward_v1",
                f"    RL reward observed range: [{audit_gate['observed_reward_minimum']:.6f}, {audit_gate['observed_reward_maximum']:.6f}]",
                f"    Reward audit: {audit_gate['status']}",
                "    SAC pilot: NOT RUN — REWARD AUDIT FAILED",
                "    Physics conditioning: NOT ENABLED",
                "    Protected test: NOT EVALUATED",
                "    Real hardware: NOT EXECUTED",
            ]
        )
    else:
        baseline = outcome.evaluation_history[0]
        final = final_validation
        lines.extend(
            [
                "## 7. Focused SAC pilot",
                "",
                "The pilot used the canonical settled state plus the 128 closest states from the existing 5B training bank. Validation used the canonical state and 32 closest states from the separate existing validation bank. Targets were 50% exact canonical and 50% sampled from x=[0.95,1.05], y=[-0.05,0.05], and canonical local z +/-0.03 m.",
                "",
                "The actor and twin critics remained 256-256-256 SiLU networks. The radial stochastic transform and one-terminal-decision target remained unchanged. Critic regression used SmoothL1/Huber with delta 1.0. Training began after 10,240 actor-generated episodes, used eight updates per 2,048 rollouts, and used no demonstrations.",
                "",
                "## 8. Pilot progression and final result",
                "",
                f"Untrained canonical tip error: {1000*baseline['canonical']['median_tip_error_m']:.3f} mm; final best-checkpoint canonical tip error: {1000*final['canonical']['median_tip_error_m']:.3f} mm.",
                "",
                f"Untrained held-out success/feasibility: {100*baseline['nearcanonical']['hard_success_rate']:.2f}% / {100*baseline['nearcanonical']['feasible_rate']:.2f}%. Final held-out success/feasibility: {100*final['nearcanonical']['hard_success_rate']:.2f}% / {100*final['nearcanonical']['feasible_rate']:.2f}%.",
                "",
                f"Collected episodes: {outcome.episodes}; gradient updates: {outcome.gradient_updates}; training runtime: {outcome.runtime_s:.3f} s.",
                "",
                f"Pilot classification: **{outcome.classification}**.",
                "",
                "## 9. Comparison with failed 5B",
                "",
                "Milestone 5B used the legacy optimizer reward divided by 100 and produced critic losses up to approximately 1e15 with zero success. This pilot uses the separately bounded RL reward and Huber critics. The comparison is methodological; the model, policy interface, and hard task gates remain fixed.",
                "",
                "## 10. Safety boundary",
                "",
                "CEM training data: **NOT USED**. Physics conditioning: **NOT ENABLED**. Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**.",
                "",
                "## Final summary",
                "",
                "    Model:",
                "        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI",
                "",
                "    Policy:",
                "        One-Shot Terminal SAC",
                "",
                "    Execution:",
                "        OPEN LOOP",
                "",
                "    Scientific success gates:",
                "        UNCHANGED",
                "",
                "    CEM training data:",
                "        NOT USED",
                "",
                "    Optimizer/reference reward:",
                "        legacy_run_online_strike_margin_tuned_v4",
                "",
                "    SAC reward:",
                "        rl_whip_reward_v1",
                "",
                "    RL reward observed range:",
                f"        [{runtime['observed_reward_minimum']:.6f}, {runtime['observed_reward_maximum']:.6f}]",
                "",
                "    Reward audit:",
                f"        {audit_gate['status']}",
                "",
                "    SAC pilot episodes:",
                f"        {outcome.episodes}",
                "",
                "    Canonical:",
                f"        tip error = {1000*final['canonical']['median_tip_error_m']:.3f} mm",
                f"        directed speed = {final['canonical']['median_directed_tip_speed_m_s']:.3f} m/s",
                f"        direction error = {final['canonical']['median_direction_error_deg']:.3f} deg",
                f"        feasible = {bool(final['canonical']['feasible_rate'] == 1.0)}",
                f"        result = {'PASS' if final['canonical']['hard_success_rate'] == 1.0 else 'FAIL'}",
                "",
                "    Near-canonical held-out:",
                f"        success = {100*final['nearcanonical']['hard_success_rate']:.2f} %",
                f"        feasible = {100*final['nearcanonical']['feasible_rate']:.2f} %",
                "",
                "    SAC pilot:",
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
        )
    report = "\n".join(lines) + "\n"
    (artifact / "MILESTONE5B2_RL_REWARD_AND_SAC_PILOT_REPORT.md").write_text(
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
        / "rl_reward_audit_sac_pilot_v1"
        / _timestamp()
    )
    artifact.mkdir(parents=True)
    shutil.copy2(config_path, artifact / "sac_pilot_config.json")
    reward_config = _reward_config(config["reward"])
    _write_json(artifact / "rl_reward_config.json", reward_config.snapshot())

    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    task_path = (PROJECT_ROOT / config["task_config"]).resolve()
    task = load_variable_duration_task(task_path)
    canonical_bank, canonical_spec = _canonical_setup(simulator, task)

    audit_config = config["reward_audit"]
    labels = (
        "strong_success",
        "fragile_success",
        "close_wrong_direction",
        "close_slow",
        "old_aggressive_mppi",
        "failed_sac",
    )
    audit_rows = [
        audit_saved_replay(
            PROJECT_ROOT / audit_config[label], label=label, config=reward_config
        )
        for label in labels
    ]
    audit_rows.extend(
        _live_audit_rows(
            simulator,
            task,
            reward_config,
            canonical_bank,
            canonical_spec,
            int(audit_config["new_bad_random_rollouts"]),
            int(audit_config["audit_seed"]),
        )
    )
    slices = analytic_reward_slices(reward_config)
    _plot_slices(slices, artifact)
    gate = _audit_gate(audit_rows, slices)
    _write_json(
        artifact / "reward_audit_dataset_manifest.json",
        {
            "existing_artifact_rows": labels,
            "new_safe_hover_rollouts": 1,
            "new_random_rollouts": int(audit_config["new_bad_random_rollouts"]),
            "new_cem_runs": 0,
            "cem_artifacts_used_for_training": False,
            "sources": {row["label"]: row["source_replay"] for row in audit_rows},
        },
    )
    _write_json(
        artifact / "reward_audit_results.json",
        {"rows": audit_rows, "gate": gate},
    )
    print("REWARD_AUDIT " + json.dumps(gate, sort_keys=True), flush=True)
    if gate["status"] != "PASS":
        _write_report(
            artifact=artifact,
            config=config,
            audit_rows=audit_rows,
            audit_gate=gate,
        )
        return {
            "reward_audit": "FAIL",
            "sac_pilot": "NOT RUN — REWARD AUDIT FAILED",
            "artifact_directory": str(artifact),
            "report": str(REPORT_PATH),
        }

    source = (PROJECT_ROOT / config["source_5b_training_artifact"]).resolve()
    original_training = InitialStateBank.load(
        source / "training_state_bank.npz", source / "training_state_bank_manifest.json"
    )
    original_validation = InitialStateBank.load(
        source / "validation_state_bank.npz", source / "validation_state_bank_manifest.json"
    )
    normalizer = FixedContextNormalizer.load(source / "context_normalizer.json")
    shutil.copy2(source / "context_normalizer.json", artifact / "context_normalizer.json")
    nearest = config["pilot_state_subset"]
    train_indices, train_distances = nearest_state_indices(
        simulator,
        original_training,
        canonical_bank,
        canonical_spec.target_position_local_m[0],
        canonical_spec.desired_direction_local[0],
        normalizer,
        count=int(nearest["training_nearest_count"]),
    )
    validation_indices, validation_distances = nearest_state_indices(
        simulator,
        original_validation,
        canonical_bank,
        canonical_spec.target_position_local_m[0],
        canonical_spec.desired_direction_local[0],
        normalizer,
        count=int(nearest["validation_nearest_count"]),
    )
    near_training = subset_state_bank(
        original_training, train_indices, label="pilot_training_128_nearest"
    )
    training_bank = concatenate_state_banks(
        canonical_bank, near_training, label="canonical_plus_128_nearest"
    )
    validation_bank = subset_state_bank(
        original_validation, validation_indices, label="pilot_validation_32_nearest"
    )
    training_bank.save(
        artifact / "pilot_training_state_bank.npz",
        artifact / "pilot_training_state_bank_manifest.json",
    )
    validation_bank.save(
        artifact / "pilot_validation_state_bank.npz",
        artifact / "pilot_validation_state_bank_manifest.json",
    )
    _write_json(
        artifact / "nearest_state_selection.json",
        {
            "training_source_indices": train_indices.tolist(),
            "training_normalized_distances": train_distances.tolist(),
            "validation_source_indices": validation_indices.tolist(),
            "validation_normalized_distances": validation_distances.tolist(),
            "canonical_state_included_at_training_index": 0,
        },
    )
    validation_generator = torch.Generator().manual_seed(
        int(config["validation"]["target_seed"])
    )
    heldout_spec = sample_pilot_context_specification(
        validation_bank,
        count=int(config["validation"]["nearcanonical_count"]),
        generator=validation_generator,
        canonical_target_local_m=canonical_spec.target_position_local_m[0],
        canonical_fraction=0.50,
        split="nearcanonical_heldout",
    )
    np.savez_compressed(
        artifact / "pilot_validation_contexts.npz",
        state_indices=heldout_spec.state_indices.numpy(),
        target_position_local_m=heldout_spec.target_position_local_m.numpy(),
        desired_direction_local=heldout_spec.desired_direction_local.numpy(),
        excluded_from_training=np.asarray(True),
    )

    sac = config["sac"]
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
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
    outcome = run_focused_sac_pilot(
        agent=agent,
        normalizer=normalizer,
        replay=replay,
        simulator=simulator,
        task=task,
        reward_config=reward_config,
        training_bank=training_bank,
        canonical_bank=canonical_bank,
        validation_bank=validation_bank,
        canonical_specification=canonical_spec,
        heldout_specification=heldout_spec,
        canonical_target_local_m=canonical_spec.target_position_local_m[0],
        config=config,
        artifact_directory=artifact,
    )
    restore_generator = torch.Generator().manual_seed(0)
    load_training_checkpoint(
        outcome.best_checkpoint, agent=agent, context_generator=restore_generator
    )
    final_validation = {
        "canonical": evaluate_pilot_policy(
            agent,
            normalizer,
            simulator,
            task,
            reward_config,
            canonical_bank,
            canonical_spec,
        ),
        "nearcanonical": evaluate_pilot_policy(
            agent,
            normalizer,
            simulator,
            task,
            reward_config,
            validation_bank,
            heldout_spec,
        ),
    }
    _write_json(artifact / "pilot_training_history.json", list(outcome.training_history))
    _write_json(artifact / "pilot_evaluation_history.json", list(outcome.evaluation_history))
    _write_json(artifact / "final_validation_metrics.json", final_validation)
    torch.save(agent.actor.state_dict(), artifact / "best_actor.pt")
    torch.save(agent.critic1.state_dict(), artifact / "best_critic1.pt")
    torch.save(agent.critic2.state_dict(), artifact / "best_critic2.pt")

    canonical_result, canonical_replay = _record_policy(
        agent,
        normalizer,
        simulator,
        task,
        reward_config,
        canonical_bank,
        canonical_spec,
    )
    save_command_csv(artifact / "canonical_fullstate_command.csv", canonical_replay)
    save_replay_npz(artifact / "canonical_final_replay.npz", canonical_replay, task)
    canonical_metrics = final_replay_metrics(canonical_replay, task, settings)
    canonical_metrics.update(
        {
            "policy": "One-Shot Terminal SAC",
            "reward_profile": reward_config.profile,
            "rl_reward": float(canonical_result.reward[0]),
            "reward_components": canonical_result.row(0)["reward_components"],
            "optimized_duration_s": float(canonical_result.decoded_action.duration_s[0]),
            "task_classification": "PASS"
            if bool(canonical_result.task_success[0])
            else "FAIL",
            "policy_query_count": 1,
            "execution": "OPEN_LOOP",
        }
    )
    _write_json(artifact / "canonical_final_metrics.json", canonical_metrics)

    # Save one representative held-out deterministic replay: prefer a success,
    # otherwise choose the highest RL reward under the frozen validation set.
    held_context = build_context_from_specification(simulator, validation_bank, heldout_spec)
    with torch.no_grad():
        held_action = agent.actor(
            normalizer.normalize(held_context.to_tensor()), deterministic=True
        ).deterministic_mean_action
        held_result = __import__(
            "learning.one_shot_env", fromlist=["evaluate_open_loop_batch"]
        ).evaluate_open_loop_batch(
            simulator,
            held_context,
            held_action,
            task,
            rl_reward_config=reward_config,
        )
    success_indices = torch.nonzero(held_result.task_success, as_tuple=False).reshape(-1)
    representative_index = (
        int(success_indices[0])
        if success_indices.numel()
        else int(torch.argmax(held_result.reward))
    )
    representative_spec = ContextSpecification(
        heldout_spec.state_indices[representative_index : representative_index + 1],
        heldout_spec.target_position_local_m[representative_index : representative_index + 1],
        heldout_spec.desired_direction_local[representative_index : representative_index + 1],
        "representative_nearcanonical",
    )
    _, representative_replay = _record_policy(
        agent,
        normalizer,
        simulator,
        task,
        reward_config,
        validation_bank,
        representative_spec,
    )
    save_replay_npz(
        artifact / "representative_nearcanonical_replay.npz",
        representative_replay,
        task,
    )

    training_rewards = [
        value
        for row in outcome.training_history
        for value in (row["minimum_reward"], row["maximum_reward"])
    ]
    observed_minimum = min(
        gate["observed_reward_minimum"], min(training_rewards, default=float("inf"))
    )
    observed_maximum = max(
        gate["observed_reward_maximum"], max(training_rewards, default=-float("inf"))
    )
    runtime = {
        "overall_runtime_s": time.perf_counter() - overall_start,
        "pilot_training_runtime_s": outcome.runtime_s,
        "collected_episodes": outcome.episodes,
        "gradient_updates": outcome.gradient_updates,
        "observed_reward_minimum": observed_minimum,
        "observed_reward_maximum": observed_maximum,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(simulator.device) / 1024**2,
        "device": torch.cuda.get_device_name(simulator.device),
    }
    _write_json(artifact / "pilot_runtime.json", runtime)
    source_paths = [
        config_path,
        task_path,
        PROJECT_ROOT / "planning" / "rl_reward.py",
        PROJECT_ROOT / "planning" / "variable_duration.py",
        PROJECT_ROOT / "learning" / "one_shot_env.py",
        PROJECT_ROOT / "learning" / "reward_audit.py",
        PROJECT_ROOT / "learning" / "pilot_training.py",
        PROJECT_ROOT / "learning" / "sac.py",
        Path(__file__),
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone5b2_source_hash_manifest_v1",
            "sha256": {
                str(path.relative_to(PROJECT_ROOT)): _sha256(path)
                for path in source_paths
            },
            "model_freeze": config["model_freeze"],
            "protected_test": "NOT EVALUATED",
            "real_hardware": "NOT EXECUTED",
        },
    )
    _write_report(
        artifact=artifact,
        config=config,
        audit_rows=audit_rows,
        audit_gate=gate,
        outcome=outcome,
        final_validation=final_validation,
        canonical_metrics=canonical_metrics,
        runtime=runtime,
    )
    result = {
        "reward_audit": gate["status"],
        "sac_pilot": outcome.classification,
        "episodes": outcome.episodes,
        "canonical_success": bool(canonical_result.task_success[0]),
        "nearcanonical_success_rate": final_validation["nearcanonical"][
            "hard_success_rate"
        ],
        "artifact_directory": str(artifact),
        "report": str(REPORT_PATH),
        "protected_test": "NOT EVALUATED",
        "real_hardware": "NOT EXECUTED",
    }
    print("MILESTONE5B2_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    run(arguments.config.resolve())


if __name__ == "__main__":
    main()
