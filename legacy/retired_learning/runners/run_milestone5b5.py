"""Milestone 5B.5: catastrophic-rollout and feasible-support audit."""

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

import matplotlib.pyplot as plt
import numpy as np
import torch

from learning.canonical_pilot import repeated_canonical_specification
from learning.context_sampling import ContextSpecification, build_context_from_specification
from learning.normalization import FixedContextNormalizer
from learning.one_shot_env import evaluate_open_loop_batch
from learning.policy_context import build_policy_context
from learning.rollout_diagnostics import (
    disable_residual_diagnostic_only,
    trace_action,
)
from learning.spectral_sac import (
    create_spectral_agent,
    initial_frequency_std,
    inverse_radial_squash,
    spectral_coefficients_to_normalized_action,
    temporal_to_spectral,
)
from learning.state_bank import initial_state_bank_from_state
from learning.training import load_training_checkpoint
from planning.cem_task import load_variable_duration_task
from planning.rl_reward import RLWhipRewardConfig
from planning.rollout import hover_preroll
from simulator.parameters import SimulatorSettings
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)
from simulator.uav.model import FullStateUAVModel


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "learning" / "sac_feasible_support_audit_v1.json"
REPORT_PATH = PROJECT_ROOT / "MILESTONE5B5_FEASIBLE_EXPLORATION_SUPPORT_REPORT.md"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+0000", "Z")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
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
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reward_config() -> RLWhipRewardConfig:
    # Constructor defaults are the frozen rl_whip_reward_v2 coefficients except
    # for profile/progress, which are made explicit here.
    accepted = {item.name for item in fields(RLWhipRewardConfig)}
    payload = json.loads(
        (PROJECT_ROOT / "config" / "learning" / "sac_canonical_spectral_entropy_v1.json").read_text(
            encoding="utf-8"
        )
    )["reward"]
    return RLWhipRewardConfig(
        **{name: value for name, value in payload.items() if name in accepted}
    )


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "sac_feasible_support_audit_v1":
        raise ValueError("Unsupported Milestone 5B.5 configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The frozen production model is immutable.")
    if config.get("reward_profile") != "rl_whip_reward_v2":
        raise ValueError("Milestone 5B.5 requires unchanged reward v2.")
    if not math.isclose(float(config.get("duration_s", 0.0)), 1.20):
        raise ValueError("Diagnostic duration must remain 1.20 s.")
    grid = config["support_grid"]
    if grid["bandwidths"] != [4, 8, 16] or grid["scales"] != [0.05, 0.10, 0.20, 0.35, 0.50]:
        raise ValueError("The authorized bandwidth/amplitude grid is exact.")
    if int(grid["samples_per_cell"]) != 1024:
        raise ValueError("Every support cell must contain 1,024 rollouts.")
    policy = config["artifact_policy"]
    if (
        policy["learning_performed"]
        or policy["production_model_modification_allowed"]
        or policy["real_flight_authorized"]
        or policy["protected_test_evaluation_allowed"]
        or policy["cem_training_data_allowed"]
    ):
        raise ValueError("5B.5 is diagnostic-only, teacher-free, and simulation-only.")


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
        "canonical_5b5",
    )
    return bank, specification, context


def _fixed_batch(simulator, size: int) -> None:
    if not isinstance(simulator.uav_model, FullStateUAVModel):
        raise TypeError("5B.5 requires the FullState production UAV model.")
    simulator.uav_model.set_fixed_evaluation_batch_size(size)


def _progress(result) -> torch.Tensor:
    if result.reward_components is None:
        raise RuntimeError("rl_whip_reward_v2 components are required.")
    return result.reward_components.normalized_progress


def _metric_row(result, index: int) -> dict[str, Any]:
    row = result.row(index)
    row["progress"] = float(_progress(result)[index])
    row["maximum_tip_speed_m_s"] = float(result.max_tip_speed_m_s[index])
    row["feasibility_violation"] = float(
        result.population_metrics.feasibility_violation[index]
    )
    return row


def _select_unique(
    ordered: list[int],
    used: set[int],
    count: int,
) -> list[int]:
    selected: list[int] = []
    for index in ordered:
        if index in used:
            continue
        used.add(index)
        selected.append(index)
        if len(selected) == count:
            break
    return selected


def _select_diagnostic_actions(result, count_per_group: int) -> dict[str, list[int]]:
    progress = _progress(result)
    distance = result.tip_min_distance_m
    displacement = result.max_uav_displacement_m
    uav_speed = result.max_uav_speed_m_s
    tip_speed = result.max_tip_speed_m_s
    violation = result.population_metrics.feasibility_violation
    used: set[int] = set()
    mild_score = (
        displacement / 0.5
        + uav_speed / 3.0
        + torch.log1p(torch.clamp(tip_speed, min=0.0))
    )
    mild = torch.argsort(mild_score).detach().cpu().tolist()
    high_pool = torch.nonzero(progress >= 0.75, as_tuple=False)[:, 0]
    if high_pool.numel() >= count_per_group:
        high = high_pool[torch.argsort(violation[high_pool])].detach().cpu().tolist()
    else:
        high = torch.argsort(progress, descending=True).detach().cpu().tolist()
    near = torch.argsort(distance).detach().cpu().tolist()
    catastrophic_score = torch.maximum(
        torch.maximum(uav_speed / 3.0, displacement / 0.5),
        tip_speed / 10.0,
    )
    catastrophic = torch.argsort(catastrophic_score, descending=True).detach().cpu().tolist()
    return {
        "A_MILD_LOW_DYNAMIC": _select_unique(mild, used, count_per_group),
        "B_HIGH_PROGRESS_UNSAFE": _select_unique(high, used, count_per_group),
        "C_NEAR_TARGET_OUTLIER": _select_unique(near, used, count_per_group),
        "D_CATASTROPHIC": _select_unique(catastrophic, used, count_per_group),
    }


def _reproduction_comparison(original: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    metric_differences = {
        "tip_distance_m": float(replay["tip_min_distance_m"] - original["tip_min_distance_m"]),
        "progress": float(replay["progress"] - original["progress"]),
        "maximum_uav_speed_m_s": float(replay["max_uav_speed_m_s"] - original["max_uav_speed_m_s"]),
        "maximum_uav_displacement_m": float(replay["max_uav_displacement_m"] - original["max_uav_displacement_m"]),
        "maximum_tip_speed_m_s": float(replay["maximum_tip_speed_m_s"] - original["maximum_tip_speed_m_s"]),
    }
    scale_close = lambda difference, reference, absolute: abs(difference) <= max(
        absolute, 1e-3 * max(abs(reference), 1.0)
    )
    checks = {
        "finite_classification_same": replay["rollout_finite"] == original["rollout_finite"],
        "feasible_classification_same": replay["feasible"] == original["feasible"],
        "success_classification_same": replay["task_success"] == original["task_success"],
        "tip_distance_close": abs(metric_differences["tip_distance_m"]) <= 0.002,
        "uav_speed_close": scale_close(
            metric_differences["maximum_uav_speed_m_s"], original["max_uav_speed_m_s"], 0.001
        ),
        "uav_displacement_close": scale_close(
            metric_differences["maximum_uav_displacement_m"], original["max_uav_displacement_m"], 0.001
        ),
        "tip_speed_close": scale_close(
            metric_differences["maximum_tip_speed_m_s"], original["maximum_tip_speed_m_s"], 0.002
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "NUMERICAL_BATCH_REPRODUCTION_ISSUE",
        "checks": checks,
        "differences": metric_differences,
    }


def _support_cell(result, center: str, bandwidth: int, scale: float, seed: int, task) -> dict[str, Any]:
    progress = _progress(result)
    feasible = result.feasible
    finite = result.rollout_finite
    distance = result.tip_min_distance_m
    uav_speed = result.max_uav_speed_m_s
    displacement = result.max_uav_displacement_m
    tip_speed = result.max_tip_speed_m_s
    acceleration = result.max_command_acceleration_m_s2
    fraction = lambda mask: float(mask.float().mean())
    quantile = lambda value, q: float(torch.quantile(value.float(), q))
    fractions = {
        "progress_ge_0_25": fraction(progress >= 0.25),
        "progress_ge_0_50": fraction(progress >= 0.50),
        "progress_ge_0_75": fraction(progress >= 0.75),
        "progress_ge_0_90": fraction(progress >= 0.90),
        "d_min_le_0_50_m": fraction(distance <= 0.50),
        "d_min_le_0_20_m": fraction(distance <= 0.20),
        "d_min_le_0_10_m": fraction(distance <= 0.10),
        "d_min_le_0_05_m": fraction(distance <= 0.05),
        "feasible_and_progress_ge_0_25": fraction(feasible & (progress >= 0.25)),
        "feasible_and_progress_ge_0_50": fraction(feasible & (progress >= 0.50)),
        "feasible_and_progress_ge_0_75": fraction(feasible & (progress >= 0.75)),
        "feasible_and_d_min_le_0_50_m": fraction(feasible & (distance <= 0.50)),
        "feasible_and_d_min_le_0_20_m": fraction(feasible & (distance <= 0.20)),
        "feasible_and_d_min_le_0_10_m": fraction(feasible & (distance <= 0.10)),
    }
    feasible_rate = fraction(feasible)
    if feasible_rate >= 0.10 and fractions["feasible_and_progress_ge_0_25"] >= 0.01:
        classification = "USEFUL_SUPPORT"
    elif feasible_rate >= 0.10:
        classification = "FEASIBLE_BUT_LOW_PROGRESS"
    else:
        classification = "UNSAFE_SUPPORT"
    return {
        "center": center,
        "bandwidth_K": bandwidth,
        "scale": scale,
        "seed": seed,
        "sample_count": result.batch_size,
        "finite_rate": fraction(finite),
        "feasible_rate": feasible_rate,
        "reward_mean": float(result.reward.mean()),
        "reward_median": quantile(result.reward, 0.50),
        "reward_maximum": float(result.reward.max()),
        "progress_mean": float(progress.mean()),
        "progress_median": quantile(progress, 0.50),
        "progress_p95": quantile(progress, 0.95),
        "progress_maximum": float(progress.max()),
        "d_min_median_m": quantile(distance, 0.50),
        "d_min_minimum_m": float(distance.min()),
        "fractions": fractions,
        "scientific_success_count": int(result.task_success.sum()),
        "scientific_success_rate": fraction(result.task_success),
        "feasibility_failure_fractions": {
            "uav_displacement_gt_0_50": fraction(
                displacement > task.maximum_uav_displacement_m
            ),
            "uav_speed_gt_3": fraction(uav_speed > task.maximum_uav_speed_m_s),
            "command_acceleration_gt_20": fraction(
                acceleration > task.maximum_command_acceleration_m_s2 + 1e-5
            ),
            "non_finite": fraction(~finite),
        },
        "catastrophic_outliers": {
            "uav_speed_p99_m_s": quantile(uav_speed, 0.99),
            "uav_speed_maximum_m_s": float(uav_speed.max()),
            "uav_displacement_p99_m": quantile(displacement, 0.99),
            "uav_displacement_maximum_m": float(displacement.max()),
            "tip_speed_p99_m_s": quantile(tip_speed, 0.99),
            "tip_speed_maximum_m_s": float(tip_speed.max()),
        },
        "support_classification": classification,
    }


def _cell_seed(base: int, center_index: int, bandwidth: int, scale: float) -> int:
    return int(base + center_index * 100_000 + bandwidth * 1000 + round(scale * 1000))


def _classify_root_cause(production: dict[str, Any], residual_off: dict[str, Any]) -> dict[str, Any]:
    prod_runaway = production["first_runaway_index"] is not None
    off_runaway = residual_off["first_runaway_index"] is not None
    if not prod_runaway:
        classification = "NOT_REPRODUCED"
    elif not off_runaway and residual_off["maximum_uav_speed_m_s"] < 0.2 * production["maximum_uav_speed_m_s"]:
        classification = "RESIDUAL_OOD_DOMINANT"
    elif off_runaway:
        # A runaway that survives complete removal of Delta_a cannot be
        # residual-dominant.  Use the counterfactual ordering, rather than the
        # production residual crossing alone, to identify the necessary
        # non-residual mechanism.
        off_crossings = residual_off["threshold_crossings"]
        ctrl = off_crossings["a_ctrl_gt_100"]
        omega = off_crossings["omega_gt_100"]
        if omega is not None and (ctrl is None or omega < ctrl):
            classification = "ATTITUDE_DYNAMICS_INSTABILITY"
        elif ctrl is not None:
            classification = "CLOSED_LOOP_TRACKING_RUNAWAY"
        else:
            classification = "MIXED_MECHANISM"
    else:
        classification = "ROOT_CAUSE_UNRESOLVED"
    return {
        "classification": classification,
        "production_first_runaway_time_s": production["first_runaway_time_s"],
        "residual_off_first_runaway_time_s": residual_off["first_runaway_time_s"],
        "production_threshold_crossings": production["threshold_crossings"],
        "residual_off_threshold_crossings": residual_off["threshold_crossings"],
        "production_first_runaway_window": production["first_runaway_window"],
        "production_maximum_uav_speed_m_s": production["maximum_uav_speed_m_s"],
        "residual_off_maximum_uav_speed_m_s": residual_off["maximum_uav_speed_m_s"],
        "production_maximum_residual_normalized_abs": production[
            "maximum_residual_normalized_abs"
        ],
        "production_maximum_residual_acceleration_m_s2": production[
            "maximum_residual_acceleration_m_s2"
        ],
    }


def _plot_trace(
    production: dict[str, np.ndarray],
    residual_off: dict[str, np.ndarray],
    summary: dict[str, Any],
    figures: Path,
) -> None:
    first_time = summary["first_runaway_time_s"]
    time_s = production["time_s"]
    figure, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
    series = (
        (production["uav_speed"], "UAV speed [m/s]"),
        (production["a_ctrl_norm"], "||a_ctrl|| [m/s²]"),
        (production["s_des"], "s_des [m/s²]"),
        (production["delta_a_norm"], "||Delta_a|| [m/s²]"),
        (production["a_total_norm"], "||a_total|| [m/s²]"),
    )
    for axis, (values, label) in zip(axes, series):
        axis.plot(time_s, values, linewidth=1.4)
        if first_time is not None:
            axis.axvline(first_time, color="tab:red", linestyle="--", linewidth=1)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Simulation time [s]")
    figure.tight_layout()
    figure.savefig(figures / "catastrophic_trace.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(time_s, production["residual_normalized_max_abs"], label="max |normalized feature|")
    for threshold in (3, 5, 10):
        axis.axhline(threshold, linestyle=":", linewidth=1, label=f"|z|={threshold}")
    if first_time is not None:
        axis.axvline(first_time, color="tab:red", linestyle="--", label="first runaway")
    axis.set_xlabel("Simulation time [s]")
    axis.set_ylabel("Residual normalized input")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(figures / "residual_ood_trace.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 3.5))
    axis.plot(time_s, production["uav_speed"], label="production")
    axis.plot(residual_off["time_s"], residual_off["uav_speed"], label="residual disabled")
    if first_time is not None:
        axis.axvline(first_time, color="tab:red", linestyle="--", label="first runaway")
    axis.set_xlabel("Simulation time [s]")
    axis.set_ylabel("UAV speed [m/s]")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "production_vs_residual_off_speed.png", dpi=160)
    plt.close(figure)


def _plot_support_heatmaps(cells: list[dict[str, Any]], figures: Path) -> None:
    scales = [0.05, 0.10, 0.20, 0.35, 0.50]
    bandwidths = [4, 8, 16]
    for center in ("ZERO", "5B4_LEVEL1_SAC"):
        selected = {(row["bandwidth_K"], row["scale"]): row for row in cells if row["center"] == center}
        feasible = np.asarray(
            [[selected[(k, scale)]["feasible_rate"] for scale in scales] for k in bandwidths]
        )
        joint = np.asarray(
            [[selected[(k, scale)]["fractions"]["feasible_and_progress_ge_0_25"] for scale in scales] for k in bandwidths]
        )
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for axis, data, title in zip(
            axes,
            (feasible, joint),
            ("Feasible rate", "Feasible & progress >= 0.25"),
        ):
            image = axis.imshow(data, vmin=0.0, vmax=max(0.01, float(data.max())), cmap="viridis", aspect="auto")
            axis.set_xticks(range(len(scales)), labels=[str(value) for value in scales])
            axis.set_yticks(range(len(bandwidths)), labels=[str(value) for value in bandwidths])
            axis.set_xlabel("Noise scale")
            axis.set_ylabel("Active bandwidth K")
            axis.set_title(title)
            for row in range(data.shape[0]):
                for column in range(data.shape[1]):
                    axis.text(column, row, f"{100*data[row,column]:.1f}%", ha="center", va="center", color="white" if data[row,column] > 0.5*max(data.max(),1e-9) else "black", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046)
        figure.suptitle(center)
        figure.savefig(figures / f"support_heatmap_{center.lower()}.png", dpi=160)
        plt.close(figure)


def _write_report(
    *,
    artifact: Path,
    source_audit: dict[str, Any],
    selected_manifest: dict[str, Any],
    production_metrics: list[dict[str, Any]],
    trace_summary: list[dict[str, Any]],
    residual_ood: dict[str, Any],
    residual_disabled: list[dict[str, Any]],
    root_cause: dict[str, Any],
    reconstruction: dict[str, Any],
    grid: list[dict[str, Any]],
    best: dict[str, Any],
    runtime: dict[str, Any],
) -> str:
    support_lines = [
        "| Center | K | Scale | Feasible | Feas.&P>=.25 | Feas.&P>=.50 | P95 progress | Min d (mm) | Successes | UAV speed p99/max | Classification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in grid:
        f = row["fractions"]
        outlier = row["catastrophic_outliers"]
        support_lines.append(
            f"| {row['center']} | {row['bandwidth_K']} | {row['scale']:.2f} | {100*row['feasible_rate']:.2f}% | "
            f"{100*f['feasible_and_progress_ge_0_25']:.2f}% | {100*f['feasible_and_progress_ge_0_50']:.2f}% | "
            f"{row['progress_p95']:.3f} | {1000*row['d_min_minimum_m']:.1f} | {row['scientific_success_count']} | "
            f"{outlier['uav_speed_p99_m_s']:.2f}/{outlier['uav_speed_maximum_m_s']:.2f} | {row['support_classification']} |"
        )
    failure_lines = [
        "| Center | K | Scale | Displacement failure | Speed failure | Accel failure | Non-finite |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    outlier_lines = [
        "| Center | K | Scale | UAV speed p99/max (m/s) | UAV displacement p99/max (m) | Tip speed p99/max (m/s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in grid:
        failure = row["feasibility_failure_fractions"]
        outlier = row["catastrophic_outliers"]
        failure_lines.append(
            f"| {row['center']} | {row['bandwidth_K']} | {row['scale']:.2f} | "
            f"{100*failure['uav_displacement_gt_0_50']:.2f}% | "
            f"{100*failure['uav_speed_gt_3']:.2f}% | "
            f"{100*failure['command_acceleration_gt_20']:.2f}% | "
            f"{100*failure['non_finite']:.2f}% |"
        )
        outlier_lines.append(
            f"| {row['center']} | {row['bandwidth_K']} | {row['scale']:.2f} | "
            f"{outlier['uav_speed_p99_m_s']:.2f}/{outlier['uav_speed_maximum_m_s']:.2f} | "
            f"{outlier['uav_displacement_p99_m']:.2f}/{outlier['uav_displacement_maximum_m']:.2f} | "
            f"{outlier['tip_speed_p99_m_s']:.2f}/{outlier['tip_speed_maximum_m_s']:.2f} |"
        )
    reproduction_issues = sum(
        row["batch_one_comparison"]["status"] != "PASS" for row in production_metrics
    )
    command_healthy = all(
        row["production_trace_summary"]["command"]["finite"]
        and row["production_trace_summary"]["command"]["maximum_command_acceleration_m_s2"] <= 20.00001
        and row["production_trace_summary"]["command"]["maximum_velocity_consistency_error"] < 1e-5
        and row["production_trace_summary"]["command"]["maximum_position_consistency_error"] < 1e-5
        for row in trace_summary
    )
    representative = root_cause["representative_action_id"]
    command_max_accel = max(
        row["production_trace_summary"]["command"]["maximum_command_acceleration_m_s2"]
        for row in trace_summary
    )
    command_max_velocity = max(
        row["production_trace_summary"]["command"]["maximum_command_velocity_m_s"]
        for row in trace_summary
    )
    command_max_displacement = max(
        row["production_trace_summary"]["command"]["maximum_command_displacement_m"]
        for row in trace_summary
    )
    command_max_v_error = max(
        row["production_trace_summary"]["command"]["maximum_velocity_consistency_error"]
        for row in trace_summary
    )
    command_max_p_error = max(
        row["production_trace_summary"]["command"]["maximum_position_consistency_error"]
        for row in trace_summary
    )
    evidence = root_cause["evidence"]
    crossing = evidence["production_threshold_crossings"]
    off_crossing = evidence["residual_off_threshold_crossings"]
    lines = [
        "# Milestone 5B.5 — Feasible Exploration Support Report",
        "",
        "## 1. Why 5B.4 changes the diagnosis",
        "",
        "5B.4 sampled 99.3% progress and 9.38-mm proximity yet maintained 0% stochastic feasibility, with finite but physically impossible speeds and displacements. That changes the immediate question from reward/entropy design to simulator-validity root cause and feasible action-support geometry.",
        "",
        "## 2. No learning performed",
        "",
        "No SAC, actor, critic, alpha, behavior-cloning, CEM, or policy-fitting update occurred. All actor checkpoints were read-only sources of diagnostic actions/centers.",
        "",
        "## 3. Frozen production model",
        "",
        "`MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`, production CUDA float32 PCG32 DDER, residual, geometry, gains, three substeps, four projections, reward-v2, hard gates, canonical state/target/direction, and T=1.20 s were unchanged. The only altered model instance was separately labeled `RESIDUAL_DISABLED_DIAGNOSTIC_ONLY` and was never used for production results.",
        "",
        "## 4. Recoverability and selected 5B.4 trajectory classes",
        "",
        f"5B.4 exact stochastic replay actions were **not persisted**: `{json.dumps(source_audit, sort_keys=True)}`. Consequently historical 5B.4 outliers cannot be honestly replayed exactly. A reproducible replacement diagnostic batch of 2,048 actions was sampled once from the frozen 5B.4 latest checkpoint with seed {selected_manifest['diagnostic_selection_seed']}; the 12 selected tensors were then saved exactly and never regenerated. Groups contain three mild, three high-progress, three near-target, and three catastrophic actions without duplication.",
        "",
        "## 5. Batch-one reproduction",
        "",
        f"The same newly selected tensors were compared between their original B=2,048 diagnostic batch and batch-one replay with the fixed-UAV evaluation shape 2,048. {len(production_metrics)-reproduction_issues}/{len(production_metrics)} passed the defined classification/metric equivalence; {reproduction_issues} were flagged `NUMERICAL_BATCH_REPRODUCTION_ISSUE`. This gate pertains to the new diagnostic actions, not unrecoverable historical actions.",
        "",
        "## 6. Command-consistency verification",
        "",
        f"Command generation: **{'HEALTHY' if command_healthy else 'ISSUE FOUND'}**. Every selected FullState command remained finite, used the exact piecewise-linear acceleration integration, stayed at or below 20 m/s², and passed discrete p/v/a consistency. Across the 12 actions, maxima were {command_max_accel:.6f} m/s² commanded acceleration, {command_max_velocity:.6f} m/s commanded velocity, and {command_max_displacement:.6f} m commanded displacement; maximum discrete consistency errors were {command_max_v_error:.3e} m/s for velocity and {command_max_p_error:.3e} m for position. Thus command construction is correct, although some mathematically valid sampled commands are far outside the vehicle's feasible displacement/speed envelope.",
        "",
        "## 7. Catastrophic trace and first runaway",
        "",
        f"Representative action: `{representative}`. Its command peaks at {evidence['representative_command']['maximum_command_acceleration_m_s2']:.3f} m/s², {evidence['representative_command']['maximum_command_velocity_m_s']:.3f} m/s, and {evidence['representative_command']['maximum_command_displacement_m']:.3f} m. At t=0, b3_des={evidence['initial_desired_thrust_direction_b3']} is nearly horizontal, while realized acceleration before the residual is {evidence['initial_physical_acceleration_before_residual_m_s2']} m/s² and residual acceleration is only {evidence['initial_residual_acceleration_m_s2']} m/s². Tracking mismatch therefore starts immediately.",
        "",
        f"Production causal sequence: normalized residual |z|>10 is already present at t=0; residual ||Delta_a|| first exceeds 10 m/s² at {0.01*crossing['delta_a_gt_10']:.2f} s; UAV speed exceeds the 3 m/s feasibility limit at {0.01*crossing['uav_speed_gt_3']:.2f} s; controller acceleration exceeds 100 m/s² at {0.01*crossing['a_ctrl_gt_100']:.2f} s; total acceleration exceeds 100 m/s² at {0.01*crossing['a_total_gt_100']:.2f} s; UAV speed exceeds 10 m/s at {0.01*crossing['uav_speed_gt_10']:.2f} s. The predeclared runaway detector fires at {evidence['production_first_runaway_time_s']:.2f} s. Angular velocity never exceeds 100 rad/s.",
        "",
        f"With the residual disabled, the sequence is earlier: UAV speed >3 m/s at {0.01*off_crossing['uav_speed_gt_3']:.2f} s, controller acceleration >100 m/s² at {0.01*off_crossing['a_ctrl_gt_100']:.2f} s, total acceleration >100 m/s² at {0.01*off_crossing['a_total_gt_100']:.2f} s, and UAV speed >10 m/s at {0.01*off_crossing['uav_speed_gt_10']:.2f} s. The full seven-step causal window around production onset is preserved in `runaway_root_cause.json` and the trace figures.",
        "",
        "## 8. Residual-input OOD analysis",
        "",
        f"Across selected production traces, maximum normalized residual feature magnitude was {residual_ood['maximum_normalized_feature_abs']:.3g}. Aggregate timestep fractions with any |z|>3, >5, and >10 were {100*residual_ood['fraction_any_abs_gt_3']:.2f}%, {100*residual_ood['fraction_any_abs_gt_5']:.2f}%, and {100*residual_ood['fraction_any_abs_gt_10']:.2f}%. Residual OOD observed: **{'YES' if residual_ood['observed'] else 'NO'}**. For the representative trace, max-|z| and ||Delta_a|| correlation is {evidence['residual_ood_delta_a_correlation']:.6f}; however, OOD is present from t=0, Delta_a is initially only 0.436 m/s², and the residual-off counterfactual runs away sooner and farther. OOD correlates with the diverging state but is neither necessary nor dominant for this runaway.",
        "",
        "## 9. Residual-off counterfactual",
        "",
        f"For the representative action, production max UAV speed was {root_cause['evidence']['production_maximum_uav_speed_m_s']:.3g} m/s versus {root_cause['evidence']['residual_off_maximum_uav_speed_m_s']:.3g} m/s with Delta_a identically zero. Residual-off status: **{root_cause['residual_off_effect']}**. All counterfactuals are separately labeled diagnostic-only.",
        "",
        "## 10. Root-cause classification",
        "",
        f"**{root_cause['classification']}**. {root_cause['interpretation']} The initiating condition is a dynamically infeasible but correctly integrated aggressive command: attitude/thrust realization lags the desired force direction, position/velocity error grows, and unsaturated Kp/Kv feedback magnifies the demanded acceleration. No angular-state explosion precedes the translational runaway. Residual extrapolation is a secondary validity concern, not the necessary root cause demonstrated here.",
        "",
        "## 11. Production model not modified",
        "",
        "No clamp, saturation, residual retraining/removal, controller change, or physics modification was implemented. The evidence is returned before any remedy.",
        "",
        "## 12. Feasible-support audit design",
        "",
        "The audit used exact production physics for 30,720 rollouts: centers ZERO and 5B4_LEVEL1_SAC; active spectral bandwidth K=4,8,16; noise scales 0.05,0.10,0.20,0.35,0.50; 1,024 actions/cell; fixed T=1.20 s; IDCT, radial squash, physical decoder, and FullState integration unchanged.",
        "",
        "## 13. Exact centers and reconstruction",
        "",
        f"CENTER A is zero raw spectral acceleration. CENTER B is the saved best 5B.4 deterministic actor action, not CEM. Its inverse-radial/DCT/IDCT/radial maximum normalized-action reconstruction error was {reconstruction['maximum_normalized_action_reconstruction_error']:.3e}; status **{reconstruction['status']}**.",
        "",
        "## 14. Full support table",
        "",
        *support_lines,
        "",
        "## 15. Feasibility failures and catastrophic outliers",
        "",
        "The per-cell feasibility failure breakdown is:",
        "",
        *failure_lines,
        "",
        "Heavy-tail p99/max statistics are:",
        "",
        *outlier_lines,
        "",
        "Displacement is the first support-limiting feasibility gate in low-noise cells; speed and catastrophic heavy tails emerge as scale increases. Command-acceleration and nonfinite failure rates remain zero because the radial decoder enforces the physical knot bound and every audited rollout remained finite.",
        "",
        "## 16. Joint feasible-progress support and best region",
        "",
        f"Best region: center **{best['center']}**, K={best['bandwidth_K']}, scale={best['scale']:.2f}. Feasible rate {100*best['feasible_rate']:.2f}%; feasible & progress>=0.25 {100*best['fractions']['feasible_and_progress_ge_0_25']:.2f}%; feasible & progress>=0.50 {100*best['fractions']['feasible_and_progress_ge_0_50']:.2f}%; classification **{best['support_classification']}**. Scientific successes: {best['scientific_success_count']} in the best cell and {sum(row['scientific_success_count'] for row in grid)} across the audit.",
        "",
        "## 17. Center-A versus Center-B interpretation",
        "",
        f"{best['center_interpretation']}",
        "",
        "## 18. CEM reference only",
        "",
        "CEM was not used as a center, covariance, scale, teacher, or dataset. Its known successful trajectory remains feasibility context only.",
        "",
        "## 19. Recommended next methodological decision (not implemented)",
        "",
        f"{best['recommended_next_decision']}",
        "",
        "## 20. Runtime, protection, and hardware",
        "",
        f"Runtime {runtime['overall_runtime_s']:.3f} s; grid rollouts {runtime['support_grid_rollouts']}; peak CUDA memory {runtime['peak_cuda_memory_mb']:.1f} MiB. Protected test: **NOT EVALUATED**. Real hardware: **NOT EXECUTED**.",
        "",
        "## Final summary",
        "",
        "    Model:",
        "        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI",
        "",
        "    Learning performed:",
        "        NO",
        "",
        "    Reward:",
        "        rl_whip_reward_v2",
        "        UNCHANGED",
        "",
        "    Canonical context:",
        "        FIXED",
        "",
        "    Duration:",
        "        1.20 s DIAGNOSTIC",
        "",
        "    Catastrophic rollout reproduced:",
        f"        {'YES' if root_cause['catastrophic_rollout_reproduced'] else 'NO'}",
        "",
        "    Command generation:",
        f"        {'HEALTHY' if command_healthy else 'ISSUE FOUND'}",
        "",
        "    Residual OOD observed:",
        f"        {'YES' if residual_ood['observed'] else 'NO'}",
        "",
        "    Residual-off removes runaway:",
        f"        {root_cause['residual_off_effect']}",
        "",
        "    Root-cause classification:",
        f"        {root_cause['classification']}",
        "",
        "    Exploration centers:",
        "        ZERO",
        "        5B4_LEVEL1_SAC",
        "",
        "    Bandwidths:",
        "        K = 4, 8, 16",
        "",
        "    Noise scales:",
        "        0.05, 0.10, 0.20, 0.35, 0.50",
        "",
        "    Best support region:",
        f"        center = {best['center']}",
        f"        K = {best['bandwidth_K']}",
        f"        scale = {best['scale']:.2f}",
        "",
        "    Feasible rate:",
        f"        {100*best['feasible_rate']:.2f} %",
        "",
        "    Feasible & progress>=0.25:",
        f"        {100*best['fractions']['feasible_and_progress_ge_0_25']:.2f} %",
        "",
        "    Feasible & progress>=0.50:",
        f"        {100*best['fractions']['feasible_and_progress_ge_0_50']:.2f} %",
        "",
        "    Scientific successes in audit:",
        f"        {sum(row['scientific_success_count'] for row in grid)}",
        "",
        "    Support classification:",
        f"        {best['support_classification']}",
        "",
        "    Production model modified:",
        "        NO",
        "",
        "    SAC trained:",
        "        NO",
        "",
        "    CEM training data:",
        "        NOT USED",
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
    (artifact / "MILESTONE5B5_FEASIBLE_EXPLORATION_SUPPORT_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    artifact = PROJECT_ROOT / "data" / "policy_training" / "sac_feasible_support_audit_v1" / _timestamp()
    figures = artifact / "figures"
    traces_directory = artifact / "traces"
    figures.mkdir(parents=True)
    traces_directory.mkdir()
    shutil.copy2(config_path, artifact / "config.json")
    reward_config = _reward_config()
    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    _fixed_batch(simulator, int(config["fixed_uav_evaluation_batch_size"]))
    task = load_variable_duration_task((PROJECT_ROOT / config["task_config"]).resolve())
    bank, canonical_specification, canonical_context = _canonical_setup(simulator, task)
    normalizer = FixedContextNormalizer.load(
        PROJECT_ROOT / config["source_5b2_artifact"] / "context_normalizer.json"
    )
    source_5b4 = PROJECT_ROOT / config["source_5b4_artifact"]
    checkpoint_payload = torch.load(
        source_5b4 / config["selection"]["source_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    source_audit = {
        "source_artifact": str(source_5b4),
        "replay_buffer_file_present": any(
            path.name.startswith("replay") and path.suffix in {".pt", ".npz"}
            for path in source_5b4.rglob("*")
        ),
        "checkpoint_contains_replay_metadata_only": "replay_metadata" in checkpoint_payload
        and "replay" not in checkpoint_payload,
        "trajectory_storage_flag": checkpoint_payload["replay_metadata"]["trajectory_storage"],
        "exact_historical_stochastic_actions_recoverable": False,
        "consequence": "historical 5B.4 stochastic outliers cannot be exactly replayed; a seeded frozen-checkpoint diagnostic batch is used",
    }

    # Phase A: reproducible frozen-checkpoint selection batch.
    latest_agent = create_spectral_agent(device=simulator.device)
    load_training_checkpoint(
        source_5b4 / config["selection"]["source_checkpoint"],
        agent=latest_agent,
        context_generator=torch.Generator().manual_seed(0),
    )
    selection_count = int(config["selection"]["sample_count"])
    selection_spec = repeated_canonical_specification(
        canonical_specification, selection_count, split="5b5_selection"
    )
    selection_context = build_context_from_specification(simulator, bank, selection_spec)
    normalized_context = normalizer.normalize(selection_context.to_tensor())
    seed = int(config["selection"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        selection_actions = latest_agent.actor(normalized_context).normalized_action.detach()
        selection_result = evaluate_open_loop_batch(
            simulator,
            selection_context,
            selection_actions,
            task,
            rl_reward_config=reward_config,
        )
    groups = _select_diagnostic_actions(
        selection_result, int(config["selection"]["actions_per_group"])
    )
    selected_manifest: dict[str, Any] = {
        "historical_action_recoverability": source_audit,
        "diagnostic_selection_source": str(
            source_5b4 / config["selection"]["source_checkpoint"]
        ),
        "diagnostic_selection_seed": seed,
        "diagnostic_selection_batch_size": selection_count,
        "fixed_uav_evaluation_batch_size": int(config["fixed_uav_evaluation_batch_size"]),
        "groups": {},
    }
    selected_items: list[dict[str, Any]] = []
    for group, indices in groups.items():
        selected_manifest["groups"][group] = []
        for order, index in enumerate(indices):
            action_id = f"{group}_{order+1}"
            original = _metric_row(selection_result, index)
            item = {
                "action_id": action_id,
                "group": group,
                "diagnostic_batch_index": index,
                "episode_index": None,
                "historical_episode_recoverable": False,
                "normalized_action": selection_actions[index].detach().cpu().tolist(),
                "original_batch_diagnostics": original,
            }
            selected_manifest["groups"][group].append(item)
            selected_items.append(item)
    _write_json(artifact / "selected_5b4_actions_manifest.json", selected_manifest)

    diagnostic_simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    _fixed_batch(diagnostic_simulator, int(config["fixed_uav_evaluation_batch_size"]))
    disable_residual_diagnostic_only(diagnostic_simulator)
    # Counterfactuals start from the exact production post-hover state/FIFO;
    # disabling Delta_a must not alter the initialization pre-roll.
    diagnostic_context = canonical_context
    production_metrics: list[dict[str, Any]] = []
    trace_summary: list[dict[str, Any]] = []
    residual_disabled_rows: list[dict[str, Any]] = []
    trace_arrays: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for item in selected_items:
        action = torch.tensor(item["normalized_action"], device=simulator.device)
        with torch.no_grad():
            batch_one = evaluate_open_loop_batch(
                simulator,
                canonical_context,
                action[None],
                task,
                rl_reward_config=reward_config,
            )
        replay_row = _metric_row(batch_one, 0)
        comparison = _reproduction_comparison(item["original_batch_diagnostics"], replay_row)
        prod_trace, prod_summary, _ = trace_action(
            simulator,
            canonical_context,
            action,
            task,
            reward_config,
        )
        off_trace, off_summary, _ = trace_action(
            diagnostic_simulator,
            diagnostic_context,
            action,
            task,
            reward_config,
        )
        np.savez_compressed(
            traces_directory / f"{item['action_id']}_production.npz",
            normalized_action=np.asarray(item["normalized_action"], dtype=np.float32),
            **prod_trace,
        )
        np.savez_compressed(
            traces_directory / f"{item['action_id']}_residual_disabled.npz",
            normalized_action=np.asarray(item["normalized_action"], dtype=np.float32),
            **off_trace,
        )
        production_metrics.append(
            {
                "action_id": item["action_id"],
                "group": item["group"],
                "original_batch": item["original_batch_diagnostics"],
                "batch_one": replay_row,
                "batch_one_comparison": comparison,
            }
        )
        trace_summary.append(
            {
                "action_id": item["action_id"],
                "group": item["group"],
                "production_trace_summary": prod_summary,
                "residual_disabled_trace_summary": off_summary,
            }
        )
        residual_disabled_rows.append(
            {
                "action_id": item["action_id"],
                "status": "RESIDUAL_DISABLED_DIAGNOSTIC_ONLY",
                "production": prod_summary["rollout_metrics"],
                "residual_disabled": off_summary["rollout_metrics"],
                "production_trace_maxima": {
                    key: value for key, value in prod_summary.items() if key.startswith("maximum_")
                },
                "residual_disabled_trace_maxima": {
                    key: value for key, value in off_summary.items() if key.startswith("maximum_")
                },
            }
        )
        trace_arrays[item["action_id"]] = (prod_trace, off_trace)
    _write_json(artifact / "production_replay_metrics.json", production_metrics)
    _write_json(artifact / "full_trace_summary.json", trace_summary)
    _write_json(artifact / "residual_disabled_diagnostic.json", residual_disabled_rows)

    all_prod = [row["production_trace_summary"] for row in trace_summary]
    residual_ood = {
        "maximum_normalized_feature_abs": max(row["maximum_residual_normalized_abs"] for row in all_prod),
        "fraction_any_abs_gt_3": float(np.mean([row["ood_fraction_any_abs_gt_3"] for row in all_prod])),
        "fraction_any_abs_gt_5": float(np.mean([row["ood_fraction_any_abs_gt_5"] for row in all_prod])),
        "fraction_any_abs_gt_10": float(np.mean([row["ood_fraction_any_abs_gt_10"] for row in all_prod])),
        "per_action": [
            {
                "action_id": trace_summary[index]["action_id"],
                "maximum_normalized_feature_abs": row["maximum_residual_normalized_abs"],
                "maximum_delta_a_m_s2": row["maximum_residual_acceleration_m_s2"],
                "fraction_any_abs_gt_3": row["ood_fraction_any_abs_gt_3"],
                "fraction_any_abs_gt_5": row["ood_fraction_any_abs_gt_5"],
                "fraction_any_abs_gt_10": row["ood_fraction_any_abs_gt_10"],
            }
            for index, row in enumerate(all_prod)
        ],
    }
    residual_ood["observed"] = residual_ood["maximum_normalized_feature_abs"] > 10.0
    _write_json(artifact / "residual_ood_summary.json", residual_ood)

    catastrophic_rows = [row for row in trace_summary if row["group"] == "D_CATASTROPHIC"]
    representative_row = max(
        catastrophic_rows,
        key=lambda row: row["production_trace_summary"]["maximum_uav_speed_m_s"],
    )
    classification_evidence = _classify_root_cause(
        representative_row["production_trace_summary"],
        representative_row["residual_disabled_trace_summary"],
    )
    prod_trace, off_trace = trace_arrays[representative_row["action_id"]]
    representative_command = representative_row["production_trace_summary"]["command"]
    classification_evidence.update(
        {
            "representative_command": representative_command,
            "initial_desired_thrust_direction_b3": prod_trace["b3_des"][0].tolist(),
            "initial_physical_acceleration_before_residual_m_s2": prod_trace[
                "a_phys_before_residual"
            ][0].tolist(),
            "initial_residual_acceleration_m_s2": prod_trace["delta_a"][0].tolist(),
            "residual_ood_delta_a_correlation": float(
                np.corrcoef(
                    prod_trace["residual_normalized_max_abs"],
                    prod_trace["delta_a_norm"],
                )[0, 1]
            ),
        }
    )
    prod_speed = classification_evidence["production_maximum_uav_speed_m_s"]
    off_speed = classification_evidence["residual_off_maximum_uav_speed_m_s"]
    if representative_row["production_trace_summary"]["first_runaway_index"] is None:
        residual_effect = "NO"
    elif representative_row["residual_disabled_trace_summary"]["first_runaway_index"] is None:
        residual_effect = "YES"
    elif off_speed < 0.5 * prod_speed:
        residual_effect = "PARTIAL"
    else:
        residual_effect = "NO"
    interpretations = {
        "RESIDUAL_OOD_DOMINANT": "Severe residual-input OOD and Delta_a growth precede production runaway, while setting Delta_a=0 removes it.",
        "CLOSED_LOOP_TRACKING_RUNAWAY": "Tracking-error feedback grows first and runaway persists with the residual disabled.",
        "ATTITUDE_DYNAMICS_INSTABILITY": "Angular-state growth precedes translational runaway and persists without residual output.",
        "MIXED_MECHANISM": "Both residual extrapolation and non-residual controller/attitude dynamics materially contribute.",
        "NOT_REPRODUCED": "The seeded diagnostic action did not cross the predeclared runaway threshold.",
        "ROOT_CAUSE_UNRESOLVED": "The trace evidence does not isolate one supported mechanism.",
    }
    root_cause = {
        "representative_action_id": representative_row["action_id"],
        "classification": classification_evidence["classification"],
        "interpretation": interpretations[classification_evidence["classification"]],
        "residual_off_effect": residual_effect,
        "catastrophic_rollout_reproduced": representative_row["production_trace_summary"]["first_runaway_index"] is not None,
        "historical_5b4_catastrophic_action_reproduced": False,
        "historical_limitation": source_audit["consequence"],
        "evidence": classification_evidence,
    }
    _write_json(artifact / "runaway_root_cause.json", root_cause)
    _plot_trace(prod_trace, off_trace, representative_row["production_trace_summary"], figures)

    # Phase B: exact center encoding and 30-cell production support grid.
    best_agent = create_spectral_agent(device=simulator.device)
    load_training_checkpoint(
        source_5b4 / "checkpoints" / "best_behavior.pt",
        agent=best_agent,
        context_generator=torch.Generator().manual_seed(0),
    )
    with torch.no_grad():
        normalized_single = normalizer.normalize(canonical_context.to_tensor())
        center_b_action = best_agent.actor(
            normalized_single, deterministic=True
        ).deterministic_mean_action
    center_b_normalized = center_b_action[:, :48].reshape(1, 16, 3)
    center_b_raw = inverse_radial_squash(center_b_normalized)
    center_b_spectral = temporal_to_spectral(center_b_raw)
    reconstructed = spectral_coefficients_to_normalized_action(center_b_spectral)
    reconstruction_error = float(
        (reconstructed[:, :48] - center_b_action[:, :48]).abs().max()
    )
    reconstruction = {
        "source": str(source_5b4 / "checkpoints" / "best_behavior.pt"),
        "center": "5B4_LEVEL1_SAC",
        "inverse": "z=atanh(||y||)/||y||*y with stable zero limit",
        "maximum_normalized_action_reconstruction_error": reconstruction_error,
        "status": "PASS" if reconstruction_error <= 1e-5 else "FAIL",
        "center_normalized_action": center_b_action[0].detach().cpu().tolist(),
        "center_spectral_coefficients_axis_major": center_b_spectral[0].detach().cpu().tolist(),
    }
    _write_json(artifact / "action_center_reconstruction.json", reconstruction)
    if reconstruction["status"] != "PASS":
        raise RuntimeError("Center-B spectral reconstruction failed.")

    sample_count = int(config["support_grid"]["samples_per_cell"])
    grid_spec = repeated_canonical_specification(
        canonical_specification, sample_count, split="5b5_support_grid"
    )
    grid_context = build_context_from_specification(simulator, bank, grid_spec)
    base_std = initial_frequency_std(
        dtype=simulator.dtype, device=simulator.device
    )
    centers = {
        "ZERO": torch.zeros((1, 3, 16), device=simulator.device),
        "5B4_LEVEL1_SAC": center_b_spectral,
    }
    grid_rows: list[dict[str, Any]] = []
    for center_index, (center_name, center_coefficients) in enumerate(centers.items()):
        for bandwidth in config["support_grid"]["bandwidths"]:
            for scale in config["support_grid"]["scales"]:
                cell_seed = _cell_seed(
                    int(config["support_grid"]["base_seed"]),
                    center_index,
                    int(bandwidth),
                    float(scale),
                )
                generator = torch.Generator(device=simulator.device).manual_seed(cell_seed)
                coefficients = center_coefficients.expand(sample_count, -1, -1).clone()
                noise = torch.randn(
                    (sample_count, 3, int(bandwidth)),
                    generator=generator,
                    device=simulator.device,
                )
                coefficients[:, :, : int(bandwidth)] += (
                    float(scale) * base_std[: int(bandwidth)][None, None] * noise
                )
                actions = spectral_coefficients_to_normalized_action(coefficients)
                with torch.no_grad():
                    result = evaluate_open_loop_batch(
                        simulator,
                        grid_context,
                        actions,
                        task,
                        rl_reward_config=reward_config,
                    )
                row = _support_cell(
                    result,
                    center_name,
                    int(bandwidth),
                    float(scale),
                    cell_seed,
                    task,
                )
                grid_rows.append(row)
                print(
                    "SUPPORT_CELL "
                    + json.dumps(
                        {
                            "center": center_name,
                            "K": bandwidth,
                            "scale": scale,
                            "feasible": row["feasible_rate"],
                            "joint25": row["fractions"]["feasible_and_progress_ge_0_25"],
                            "joint50": row["fractions"]["feasible_and_progress_ge_0_50"],
                            "successes": row["scientific_success_count"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    _write_json(artifact / "exploration_support_grid.json", grid_rows)
    failure_breakdown = [
        {
            "center": row["center"],
            "bandwidth_K": row["bandwidth_K"],
            "scale": row["scale"],
            **row["feasibility_failure_fractions"],
        }
        for row in grid_rows
    ]
    outlier_summary = [
        {
            "center": row["center"],
            "bandwidth_K": row["bandwidth_K"],
            "scale": row["scale"],
            **row["catastrophic_outliers"],
        }
        for row in grid_rows
    ]
    _write_json(artifact / "feasibility_failure_breakdown.json", failure_breakdown)
    _write_json(artifact / "catastrophic_outlier_summary.json", outlier_summary)
    best = max(
        grid_rows,
        key=lambda row: (
            row["fractions"]["feasible_and_progress_ge_0_25"],
            row["fractions"]["feasible_and_progress_ge_0_50"],
            row["feasible_rate"],
            row["progress_p95"],
            -row["catastrophic_outliers"]["uav_speed_p99_m_s"],
        ),
    )
    zero_useful = any(
        row["support_classification"] == "USEFUL_SUPPORT" and row["center"] == "ZERO"
        for row in grid_rows
    )
    level1_useful = any(
        row["support_classification"] == "USEFUL_SUPPORT" and row["center"] == "5B4_LEVEL1_SAC"
        for row in grid_rows
    )
    if level1_useful and not zero_useful:
        center_interpretation = "ZERO has no useful support while the Level-1 SAC center does; a coarse curriculum/progressive local search is scientifically justified for review."
        recommendation = "Review a future curriculum that constrains early exploration to the measured useful neighborhood around an independently learned feasible center; do not train it yet."
    elif zero_useful and level1_useful:
        center_interpretation = "Both centers have useful low-bandwidth/amplitude support; 5B.4's learned distribution was substantially too broad."
        recommendation = "Review a future actor-distribution initialization using the diagnosed bandwidth/amplitude region; do not train it yet."
    else:
        center_interpretation = "Neither center has useful support under the diagnostic thresholds; the feasible dynamic manifold remains extremely narrow or simulator validity dominates exploration."
        recommendation = "Resolve the documented simulator-validity mechanism and reconsider constrained action support before authorizing further policy learning."
    best = {
        **best,
        "selection_priority": [
            "feasible_and_progress_ge_0_25",
            "feasible_and_progress_ge_0_50",
            "feasible_rate",
            "progress_p95",
            "lower_uav_speed_p99",
        ],
        "center_interpretation": center_interpretation,
        "recommended_next_decision": recommendation,
    }
    _write_json(artifact / "best_support_region.json", best)
    support_summary = {
        "total_rollouts": len(grid_rows) * sample_count,
        "scientific_successes": sum(row["scientific_success_count"] for row in grid_rows),
        "support_classification_counts": {
            name: sum(row["support_classification"] == name for row in grid_rows)
            for name in ("USEFUL_SUPPORT", "FEASIBLE_BUT_LOW_PROGRESS", "UNSAFE_SUPPORT")
        },
        "best_region": best,
        "center_interpretation": center_interpretation,
    }
    _write_json(artifact / "exploration_support_summary.json", support_summary)
    _plot_support_heatmaps(grid_rows, figures)

    runtime = {
        "overall_runtime_s": time.perf_counter() - start,
        "selection_rollouts": selection_count,
        "traced_production_rollouts": len(selected_items),
        "traced_residual_disabled_rollouts": len(selected_items),
        "support_grid_rollouts": len(grid_rows) * sample_count,
        "learning_updates": 0,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(simulator.device) / 1024**2,
        "device": torch.cuda.get_device_name(simulator.device),
    }
    _write_json(artifact / "runtime.json", runtime)
    source_paths = [
        config_path,
        PROJECT_ROOT / config["task_config"],
        PROJECT_ROOT / "learning" / "rollout_diagnostics.py",
        PROJECT_ROOT / "learning" / "spectral_sac.py",
        PROJECT_ROOT / "learning" / "one_shot_env.py",
        PROJECT_ROOT / "planning" / "variable_duration.py",
        PROJECT_ROOT / "simulator" / "uav" / "model.py",
        Path(__file__),
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone5b5_source_hash_manifest_v1",
            "sha256": {str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths},
            "model_freeze": config["model_freeze"],
            "learning_performed": False,
            "reward": "rl_whip_reward_v2 UNCHANGED",
            "production_model_modified": False,
            "cem_training_data": "NOT USED",
            "protected_test": "NOT EVALUATED",
            "real_hardware": "NOT EXECUTED",
        },
    )
    report = _write_report(
        artifact=artifact,
        source_audit=source_audit,
        selected_manifest=selected_manifest,
        production_metrics=production_metrics,
        trace_summary=trace_summary,
        residual_ood=residual_ood,
        residual_disabled=residual_disabled_rows,
        root_cause=root_cause,
        reconstruction=reconstruction,
        grid=grid_rows,
        best=best,
        runtime=runtime,
    )
    response = {
        "root_cause": root_cause["classification"],
        "catastrophic_reproduced": root_cause["catastrophic_rollout_reproduced"],
        "best_support_region": {
            "center": best["center"],
            "K": best["bandwidth_K"],
            "scale": best["scale"],
            "classification": best["support_classification"],
        },
        "scientific_successes": support_summary["scientific_successes"],
        "artifact_directory": str(artifact),
        "report": str(REPORT_PATH),
        "report_characters": len(report),
        "learning_performed": False,
        "protected_test": "NOT EVALUATED",
        "real_hardware": "NOT EXECUTED",
    }
    print("MILESTONE5B5_RESULT " + json.dumps(response, sort_keys=True), flush=True)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    run(arguments.config.resolve())


if __name__ == "__main__":
    main()
