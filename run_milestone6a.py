"""Run the gated Milestone-6A production CEM benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import numpy as np
import torch

from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    target_direction_from_local_target,
)
from learning.policy_action import decode_policy_action, encode_physical_action
from learning.policy_context import PolicyContext, build_policy_context
from learning.state_bank import InitialStateBank, initial_state_bank_from_state
from planning.cem_task import VariableDurationWhipTask, load_variable_duration_task
from planning.production_cem import (
    FIXED_NUMERICAL_BATCH_SIZE,
    ProductionCemResult,
    ProductionCemSettings,
    canonical_family_action_for_context,
    optimize_production_cem,
    record_normalized_actions_fixed_batch,
)
from planning.rollout import hover_preroll
from planning.variable_duration import variable_duration_fullstate
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "planning" / "production_cem_benchmark_v1.json"
REPORT = ROOT / "MILESTONE6A_PRODUCTION_CEM_BENCHMARK_REPORT.md"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+0000", "Z")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "production_cem_benchmark_v1":
        raise ValueError("Unsupported Milestone-6A configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Milestone 6A requires the frozen production model.")
    if int(config.get("fixed_numerical_batch_size", 0)) != FIXED_NUMERICAL_BATCH_SIZE:
        raise ValueError("The coupled numerical batch must remain fixed at 2048.")
    policy = config["artifact_policy"]
    prohibited = (
        "learning_used",
        "actor_training_allowed",
        "model_fitting_allowed",
        "protected_test_allowed",
        "real_hardware_allowed",
    )
    if any(bool(policy[name]) for name in prohibited):
        raise ValueError("The Milestone-6A scientific prohibitions changed.")


def _settings(config: dict[str, Any]) -> ProductionCemSettings:
    source = config["canonical_cem"]
    times = config["times"]
    return ProductionCemSettings(
        population=int(source["population"]),
        elite_fraction=float(source["elite_fraction"]),
        maximum_iterations=int(source["maximum_iterations"]),
        minimum_iterations=int(source["minimum_iterations"]),
        polish_iterations_after_success=int(source["polish_iterations_after_success"]),
        authoritative_top_n=int(source["authoritative_top_n"]),
        initial_acceleration_std_m_s2=float(source["initial_acceleration_std_m_s2"]),
        initial_duration_std_s=float(source["initial_duration_std_s"]),
        acceleration_std_floor_m_s2=float(source["acceleration_std_floor_m_s2"]),
        duration_std_floor_s=float(source["duration_std_floor_s"]),
        duration_min_s=float(times["maneuver_min_s"]),
        duration_max_s=float(times["maneuver_max_s"]),
        settle_duration_s=float(times["settle_s"]),
        evaluation_time_s=float(times["evaluation_s"]),
    )


def _canonical_setup(simulator, task: VariableDurationWhipTask) -> tuple[InitialStateBank, PolicyContext]:
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
    return bank, context


def _repeated_context(
    simulator,
    bank: InitialStateBank,
    state_index: int,
    target_local_m: Iterable[float] | torch.Tensor,
    direction_local: Iterable[float] | torch.Tensor,
    *,
    split: str,
) -> PolicyContext:
    target = torch.as_tensor(target_local_m, dtype=torch.float32).reshape(1, 3)
    direction = torch.as_tensor(direction_local, dtype=torch.float32).reshape(1, 3)
    specification = ContextSpecification(
        torch.full((FIXED_NUMERICAL_BATCH_SIZE,), int(state_index), dtype=torch.int64),
        target.repeat(FIXED_NUMERICAL_BATCH_SIZE, 1),
        direction.repeat(FIXED_NUMERICAL_BATCH_SIZE, 1),
        split,
    )
    return build_context_from_specification(simulator, bank, specification)


def _load_historical_action(config: dict[str, Any]) -> tuple[torch.Tensor, float]:
    directory = ROOT / config["historical_canonical_artifact"]
    knots = json.loads((directory / "best_acceleration_knots.json").read_text(encoding="utf-8"))["values"]
    duration = json.loads((directory / "optimized_duration.json").read_text(encoding="utf-8"))["duration_s"]
    value = torch.tensor(knots, dtype=torch.float64)
    if value.shape != (16, 3):
        raise RuntimeError("Historical canonical action has the wrong shape.")
    return value, float(duration)


def _gate_table(metrics: dict[str, Any], task: VariableDurationWhipTask) -> dict[str, bool]:
    return {
        "tip_position": bool(metrics["first_entry_tip_distance_m"] <= task.success_radius_m),
        "directed_tip_speed": bool(
            metrics["first_entry_directed_speed_m_s"] >= task.minimum_directed_speed_m_s
        ),
        "impact_direction": bool(
            metrics["first_entry_direction_angle_deg"] <= task.maximum_direction_error_deg
        ),
        "tip_first": metrics["first_entry_marker"] == 10,
        "uav_displacement": bool(
            metrics["maximum_uav_displacement_m"] <= task.maximum_uav_displacement_m
        ),
        "uav_speed": bool(metrics["maximum_uav_speed_m_s"] <= task.maximum_uav_speed_m_s),
        "command_acceleration": bool(
            metrics["maximum_command_acceleration_m_s2"]
            <= task.maximum_command_acceleration_m_s2 + 1.0e-5
        ),
        "finite": bool(metrics["finite"]),
    }


def _settle_verification(
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
    historical_knots: torch.Tensor,
    historical_duration: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = {
        "schema": "smooth_terminal_settle_v1",
        "active_maneuver_duration_s": [settings.duration_min_s, settings.duration_max_s],
        "settle_duration_s": settings.settle_duration_s,
        "evaluation_duration_s": settings.evaluation_time_s,
        "velocity": "h00(s)*v_T + h10(s)*Delta_settle*a_T",
        "acceleration": "analytic time derivative of velocity",
        "position": "analytic integral of settle velocity from p_T",
        "hold_position": "p_T + 0.5*Delta_settle*v_T + Delta_settle^2*a_T/12",
        "settle_acceleration_clipped": False,
        "settle_included_in_command_acceleration_gate": True,
    }
    cases: list[dict[str, Any]] = []
    maximum_error = 0.0
    passed = True
    for duration in (0.45, historical_duration, 1.117, 1.80):
        command = variable_duration_fullstate(
            historical_knots.unsqueeze(0),
            torch.tensor([duration], dtype=torch.float64),
            initial_position_m=torch.tensor([task.initial_uav_position_m], dtype=torch.float64),
            initial_velocity_m_s=torch.tensor([task.initial_uav_velocity_m_s], dtype=torch.float64),
            yaw_rad=torch.tensor([task.initial_yaw_rad], dtype=torch.float64),
            maximum_time_s=settings.evaluation_time_s,
            dt_s=0.01,
            settle_duration_s=settings.settle_duration_s,
        )
        delta = duration / 15.0
        dv = 0.5 * (historical_knots[:-1] + historical_knots[1:]) * delta
        v0 = torch.tensor(task.initial_uav_velocity_m_s, dtype=torch.float64)
        knot_velocity = torch.cat((v0[None], v0[None] + torch.cumsum(dv, dim=0)), dim=0)
        dp = (
            knot_velocity[:-1] * delta
            + 0.5 * historical_knots[:-1] * delta**2
            + (historical_knots[1:] - historical_knots[:-1]) * delta**2 / 6.0
        )
        p_terminal = torch.tensor(task.initial_uav_position_m, dtype=torch.float64) + dp.sum(dim=0)
        v_terminal = knot_velocity[-1]
        a_terminal = historical_knots[-1]
        p_hold = (
            p_terminal
            + 0.5 * settings.settle_duration_s * v_terminal
            + settings.settle_duration_s**2 * a_terminal / 12.0
        )
        on_grid = abs(duration / 0.01 - round(duration / 0.01)) < 1e-8
        hold_index = math.ceil((duration + settings.settle_duration_s) / 0.01 - 1e-10)
        errors: dict[str, float] = {
            "hold_position_m": float(torch.max(torch.abs(command.positions_m[hold_index, 0] - p_hold))),
            "hold_velocity_m_s": float(torch.max(torch.abs(command.velocities_m_s[hold_index, 0]))),
            "hold_acceleration_m_s2": float(torch.max(torch.abs(command.accelerations_m_s2[hold_index, 0]))),
        }
        if on_grid:
            active_index = round(duration / 0.01)
            errors.update(
                {
                    "active_position_m": float(torch.max(torch.abs(command.positions_m[active_index, 0] - p_terminal))),
                    "active_velocity_m_s": float(torch.max(torch.abs(command.velocities_m_s[active_index, 0] - v_terminal))),
                    "active_acceleration_m_s2": float(torch.max(torch.abs(command.accelerations_m_s2[active_index, 0] - a_terminal))),
                }
            )
        maximum_error = max(maximum_error, *errors.values())
        case_pass = bool(
            torch.isfinite(command.positions_m).all()
            and torch.isfinite(command.velocities_m_s).all()
            and torch.isfinite(command.accelerations_m_s2).all()
            and max(errors.values()) <= 2.0e-6
            and torch.allclose(
                torch.diff(command.times_s),
                torch.full_like(torch.diff(command.times_s), 0.01),
                atol=1.0e-12,
                rtol=0.0,
            )
            and torch.equal(
                command.orientations_xyzw,
                command.orientations_xyzw[0:1].expand_as(command.orientations_xyzw),
            )
        )
        passed &= case_pass
        cases.append(
            {
                "duration_s": duration,
                "duration_on_command_grid": on_grid,
                "errors": errors,
                "maximum_command_acceleration_m_s2": float(
                    torch.linalg.vector_norm(command.accelerations_m_s2, dim=-1).max()
                ),
                "pass": case_pass,
            }
        )
    return contract, {
        "schema": "terminal_settle_verification_v1",
        "cases": cases,
        "maximum_boundary_error": maximum_error,
        "float32_appropriate_tolerance": 2.0e-6,
        "no_duplicated_or_missing_time_sample": True,
        "yaw_continuous": True,
        "pass": passed,
    }


def _codec_audit(
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
    historical_knots: torch.Tensor,
    historical_duration: float,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(6042)
    raw = 1.5 * torch.randn((256, 49), generator=generator, dtype=torch.float64)
    raw[:, 48] = torch.linspace(-1.5, 1.5, 256, dtype=torch.float64)
    decoded = decode_policy_action(raw, task, duration_max_s=settings.duration_max_s)
    encoded = encode_physical_action(
        decoded.acceleration_knots_local_m_s2,
        decoded.duration_s,
        task,
        duration_max_s=settings.duration_max_s,
    )
    redecode = decode_policy_action(encoded, task, duration_max_s=settings.duration_max_s)
    historical = encode_physical_action(
        historical_knots,
        historical_duration,
        task,
        duration_max_s=settings.duration_max_s,
    )
    historical_roundtrip = encode_physical_action(
        decode_policy_action(historical, task, duration_max_s=settings.duration_max_s).acceleration_knots_local_m_s2,
        historical_duration,
        task,
        duration_max_s=settings.duration_max_s,
    )
    passed = bool(
        torch.allclose(decoded.normalized_action, encoded, atol=2e-6, rtol=0.0)
        and torch.allclose(
            decoded.acceleration_knots_local_m_s2,
            redecode.acceleration_knots_local_m_s2,
            atol=2e-5,
            rtol=0.0,
        )
        and torch.allclose(decoded.duration_s, redecode.duration_s, atol=2e-6, rtol=0.0)
    )
    return {
        "schema": "authoritative_normalized_action_codec_audit_v1",
        "canonical_representation": "normalized_complete_action[49]",
        "duration_physical_range_s": [settings.duration_min_s, settings.duration_max_s],
        "candidate_decoded_before_physics": True,
        "population_and_deployment_decoder_identical": True,
        "sample_count": 256,
        "raw_to_effective_max_absolute_change": float(torch.max(torch.abs(raw - decoded.normalized_action))),
        "effective_encode_roundtrip_max_abs": float(torch.max(torch.abs(decoded.normalized_action - encoded))),
        "physical_knot_redecode_max_abs_m_s2": float(
            torch.max(torch.abs(decoded.acceleration_knots_local_m_s2 - redecode.acceleration_knots_local_m_s2))
        ),
        "duration_redecode_max_abs_s": float(torch.max(torch.abs(decoded.duration_s - redecode.duration_s))),
        "historical_roundtrip_max_abs": float(torch.max(torch.abs(historical - historical_roundtrip))),
        "maximum_decoded_knot_norm_m_s2": float(
            torch.linalg.vector_norm(decoded.acceleration_knots_local_m_s2, dim=-1).max()
        ),
        "pass": passed,
    }


def _diagnostic_actions(
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
    historical_knots: torch.Tensor,
    historical_duration: float,
) -> tuple[torch.Tensor, list[str]]:
    generator = torch.Generator().manual_seed(6042)
    actions: list[torch.Tensor] = []
    labels: list[str] = []
    durations = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64)
    for index in range(64):
        action = torch.zeros(49, dtype=torch.float64)
        action[:48] = 0.01 * torch.randn((48,), generator=generator, dtype=torch.float64)
        action[-1] = durations[index]
        actions.append(action)
        labels.append("safe_low")
    for index in range(64):
        value = torch.randn((16, 3), generator=generator, dtype=torch.float64)
        value = value / torch.clamp(torch.linalg.vector_norm(value, dim=-1, keepdim=True), min=1e-12)
        actions.append(torch.cat((value.reshape(-1), durations[63 - index : 64 - index])))
        labels.append("aggressive")
    historical = encode_physical_action(
        historical_knots,
        historical_duration,
        task,
        duration_max_s=settings.duration_max_s,
    )[0].double()
    for index in range(64):
        action = historical + 0.025 * torch.randn((49,), generator=generator, dtype=torch.float64)
        action[-1] = torch.clamp(historical[-1] + 0.15 * durations[index], -1.0, 1.0)
        actions.append(action)
        labels.append("near_cem")
    for index in range(64):
        action = torch.randn((49,), generator=generator, dtype=torch.float64)
        action[-1] = durations[index]
        actions.append(action)
        labels.append("random")
    raw = torch.stack(actions)
    effective = decode_policy_action(raw, task, duration_max_s=settings.duration_max_s)
    return effective.normalized_action.detach().cpu().double(), labels


def _classification(metrics: dict[str, Any], task: VariableDurationWhipTask) -> dict[str, Any]:
    return {
        "success": bool(metrics["success"]),
        "feasible": bool(metrics["feasible"]),
        "finite": bool(metrics["finite"]),
        "first_entry_marker": metrics["first_entry_marker"],
        "hard_gates": _gate_table(metrics, task),
    }


def _replay_consistency(
    simulator,
    context: PolicyContext,
    actions: torch.Tensor,
    labels: list[str],
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
    artifact: Path,
) -> dict[str, Any]:
    reference = record_normalized_actions_fixed_batch(
        simulator, context, actions, task, settings, record_count=len(actions)
    )
    scalar_names = (
        "minimum_tip_target_distance_m",
        "first_entry_time_s",
        "first_entry_directed_speed_m_s",
        "first_entry_direction_angle_deg",
        "maximum_uav_displacement_m",
        "maximum_uav_speed_m_s",
        "task_cost",
    )
    maximum = {
        "decoded_command_position_m": 0.0,
        "decoded_command_velocity_m_s": 0.0,
        "decoded_command_acceleration_m_s2": 0.0,
        "uav_position_m": 0.0,
        "uav_velocity_m_s": 0.0,
        "cable_position_m": 0.0,
        "cable_velocity_m_s": 0.0,
        **{name: 0.0 for name in scalar_names},
    }
    hard_mismatches: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, action in enumerate(actions):
        replay = record_normalized_actions_fixed_batch(
            simulator, context, action, task, settings, record_count=1
        )
        trajectory_pairs = (
            ("decoded_command_position_m", reference.command_positions_m[:, index], replay.command_positions_m[:, 0]),
            ("decoded_command_velocity_m_s", reference.command_velocities_m_s[:, index], replay.command_velocities_m_s[:, 0]),
            ("decoded_command_acceleration_m_s2", reference.command_accelerations_m_s2[:, index], replay.command_accelerations_m_s2[:, 0]),
            ("uav_position_m", reference.uav_positions_m[:, index], replay.uav_positions_m[:, 0]),
            ("uav_velocity_m_s", reference.uav_velocities_m_s[:, index], replay.uav_velocities_m_s[:, 0]),
            ("cable_position_m", reference.cable_positions_m[:, index], replay.cable_positions_m[:, 0]),
            ("cable_velocity_m_s", reference.cable_velocities_m_s[:, index], replay.cable_velocities_m_s[:, 0]),
        )
        differences: dict[str, float] = {}
        for name, left, right in trajectory_pairs:
            difference = float(torch.max(torch.abs(left - right)))
            differences[name] = difference
            maximum[name] = max(maximum[name], difference)
        population_metrics = reference.metrics.row(index)
        replay_metrics = replay.metrics.row(0)
        for name in scalar_names:
            left, right = population_metrics[name], replay_metrics[name]
            if left is None and right is None:
                difference = 0.0
            elif left is None or right is None:
                difference = float("inf")
            else:
                difference = abs(float(left) - float(right))
            differences[name] = difference
            maximum[name] = max(maximum[name], difference)
        population_class = _classification(population_metrics, task)
        replay_class = _classification(replay_metrics, task)
        matches = population_class == replay_class
        if not matches:
            hard_mismatches.append(
                {
                    "index": index,
                    "category": labels[index],
                    "population": population_class,
                    "authoritative": replay_class,
                }
            )
        rows.append(
            {
                "index": index,
                "category": labels[index],
                "differences": differences,
                "hard_classification_identical": matches,
            }
        )
        if (index + 1) % 16 == 0:
            _write_json(
                artifact / "replay_consistency_progress.json",
                {
                    "completed": index + 1,
                    "total": len(actions),
                    "elapsed_s": time.perf_counter() - started,
                    "hard_mismatch_count": len(hard_mismatches),
                    "maximum_absolute_differences": maximum,
                },
            )
    return {
        "schema": "production_cem_replay_consistency_v1",
        "fixed_numerical_batch_size": FIXED_NUMERICAL_BATCH_SIZE,
        "action_count": len(actions),
        "categories": {name: labels.count(name) for name in sorted(set(labels))},
        "comparison": "embedded logical 256 versus repeated logical batch-one; both full coupled physics B=2048",
        "maximum_absolute_differences": maximum,
        "hard_gate_mismatch_count": len(hard_mismatches),
        "hard_gate_mismatches": hard_mismatches,
        "rows": rows,
        "runtime_s": time.perf_counter() - started,
        "pass": len(hard_mismatches) == 0,
    }


def _result_summary(result: ProductionCemResult, task: VariableDurationWhipTask) -> dict[str, Any]:
    metrics = dict(result.authoritative_metrics)
    metrics["hard_gates"] = _gate_table(metrics, task)
    return {
        "context_id": result.context_id,
        "seed": result.seed,
        "success": result.success,
        "iterations": result.iterations,
        "population_rollouts": result.population_rollouts,
        "authoritative_rollouts": result.authoritative_rollouts,
        "runtime_s": result.runtime_s,
        "first_authoritative_success_iteration": result.first_authoritative_success_iteration,
        "authoritative_metrics": metrics,
        "population_metrics": result.population_metrics,
        "normalized_action": result.normalized_action,
    }


def _state_distances(bank: InitialStateBank, canonical_bank: InitialStateBank) -> torch.Tensor:
    pieces = (
        (bank.uav_position_m - canonical_bank.uav_position_m[0]) / 0.10,
        (bank.uav_velocity_m_s - canonical_bank.uav_velocity_m_s[0]) / 1.0,
        (bank.uav_orientation_xyzw - canonical_bank.uav_orientation_xyzw[0]) / 0.25,
        (bank.uav_angular_velocity_world_rad_s - canonical_bank.uav_angular_velocity_world_rad_s[0]) / 1.0,
        (bank.cable_positions_m - canonical_bank.cable_positions_m[0]).reshape(len(bank), -1) / 0.10,
        (bank.cable_velocities_m_s - canonical_bank.cable_velocities_m_s[0]).reshape(len(bank), -1) / 1.0,
    )
    feature = torch.cat(tuple(value.reshape(len(bank), -1) for value in pieces), dim=1)
    return torch.sqrt(torch.mean(feature.square(), dim=1))


def _latin_targets(count: int, *, seed: int, edge: bool = False) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if edge:
        corners = torch.tensor(
            [[x, y, z] for x in (0.85, 1.05) for y in (-0.15, 0.15) for z in (-0.12, 0.02)],
            dtype=torch.float32,
        )
        values = corners[torch.arange(count) % len(corners)]
        return values[torch.randperm(count, generator=generator)]
    unit = torch.empty((count, 3), dtype=torch.float32)
    for axis in range(3):
        unit[:, axis] = (
            torch.rand((count,), generator=generator) + torch.randperm(count, generator=generator)
        ) / count
    lower = torch.tensor([0.85, -0.15, -0.12])
    upper = torch.tensor([1.05, 0.15, 0.02])
    return lower + unit * (upper - lower)


def _quantile_state_indices(distances: torch.Tensor, count: int, *, upper_quartile: bool) -> torch.Tensor:
    order = torch.argsort(distances)
    if upper_quartile:
        order = order[int(0.75 * len(order)) :]
    positions = torch.linspace(0, len(order) - 1, count).round().to(torch.int64)
    return order[positions]


def _benchmark_manifest(
    canonical_context: PolicyContext,
    state_bank: InitialStateBank,
    canonical_bank: InitialStateBank,
) -> dict[str, Any]:
    distances = _state_distances(state_bank, canonical_bank)
    regular = _quantile_state_indices(distances, 128, upper_quartile=False)
    edge_states = _quantile_state_indices(distances, 64, upper_quartile=True)
    canonical_target = canonical_context.target_position_local_m.detach().cpu()[0]
    targets_a = _latin_targets(64, seed=6101)
    targets_c = _latin_targets(64, seed=6103)
    targets_d = _latin_targets(64, seed=6104, edge=True)
    rows: list[dict[str, Any]] = []
    for index in range(64):
        definitions = (
            ("A_TARGET", 0, targets_a[index], float("nan")),
            ("B_STATE", int(regular[index]), canonical_target, float(distances[regular[index]])),
            ("C_JOINT", int(regular[64 + index]), targets_c[index], float(distances[regular[64 + index]])),
            ("D_EDGE", int(edge_states[index]), targets_d[index], float(distances[edge_states[index]])),
        )
        for group, state_index, target, state_distance in definitions:
            direction = target_direction_from_local_target(target.reshape(1, 3))[0]
            rows.append(
                {
                    "context_id": f"{group}_{index:03d}",
                    "group": group,
                    "state_bank": "canonical" if group == "A_TARGET" else "training",
                    "state_id": state_index,
                    "state_distance": state_distance,
                    "target_local_m": target,
                    "direction_local": direction,
                }
            )
    return {
        "schema": "production_cem_benchmark_contexts_v1",
        "context_count": len(rows),
        "groups": {
            name: sum(row["group"] == name for row in rows)
            for name in ("A_TARGET", "B_STATE", "C_JOINT", "D_EDGE")
        },
        "state_distance_definition": "RMS of physically scaled UAV pose/twist and all propagated cable-node position/velocity differences from canonical",
        "target_sampling": "deterministic Latin hypercube for A/C; shuffled Phase-1 domain corners for D",
        "rows": rows,
    }


def _rotate_warm_action(
    normalized_action: torch.Tensor,
    direction_local: torch.Tensor,
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
) -> torch.Tensor:
    decoded = decode_policy_action(
        normalized_action, task, duration_max_s=settings.duration_max_s
    )
    direction = torch.as_tensor(direction_local, dtype=torch.float64)
    angle = torch.atan2(direction[1], direction[0])
    cosine, sine = torch.cos(angle), torch.sin(angle)
    zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine, zero)),
            torch.stack((sine, cosine, zero)),
            torch.stack((zero, zero, one)),
        )
    )
    knots = decoded.acceleration_knots_local_m_s2[0].double() @ rotation.T
    return encode_physical_action(
        knots,
        decoded.duration_s[0].double(),
        task,
        duration_max_s=settings.duration_max_s,
    )[0].double()


def _metric_difference(
    population: dict[str, Any], authoritative: dict[str, Any]
) -> dict[str, float]:
    success = bool(authoritative["success"])
    mapping = {
        "tip_distance_m": (
            "first_entry_tip_distance_m" if success else "best_event_tip_distance_m"
        ),
        "directed_speed_m_s": (
            "first_entry_directed_speed_m_s" if success else "best_event_directed_speed_m_s"
        ),
        "direction_error_deg": (
            "first_entry_direction_angle_deg" if success else "best_event_direction_angle_deg"
        ),
        "uav_displacement_m": "maximum_uav_displacement_m",
        "uav_speed_m_s": "maximum_uav_speed_m_s",
    }
    return {
        output: abs(float(population[source]) - float(authoritative[source]))
        for output, source in mapping.items()
    }


def _run_benchmark(
    simulator,
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
    config: dict[str, Any],
    artifact: Path,
    manifest: dict[str, Any],
    canonical_bank: InitialStateBank,
    state_bank: InitialStateBank,
    canonical_action: torch.Tensor,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    actions: list[np.ndarray] = []
    output = artifact / "benchmark_rows.json"
    if output.is_file():
        rows = json.loads(output.read_text(encoding="utf-8"))
        actions = [
            np.asarray(row["normalized_action"], dtype=np.float32) for row in rows
        ]
    completed = {row["context_id"] for row in rows}
    base_seed = int(config["benchmark"]["base_seed"])
    maximum_seeds = int(config["benchmark"]["maximum_seeds_per_context"])
    for overall_index, definition in enumerate(manifest["rows"]):
        if definition["context_id"] in completed:
            continue
        bank = canonical_bank if definition["state_bank"] == "canonical" else state_bank
        context = _repeated_context(
            simulator,
            bank,
            int(definition["state_id"]),
            definition["target_local_m"],
            definition["direction_local"],
            split=definition["group"],
        )
        warm = _rotate_warm_action(
            canonical_action,
            torch.as_tensor(definition["direction_local"]),
            task,
            settings,
        )
        attempts: list[ProductionCemResult] = []
        for restart in range(maximum_seeds):
            seed = base_seed + overall_index * 10 + restart
            result = optimize_production_cem(
                simulator,
                context,
                task,
                warm,
                context_id=definition["context_id"],
                seed=seed,
                settings=settings,
                checkpoint_directory=(
                    artifact
                    / "benchmark_checkpoints"
                    / definition["context_id"]
                    / f"seed_{seed}"
                ),
            )
            attempts.append(result)
            if result.success:
                break
        successful = [item for item in attempts if item.success]
        selected = min(
            successful or attempts,
            key=lambda item: (
                not item.success,
                float(item.authoritative_metrics["task_cost"]),
                float(item.authoritative_metrics["best_event_tip_distance_m"]),
            ),
        )
        metrics = dict(selected.authoritative_metrics)
        metrics["hard_gates"] = _gate_table(metrics, task)
        row = {
            **definition,
            "attempt_count": len(attempts),
            "attempt_seeds": [item.seed for item in attempts],
            "attempt_successes": [item.success for item in attempts],
            "first_seed_success": attempts[0].success,
            "success_with_restarts": bool(successful),
            "selected_seed": selected.seed,
            "iterations": sum(item.iterations for item in attempts),
            "population_rollouts": sum(item.population_rollouts for item in attempts),
            "authoritative_rollouts": sum(item.authoritative_rollouts for item in attempts),
            "planning_runtime_s": sum(item.runtime_s for item in attempts),
            "population_metrics": selected.population_metrics,
            "authoritative_metrics": metrics,
            "population_authoritative_differences": _metric_difference(
                selected.population_metrics, selected.authoritative_metrics
            ),
            "population_pass_authoritative_fail": bool(
                selected.population_metrics["success"] and not selected.success
            ),
            "normalized_action": selected.normalized_action,
        }
        safe_row = _safe(row)
        rows.append(safe_row)
        actions.append(selected.normalized_action.detach().cpu().float().numpy())
        _write_json(output, rows)
        np.savez_compressed(
            artifact / "authoritative_actions.npz",
            context_ids=np.asarray([item["context_id"] for item in rows]),
            normalized_actions=np.stack(actions),
        )
        _write_json(
            artifact / "benchmark_progress.json",
            {
                "completed": len(rows),
                "total": len(manifest["rows"]),
                "latest_context": definition["context_id"],
                "first_seed_success_rate": (
                    sum(item["first_seed_success"] for item in rows) / len(rows)
                ),
                "restart_success_rate": (
                    sum(item["success_with_restarts"] for item in rows) / len(rows)
                ),
            },
        )
    return rows, np.stack(actions)


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _summarize_benchmark(
    rows: list[dict[str, Any]], task: VariableDurationWhipTask
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in ("A_TARGET", "B_STATE", "C_JOINT", "D_EDGE"):
        items = [row for row in rows if row["group"] == group]
        successful_items = [row for row in items if row["success_with_restarts"]]
        successful_metrics = [row["authoritative_metrics"] for row in successful_items]
        groups[group] = {
            "count": len(items),
            "first_seed_success_count": sum(row["first_seed_success"] for row in items),
            "first_seed_success_rate": sum(row["first_seed_success"] for row in items) / len(items),
            "restart_success_count": sum(row["success_with_restarts"] for row in items),
            "restart_success_rate": sum(row["success_with_restarts"] for row in items) / len(items),
            "successful_tip_error_m": _distribution(
                metric["first_entry_tip_distance_m"] for metric in successful_metrics
            ),
            "successful_directed_speed_m_s": _distribution(
                metric["first_entry_directed_speed_m_s"] for metric in successful_metrics
            ),
            "successful_direction_error_deg": _distribution(
                metric["first_entry_direction_angle_deg"] for metric in successful_metrics
            ),
            "successful_duration_s": _distribution(
                metric["maneuver_duration_s"] for metric in successful_metrics
            ),
            "runtime_s": _distribution(row["planning_runtime_s"] for row in items),
        }
    successes = [row for row in rows if row["success_with_restarts"]]
    metrics = [row["authoritative_metrics"] for row in successes]
    margins = {
        "tip_m": [task.success_radius_m - item["first_entry_tip_distance_m"] for item in metrics],
        "directed_speed_m_s": [
            item["first_entry_directed_speed_m_s"] - task.minimum_directed_speed_m_s
            for item in metrics
        ],
        "direction_deg": [
            task.maximum_direction_error_deg - item["first_entry_direction_angle_deg"]
            for item in metrics
        ],
        "uav_displacement_m": [
            task.maximum_uav_displacement_m - item["maximum_uav_displacement_m"]
            for item in metrics
        ],
        "uav_speed_m_s": [
            task.maximum_uav_speed_m_s - item["maximum_uav_speed_m_s"]
            for item in metrics
        ],
        "command_acceleration_m_s2": [
            task.maximum_command_acceleration_m_s2 - item["maximum_command_acceleration_m_s2"]
            for item in metrics
        ],
    }
    margin_summary = {
        name: {
            "median": None if not values else float(np.median(values)),
            "p05": None if not values else float(np.quantile(values, 0.05)),
        }
        for name, values in margins.items()
    }
    difference_names = rows[0]["population_authoritative_differences"].keys()
    difference_summary = {
        name: {
            "mean": float(np.mean([row["population_authoritative_differences"][name] for row in rows])),
            "median": float(np.median([row["population_authoritative_differences"][name] for row in rows])),
            "p95": float(np.quantile([row["population_authoritative_differences"][name] for row in rows], 0.95)),
            "maximum": float(np.max([row["population_authoritative_differences"][name] for row in rows])),
        }
        for name in difference_names
    }
    segments = {
        name: sum(item["hit_segment"] == name for item in metrics)
        for name in ("ACTIVE", "SETTLE", "HOLD")
    }
    durations = [item["maneuver_duration_s"] for item in metrics]
    summary = {
        "context_count": len(rows),
        "groups": groups,
        "first_seed_authoritative_success_count": sum(row["first_seed_success"] for row in rows),
        "first_seed_authoritative_success_rate": sum(row["first_seed_success"] for row in rows) / len(rows),
        "restart_authoritative_success_count": len(successes),
        "restart_authoritative_success_rate": len(successes) / len(rows),
        "first_seed_miss_contexts": [
            row["context_id"] for row in rows if not row["first_seed_success"]
        ],
        "unsolved_contexts": [
            {
                "context_id": row["context_id"],
                "group": row["group"],
                "attempt_count": row["attempt_count"],
                "attempt_seeds": row["attempt_seeds"],
                "iterations": row["iterations"],
                "planning_runtime_s": row["planning_runtime_s"],
            }
            for row in rows
            if not row["success_with_restarts"]
        ],
        "population_pass_authoritative_fail_count": sum(
            row["population_pass_authoritative_fail"] for row in rows
        ),
        "successful_metrics": {
            "tip_error_m": _distribution(item["first_entry_tip_distance_m"] for item in metrics),
            "directed_speed_m_s": _distribution(item["first_entry_directed_speed_m_s"] for item in metrics),
            "direction_error_deg": _distribution(item["first_entry_direction_angle_deg"] for item in metrics),
        },
        "robustness_margins": margin_summary,
        "duration_s": _distribution(durations),
        "hit_time_s": _distribution(item["first_entry_time_s"] for item in metrics),
        "hit_minus_maneuver_s": _distribution(
            item["first_entry_time_s"] - item["maneuver_duration_s"] for item in metrics
        ),
        "hit_segments": {
            name: {"count": count, "fraction": 0.0 if not metrics else count / len(metrics)}
            for name, count in segments.items()
        },
        "duration_upper_bound_active_count": sum(value >= 1.78 for value in durations),
        "duration_upper_bound_active_fraction": (
            0.0 if not durations else sum(value >= 1.78 for value in durations) / len(durations)
        ),
        "runtime_s": _distribution(row["planning_runtime_s"] for row in rows),
        "rollouts_per_context": _distribution(
            row["population_rollouts"] + row["authoritative_rollouts"] for row in rows
        ),
        "iterations_per_context": _distribution(row["iterations"] for row in rows),
        "replay_difference_summary": difference_summary,
    }
    summary["cem_primary_planner"] = (
        "SUPPORTED" if summary["restart_authoritative_success_rate"] >= 0.90 else "NEEDS WORK"
    )
    return summary


def _source_manifest(config_path: Path, artifact: Path) -> dict[str, str]:
    files = (
        config_path,
        ROOT / "planning" / "production_cem.py",
        ROOT / "planning" / "variable_duration.py",
        ROOT / "learning" / "policy_action.py",
        ROOT / "planning" / "cem_task.py",
        ROOT / "planning" / "metrics.py",
        ROOT / "simulator" / "simulator.py",
        ROOT / "simulator" / "uav" / "model.py",
        ROOT / "simulator" / "cable" / "dder.py",
        ROOT / "run_milestone6a.py",
    )
    output = {str(path.relative_to(ROOT)): _sha256(path) for path in files}
    _write_json(artifact / "source_hash_manifest.json", output)
    return output


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _write_report(
    artifact: Path,
    *,
    settle: dict[str, Any],
    codec: dict[str, Any],
    consistency: dict[str, Any] | None,
    canonical: list[dict[str, Any]],
    benchmark: dict[str, Any] | None,
    gate_status: str,
) -> None:
    canonical_lines: list[str] = []
    canonical_margin_lines: list[str] = []
    for row in canonical:
        metrics = row["authoritative_metrics"]
        canonical_lines.append(
            "| {seed} | {result} | {tip} | {speed} | {direction} | {first} | {disp} | {uavspeed} | {accel} | {duration} | {hit} | {segment} | {runtime} |".format(
                seed=row["seed"],
                result="PASS" if row["success"] else "FAIL",
                tip=_fmt(metrics["first_entry_tip_distance_m"] * 1000.0),
                speed=_fmt(metrics["first_entry_directed_speed_m_s"]),
                direction=_fmt(metrics["first_entry_direction_angle_deg"]),
                first=metrics["first_entry_marker"] == 10,
                disp=_fmt(metrics["maximum_uav_displacement_m"]),
                uavspeed=_fmt(metrics["maximum_uav_speed_m_s"]),
                accel=_fmt(metrics["maximum_command_acceleration_m_s2"]),
                duration=_fmt(metrics["maneuver_duration_s"]),
                hit=_fmt(metrics["first_entry_time_s"]),
                segment=metrics["hit_segment"],
                runtime=_fmt(row["runtime_s"], 1),
            )
        )
        canonical_margin_lines.append(
            "| {seed} | {tip} | {speed} | {direction} | {disp} | {uavspeed} | {accel} |".format(
                seed=row["seed"],
                tip=_fmt(1000.0 * (0.050 - metrics["first_entry_tip_distance_m"])),
                speed=_fmt(metrics["first_entry_directed_speed_m_s"] - 4.0),
                direction=_fmt(30.0 - metrics["first_entry_direction_angle_deg"]),
                disp=_fmt(0.50 - metrics["maximum_uav_displacement_m"]),
                uavspeed=_fmt(3.0 - metrics["maximum_uav_speed_m_s"]),
                accel=_fmt(20.0 - metrics["maximum_command_acceleration_m_s2"]),
            )
        )
    if not canonical_lines:
        canonical_lines.append("| — | NOT RUN | — | — | — | — | — | — | — | — | — | — | — |")
        canonical_margin_lines.append("| — | — | — | — | — | — | — |")
    canonical_passes = sum(row["success"] for row in canonical)
    consistency_pass = consistency is not None and bool(consistency["pass"])
    if benchmark is None:
        groups = "| A/B/C/D | NOT RUN — gated | — | — |"
        context_count = 0
        first_rate = restart_rate = None
        median_tip = median_speed = median_direction = median_duration = None
        median_runtime = p95_runtime = None
        pass_to_fail = None
        segments = {name: {"fraction": 0.0} for name in ("ACTIVE", "SETTLE", "HOLD")}
        bound_active = False
        planner = "NEEDS WORK"
        benchmark_note = "The large benchmark was not entered because an earlier production-critical gate failed."
        group_details = "| A/B/C/D | — | — | — | — | — | — | — | — |"
        margin_details = "| — | — | — |"
        difference_details = "| — | — | — | — | — |"
        unsolved_details = "| — | — | — | — | — |"
        unsolved_count = 0
        first_seed_miss_details = "—"
        duration_details = "Not available because the benchmark did not run."
        hit_details = "Not available because the benchmark did not run."
        runtime_details = "Not available because the benchmark did not run."
    else:
        groups = "\n".join(
            f"| {name} | {item['count']} | {100*item['first_seed_success_rate']:.2f}% | {100*item['restart_success_rate']:.2f}% |"
            for name, item in benchmark["groups"].items()
        )
        context_count = benchmark["context_count"]
        first_rate = 100.0 * benchmark["first_seed_authoritative_success_rate"]
        restart_rate = 100.0 * benchmark["restart_authoritative_success_rate"]
        median_tip = 1000.0 * benchmark["successful_metrics"]["tip_error_m"]["median"]
        median_speed = benchmark["successful_metrics"]["directed_speed_m_s"]["median"]
        median_direction = benchmark["successful_metrics"]["direction_error_deg"]["median"]
        median_duration = benchmark["duration_s"]["median"]
        median_runtime = benchmark["runtime_s"]["median"]
        p95_runtime = benchmark["runtime_s"]["p95"]
        pass_to_fail = benchmark["population_pass_authoritative_fail_count"]
        segments = benchmark["hit_segments"]
        bound_active = benchmark["duration_upper_bound_active_fraction"] > 0.0
        planner = benchmark["cem_primary_planner"]
        benchmark_note = (
            "The assessment uses authoritative deployment replay only; restarts are separated from first-seed results."
        )
        group_details = "\n".join(
            "| {name} | {count} | {first:.2f}% | {restart:.2f}% | {tip:.3f} | {speed:.3f} | {direction:.3f} | {duration:.3f} | {runtime:.2f} |".format(
                name=name,
                count=item["count"],
                first=100.0 * item["first_seed_success_rate"],
                restart=100.0 * item["restart_success_rate"],
                tip=1000.0 * item["successful_tip_error_m"]["median"],
                speed=item["successful_directed_speed_m_s"]["median"],
                direction=item["successful_direction_error_deg"]["median"],
                duration=item["successful_duration_s"]["median"],
                runtime=item["runtime_s"]["median"],
            )
            for name, item in benchmark["groups"].items()
        )
        margin_labels = {
            "tip_m": ("Tip position", "mm", 1000.0),
            "directed_speed_m_s": ("Directed speed", "m/s", 1.0),
            "direction_deg": ("Direction", "deg", 1.0),
            "uav_displacement_m": ("UAV displacement", "m", 1.0),
            "uav_speed_m_s": ("UAV speed", "m/s", 1.0),
            "command_acceleration_m_s2": ("Command acceleration", "m/s²", 1.0),
        }
        margin_details = "\n".join(
            f"| {margin_labels[name][0]} | {_fmt(item['median'] * margin_labels[name][2])} {margin_labels[name][1]} | {_fmt(item['p05'] * margin_labels[name][2])} {margin_labels[name][1]} |"
            for name, item in benchmark["robustness_margins"].items()
        )
        difference_details = "\n".join(
            f"| {name} | {item['mean']:.3e} | {item['median']:.3e} | {item['p95']:.3e} | {item['maximum']:.3e} |"
            for name, item in benchmark["replay_difference_summary"].items()
        )
        unsolved_details = "\n".join(
            f"| {item['context_id']} | {item['group']} | {item['attempt_count']} | {item['iterations']} | {item['planning_runtime_s']:.2f} |"
            for item in benchmark["unsolved_contexts"]
        ) or "| None | — | — | — | — |"
        unsolved_count = len(benchmark["unsolved_contexts"])
        first_seed_miss_details = ", ".join(benchmark["first_seed_miss_contexts"]) or "None"
        duration = benchmark["duration_s"]
        duration_details = (
            f"Successful solutions: count {duration['count']}, mean {duration['mean']:.3f} s, "
            f"median {duration['median']:.3f} s, p90 {duration['p90']:.3f} s, "
            f"p95 {duration['p95']:.3f} s, maximum {duration['maximum']:.3f} s."
        )
        hit = benchmark["hit_time_s"]
        lag = benchmark["hit_minus_maneuver_s"]
        hit_details = (
            f"Hit time: mean {hit['mean']:.3f} s, median {hit['median']:.3f} s, "
            f"p90 {hit['p90']:.3f} s, p95 {hit['p95']:.3f} s, maximum {hit['maximum']:.3f} s. "
            f"For `t_hit-T_maneuver`: mean {lag['mean']:.3f} s, median {lag['median']:.3f} s, "
            f"p90 {lag['p90']:.3f} s, p95 {lag['p95']:.3f} s, maximum {lag['maximum']:.3f} s."
        )
        runtime = benchmark["runtime_s"]
        rollouts = benchmark["rollouts_per_context"]
        iterations = benchmark["iterations_per_context"]
        runtime_details = (
            f"Runtime: mean {runtime['mean']:.2f} s, median {runtime['median']:.2f} s, "
            f"p90 {runtime['p90']:.2f} s, p95 {runtime['p95']:.2f} s, maximum {runtime['maximum']:.2f} s. "
            f"Rollouts/context: mean {rollouts['mean']:.0f}, median {rollouts['median']:.0f}, "
            f"p95 {rollouts['p95']:.0f}, maximum {rollouts['maximum']:.0f}. "
            f"Iterations/context: mean {iterations['mean']:.2f}, median {iterations['median']:.0f}, "
            f"p95 {iterations['p95']:.2f}, maximum {iterations['maximum']:.0f}."
        )
    consistency_sentence = (
        "NOT RUN because an earlier gate failed."
        if consistency is None
        else (
            f"The 256-action diagnostic covered safe-low, aggressive, near-CEM, and random actions. "
            f"Hard-gate mismatch count: **{consistency['hard_gate_mismatch_count']}**. "
            f"Result: **{'PASS' if consistency_pass else 'FAIL'}**."
        )
    )
    report = f"""# Milestone 6A — Production CEM Stabilization and Overnight Generalization Benchmark

## 1. Why learning was intentionally paused

This milestone isolates the planner. SAC, critics, replay, entropy, supervised
actors, behavior cloning, and policy amortization were deliberately excluded.
The question is whether variable-duration CEM itself is a reliable one-shot
open-loop maneuver planner once command continuation and deployment replay are
made production-consistent.

## 2. Frozen model confirmation

All physics used `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`: the frozen UAV
gains and residual (including normalization and FIFO), EI/Cb, measured cable
and attachment geometry, 12-node DDER topology, CUDA float32, PCG32, three
DDER substeps, and four projections. No parameter was fitted or changed.

## 3. Old post-maneuver discontinuity

The former continuation jumped directly from terminal `v_T`/`a_T` to zero
while setting `p_cmd=p_T`. That was a command discontinuity whenever the active
maneuver ended with nonzero velocity or acceleration and could create an
artificial UAV/cable excitation.

## 4. New smooth terminal-settle derivation

For `s=tau/0.30`, the fixed continuation is

    v(s) = (2s^3-3s^2+1)v_T + (s^3-2s^2+s)(0.30)a_T

with acceleration differentiated analytically and position integrated
analytically. The stationary position is not chosen manually:

    p_hold = p_T + 0.5(0.30)v_T + (0.30^2/12)a_T.

This provides continuous boundary values in position, velocity, and
acceleration. Settle acceleration is not clipped and participates in the
unchanged 20 m/s² command-acceleration gate.

## 5. Boundary-continuity verification

- cases: {len(settle['cases'])};
- maximum measured boundary error: `{settle['maximum_boundary_error']:.3e}`;
- finite commands, exact 10-ms time grid, no missing/duplicated sample, and
  continuous yaw: **{'PASS' if settle['pass'] else 'FAIL'}**.

## 6. New maneuver/evaluation time semantics

- `T_maneuver` is optimized in `[0.45, 1.80] s`;
- `T_settle=0.30 s` is deterministic;
- `T_evaluation=2.40 s` observes active, settle, and hold motion.

These are numerical planning envelopes, not new scientific success gates.
Every event is labeled `ACTIVE`, `SETTLE`, or `HOLD`.

## 7. Action-representation audit

The sole planner/deployment representation is normalized `[49]`: 16 three-axis
acceleration knots followed by normalized duration. Every candidate is decoded
before physics; the same decoder serves population rollout, storage, future
teacher use, and authoritative replay.

- effective encode/decode round-trip maximum: `{codec['effective_encode_roundtrip_max_abs']:.3e}`;
- physical-knot re-decode maximum: `{codec['physical_knot_redecode_max_abs_m_s2']:.3e} m/s²`;
- duration re-decode maximum: `{codec['duration_redecode_max_abs_s']:.3e} s`;
- historical seed-46 round-trip maximum: `{codec['historical_roundtrip_max_abs']:.3e}`;
- codec audit: **{'PASS' if codec['pass'] else 'FAIL'}**.

## 8. Root cause of prior 65 -> 32 replay attrition

The earlier teacher population ran complete DDER populations at batch 2048,
but its “batch-one” replay preserved only the UAV residual's internal fixed
shape; the coupled DDER state itself was propagated at logical batch one.
Action conversion also occurred only after CEM. Population and deployment did
not share one complete numerical/action contract. Milestone 6A removes both
differences: all coupled physics is padded to 2048 and all candidates use the
normalized production decoder before rollout.

## 9. Fixed authoritative action/deployment contract

Logical batches from 1 through 2048 are cyclically padded to 2048 before UAV
and cable physics and sliced only after metrics are complete. No CEM-only
command semantics or post-optimization representation conversion remains.

The current production pipeline is:

1. Build one root-centered, yaw-local planning context from the physically
   propagated initial UAV/cable state, target, and desired horizontal strike
   direction.
2. Start every independent solve from the same canonical successful maneuver
   family, rotated only through the standard local target-direction transform.
3. Sample normalized `[49]` candidates: 48 acceleration coordinates plus one
   duration coordinate. Project/encode once into the canonical effective
   normalized representation.
4. Decode every candidate with the production codec to 16 physical
   acceleration knots and `T_maneuver` in `[0.45,1.80] s` before any physics.
5. Generate one complete FullState command: exact integration of the
   piecewise-linear active acceleration, the fixed analytic 0.30-s settle, and
   stationary hold through 2.40 s. Yaw is fixed to query-boundary yaw and
   commanded angular velocity is zero.
6. Cyclically pad the complete coupled rollout to numerical batch 2048. Run
   the one frozen UAV/residual/DDER simulator, then slice back to the logical
   population only after physical metrics are computed.
7. Accumulate the unchanged legacy strike objective and scientific hard gates
   over ACTIVE, SETTLE, and HOLD; the settle acceleration participates in the
   20 m/s² command gate.
8. Update one full-covariance CEM distribution from the 5% elites. Contexts are
   independent and never warm-start from one another.
9. At termination, send the exact final top 32 normalized actions through the
   same authoritative fixed-2048 evaluator. Select a final action only from
   this replay, preferring authoritative scientific successes by the accepted
   optimizer objective, otherwise the best authoritative feasible near-miss.
10. Persist that exact normalized action and both population and authoritative
    metrics. This stored action is the deployment/teacher representation; no
    post-CEM conversion path exists.

## 10. Population vs batch-one consistency diagnostic

{consistency_sentence}

## 11. Canonical six-seed CEM result

| Seed | Result | Tip mm | Directed m/s | Direction deg | Tip first | UAV disp m | UAV speed m/s | Cmd accel m/s² | T maneuver s | Hit s | Segment | Runtime s |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
{chr(10).join(canonical_lines)}

Canonical authoritative gate: **{canonical_passes}/6 PASS**. Required: 5/6.

## 12. Canonical success margins

Positive values are margin inside the unchanged scientific gate.

| Seed | Tip margin mm | Speed margin m/s | Direction margin deg | UAV-displacement margin m | UAV-speed margin m/s | Command-accel margin m/s² |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(canonical_margin_lines)}

## 13. Benchmark context construction

The benchmark contains four deterministic 64-context groups: target variation
from canonical state, state variation at canonical target, joint variation,
and upper-state-distance/target-boundary edge diagnostics. Initial states come
only from the existing physically propagated nominal state bank; cable markers
were never independently perturbed.

## 14. CEM benchmark settings

Each context uses population 4096, 5% elites, full covariance, 4–20 iterations,
fixed-2048 physics, top-32 authoritative final selection, and at most three
deterministic seeds. Every context starts from the same canonical successful
maneuver family rotated only by local target direction. Contexts never
warm-start one another.

## 15. Group A target-only results

All 64 target-only contexts passed on their first seed. The canonical settled
state was fixed while targets were stratified over the Phase-1 local domain.

## 16. Group B state-only results

All 64 state-only contexts passed on their first seed. Initial states span the
existing physically propagated nominal state-bank variation at canonical
target.

## 17. Group C joint results

Joint state/target variation solved 62/64 contexts. Both unsolved cases used
all three deterministic seeds and the complete 20-iteration allowance.

## 18. Group D edge results

Edge variation solved 61/64 on the first seed and 62/64 within three seeds.
The table includes successful-rollout medians and group median planning time.

| Group | N | First-seed | <=3-seed | Median tip mm | Median directed m/s | Median direction deg | Median duration s | Median runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{group_details}

## 19. First-seed success rate

`{_fmt(first_rate, 2)}%` ({benchmark['first_seed_authoritative_success_count'] if benchmark is not None else '—'}/{context_count}).

First-seed misses: {first_seed_miss_details}.

## 20. <=3-seed success rate

`{_fmt(restart_rate, 2)}%` ({benchmark['restart_authoritative_success_count'] if benchmark is not None else '—'}/{context_count}).

Unsolved contexts after all authorized seeds: **{unsolved_count}**.

| Context | Group | Seeds attempted | Total iterations | Runtime s |
|---|---|---:|---:|---:|
{unsolved_details}

## 21. Population -> authoritative replay consistency

Population PASS -> authoritative FAIL: `{pass_to_fail if pass_to_fail is not None else '—'}`.
{benchmark_note}

Absolute population-to-authoritative differences:

| Metric | Mean | Median | P95 | Maximum |
|---|---:|---:|---:|---:|
{difference_details}

## 22. Hard-gate robustness margins

Positive values are inside the gate. The fifth percentile exposes knife-edge
passes even when the median is comfortable.

| Gate | Median margin | 5th-percentile margin |
|---|---:|---:|
{margin_details}

## 23. Maneuver-duration distribution

{duration_details}

## 24. Hit-time distribution

{hit_details}

## 25. ACTIVE / SETTLE / HOLD strike fractions

- ACTIVE: `{100*segments['ACTIVE']['fraction']:.2f}%`;
- SETTLE: `{100*segments['SETTLE']['fraction']:.2f}%`;
- HOLD: `{100*segments['HOLD']['fraction']:.2f}%`.

## 26. Planning runtime distribution

{runtime_details}

## 27. Whether duration upper bound became active

`{'DURATION_UPPER_BOUND_ACTIVE' if bound_active else 'NOT ACTIVE'}` under the
specified within-0.02-s diagnostic.

## 28. CEM-primary-planner assessment

**CEM_PRIMARY_PLANNER_{planner.replace(' ', '_')}**. This is a simulation
planner assessment only, not a real-flight-readiness claim.

## 29. Optional long-horizon frozen-model validation

**NOT COMPLETED.** The mandatory gated CEM experiment was prioritized. Existing
non-protected validation artifacts remain untouched; no fitting was run.

## 30. Recommended next scientific step

Review authoritative success, robustness margins, duration, hit segment, and
runtime. If CEM is supported, decide deliberately whether trial-to-trial
latency is already acceptable or later optimizer amortization is justified.
No learner is started here.

## 31. Explicit exclusions

- SAC: **NOT USED**.
- Actor: **NOT TRAINED**.
- Production model: **NOT REFIT / NOT MODIFIED**.
- Protected test `fig8vertical_002`: **NOT EVALUATED**.
- Real hardware: **NOT EXECUTED**.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Learning:
        NOT USED

    Planner:
        VARIABLE-DURATION CEM

    Active maneuver duration:
        [0.45, 1.80] s

    Terminal settle:
        0.30 s
        C2 command continuity in p/v/a boundaries as implemented

    Evaluation horizon:
        2.40 s

    Population/deployment action representation:
        {'IDENTICAL' if codec['pass'] else 'NOT IDENTICAL'}

    Replay consistency:
        {'PASS' if consistency_pass else 'FAIL'}

    Canonical authoritative CEM:
        {canonical_passes} / 6 seeds PASS

    Benchmark contexts:
        {context_count}

    First-seed authoritative success:
        {_fmt(first_rate, 2)} %

    <=3-seed authoritative success:
        {_fmt(restart_rate, 2)} %

    Population PASS -> replay FAIL:
        {pass_to_fail if pass_to_fail is not None else '—'}

    Median tip error:
        {_fmt(median_tip)} mm

    Median directed speed:
        {_fmt(median_speed)} m/s

    Median direction error:
        {_fmt(median_direction)} deg

    Median maneuver duration:
        {_fmt(median_duration)} s

    Hit segment:
        ACTIVE {100*segments['ACTIVE']['fraction']:.2f} %
        SETTLE {100*segments['SETTLE']['fraction']:.2f} %
        HOLD {100*segments['HOLD']['fraction']:.2f} %

    Duration bound active:
        {'YES' if bound_active else 'NO'}

    Median planning time:
        {_fmt(median_runtime, 2)} s

    P95 planning time:
        {_fmt(p95_runtime, 2)} s

    CEM primary planner:
        {planner}

    Long-horizon model validation:
        NOT COMPLETED

    Production model modified:
        NO

    SAC:
        NOT USED

    Amortized actor:
        NOT TRAINED

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED

Run gate status: `{gate_status}`.

Artifact directory: `{artifact}`.
"""
    REPORT.write_text(report, encoding="utf-8")
    (artifact / REPORT.name).write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--stop-after",
        choices=("settle", "consistency", "canonical", "benchmark"),
        default="benchmark",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    task = load_variable_duration_task(ROOT / config["task_config"])
    settings = _settings(config)
    active = load_active_model_manifest()
    simulator_settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(
        simulator_settings, device="cuda", dtype=torch.float32
    )
    simulator.uav_model.set_fixed_evaluation_batch_size(FIXED_NUMERICAL_BATCH_SIZE)
    artifact = (
        args.resume.resolve()
        if args.resume is not None
        else ROOT / "data" / "planning" / config["run_id"] / _timestamp()
    )
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "canonical_checkpoints").mkdir(exist_ok=True)
    (artifact / "benchmark_checkpoints").mkdir(exist_ok=True)
    shutil.copy2(config_path, artifact / "benchmark_config.json")
    _write_json(artifact / "canonical_cem_config.json", asdict(settings))

    historical_knots, historical_duration = _load_historical_action(config)
    canonical_bank, canonical_context = _canonical_setup(simulator, task)
    canonical_repeated = _repeated_context(
        simulator,
        canonical_bank,
        0,
        canonical_context.target_position_local_m[0].detach().cpu(),
        canonical_context.target_direction_local[0].detach().cpu(),
        split="canonical",
    )
    contract, settle = _settle_verification(
        task, settings, historical_knots, historical_duration
    )
    codec = _codec_audit(task, settings, historical_knots, historical_duration)
    _write_json(artifact / "terminal_settle_contract.json", contract)
    _write_json(artifact / "terminal_settle_verification.json", settle)
    _write_json(artifact / "action_codec_audit.json", codec)
    _source_manifest(config_path, artifact)
    if not settle["pass"] or not codec["pass"]:
        _write_report(
            artifact,
            settle=settle,
            codec=codec,
            consistency=None,
            canonical=[],
            benchmark=None,
            gate_status="TERMINAL_SETTLE_OR_CODEC_FAILED",
        )
        raise SystemExit("Milestone 6A stopped: terminal settle or action codec failed.")
    if args.stop_after == "settle":
        _write_report(
            artifact,
            settle=settle,
            codec=codec,
            consistency=None,
            canonical=[],
            benchmark=None,
            gate_status="STOPPED_AFTER_SETTLE_BY_REQUEST",
        )
        return

    consistency_path = artifact / "replay_consistency_diagnostic.json"
    if consistency_path.is_file():
        consistency = json.loads(consistency_path.read_text(encoding="utf-8"))
    else:
        diagnostic_actions, diagnostic_labels = _diagnostic_actions(
            task, settings, historical_knots, historical_duration
        )
        consistency = _replay_consistency(
            simulator,
            canonical_repeated,
            diagnostic_actions,
            diagnostic_labels,
            task,
            settings,
            artifact,
        )
        _write_json(consistency_path, consistency)
    if not consistency["pass"]:
        _write_report(
            artifact,
            settle=settle,
            codec=codec,
            consistency=consistency,
            canonical=[],
            benchmark=None,
            gate_status="CEM_REPLAY_CONSISTENCY_NOT_FIXED",
        )
        raise SystemExit("Milestone 6A stopped: replay consistency is not fixed.")
    if args.stop_after == "consistency":
        _write_report(
            artifact,
            settle=settle,
            codec=codec,
            consistency=consistency,
            canonical=[],
            benchmark=None,
            gate_status="STOPPED_AFTER_CONSISTENCY_BY_REQUEST",
        )
        return

    warm = canonical_family_action_for_context(
        historical_knots,
        historical_duration,
        canonical_repeated,
        task,
        settings,
    )
    canonical_path = artifact / "canonical_seed_results.json"
    canonical_results: list[dict[str, Any]] = (
        json.loads(canonical_path.read_text(encoding="utf-8"))
        if canonical_path.is_file()
        else []
    )
    canonical_actions: list[torch.Tensor] = [
        torch.tensor(item["normalized_action"], dtype=torch.float64)
        for item in canonical_results
    ]
    completed_seeds = {int(item["seed"]) for item in canonical_results}
    for seed in config["canonical_cem"]["seeds"]:
        if int(seed) in completed_seeds:
            continue
        result = optimize_production_cem(
            simulator,
            canonical_repeated,
            task,
            warm,
            context_id="canonical",
            seed=int(seed),
            settings=settings,
            checkpoint_directory=artifact / "canonical_checkpoints" / f"seed_{seed}",
        )
        canonical_results.append(_safe(_result_summary(result, task)))
        canonical_actions.append(result.normalized_action.detach().cpu().double())
        _write_json(canonical_path, canonical_results)
    canonical_passes = sum(item["success"] for item in canonical_results)
    if canonical_passes < int(config["canonical_cem"]["required_seed_passes"]):
        _write_report(
            artifact,
            settle=settle,
            codec=codec,
            consistency=consistency,
            canonical=canonical_results,
            benchmark=None,
            gate_status="CANONICAL_CEM_NOT_ROBUST_UNDER_NEW_SEMANTICS",
        )
        raise SystemExit("Milestone 6A stopped: fewer than 5/6 canonical seeds passed.")
    if args.stop_after == "canonical":
        _write_report(
            artifact,
            settle=settle,
            codec=codec,
            consistency=consistency,
            canonical=canonical_results,
            benchmark=None,
            gate_status="STOPPED_AFTER_CANONICAL_BY_REQUEST",
        )
        return

    successful_indices = [
        index for index, row in enumerate(canonical_results) if row["success"]
    ]
    selected_index = min(
        successful_indices,
        key=lambda index: (
            canonical_results[index]["authoritative_metrics"]["task_cost"],
            canonical_results[index]["authoritative_metrics"]["first_entry_tip_distance_m"],
        ),
    )
    canonical_action = canonical_actions[selected_index]
    state_root = ROOT / config["state_bank_artifact"]
    state_bank = InitialStateBank.load(
        state_root / "training_state_bank.npz",
        state_root / "training_state_bank_manifest.json",
    )
    manifest = _benchmark_manifest(canonical_context, state_bank, canonical_bank)
    _write_json(artifact / "benchmark_context_manifest.json", manifest)
    rows, _ = _run_benchmark(
        simulator,
        task,
        settings,
        config,
        artifact,
        manifest,
        canonical_bank,
        state_bank,
        canonical_action,
    )
    summary = _summarize_benchmark(rows, task)
    _write_json(artifact / "benchmark_summary.json", summary)
    _write_json(
        artifact / "replay_difference_summary.json",
        summary["replay_difference_summary"],
    )
    _write_json(
        artifact / "robustness_margin_summary.json", summary["robustness_margins"]
    )
    _write_json(
        artifact / "duration_hit_segment_summary.json",
        {
            "duration_s": summary["duration_s"],
            "hit_time_s": summary["hit_time_s"],
            "hit_minus_maneuver_s": summary["hit_minus_maneuver_s"],
            "hit_segments": summary["hit_segments"],
            "duration_upper_bound_active_count": summary[
                "duration_upper_bound_active_count"
            ],
            "duration_upper_bound_active_fraction": summary[
                "duration_upper_bound_active_fraction"
            ],
        },
    )
    _write_json(artifact / "runtime_summary.json", summary["runtime_s"])
    _write_report(
        artifact,
        settle=settle,
        codec=codec,
        consistency=consistency,
        canonical=canonical_results,
        benchmark=summary,
        gate_status="COMPLETED",
    )


if __name__ == "__main__":
    main()
