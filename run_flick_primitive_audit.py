"""Canonical feasibility audits for compact smooth two-pulse flicks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from learning.cem_teacher_support import (
    load_fixed_production_environment,
    production_cem_settings,
)
from learning.context_sampling import ContextSpecification, build_context_from_specification
from learning.policy_action import decode_policy_action
from planning.flick_primitive import (
    DIRECTED_FLICK_PARAMETER_NAMES,
    FLICK_PARAMETER_NAMES,
    FlickCemResult,
    FlickCemSettings,
    FlickPrimitiveBounds,
    directed_flick_parameters_to_knots,
    encode_directed_flick_as_production_action,
    encode_flick_as_production_action,
    flick_parameters_to_knots,
    optimize_flick_cem,
    project_directed_flick_parameters,
    project_flick_parameters,
)
from planning.production_cem import (
    FIXED_NUMERICAL_BATCH_SIZE,
    record_normalized_actions_fixed_batch,
)
from simulator.production import active_model_paths, load_active_model_manifest


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "planning" / "five_parameter_flick_audit_v1.json"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter_names(config: dict[str, Any]) -> tuple[str, ...]:
    primitive_type = config.get("primitive_type")
    if primitive_type == "shared_axis_5d":
        return FLICK_PARAMETER_NAMES
    if primitive_type == "independent_axes_7d":
        return DIRECTED_FLICK_PARAMETER_NAMES
    raise ValueError("Unsupported compact flick primitive type.")


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") not in (
        "five_parameter_flick_audit_v1",
        "seven_parameter_flick_audit_v1",
    ):
        raise ValueError("Unsupported compact flick configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The flick audit requires the frozen production model.")
    if int(config.get("fixed_numerical_batch_size", 0)) != FIXED_NUMERICAL_BATCH_SIZE:
        raise ValueError("The fixed numerical batch must remain 2048.")
    if tuple(config["primitive"]["parameter_names"]) != _parameter_names(config):
        raise ValueError("The configured primitive parameter schema is inconsistent.")
    policy = config["artifact_policy"]
    if any(
        bool(policy[name])
        for name in (
            "learning_used",
            "new_policy_training",
            "production_model_modified",
            "protected_test_allowed",
            "real_hardware_allowed",
        )
    ):
        raise ValueError("A prohibited action was enabled in the flick audit.")
    if policy["protected_test_id"] != "fig8vertical_002":
        raise ValueError("Protected-test identity changed.")


def _bounds(config: dict[str, Any]) -> FlickPrimitiveBounds:
    source = config["primitive"]
    return FlickPrimitiveBounds(
        azimuth_offset_min_rad=float(source["azimuth_offset_rad"][0]),
        azimuth_offset_max_rad=float(source["azimuth_offset_rad"][1]),
        elevation_min_rad=float(source["elevation_rad"][0]),
        elevation_max_rad=float(source["elevation_rad"][1]),
        pulse_min_m_s2=float(source["pulse_m_s2"][0]),
        pulse_max_m_s2=float(source["pulse_m_s2"][1]),
        duration_min_s=float(source["maneuver_duration_s"][0]),
        duration_max_s=float(source["maneuver_duration_s"][1]),
        switch_fraction=float(source["switch_fraction"]),
    )


def _cem_settings(config: dict[str, Any], maximum_iterations: int | None) -> FlickCemSettings:
    source = config["cem"]
    configured_maximum = int(source["maximum_iterations"])
    selected_maximum = configured_maximum if maximum_iterations is None else int(maximum_iterations)
    if selected_maximum < int(source["minimum_iterations"]):
        raise ValueError("The requested maximum is below the required minimum iterations.")
    return FlickCemSettings(
        population=int(source["population"]),
        elite_fraction=float(source["elite_fraction"]),
        maximum_iterations=selected_maximum,
        minimum_iterations=int(source["minimum_iterations"]),
        polish_iterations_after_success=int(source["polish_iterations_after_success"]),
        authoritative_top_n=int(source["authoritative_top_n"]),
        old_distribution_weight=float(source["old_distribution_weight"]),
        elite_distribution_weight=float(source["elite_distribution_weight"]),
        covariance_jitter=float(source["covariance_jitter"]),
        initial_mean=tuple(float(value) for value in source["initial_mean"]),
        initial_std=tuple(float(value) for value in source["initial_std"]),
        std_floor=tuple(float(value) for value in source["std_floor"]),
    )


def _canonical_context(simulator, canonical_bank, config: dict[str, Any]):
    source = config["canonical_context"]
    specification = ContextSpecification(
        torch.full(
            (FIXED_NUMERICAL_BATCH_SIZE,), int(source["state_index"]), dtype=torch.int64
        ),
        torch.tensor(
            [source["target_local_m"]] * FIXED_NUMERICAL_BATCH_SIZE,
            dtype=torch.float32,
        ),
        torch.tensor(
            [source["direction_local"]] * FIXED_NUMERICAL_BATCH_SIZE,
            dtype=torch.float32,
        ),
        "compact_flick_canonical",
    )
    return build_context_from_specification(simulator, canonical_bank, specification)


def _parameterization_verification(
    config: dict[str, Any],
    task,
    production_settings,
    bounds: FlickPrimitiveBounds,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(5042)
    directed = config["primitive_type"] == "independent_axes_7d"
    dimension = 7 if directed else 5
    raw = torch.randn((256, dimension), generator=generator, dtype=torch.float64)
    if directed:
        raw[:, (0, 2)] *= math.pi
        raw[:, (1, 3)] *= 0.8
        raw[:, 4:6] = 10.0 + 8.0 * raw[:, 4:6]
        raw[:, 6] = 1.10 + 0.40 * raw[:, 6]
        parameters = project_directed_flick_parameters(raw, bounds)
        knot_decoder = directed_flick_parameters_to_knots
        action_encoder = encode_directed_flick_as_production_action
    else:
        raw[:, 0] *= math.pi
        raw[:, 1] *= 0.8
        raw[:, 2:4] = 10.0 + 8.0 * raw[:, 2:4]
        raw[:, 4] = 1.10 + 0.40 * raw[:, 4]
        parameters = project_flick_parameters(raw, bounds)
        knot_decoder = flick_parameters_to_knots
        action_encoder = encode_flick_as_production_action
    directions = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64).expand(256, 3)
    expected_knots, expected_duration = knot_decoder(
        parameters, directions, bounds=bounds
    )
    encoded = action_encoder(
        parameters, directions, task, production_settings, bounds=bounds
    )
    decoded = decode_policy_action(
        encoded, task, duration_max_s=production_settings.duration_max_s
    )
    maximum_knot_difference = float(
        torch.max(torch.abs(expected_knots - decoded.acceleration_knots_local_m_s2))
    )
    maximum_duration_difference = float(
        torch.max(torch.abs(expected_duration - decoded.duration_s))
    )
    endpoint_acceleration = torch.maximum(
        torch.linalg.vector_norm(expected_knots[:, 0], dim=-1),
        torch.linalg.vector_norm(expected_knots[:, -1], dim=-1),
    )
    return {
        "schema": f"{dimension}_parameter_flick_verification_v1",
        "sample_count": 256,
        "parameter_dimension": dimension,
        "primitive_type": config["primitive_type"],
        "knot_count": int(expected_knots.shape[1]),
        "normalized_action_dimension": int(encoded.shape[1]),
        "all_finite": bool(
            torch.isfinite(expected_knots).all()
            and torch.isfinite(expected_duration).all()
            and torch.isfinite(encoded).all()
        ),
        "maximum_knot_norm_m_s2": float(
            torch.linalg.vector_norm(expected_knots, dim=-1).max()
        ),
        "maximum_endpoint_acceleration_m_s2": float(endpoint_acceleration.max()),
        "maximum_codec_knot_difference_m_s2": maximum_knot_difference,
        "maximum_codec_duration_difference_s": maximum_duration_difference,
        "switch_fraction": bounds.switch_fraction,
        "production_codec_used": True,
        "pass": bool(
            torch.isfinite(encoded).all()
            and float(torch.linalg.vector_norm(expected_knots, dim=-1).max())
            <= task.maximum_command_acceleration_m_s2 + 1.0e-10
            and float(endpoint_acceleration.max()) <= 1.0e-10
            and maximum_knot_difference <= 1.0e-12
            and maximum_duration_difference <= 1.0e-12
        ),
    }


def _hard_gates(metrics: dict[str, Any], task) -> dict[str, bool]:
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


def _result_row(
    result: FlickCemResult, task, parameter_names: tuple[str, ...]
) -> dict[str, Any]:
    metrics = dict(result.authoritative_metrics)
    metrics["hard_gates"] = _hard_gates(metrics, task)
    return {
        "context_id": result.context_id,
        "seed": result.seed,
        "success": result.success,
        "parameters": dict(zip(parameter_names, result.parameters.tolist(), strict=True)),
        "normalized_action": result.normalized_action.tolist(),
        "authoritative_metrics": metrics,
        "iterations": result.iterations,
        "population_rollouts": result.population_rollouts,
        "authoritative_rollouts": result.authoritative_rollouts,
        "runtime_s": result.runtime_s,
        "first_success_iteration": result.first_success_iteration,
        "history": list(result.history),
    }


def _best_result(results: list[FlickCemResult], task) -> FlickCemResult:
    def key(item: FlickCemResult) -> tuple[float, float, float, float, float]:
        metrics = item.authoritative_metrics
        success = bool(metrics["success"])
        feasible = bool(metrics["feasible"])
        tip_entry = metrics["first_entry_marker"] == 10
        speed_deficit = max(
            0.0,
            task.minimum_directed_speed_m_s
            - float(metrics["first_entry_directed_speed_m_s"]),
        ) / task.minimum_directed_speed_m_s
        direction_deficit = max(
            0.0,
            float(metrics["first_entry_direction_angle_deg"])
            - task.maximum_direction_error_deg,
        ) / task.maximum_direction_error_deg
        category = (
            0.0 if success else 1.0 if feasible and tip_entry else 2.0 if feasible else 3.0
        )
        return (
            category,
            speed_deficit**2 + direction_deficit**2
            if tip_entry
            else float(metrics["minimum_tip_target_distance_m"]),
            float(metrics["first_entry_tip_distance_m"])
            if tip_entry
            else float(metrics["task_cost"]),
            float(metrics["first_entry_direction_angle_deg"]) if tip_entry else 0.0,
            -float(metrics["first_entry_directed_speed_m_s"]) if tip_entry else 0.0,
        )

    return min(results, key=key)


def _report(
    artifact: Path,
    config: dict[str, Any],
    results: list[FlickCemResult],
    verification: dict[str, Any],
    best: FlickCemResult,
    parameter_names: tuple[str, ...],
) -> str:
    pass_count = sum(result.success for result in results)
    family_support = pass_count >= 1
    robust_support = pass_count >= 5 and len(results) >= 6
    metrics = best.authoritative_metrics
    rows = [
        "| Seed | PASS | Minimum tip mm | Entry directed m/s | Entry direction deg | UAV disp m | UAV speed m/s | Duration s | Runtime s |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        item = result.authoritative_metrics
        rows.append(
            f"| {result.seed} | {'YES' if result.success else 'NO'} | "
            f"{1000.0 * float(item['minimum_tip_target_distance_m']):.2f} | "
            f"{float(item['first_entry_directed_speed_m_s']):.3f} | "
            f"{float(item['first_entry_direction_angle_deg']):.2f} | "
            f"{float(item['maximum_uav_displacement_m']):.3f} | "
            f"{float(item['maximum_uav_speed_m_s']):.3f} | "
            f"{float(item['maneuver_duration_s']):.3f} | {result.runtime_s:.1f} |"
        )
    prefix = "FIVE_PARAMETER" if len(parameter_names) == 5 else "SEVEN_PARAMETER"
    classification = (
        f"{prefix}_FLICK_ROBUST_SUPPORT"
        if robust_support
        else f"{prefix}_FLICK_SUPPORT_FOUND"
        if family_support
        else f"{prefix}_FLICK_SUPPORT_NOT_FOUND"
    )
    dimension = len(parameter_names)
    primitive_title = (
        "Five-Parameter Shared-Axis Flick"
        if dimension == 5
        else "Seven-Parameter Independent-Axes Flick"
    )
    text = f"""# {primitive_title} Feasibility Report

## Purpose

This audit tests whether a deliberately small, interpretable {dimension}-D maneuver family contains a scientifically valid canonical cable strike. No policy was trained. The production model, action codec, smooth settle, fixed-2048 numerical contract, and scientific gates were unchanged.

## Primitive

The parameters are `{', '.join(parameter_names)}`. Both acceleration pulses use smooth `sin^2` lobes around a fixed half-time reversal. Directions are defined relative to the requested horizontal target direction, so commanded lateral acceleration continues to exercise the production UAV roll/pitch dynamics.

Every primitive is decoded to 16 acceleration knots and encoded immediately as the normalized production action `[49]`. There is no primitive-specific simulator path.

## Contract verification

- Verification: **{'PASS' if verification['pass'] else 'FAIL'}**
- Random samples: {verification['sample_count']}
- Maximum acceleration-knot norm: {verification['maximum_knot_norm_m_s2']:.6f} m/s^2
- Maximum start/end acceleration: {verification['maximum_endpoint_acceleration_m_s2']:.3e} m/s^2
- Maximum codec knot difference: {verification['maximum_codec_knot_difference_m_s2']:.3e} m/s^2
- Maximum codec duration difference: {verification['maximum_codec_duration_difference_s']:.3e} s

## Canonical CEM audit

Population is exactly 2048. CEM optimizes only {dimension} primitive parameters. Top candidates and the final selected action are replayed authoritatively through logical batch-one with cyclic padding to 2048.

The audit ranks unchanged scientific success first, then feasible tip-first entries by their speed/direction hard-gate deficit, then other feasible candidates by true minimum tip distance. The legacy production planner reward and simulator are not modified.

{chr(10).join(rows)}

Canonical authoritative successes: **{pass_count}/{len(results)}**.

## Best authoritative maneuver

- Seed: {best.seed}
- Parameters: `{json.dumps(dict(zip(parameter_names, best.parameters.tolist(), strict=True)))}`
- Scientific success: **{'YES' if best.success else 'NO'}**
- Feasible: **{'YES' if metrics['feasible'] else 'NO'}**
- Minimum tip distance: {1000.0 * float(metrics['minimum_tip_target_distance_m']):.2f} mm
- Entry tip distance: {1000.0 * float(metrics['first_entry_tip_distance_m']):.2f} mm
- Entry directed tip speed: {float(metrics['first_entry_directed_speed_m_s']):.3f} m/s
- Entry direction error: {float(metrics['first_entry_direction_angle_deg']):.2f} deg
- Tip-first marker: {metrics['first_entry_marker']}
- UAV displacement: {float(metrics['maximum_uav_displacement_m']):.3f} m
- UAV speed: {float(metrics['maximum_uav_speed_m_s']):.3f} m/s
- Command acceleration: {float(metrics['maximum_command_acceleration_m_s2']):.3f} m/s^2
- Maneuver duration: {float(metrics['maneuver_duration_s']):.3f} s
- Hit segment: {metrics['hit_segment']}

## Interpretation

Family support is **{'present' if family_support else 'not demonstrated'}**. Robust six-seed support is **{'present' if robust_support else 'not demonstrated'}**. This audit intentionally stops at the representation question; no forward model or other learner is authorized by this result alone.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Learning:
        NOT USED

    Primitive:
        {primitive_title.upper()}

    Primitive dimensions:
        {dimension}

    Production action representation:
        NORMALIZED [49]

    Canonical authoritative CEM:
        {pass_count} / {len(results)} seeds PASS

    Family support:
        {'YES' if family_support else 'NO'}

    Robust support:
        {'YES' if robust_support else 'NO'}

    Classification:
        {classification}

    Production model modified:
        NO

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    artifact_report = artifact / (
        "FIVE_PARAMETER_FLICK_FEASIBILITY_REPORT.md"
        if dimension == 5
        else "SEVEN_PARAMETER_FLICK_FEASIBILITY_REPORT.md"
    )
    root_report = ROOT / artifact_report.name
    artifact_report.write_text(text, encoding="utf-8")
    root_report.write_text(text, encoding="utf-8")
    return classification


def run(
    config_path: Path,
    *,
    seeds: list[int] | None = None,
    maximum_iterations: int | None = None,
    artifact_directory: Path | None = None,
) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    parameter_names = _parameter_names(config)
    directed = config["primitive_type"] == "independent_axes_7d"
    selected_seeds = list(config["cem"]["seeds"] if seeds is None else seeds)
    if not selected_seeds:
        raise ValueError("At least one deterministic seed is required.")
    artifact = (
        ROOT / "data" / "planning" / str(config["run_id"]) / _timestamp()
        if artifact_directory is None
        else artifact_directory.resolve()
    )
    artifact.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, artifact / "config.json")
    _write_json(
        artifact / "run_status.json",
        {
            "status": "STARTING",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": selected_seeds,
            "completed_seeds": 0,
        },
    )

    simulator, task, _training_bank, _validation_bank, canonical_bank, _settings = (
        load_fixed_production_environment(config)
    )
    production_settings = replace(
        production_cem_settings(config), population=FIXED_NUMERICAL_BATCH_SIZE
    )
    bounds = _bounds(config)
    if bounds.pulse_max_m_s2 != task.maximum_command_acceleration_m_s2:
        raise RuntimeError("Primitive pulse bound differs from the scientific command gate.")
    flick_settings = _cem_settings(config, maximum_iterations)
    context = _canonical_context(simulator, canonical_bank, config)
    verification = _parameterization_verification(
        config, task, production_settings, bounds
    )
    _write_json(artifact / "primitive_contract_verification.json", verification)
    if not verification["pass"]:
        raise RuntimeError("Compact flick primitive contract verification failed.")

    results: list[FlickCemResult] = []
    for seed in selected_seeds:
        print(f"{len(parameter_names)}-parameter flick: seed {seed}", flush=True)
        result = optimize_flick_cem(
            simulator,
            context,
            task,
            context_id="canonical",
            seed=int(seed),
            production_settings=production_settings,
            flick_settings=flick_settings,
            bounds=bounds,
            checkpoint_directory=artifact / "checkpoints" / f"seed_{seed}",
            parameter_projector=(
                project_directed_flick_parameters if directed else project_flick_parameters
            ),
            action_encoder=(
                encode_directed_flick_as_production_action
                if directed
                else encode_flick_as_production_action
            ),
            duration_index=(6 if directed else 4),
        )
        results.append(result)
        rows = [_result_row(item, task, parameter_names) for item in results]
        _write_json(artifact / "canonical_seed_results.json", rows)
        _write_json(
            artifact / "run_status.json",
            {
                "status": "RUNNING",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seeds": selected_seeds,
                "completed_seeds": len(results),
                "latest_seed": seed,
                "latest_success": result.success,
            },
        )
        print(
            f"seed {seed}: pass={result.success} "
            f"min_tip={1000.0 * float(result.authoritative_metrics['minimum_tip_target_distance_m']):.1f}mm "
            f"entry_directed={float(result.authoritative_metrics['first_entry_directed_speed_m_s']):.2f}m/s",
            flush=True,
        )

    best = _best_result(results, task)
    np.save(artifact / "best_normalized_action.npy", best.normalized_action.numpy())
    _write_json(
        artifact / "best_flick_parameters.json",
        dict(zip(parameter_names, best.parameters.tolist(), strict=True)),
    )
    trajectory = record_normalized_actions_fixed_batch(
        simulator,
        context,
        best.normalized_action[None],
        task,
        production_settings,
        record_count=1,
    )
    np.savez_compressed(
        artifact / "best_authoritative_replay.npz",
        times_s=trajectory.times_s.numpy(),
        command_positions_m=trajectory.command_positions_m[:, 0].numpy(),
        command_velocities_m_s=trajectory.command_velocities_m_s[:, 0].numpy(),
        command_accelerations_m_s2=trajectory.command_accelerations_m_s2[:, 0].numpy(),
        uav_positions_m=trajectory.uav_positions_m[:, 0].numpy(),
        uav_velocities_m_s=trajectory.uav_velocities_m_s[:, 0].numpy(),
        cable_positions_m=trajectory.cable_positions_m[:, 0].numpy(),
        cable_velocities_m_s=trajectory.cable_velocities_m_s[:, 0].numpy(),
    )
    _write_json(
        artifact / "best_authoritative_metrics.json",
        {**best.authoritative_metrics, "hard_gates": _hard_gates(best.authoritative_metrics, task)},
    )
    classification = _report(
        artifact, config, results, verification, best, parameter_names
    )

    manifest_paths = {
        "runner": Path(__file__),
        "primitive": ROOT / "planning" / "flick_primitive.py",
        "production_cem": ROOT / "planning" / "production_cem.py",
        "variable_duration": ROOT / "planning" / "variable_duration.py",
        "policy_action": ROOT / "learning" / "policy_action.py",
        "task": ROOT / config["task_config"],
        "config": config_path,
    }
    for name, path in active_model_paths(load_active_model_manifest()).items():
        if path.is_file():
            manifest_paths[f"active_{name}"] = path
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": f"{len(parameter_names)}_parameter_flick_source_hashes_v1",
            "files": {
                name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                for name, path in manifest_paths.items()
            },
        },
    )
    _write_json(
        artifact / "run_status.json",
        {
            "status": "COMPLETE",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": selected_seeds,
            "completed_seeds": len(results),
            "success_count": sum(result.success for result in results),
            "classification": classification,
        },
    )
    print(f"artifact: {artifact}", flush=True)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--maximum-iterations", type=int)
    parser.add_argument("--artifact-directory", type=Path)
    arguments = parser.parse_args()
    with torch.no_grad():
        run(
            arguments.config.resolve(),
            seeds=arguments.seeds,
            maximum_iterations=arguments.maximum_iterations,
            artifact_directory=arguments.artifact_directory,
        )


if __name__ == "__main__":
    main()
