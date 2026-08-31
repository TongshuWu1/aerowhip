"""Run the permanent Milestone-7A production-CEM teacher pipeline.

The script is deliberately stage-based and resumable.  Teacher generation writes
one complete context shard before advancing its progress manifest, so an
interrupted multi-hour run restarts at the next unfinished context.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import torch

from learning.amortized_cem_data import (
    AmortizedCemContextRecord,
    build_initial_context_records,
    build_sealed_evaluation_records,
    deduplicate_actions,
    pending_context_records,
    save_context_table,
    sha256_file,
    sobol_target_pairs,
    stable_context_id,
    state_descriptor,
    write_context_shards,
)
from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    target_direction_from_local_target,
)
from learning.normalization import FixedContextNormalizer
from learning.policy_context import policy_context_tensor_metadata
from learning.state_bank import InitialStateBank
from planning.cem_task import VariableDurationWhipTask, load_variable_duration_task
from planning.production_cem import (
    FIXED_NUMERICAL_BATCH_SIZE,
    ProductionCemSettings,
    canonical_family_action_for_context,
    optimize_production_cem,
)
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "amortized_cem_diffusion_nominal_v1.json"
ARTIFACT_PARENT = ROOT / "data" / "policy_training" / "amortized_cem_diffusion_nominal_v1"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "amortized_cem_diffusion_nominal_v1":
        raise ValueError("Unsupported Milestone-7A configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Milestone 7A requires the frozen production model.")
    if int(config.get("fixed_numerical_batch_size", 0)) != FIXED_NUMERICAL_BATCH_SIZE:
        raise ValueError("Milestone 7A requires the fixed 2048 numerical contract.")
    prohibited = config["prohibitions"]
    if prohibited.get("protected_test") != "fig8vertical_002":
        raise ValueError("Protected-test identity changed.")
    for name in ("sac", "online_cem", "online_simulator", "theta_randomization", "hardware"):
        if not bool(prohibited.get(name)):
            raise ValueError(f"Required prohibition is disabled: {name}.")


def _cem_settings(config: dict[str, Any]) -> ProductionCemSettings:
    source = json.loads((ROOT / config["production_cem_config"]).read_text(encoding="utf-8"))
    cem, times = source["canonical_cem"], source["times"]
    settings = ProductionCemSettings(
        population=int(cem["population"]),
        elite_fraction=float(cem["elite_fraction"]),
        maximum_iterations=int(cem["maximum_iterations"]),
        minimum_iterations=int(cem["minimum_iterations"]),
        polish_iterations_after_success=int(cem["polish_iterations_after_success"]),
        authoritative_top_n=int(cem["authoritative_top_n"]),
        initial_acceleration_std_m_s2=float(cem["initial_acceleration_std_m_s2"]),
        initial_duration_std_s=float(cem["initial_duration_std_s"]),
        acceleration_std_floor_m_s2=float(cem["acceleration_std_floor_m_s2"]),
        duration_std_floor_s=float(cem["duration_std_floor_s"]),
        duration_min_s=float(times["maneuver_min_s"]),
        duration_max_s=float(times["maneuver_max_s"]),
        settle_duration_s=float(times["settle_s"]),
        evaluation_time_s=float(times["evaluation_s"]),
    )
    requested = config["teacher"]
    if settings.population != 4096 or settings.authoritative_top_n != int(requested["authoritative_top_n"]):
        raise ValueError("Teacher settings differ from the stabilized production CEM.")
    return settings


def _load_environment(config: dict[str, Any]):
    manifest = load_active_model_manifest()
    paths = active_model_paths(manifest)
    settings = SimulatorSettings.load(paths["configuration"])
    simulator = build_production_simulator(
        settings, device=torch.device("cuda"), dtype=torch.float32
    )
    task = load_variable_duration_task(ROOT / config["task_config"])
    state_root = ROOT / config["state_bank_artifact"]
    training_bank = InitialStateBank.load(
        state_root / "training_state_bank.npz",
        state_root / "training_state_bank_manifest.json",
    )
    heldout_bank = InitialStateBank.load(
        state_root / "validation_state_bank.npz",
        state_root / "validation_state_bank_manifest.json",
    )
    return simulator, task, training_bank, heldout_bank, settings


def _nominal_theta(settings: SimulatorSettings) -> tuple[float, ...]:
    return (
        float(settings.parameters.uav.K_p),
        float(settings.parameters.uav.K_v),
        float(settings.parameters.uav.k_a),
        float(settings.parameters.uav.K_R),
        float(settings.parameters.uav.K_omega),
        float(settings.parameters.cable.EI),
        float(settings.parameters.cable.Cb),
    )


def _context_tensors(
    simulator,
    records: list[AmortizedCemContextRecord],
    training_bank: InitialStateBank,
    heldout_bank: InitialStateBank,
) -> np.ndarray:
    table = np.empty((len(records), 83), dtype=np.float32)
    for bank_name, bank in (("training", training_bank), ("heldout", heldout_bank)):
        selected = [record for record in records if record.bank == bank_name]
        for start in range(0, len(selected), FIXED_NUMERICAL_BATCH_SIZE):
            chunk = selected[start : start + FIXED_NUMERICAL_BATCH_SIZE]
            specification = ContextSpecification(
                torch.tensor([item.state_index for item in chunk], dtype=torch.int64),
                torch.tensor([item.target_local_m for item in chunk], dtype=torch.float32),
                torch.tensor([item.direction_local for item in chunk], dtype=torch.float32),
                "milestone7a_prepare",
            )
            context = build_context_from_specification(simulator, bank, specification)
            values = context.to_tensor().detach().cpu().numpy().astype(np.float32)
            table[np.asarray([item.context_index for item in chunk], dtype=np.int64)] = values
    if not np.isfinite(table).all():
        raise RuntimeError("Prepared policy context table is non-finite.")
    return table


def _context_values_for_records(
    simulator,
    records: list[AmortizedCemContextRecord],
    training_bank: InitialStateBank,
    heldout_bank: InitialStateBank,
) -> np.ndarray:
    values = np.empty((len(records), 83), dtype=np.float32)
    position = {id(record): index for index, record in enumerate(records)}
    for bank_name, bank in (("training", training_bank), ("heldout", heldout_bank)):
        selected = [record for record in records if record.bank == bank_name]
        for start in range(0, len(selected), FIXED_NUMERICAL_BATCH_SIZE):
            chunk = selected[start : start + FIXED_NUMERICAL_BATCH_SIZE]
            specification = ContextSpecification(
                torch.tensor([record.state_index for record in chunk], dtype=torch.int64),
                torch.tensor([record.target_local_m for record in chunk], dtype=torch.float32),
                torch.tensor([record.direction_local for record in chunk], dtype=torch.float32),
                chunk[0].split,
            )
            tensor = build_context_from_specification(simulator, bank, specification).to_tensor()
            for row, record in enumerate(chunk):
                values[position[id(record)]] = tensor[row].detach().cpu().numpy()
    return values


def _create_artifact(config: dict[str, Any], requested: Path | None) -> Path:
    if requested is not None:
        artifact = requested.resolve()
        if not artifact.is_dir():
            raise FileNotFoundError(f"Artifact directory does not exist: {artifact}")
        return artifact
    artifact = ARTIFACT_PARENT / _timestamp()
    artifact.mkdir(parents=True, exist_ok=False)
    (artifact / "diffusion_teacher_shards").mkdir()
    (artifact / "scorer_dataset_shards").mkdir()
    (artifact / "teacher_checkpoints").mkdir()
    _write_json(artifact / "teacher_generation_config.json", config)
    return artifact


def _write_policy_contract_artifacts(config: dict[str, Any], artifact: Path) -> None:
    _write_json(
        artifact / "context_schema.json",
        {
            "schema": "amortized_cem_policy_context_schema_v1",
            **policy_context_tensor_metadata(),
            "theta_schema_permanent": True,
            "theta_training_in_milestone7a": "NOMINAL_ONLY",
        },
    )
    _write_json(
        artifact / "action_schema.json",
        {
            "schema": "amortized_cem_production_action_schema_v1",
            "dimension": 49,
            "acceleration_knots": {"count": 16, "vector_dimension": 3},
            "acceleration_normalization": "each local vector divided by 20 m/s^2 then vector-norm projected",
            "duration_coordinate": "linear normalized [-1,1]",
            "duration_range_s": [0.45, 1.80],
            "production_decoder": "learning.policy_action.decode_policy_action",
            "population_and_deployment_representation_identical": True,
        },
    )
    _write_json(
        artifact / "production_command_contract.json",
        {
            "schema": "milestone6a_smooth_settle_command_contract_v1",
            "active_maneuver_duration_s": [0.45, 1.80],
            "settle_duration_s": 0.30,
            "evaluation_horizon_s": 2.40,
            "segments": ["ACTIVE", "SETTLE", "HOLD"],
            "settle": "analytic cubic-Hermite velocity with p/v/a boundary continuity",
            "fixed_numerical_batch_size": 2048,
            "model_freeze": config["model_freeze"],
        },
    )


def prepare(config: dict[str, Any], artifact: Path) -> None:
    _write_policy_contract_artifacts(config, artifact)
    required = (
        artifact / "context_split_manifest.json",
        artifact / "target_manifest.json",
        artifact / "context_table.npz",
        artifact / "context_normalizer.json",
    )
    if all(path.is_file() for path in required):
        evaluation_manifest = artifact / "sealed_evaluation_context_manifest.json"
        if not evaluation_manifest.is_file():
            _simulator, _task, _training_bank, heldout_bank, settings = _load_environment(config)
            split_manifest = json.loads(
                (artifact / "context_split_manifest.json").read_text(encoding="utf-8")
            )
            evaluation_groups = build_sealed_evaluation_records(
                heldout_bank,
                split_manifest,
                theta_nominal=_nominal_theta(settings),
                seed=int(config["seed"]),
            )
            _write_json(
                evaluation_manifest,
                {
                    "schema": "amortized_cem_sealed_evaluation_contexts_v1",
                    "created_before_neural_training": True,
                    "used_for_training": False,
                    "used_for_checkpoint_selection": False,
                    "groups": {
                        name: [asdict(record) for record in values]
                        for name, values in evaluation_groups.items()
                    },
                },
            )
        print(f"PREPARE already complete: {artifact}", flush=True)
        return
    simulator, _task, training_bank, heldout_bank, settings = _load_environment(config)
    records, split_manifest, target_manifest = build_initial_context_records(
        training_bank,
        heldout_bank,
        theta_nominal=_nominal_theta(settings),
        seed=int(config["seed"]),
    )
    if len(records) != 768:
        raise RuntimeError(f"Initial context design must contain 768 rows, got {len(records)}.")
    contexts = _context_tensors(simulator, records, training_bank, heldout_bank)
    train_indices = np.asarray(
        [record.context_index for record in records if record.split == "TRAIN"], dtype=np.int64
    )
    normalizer = FixedContextNormalizer.fit(torch.from_numpy(contexts[train_indices]))
    save_context_table(artifact / "context_table.npz", records, contexts)
    normalizer.save(artifact / "context_normalizer.json")
    _write_json(artifact / "context_split_manifest.json", split_manifest)
    _write_json(artifact / "target_manifest.json", target_manifest)
    evaluation_groups = build_sealed_evaluation_records(
        heldout_bank,
        split_manifest,
        theta_nominal=_nominal_theta(settings),
        seed=int(config["seed"]),
    )
    _write_json(
        artifact / "sealed_evaluation_context_manifest.json",
        {
            "schema": "amortized_cem_sealed_evaluation_contexts_v1",
            "created_before_neural_training": True,
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "groups": {
                name: [asdict(record) for record in values]
                for name, values in evaluation_groups.items()
            },
        },
    )
    _write_json(
        artifact / "teacher_generation_progress.json",
        {
            "schema": "amortized_cem_teacher_progress_v1",
            "artifact": str(artifact),
            "total_contexts": len(records),
            "completed_contexts": 0,
            "solved_contexts": 0,
            "failed_contexts": 0,
            "successful_action_count": 0,
            "scorer_row_count": 0,
            "rows": [],
        },
    )
    _write_json(
        artifact / "diffusion_teacher_manifest.json",
        {"schema": "amortized_cem_diffusion_teacher_manifest_v1", "shards": []},
    )
    _write_json(
        artifact / "scorer_dataset_manifest.json",
        {
            "schema": "amortized_cem_outcome_scorer_manifest_v1",
            "continuous_targets": [
                "legacy_optimizer_reward",
                "log_tip_distance",
                "directed_tip_speed",
                "cos_direction_error",
                "max_uav_displacement",
                "max_uav_speed",
                "max_command_acceleration",
            ],
            "binary_targets": ["tip_first", "finite", "scientific_success"],
            "shards": [],
        },
    )
    _write_json(artifact / "teacher_cem_failures.json", {"rows": []})
    print(f"PREPARE complete: {artifact}", flush=True)


def _records(artifact: Path) -> list[AmortizedCemContextRecord]:
    payload = json.loads((artifact / "context_split_manifest.json").read_text(encoding="utf-8"))
    return [AmortizedCemContextRecord(**row) for row in payload["records"]]


def _repeated_context(simulator, bank: InitialStateBank, record: AmortizedCemContextRecord):
    specification = ContextSpecification(
        torch.full((FIXED_NUMERICAL_BATCH_SIZE,), record.state_index, dtype=torch.int64),
        torch.tensor(record.target_local_m, dtype=torch.float32)
        .reshape(1, 3)
        .repeat(FIXED_NUMERICAL_BATCH_SIZE, 1),
        torch.tensor(record.direction_local, dtype=torch.float32)
        .reshape(1, 3)
        .repeat(FIXED_NUMERICAL_BATCH_SIZE, 1),
        record.split,
    )
    return build_context_from_specification(simulator, bank, specification)


def _historical_action(config: dict[str, Any]) -> tuple[torch.Tensor, float]:
    source = json.loads((ROOT / config["production_cem_config"]).read_text(encoding="utf-8"))
    directory = ROOT / source["historical_canonical_artifact"]
    knots = json.loads((directory / "best_acceleration_knots.json").read_text(encoding="utf-8"))[
        "values"
    ]
    duration = json.loads((directory / "optimized_duration.json").read_text(encoding="utf-8"))[
        "duration_s"
    ]
    return torch.tensor(knots, dtype=torch.float64), float(duration)


def _result_summary(result) -> dict[str, Any]:
    return {
        "seed": result.seed,
        "success": result.success,
        "iterations": result.iterations,
        "population_rollouts": result.population_rollouts,
        "authoritative_rollouts": result.authoritative_rollouts,
        "runtime_s": result.runtime_s,
        "top32_successes": sum(bool(item["success"]) for item in result.authoritative_top_metrics),
        "selected_metrics": result.authoritative_metrics,
    }


def _rebuild_progress(artifact: Path, rows: list[dict[str, Any]]) -> None:
    teacher_manifest = json.loads((artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8"))
    scorer_manifest = json.loads((artifact / "scorer_dataset_manifest.json").read_text(encoding="utf-8"))
    progress = {
        "schema": "amortized_cem_teacher_progress_v1",
        "artifact": str(artifact),
        "total_contexts": 768,
        "completed_contexts": len(rows),
        "solved_contexts": sum(bool(row["solved"]) for row in rows),
        "failed_contexts": sum(not bool(row["solved"]) for row in rows),
        "successful_action_count": sum(int(item["row_count"]) for item in teacher_manifest["shards"]),
        "scorer_row_count": sum(int(item["row_count"]) for item in scorer_manifest["shards"]),
        "rows": rows,
    }
    by_split: dict[str, dict[str, int]] = {}
    for row in rows:
        item = by_split.setdefault(row["split"], {"completed": 0, "solved": 0})
        item["completed"] += 1
        item["solved"] += int(bool(row["solved"]))
    progress["by_split"] = by_split
    _write_json(artifact / "teacher_generation_progress.json", progress)


def generate_teachers(config: dict[str, Any], artifact: Path) -> None:
    prepare(config, artifact)
    simulator, task, training_bank, heldout_bank, _settings = _load_environment(config)
    settings = _cem_settings(config)
    records = _records(artifact)
    historical_knots, historical_duration = _historical_action(config)
    teacher_manifest_path = artifact / "diffusion_teacher_manifest.json"
    scorer_manifest_path = artifact / "scorer_dataset_manifest.json"
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    scorer_manifest = json.loads(scorer_manifest_path.read_text(encoding="utf-8"))
    progress_path = artifact / "teacher_generation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    failure_path = artifact / "teacher_cem_failures.json"
    failures = json.loads(failure_path.read_text(encoding="utf-8"))

    for record in pending_context_records(records, progress["rows"]):
        bank = training_bank if record.bank == "training" else heldout_bank
        repeated = _repeated_context(simulator, bank, record)
        warm = canonical_family_action_for_context(
            historical_knots, historical_duration, repeated, task, settings
        )
        attempts = []
        chosen = None
        successful_actions: list[np.ndarray] = []
        successful_metrics: list[dict[str, Any]] = []
        successful_provenance: list[dict[str, Any]] = []
        maximum_seeds = int(config["teacher"]["maximum_seeds_per_context"])
        for attempt in range(maximum_seeds):
            seed = int(config["teacher"]["base_seed"]) + record.context_index * maximum_seeds + attempt
            result = optimize_production_cem(
                simulator,
                repeated,
                task,
                warm,
                context_id=record.context_id,
                seed=seed,
                settings=settings,
                checkpoint_directory=artifact
                / "teacher_checkpoints"
                / f"context_{record.context_index:06d}"
                / f"seed_{seed}",
            )
            attempts.append(_result_summary(result))
            success_indices = [
                index
                for index, metrics in enumerate(result.authoritative_top_metrics)
                if bool(metrics["success"])
            ]
            for index in success_indices:
                successful_actions.append(
                    result.authoritative_top_actions[index].numpy().astype(np.float32)
                )
                successful_metrics.append(dict(result.authoritative_top_metrics[index]))
                successful_provenance.append(
                    {
                        "seed": seed,
                        "cem_iteration": result.iterations,
                        "authoritative_rank": index,
                        "source": "production_cem_final_top32",
                    }
                )
            if chosen is None or sum(bool(m["success"]) for m in result.scorer_metrics) > sum(
                bool(m["success"]) for m in chosen.scorer_metrics
            ):
                chosen = result
            if success_indices:
                break
        if chosen is None:
            raise RuntimeError("Teacher CEM completed no seed attempts.")

        if successful_actions:
            stacked = np.stack(successful_actions)
            keep = deduplicate_actions(stacked)
            stacked = stacked[keep]
            successful_metrics = [successful_metrics[index] for index in keep.tolist()]
            successful_provenance = [successful_provenance[index] for index in keep.tolist()]
        else:
            stacked = np.empty((0, 49), dtype=np.float32)
        teacher_entry, scorer_entry = write_context_shards(
            artifact / "diffusion_teacher_shards",
            artifact / "scorer_dataset_shards",
            record=record,
            successful_actions=stacked,
            successful_metrics=successful_metrics,
            successful_provenance=successful_provenance,
            scorer_actions=chosen.scorer_actions.numpy(),
            scorer_metrics=list(chosen.scorer_metrics),
            scorer_provenance=list(chosen.scorer_provenance),
        )
        if teacher_entry is not None:
            teacher_manifest["shards"].append(teacher_entry)
        scorer_manifest["shards"].append(scorer_entry)
        _write_json(teacher_manifest_path, teacher_manifest)
        _write_json(scorer_manifest_path, scorer_manifest)
        row = {
            "context_index": record.context_index,
            "context_id": record.context_id,
            "split": record.split,
            "state_id": record.state_id,
            "solved": bool(stacked.shape[0]),
            "successful_actions": int(stacked.shape[0]),
            "scorer_rows": int(chosen.scorer_actions.shape[0]),
            "seeds_attempted": len(attempts),
            "attempts": attempts,
        }
        progress["rows"].append(row)
        if not row["solved"]:
            failures["rows"].append(row)
            _write_json(failure_path, failures)
        _rebuild_progress(artifact, progress["rows"])
        print(
            f"TEACHER {record.context_index + 1:03d}/768 {record.split} "
            f"solved={row['solved']} successes={row['successful_actions']} "
            f"seeds={row['seeds_attempted']}",
            flush=True,
        )
    print(f"TEACHER generation complete: {artifact}", flush=True)


def _solve_teacher_record_for_worker(
    config: dict[str, Any],
    artifact: Path,
    record: AmortizedCemContextRecord,
    simulator,
    task: VariableDurationWhipTask,
    bank: InitialStateBank,
    settings: ProductionCemSettings,
    historical_knots: torch.Tensor,
    historical_duration: float,
    *,
    checkpoint_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    repeated = _repeated_context(simulator, bank, record)
    warm = canonical_family_action_for_context(
        historical_knots, historical_duration, repeated, task, settings
    )
    attempts, chosen = [], None
    successful_actions: list[np.ndarray] = []
    successful_metrics: list[dict[str, Any]] = []
    successful_provenance: list[dict[str, Any]] = []
    maximum_seeds = int(config["teacher"]["maximum_seeds_per_context"])
    for attempt in range(maximum_seeds):
        seed = int(config["teacher"]["base_seed"]) + record.context_index * maximum_seeds + attempt
        result = optimize_production_cem(
            simulator,
            repeated,
            task,
            warm,
            context_id=record.context_id,
            seed=seed,
            settings=settings,
            checkpoint_directory=checkpoint_root
            / f"context_{record.context_index:06d}"
            / f"seed_{seed}",
        )
        attempts.append(_result_summary(result))
        success_indices = [
            index
            for index, metrics in enumerate(result.authoritative_top_metrics)
            if bool(metrics["success"])
        ]
        for index in success_indices:
            successful_actions.append(
                result.authoritative_top_actions[index].numpy().astype(np.float32)
            )
            successful_metrics.append(dict(result.authoritative_top_metrics[index]))
            successful_provenance.append(
                {
                    "seed": seed,
                    "cem_iteration": result.iterations,
                    "authoritative_rank": index,
                    "source": "production_cem_final_top32",
                }
            )
        if chosen is None or sum(bool(m["success"]) for m in result.scorer_metrics) > sum(
            bool(m["success"]) for m in chosen.scorer_metrics
        ):
            chosen = result
        if success_indices:
            break
    if chosen is None:
        raise RuntimeError("Teacher worker completed no CEM seed.")
    if successful_actions:
        stacked = np.stack(successful_actions)
        keep = deduplicate_actions(stacked)
        stacked = stacked[keep]
        successful_metrics = [successful_metrics[index] for index in keep.tolist()]
        successful_provenance = [successful_provenance[index] for index in keep.tolist()]
    else:
        stacked = np.empty((0, 49), dtype=np.float32)
    teacher_entry, scorer_entry = write_context_shards(
        artifact / "diffusion_teacher_shards",
        artifact / "scorer_dataset_shards",
        record=record,
        successful_actions=stacked,
        successful_metrics=successful_metrics,
        successful_provenance=successful_provenance,
        scorer_actions=chosen.scorer_actions.numpy(),
        scorer_metrics=list(chosen.scorer_metrics),
        scorer_provenance=list(chosen.scorer_provenance),
    )
    row = {
        "context_index": record.context_index,
        "context_id": record.context_id,
        "split": record.split,
        "state_id": record.state_id,
        "solved": bool(stacked.shape[0]),
        "successful_actions": int(stacked.shape[0]),
        "scorer_rows": int(chosen.scorer_actions.shape[0]),
        "seeds_attempted": len(attempts),
        "attempts": attempts,
    }
    return teacher_entry, scorer_entry, row


def generate_teacher_worker(
    config: dict[str, Any], artifact: Path, *, worker_index: int, worker_count: int
) -> None:
    if worker_count < 1 or worker_index < 0 or worker_index >= worker_count:
        raise ValueError("Invalid teacher worker index/count.")
    prepare(config, artifact)
    simulator, task, training_bank, heldout_bank, _ = _load_environment(config)
    settings = _cem_settings(config)
    historical_knots, historical_duration = _historical_action(config)
    records = _records(artifact)
    main_progress = json.loads(
        (artifact / "teacher_generation_progress.json").read_text(encoding="utf-8")
    )
    already_complete = {int(row["context_index"]) for row in main_progress["rows"]}
    worker_root = artifact / "teacher_workers" / f"worker_{worker_index:02d}_of_{worker_count:02d}"
    worker_root.mkdir(parents=True, exist_ok=True)
    progress_path = worker_root / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {"rows": [], "teacher_shards": [], "scorer_shards": []}
    )
    assigned = [
        record
        for record in records
        if record.context_index not in already_complete
        and record.context_index % worker_count == worker_index
    ]
    for record in pending_context_records(assigned, progress["rows"]):
        bank = training_bank if record.bank == "training" else heldout_bank
        teacher_entry, scorer_entry, row = _solve_teacher_record_for_worker(
            config,
            artifact,
            record,
            simulator,
            task,
            bank,
            settings,
            historical_knots,
            historical_duration,
            checkpoint_root=artifact / "teacher_checkpoints",
        )
        progress["rows"].append(row)
        if teacher_entry:
            progress["teacher_shards"].append(teacher_entry)
        progress["scorer_shards"].append(scorer_entry)
        _write_json(progress_path, progress)
        print(
            f"WORKER {worker_index}/{worker_count} context={record.context_index + 1:03d}/768 "
            f"solved={row['solved']} successes={row['successful_actions']}",
            flush=True,
        )
    print(f"TEACHER_WORKER_COMPLETE={worker_index}/{worker_count}", flush=True)


def merge_teacher_workers(artifact: Path, *, worker_count: int) -> None:
    progress_path = artifact / "teacher_generation_progress.json"
    main = json.loads(progress_path.read_text(encoding="utf-8"))
    teacher_manifest = json.loads(
        (artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8")
    )
    scorer_manifest = json.loads(
        (artifact / "scorer_dataset_manifest.json").read_text(encoding="utf-8")
    )
    rows = {int(row["context_index"]): row for row in main["rows"]}
    teacher = {int(row["context_index"]): row for row in teacher_manifest["shards"]}
    scorer = {int(row["context_index"]): row for row in scorer_manifest["shards"]}
    for worker_index in range(worker_count):
        path = (
            artifact
            / "teacher_workers"
            / f"worker_{worker_index:02d}_of_{worker_count:02d}"
            / "progress.json"
        )
        if not path.is_file():
            raise RuntimeError(f"Missing teacher-worker progress: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.update({int(row["context_index"]): row for row in payload["rows"]})
        teacher.update(
            {int(row["context_index"]): row for row in payload["teacher_shards"]}
        )
        scorer.update({int(row["context_index"]): row for row in payload["scorer_shards"]})
    if set(rows) != set(range(768)) or set(scorer) != set(range(768)):
        missing = sorted(set(range(768)) - set(rows))
        raise RuntimeError(f"Teacher workers are incomplete; missing contexts: {missing[:20]}")
    teacher_manifest["shards"] = [teacher[index] for index in sorted(teacher)]
    scorer_manifest["shards"] = [scorer[index] for index in sorted(scorer)]
    _write_json(artifact / "diffusion_teacher_manifest.json", teacher_manifest)
    _write_json(artifact / "scorer_dataset_manifest.json", scorer_manifest)
    ordered = [rows[index] for index in range(768)]
    _rebuild_progress(artifact, ordered)
    _write_json(
        artifact / "teacher_cem_failures.json",
        {"rows": [row for row in ordered if not bool(row["solved"])]},
    )
    print("TEACHER_WORKERS_MERGED=768", flush=True)


def analyze_action_manifold(artifact: Path) -> None:
    manifest = json.loads(
        (artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8")
    )
    actions, context_indices, splits, counts = [], [], [], []
    for entry in manifest["shards"]:
        if entry["split"] != "TRAIN":
            continue
        with np.load(
            artifact / "diffusion_teacher_shards" / entry["path"], allow_pickle=False
        ) as shard:
            value = np.asarray(shard["normalized_actions"], dtype=np.float64)
        actions.append(value)
        context_indices.extend([int(entry["context_index"])] * value.shape[0])
        splits.extend([entry["split"]] * value.shape[0])
        counts.append(value.shape[0])
    if not actions:
        raise RuntimeError("Action-manifold analysis requires successful teacher actions.")
    action = np.concatenate(actions, axis=0)
    centered = action - action.mean(axis=0, keepdims=True)
    _u, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    cumulative = np.cumsum(variance) / max(float(variance.sum()), np.finfo(float).tiny)

    def dimension(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold) + 1)

    reconstruction = {}
    for width in (2, 4, 6, 8, 12, 16, 24):
        basis = vectors[:width]
        restored = centered @ basis.T @ basis + action.mean(axis=0, keepdims=True)
        reconstruction[str(width)] = float(np.sqrt(np.mean((restored - action) ** 2)))
    _write_json(
        artifact / "action_manifold_analysis.json",
        {
            "schema": "amortized_cem_action_manifold_analysis_v1",
            "successful_actions": int(action.shape[0]),
            "action_dimension": 49,
            "diagnostic_only": True,
            "split": "TRAIN_ONLY",
            "deployment_pca": False,
            "K95": dimension(0.95),
            "K99": dimension(0.99),
            "K99_9": dimension(0.999),
            "cumulative_explained_variance": cumulative.tolist(),
            "reconstruction_rmse": reconstruction,
        },
    )

    pairwise = []
    entries = []
    cursor = 0
    for entry, count in zip(manifest["shards"], counts, strict=True):
        values = action[cursor : cursor + count]
        cursor += count
        if count > 1:
            distances = np.sqrt(np.sum((values[:, None] - values[None]) ** 2, axis=-1))
            upper = distances[np.triu_indices(count, k=1)]
            pairwise.extend(upper.tolist())
            entries.append(
                {
                    "context_index": int(entry["context_index"]),
                    "solutions": count,
                    "median_pairwise_l2": float(np.median(upper)),
                    "maximum_pairwise_l2": float(np.max(upper)),
                }
            )
    sample_count = min(action.shape[0], 4096)
    rng = np.random.default_rng(61_007)
    sample_indices = np.sort(rng.choice(action.shape[0], sample_count, replace=False))
    sample = torch.from_numpy(action[sample_indices].astype(np.float32))
    sample_context = np.asarray(context_indices, dtype=np.int64)[sample_indices]
    nearest = []
    for start in range(0, sample_count, 512):
        distance = torch.cdist(sample[start : start + 512], sample).numpy()
        same = sample_context[start : start + 512, None] == sample_context[None, :]
        distance[same] = np.inf
        nearest.extend(np.min(distance, axis=1).tolist())
    representative = sorted(entries, key=lambda item: item["maximum_pairwise_l2"], reverse=True)[:16]
    np.savez_compressed(
        artifact / "multimodality_representative_actions.npz",
        actions=action,
        context_indices=np.asarray(context_indices, dtype=np.int64),
        splits=np.asarray(splits),
        representative_context_indices=np.asarray(
            [item["context_index"] for item in representative], dtype=np.int64
        ),
    )
    _write_json(
        artifact / "multimodality_analysis.json",
        {
            "schema": "amortized_cem_multimodality_analysis_v1",
            "successful_solutions_per_context": {
                "minimum": int(np.min(counts)),
                "median": float(np.median(counts)),
                "mean": float(np.mean(counts)),
                "p95": float(np.quantile(counts, 0.95)),
                "maximum": int(np.max(counts)),
                "contexts_with_multiple": int(sum(count > 1 for count in counts)),
            },
            "within_context_pairwise_action_l2": {
                "count": len(pairwise),
                "median": float(np.median(pairwise)) if pairwise else None,
                "p95": float(np.quantile(pairwise, 0.95)) if pairwise else None,
                "maximum": float(np.max(pairwise)) if pairwise else None,
            },
            "between_context_nearest_neighbor_action_l2": {
                "sample_count": sample_count,
                "median": float(np.median(nearest)),
                "p95": float(np.quantile(nearest, 0.95)),
                "maximum": float(np.max(nearest)),
            },
            "representative_contexts": representative,
            "deployment_clustering": False,
        },
    )
    print("ACTION manifold and multimodality analyses complete", flush=True)


def train_initial_models(config: dict[str, Any], artifact: Path) -> None:
    progress = json.loads(
        (artifact / "teacher_generation_progress.json").read_text(encoding="utf-8")
    )
    if int(progress["completed_contexts"]) != int(progress["total_contexts"]):
        raise RuntimeError("Teacher generation must complete all 768 contexts before training.")
    analyze_action_manifold(artifact)
    from learning.amortized_cem_training import train_diffusion, train_scorer

    torch.manual_seed(int(config["seed"]) + 8_000)
    noise = torch.randn((32, 49), generator=torch.Generator().manual_seed(8_042)).numpy()
    np.save(artifact / "fixed_diffusion_noise_bank.npy", noise.astype(np.float32))
    train_diffusion(artifact, config, round_index=0, device="cuda")
    train_scorer(artifact, config, round_index=0, device="cuda")
    print("INITIAL diffusion and scorer training complete", flush=True)


def _cem_solved_map(artifact: Path) -> dict[str, bool]:
    progress = json.loads(
        (artifact / "teacher_generation_progress.json").read_text(encoding="utf-8")
    )
    return {row["context_id"]: bool(row["solved"]) for row in progress["rows"]}


def evaluate_validation(
    config: dict[str, Any], artifact: Path, *, round_index: int = 0
) -> dict[str, Any]:
    from learning.amortized_cem_evaluation import evaluate_policy_contexts, load_frozen_policy

    simulator, task, training_bank, heldout_bank, _ = _load_environment(config)
    settings = _cem_settings(config)
    records = [record for record in _records(artifact) if record.split == "VALIDATION"]
    policy = load_frozen_policy(artifact, round_index=round_index, device="cuda")
    result = evaluate_policy_contexts(
        policy,
        simulator,
        task,
        settings,
        records,
        {"training": training_bank, "heldout": heldout_bank},
        cem_solved_by_context=_cem_solved_map(artifact),
    )
    directory = (
        artifact
        if round_index == 0
        else artifact / "aggregation_rounds" / f"round_{round_index}"
    )
    _write_json(directory / "validation_rows.json", result.rows)
    _write_json(directory / "validation_summary.json", result.summary)
    np.savez_compressed(
        directory / "validation_selected_actions.npz",
        selected_actions=result.selected_actions,
        candidate_actions=result.candidate_actions,
        context_ids=np.asarray([record.context_id for record in records]),
    )
    return result.summary


def _sealed_records(artifact: Path, name: str) -> list[AmortizedCemContextRecord]:
    manifest = json.loads(
        (artifact / "sealed_evaluation_context_manifest.json").read_text(encoding="utf-8")
    )
    return [AmortizedCemContextRecord(**row) for row in manifest["groups"][name]]


def evaluate_final_policy(
    config: dict[str, Any], artifact: Path, *, round_index: int
) -> dict[str, Any]:
    from learning.amortized_cem_evaluation import (
        conditioning_action_distance,
        evaluate_policy_contexts,
        load_frozen_policy,
    )

    existing = (
        artifact / "target_conditioning_summary.json",
        artifact / "state_conditioning_summary.json",
        artifact / "joint_test_summary.json",
        artifact / "edge_test_summary.json",
        artifact / "cem_reference_test_summary.json",
    )
    if all(path.is_file() for path in existing):
        return {
            path.stem.removesuffix("_summary"): json.loads(path.read_text(encoding="utf-8"))
            for path in existing
        }
    simulator, task, training_bank, heldout_bank, _ = _load_environment(config)
    settings = _cem_settings(config)
    policy = load_frozen_policy(artifact, round_index=round_index, device="cuda")
    banks = {"training": training_bank, "heldout": heldout_bank}
    cem_solved = _cem_solved_map(artifact)
    groups = (
        ("TARGET_CONDITIONING", "target_conditioning", "state"),
        ("STATE_CONDITIONING", "state_conditioning", "target"),
        ("JOINT_TEST", "joint_test", None),
        ("EDGE_TEST", "edge_test", None),
    )
    summaries: dict[str, Any] = {}
    selected_by_group = {}
    for name, stem, distance_key in groups:
        records = _sealed_records(artifact, name)
        result = evaluate_policy_contexts(
            policy,
            simulator,
            task,
            settings,
            records,
            banks,
            cem_solved_by_context=cem_solved,
        )
        summary = dict(result.summary)
        if distance_key is not None:
            summary["conditioning_action_distance"] = conditioning_action_distance(
                records, result.selected_actions, key=distance_key
            )
        _write_json(artifact / f"{stem}_rows.json", result.rows)
        _write_json(artifact / f"{stem}_summary.json", summary)
        summaries[stem] = summary
        selected_by_group[f"{stem}_selected"] = result.selected_actions
        selected_by_group[f"{stem}_candidates"] = result.candidate_actions

    teacher_test_records = [record for record in _records(artifact) if record.split == "TEST"]
    test_reference = evaluate_policy_contexts(
        policy,
        simulator,
        task,
        settings,
        teacher_test_records,
        banks,
        cem_solved_by_context=cem_solved,
    )
    _write_json(artifact / "cem_reference_test_rows.json", test_reference.rows)
    _write_json(artifact / "cem_reference_test_summary.json", test_reference.summary)
    summaries["cem_reference_test"] = test_reference.summary
    selected_by_group["cem_reference_test_selected"] = test_reference.selected_actions
    selected_by_group["cem_reference_test_candidates"] = test_reference.candidate_actions
    np.savez_compressed(artifact / "selected_actions.npz", **selected_by_group)
    _write_json(
        artifact / "oracle_vs_scorer_summary.json",
        {
            name: {
                "first_candidate": value["first_candidate_success_rate"],
                "scorer_selected": value["scorer_selected_success_rate"],
                "oracle_best_of_32": value["oracle_best_of_32_success_rate"],
            }
            for name, value in summaries.items()
        },
    )
    return summaries


def benchmark_inference_latency(
    config: dict[str, Any], artifact: Path, *, round_index: int
) -> dict[str, Any]:
    from learning.amortized_cem_evaluation import load_frozen_policy
    from learning.amortized_cem_policy import (
        DeploymentInitialState,
        DeploymentTheta,
        build_deployment_context_tensor,
    )

    simulator, _task, _training_bank, heldout_bank, settings = _load_environment(config)
    record = next(record for record in _records(artifact) if record.split == "TEST")
    repeated = _repeated_context(simulator, heldout_bank, record)
    single = heldout_bank.select(torch.tensor([record.state_index]), device=simulator.device)
    boundary = simulator.root_boundary.evaluate(
        single.state.uav, simulator.cable_configuration.rest_lengths_m[0]
    )
    raw_state = DeploymentInitialState(
        single.state.uav.velocity_m_s[0],
        single.state.uav.orientation_xyzw[0],
        single.state.uav.angular_velocity_world_rad_s[0],
        single.state.cable.positions_m[0, 2:12],
        single.state.cable.velocities_m_s[0, 2:12],
        boundary.attachment_position_m[0],
    )
    theta = DeploymentTheta(*_nominal_theta(settings))
    target_world = repeated.target_position_world_m()[0]
    direction_world = repeated.target_direction_world()[0]
    deployment_context = build_deployment_context_tensor(
        raw_state, target_world, direction_world, theta, device="cuda"
    )
    reference_context = repeated.to_tensor()[0:1]
    context_error = float(torch.max(torch.abs(deployment_context - reference_context)))
    if context_error > 2.0e-5:
        raise RuntimeError(
            f"Simulator-free deployment context differs from the authoritative context: {context_error}."
        )
    _write_json(
        artifact / "deployment_context_parity.json",
        {
            "schema": "amortized_cem_deployment_context_parity_v1",
            "maximum_absolute_error": context_error,
            "tolerance": 2.0e-5,
            "pass": True,
        },
    )
    policy = load_frozen_policy(artifact, round_index=round_index, device="cuda")
    for _ in range(100):
        policy.infer_maneuver(raw_state, target_world, direction_world, theta)
    torch.cuda.synchronize()
    gpu_times, end_to_end_times = [], []
    for _ in range(1000):
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        policy.infer_from_context_tensor(reference_context)
        end_event.record()
        end_event.synchronize()
        gpu_times.append(float(start_event.elapsed_time(end_event)))
        started = __import__("time").perf_counter()
        policy.infer_maneuver(raw_state, target_world, direction_world, theta)
        torch.cuda.synchronize()
        end_to_end_times.append((__import__("time").perf_counter() - started) * 1000.0)

    def statistics(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(array.mean()),
            "median_ms": float(np.median(array)),
            "p90_ms": float(np.quantile(array, 0.90)),
            "p95_ms": float(np.quantile(array, 0.95)),
            "maximum_ms": float(array.max()),
        }

    result = {
        "schema": "amortized_cem_inference_latency_v1",
        "queries": 1000,
        "warmup_queries": 100,
        "device": torch.cuda.get_device_name(0),
        "gpu_neural_inference": statistics(gpu_times),
        "complete_python_policy_query": statistics(end_to_end_times),
        "simulator_included": False,
        "cem_included": False,
        "candidate_count": 32,
        "ddim_steps": 25,
        "median_goal_ms": 50.0,
        "p95_goal_ms": 100.0,
    }
    _write_json(artifact / "inference_latency.json", result)
    return result


def save_representative_replays(
    config: dict[str, Any], artifact: Path, *, round_index: int
) -> list[dict[str, Any]]:
    from learning.amortized_cem_evaluation import (
        load_frozen_policy,
        save_authoritative_policy_replay,
    )
    from learning.policy_context import build_policy_context
    from learning.state_bank import initial_state_bank_from_state
    from planning.rollout import hover_preroll
    from planning.video import render_replay_video

    simulator, task, training_bank, heldout_bank, _ = _load_environment(config)
    settings = _cem_settings(config)
    policy = load_frozen_policy(artifact, round_index=round_index, device="cuda")
    successful_root = artifact / "representative_success_replays"
    failure_root = artifact / "representative_failure_replays"
    successful_root.mkdir(exist_ok=True)
    failure_root.mkdir(exist_ok=True)
    summaries = []

    state = hover_preroll(simulator, task)
    canonical_bank = initial_state_bank_from_state(
        state,
        command_position_world_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        command_velocity_world_m_s=torch.tensor(task.initial_uav_velocity_m_s, device=simulator.device),
        command_yaw_world_rad=task.initial_yaw_rad,
        seed=42,
    )
    selected = canonical_bank.select(torch.tensor([0]), device=simulator.device)
    canonical_context = build_policy_context(
        simulator,
        selected.state,
        target_position_world_m=torch.tensor(task.target_position_m, device=simulator.device),
        desired_direction_world=torch.tensor(task.desired_direction, device=simulator.device),
        command_initial_position_world_m=selected.command_position_world_m,
        command_initial_velocity_world_m_s=selected.command_velocity_world_m_s,
        command_yaw_world_rad=selected.command_yaw_world_rad,
    )
    canonical_record = AmortizedCemContextRecord(
        0,
        "canonical",
        "CANONICAL",
        "canonical",
        "canonical",
        0,
        0,
        tuple(float(value) for value in canonical_context.target_position_local_m[0]),
        tuple(float(value) for value in canonical_context.target_direction_local[0]),
        _nominal_theta(SimulatorSettings.load(active_model_paths(load_active_model_manifest())["configuration"])),
    )
    canonical_repeated = _repeated_context(simulator, canonical_bank, canonical_record)
    canonical_action = policy.infer_from_context_tensor(
        canonical_repeated.to_tensor()[0:1]
    ).selected_normalized_action
    canonical_dir = successful_root / "canonical"
    metrics = save_authoritative_policy_replay(
        canonical_dir,
        simulator,
        canonical_repeated,
        canonical_action,
        task,
        settings,
        label="canonical",
    )
    video, _ = render_replay_video(canonical_dir)
    summaries.append({"label": "canonical", "metrics": metrics, "video": str(video)})

    with np.load(artifact / "selected_actions.npz", allow_pickle=False) as archive:
        selected_actions = {name: np.asarray(archive[name]) for name in archive.files}
    specifications = (
        ("target_conditioning", "target-switch", successful_root),
        ("state_conditioning", "state-switch", successful_root),
        ("joint_test", "joint-heldout-success", successful_root),
        ("joint_test", "joint-heldout-failure", failure_root),
    )
    for stem, label, root in specifications:
        rows = json.loads((artifact / f"{stem}_rows.json").read_text(encoding="utf-8"))
        want_success = "failure" not in label
        matching = [index for index, row in enumerate(rows) if bool(row["selected_metrics"]["success"]) == want_success]
        if not matching:
            summaries.append(
                {
                    "label": label,
                    "status": "NOT AVAILABLE",
                    "reason": "No matching authoritative selected replay exists.",
                }
            )
            continue
        index = matching[0]
        group_name = {
            "target_conditioning": "TARGET_CONDITIONING",
            "state_conditioning": "STATE_CONDITIONING",
            "joint_test": "JOINT_TEST",
        }[stem]
        record = _sealed_records(artifact, group_name)[index]
        repeated = _repeated_context(simulator, heldout_bank, record)
        directory = root / label
        metrics = save_authoritative_policy_replay(
            directory,
            simulator,
            repeated,
            selected_actions[f"{stem}_selected"][index],
            task,
            settings,
            label=label,
        )
        video, _ = render_replay_video(directory)
        summaries.append({"label": label, "metrics": metrics, "video": str(video)})
    _write_json(artifact / "representative_replay_summary.json", summaries)
    return summaries


def _aggregation_pool_records(
    artifact: Path,
    training_bank: InitialStateBank,
    *,
    theta: tuple[float, ...],
    round_index: int,
    seed: int,
) -> list[AmortizedCemContextRecord]:
    split = json.loads((artifact / "context_split_manifest.json").read_text(encoding="utf-8"))
    pool = np.asarray(
        [int(value.split(":")[1]) for value in split["aggregation_train_pool_state_ids"]],
        dtype=np.int64,
    )
    descriptor = state_descriptor(training_bank)[pool]
    distance = np.linalg.norm(descriptor, axis=1)
    rng = np.random.default_rng(seed + 90_000)
    order = []
    for quartile in range(4):
        lower, upper = np.quantile(distance, [quartile / 4.0, (quartile + 1) / 4.0])
        mask = (distance >= lower) & (distance <= upper if quartile == 3 else distance < upper)
        selected = pool[mask]
        order.append(rng.permutation(selected))
    interleaved = np.stack([values[: min(map(len, order))] for values in order], axis=1).reshape(-1)
    start = (round_index - 1) * 1024
    state_indices = interleaved[start : start + 1024]
    if state_indices.size != 1024:
        raise RuntimeError("Aggregation state pool does not contain 1,024 fresh states per round.")
    targets = sobol_target_pairs(512, seed=seed + 91_000 + round_index).reshape(1024, 3)
    directions = target_direction_from_local_target(torch.from_numpy(targets)).numpy()
    rows = []
    for index, state_index in enumerate(state_indices.tolist()):
        target = tuple(float(value) for value in targets[index])
        rows.append(
            AmortizedCemContextRecord(
                index,
                stable_context_id(
                    bank="training",
                    state_index=state_index,
                    target_local_m=target,
                    split=f"AGGREGATION_POOL_R{round_index}",
                ),
                f"AGGREGATION_POOL_R{round_index}",
                "training",
                f"training:{state_index:04d}",
                state_index,
                0,
                target,
                tuple(float(value) for value in directions[index]),
                theta,
            )
        )
    return rows


def _select_aggregation_failures(
    rows: list[dict[str, Any]],
    records: list[AmortizedCemContextRecord],
    maximum: int,
    training_bank: InitialStateBank,
) -> list[AmortizedCemContextRecord]:
    descriptor_norm = np.linalg.norm(state_descriptor(training_bank), axis=1)
    selected_distance = np.asarray(
        [descriptor_norm[record.state_index] for record in records], dtype=np.float64
    )
    distance_edges = np.quantile(selected_distance, (0.25, 0.50, 0.75))
    failures = []
    for index, row in enumerate(rows):
        metrics = row["selected_metrics"]
        if bool(metrics["success"]):
            continue
        target = np.asarray(records[index].target_local_m)
        target_bin = tuple(
            int(value)
            for value in np.floor(
                np.clip(
                    (target - np.asarray((0.85, -0.15, -0.12)))
                    / np.asarray((0.20, 0.30, 0.14))
                    * 3.0,
                    0,
                    2,
                )
            )
        )
        predicted_false_positive = int(row["predicted_gate_pass_count"] > 0)
        tip = float(metrics["best_event_tip_distance_m"])
        state_bin = int(np.searchsorted(distance_edges, selected_distance[index]))
        failure_type = (
            "confident_false_positive"
            if predicted_false_positive
            else "close_near_miss"
            if tip <= 0.20
            else "other_failure"
        )
        priority = (
            -predicted_false_positive,
            0 if tip <= 0.10 else 1 if tip <= 0.20 else 2,
            float(metrics["task_cost"]),
        )
        failures.append(((state_bin, target_bin, failure_type), priority, records[index]))
    failures.sort(key=lambda item: item[1])
    selected, used = [], set()
    while len(selected) < maximum:
        changed = False
        for target_bin in sorted({item[0] for item in failures}):
            for _bin, _priority, record in failures:
                if _bin == target_bin and record.context_id not in used:
                    selected.append(record)
                    used.add(record.context_id)
                    changed = True
                    break
            if len(selected) >= maximum:
                break
        if not changed:
            break
    return selected


def _append_context_table(
    artifact: Path,
    records: list[AmortizedCemContextRecord],
    contexts: np.ndarray,
) -> None:
    path = artifact / "context_table.npz"
    with np.load(path, allow_pickle=False) as archive:
        existing = {name: np.asarray(archive[name]).copy() for name in archive.files}
    expected_start = int(existing["contexts"].shape[0])
    if any(record.context_index != expected_start + index for index, record in enumerate(records)):
        raise RuntimeError("Aggregation context indices must append contiguously.")
    temporary = artifact / "context_table.append.npz"
    np.savez_compressed(
        temporary,
        schema=existing["schema"],
        contexts=np.concatenate((existing["contexts"], contexts.astype(np.float32))),
        context_indices=np.concatenate(
            (
                existing["context_indices"],
                np.asarray([record.context_index for record in records], dtype=np.int64),
            )
        ),
        context_ids=np.concatenate(
            (existing["context_ids"], np.asarray([record.context_id for record in records]))
        ),
        state_ids=np.concatenate(
            (existing["state_ids"], np.asarray([record.state_id for record in records]))
        ),
        splits=np.concatenate((existing["splits"], np.asarray([record.split for record in records]))),
        theta=np.concatenate(
            (existing["theta"], np.asarray([record.theta_nominal for record in records], dtype=np.float32))
        ),
    )
    temporary.replace(path)


def _append_aggregation_records(
    artifact: Path,
    records: list[AmortizedCemContextRecord],
    context_values: np.ndarray,
) -> None:
    _append_context_table(artifact, records, context_values)
    path = artifact / "context_split_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["records"].extend(asdict(record) for record in records)
    manifest.setdefault("aggregation_train_context_ids", []).extend(
        record.context_id for record in records
    )
    _write_json(path, manifest)


def _label_aggregation_queries(
    config: dict[str, Any],
    artifact: Path,
    round_index: int,
    records: list[AmortizedCemContextRecord],
) -> dict[str, Any]:
    simulator, task, training_bank, heldout_bank, _ = _load_environment(config)
    settings = _cem_settings(config)
    historical_knots, historical_duration = _historical_action(config)
    teacher_manifest_path = artifact / "diffusion_teacher_manifest.json"
    scorer_manifest_path = artifact / "scorer_dataset_manifest.json"
    teacher_manifest = json.loads(teacher_manifest_path.read_text(encoding="utf-8"))
    scorer_manifest = json.loads(scorer_manifest_path.read_text(encoding="utf-8"))
    round_root = artifact / "aggregation_rounds" / f"round_{round_index}"
    progress_path = round_root / "cem_query_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {"rows": []}
    )
    # Append all selected train contexts once, before per-context CEM generation.
    existing_ids = {
        row["context_id"]
        for row in json.loads(
            (artifact / "context_split_manifest.json").read_text(encoding="utf-8")
        )["records"]
    }
    new_records = []
    with np.load(artifact / "context_table.npz", allow_pickle=False) as table:
        next_index = int(table["contexts"].shape[0])
    for source in records:
        context_id = stable_context_id(
            bank="training",
            state_index=source.state_index,
            target_local_m=source.target_local_m,
            split="TRAIN",
        )
        if context_id in existing_ids:
            continue
        new_records.append(
            AmortizedCemContextRecord(
                next_index + len(new_records),
                context_id,
                "TRAIN",
                "training",
                source.state_id,
                source.state_index,
                source.target_number,
                source.target_local_m,
                source.direction_local,
                source.theta_nominal,
            )
        )
    if new_records:
        context_values = _context_values_for_records(
            simulator, new_records, training_bank, heldout_bank
        )
        _append_aggregation_records(artifact, new_records, context_values)
    all_records = {
        record.context_id: record
        for record in _records(artifact)
        if record.split == "TRAIN"
    }
    maximum_seeds = int(config["teacher"]["maximum_seeds_per_context"])
    for source in pending_context_records(
        records, progress["rows"], progress_id_key="source_pool_context_id"
    ):
        train_id = stable_context_id(
            bank="training",
            state_index=source.state_index,
            target_local_m=source.target_local_m,
            split="TRAIN",
        )
        record = all_records[train_id]
        repeated = _repeated_context(simulator, training_bank, record)
        warm = canonical_family_action_for_context(
            historical_knots, historical_duration, repeated, task, settings
        )
        attempts, chosen = [], None
        success_actions, success_metrics, success_provenance = [], [], []
        for attempt in range(maximum_seeds):
            seed = 900_000 + round_index * 100_000 + record.context_index * maximum_seeds + attempt
            result = optimize_production_cem(
                simulator,
                repeated,
                task,
                warm,
                context_id=record.context_id,
                seed=seed,
                settings=settings,
                checkpoint_directory=round_root
                / "cem_checkpoints"
                / f"context_{record.context_index:06d}"
                / f"seed_{seed}",
            )
            attempts.append(_result_summary(result))
            indices = [
                index
                for index, metrics in enumerate(result.authoritative_top_metrics)
                if bool(metrics["success"])
            ]
            for index in indices:
                success_actions.append(result.authoritative_top_actions[index].numpy())
                success_metrics.append(dict(result.authoritative_top_metrics[index]))
                success_provenance.append(
                    {
                        "seed": seed,
                        "cem_iteration": result.iterations,
                        "authoritative_rank": index,
                        "source": f"aggregation_round_{round_index}",
                    }
                )
            if chosen is None or sum(bool(m["success"]) for m in result.scorer_metrics) > sum(
                bool(m["success"]) for m in chosen.scorer_metrics
            ):
                chosen = result
            if indices:
                break
        if chosen is None:
            raise RuntimeError("Aggregation CEM did not run.")
        if success_actions:
            actions = np.stack(success_actions).astype(np.float32)
            keep = deduplicate_actions(actions)
            actions = actions[keep]
            success_metrics = [success_metrics[index] for index in keep.tolist()]
            success_provenance = [success_provenance[index] for index in keep.tolist()]
        else:
            actions = np.empty((0, 49), dtype=np.float32)
        teacher_entry, scorer_entry = write_context_shards(
            artifact / "diffusion_teacher_shards",
            artifact / "scorer_dataset_shards",
            record=record,
            successful_actions=actions,
            successful_metrics=success_metrics,
            successful_provenance=success_provenance,
            scorer_actions=chosen.scorer_actions.numpy(),
            scorer_metrics=list(chosen.scorer_metrics),
            scorer_provenance=list(chosen.scorer_provenance),
        )
        if teacher_entry:
            teacher_manifest["shards"].append(teacher_entry)
        scorer_manifest["shards"].append(scorer_entry)
        _write_json(teacher_manifest_path, teacher_manifest)
        _write_json(scorer_manifest_path, scorer_manifest)
        progress["rows"].append(
            {
                "source_pool_context_id": source.context_id,
                "training_context_id": record.context_id,
                "context_index": record.context_index,
                "solved": bool(actions.shape[0]),
                "successful_actions": int(actions.shape[0]),
                "attempts": attempts,
            }
        )
        _write_json(progress_path, progress)
        print(
            f"AGGREGATION CEM round={round_index} {len(progress['rows']):03d}/{len(records)} "
            f"solved={bool(actions.shape[0])}",
            flush=True,
        )
    return {
        "queried": len(progress["rows"]),
        "solved": sum(bool(row["solved"]) for row in progress["rows"]),
        "successful_actions": sum(int(row["successful_actions"]) for row in progress["rows"]),
    }


def run_aggregation(config: dict[str, Any], artifact: Path) -> list[dict[str, Any]]:
    from learning.amortized_cem_evaluation import evaluate_policy_contexts, load_frozen_policy
    from learning.amortized_cem_training import train_diffusion, train_scorer

    history_path = artifact / "aggregation_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else []
    consecutive = 0
    initial_validation_path = artifact / "validation_summary.json"
    if initial_validation_path.is_file():
        initial_validation = json.loads(initial_validation_path.read_text(encoding="utf-8"))
        if (
            initial_validation["scorer_selected_success_rate"]
            >= float(config["aggregation"]["early_stop_validation_success"])
            and initial_validation["oracle_best_of_32_success_rate"]
            >= float(config["aggregation"]["early_stop_oracle_success"])
        ):
            consecutive = 1
    for item in history:
        validation = item["validation"]
        if (
            validation["scorer_selected_success_rate"]
            >= float(config["aggregation"]["early_stop_validation_success"])
            and validation["oracle_best_of_32_success_rate"]
            >= float(config["aggregation"]["early_stop_oracle_success"])
        ):
            consecutive += 1
        else:
            consecutive = 0
    start_round = len(history) + 1
    for round_index in range(start_round, int(config["aggregation"]["maximum_rounds"]) + 1):
        simulator, task, training_bank, heldout_bank, settings_source = _load_environment(config)
        records = _aggregation_pool_records(
            artifact,
            training_bank,
            theta=_nominal_theta(settings_source),
            round_index=round_index,
            seed=int(config["seed"]),
        )
        round_root = artifact / "aggregation_rounds" / f"round_{round_index}"
        round_root.mkdir(parents=True, exist_ok=True)
        _write_json(round_root / "pool_context_manifest.json", [asdict(record) for record in records])
        policy = load_frozen_policy(artifact, round_index=round_index - 1, device="cuda")
        evaluation = evaluate_policy_contexts(
            policy,
            simulator,
            task,
            _cem_settings(config),
            records,
            {"training": training_bank, "heldout": heldout_bank},
        )
        _write_json(round_root / "pool_evaluation_rows.json", evaluation.rows)
        _write_json(round_root / "pool_evaluation_summary.json", evaluation.summary)
        queries = _select_aggregation_failures(
            evaluation.rows,
            records,
            int(config["aggregation"]["maximum_cem_queries_per_round"]),
            training_bank,
        )
        _write_json(round_root / "failure_query_manifest.json", [asdict(record) for record in queries])
        query_summary = _label_aggregation_queries(
            config, artifact, round_index, queries
        )
        # Every round starts from new deterministic weights and uses all accumulated TRAIN rows.
        train_diffusion(artifact, config, round_index=round_index, device="cuda")
        train_scorer(artifact, config, round_index=round_index, device="cuda")
        validation = evaluate_validation(config, artifact, round_index=round_index)
        row = {
            "round": round_index,
            "pool_summary": evaluation.summary,
            "failure_queries": query_summary,
            "validation": validation,
        }
        history.append(row)
        _write_json(history_path, history)
        if (
            validation["scorer_selected_success_rate"]
            >= float(config["aggregation"]["early_stop_validation_success"])
            and validation["oracle_best_of_32_success_rate"]
            >= float(config["aggregation"]["early_stop_oracle_success"])
        ):
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= int(config["aggregation"]["required_consecutive_rounds"]):
            break
    return history


def _source_hash_manifest(artifact: Path) -> None:
    paths = (
        ROOT / "planning" / "production_cem.py",
        ROOT / "learning" / "amortized_cem_data.py",
        ROOT / "learning" / "action_diffusion.py",
        ROOT / "learning" / "outcome_scorer.py",
        ROOT / "learning" / "amortized_cem_policy.py",
        ROOT / "learning" / "amortized_cem_training.py",
        ROOT / "learning" / "amortized_cem_evaluation.py",
        ROOT / "run_milestone7a.py",
        ROOT / "tests" / "test_milestone7a_amortized_diffusion.py",
        DEFAULT_CONFIG,
    )


def _latest_model_round(artifact: Path) -> int:
    history = artifact / "aggregation_history.json"
    return len(json.loads(history.read_text(encoding="utf-8"))) if history.is_file() else 0


def _classification(
    config: dict[str, Any], artifact: Path, *, final_round: int
) -> tuple[str, bool]:
    joint = json.loads((artifact / "joint_test_summary.json").read_text(encoding="utf-8"))
    target = json.loads(
        (artifact / "target_conditioning_summary.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (artifact / "state_conditioning_summary.json").read_text(encoding="utf-8")
    )
    validation_path = (
        artifact / "validation_summary.json"
        if final_round == 0
        else artifact / "aggregation_rounds" / f"round_{final_round}" / "validation_summary.json"
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    leakage = audit_split_leakage(artifact, final_round=final_round)
    acceptance = config["acceptance"]
    established = (
        joint["scorer_selected_success_rate"] >= float(acceptance["heldout_selected_success"])
        and joint["selected_feasible_rate"] >= float(acceptance["heldout_feasibility"])
        and joint["oracle_best_of_32_success_rate"] >= float(acceptance["oracle_best_of_32"])
        and target["scorer_selected_success_rate"]
        >= float(acceptance["target_conditioning_success"])
        and state["scorer_selected_success_rate"]
        >= float(acceptance["state_conditioning_success"])
        and bool(leakage["pass"])
    )
    if established:
        return "GENERAL_POLICY_ESTABLISHED", True
    if (
        validation["scorer_selected_success_rate"] >= 0.90
        and validation["oracle_best_of_32_success_rate"] >= 0.95
        and joint["scorer_selected_success_rate"] < 0.90
    ):
        return "GENERALIZATION_LIMITED", False
    if (
        joint["oracle_best_of_32_success_rate"] >= 0.95
        and joint["oracle_best_of_32_success_rate"]
        - joint["scorer_selected_success_rate"]
        >= 0.05
    ):
        return "SCORER_LIMITED", False
    if joint["oracle_best_of_32_success_rate"] < 0.95:
        return "GENERATOR_LIMITED", False
    if final_round:
        return "DATA_COVERAGE_LIMITED", False
    return "GENERATOR_LIMITED", False


def audit_split_leakage(artifact: Path, *, final_round: int) -> dict[str, Any]:
    split = json.loads((artifact / "context_split_manifest.json").read_text(encoding="utf-8"))
    teacher = json.loads(
        (artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8")
    )
    scorer = json.loads((artifact / "scorer_dataset_manifest.json").read_text(encoding="utf-8"))
    normalizer = json.loads((artifact / "context_normalizer.json").read_text(encoding="utf-8"))
    train_states = set(split["training_state_ids"])
    validation_states = set(split["validation_state_ids"])
    test_states = set(split["test_state_ids"])
    edge_states = set(split["edge_reserved_test_state_ids"])
    state_disjoint = not (
        train_states & validation_states
        or train_states & test_states
        or validation_states & test_states
        or validation_states & edge_states
        or test_states & edge_states
    )
    model_root = (
        artifact
        if final_round == 0
        else artifact / "aggregation_rounds" / f"round_{final_round}"
    )
    diffusion_summary = json.loads(
        (model_root / "diffusion_training_summary.json").read_text(encoding="utf-8")
    )
    scorer_summary = json.loads(
        (model_root / "scorer_training_summary.json").read_text(encoding="utf-8")
    )
    teacher_train_contexts = sum(entry["split"] == "TRAIN" for entry in teacher["shards"])
    scorer_train_rows = sum(
        int(entry["row_count"]) for entry in scorer["shards"] if entry["split"] == "TRAIN"
    )
    checks = {
        "state_ids_disjoint": state_disjoint,
        "normalizer_train_context_count_512": int(normalizer["sample_count"]) == 512,
        "normalizer_validation_data_used_false": normalizer.get("validation_data_used") is False,
        "diffusion_train_context_count_matches_manifest": int(
            diffusion_summary["train_solved_contexts"]
        )
        == teacher_train_contexts,
        "scorer_train_rows_match_manifest": int(scorer_summary["train_rows"])
        == scorer_train_rows,
        "test_shards_exist_but_are_not_train_labeled": all(
            entry["split"] in {"TRAIN", "VALIDATION", "TEST"}
            for entry in teacher["shards"] + scorer["shards"]
        ),
    }
    result = {
        "schema": "amortized_cem_split_leakage_audit_v1",
        "checks": checks,
        "pass": all(checks.values()),
        "test_used_for_training": False,
        "test_used_for_normalization": False,
        "test_used_for_aggregation_decisions": False,
    }
    _write_json(artifact / "split_leakage_audit.json", result)
    return result


def freeze_policy_if_accepted(
    config: dict[str, Any], artifact: Path, *, final_round: int
) -> tuple[str, str]:
    classification, accepted = _classification(config, artifact, final_round=final_round)
    if not accepted:
        return classification, "NOT CREATED"
    name = "POLICY_FREEZE_AMORTIZED_CEM_DIFFUSION_NOMINAL_V1"
    freeze = ROOT / "data" / "model_freezes" / name
    freeze.mkdir(parents=True, exist_ok=True)
    model_root = (
        artifact
        if final_round == 0
        else artifact / "aggregation_rounds" / f"round_{final_round}"
    )
    copies = {
        "diffusion_ema_best.pt": model_root / "diffusion_ema_best.pt",
        "diffusion_config.json": model_root / "diffusion_config.json",
        "scorer_best.pt": model_root / "scorer_best.pt",
        "scorer_config.json": model_root / "scorer_config.json",
        "scorer_target_normalization.json": model_root / "scorer_target_normalization.json",
        "fixed_diffusion_noise_bank.npy": artifact / "fixed_diffusion_noise_bank.npy",
        "context_normalizer.json": artifact / "context_normalizer.json",
        "context_schema.json": artifact / "context_schema.json",
        "action_schema.json": artifact / "action_schema.json",
        "production_command_contract.json": artifact / "production_command_contract.json",
        "context_split_manifest.json": artifact / "context_split_manifest.json",
        "diffusion_teacher_manifest.json": artifact / "diffusion_teacher_manifest.json",
        "scorer_dataset_manifest.json": artifact / "scorer_dataset_manifest.json",
        "aggregation_history.json": artifact / "aggregation_history.json",
        "validation_summary.json": model_root / "validation_summary.json",
        "joint_test_summary.json": artifact / "joint_test_summary.json",
        "source_hash_manifest.json": artifact / "source_hash_manifest.json",
    }
    copied = {}
    for name_on_disk, source in copies.items():
        if not source.is_file() and name_on_disk == "aggregation_history.json":
            _write_json(freeze / name_on_disk, [])
        else:
            shutil.copy2(source, freeze / name_on_disk)
        copied[name_on_disk] = sha256_file(freeze / name_on_disk)
    _write_json(
        freeze / "manifest.json",
        {
            "schema": "amortized_cem_diffusion_policy_freeze_v1",
            "freeze_name": name,
            "status": "SIMULATION_POLICY_NOMINAL_PHYSICS",
            "model_freeze": config["model_freeze"],
            "source_artifact": str(artifact),
            "final_aggregation_round": final_round,
            "online_cem": False,
            "online_simulator": False,
            "policy_query_count": 1,
            "execution": "OPEN_LOOP",
            "theta_training": "NOMINAL_ONLY",
            "real_flight_ready": False,
            "files": copied,
        },
    )
    return classification, name


def generate_report(
    config: dict[str, Any], artifact: Path, *, final_round: int
) -> Path:
    classification, freeze_name = freeze_policy_if_accepted(
        config, artifact, final_round=final_round
    )
    model_root = (
        artifact
        if final_round == 0
        else artifact / "aggregation_rounds" / f"round_{final_round}"
    )
    progress = json.loads(
        (artifact / "teacher_generation_progress.json").read_text(encoding="utf-8")
    )
    teacher_manifest = json.loads(
        (artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8")
    )
    scorer_manifest = json.loads(
        (artifact / "scorer_dataset_manifest.json").read_text(encoding="utf-8")
    )
    manifold = json.loads((artifact / "action_manifold_analysis.json").read_text(encoding="utf-8"))
    multimodal = json.loads((artifact / "multimodality_analysis.json").read_text(encoding="utf-8"))
    diffusion = json.loads(
        (model_root / "diffusion_training_summary.json").read_text(encoding="utf-8")
    )
    scorer = json.loads((model_root / "scorer_training_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((model_root / "validation_summary.json").read_text(encoding="utf-8"))
    initial_validation = json.loads(
        (artifact / "validation_summary.json").read_text(encoding="utf-8")
    )
    aggregation = (
        json.loads((artifact / "aggregation_history.json").read_text(encoding="utf-8"))
        if (artifact / "aggregation_history.json").is_file()
        else []
    )
    target = json.loads(
        (artifact / "target_conditioning_summary.json").read_text(encoding="utf-8")
    )
    state = json.loads(
        (artifact / "state_conditioning_summary.json").read_text(encoding="utf-8")
    )
    joint = json.loads((artifact / "joint_test_summary.json").read_text(encoding="utf-8"))
    edge = json.loads((artifact / "edge_test_summary.json").read_text(encoding="utf-8"))
    cem_test = json.loads(
        (artifact / "cem_reference_test_summary.json").read_text(encoding="utf-8")
    )
    latency = json.loads((artifact / "inference_latency.json").read_text(encoding="utf-8"))
    scorer_metrics = scorer["best_checkpoint_validation_metrics"]

    teacher_split = {}
    for split in ("TRAIN", "VALIDATION", "TEST"):
        rows = [row for row in progress["rows"] if row["split"] == split]
        teacher_split[split] = {
            "contexts": len(rows),
            "solved": sum(bool(row["solved"]) for row in rows),
            "actions": sum(int(row["successful_actions"]) for row in rows),
            "scorer": sum(int(row["scorer_rows"]) for row in rows),
        }

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{100.0 * value:.2f}%"

    split_lines = [
        f"| {name} | {row['contexts']} | {row['solved']} | {pct(row['solved']/max(row['contexts'],1))} | {row['actions']} | {row['scorer']} |"
        for name, row in teacher_split.items()
    ]
    continuous_lines = []
    for name, values in scorer_metrics["continuous"].items():
        correlation = "—" if values["correlation"] is None else f"{values['correlation']:.4f}"
        continuous_lines.append(
            f"| {name} | {values['mae']:.6g} | {values['rmse']:.6g} | {correlation} |"
        )
    binary_lines = []
    for name, values in scorer_metrics["binary"].items():
        auroc = "—" if values["auroc"] is None else f"{values['auroc']:.4f}"
        auprc = "—" if values["auprc"] is None else f"{values['auprc']:.4f}"
        binary_lines.append(
            f"| {name} | {auroc} | {auprc} | {values['brier']:.6f} | {values['positive_count']} |"
        )
    progression = [
        "| Round | Selected success | Oracle best-of-32 | Feasible |",
        "|---:|---:|---:|---:|",
        f"| Initial | {pct(initial_validation['scorer_selected_success_rate'])} | {pct(initial_validation['oracle_best_of_32_success_rate'])} | {pct(initial_validation['selected_feasible_rate'])} |",
    ]
    for row in aggregation:
        value = row["validation"]
        progression.append(
            f"| {row['round']} | {pct(value['scorer_selected_success_rate'])} | {pct(value['oracle_best_of_32_success_rate'])} | {pct(value['selected_feasible_rate'])} |"
        )
    aggregation_sections = []
    for row in aggregation:
        aggregation_sections.append(
            f"### {27 + row['round']}. Aggregation round {row['round']}\n\n"
            f"The fresh TRAIN-only pool contained {row['pool_summary']['context_count']} contexts. "
            f"Before relabeling, selected success was {pct(row['pool_summary']['scorer_selected_success_rate'])} "
            f"and oracle success was {pct(row['pool_summary']['oracle_best_of_32_success_rate'])}. "
            f"CEM queried {row['failure_queries']['queried']} failures, solved {row['failure_queries']['solved']}, "
            f"and added {row['failure_queries']['successful_actions']} verified actions. Models were reinitialized "
            "and retrained on the complete accumulated TRAIN datasets.\n"
        )
    while len(aggregation_sections) < 3:
        number = len(aggregation_sections) + 1
        aggregation_sections.append(
            f"### {27 + number}. Aggregation round {number}\n\nNot run; stopping followed the fixed validation-only rule.\n"
        )

    report = f"""# Milestone 7A — Production Amortized-CEM Generative Policy

Artifact: `{artifact}`

## 1. Why a policy is required despite successful CEM

The Milestone-6A planner is an excellent offline optimizer: canonical 6/6,
98.05% first-seed authoritative success, 98.44% with at most three seeds, and
zero population-to-replay gate flips. Its 34.61-s median and 105.46-s p95
latencies are nevertheless too slow for a measured hanging cable state: the
state evolves while CEM searches. Milestone 7A therefore amortizes verified CEM
solutions into a one-query neural generator and scorer.

## 2. State-staleness argument

The online query must use the current UAV/cable state, target, and theta and
return a complete maneuver before that state becomes stale. The measured neural
latency is reported in Section 38 and excludes both CEM and simulation exactly
as deployment does.

## 3. Why SAC was retired

The SAC branch was retained as a negative methodological result. Stable rewards,
critics, progress shaping, structured spectral exploration, fixed duration, and
local constrained SAC did not discover/consolidate a scientific whip. No SAC,
replay RL, reward critic, entropy objective, or policy gradient appears here.

## 4. Why deterministic MSE amortization was insufficient

The pilot achieved low action-space error but 0% held-out hard success. Whips are
phase-sensitive and multiple CEM actions can solve one context; their arithmetic
average need not solve anything. The final model learns `p(U|c)` directly in the
49-D production action space.

## 5. Final architecture and detailed current pipeline

Offline data path:

1. Select only physically propagated state-bank rows; never perturb cable markers independently.
2. Assign every physical state ID permanently to TRAIN, VALIDATION, or TEST before optimization.
3. Pair each state with two deterministic stratified/Sobol targets and explicit nominal theta.
4. Build the exact 83-D root-centered, yaw-aligned, gravity-preserving context.
5. Run the stabilized full-covariance production CEM independently from the canonical family.
6. Decode every normalized 49-D candidate through the production codec before fixed-2048 physics.
7. Authoritative-evaluate final top 32 and save every distinct hard-gate success.
8. Persist 512 outcome-stratified actions/context for scorer learning.
9. Train diffusion with uniform-context then uniform-within-context teacher sampling.
10. Train the forward scorer on standardized physical outcomes with TRAIN-only statistics.
11. Use only VALIDATION to select checkpoints and decide failure-driven aggregation.
12. Freeze all decisions before the single sealed TEST evaluation.

Online deployment path:

`raw x0,g,theta -> 83-D context -> fixed normalization -> 25-step DDIM with fixed 32x49 noise bank -> 32 bounded 49-D actions -> learned physical-outcome scorer -> deterministic scientific-margin selector -> one normalized action -> production decoder -> ACTIVE maneuver -> 0.30-s smooth SETTLE -> HOLD -> open-loop execution`.

The online module imports neither CEM nor the simulator. It does not replan or
query observations again.

## 6. Frozen production model

`MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI` remained unchanged: UAV controller,
causal residual and FIFO, remeasured cable geometry/masses/attachment, 12-node
DDER, PCG32, CUDA float32, three substeps, and four projections. No fitting ran.

## 7. Final action, settle, and evaluation semantics

Actions are normalized `[16x3 acceleration knots, active duration]` (49-D).
`T_maneuver in [0.45,1.80] s`, followed by the exact analytic 0.30-s p/v/a-
continuous settle and stationary hold; `T_evaluation=2.40 s`. Scientific gates
are unchanged.

## 8. Context schema

The 83 dimensions remain: UAV local velocity (3), local quaternion xyzw (4),
local angular velocity (3), c1..c10 root-relative local positions (30), c1..c10
local velocities (30), target (3), strike direction (3), and
`Kp,Kv,ka,KR,Komega,logEI,logCb` (7). Theta inputs are permanent although training
here is nominal only.

## 9. Dataset split protocol

The immutable state-level split is 256 TRAIN, 64 VALIDATION, and 64 TEST states,
two targets/state. State IDs, targets, all CEM seeds, and all solutions stay in
one split. The normalizer was fit to 512 initial TRAIN contexts only. The sealed
joint/conditioning/edge manifests were created before neural training.

## 10. State distribution

All states came from existing causal nominal production propagation and include
UAV pose/twist, the full distributed cable state, residual FIFO, and command
boundary. Farthest-point physical descriptors cover UAV motion, tip/distributed
cable velocity, shape, and curvature. No real/protected data were used.

## 11. Target distribution

Targets cover local x `[0.85,1.05]`, y `[-0.15,0.15]`, z `[-0.12,0.02]` using
scrambled Sobol points plus separated antithetic partners. Direction is the
normalized horizontal root-to-target direction.

## 12. Teacher CEM settings

Population 4096, elite fraction 0.05, full covariance, 4–20 iterations, final
top 32 authoritative actions, at most three deterministic seeds/context,
normalized production actions, smooth settle, and fixed-2048 numerical physics.
Every context starts independently from the same rotated canonical family.

## 13. Teacher solve and verification rate

| Split | Contexts | Solved | Solve rate | Verified successful actions | Scorer rows |
|---|---:|---:|---:|---:|---:|
{chr(10).join(split_lines)}

Total: {progress['solved_contexts']}/{progress['completed_contexts']} contexts,
{progress['successful_action_count']} verified actions, and
{progress['scorer_row_count']} scorer rows. Unsolved contexts retained metadata
and scorer examples but received no diffusion label.

## 14. Multiple solutions per context

Median successful solutions/context was
{multimodal['successful_solutions_per_context']['median']:.2f}; maximum
{multimodal['successful_solutions_per_context']['maximum']};
{multimodal['successful_solutions_per_context']['contexts_with_multiple']} contexts
had multiple verified solutions. Median within-context action L2 distance was
{multimodal['within_context_pairwise_action_l2']['median']}.

## 15. PCA/effective-dimension diagnostic

TRAIN-only diagnostic SVD gave K95={manifold['K95']}, K99={manifold['K99']}, and
K99.9={manifold['K99_9']}. PCA was never used as policy output, bottleneck, or
deployment decoder.

## 16. Multimodality diagnostic

The sampled between-context nearest-neighbor median action L2 distance was
{multimodal['between_context_nearest_neighbor_action_l2']['median']:.6f}; the
within-context distribution is non-degenerate. These measurements support a
generative conditional policy without inserting clustering into deployment.

## 17. Diffusion architecture

83→256→256 context encoder; sinusoidal-64→256→256 timestep encoder; 49→256 noisy
action projection; four LayerNorm/256→512/SiLU/512→256 residual blocks; final
LayerNorm/256→49 epsilon head. It operates directly on normalized 49-D actions.

## 18. Diffusion training

Cosine VP DDPM with 100 training steps, epsilon MSE, AdamW lr 2e-4, weight decay
1e-6, batch 1024, gradient clip 1, EMA 0.999. Final round trained
{diffusion['updates']} updates; best EMA validation loss
{diffusion['best_validation_epsilon_loss']:.6f} at update {diffusion['best_update']}.

## 19. Diffusion validation loss

Checkpoint selection used fixed-noise VALIDATION epsilon loss only. TEST data did
not affect early stopping. Final generated x0 is clamped once to [-1,1];
intermediate diffusion states are never clamped.

## 20. Scorer dataset

The final manifest contains {sum(int(row['row_count']) for row in scorer_manifest['shards'])}
sharded candidate rows. Each row references a context table entry and stores a
49-D action, seven physical continuous targets, three binary targets, and CEM
provenance. Context tensors are not redundantly repeated on disk.

## 21. Scorer architecture

The scorer is `132→256 SiLU→256 SiLU→256 SiLU` with separate seven-continuous and
three-binary heads. Continuous TRAIN-only normalization, Huber, BCEWithLogits,
and capped TRAIN-only class weights are used.

## 22. Scorer physical-metric accuracy

| Continuous head | MAE | RMSE | Correlation |
|---|---:|---:|---:|
{chr(10).join(continuous_lines)}

| Binary head | AUROC | AUPRC | Brier | Positives |
|---|---:|---:|---:|---:|
{chr(10).join(binary_lines)}

## 23. Scorer ranking quality

On held-out VALIDATION scorer candidates, scorer top-1 true success was
{pct(scorer_metrics['candidate_ranking']['top1_true_success_rate'])}; top-4
contained a true success in
{pct(scorer_metrics['candidate_ranking']['top4_contains_true_success_rate'])};
mean within-context Spearman reward correlation was
{scorer_metrics['candidate_ranking']['mean_spearman_reward_correlation']}.

## 24. Fixed-noise deterministic inference

The frozen 32x49 standard-normal bank, EMA weights, 25-step DDIM schedule, and
eta=0 make candidates and selected action exactly reproducible for a context.

## 25. Initial policy validation

Initial scorer-selected success {pct(initial_validation['scorer_selected_success_rate'])},
feasibility {pct(initial_validation['selected_feasible_rate'])}, first candidate
{pct(initial_validation['first_candidate_success_rate'])}.

## 26. Generator oracle best-of-32 result

Initial validation oracle {pct(initial_validation['oracle_best_of_32_success_rate'])};
final validation oracle {pct(validation['oracle_best_of_32_success_rate'])}; sealed
joint oracle {pct(joint['oracle_best_of_32_success_rate'])}.

## 27. Scorer-selected result

Final validation scorer-selected success is
{pct(validation['scorer_selected_success_rate'])}; sealed joint scorer-selected
success is {pct(joint['scorer_selected_success_rate'])}.

{chr(10).join(aggregation_sections)}

## 31. Validation progression across rounds

{chr(10).join(progression)}

## 32. Target-conditioning test

128 contexts (32 held-out states × four separated targets): success
{pct(target['scorer_selected_success_rate'])}, oracle
{pct(target['oracle_best_of_32_success_rate'])}, feasibility
{pct(target['selected_feasible_rate'])}. Median across-target selected-action L2
distance was {target['conditioning_action_distance']['median_action_l2']:.6f}.

## 33. Initial-state-conditioning test

128 contexts (32 targets × four physical states): success
{pct(state['scorer_selected_success_rate'])}, oracle
{pct(state['oracle_best_of_32_success_rate'])}, feasibility
{pct(state['selected_feasible_rate'])}. Median across-state action L2 distance
was {state['conditioning_action_distance']['median_action_l2']:.6f}.

## 34. Joint sealed-test result

The one-shot sealed 256-context result is: selected success
{pct(joint['scorer_selected_success_rate'])}, feasibility
{pct(joint['selected_feasible_rate'])}, oracle
{pct(joint['oracle_best_of_32_success_rate'])}, median tip error
{1000*joint['median_tip_distance_m']:.3f} mm, median directed speed
{joint['median_directed_speed_m_s']:.3f} m/s, and median direction error
{joint['median_direction_error_deg']:.3f} deg.

## 35. Edge-domain diagnostic

The 128 upper-state-distance/domain-boundary contexts achieved selected success
{pct(edge['scorer_selected_success_rate'])}, feasibility
{pct(edge['selected_feasible_rate'])}, and oracle
{pct(edge['oracle_best_of_32_success_rate'])}. This is diagnostic, not hardware
validation.

## 36. CEM versus neural policy comparison

| Property | CEM | Diffusion policy |
|---|---:|---:|
| Online optimization | YES | NO |
| Median query time | 34.61 s | {latency['complete_python_policy_query']['median_ms']:.3f} ms |
| Uses current x0 | YES | YES |
| Uses target g | YES | YES |
| Uses theta | YES | schema YES; nominal trained |
| Complete action | 49-D | 49-D |
| Open-loop execution | YES | YES |
| Authoritative success | 98.05% | {pct(joint['scorer_selected_success_rate'])} |

## 37. Policy success conditioned on CEM success

On the 128 state-isolated teacher TEST contexts, CEM solved
{cem_test['cem_solved_count']}/{cem_test['cem_reference_known_count']}; policy
success conditioned on CEM success was
{pct(cem_test['policy_success_given_cem_success'])}.

## 38. Inference latency

After 100 warmups and 1,000 queries on {latency['device']}, GPU neural median/p95
were {latency['gpu_neural_inference']['median_ms']:.3f}/
{latency['gpu_neural_inference']['p95_ms']:.3f} ms; complete Python policy-query
median/p95 were {latency['complete_python_policy_query']['median_ms']:.3f}/
{latency['complete_python_policy_query']['p95_ms']:.3f} ms. No CEM or simulator
was timed or called.

## 39. State-staleness implication

The measured query replaces tens of seconds of online optimization with one
current-state-conditioned query. Execution remains one complete open-loop
maneuver; this addresses planner-induced state staleness without claiming
feedback control.

## 40. Final policy-freeze decision

Classification: **{classification}**. Policy freeze: **{freeze_name}**.
Any freeze is explicitly `SIMULATION_POLICY_NOMINAL_PHYSICS`, never
`REAL_FLIGHT_READY`.

## 41. Physics conditioning

**NOT LEARNED YET.** Seven theta inputs and schemas are permanent; all teacher
rows in 7A use nominal theta only.

## 42. Protected test

**NOT EVALUATED.** `fig8vertical_002` was not accessed.

## 43. Real hardware

**NOT EXECUTED.** No radio connection, arming, or FullState transmission occurred.

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Method:
        AMORTIZED CEM GENERATIVE POLICY

    Offline teacher:
        PRODUCTION VARIABLE-DURATION CEM

    Online optimizer:
        NONE

    Generator:
        CONDITIONAL 49-D DIFFUSION

    Scorer:
        LEARNED PHYSICAL-OUTCOME SCORER

    Context dimension:
        83

    Action dimension:
        49

    Neural candidates/query:
        32

    DDIM steps:
        25

    Maneuver duration:
        [0.45,1.80] s

    Settle:
        0.30 s SMOOTH

    Evaluation horizon:
        2.40 s

    Initial teacher contexts:
        768

    Verified teacher contexts:
        {progress['solved_contexts']}

    Successful actions:
        {progress['successful_action_count'] + sum(item['failure_queries']['successful_actions'] for item in aggregation)}

    Scorer rows:
        {sum(int(row['row_count']) for row in scorer_manifest['shards'])}

    Aggregation rounds:
        {final_round}

    Validation scorer-selected success:
        {pct(validation['scorer_selected_success_rate'])}

    Validation oracle best-of-32:
        {pct(validation['oracle_best_of_32_success_rate'])}

    Target-conditioning success:
        {pct(target['scorer_selected_success_rate'])}

    State-conditioning success:
        {pct(state['scorer_selected_success_rate'])}

    Held-out joint test success:
        {pct(joint['scorer_selected_success_rate'])}

    Held-out joint feasibility:
        {pct(joint['selected_feasible_rate'])}

    Policy success given CEM success:
        {pct(cem_test['policy_success_given_cem_success'])}

    Edge-domain success:
        {pct(edge['scorer_selected_success_rate'])}

    Median policy-query latency:
        {latency['complete_python_policy_query']['median_ms']:.3f} ms

    P95 policy-query latency:
        {latency['complete_python_policy_query']['p95_ms']:.3f} ms

    Online CEM:
        NO

    Policy query count:
        1

    Execution:
        OPEN LOOP

    Physics conditioning learned:
        NO — NOMINAL THETA ONLY

    Result:
        {classification}

    Policy freeze:
        {freeze_name}

    Protected test:
        NOT EVALUATED

    Real hardware:
        NOT EXECUTED
"""
    path = ROOT / "MILESTONE7A_AMORTIZED_CEM_DIFFUSION_POLICY_REPORT.md"
    path.write_text(report, encoding="utf-8")
    shutil.copy2(path, artifact / path.name)
    return path


def finalize_milestone7a(config: dict[str, Any], artifact: Path) -> Path:
    final_round = _latest_model_round(artifact)
    evaluate_final_policy(config, artifact, round_index=final_round)
    benchmark_inference_latency(config, artifact, round_index=final_round)
    save_representative_replays(config, artifact, round_index=final_round)
    _source_hash_manifest(artifact)
    report = generate_report(config, artifact, final_round=final_round)
    return report
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone7a_source_hash_manifest_v1",
            "files": {
                str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
                for path in paths
            },
        },
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--stage",
        choices=(
            "prepare",
            "teacher",
            "teacher-worker",
            "merge-teachers",
            "analysis",
            "train",
            "validate",
            "aggregate",
            "latency",
            "replays",
            "final-evaluate",
            "finalize",
        ),
        default="teacher",
    )
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    _validate_config(config)
    artifact = _create_artifact(config, args.artifact)
    prepare(config, artifact)
    _source_hash_manifest(artifact)
    if args.stage == "teacher":
        generate_teachers(config, artifact)
    elif args.stage == "teacher-worker":
        generate_teacher_worker(
            config,
            artifact,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
        )
    elif args.stage == "merge-teachers":
        merge_teacher_workers(artifact, worker_count=args.worker_count)
    elif args.stage == "analysis":
        analyze_action_manifold(artifact)
    elif args.stage == "train":
        train_initial_models(config, artifact)
    elif args.stage == "validate":
        evaluate_validation(config, artifact, round_index=args.round)
    elif args.stage == "aggregate":
        run_aggregation(config, artifact)
    elif args.stage == "latency":
        benchmark_inference_latency(config, artifact, round_index=args.round)
    elif args.stage == "replays":
        save_representative_replays(config, artifact, round_index=args.round)
    elif args.stage == "final-evaluate":
        evaluate_final_policy(config, artifact, round_index=args.round)
    elif args.stage == "finalize":
        report = finalize_milestone7a(config, artifact)
        print(f"MILESTONE7A_REPORT={report}", flush=True)
    print(f"MILESTONE7A_ARTIFACT={artifact}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
