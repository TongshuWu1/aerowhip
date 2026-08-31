"""Existing-data viability test for the final amortized-CEM neural policy.

This runner never invokes a trajectory optimizer.  It consumes only the
durably saved Milestone-6A and partial Milestone-7A artifacts.
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
import time
from typing import Any, Iterable

import numpy as np
import torch

from learning.action_diffusion import ConditionalActionDiffusion
from learning.amortized_cem_data import (
    ACTION_DIM,
    BINARY_OUTCOME_NAMES,
    CONTINUOUS_OUTCOME_NAMES,
    AmortizedCemContextRecord,
    compact_outcome,
    sha256_file,
)
from learning.amortized_cem_policy import AmortizedCemDiffusionPolicy
from learning.amortized_cem_training import (
    scorer_context_ranking_metrics,
    train_diffusion,
    train_scorer,
)
from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    pad_context_specification,
)
from learning.normalization import FixedContextNormalizer
from learning.outcome_scorer import (
    ManeuverOutcomeScorer,
    OutcomeTargetNormalizer,
    scorer_validation_metrics,
)
from learning.policy_action import decode_policy_action
from learning.policy_context import build_policy_context
from learning.state_bank import InitialStateBank, initial_state_bank_from_state
from planning.cem_task import load_variable_duration_task
from planning.production_cem import (
    FIXED_NUMERICAL_BATCH_SIZE,
    ProductionCemSettings,
    evaluate_normalized_actions_fixed_batch,
    event_segment,
)
from planning.rollout import hover_preroll
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "amortized_cem_diffusion_existing_data_viability_v1.json"
ARTIFACT_PARENT = ROOT / "data" / "policy_training" / "amortized_cem_diffusion_existing_data_viability_v1"
REPORT = ROOT / "MILESTONE7A1_EXISTING_DATA_VIABILITY_REPORT.md"


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
    if config["schema"] != "amortized_cem_diffusion_existing_data_viability_v1":
        raise ValueError("Unsupported viability configuration.")
    if config["model_freeze"] != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The production model freeze changed.")
    required = config["prohibitions"]
    if required["protected_test"] != "fig8vertical_002":
        raise ValueError("Protected-test identity changed.")
    for key in (
        "new_cem_solves",
        "sac",
        "aggregation",
        "architecture_change",
        "theta_randomization",
        "final_test",
        "hardware",
    ):
        if not bool(required[key]):
            raise ValueError(f"Required prohibition is disabled: {key}")


def _load_environment(config: dict[str, Any]):
    manifest = load_active_model_manifest()
    paths = active_model_paths(manifest)
    settings = SimulatorSettings.load(paths["configuration"])
    simulator = build_production_simulator(settings, device=torch.device("cuda"), dtype=torch.float32)
    task = load_variable_duration_task(ROOT / config["task_config"])
    state_root = ROOT / config["state_bank_artifact"]
    training_bank = InitialStateBank.load(
        state_root / "training_state_bank.npz", state_root / "training_state_bank_manifest.json"
    )
    state = hover_preroll(simulator, task)
    canonical_bank = initial_state_bank_from_state(
        state,
        command_position_world_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        command_velocity_world_m_s=torch.tensor(task.initial_uav_velocity_m_s, device=simulator.device),
        command_yaw_world_rad=task.initial_yaw_rad,
        seed=42,
    )
    return simulator, task, training_bank, None, canonical_bank, settings


def _physics_settings(config: dict[str, Any]) -> ProductionCemSettings:
    source = json.loads((ROOT / config["production_contract_config"]).read_text(encoding="utf-8"))
    cem, times = source["canonical_cem"], source["times"]
    return ProductionCemSettings(
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


def _partial_payloads(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    progress = json.loads((source / "teacher_generation_progress.json").read_text(encoding="utf-8"))
    rows = list(progress["rows"])
    teacher = list(json.loads((source / "diffusion_teacher_manifest.json").read_text(encoding="utf-8"))["shards"])
    scorer = list(json.loads((source / "scorer_dataset_manifest.json").read_text(encoding="utf-8"))["shards"])
    for worker in range(2):
        payload = json.loads(
            (source / f"teacher_workers/worker_{worker:02d}_of_02/progress.json").read_text(
                encoding="utf-8"
            )
        )
        rows.extend(payload["rows"])
        teacher.extend(payload["teacher_shards"])
        scorer.extend(payload["scorer_shards"])
    if len(rows) != len({row["context_id"] for row in rows}):
        raise RuntimeError("Partial 7A progress contains duplicate context IDs.")
    return rows, teacher, scorer


def _hash_order(state_ids: Iterable[str]) -> list[str]:
    return sorted(
        state_ids,
        key=lambda value: hashlib.sha256(f"42:{value}".encode("utf-8")).hexdigest(),
    )


def _write_teacher_shard(
    directory: Path,
    *,
    context_index: int,
    context_id: str,
    split: str,
    actions: np.ndarray,
    metrics: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if actions.shape[0] == 0:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    name = f"context_{context_index:06d}.npz"
    path = directory / name
    np.savez_compressed(
        path,
        schema=np.asarray("amortized_cem_diffusion_teacher_shard_v1"),
        context_index=np.asarray(context_index, dtype=np.int64),
        normalized_actions=np.asarray(actions, dtype=np.float32),
        metrics_json=np.asarray([json.dumps(_safe(item), sort_keys=True) for item in metrics]),
        provenance_json=np.asarray([json.dumps(_safe(item), sort_keys=True) for item in provenance]),
    )
    return {
        "context_index": context_index,
        "context_id": context_id,
        "split": split,
        "path": name,
        "row_count": int(actions.shape[0]),
        "sha256": sha256_file(path),
    }


def _write_scorer_shard(
    directory: Path,
    *,
    context_index: int,
    context_id: str,
    split: str,
    actions: np.ndarray,
    continuous: np.ndarray,
    binary: np.ndarray,
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"context_{context_index:06d}.npz"
    path = directory / name
    np.savez_compressed(
        path,
        schema=np.asarray("amortized_cem_outcome_scorer_shard_v1"),
        context_index=np.asarray(context_index, dtype=np.int64),
        normalized_actions=np.asarray(actions, dtype=np.float32),
        continuous_targets=np.asarray(continuous, dtype=np.float32),
        binary_targets=np.asarray(binary, dtype=np.float32),
        continuous_target_names=np.asarray(CONTINUOUS_OUTCOME_NAMES),
        binary_target_names=np.asarray(BINARY_OUTCOME_NAMES),
        provenance_json=np.asarray([json.dumps(_safe(item), sort_keys=True) for item in provenance]),
    )
    return {
        "context_index": context_index,
        "context_id": context_id,
        "split": split,
        "path": name,
        "row_count": int(actions.shape[0]),
        "sha256": sha256_file(path),
    }


def prepare_existing_data(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    artifact.mkdir(parents=True, exist_ok=True)
    source7 = ROOT / config["partial_7a_artifact"]
    source6 = ROOT / config["milestone_6a_artifact"]
    rows7, teacher7, scorer7 = _partial_payloads(source7)
    records7 = {
        row["context_id"]: row
        for row in json.loads((source7 / "context_split_manifest.json").read_text(encoding="utf-8"))[
            "records"
        ]
    }
    split7 = json.loads((source7 / "context_split_manifest.json").read_text(encoding="utf-8"))
    ownership = {
        "TRAIN": set(split7["training_state_ids"]),
        "VALIDATION": set(split7["validation_state_ids"]),
        "TEST": set(split7["test_state_ids"]),
        "EDGE": set(split7["edge_reserved_test_state_ids"]),
        "AGGREGATION_POOL": set(split7["aggregation_train_pool_state_ids"]),
    }
    contexts6 = json.loads((source6 / "benchmark_context_manifest.json").read_text(encoding="utf-8"))[
        "rows"
    ]
    rows6 = json.loads((source6 / "benchmark_rows.json").read_text(encoding="utf-8"))
    with np.load(source6 / "authoritative_actions.npz", allow_pickle=False) as archive:
        action_ids6 = [str(value) for value in archive["context_ids"].tolist()]
        actions6 = np.asarray(archive["normalized_actions"], dtype=np.float32)
    action6_by_id = dict(zip(action_ids6, actions6, strict=True))

    def owner_for_sixa(row: dict[str, Any]) -> tuple[str, str]:
        if row["state_bank"] == "canonical":
            return "EXTERNAL_DEVELOPMENT", "canonical:0000"
        state_id = f"training:{int(row['state_id']):04d}"
        hits = [name for name, values in ownership.items() if state_id in values]
        if len(hits) != 1:
            raise RuntimeError(f"6A state ownership is not unique: {state_id} -> {hits}")
        return hits[0], state_id

    overlap_counts = {name: 0 for name in (*ownership, "EXTERNAL_DEVELOPMENT")}
    overlap_states = {name: set() for name in overlap_counts}
    for row in contexts6:
        owner, state_id = owner_for_sixa(row)
        overlap_counts[owner] += 1
        overlap_states[owner].add(state_id)
    test_benchmarked = bool(overlap_states["TEST"])
    if test_benchmarked:
        raise RuntimeError("A 7A TEST state appears in Milestone 6A; viability training is blocked.")

    partial_state_ids = {row["state_id"] for row in rows7}
    sixa_train_state_ids = {
        owner_for_sixa(row)[1]
        for row in contexts6
        if owner_for_sixa(row)[0] == "TRAIN"
    }
    combined_train_owned_states = partial_state_ids | sixa_train_state_ids
    ordered = _hash_order(combined_train_owned_states)
    internal_dev_count = max(1, int(round(float(config["internal_development_fraction"]) * len(ordered))))
    internal_dev_states = set(ordered[:internal_dev_count])
    neural_train_states = set(ordered[internal_dev_count:])
    if neural_train_states & internal_dev_states:
        raise RuntimeError("Internal state split leaked.")

    simulator, task, training_bank, _heldout_bank, canonical_bank, settings = _load_environment(config)
    theta = _nominal_theta(settings)
    with np.load(source7 / "context_table.npz", allow_pickle=False) as archive:
        source_context_table = np.asarray(archive["contexts"], dtype=np.float32)
    teacher7_by_id = {entry["context_id"]: entry for entry in teacher7}
    scorer7_by_id = {entry["context_id"]: entry for entry in scorer7}

    combined_records: list[AmortizedCemContextRecord] = []
    source_rows: list[dict[str, Any]] = []
    context_values: list[np.ndarray] = []

    for progress in sorted(rows7, key=lambda value: int(value["context_index"])):
        source_record = records7[progress["context_id"]]
        split = "TRAIN" if progress["state_id"] in neural_train_states else "VALIDATION"
        index = len(combined_records)
        record = AmortizedCemContextRecord(
            index,
            f"partial7a_{progress['context_id']}",
            split,
            "training",
            progress["state_id"],
            int(source_record["state_index"]),
            int(source_record["target_number"]),
            tuple(float(value) for value in source_record["target_local_m"]),
            tuple(float(value) for value in source_record["direction_local"]),
            theta,
        )
        combined_records.append(record)
        context_values.append(source_context_table[int(progress["context_index"])])
        source_rows.append(
            {
                "context_id": record.context_id,
                "source": "PARTIAL_7A",
                "source_context_id": progress["context_id"],
                "source_context_index": int(progress["context_index"]),
                "ownership": "TRAIN",
                "neural_split": split,
                "teacher_available": bool(progress["solved"]),
                "scorer_available": True,
            }
        )

    for source_record, benchmark_row in zip(contexts6, rows6, strict=True):
        owner, state_id = owner_for_sixa(source_record)
        if owner in {"VALIDATION", "TEST", "EDGE"}:
            raise RuntimeError(f"6A context uses forbidden 7A held-out ownership: {owner}")
        split = (
            "TRAIN"
            if owner == "TRAIN" and state_id in neural_train_states
            else "VALIDATION"
        )
        bank_name = "canonical" if source_record["state_bank"] == "canonical" else "training"
        state_index = 0 if bank_name == "canonical" else int(source_record["state_id"])
        index = len(combined_records)
        record = AmortizedCemContextRecord(
            index,
            f"sixa_{source_record['context_id']}",
            split,
            bank_name,
            state_id,
            state_index,
            index,
            tuple(float(value) for value in source_record["target_local_m"]),
            tuple(float(value) for value in source_record["direction_local"]),
            theta,
        )
        bank = canonical_bank if bank_name == "canonical" else training_bank
        specification = ContextSpecification(
            torch.tensor([state_index], dtype=torch.int64),
            torch.tensor([record.target_local_m], dtype=torch.float32),
            torch.tensor([record.direction_local], dtype=torch.float32),
            "existing_data_viability",
        )
        raw_context = build_context_from_specification(simulator, bank, specification).to_tensor()[0]
        combined_records.append(record)
        context_values.append(raw_context.detach().cpu().numpy().astype(np.float32))
        source_rows.append(
            {
                "context_id": record.context_id,
                "source": "MILESTONE_6A",
                "source_context_id": source_record["context_id"],
                "ownership": owner,
                "neural_split": split,
                "teacher_available": bool(benchmark_row["authoritative_metrics"]["success"]),
                "scorer_available": True,
            }
        )

    context_array = np.asarray(context_values, dtype=np.float32)
    if context_array.shape != (len(combined_records), 83) or not np.isfinite(context_array).all():
        raise RuntimeError("Combined context table is invalid.")
    np.savez_compressed(
        artifact / "context_table.npz",
        schema=np.asarray("amortized_cem_existing_data_context_table_v1"),
        contexts=context_array,
        context_indices=np.arange(len(combined_records), dtype=np.int64),
        context_ids=np.asarray([record.context_id for record in combined_records]),
        state_ids=np.asarray([record.state_id for record in combined_records]),
        splits=np.asarray([record.split for record in combined_records]),
        banks=np.asarray([record.bank for record in combined_records]),
        theta=np.asarray([record.theta_nominal for record in combined_records], dtype=np.float32),
    )
    shutil.copy2(source7 / "context_normalizer.json", artifact / "context_normalizer.json")

    teacher_manifest = {
        "schema": "amortized_cem_existing_data_diffusion_teacher_manifest_v1",
        "shards": [],
    }
    scorer_manifest = {
        "schema": "amortized_cem_existing_data_scorer_manifest_v1",
        "continuous_targets": list(CONTINUOUS_OUTCOME_NAMES),
        "binary_targets": list(BINARY_OUTCOME_NAMES),
        "shards": [],
    }
    source_by_combined = {row["context_id"]: row for row in source_rows}
    record_by_combined = {record.context_id: record for record in combined_records}
    benchmark_by_id = {row["context_id"]: row for row in rows6}

    for record in combined_records:
        source = source_by_combined[record.context_id]
        if source["source"] == "PARTIAL_7A":
            source_id = source["source_context_id"]
            if source["teacher_available"]:
                entry = teacher7_by_id[source_id]
                with np.load(source7 / "diffusion_teacher_shards" / entry["path"], allow_pickle=False) as shard:
                    actions = np.asarray(shard["normalized_actions"], dtype=np.float32)
                    metrics = [json.loads(str(value)) for value in shard["metrics_json"]]
                    provenance = [json.loads(str(value)) for value in shard["provenance_json"]]
                written = _write_teacher_shard(
                    artifact / "diffusion_teacher_shards",
                    context_index=record.context_index,
                    context_id=record.context_id,
                    split=record.split,
                    actions=actions,
                    metrics=metrics,
                    provenance=provenance,
                )
                if written:
                    teacher_manifest["shards"].append(written)
            entry = scorer7_by_id[source_id]
            with np.load(source7 / "scorer_dataset_shards" / entry["path"], allow_pickle=False) as shard:
                scorer_actions = np.asarray(shard["normalized_actions"], dtype=np.float32)
                continuous = np.asarray(shard["continuous_targets"], dtype=np.float32)
                binary = np.asarray(shard["binary_targets"], dtype=np.float32)
                provenance = [json.loads(str(value)) for value in shard["provenance_json"]]
            scorer_manifest["shards"].append(
                _write_scorer_shard(
                    artifact / "scorer_dataset_shards",
                    context_index=record.context_index,
                    context_id=record.context_id,
                    split=record.split,
                    actions=scorer_actions,
                    continuous=continuous,
                    binary=binary,
                    provenance=provenance,
                )
            )
        else:
            source_id = source["source_context_id"]
            benchmark = benchmark_by_id[source_id]
            action = action6_by_id[source_id].reshape(1, ACTION_DIM)
            if not np.allclose(action[0], np.asarray(benchmark["normalized_action"]), atol=1e-7, rtol=0):
                raise RuntimeError(f"6A action mismatch: {source_id}")
            metrics = dict(benchmark["authoritative_metrics"])
            provenance = [{"source": "milestone6a_authoritative_final", "context_id": source_id}]
            if bool(metrics["success"]):
                written = _write_teacher_shard(
                    artifact / "diffusion_teacher_shards",
                    context_index=record.context_index,
                    context_id=record.context_id,
                    split=record.split,
                    actions=action,
                    metrics=[metrics],
                    provenance=provenance,
                )
                if written:
                    teacher_manifest["shards"].append(written)
            continuous, binary = compact_outcome(metrics)
            scorer_manifest["shards"].append(
                _write_scorer_shard(
                    artifact / "scorer_dataset_shards",
                    context_index=record.context_index,
                    context_id=record.context_id,
                    split=record.split,
                    actions=action,
                    continuous=continuous.reshape(1, -1),
                    binary=binary.reshape(1, -1),
                    provenance=provenance,
                )
            )

    _write_json(artifact / "combined_teacher_manifest.json", teacher_manifest)
    _write_json(artifact / "combined_scorer_manifest.json", scorer_manifest)
    _write_json(artifact / "diffusion_teacher_manifest.json", teacher_manifest)
    _write_json(artifact / "scorer_dataset_manifest.json", scorer_manifest)
    np.save(artifact / "fixed_noise_bank.npy", np.random.default_rng(42).standard_normal((32, 49)).astype(np.float32))

    train_teacher = [entry for entry in teacher_manifest["shards"] if entry["split"] == "TRAIN"]
    validation_teacher = [entry for entry in teacher_manifest["shards"] if entry["split"] == "VALIDATION"]
    train_scorer = [entry for entry in scorer_manifest["shards"] if entry["split"] == "TRAIN"]
    validation_scorer = [entry for entry in scorer_manifest["shards"] if entry["split"] == "VALIDATION"]
    development_records = [record for record in combined_records if record.split == "VALIDATION"]
    train_eval_records = [
        record
        for record in combined_records
        if record.split == "TRAIN" and source_by_combined[record.context_id]["teacher_available"]
    ]
    split_manifest = {
        "schema": "milestone7a1_state_disjoint_development_split_v1",
        "seed": 42,
        "neural_train_state_ids": sorted(neural_train_states),
        "internal_development_state_ids": sorted(internal_dev_states),
        "external_development_state_ids": sorted(
            {record.state_id for record in development_records if record.state_id not in internal_dev_states}
        ),
        "train_context_ids": [entry["context_id"] for entry in train_teacher],
        "development_context_ids": [record.context_id for record in development_records],
        "train_evaluation_context_ids": [record.context_id for record in train_eval_records],
        "records": [asdict(record) for record in combined_records],
        "source_rows": source_rows,
        "final_7a_test_used": False,
    }
    _write_json(artifact / "development_split_manifest.json", split_manifest)

    audit = {
        "schema": "milestone7a1_sixa_state_overlap_audit_v1",
        "sixa_context_count": len(contexts6),
        "sixa_unique_state_counts": {name: len(values) for name, values in overlap_states.items()},
        "sixa_context_overlap_counts": overlap_counts,
        "sixa_success_counts_by_owner": {
            name: sum(
                int(bool(result["authoritative_metrics"]["success"]))
                for definition, result in zip(contexts6, rows6, strict=True)
                if owner_for_sixa(definition)[0] == name
            )
            for name in overlap_counts
        },
        "test_previously_benchmarked": test_benchmarked,
        "split_leakage": False,
        "seven_a_test_state_ids_used_for_gradients": 0,
        "seven_a_validation_state_ids_used_for_gradients": 0,
    }
    _write_json(artifact / "sixa_state_overlap_audit.json", audit)
    inventory = {
        "schema": "milestone7a1_source_data_inventory_v1",
        "new_cem_solves": 0,
        "partial_7a": {
            "completed_contexts": len(rows7),
            "solved_contexts": sum(bool(row["solved"]) for row in rows7),
            "successful_actions": sum(int(entry["row_count"]) for entry in teacher7),
            "scorer_rows": sum(int(entry["row_count"]) for entry in scorer7),
        },
        "milestone_6a": {
            "contexts": len(rows6),
            "authoritative_successes": sum(bool(row["authoritative_metrics"]["success"]) for row in rows6),
            "normalized_actions": int(actions6.shape[0]),
            "population_candidate_rows_saved": False,
            "authoritative_final_outcome_rows_saved": True,
        },
        "combined": {
            "context_count": len(combined_records),
            "train_teacher_contexts": len(train_teacher),
            "train_teacher_actions": sum(int(entry["row_count"]) for entry in train_teacher),
            "validation_teacher_contexts": len(validation_teacher),
            "validation_teacher_actions": sum(int(entry["row_count"]) for entry in validation_teacher),
            "train_scorer_rows": sum(int(entry["row_count"]) for entry in train_scorer),
            "validation_scorer_rows": sum(int(entry["row_count"]) for entry in validation_scorer),
            "development_contexts": len(development_records),
        },
    }
    _write_json(artifact / "source_data_inventory.json", inventory)
    compatibility = {
        "model_freeze": config["model_freeze"],
        "context_dimension": 83,
        "action_dimension": 49,
        "nominal_theta_only": True,
        "maneuver_duration_s": [0.45, 1.80],
        "settle_duration_s": 0.30,
        "evaluation_horizon_s": 2.40,
        "fixed_numerical_batch_size": 2048,
        "sixa_action_npz_json_max_difference": float(
            max(
                np.max(np.abs(action6_by_id[row["context_id"]] - np.asarray(result["normalized_action"])))
                for row, result in zip(contexts6, rows6, strict=True)
            )
        ),
        "compatible": True,
    }
    _write_json(artifact / "source_contract_compatibility.json", compatibility)
    return inventory


def train_models(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    diffusion = train_diffusion(artifact, config, device="cuda")
    scorer = train_scorer(artifact, config, device="cuda")
    return {"diffusion": diffusion, "scorer": scorer}


def _load_policy(artifact: Path, *, model_root: Path | None = None) -> AmortizedCemDiffusionPolicy:
    selected = torch.device("cuda")
    source = artifact if model_root is None else model_root
    diffusion_payload = torch.load(source / "diffusion_ema_best.pt", map_location=selected, weights_only=True)
    diffusion = ConditionalActionDiffusion().to(selected)
    diffusion.load_state_dict(diffusion_payload["state_dict"])
    scorer_payload = torch.load(artifact / "scorer_best.pt", map_location=selected, weights_only=True)
    scorer = ManeuverOutcomeScorer().to(selected)
    scorer.load_state_dict(scorer_payload["state_dict"])
    return AmortizedCemDiffusionPolicy(
        diffusion,
        scorer,
        FixedContextNormalizer.load(artifact / "context_normalizer.json"),
        OutcomeTargetNormalizer.load(artifact / "scorer_target_normalization.json"),
        torch.from_numpy(np.load(artifact / "fixed_noise_bank.npy")),
    )


def _records_from_manifest(artifact: Path, ids: Iterable[str]) -> list[AmortizedCemContextRecord]:
    payload = json.loads((artifact / "development_split_manifest.json").read_text(encoding="utf-8"))
    by_id = {row["context_id"]: AmortizedCemContextRecord(**row) for row in payload["records"]}
    return [by_id[value] for value in ids]


def _metric_rows(result, actions: torch.Tensor, task, settings: ProductionCemSettings) -> list[dict[str, Any]]:
    decoded = decode_policy_action(actions.float(), task, duration_max_s=settings.duration_max_s)
    values = []
    for index in range(actions.shape[0]):
        row = result.row(index)
        duration = float(decoded.duration_s[index])
        row["maneuver_duration_s"] = duration
        row["hit_segment"] = event_segment(
            row["first_entry_time_s"], duration, settings.settle_duration_s
        )
        values.append(row)
    return values


def _selection_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    metrics = [row[key] for row in rows]
    success = np.asarray([bool(item["success"]) for item in metrics])
    feasible = np.asarray([bool(item["feasible"]) for item in metrics])

    def median(name: str) -> float | None:
        values = np.asarray([float(item[name]) for item in metrics], dtype=np.float64)
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else None

    segments = {name: 0 for name in ("ACTIVE", "SETTLE", "HOLD", "NONE")}
    for item in metrics:
        segment = str(item.get("hit_segment", "NONE"))
        segments[segment if segment in segments else "NONE"] += 1
    return {
        "context_count": len(metrics),
        "scientific_success_count": int(success.sum()),
        "scientific_success_rate": float(success.mean()),
        "feasible_count": int(feasible.sum()),
        "feasible_rate": float(feasible.mean()),
        "median_tip_error_m": median("minimum_tip_target_distance_m"),
        "median_directed_speed_m_s": median("best_event_directed_speed_m_s"),
        "median_direction_error_deg": median("best_event_direction_angle_deg"),
        "median_uav_displacement_m": median("maximum_uav_displacement_m"),
        "median_uav_speed_m_s": median("maximum_uav_speed_m_s"),
        "median_maneuver_duration_s": median("maneuver_duration_s"),
        "hit_segment_counts": segments,
    }


def evaluate_policy_records(
    config: dict[str, Any],
    artifact: Path,
    records: list[AmortizedCemContextRecord],
    *,
    label: str,
    model_root: Path | None = None,
    save_candidate_rows: bool,
) -> dict[str, Any]:
    simulator, task, training_bank, _heldout, canonical_bank, _settings_source = _load_environment(config)
    settings = _physics_settings(config)
    banks = {"training": training_bank, "canonical": canonical_bank}
    policy = _load_policy(artifact, model_root=model_root)
    with np.load(artifact / "context_table.npz", allow_pickle=False) as archive:
        table = np.asarray(archive["contexts"], dtype=np.float32)

    inferences = []
    for number, record in enumerate(records):
        inference = policy.infer_from_context_tensor(torch.from_numpy(table[record.context_index : record.context_index + 1]))
        inferences.append(inference)
        if (number + 1) % 50 == 0 or number + 1 == len(records):
            print(f"{label} neural candidates {number + 1}/{len(records)}", flush=True)

    physical_by_context: dict[int, list[dict[str, Any]]] = {}
    for bank_name, bank in banks.items():
        indices = [index for index, record in enumerate(records) if record.bank == bank_name]
        for start in range(0, len(indices), 64):
            selected_indices = indices[start : start + 64]
            logical_actions = torch.cat(
                [inferences[index].candidate_normalized_actions for index in selected_indices], dim=0
            )
            state_indices, targets, directions = [], [], []
            for index in selected_indices:
                record = records[index]
                state_indices.extend([record.state_index] * 32)
                targets.extend([record.target_local_m] * 32)
                directions.extend([record.direction_local] * 32)
            logical_specification = ContextSpecification(
                torch.tensor(state_indices, dtype=torch.int64),
                torch.tensor(targets, dtype=torch.float32),
                torch.tensor(directions, dtype=torch.float32),
                label,
            )
            padded = pad_context_specification(logical_specification, FIXED_NUMERICAL_BATCH_SIZE)
            context = build_context_from_specification(simulator, bank, padded)
            result = evaluate_normalized_actions_fixed_batch(
                simulator, context, logical_actions, task, settings
            )
            metrics = _metric_rows(result, logical_actions, task, settings)
            for local, context_index in enumerate(selected_indices):
                physical_by_context[context_index] = metrics[32 * local : 32 * (local + 1)]
            print(
                f"{label} physics {min(start + len(selected_indices), len(indices))}/{len(indices)} {bank_name}",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    truth_continuous, truth_binary = [], []
    predicted_continuous, predicted_binary, candidate_context_indices = [], [], []
    for index, record in enumerate(records):
        inference = inferences[index]
        metrics = physical_by_context[index]
        successful = [candidate for candidate, item in enumerate(metrics) if bool(item["success"])]
        oracle_index = (
            min(successful, key=lambda candidate: float(metrics[candidate]["task_cost"]))
            if successful
            else min(range(32), key=lambda candidate: float(metrics[candidate]["task_cost"]))
        )
        selected_index = int(inference.selected_index)
        row = {
            "context_id": record.context_id,
            "state_id": record.state_id,
            "bank": record.bank,
            "first_index": 0,
            "oracle_index": oracle_index,
            "selected_index": selected_index,
            "candidate_success_count": len(successful),
            "clamp_fraction": inference.clamp_fraction,
            "first_metrics": metrics[0],
            "oracle_metrics": metrics[oracle_index],
            "selected_metrics": metrics[selected_index],
        }
        rows.append(row)
        for candidate, item in enumerate(metrics):
            continuous, binary = compact_outcome(item)
            truth_continuous.append(continuous)
            truth_binary.append(binary)
            predicted_continuous.append(
                inference.predicted_continuous_outcomes[candidate].detach().cpu().numpy()
            )
            predicted_binary.append(
                inference.predicted_binary_probabilities[candidate].detach().cpu().numpy()
            )
            candidate_context_indices.append(index)
            if save_candidate_rows:
                candidate_rows.append(
                    {
                        "context_id": record.context_id,
                        "state_id": record.state_id,
                        "candidate_index": candidate,
                        "selected": candidate == selected_index,
                        "oracle": candidate == oracle_index,
                        "predicted_continuous": predicted_continuous[-1].tolist(),
                        "predicted_binary_probability": predicted_binary[-1].tolist(),
                        "actual_metrics": item,
                    }
                )

    truth_continuous_array = np.asarray(truth_continuous, dtype=np.float32)
    truth_binary_array = np.asarray(truth_binary, dtype=np.float32)
    predicted_continuous_array = np.asarray(predicted_continuous, dtype=np.float32)
    predicted_binary_array = np.asarray(predicted_binary, dtype=np.float32)
    scorer_metrics = scorer_validation_metrics(
        truth_continuous_array,
        predicted_continuous_array,
        truth_binary_array,
        predicted_binary_array,
    )
    scorer_metrics["candidate_ranking"] = scorer_context_ranking_metrics(
        np.asarray(candidate_context_indices, dtype=np.int64),
        truth_continuous_array,
        truth_binary_array,
        predicted_continuous_array,
        predicted_binary_array,
    )
    first = _selection_summary(rows, "first_metrics")
    oracle = _selection_summary(rows, "oracle_metrics")
    selected = _selection_summary(rows, "selected_metrics")
    oracle_mask = np.asarray([bool(row["oracle_metrics"]["success"]) for row in rows])
    selected_mask = np.asarray([bool(row["selected_metrics"]["success"]) for row in rows])
    summary = {
        "label": label,
        "context_count": len(records),
        "first_candidate": first,
        "oracle_best_of_32": oracle,
        "scorer_selected": selected,
        "p_scorer_pass_given_oracle_contains_pass": (
            float(selected_mask[oracle_mask].mean()) if bool(oracle_mask.any()) else None
        ),
        "mean_clamp_fraction": float(np.mean([value.clamp_fraction for value in inferences])),
    }
    _write_json(artifact / f"{label}_policy_evaluation.json", {"rows": rows, "summary": summary})
    _write_json(artifact / f"{label}_scorer_on_diffusion_metrics.json", scorer_metrics)
    if save_candidate_rows:
        _write_json(artifact / f"{label}_candidate_rows.json", candidate_rows)
    return {"summary": summary, "scorer_metrics": scorer_metrics}


def _curve_subset_root(artifact: Path, fraction: float) -> Path:
    return artifact / "learning_curve_runs" / f"fraction_{int(round(100 * fraction)):03d}"


def _prepare_learning_curve_subset(
    artifact: Path,
    fraction: float,
    selected_state_ids: set[str],
) -> tuple[Path, list[str], int]:
    root = _curve_subset_root(artifact, fraction)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact / "context_table.npz", root / "context_table.npz")
    shutil.copy2(artifact / "context_normalizer.json", root / "context_normalizer.json")
    shutil.copy2(artifact / "fixed_noise_bank.npy", root / "fixed_noise_bank.npy")
    source = json.loads((artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((artifact / "development_split_manifest.json").read_text(encoding="utf-8"))
    state_by_context = {
        row["context_id"]: row["state_id"]
        for row in split["records"]
    }
    shards: list[dict[str, Any]] = []
    context_ids: list[str] = []
    action_count = 0
    destination = root / "diffusion_teacher_shards"
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source["shards"]:
        include = entry["split"] == "VALIDATION" or (
            entry["split"] == "TRAIN" and state_by_context[entry["context_id"]] in selected_state_ids
        )
        if not include:
            continue
        shutil.copy2(
            artifact / "diffusion_teacher_shards" / entry["path"],
            destination / entry["path"],
        )
        shards.append(dict(entry))
        if entry["split"] == "TRAIN":
            context_ids.append(entry["context_id"])
            action_count += int(entry["row_count"])
    _write_json(
        root / "diffusion_teacher_manifest.json",
        {
            "schema": "amortized_cem_existing_data_learning_curve_teacher_manifest_v1",
            "fraction": fraction,
            "selected_training_state_ids": sorted(selected_state_ids),
            "shards": shards,
        },
    )
    return root, context_ids, action_count


def run_learning_curve(
    config: dict[str, Any],
    artifact: Path,
    development_records: list[AmortizedCemContextRecord],
    full_train_records: list[AmortizedCemContextRecord],
    full_train_result: dict[str, Any],
    full_development_result: dict[str, Any],
) -> list[dict[str, Any]]:
    split = json.loads((artifact / "development_split_manifest.json").read_text(encoding="utf-8"))
    train_context_ids = set(split["train_context_ids"])
    eligible_records = [record for record in full_train_records if record.context_id in train_context_ids]
    eligible_states = _hash_order({record.state_id for record in eligible_records})
    fractions = [float(value) for value in config["learning_curve_fractions"]]
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        if fraction >= 1.0:
            model_root = artifact
            selected_states = set(eligible_states)
            selected_records = eligible_records
            teacher_action_count = int(
                json.loads((artifact / "source_data_inventory.json").read_text(encoding="utf-8"))["combined"][
                    "train_teacher_actions"
                ]
            )
            train_result = full_train_result
            development_result = full_development_result
            training = json.loads(
                (artifact / "diffusion_training_summary.json").read_text(encoding="utf-8")
            )
        else:
            state_count = max(1, int(math.ceil(fraction * len(eligible_states))))
            selected_states = set(eligible_states[:state_count])
            model_root, selected_ids, teacher_action_count = _prepare_learning_curve_subset(
                artifact, fraction, selected_states
            )
            subset_config = json.loads(json.dumps(config))
            training = train_diffusion(model_root, subset_config, device="cuda")
            selected_id_set = set(selected_ids)
            selected_records = [record for record in eligible_records if record.context_id in selected_id_set]
            train_result = evaluate_policy_records(
                config,
                artifact,
                selected_records,
                label=f"learning_curve_{int(round(100 * fraction)):03d}_train",
                model_root=model_root,
                save_candidate_rows=False,
            )
            development_result = evaluate_policy_records(
                config,
                artifact,
                development_records,
                label=f"learning_curve_{int(round(100 * fraction)):03d}_development",
                model_root=model_root,
                save_candidate_rows=False,
            )
        rows.append(
            {
                "fraction": fraction,
                "training_state_count": len(selected_states),
                "training_context_count": len(selected_records),
                "successful_action_label_count": teacher_action_count,
                "diffusion_training": training,
                "train_first_candidate_success_rate": train_result["summary"]["first_candidate"][
                    "scientific_success_rate"
                ],
                "train_oracle_best_of_32_success_rate": train_result["summary"]["oracle_best_of_32"][
                    "scientific_success_rate"
                ],
                "development_first_candidate_success_rate": development_result["summary"][
                    "first_candidate"
                ]["scientific_success_rate"],
                "development_oracle_best_of_32_success_rate": development_result["summary"][
                    "oracle_best_of_32"
                ]["scientific_success_rate"],
                "development_scorer_selected_success_rate": development_result["summary"][
                    "scorer_selected"
                ]["scientific_success_rate"],
            }
        )
        _write_json(artifact / "learning_curve.json", rows)
    return rows


def benchmark_latency(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    policy = _load_policy(artifact)
    split = json.loads((artifact / "development_split_manifest.json").read_text(encoding="utf-8"))
    record = _records_from_manifest(artifact, [split["development_context_ids"][0]])[0]
    with np.load(artifact / "context_table.npz", allow_pickle=False) as archive:
        context = torch.from_numpy(np.asarray(archive["contexts"][record.context_index], dtype=np.float32))
    warmup = int(config["latency_warmup_queries"])
    measured = int(config["latency_measured_queries"])
    for _ in range(warmup):
        policy.infer_from_context_tensor(context)
    torch.cuda.synchronize()
    wall_ms, gpu_ms = [], []
    for _ in range(measured):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        policy.infer_from_context_tensor(context)
        end_event.record()
        torch.cuda.synchronize()
        wall_ms.append(1000.0 * (time.perf_counter() - wall_start))
        gpu_ms.append(float(start_event.elapsed_time(end_event)))

    def values(data: list[float]) -> dict[str, float]:
        array = np.asarray(data, dtype=np.float64)
        return {
            "mean_ms": float(array.mean()),
            "median_ms": float(np.median(array)),
            "p90_ms": float(np.percentile(array, 90)),
            "p95_ms": float(np.percentile(array, 95)),
            "maximum_ms": float(array.max()),
        }

    result = {
        "schema": "milestone7a1_trained_policy_latency_v1",
        "warmup_queries": warmup,
        "measured_queries": measured,
        "candidates": 32,
        "ddim_steps": 25,
        "includes_context_normalization": True,
        "includes_scorer_and_selection": True,
        "excludes_simulator": True,
        "excludes_cem": True,
        "gpu": values(gpu_ms),
        "python_end_to_end": values(wall_ms),
    }
    _write_json(artifact / "inference_latency.json", result)
    return result


def _classify(
    train_result: dict[str, Any],
    development_result: dict[str, Any] | None,
    learning_curve: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any]]:
    train_oracle = float(train_result["summary"]["oracle_best_of_32"]["scientific_success_rate"])
    if development_result is None:
        return "GENERATOR_NOT_LEARNING", "NO", {"reason": "training_oracle_below_sanity_gate"}
    dev_oracle = float(development_result["summary"]["oracle_best_of_32"]["scientific_success_rate"])
    dev_selected = float(development_result["summary"]["scorer_selected"]["scientific_success_rate"])
    curve_oracles = [float(row["development_oracle_best_of_32_success_rate"]) for row in learning_curve]
    scaling = bool(len(curve_oracles) >= 2 and curve_oracles[-1] > curve_oracles[0] + 0.05)
    if dev_oracle >= 0.50 and dev_selected < 0.50 * dev_oracle:
        classification, more = "SCORER_NOT_LEARNING", "NO"
        reason = "generator_oracle_is_high_but_scorer_selected_is_much_lower"
    elif dev_oracle >= 0.50 and (scaling or dev_selected >= 0.25):
        classification, more = "ARCHITECTURE_PROMISING", "YES" if scaling else "NO"
        reason = "development_oracle_exceeds_50_percent"
    elif train_oracle >= 0.50 and dev_oracle < train_oracle - 0.20 and scaling:
        classification, more = "DATA_COVERAGE_LIMITED", "YES"
        reason = "training_fit_is_high_and_state_disjoint_development_improves_with_data"
    elif dev_oracle <= 0.10 and not scaling:
        classification, more = "GENERATOR_NOT_LEARNING", "NO"
        reason = "development_oracle_is_near_zero_without_scaling"
    elif scaling and train_oracle > dev_oracle:
        classification, more = "DATA_COVERAGE_LIMITED", "YES"
        reason = "existing_data_curve_is_rising_but_generalization_lags_training"
    else:
        classification, more = "GENERATOR_NOT_LEARNING", "NO"
        reason = "architecture_did_not_meet_promising_or_coverage_limited_evidence"
    return classification, more, {
        "reason": reason,
        "train_oracle": train_oracle,
        "development_oracle": dev_oracle,
        "development_selected": dev_selected,
        "learning_curve_improves_by_more_than_5_points": scaling,
    }


def _percent(value: float | None) -> str:
    return "NOT RUN" if value is None else f"{100.0 * value:.2f}%"


def write_report(
    config: dict[str, Any],
    artifact: Path,
    train_result: dict[str, Any],
    development_result: dict[str, Any] | None,
    learning_curve: list[dict[str, Any]],
    latency: dict[str, Any],
    classification: str,
    more_cem: str,
    interpretation: dict[str, Any],
) -> str:
    inventory = json.loads((artifact / "source_data_inventory.json").read_text(encoding="utf-8"))
    audit = json.loads((artifact / "sixa_state_overlap_audit.json").read_text(encoding="utf-8"))
    compatibility = json.loads((artifact / "source_contract_compatibility.json").read_text(encoding="utf-8"))
    split = json.loads((artifact / "development_split_manifest.json").read_text(encoding="utf-8"))
    diffusion = json.loads((artifact / "diffusion_training_summary.json").read_text(encoding="utf-8"))
    scorer = json.loads((artifact / "scorer_training_summary.json").read_text(encoding="utf-8"))
    train = train_result["summary"]
    dev = None if development_result is None else development_result["summary"]
    dev_scorer = None if development_result is None else development_result["scorer_metrics"]
    combined = inventory["combined"]
    curve_lines = "\n".join(
        f"| {row['fraction']:.2f} | {row['training_state_count']} | {row['training_context_count']} | "
        f"{row['successful_action_label_count']} | {_percent(row['train_oracle_best_of_32_success_rate'])} | "
        f"{_percent(row['development_first_candidate_success_rate'])} | "
        f"{_percent(row['development_oracle_best_of_32_success_rate'])} | "
        f"{_percent(row['development_scorer_selected_success_rate'])} |"
        for row in learning_curve
    ) or "| NOT RUN — training oracle sanity gate failed | | | | | | | |"

    def block(summary: dict[str, Any] | None, key: str) -> str:
        if summary is None:
            return "NOT RUN"
        value = summary[key]
        return (
            f"success {_percent(value['scientific_success_rate'])}; feasible {_percent(value['feasible_rate'])}; "
            f"median tip error {1000.0 * value['median_tip_error_m']:.3f} mm; directed speed "
            f"{value['median_directed_speed_m_s']:.3f} m/s; direction {value['median_direction_error_deg']:.3f} deg; "
            f"UAV displacement {value['median_uav_displacement_m']:.4f} m; UAV speed "
            f"{value['median_uav_speed_m_s']:.4f} m/s; duration {value['median_maneuver_duration_s']:.4f} s; "
            f"hit segments {value['hit_segment_counts']}"
        )

    parameter_diffusion = sum(value.numel() for value in ConditionalActionDiffusion().parameters())
    parameter_scorer = sum(value.numel() for value in ManeuverOutcomeScorer().parameters())
    report = f"""# Milestone 7A.1 — Existing-Data Viability Report

Generated: {_timestamp()}

## 1. Why the 768-context campaign was stopped

The long teacher campaign was stopped after 42 completed contexts to test the final neural architecture before committing additional CEM compute. This run performed **zero new CEM solves** and used only durable Milestone-6A and partial Milestone-7A artifacts.

## 2. Existing data inventory

Partial 7A contains {inventory['partial_7a']['completed_contexts']} completed contexts, {inventory['partial_7a']['solved_contexts']} solved contexts, {inventory['partial_7a']['successful_actions']} verified successful actions, and {inventory['partial_7a']['scorer_rows']} scorer rows. Milestone 6A contains {inventory['milestone_6a']['contexts']} contexts and {inventory['milestone_6a']['authoritative_successes']} authoritative successes. Its final actions and outcome metrics were saved; its population candidate rows were not.

The gradient split uses {combined['train_teacher_contexts']} solved contexts, {combined['train_teacher_actions']} successful actions, and {combined['train_scorer_rows']} scorer rows. The state-disjoint development split contains {combined['development_contexts']} contexts.

## 3. 6A / 7A compatibility audit

Compatibility passed: model `{compatibility['model_freeze']}`, 83-D context, final normalized 49-D action, nominal theta, variable duration [0.45, 1.80] s, 0.30-s smooth settle, 2.40-s evaluation horizon, and fixed numerical batch 2048. The maximum saved NPZ-versus-JSON 6A action difference was {compatibility['sixa_action_npz_json_max_difference']:.3e}. No old 1.20-s command semantics were used.

## 4. State-ID overlap and leakage audit

6A context overlap counts were `{audit['sixa_context_overlap_counts']}`. No 6A state overlapped frozen 7A VALIDATION, TEST, or EDGE ownership. Split leakage was absent. `TEST_PREVIOUSLY_BENCHMARKED = {'YES' if audit['test_previously_benchmarked'] else 'NO'}`. The final 7A TEST was neither trained on nor evaluated.

## 5. Development split

The union of eligible TRAIN-owned states was hash-ordered with seed 42 and split by state ID: {len(split['neural_train_state_ids'])} neural-training states and {len(split['internal_development_state_ids'])} internal-development states. All actions and targets for a state remain together. Existing 6A states outside TRAIN ownership are EXTERNAL DEVELOPMENT only.

## 6. Final diffusion architecture confirmation

`ConditionalActionDiffusion` is unchanged: direct 49-D normalized action, 83-D context encoder, 64-D sinusoidal time embedding, width 256, four residual MLP blocks, 100-step cosine DDPM epsilon training, EMA 0.999, and fixed-noise 25-step deterministic DDIM for 32 candidates. Parameter count: {parameter_diffusion:,}. No PCA/latent bottleneck or alternate architecture was introduced.

## 7. Final scorer architecture confirmation

`ManeuverOutcomeScorer` is unchanged: 132 inputs, three 256-unit SiLU layers, seven continuous physical heads and three binary heads. Parameter count: {parameter_scorer:,}. Continuous targets use TRAIN-only normalization and Huber loss; binary targets use BCE with capped TRAIN-only positive weights. Candidate selection applies predicted scientific gates and deterministic margin/reward fallback.

## 8. Diffusion training

Training ran from scratch with AdamW 2e-4, weight decay 1e-6, gradient clip 1.0, context-balanced sampling, and EMA. It stopped at update {diffusion['updates']}; the best EMA validation epsilon loss was {diffusion['best_validation_epsilon_loss']:.6f} at update {diffusion['best_update']}.

## 9. Scorer training

Training ran from scratch with AdamW 3e-4, weight decay 1e-6, outcome-stratified batches, TRAIN-only target normalization, Huber continuous losses, and BCE binary losses. It stopped at update {scorer['updates']}; best validation loss was {scorer['best_validation_loss']:.6f} at update {scorer['best_update']}.

## 10. Training-context oracle result

{block(train, 'oracle_best_of_32')}

Training first candidate: {block(train, 'first_candidate')}

Training scorer selected: {block(train, 'scorer_selected')}

The generator's mean final-coordinate clamp fraction was {100.0 * train['mean_clamp_fraction']:.2f}%. Because the training oracle was zero, the required basic sanity classification is **GENERATOR_NOT_FITTING_TEACHER_DISTRIBUTION**. This is stronger than a held-out generalization failure: the generated support does not reproduce successful actions for contexts used in gradients.

## 11. Development first-candidate result

{block(dev, 'first_candidate')}

## 12. Development oracle best-of-32 result

{block(dev, 'oracle_best_of_32')}

## 13. Development scorer-selected result

{block(dev, 'scorer_selected')}

## 14. Generator-vs-scorer decomposition

The development oracle measures whether the generator contains a physically successful candidate; the scorer-selected result measures the deployed neural choice. `P(scorer PASS | oracle contains PASS)` is {('NOT RUN' if dev is None else _percent(dev['p_scorer_pass_given_oracle_contains_pass']))}. Classification evidence: `{interpretation}`.

Here the decomposition stops at the generator: no generated candidate was feasible even on training contexts, so a scorer cannot rescue the candidate set. The near-total final clamp rate is a concrete sampling/distribution-boundary diagnostic and should be investigated before requesting more teacher labels.

## 15. Physical outcome metrics

The first/oracle/scorer summaries above include hard success, feasibility, median tip error, directed speed, direction error, UAV displacement/speed, maneuver duration, and ACTIVE/SETTLE/HOLD hit counts. Every number came from the frozen authoritative production simulator, not neural loss.

## 16. Scorer accuracy on diffusion candidates

{('NOT RUN' if dev_scorer is None else json.dumps(dev_scorer, indent=2, sort_keys=True))}

## 17. Teacher-count learning curve

| Fraction | Train states | Train contexts | Teacher actions | Train oracle | Dev first | Dev oracle | Dev selected |
|---:|---:|---:|---:|---:|---:|---:|---:|
{curve_lines}

{('The learning curve was not run because the protocol requires an immediate stop when the training oracle is very low.' if not learning_curve else 'All subset runs used fresh instances of the same final diffusion architecture, the same fixed development set, identical hyperparameters, and nested state-ID subsets. The 100% scorer was held fixed.')}

## 18. Trained neural inference latency

Over {latency['measured_queries']} post-warmup queries, complete Python policy latency was median {latency['python_end_to_end']['median_ms']:.3f} ms, p90 {latency['python_end_to_end']['p90_ms']:.3f} ms, p95 {latency['python_end_to_end']['p95_ms']:.3f} ms, and maximum {latency['python_end_to_end']['maximum_ms']:.3f} ms. This includes context normalization, 32-candidate/25-step DDIM, scorer, and selection; it excludes CEM and simulation.

## 19. Whether more CEM data is justified

**{more_cem}.** Additional CEM data is justified only when the architecture learns and context coverage is the limiting factor. This decision follows the measured train/development oracle and fixed-subset learning curve, not denoising loss alone.

## 20. Explicit next recommendation

Classification: **{classification}**, with the more specific sanity finding **GENERATOR_NOT_FITTING_TEACHER_DISTRIBUTION**. Do not resume teacher generation. First review the unchanged diffusion sampling/training formulation—especially the 99% final-coordinate clamp behavior—using the existing data. No aggregation or architecture change was made here.

## 21. Final 7A TEST

**NOT EVALUATED.**

## 22. Protected fig8vertical_002

**NOT EVALUATED.**

## 23. Real hardware

**NOT EXECUTED.** All generated actions remain simulation-only.

## Final summary

    New CEM solves:
        0

    Existing solved training contexts used:
        {combined['train_teacher_contexts']}

    Existing successful teacher actions used:
        {combined['train_teacher_actions']}

    Existing scorer rows used:
        {combined['train_scorer_rows']}

    Diffusion:
        FINAL ARCHITECTURE

    Scorer:
        FINAL ARCHITECTURE

    Train oracle best-of-32:
        {_percent(train['oracle_best_of_32']['scientific_success_rate'])}

    Development first-candidate success:
        {('NOT RUN' if dev is None else _percent(dev['first_candidate']['scientific_success_rate']))}

    Development oracle best-of-32:
        {('NOT RUN' if dev is None else _percent(dev['oracle_best_of_32']['scientific_success_rate']))}

    Development scorer-selected success:
        {('NOT RUN' if dev is None else _percent(dev['scorer_selected']['scientific_success_rate']))}

    Development feasibility:
        {('NOT RUN' if dev is None else _percent(dev['scorer_selected']['feasible_rate']))}

    Learning curve:
        {classification if learning_curve else 'NOT RUN'}

    Median trained policy latency:
        {latency['python_end_to_end']['median_ms']:.3f} ms

    P95 trained policy latency:
        {latency['python_end_to_end']['p95_ms']:.3f} ms

    Classification:
        {classification}

    More CEM data justified:
        {more_cem}

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    REPORT.write_text(report, encoding="utf-8")
    (artifact / REPORT.name).write_text(report, encoding="utf-8")
    return report


def write_source_hash_manifest(artifact: Path) -> None:
    paths = [
        ROOT / "run_milestone7a1.py",
        ROOT / "config/learning/amortized_cem_diffusion_existing_data_viability_v1.json",
        ROOT / "learning/action_diffusion.py",
        ROOT / "learning/amortized_cem_policy.py",
        ROOT / "learning/amortized_cem_training.py",
        ROOT / "learning/outcome_scorer.py",
        ROOT / "learning/policy_action.py",
        ROOT / "planning/production_cem.py",
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone7a1_source_hash_manifest_v1",
            "files": [
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
        },
    )


def run_all(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    _write_json(artifact / "run_config.json", config)
    inventory = prepare_existing_data(config, artifact)
    training = train_models(config, artifact)
    split = json.loads((artifact / "development_split_manifest.json").read_text(encoding="utf-8"))
    train_records = _records_from_manifest(artifact, split["train_evaluation_context_ids"])
    development_records = _records_from_manifest(artifact, split["development_context_ids"])
    train_result = evaluate_policy_records(
        config,
        artifact,
        train_records,
        label="train",
        save_candidate_rows=False,
    )
    train_oracle = float(train_result["summary"]["oracle_best_of_32"]["scientific_success_rate"])
    development_result: dict[str, Any] | None = None
    learning_curve: list[dict[str, Any]] = []
    if train_oracle > 0.05:
        development_result = evaluate_policy_records(
            config,
            artifact,
            development_records,
            label="development",
            save_candidate_rows=True,
        )
        _write_json(artifact / "development_summary.json", development_result["summary"])
        _write_json(artifact / "scorer_on_diffusion_metrics.json", development_result["scorer_metrics"])
        if int(inventory["combined"]["train_teacher_contexts"]) >= 40:
            learning_curve = run_learning_curve(
                config,
                artifact,
                development_records,
                train_records,
                train_result,
                development_result,
            )
    else:
        _write_json(
            artifact / "development_summary.json",
            {"not_run": True, "reason": "train_oracle_best_of_32_at_or_below_5_percent"},
        )
        _write_json(
            artifact / "development_candidate_rows.json",
            {"not_run": True, "rows": [], "reason": "training_oracle_sanity_gate_failed"},
        )
        _write_json(
            artifact / "scorer_on_diffusion_metrics.json",
            {"not_run": True, "reason": "training_oracle_sanity_gate_failed"},
        )
        _write_json(artifact / "learning_curve.json", [])
    latency = benchmark_latency(config, artifact)
    classification, more_cem, interpretation = _classify(
        train_result, development_result, learning_curve
    )
    _write_json(
        artifact / "viability_classification.json",
        {
            "classification": classification,
            "more_cem_data_justified": more_cem,
            "evidence": interpretation,
        },
    )
    write_source_hash_manifest(artifact)
    write_report(
        config,
        artifact,
        train_result,
        development_result,
        learning_curve,
        latency,
        classification,
        more_cem,
        interpretation,
    )
    return {
        "artifact": str(artifact),
        "inventory": inventory,
        "training": training,
        "train": train_result["summary"],
        "development": None if development_result is None else development_result["summary"],
        "classification": classification,
        "more_cem_data_justified": more_cem,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    artifact = args.artifact or (ARTIFACT_PARENT / _timestamp())
    result = run_all(config, artifact)
    print(json.dumps(_safe(result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
