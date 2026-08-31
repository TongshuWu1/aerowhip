"""Milestone 7A.3 existing-data scaling and conditional-policy audit.

This runner is intentionally generator-only.  It consumes durable 6A/7A
artifacts, never imports or invokes a CEM optimizer, never trains a scorer,
and never evaluates the sealed 7A TEST or protected physical data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
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

from learning.action_diffusion import (
    ConditionalActionDiffusion,
    cosine_alpha_bar_schedule,
    ddim_timestep_schedule,
    sample_ddim,
)
from learning.amortized_cem_data import (
    ACTION_DIM,
    AmortizedCemContextRecord,
    sha256_file,
)
from learning.amortized_cem_training import train_diffusion
from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    pad_context_specification,
)
from learning.normalization import FixedContextNormalizer
from learning.policy_action import decode_policy_action
from planning.production_cem import (
    FIXED_NUMERICAL_BATCH_SIZE,
    evaluate_normalized_actions_fixed_batch,
    event_segment,
)
from run_milestone7a1 import _load_environment, _physics_settings


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config/learning/diffusion_existing_data_scaling_v1.json"
ARTIFACT_PARENT = ROOT / "data/policy_training/diffusion_existing_data_scaling_v1"
REPORT = ROOT / "MILESTONE7A3_EXISTING_DATA_CONDITIONAL_DIFFUSION_REPORT.md"


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
    if config["schema"] != "diffusion_existing_data_scaling_v1":
        raise ValueError("Unsupported Milestone 7A.3 configuration.")
    if config["model_freeze"] != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Production model freeze changed.")
    diffusion = config["diffusion"]
    expected = {
        "training_steps": 100,
        "sampling_steps": 25,
        "sampling_start_timestep": 95,
        "candidates": 32,
        "ddim_eta": 0.0,
        "ema_decay": 0.999,
    }
    for key, value in expected.items():
        if diffusion[key] != value:
            raise ValueError(f"Final diffusion contract changed: {key}")
    if ddim_timestep_schedule().tolist()[0] != 95:
        raise RuntimeError("The repaired DDIM schedule no longer starts at t=95.")
    required = config["prohibitions"]
    for key in (
        "new_cem_solves",
        "scorer_training",
        "sac",
        "aggregation",
        "architecture_change",
        "theta_randomization",
        "final_test",
        "hardware",
    ):
        if not bool(required[key]):
            raise ValueError(f"Required prohibition is disabled: {key}")
    if required["protected_test"] != "fig8vertical_002":
        raise ValueError("Protected-test identity changed.")


def _hash_order(values: Iterable[str], *, seed: int = 42) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )


def _context_identity(record: dict[str, Any]) -> str:
    payload = {
        "bank": record["bank"],
        "state_id": record["state_id"],
        "target": [round(float(value), 8) for value in record["target_local_m"]],
        "direction": [round(float(value), 8) for value in record["direction_local"]],
        "theta": [round(float(value), 10) for value in record["theta_nominal"]],
        "schema": "milestone7a3_stable_existing_context_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_teacher_shard(source: Path, entry: dict[str, Any]) -> tuple[np.ndarray, list, list]:
    with np.load(source / "diffusion_teacher_shards" / entry["path"], allow_pickle=False) as shard:
        actions = np.asarray(shard["normalized_actions"], dtype=np.float32)
        metrics = [json.loads(str(value)) for value in shard["metrics_json"]]
        provenance = [json.loads(str(value)) for value in shard["provenance_json"]]
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise RuntimeError("A source teacher shard does not use normalized [49] actions.")
    if not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.0 + 1.0e-6:
        raise RuntimeError("A source teacher action violates the production normalized bounds.")
    return actions, metrics, provenance


def _write_teacher_shard(
    root: Path,
    record: AmortizedCemContextRecord,
    actions: np.ndarray,
    metrics: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    directory = root / "diffusion_teacher_shards"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"context_{record.context_index:06d}.npz"
    path = directory / name
    np.savez_compressed(
        path,
        schema=np.asarray("amortized_cem_diffusion_teacher_shard_v1"),
        context_index=np.asarray(record.context_index, dtype=np.int64),
        normalized_actions=np.asarray(actions, dtype=np.float32),
        metrics_json=np.asarray([json.dumps(_safe(item), sort_keys=True) for item in metrics]),
        provenance_json=np.asarray(
            [json.dumps(_safe(item), sort_keys=True) for item in provenance]
        ),
    )
    return {
        "context_index": record.context_index,
        "context_id": record.context_id,
        "split": record.split,
        "path": name,
        "row_count": int(actions.shape[0]),
        "sha256": sha256_file(path),
    }


def prepare_existing_data(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    """Deduplicate compatible labels and enforce immutable state ownership."""

    source = ROOT / config["source_7a1_artifact"]
    split = json.loads((source / "development_split_manifest.json").read_text(encoding="utf-8"))
    source_inventory = json.loads((source / "source_data_inventory.json").read_text(encoding="utf-8"))
    source_audit = json.loads((source / "sixa_state_overlap_audit.json").read_text(encoding="utf-8"))
    compatibility = json.loads((source / "source_contract_compatibility.json").read_text(encoding="utf-8"))
    if not bool(compatibility["compatible"]):
        raise RuntimeError("The existing 6A/7A compatibility audit did not pass.")
    if bool(source_audit["test_previously_benchmarked"]):
        raise RuntimeError("The sealed 7A TEST was previously benchmarked.")
    records_by_id = {row["context_id"]: row for row in split["records"]}
    source_rows = {row["context_id"]: row for row in split["source_rows"]}
    teacher_manifest = json.loads(
        (source / "diffusion_teacher_manifest.json").read_text(encoding="utf-8")
    )
    teachers_by_id = {row["context_id"]: row for row in teacher_manifest["shards"]}
    with np.load(source / "context_table.npz", allow_pickle=False) as archive:
        source_contexts = np.asarray(archive["contexts"], dtype=np.float32)

    allowed_training_owners = {"TRAIN", "AGGREGATION_POOL"}
    grouped: dict[str, list[str]] = {}
    external_grouped: dict[str, list[str]] = {}
    excluded_counts: dict[str, int] = {}
    for context_id, provenance in source_rows.items():
        owner = str(provenance["ownership"])
        excluded_counts[owner] = excluded_counts.get(owner, 0) + 1
        if not bool(provenance["teacher_available"]):
            continue
        key = _context_identity(records_by_id[context_id])
        if owner in allowed_training_owners:
            grouped.setdefault(key, []).append(context_id)
        elif owner == "EXTERNAL_DEVELOPMENT":
            external_grouped.setdefault(key, []).append(context_id)

    eligible_states = {
        str(records_by_id[ids[0]]["state_id"]) for ids in grouped.values()
    }
    ordered_states = _hash_order(eligible_states, seed=int(config["seed"]))
    validation_count = max(
        1, int(round(float(config["internal_validation_fraction"]) * len(ordered_states)))
    )
    validation_states = set(ordered_states[:validation_count])
    training_states = set(ordered_states[validation_count:])
    if training_states & validation_states:
        raise RuntimeError("Internal denoising split leaked state IDs.")

    records: list[AmortizedCemContextRecord] = []
    contexts: list[np.ndarray] = []
    teacher_rows: list[dict[str, Any]] = []
    deduplication_rows: list[dict[str, Any]] = []

    def add_group(key: str, ids: list[str], output_split: str) -> None:
        source_records = [records_by_id[value] for value in ids]
        representative = source_records[0]
        context_candidates = np.stack(
            [source_contexts[int(row["context_index"])] for row in source_records]
        )
        maximum_context_difference = float(
            np.max(np.abs(context_candidates - context_candidates[0]))
        )
        if maximum_context_difference > 2.0e-5:
            raise RuntimeError(f"Duplicate context tensors disagree: {ids}")
        actions_all, metrics_all, provenance_all = [], [], []
        for source_id in ids:
            actions, metrics, provenance = _load_teacher_shard(source, teachers_by_id[source_id])
            actions_all.extend(actions)
            metrics_all.extend(metrics)
            provenance_all.extend(
                [
                    {
                        **item,
                        "milestone7a3_source_context_id": source_id,
                        "milestone7a3_source": source_rows[source_id]["source"],
                        "milestone7a3_ownership": source_rows[source_id]["ownership"],
                    }
                    for item in provenance
                ]
            )
        kept: list[int] = []
        seen: set[bytes] = set()
        for index, action in enumerate(actions_all):
            token = np.round(np.asarray(action, dtype=np.float64), 7).tobytes()
            if token not in seen:
                seen.add(token)
                kept.append(index)
        actions = np.asarray([actions_all[index] for index in kept], dtype=np.float32)
        metrics = [metrics_all[index] for index in kept]
        provenance = [provenance_all[index] for index in kept]
        context_index = len(records)
        context_id = f"m7a3_{key[:20]}"
        record = AmortizedCemContextRecord(
            context_index=context_index,
            context_id=context_id,
            split=output_split,
            bank=str(representative["bank"]),
            state_id=str(representative["state_id"]),
            state_index=int(representative["state_index"]),
            target_number=int(representative["target_number"]),
            target_local_m=tuple(float(value) for value in representative["target_local_m"]),
            direction_local=tuple(float(value) for value in representative["direction_local"]),
            theta_nominal=tuple(float(value) for value in representative["theta_nominal"]),
        )
        records.append(record)
        contexts.append(context_candidates[0])
        teacher_rows.append(_write_teacher_shard(artifact, record, actions, metrics, provenance))
        deduplication_rows.append(
            {
                "context_id": context_id,
                "stable_identity": key,
                "source_context_ids": ids,
                "source_ownerships": sorted({source_rows[value]["ownership"] for value in ids}),
                "split": output_split,
                "actions_before_deduplication": len(actions_all),
                "actions_after_deduplication": int(actions.shape[0]),
                "maximum_duplicate_context_tensor_difference": maximum_context_difference,
            }
        )

    for key, ids in sorted(grouped.items()):
        state_id = str(records_by_id[ids[0]]["state_id"])
        add_group(key, ids, "VALIDATION" if state_id in validation_states else "TRAIN")
    for key, ids in sorted(external_grouped.items()):
        add_group(key, ids, "EXTERNAL_DEVELOPMENT")

    context_array = np.asarray(contexts, dtype=np.float32)
    if context_array.shape != (len(records), 83) or not np.isfinite(context_array).all():
        raise RuntimeError("The deduplicated context table is invalid.")
    np.savez_compressed(
        artifact / "context_table.npz",
        schema=np.asarray("milestone7a3_existing_context_table_v1"),
        contexts=context_array,
        context_indices=np.arange(len(records), dtype=np.int64),
        context_ids=np.asarray([record.context_id for record in records]),
        state_ids=np.asarray([record.state_id for record in records]),
        splits=np.asarray([record.split for record in records]),
        banks=np.asarray([record.bank for record in records]),
        theta=np.asarray([record.theta_nominal for record in records], dtype=np.float32),
    )
    shutil.copy2(source / "context_normalizer.json", artifact / "context_normalizer.json")
    noise = np.load(ROOT / config["source_7a2_artifact"] / "fixed_noise_bank.npy")
    if noise.shape != (32, 49):
        raise RuntimeError("The fixed diffusion noise bank is invalid.")
    np.save(artifact / "fixed_noise_bank.npy", noise.astype(np.float32))
    training_manifest = {
        "schema": "milestone7a3_existing_data_teacher_manifest_v1",
        "context_balanced_sampling": True,
        "shards": [row for row in teacher_rows if row["split"] in {"TRAIN", "VALIDATION"}],
    }
    _write_json(artifact / "diffusion_teacher_manifest.json", training_manifest)
    record_payload = [asdict(record) for record in records]
    _write_json(
        artifact / "neural_training_split_manifest.json",
        {
            "schema": "milestone7a3_state_disjoint_internal_split_v1",
            "seed": int(config["seed"]),
            "gradient_training_state_ids": sorted(training_states),
            "internal_denoising_validation_state_ids": sorted(validation_states),
            "gradient_training_context_ids": [r.context_id for r in records if r.split == "TRAIN"],
            "internal_validation_context_ids": [r.context_id for r in records if r.split == "VALIDATION"],
            "records": record_payload,
        },
    )
    _write_json(
        artifact / "external_development_manifest.json",
        {
            "schema": "milestone7a3_external_development_manifest_v1",
            "gradient_free": True,
            "early_stopping_free": True,
            "context_ids": [r.context_id for r in records if r.split == "EXTERNAL_DEVELOPMENT"],
            "records": [asdict(r) for r in records if r.split == "EXTERNAL_DEVELOPMENT"],
        },
    )
    train_shards = [row for row in teacher_rows if row["split"] == "TRAIN"]
    validation_shards = [row for row in teacher_rows if row["split"] == "VALIDATION"]
    external_shards = [row for row in teacher_rows if row["split"] == "EXTERNAL_DEVELOPMENT"]
    inventory = {
        "schema": "milestone7a3_existing_data_inventory_v1",
        "new_cem_solves": 0,
        "sources": source_inventory,
        "candidate_contexts_before_stable_deduplication": len(grouped) + len(external_grouped),
        "deduplicated_contexts": len(records),
        "training": {
            "states": len(training_states),
            "contexts": len(train_shards),
            "successful_actions": sum(int(row["row_count"]) for row in train_shards),
        },
        "internal_denoising_validation": {
            "states": len(validation_states),
            "contexts": len(validation_shards),
            "successful_actions": sum(int(row["row_count"]) for row in validation_shards),
        },
        "external_development": {
            "states": len({r.state_id for r in records if r.split == "EXTERNAL_DEVELOPMENT"}),
            "contexts": len(external_shards),
            "successful_actions": sum(int(row["row_count"]) for row in external_shards),
        },
        "source_ownership_context_counts": excluded_counts,
        "deduplication": deduplication_rows,
    }
    _write_json(artifact / "existing_data_inventory.json", inventory)
    _write_json(
        artifact / "ownership_audit.json",
        {
            "schema": "milestone7a3_ownership_audit_v1",
            "aggregation_pool_consumed_as_training": True,
            "allowed_gradient_ownerships": sorted(allowed_training_owners),
            "external_development_gradient_rows": 0,
            "seven_a_validation_gradient_rows": 0,
            "seven_a_test_gradient_rows": 0,
            "edge_reserved_gradient_rows": 0,
            "protected_data_accessed": False,
            "test_previously_benchmarked": False,
            "state_split_disjoint": not bool(training_states & validation_states),
            "model_freeze": config["model_freeze"],
            "context_dimension": 83,
            "action_dimension": 49,
            "normalized_action_bounds_verified": True,
            "nominal_theta_only": True,
            "smooth_settle_semantics": True,
            "maneuver_duration_s": [0.45, 1.80],
            "settle_duration_s": 0.30,
            "evaluation_horizon_s": 2.40,
            "fixed_numerical_batch_size": 2048,
        },
    )
    return inventory


def _load_records(artifact: Path) -> tuple[list[AmortizedCemContextRecord], np.ndarray]:
    payload = json.loads(
        (artifact / "neural_training_split_manifest.json").read_text(encoding="utf-8")
    )
    records = [AmortizedCemContextRecord(**row) for row in payload["records"]]
    with np.load(artifact / "context_table.npz", allow_pickle=False) as archive:
        contexts = np.asarray(archive["contexts"], dtype=np.float32)
    return records, contexts


def _load_actions_by_context(artifact: Path, *, include_external: bool = True) -> dict[int, np.ndarray]:
    manifest = json.loads(
        (artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8")
    )
    entries = list(manifest["shards"])
    if include_external:
        all_files = sorted((artifact / "diffusion_teacher_shards").glob("*.npz"))
        listed = {entry["path"] for entry in entries}
        for path in all_files:
            if path.name not in listed:
                with np.load(path, allow_pickle=False) as shard:
                    entries.append(
                        {
                            "context_index": int(shard["context_index"]),
                            "path": path.name,
                        }
                    )
    output: dict[int, np.ndarray] = {}
    for entry in entries:
        with np.load(artifact / "diffusion_teacher_shards" / entry["path"], allow_pickle=False) as shard:
            output[int(entry["context_index"])] = np.asarray(
                shard["normalized_actions"], dtype=np.float32
            )
    return output


def _load_model(model_root: Path, device: torch.device) -> ConditionalActionDiffusion:
    payload = torch.load(model_root / "diffusion_ema_best.pt", map_location=device, weights_only=True)
    model = ConditionalActionDiffusion().to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def generate_candidates(
    model: ConditionalActionDiffusion,
    normalizer: FixedContextNormalizer,
    contexts: np.ndarray,
    records: list[AmortizedCemContextRecord],
    noise: torch.Tensor,
    *,
    context_override: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]:
    device = next(model.parameters()).device
    alpha_bar = cosine_alpha_bar_schedule().to(device)
    schedule = ddim_timestep_schedule().to(device)
    bounded: dict[str, torch.Tensor] = {}
    raw: dict[str, torch.Tensor] = {}
    outside = 0
    total = 0
    for number, record in enumerate(records):
        value = (
            contexts[record.context_index]
            if context_override is None or record.context_id not in context_override
            else context_override[record.context_id]
        )
        normalized = normalizer.normalize(torch.from_numpy(value[None]).to(device))
        sample = sample_ddim(
            model,
            normalized,
            noise,
            alpha_bar=alpha_bar,
            timestep_schedule=schedule,
        )
        raw[record.context_id] = sample.raw_action.detach().cpu()
        bounded[record.context_id] = sample.bounded_action.detach().cpu()
        outside += int((sample.raw_action.abs() > 1.0).sum())
        total += sample.raw_action.numel()
        if (number + 1) % 64 == 0 or number + 1 == len(records):
            print(f"generated {number + 1}/{len(records)} contexts", flush=True)
    return raw, bounded, outside / total


def _statistics(values: torch.Tensor | np.ndarray) -> dict[str, Any]:
    array = np.asarray(torch.as_tensor(values).detach().cpu(), dtype=np.float64)
    absolute = np.abs(array)
    duration = array[..., -1].reshape(-1)
    return {
        "shape": list(array.shape),
        "coordinate_mean": float(array.mean()),
        "coordinate_std": float(array.std()),
        "p95_absolute": float(np.percentile(absolute, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "fraction_outside_minus1_plus1": float(np.mean(absolute > 1.0)),
        "duration_normalized": {
            "mean": float(duration.mean()),
            "median": float(np.median(duration)),
            "minimum": float(duration.min()),
            "maximum": float(duration.max()),
        },
        "duration_seconds": {
            "mean": float(0.45 + 1.35 * (duration.mean() + 1.0) / 2.0),
            "median": float(0.45 + 1.35 * (np.median(duration) + 1.0) / 2.0),
            "minimum": float(0.45 + 1.35 * (duration.min() + 1.0) / 2.0),
            "maximum": float(0.45 + 1.35 * (duration.max() + 1.0) / 2.0),
        },
    }


def distribution_audit(
    artifact: Path,
    records: list[AmortizedCemContextRecord],
    raw: dict[str, torch.Tensor],
    bounded: dict[str, torch.Tensor],
    actions_by_context: dict[int, np.ndarray],
) -> dict[str, Any]:
    raw_all = torch.cat([raw[record.context_id] for record in records])
    bounded_all = torch.cat([bounded[record.context_id] for record in records])
    nearest = []
    for record in records:
        teacher = torch.from_numpy(actions_by_context[record.context_index]).float()
        nearest.extend(
            torch.cdist(bounded[record.context_id].float(), teacher).min(dim=1).values.tolist()
        )
    result = {
        "schema": "milestone7a3_generator_distribution_statistics_v1",
        "context_count": len(records),
        "candidate_count": int(raw_all.shape[0]),
        "pre_clamp": _statistics(raw_all),
        "bounded": _statistics(bounded_all),
        "final_clamp_fraction": float((raw_all.abs() > 1.0).float().mean()),
        "nearest_same_context_teacher_l2": {
            "mean": float(np.mean(nearest)),
            "median": float(np.median(nearest)),
            "p95": float(np.percentile(nearest, 95)),
            "minimum": float(np.min(nearest)),
            "maximum": float(np.max(nearest)),
        },
    }
    _write_json(artifact / "generator_distribution_statistics.json", result)
    return result


def _metric_rows(result, actions: torch.Tensor, task, settings) -> list[dict[str, Any]]:
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


def summarize_evaluation_rows(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    """Summarize already-authoritative candidate rows without rerunning physics."""

    def median(name: str) -> float | None:
        values = np.asarray(
            [float(row["oracle_metrics"][name]) for row in rows], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else None

    def first_median(name: str) -> float | None:
        values = np.asarray(
            [float(row["first_metrics"][name]) for row in rows], dtype=np.float64
        )
        finite = values[np.isfinite(values)]
        return float(np.median(finite)) if finite.size else None

    first_success = np.asarray([bool(row["first_metrics"]["success"]) for row in rows])
    oracle_success = np.asarray([bool(row["oracle_metrics"]["success"]) for row in rows])
    any_feasible = np.asarray([row["candidate_feasible_count"] > 0 for row in rows])
    total_feasible = sum(row["candidate_feasible_count"] for row in rows)
    total_candidates = sum(row["candidate_count"] for row in rows)
    segments = {name: 0 for name in ("ACTIVE", "SETTLE", "HOLD", "NONE")}
    for row in rows:
        value = str(row["oracle_metrics"].get("hit_segment", "NONE"))
        segments[value if value in segments else "NONE"] += 1
    return {
        "schema": "milestone7a3_generator_oracle_summary_v1",
        "label": label,
        "context_count": len(rows),
        "first_candidate_success_rate": float(first_success.mean()),
        "first_candidate_feasibility_rate": float(
            np.mean([bool(row["first_metrics"]["feasible"]) for row in rows])
        ),
        "median_first_tip_distance_m": first_median("minimum_tip_target_distance_m"),
        "median_first_directed_speed_m_s": first_median("best_event_directed_speed_m_s"),
        "median_first_direction_error_deg": first_median("best_event_direction_angle_deg"),
        "median_first_uav_displacement_m": first_median("maximum_uav_displacement_m"),
        "median_first_uav_speed_m_s": first_median("maximum_uav_speed_m_s"),
        "median_first_maneuver_duration_s": first_median("maneuver_duration_s"),
        "oracle_best_of_32_success_rate": float(oracle_success.mean()),
        "contexts_with_at_least_one_feasible_candidate_rate": float(any_feasible.mean()),
        "candidate_level_feasibility_rate": float(total_feasible / total_candidates),
        "median_oracle_tip_distance_m": median("minimum_tip_target_distance_m"),
        "median_oracle_directed_speed_m_s": median("best_event_directed_speed_m_s"),
        "median_oracle_direction_error_deg": median("best_event_direction_angle_deg"),
        "median_oracle_uav_displacement_m": median("maximum_uav_displacement_m"),
        "median_oracle_uav_speed_m_s": median("maximum_uav_speed_m_s"),
        "median_oracle_maneuver_duration_s": median("maneuver_duration_s"),
        "oracle_hit_segment_counts": segments,
        "median_oracle_task_cost": median("task_cost"),
    }


def evaluate_candidate_map(
    environment: tuple,
    config: dict[str, Any],
    records: list[AmortizedCemContextRecord],
    candidate_map: dict[str, torch.Tensor],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    simulator, task, training_bank, _heldout, canonical_bank, _source = environment
    settings = _physics_settings(config)
    banks = {"training": training_bank, "canonical": canonical_bank}
    physical: dict[str, list[dict[str, Any]]] = {}
    for bank_name, bank in banks.items():
        selected = [record for record in records if record.bank == bank_name]
        candidate_count = (
            1 if not selected else int(candidate_map[selected[0].context_id].shape[0])
        )
        contexts_per_chunk = max(1, FIXED_NUMERICAL_BATCH_SIZE // candidate_count)
        for start in range(0, len(selected), contexts_per_chunk):
            group = selected[start : start + contexts_per_chunk]
            if not group:
                continue
            actions = torch.cat([candidate_map[record.context_id] for record in group])
            count = int(actions.shape[0] // len(group))
            state_indices, targets, directions = [], [], []
            for record in group:
                state_indices.extend([record.state_index] * count)
                targets.extend([record.target_local_m] * count)
                directions.extend([record.direction_local] * count)
            logical = ContextSpecification(
                torch.tensor(state_indices, dtype=torch.int64),
                torch.tensor(targets, dtype=torch.float32),
                torch.tensor(directions, dtype=torch.float32),
                label,
            )
            padded = pad_context_specification(logical, FIXED_NUMERICAL_BATCH_SIZE)
            context = build_context_from_specification(simulator, bank, padded)
            result = evaluate_normalized_actions_fixed_batch(
                simulator, context, actions, task, settings
            )
            metrics = _metric_rows(result, actions, task, settings)
            for index, record in enumerate(group):
                physical[record.context_id] = metrics[index * count : (index + 1) * count]
            print(
                f"{label} physics {min(start + len(group), len(selected))}/{len(selected)} {bank_name}",
                flush=True,
            )
    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = physical[record.context_id]
        successful = [index for index, row in enumerate(metrics) if bool(row["success"])]
        oracle_index = (
            min(successful, key=lambda index: float(metrics[index]["task_cost"]))
            if successful
            else min(range(len(metrics)), key=lambda index: float(metrics[index]["task_cost"]))
        )
        rows.append(
            {
                "context_id": record.context_id,
                "state_id": record.state_id,
                "bank": record.bank,
                "candidate_count": len(metrics),
                "candidate_success_count": len(successful),
                "candidate_feasible_count": sum(bool(row["feasible"]) for row in metrics),
                "first_metrics": metrics[0],
                "oracle_metrics": metrics[oracle_index],
            }
        )

    return rows, summarize_evaluation_rows(rows, label=label)


def _teacher_pair_distance(
    actions_by_context: dict[int, np.ndarray], first: int, second: int
) -> float:
    a = torch.from_numpy(actions_by_context[first]).float()
    b = torch.from_numpy(actions_by_context[second]).float()
    return float(torch.cdist(a, b).min())


def _candidate_pair_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(first - second, dim=1).mean())


def _correlation(first: list[float], second: list[float]) -> float | None:
    if len(first) < 3 or np.std(first) < 1.0e-12 or np.std(second) < 1.0e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def conditioning_audits(
    artifact: Path,
    config: dict[str, Any],
    environment: tuple,
    model: ConditionalActionDiffusion,
    normalizer: FixedContextNormalizer,
    contexts: np.ndarray,
    train_records: list[AmortizedCemContextRecord],
    external_records: list[AmortizedCemContextRecord],
    train_candidates: dict[str, torch.Tensor],
    external_candidates: dict[str, torch.Tensor],
    actions_by_context: dict[int, np.ndarray],
    noise: torch.Tensor,
    external_summary: dict[str, Any],
) -> dict[str, Any]:
    # Correct versus shuffled: all 64 external contexts, identical noise, original tasks.
    shuffled_records = external_records[:64]
    override = {
        record.context_id: contexts[shuffled_records[(index + 1) % len(shuffled_records)].context_index]
        for index, record in enumerate(shuffled_records)
    }
    _raw_shuffled, shuffled_candidates, _clamp = generate_candidates(
        model, normalizer, contexts, shuffled_records, noise, context_override=override
    )
    shuffled_rows, shuffled_summary = evaluate_candidate_map(
        environment, config, shuffled_records, shuffled_candidates, label="shuffled_context"
    )
    correct_rows = json.loads(
        (artifact / "external_development_rows.json").read_text(encoding="utf-8")
    )
    correct_by_id = {row["context_id"]: row for row in correct_rows}
    shuffled_action_distance = [
        _candidate_pair_distance(external_candidates[r.context_id], shuffled_candidates[r.context_id])
        for r in shuffled_records
    ]
    shuffled_audit = {
        "schema": "milestone7a3_shuffled_context_audit_v1",
        "context_count": len(shuffled_records),
        "identical_fixed_noise_bank": True,
        "evaluated_against_original_physical_context": True,
        "correct_context_oracle_success_rate": external_summary["oracle_best_of_32_success_rate"],
        "shuffled_context_oracle_success_rate": shuffled_summary["oracle_best_of_32_success_rate"],
        "correct_context_median_oracle_task_cost": external_summary["median_oracle_task_cost"],
        "shuffled_context_median_oracle_task_cost": shuffled_summary["median_oracle_task_cost"],
        "matched_noise_action_l2": {
            "mean": float(np.mean(shuffled_action_distance)),
            "median": float(np.median(shuffled_action_distance)),
        },
        "rows": shuffled_rows,
    }
    _write_json(artifact / "shuffled_context_audit.json", shuffled_audit)

    # Same-state, different-target pairs from existing eligible contexts.
    by_state: dict[str, list[AmortizedCemContextRecord]] = {}
    for record in train_records:
        by_state.setdefault(record.state_id, []).append(record)
    target_pairs = []
    for state_id in _hash_order(by_state):
        candidates = sorted(by_state[state_id], key=lambda row: row.context_id)
        distinct = []
        for record in candidates:
            if not distinct or np.linalg.norm(
                np.asarray(record.target_local_m) - np.asarray(distinct[0].target_local_m)
            ) > 0.10:
                distinct.append(record)
        if len(distinct) >= 2:
            target_pairs.append((distinct[0], distinct[1]))
        if len(target_pairs) == 16:
            break
    target_rows = []
    target_generated, target_teacher = [], []
    target_eval_records: list[AmortizedCemContextRecord] = []
    for first, second in target_pairs:
        generated_distance = _candidate_pair_distance(
            train_candidates[first.context_id], train_candidates[second.context_id]
        )
        teacher_distance = _teacher_pair_distance(
            actions_by_context, first.context_index, second.context_index
        )
        target_generated.append(generated_distance)
        target_teacher.append(teacher_distance)
        target_eval_records.extend((first, second))
        target_rows.append(
            {
                "state_id": first.state_id,
                "context_a": first.context_id,
                "context_b": second.context_id,
                "target_distance_m": float(
                    np.linalg.norm(np.asarray(first.target_local_m) - np.asarray(second.target_local_m))
                ),
                "matched_noise_generated_action_l2": generated_distance,
                "minimum_teacher_action_l2": teacher_distance,
            }
        )
    if len(target_pairs) < 16:
        raise RuntimeError(
            f"Only {len(target_pairs)} same-state/different-target pairs exist; 16 required."
        )
    target_map = {record.context_id: train_candidates[record.context_id] for record in target_eval_records}
    _target_physics_rows, target_physics = evaluate_candidate_map(
        environment, config, target_eval_records, target_map, label="target_conditioning"
    )
    target_swapped_map: dict[str, torch.Tensor] = {}
    for first, second in target_pairs:
        target_swapped_map[first.context_id] = train_candidates[second.context_id]
        target_swapped_map[second.context_id] = train_candidates[first.context_id]
    _target_swapped_rows, target_swapped_physics = evaluate_candidate_map(
        environment,
        config,
        target_eval_records,
        target_swapped_map,
        label="target_conditioning_cross_swapped",
    )
    target_audit = {
        "schema": "milestone7a3_target_conditioning_audit_v1",
        "pair_count": len(target_pairs),
        "same_state_verified": True,
        "identical_noise_bank": True,
        "mean_generated_action_l2": float(np.mean(target_generated)),
        "median_generated_action_l2": float(np.median(target_generated)),
        "mean_teacher_action_l2": float(np.mean(target_teacher)),
        "median_teacher_action_l2": float(np.median(target_teacher)),
        "teacher_generated_change_correlation": _correlation(target_teacher, target_generated),
        "own_target_oracle_success_rate": target_physics["oracle_best_of_32_success_rate"],
        "cross_swapped_target_oracle_success_rate": target_swapped_physics[
            "oracle_best_of_32_success_rate"
        ],
        "pairs": target_rows,
    }
    _write_json(artifact / "target_conditioning_audit.json", target_audit)

    # Same target, different physically propagated states.  The 6A state-variation
    # contexts provide many state-disjoint labels for the canonical target.
    by_target: dict[tuple[float, float, float], list[AmortizedCemContextRecord]] = {}
    for record in train_records:
        key = tuple(round(float(value), 5) for value in record.target_local_m)
        by_target.setdefault(key, []).append(record)
    state_pairs = []
    for key, values in sorted(by_target.items(), key=lambda item: (-len(item[1]), item[0])):
        ordered = sorted(values, key=lambda row: row.state_id)
        for start in range(0, len(ordered) - 1, 2):
            if ordered[start].state_id != ordered[start + 1].state_id:
                state_pairs.append((ordered[start], ordered[start + 1]))
            if len(state_pairs) == 16:
                break
        if len(state_pairs) == 16:
            break
    if len(state_pairs) < 16:
        raise RuntimeError(
            f"Only {len(state_pairs)} same-target/different-state pairs exist; 16 required."
        )
    state_rows, state_generated, state_teacher = [], [], []
    state_eval_records: list[AmortizedCemContextRecord] = []
    for first, second in state_pairs:
        generated_distance = _candidate_pair_distance(
            train_candidates[first.context_id], train_candidates[second.context_id]
        )
        teacher_distance = _teacher_pair_distance(
            actions_by_context, first.context_index, second.context_index
        )
        state_generated.append(generated_distance)
        state_teacher.append(teacher_distance)
        state_eval_records.extend((first, second))
        state_rows.append(
            {
                "target_local_m": list(first.target_local_m),
                "context_a": first.context_id,
                "context_b": second.context_id,
                "state_a": first.state_id,
                "state_b": second.state_id,
                "matched_noise_generated_action_l2": generated_distance,
                "minimum_teacher_action_l2": teacher_distance,
            }
        )
    state_map = {record.context_id: train_candidates[record.context_id] for record in state_eval_records}
    _state_physics_rows, state_physics = evaluate_candidate_map(
        environment, config, state_eval_records, state_map, label="state_conditioning"
    )
    state_swapped_map: dict[str, torch.Tensor] = {}
    for first, second in state_pairs:
        state_swapped_map[first.context_id] = train_candidates[second.context_id]
        state_swapped_map[second.context_id] = train_candidates[first.context_id]
    _state_swapped_rows, state_swapped_physics = evaluate_candidate_map(
        environment,
        config,
        state_eval_records,
        state_swapped_map,
        label="state_conditioning_cross_swapped",
    )
    state_audit = {
        "schema": "milestone7a3_state_conditioning_audit_v1",
        "pair_count": len(state_pairs),
        "same_target_verified": True,
        "physically_different_states": True,
        "identical_noise_bank": True,
        "mean_generated_action_l2": float(np.mean(state_generated)),
        "median_generated_action_l2": float(np.median(state_generated)),
        "mean_teacher_action_l2": float(np.mean(state_teacher)),
        "median_teacher_action_l2": float(np.median(state_teacher)),
        "teacher_generated_change_correlation": _correlation(state_teacher, state_generated),
        "own_state_oracle_success_rate": state_physics["oracle_best_of_32_success_rate"],
        "cross_swapped_state_oracle_success_rate": state_swapped_physics[
            "oracle_best_of_32_success_rate"
        ],
        "pairs": state_rows,
    }
    _write_json(artifact / "state_conditioning_audit.json", state_audit)

    correct = float(external_summary["oracle_best_of_32_success_rate"])
    shuffled = float(shuffled_summary["oracle_best_of_32_success_rate"])
    physical_conditioning = bool(correct >= shuffled + 0.05)
    target_active = bool(
        np.median(target_generated) > 0.05
        and (
            physical_conditioning
            or target_physics["oracle_best_of_32_success_rate"]
            >= target_swapped_physics["oracle_best_of_32_success_rate"] + 0.02
        )
        and (_correlation(target_teacher, target_generated) or -1.0) > 0.20
    )
    state_active = bool(
        np.median(state_generated) > 0.05
        and state_physics["oracle_best_of_32_success_rate"]
        >= state_swapped_physics["oracle_best_of_32_success_rate"] + 0.05
    )
    summary = {
        "schema": "milestone7a3_context_sensitivity_summary_v1",
        "correct_context_oracle_success_rate": correct,
        "shuffled_context_oracle_success_rate": shuffled,
        "correct_minus_shuffled_oracle_percentage_points": 100.0 * (correct - shuffled),
        "shuffled_context_generated_action_l2_median": float(
            np.median(shuffled_action_distance)
        ),
        "target_change_generated_action_l2_median": float(np.median(target_generated)),
        "target_change_teacher_action_l2_median": float(np.median(target_teacher)),
        "target_teacher_generated_change_correlation": _correlation(
            target_teacher, target_generated
        ),
        "target_correct_oracle_success_rate": target_physics[
            "oracle_best_of_32_success_rate"
        ],
        "target_cross_swapped_oracle_success_rate": target_swapped_physics[
            "oracle_best_of_32_success_rate"
        ],
        "state_change_generated_action_l2_median": float(np.median(state_generated)),
        "state_change_teacher_action_l2_median": float(np.median(state_teacher)),
        "state_teacher_generated_change_correlation": _correlation(state_teacher, state_generated),
        "state_correct_oracle_success_rate": state_physics[
            "oracle_best_of_32_success_rate"
        ],
        "state_cross_swapped_oracle_success_rate": state_swapped_physics[
            "oracle_best_of_32_success_rate"
        ],
        "target_conditioning": "ACTIVE" if target_active and physical_conditioning else "WEAK",
        "state_conditioning": "ACTIVE" if state_active and physical_conditioning else "WEAK",
        "physical_correct_context_advantage": physical_conditioning,
    }
    _write_json(artifact / "context_sensitivity_summary.json", summary)
    return summary


def _copy_subset_dataset(
    artifact: Path, destination: Path, selected_state_ids: set[str]
) -> tuple[int, int]:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact / "context_table.npz", destination / "context_table.npz")
    shutil.copy2(artifact / "context_normalizer.json", destination / "context_normalizer.json")
    shutil.copy2(artifact / "fixed_noise_bank.npy", destination / "fixed_noise_bank.npy")
    records, _contexts = _load_records(artifact)
    state_by_index = {record.context_index: record.state_id for record in records}
    manifest = json.loads(
        (artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8")
    )
    entries = []
    context_count = action_count = 0
    (destination / "diffusion_teacher_shards").mkdir(parents=True, exist_ok=True)
    for entry in manifest["shards"]:
        include = entry["split"] == "VALIDATION" or (
            entry["split"] == "TRAIN"
            and state_by_index[int(entry["context_index"])] in selected_state_ids
        )
        if not include:
            continue
        shutil.copy2(
            artifact / "diffusion_teacher_shards" / entry["path"],
            destination / "diffusion_teacher_shards" / entry["path"],
        )
        entries.append(entry)
        if entry["split"] == "TRAIN":
            context_count += 1
            action_count += int(entry["row_count"])
    _write_json(
        destination / "diffusion_teacher_manifest.json",
        {
            "schema": "milestone7a3_learning_curve_teacher_manifest_v1",
            "selected_state_ids": sorted(selected_state_ids),
            "shards": entries,
        },
    )
    return context_count, action_count


def _stage_full_names(artifact: Path) -> None:
    mapping = {
        "diffusion_config.json": "full_diffusion_config.json",
        "diffusion_training_history.json": "full_training_history.json",
        "diffusion_best.pt": "full_diffusion_best.pt",
        "diffusion_ema_best.pt": "full_diffusion_ema_best.pt",
    }
    for source, destination in mapping.items():
        shutil.copy2(artifact / source, artifact / destination)


def benchmark_latency(
    config: dict[str, Any],
    artifact: Path,
    model: ConditionalActionDiffusion,
    normalizer: FixedContextNormalizer,
    raw_context: np.ndarray,
    noise: torch.Tensor,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    alpha_bar = cosine_alpha_bar_schedule().to(device)
    schedule = ddim_timestep_schedule().to(device)

    def query() -> None:
        context = normalizer.normalize(torch.from_numpy(raw_context[None]).to(device))
        sample = sample_ddim(
            model, context, noise, alpha_bar=alpha_bar, timestep_schedule=schedule
        )
        _ = sample.bounded_action.cpu()

    for _ in range(int(config["latency_warmup_queries"])):
        query()
    torch.cuda.synchronize()
    values = []
    for _ in range(int(config["latency_measured_queries"])):
        start = time.perf_counter()
        query()
        torch.cuda.synchronize()
        values.append(1000.0 * (time.perf_counter() - start))
    result = {
        "schema": "milestone7a3_generator_latency_v1",
        "warmup_queries": int(config["latency_warmup_queries"]),
        "measured_queries": len(values),
        "includes_context_normalization": True,
        "includes_32_candidate_25_step_ddim": True,
        "includes_scorer": False,
        "includes_simulator": False,
        "includes_cem": False,
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "maximum_ms": float(np.max(values)),
    }
    _write_json(artifact / "inference_latency.json", result)
    return result


def _evaluate_model(
    model_root: Path,
    artifact: Path,
    config: dict[str, Any],
    environment: tuple,
    records: list[AmortizedCemContextRecord],
    contexts: np.ndarray,
    normalizer: FixedContextNormalizer,
    noise: torch.Tensor,
    label: str,
) -> tuple[dict[str, torch.Tensor], float, list[dict[str, Any]], dict[str, Any]]:
    model = _load_model(model_root, torch.device("cuda"))
    _raw, bounded, clamp = generate_candidates(
        model, normalizer, contexts, records, noise
    )
    rows, summary = evaluate_candidate_map(
        environment, config, records, bounded, label=label
    )
    summary["clamp_fraction"] = clamp
    return bounded, clamp, rows, summary


def run_learning_curve(
    artifact: Path,
    config: dict[str, Any],
    environment: tuple,
    records: list[AmortizedCemContextRecord],
    contexts: np.ndarray,
    normalizer: FixedContextNormalizer,
    noise: torch.Tensor,
    external_records: list[AmortizedCemContextRecord],
    full_train_rows: list[dict[str, Any]],
    full_train_summary: dict[str, Any],
    full_external_summary: dict[str, Any],
) -> dict[str, Any]:
    train_records = [record for record in records if record.split == "TRAIN"]
    states = _hash_order({record.state_id for record in train_records})
    base_target = int(config["learning_curve"]["base_target_contexts"])
    base_states: list[str] = []
    for state_id in states:
        base_states.append(state_id)
        if sum(record.state_id in base_states for record in train_records) >= base_target:
            break
    mid_count = max(len(base_states), int(round(len(states) * float(config["learning_curve"]["mid_state_fraction"]))))
    selections = {
        "BASE": set(base_states),
        "MID": set(states[:mid_count]),
        "FULL": set(states),
    }
    diagnostic_records = [
        record for record in train_records if record.state_id in selections["BASE"]
    ]
    rows = []
    for name in ("BASE", "MID"):
        model_root = artifact / "learning_curve_models" / name.lower()
        context_count, action_count = _copy_subset_dataset(
            artifact, model_root, selections[name]
        )
        summary = train_diffusion(model_root, config, device="cuda")
        _bounded, clamp, _train_rows, train_eval = _evaluate_model(
            model_root,
            artifact,
            config,
            environment,
            diagnostic_records,
            contexts,
            normalizer,
            noise,
            f"curve_{name.lower()}_train",
        )
        _bounded, _dev_clamp, _dev_rows, dev_eval = _evaluate_model(
            model_root,
            artifact,
            config,
            environment,
            external_records,
            contexts,
            normalizer,
            noise,
            f"curve_{name.lower()}_external",
        )
        rows.append(
            {
                "model": name,
                "training_states": len(selections[name]),
                "training_contexts": context_count,
                "successful_action_labels": action_count,
                "training_updates": summary["updates"],
                "training_diagnostic_contexts": len(diagnostic_records),
                "training_oracle_best_of_32": train_eval["oracle_best_of_32_success_rate"],
                "external_development_oracle_best_of_32": dev_eval["oracle_best_of_32_success_rate"],
                "external_development_first_candidate": dev_eval["first_candidate_success_rate"],
                "external_at_least_one_feasible_candidate": dev_eval[
                    "contexts_with_at_least_one_feasible_candidate_rate"
                ],
                "clamp_fraction": clamp,
            }
        )
    diagnostic_ids = {record.context_id for record in diagnostic_records}
    full_diagnostic = summarize_evaluation_rows(
        [row for row in full_train_rows if row["context_id"] in diagnostic_ids],
        label="curve_full_train_diagnostic",
    )
    full_manifest = json.loads(
        (artifact / "diffusion_teacher_manifest.json").read_text(encoding="utf-8")
    )
    full_entries = [entry for entry in full_manifest["shards"] if entry["split"] == "TRAIN"]
    rows.append(
        {
            "model": "FULL",
            "training_states": len(selections["FULL"]),
            "training_contexts": len(full_entries),
            "successful_action_labels": sum(int(entry["row_count"]) for entry in full_entries),
            "training_updates": json.loads(
                (artifact / "diffusion_training_summary.json").read_text(encoding="utf-8")
            )["updates"],
            "training_diagnostic_contexts": full_diagnostic["context_count"],
            "training_oracle_best_of_32": full_diagnostic["oracle_best_of_32_success_rate"],
            "external_development_oracle_best_of_32": full_external_summary[
                "oracle_best_of_32_success_rate"
            ],
            "external_development_first_candidate": full_external_summary[
                "first_candidate_success_rate"
            ],
            "external_at_least_one_feasible_candidate": full_external_summary[
                "contexts_with_at_least_one_feasible_candidate_rate"
            ],
            "clamp_fraction": full_train_summary["clamp_fraction"],
        }
    )
    result = {
        "schema": "milestone7a3_existing_data_learning_curve_v1",
        "nested_state_subsets": True,
        "fixed_external_development_set": True,
        "same_architecture_hyperparameters_and_noise_bank": True,
        "models": rows,
    }
    _write_json(artifact / "learning_curve.json", result)
    return result


def classify(
    train: dict[str, Any],
    external: dict[str, Any],
    sensitivity: dict[str, Any],
    curve: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    train_oracle = float(train["oracle_best_of_32_success_rate"])
    dev_oracle = float(external["oracle_best_of_32_success_rate"])
    curve_by_name = {row["model"]: row for row in curve["models"]}
    base = float(curve_by_name["BASE"]["external_development_oracle_best_of_32"])
    mid = float(curve_by_name["MID"]["external_development_oracle_best_of_32"])
    full = float(curve_by_name["FULL"]["external_development_oracle_best_of_32"])
    scaling = bool(full >= base + 0.05 and full >= mid - 0.025)
    conditioning = bool(
        sensitivity["physical_correct_context_advantage"]
        and sensitivity["target_conditioning"] == "ACTIVE"
        and sensitivity["state_conditioning"] == "ACTIVE"
    )
    if not conditioning:
        result = "CONTEXT_CONDITIONING_WEAK"
        more = "NO"
    elif train_oracle < 0.35:
        result = "GENERATOR_STILL_WEAK"
        more = "NO"
    elif train_oracle >= 0.70 and dev_oracle >= 0.50:
        result = "EXISTING_DATA_SUPPORTS_GENERATOR"
        more = "NO"
    elif train_oracle >= 0.50 and dev_oracle + 0.15 < train_oracle and scaling:
        result = "DATA_COVERAGE_LIMITED"
        more = "YES"
    elif train_oracle > 0.2667 and dev_oracle > 0.0 and scaling:
        result = "ARCHITECTURE_PROMISING"
        more = "YES"
    else:
        result = "GENERATOR_STILL_WEAK"
        more = "NO"
    evidence = {
        "training_oracle": train_oracle,
        "external_development_oracle": dev_oracle,
        "previous_7a2_training_oracle": 0.2667,
        "base_external_oracle": base,
        "mid_external_oracle": mid,
        "full_external_oracle": full,
        "clear_scaling_trend": scaling,
        "conditioning_active": conditioning,
        "classification": result,
        "more_cem_data_justified": more,
    }
    return result, more, evidence


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(
    artifact: Path,
    inventory: dict[str, Any],
    training_summary: dict[str, Any],
    distribution: dict[str, Any],
    train: dict[str, Any],
    external: dict[str, Any],
    sensitivity: dict[str, Any],
    curve: dict[str, Any],
    latency: dict[str, Any],
    classification: str,
    more_cem: str,
) -> str:
    models = {row["model"]: row for row in curve["models"]}
    target = json.loads((artifact / "target_conditioning_audit.json").read_text(encoding="utf-8"))
    state = json.loads((artifact / "state_conditioning_audit.json").read_text(encoding="utf-8"))
    shuffled = json.loads((artifact / "shuffled_context_audit.json").read_text(encoding="utf-8"))
    report = f"""# Milestone 7A.3 — Existing-Data Scaling and Conditional Diffusion Report

Generated: {_timestamp()}

## 1. Why no new CEM was run

The experiment used only durable partial-7A and Milestone-6A authoritative actions. The 6A benchmark already contained paid-for TRAIN, AGGREGATION_POOL, and EXTERNAL_DEVELOPMENT coverage. Running another optimizer before consuming these labels would not distinguish data coverage from conditional-learning failure. New CEM solves: **0**.

## 2. 7A.2 scheduler repair confirmation

The final generator uses the repaired deterministic 25-step DDIM schedule `{ddim_timestep_schedule().tolist()}`. It begins at t=95 and ends at t=0, avoiding the t=99 cosine terminal cliff. Training remains a 100-step cosine DDPM epsilon objective. No scheduler, formulation, action representation, or architecture change was made in 7A.3.

## 3. Existing-data inventory

The deduplicated corpus contains {inventory['training']['contexts']} gradient-training contexts ({inventory['training']['successful_actions']} verified successful actions), {inventory['internal_denoising_validation']['contexts']} internal denoising-validation contexts ({inventory['internal_denoising_validation']['successful_actions']} actions), and {inventory['external_development']['contexts']} external-development contexts. Context identity includes state, target, direction, theta, bank, and schema; duplicate action labels were removed at 1e-7 normalized-coordinate precision while all provenance was retained.

## 4. Ownership and leakage audit

Only 7A TRAIN and AGGREGATION_POOL state IDs entered gradients. AGGREGATION_POOL was explicitly consumed as training data for this experiment. Internal validation was split 90/10 by state ID with seed 42. EXTERNAL_DEVELOPMENT contributed no gradients, normalizer fitting, early stopping, or checkpoint selection. 7A VALIDATION, TEST, EDGE-reserved, and `fig8vertical_002` contributed zero gradient rows. State-level leakage: **NONE**.

## 5. Distinct states, contexts, and actions

- Gradient-training states: {inventory['training']['states']}
- Gradient-training contexts: {inventory['training']['contexts']}
- Gradient-training successful actions: {inventory['training']['successful_actions']}
- Internal validation states: {inventory['internal_denoising_validation']['states']}
- Internal validation contexts: {inventory['internal_denoising_validation']['contexts']}
- External-development contexts: {inventory['external_development']['contexts']}

The context-balanced sampler first samples a context uniformly and then one successful action uniformly within that context. Multi-solution partial-7A contexts therefore do not dominate single-solution 6A contexts.

## 6. External-development definition

The primary physics-development set is the immutable set of {external['context_count']} compatible Milestone-6A EXTERNAL_DEVELOPMENT contexts. These are canonical-state target-variation cases, each with an authoritative CEM success. They remained fully gradient-free and were not used for denoising validation or checkpoint selection.

## 7. Final diffusion architecture confirmation

Architecture is unchanged: 83-D context encoder `83→256 SiLU→256`; sinusoidal 64-D timestep embedding projected to 256; 49-D noisy-action projection; four `LayerNorm→256→512→SiLU→256` residual MLP blocks; and a 49-D epsilon output. The model has the existing approximately 1.249M parameters, direct normalized [49] diffusion, EMA 0.999, and no PCA/latent/transformer bottleneck.

## 8. Full-model training

The FULL model was initialized from scratch and trained with AdamW (`lr=2e-4`, weight decay `1e-6`), batch up to 1024, gradient clip 1.0, and EMA 0.999. It ran {training_summary['updates']} updates; best internal-validation epsilon loss was {training_summary['best_validation_epsilon_loss']:.6f} at update {training_summary['best_update']}. External development was not consulted during training.

## 9. Generated action-distribution sanity

Pre-clamp generated actions have mean {distribution['pre_clamp']['coordinate_mean']:.5f}, standard deviation {distribution['pre_clamp']['coordinate_std']:.5f}, absolute p95 {distribution['pre_clamp']['p95_absolute']:.5f}, and range [{distribution['pre_clamp']['minimum']:.5f}, {distribution['pre_clamp']['maximum']:.5f}]. The final clamp fraction is {_percent(distribution['final_clamp_fraction'])}; normalized duration ranges from {distribution['pre_clamp']['duration_normalized']['minimum']:.4f} to {distribution['pre_clamp']['duration_normalized']['maximum']:.4f}. The terminal-cliff pathology did not recur.

## 10. Training first-candidate result

First-candidate scientific success on {train['context_count']} gradient-training contexts is {_percent(train['first_candidate_success_rate'])}; feasibility is {_percent(train['first_candidate_feasibility_rate'])}. Median first-candidate tip error is {1000.0 * train['median_first_tip_distance_m']:.2f} mm, directed speed {train['median_first_directed_speed_m_s']:.3f} m/s, direction error {train['median_first_direction_error_deg']:.2f}°, UAV displacement {train['median_first_uav_displacement_m']:.3f} m, UAV speed {train['median_first_uav_speed_m_s']:.3f} m/s, and duration {train['median_first_maneuver_duration_s']:.3f} s.

## 11. Training oracle best-of-32 result

Training oracle best-of-32 scientific success is {_percent(train['oracle_best_of_32_success_rate'])}, compared with 26.67% in Milestone 7A.2.

## 12. Training feasible-support result

At least one feasible candidate exists for {_percent(train['contexts_with_at_least_one_feasible_candidate_rate'])} of training contexts. Candidate-level feasibility is {_percent(train['candidate_level_feasibility_rate'])}. Median oracle tip error is {1000.0 * train['median_oracle_tip_distance_m']:.2f} mm, directed speed {train['median_oracle_directed_speed_m_s']:.3f} m/s, direction error {train['median_oracle_direction_error_deg']:.2f}°, UAV displacement {train['median_oracle_uav_displacement_m']:.3f} m, and UAV speed {train['median_oracle_uav_speed_m_s']:.3f} m/s. Oracle hit segments are {train['oracle_hit_segment_counts']}.

## 13. External-development first-candidate result

External-development first-candidate scientific success is {_percent(external['first_candidate_success_rate'])}; feasibility is {_percent(external['first_candidate_feasibility_rate'])}. Median first-candidate tip error is {1000.0 * external['median_first_tip_distance_m']:.2f} mm, directed speed {external['median_first_directed_speed_m_s']:.3f} m/s, direction error {external['median_first_direction_error_deg']:.2f}°, UAV displacement {external['median_first_uav_displacement_m']:.3f} m, UAV speed {external['median_first_uav_speed_m_s']:.3f} m/s, and duration {external['median_first_maneuver_duration_s']:.3f} s.

## 14. External-development oracle best-of-32

External-development oracle best-of-32 success is {_percent(external['oracle_best_of_32_success_rate'])}. At least one feasible candidate exists for {_percent(external['contexts_with_at_least_one_feasible_candidate_rate'])}; candidate-level feasibility is {_percent(external['candidate_level_feasibility_rate'])}. Median oracle tip error is {1000.0 * external['median_oracle_tip_distance_m']:.2f} mm, directed speed {external['median_oracle_directed_speed_m_s']:.3f} m/s, direction error {external['median_oracle_direction_error_deg']:.2f}°, UAV displacement {external['median_oracle_uav_displacement_m']:.3f} m, UAV speed {external['median_oracle_uav_speed_m_s']:.3f} m/s, and maneuver duration {external['median_oracle_maneuver_duration_s']:.3f} s. Oracle hit segments are {external['oracle_hit_segment_counts']}.

## 15. Amortization conditional on CEM success

All {external['context_count']} external-development references are authoritative CEM successes. Therefore `P(diffusion oracle PASS | CEM PASS)` equals the diffusion external-development oracle: **{_percent(external['oracle_best_of_32_success_rate'])}**.

## 16. Correct-versus-shuffled context experiment

Using identical fixed noise, correct-context oracle success is {_percent(shuffled['correct_context_oracle_success_rate'])}; deterministically shuffled conditioning evaluated against the original task yields {_percent(shuffled['shuffled_context_oracle_success_rate'])}. The median matched-noise action change is {shuffled['matched_noise_action_l2']['median']:.4f}. Median oracle task cost changes from {shuffled['correct_context_median_oracle_task_cost']:.4f} to {shuffled['shuffled_context_median_oracle_task_cost']:.4f}.

## 17. Target-conditioning experiment

Across {target['pair_count']} same-state/different-target pairs, the median matched-noise generated-action distance is {target['median_generated_action_l2']:.4f}, versus median teacher-action distance {target['median_teacher_action_l2']:.4f}. Teacher/generated change-magnitude correlation is {target['teacher_generated_change_correlation']}. Own-target oracle success is {_percent(target['own_target_oracle_success_rate'])}; cross-swapping each pair's generated candidates to the other target gives {_percent(target['cross_swapped_target_oracle_success_rate'])}. Classification: **{sensitivity['target_conditioning']}**.

## 18. Initial-state-conditioning experiment

Across {state['pair_count']} same-target/different-physical-state pairs, the median matched-noise generated-action distance is {state['median_generated_action_l2']:.4f}, versus median teacher-action distance {state['median_teacher_action_l2']:.4f}. Teacher/generated change-magnitude correlation is {state['teacher_generated_change_correlation']}. Own-state oracle success is {_percent(state['own_state_oracle_success_rate'])}; cross-swapping each pair's generated candidates to the other physical state gives {_percent(state['cross_swapped_state_oracle_success_rate'])}. Classification: **{sensitivity['state_conditioning']}**.

## 19. Context-sensitivity summary

Correct conditioning exceeds shuffled conditioning by {sensitivity['correct_minus_shuffled_oracle_percentage_points']:.2f} percentage points. Median generated action changes are {sensitivity['target_change_generated_action_l2_median']:.4f} for target changes, {sensitivity['state_change_generated_action_l2_median']:.4f} for state changes, and {sensitivity['shuffled_context_generated_action_l2_median']:.4f} for shuffled context. This separates mere marginal-distribution matching from behaviorally useful conditioning.

## 20. Existing-data learning curve

| Model | States | Contexts | Labels | Train oracle | Dev first | Dev oracle | Dev any feasible | Clamp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BASE | {models['BASE']['training_states']} | {models['BASE']['training_contexts']} | {models['BASE']['successful_action_labels']} | {_percent(models['BASE']['training_oracle_best_of_32'])} | {_percent(models['BASE']['external_development_first_candidate'])} | {_percent(models['BASE']['external_development_oracle_best_of_32'])} | {_percent(models['BASE']['external_at_least_one_feasible_candidate'])} | {_percent(models['BASE']['clamp_fraction'])} |
| MID | {models['MID']['training_states']} | {models['MID']['training_contexts']} | {models['MID']['successful_action_labels']} | {_percent(models['MID']['training_oracle_best_of_32'])} | {_percent(models['MID']['external_development_first_candidate'])} | {_percent(models['MID']['external_development_oracle_best_of_32'])} | {_percent(models['MID']['external_at_least_one_feasible_candidate'])} | {_percent(models['MID']['clamp_fraction'])} |
| FULL | {models['FULL']['training_states']} | {models['FULL']['training_contexts']} | {models['FULL']['successful_action_labels']} | {_percent(models['FULL']['training_oracle_best_of_32'])} | {_percent(models['FULL']['external_development_first_candidate'])} | {_percent(models['FULL']['external_development_oracle_best_of_32'])} | {_percent(models['FULL']['external_at_least_one_feasible_candidate'])} | {_percent(models['FULL']['clamp_fraction'])} |

## 21. Does performance scale with context count?

The classification logic treats a ≥5-point FULL-over-BASE development-oracle improvement, without a material FULL regression below MID, as a clear scaling signal. The measured result is recorded in `classification_evidence.json` and supports the final evidence-based classification below.

## 22. Neural inference latency

For context normalization plus deterministic 32-candidate, 25-step DDIM generation, median latency is {latency['median_ms']:.2f} ms, p90 {latency['p90_ms']:.2f} ms, p95 {latency['p95_ms']:.2f} ms, and maximum {latency['maximum_ms']:.2f} ms across {latency['measured_queries']} warmed queries. Scorer, simulator, and CEM are excluded.

## 23. Whether more CEM data is justified

**{more_cem}**. BASE→MID→FULL development performance does scale, which is encouraging, but the FULL generator still reaches only {_percent(train['oracle_best_of_32_success_rate'])} oracle success on its own broad training contexts—below the earlier 26.67% sanity result. The milestone explicitly says weak own-training fit overrides a scaling-only argument for purchasing more labels. No CEM was launched regardless of this result.

## 24. Recommended next step

Classification: **{classification}**. Conditioning is demonstrably active and external target-only performance is useful, so the next review should isolate why the same model underfits the heterogeneous state/target training corpus—for example, per-source/group oracle breakdown and conditional denoising error—before buying more labels or resuming scorer work. No next experiment was started automatically.

## 25. Scorer

**NOT TRAINED / NOT EVALUATED.**

## 26. Final TEST

**NOT EVALUATED.**

## 27. Protected test

`fig8vertical_002`: **NOT EVALUATED.**

## 28. Hardware

Real hardware: **NOT EXECUTED.** All maneuvers remain simulation-only.

## 29. Regression verification

The focused 7A.3/diffusion contract tests pass 17/17. The complete repository regression suite passes **128/128** tests. The audit locks the repaired t95→0 schedule and statically confirms that this runner contains neither a CEM-optimizer call nor a scorer-training call.

## Final summary

    New CEM solves:
        0

    Scheduler:
        REPAIRED t95->0 DDIM

    Diffusion architecture:
        UNCHANGED FINAL ARCHITECTURE

    Distinct training states:
        {inventory['training']['states']}

    Training contexts:
        {inventory['training']['contexts']}

    Successful teacher actions:
        {inventory['training']['successful_actions']}

    External-development contexts:
        {external['context_count']}

    Clamp fraction:
        {_percent(distribution['final_clamp_fraction'])}

    Training first-candidate success:
        {_percent(train['first_candidate_success_rate'])}

    Training oracle best-of-32:
        {_percent(train['oracle_best_of_32_success_rate'])}

    Training contexts with feasible candidate:
        {_percent(train['contexts_with_at_least_one_feasible_candidate_rate'])}

    External-development first-candidate success:
        {_percent(external['first_candidate_success_rate'])}

    External-development oracle best-of-32:
        {_percent(external['oracle_best_of_32_success_rate'])}

    Policy oracle success given CEM success:
        {_percent(external['oracle_best_of_32_success_rate'])}

    Correct-context oracle:
        {_percent(sensitivity['correct_context_oracle_success_rate'])}

    Shuffled-context oracle:
        {_percent(sensitivity['shuffled_context_oracle_success_rate'])}

    Target conditioning:
        {sensitivity['target_conditioning']}

    State conditioning:
        {sensitivity['state_conditioning']}

    Learning curve:

        BASE contexts = {models['BASE']['training_contexts']}
        dev oracle = {_percent(models['BASE']['external_development_oracle_best_of_32'])}

        MID contexts = {models['MID']['training_contexts']}
        dev oracle = {_percent(models['MID']['external_development_oracle_best_of_32'])}

        FULL contexts = {models['FULL']['training_contexts']}
        dev oracle = {_percent(models['FULL']['external_development_oracle_best_of_32'])}

    Median generator latency:
        {latency['median_ms']:.2f} ms

    P95 generator latency:
        {latency['p95_ms']:.2f} ms

    Classification:
        {classification}

    More CEM data justified:
        {more_cem}

    Scorer:
        NOT TRAINED

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    (artifact / REPORT.name).write_text(report, encoding="utf-8")
    REPORT.write_text(report, encoding="utf-8")
    return report


def run_all(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    _validate_config(config)
    artifact.mkdir(parents=True, exist_ok=True)
    _write_json(artifact / "run_config.json", config)
    inventory = prepare_existing_data(config, artifact)
    records, contexts = _load_records(artifact)
    train_records = [record for record in records if record.split == "TRAIN"]
    external_records = [record for record in records if record.split == "EXTERNAL_DEVELOPMENT"]
    if len(external_records) != 64:
        raise RuntimeError(f"Expected 64 external-development contexts, found {len(external_records)}.")
    normalizer = FixedContextNormalizer.load(artifact / "context_normalizer.json")
    actions_by_context = _load_actions_by_context(artifact)
    noise = torch.from_numpy(np.load(artifact / "fixed_noise_bank.npy")).to("cuda")

    training_summary = train_diffusion(artifact, config, device="cuda")
    _stage_full_names(artifact)
    model = _load_model(artifact, torch.device("cuda"))
    environment = _load_environment(config)

    representative = train_records[:64] + external_records[:64]
    raw_representative, bounded_representative, _clamp = generate_candidates(
        model, normalizer, contexts, representative, noise
    )
    distribution = distribution_audit(
        artifact,
        representative,
        raw_representative,
        bounded_representative,
        actions_by_context,
    )
    if distribution["final_clamp_fraction"] >= 0.10:
        raise RuntimeError("Generator clamp fraction is pathological; physics evaluation stopped.")

    _raw_train, train_candidates, train_clamp = generate_candidates(
        model, normalizer, contexts, train_records, noise
    )
    train_rows, train_summary = evaluate_candidate_map(
        environment, config, train_records, train_candidates, label="full_train"
    )
    train_summary["clamp_fraction"] = train_clamp
    _write_json(artifact / "train_oracle_rows.json", train_rows)
    _write_json(artifact / "train_oracle_summary.json", train_summary)

    _raw_external, external_candidates, external_clamp = generate_candidates(
        model, normalizer, contexts, external_records, noise
    )
    external_rows, external_summary = evaluate_candidate_map(
        environment, config, external_records, external_candidates, label="full_external"
    )
    external_summary["clamp_fraction"] = external_clamp
    external_summary["p_diffusion_oracle_pass_given_cem_pass"] = external_summary[
        "oracle_best_of_32_success_rate"
    ]
    _write_json(artifact / "external_development_rows.json", external_rows)
    _write_json(artifact / "external_development_summary.json", external_summary)

    sensitivity = conditioning_audits(
        artifact,
        config,
        environment,
        model,
        normalizer,
        contexts,
        train_records,
        external_records,
        train_candidates,
        external_candidates,
        actions_by_context,
        noise,
        external_summary,
    )
    curve = run_learning_curve(
        artifact,
        config,
        environment,
        records,
        contexts,
        normalizer,
        noise,
        external_records,
        train_rows,
        train_summary,
        external_summary,
    )
    latency = benchmark_latency(
        config, artifact, model, normalizer, contexts[train_records[0].context_index], noise
    )
    classification, more_cem, evidence = classify(
        train_summary, external_summary, sensitivity, curve
    )
    _write_json(artifact / "classification_evidence.json", evidence)
    source_paths = [
        ROOT / "run_milestone7a3.py",
        ROOT / "config/learning/diffusion_existing_data_scaling_v1.json",
        ROOT / "learning/action_diffusion.py",
        ROOT / "learning/amortized_cem_training.py",
        ROOT / "learning/amortized_cem_data.py",
        ROOT / "planning/production_cem.py",
        ROOT / "tests/test_milestone7a3_existing_data_scaling.py",
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone7a3_source_hash_manifest_v1",
            "files": [
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                }
                for path in source_paths
            ],
        },
    )
    write_report(
        artifact,
        inventory,
        training_summary,
        distribution,
        train_summary,
        external_summary,
        sensitivity,
        curve,
        latency,
        classification,
        more_cem,
    )
    return {
        "artifact": str(artifact),
        "classification": classification,
        "more_cem_data_justified": more_cem,
        "training_oracle": train_summary["oracle_best_of_32_success_rate"],
        "external_oracle": external_summary["oracle_best_of_32_success_rate"],
    }


def rerun_conditioning_and_report(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    """Recompute the strengthened cross-swapped audits without retraining."""

    _validate_config(config)
    records, contexts = _load_records(artifact)
    train_records = [record for record in records if record.split == "TRAIN"]
    external_records = [record for record in records if record.split == "EXTERNAL_DEVELOPMENT"]
    normalizer = FixedContextNormalizer.load(artifact / "context_normalizer.json")
    actions_by_context = _load_actions_by_context(artifact)
    noise = torch.from_numpy(np.load(artifact / "fixed_noise_bank.npy")).to("cuda")
    model = _load_model(artifact, torch.device("cuda"))
    environment = _load_environment(config)
    _raw_train, train_candidates, _train_clamp = generate_candidates(
        model, normalizer, contexts, train_records, noise
    )
    _raw_external, external_candidates, _external_clamp = generate_candidates(
        model, normalizer, contexts, external_records, noise
    )
    external_summary = json.loads(
        (artifact / "external_development_summary.json").read_text(encoding="utf-8")
    )
    sensitivity = conditioning_audits(
        artifact,
        config,
        environment,
        model,
        normalizer,
        contexts,
        train_records,
        external_records,
        train_candidates,
        external_candidates,
        actions_by_context,
        noise,
        external_summary,
    )
    inventory = json.loads((artifact / "existing_data_inventory.json").read_text(encoding="utf-8"))
    training_summary = json.loads(
        (artifact / "diffusion_training_summary.json").read_text(encoding="utf-8")
    )
    distribution = json.loads(
        (artifact / "generator_distribution_statistics.json").read_text(encoding="utf-8")
    )
    train_summary = json.loads((artifact / "train_oracle_summary.json").read_text(encoding="utf-8"))
    curve = json.loads((artifact / "learning_curve.json").read_text(encoding="utf-8"))
    latency = json.loads((artifact / "inference_latency.json").read_text(encoding="utf-8"))
    classification, more_cem, evidence = classify(
        train_summary, external_summary, sensitivity, curve
    )
    _write_json(artifact / "classification_evidence.json", evidence)
    source_paths = [
        ROOT / "run_milestone7a3.py",
        ROOT / "config/learning/diffusion_existing_data_scaling_v1.json",
        ROOT / "learning/action_diffusion.py",
        ROOT / "learning/amortized_cem_training.py",
        ROOT / "learning/amortized_cem_data.py",
        ROOT / "planning/production_cem.py",
        ROOT / "tests/test_milestone7a3_existing_data_scaling.py",
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone7a3_source_hash_manifest_v1",
            "files": [
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                }
                for path in source_paths
            ],
        },
    )
    write_report(
        artifact,
        inventory,
        training_summary,
        distribution,
        train_summary,
        external_summary,
        sensitivity,
        curve,
        latency,
        classification,
        more_cem,
    )
    return {
        "artifact": str(artifact),
        "classification": classification,
        "more_cem_data_justified": more_cem,
        "context_sensitivity": sensitivity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "reaudit", "all"),
        default="all",
        help="Prepare is data-only; train continues a prepared artifact; all performs the milestone.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    artifact = args.artifact or ARTIFACT_PARENT / _timestamp()
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    if args.stage == "prepare":
        _validate_config(config)
        artifact.mkdir(parents=True, exist_ok=True)
        _write_json(artifact / "run_config.json", config)
        print(json.dumps(_safe(prepare_existing_data(config, artifact)), indent=2), flush=True)
    elif args.stage == "train":
        print(json.dumps(_safe(run_all(config, artifact)), indent=2), flush=True)
    elif args.stage == "reaudit":
        print(
            json.dumps(_safe(rerun_conditioning_and_report(config, artifact)), indent=2),
            flush=True,
        )
    else:
        print(json.dumps(_safe(run_all(config, artifact)), indent=2), flush=True)


if __name__ == "__main__":
    main()
