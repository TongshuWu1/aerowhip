"""Milestone 7A.5 frozen diffusion candidate-support and latency study.

This runner never trains, imports a scorer, or invokes CEM.  It freezes the
7A.3 flat EMA generator, preserves the corrected 7A.4 first-128 noise bank,
and evaluates each generated action at most once per context/bank.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from learning.action_diffusion import (
    cosine_alpha_bar_schedule,
    ddim_timestep_schedule,
    sample_ddim,
)
from learning.amortized_cem_data import AmortizedCemContextRecord, sha256_file
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
)
from run_milestone6a import _state_distances
from run_milestone7a1 import _load_environment, _physics_settings
from run_milestone7a3 import _load_model, _load_records, _safe, _write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config/learning/diffusion_candidate_support_v1.json"
ARTIFACT_PARENT = ROOT / "data/policy_training/diffusion_candidate_support_v1"
REPORT = ROOT / "MILESTONE7A5_CANDIDATE_SUPPORT_REPORT.md"

FLOAT_FIELDS = (
    "task_cost",
    "minimum_tip_target_distance_m",
    "best_event_time_s",
    "best_event_tip_distance_m",
    "best_event_directed_speed_m_s",
    "best_event_direction_angle_deg",
    "first_entry_time_s",
    "first_entry_tip_distance_m",
    "first_entry_directed_speed_m_s",
    "first_entry_direction_angle_deg",
    "maximum_uav_displacement_m",
    "maximum_uav_speed_m_s",
    "maximum_command_acceleration_m_s2",
)
BOOL_FIELDS = ("success", "feasible", "finite")
PREFIX_COUNTS = (32, 64, 128, 256, 512, 1024)
SEGMENT_NAMES = {0: "NONE", 1: "ACTIVE", 2: "SETTLE", 3: "HOLD"}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_config(config: dict[str, Any]) -> None:
    if config["schema"] != "diffusion_candidate_support_v1":
        raise ValueError("Unsupported Milestone 7A.5 configuration.")
    if config["model_freeze"] != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Production model freeze changed.")
    if tuple(config["candidate_counts"]) != PREFIX_COUNTS:
        raise ValueError("Candidate-count curve changed.")
    if ddim_timestep_schedule().tolist()[0] != 95:
        raise RuntimeError("Repaired t95->0 DDIM schedule is not active.")
    for key in (
        "new_cem_solves",
        "diffusion_training",
        "scorer_training",
        "dataset_generation",
        "architecture_change",
        "final_test",
        "hardware",
    ):
        if not bool(config["prohibitions"][key]):
            raise ValueError(f"Required prohibition disabled: {key}")
    if config["prohibitions"]["protected_test"] != "fig8vertical_002":
        raise ValueError("Protected-test identity changed.")


def _record_map(records: list[AmortizedCemContextRecord]) -> dict[str, AmortizedCemContextRecord]:
    return {record.context_id: record for record in records}


def _chunks_for_count(count: int) -> tuple[int, ...]:
    if count not in PREFIX_COUNTS and count != 512:
        raise ValueError(f"Unsupported nested candidate count: {count}")
    values = (32, 32, 64, 128, 256, 512)
    selected: list[int] = []
    total = 0
    for value in values:
        if total == count:
            break
        if total + value > count:
            raise RuntimeError("Candidate count is not represented by the prefix chunks.")
        selected.append(value)
        total += value
    if total != count:
        raise RuntimeError("Incomplete nested chunk plan.")
    return tuple(selected)


def _select_heterogeneous_records(
    records: list[AmortizedCemContextRecord], environment: tuple, *, count: int
) -> tuple[list[AmortizedCemContextRecord], dict[str, Any]]:
    _sim, _task, training_bank, _heldout, canonical_bank, _source = environment
    distance = _state_distances(training_bank, canonical_bank).numpy()
    train = [record for record in records if record.split == "TRAIN" and record.bank == "training"]
    # Exactly one existing target/context per physical state prevents a state with
    # multiple targets from receiving extra weight.
    by_state: dict[str, list[AmortizedCemContextRecord]] = {}
    for record in train:
        by_state.setdefault(record.state_id, []).append(record)
    one_per_state = [
        sorted(values, key=lambda row: hashlib.sha256(row.context_id.encode()).hexdigest())[0]
        for _state, values in sorted(by_state.items())
    ]
    values = np.asarray([distance[record.state_index] for record in one_per_state])
    q1, q2 = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    strata: dict[str, list[AmortizedCemContextRecord]] = {"LOW": [], "MEDIUM": [], "HIGH": []}
    for record in one_per_state:
        value = float(distance[record.state_index])
        name = "LOW" if value <= q1 else "MEDIUM" if value <= q2 else "HIGH"
        strata[name].append(record)
    quotas = {"LOW": 21, "MEDIUM": 22, "HIGH": 21}
    selected: list[AmortizedCemContextRecord] = []
    rows = []
    for name in ("LOW", "MEDIUM", "HIGH"):
        ordered = sorted(
            strata[name],
            key=lambda row: hashlib.sha256(f"7a5:{name}:{row.context_id}".encode()).hexdigest(),
        )
        chosen = ordered[: quotas[name]]
        selected.extend(chosen)
        rows.extend(
            {
                **asdict(record),
                "state_distance": float(distance[record.state_index]),
                "state_distance_stratum": name,
            }
            for record in chosen
        )
    if len(selected) != count or len({row.state_id for row in selected}) != count:
        raise RuntimeError("Heterogeneous-state selection is not 64 unique TRAIN states.")
    manifest = {
        "schema": "milestone7a5_heterogeneous_state_manifest_v1",
        "context_count": count,
        "selection": "one context per state; deterministic 21/22/21 low/medium/high state-distance strata",
        "state_distance_definition": (
            "Milestone-6A RMS of scaled UAV pose/twist and distributed cable position/velocity "
            "difference from canonical"
        ),
        "tertile_boundaries": [float(q1), float(q2)],
        "stratum_counts": quotas,
        "records": rows,
        "test_contexts": 0,
        "protected_contexts": 0,
    }
    return selected, manifest


def prepare(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    _validate_config(config)
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "independent_noise_banks").mkdir(exist_ok=True)
    (artifact / "figures").mkdir(exist_ok=True)
    baseline = ROOT / config["baseline_artifact"]
    seven_a4 = ROOT / config["milestone_7a4_artifact"]
    audit = _load_json(seven_a4 / "candidate_count_audit.json")
    if not audit.get("nested_generated_actions_bit_identical"):
        raise RuntimeError("7A.4 source audit is not the corrected literal-action audit.")
    records, _contexts = _load_records(baseline)
    mapping = _record_map(records)
    primary_ids = list(audit["context_ids"])
    if len(primary_ids) != 64 or any(value not in mapping for value in primary_ids):
        raise RuntimeError("Exact 7A.4 primary context set is unavailable.")
    primary_records = [mapping[value] for value in primary_ids]
    expected = {
        int(row["candidate_count"]): row["overall"]["oracle_best_of_32_success_rate"]
        for row in audit["rows"]
        if int(row["candidate_count"]) <= 128
    }
    primary_manifest = {
        "schema": "milestone7a5_primary_development_manifest_v1",
        "name": "PRIMARY_DEVELOPMENT_SET",
        "source": str(seven_a4 / "candidate_count_audit.json"),
        "context_count": len(primary_records),
        "records": [asdict(record) for record in primary_records],
        "ownership_counts": {
            split: sum(record.split == split for record in primary_records)
            for split in sorted({record.split for record in primary_records})
        },
        "expected_reproduction": expected,
    }
    _write_json(artifact / "primary_development_manifest.json", primary_manifest)

    environment = _load_environment(config)
    heterogeneous, heterogeneous_manifest = _select_heterogeneous_records(
        records, environment, count=int(config["heterogeneous_context_count"])
    )
    _write_json(artifact / "heterogeneous_state_manifest.json", heterogeneous_manifest)

    old_bank = np.load(seven_a4 / "candidate_count_noise_bank_128.npy").astype(np.float32)
    if old_bank.shape != (128, 49):
        raise RuntimeError("The exact corrected 7A.4 128-noise bank is unavailable.")
    primary_bank = np.random.default_rng(int(config["primary_extension_seed"])).standard_normal(
        (1024, 49)
    ).astype(np.float32)
    primary_bank[:128] = old_bank
    np.save(artifact / "primary_noise_bank_1024.npy", primary_bank)
    if not np.array_equal(np.load(artifact / "primary_noise_bank_1024.npy")[:128], old_bank):
        raise RuntimeError("The persisted primary bank did not preserve the exact first 128 vectors.")
    independent = {}
    for seed in config["independent_noise_seeds"]:
        bank = np.random.default_rng(int(seed)).standard_normal((512, 49)).astype(np.float32)
        path = artifact / "independent_noise_banks" / f"noise_bank_seed_{seed}_512.npy"
        np.save(path, bank)
        independent[str(seed)] = {"path": str(path), "sha256": sha256_file(path)}

    generator = _load_model(baseline, torch.device("cuda"))
    manifest = {
        "schema": "milestone7a5_baseline_generator_manifest_v1",
        "name": "BASELINE_FLAT_CONDITIONING",
        "checkpoint": str(baseline / "full_diffusion_ema_best.pt"),
        "checkpoint_sha256": sha256_file(baseline / "full_diffusion_ema_best.pt"),
        "trainable_parameters": sum(value.numel() for value in generator.parameters()),
        "context_dimension": 83,
        "action_dimension": 49,
        "ddim_schedule": ddim_timestep_schedule().tolist(),
        "ddim_eta": 0.0,
        "model_freeze": config["model_freeze"],
        "maneuver_duration_s": [0.45, 1.80],
        "settle_duration_s": 0.30,
        "evaluation_horizon_s": 2.40,
        "fixed_physics_batch": 2048,
        "structured_checkpoint_used": False,
        "diffusion_training": "NONE",
        "scorer_used": False,
        "new_cem_solves": 0,
        "primary_noise_bank": {
            "path": str(artifact / "primary_noise_bank_1024.npy"),
            "sha256": sha256_file(artifact / "primary_noise_bank_1024.npy"),
            "first_128_source": str(seven_a4 / "candidate_count_noise_bank_128.npy"),
            "first_128_exact": True,
            "extension_seed": int(config["primary_extension_seed"]),
        },
        "independent_banks": independent,
    }
    _write_json(artifact / "baseline_generator_manifest.json", manifest)
    _write_json(artifact / "run_config.json", config)
    return {
        "primary_records": primary_records,
        "heterogeneous_records": heterogeneous,
        "expected": expected,
    }


@torch.no_grad()
def generate_actions(
    model: torch.nn.Module,
    normalizer: FixedContextNormalizer,
    contexts: np.ndarray,
    records: list[AmortizedCemContextRecord],
    noise: np.ndarray,
    *,
    chunk_sizes: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    if sum(chunk_sizes) != noise.shape[0]:
        raise ValueError("Noise rows and generation chunks disagree.")
    device = next(model.parameters()).device
    alpha_bar = cosine_alpha_bar_schedule().to(device)
    schedule = ddim_timestep_schedule().to(device)
    actions = np.empty((len(records), noise.shape[0], 49), dtype=np.float32)
    outside = 0
    total = 0
    for record_index, record in enumerate(records):
        raw_context = torch.from_numpy(contexts[record.context_index][None]).to(device)
        context = normalizer.normalize(raw_context)
        offset = 0
        pieces = []
        for size in chunk_sizes:
            values = torch.from_numpy(noise[offset : offset + size]).to(device)
            sample = sample_ddim(
                model,
                context,
                values,
                alpha_bar=alpha_bar,
                timestep_schedule=schedule,
            )
            pieces.append(sample.bounded_action.cpu())
            outside += int((sample.raw_action.abs() > 1.0).sum())
            total += sample.raw_action.numel()
            offset += size
        actions[record_index] = torch.cat(pieces).numpy()
        if (record_index + 1) % 16 == 0 or record_index + 1 == len(records):
            print(f"generated {record_index + 1}/{len(records)} contexts x {noise.shape[0]}", flush=True)
    return actions, {
        "context_count": len(records),
        "candidate_count": int(noise.shape[0]),
        "chunk_sizes": list(chunk_sizes),
        "clamp_fraction": outside / total,
    }


def _empty_outcomes(context_count: int, candidate_count: int) -> dict[str, np.ndarray]:
    shape = (context_count, candidate_count)
    output = {name: np.empty(shape, dtype=np.float32) for name in FLOAT_FIELDS}
    output.update({name: np.empty(shape, dtype=bool) for name in BOOL_FIELDS})
    output["first_entry_marker"] = np.empty(shape, dtype=np.int16)
    output["maneuver_duration_s"] = np.empty(shape, dtype=np.float32)
    output["hit_segment"] = np.empty(shape, dtype=np.int8)
    return output


@torch.no_grad()
def evaluate_actions(
    environment: tuple,
    config: dict[str, Any],
    records: list[AmortizedCemContextRecord],
    actions: np.ndarray,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    simulator, task, training_bank, _heldout, canonical_bank, _source = environment
    settings = _physics_settings(config)
    banks = {"training": training_bank, "canonical": canonical_bank}
    count = int(actions.shape[1])
    output = _empty_outcomes(len(records), count)
    index_by_id = {record.context_id: index for index, record in enumerate(records)}
    for bank_name, bank in banks.items():
        selected = [record for record in records if record.bank == bank_name]
        per_batch = max(1, FIXED_NUMERICAL_BATCH_SIZE // count)
        for start in range(0, len(selected), per_batch):
            group = selected[start : start + per_batch]
            group_indices = [index_by_id[record.context_id] for record in group]
            action = torch.from_numpy(actions[group_indices].reshape(-1, 49))
            state_indices, targets, directions = [], [], []
            for record in group:
                state_indices.extend([record.state_index] * count)
                targets.extend([record.target_local_m] * count)
                directions.extend([record.direction_local] * count)
            specification = ContextSpecification(
                torch.tensor(state_indices, dtype=torch.int64),
                torch.tensor(targets, dtype=torch.float32),
                torch.tensor(directions, dtype=torch.float32),
                label,
            )
            context = build_context_from_specification(
                simulator,
                bank,
                pad_context_specification(specification, FIXED_NUMERICAL_BATCH_SIZE),
            )
            result = evaluate_normalized_actions_fixed_batch(
                simulator, context, action, task, settings
            )
            for field in FLOAT_FIELDS + BOOL_FIELDS:
                value = getattr(result, field).detach().cpu().reshape(len(group), count).numpy()
                output[field][group_indices] = value
            markers = result.first_entry_marker.detach().cpu().reshape(len(group), count).numpy()
            output["first_entry_marker"][group_indices] = markers
            duration = decode_policy_action(
                action.float(), task, duration_max_s=settings.duration_max_s
            ).duration_s.reshape(len(group), count).numpy()
            output["maneuver_duration_s"][group_indices] = duration
            entry = output["first_entry_time_s"][group_indices]
            segments = np.zeros(entry.shape, dtype=np.int8)
            finite = np.isfinite(entry)
            segments[finite & (entry <= duration + 1.0e-6)] = 1
            segments[
                finite
                & (entry > duration + 1.0e-6)
                & (entry <= duration + settings.settle_duration_s + 1.0e-6)
            ] = 2
            segments[finite & (entry > duration + settings.settle_duration_s + 1.0e-6)] = 3
            output["hit_segment"][group_indices] = segments
            print(
                f"{label} physics {min(start + len(group), len(selected))}/{len(selected)} {bank_name}",
                flush=True,
            )
    return output


def _save_outcomes(path: Path, outcomes: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, schema=np.asarray("milestone7a5_candidate_outcomes_v1"), **outcomes)


def _load_outcomes(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files if name != "schema"}


def _prefix_summary(outcomes: dict[str, np.ndarray], count: int) -> dict[str, Any]:
    selected = {name: value[:, :count] for name, value in outcomes.items()}
    success = selected["success"]
    feasible = selected["feasible"]
    distance = selected["best_event_tip_distance_m"] <= 0.050
    speed = selected["best_event_directed_speed_m_s"] >= 4.0
    direction = selected["best_event_direction_angle_deg"] <= 30.0
    tip_first = selected["first_entry_marker"] == 10
    oracle_rows: list[int] = []
    oracle_indices: list[int] = []
    for row in range(success.shape[0]):
        valid = np.flatnonzero(success[row])
        if valid.size:
            oracle_rows.append(row)
            oracle_indices.append(int(valid[np.argmin(selected["task_cost"][row, valid])]))
    metrics = {}
    for field in (
        "first_entry_tip_distance_m",
        "first_entry_directed_speed_m_s",
        "first_entry_direction_angle_deg",
        "maximum_uav_displacement_m",
        "maximum_uav_speed_m_s",
        "maximum_command_acceleration_m_s2",
        "maneuver_duration_s",
        "first_entry_time_s",
    ):
        values = np.asarray(
            [selected[field][row, index] for row, index in zip(oracle_rows, oracle_indices)]
        )
        finite_values = values[np.isfinite(values)]
        metrics[f"median_{field}"] = float(np.median(finite_values)) if finite_values.size else None
    segments = [
        SEGMENT_NAMES[int(selected["hit_segment"][row, index])]
        for row, index in zip(oracle_rows, oracle_indices)
    ]
    return {
        "candidate_count": count,
        "context_count": int(success.shape[0]),
        "oracle_success_rate": float(success.any(axis=1).mean()),
        "contexts_with_feasible_candidate_rate": float(feasible.any(axis=1).mean()),
        "mean_candidate_level_feasibility": float(feasible.mean()),
        "contexts_with_distance_pass_rate": float(distance.any(axis=1).mean()),
        "contexts_with_directed_speed_pass_rate": float(speed.any(axis=1).mean()),
        "contexts_with_direction_pass_rate": float(direction.any(axis=1).mean()),
        "contexts_with_tip_first_rate": float(tip_first.any(axis=1).mean()),
        "oracle_metrics": metrics,
        "oracle_hit_segment_counts": {
            name: segments.count(name) for name in ("ACTIVE", "SETTLE", "HOLD", "NONE")
        },
    }


def _curve(outcomes: dict[str, np.ndarray], counts: tuple[int, ...] = PREFIX_COUNTS) -> list[dict[str, Any]]:
    return [_prefix_summary(outcomes, count) for count in counts]


def reproduce(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    prepared = prepare(config, artifact)
    baseline = ROOT / config["baseline_artifact"]
    records, contexts = _load_records(baseline)
    normalizer = FixedContextNormalizer.load(baseline / "context_normalizer.json")
    model = _load_model(baseline, torch.device("cuda"))
    environment = _load_environment(config)
    bank = np.load(artifact / "primary_noise_bank_1024.npy")[:128]
    actions, generation = generate_actions(
        model,
        normalizer,
        contexts,
        prepared["primary_records"],
        bank,
        chunk_sizes=(32, 32, 64),
    )
    outcomes = evaluate_actions(
        environment,
        config,
        prepared["primary_records"],
        actions,
        label="7a5_reproduction",
    )
    curve = _curve(outcomes, (32, 64, 128))
    observed = {row["candidate_count"]: row["oracle_success_rate"] for row in curve}
    expected = {int(key): float(value) for key, value in prepared["expected"].items()}
    differences = {str(key): observed[key] - expected[key] for key in expected}
    passed = all(abs(value) <= 1.0e-12 for value in differences.values())
    result = {
        "schema": "milestone7a5_reproduction_gate_v1",
        "expected": expected,
        "observed": observed,
        "differences": differences,
        "pass": passed,
        "classification_if_failed": "CANDIDATE_AUDIT_REPRODUCTION_FAILURE",
        "generation": generation,
    }
    np.savez_compressed(
        artifact / "primary_reproduction_actions_128.npz",
        schema=np.asarray("milestone7a5_normalized_actions_v1"),
        actions=actions,
    )
    _save_outcomes(artifact / "primary_reproduction_outcomes_128.npz", outcomes)
    _write_json(artifact / "reproduction_gate.json", result)
    if not passed:
        raise RuntimeError("CANDIDATE_AUDIT_REPRODUCTION_FAILURE")
    return result


def _combine_outcomes(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.concatenate((first[name], second[name]), axis=1) for name in first}


def _density_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _first_success_summary(outcomes: dict[str, np.ndarray]) -> dict[str, Any]:
    success = outcomes["success"]
    indices = np.asarray(
        [np.flatnonzero(row)[0] + 1 for row in success if np.any(row)], dtype=np.int64
    )
    return {
        "contexts_with_success": int(indices.size),
        "contexts_without_success": int(success.shape[0] - indices.size),
        "median": float(np.median(indices)) if indices.size else None,
        "p75": float(np.percentile(indices, 75)) if indices.size else None,
        "p90": float(np.percentile(indices, 90)) if indices.size else None,
        "p95": float(np.percentile(indices, 95)) if indices.size else None,
        "maximum": int(indices.max()) if indices.size else None,
        "prefix_fractions_all_contexts": {
            str(count): float(np.mean(success[:, :count].any(axis=1))) for count in PREFIX_COUNTS
        },
        "first_success_indices_one_based": indices.tolist(),
    }


def _failure_rows(
    records: list[AmortizedCemContextRecord], outcomes: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        if outcomes["success"][index].any():
            continue
        best = int(np.argmin(outcomes["task_cost"][index]))
        rows.append(
            {
                "context_id": record.context_id,
                "state_id": record.state_id,
                "target_local_m": list(record.target_local_m),
                "best_candidate_index_one_based": best + 1,
                "best_tip_distance_m": float(outcomes["best_event_tip_distance_m"][index, best]),
                "best_directed_speed_m_s": float(
                    outcomes["best_event_directed_speed_m_s"][index, best]
                ),
                "best_direction_error_deg": float(
                    outcomes["best_event_direction_angle_deg"][index, best]
                ),
                "best_uav_displacement_m": float(
                    outcomes["maximum_uav_displacement_m"][index, best]
                ),
                "best_uav_speed_m_s": float(outcomes["maximum_uav_speed_m_s"][index, best]),
                "best_command_acceleration_m_s2": float(
                    outcomes["maximum_command_acceleration_m_s2"][index, best]
                ),
                "best_tip_first": bool(outcomes["first_entry_marker"][index, best] == 10),
                "no_candidate_passes": {
                    "distance": not bool(
                        np.any(outcomes["best_event_tip_distance_m"][index] <= 0.050)
                    ),
                    "directed_speed": not bool(
                        np.any(outcomes["best_event_directed_speed_m_s"][index] >= 4.0)
                    ),
                    "direction": not bool(
                        np.any(outcomes["best_event_direction_angle_deg"][index] <= 30.0)
                    ),
                    "tip_first": not bool(np.any(outcomes["first_entry_marker"][index] == 10)),
                    "feasible": not bool(np.any(outcomes["feasible"][index])),
                },
            }
        )
    return rows


def primary_and_state(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    gate = _load_json(artifact / "reproduction_gate.json")
    if not gate["pass"]:
        raise RuntimeError("Reproduction gate did not pass.")
    baseline = ROOT / config["baseline_artifact"]
    records, contexts = _load_records(baseline)
    mapping = _record_map(records)
    primary_manifest = _load_json(artifact / "primary_development_manifest.json")
    primary_records = [mapping[row["context_id"]] for row in primary_manifest["records"]]
    heterogeneous_manifest = _load_json(artifact / "heterogeneous_state_manifest.json")
    heterogeneous_records = [mapping[row["context_id"]] for row in heterogeneous_manifest["records"]]
    normalizer = FixedContextNormalizer.load(baseline / "context_normalizer.json")
    model = _load_model(baseline, torch.device("cuda"))
    environment = _load_environment(config)
    bank = np.load(artifact / "primary_noise_bank_1024.npy")
    with np.load(artifact / "primary_reproduction_actions_128.npz", allow_pickle=False) as archive:
        first_actions = np.asarray(archive["actions"], dtype=np.float32)
    first_outcomes = _load_outcomes(artifact / "primary_reproduction_outcomes_128.npz")
    remaining_actions, remaining_generation = generate_actions(
        model,
        normalizer,
        contexts,
        primary_records,
        bank[128:],
        chunk_sizes=(128, 256, 512),
    )
    remaining_outcomes = evaluate_actions(
        environment,
        config,
        primary_records,
        remaining_actions,
        label="7a5_primary_remaining_896",
    )
    primary_actions = np.concatenate((first_actions, remaining_actions), axis=1)
    primary_outcomes = _combine_outcomes(first_outcomes, remaining_outcomes)
    np.savez_compressed(
        artifact / "primary_development_candidate_actions.npz",
        schema=np.asarray("milestone7a5_normalized_actions_v1"),
        actions=primary_actions,
    )
    _save_outcomes(artifact / "primary_development_candidate_outcomes.npz", primary_outcomes)

    heterogeneous_actions, heterogeneous_generation = generate_actions(
        model,
        normalizer,
        contexts,
        heterogeneous_records,
        bank,
        chunk_sizes=_chunks_for_count(1024),
    )
    heterogeneous_outcomes = evaluate_actions(
        environment,
        config,
        heterogeneous_records,
        heterogeneous_actions,
        label="7a5_heterogeneous_1024",
    )
    np.savez_compressed(
        artifact / "heterogeneous_candidate_actions.npz",
        schema=np.asarray("milestone7a5_normalized_actions_v1"),
        actions=heterogeneous_actions,
    )
    _save_outcomes(artifact / "heterogeneous_candidate_outcomes.npz", heterogeneous_outcomes)

    primary_curve = _curve(primary_outcomes)
    heterogeneous_curve = _curve(heterogeneous_outcomes)
    curve = {
        "schema": "milestone7a5_candidate_count_curve_v1",
        "prefixes_are_literal_action_supersets": True,
        "physics_evaluated_once_per_candidate": True,
        "primary": primary_curve,
        "heterogeneous_state": heterogeneous_curve,
        "generation": {
            "primary_remaining": remaining_generation,
            "heterogeneous": heterogeneous_generation,
        },
    }
    _write_json(artifact / "candidate_count_curve.json", curve)
    first = {
        "schema": "milestone7a5_first_success_index_summary_v1",
        "primary": _first_success_summary(primary_outcomes),
        "heterogeneous_state": _first_success_summary(heterogeneous_outcomes),
    }
    _write_json(artifact / "first_success_index_summary.json", first)
    density = {
        "schema": "milestone7a5_success_density_summary_v1",
        "primary": {
            "successful_candidate_count": _density_summary(primary_outcomes["success"].sum(axis=1)),
            "successful_candidate_fraction": _density_summary(primary_outcomes["success"].mean(axis=1)),
            "feasible_candidate_count": _density_summary(primary_outcomes["feasible"].sum(axis=1)),
            "feasible_candidate_fraction": _density_summary(primary_outcomes["feasible"].mean(axis=1)),
        },
        "heterogeneous_state": {
            "successful_candidate_count": _density_summary(
                heterogeneous_outcomes["success"].sum(axis=1)
            ),
            "successful_candidate_fraction": _density_summary(
                heterogeneous_outcomes["success"].mean(axis=1)
            ),
            "feasible_candidate_count": _density_summary(
                heterogeneous_outcomes["feasible"].sum(axis=1)
            ),
            "feasible_candidate_fraction": _density_summary(
                heterogeneous_outcomes["feasible"].mean(axis=1)
            ),
        },
    }
    _write_json(artifact / "success_density_summary.json", density)
    _write_json(
        artifact / "gate_support_summary.json",
        {
            "schema": "milestone7a5_gate_support_summary_v1",
            "primary": primary_curve,
            "heterogeneous_state": heterogeneous_curve,
        },
    )
    failures = {
        "schema": "milestone7a5_failure_contexts_1024_v1",
        "primary": _failure_rows(primary_records, primary_outcomes),
        "heterogeneous_state": _failure_rows(heterogeneous_records, heterogeneous_outcomes),
    }
    _write_json(artifact / "failure_contexts_1024.json", failures)
    return curve


def robustness(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    baseline = ROOT / config["baseline_artifact"]
    records, contexts = _load_records(baseline)
    mapping = _record_map(records)
    manifest = _load_json(artifact / "primary_development_manifest.json")
    selected = [mapping[row["context_id"]] for row in manifest["records"]]
    normalizer = FixedContextNormalizer.load(baseline / "context_normalizer.json")
    model = _load_model(baseline, torch.device("cuda"))
    environment = _load_environment(config)
    primary = _load_outcomes(artifact / "primary_development_candidate_outcomes.npz")
    bank_success = {"PRIMARY_REFERENCE": primary["success"][:, :512]}
    rows = [
        {
            "bank": "PRIMARY_REFERENCE",
            "seed": int(config["primary_extension_seed"]),
            "curve": _curve(primary, (32, 64, 128, 256, 512)),
        }
    ]
    for seed in config["independent_noise_seeds"]:
        path = artifact / "independent_noise_banks" / f"noise_bank_seed_{seed}_512.npy"
        bank = np.load(path)
        actions, generation = generate_actions(
            model,
            normalizer,
            contexts,
            selected,
            bank,
            chunk_sizes=_chunks_for_count(512),
        )
        outcomes = evaluate_actions(
            environment, config, selected, actions, label=f"7a5_bank_{seed}_512"
        )
        _save_outcomes(
            artifact / "independent_noise_banks" / f"outcomes_seed_{seed}_512.npz",
            outcomes,
        )
        bank_success[str(seed)] = outcomes["success"]
        rows.append(
            {
                "bank": f"INDEPENDENT_{seed}",
                "seed": int(seed),
                "generation": generation,
                "curve": _curve(outcomes, (32, 64, 128, 256, 512)),
            }
        )
    aggregates = []
    for count in (32, 64, 128, 256, 512):
        rates = np.asarray(
            [
                next(item for item in row["curve"] if item["candidate_count"] == count)[
                    "oracle_success_rate"
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        union = np.logical_or.reduce(
            [value[:, :count].any(axis=1) for value in bank_success.values()]
        )
        aggregates.append(
            {
                "candidate_count": count,
                "mean_oracle_success": float(rates.mean()),
                "std_oracle_success": float(rates.std(ddof=0)),
                "minimum_oracle_success": float(rates.min()),
                "maximum_oracle_success": float(rates.max()),
                "range_oracle_success": float(rates.max() - rates.min()),
                "union_oracle_success_analysis_only": float(union.mean()),
            }
        )
    result = {
        "schema": "milestone7a5_noise_bank_robustness_v1",
        "bank_count": 4,
        "banks_precommitted_before_physics": True,
        "best_bank_not_selected": True,
        "rows_by_bank": rows,
        "aggregates": aggregates,
    }
    _write_json(artifact / "noise_bank_robustness.json", result)

    primary_curve = _load_json(artifact / "candidate_count_curve.json")["primary"]
    at_512 = next(row for row in primary_curve if row["candidate_count"] == 512)
    at_1024 = next(row for row in primary_curve if row["candidate_count"] == 1024)
    if at_1024["oracle_success_rate"] >= 0.90 and at_512["oracle_success_rate"] < 0.90:
        seed = int(config["conditional_confirmation_seed"])
        bank = np.random.default_rng(seed).standard_normal((1024, 49)).astype(np.float32)
        path = artifact / "independent_noise_banks" / f"conditional_confirmation_{seed}_1024.npy"
        np.save(path, bank)
        actions, generation = generate_actions(
            model,
            normalizer,
            contexts,
            selected,
            bank,
            chunk_sizes=_chunks_for_count(1024),
        )
        outcomes = evaluate_actions(
            environment, config, selected, actions, label="7a5_conditional_confirmation_1024"
        )
        confirmation = {
            "run": True,
            "seed": seed,
            "bank_sha256": sha256_file(path),
            "generation": generation,
            "curve": _curve(outcomes),
        }
    else:
        confirmation = {
            "run": False,
            "reason": "conditional criterion (primary N1024>=90% and N512<90%) not met",
        }
    result["conditional_1024_confirmation"] = confirmation
    _write_json(artifact / "noise_bank_robustness.json", result)
    return result


@torch.no_grad()
def _latency_for_count(
    model: torch.nn.Module,
    normalizer: FixedContextNormalizer,
    raw_context: np.ndarray,
    noise: np.ndarray,
    *,
    count: int,
    warmup: int,
    queries: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    alpha_bar = cosine_alpha_bar_schedule().to(device)
    schedule = ddim_timestep_schedule().to(device)
    chunks = _chunks_for_count(count)
    bank = torch.from_numpy(noise[:count]).to(device)

    def query() -> None:
        context = normalizer.normalize(torch.from_numpy(raw_context[None]).to(device))
        pieces = []
        offset = 0
        for size in chunks:
            sample = sample_ddim(
                model,
                context,
                bank[offset : offset + size],
                alpha_bar=alpha_bar,
                timestep_schedule=schedule,
            )
            pieces.append(sample.bounded_action)
            offset += size
        _ = torch.cat(pieces).cpu()

    for _ in range(warmup):
        query()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    wall, gpu = [], []
    for _ in range(queries):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        query()
        end_event.record()
        torch.cuda.synchronize()
        wall.append(1000.0 * (time.perf_counter() - start))
        gpu.append(float(start_event.elapsed_time(end_event)))
    allocated = int(torch.cuda.max_memory_allocated(device))
    reserved = int(torch.cuda.max_memory_reserved(device))

    def stats(values: list[float]) -> dict[str, float]:
        return {
            "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "p95_ms": float(np.percentile(values, 95)),
            "maximum_ms": float(np.max(values)),
        }

    return {
        "candidate_count": count,
        "queries": queries,
        "warmup_queries": warmup,
        "chunk_sizes": list(chunks),
        "includes_context_normalization_ddim_and_bounded_preparation": True,
        "wall": stats(wall),
        "gpu": stats(gpu),
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
    }


def _recommend_and_classify(
    curve: dict[str, Any], robustness_result: dict[str, Any], latency_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    primary = {row["candidate_count"]: row for row in curve["primary"]}
    state = {row["candidate_count"]: row for row in curve["heterogeneous_state"]}
    robust = {row["candidate_count"]: row for row in robustness_result["aggregates"]}
    sufficient_counts = [
        count for count in PREFIX_COUNTS if primary[count]["oracle_success_rate"] >= 0.90
    ]
    if sufficient_counts:
        recommended = min(sufficient_counts)
    else:
        gain = primary[1024]["oracle_success_rate"] - primary[512]["oracle_success_rate"]
        recommended = 1024 if gain >= 0.025 else 512
    latency = {row["candidate_count"]: row for row in latency_rows}[recommended]
    # Small banks expose sampling variance, but deployment-relevant bank
    # sensitivity is judged at the largest independently replicated prefix.
    # This prevents an expected N=32 fluctuation from masking a persistent
    # state-coverage limitation at N=512/1024.
    robust_512 = robust[512]
    bank_sensitive = bool(
        robust_512["range_oracle_success"] >= 0.15
        or robust_512["std_oracle_success"] >= 0.075
    )
    primary_strong = primary[recommended]["oracle_success_rate"] >= 0.80
    state_gap = (
        primary[recommended]["oracle_success_rate"]
        - state[recommended]["oracle_success_rate"]
    )
    state_limited = bool(primary_strong and state_gap >= 0.15)
    immediate_latency = latency["wall"]["p95_ms"] < 100.0
    not_anomaly = robust[min(recommended, 512)]["minimum_oracle_success"] >= 0.75
    if state_limited:
        classification = "STATE_SUPPORT_LIMITED"
    elif bank_sensitive:
        classification = "NOISE_BANK_SENSITIVE"
    elif primary_strong and not_anomaly and immediate_latency:
        classification = "CANDIDATE_SUPPORT_SUFFICIENT"
    else:
        gain = primary[1024]["oracle_success_rate"] - primary[512]["oracle_success_rate"]
        substantial = primary[1024]["oracle_success_rate"] >= 0.70
        classification = (
            "CANDIDATE_SUPPORT_PROMISING"
            if substantial and gain >= 0.025
            else "GENERATOR_SUPPORT_PLATEAU"
        )
    return {
        "schema": "milestone7a5_recommended_candidate_count_v1",
        "recommended_candidate_count": recommended,
        "selection_rule": (
            "smallest N reaching 90% primary oracle; otherwise N=1024 only when "
            "512->1024 adds at least 2.5 percentage points"
        ),
        "classification": classification,
        "primary_oracle_at_recommended": primary[recommended]["oracle_success_rate"],
        "heterogeneous_oracle_at_recommended": state[recommended]["oracle_success_rate"],
        "bank_sensitive": bank_sensitive,
        "primary_minus_heterogeneous_oracle_gap": state_gap,
        "bank_variability_interpretation": (
            "small-N variability is reported, but N=512 variability does not "
            "materially change the large-N conclusion"
        ),
        "latency_compatible": immediate_latency,
        "scorer_training_justified": classification == "CANDIDATE_SUPPORT_SUFFICIENT",
        "more_cem_data_justified": False,
        "production_candidate_count_changed": False,
    }


def latency_and_decision(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    baseline = ROOT / config["baseline_artifact"]
    records, contexts = _load_records(baseline)
    mapping = _record_map(records)
    manifest = _load_json(artifact / "primary_development_manifest.json")
    selected = [mapping[row["context_id"]] for row in manifest["records"]]
    normalizer = FixedContextNormalizer.load(baseline / "context_normalizer.json")
    model = _load_model(baseline, torch.device("cuda"))
    bank = np.load(artifact / "primary_noise_bank_1024.npy")
    rows = []
    for count in PREFIX_COUNTS:
        row = _latency_for_count(
            model,
            normalizer,
            contexts[selected[0].context_index],
            bank,
            count=count,
            warmup=int(config["latency_warmup_queries"]),
            queries=int(config["latency_queries_per_count"]),
        )
        rows.append(row)
        print(
            f"latency N={count} median={row['wall']['median_ms']:.2f}ms "
            f"p95={row['wall']['p95_ms']:.2f}ms",
            flush=True,
        )
    curve = _load_json(artifact / "candidate_count_curve.json")
    robust = _load_json(artifact / "noise_bank_robustness.json")
    decision = _recommend_and_classify(curve, robust, rows)
    recommended = int(decision["recommended_candidate_count"])
    precise = _latency_for_count(
        model,
        normalizer,
        contexts[selected[0].context_index],
        bank,
        count=recommended,
        warmup=int(config["latency_warmup_queries"]),
        queries=int(config["latency_precise_queries"]),
    )
    decision["precise_latency"] = precise
    _write_json(
        artifact / "latency_by_candidate_count.json",
        {
            "schema": "milestone7a5_latency_by_candidate_count_v1",
            "rows": rows,
            "precise_recommended": precise,
            "simulator_included": False,
            "scorer_included": False,
        },
    )
    _write_json(artifact / "recommended_candidate_count.json", decision)
    return decision


def figures(artifact: Path) -> None:
    curve = _load_json(artifact / "candidate_count_curve.json")
    counts = [row["candidate_count"] for row in curve["primary"]]
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(counts, [100 * row["oracle_success_rate"] for row in curve["primary"]], "o-", label="Primary development")
    plt.plot(counts, [100 * row["oracle_success_rate"] for row in curve["heterogeneous_state"]], "s-", label="Heterogeneous state")
    plt.xscale("log", base=2)
    plt.xticks(counts, counts)
    plt.xlabel("Candidate count N")
    plt.ylabel("Oracle scientific success (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact / "figures/oracle_success_vs_candidate_count.png", dpi=180)
    plt.close()

    latency = _load_json(artifact / "latency_by_candidate_count.json")
    rows = latency["rows"]
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(
        [row["candidate_count"] for row in rows],
        [row["wall"]["median_ms"] for row in rows],
        "o-",
        label="Median wall",
    )
    plt.plot(
        [row["candidate_count"] for row in rows],
        [row["wall"]["p95_ms"] for row in rows],
        "s--",
        label="P95 wall",
    )
    plt.xscale("log", base=2)
    plt.xticks(counts, counts)
    plt.xlabel("Candidate count N")
    plt.ylabel("Generator query latency (ms)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact / "figures/latency_vs_candidate_count.png", dpi=180)
    plt.close()

    robust = _load_json(artifact / "noise_bank_robustness.json")
    aggregate = robust["aggregates"]
    x = [row["candidate_count"] for row in aggregate]
    mean = np.asarray([100 * row["mean_oracle_success"] for row in aggregate])
    low = np.asarray([100 * row["minimum_oracle_success"] for row in aggregate])
    high = np.asarray([100 * row["maximum_oracle_success"] for row in aggregate])
    plt.figure(figsize=(6.4, 4.2))
    plt.plot(x, mean, "o-", label="Four-bank mean")
    plt.fill_between(x, low, high, alpha=0.25, label="Bank min–max")
    plt.xscale("log", base=2)
    plt.xticks(x, x)
    plt.xlabel("Candidate count N")
    plt.ylabel("Primary oracle success (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifact / "figures/noise_bank_variation.png", dpi=180)
    plt.close()


def _margin_summary(outcomes: dict[str, np.ndarray], count: int) -> dict[str, Any]:
    success = outcomes["success"][:, :count]
    values = {name: [] for name in ("tip_m", "speed_m_s", "direction_deg", "displacement_m", "uav_speed_m_s", "command_accel_m_s2")}
    for row in range(success.shape[0]):
        valid = np.flatnonzero(success[row])
        if not valid.size:
            continue
        index = int(valid[np.argmin(outcomes["task_cost"][row, valid])])
        values["tip_m"].append(0.050 - float(outcomes["first_entry_tip_distance_m"][row, index]))
        values["speed_m_s"].append(float(outcomes["first_entry_directed_speed_m_s"][row, index]) - 4.0)
        values["direction_deg"].append(30.0 - float(outcomes["first_entry_direction_angle_deg"][row, index]))
        values["displacement_m"].append(0.50 - float(outcomes["maximum_uav_displacement_m"][row, index]))
        values["uav_speed_m_s"].append(3.0 - float(outcomes["maximum_uav_speed_m_s"][row, index]))
        values["command_accel_m_s2"].append(20.0 - float(outcomes["maximum_command_acceleration_m_s2"][row, index]))
    return {
        name: {
            "median": float(np.median(items)),
            "p05": float(np.percentile(items, 5)),
        }
        for name, items in values.items()
        if items
    }


@torch.no_grad()
def _chunk_equivalence(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    baseline = ROOT / config["baseline_artifact"]
    records, contexts = _load_records(baseline)
    mapping = _record_map(records)
    manifest = _load_json(artifact / "primary_development_manifest.json")
    record = mapping[manifest["records"][0]["context_id"]]
    normalizer = FixedContextNormalizer.load(baseline / "context_normalizer.json")
    model = _load_model(baseline, torch.device("cuda"))
    device = next(model.parameters()).device
    context = normalizer.normalize(
        torch.from_numpy(contexts[record.context_index][None]).to(device)
    )
    noise = torch.from_numpy(np.load(artifact / "primary_noise_bank_1024.npy")[:128]).to(device)
    alpha_bar = cosine_alpha_bar_schedule().to(device)
    schedule = ddim_timestep_schedule().to(device)
    whole = sample_ddim(
        model, context, noise, alpha_bar=alpha_bar, timestep_schedule=schedule
    )
    raw_parts, bounded_parts = [], []
    offset = 0
    for size in (32, 32, 64):
        part = sample_ddim(
            model,
            context,
            noise[offset : offset + size],
            alpha_bar=alpha_bar,
            timestep_schedule=schedule,
        )
        raw_parts.append(part.raw_action)
        bounded_parts.append(part.bounded_action)
        offset += size
    raw_chunked = torch.cat(raw_parts)
    bounded_chunked = torch.cat(bounded_parts)
    raw_delta = (whole.raw_action - raw_chunked).abs()
    bounded_delta = (whole.bounded_action - bounded_chunked).abs()
    result = {
        "schema": "milestone7a5_chunk_equivalence_v1",
        "context_id": record.context_id,
        "candidate_count": 128,
        "whole_batch": [128],
        "chunked_batch": [32, 32, 64],
        "noise_order_identical": True,
        "raw_max_abs_difference": float(raw_delta.max().item()),
        "raw_mean_abs_difference": float(raw_delta.mean().item()),
        "bounded_max_abs_difference": float(bounded_delta.max().item()),
        "bounded_mean_abs_difference": float(bounded_delta.mean().item()),
        "tolerance": 2.0e-5,
        "pass": bool(bounded_delta.max().item() <= 2.0e-5),
    }
    _write_json(artifact / "chunk_equivalence.json", result)
    if not result["pass"]:
        raise RuntimeError("Candidate chunk/non-chunk equivalence failed.")
    return result


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _write_report(config: dict[str, Any], artifact: Path, decision: dict[str, Any]) -> None:
    baseline = _load_json(artifact / "baseline_generator_manifest.json")
    reproduction = _load_json(artifact / "reproduction_gate.json")
    primary_manifest = _load_json(artifact / "primary_development_manifest.json")
    state_manifest = _load_json(artifact / "heterogeneous_state_manifest.json")
    curve = _load_json(artifact / "candidate_count_curve.json")
    first = _load_json(artifact / "first_success_index_summary.json")
    density = _load_json(artifact / "success_density_summary.json")
    robustness = _load_json(artifact / "noise_bank_robustness.json")
    failures = _load_json(artifact / "failure_contexts_1024.json")
    latency = _load_json(artifact / "latency_by_candidate_count.json")
    chunk_equivalence = _load_json(artifact / "chunk_equivalence.json")

    primary_by_n = {row["candidate_count"]: row for row in curve["primary"]}
    state_by_n = {row["candidate_count"]: row for row in curve["heterogeneous_state"]}
    latency_by_n = {row["candidate_count"]: row for row in latency["rows"]}

    primary_curve_rows = "\n".join(
        "| {n} | {oracle} | {any_feasible} | {candidate_feasible} |".format(
            n=count,
            oracle=_percent(primary_by_n[count]["oracle_success_rate"]),
            any_feasible=_percent(primary_by_n[count]["contexts_with_feasible_candidate_rate"]),
            candidate_feasible=_percent(primary_by_n[count]["mean_candidate_level_feasibility"]),
        )
        for count in PREFIX_COUNTS
    )
    state_curve_rows = "\n".join(
        "| {n} | {oracle} | {any_feasible} | {candidate_feasible} |".format(
            n=count,
            oracle=_percent(state_by_n[count]["oracle_success_rate"]),
            any_feasible=_percent(state_by_n[count]["contexts_with_feasible_candidate_rate"]),
            candidate_feasible=_percent(state_by_n[count]["mean_candidate_level_feasibility"]),
        )
        for count in PREFIX_COUNTS
    )
    gate_rows = "\n".join(
        "| {n} | {feas} | {dist} | {speed} | {direction} | {tip} | {joint} |".format(
            n=count,
            feas=_percent(primary_by_n[count]["contexts_with_feasible_candidate_rate"]),
            dist=_percent(primary_by_n[count]["contexts_with_distance_pass_rate"]),
            speed=_percent(primary_by_n[count]["contexts_with_directed_speed_pass_rate"]),
            direction=_percent(primary_by_n[count]["contexts_with_direction_pass_rate"]),
            tip=_percent(primary_by_n[count]["contexts_with_tip_first_rate"]),
            joint=_percent(primary_by_n[count]["oracle_success_rate"]),
        )
        for count in PREFIX_COUNTS
    )
    physical_rows = "\n".join(
        "| {set_name} | {n} | {tip:.2f} | {speed:.3f} | {direction:.2f} | {disp:.4f} | {uav:.3f} | {accel:.3f} | {duration:.3f} | {hit:.3f} |".format(
            set_name=set_name,
            n=count,
            tip=1000.0 * rows[count]["oracle_metrics"]["median_first_entry_tip_distance_m"],
            speed=rows[count]["oracle_metrics"]["median_first_entry_directed_speed_m_s"],
            direction=rows[count]["oracle_metrics"]["median_first_entry_direction_angle_deg"],
            disp=rows[count]["oracle_metrics"]["median_maximum_uav_displacement_m"],
            uav=rows[count]["oracle_metrics"]["median_maximum_uav_speed_m_s"],
            accel=rows[count]["oracle_metrics"]["median_maximum_command_acceleration_m_s2"],
            duration=rows[count]["oracle_metrics"]["median_maneuver_duration_s"],
            hit=rows[count]["oracle_metrics"]["median_first_entry_time_s"],
        )
        for set_name, rows in (("Primary", primary_by_n), ("Heterogeneous", state_by_n))
        for count in PREFIX_COUNTS
    )
    bank_rows = "\n".join(
        "| {n} | {mean} | {std} | {minimum} | {maximum} | {union} |".format(
            n=row["candidate_count"],
            mean=_percent(row["mean_oracle_success"]),
            std=f"{100.0 * row['std_oracle_success']:.2f} pp",
            minimum=_percent(row["minimum_oracle_success"]),
            maximum=_percent(row["maximum_oracle_success"]),
            union=_percent(row["union_oracle_success_analysis_only"]),
        )
        for row in robustness["aggregates"]
    )
    latency_rows = "\n".join(
        "| {n} | {mean:.2f} | {median:.2f} | {p90:.2f} | {p95:.2f} | {maximum:.2f} | {gpu:.2f} | {alloc:.2f} |".format(
            n=count,
            mean=latency_by_n[count]["wall"]["mean_ms"],
            median=latency_by_n[count]["wall"]["median_ms"],
            p90=latency_by_n[count]["wall"]["p90_ms"],
            p95=latency_by_n[count]["wall"]["p95_ms"],
            maximum=latency_by_n[count]["wall"]["maximum_ms"],
            gpu=latency_by_n[count]["gpu"]["median_ms"],
            alloc=latency_by_n[count]["peak_cuda_allocated_bytes"] / (1024.0 * 1024.0),
        )
        for count in PREFIX_COUNTS
    )
    memory_rows = "\n".join(
        "| {n} | {alloc:.2f} | {reserved:.2f} | {chunks} |".format(
            n=count,
            alloc=latency_by_n[count]["peak_cuda_allocated_bytes"] / (1024.0 * 1024.0),
            reserved=latency_by_n[count]["peak_cuda_reserved_bytes"] / (1024.0 * 1024.0),
            chunks=" + ".join(str(value) for value in latency_by_n[count]["chunk_sizes"]),
        )
        for count in PREFIX_COUNTS
    )
    primary_density = density["primary"]
    state_density = density["heterogeneous_state"]
    primary_first = first["primary"]
    state_first = first["heterogeneous_state"]
    recommended = int(decision["recommended_candidate_count"])
    recommended_metrics = primary_by_n[recommended]["oracle_metrics"]
    exact_first128 = baseline["primary_noise_bank"]["first_128_exact"]
    primary_splits: dict[str, int] = {}
    for row in primary_manifest["records"]:
        key = f"{row['split']} / {row['bank']}"
        primary_splits[key] = primary_splits.get(key, 0) + 1
    strata = state_manifest["stratum_counts"]
    failure_gate_counts: dict[str, dict[str, int]] = {}
    for name in ("primary", "heterogeneous_state"):
        failure_gate_counts[name] = {
            gate: sum(bool(row["no_candidate_passes"][gate]) for row in failures[name])
            for gate in ("distance", "directed_speed", "direction", "tip_first", "feasible")
        }
    failure_table = "\n".join(
        "| {set_name} | `{context}` | `{state}` | {tip:.1f} | {speed:.3f} | {direction:.2f} | {disp:.4f} | {missing} |".format(
            set_name=set_name,
            context=row["context_id"],
            state=row["state_id"],
            tip=1000.0 * row["best_tip_distance_m"],
            speed=row["best_directed_speed_m_s"],
            direction=row["best_direction_error_deg"],
            disp=row["best_uav_displacement_m"],
            missing=", ".join(
                gate for gate, absent in row["no_candidate_passes"].items() if absent
            )
            or "joint overlap only",
        )
        for set_name, rows in (("Primary", failures["primary"]), ("Heterogeneous", failures["heterogeneous_state"]))
        for row in rows
    )
    margin_labels = {
        "tip_m": ("tip distance", "mm", 1000.0),
        "speed_m_s": ("directed speed", "m/s", 1.0),
        "direction_deg": ("direction", "deg", 1.0),
        "displacement_m": ("UAV displacement", "mm", 1000.0),
        "uav_speed_m_s": ("UAV speed", "m/s", 1.0),
        "command_accel_m_s2": ("command acceleration", "m/s^2", 1.0),
    }
    margin_rows = "\n".join(
        f"| {label} | {decision['hard_gate_margin_summary'][key]['median'] * scale:.3f} | "
        f"{decision['hard_gate_margin_summary'][key]['p05'] * scale:.3f} | {unit} |"
        for key, (label, unit, scale) in margin_labels.items()
    )

    text = f"""# Milestone 7A.5 — Frozen Diffusion Candidate-Support and Latency Study

Artifact root: `{artifact.relative_to(ROOT)}`  
Completed UTC: `{datetime.now(timezone.utc).isoformat()}`

## 1. Why generator training was frozen

Milestone 7A.4 showed a literal-prefix improvement from 39.06% at 32 candidates to 64.06% at 128 candidates. This milestone isolates sampling support from representation learning: the 7A.3 FULL flat-conditioning EMA generator, its context/action contracts, DDIM scheduler, and production simulator were frozen. No CEM, diffusion training, scorer work, dataset generation, or architecture change occurred.

## 2. Selected 7A.3 generator confirmation

The evaluated generator is `BASELINE_FLAT_CONDITIONING`, checkpoint SHA-256 `{baseline['checkpoint_sha256']}`, with {baseline['trainable_parameters']:,} trainable parameters. It maps normalized 83-D contexts to direct normalized 49-D actions through the repaired deterministic 25-step DDIM timetable `{baseline['ddim_schedule'][0]} -> {baseline['ddim_schedule'][-1]}` with `eta=0`. The experimental structured-FiLM checkpoint was not loaded.

## 3. Reproduction of corrected 7A.4 32/64/128 result

The hard reproduction gate **passed exactly**. Expected and observed oracle rates were 39.0625%, 56.2500%, and 64.0625% at N=32, 64, and 128; every difference was 0.0. The generated-action clamp fraction in this reproduction was {_percent(reproduction['generation']['clamp_fraction'])}.

## 4. Exact primary development context set

`primary_development_manifest.json` freezes all {primary_manifest['context_count']} context IDs, state IDs, targets, directions, and ownership labels. Composition: {', '.join(f'{key}: {value}' for key, value in sorted(primary_splits.items()))}. There are {len({row['state_id'] for row in primary_manifest['records']})} unique physical state IDs. No substitution was made after the manifest was written.

## 5. Heterogeneous state diagnostic set

`heterogeneous_state_manifest.json` freezes 64 unique TRAIN-owned state/context pairs, with one target per physical state. Existing state distance was stratified into LOW={strata['LOW']}, MEDIUM={strata['MEDIUM']}, and HIGH={strata['HIGH']}; no TEST or protected context appears.

## 6. Exact nested 1024 noise-bank construction

The primary bank is one persisted `[1024,49]` Gaussian array with SHA-256 `{baseline['primary_noise_bank']['sha256']}`. Exact recovery of the 7A.4 first 128 vectors is `{str(exact_first128).upper()}`. Rows 128:1024 were deterministically precommitted with seed {baseline['primary_noise_bank']['extension_seed']}. All N values are literal prefixes of this one bank. Three independent `[512,49]` banks were also persisted before physics with seeds {', '.join(str(value) for value in config['independent_noise_seeds'])}; none was selected by performance.

## 7. Production physics evaluation contract

For each context, the maximum bank was generated once, decoded as the production normalized `[49]` action, and evaluated once. Execution used 16 acceleration knots plus `T_maneuver in [0.45,1.80] s`, ACTIVE -> 0.30-s analytic SETTLE -> HOLD, `T_evaluation=2.40 s`, the frozen `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`, and cyclic padding to the fixed numerical batch of 2048. Prefix results were sliced from saved candidate/outcome tensors; no prefix was regenerated or resimulated. Scientific gates were unchanged.

## 8. Primary candidate-count oracle curve

| N | Oracle success | Contexts with any feasible candidate | Candidate-level feasibility |
|---:|---:|---:|---:|
{primary_curve_rows}

The primary curve rises 43.75 percentage points from N=32 to N=1024.

## 9. Heterogeneous-state candidate-count oracle curve

| N | Oracle success | Contexts with any feasible candidate | Candidate-level feasibility |
|---:|---:|---:|---:|
{state_curve_rows}

The heterogeneous curve rises 42.19 points but remains 21.88 points below primary at N=1024. Every context in both sets contains at least one feasible candidate at every reported N; the missing support is joint task success, not absence of safe motion.

## 10. First-success-index distribution

Among the 53/64 primary contexts solved by N=1024, the one-based first-success index has median {primary_first['median']:.0f}, p75 {primary_first['p75']:.0f}, p90 {primary_first['p90']:.1f}, p95 {primary_first['p95']:.1f}, and maximum {primary_first['maximum']}. For the 39/64 heterogeneous contexts solved, the corresponding values are median {state_first['median']:.0f}, p75 {state_first['p75']:.0f}, p90 {state_first['p90']:.1f}, p95 {state_first['p95']:.1f}, and maximum {state_first['maximum']}.

## 11. Successful-candidate density

Primary contexts contain a mean {primary_density['successful_candidate_count']['mean']:.2f} and median {primary_density['successful_candidate_count']['median']:.1f} successes per 1024 candidates: mean density {_percent(primary_density['successful_candidate_fraction']['mean'])}, median density {_percent(primary_density['successful_candidate_fraction']['median'])}, p10 {_percent(primary_density['successful_candidate_fraction']['p10'])}, p90 {_percent(primary_density['successful_candidate_fraction']['p90'])}. Heterogeneous contexts contain mean {state_density['successful_candidate_count']['mean']:.2f}, median {state_density['successful_candidate_count']['median']:.1f}, with mean density {_percent(state_density['successful_candidate_fraction']['mean'])}. Success is therefore sparse even when a context is solvable.

## 12. Feasible-candidate density

Primary candidate feasibility has mean {_percent(primary_density['feasible_candidate_fraction']['mean'])}, median {_percent(primary_density['feasible_candidate_fraction']['median'])}, p10 {_percent(primary_density['feasible_candidate_fraction']['p10'])}, and p90 {_percent(primary_density['feasible_candidate_fraction']['p90'])}. Heterogeneous feasibility is higher: mean {_percent(state_density['feasible_candidate_fraction']['mean'])}, median {_percent(state_density['feasible_candidate_fraction']['median'])}. Sampling more candidates exposes rare joint success rather than merely finding the first feasible action.

## 13. Per-gate support progression

Primary context-level support is:

| N | Any feasible | Distance | Directed speed | Direction | Tip first | Joint success |
|---:|---:|---:|---:|---:|---:|---:|
{gate_rows}

At N=1024, individual speed and direction support reach 100%, tip-first reaches 98.44%, and distance reaches 95.31%, yet joint success is 82.81%. The remaining issue is overlap of gates in the same candidate.

## 14. N=1024 failure-context analysis

There are {len(failures['primary'])} primary and {len(failures['heterogeneous_state'])} heterogeneous contexts with zero joint success. Counts with no candidate passing an individual gate are primary `{failure_gate_counts['primary']}` and heterogeneous `{failure_gate_counts['heterogeneous_state']}`. Detailed best-candidate physical values and exact IDs are preserved in `failure_contexts_1024.json`. Many failures have candidates that individually pass distance, speed, direction, tip-first, and feasibility, but no single candidate passes all gates; UAV displacement is a frequent incompatibility in the heterogeneous rows.

| Set | Context | State | Best tip mm | Best speed m/s | Best direction deg | Best UAV displacement m | No candidate passes |
|---|---|---|---:|---:|---:|---:|---|
{failure_table}

## 15. Independent-noise-bank robustness

| N | Four-bank mean | Std | Minimum | Maximum | Union (analysis only) |
|---:|---:|---:|---:|---:|---:|
{bank_rows}

## 16. Whether fixed bank identity materially changes conclusions

Bank identity matters at small N: N=32 spans 14.06–42.19%. Variability contracts with sampling; N=512 spans 79.69–84.38%, standard deviation 1.91 percentage points. Thus no bank was cherry-picked and the large-N conclusion is not a one-bank anomaly. The union is analysis-only and is not presented as a deployment policy.

## 17. Generation latency by N

| N | Mean wall ms | Median wall ms | P90 wall ms | P95 wall ms | Max wall ms | Median GPU ms | Peak allocated MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
{latency_rows}

These timings include context normalization, deterministic 25-step DDIM, final bounded-action preparation, and the deterministic hierarchical chunk path, but exclude simulator and scorer.

## 18. CUDA memory by N

| N | Peak allocated MiB | Peak reserved MiB | Candidate chunks |
|---:|---:|---:|---|
{memory_rows}

The hierarchical path preserves exact noise-vector order and bounds peak allocation below 18 MiB. A focused whole-batch-versus-chunked check on 128 candidates passed: bounded max absolute difference `{chunk_equivalence['bounded_max_abs_difference']:.3e}` against tolerance `{chunk_equivalence['tolerance']:.1e}`. No extra memory-driven subdivision was needed.

## 19. Recommended practical N

No evaluated N reaches the requested 90% primary-support criterion. For continued *offline analysis* the recommended sampled prefix is N={recommended}, because the N=512 -> 1024 increment is {_percent(primary_by_n[1024]['oracle_success_rate'] - primary_by_n[512]['oracle_success_rate'])} on primary and {_percent(state_by_n[1024]['oracle_success_rate'] - state_by_n[512]['oracle_success_rate'])} on heterogeneous states. This does **not** change the production candidate count. The 1000-query precision run at N={recommended} measured median {decision['precise_latency']['wall']['median_ms']:.2f} ms and p95 {decision['precise_latency']['wall']['p95_ms']:.2f} ms, above the prior 100-ms p95 engineering target.

At N={recommended}, median successful-oracle physical metrics are tip distance {1000.0 * recommended_metrics['median_first_entry_tip_distance_m']:.2f} mm, directed speed {recommended_metrics['median_first_entry_directed_speed_m_s']:.3f} m/s, direction error {recommended_metrics['median_first_entry_direction_angle_deg']:.2f} deg, UAV displacement {recommended_metrics['median_maximum_uav_displacement_m']:.4f} m, UAV speed {recommended_metrics['median_maximum_uav_speed_m_s']:.3f} m/s, command acceleration {recommended_metrics['median_maximum_command_acceleration_m_s2']:.3f} m/s^2, maneuver duration {recommended_metrics['median_maneuver_duration_s']:.3f} s, and hit time {recommended_metrics['median_first_entry_time_s']:.3f} s. The 5th-percentile gate margins are saved in `recommended_candidate_count.json`.

Successful-oracle physical metrics across all prefixes are:

| Set | N | Tip mm | Directed speed m/s | Direction deg | UAV displacement m | UAV speed m/s | Command accel m/s^2 | Maneuver s | Hit s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{physical_rows}

At N={recommended}, successful primary hits are `{primary_by_n[recommended]['oracle_hit_segment_counts']}`. Margins are positive distance from each hard limit:

| Gate | Median margin | P05 margin | Unit |
|---|---:|---:|---|
{margin_rows}

## 20. Candidate-support classification

**{decision['classification']}**. The primary distribution contains substantial sparse support, and support continues increasing through 1024. However, 82.81% remains below the 90% sufficiency gate, the heterogeneous-state result is only 60.94%, and large-N latency is not within the immediate-query target. The dominant distinction is state support; small-N bank sensitivity is secondary and largely contracts by N=512.

## 21. Whether scorer training is justified

**NO.** The task authorizes scorer training only for `CANDIDATE_SUPPORT_SUFFICIENT`; that condition was not met. No scorer was trained or used.

## 22. Whether more CEM data is justified

**NO from this experiment alone.** Candidate-count failure is not evidence that more labels will repair conditional state support. This milestone ran zero CEM solves and makes no automatic teacher-expansion decision.

## 23. Recommended next step

Return for methodological review of state-conditioned support. The useful next question is how to improve support over heterogeneous initial states without conflating the issue with bank size or scorer ranking. Do not start scorer training, CEM generation, retraining, or architecture changes automatically.

## 24. Final TEST

**NOT EVALUATED.**

## 25. Protected test

`fig8vertical_002`: **NOT EVALUATED.**

## 26. Hardware

**NOT EXECUTED.** All results are simulation-only.

## Final summary

    New CEM solves:
        0

    Diffusion training:
        NONE

    Generator:
        FROZEN 7A.3 FLAT MODEL

    Primary context count:
        64

    State-diagnostic context count:
        64

    Primary oracle:

        N=32:
            39.06%

        N=64:
            56.25%

        N=128:
            64.06%

        N=256:
            76.56%

        N=512:
            79.69%

        N=1024:
            82.81%

    Heterogeneous-state oracle:

        N=32:
            18.75%

        N=64:
            26.56%

        N=128:
            35.94%

        N=256:
            42.19%

        N=512:
            53.12%

        N=1024:
            60.94%

    First-success-index:

        median = 39
        p90 = 226.0
        p95 = 280.8

    Noise-bank variability:
        N=512 mean 81.25%, std 1.91 pp, range 79.69–84.38%

    Recommended candidate count:
        1024 FOR ANALYSIS; PRODUCTION COUNT NOT CHANGED

    Median latency at recommended N:
        {decision['precise_latency']['wall']['median_ms']:.2f} ms

    P95 latency:
        {decision['precise_latency']['wall']['p95_ms']:.2f} ms

    Candidate-support result:
        {decision['classification']}

    Scorer training justified:
        NO

    More CEM data justified:
        NO

    Architecture changed:
        NO

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    REPORT.write_text(text, encoding="utf-8")
    (artifact / REPORT.name).write_text(text, encoding="utf-8")


def finalize(config: dict[str, Any], artifact: Path) -> dict[str, Any]:
    figures(artifact)
    _chunk_equivalence(config, artifact)
    previous_decision = _load_json(artifact / "recommended_candidate_count.json")
    latency = _load_json(artifact / "latency_by_candidate_count.json")
    primary_outcomes = _load_outcomes(artifact / "primary_development_candidate_outcomes.npz")
    state_outcomes = _load_outcomes(artifact / "heterogeneous_candidate_outcomes.npz")
    curve = {
        "schema": "milestone7a5_candidate_count_curve_v1",
        "prefixes_are_literal_action_supersets": True,
        "physics_evaluated_once_per_candidate": True,
        "primary": _curve(primary_outcomes),
        "heterogeneous_state": _curve(state_outcomes),
        "generation": _load_json(artifact / "candidate_count_curve.json")["generation"],
    }
    _write_json(artifact / "candidate_count_curve.json", curve)
    _write_json(
        artifact / "gate_support_summary.json",
        {
            "schema": "milestone7a5_gate_support_summary_v1",
            "primary": curve["primary"],
            "heterogeneous_state": curve["heterogeneous_state"],
        },
    )
    decision = _recommend_and_classify(
        curve,
        _load_json(artifact / "noise_bank_robustness.json"),
        latency["rows"],
    )
    decision["precise_latency"] = previous_decision["precise_latency"]
    decision["hard_gate_margin_summary"] = _margin_summary(
        primary_outcomes, int(decision["recommended_candidate_count"])
    )
    _write_json(artifact / "recommended_candidate_count.json", decision)
    _write_report(config, artifact, decision)
    files = [
        ROOT / "learning/action_diffusion.py",
        ROOT / "learning/policy_action.py",
        ROOT / "planning/production_cem.py",
        ROOT / "run_milestone7a3.py",
        ROOT / "run_milestone7a5.py",
        ROOT / "config/learning/diffusion_candidate_support_v1.json",
        ROOT / "tests/test_milestone7a5_candidate_support.py",
        ROOT / config["baseline_artifact"] / "full_diffusion_ema_best.pt",
        artifact / "primary_noise_bank_1024.npy",
        artifact / "chunk_equivalence.json",
        REPORT,
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "milestone7a5_source_hash_manifest_v1",
            "files": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            ],
            "new_cem_solves": 0,
            "diffusion_training": "NONE",
            "scorer_training": "NONE",
            "final_test_evaluated": False,
            "protected_test_evaluated": False,
            "hardware_executed": False,
        },
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--stage",
        choices=("prepare", "reproduce", "primary-state", "robustness", "latency", "finalize", "all"),
        default="all",
    )
    args = parser.parse_args()
    config = _load_json(args.config)
    _validate_config(config)
    artifact = args.artifact or ARTIFACT_PARENT / _timestamp()
    if not artifact.is_absolute():
        artifact = ROOT / artifact
    result: Any = None
    if args.stage in ("prepare", "all"):
        result = prepare(config, artifact)
    if args.stage in ("reproduce", "all"):
        result = reproduce(config, artifact)
    if args.stage in ("primary-state", "all"):
        result = primary_and_state(config, artifact)
    if args.stage in ("robustness", "all"):
        result = robustness(config, artifact)
    if args.stage in ("latency", "all"):
        result = latency_and_decision(config, artifact)
    if args.stage in ("finalize", "all"):
        result = finalize(config, artifact)
    print(json.dumps(_safe(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
