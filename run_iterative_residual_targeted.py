"""Targeted continuation of the IRP-style residual-outcome experiment.

This runner performs no CEM.  It creates local production-simulator responses
around failed initializer actions, saves compact tip/UAV trajectories, and
fine-tunes the existing delta-outcome model on a mixture of the original
teacher-neighborhood data and the new failed-initializer neighborhoods.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from learning.cem_teacher_support import (
    evaluate_teacher_action_candidates,
    load_fixed_production_environment,
    load_production_cem_teachers,
    production_cem_settings,
)
from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    pad_context_specification,
)
from learning.iterative_residual import (
    IterativeResidualOutcomeModel,
    OutcomeNormalizer,
    deterministic_compact_candidates,
    outcome_arrays_from_archive,
    outcome_arrays_from_rows,
)
from planning.production_cem import FIXED_NUMERICAL_BATCH_SIZE, record_normalized_actions_fixed_batch
from run_iterative_residual_policy import (
    _load_existing_data,
    _sample_pairs,
    _select_corrections,
    _summarize_rows,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "iterative_residual_targeted_v1.json"
ARTIFACT_PARENT = ROOT / "data" / "policy_training" / "iterative_residual_targeted_v1"
ROOT_REPORT = ROOT / "ITERATIVE_RESIDUAL_TARGETED_TRAINING_REPORT.md"


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


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "iterative_residual_targeted_v1":
        raise ValueError("Unsupported targeted residual configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("Frozen production model changed.")
    for key in ("new_cem_solves", "sac", "diffusion", "final_test", "theta_randomization", "hardware"):
        if not bool(config["prohibitions"][key]):
            raise ValueError(f"Required prohibition is disabled: {key}")
    if config["prohibitions"]["protected_test"] != "fig8vertical_002":
        raise ValueError("Protected-test identity changed.")


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _stratified_select(records, eligible: list[int], count: int, seed: int) -> list[int]:
    buckets: dict[str, list[int]] = {}
    for index in eligible:
        buckets.setdefault(records[index].group, []).append(index)
    for group in buckets:
        buckets[group].sort(key=lambda index: _stable_rank(seed, records[index].context_id))
    selected: list[int] = []
    groups = sorted(buckets)
    while len(selected) < min(count, len(eligible)):
        progressed = False
        for group in groups:
            if buckets[group] and len(selected) < count:
                selected.append(buckets[group].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _select_failed_contexts(config, data, records) -> dict[str, Any]:
    source = ROOT / config["milestone_7b_artifact"]
    payload = json.loads((source / "one_action_evaluation.json").read_text(encoding="utf-8"))
    success_by_id = {
        str(row["context_id"]): bool(row["methods"]["DETERMINISTIC_RESIDUAL_MLP"]["success"])
        for row in payload["authoritative_rows"]
    }
    train_ids = set(data["split"]["training_context_ids"])
    dev_ids = set(data["split"]["development_context_ids"])
    train_eligible = [
        index for index, record in enumerate(records)
        if record.context_id in train_ids and not success_by_id[record.context_id]
    ]
    dev_eligible = [
        index for index, record in enumerate(records)
        if record.context_id in dev_ids and not success_by_id[record.context_id]
    ]
    settings = config["targeted_data"]
    train_selected = _stratified_select(
        records, train_eligible, int(settings["training_failed_contexts"]), int(config["seed"]) + 10
    )
    dev_selected = _stratified_select(
        records, dev_eligible, int(settings["development_failed_contexts"]), int(config["seed"]) + 20
    )
    return {
        "schema": "failed_initializer_targeted_context_manifest_v1",
        "training_eligible_failures": len(train_eligible),
        "development_eligible_failures": len(dev_eligible),
        "training_indices": train_selected,
        "development_indices": dev_selected,
        "training_context_ids": [records[index].context_id for index in train_selected],
        "development_context_ids": [records[index].context_id for index in dev_selected],
        "state_overlap": sorted(
            {records[index].state_id for index in train_selected}
            & {records[index].state_id for index in dev_selected}
        ),
        "all_selected_initially_failed": True,
        "selection": "deterministic group-stratified stable hash",
    }


def _trajectory_feature(traj, repeated_context, record, stride: int) -> tuple[np.ndarray, np.ndarray]:
    sample_index = torch.arange(0, len(traj.times_s), stride, dtype=torch.int64)
    if int(sample_index[-1]) != len(traj.times_s) - 1:
        sample_index = torch.cat((sample_index, torch.tensor([len(traj.times_s) - 1])))
    count = int(traj.uav_positions_m.shape[1])
    rotation = repeated_context.frame.rotation_world_from_local[:count].detach().cpu()
    world_to_local = rotation.transpose(-1, -2)
    root = repeated_context.frame.root_position_world_m[:count].detach().cpu()
    target = torch.tensor(record.target_local_m, dtype=traj.uav_positions_m.dtype)
    tip_world = traj.cable_positions_m[sample_index, :, -1]
    tip_velocity_world = traj.cable_velocities_m_s[sample_index, :, -1]
    uav_world = traj.uav_positions_m[sample_index]
    tip_local = torch.einsum("bij,tbj->tbi", world_to_local, tip_world - root[None])
    tip_velocity_local = torch.einsum("bij,tbj->tbi", world_to_local, tip_velocity_world)
    uav_displacement_local = torch.einsum(
        "bij,tbj->tbi", world_to_local, uav_world - traj.uav_positions_m[0][None]
    )
    feature = torch.cat((tip_local - target, tip_velocity_local, uav_displacement_local), dim=-1)
    return feature.permute(1, 0, 2).numpy().astype(np.float32), traj.times_s[sample_index].numpy().astype(np.float32)


def _outcomes_from_trajectory(traj) -> tuple[np.ndarray, np.ndarray]:
    result = traj.metrics
    archive = {
        "best_event_tip_distance_m": result.best_event_tip_distance_m.numpy(),
        "best_event_directed_speed_m_s": result.best_event_directed_speed_m_s.numpy(),
        "best_event_direction_angle_deg": result.best_event_direction_angle_deg.numpy(),
        "maximum_uav_displacement_m": result.maximum_uav_displacement_m.numpy(),
        "maximum_uav_speed_m_s": result.maximum_uav_speed_m_s.numpy(),
        "maximum_command_acceleration_m_s2": result.maximum_command_acceleration_m_s2.numpy(),
        "first_entry_marker": result.first_entry_marker.numpy(),
        "feasible": result.feasible.numpy(),
        "success": result.success.numpy(),
    }
    return outcome_arrays_from_archive(archive)


def _generate_targeted_data(config, artifact, data, records, manifest, basis, environment) -> dict[str, Any]:
    simulator, task, training_bank, _heldout, canonical_bank, _settings = environment
    physics_settings = production_cem_settings(config)
    settings = config["targeted_data"]
    shard_root = artifact / "targeted_rollout_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    indices = manifest["training_indices"] + manifest["development_indices"]
    started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    for ordinal, index in enumerate(indices, start=1):
        record = records[index]
        split = "TRAIN" if index in set(manifest["training_indices"]) else "DEVELOPMENT"
        shard = shard_root / f"{record.context_id}.npz"
        if shard.exists():
            with np.load(shard, allow_pickle=False) as saved:
                summaries.append(json.loads(str(saved["summary_json"])))
            continue
        candidates, _delta = deterministic_compact_candidates(
            torch.from_numpy(data["initial_actions"][index]),
            basis,
            candidate_count=int(settings["candidates_per_context"]),
            coordinate_rms_scales=tuple(float(value) for value in settings["coordinate_rms_scales"]),
            seed=int(config["seed"]) + 1_000_000 + index,
        )
        count = int(candidates.shape[0])
        bank = canonical_bank if record.bank == "canonical" else training_bank
        specification = ContextSpecification(
            torch.full((count,), record.state_index, dtype=torch.int64),
            torch.tensor([record.target_local_m], dtype=torch.float32).expand(count, -1),
            torch.tensor([record.direction_local], dtype=torch.float32).expand(count, -1),
            "targeted_failed_initializer_response",
        )
        repeated_context = build_context_from_specification(
            simulator,
            bank,
            pad_context_specification(specification, FIXED_NUMERICAL_BATCH_SIZE),
        )
        trajectory = record_normalized_actions_fixed_batch(
            simulator,
            repeated_context,
            candidates,
            task,
            physics_settings,
            record_count=count,
        )
        trajectory_feature, trajectory_times = _trajectory_feature(
            trajectory, repeated_context, record, int(settings["trajectory_stride_steps"])
        )
        continuous, binary = _outcomes_from_trajectory(trajectory)
        summary = {
            "context_id": record.context_id,
            "state_id": record.state_id,
            "group": record.group,
            "split": split,
            "candidate_count": count,
            "scientific_success_count": int(binary[:, 2].sum()),
            "feasible_count": int(binary[:, 1].sum()),
            "initial_action_success": bool(binary[0, 2]),
            "trajectory_samples": int(trajectory_feature.shape[1]),
        }
        np.savez_compressed(
            shard,
            schema=np.asarray("targeted_failed_initializer_response_v1"),
            context_id=np.asarray(record.context_id),
            state_id=np.asarray(record.state_id),
            split=np.asarray(split),
            original_index=np.asarray(index, dtype=np.int64),
            normalized_context=data["normalized_contexts"][index],
            initial_action=data["initial_actions"][index],
            normalized_actions=candidates.numpy(),
            trajectory_times_s=trajectory_times,
            trajectory_features=trajectory_feature,
            continuous_outcomes=continuous,
            binary_outcomes=binary,
            summary_json=np.asarray(json.dumps(summary, sort_keys=True)),
        )
        summaries.append(summary)
        print(
            f"targeted rollout {ordinal}/{len(indices)} {record.context_id}: "
            f"success={summary['scientific_success_count']}/{count}",
            flush=True,
        )
    output = {
        "schema": "targeted_failed_initializer_dataset_summary_v1",
        "contexts": len(summaries),
        "training_contexts": sum(row["split"] == "TRAIN" for row in summaries),
        "development_contexts": sum(row["split"] == "DEVELOPMENT" for row in summaries),
        "candidate_rows": sum(row["candidate_count"] for row in summaries),
        "successful_rows": sum(row["scientific_success_count"] for row in summaries),
        "feasible_rows": sum(row["feasible_count"] for row in summaries),
        "trajectory_feature_schema": "25 samples x [tip-target position(3), tip velocity(3), UAV displacement(3)]",
        "new_cem_solves": 0,
        "runtime_s": time.perf_counter() - started,
        "rows": summaries,
    }
    _write_json(artifact / "targeted_dataset_summary.json", output)
    return output


def _load_targeted_tensors(artifact: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    ids = manifest["training_context_ids"] + manifest["development_context_ids"]
    actions, continuous, binary, contexts, original = [], [], [], [], []
    for context_id in ids:
        with np.load(artifact / "targeted_rollout_shards" / f"{context_id}.npz", allow_pickle=False) as shard:
            actions.append(np.asarray(shard["normalized_actions"], dtype=np.float32))
            continuous.append(np.asarray(shard["continuous_outcomes"], dtype=np.float32))
            binary.append(np.asarray(shard["binary_outcomes"], dtype=np.float32))
            contexts.append(np.asarray(shard["normalized_context"], dtype=np.float32))
            original.append(int(shard["original_index"]))
    train_count = len(manifest["training_context_ids"])
    return {
        "context_ids": np.asarray(ids),
        "normalized_contexts": np.stack(contexts),
        "perturbation_actions": np.stack(actions),
        "continuous": np.stack(continuous),
        "binary": np.stack(binary),
        "train_indices": np.arange(train_count, dtype=np.int64),
        "dev_indices": np.arange(train_count, len(ids), dtype=np.int64),
        "original_indices": np.asarray(original, dtype=np.int64),
    }


def _load_source_model(config: dict[str, Any]) -> tuple[IterativeResidualOutcomeModel, OutcomeNormalizer]:
    source = ROOT / config["source_irp_artifact"]
    checkpoint = torch.load(
        source / "iterative_residual_outcome_model_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = IterativeResidualOutcomeModel(int(checkpoint["hidden_dimension"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    payload = json.loads((source / "outcome_normalizer.json").read_text(encoding="utf-8"))
    normalizer = OutcomeNormalizer(
        torch.tensor(payload["mean"], dtype=torch.float32),
        torch.tensor(payload["standard_deviation"], dtype=torch.float32),
    )
    return model, normalizer


def _fine_tune(config, original, targeted, artifact) -> tuple[IterativeResidualOutcomeModel, OutcomeNormalizer, dict[str, Any]]:
    settings = config["fine_tuning"]
    device = torch.device("cuda")
    model, normalizer = _load_source_model(config)
    model.to(device)
    original_actions = torch.from_numpy(original["perturbation_actions"]).to(device)
    original_continuous = torch.from_numpy(original["continuous"]).to(device)
    original_binary = torch.from_numpy(original["binary"]).to(device)
    original_contexts = torch.from_numpy(original["normalized_contexts"]).to(device)
    original_train = torch.from_numpy(original["train_indices"]).to(device)
    targeted_actions = torch.from_numpy(targeted["perturbation_actions"]).to(device)
    targeted_continuous = torch.from_numpy(targeted["continuous"]).to(device)
    targeted_binary = torch.from_numpy(targeted["binary"]).to(device)
    targeted_contexts = torch.from_numpy(targeted["normalized_contexts"]).to(device)
    targeted_train = torch.from_numpy(targeted["train_indices"]).to(device)
    targeted_dev = torch.from_numpy(targeted["dev_indices"]).to(device)
    combined_binary = torch.cat(
        (original_binary[original_train].reshape(-1, 3), targeted_binary[targeted_train].reshape(-1, 3))
    )
    prevalence = combined_binary.mean(dim=0)
    positive_weight = torch.clamp((1.0 - prevalence) / torch.clamp(prevalence, min=1.0e-4), max=20.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 300)
    validation_generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 301)
    validation = _sample_pairs(
        targeted_dev,
        targeted_actions,
        targeted_continuous,
        targeted_binary,
        targeted_contexts,
        normalizer,
        count=65536,
        generator=validation_generator,
        teacher_anchor_probability=0.0,
        outcome_balanced_target_sampling=True,
    )

    def validation_metrics() -> tuple[float, float, float]:
        model.eval()
        with torch.no_grad():
            predicted_continuous, predicted_binary = model(*validation[:4])
            continuous_loss = F.smooth_l1_loss(predicted_continuous, validation[4])
            binary_loss = F.binary_cross_entropy_with_logits(
                predicted_binary, validation[5], pos_weight=positive_weight
            )
        model.train()
        return float(continuous_loss + binary_loss), float(continuous_loss), float(binary_loss)

    initial_validation, initial_continuous, initial_binary = validation_metrics()
    best_loss = initial_validation
    best_update = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = [{
        "update": 0,
        "validation_loss": initial_validation,
        "validation_continuous_loss": initial_continuous,
        "validation_binary_loss": initial_binary,
    }]
    batch_size = int(settings["batch_size"])
    original_count = int(round(batch_size * float(settings["original_data_fraction"])))
    targeted_count = batch_size - original_count
    started = time.perf_counter()
    for update in range(1, int(settings["maximum_updates"]) + 1):
        old = _sample_pairs(
            original_train,
            original_actions,
            original_continuous,
            original_binary,
            original_contexts,
            normalizer,
            count=original_count,
            generator=generator,
        )
        new = _sample_pairs(
            targeted_train,
            targeted_actions,
            targeted_continuous,
            targeted_binary,
            targeted_contexts,
            normalizer,
            count=targeted_count,
            generator=generator,
            teacher_anchor_probability=0.0,
            outcome_balanced_target_sampling=True,
        )
        batch = tuple(torch.cat((old[index], new[index]), dim=0) for index in range(6))
        predicted_continuous, predicted_binary = model(*batch[:4])
        continuous_loss = F.smooth_l1_loss(predicted_continuous, batch[4])
        binary_loss = F.binary_cross_entropy_with_logits(
            predicted_binary, batch[5], pos_weight=positive_weight
        )
        loss = continuous_loss + binary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"])))
        optimizer.step()
        if update % int(settings["validation_interval"]) == 0:
            value, continuous_value, binary_value = validation_metrics()
            row = {
                "update": update,
                "minibatch_loss": float(loss.detach()),
                "validation_loss": value,
                "validation_continuous_loss": continuous_value,
                "validation_binary_loss": binary_value,
                "gradient_norm_before_clip": gradient_norm,
            }
            history.append(row)
            print(f"fine-tune {update}: train={float(loss.detach()):.5f} dev={value:.5f}", flush=True)
            if value < best_loss - 1.0e-6:
                best_loss = value
                best_update = update
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            if update >= int(settings["minimum_updates"]) and update - best_update >= int(settings["patience_updates"]):
                break
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "schema": "iterative_residual_targeted_model_v1",
            "model_state_dict": best_state,
            "hidden_dimension": int(settings["hidden_dimension"]),
            "best_update": best_update,
            "best_validation_loss": best_loss,
            "source_irp_artifact": config["source_irp_artifact"],
        },
        artifact / "targeted_residual_outcome_model_best.pt",
    )
    summary = {
        "schema": "iterative_residual_targeted_training_history_v1",
        "configuration": settings,
        "initial_validation_loss": initial_validation,
        "best_validation_loss": best_loss,
        "best_update": best_update,
        "updates_completed": history[-1]["update"],
        "runtime_s": time.perf_counter() - started,
        "original_unique_rows": int(len(original["train_indices"]) * original_actions.shape[1]),
        "targeted_unique_rows": int(len(targeted["train_indices"]) * targeted_actions.shape[1]),
        "history": history,
    }
    _write_json(artifact / "targeted_training_history.json", summary)
    return model, normalizer, summary


def _evaluate_iterative(config, data, records, basis, environment, model, normalizer, artifact) -> dict[str, Any]:
    simulator, task, training_bank, _heldout, canonical_bank, _settings = environment
    physics_settings = production_cem_settings(config)
    current_actions = data["initial_actions"].copy()
    split_ids = {
        "TRAIN": set(data["split"]["training_context_ids"]),
        "DEVELOPMENT": set(data["split"]["development_context_ids"]),
    }
    action_history = [current_actions.copy()]
    iterations: list[dict[str, Any]] = []
    rows = evaluate_teacher_action_candidates(
        simulator, task, physics_settings, records, current_actions[:, None], training_bank, canonical_bank,
        label="targeted_irp_execution_0",
    )
    current_continuous, current_binary = outcome_arrays_from_rows(rows)
    iterations.append({"execution_count": 1, "correction_iteration": 0, "summary": _summarize_rows(records, rows, split_ids)})
    selection_history = []
    for iteration in range(1, int(config["residual_search"]["maximum_iterations"]) + 1):
        current_actions, selection = _select_corrections(
            model, normalizer, data["normalized_contexts"], current_actions,
            current_continuous, current_binary, basis, config, iteration=iteration,
        )
        selection_history.append(selection)
        rows = evaluate_teacher_action_candidates(
            simulator, task, physics_settings, records, current_actions[:, None], training_bank, canonical_bank,
            label=f"targeted_irp_execution_{iteration}",
        )
        current_continuous, current_binary = outcome_arrays_from_rows(rows)
        action_history.append(current_actions.copy())
        summary = _summarize_rows(records, rows, split_ids)
        iterations.append({"execution_count": iteration + 1, "correction_iteration": iteration, "summary": summary})
        print(
            f"targeted execution {iteration + 1}: train={100*summary['TRAIN']['scientific_success_rate']:.2f}% "
            f"dev={100*summary['DEVELOPMENT']['scientific_success_rate']:.2f}%",
            flush=True,
        )
    np.savez_compressed(
        artifact / "targeted_executed_actions.npz",
        schema=np.asarray("iterative_residual_targeted_executed_actions_v1"),
        context_ids=data["context_ids"],
        normalized_actions=np.stack(action_history, axis=1),
    )
    _write_json(artifact / "targeted_selection_diagnostics.json", selection_history)
    best = max(
        iterations,
        key=lambda item: (
            item["summary"]["DEVELOPMENT"]["scientific_success_rate"],
            item["summary"]["DEVELOPMENT"]["feasible_rate"],
            -item["execution_count"],
        ),
    )
    result = {
        "schema": "iterative_residual_targeted_authoritative_evaluation_v1",
        "iterations": iterations,
        "recommended_execution_count": best["execution_count"],
        "recommended_correction_count": best["execution_count"] - 1,
        "candidate_corrections_ranked_by_simulator": False,
    }
    _write_json(artifact / "targeted_iterative_evaluation.json", result)
    return result


def _write_report(artifact, config, dataset, training, evaluation) -> str:
    baseline = evaluation["iterations"][0]["summary"]
    best = evaluation["iterations"][evaluation["recommended_execution_count"] - 1]["summary"]
    train_gain = 100 * (best["TRAIN"]["scientific_success_rate"] - baseline["TRAIN"]["scientific_success_rate"])
    dev_gain = 100 * (best["DEVELOPMENT"]["scientific_success_rate"] - baseline["DEVELOPMENT"]["scientific_success_rate"])
    source_dev = 43.24
    best_dev = 100 * best["DEVELOPMENT"]["scientific_success_rate"]
    if best_dev >= 50.0 and best_dev > source_dev:
        classification = "TARGETED_RESIDUAL_TRAINING_PROMISING"
    elif best_dev > source_dev:
        classification = "TARGETED_RESIDUAL_TRAINING_PARTIAL"
    else:
        classification = "TARGETED_RESIDUAL_DATA_NO_IMPROVEMENT"
    report = f"""# Targeted Iterative Residual Training Report

## Outcome

This continuation trained the same 372k-parameter residual-outcome architecture on simulator responses centered on actions that actually failed. It ran zero CEM solves and preserved the frozen production model/action contract.

## Targeted data

- Failed training contexts: {dataset['training_contexts']}
- Failed state-disjoint development contexts: {dataset['development_contexts']}
- Candidate responses/context: {config['targeted_data']['candidates_per_context']}
- New physical response rows: {dataset['candidate_rows']}
- Successful response rows: {dataset['successful_rows']}
- Feasible response rows: {dataset['feasible_rows']}
- Saved trajectory feature: 25 time samples of tip-target position, tip velocity, and UAV displacement
- New CEM solves: 0

Candidate corrections were generated in the existing 13-D basis around the failed 7B initializer, then every response was propagated by the unchanged full production simulator.

## Training

The prior IRP checkpoint was continued with 50% original teacher-neighborhood pairs and 50% targeted failed-initializer pairs. Outcome-balanced target sampling prevents the rare useful targeted responses from disappearing. Best targeted development loss: {training['best_validation_loss']:.6f} at update {training['best_update']} (initial {training['initial_validation_loss']:.6f}).

## Authoritative correction result

| Executions | Train success | Train feasible | Development success | Development feasible |
|---:|---:|---:|---:|---:|
"""
    for item in evaluation["iterations"]:
        summary = item["summary"]
        report += (
            f"| {item['execution_count']} | {100*summary['TRAIN']['scientific_success_rate']:.2f}% | "
            f"{100*summary['TRAIN']['feasible_rate']:.2f}% | "
            f"{100*summary['DEVELOPMENT']['scientific_success_rate']:.2f}% | "
            f"{100*summary['DEVELOPMENT']['feasible_rate']:.2f}% |\n"
        )
    report += f"""

Best stopping point: {evaluation['recommended_correction_count']} corrections. At that point train success improved by {train_gain:+.2f} percentage points and development success by {dev_gain:+.2f} points from the one-action initializer.

The previous teacher-neighborhood IRP reached 43.24% development success. This targeted continuation reached {best_dev:.2f}%. Classification: **{classification}**.

## Interpretation

This isolates whether relevant residual-response coverage—not merely longer neural optimization—was limiting the previous model. The trajectory features are durably saved but were not yet added to the network input, so any improvement comes strictly from moving the training distribution around the real failure actions. Candidate ranking remained neural-only; unselected candidates were not searched with physics during evaluation.

## Restrictions

- Production model modified: **NO**
- CEM: **NOT RUN**
- SAC/diffusion/scorer: **NOT USED**
- Final TEST: **NOT EVALUATED**
- Protected test: **NOT EVALUATED**
- Hardware: **NOT EXECUTED**

## Final summary

    New CEM solves:
        0

    New targeted simulator rows:
        {dataset['candidate_rows']}

    Training updates:
        {training['updates_completed']}

    Initial development success:
        {100*baseline['DEVELOPMENT']['scientific_success_rate']:.2f}%

    Best development success:
        {best_dev:.2f}%

    Recommended corrections:
        {evaluation['recommended_correction_count']}

    Result:
        {classification}

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    (artifact / "ITERATIVE_RESIDUAL_TARGETED_TRAINING_REPORT.md").write_text(report, encoding="utf-8")
    ROOT_REPORT.write_text(report, encoding="utf-8")
    return classification


def run(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    artifact = ARTIFACT_PARENT / _timestamp()
    artifact.mkdir(parents=True, exist_ok=False)
    _write_json(artifact / "config.json", config)
    data = _load_existing_data(config)
    records, inventory = load_production_cem_teachers(config)
    if [record.context_id for record in records] != data["context_ids"].tolist():
        raise RuntimeError("Teacher order changed.")
    manifest = _select_failed_contexts(config, data, records)
    if manifest["state_overlap"]:
        raise RuntimeError("Targeted train/development states overlap.")
    _write_json(artifact / "targeted_context_manifest.json", manifest)
    basis = torch.load(ROOT / config["source_irp_artifact"] / "compact_residual_basis.pt", weights_only=True)
    environment = load_fixed_production_environment(config)
    dataset_summary = _generate_targeted_data(
        config, artifact, data, records, manifest, basis, environment
    )
    targeted = _load_targeted_tensors(artifact, manifest)
    model, normalizer, training = _fine_tune(config, data, targeted, artifact)
    evaluation = _evaluate_iterative(
        config, data, records, basis, environment, model, normalizer, artifact
    )
    classification = _write_report(artifact, config, dataset_summary, training, evaluation)
    _write_json(
        artifact / "existing_data_inventory.json",
        {**inventory, "source_irp_artifact": config["source_irp_artifact"], "new_cem_solves": 0},
    )
    hash_inputs = [
        Path(__file__),
        ROOT / "run_iterative_residual_policy.py",
        ROOT / "learning" / "iterative_residual.py",
        config_path,
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {"schema": "targeted_irp_source_hash_manifest_v1", "files": {
            str(path.relative_to(ROOT)): _sha256(path) for path in hash_inputs
        }},
    )
    print(f"classification={classification}", flush=True)
    print(f"artifact={artifact}", flush=True)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config if args.config.is_absolute() else ROOT / args.config)


if __name__ == "__main__":
    main()
