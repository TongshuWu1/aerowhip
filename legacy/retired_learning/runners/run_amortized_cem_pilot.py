"""Amortized CEM pilot: offline teacher labels, then supervised one-shot policy.

This runner intentionally contains no critic, replay buffer, entropy term, or
online optimizer.  CEM is used only while constructing the offline dataset.
The deployed artifact is one 83-D -> 49-D feed-forward network query.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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

from learning.amortized_policy import (
    AmortizedTrajectoryActor,
    action_error_metrics,
    save_teacher_dataset,
    train_amortized_actor,
)
from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    target_direction_from_local_target,
)
from learning.normalization import FixedContextNormalizer
from learning.one_shot_env import evaluate_open_loop_batch
from learning.policy_action import encode_physical_action
from learning.policy_context import PolicyContext, build_policy_context, policy_context_tensor_metadata
from learning.state_bank import InitialStateBank, initial_state_bank_from_state
from planning.artifacts import final_replay_metrics, save_command_csv, save_replay_npz
from planning.cem_task import load_variable_duration_task
from planning.results import PlanningResult
from planning.rollout import hover_preroll
from planning.teacher_cem import TeacherCemSettings, optimize_teacher_context
from planning.video import render_replay_video
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest
from simulator.uav import FullStateUAVModel


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "amortized_cem_nominal_pilot_v1.json"
REPORT_PATH = ROOT / "AMORTIZED_CEM_ONESHOT_POLICY_PILOT_REPORT.md"


@dataclass(frozen=True, slots=True)
class ContextRecord:
    context_id: str
    split: str
    bank_name: str
    state_index: int
    target_local_m: tuple[float, float, float]
    direction_local: tuple[float, float, float]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+0000", "Z")


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "amortized_cem_nominal_pilot_v1":
        raise ValueError("Unsupported amortized-CEM pilot configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The production model freeze is immutable.")
    if config.get("context_dimension") != 83 or config.get("action_dimension") != 49:
        raise ValueError("The verified one-shot 83-D/49-D contract changed.")
    if config.get("physics") != "NOMINAL_ONLY":
        raise ValueError("This pilot permits nominal physics only.")
    policy = config["artifact_policy"]
    if bool(policy["real_flight_authorized"]) or bool(policy["protected_test_evaluation_allowed"]):
        raise ValueError("The pilot is simulation-only and may not access the protected test.")
    if bool(policy["sac_used"]) or bool(policy["cem_used_online"]):
        raise ValueError("Neither SAC nor online CEM belongs in this pilot.")


def _canonical_setup(simulator, task) -> tuple[InitialStateBank, PolicyContext]:
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


def _nearest_indices(
    simulator,
    bank: InitialStateBank,
    canonical_context: PolicyContext,
    normalizer: FixedContextNormalizer,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = canonical_context.target_position_local_m.detach().cpu()
    direction = canonical_context.target_direction_local.detach().cpu()
    reference = normalizer.normalize(canonical_context.to_tensor())[0, :70]
    distances: list[torch.Tensor] = []
    for start in range(0, len(bank), 2048):
        stop = min(start + 2048, len(bank))
        rows = stop - start
        specification = ContextSpecification(
            torch.arange(start, stop), target.repeat(rows, 1), direction.repeat(rows, 1), "nearest_state_audit"
        )
        context = build_context_from_specification(simulator, bank, specification)
        feature = normalizer.normalize(context.to_tensor())[:, :70]
        distances.append(torch.sqrt(torch.mean((feature - reference).square(), dim=-1)).cpu())
    all_distances = torch.cat(distances)
    order = torch.argsort(all_distances)
    return order[:count], all_distances[order[:count]]


def _targets(
    canonical_target: torch.Tensor,
    *,
    count: int,
    half_width: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # Symmetric antithetic offsets keep every split centered on the known mode.
    half = (count + 1) // 2
    offset = (2.0 * torch.rand((half, 3), generator=generator) - 1.0) * half_width
    offset = torch.cat((offset, -offset), dim=0)[:count]
    return canonical_target.reshape(1, 3) + offset


def _make_records(
    config: dict[str, Any],
    canonical_context: PolicyContext,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
) -> tuple[list[ContextRecord], dict[str, Any]]:
    data = config["dataset_pilot"]
    per_state = int(data["targets_per_state"])
    train_state_count = int(data["training_state_count"])
    val_state_count = int(data["validation_state_count"])
    test_state_count = int(data["test_state_count"])
    canonical_target = canonical_context.target_position_local_m.detach().cpu()[0]
    half_width = torch.tensor(data["target_offset_half_width_m"], dtype=torch.float32)
    records: list[ContextRecord] = []

    if bool(data["include_canonical_training_context"]):
        direction = canonical_context.target_direction_local.detach().cpu()[0]
        records.append(ContextRecord("train_canonical", "train", "canonical", 0, tuple(canonical_target.tolist()), tuple(direction.tolist())))

    groups = (
        ("train", "training", train_indices[:train_state_count], 1001),
        ("validation", "validation", validation_indices[:val_state_count], 2001),
        (
            "test",
            "validation",
            validation_indices[val_state_count : val_state_count + test_state_count],
            3001,
        ),
    )
    for split, bank_name, indices, seed in groups:
        targets = _targets(canonical_target, count=len(indices) * per_state, half_width=half_width, seed=seed)
        directions = target_direction_from_local_target(targets)
        cursor = 0
        for local_state_number, state_index in enumerate(indices.tolist()):
            for target_number in range(per_state):
                records.append(
                    ContextRecord(
                        f"{split}_{local_state_number:03d}_{target_number:02d}",
                        split,
                        bank_name,
                        int(state_index),
                        tuple(targets[cursor].tolist()),
                        tuple(directions[cursor].tolist()),
                    )
                )
                cursor += 1
    manifest = {
        "schema": "amortized_cem_context_split_v1",
        "designed_counts": {split: sum(row.split == split for row in records) for split in ("train", "validation", "test")},
        "state_overlap": {
            "training_vs_validation": False,
            "training_vs_test": False,
            "validation_vs_test": bool(set(validation_indices[:val_state_count].tolist()) & set(validation_indices[val_state_count : val_state_count + test_state_count].tolist())),
        },
        "records": [asdict(row) for row in records],
    }
    return records, manifest


def _build_record_context(simulator, banks: dict[str, InitialStateBank], record: ContextRecord) -> PolicyContext:
    target = torch.tensor(record.target_local_m, dtype=torch.float32).reshape(1, 3)
    direction = torch.tensor(record.direction_local, dtype=torch.float32).reshape(1, 3)
    specification = ContextSpecification(torch.tensor([record.state_index]), target, direction, record.split)
    return build_context_from_specification(simulator, banks[record.bank_name], specification)


def _teacher_settings(config: dict[str, Any]) -> TeacherCemSettings:
    teacher = config["teacher_cem"]
    keys = TeacherCemSettings.__dataclass_fields__.keys()
    return TeacherCemSettings(**{key: teacher[key] for key in keys if key in teacher})


def _load_canonical_decision(config: dict[str, Any], task) -> torch.Tensor:
    source = ROOT / config["canonical_teacher_artifact"]
    knots = json.loads((source / "best_acceleration_knots.json").read_text(encoding="utf-8"))["values"]
    duration = json.loads((source / "optimized_duration.json").read_text(encoding="utf-8"))["duration_s"]
    decision = torch.cat((torch.tensor(knots, dtype=torch.float64).reshape(-1), torch.tensor([duration], dtype=torch.float64)))
    if decision.shape != (49,):
        raise RuntimeError("Canonical CEM teacher action has the wrong shape.")
    return decision


def _evaluate_actor_rows(
    actor: AmortizedTrajectoryActor,
    normalizer: FixedContextNormalizer,
    simulator,
    task,
    banks: dict[str, InitialStateBank],
    labels: list[dict[str, Any]],
    evaluation_time_s: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    actor.eval()
    for item in labels:
        record = ContextRecord(**item["record"])
        context = _build_record_context(simulator, banks, record)
        with torch.no_grad():
            prediction = actor(normalizer.normalize(context.to_tensor()))
            episode = evaluate_open_loop_batch(
                simulator, context, prediction, task, evaluation_time_s=evaluation_time_s
            )
        result = episode.row(0)
        result.update(
            {
                "context_id": record.context_id,
                "split": record.split,
                "teacher_success": True,
                "normalized_action_error": action_error_metrics(
                    prediction.cpu(), torch.tensor(item["normalized_action"], dtype=torch.float32).reshape(1, 49)
                ),
                "predicted_normalized_action": prediction.detach().cpu()[0].tolist(),
            }
        )
        rows.append(result)
    return rows


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "success_rate": None, "feasible_rate": None}
    tip = np.asarray([row["tip_min_distance_m"] for row in rows], dtype=float)
    return {
        "count": len(rows),
        "success_count": sum(bool(row["task_success"]) for row in rows),
        "success_rate": float(np.mean([row["task_success"] for row in rows])),
        "feasible_rate": float(np.mean([row["feasible"] for row in rows])),
        "median_tip_min_distance_m": float(np.median(tip)),
        "mean_tip_min_distance_m": float(np.mean(tip)),
        "median_directed_tip_speed_m_s": float(np.median([row["directed_tip_speed_m_s"] for row in rows])),
        "median_direction_error_deg": float(np.median([row["direction_error_deg"] for row in rows])),
        "mean_duration_s": float(np.mean([row["duration_s"] for row in rows])),
    }


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    start = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    artifact = ROOT / "data" / "policy_training" / config["run_id"] / _timestamp()
    artifact.mkdir(parents=True, exist_ok=False)
    (artifact / "teacher_checkpoints").mkdir()
    _write_json(artifact / "config.json", config)
    _write_json(artifact / "context_schema.json", policy_context_tensor_metadata())

    active = load_active_model_manifest()
    settings = SimulatorSettings.load(active_model_paths(active)["configuration"])
    simulator = build_production_simulator(settings, device="cuda", dtype=torch.float32)
    if not isinstance(simulator.uav_model, FullStateUAVModel):
        raise TypeError("The pilot requires the production FullState UAV model.")
    simulator.uav_model.set_fixed_evaluation_batch_size(2048)
    task = load_variable_duration_task(ROOT / config["task_config"])
    task.validate_for_dt(simulator.dt_s)
    if task.model_freeze != config["model_freeze"]:
        raise ValueError("Task and requested model freeze differ.")
    _write_json(artifact / "task_config_snapshot.json", task.snapshot())

    bank_source = ROOT / config["state_bank_artifact"]
    training_bank = InitialStateBank.load(bank_source / "training_state_bank.npz", bank_source / "training_state_bank_manifest.json")
    validation_bank = InitialStateBank.load(bank_source / "validation_state_bank.npz", bank_source / "validation_state_bank_manifest.json")
    normalizer = FixedContextNormalizer.load(bank_source / "context_normalizer.json")
    shutil.copy2(bank_source / "context_normalizer.json", artifact / "context_normalizer.json")
    canonical_bank, canonical_context = _canonical_setup(simulator, task)
    banks = {"canonical": canonical_bank, "training": training_bank, "validation": validation_bank}

    data_config = config["dataset_pilot"]
    train_indices, train_distances = _nearest_indices(
        simulator, training_bank, canonical_context, normalizer, int(data_config["training_state_count"])
    )
    validation_total = int(data_config["validation_state_count"]) + int(data_config["test_state_count"])
    validation_indices, validation_distances = _nearest_indices(
        simulator, validation_bank, canonical_context, normalizer, validation_total
    )
    records, split_manifest = _make_records(config, canonical_context, train_indices, validation_indices)
    split_manifest["nearest_state_distances"] = {
        "training": train_distances.tolist(),
        "validation_and_test": validation_distances.tolist(),
    }
    _write_json(artifact / "context_split_manifest.json", split_manifest)

    warm = _load_canonical_decision(config, task)
    teacher_settings = _teacher_settings(config)
    labels: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for ordinal, record in enumerate(records):
        context = _build_record_context(simulator, banks, record)
        result = optimize_teacher_context(
            simulator,
            context,
            task,
            warm,
            context_id=record.context_id,
            seed=seed * 10000 + ordinal,
            settings=teacher_settings,
            artifact_directory=artifact / "teacher_checkpoints" / record.context_id,
        )
        attempt = {
            "record": asdict(record),
            "teacher_success_in_population": result.success,
            "iterations": result.iterations,
            "rollouts": result.rollouts,
            "runtime_s": result.runtime_s,
            "population_metrics": result.metrics,
            "batch_one_replay_success": False,
        }
        if result.decision_local is not None:
            knots = result.decision_local[:-1].reshape(task.cem.knot_count, 3).to(torch.float32)
            normalized = encode_physical_action(knots, result.decision_local[-1], task)
            verification = evaluate_open_loop_batch(
                simulator,
                context,
                normalized,
                task,
                evaluation_time_s=teacher_settings.evaluation_time_s,
            )
            replay_row = verification.row(0)
            attempt["batch_one_replay"] = replay_row
            attempt["batch_one_replay_success"] = bool(replay_row["task_success"])
            if bool(replay_row["task_success"]):
                labels.append(
                    {
                        "record": asdict(record),
                        "context_tensor": context.to_tensor().detach().cpu()[0].tolist(),
                        "normalized_action": normalized.detach().cpu()[0].tolist(),
                        "physical_decision_local": result.decision_local.tolist(),
                        "teacher_metrics": replay_row,
                        "teacher_iterations": result.iterations,
                        "teacher_runtime_s": result.runtime_s,
                    }
                )
        attempts.append(attempt)
        _write_json(artifact / "teacher_attempts.json", attempts)
        _write_json(artifact / "verified_teacher_labels.json", labels)

    verified_counts = {split: sum(item["record"]["split"] == split for item in labels) for split in ("train", "validation", "test")}
    if verified_counts["train"] < 8 or verified_counts["validation"] < 4 or verified_counts["test"] < 4:
        raise RuntimeError(f"Insufficient verified teacher labels for a meaningful pilot: {verified_counts}")
    contexts = torch.tensor([item["context_tensor"] for item in labels], dtype=torch.float32)
    actions = torch.tensor([item["normalized_action"] for item in labels], dtype=torch.float32)
    save_teacher_dataset(
        artifact / "teacher_dataset.npz",
        contexts=contexts,
        actions=actions,
        split=np.asarray([item["record"]["split"] for item in labels]),
        context_ids=np.asarray([item["record"]["context_id"] for item in labels]),
        state_indices=np.asarray([item["record"]["state_index"] for item in labels]),
        target_positions_local_m=np.asarray([item["record"]["target_local_m"] for item in labels]),
        target_directions_local=np.asarray([item["record"]["direction_local"] for item in labels]),
        teacher_metrics_json=np.asarray([json.dumps(item["teacher_metrics"], sort_keys=True) for item in labels]),
    )

    split = np.asarray([item["record"]["split"] for item in labels])
    train_mask = torch.from_numpy(split == "train")
    validation_mask = torch.from_numpy(split == "validation")
    actor = AmortizedTrajectoryActor().to(simulator.device)
    actor_config = config["supervised_actor"]
    training = train_amortized_actor(
        actor,
        normalizer,
        contexts[train_mask],
        actions[train_mask],
        contexts[validation_mask],
        actions[validation_mask],
        seed=seed,
        learning_rate=float(actor_config["learning_rate"]),
        weight_decay=float(actor_config["weight_decay"]),
        batch_size=int(actor_config["batch_size"]),
        maximum_epochs=int(actor_config["maximum_epochs"]),
        patience_epochs=int(actor_config["patience_epochs"]),
        checkpoint_path=artifact / "best_actor.pt",
    )
    _write_json(artifact / "supervised_training_history.json", list(training.training_history))
    _write_json(artifact / "supervised_training_summary.json", asdict(training))

    evaluation_time_s = float(config["time_contract"]["evaluation_time_s"])
    evaluation_rows = _evaluate_actor_rows(actor, normalizer, simulator, task, banks, labels, evaluation_time_s)
    _write_json(artifact / "policy_evaluation_rows.json", evaluation_rows)
    split_summaries = {
        name: _split_summary([row for row in evaluation_rows if row["split"] == name])
        for name in ("train", "validation", "test")
    }

    canonical_prediction = actor(normalizer.normalize(canonical_context.to_tensor()))
    canonical_episode = evaluate_open_loop_batch(
        simulator,
        canonical_context,
        canonical_prediction,
        task,
        record_trajectory=True,
        evaluation_time_s=evaluation_time_s,
    )
    if canonical_episode.trajectory is None:
        raise RuntimeError("Canonical authoritative replay was not recorded.")
    replay = canonical_episode.trajectory
    save_command_csv(artifact / "canonical_fullstate_command.csv", replay)
    save_replay_npz(artifact / "canonical_final_replay.npz", replay, task)
    save_replay_npz(artifact / "final_replay.npz", replay, task)
    final = final_replay_metrics(replay, task, settings)
    final.update(
        {
            "optimizer": "Amortized CEM actor",
            "task_classification": "PASS" if canonical_episode.row(0)["task_success"] else "FAIL",
            "maneuver_duration_s": canonical_episode.row(0)["duration_s"],
            "evaluation_time_s": evaluation_time_s,
            "policy_row": canonical_episode.row(0),
        }
    )
    _write_json(artifact / "canonical_final_metrics.json", final)
    _write_json(artifact / "final_metrics.json", final)
    _write_json(artifact / "evaluation_summary.json", split_summaries)
    np.save(artifact / "canonical_predicted_normalized_action.npy", canonical_prediction.detach().cpu().numpy())

    planning_result = PlanningResult(
        task_id=task.task_id,
        task_label="Amortized CEM one-shot actor",
        directory=artifact,
        task_config=task.snapshot(),
        metrics=final,
        iteration_history=(),
    )
    video_source, video_metadata = render_replay_video(planning_result, overwrite=True)
    video = artifact / "amortized_cem_actor_canonical_final_replay.mp4"
    if video_source != video:
        shutil.move(video_source, video)
    video_metadata["video_path"] = str(video)
    _write_json(artifact / "video_metadata.json", video_metadata)

    source_files = (
        Path(__file__),
        ROOT / "planning" / "teacher_cem.py",
        ROOT / "planning" / "variable_duration.py",
        ROOT / "learning" / "amortized_policy.py",
        ROOT / "learning" / "one_shot_env.py",
        ROOT / "learning" / "policy_context.py",
        ROOT / "learning" / "policy_action.py",
        config_path.resolve(),
    )
    _write_json(
        artifact / "source_hash_manifest.json",
        {str(path.resolve().relative_to(ROOT)): _sha256(path.resolve()) for path in source_files},
    )
    result = {
        "artifact_directory": str(artifact),
        "designed_context_counts": split_manifest["designed_counts"],
        "verified_teacher_counts": verified_counts,
        "teacher_success_rate": len(labels) / len(records),
        "training": asdict(training),
        "policy_evaluation": split_summaries,
        "canonical": canonical_episode.row(0),
        "video": str(video),
        "runtime_s": time.perf_counter() - start,
        "sac_used": False,
        "cem_online": False,
        "physics_conditioning": "NOT_ENABLED",
        "protected_test": "NOT_EVALUATED",
        "real_hardware": "NOT_EXECUTED",
    }
    _write_json(artifact / "run_summary.json", result)
    print(json.dumps(_safe(result), indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
