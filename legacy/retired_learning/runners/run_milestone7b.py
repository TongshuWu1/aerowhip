"""Milestone 7B: simple deterministic amortization of existing CEM solutions.

The runner intentionally performs no optimization in the simulator.  It first
audits whether fixed cubic splines preserve the already-authoritative 6A
maneuvers, then trains one small residual MLP and compares one-action baselines.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
import torch.nn.functional as F

from learning.cem_residual_policy import DeterministicCemResidualPolicy
from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    pad_context_specification,
)
from learning.normalization import FixedContextNormalizer
from learning.policy_action import decode_policy_action, encode_physical_action
from learning.state_bank import InitialStateBank, initial_state_bank_from_state
from learning.trajectory_primitive import CompleteActionCodec, FixedCubicSplineActionCodec
from planning.rollout import hover_preroll
from planning.production_cem import (
    FIXED_NUMERICAL_BATCH_SIZE,
    evaluate_normalized_actions_fixed_batch,
    event_segment,
)
from run_milestone7a1 import _physics_settings
from simulator.parameters import SimulatorSettings
from simulator.production import active_model_paths, build_production_simulator, load_active_model_manifest


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "structured_cem_residual_policy_v1.json"
ARTIFACT_PARENT = ROOT / "data" / "policy_training" / "structured_cem_residual_policy_v1"
REPORT = ROOT / "MILESTONE7B_STRUCTURED_CEM_RESIDUAL_POLICY_REPORT.md"


@dataclass(frozen=True, slots=True)
class TeacherRecord:
    context_id: str
    group: str
    bank: str
    state_index: int
    state_id: str
    target_local_m: tuple[float, float, float]
    direction_local: tuple[float, float, float]
    action: np.ndarray
    source_metrics: dict[str, Any]


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
    if config.get("schema") != "structured_cem_residual_policy_v1":
        raise ValueError("Unsupported Milestone-7B configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The frozen production model changed.")
    prohibited = config["prohibitions"]
    for key in (
        "new_cem_solves",
        "diffusion",
        "scorer",
        "sac",
        "final_test",
        "theta_randomization",
        "hardware",
    ):
        if not bool(prohibited[key]):
            raise ValueError(f"Required prohibition is disabled: {key}")
    if prohibited["protected_test"] != "fig8vertical_002":
        raise ValueError("Protected-test identity changed.")


def _load_fixed_environment(config: dict[str, Any]):
    """Recreate the exact Milestone-6A pre-roll ordering and batch contract."""

    manifest = load_active_model_manifest()
    paths = active_model_paths(manifest)
    simulator_settings = SimulatorSettings.load(paths["configuration"])
    simulator = build_production_simulator(
        simulator_settings, device=torch.device("cuda"), dtype=torch.float32
    )
    # This must precede hover pre-roll.  The frozen residual is numerically
    # shape-sensitive at float32, and Milestone 6A settled the canonical state
    # under the fixed-2048 residual evaluation contract.
    simulator.uav_model.set_fixed_evaluation_batch_size(FIXED_NUMERICAL_BATCH_SIZE)
    from planning.cem_task import load_variable_duration_task

    task = load_variable_duration_task(ROOT / config["task_config"])
    state_root = ROOT / config["state_bank_artifact"]
    training_bank = InitialStateBank.load(
        state_root / "training_state_bank.npz",
        state_root / "training_state_bank_manifest.json",
    )
    state = hover_preroll(simulator, task)
    canonical_bank = initial_state_bank_from_state(
        state,
        command_position_world_m=torch.tensor(task.initial_uav_position_m, device=simulator.device),
        command_velocity_world_m_s=torch.tensor(task.initial_uav_velocity_m_s, device=simulator.device),
        command_yaw_world_rad=task.initial_yaw_rad,
        seed=42,
    )
    return simulator, task, training_bank, None, canonical_bank, simulator_settings


def _load_records(config: dict[str, Any]) -> tuple[list[TeacherRecord], dict[str, Any]]:
    source = ROOT / config["milestone_6a_artifact"]
    manifest = json.loads((source / "benchmark_context_manifest.json").read_text(encoding="utf-8"))
    rows = json.loads((source / "benchmark_rows.json").read_text(encoding="utf-8"))
    with np.load(source / "authoritative_actions.npz", allow_pickle=False) as archive:
        action_ids = [str(value) for value in archive["context_ids"].tolist()]
        actions = np.asarray(archive["normalized_actions"], dtype=np.float32)
    if len(manifest["rows"]) != len(rows) or len(rows) != len(actions):
        raise RuntimeError("Milestone-6A artifact row counts disagree.")
    action_by_id = dict(zip(action_ids, actions, strict=True))
    row_by_id = {row["context_id"]: row for row in rows}
    records: list[TeacherRecord] = []
    failed: list[str] = []
    maximum_action_difference = 0.0
    for definition in manifest["rows"]:
        context_id = str(definition["context_id"])
        row = row_by_id[context_id]
        saved = action_by_id[context_id]
        inline = np.asarray(row["normalized_action"], dtype=np.float32)
        maximum_action_difference = max(maximum_action_difference, float(np.max(np.abs(saved - inline))))
        if not bool(row["authoritative_metrics"]["success"]):
            failed.append(context_id)
            continue
        bank = str(definition["state_bank"])
        state_index = int(definition["state_id"])
        state_id = "canonical:0000" if bank == "canonical" else f"training:{state_index:04d}"
        records.append(
            TeacherRecord(
                context_id=context_id,
                group=str(definition["group"]),
                bank=bank,
                state_index=state_index,
                state_id=state_id,
                target_local_m=tuple(float(value) for value in definition["target_local_m"]),
                direction_local=tuple(float(value) for value in definition["direction_local"]),
                action=saved.copy(),
                source_metrics=dict(row["authoritative_metrics"]),
            )
        )
    if len(records) != 252:
        raise RuntimeError(f"Expected 252 verified Milestone-6A teachers, found {len(records)}.")
    inventory = {
        "schema": "structured_cem_existing_teacher_inventory_v1",
        "source": str(source),
        "designed_contexts": len(rows),
        "authoritative_successes": len(records),
        "authoritative_failures": len(failed),
        "failed_context_ids": failed,
        "unique_state_ids": len({record.state_id for record in records}),
        "group_counts": {
            group: sum(record.group == group for record in records)
            for group in ("A_TARGET", "B_STATE", "C_JOINT", "D_EDGE")
        },
        "one_action_per_context": True,
        "maximum_npz_vs_json_action_difference": maximum_action_difference,
        "action_schema": "normalized production [49]",
        "command_semantics": "ACTIVE -> 0.30-s analytic SETTLE -> HOLD; evaluation 2.40 s",
        "model_freeze": config["model_freeze"],
        "cem_family": "single independently rotated canonical Milestone-6A warm-start family",
        "partial_7a_multisolution_labels_used": False,
        "new_cem_solves": 0,
    }
    return records, inventory


def _load_canonical_seed_action(config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    source = ROOT / config["milestone_6a_artifact"]
    rows = json.loads((source / "canonical_seed_results.json").read_text(encoding="utf-8"))
    successful = [row for row in rows if bool(row["success"])]
    selected = min(
        successful,
        key=lambda row: (
            float(row["authoritative_metrics"]["task_cost"]),
            float(row["authoritative_metrics"]["first_entry_tip_distance_m"]),
        ),
    )
    return torch.tensor(selected["normalized_action"], dtype=torch.float64), selected


def _rotate_normalized_action(
    action: torch.Tensor,
    direction_local: Iterable[float],
    task,
    settings,
) -> torch.Tensor:
    decoded = decode_policy_action(action, task, duration_max_s=settings.duration_max_s)
    direction = torch.tensor(tuple(direction_local), dtype=torch.float64)
    angle = torch.atan2(direction[1], direction[0])
    cosine, sine = torch.cos(angle), torch.sin(angle)
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine, torch.zeros_like(cosine))),
            torch.stack((sine, cosine, torch.zeros_like(cosine))),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    knots = decoded.acceleration_knots_local_m_s2[0].double() @ rotation.T
    return encode_physical_action(
        knots,
        decoded.duration_s[0].double(),
        task,
        duration_max_s=settings.duration_max_s,
    )[0].double()


def _context_table(environment: tuple, records: list[TeacherRecord]) -> np.ndarray:
    simulator, _task, training_bank, _heldout, canonical_bank, _source = environment
    values: list[np.ndarray] = []
    for record in records:
        bank = canonical_bank if record.bank == "canonical" else training_bank
        specification = ContextSpecification(
            torch.tensor([record.state_index], dtype=torch.int64),
            torch.tensor([record.target_local_m], dtype=torch.float32),
            torch.tensor([record.direction_local], dtype=torch.float32),
            "structured_cem_teacher",
        )
        context = build_context_from_specification(simulator, bank, specification)
        values.append(context.to_tensor()[0].detach().cpu().numpy().astype(np.float32))
    result = np.stack(values)
    if result.shape != (len(records), 83) or not np.isfinite(result).all():
        raise RuntimeError("Policy context table is invalid.")
    return result


def _evaluate_actions(
    environment: tuple,
    config: dict[str, Any],
    records: list[TeacherRecord],
    actions: np.ndarray,
    *,
    label: str,
) -> list[list[dict[str, Any]]]:
    """Evaluate R x N actions with varied contexts under the fixed-2048 contract."""

    simulator, task, training_bank, _heldout, canonical_bank, _source = environment
    settings = _physics_settings(config)
    if actions.ndim != 3 or actions.shape[0] != len(records) or actions.shape[2] != 49:
        raise ValueError("Evaluation actions must have shape contexts x candidates x 49.")
    candidate_count = int(actions.shape[1])
    rows: list[list[dict[str, Any]] | None] = [None] * len(records)
    record_index = {record.context_id: index for index, record in enumerate(records)}
    for bank_name, bank in (("canonical", canonical_bank), ("training", training_bank)):
        selected = [record for record in records if record.bank == bank_name]
        per_batch = max(1, FIXED_NUMERICAL_BATCH_SIZE // candidate_count)
        for start in range(0, len(selected), per_batch):
            group = selected[start : start + per_batch]
            indices = [record_index[record.context_id] for record in group]
            flat_actions = torch.from_numpy(actions[indices].reshape(-1, 49)).float()
            state_indices: list[int] = []
            targets: list[tuple[float, float, float]] = []
            directions: list[tuple[float, float, float]] = []
            for record in group:
                state_indices.extend([record.state_index] * candidate_count)
                targets.extend([record.target_local_m] * candidate_count)
                directions.extend([record.direction_local] * candidate_count)
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
                simulator, context, flat_actions, task, settings
            )
            decoded = decode_policy_action(
                flat_actions, task, duration_max_s=settings.duration_max_s
            )
            for local_index, record in enumerate(group):
                output: list[dict[str, Any]] = []
                for candidate in range(candidate_count):
                    flat_index = local_index * candidate_count + candidate
                    metric = result.row(flat_index)
                    duration = float(decoded.duration_s[flat_index])
                    metric["maneuver_duration_s"] = duration
                    metric["hit_segment"] = event_segment(
                        metric["first_entry_time_s"], duration, settings.settle_duration_s
                    )
                    output.append(metric)
                rows[record_index[record.context_id]] = output
            print(
                f"{label}: {min(start + len(group), len(selected))}/{len(selected)} {bank_name}",
                flush=True,
            )
    if any(row is None for row in rows):
        raise RuntimeError("An evaluation context was not populated.")
    return [row for row in rows if row is not None]


def _primitive_audit(
    environment: tuple,
    config: dict[str, Any],
    records: list[TeacherRecord],
) -> tuple[Any, dict[str, Any]]:
    teacher = torch.from_numpy(np.stack([record.action for record in records])).double()
    codecs = [FixedCubicSplineActionCodec(int(count)) for count in config["primitive_control_points"]]
    candidates = [teacher.float().numpy()]
    reconstruction: dict[int, torch.Tensor] = {}
    for codec in codecs:
        decoded = codec.decode(codec.encode(teacher))
        reconstruction[codec.control_point_count] = decoded
        candidates.append(decoded.float().numpy())
    stacked = np.stack(candidates, axis=1)
    metrics = _evaluate_actions(environment, config, records, stacked, label="primitive_audit")
    original_success = np.asarray([row[0]["success"] for row in metrics], dtype=bool)
    if not bool(original_success.all()):
        failures = [records[index].context_id for index in np.flatnonzero(~original_success)]
        raise RuntimeError(f"Authoritative teacher replay changed for contexts: {failures}")
    entries = []
    selected_codec: Any = CompleteActionCodec()
    selected_name = "FULL_49D"
    threshold = float(config["primitive_success_retention_threshold"])
    for candidate_index, codec in enumerate(codecs, start=1):
        reconstructed = reconstruction[codec.control_point_count]
        success = np.asarray([row[candidate_index]["success"] for row in metrics], dtype=bool)
        feasible = np.asarray([row[candidate_index]["feasible"] for row in metrics], dtype=bool)
        difference = reconstructed - teacher
        entry = {
            "control_points_per_axis": codec.control_point_count,
            "latent_dimension": codec.latent_dimension,
            "teacher_contexts": len(records),
            "authoritative_success_count": int(success.sum()),
            "authoritative_success_retention": float(success.mean()),
            "authoritative_feasible_count": int(feasible.sum()),
            "normalized_action_rmse": float(torch.sqrt(torch.mean(difference.square()))),
            "normalized_action_max_abs_error": float(difference.abs().max()),
            "duration_max_abs_error": float(difference[:, -1].abs().max()),
            "passes_90_percent_gate": bool(success.mean() >= threshold),
        }
        entries.append(entry)
        if selected_name == "FULL_49D" and entry["passes_90_percent_gate"]:
            selected_codec = codec
            selected_name = f"CUBIC_BSPLINE_{codec.control_point_count}_PER_AXIS"
    audit = {
        "schema": "structured_cem_primitive_replay_audit_v1",
        "original_teacher_replay_success": int(original_success.sum()),
        "original_teacher_count": len(records),
        "required_success_retention": threshold,
        "representations": entries,
        "selected_representation": selected_name,
        "selected_latent_dimension": selected_codec.latent_dimension,
        "fallback_to_full_49d": selected_name == "FULL_49D",
        "duration_preserved_exactly": True,
        "production_decoder_used": True,
        "production_physics_used": True,
    }
    return selected_codec, audit


def _state_split(records: list[TeacherRecord], fraction: float, seed: int) -> dict[str, Any]:
    state_ids = sorted({record.state_id for record in records if record.state_id != "canonical:0000"})
    ranked = sorted(
        state_ids,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    dev_count = max(1, int(round(fraction * len(ranked))))
    dev_states = set(ranked[:dev_count])
    train_states = set(ranked[dev_count:]) | {"canonical:0000"}
    if train_states & dev_states:
        raise RuntimeError("State-level train/development split leaked.")
    return {
        "schema": "structured_cem_state_disjoint_development_split_v1",
        "seed": seed,
        "development_fraction": fraction,
        "training_state_ids": sorted(train_states),
        "development_state_ids": sorted(dev_states),
        "training_context_ids": [record.context_id for record in records if record.state_id in train_states],
        "development_context_ids": [record.context_id for record in records if record.state_id in dev_states],
        "state_overlap": [],
        "canonical_state_assignment": "TRAIN",
        "final_7a_test_used": False,
    }


def _train_policy(
    config: dict[str, Any],
    normalized_contexts: torch.Tensor,
    center_latents: torch.Tensor,
    teacher_latents: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    artifact: Path,
) -> tuple[DeterministicCemResidualPolicy, dict[str, Any]]:
    settings = config["policy"]
    torch.manual_seed(int(config["seed"]))
    device = torch.device("cuda")
    model = DeterministicCemResidualPolicy(
        teacher_latents.shape[1], int(settings["hidden_dimension"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    contexts = normalized_contexts.to(device)
    centers = center_latents.to(device)
    teachers = teacher_latents.to(device)
    train_indices = train_indices.to(device)
    dev_indices = dev_indices.to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    batch_size = min(int(settings["batch_size"]), int(train_indices.numel()))
    best_loss = float("inf")
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for update in range(1, int(settings["maximum_updates"]) + 1):
        selection = torch.randint(
            int(train_indices.numel()), (batch_size,), generator=generator
        ).to(device)
        index = train_indices[selection]
        prediction = model(contexts[index], centers[index])
        loss = F.huber_loss(
            prediction,
            teachers[index],
            delta=float(settings["huber_delta"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
        )
        optimizer.step()
        if update % int(settings["validation_interval"]) == 0:
            model.eval()
            with torch.no_grad():
                train_prediction = model(contexts[train_indices], centers[train_indices])
                dev_prediction = model(contexts[dev_indices], centers[dev_indices])
                train_loss = float(
                    F.huber_loss(train_prediction, teachers[train_indices], delta=1.0)
                )
                dev_loss = float(F.huber_loss(dev_prediction, teachers[dev_indices], delta=1.0))
                train_rmse = float(torch.sqrt(torch.mean((train_prediction - teachers[train_indices]) ** 2)))
                dev_rmse = float(torch.sqrt(torch.mean((dev_prediction - teachers[dev_indices]) ** 2)))
            model.train()
            history.append(
                {
                    "update": update,
                    "minibatch_huber": float(loss.detach()),
                    "train_huber": train_loss,
                    "development_huber": dev_loss,
                    "train_latent_rmse": train_rmse,
                    "development_latent_rmse": dev_rmse,
                    "gradient_norm_before_clip": gradient_norm,
                }
            )
            if dev_loss < best_loss - 1.0e-9:
                best_loss = dev_loss
                best_update = update
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
            if (
                update >= int(settings["minimum_updates"])
                and update - best_update >= int(settings["patience_updates"])
            ):
                break
    if best_state is None:
        raise RuntimeError("Residual policy training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "schema": "deterministic_cem_residual_policy_v1",
            "model_state_dict": best_state,
            "latent_dimension": int(teacher_latents.shape[1]),
            "hidden_dimension": int(settings["hidden_dimension"]),
            "best_update": best_update,
            "best_development_huber": best_loss,
        },
        artifact / "residual_policy_best.pt",
    )
    result = {
        "schema": "deterministic_cem_residual_training_history_v1",
        "configuration": settings,
        "train_context_count": int(train_indices.numel()),
        "development_context_count": int(dev_indices.numel()),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "updates_completed": history[-1]["update"],
        "best_update": best_update,
        "best_development_huber": best_loss,
        "runtime_s": time.perf_counter() - started,
        "history": history,
    }
    return model, result


def _nearest_neighbor_actions(
    normalized_contexts: torch.Tensor,
    teacher_actions: torch.Tensor,
    train_indices: list[int],
    query_indices: list[int],
) -> tuple[torch.Tensor, list[str]]:
    train = torch.tensor(train_indices, dtype=torch.int64)
    output: list[torch.Tensor] = []
    sources: list[str] = []
    for query in query_indices:
        eligible = train[train != query]
        if eligible.numel() == 0:
            eligible = train
        distance = torch.linalg.vector_norm(
            normalized_contexts[eligible] - normalized_contexts[query], dim=-1
        )
        selected = int(eligible[int(torch.argmin(distance))])
        output.append(teacher_actions[selected])
        sources.append(str(selected))
    return torch.stack(output), sources


def _train_memorization_control(
    config: dict[str, Any],
    normalized_contexts: torch.Tensor,
    center_latents: torch.Tensor,
    teacher_latents: torch.Tensor,
    train_indices: torch.Tensor,
    artifact: Path,
) -> tuple[DeterministicCemResidualPolicy, dict[str, Any]]:
    """Fit the same architecture to TRAIN only as a bounded capacity control."""

    settings = config["policy"]
    torch.manual_seed(int(config["seed"]) + 7000)
    device = torch.device("cuda")
    model = DeterministicCemResidualPolicy(
        teacher_latents.shape[1], int(settings["hidden_dimension"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    contexts = normalized_contexts.to(device)
    centers = center_latents.to(device)
    teachers = teacher_latents.to(device)
    indices = train_indices.to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for update in range(1, int(settings["memorization_updates"]) + 1):
        prediction = model(contexts[indices], centers[indices])
        loss = F.huber_loss(
            prediction, teachers[indices], delta=float(settings["huber_delta"])
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
        )
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
        if update % 500 == 0 or update == 1:
            history.append(
                {
                    "update": update,
                    "train_huber": value,
                    "train_latent_rmse": float(
                        torch.sqrt(torch.mean((prediction.detach() - teachers[indices]) ** 2))
                    ),
                    "gradient_norm_before_clip": gradient_norm,
                }
            )
    if best_state is None:
        raise RuntimeError("Memorization control produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(
        {
            "schema": "deterministic_cem_residual_memorization_control_v1",
            "model_state_dict": best_state,
            "latent_dimension": int(teacher_latents.shape[1]),
            "hidden_dimension": int(settings["hidden_dimension"]),
            "updates": int(settings["memorization_updates"]),
            "best_train_huber": best_loss,
        },
        artifact / "residual_policy_train_memorization.pt",
    )
    summary = {
        "schema": "deterministic_cem_residual_memorization_history_v1",
        "purpose": "capacity diagnostic only; not a deployment checkpoint",
        "train_context_count": int(indices.numel()),
        "updates": int(settings["memorization_updates"]),
        "best_train_huber": best_loss,
        "runtime_s": time.perf_counter() - started,
        "history": history,
    }
    return model, summary


def _summarize_method(
    records: list[TeacherRecord],
    rows: list[list[dict[str, Any]]],
    method_index: int,
    split_by_id: dict[str, str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("TRAIN", "DEVELOPMENT", "ALL"):
        indices = [
            index
            for index, record in enumerate(records)
            if split == "ALL" or split_by_id[record.context_id] == split
        ]
        metrics = [rows[index][method_index] for index in indices]
        successes = np.asarray([bool(row["success"]) for row in metrics], dtype=bool)
        feasible = np.asarray([bool(row["feasible"]) for row in metrics], dtype=bool)

        def median(name: str) -> float | None:
            values = np.asarray([float(row[name]) for row in metrics], dtype=np.float64)
            values = values[np.isfinite(values)]
            return float(np.median(values)) if values.size else None

        result[split] = {
            "context_count": len(indices),
            "scientific_success_count": int(successes.sum()),
            "scientific_success_rate": float(successes.mean()),
            "feasible_count": int(feasible.sum()),
            "feasible_rate": float(feasible.mean()),
            "median_minimum_tip_distance_m": median("minimum_tip_target_distance_m"),
            "median_best_event_directed_speed_m_s": median("best_event_directed_speed_m_s"),
            "median_best_event_direction_error_deg": median("best_event_direction_angle_deg"),
            "median_maximum_uav_displacement_m": median("maximum_uav_displacement_m"),
            "median_maximum_uav_speed_m_s": median("maximum_uav_speed_m_s"),
            "median_maneuver_duration_s": median("maneuver_duration_s"),
            "by_group": {
                group: {
                    "count": sum(records[index].group == group for index in indices),
                    "success_rate": float(
                        np.mean(
                            [
                                rows[index][method_index]["success"]
                                for index in indices
                                if records[index].group == group
                            ]
                        )
                    )
                    if any(records[index].group == group for index in indices)
                    else None,
                }
                for group in ("A_TARGET", "B_STATE", "C_JOINT", "D_EDGE")
            },
        }
    return result


def _conditioning_audit(
    environment: tuple,
    config: dict[str, Any],
    records: list[TeacherRecord],
    policy_actions: np.ndarray,
) -> dict[str, Any]:
    index_by_group = {
        group: [index for index, record in enumerate(records) if record.group == group]
        for group in ("A_TARGET", "B_STATE")
    }
    audit: dict[str, Any] = {}
    for name, group in (("target", "A_TARGET"), ("state", "B_STATE")):
        indices = index_by_group[group]
        if len(indices) < 2:
            continue
        shifted = indices[1:] + indices[:1]
        subset_records = [records[index] for index in indices]
        candidates = np.stack(
            (
                policy_actions[indices],
                policy_actions[shifted],
            ),
            axis=1,
        )
        metrics = _evaluate_actions(
            environment, config, subset_records, candidates, label=f"{name}_conditioning_audit"
        )
        correct = np.asarray([row[0]["success"] for row in metrics], dtype=bool)
        swapped = np.asarray([row[1]["success"] for row in metrics], dtype=bool)
        distance = np.linalg.norm(policy_actions[indices] - policy_actions[shifted], axis=1)
        audit[name] = {
            "context_count": len(indices),
            "correct_action_success_rate": float(correct.mean()),
            "swapped_action_success_rate": float(swapped.mean()),
            "median_normalized_action_l2_change": float(np.median(distance)),
            "conditioning": "ACTIVE" if float(np.median(distance)) > 1.0e-4 else "INACTIVE",
        }
    return audit


def _source_hash_manifest(artifact: Path) -> dict[str, Any]:
    paths = [
        ROOT / "learning" / "trajectory_primitive.py",
        ROOT / "learning" / "cem_residual_policy.py",
        ROOT / "run_milestone7b.py",
        ROOT / "config" / "learning" / "structured_cem_residual_policy_v1.json",
        ROOT / "learning" / "policy_action.py",
        ROOT / "learning" / "policy_context.py",
        ROOT / "planning" / "production_cem.py",
        artifact / "residual_policy_best.pt",
        artifact / "residual_policy_train_memorization.pt",
    ]
    return {
        "schema": "structured_cem_residual_source_hash_manifest_v1",
        "files": [
            {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": _sha256(path)}
            for path in paths
            if path.is_file()
        ],
    }


def _percent(value: float | None) -> str:
    return "NOT RUN" if value is None else f"{100.0 * value:.2f}%"


def _write_report(
    artifact: Path,
    inventory: dict[str, Any],
    primitive: dict[str, Any],
    split: dict[str, Any],
    training: dict[str, Any],
    evaluation: dict[str, Any],
    conditioning: dict[str, Any],
) -> None:
    warm = evaluation["methods"]["TARGET_CONDITIONED_CEM_WARM_START"]
    nearest = evaluation["methods"]["NEAREST_NEIGHBOR_RETRIEVAL"]
    policy = evaluation["methods"]["DETERMINISTIC_RESIDUAL_MLP"]
    memorized = evaluation["methods"]["TRAIN_MEMORIZATION_CONTROL"]
    report = f"""# Milestone 7B — Structured CEM Residual Policy Report

## 1. Decision and scope

The diffusion/scorer branch was retired from the active experiment because increasing the frozen generator from 32 to 1,024 candidates still achieved only 60.94% oracle success on heterogeneous states. This run tests the simpler question directly: can one deterministic network interpolate a consistent family of authoritative CEM maneuvers? It ran **zero new CEM solves**, no diffusion, no scorer, no SAC, no protected test, and no hardware.

## 2. Current production pipeline

The active scientific path is:

1. A physically propagated UAV/cable state is converted to the fixed root-centered, yaw-aligned 83-D `PolicyContext`.
2. The goal rotates one canonical Milestone-6A maneuver family into the local strike direction. This is the deterministic target-conditioned center.
3. A small MLP reads the normalized 83-D context and predicts one residual around that center.
4. The selected maneuver coordinate is decoded to the authoritative normalized 49-D production action: 16 three-axis acceleration knots plus maneuver duration.
5. The unchanged decoder generates `ACTIVE -> 0.30-s analytic SETTLE -> HOLD`, with a 2.40-s evaluation horizon.
6. The unchanged frozen UAV/residual/DDER simulator runs through the fixed 2,048-row numerical contract.
7. The unchanged hard scientific gates decide success.

There is one network query, one returned action, and one open-loop execution. CEM appears only offline in the already-existing teacher artifacts.

## 3. Existing teacher data

- Source contexts: {inventory['designed_contexts']}
- Authoritative successful contexts used: {inventory['authoritative_successes']}
- Unique physical state IDs: {inventory['unique_state_ids']}
- Labels/context: exactly one
- Action representation: normalized production 49-D
- Multi-solution partial-7A labels used: no

The one-label rule deliberately avoids averaging several CEM modes for one context.

## 4. Frozen scientific contract

Model: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. The production action codec, 0.45–1.80-s active duration, smooth settle, 2.40-s observation horizon, fixed-2048 CUDA path, and all hard gates were unchanged.

## 5. Primitive reconstruction audit

Before learning, every teacher and every spline reconstruction was replayed through authoritative physics. The exact teacher replay reproduced {primitive['original_teacher_replay_success']}/{primitive['original_teacher_count']} successes.

| Control points/axis | Latent dim | Replay successes | Retention | Action RMSE | Gate |
|---:|---:|---:|---:|---:|:---:|
"""
    for row in primitive["representations"]:
        report += (
            f"| {row['control_points_per_axis']} | {row['latent_dimension']} | "
            f"{row['authoritative_success_count']}/{row['teacher_contexts']} | "
            f"{100.0 * row['authoritative_success_retention']:.2f}% | "
            f"{row['normalized_action_rmse']:.5f} | "
            f"{'PASS' if row['passes_90_percent_gate'] else 'FAIL'} |\n"
        )
    report += f"""

Selected regression representation: **{primitive['selected_representation']}** ({primitive['selected_latent_dimension']} dimensions). The smallest spline was accepted only if it retained at least 90% of the original authoritative successes; otherwise the full 49-D production action was retained.

## 6. State-disjoint split

- Training states: {len(split['training_state_ids'])}
- Development states: {len(split['development_state_ids'])}
- Training contexts: {len(split['training_context_ids'])}
- Development contexts: {len(split['development_context_ids'])}
- State overlap: none
- Canonical state: training only
- Final 7A TEST: not evaluated

## 7. Deterministic residual policy

The policy is an 83 -> 256 SiLU -> 256 SiLU -> {primitive['selected_latent_dimension']} MLP with {training['parameter_count']:,} parameters. Its output layer starts at zero, so initialization exactly reproduces the rotated CEM-family center. It was trained with Huber regression on the teacher residual, AdamW at 3e-4, and state-disjoint early stopping. Best update: {training['best_update']}; best development Huber: {training['best_development_huber']:.6f}.

## 8. One-action authoritative comparison

| Method | Train success | Development success | Development feasible |
|---|---:|---:|---:|
| Target-conditioned CEM warm start | {_percent(warm['TRAIN']['scientific_success_rate'])} | {_percent(warm['DEVELOPMENT']['scientific_success_rate'])} | {_percent(warm['DEVELOPMENT']['feasible_rate'])} |
| Nearest-neighbor retrieval | {_percent(nearest['TRAIN']['scientific_success_rate'])} | {_percent(nearest['DEVELOPMENT']['scientific_success_rate'])} | {_percent(nearest['DEVELOPMENT']['feasible_rate'])} |
| Deterministic residual MLP | {_percent(policy['TRAIN']['scientific_success_rate'])} | {_percent(policy['DEVELOPMENT']['scientific_success_rate'])} | {_percent(policy['DEVELOPMENT']['feasible_rate'])} |
| Train-only memorization control | {_percent(memorized['TRAIN']['scientific_success_rate'])} | diagnostic only | diagnostic only |

These are not action-space claims: every number is one decoded action replayed in the authoritative production simulator.

## 9. Physical development metrics for the residual MLP

- Median minimum tip distance: {1000.0 * policy['DEVELOPMENT']['median_minimum_tip_distance_m']:.2f} mm
- Median directed speed at best event: {policy['DEVELOPMENT']['median_best_event_directed_speed_m_s']:.3f} m/s
- Median direction error: {policy['DEVELOPMENT']['median_best_event_direction_error_deg']:.2f} deg
- Median UAV displacement: {policy['DEVELOPMENT']['median_maximum_uav_displacement_m']:.4f} m
- Median UAV speed: {policy['DEVELOPMENT']['median_maximum_uav_speed_m_s']:.4f} m/s

## 10. Conditioning checks

Target-conditioned correct-action success was {_percent(conditioning.get('target', {}).get('correct_action_success_rate'))}; target-swapped action success was {_percent(conditioning.get('target', {}).get('swapped_action_success_rate'))}. State-conditioned correct-action success was {_percent(conditioning.get('state', {}).get('correct_action_success_rate'))}; state-swapped action success was {_percent(conditioning.get('state', {}).get('swapped_action_success_rate'))}. These controlled swaps keep the original physical context and change only which predicted action is executed.

## 11. Interpretation

{evaluation['interpretation']}

## 12. Active versus historical code

The production path for this experiment is limited to `PolicyContext`, the fixed normalizer, the deterministic residual policy, the authoritative 49-D codec, smooth command continuation, and the production simulator. Historical SAC, diffusion, scorer, and experimental structured-FiLM modules were not imported by the deployment policy and were not deleted; they remain only for reproduction of earlier negative results.

## 13. Restrictions

- New CEM solves: **0**
- Diffusion/scorer/SAC: **NOT USED**
- Production model modified: **NO**
- Final TEST: **NOT EVALUATED**
- Protected `fig8vertical_002`: **NOT EVALUATED**
- Hardware: **NOT EXECUTED**

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Method:
        DETERMINISTIC CEM-FAMILY RESIDUAL POLICY

    New CEM solves:
        0

    Teacher contexts:
        {inventory['authoritative_successes']}

    Teacher labels/context:
        1

    Selected representation:
        {primitive['selected_representation']}

    Latent dimension:
        {primitive['selected_latent_dimension']}

    Training contexts:
        {len(split['training_context_ids'])}

    Development contexts:
        {len(split['development_context_ids'])}

    Warm-start development success:
        {_percent(warm['DEVELOPMENT']['scientific_success_rate'])}

    Nearest-neighbor development success:
        {_percent(nearest['DEVELOPMENT']['scientific_success_rate'])}

    Residual-MLP training success:
        {_percent(policy['TRAIN']['scientific_success_rate'])}

    Residual-MLP development success:
        {_percent(policy['DEVELOPMENT']['scientific_success_rate'])}

    Residual-MLP development feasibility:
        {_percent(policy['DEVELOPMENT']['feasible_rate'])}

    Train-only memorization success:
        {_percent(memorized['TRAIN']['scientific_success_rate'])}

    Result:
        {evaluation['classification']}

    More CEM data justified:
        {evaluation['more_cem_data_justified']}

    Online CEM:
        NO

    Policy query count:
        1

    Execution:
        OPEN LOOP

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    REPORT.write_text(report, encoding="utf-8")
    shutil.copy2(REPORT, artifact / REPORT.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--report-only",
        type=Path,
        help="Render the report from one completed artifact without rerunning physics.",
    )
    args = parser.parse_args()
    if args.report_only is not None:
        artifact = args.report_only if args.report_only.is_absolute() else ROOT / args.report_only
        inventory = json.loads((artifact / "existing_teacher_inventory.json").read_text(encoding="utf-8"))
        primitive = json.loads((artifact / "primitive_reconstruction_audit.json").read_text(encoding="utf-8"))
        split = json.loads((artifact / "state_disjoint_split_manifest.json").read_text(encoding="utf-8"))
        training = json.loads((artifact / "training_history.json").read_text(encoding="utf-8"))
        evaluation = json.loads((artifact / "one_action_evaluation.json").read_text(encoding="utf-8"))
        conditioning = json.loads((artifact / "conditioning_audit.json").read_text(encoding="utf-8"))
        _write_report(artifact, inventory, primitive, split, training, evaluation, conditioning)
        print(json.dumps({"artifact": str(artifact), "report": str(REPORT)}, indent=2))
        return
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    artifact = args.artifact or ARTIFACT_PARENT / _timestamp()
    artifact = artifact if artifact.is_absolute() else ROOT / artifact
    artifact.mkdir(parents=True, exist_ok=False)
    _write_json(artifact / "config.json", config)

    records, inventory = _load_records(config)
    _write_json(artifact / "existing_teacher_inventory.json", inventory)
    environment = _load_fixed_environment(config)
    task = environment[1]
    physics = _physics_settings(config)
    contexts = _context_table(environment, records)
    np.savez_compressed(
        artifact / "teacher_dataset.npz",
        schema=np.asarray("structured_cem_teacher_dataset_v1"),
        context_ids=np.asarray([record.context_id for record in records]),
        state_ids=np.asarray([record.state_id for record in records]),
        contexts=contexts,
        normalized_actions=np.stack([record.action for record in records]),
    )

    codec, primitive = _primitive_audit(environment, config, records)
    _write_json(artifact / "primitive_reconstruction_audit.json", primitive)

    split = _state_split(
        records, float(config["development_state_fraction"]), int(config["seed"])
    )
    _write_json(artifact / "state_disjoint_split_manifest.json", split)
    split_by_id = {
        context_id: "TRAIN" for context_id in split["training_context_ids"]
    } | {
        context_id: "DEVELOPMENT" for context_id in split["development_context_ids"]
    }
    train_indices = [
        index for index, record in enumerate(records) if split_by_id[record.context_id] == "TRAIN"
    ]
    dev_indices = [
        index
        for index, record in enumerate(records)
        if split_by_id[record.context_id] == "DEVELOPMENT"
    ]

    source7 = ROOT / config["partial_7a_artifact"]
    normalizer_path = source7 / "context_normalizer.json"
    shutil.copy2(normalizer_path, artifact / "context_normalizer.json")
    normalizer = FixedContextNormalizer.load(normalizer_path)
    raw_context = torch.from_numpy(contexts).float()
    normalized_context = normalizer.normalize(raw_context)
    teacher_action = torch.from_numpy(np.stack([record.action for record in records])).float()
    canonical_action, canonical_source = _load_canonical_seed_action(config)
    center_action = torch.stack(
        [
            _rotate_normalized_action(canonical_action, record.direction_local, task, physics)
            for record in records
        ]
    ).float()
    teacher_latent = codec.encode(teacher_action)
    center_latent = codec.encode(center_action)
    _write_json(
        artifact / "center_definition.json",
        {
            "schema": "target_conditioned_cem_family_center_v1",
            "canonical_source_seed": canonical_source["seed"],
            "canonical_source_success": canonical_source["success"],
            "transformation": "rotate canonical local acceleration knots about +Z into target direction",
            "duration_changed_by_context": False,
            "online_optimizer": False,
        },
    )

    model, training = _train_policy(
        config,
        normalized_context,
        center_latent,
        teacher_latent,
        torch.tensor(train_indices, dtype=torch.int64),
        torch.tensor(dev_indices, dtype=torch.int64),
        artifact,
    )
    _write_json(artifact / "training_history.json", training)
    memorization_model, memorization_history = _train_memorization_control(
        config,
        normalized_context,
        center_latent,
        teacher_latent,
        torch.tensor(train_indices, dtype=torch.int64),
        artifact,
    )
    _write_json(artifact / "memorization_training_history.json", memorization_history)
    with torch.no_grad():
        predicted_latent = model(
            normalized_context.cuda(), center_latent.cuda()
        ).cpu()
        memorized_latent = memorization_model(
            normalized_context.cuda(), center_latent.cuda()
        ).cpu()
    predicted_action = codec.decode(predicted_latent).float()
    memorized_action = codec.decode(memorized_latent).float()
    nearest_action, nearest_sources = _nearest_neighbor_actions(
        normalized_context, teacher_action, train_indices, list(range(len(records)))
    )
    method_names = (
        "TARGET_CONDITIONED_CEM_WARM_START",
        "NEAREST_NEIGHBOR_RETRIEVAL",
        "DETERMINISTIC_RESIDUAL_MLP",
        "TRAIN_MEMORIZATION_CONTROL",
    )
    method_actions = torch.stack(
        (center_action, nearest_action, predicted_action, memorized_action), dim=1
    )
    np.savez_compressed(
        artifact / "predicted_actions.npz",
        schema=np.asarray("structured_cem_one_action_predictions_v1"),
        context_ids=np.asarray([record.context_id for record in records]),
        method_names=np.asarray(method_names),
        normalized_actions=method_actions.numpy(),
        nearest_neighbor_source_indices=np.asarray(nearest_sources),
    )
    rows = _evaluate_actions(
        environment,
        config,
        records,
        method_actions.numpy(),
        label="one_action_baselines",
    )
    methods = {
        name: _summarize_method(records, rows, index, split_by_id)
        for index, name in enumerate(method_names)
    }
    policy_train = methods["DETERMINISTIC_RESIDUAL_MLP"]["TRAIN"]
    policy_dev = methods["DETERMINISTIC_RESIDUAL_MLP"]["DEVELOPMENT"]
    memorized_train = methods["TRAIN_MEMORIZATION_CONTROL"]["TRAIN"]
    warm_dev = methods["TARGET_CONDITIONED_CEM_WARM_START"]["DEVELOPMENT"]
    if memorized_train["scientific_success_rate"] < 0.50:
        classification = "DETERMINISTIC_POLICY_NOT_FITTING_TEACHERS"
        more_cem = "NO"
        interpretation = (
            "Even the bounded train-only memorization control did not reproduce half of its training maneuvers in physics. "
            "Additional CEM labels are therefore not justified; the regression representation/objective must be reviewed."
        )
    elif policy_dev["scientific_success_rate"] >= 0.80:
        classification = "SIMPLE_AMORTIZATION_PROMISING"
        more_cem = "NO — VALIDATE THIS SIMPLE POLICY FIRST"
        interpretation = (
            "A single deterministic residual action generalized strongly on state-disjoint development contexts. "
            "The simple amortization hypothesis is supported; the next step should be a frozen, larger held-out evaluation, not a more complex learner."
        )
    elif memorized_train["scientific_success_rate"] >= 0.80 and policy_dev["scientific_success_rate"] < 0.50:
        classification = "TRAIN_FIT_HIGH_GENERALIZATION_UNRESOLVED"
        more_cem = "NO — COVERAGE SCALING NOT YET DEMONSTRATED"
        interpretation = (
            "The same MLP can memorize the training family, but the development-selected checkpoint loses substantial "
            "success on state-disjoint contexts. This isolates generalization rather than raw capacity, but it does not "
            "yet demonstrate a context-coverage scaling trend; new CEM is therefore not authorized by this result alone."
        )
    elif policy_dev["scientific_success_rate"] > warm_dev["scientific_success_rate"]:
        classification = "SIMPLE_AMORTIZATION_PARTIAL"
        more_cem = "CANNOT JUDGE"
        interpretation = (
            "The learned residual improves over the fixed target-conditioned warm start but is not yet a reliable planner. "
            "Inspect group-wise errors before buying more labels."
        )
    else:
        classification = "SIMPLE_AMORTIZATION_NOT_ESTABLISHED"
        more_cem = "NO"
        interpretation = (
            "The residual MLP did not improve the one-action state-disjoint result over the nonlearned center. "
            "More CEM data is not justified by this experiment."
        )
    evaluation = {
        "schema": "structured_cem_one_action_evaluation_v1",
        "method_order": list(method_names),
        "methods": methods,
        "classification": classification,
        "more_cem_data_justified": more_cem,
        "interpretation": interpretation,
        "new_cem_solves": 0,
        "candidate_count_per_policy_query": 1,
        "authoritative_rows": [
            {
                "context_id": record.context_id,
                "state_id": record.state_id,
                "split": split_by_id[record.context_id],
                "group": record.group,
                "methods": {name: rows[index][method] for method, name in enumerate(method_names)},
            }
            for index, record in enumerate(records)
        ],
    }
    _write_json(artifact / "one_action_evaluation.json", evaluation)
    conditioning = _conditioning_audit(
        environment, config, records, predicted_action.numpy()
    )
    _write_json(artifact / "conditioning_audit.json", conditioning)
    _write_json(artifact / "source_hash_manifest.json", _source_hash_manifest(artifact))
    _write_report(artifact, inventory, primitive, split, training, evaluation, conditioning)
    print(json.dumps({
        "artifact": str(artifact),
        "report": str(REPORT),
        "representation": primitive["selected_representation"],
        "train_success": policy_train["scientific_success_rate"],
        "development_success": policy_dev["scientific_success_rate"],
        "classification": classification,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
