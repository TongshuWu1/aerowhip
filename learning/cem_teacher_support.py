"""Shared loading contract for existing production-CEM teacher actions.

This module contains no learning algorithm.  It isolates the durable
Milestone-6A environment/action contract from retired policy experiments so
current diagnostics never import a historical milestone runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    pad_context_specification,
)
from learning.policy_action import decode_policy_action
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
from simulator.production import (
    active_model_paths,
    build_production_simulator,
    load_active_model_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_fixed_production_environment(config: dict[str, Any]):
    """Recreate the settled Milestone-6A environment and batch ordering."""

    manifest = load_active_model_manifest()
    paths = active_model_paths(manifest)
    simulator_settings = SimulatorSettings.load(paths["configuration"])
    simulator = build_production_simulator(
        simulator_settings, device=torch.device("cuda"), dtype=torch.float32
    )
    # Set this before hover pre-roll: the frozen residual path is numerically
    # shape-sensitive and the canonical state was settled under this contract.
    simulator.uav_model.set_fixed_evaluation_batch_size(FIXED_NUMERICAL_BATCH_SIZE)
    task = load_variable_duration_task(_project_path(config["task_config"]))
    state_root = _project_path(config["state_bank_artifact"])
    training_bank = InitialStateBank.load(
        state_root / "training_state_bank.npz",
        state_root / "training_state_bank_manifest.json",
    )
    state = hover_preroll(simulator, task)
    canonical_bank = initial_state_bank_from_state(
        state,
        command_position_world_m=torch.tensor(
            task.initial_uav_position_m, device=simulator.device
        ),
        command_velocity_world_m_s=torch.tensor(
            task.initial_uav_velocity_m_s, device=simulator.device
        ),
        command_yaw_world_rad=task.initial_yaw_rad,
        seed=42,
    )
    return simulator, task, training_bank, None, canonical_bank, simulator_settings


def production_cem_settings(config: dict[str, Any]) -> ProductionCemSettings:
    """Load the final maneuver/settle/evaluation contract from Milestone 6A."""

    source = json.loads(
        _project_path(config["production_contract_config"]).read_text(encoding="utf-8")
    )
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


def load_production_cem_teachers(
    config: dict[str, Any],
) -> tuple[list[TeacherRecord], dict[str, Any]]:
    """Load verified Milestone-6A actions without re-running CEM or physics."""

    source = _project_path(config["milestone_6a_artifact"])
    manifest = json.loads(
        (source / "benchmark_context_manifest.json").read_text(encoding="utf-8")
    )
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
        maximum_action_difference = max(
            maximum_action_difference, float(np.max(np.abs(saved - inline)))
        )
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

    inventory = {
        "schema": "production_cem_teacher_inventory_v1",
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
        "new_cem_solves": 0,
    }
    return records, inventory


def build_teacher_context_table(
    simulator,
    records: list[TeacherRecord],
    training_bank: InitialStateBank,
    canonical_bank: InitialStateBank,
) -> np.ndarray:
    """Build the exact 83-D production context for each saved teacher."""

    values: list[np.ndarray] = []
    for record in records:
        bank = canonical_bank if record.bank == "canonical" else training_bank
        specification = ContextSpecification(
            torch.tensor([record.state_index], dtype=torch.int64),
            torch.tensor([record.target_local_m], dtype=torch.float32),
            torch.tensor([record.direction_local], dtype=torch.float32),
            "production_cem_teacher",
        )
        context = build_context_from_specification(simulator, bank, specification)
        values.append(context.to_tensor()[0].detach().cpu().numpy().astype(np.float32))
    result = np.stack(values)
    if result.shape != (len(records), 83) or not np.isfinite(result).all():
        raise RuntimeError("Production teacher context table is invalid.")
    return result


def evaluate_teacher_action_candidates(
    simulator,
    task,
    settings: ProductionCemSettings,
    records: list[TeacherRecord],
    actions: np.ndarray,
    training_bank: InitialStateBank,
    canonical_bank: InitialStateBank,
    *,
    label: str,
) -> list[list[dict[str, Any]]]:
    """Authoritatively evaluate ``contexts x candidates x 49`` actions.

    This is the permanent varied-context counterpart of
    :func:`evaluate_normalized_actions_fixed_batch`.  It preserves the exact
    fixed-2048 production contract and never invokes CEM.
    """

    value = np.asarray(actions, dtype=np.float32)
    if value.ndim != 3 or value.shape != (len(records), value.shape[1], 49):
        raise ValueError("Evaluation actions must have shape contexts x candidates x 49.")
    if not np.isfinite(value).all():
        raise ValueError("Evaluation actions must be finite.")
    candidate_count = int(value.shape[1])
    if candidate_count < 1 or candidate_count > FIXED_NUMERICAL_BATCH_SIZE:
        raise ValueError("Candidate count must lie in [1, 2048].")

    rows: list[list[dict[str, Any]] | None] = [None] * len(records)
    record_index = {record.context_id: index for index, record in enumerate(records)}
    for bank_name, bank in (("canonical", canonical_bank), ("training", training_bank)):
        selected = [record for record in records if record.bank == bank_name]
        contexts_per_batch = max(1, FIXED_NUMERICAL_BATCH_SIZE // candidate_count)
        for start in range(0, len(selected), contexts_per_batch):
            group = selected[start : start + contexts_per_batch]
            indices = [record_index[record.context_id] for record in group]
            flat_actions = torch.from_numpy(value[indices].reshape(-1, 49))
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
            repeated_context = build_context_from_specification(
                simulator,
                bank,
                pad_context_specification(specification, FIXED_NUMERICAL_BATCH_SIZE),
            )
            result = evaluate_normalized_actions_fixed_batch(
                simulator,
                repeated_context,
                flat_actions,
                task,
                settings,
            )
            decoded = decode_policy_action(
                flat_actions,
                task,
                duration_max_s=settings.duration_max_s,
            )
            for local_index, record in enumerate(group):
                output: list[dict[str, Any]] = []
                for candidate in range(candidate_count):
                    flat_index = local_index * candidate_count + candidate
                    metric = result.row(flat_index)
                    duration = float(decoded.duration_s[flat_index])
                    metric["maneuver_duration_s"] = duration
                    metric["hit_segment"] = event_segment(
                        metric["first_entry_time_s"],
                        duration,
                        settings.settle_duration_s,
                    )
                    output.append(metric)
                rows[record_index[record.context_id]] = output
    if any(row is None for row in rows):
        raise RuntimeError("An authoritative evaluation context was not populated.")
    return [row for row in rows if row is not None]
