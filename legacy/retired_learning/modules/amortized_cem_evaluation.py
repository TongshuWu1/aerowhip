"""Offline production-physics evaluation for the simulator-free neural policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .action_diffusion import ConditionalActionDiffusion
from .amortized_cem_data import AmortizedCemContextRecord
from .amortized_cem_policy import AmortizedCemDiffusionPolicy
from .context_sampling import ContextSpecification, build_context_from_specification
from .normalization import FixedContextNormalizer
from .outcome_scorer import ManeuverOutcomeScorer, OutcomeTargetNormalizer
from .policy_action import decode_policy_action
from .state_bank import InitialStateBank
from planning.cem_task import VariableDurationWhipTask
from planning.production_cem import (
    FIXED_NUMERICAL_BATCH_SIZE,
    ProductionCemSettings,
    evaluate_normalized_actions_fixed_batch,
    event_segment,
    record_normalized_actions_fixed_batch,
)


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    rows: list[dict[str, Any]]
    selected_actions: np.ndarray
    candidate_actions: np.ndarray
    summary: dict[str, Any]


def load_frozen_policy(
    artifact: str | Path,
    *,
    round_index: int = 0,
    device: torch.device | str = "cuda",
) -> AmortizedCemDiffusionPolicy:
    root = Path(artifact)
    model_root = root if round_index == 0 else root / "aggregation_rounds" / f"round_{round_index}"
    selected = torch.device(device)
    diffusion_payload = torch.load(
        model_root / "diffusion_ema_best.pt", map_location=selected, weights_only=True
    )
    diffusion = ConditionalActionDiffusion().to(selected)
    diffusion.load_state_dict(diffusion_payload["state_dict"])
    scorer_payload = torch.load(
        model_root / "scorer_best.pt", map_location=selected, weights_only=True
    )
    scorer = ManeuverOutcomeScorer().to(selected)
    scorer.load_state_dict(scorer_payload["state_dict"])
    return AmortizedCemDiffusionPolicy(
        diffusion,
        scorer,
        FixedContextNormalizer.load(root / "context_normalizer.json"),
        OutcomeTargetNormalizer.load(model_root / "scorer_target_normalization.json"),
        torch.from_numpy(np.load(root / "fixed_diffusion_noise_bank.npy")),
    )


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


def _candidate_metrics(result, actions, task, settings) -> list[dict[str, Any]]:
    decoded = decode_policy_action(
        actions.float(), task, duration_max_s=settings.duration_max_s
    )
    rows = []
    for index in range(actions.shape[0]):
        row = result.row(index)
        duration = float(decoded.duration_s[index])
        row["maneuver_duration_s"] = duration
        row["hit_segment"] = event_segment(
            row["first_entry_time_s"], duration, settings.settle_duration_s
        )
        rows.append(row)
    return rows


def summarize_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    selected_success = np.asarray([row["selected_metrics"]["success"] for row in rows], dtype=bool)
    selected_feasible = np.asarray([row["selected_metrics"]["feasible"] for row in rows], dtype=bool)
    first_success = np.asarray([row["first_candidate_success"] for row in rows], dtype=bool)
    oracle_success = np.asarray([row["oracle_success"] for row in rows], dtype=bool)

    def values(name: str) -> np.ndarray:
        return np.asarray([float(row["selected_metrics"][name]) for row in rows], dtype=np.float64)

    segments = {name: 0 for name in ("ACTIVE", "SETTLE", "HOLD", "NONE")}
    for row in rows:
        segment = str(row["selected_metrics"].get("hit_segment", "NONE"))
        segments[segment if segment in segments else "NONE"] += 1
    cem_known = np.asarray([row.get("cem_teacher_solved") for row in rows], dtype=object)
    known_mask = np.asarray([value is not None for value in cem_known], dtype=bool)
    cem_pass = np.asarray([bool(value) if value is not None else False for value in cem_known])
    conditional = None
    if bool(cem_pass.any()):
        conditional = float(selected_success[cem_pass].mean())
    return {
        "context_count": count,
        "first_candidate_success_rate": float(first_success.mean()),
        "scorer_selected_success_rate": float(selected_success.mean()),
        "oracle_best_of_32_success_rate": float(oracle_success.mean()),
        "selected_feasible_rate": float(selected_feasible.mean()),
        "median_tip_distance_m": float(np.median(values("minimum_tip_target_distance_m"))),
        "median_directed_speed_m_s": float(np.median(values("best_event_directed_speed_m_s"))),
        "median_direction_error_deg": float(np.median(values("best_event_direction_angle_deg"))),
        "median_uav_displacement_m": float(np.median(values("maximum_uav_displacement_m"))),
        "median_maneuver_duration_s": float(np.median(values("maneuver_duration_s"))),
        "hit_segment_counts": segments,
        "mean_clamp_fraction": float(np.mean([row["clamp_fraction"] for row in rows])),
        "cem_reference_known_count": int(known_mask.sum()),
        "cem_solved_count": int(cem_pass.sum()),
        "policy_success_given_cem_success": conditional,
    }


def evaluate_policy_contexts(
    policy: AmortizedCemDiffusionPolicy,
    simulator,
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
    records: list[AmortizedCemContextRecord],
    banks: dict[str, InitialStateBank],
    *,
    cem_solved_by_context: dict[str, bool] | None = None,
) -> PolicyEvaluation:
    rows, selected_actions, candidate_actions = [], [], []
    cem_reference = cem_solved_by_context or {}
    for number, record in enumerate(records):
        repeated = _repeated_context(simulator, banks[record.bank], record)
        raw_context = repeated.to_tensor()[0:1]
        started = time.perf_counter()
        inference = policy.infer_from_context_tensor(raw_context)
        inference_time = time.perf_counter() - started
        candidates = inference.candidate_normalized_actions
        physical = evaluate_normalized_actions_fixed_batch(
            simulator, repeated, candidates, task, settings
        )
        metrics = _candidate_metrics(physical, candidates, task, settings)
        selected = metrics[inference.selected_index]
        successful = [index for index, value in enumerate(metrics) if bool(value["success"])]
        oracle_index = (
            min(successful, key=lambda index: float(metrics[index]["task_cost"]))
            if successful
            else min(range(32), key=lambda index: float(metrics[index]["task_cost"]))
        )
        rows.append(
            {
                "context_index": record.context_index,
                "context_id": record.context_id,
                "split": record.split,
                "state_id": record.state_id,
                "target_local_m": list(record.target_local_m),
                "selected_index": inference.selected_index,
                "clamp_fraction": inference.clamp_fraction,
                "predicted_gate_pass_count": int(inference.predicted_gate_pass.sum()),
                "predicted_selected_outcomes": inference.predicted_continuous_outcomes[
                    inference.selected_index
                ].cpu().tolist(),
                "predicted_selected_binary": inference.predicted_binary_probabilities[
                    inference.selected_index
                ].cpu().tolist(),
                "policy_query_time_s": inference_time,
                "first_candidate_success": bool(metrics[0]["success"]),
                "selected_metrics": selected,
                "oracle_success": bool(successful),
                "oracle_index": oracle_index,
                "oracle_metrics": metrics[oracle_index],
                "candidate_success_count": len(successful),
                "cem_teacher_solved": cem_reference.get(record.context_id),
            }
        )
        selected_actions.append(candidates[inference.selected_index].cpu().numpy())
        candidate_actions.append(candidates.cpu().numpy())
        print(
            f"POLICY EVAL {number + 1:03d}/{len(records)} {record.split} "
            f"selected={bool(selected['success'])} oracle={bool(successful)}",
            flush=True,
        )
    summary = summarize_policy_rows(rows)
    return PolicyEvaluation(
        rows,
        np.stack(selected_actions).astype(np.float32),
        np.stack(candidate_actions).astype(np.float32),
        summary,
    )


def conditioning_action_distance(
    records: list[AmortizedCemContextRecord], selected_actions: np.ndarray, *, key: str
) -> dict[str, float]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        group = record.state_id if key == "state" else f"target_{record.target_number:03d}"
        groups.setdefault(group, []).append(index)
    distances = []
    for indices in groups.values():
        values = selected_actions[np.asarray(indices)]
        pair = np.sqrt(np.sum((values[:, None] - values[None]) ** 2, axis=-1))
        distances.extend(pair[np.triu_indices(values.shape[0], k=1)].tolist())
    return {
        "pair_count": len(distances),
        "mean_action_l2": float(np.mean(distances)),
        "median_action_l2": float(np.median(distances)),
        "minimum_action_l2": float(np.min(distances)),
        "maximum_action_l2": float(np.max(distances)),
    }


def save_authoritative_policy_replay(
    directory: str | Path,
    simulator,
    repeated_context,
    normalized_action: torch.Tensor,
    task: VariableDurationWhipTask,
    settings: ProductionCemSettings,
    *,
    label: str,
) -> dict[str, Any]:
    """Record one full fixed-2048 production replay and a video-compatible view."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    action = torch.as_tensor(normalized_action, dtype=torch.float32).reshape(1, 49)
    trajectory = record_normalized_actions_fixed_batch(
        simulator, repeated_context, action, task, settings, record_count=1
    )
    target = repeated_context.target_position_world_m()[0].detach().cpu()
    direction = repeated_context.target_direction_world()[0].detach().cpu()
    tip = trajectory.cable_positions_m[:, 0, 11]
    tip_velocity = trajectory.cable_velocities_m_s[:, 0, 11]
    difference = tip - target
    distance = torch.linalg.vector_norm(difference, dim=-1)
    speed = torch.linalg.vector_norm(tip_velocity, dim=-1)
    directed = torch.sum(tip_velocity * direction, dim=-1)
    cosine = directed / torch.clamp(speed, min=1.0e-9)
    angle = torch.rad2deg(torch.acos(torch.clamp(cosine, -1.0, 1.0)))
    non_tip = torch.linalg.vector_norm(
        trajectory.cable_positions_m[:, 0, 2:11] - target.reshape(1, 1, 3), dim=-1
    ).min(dim=-1).values
    uav = trajectory.uav_positions_m[:, 0]
    uav_velocity = trajectory.uav_velocities_m_s[:, 0]
    initial_uav = repeated_context.command_initial_position_world_m[0].detach().cpu()
    duration = float(
        decode_policy_action(action, task, duration_max_s=settings.duration_max_s).duration_s[0]
    )
    metrics = trajectory.metrics.row(0)
    metrics["maneuver_duration_s"] = duration
    metrics["hit_segment"] = event_segment(
        metrics["first_entry_time_s"], duration, settings.settle_duration_s
    )
    metrics.update(
        {
            "task_id": f"amortized_cem_diffusion_{label}",
            "optimizer": "Amortized CEM Diffusion (no online optimizer)",
            "task_classification": "PASS" if metrics["success"] else "FAIL",
            "authorization": "SIMULATION_ONLY",
            "real_flight_authorized": False,
        }
    )
    np.savez_compressed(
        root / "final_replay.npz",
        authorization=np.asarray("SIMULATION_ONLY"),
        real_flight_authorized=np.asarray(False),
        task_id=np.asarray(metrics["task_id"]),
        normalized_action=action.cpu().numpy()[0],
        time_s=trajectory.times_s.numpy(),
        uav_position_m=uav.numpy(),
        uav_velocity_m_s=uav_velocity.numpy(),
        cable_position_m=trajectory.cable_positions_m[:, 0].numpy(),
        cable_velocity_m_s=trajectory.cable_velocities_m_s[:, 0].numpy(),
        c1_c10_position_m=trajectory.cable_positions_m[:, 0, 2:12].numpy(),
        c1_c10_velocity_m_s=trajectory.cable_velocities_m_s[:, 0, 2:12].numpy(),
        p_cmd_m=trajectory.command_positions_m[:, 0].numpy(),
        v_cmd_m_s=trajectory.command_velocities_m_s[:, 0].numpy(),
        a_cmd_m_s2=trajectory.command_accelerations_m_s2[:, 0].numpy(),
        tip_target_distance_m=distance.numpy(),
        tip_speed_m_s=speed.numpy(),
        directed_tip_speed_m_s=directed.numpy(),
        impact_direction_error_deg=angle.numpy(),
        minimum_non_tip_target_distance_m=non_tip.numpy(),
        uav_speed_m_s=torch.linalg.vector_norm(uav_velocity, dim=-1).numpy(),
        uav_displacement_m=torch.linalg.vector_norm(uav - initial_uav, dim=-1).numpy(),
    )
    task_snapshot = {
        "task_id": metrics["task_id"],
        "target": {
            "position_m": target.tolist(),
            "desired_impact_direction": direction.tolist(),
            "success_radius_m": task.success_radius_m,
        },
        "artifact_policy": {
            "authorization": "SIMULATION_ONLY",
            "real_flight_authorized": False,
            "protected_test_evaluation_allowed": False,
        },
    }
    (root / "task_config_snapshot.json").write_text(
        json.dumps(task_snapshot, indent=2) + "\n", encoding="utf-8"
    )
    (root / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "cem_iteration_history.json").write_text("[]\n", encoding="utf-8")
    return metrics
