"""Feasibility-first candidate ordering shared by production CEM stages."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from .cem_task import VariableDurationWhipTask
from .metrics import PopulationRolloutResult


def _metric_arrays(result: PopulationRolloutResult) -> dict[str, np.ndarray]:
    names = (
        "task_cost",
        "feasible",
        "feasibility_violation",
        "success",
        "finite",
        "best_event_tip_distance_m",
        "best_event_direction_angle_deg",
        "first_entry_tip_distance_m",
        "first_entry_direction_angle_deg",
        "maximum_uav_displacement_m",
        "maximum_uav_speed_m_s",
    )
    available = {item.name for item in fields(PopulationRolloutResult)}
    if not set(names).issubset(available):
        raise RuntimeError("Production rollout result no longer satisfies selection contract.")
    return {name: getattr(result, name).detach().cpu().numpy() for name in names}


def _reward_ranked_successes(task: VariableDurationWhipTask | None) -> bool:
    objective = None if task is None else task.legacy_run_online_objective
    return bool(
        objective is not None
        and objective.profile == "legacy_run_online_strike_margin_tuned_v4"
    )


def feasibility_elite_order(
    result: PopulationRolloutResult,
    task: VariableDurationWhipTask | None = None,
) -> np.ndarray:
    """Return a global lexicographic success/feasible/infeasible ordering."""

    data = _metric_arrays(result)
    success = data["success"].astype(bool)
    feasible = data["feasible"].astype(bool)
    finite = data["finite"].astype(bool)
    category = np.where(success, 0, np.where(feasible, 1, 2)).astype(np.int64)
    category = np.where(finite, category, 3)

    tip = np.where(
        success, data["first_entry_tip_distance_m"], data["best_event_tip_distance_m"]
    )
    direction = np.where(
        success,
        data["first_entry_direction_angle_deg"],
        data["best_event_direction_angle_deg"],
    )
    task_cost = np.nan_to_num(
        data["task_cost"], nan=1e30, posinf=1e30, neginf=-1e30
    )
    violation = np.nan_to_num(
        data["feasibility_violation"], nan=1e30, posinf=1e30
    )
    slack = np.minimum(
        1.0 - data["maximum_uav_displacement_m"] / 0.5,
        1.0 - data["maximum_uav_speed_m_s"] / 3.0,
    )
    if _reward_ranked_successes(task):
        key1 = np.where(success, task_cost, np.where(feasible, task_cost, violation))
        key2 = np.where(success, tip, np.where(feasible, tip, task_cost))
        key3 = direction
        key4 = np.where(success, -slack, np.zeros_like(slack))
        return np.lexsort((key4, key3, key2, key1, category))
    key1 = np.where(success, tip, np.where(feasible, task_cost, violation))
    key2 = np.where(success, direction, np.where(feasible, tip, task_cost))
    key3 = np.where(success, -slack, direction)
    return np.lexsort((key3, key2, key1, category))
