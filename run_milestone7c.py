"""Audit the physical success basin around existing Milestone-6A CEM teachers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from learning.action_robustness import build_action_perturbation_bank
from learning.cem_teacher_support import (
    TeacherRecord,
    load_fixed_production_environment,
    load_production_cem_teachers,
    production_cem_settings,
)
from learning.context_sampling import (
    ContextSpecification,
    build_context_from_specification,
    pad_context_specification,
)
from learning.policy_action import decode_policy_action
from planning.production_cem import (
    FIXED_NUMERICAL_BATCH_SIZE,
    evaluate_normalized_actions_fixed_batch,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "cem_teacher_robustness_audit_v1.json"
ARTIFACT_PARENT = ROOT / "data" / "policy_training" / "cem_teacher_robustness_audit_v1"
REPORT = ROOT / "MILESTONE7C_CEM_TEACHER_ROBUSTNESS_REPORT.md"

FLOAT_FIELDS = (
    "task_cost",
    "minimum_tip_target_distance_m",
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
    if config.get("schema") != "cem_teacher_robustness_audit_v1":
        raise ValueError("Unsupported Milestone-7C configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The frozen production model changed.")
    for key, value in config["prohibitions"].items():
        if key == "protected_test":
            if value != "fig8vertical_002":
                raise ValueError("Protected-test identity changed.")
        elif not bool(value):
            raise ValueError(f"Required prohibition is disabled: {key}")


def _build_banks(
    config: dict[str, Any], records: list[TeacherRecord]
) -> tuple[np.ndarray, list[dict[str, object]], dict[str, Any]]:
    banks: list[np.ndarray] = []
    first_labels: list[dict[str, object]] | None = None
    for record in records:
        bank, labels = build_action_perturbation_bank(
            record.action,
            context_id=record.context_id,
            acceleration_sigmas=config["acceleration_sigmas"],
            duration_offsets_ms=config["duration_offsets_ms"],
            samples_per_level=int(config["samples_per_level"]),
            base_seed=int(config["seed"]),
        )
        if first_labels is None:
            first_labels = labels
        elif [(row["family"], row["level"]) for row in labels] != [
            (row["family"], row["level"]) for row in first_labels
        ]:
            raise RuntimeError("Perturbation-bank condition ordering changed across contexts.")
        banks.append(bank)
    actions = np.stack(banks)
    assert first_labels is not None
    teacher = np.stack([record.action for record in records])[:, None]
    difference = actions - teacher
    manifest_rows: list[dict[str, Any]] = []
    keys: list[tuple[str, float]] = []
    for label in first_labels:
        key = (str(label["family"]), float(label["level"]))
        if key not in keys:
            keys.append(key)
    for family, level in keys:
        indices = [
            index
            for index, label in enumerate(first_labels)
            if label["family"] == family and float(label["level"]) == level
        ]
        effective = difference[:, indices]
        manifest_rows.append(
            {
                "family": family,
                "level": level,
                "start": min(indices),
                "stop": max(indices) + 1,
                "candidate_count_per_context": len(indices),
                "mean_effective_acceleration_coordinate_rms": float(
                    np.sqrt(np.mean(effective[..., :48] ** 2, axis=-1)).mean()
                ),
                "median_effective_action_l2": float(
                    np.median(np.linalg.norm(effective, axis=-1))
                ),
                "median_effective_duration_offset_ms": float(
                    np.median(np.abs(effective[..., 48])) * 0.5 * 1.35 * 1000.0
                ),
            }
        )
    manifest = {
        "schema": "cem_teacher_deterministic_perturbation_manifest_v1",
        "context_count": len(records),
        "candidate_count_per_context": int(actions.shape[1]),
        "total_physics_rollouts": int(actions.shape[0] * actions.shape[1]),
        "samples_per_nonzero_level": int(config["samples_per_level"]),
        "condition_order": manifest_rows,
        "families": ["EXACT", "IID_ACCELERATION", "SMOOTH_ACCELERATION", "DURATION_ONLY"],
        "action_projection": "production-equivalent coordinate clamp then per-knot radial projection",
        "duration_range_s": 1.35,
        "new_cem_solves": 0,
    }
    return actions, first_labels, manifest


def _empty_outcomes(context_count: int, candidate_count: int) -> dict[str, np.ndarray]:
    shape = (context_count, candidate_count)
    output = {name: np.empty(shape, dtype=np.float32) for name in FLOAT_FIELDS}
    output.update({name: np.empty(shape, dtype=bool) for name in BOOL_FIELDS})
    output["first_entry_marker"] = np.empty(shape, dtype=np.int16)
    output["maneuver_duration_s"] = np.empty(shape, dtype=np.float32)
    return output


@torch.no_grad()
def _evaluate(
    environment: tuple,
    config: dict[str, Any],
    records: list[TeacherRecord],
    actions: np.ndarray,
) -> dict[str, np.ndarray]:
    simulator, task, training_bank, _heldout, canonical_bank, _source = environment
    settings = production_cem_settings(config)
    candidate_count = int(actions.shape[1])
    output = _empty_outcomes(len(records), candidate_count)
    index_by_id = {record.context_id: index for index, record in enumerate(records)}
    for bank_name, bank in (("canonical", canonical_bank), ("training", training_bank)):
        selected = [record for record in records if record.bank == bank_name]
        per_batch = max(1, FIXED_NUMERICAL_BATCH_SIZE // candidate_count)
        for start in range(0, len(selected), per_batch):
            group = selected[start : start + per_batch]
            indices = [index_by_id[record.context_id] for record in group]
            flat_actions = torch.from_numpy(actions[indices].reshape(-1, 49))
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
                "teacher_robustness_audit",
            )
            context = build_context_from_specification(
                simulator,
                bank,
                pad_context_specification(specification, FIXED_NUMERICAL_BATCH_SIZE),
            )
            result = evaluate_normalized_actions_fixed_batch(
                simulator, context, flat_actions, task, settings
            )
            for field in FLOAT_FIELDS + BOOL_FIELDS:
                value = getattr(result, field).detach().cpu().reshape(len(group), candidate_count).numpy()
                output[field][indices] = value
            output["first_entry_marker"][indices] = (
                result.first_entry_marker.detach().cpu().reshape(len(group), candidate_count).numpy()
            )
            output["maneuver_duration_s"][indices] = (
                decode_policy_action(flat_actions, task, duration_max_s=settings.duration_max_s)
                .duration_s.detach().cpu().reshape(len(group), candidate_count).numpy()
            )
            print(
                f"robustness physics: {min(start + len(group), len(selected))}/{len(selected)} {bank_name}",
                flush=True,
            )
    return output


def _condition_indices(labels: list[dict[str, object]]) -> dict[tuple[str, float], np.ndarray]:
    values: dict[tuple[str, float], list[int]] = {}
    for index, row in enumerate(labels):
        values.setdefault((str(row["family"]), float(row["level"])), []).append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in values.items()}


def _gate_arrays(outcomes: dict[str, np.ndarray], task) -> dict[str, np.ndarray]:
    entry_time = outcomes["first_entry_time_s"]
    return {
        "tip_first": outcomes["first_entry_marker"] == 10,
        "impact_window": np.isfinite(entry_time)
        & (entry_time >= float(task.impact_window_s[0]) - 1e-6)
        & (entry_time <= float(task.impact_window_s[1]) + 1e-6),
        "tip_distance": outcomes["first_entry_tip_distance_m"] <= float(task.success_radius_m),
        "directed_speed": outcomes["first_entry_directed_speed_m_s"]
        >= float(task.minimum_directed_speed_m_s),
        "direction": outcomes["first_entry_direction_angle_deg"]
        <= float(task.maximum_direction_error_deg),
        "uav_displacement": outcomes["maximum_uav_displacement_m"]
        <= float(task.maximum_uav_displacement_m),
        "uav_speed": outcomes["maximum_uav_speed_m_s"] <= float(task.maximum_uav_speed_m_s),
        "command_acceleration": outcomes["maximum_command_acceleration_m_s2"]
        <= float(task.maximum_command_acceleration_m_s2) + 1e-5,
        "finite": outcomes["finite"],
    }


def _curve_summary(
    records: list[TeacherRecord],
    labels: list[dict[str, object]],
    outcomes: dict[str, np.ndarray],
    task,
) -> dict[str, Any]:
    conditions = _condition_indices(labels)
    gates = _gate_arrays(outcomes, task)
    rows: list[dict[str, Any]] = []
    groups = sorted({record.group for record in records})
    for (family, level), indices in conditions.items():
        success = outcomes["success"][:, indices]
        feasible = outcomes["feasible"][:, indices]
        row = {
            "family": family,
            "level": level,
            "candidate_count": int(success.size),
            "mean_teacher_survival_probability": float(success.mean()),
            "median_context_survival_probability": float(np.median(success.mean(axis=1))),
            "contexts_with_any_success_rate": float(success.any(axis=1).mean()),
            "mean_feasibility_probability": float(feasible.mean()),
            "gate_pass_probabilities": {
                name: float(value[:, indices].mean()) for name, value in gates.items()
            },
            "group_survival": {
                group: float(
                    success[
                        np.asarray([record.group == group for record in records], dtype=bool)
                    ].mean()
                )
                for group in groups
            },
        }
        rows.append(row)
    return {
        "schema": "cem_teacher_robustness_curve_v1",
        "context_count": len(records),
        "curves": rows,
    }


def _largest_passing_level(
    probabilities: list[tuple[float, float]], threshold: float
) -> tuple[float, float | None]:
    passing = [level for level, probability in probabilities if probability >= threshold]
    largest = max(passing) if passing else 0.0
    failing = [level for level, probability in probabilities if level > largest and probability < threshold]
    return largest, min(failing) if failing else None


def _radii_summary(
    records: list[TeacherRecord],
    labels: list[dict[str, object]],
    outcomes: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conditions = _condition_indices(labels)
    families = ("IID_ACCELERATION", "SMOOTH_ACCELERATION", "DURATION_ONLY")
    context_rows: list[dict[str, Any]] = []
    for context_index, record in enumerate(records):
        values: dict[str, Any] = {}
        for family in families:
            levels = sorted(level for candidate_family, level in conditions if candidate_family == family)
            probabilities = [
                (
                    level,
                    float(outcomes["success"][context_index, conditions[(family, level)]].mean()),
                )
                for level in levels
            ]
            r90, next90 = _largest_passing_level(probabilities, 0.90)
            r50, next50 = _largest_passing_level(probabilities, 0.50)
            values[family] = {
                "survival_by_level": {str(level): probability for level, probability in probabilities},
                "largest_tested_level_at_90pct_survival": r90,
                "first_tested_level_below_90pct": next90,
                "largest_tested_level_at_50pct_survival": r50,
                "first_tested_level_below_50pct": next50,
            }
        context_rows.append(
            {
                "context_id": record.context_id,
                "state_id": record.state_id,
                "group": record.group,
                "families": values,
            }
        )
    aggregate: dict[str, Any] = {}
    for family in families:
        for threshold_name in (
            "largest_tested_level_at_90pct_survival",
            "largest_tested_level_at_50pct_survival",
        ):
            values = np.asarray(
                [row["families"][family][threshold_name] for row in context_rows], dtype=np.float64
            )
            aggregate.setdefault(family, {})[threshold_name] = {
                "median": float(np.median(values)),
                "p10": float(np.quantile(values, 0.10)),
                "p90": float(np.quantile(values, 0.90)),
                "zero_fraction": float((values == 0.0).mean()),
            }
    return context_rows, {
        "schema": "cem_teacher_robustness_radius_summary_v1",
        "discrete_tested_radius_definition": True,
        "context_count": len(context_rows),
        "aggregate": aggregate,
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty_like(values, dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    x = _rank(left) if rank else np.asarray(left, dtype=np.float64)
    y = _rank(right) if rank else np.asarray(right, dtype=np.float64)
    if len(x) < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _mlp_relationship(
    config: dict[str, Any],
    records: list[TeacherRecord],
    context_radii: list[dict[str, Any]],
) -> dict[str, Any]:
    source = ROOT / config["milestone_7b_artifact"]
    with np.load(source / "predicted_actions.npz", allow_pickle=False) as archive:
        context_ids = [str(value) for value in archive["context_ids"].tolist()]
        method_names = [str(value) for value in archive["method_names"].tolist()]
        predictions = np.asarray(archive["normalized_actions"], dtype=np.float32)
    if context_ids != [record.context_id for record in records]:
        raise RuntimeError("Milestone-7B prediction contexts do not match the robustness teachers.")
    teacher = np.stack([record.action for record in records])
    evaluation = json.loads((source / "one_action_evaluation.json").read_text(encoding="utf-8"))
    metric_by_id = {row["context_id"]: row for row in evaluation["authoritative_rows"]}
    split = json.loads((source / "state_disjoint_split_manifest.json").read_text(encoding="utf-8"))
    training_context_ids = set(split["training_context_ids"])
    output: dict[str, Any] = {
        "schema": "cem_teacher_robustness_vs_mlp_error_v1",
        "source": str(source),
        "methods": {},
    }
    for method in ("DETERMINISTIC_RESIDUAL_MLP", "TRAIN_MEMORIZATION_CONTROL"):
        index = method_names.index(method)
        scope_mask = np.asarray(
            [
                True
                if method == "DETERMINISTIC_RESIDUAL_MLP"
                else record.context_id in training_context_ids
                for record in records
            ],
            dtype=bool,
        )
        difference = predictions[scope_mask, index] - teacher[scope_mask]
        acceleration_rmse = np.sqrt(np.mean(difference[:, :48] ** 2, axis=1))
        action_l2 = np.linalg.norm(difference, axis=1)
        duration_ms = np.abs(difference[:, 48]) * 0.5 * 1.35 * 1000.0
        success = np.asarray(
            [
                bool(metric_by_id[record.context_id]["methods"][method]["success"])
                for record, keep in zip(records, scope_mask)
                if keep
            ]
        )
        iid_r50 = np.asarray(
            [
                row["families"]["IID_ACCELERATION"][
                    "largest_tested_level_at_50pct_survival"
                ]
                for row, keep in zip(context_radii, scope_mask)
                if keep
            ],
            dtype=np.float64,
        )
        smooth_r50 = np.asarray(
            [
                row["families"]["SMOOTH_ACCELERATION"][
                    "largest_tested_level_at_50pct_survival"
                ]
                for row, keep in zip(context_radii, scope_mask)
                if keep
            ],
            dtype=np.float64,
        )
        output["methods"][method] = {
            "evaluation_scope": "ALL_EXISTING_CONTEXTS"
            if method == "DETERMINISTIC_RESIDUAL_MLP"
            else "TRAIN_CONTEXTS_ONLY",
            "context_count": int(scope_mask.sum()),
            "scientific_success_rate": float(success.mean()),
            "acceleration_coordinate_rmse": {
                "mean": float(acceleration_rmse.mean()),
                "median": float(np.median(acceleration_rmse)),
                "p90": float(np.quantile(acceleration_rmse, 0.90)),
            },
            "complete_action_l2": {
                "median": float(np.median(action_l2)),
                "p90": float(np.quantile(action_l2, 0.90)),
            },
            "duration_error_ms": {
                "median": float(np.median(duration_ms)),
                "p90": float(np.quantile(duration_ms, 0.90)),
            },
            "fraction_acceleration_rmse_within_iid_discrete_r50": float(
                (acceleration_rmse <= iid_r50).mean()
            ),
            "fraction_acceleration_rmse_within_smooth_discrete_r50": float(
                (acceleration_rmse <= smooth_r50).mean()
            ),
            "spearman_error_vs_iid_r50": _correlation(acceleration_rmse, iid_r50, rank=True),
            "spearman_error_vs_smooth_r50": _correlation(
                acceleration_rmse, smooth_r50, rank=True
            ),
            "median_error_successful": float(np.median(acceleration_rmse[success]))
            if success.any()
            else None,
            "median_error_failed": float(np.median(acceleration_rmse[~success]))
            if (~success).any()
            else None,
        }
    return output


def _classification(curves: dict[str, Any], radii: dict[str, Any]) -> dict[str, Any]:
    by_key = {
        (row["family"], float(row["level"])): row for row in curves["curves"]
    }
    iid_001 = by_key[("IID_ACCELERATION", 0.01)]["mean_teacher_survival_probability"]
    smooth_001 = by_key[("SMOOTH_ACCELERATION", 0.01)]["mean_teacher_survival_probability"]
    iid_0005 = by_key[("IID_ACCELERATION", 0.005)]["mean_teacher_survival_probability"]
    smooth_0005 = by_key[("SMOOTH_ACCELERATION", 0.005)]["mean_teacher_survival_probability"]
    median_iid_r50 = radii["aggregate"]["IID_ACCELERATION"][
        "largest_tested_level_at_50pct_survival"
    ]["median"]
    median_smooth_r50 = radii["aggregate"]["SMOOTH_ACCELERATION"][
        "largest_tested_level_at_50pct_survival"
    ]["median"]
    if iid_001 < 0.50 and smooth_001 < 0.50 and max(median_iid_r50, median_smooth_r50) < 0.01:
        name = "TEACHER_MANIFOLD_KNIFE_EDGE"
        more_cem = "ROBUST-CEM PILOT ONLY; NO BROAD DATASET CAMPAIGN"
        recommendation = (
            "Nominal CEM actions occupy narrow physical success basins. The next authorized method should be a small "
            "16-32-context robust-CEM pilot that optimizes perturbation survival and hard-gate margins before any policy training."
        )
    elif iid_001 >= 0.80 and smooth_001 >= 0.80:
        name = "REGRESSION_LOSS_MISMATCH"
        more_cem = "NO"
        recommendation = (
            "Teachers tolerate errors at the learned policy scale, so action-coordinate Huber is the wrong objective. "
            "The next experiment should use a physics-aware local objective on existing teachers, not more CEM labels."
        )
    else:
        name = "MIXED_ROBUSTNESS_LIMITATION"
        more_cem = "TARGETED ROBUST PILOT MAY BE JUSTIFIED"
        recommendation = (
            "Robustness varies by perturbation family/context. Restrict any follow-up optimization to the fragile groups "
            "and explicitly optimize survival probability; do not resume broad teacher generation."
        )
    return {
        "schema": "cem_teacher_robustness_classification_v1",
        "classification": name,
        "iid_survival_sigma_0_005": iid_0005,
        "smooth_survival_sigma_0_005": smooth_0005,
        "iid_survival_sigma_0_01": iid_001,
        "smooth_survival_sigma_0_01": smooth_001,
        "median_iid_discrete_r50": median_iid_r50,
        "median_smooth_discrete_r50": median_smooth_r50,
        "more_cem_data_justified": more_cem,
        "recommended_next_step": recommendation,
    }


def _figures(artifact: Path, curves: dict[str, Any], mlp: dict[str, Any]) -> None:
    figures = artifact / "figures"
    figures.mkdir(exist_ok=True)
    rows = curves["curves"]
    plt.figure(figsize=(6.5, 4.0))
    for family, label in (
        ("IID_ACCELERATION", "IID knot noise"),
        ("SMOOTH_ACCELERATION", "Smooth temporal noise"),
    ):
        selected = sorted(
            (row for row in rows if row["family"] == family), key=lambda row: row["level"]
        )
        plt.plot(
            [row["level"] for row in selected],
            [100.0 * row["mean_teacher_survival_probability"] for row in selected],
            marker="o",
            label=label,
        )
    plt.axvline(0.01, color="black", linestyle="--", linewidth=1, label="~memorization error")
    plt.xlabel("Requested normalized acceleration noise σ")
    plt.ylabel("Teacher scientific-success survival (%)")
    plt.ylim(-2, 102)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "success_survival_vs_acceleration_noise.png", dpi=180)
    plt.close()

    duration = sorted(
        (row for row in rows if row["family"] == "DURATION_ONLY"), key=lambda row: row["level"]
    )
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(
        [row["level"] for row in duration],
        [100.0 * row["mean_teacher_survival_probability"] for row in duration],
        marker="o",
    )
    plt.xlabel("Absolute duration perturbation (ms)")
    plt.ylabel("Teacher scientific-success survival (%)")
    plt.ylim(-2, 102)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "success_survival_vs_duration_error.png", dpi=180)
    plt.close()

    methods = mlp["methods"]
    labels = ["Dev-selected MLP", "Train memorization"]
    values = [
        methods["DETERMINISTIC_RESIDUAL_MLP"]["acceleration_coordinate_rmse"]["median"],
        methods["TRAIN_MEMORIZATION_CONTROL"]["acceleration_coordinate_rmse"]["median"],
    ]
    plt.figure(figsize=(5.5, 3.8))
    plt.bar(labels, values)
    plt.ylabel("Median normalized acceleration RMSE")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(figures / "policy_error_scale.png", dpi=180)
    plt.close()


def _source_hashes(artifact: Path) -> dict[str, Any]:
    paths = [
        ROOT / "learning" / "action_robustness.py",
        ROOT / "learning" / "cem_teacher_support.py",
        ROOT / "run_milestone7c.py",
        ROOT / "config" / "learning" / "cem_teacher_robustness_audit_v1.json",
        ROOT / "learning" / "policy_action.py",
        ROOT / "planning" / "production_cem.py",
        artifact / "perturbed_actions.npz",
        artifact / "perturbation_outcomes.npz",
    ]
    return {
        "schema": "cem_teacher_robustness_source_hash_manifest_v1",
        "files": [
            {
                "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _write_report(
    artifact: Path,
    inventory: dict[str, Any],
    manifest: dict[str, Any],
    curves: dict[str, Any],
    radii: dict[str, Any],
    mlp: dict[str, Any],
    classification: dict[str, Any],
) -> None:
    by_key = {(row["family"], row["level"]): row for row in curves["curves"]}
    duration_rows = [row for row in curves["curves"] if row["family"] == "DURATION_ONLY"]
    policy = mlp["methods"]["DETERMINISTIC_RESIDUAL_MLP"]
    memorized = mlp["methods"]["TRAIN_MEMORIZATION_CONTROL"]
    text = f"""# Milestone 7C — CEM Teacher Robustness-Basin Audit

## 1. Why this audit was run

Milestone 7B found that exact CEM actions replayed 252/252 successes, but cubic-spline approximations retained 0/252 and a train-only MLP with roughly 0.01 normalized coordinate error retained only 35.81% physical success. This audit tests whether the CEM teachers themselves occupy narrow success basins.

## 2. Scope and prohibitions

This was a simulator/action-support diagnostic using existing artifacts only. New CEM solves: **0**. Learning: **none**. Diffusion, scorer, SAC, final TEST, protected `fig8vertical_002`, theta randomization, and hardware were not used.

## 3. Frozen production contract

The model remained `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`. Every action used the normalized production 49-D decoder, variable active duration, analytic 0.30-s settle, hold, 2.40-s evaluation horizon, and fixed-2048 CUDA physics. All hard scientific gates were unchanged.

## 4. Teacher set

- Designed Milestone-6A contexts: {inventory['designed_contexts']}
- Exact authoritative successful teachers: {inventory['authoritative_successes']}
- Unique physical state IDs: {inventory['unique_state_ids']}
- One verified action/context: yes
- Exact replay before perturbation: 252/252 PASS

## 5. Perturbation design

Each teacher was evaluated exactly plus 32 deterministic samples at every nonzero level:

- IID normalized acceleration-coordinate noise: 0.001, 0.0025, 0.005, 0.01, 0.02
- Temporally smoothed acceleration noise at the same scales
- Duration-only errors: ±1, ±2, ±5, ±10 ms

All perturbed actions were canonicalized by the production-equivalent coordinate and radial bounds before physics. Total authoritative rollouts: {manifest['total_physics_rollouts']:,}.

## 6. Acceleration-noise survival

| σ | IID survival | Smooth survival | IID feasible | Smooth feasible |
|---:|---:|---:|---:|---:|
"""
    for level in (0.001, 0.0025, 0.005, 0.01, 0.02):
        iid = by_key[("IID_ACCELERATION", level)]
        smooth = by_key[("SMOOTH_ACCELERATION", level)]
        text += (
            f"| {level:.4f} | {_percent(iid['mean_teacher_survival_probability'])} | "
            f"{_percent(smooth['mean_teacher_survival_probability'])} | "
            f"{_percent(iid['mean_feasibility_probability'])} | "
            f"{_percent(smooth['mean_feasibility_probability'])} |\n"
        )
    text += """

## 7. Duration sensitivity

| Absolute duration error | Success survival | Feasibility |
|---:|---:|---:|
"""
    for row in sorted(duration_rows, key=lambda value: value["level"]):
        text += (
            f"| {row['level']:.0f} ms | {_percent(row['mean_teacher_survival_probability'])} | "
            f"{_percent(row['mean_feasibility_probability'])} |\n"
        )
    text += f"""

## 8. Discrete robustness radii

The reported radius is the largest tested perturbation level retaining the stated per-context survival probability; it is not an interpolated continuous certificate.

- Median IID r90: {radii['aggregate']['IID_ACCELERATION']['largest_tested_level_at_90pct_survival']['median']:.4f}
- Median IID r50: {radii['aggregate']['IID_ACCELERATION']['largest_tested_level_at_50pct_survival']['median']:.4f}
- Median smooth r90: {radii['aggregate']['SMOOTH_ACCELERATION']['largest_tested_level_at_90pct_survival']['median']:.4f}
- Median smooth r50: {radii['aggregate']['SMOOTH_ACCELERATION']['largest_tested_level_at_50pct_survival']['median']:.4f}
- Median duration r50: {radii['aggregate']['DURATION_ONLY']['largest_tested_level_at_50pct_survival']['median']:.1f} ms

## 9. Relation to learned policy error

The development-selected residual MLP has median normalized acceleration-coordinate RMSE {policy['acceleration_coordinate_rmse']['median']:.5f} and scientific success {_percent(policy['scientific_success_rate'])}. Only {_percent(policy['fraction_acceleration_rmse_within_iid_discrete_r50'])} of its outputs lie within the corresponding teacher's discrete IID r50.

The train-only memorization control reduces median acceleration RMSE to {memorized['acceleration_coordinate_rmse']['median']:.5f}, yet success is {_percent(memorized['scientific_success_rate'])}; only {_percent(memorized['fraction_acceleration_rmse_within_iid_discrete_r50'])} lie within teacher IID r50. This directly connects small coordinate error to physical failure.

## 10. Hard-gate behavior

At σ=0.01, IID gate pass rates were:

"""
    for name, value in by_key[("IID_ACCELERATION", 0.01)]["gate_pass_probabilities"].items():
        text += f"- {name}: {_percent(value)}\n"
    text += f"""

The joint scientific-success rate is lower than most individual gate rates because all timing, strike, direction, and feasibility requirements must hold simultaneously.

## 11. Group heterogeneity

At IID σ=0.01:

"""
    for group, value in by_key[("IID_ACCELERATION", 0.01)]["group_survival"].items():
        text += f"- {group}: {_percent(value)}\n"
    text += f"""

## 12. Root-cause classification

**{classification['classification']}**

- IID survival at σ=0.005: {_percent(classification['iid_survival_sigma_0_005'])}
- Smooth survival at σ=0.005: {_percent(classification['smooth_survival_sigma_0_005'])}
- IID survival at σ=0.01: {_percent(classification['iid_survival_sigma_0_01'])}
- Smooth survival at σ=0.01: {_percent(classification['smooth_survival_sigma_0_01'])}

{classification['recommended_next_step']}

## 13. What should not happen next

Do not resume the broad 768-context teacher campaign, do not change policy architecture, and do not train another action-coordinate imitation model on the same nominal labels. A robustness-aware teacher pilot must succeed first if the classification supports it.

## 14. Restrictions

- Production model modified: **NO**
- New CEM solves: **0**
- Learning: **NONE**
- Final TEST: **NOT EVALUATED**
- Protected test: **NOT EVALUATED**
- Hardware: **NOT EXECUTED**

## Final summary

    Model:
        MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI

    Existing teachers:
        {inventory['authoritative_successes']}

    New CEM solves:
        0

    Learning:
        NONE

    Authoritative perturbation rollouts:
        {manifest['total_physics_rollouts']}

    Exact teacher replay:
        252 / 252 PASS

    IID survival at sigma=0.005:
        {_percent(classification['iid_survival_sigma_0_005'])}

    IID survival at sigma=0.01:
        {_percent(classification['iid_survival_sigma_0_01'])}

    Smooth survival at sigma=0.005:
        {_percent(classification['smooth_survival_sigma_0_005'])}

    Smooth survival at sigma=0.01:
        {_percent(classification['smooth_survival_sigma_0_01'])}

    Median IID discrete r50:
        {classification['median_iid_discrete_r50']}

    Median smooth discrete r50:
        {classification['median_smooth_discrete_r50']}

    Development-selected MLP median acceleration RMSE:
        {policy['acceleration_coordinate_rmse']['median']:.5f}

    Train-memorization median acceleration RMSE:
        {memorized['acceleration_coordinate_rmse']['median']:.5f}

    Root cause:
        {classification['classification']}

    More CEM data justified:
        {classification['more_cem_data_justified']}

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    REPORT.write_text(text, encoding="utf-8")
    shutil.copy2(REPORT, artifact / REPORT.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--analyze-existing",
        type=Path,
        help="Recompute summaries/report from one saved action/outcome artifact without physics.",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    if args.analyze_existing is not None:
        artifact = (
            args.analyze_existing
            if args.analyze_existing.is_absolute()
            else ROOT / args.analyze_existing
        )
        records, inventory = load_production_cem_teachers(config)
        manifest = json.loads((artifact / "perturbation_manifest.json").read_text(encoding="utf-8"))
        with np.load(artifact / "perturbed_actions.npz", allow_pickle=False) as archive:
            labels = [json.loads(str(value)) for value in archive["label_json"].tolist()]
        with np.load(artifact / "perturbation_outcomes.npz", allow_pickle=False) as archive:
            outcomes = {
                name: np.asarray(archive[name])
                for name in archive.files
                if name != "schema"
            }
        environment = load_fixed_production_environment(config)
        curves = _curve_summary(records, labels, outcomes, environment[1])
        context_radii, radii = _radii_summary(records, labels, outcomes)
        mlp = _mlp_relationship(config, records, context_radii)
        classification = _classification(curves, radii)
        _write_json(artifact / "robustness_curve.json", curves)
        _write_json(artifact / "context_robustness_radii.json", context_radii)
        _write_json(artifact / "robustness_radius_summary.json", radii)
        _write_json(artifact / "mlp_error_relationship.json", mlp)
        _write_json(artifact / "root_cause_classification.json", classification)
        _figures(artifact, curves, mlp)
        _write_json(artifact / "source_hash_manifest.json", _source_hashes(artifact))
        _write_report(artifact, inventory, manifest, curves, radii, mlp, classification)
        print(json.dumps({"artifact": str(artifact), "report": str(REPORT)}, indent=2))
        return
    artifact = args.artifact or ARTIFACT_PARENT / _timestamp()
    artifact = artifact if artifact.is_absolute() else ROOT / artifact
    artifact.mkdir(parents=True, exist_ok=False)
    _write_json(artifact / "config.json", config)

    records, inventory = load_production_cem_teachers(config)
    _write_json(artifact / "teacher_inventory.json", inventory)
    actions, labels, manifest = _build_banks(config, records)
    _write_json(artifact / "perturbation_manifest.json", manifest)
    np.savez_compressed(
        artifact / "perturbed_actions.npz",
        schema=np.asarray("cem_teacher_robustness_actions_v1"),
        context_ids=np.asarray([record.context_id for record in records]),
        normalized_actions=actions,
        label_json=np.asarray([json.dumps(_safe(row), sort_keys=True) for row in labels]),
    )

    environment = load_fixed_production_environment(config)
    outcomes = _evaluate(environment, config, records, actions)
    if not bool(outcomes["success"][:, 0].all()):
        failed = [
            records[index].context_id for index in np.flatnonzero(~outcomes["success"][:, 0])
        ]
        raise RuntimeError(f"Exact teacher replay failed before robustness analysis: {failed}")
    np.savez_compressed(
        artifact / "perturbation_outcomes.npz",
        schema=np.asarray("cem_teacher_robustness_outcomes_v1"),
        **outcomes,
    )

    task = environment[1]
    curves = _curve_summary(records, labels, outcomes, task)
    context_radii, radii = _radii_summary(records, labels, outcomes)
    mlp = _mlp_relationship(config, records, context_radii)
    classification = _classification(curves, radii)
    _write_json(artifact / "robustness_curve.json", curves)
    _write_json(artifact / "context_robustness_radii.json", context_radii)
    _write_json(artifact / "robustness_radius_summary.json", radii)
    _write_json(artifact / "mlp_error_relationship.json", mlp)
    _write_json(artifact / "root_cause_classification.json", classification)
    _figures(artifact, curves, mlp)
    _write_json(artifact / "source_hash_manifest.json", _source_hashes(artifact))
    _write_report(artifact, inventory, manifest, curves, radii, mlp, classification)
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "report": str(REPORT),
                "rollouts": manifest["total_physics_rollouts"],
                "classification": classification["classification"],
                "iid_survival_sigma_0.01": classification["iid_survival_sigma_0_01"],
                "smooth_survival_sigma_0.01": classification["smooth_survival_sigma_0_01"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
