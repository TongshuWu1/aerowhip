"""Existing-data pilot of an IRP-style cable-whip correction loop.

No CEM is run here.  A delta-outcome model is trained from the durable 7C
perturbation archive, then used to choose one compact residual correction after
each observed authoritative rollout.  The production simulator is used only
to represent the next physical execution, never to rank candidate corrections.
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
from learning.iterative_residual import (
    BINARY_OUTCOME_NAMES,
    CONTINUOUS_OUTCOME_NAMES,
    IterativeResidualOutcomeModel,
    OutcomeNormalizer,
    deterministic_compact_candidates,
    fit_compact_residual_basis,
    outcome_arrays_from_archive,
    outcome_arrays_from_rows,
    predicted_candidate_score,
)
from learning.normalization import FixedContextNormalizer


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "learning" / "iterative_residual_policy_v1.json"
ARTIFACT_PARENT = ROOT / "data" / "policy_training" / "iterative_residual_policy_v1"
ROOT_REPORT = ROOT / "ITERATIVE_RESIDUAL_POLICY_PILOT_REPORT.md"


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
    if config.get("schema") != "iterative_residual_policy_v1":
        raise ValueError("Unsupported iterative-residual configuration.")
    if config.get("model_freeze") != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
        raise ValueError("The frozen production model changed.")
    prohibited = config["prohibitions"]
    for key in (
        "new_cem_solves",
        "sac",
        "diffusion",
        "final_test",
        "theta_randomization",
        "hardware",
    ):
        if not bool(prohibited[key]):
            raise ValueError(f"Required prohibition is disabled: {key}")
    if prohibited["protected_test"] != "fig8vertical_002":
        raise ValueError("Protected-test identity changed.")


def _load_existing_data(config: dict[str, Any]) -> dict[str, Any]:
    residual_root = ROOT / config["milestone_7b_artifact"]
    robustness_root = ROOT / config["milestone_7c_artifact"]
    with np.load(residual_root / "teacher_dataset.npz", allow_pickle=False) as archive:
        context_ids = np.asarray(archive["context_ids"]).astype(str)
        state_ids = np.asarray(archive["state_ids"]).astype(str)
        contexts = np.asarray(archive["contexts"], dtype=np.float32)
        teachers = np.asarray(archive["normalized_actions"], dtype=np.float32)
    with np.load(residual_root / "predicted_actions.npz", allow_pickle=False) as archive:
        predicted_ids = np.asarray(archive["context_ids"]).astype(str)
        method_names = np.asarray(archive["method_names"]).astype(str)
        predictions = np.asarray(archive["normalized_actions"], dtype=np.float32)
    with np.load(robustness_root / "perturbed_actions.npz", allow_pickle=False) as archive:
        perturbation_ids = np.asarray(archive["context_ids"]).astype(str)
        perturbation_actions = np.asarray(archive["normalized_actions"], dtype=np.float32)
    with np.load(robustness_root / "perturbation_outcomes.npz", allow_pickle=False) as archive:
        continuous, binary = outcome_arrays_from_archive(archive)
    split = json.loads(
        (residual_root / "state_disjoint_split_manifest.json").read_text(encoding="utf-8")
    )
    normalizer = FixedContextNormalizer.load(residual_root / "context_normalizer.json")

    if not (
        np.array_equal(context_ids, predicted_ids)
        and np.array_equal(context_ids, perturbation_ids)
    ):
        raise RuntimeError("7B/7C context ordering is inconsistent.")
    if contexts.shape != (252, 83) or teachers.shape != (252, 49):
        raise RuntimeError("Saved teacher tensor contract changed.")
    if perturbation_actions.shape != (252, 449, 49):
        raise RuntimeError("Saved perturbation tensor contract changed.")
    if continuous.shape != (252, 449, 6) or binary.shape != (252, 449, 3):
        raise RuntimeError("Saved perturbation outcome contract changed.")
    method_index = np.flatnonzero(method_names == "DETERMINISTIC_RESIDUAL_MLP")
    if method_index.size != 1:
        raise RuntimeError("The 7B residual-policy initialization is missing.")
    initial_actions = predictions[:, int(method_index[0])]
    train_ids = set(str(value) for value in split["training_context_ids"])
    dev_ids = set(str(value) for value in split["development_context_ids"])
    train_indices = np.asarray(
        [index for index, value in enumerate(context_ids) if value in train_ids], dtype=np.int64
    )
    dev_indices = np.asarray(
        [index for index, value in enumerate(context_ids) if value in dev_ids], dtype=np.int64
    )
    if len(train_indices) != 215 or len(dev_indices) != 37:
        raise RuntimeError("The immutable 7B state-disjoint split changed.")
    normalized_contexts = normalizer.normalize(torch.from_numpy(contexts)).numpy()
    return {
        "context_ids": context_ids,
        "state_ids": state_ids,
        "contexts": contexts,
        "normalized_contexts": normalized_contexts.astype(np.float32),
        "teachers": teachers,
        "initial_actions": initial_actions,
        "perturbation_actions": perturbation_actions,
        "continuous": continuous,
        "binary": binary,
        "train_indices": train_indices,
        "dev_indices": dev_indices,
        "context_normalizer": normalizer,
        "split": split,
        "method_names": method_names.tolist(),
    }


def _sample_pairs(
    context_indices: torch.Tensor,
    actions: torch.Tensor,
    continuous: torch.Tensor,
    binary: torch.Tensor,
    normalized_contexts: torch.Tensor,
    outcome_normalizer: OutcomeNormalizer,
    *,
    count: int,
    generator: torch.Generator,
    teacher_anchor_probability: float = 0.25,
    outcome_balanced_target_sampling: bool = False,
) -> tuple[torch.Tensor, ...]:
    device = actions.device
    context_slot = torch.randint(
        int(context_indices.numel()), (count,), generator=generator, device=device
    )
    context = context_indices[context_slot]
    sample_count = int(actions.shape[1])
    base = torch.randint(sample_count, (count,), generator=generator, device=device)
    if outcome_balanced_target_sampling:
        target_weight = 1.0 + 2.0 * binary[:, :, 1] + 8.0 * binary[:, :, 2]
        target = torch.multinomial(
            target_weight[context],
            num_samples=1,
            replacement=True,
            generator=generator,
        ).squeeze(-1)
    else:
        target = torch.randint(sample_count, (count,), generator=generator, device=device)
    # Make improvement-to-the-exact-teacher pairs common while retaining the
    # complete local response distribution in the remaining pairs.
    if teacher_anchor_probability > 0.0:
        teacher_mask = (
            torch.rand((count,), generator=generator, device=device)
            < float(teacher_anchor_probability)
        )
        target = torch.where(teacher_mask, torch.zeros_like(target), target)
    base_action = actions[context, base]
    target_action = actions[context, target]
    base_continuous = outcome_normalizer.normalize(continuous[context, base])
    base_observed = torch.cat((base_continuous, binary[context, base]), dim=-1)
    target_continuous = outcome_normalizer.normalize(continuous[context, target])
    return (
        normalized_contexts[context],
        base_action,
        base_observed,
        target_action - base_action,
        target_continuous,
        binary[context, target],
    )


def _train_model(
    config: dict[str, Any],
    data: dict[str, Any],
    artifact: Path,
) -> tuple[IterativeResidualOutcomeModel, OutcomeNormalizer, dict[str, Any]]:
    settings = config["model"]
    device = torch.device("cuda")
    torch.manual_seed(int(config["seed"]))
    actions = torch.from_numpy(data["perturbation_actions"]).to(device)
    continuous = torch.from_numpy(data["continuous"]).to(device)
    binary = torch.from_numpy(data["binary"]).to(device)
    contexts = torch.from_numpy(data["normalized_contexts"]).to(device)
    train_indices = torch.from_numpy(data["train_indices"]).to(device)
    dev_indices = torch.from_numpy(data["dev_indices"]).to(device)
    outcome_normalizer = OutcomeNormalizer.fit(continuous[train_indices].cpu())
    _write_json(artifact / "outcome_normalizer.json", outcome_normalizer.to_json())

    prevalence = binary[train_indices].reshape(-1, binary.shape[-1]).mean(dim=0)
    positive_weight = torch.clamp((1.0 - prevalence) / torch.clamp(prevalence, min=1.0e-4), max=20.0)
    model = IterativeResidualOutcomeModel(int(settings["hidden_dimension"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    train_generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 1)
    validation_generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 2)
    validation_batch = _sample_pairs(
        dev_indices,
        actions,
        continuous,
        binary,
        contexts,
        outcome_normalizer,
        count=min(65536, int(dev_indices.numel()) * int(actions.shape[1]) * 4),
        generator=validation_generator,
    )
    best_loss = float("inf")
    best_update = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    batch_size = int(settings["batch_size"])
    for update in range(1, int(settings["maximum_updates"]) + 1):
        batch = _sample_pairs(
            train_indices,
            actions,
            continuous,
            binary,
            contexts,
            outcome_normalizer,
            count=batch_size,
            generator=train_generator,
        )
        predicted_continuous, predicted_binary = model(*batch[:4])
        continuous_loss = F.smooth_l1_loss(predicted_continuous, batch[4])
        binary_loss = F.binary_cross_entropy_with_logits(
            predicted_binary,
            batch[5],
            pos_weight=positive_weight,
        )
        loss = (
            float(settings["continuous_loss_weight"]) * continuous_loss
            + float(settings["binary_loss_weight"]) * binary_loss
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
                validation_continuous, validation_binary = model(*validation_batch[:4])
                validation_continuous_loss = F.smooth_l1_loss(
                    validation_continuous, validation_batch[4]
                )
                validation_binary_loss = F.binary_cross_entropy_with_logits(
                    validation_binary,
                    validation_batch[5],
                    pos_weight=positive_weight,
                )
                validation_loss = float(validation_continuous_loss + validation_binary_loss)
                binary_probability = torch.sigmoid(validation_binary)
                binary_accuracy = ((binary_probability >= 0.5) == (validation_batch[5] >= 0.5)).float().mean(dim=0)
                normalized_rmse = torch.sqrt(
                    torch.mean((validation_continuous - validation_batch[4]) ** 2, dim=0)
                )
            model.train()
            row = {
                "update": update,
                "minibatch_loss": float(loss.detach()),
                "minibatch_continuous_loss": float(continuous_loss.detach()),
                "minibatch_binary_loss": float(binary_loss.detach()),
                "validation_loss": validation_loss,
                "validation_continuous_loss": float(validation_continuous_loss),
                "validation_binary_loss": float(validation_binary_loss),
                "validation_continuous_normalized_rmse": normalized_rmse.tolist(),
                "validation_binary_accuracy": binary_accuracy.tolist(),
                "gradient_norm_before_clip": gradient_norm,
            }
            history.append(row)
            print(
                f"update {update}: train={float(loss.detach()):.5f} dev={validation_loss:.5f}",
                flush=True,
            )
            if validation_loss < best_loss - 1.0e-6:
                best_loss = validation_loss
                best_update = update
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            if (
                update >= int(settings["minimum_updates"])
                and update - best_update >= int(settings["patience_updates"])
            ):
                break
    if best_state is None:
        raise RuntimeError("Iterative residual training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = {
        "schema": "iterative_residual_outcome_model_v1",
        "model_state_dict": best_state,
        "hidden_dimension": int(settings["hidden_dimension"]),
        "best_update": best_update,
        "best_validation_loss": best_loss,
        "continuous_outcome_names": CONTINUOUS_OUTCOME_NAMES,
        "binary_outcome_names": BINARY_OUTCOME_NAMES,
        "outcome_normalizer": outcome_normalizer.to_json(),
    }
    torch.save(checkpoint, artifact / "iterative_residual_outcome_model_best.pt")
    summary = {
        "schema": "iterative_residual_training_history_v1",
        "configuration": settings,
        "train_context_count": len(data["train_indices"]),
        "development_context_count": len(data["dev_indices"]),
        "perturbations_per_context": int(data["perturbation_actions"].shape[1]),
        "physical_outcome_rows_used_for_gradients": int(
            len(data["train_indices"]) * data["perturbation_actions"].shape[1]
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "binary_prevalence": dict(zip(BINARY_OUTCOME_NAMES, prevalence.tolist(), strict=True)),
        "positive_class_weight": dict(zip(BINARY_OUTCOME_NAMES, positive_weight.tolist(), strict=True)),
        "updates_completed": history[-1]["update"],
        "best_update": best_update,
        "best_validation_loss": best_loss,
        "runtime_s": time.perf_counter() - started,
        "history": history,
    }
    _write_json(artifact / "training_history.json", summary)
    return model, outcome_normalizer, summary


def _prediction_diagnostic(
    model: IterativeResidualOutcomeModel,
    outcome_normalizer: OutcomeNormalizer,
    data: dict[str, Any],
) -> dict[str, Any]:
    device = next(model.parameters()).device
    actions = torch.from_numpy(data["perturbation_actions"]).to(device)
    continuous = torch.from_numpy(data["continuous"]).to(device)
    binary = torch.from_numpy(data["binary"]).to(device)
    contexts = torch.from_numpy(data["normalized_contexts"]).to(device)
    dev_indices = torch.from_numpy(data["dev_indices"]).to(device)
    generator = torch.Generator(device=device).manual_seed(9042)
    batch = _sample_pairs(
        dev_indices,
        actions,
        continuous,
        binary,
        contexts,
        outcome_normalizer,
        count=65536,
        generator=generator,
    )
    with torch.no_grad():
        predicted_continuous, predicted_binary = model(*batch[:4])
    predicted_physical = outcome_normalizer.denormalize(predicted_continuous)
    target_physical = outcome_normalizer.denormalize(batch[4])
    mae = torch.mean(torch.abs(predicted_physical - target_physical), dim=0)
    rmse = torch.sqrt(torch.mean((predicted_physical - target_physical) ** 2, dim=0))
    probability = torch.sigmoid(predicted_binary)
    accuracy = ((probability >= 0.5) == (batch[5] >= 0.5)).float().mean(dim=0)
    return {
        "schema": "iterative_residual_heldout_pair_prediction_v1",
        "pair_count": int(batch[0].shape[0]),
        "continuous_mae": dict(zip(CONTINUOUS_OUTCOME_NAMES, mae.tolist(), strict=True)),
        "continuous_rmse": dict(zip(CONTINUOUS_OUTCOME_NAMES, rmse.tolist(), strict=True)),
        "binary_accuracy": dict(zip(BINARY_OUTCOME_NAMES, accuracy.tolist(), strict=True)),
    }


def _summarize_rows(
    records,
    rows: list[list[dict[str, Any]]],
    context_ids_by_split: dict[str, set[str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split, identifiers in context_ids_by_split.items():
        metrics = [rows[index][0] for index, record in enumerate(records) if record.context_id in identifiers]
        success = np.asarray([bool(row["success"]) for row in metrics], dtype=bool)
        feasible = np.asarray([bool(row["feasible"]) for row in metrics], dtype=bool)

        def median(name: str) -> float | None:
            values = np.asarray([float(row[name]) for row in metrics], dtype=np.float64)
            values = values[np.isfinite(values)]
            return float(np.median(values)) if values.size else None

        output[split] = {
            "context_count": len(metrics),
            "scientific_success_count": int(success.sum()),
            "scientific_success_rate": float(success.mean()),
            "feasible_count": int(feasible.sum()),
            "feasible_rate": float(feasible.mean()),
            "median_tip_distance_m": median("best_event_tip_distance_m"),
            "median_directed_speed_m_s": median("best_event_directed_speed_m_s"),
            "median_direction_error_deg": median("best_event_direction_angle_deg"),
            "median_uav_displacement_m": median("maximum_uav_displacement_m"),
            "median_uav_speed_m_s": median("maximum_uav_speed_m_s"),
        }
    return output


def _select_corrections(
    model: IterativeResidualOutcomeModel,
    outcome_normalizer: OutcomeNormalizer,
    normalized_contexts: np.ndarray,
    current_actions: np.ndarray,
    current_continuous: np.ndarray,
    current_binary: np.ndarray,
    basis: torch.Tensor,
    config: dict[str, Any],
    *,
    iteration: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    search = config["residual_search"]
    device = next(model.parameters()).device
    output = current_actions.copy()
    diagnostics: list[dict[str, Any]] = []
    for index in range(len(current_actions)):
        if bool(current_binary[index, 2]):
            diagnostics.append({"context_index": index, "already_successful": True, "selected_candidate": 0})
            continue
        candidates, deltas = deterministic_compact_candidates(
            torch.from_numpy(current_actions[index]),
            basis,
            candidate_count=int(search["candidate_count"]),
            coordinate_rms_scales=tuple(float(value) for value in search["coordinate_rms_scales"]),
            seed=int(config["seed"]) + 100_000 * iteration + index,
        )
        count = int(candidates.shape[0])
        context = torch.from_numpy(normalized_contexts[index]).to(device).expand(count, -1)
        action = torch.from_numpy(current_actions[index]).to(device).expand(count, -1)
        normalized_outcome = outcome_normalizer.normalize(
            torch.from_numpy(current_continuous[index]).to(device)
        )
        observed = torch.cat(
            (normalized_outcome, torch.from_numpy(current_binary[index]).to(device)), dim=-1
        ).expand(count, -1)
        with torch.no_grad():
            predicted_continuous, predicted_binary = model(
                context,
                action,
                observed,
                deltas.to(device),
            )
            score, values = predicted_candidate_score(
                predicted_continuous,
                predicted_binary,
                outcome_normalizer,
            )
        selected = int(torch.argmax(score))
        output[index] = candidates[selected].numpy()
        diagnostics.append(
            {
                "context_index": index,
                "already_successful": False,
                "selected_candidate": selected,
                "requested_candidate_count": count,
                "selected_actual_delta_rms": float(torch.sqrt(torch.mean(deltas[selected] ** 2))),
                "predicted_score": float(score[selected]),
                "predicted_success_probability": float(values["success_probability"][selected]),
                "predicted_feasible_probability": float(values["feasible_probability"][selected]),
                "predicted_minimum_gate_margin": float(values["minimum_gate_margin"][selected]),
                "predicted_tip_distance_m": float(values["tip_distance_m"][selected]),
            }
        )
    return output, diagnostics


def _write_report(
    artifact: Path,
    config: dict[str, Any],
    inventory: dict[str, Any],
    training: dict[str, Any],
    diagnostic: dict[str, Any],
    basis_summary: dict[str, Any],
    iterative: dict[str, Any],
) -> str:
    initial = iterative["iterations"][0]["summary"]
    final = iterative["iterations"][-1]["summary"]
    train_initial, train_final = initial["TRAIN"], final["TRAIN"]
    dev_initial, dev_final = initial["DEVELOPMENT"], final["DEVELOPMENT"]
    train_gain = 100.0 * (train_final["scientific_success_rate"] - train_initial["scientific_success_rate"])
    dev_gain = 100.0 * (dev_final["scientific_success_rate"] - dev_initial["scientific_success_rate"])
    best_development_iteration = max(
        iterative["iterations"],
        key=lambda item: (
            item["summary"]["DEVELOPMENT"]["scientific_success_rate"],
            item["summary"]["DEVELOPMENT"]["feasible_rate"],
            -item["execution_count"],
        ),
    )
    if dev_final["scientific_success_rate"] >= 0.50 and dev_gain >= 15.0:
        classification = "IRP_STYLE_REFINEMENT_PROMISING"
    elif train_gain > 0.0 or dev_gain > 0.0:
        classification = "IRP_STYLE_REFINEMENT_PARTIAL"
    else:
        classification = "IRP_STYLE_REFINEMENT_NOT_WORKING"
    iterative["classification"] = classification
    iterative["train_success_gain_percentage_points"] = train_gain
    iterative["development_success_gain_percentage_points"] = dev_gain
    iterative["recommended_execution_count"] = best_development_iteration["execution_count"]
    iterative["recommended_correction_count"] = best_development_iteration["execution_count"] - 1
    _write_json(artifact / "iterative_evaluation_summary.json", iterative)

    report = f"""# Iterative Residual Policy Pilot Report

## 1. Decision

This pilot tests the actual iterative-residual mechanism used by successful dynamic-rope work, rather than another direct action-regression policy. It uses **zero new CEM solves**, no SAC, no diffusion, no scorer, no final test, no protected data, and no hardware.

## 2. Detailed current pipeline

The complete experimental path is:

1. A physically propagated UAV/cable state and target are converted to the frozen root-centered, yaw-aligned 83-D `PolicyContext`.
2. The already-trained 7B deterministic residual MLP supplies the initial complete normalized 49-D maneuver. It is only an initializer; its development success was 16.22%.
3. That maneuver is decoded by the production codec into 16 three-axis acceleration knots and `T_maneuver` in [0.45, 1.80] s.
4. The command executes as `ACTIVE -> 0.30-s analytic SETTLE -> HOLD`, observed to 2.40 s in the fixed-2048 frozen simulator.
5. The observed rollout is compressed into six physical continuous outcomes (tip distance, directed speed, direction cosine, UAV displacement, UAV speed, command acceleration) plus tip-first, feasibility, and scientific-success flags.
6. A delta-outcome MLP receives normalized context, current action, observed outcome, and one proposed action correction. It predicts the physical result of that correction.
7. Exactly {config['residual_search']['candidate_count']} corrections are generated in a {basis_summary['rank']}-D local basis fitted from existing CEM-family residuals. The authoritative action remains 49-D; the compact basis changes only local search.
8. Candidates are ranked using predicted unchanged hard-gate margins. Neither CEM nor the production simulator participates in ranking.
9. One correction is selected and executed. Steps 5-9 repeat up to {config['residual_search']['maximum_iterations']} times, stopping per context on scientific success.

This is an iterative physical-trial method, not the earlier one-query deployment objective. It is deliberately tested now because direct one-query imitation failed on knife-edge CEM teachers.

## 3. Existing data

- Teacher contexts: {inventory['authoritative_successes']}
- State-disjoint training contexts: {training['train_context_count']}
- State-disjoint development contexts: {training['development_context_count']}
- Saved perturbations/context: {training['perturbations_per_context']}
- Physical outcome rows used for gradients: {training['physical_outcome_rows_used_for_gradients']}
- New CEM solves: 0

The data are the existing 7C exact/IID/smooth/duration perturbations evaluated under the final production action, settle, and physics contract.

## 4. Compact correction primitive

The correction basis has {basis_summary['rank']} dimensions. It is an orthonormal SVD basis of the training-only difference between the 7B initial action and the exact CEM teacher action. It does not approximate or reconstruct the complete teacher action, which avoids the spline reconstruction failure from 7B. Proposed corrections remain bounded and are canonicalized by the production 49-D codec.

## 5. Learned delta-outcome model

Architecture: `(83 context + 49 current action + 9 observed outcomes + 49 delta) -> {config['model']['hidden_dimension']} SiLU -> {config['model']['hidden_dimension']} SiLU -> {config['model']['hidden_dimension']} SiLU`, followed by six continuous and three binary heads. Parameters: {training['parameter_count']:,}. Best development loss: {training['best_validation_loss']:.6f} at update {training['best_update']}.

Held-out pair prediction used {diagnostic['pair_count']} development pairs. Binary accuracy was {diagnostic['binary_accuracy']}.

## 6. Authoritative iterative result

| Executions | Train success | Train feasible | Development success | Development feasible |
|---:|---:|---:|---:|---:|
"""
    for item in iterative["iterations"]:
        summary = item["summary"]
        report += (
            f"| {item['execution_count']} | "
            f"{100*summary['TRAIN']['scientific_success_rate']:.2f}% | "
            f"{100*summary['TRAIN']['feasible_rate']:.2f}% | "
            f"{100*summary['DEVELOPMENT']['scientific_success_rate']:.2f}% | "
            f"{100*summary['DEVELOPMENT']['feasible_rate']:.2f}% |\n"
        )
    report += f"""

Every entry is the one model-selected maneuver actually replayed through the complete production simulator. Candidate actions that were not selected were not simulated.

## 7. Interpretation

Train success changed by {train_gain:+.2f} percentage points and state-disjoint development success changed by {dev_gain:+.2f} points. Classification: **{classification}**.

This result answers whether learned local physical response plus iterative correction is more useful than direct coordinate imitation. It does not establish a one-query policy and it does not authorize hardware. If refinement is weak, the next issue is the saved perturbation experiment: it records outcome summaries around successful teachers, not complete tip trajectories around arbitrary failed starting maneuvers. More CEM winners would not repair that mismatch.

Development success first reached its maximum after {best_development_iteration['execution_count']} executions ({best_development_iteration['execution_count'] - 1} corrections). Later corrections did not add development successes and reduced failure-row feasibility, so the practical stopping point for this frozen pilot is {best_development_iteration['execution_count'] - 1} corrections. The next scientifically useful dataset would record complete tip trajectories and local responses around the actual failed initializer—not generate more nominal CEM winners.

## 8. Scientific restrictions

- Model: `MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI`
- Production model modified: **NO**
- New CEM solves: **0**
- SAC/diffusion/scorer: **NOT USED**
- Final TEST: **NOT EVALUATED**
- Protected `fig8vertical_002`: **NOT EVALUATED**
- Hardware: **NOT EXECUTED**

## Final summary

    Method:
        ITERATIVE RESIDUAL OUTCOME MODEL

    Initial maneuver:
        7B DETERMINISTIC RESIDUAL MLP

    Correction dimension:
        {basis_summary['rank']}

    Candidate corrections/iteration:
        {config['residual_search']['candidate_count']}

    Maximum correction iterations:
        {config['residual_search']['maximum_iterations']}

    Recommended corrections for frozen pilot:
        {best_development_iteration['execution_count'] - 1}

    New CEM solves:
        0

    Training contexts:
        {training['train_context_count']}

    Development contexts:
        {training['development_context_count']}

    Initial train success:
        {100*train_initial['scientific_success_rate']:.2f}%

    Final train success:
        {100*train_final['scientific_success_rate']:.2f}%

    Initial development success:
        {100*dev_initial['scientific_success_rate']:.2f}%

    Final development success:
        {100*dev_final['scientific_success_rate']:.2f}%

    Classification:
        {classification}

    Final TEST:
        NOT EVALUATED

    Protected test:
        NOT EVALUATED

    Hardware:
        NOT EXECUTED
"""
    (artifact / "ITERATIVE_RESIDUAL_POLICY_PILOT_REPORT.md").write_text(report, encoding="utf-8")
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
        raise RuntimeError("Production teacher order disagrees with learned-data order.")
    _write_json(
        artifact / "existing_data_inventory.json",
        {
            **inventory,
            "robustness_rows": int(np.prod(data["continuous"].shape[:2])),
            "train_contexts": int(len(data["train_indices"])),
            "development_contexts": int(len(data["dev_indices"])),
            "state_overlap": data["split"]["state_overlap"],
        },
    )

    model, outcome_normalizer, training = _train_model(config, data, artifact)
    diagnostic = _prediction_diagnostic(model, outcome_normalizer, data)
    _write_json(artifact / "heldout_prediction_diagnostic.json", diagnostic)

    train_indices = torch.from_numpy(data["train_indices"])
    basis = fit_compact_residual_basis(
        torch.from_numpy(data["initial_actions"])[train_indices],
        torch.from_numpy(data["teachers"])[train_indices],
        rank=int(config["residual_search"]["basis_rank"]),
    )
    torch.save(basis, artifact / "compact_residual_basis.pt")
    training_residual = torch.from_numpy(data["teachers"])[train_indices] - torch.from_numpy(data["initial_actions"])[train_indices]
    projected = (training_residual @ basis.T) @ basis
    basis_summary = {
        "schema": "compact_cem_family_residual_basis_v1",
        "rank": int(basis.shape[0]),
        "action_dimension": int(basis.shape[1]),
        "training_context_count": int(len(train_indices)),
        "orthonormal_max_error": float((basis @ basis.T - torch.eye(len(basis))).abs().max()),
        "training_residual_energy_retained": float(projected.square().sum() / training_residual.square().sum()),
        "complete_action_representation_changed": False,
    }
    _write_json(artifact / "compact_residual_basis.json", basis_summary)

    environment = load_fixed_production_environment(config)
    simulator, task, training_bank, _heldout, canonical_bank, _settings = environment
    physics_settings = production_cem_settings(config)
    current_actions = data["initial_actions"].copy()
    split_ids = {
        "TRAIN": set(data["split"]["training_context_ids"]),
        "DEVELOPMENT": set(data["split"]["development_context_ids"]),
    }
    action_history = [current_actions.copy()]
    selection_history: list[list[dict[str, Any]]] = []
    iteration_rows: list[dict[str, Any]] = []

    rows = evaluate_teacher_action_candidates(
        simulator,
        task,
        physics_settings,
        records,
        current_actions[:, None, :],
        training_bank,
        canonical_bank,
        label="iterative_residual_execution_0",
    )
    current_continuous, current_binary = outcome_arrays_from_rows(rows)
    iteration_rows.append(
        {"execution_count": 1, "correction_iteration": 0, "summary": _summarize_rows(records, rows, split_ids)}
    )
    for iteration in range(1, int(config["residual_search"]["maximum_iterations"]) + 1):
        current_actions, selection = _select_corrections(
            model,
            outcome_normalizer,
            data["normalized_contexts"],
            current_actions,
            current_continuous,
            current_binary,
            basis,
            config,
            iteration=iteration,
        )
        selection_history.append(selection)
        rows = evaluate_teacher_action_candidates(
            simulator,
            task,
            physics_settings,
            records,
            current_actions[:, None, :],
            training_bank,
            canonical_bank,
            label=f"iterative_residual_execution_{iteration}",
        )
        current_continuous, current_binary = outcome_arrays_from_rows(rows)
        action_history.append(current_actions.copy())
        summary = _summarize_rows(records, rows, split_ids)
        iteration_rows.append(
            {
                "execution_count": iteration + 1,
                "correction_iteration": iteration,
                "summary": summary,
            }
        )
        print(
            f"execution {iteration + 1}: train={100*summary['TRAIN']['scientific_success_rate']:.2f}% "
            f"dev={100*summary['DEVELOPMENT']['scientific_success_rate']:.2f}%",
            flush=True,
        )
    np.savez_compressed(
        artifact / "executed_actions.npz",
        schema=np.asarray("iterative_residual_executed_actions_v1"),
        context_ids=data["context_ids"],
        normalized_actions=np.stack(action_history, axis=1),
    )
    _write_json(artifact / "selection_diagnostics.json", selection_history)
    iterative = {
        "schema": "iterative_residual_authoritative_evaluation_v1",
        "initial_method": "DETERMINISTIC_RESIDUAL_MLP",
        "candidate_corrections_ranked_by_simulator": False,
        "one_authoritative_execution_per_iteration": True,
        "maximum_correction_iterations": int(config["residual_search"]["maximum_iterations"]),
        "iterations": iteration_rows,
    }
    classification = _write_report(
        artifact, config, inventory, training, diagnostic, basis_summary, iterative
    )
    hash_inputs = [
        Path(__file__),
        ROOT / "learning" / "iterative_residual.py",
        ROOT / "learning" / "cem_teacher_support.py",
        config_path,
        ROOT / config["milestone_7b_artifact"] / "teacher_dataset.npz",
        ROOT / config["milestone_7c_artifact"] / "perturbed_actions.npz",
        ROOT / config["milestone_7c_artifact"] / "perturbation_outcomes.npz",
    ]
    _write_json(
        artifact / "source_hash_manifest.json",
        {
            "schema": "iterative_residual_source_hash_manifest_v1",
            "files": {str(path.relative_to(ROOT)): _sha256(path) for path in hash_inputs},
        },
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
