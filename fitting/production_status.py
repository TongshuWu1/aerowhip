"""Read-only model, dataset, fit, validation, and freeze summaries for the UI."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from experimental_data.io import sha256_file
from simulator.parameters import SimulatorSettings
from simulator.production import (
    PROJECT_ROOT,
    active_model_paths,
    load_active_model_manifest,
    resolve_portable_artifact_reference,
    verify_active_geometry,
    verify_active_parameters,
)

from .config import FitConfiguration, load_fit_configuration
from .dataset import Dataset, load_dataset
from .episodes import build_physical_episodes


ROLE_LABELS = {
    "training": "Training",
    "validation": "Provisional Validation",
    "untouched_test": "Protected Test",
    "ignore": "Ignore",
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_active_model_summary(settings: SimulatorSettings) -> dict[str, Any]:
    manifest = load_active_model_manifest()
    verify_active_geometry(settings, manifest)
    verify_active_parameters(settings, manifest)
    freeze = active_model_paths(manifest)["uav_residual_freeze"]
    freeze_manifest = _json(freeze / "manifest.json")
    residual_fit = _json(
        resolve_portable_artifact_reference(freeze_manifest["fit_directory"])
        / "residual_fit.json"
    )
    cable = settings.cable_configuration
    return {
        "uav_model": "Attitude-coupled effective FullState model",
        "uav_correction": "Causal acceleration residual",
        "residual_active": True,
        "cable_model": "12-node isotropic DDER",
        "coupling": "one-way UAV → cable",
        "attachment": "rigid two-node centerline clamp",
        "observations": "c1...c10",
        "cable_length_m": cable.length_m,
        "attachment_offset_m": abs(settings.attachment_offset_body_m[2]),
        "backend": "CUDA float32 / PCG32 / fused projection",
        "node_count": cable.node_count,
        "edge_count": cable.edge_count,
        "rest_lengths_m": list(cable.rest_lengths_m),
        "marker_node_mapping": list(cable.marker_node_indices[1:]),
        "substeps": cable.substeps,
        "position_projections": cable.constraint_iterations,
        "total_modeled_mass_kg": sum(cable.vertex_masses_kg),
        "residual_architecture": residual_fit["architecture"],
        "residual_parameter_count": residual_fit["parameter_count"],
        "residual_hash": freeze_manifest["artifact_hashes"]["residual_weights.pt"],
        "geometry_version": manifest["geometry_version"],
        "model_integrity": manifest.get("model_integrity", "Verified"),
    }


def _cable_eligibility_by_take(cable_fit: Path) -> dict[str, dict[str, int]]:
    audit = _json(cable_fit / "cable_episode_eligibility.json")
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "rejected": 0})
    for role in ("training", "validation"):
        for item in audit.get(role, []):
            key = "accepted" if bool(item["accepted"]) else "rejected"
            grouped[str(item["take_id"])][key] += 1
    return dict(grouped)


def get_dataset_role_summary(
    dataset: Dataset | None = None,
    config: FitConfiguration | None = None,
) -> list[dict[str, Any]]:
    selected = load_dataset() if dataset is None else dataset
    fit_config = load_fit_configuration() if config is None else config
    cable_fit = active_model_paths()["development_cable_fit"]
    cable_status = _cable_eligibility_by_take(cable_fit)
    rows = []
    for take in selected.takes:
        episodes, _ = build_physical_episodes(take, fit_config)
        residual_count = 0
        residual_duration = 0.0
        for episode in episodes:
            start = episode.residual_eligible_start_index
            if start is None:
                continue
            residual_count += 1
            residual_duration += float(
                take.arrays["time_s"][episode.end_index]
                - take.arrays["time_s"][start]
            )
        eligibility = cable_status.get(take.take_id)
        if take.role == "untouched_test":
            cable_text = "PROTECTED — NOT EVALUATED"
        elif eligibility is None:
            cable_text = "not in pinned cable-fit artifact"
        else:
            cable_text = (
                f"{eligibility['accepted']} eligible"
                + (f" / {eligibility['rejected']} rejected" if eligibility["rejected"] else "")
            )
        rows.append(
            {
                "take_id": take.take_id,
                "role": ROLE_LABELS[take.role],
                "physical_episode_count": len(episodes),
                "physical_duration_s": sum(item.duration_s for item in episodes),
                "residual_eligible_episode_count": residual_count,
                "residual_eligible_duration_s": residual_duration,
                "cable_status": cable_text,
            }
        )
    return rows


def get_latest_fit_summary(settings: SimulatorSettings) -> dict[str, Any]:
    manifest = load_active_model_manifest()
    paths = active_model_paths(manifest)
    freeze = paths["uav_residual_freeze"]
    freeze_manifest = _json(freeze / "manifest.json")
    fit = resolve_portable_artifact_reference(freeze_manifest["fit_directory"])
    comparison = _json(fit / "comparison_summary.json")
    physical = _json(freeze / "physical_parameters.json")
    residual = comparison["residual_fit"]
    cable_fit = _json(paths["development_cable_fit"] / "fitted_cable_parameters.json")
    ready = bool(manifest.get("ready_for_mppi", False))
    cable_summary_path = paths["development_cable_fit"] / "summary.json"
    cable_status_path = paths["development_cable_fit"] / "execution_status.json"
    cable_summary = (
        _json(cable_summary_path)
        if cable_summary_path.exists()
        else _json(cable_status_path)
    )
    same_p = comparison["uav_same_suffix_physics"]["validation"]
    same_pr = comparison["uav_same_suffix_residual"]["validation"]
    return {
        "uav": {
            "parameters": {key: physical[key] for key in ("K_p", "K_v", "k_a", "K_R", "K_omega")},
            "artifact": str(fit),
            "created_utc": comparison["created_utc"],
            "training_objective": comparison["uav_fit"]["best_training_objective"],
            "training_position_rmse_m": comparison["uav_physics_metrics"]["training"]["position_rmse_m"],
            "training_orientation_rmse_deg": comparison["uav_physics_metrics"]["training"]["orientation_rmse_deg"],
            "source_status": "frozen component",
        },
        "residual": {
            "active": True,
            "type": "causal translational acceleration residual",
            "history_ms": 100,
            "architecture": "90 → 32 → 32 → 3",
            "parameter_count": residual["parameter_count"],
            "training_objective": residual["best_training_objective"],
            "validation_position_rmse_physics_m": same_p["position_rmse_m"],
            "validation_position_rmse_pr_m": same_pr["position_rmse_m"],
            "hash": residual["network_state_sha256"],
            "normalization": str(freeze / "residual_normalization.json"),
        },
        "cable": {
            "EI": cable_fit["fitted_parameters"]["EI"],
            "Cb": cable_fit["fitted_parameters"]["Cb"],
            "EI_status": cable_fit["identifiability"]["EI"]["status"],
            "Cb_status": cable_fit["identifiability"]["Cb"]["status"],
            "training_objective": cable_fit["best_training_objective"],
            "fit_geometry_version": manifest["development_cable_fit_geometry_version"],
            "active_geometry_version": manifest["geometry_version"],
            "geometry_matches_active": (
                manifest["development_cable_fit_geometry_version"]
                == manifest["geometry_version"]
            ),
            "production_status": (
                "VALIDATED PRODUCTION VALUES — READY FOR MPPI"
                if ready
                else "DEVELOPMENT VALUES — GEOMETRY REFIT REQUIRED"
            ),
            "topology": "12 nodes / c1...c10 at nodes 2...11",
            "method": "measured UAV boundary → DDER; complete PhysicalEpisodes",
            "artifact": str(paths["development_cable_fit"]),
            "created_utc": cable_summary.get(
                "created_utc", cable_summary.get("completed_utc", "unknown")
            ),
            "loss_landscape": str(
                paths["development_cable_fit"]
                / (
                    "ei_cb_loss_landscape.png"
                    if ready
                    else "final_ei_cb_loss_landscape.png"
                )
            ),
            "profile_curves": str(
                paths["development_cable_fit"]
                / ("ei_profile.png" if ready else "ei_cb_profile_curves.png")
            ),
        },
    }


def get_validation_summary() -> dict[str, Any]:
    manifest = load_active_model_manifest()
    artifact = active_model_paths(manifest)["development_cable_fit"]
    ready = bool(manifest.get("ready_for_mppi", False))
    conditional = _json(
        artifact
        / (
            "conditional_task_horizon_validation.json"
            if ready
            else "conditional_12node.json"
        )
    )
    end_to_end = _json(
        artifact
        / (
            "pr_end_to_end_task_horizon_validation.json"
            if ready
            else "end_to_end_pr_12node.json"
        )
    )
    if ready:
        acceptance = _json(artifact / "pre_mppi_acceptance.json")
        thresholds = acceptance["thresholds"]
    else:
        fit_config = _json(artifact / "fit_config.json")
        thresholds = fit_config["task_horizon_acceptance_thresholds"]
    lead = end_to_end["aggregate_equal_take_rmse"]["lead_time"]
    horizon = lead["0.7"]
    fatal = [
        item
        for item in end_to_end["failed_episodes"]
        if item["reason"] != "cable_initialization_ineligible"
    ]
    checks = {
        key: float(horizon[key]) <= float(limit) for key, limit in thresholds.items()
    }
    return {
        "artifact": str(artifact),
        "artifact_geometry_version": manifest["development_cable_fit_geometry_version"],
        "active_geometry_version": manifest["geometry_version"],
        "geometry_matches_active": (
            manifest["development_cable_fit_geometry_version"]
            == manifest["geometry_version"]
        ),
        "conditional": conditional["validation"],
        "end_to_end": end_to_end["aggregate_equal_take_rmse"],
        "task_horizon_s": 0.7,
        "saved_artifact_pass": not fatal and all(checks.values()),
        "active_model_pass": ready and not fatal and all(checks.values()),
        "thresholds": thresholds,
        "threshold_checks": checks,
        "failed_or_ineligible_episodes": end_to_end["failed_episodes"],
    }


def get_active_model_freeze() -> dict[str, Any]:
    manifest = load_active_model_manifest()
    paths = active_model_paths(manifest)
    component_manifest = _json(paths["uav_residual_freeze"] / "manifest.json")
    ready = bool(manifest.get("ready_for_mppi", False))
    production_freeze = (
        (PROJECT_ROOT / str(manifest["production_freeze"])).resolve()
        if ready and manifest.get("production_freeze")
        else None
    )
    freeze_manifest = (
        _json(production_freeze / "manifest.json")
        if production_freeze is not None
        else component_manifest
    )
    source_manifest = (
        _json(production_freeze / "source_hash_manifest.json")
        if production_freeze is not None
        else freeze_manifest["source"]
    )
    return {
        "name": manifest["model_name"],
        "status": manifest["status"],
        "ready_for_mppi": manifest["ready_for_mppi"],
        "reason_not_ready": manifest["reason_not_ready"],
        "geometry_version": manifest["geometry_version"],
        "uav_parameter_source": str(paths["uav_residual_freeze"] / "physical_parameters.json"),
        "residual_hash": component_manifest["artifact_hashes"]["residual_weights.pt"],
        "EI": _json(paths["development_cable_fit"] / "fitted_cable_parameters.json")["fitted_parameters"]["EI"],
        "Cb": _json(paths["development_cable_fit"] / "fitted_cable_parameters.json")["fitted_parameters"]["Cb"],
        "topology": "12-node DDER",
        "source_hash": source_manifest["aggregate_source_sha256"],
        "dataset_role_snapshot": str(
            (production_freeze if production_freeze is not None else paths["uav_residual_freeze"])
            / "dataset_snapshot.json"
        ),
        "protected_test_status": "fig8vertical_002 — PROTECTED / NOT EVALUATED",
        "manifest_path": str(PROJECT_ROOT / "config" / "active_model.json"),
        "component_created_utc": freeze_manifest["created_utc"],
        "component_manifest_sha256": sha256_file(
            (production_freeze / "manifest.json")
            if production_freeze is not None
            else (paths["uav_residual_freeze"] / "manifest.json")
        ),
        "model_integrity": manifest.get("model_integrity", "Verified"),
        "production_freeze": (
            str(production_freeze) if production_freeze is not None else None
        ),
    }
