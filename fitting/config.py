"""Visible, validated identification configuration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


PARAMETER_NAMES = ("K_p", "K_v", "k_a", "K_R", "K_omega", "EI", "Cb")
DEFAULT_PATH = Path(__file__).with_name("default_fit.json")


@dataclass(frozen=True, slots=True)
class FitConfiguration:
    initialization_history_frames: int
    maximum_initialization_rmse_m: float
    robust_cable_scale_m: float
    joint_weights: dict[str, float]
    normalization_scales: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    optimizer: dict[str, float | int]
    uav_residual_ablation: dict[str, object]
    validation_lead_times_s: tuple[float, ...]
    source_payload: dict[str, object]


def load_fit_configuration(path: str | Path = DEFAULT_PATH) -> FitConfiguration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "aerial_cable_joint_fit_config_v1":
        raise ValueError("Unsupported fit configuration schema.")
    raw_bounds = payload["parameter_bounds_provisional"]
    bounds = {name: tuple(float(value) for value in raw_bounds[name]) for name in PARAMETER_NAMES}
    for name, (lower, upper) in bounds.items():
        if not 0.0 < lower < upper:
            raise ValueError(f"Invalid positive provisional bounds for {name}.")
    history = int(payload["initialization_history_frames"])
    if history < 3:
        raise ValueError("Causal initialization requires at least three frames.")
    residual = payload.get("uav_residual_ablation", {})
    if int(residual.get("history_samples", 0)) != 10:
        raise ValueError("Milestone 3A.2 requires exactly 10 residual history samples.")
    if list(residual.get("hidden_dimensions", [])) != [32, 32]:
        raise ValueError("Milestone 3A.2 residual hidden dimensions must be [32, 32].")
    if list(residual.get("input_features", [])) != ["e_p", "e_v", "a_cmd"]:
        raise ValueError("Milestone 3A.2 residual features must be e_p, e_v, a_cmd.")
    residual_optimizer = dict(residual.get("optimizer", {}))
    required_optimizer = {
        "seed", "learning_rate", "updates", "windows_per_take_per_update",
        "full_training_evaluation_interval", "gradient_clip",
    }
    if not required_optimizer.issubset(residual_optimizer):
        raise ValueError("Residual optimizer configuration is incomplete.")
    return FitConfiguration(
        initialization_history_frames=history,
        maximum_initialization_rmse_m=float(payload["maximum_initialization_rmse_m"]),
        robust_cable_scale_m=float(payload["robust_cable_scale_m"]),
        joint_weights={key: float(value) for key, value in payload["joint_weights"].items()},
        normalization_scales={key: float(value) for key, value in payload["normalization_scales"].items()},
        bounds=bounds,
        optimizer={key: value for key, value in payload["optimizer"].items()},
        uav_residual_ablation=dict(residual),
        validation_lead_times_s=tuple(float(value) for value in payload["validation_lead_times_s"]),
        source_payload=payload,
    )
