"""CUDA system identification for the free-cable DDER transition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from cable_twin.der import (
    BatchedDderModel,
    DDER_ASSUMPTIONS,
    DDER_EQUATIONS,
    DDER_MODEL_FAMILY,
    DderModelParameters,
)

from .reference import DderReferenceDataset


@dataclass(frozen=True, slots=True)
class IdentificationSettings:
    epochs: int = 80
    rollout_steps: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    observation_sigma_floor_m: float = 0.003
    maximum_dt_s: float = 0.05
    minimum_motion_rms_m_s: float = 0.01
    minimum_shape_excitation_rms_m: float = 0.005
    bending_stiffness_initial_n_m2: float = 1.0e-4
    bending_stiffness_min_n_m2: float = 1.0e-7
    bending_stiffness_max_n_m2: float = 1.0e-2
    velocity_damping_initial_s_inv: float = 2.0
    velocity_damping_min_s_inv: float = 0.0
    velocity_damping_max_s_inv: float = 30.0
    coarse_stiffness_samples: int = 13
    coarse_damping_samples: int = 9
    coarse_refinement_samples: int = 17
    coarse_refinement_levels: int = 4
    substeps: int = 2
    constraint_iterations: int = 8
    constraint_relaxation: float = 1.0
    seed: int = 1729

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.rollout_steps <= 0 or self.batch_size <= 0:
            raise ValueError("identification counts must be positive")
        for name in (
            "learning_rate",
            "observation_sigma_floor_m",
            "maximum_dt_s",
            "minimum_motion_rms_m_s",
            "minimum_shape_excitation_rms_m",
            "bending_stiffness_initial_n_m2",
            "bending_stiffness_min_n_m2",
            "bending_stiffness_max_n_m2",
            "velocity_damping_initial_s_inv",
            "velocity_damping_max_s_inv",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.velocity_damping_min_s_inv < 0.0:
            raise ValueError("velocity damping fitting bound must be nonnegative")
        if self.bending_stiffness_max_n_m2 <= self.bending_stiffness_min_n_m2:
            raise ValueError("bending stiffness bounds are invalid")
        if self.velocity_damping_max_s_inv <= self.velocity_damping_min_s_inv:
            raise ValueError("velocity damping bounds are invalid")
        if not (
            self.bending_stiffness_min_n_m2
            <= self.bending_stiffness_initial_n_m2
            <= self.bending_stiffness_max_n_m2
        ):
            raise ValueError("initial bending stiffness must lie within its bounds")
        if not (
            self.velocity_damping_min_s_inv
            <= self.velocity_damping_initial_s_inv
            <= self.velocity_damping_max_s_inv
        ):
            raise ValueError("initial damping must lie within its bounds")
        if self.coarse_stiffness_samples < 2 or self.coarse_damping_samples < 2:
            raise ValueError("coarse identification grids require at least two samples")
        if (
            self.coarse_refinement_samples < 3
            or self.coarse_refinement_levels <= 0
        ):
            raise ValueError("coarse refinement settings are invalid")
        if self.substeps <= 0 or self.constraint_iterations <= 0:
            raise ValueError("DDER solver counts must be positive")
        if not 0.0 < self.constraint_relaxation <= 1.0:
            raise ValueError("constraint_relaxation must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    parameters: DderModelParameters
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _TrainingWindows:
    start_positions: torch.Tensor
    start_velocities: torch.Tensor
    target_positions: torch.Tensor
    target_endpoints: torch.Tensor
    target_weights: torch.Tensor
    dt_s: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.start_positions.shape[0])


def _sequence_ranges(sequence_ids: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(sequence_ids) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(sequence_ids)]))
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _build_training_windows(
    datasets: Sequence[DderReferenceDataset],
    settings: IdentificationSettings,
    device: torch.device,
) -> _TrainingWindows:
    starts_q = []
    starts_v = []
    targets = []
    endpoints = []
    weights = []
    timesteps = []
    sigma_floor2 = settings.observation_sigma_floor_m**2
    for dataset in datasets:
        for sequence_start, sequence_stop in _sequence_ranges(dataset.sequence_id_i32):
            available = sequence_stop - sequence_start
            if available <= settings.rollout_steps:
                continue
            for start in range(sequence_start, sequence_stop - settings.rollout_steps):
                stop = start + settings.rollout_steps + 1
                timestamp_s = (
                    dataset.timestamps_ns_i64[start:stop].astype(np.float64) * 1.0e-9
                )
                dt = np.diff(timestamp_s)
                if np.any(dt <= 0.0) or np.any(dt > settings.maximum_dt_s):
                    continue
                target = dataset.centerlines_m_f32[start + 1 : stop]
                covariance = dataset.covariance_m2_f32[start + 1 : stop]
                variance = np.trace(covariance, axis1=-2, axis2=-1) / 3.0
                observed = dataset.node_observed_bool[start + 1 : stop]
                weight = observed.astype(np.float32) / np.maximum(
                    variance,
                    sigma_floor2,
                )
                weight[:, (0, -1)] = 0.0
                if not np.any(weight > 0.0):
                    continue
                starts_q.append(dataset.centerlines_m_f32[start])
                starts_v.append(dataset.velocities_m_s_f32[start])
                targets.append(target)
                endpoints.append(target[:, (0, -1)])
                weights.append(weight)
                timesteps.append(dt.astype(np.float32))
    if not starts_q:
        raise ValueError("No valid contiguous DDER training windows were found")
    return _TrainingWindows(
        start_positions=torch.as_tensor(np.stack(starts_q), device=device),
        start_velocities=torch.as_tensor(np.stack(starts_v), device=device),
        target_positions=torch.as_tensor(np.stack(targets), device=device),
        target_endpoints=torch.as_tensor(np.stack(endpoints), device=device),
        target_weights=torch.as_tensor(np.stack(weights), device=device),
        dt_s=torch.as_tensor(np.stack(timesteps), device=device),
    )


def _inverse_sigmoid(value: float) -> float:
    clipped = min(max(value, 1.0e-6), 1.0 - 1.0e-6)
    return math.log(clipped / (1.0 - clipped))


def _bounded_log_parameter(
    raw: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    log_lower = math.log(lower)
    log_upper = math.log(upper)
    return torch.exp(log_lower + (log_upper - log_lower) * torch.sigmoid(raw))


def _bounded_linear_parameter(
    raw: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


def _rollout_loss(
    model: BatchedDderModel,
    windows: _TrainingWindows,
    indices: torch.Tensor,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = windows.start_positions[indices]
    v = windows.start_velocities[indices]
    weighted_error = torch.zeros((), dtype=q.dtype, device=q.device)
    weight_total = torch.zeros((), dtype=q.dtype, device=q.device)
    squared_error = torch.zeros((), dtype=q.dtype, device=q.device)
    observed_total = torch.zeros((), dtype=q.dtype, device=q.device)
    for step in range(windows.target_positions.shape[1]):
        q, v = model.step(
            q,
            v,
            windows.target_endpoints[indices, step],
            windows.dt_s[indices, step],
            bending_stiffness_n_m2=stiffness,
            velocity_damping_s_inv=damping,
        )
        residual2 = torch.sum(
            (q - windows.target_positions[indices, step]) ** 2,
            dim=-1,
        )
        weight = windows.target_weights[indices, step]
        weighted_error = weighted_error + torch.sum(weight * residual2)
        weight_total = weight_total + 3.0 * torch.sum(weight)
        observed = weight > 0.0
        squared_error = squared_error + torch.sum(residual2[observed])
        observed_total = observed_total + 3.0 * torch.sum(observed)
    return (
        weighted_error / torch.clamp(weight_total, min=1.0),
        squared_error,
        observed_total,
    )


def _coarse_initialization(
    model: BatchedDderModel,
    windows: _TrainingWindows,
    settings: IdentificationSettings,
) -> tuple[float, float, float]:
    stiffness_values = np.unique(
        np.concatenate(
            (
                np.geomspace(
                    settings.bending_stiffness_min_n_m2,
                    settings.bending_stiffness_max_n_m2,
                    settings.coarse_stiffness_samples,
                ),
                [settings.bending_stiffness_initial_n_m2],
            )
        )
    )
    damping_values = np.unique(
        np.concatenate(
            (
                np.linspace(
                    settings.velocity_damping_min_s_inv,
                    settings.velocity_damping_max_s_inv,
                    settings.coarse_damping_samples,
                ),
                [settings.velocity_damping_initial_s_inv],
            )
        )
    )
    indices = torch.arange(windows.count, device=windows.start_positions.device)
    evaluated: dict[tuple[float, float], float] = {}

    def evaluate(stiffness: float, damping: float) -> None:
        stiffness = float(
            np.clip(
                stiffness,
                settings.bending_stiffness_min_n_m2,
                settings.bending_stiffness_max_n_m2,
            )
        )
        damping = float(
            np.clip(
                damping,
                settings.velocity_damping_min_s_inv,
                settings.velocity_damping_max_s_inv,
            )
        )
        key = (float(f"{stiffness:.12g}"), float(f"{damping:.12g}"))
        if key in evaluated:
            return
        loss, _squared_error, _observed_total = _rollout_loss(
            model,
            windows,
            indices,
            torch.as_tensor(stiffness, dtype=torch.float32, device=indices.device),
            torch.as_tensor(damping, dtype=torch.float32, device=indices.device),
        )
        value = float(loss.cpu())
        if math.isfinite(value):
            evaluated[key] = value

    with torch.no_grad():
        for stiffness in stiffness_values:
            for damping in damping_values:
                evaluate(float(stiffness), float(damping))
        stiffness_ratio = (
            settings.bending_stiffness_max_n_m2
            / settings.bending_stiffness_min_n_m2
        ) ** (1.0 / (settings.coarse_stiffness_samples - 1))
        damping_step = (
            settings.velocity_damping_max_s_inv
            - settings.velocity_damping_min_s_inv
        ) / (settings.coarse_damping_samples - 1)
        for damping in damping_values:
            damping_results = sorted(
                (loss, stiffness)
                for (stiffness, candidate_damping), loss in evaluated.items()
                if math.isclose(candidate_damping, float(damping), abs_tol=1.0e-12)
            )
            _loss, stiffness = damping_results[0]
            for level in range(settings.coarse_refinement_levels + 1):
                log_radius = math.log(stiffness_ratio) / (2**level)
                for local_ei in np.exp(
                    np.linspace(
                        math.log(stiffness) - log_radius,
                        math.log(stiffness) + log_radius,
                        settings.coarse_refinement_samples,
                    )
                ):
                    evaluate(float(local_ei), float(damping))
                _loss, stiffness = min(
                    (loss, candidate_stiffness)
                    for (candidate_stiffness, candidate_damping), loss in evaluated.items()
                    if math.isclose(
                        candidate_damping,
                        float(damping),
                        abs_tol=1.0e-12,
                    )
                )

        for level in range(settings.coarse_refinement_levels):
            _best_loss, best_stiffness, best_damping = min(
                (loss, stiffness, damping)
                for (stiffness, damping), loss in evaluated.items()
            )
            log_radius = math.log(stiffness_ratio) / (2 ** (level + 1))
            damping_radius = damping_step / (2 ** (level + 1))
            local_stiffness = np.exp(
                np.linspace(
                    math.log(best_stiffness) - log_radius,
                    math.log(best_stiffness) + log_radius,
                    settings.coarse_refinement_samples,
                )
            )
            local_damping = np.linspace(
                best_damping - damping_radius,
                best_damping + damping_radius,
                settings.coarse_refinement_samples,
            )
            for local_ei in local_stiffness:
                for local_c in local_damping:
                    evaluate(float(local_ei), float(local_c))
    if not evaluated:
        raise RuntimeError("DDER coarse parameter search produced no finite rollout")
    loss, stiffness, damping = min(
        (loss, stiffness, damping)
        for (stiffness, damping), loss in evaluated.items()
    )
    return loss, stiffness, damping


def identify_free_cable(
    datasets: Sequence[DderReferenceDataset],
    *,
    linear_density_kg_m: float,
    cable_radius_m: float,
    gravity_camera_m_s2: tuple[float, float, float],
    settings: IdentificationSettings,
    device: torch.device,
    reference_paths: Sequence[Path] = (),
) -> IdentificationResult:
    """Identify bending stiffness and effective damping from free-cable motion."""

    if not datasets:
        raise ValueError("At least one DDER reference dataset is required")
    if not math.isfinite(linear_density_kg_m) or linear_density_kg_m <= 0.0:
        raise ValueError("linear_density_kg_m must be measured and positive")
    if not math.isfinite(cable_radius_m) or cable_radius_m <= 0.0:
        raise ValueError("cable_radius_m must be measured and positive")
    node_count = datasets[0].node_count
    cable_length = datasets[0].cable_length_m
    if any(
        dataset.node_count != node_count
        or not math.isclose(dataset.cable_length_m, cable_length, abs_tol=1.0e-9)
        for dataset in datasets
    ):
        raise ValueError("All DDER reference datasets must share node count and length")
    motion_rms = math.sqrt(
        float(
            np.mean(
                np.concatenate(
                    [dataset.velocities_m_s_f32.reshape(-1) for dataset in datasets]
                )
                ** 2
            )
        )
    )
    if motion_rms < settings.minimum_motion_rms_m_s:
        raise ValueError(
            "Reference motion is insufficient for dynamic identification: "
            f"rms={motion_rms:.6f} m/s"
        )
    excitation_squared = []
    for dataset in datasets:
        for start, stop in _sequence_ranges(dataset.sequence_id_i32):
            sequence = dataset.centerlines_m_f32[start:stop]
            excitation_squared.append(
                (sequence - np.mean(sequence, axis=0, keepdims=True)).reshape(-1) ** 2
            )
    shape_excitation_rms = math.sqrt(
        float(np.mean(np.concatenate(excitation_squared)))
    )
    if shape_excitation_rms < settings.minimum_shape_excitation_rms_m:
        raise ValueError(
            "Reference shape excitation is insufficient for identification: "
            f"rms={shape_excitation_rms:.6f} m"
        )

    windows = _build_training_windows(datasets, settings, device)
    base_parameters = DderModelParameters(
        node_count=node_count,
        cable_length_m=cable_length,
        cable_radius_m=cable_radius_m,
        linear_density_kg_m=linear_density_kg_m,
        bending_stiffness_n_m2=settings.bending_stiffness_initial_n_m2,
        velocity_damping_s_inv=settings.velocity_damping_initial_s_inv,
        gravity_camera_m_s2=gravity_camera_m_s2,
        substeps=settings.substeps,
        constraint_iterations=settings.constraint_iterations,
        constraint_relaxation=settings.constraint_relaxation,
    )
    model = BatchedDderModel(base_parameters)
    coarse_loss, coarse_stiffness, coarse_damping = _coarse_initialization(
        model,
        windows,
        settings,
    )
    stiffness_fraction = (
        math.log(coarse_stiffness)
        - math.log(settings.bending_stiffness_min_n_m2)
    ) / (
        math.log(settings.bending_stiffness_max_n_m2)
        - math.log(settings.bending_stiffness_min_n_m2)
    )
    damping_fraction = (
        coarse_damping
        - settings.velocity_damping_min_s_inv
    ) / (
        settings.velocity_damping_max_s_inv
        - settings.velocity_damping_min_s_inv
    )
    raw_stiffness = torch.tensor(
        _inverse_sigmoid(stiffness_fraction),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    raw_damping = torch.tensor(
        _inverse_sigmoid(damping_fraction),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        (raw_stiffness, raw_damping),
        lr=settings.learning_rate,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(settings.seed)
    history = []
    best = {
        "epoch": 0,
        "loss": coarse_loss,
        "bending_stiffness_n_m2": coarse_stiffness,
        "velocity_damping_s_inv": coarse_damping,
    }
    all_indices = torch.arange(windows.count, device=device)
    for epoch in range(settings.epochs):
        permutation = torch.randperm(
            windows.count,
            generator=generator,
            device=device,
        )
        epoch_loss = torch.zeros((), dtype=torch.float32, device=device)
        batch_count = 0
        for start in range(0, windows.count, settings.batch_size):
            indices = permutation[start : start + settings.batch_size]
            stiffness = _bounded_log_parameter(
                raw_stiffness,
                settings.bending_stiffness_min_n_m2,
                settings.bending_stiffness_max_n_m2,
            )
            damping = _bounded_linear_parameter(
                raw_damping,
                settings.velocity_damping_min_s_inv,
                settings.velocity_damping_max_s_inv,
            )
            optimizer.zero_grad(set_to_none=True)
            loss, _squared_error, _observed_total = _rollout_loss(
                model,
                windows,
                indices,
                stiffness,
                damping,
            )
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.detach()
            batch_count += 1
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == settings.epochs:
            current_stiffness = _bounded_log_parameter(
                raw_stiffness,
                settings.bending_stiffness_min_n_m2,
                settings.bending_stiffness_max_n_m2,
            ).detach()
            current_damping = _bounded_linear_parameter(
                raw_damping,
                settings.velocity_damping_min_s_inv,
                settings.velocity_damping_max_s_inv,
            ).detach()
            with torch.no_grad():
                evaluation_loss, _squared_error, _observed_total = _rollout_loss(
                    model,
                    windows,
                    all_indices,
                    current_stiffness,
                    current_damping,
                )
            evaluation_value = float(evaluation_loss.cpu())
            stiffness_value = float(current_stiffness.cpu())
            damping_value = float(current_damping.cpu())
            if evaluation_value < best["loss"]:
                best = {
                    "epoch": epoch + 1,
                    "loss": evaluation_value,
                    "bending_stiffness_n_m2": stiffness_value,
                    "velocity_damping_s_inv": damping_value,
                }
            history.append(
                {
                    "epoch": epoch + 1,
                    "training_batch_loss": float(
                        (epoch_loss / max(1, batch_count)).cpu()
                    ),
                    "evaluation_loss": evaluation_value,
                    "bending_stiffness_n_m2": stiffness_value,
                    "velocity_damping_s_inv": damping_value,
                }
            )

    stiffness = torch.as_tensor(
        best["bending_stiffness_n_m2"],
        dtype=torch.float32,
        device=device,
    )
    damping = torch.as_tensor(
        best["velocity_damping_s_inv"],
        dtype=torch.float32,
        device=device,
    )
    with torch.no_grad():
        final_loss, squared_error, observed_total = _rollout_loss(
            model,
            windows,
            all_indices,
            stiffness,
            damping,
        )
    rmse_m = torch.sqrt(squared_error / torch.clamp(observed_total, min=1.0))
    parameters = DderModelParameters(
        node_count=node_count,
        cable_length_m=cable_length,
        cable_radius_m=cable_radius_m,
        linear_density_kg_m=linear_density_kg_m,
        bending_stiffness_n_m2=float(stiffness.cpu()),
        velocity_damping_s_inv=float(damping.cpu()),
        gravity_camera_m_s2=gravity_camera_m_s2,
        substeps=settings.substeps,
        constraint_iterations=settings.constraint_iterations,
        constraint_relaxation=settings.constraint_relaxation,
    )
    metadata = {
        "method": "covariance_weighted_multistep_free_cable_dder_identification",
        "model_family": DDER_MODEL_FAMILY,
        "equations": DDER_EQUATIONS,
        "assumptions": dict(DDER_ASSUMPTIONS),
        "differentiated_quantities": [
            "geometric_bending_force",
            "constraint_projection",
            "multistep_rollout",
        ],
        "references": [
            {
                "path": (
                    str(Path(reference_paths[index]).resolve())
                    if index < len(reference_paths)
                    else None
                ),
                "artifact_sha256": dataset.metadata.get("artifact_sha256"),
                "recording_id": dataset.metadata.get("recording_id"),
                "cable_id": dataset.cable_id,
                "frame_count": dataset.frame_count,
            }
            for index, dataset in enumerate(datasets)
        ],
        "reference_count": len(datasets),
        "training_window_count": windows.count,
        "rollout_steps": settings.rollout_steps,
        "motion_rms_m_s": motion_rms,
        "shape_excitation_rms_m": shape_excitation_rms,
        "coarse_initialization": {
            "loss": coarse_loss,
            "bending_stiffness_n_m2": coarse_stiffness,
            "velocity_damping_s_inv": coarse_damping,
        },
        "selected_checkpoint": best,
        "final_covariance_weighted_loss": float(final_loss.cpu()),
        "final_observed_rmse_m": float(rmse_m.cpu()),
        "settings": asdict(settings),
        "torch_version": torch.__version__,
        "device": str(device),
        "history": history,
    }
    return IdentificationResult(parameters=parameters, metadata=metadata)


__all__ = [
    "IdentificationResult",
    "IdentificationSettings",
    "identify_free_cable",
]
