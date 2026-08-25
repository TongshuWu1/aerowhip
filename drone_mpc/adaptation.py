"""Tip-only DDER output-error identification for the one-attached cable.

This module is deliberately narrower than a full cable-state observer.  It
assumes that an experiment starts from the known, settled hanging state and
uses only the measured attachment and free-tip trajectories to update the two
tip-observable material parameters, EI and Cb.  No simulated interior cable
state is part of the measurement contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time as wall_time
from typing import Sequence

import numpy as np
import torch

from cable_twin.shared.dder import DderState, START_PINNED_FREE_END

from .model import CableModelSnapshot
from .reduced import reduce_cable_model


ADAPTATION_SCHEMA = "known_start_tip_only_dder_identification_v1"
PARAMETER_NAMES = ("log_ei_scale", "log_cb_scale")


def _readonly_float_array(
    value: np.ndarray | Sequence[float],
    *,
    name: str,
    ndim: int,
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D array.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TipOnlyMeasurements:
    """The complete information available to the online cable adapter.

    Arrays contain no interior cable positions or velocities.  The commanded
    acceleration is retained for provenance; the measured attachment motion is
    the actual boundary input to DDER.
    """

    time_s: np.ndarray
    attachment_positions_m: np.ndarray
    free_tip_positions_m: np.ndarray
    commanded_accelerations_m_s2: np.ndarray
    source: str = "measurement"

    def __post_init__(self) -> None:
        time = _readonly_float_array(self.time_s, name="time_s", ndim=1)
        attachment = _readonly_float_array(
            self.attachment_positions_m,
            name="attachment_positions_m",
            ndim=2,
        )
        tip = _readonly_float_array(
            self.free_tip_positions_m,
            name="free_tip_positions_m",
            ndim=2,
        )
        acceleration = _readonly_float_array(
            self.commanded_accelerations_m_s2,
            name="commanded_accelerations_m_s2",
            ndim=2,
        )
        frame_count = len(time)
        if frame_count < 3:
            raise ValueError("Tip-only identification needs at least three frames.")
        if attachment.shape != (frame_count, 3) or tip.shape != (frame_count, 3):
            raise ValueError("Attachment and free-tip measurements must have shape Tx3.")
        if acceleration.shape != (frame_count - 1, 3):
            raise ValueError("Commanded accelerations must have shape (T-1)x3.")
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("Measurement timestamps must be strictly increasing.")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Measurement source must be a non-empty string.")
        object.__setattr__(self, "time_s", time)
        object.__setattr__(self, "attachment_positions_m", attachment)
        object.__setattr__(self, "free_tip_positions_m", tip)
        object.__setattr__(self, "commanded_accelerations_m_s2", acceleration)

    @property
    def frame_count(self) -> int:
        return len(self.time_s)


@dataclass(frozen=True, slots=True)
class TipAdaptationSettings:
    """Fixed choices for bounded two-parameter output-error fitting."""

    optimizer_iterations: int = 6
    measurement_noise_std_m: float = 0.002
    prior_log_scale_std: float = math.log(1.5)
    ei_scale_bounds: tuple[float, float] = (0.5, 2.0)
    cb_scale_bounds: tuple[float, float] = (0.5, 2.0)
    sensitivity_log_step: float = 0.01
    minimum_relative_information_eigenvalue: float = 1.0
    maximum_information_condition: float = 1.0e4
    initial_lm_damping: float = 1.0e-3
    convergence_relative_tolerance: float = 1.0e-4
    convergence_step_tolerance: float = 1.0e-3

    def __post_init__(self) -> None:
        if self.optimizer_iterations < 1:
            raise ValueError("optimizer_iterations must be positive.")
        positive = (
            self.measurement_noise_std_m,
            self.prior_log_scale_std,
            self.sensitivity_log_step,
            self.minimum_relative_information_eigenvalue,
            self.maximum_information_condition,
            self.initial_lm_damping,
            self.convergence_relative_tolerance,
            self.convergence_step_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Adaptation scales, limits, and tolerances must be positive.")
        for name, bounds in (
            ("EI", self.ei_scale_bounds),
            ("Cb", self.cb_scale_bounds),
        ):
            if (
                len(bounds) != 2
                or not all(math.isfinite(value) and value > 0.0 for value in bounds)
                or bounds[0] >= bounds[1]
                or not bounds[0] <= 1.0 <= bounds[1]
            ):
                raise ValueError(f"{name} scale bounds must be positive and contain 1.")


@dataclass(frozen=True, slots=True)
class TipAdaptationResult:
    schema: str
    model_sha256: str
    provisional_model: bool
    frame_count: int
    duration_s: float
    update_applied: bool
    freeze_reason: str | None
    nominal_ei_n_m2: float
    nominal_cb_n_m2_s: float
    estimated_ei_n_m2: float
    estimated_cb_n_m2_s: float
    estimated_ei_scale: float
    estimated_cb_scale: float
    initial_tip_rmse_m: float
    final_tip_rmse_m: float
    best_iteration: int
    evaluations: int
    normalized_sensitivity_singular_values: tuple[float, float]
    relative_information_eigenvalues: tuple[float, float]
    information_condition: float
    log_parameter_standard_deviations: tuple[float, float]
    log_parameter_correlation: float
    ei_scale_95_interval: tuple[float, float]
    cb_scale_95_interval: tuple[float, float]
    runtime_s: float
    loss_history: tuple[float, ...]
    predicted_tip_positions_m: np.ndarray

    def __post_init__(self) -> None:
        prediction = _readonly_float_array(
            self.predicted_tip_positions_m,
            name="predicted_tip_positions_m",
            ndim=2,
        )
        if prediction.shape != (self.frame_count, 3):
            raise ValueError("Predicted tip trajectory must have shape frame_count x 3.")
        object.__setattr__(self, "predicted_tip_positions_m", prediction)


@dataclass(frozen=True, slots=True)
class SyntheticExcitationSettings:
    duration_s: float = 2.0
    dt_s: float = 0.02
    maximum_acceleration_m_s2: float = 6.0
    attachment_start_m: tuple[float, float, float] = (0.0, 0.0, -0.10)
    tip_noise_std_m: float = 0.0
    noise_seed: int = 0
    excitation_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.duration_s)
            or self.duration_s <= 0.0
            or not math.isfinite(self.dt_s)
            or self.dt_s <= 0.0
            or not math.isclose(
                self.duration_s / self.dt_s,
                round(self.duration_s / self.dt_s),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError("Excitation duration must be a positive integer number of steps.")
        if not math.isfinite(self.maximum_acceleration_m_s2) or self.maximum_acceleration_m_s2 < 0.0:
            raise ValueError("maximum_acceleration_m_s2 must be finite and non-negative.")
        if not math.isfinite(self.tip_noise_std_m) or self.tip_noise_std_m < 0.0:
            raise ValueError("tip_noise_std_m must be finite and non-negative.")
        if not math.isfinite(self.excitation_phase_rad):
            raise ValueError("excitation_phase_rad must be finite.")
        if len(self.attachment_start_m) != 3 or not all(
            math.isfinite(value) for value in self.attachment_start_m
        ):
            raise ValueError("attachment_start_m must contain three finite values.")


def fixed_solver_model(
    source: CableModelSnapshot,
    *,
    node_count: int = 15,
    substeps: int = 2,
    constraint_iterations: int = 4,
    bending_stiffness_scale: float = 1.0,
    bending_damping_scale: float = 1.0,
) -> CableModelSnapshot:
    """Create one explicit, immutable solver configuration for all cases."""

    return reduce_cable_model(
        source,
        node_count=node_count,
        substeps=substeps,
        constraint_iterations=constraint_iterations,
        bending_stiffness_scale=bending_stiffness_scale,
        bending_damping_scale=bending_damping_scale,
    )


def _known_hanging_state(
    model: CableModelSnapshot,
    attachment_m: torch.Tensor,
    batch_size: int = 1,
) -> DderState:
    gravity = torch.as_tensor(
        model.model.parameters.gravity_camera_m_s2,
        dtype=attachment_m.dtype,
        device=attachment_m.device,
    )
    gravity_norm = torch.linalg.vector_norm(gravity)
    if not bool(torch.isfinite(gravity_norm).detach().cpu()) or float(gravity_norm) <= 0.0:
        raise ValueError("The cable model must contain non-zero finite gravity.")
    down = gravity / gravity_norm
    material = torch.as_tensor(
        model.rod_material_coordinates_m,
        dtype=attachment_m.dtype,
        device=attachment_m.device,
    )
    if attachment_m.shape != (3,):
        raise ValueError("Initial attachment must be a three-vector.")
    positions = attachment_m[None, None] + material[None, :, None] * down[None, None]
    positions = positions.repeat(batch_size, 1, 1)
    return DderState(positions, torch.zeros_like(positions))


def _predict_tip_batch(
    model: CableModelSnapshot,
    time_s: torch.Tensor,
    attachment_positions_m: torch.Tensor,
    *,
    log_scales: torch.Tensor,
) -> torch.Tensor:
    if time_s.ndim != 1 or attachment_positions_m.shape != (len(time_s), 3):
        raise ValueError("Prediction inputs must contain T timestamps and Tx3 attachments.")
    if log_scales.ndim != 2 or log_scales.shape[1] != 2:
        raise ValueError("log_scales must have shape Bx2.")
    batch_size = int(log_scales.shape[0])
    state = _known_hanging_state(model, attachment_positions_m[0], batch_size)
    tips = [state.positions_m[:, -1]]
    nominal_ei = torch.as_tensor(
        model.bending_stiffness_n_m2,
        dtype=attachment_positions_m.dtype,
        device=attachment_positions_m.device,
    )
    nominal_cb = torch.as_tensor(
        model.bending_damping_n_m2_s,
        dtype=attachment_positions_m.dtype,
        device=attachment_positions_m.device,
    )
    scales = torch.exp(log_scales)
    ei = nominal_ei * scales[:, 0]
    cb = nominal_cb * scales[:, 1]
    for frame_index in range(1, len(time_s)):
        dt = time_s[frame_index] - time_s[frame_index - 1]
        state = model.model.step(
            state,
            attachment_positions_m[frame_index][None, None].repeat(batch_size, 1, 1),
            dt.repeat(batch_size),
            create_graph=False,
            bending_stiffness_n_m2=ei,
            bending_damping_n_m2_s=cb,
            _validate=frame_index == 1,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        tips.append(state.positions_m[:, -1])
    return torch.stack(tips, dim=1)


def _prediction_numpy(
    model: CableModelSnapshot,
    measurements: TipOnlyMeasurements,
    log_scales: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    time = torch.tensor(measurements.time_s, dtype=torch.float64, device=device)
    attachment = torch.tensor(
        measurements.attachment_positions_m,
        dtype=torch.float64,
        device=device,
    )
    logs = torch.tensor(log_scales[None], dtype=torch.float64, device=device)
    prediction = _predict_tip_batch(
        model,
        time,
        attachment,
        log_scales=logs,
    )
    return prediction[0].detach().cpu().numpy()


def predict_tip_positions(
    model: CableModelSnapshot,
    measurements: TipOnlyMeasurements,
    *,
    ei_scale: float = 1.0,
    cb_scale: float = 1.0,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Predict a tip record from its measured boundary and chosen parameters."""

    if (
        not math.isfinite(ei_scale)
        or ei_scale <= 0.0
        or not math.isfinite(cb_scale)
        or cb_scale <= 0.0
    ):
        raise ValueError("EI and Cb scales must be finite and positive.")
    prediction = _prediction_numpy(
        model,
        measurements,
        np.log(np.asarray((ei_scale, cb_scale), dtype=np.float64)),
        torch.device(device),
    )
    prediction.setflags(write=False)
    return prediction


def _sensitivity_diagnostics(
    model: CableModelSnapshot,
    measurements: TipOnlyMeasurements,
    log_scales: np.ndarray,
    settings: TipAdaptationSettings,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
    step = settings.sensitivity_log_step
    perturbations = np.repeat(np.asarray(log_scales)[None], 5, axis=0)
    perturbations[1, 0] += step
    perturbations[2, 0] -= step
    perturbations[3, 1] += step
    perturbations[4, 1] -= step
    time = torch.tensor(measurements.time_s, dtype=torch.float64, device=device)
    attachment = torch.tensor(
        measurements.attachment_positions_m,
        dtype=torch.float64,
        device=device,
    )
    predictions = _predict_tip_batch(
        model,
        time,
        attachment,
        log_scales=torch.tensor(perturbations, dtype=torch.float64, device=device),
    ).detach().cpu().numpy()
    jacobian = np.stack(
        (
            ((predictions[1, 1:] - predictions[2, 1:]) / (2.0 * step)).reshape(-1),
            ((predictions[3, 1:] - predictions[4, 1:]) / (2.0 * step)).reshape(-1),
        ),
        axis=1,
    )
    normalized = jacobian / settings.measurement_noise_std_m
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular[-1] <= np.finfo(np.float64).eps:
        condition = math.inf
    else:
        # The information matrix is J.T @ J, so its spectral condition is the
        # square of the whitened sensitivity-Jacobian condition.
        condition = float((singular[0] / singular[-1]) ** 2)
    information = normalized.T @ normalized
    prior_std = settings.prior_log_scale_std
    relative_information = (prior_std * prior_std) * information
    relative_eigenvalues = np.linalg.eigvalsh(relative_information)
    posterior_precision = information + np.eye(2) / (prior_std * prior_std)
    covariance = np.linalg.inv(posterior_precision)
    correlation = float(
        covariance[0, 1] / math.sqrt(covariance[0, 0] * covariance[1, 1])
    )
    return (
        predictions[0],
        jacobian,
        singular,
        relative_eigenvalues,
        condition,
        covariance,
        correlation,
    )


def estimate_tip_only_parameters(
    model: CableModelSnapshot,
    measurements: TipOnlyMeasurements,
    settings: TipAdaptationSettings = TipAdaptationSettings(),
    *,
    device: str | torch.device = "cuda",
) -> TipAdaptationResult:
    """Estimate EI/Cb from one causal, known-start attachment/tip record.

    The MAP problem is solved in log-parameter space with a bounded, two-
    parameter Levenberg--Marquardt iteration.  Central-difference trajectories
    are evaluated as one batched DDER rollout.  The information test is run
    before optimization, so an uninformative record cannot manufacture a
    prior-driven parameter update.
    """

    started = wall_time.perf_counter()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected for adaptation but is unavailable.")
    maximum_dt = float(np.max(np.diff(measurements.time_s)))
    maximum_ei = model.model.maximum_stable_bending_stiffness(
        maximum_dt,
        pinned_endpoints=START_PINNED_FREE_END,
    )
    if model.bending_stiffness_n_m2 * settings.ei_scale_bounds[1] > maximum_ei:
        raise ValueError(
            "The fixed adaptation solver is unstable at the upper EI bound; "
            "increase its fixed substep count before generating any cases."
        )

    observed = measurements.free_tip_positions_m
    prior_std = settings.prior_log_scale_std
    prior_precision = np.eye(2, dtype=np.float64) / (prior_std * prior_std)
    lower = np.log(
        np.asarray((settings.ei_scale_bounds[0], settings.cb_scale_bounds[0]))
    )
    upper = np.log(
        np.asarray((settings.ei_scale_bounds[1], settings.cb_scale_bounds[1]))
    )

    def objective(prediction: np.ndarray, logs: np.ndarray) -> float:
        residual = (prediction[1:] - observed[1:]).reshape(-1)
        whitened = residual / settings.measurement_noise_std_m
        return float(
            0.5 * np.dot(whitened, whitened)
            + 0.5 * logs @ prior_precision @ logs
        )

    logs = np.zeros(2, dtype=np.float64)
    (
        nominal_prediction,
        jacobian,
        singular,
        relative_eigenvalues,
        condition,
        covariance,
        correlation,
    ) = _sensitivity_diagnostics(
        model,
        measurements,
        logs,
        settings,
        target_device,
    )
    initial_rmse = float(
        np.sqrt(np.mean(np.sum(np.square(nominal_prediction - observed), axis=1)))
    )
    informative = (
        float(relative_eigenvalues[0])
        >= settings.minimum_relative_information_eigenvalue
        and condition <= settings.maximum_information_condition
    )
    standard_deviations = tuple(
        float(math.sqrt(max(value, 0.0))) for value in np.diag(covariance)
    )
    if not informative:
        prior_interval = (
            max(
                settings.ei_scale_bounds[0],
                float(math.exp(-1.96 * prior_std)),
            ),
            min(
                settings.ei_scale_bounds[1],
                float(math.exp(+1.96 * prior_std)),
            ),
        )
        cb_prior_interval = (
            max(
                settings.cb_scale_bounds[0],
                float(math.exp(-1.96 * prior_std)),
            ),
            min(
                settings.cb_scale_bounds[1],
                float(math.exp(+1.96 * prior_std)),
            ),
        )
        reason = (
            "insufficient tip sensitivity: the EI/Cb update is frozen because "
            "the measured maneuver does not independently excite both parameters"
        )
        return TipAdaptationResult(
            schema=ADAPTATION_SCHEMA,
            model_sha256=model.sha256,
            provisional_model=model.provisional,
            frame_count=measurements.frame_count,
            duration_s=float(measurements.time_s[-1] - measurements.time_s[0]),
            update_applied=False,
            freeze_reason=reason,
            nominal_ei_n_m2=model.bending_stiffness_n_m2,
            nominal_cb_n_m2_s=model.bending_damping_n_m2_s,
            estimated_ei_n_m2=model.bending_stiffness_n_m2,
            estimated_cb_n_m2_s=model.bending_damping_n_m2_s,
            estimated_ei_scale=1.0,
            estimated_cb_scale=1.0,
            initial_tip_rmse_m=initial_rmse,
            final_tip_rmse_m=initial_rmse,
            best_iteration=0,
            evaluations=1,
            normalized_sensitivity_singular_values=(
                float(singular[0]),
                float(singular[1]),
            ),
            relative_information_eigenvalues=(
                float(relative_eigenvalues[0]),
                float(relative_eigenvalues[1]),
            ),
            information_condition=condition,
            log_parameter_standard_deviations=(prior_std, prior_std),
            log_parameter_correlation=0.0,
            ei_scale_95_interval=prior_interval,
            cb_scale_95_interval=cb_prior_interval,
            runtime_s=wall_time.perf_counter() - started,
            loss_history=(objective(nominal_prediction, logs),),
            predicted_tip_positions_m=nominal_prediction,
        )

    damping = settings.initial_lm_damping
    prediction = nominal_prediction
    current_objective = objective(prediction, logs)
    best_logs = logs.copy()
    best_prediction = prediction.copy()
    best_objective = current_objective
    best_iteration = 0
    history = [current_objective]
    evaluations = 1
    for iteration in range(1, settings.optimizer_iterations + 1):
        residual = (prediction[1:] - observed[1:]).reshape(-1)
        whitened_jacobian = jacobian / settings.measurement_noise_std_m
        whitened_residual = residual / settings.measurement_noise_std_m
        hessian = whitened_jacobian.T @ whitened_jacobian + prior_precision
        gradient = whitened_jacobian.T @ whitened_residual + prior_precision @ logs
        diagonal = np.maximum(np.diag(hessian), 1.0)
        step = -np.linalg.solve(hessian + damping * np.diag(diagonal), gradient)
        standardized_step = float(np.linalg.norm(step / prior_std))
        if standardized_step < settings.convergence_step_tolerance:
            break
        candidate_logs = np.clip(logs + step, lower, upper)
        candidate_prediction = _prediction_numpy(
            model,
            measurements,
            candidate_logs,
            target_device,
        )
        evaluations += 1
        candidate_objective = objective(candidate_prediction, candidate_logs)
        if candidate_objective < current_objective:
            relative_improvement = (
                current_objective - candidate_objective
            ) / max(abs(current_objective), 1.0)
            logs = candidate_logs
            prediction = candidate_prediction
            current_objective = candidate_objective
            damping = max(damping / 3.0, 1.0e-12)
            if candidate_objective < best_objective:
                best_objective = candidate_objective
                best_logs = candidate_logs.copy()
                best_prediction = candidate_prediction.copy()
                best_iteration = iteration
            history.append(candidate_objective)
            if (
                relative_improvement < settings.convergence_relative_tolerance
                and standardized_step < settings.convergence_step_tolerance
            ):
                break
            if iteration < settings.optimizer_iterations:
                (
                    prediction,
                    jacobian,
                    _singular,
                    _relative_eigenvalues,
                    _condition,
                    _covariance,
                    _correlation,
                ) = _sensitivity_diagnostics(
                    model,
                    measurements,
                    logs,
                    settings,
                    target_device,
                )
        else:
            damping = min(damping * 10.0, 1.0e12)
            history.append(current_objective)

    (
        _final_prediction_at_center,
        _final_jacobian,
        final_singular,
        final_relative_eigenvalues,
        final_condition,
        final_covariance,
        final_correlation,
    ) = _sensitivity_diagnostics(
        model,
        measurements,
        best_logs,
        settings,
        target_device,
    )
    final_rmse = float(
        np.sqrt(np.mean(np.sum(np.square(best_prediction - observed), axis=1)))
    )
    final_informative = (
        float(final_relative_eigenvalues[0])
        >= settings.minimum_relative_information_eigenvalue
        and final_condition <= settings.maximum_information_condition
    )
    if not final_informative:
        prior_interval = (
            max(
                settings.ei_scale_bounds[0],
                float(math.exp(-1.96 * prior_std)),
            ),
            min(
                settings.ei_scale_bounds[1],
                float(math.exp(+1.96 * prior_std)),
            ),
        )
        cb_prior_interval = (
            max(
                settings.cb_scale_bounds[0],
                float(math.exp(-1.96 * prior_std)),
            ),
            min(
                settings.cb_scale_bounds[1],
                float(math.exp(+1.96 * prior_std)),
            ),
        )
        return TipAdaptationResult(
            schema=ADAPTATION_SCHEMA,
            model_sha256=model.sha256,
            provisional_model=model.provisional,
            frame_count=measurements.frame_count,
            duration_s=float(measurements.time_s[-1] - measurements.time_s[0]),
            update_applied=False,
            freeze_reason=(
                "candidate update rejected: local EI/Cb information at the "
                "fitted parameters is insufficient or ill-conditioned"
            ),
            nominal_ei_n_m2=model.bending_stiffness_n_m2,
            nominal_cb_n_m2_s=model.bending_damping_n_m2_s,
            estimated_ei_n_m2=model.bending_stiffness_n_m2,
            estimated_cb_n_m2_s=model.bending_damping_n_m2_s,
            estimated_ei_scale=1.0,
            estimated_cb_scale=1.0,
            initial_tip_rmse_m=initial_rmse,
            final_tip_rmse_m=initial_rmse,
            best_iteration=best_iteration,
            evaluations=evaluations,
            normalized_sensitivity_singular_values=(
                float(final_singular[0]),
                float(final_singular[1]),
            ),
            relative_information_eigenvalues=(
                float(final_relative_eigenvalues[0]),
                float(final_relative_eigenvalues[1]),
            ),
            information_condition=final_condition,
            log_parameter_standard_deviations=(prior_std, prior_std),
            log_parameter_correlation=0.0,
            ei_scale_95_interval=prior_interval,
            cb_scale_95_interval=cb_prior_interval,
            runtime_s=wall_time.perf_counter() - started,
            loss_history=tuple(history),
            predicted_tip_positions_m=nominal_prediction,
        )
    scales = np.exp(best_logs)
    standard_deviations = tuple(
        float(math.sqrt(max(value, 0.0))) for value in np.diag(final_covariance)
    )

    def scale_interval(index: int) -> tuple[float, float]:
        bounds = settings.ei_scale_bounds if index == 0 else settings.cb_scale_bounds
        return (
            max(
                bounds[0],
                float(math.exp(best_logs[index] - 1.96 * standard_deviations[index])),
            ),
            min(
                bounds[1],
                float(math.exp(best_logs[index] + 1.96 * standard_deviations[index])),
            ),
        )

    return TipAdaptationResult(
        schema=ADAPTATION_SCHEMA,
        model_sha256=model.sha256,
        provisional_model=model.provisional,
        frame_count=measurements.frame_count,
        duration_s=float(measurements.time_s[-1] - measurements.time_s[0]),
        update_applied=True,
        freeze_reason=None,
        nominal_ei_n_m2=model.bending_stiffness_n_m2,
        nominal_cb_n_m2_s=model.bending_damping_n_m2_s,
        estimated_ei_n_m2=model.bending_stiffness_n_m2 * float(scales[0]),
        estimated_cb_n_m2_s=model.bending_damping_n_m2_s * float(scales[1]),
        estimated_ei_scale=float(scales[0]),
        estimated_cb_scale=float(scales[1]),
        initial_tip_rmse_m=initial_rmse,
        final_tip_rmse_m=final_rmse,
        best_iteration=best_iteration,
        evaluations=evaluations,
        normalized_sensitivity_singular_values=(
            float(final_singular[0]),
            float(final_singular[1]),
        ),
        relative_information_eigenvalues=(
            float(final_relative_eigenvalues[0]),
            float(final_relative_eigenvalues[1]),
        ),
        information_condition=final_condition,
        log_parameter_standard_deviations=standard_deviations,  # type: ignore[arg-type]
        log_parameter_correlation=final_correlation,
        ei_scale_95_interval=scale_interval(0),
        cb_scale_95_interval=scale_interval(1),
        runtime_s=wall_time.perf_counter() - started,
        loss_history=tuple(history),
        predicted_tip_positions_m=best_prediction,
    )


def _multisine_accelerations(
    time_mid_s: np.ndarray,
    maximum_m_s2: float,
    phase_rad: float,
) -> np.ndarray:
    if maximum_m_s2 == 0.0:
        return np.zeros((len(time_mid_s), 3), dtype=np.float64)
    t = time_mid_s
    raw = np.column_stack(
        (
            np.sin(2.0 * np.pi * 0.70 * t + phase_rad)
            + 0.45 * np.sin(2.0 * np.pi * 1.35 * t + 0.3 - 0.4 * phase_rad),
            0.75 * np.sin(2.0 * np.pi * 0.95 * t + 0.7 + 0.6 * phase_rad),
            0.30 * np.sin(2.0 * np.pi * 1.20 * t + 1.1 - 0.8 * phase_rad),
        )
    )
    raw -= np.mean(raw, axis=0, keepdims=True)
    peak = float(np.max(np.linalg.norm(raw, axis=1)))
    return raw * (maximum_m_s2 / peak)


def make_synthetic_tip_measurements(
    hidden_model: CableModelSnapshot,
    settings: SyntheticExcitationSettings = SyntheticExcitationSettings(),
    *,
    device: str | torch.device = "cpu",
) -> TipOnlyMeasurements:
    """Generate a deterministic hidden-plant record without exposing its nodes."""

    frame_count = int(round(settings.duration_s / settings.dt_s)) + 1
    time = np.arange(frame_count, dtype=np.float64) * settings.dt_s
    accelerations = _multisine_accelerations(
        0.5 * (time[:-1] + time[1:]),
        settings.maximum_acceleration_m_s2,
        settings.excitation_phase_rad,
    )
    attachment = np.empty((frame_count, 3), dtype=np.float64)
    attachment[0] = np.asarray(settings.attachment_start_m, dtype=np.float64)
    velocity = np.zeros(3, dtype=np.float64)
    for frame_index, acceleration in enumerate(accelerations, start=1):
        attachment[frame_index] = (
            attachment[frame_index - 1]
            + settings.dt_s * velocity
            + 0.5 * settings.dt_s * settings.dt_s * acceleration
        )
        velocity = velocity + settings.dt_s * acceleration
    target_device = torch.device(device)
    time_tensor = torch.tensor(time, dtype=torch.float64, device=target_device)
    attachment_tensor = torch.tensor(
        attachment,
        dtype=torch.float64,
        device=target_device,
    )
    tip = _predict_tip_batch(
        hidden_model,
        time_tensor,
        attachment_tensor,
        log_scales=torch.zeros((1, 2), dtype=torch.float64, device=target_device),
    )[0].detach().cpu().numpy()
    if settings.tip_noise_std_m > 0.0:
        generator = np.random.default_rng(settings.noise_seed)
        tip = tip + generator.normal(0.0, settings.tip_noise_std_m, tip.shape)
    return TipOnlyMeasurements(
        time_s=time,
        attachment_positions_m=attachment,
        free_tip_positions_m=tip,
        commanded_accelerations_m_s2=accelerations,
        source=f"synthetic hidden plant {hidden_model.sha256}",
    )


__all__ = (
    "ADAPTATION_SCHEMA",
    "PARAMETER_NAMES",
    "SyntheticExcitationSettings",
    "TipAdaptationResult",
    "TipAdaptationSettings",
    "TipOnlyMeasurements",
    "estimate_tip_only_parameters",
    "fixed_solver_model",
    "make_synthetic_tip_measurements",
    "predict_tip_positions",
)
