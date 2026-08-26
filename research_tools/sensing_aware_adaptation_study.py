"""OptiTrack-compatible causal sensing study for DDER EI/Cb adaptation.

This tool preserves the validated physical estimator and varies only the
observation/state-estimation path.  Simulator truth is used to generate
position-only measurements and to score results; it is never placed in a
``CableObservation`` or passed to the sensing-aware adapter.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Literal, Sequence

import numpy as np
import torch

from cable_twin.shared.dder import DderState

from drone_mpc.cable_observation import (
    CausalCableStateEstimator,
    DderInextensibilityVelocityProjector,
    SimulatedOptiTrackSource,
    StreamingSimulatedOptiTrackSource,
    distributed_observation_from_estimate,
)
from drone_mpc.distributed_adaptation import (
    DistributedAdaptationSettings,
    DistributedObservation,
    DistributedParameterFitter,
    OnlineAdaptationMonitor,
    ParameterEstimate,
    RollingDistributedBuffer,
    _information,
    _residual_vector,
    _segment_from_observations,
    select_informative_segments,
    snapshot_for_estimate,
)
from drone_mpc.receding_mppi import RecedingMppiSettings, run_receding_horizon_mppi
from drone_mpc.reduced import reduce_cable_model
from drone_mpc.simulator import DroneCableState, WhipSimulator
from research_tools.adaptation_identifiability_diagnostics import PHASE_STARTS
from research_tools.distributed_adaptation_study import (
    _control_row,
    _prediction_rows,
    _two_second_controls,
    _truth_model,
    generate_truth_rollout,
    observations_from_rollout,
)
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, Workload, load_workload


SCHEMA = "sensing_aware_dder_adaptation_study_v1"
DEFAULT_OUTPUT_DIRECTORY = Path("reports/sensing_aware_adaptation_data")


@dataclass(frozen=True, slots=True)
class SyntheticSensingSettings:
    measurement_rate_hz: float = 100.0
    polynomial_degree: int = 2
    history_length: int = 7
    position_output: Literal["polynomial", "latest_measurement"] = "polynomial"
    position_noise_std_m: float = 0.0
    latency_s: float = 0.0
    independent_dropout_probability: float = 0.0
    project_inextensible_velocity: bool = False
    dropout_burst_start_s: float | None = None
    dropout_burst_length_frames: int = 0


class SensedControllerStateProvider:
    """Bridge streaming position observations to the unchanged MPPI state API."""

    def __init__(
        self,
        workload: Workload,
        sensing: SyntheticSensingSettings,
        initial_state: DroneCableState,
        *,
        sensor_seed: int,
    ) -> None:
        self.workload = workload
        self.source = StreamingSimulatedOptiTrackSource(
            measurement_rate_hz=sensing.measurement_rate_hz,
            position_noise_std_m=sensing.position_noise_std_m,
            latency_s=sensing.latency_s,
            independent_dropout_probability=sensing.independent_dropout_probability,
            random_seed=sensor_seed,
        )
        self.estimator = CausalCableStateEstimator(
            polynomial_degree=sensing.polynomial_degree,
            history_length=sensing.history_length,
            position_output=sensing.position_output,
            velocity_projection=(
                DderInextensibilityVelocityProjector(workload.controller)
                if sensing.project_inextensible_velocity
                else None
            ),
        )
        self._last_truth_time_s = -math.inf
        self._latest = None
        self.measurement_age_s: list[float] = []
        self.estimator_compute_time_s: list[float] = []
        self._ingest_truth(0.0, initial_state, prime=True)

    def _truth_positions(self, state: DroneCableState) -> tuple[np.ndarray, np.ndarray]:
        cable = state.cable.positions_m[0].detach().cpu().numpy()
        root = cable[0]
        return root, cable

    def _ingest_truth(
        self, time_s: float, state: DroneCableState, *, prime: bool = False
    ) -> None:
        if time_s > self._last_truth_time_s + 1.0e-12:
            root, cable = self._truth_positions(state)
            if prime:
                self.source.prime(time_s, root, cable, prehistory_s=0.10)
            else:
                self.source.append_truth_positions(time_s, root, cable)
            self._last_truth_time_s = time_s
        for observation in self.source.observations_arrived_by(time_s):
            self._latest = self.estimator.update(observation)

    def __call__(self, time_s: float, plant_state: DroneCableState) -> DroneCableState:
        self._ingest_truth(time_s, plant_state)
        if self._latest is None:
            raise RuntimeError("No causal cable state is available at controller time.")
        estimate = self._latest
        self.measurement_age_s.append(time_s - estimate.sample_timestamp_s)
        self.estimator_compute_time_s.append(estimate.estimator_compute_time_s)
        dtype = plant_state.drone_position_m.dtype
        device = plant_state.drone_position_m.device
        positions = torch.tensor(
            np.array(estimate.cable_positions_m[None], copy=True),
            dtype=dtype,
            device=device,
        )
        velocities = torch.tensor(
            np.array(estimate.cable_velocities_m_s[None], copy=True),
            dtype=dtype,
            device=device,
        )
        drop = torch.tensor(
            (0.0, 0.0, self.workload.simulation.attachment_drop_m),
            dtype=dtype,
            device=device,
        )
        return DroneCableState(
            drone_position_m=positions[:, 0] + drop,
            drone_velocity_m_s=velocities[:, 0],
            cable=DderState(positions, velocities),
        )

    def ingest_plant_rollout(self, start_time_s: float, rollout) -> None:
        """Feed every realized physics frame to the private simulated sensor."""

        local_time = rollout.time_s.detach().cpu().numpy()
        positions = rollout.cable_positions_m[0].detach().cpu().numpy()
        for frame in range(1, len(local_time)):
            absolute_time = start_time_s + float(local_time[frame])
            root = positions[frame, 0]
            self.source.append_truth_positions(
                absolute_time, root, positions[frame]
            )
        # Measurement arrivals are consumed at the next controller query so
        # samples with synthetic latency remain causal.


def _serialize(value):
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _interpolate_path(
    sample_time_s: np.ndarray, truth_time_s: np.ndarray, values: np.ndarray
) -> np.ndarray:
    flat = values.reshape(len(truth_time_s), -1)
    output = np.stack(
        [np.interp(sample_time_s, truth_time_s, flat[:, index]) for index in range(flat.shape[1])],
        axis=1,
    )
    return output.reshape((len(sample_time_s),) + values.shape[1:])


def _rollout_numpy(rollout) -> dict[str, np.ndarray]:
    return {
        "time_s": rollout.time_s.detach().cpu().numpy().astype(np.float64),
        "attachment_position_m": rollout.attachment_positions_m[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "cable_position_m": rollout.cable_positions_m[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "cable_velocity_m_s": rollout.cable_velocities_m_s[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
        "controls_m_s2": rollout.accelerations_m_s2[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64),
    }


def reconstruct_sensed_observations(
    rollout,
    workload: Workload,
    sensing: SyntheticSensingSettings,
    *,
    sensor_seed: int,
) -> tuple[tuple[DistributedObservation, ...], dict[str, float], tuple[dict[str, object], ...]]:
    """Generate 100 Hz measurements, estimate causally, and emit at 50 Hz physics times."""

    truth = _rollout_numpy(rollout)
    source = SimulatedOptiTrackSource(
        truth["time_s"],
        truth["attachment_position_m"],
        truth["cable_position_m"],
        measurement_rate_hz=sensing.measurement_rate_hz,
        position_noise_std_m=sensing.position_noise_std_m,
        latency_s=sensing.latency_s,
        random_seed=sensor_seed,
        independent_dropout_probability=sensing.independent_dropout_probability,
        forced_invalid_mask=(
            _forced_dropout_mask(
                truth["time_s"],
                sensing.measurement_rate_hz,
                sensing.dropout_burst_start_s,
                sensing.dropout_burst_length_frames,
            )
            if sensing.dropout_burst_length_frames > 0
            else None
        ),
    )
    estimator = CausalCableStateEstimator(
        polynomial_degree=sensing.polynomial_degree,
        history_length=sensing.history_length,
        position_output=sensing.position_output,
        velocity_projection=(
            DderInextensibilityVelocityProjector(workload.controller)
            if sensing.project_inextensible_velocity
            else None
        ),
    )
    sample_times = np.asarray(
        [item.sample_timestamp_s for item in source.all_observations], dtype=np.float64
    )
    truth_positions = _interpolate_path(
        sample_times, truth["time_s"], truth["cable_position_m"]
    )
    truth_velocities = _interpolate_path(
        sample_times, truth["time_s"], truth["cable_velocity_m_s"]
    )
    truth_root = _interpolate_path(
        sample_times, truth["time_s"], truth["attachment_position_m"]
    )
    # Root velocity equals DDER node-zero velocity in the prescribed boundary state.
    truth_root_velocity = truth_velocities[:, 0]
    estimated_rows = []
    adaptation_observations: list[DistributedObservation] = []
    target = np.asarray(workload.problem.target_position_m, dtype=np.float64)
    physics_dt = workload.simulation.simulation_dt_s
    for index, measurement in enumerate(source.all_observations):
        estimate = estimator.update(measurement)
        marker_position_error = estimate.cable_positions_m[1:] - truth_positions[index, 1:]
        marker_velocity_error = estimate.cable_velocities_m_s[1:] - truth_velocities[index, 1:]
        estimated_rows.append(
            {
                "sample_timestamp_s": estimate.sample_timestamp_s,
                "arrival_timestamp_s": estimate.arrival_timestamp_s,
                "measurement_age_s": estimate.measurement_age_s,
                "marker_position_squared_error": float(np.mean(marker_position_error**2)),
                "marker_velocity_squared_error": float(np.mean(marker_velocity_error**2)),
                "tip_position_error_m": float(np.linalg.norm(marker_position_error[-1])),
                "tip_velocity_error_m_s": float(np.linalg.norm(marker_velocity_error[-1])),
                "root_velocity_error_m_s": float(
                    np.linalg.norm(estimate.cable_velocities_m_s[0] - truth_root_velocity[index])
                ),
                "compute_time_s": estimate.estimator_compute_time_s,
                "measured_fraction": float(np.mean(estimate.measurement_validity_mask)),
            }
        )
        physics_index = round((estimate.sample_timestamp_s - truth["time_s"][0]) / physics_dt)
        physics_time = truth["time_s"][0] + physics_index * physics_dt
        if abs(estimate.sample_timestamp_s - physics_time) > 1.0e-7:
            continue
        control_index = min(max(physics_index, 0), len(truth["controls_m_s2"]) - 1)
        contact = bool(
            np.any(
                np.linalg.norm(estimate.cable_positions_m - target[None], axis=1)
                <= workload.problem.maximum_tip_error_m
            )
        )
        adaptation_observations.append(
            distributed_observation_from_estimate(
                estimate,
                active_parameter_estimate=ParameterEstimate(),
                executed_action_m_s2=truth["controls_m_s2"][control_index],
                attachment_drop_m=workload.simulation.attachment_drop_m,
                contact=contact,
            )
        )
    rows = tuple(estimated_rows)
    stable = rows[min(len(rows), sensing.history_length) :]
    metrics = {
        "all_marker_position_rmse_m": float(
            math.sqrt(np.mean([row["marker_position_squared_error"] for row in stable]))
        ),
        "all_marker_velocity_rmse_m_s": float(
            math.sqrt(np.mean([row["marker_velocity_squared_error"] for row in stable]))
        ),
        "tip_position_rmse_m": float(
            math.sqrt(np.mean([float(row["tip_position_error_m"]) ** 2 for row in stable]))
        ),
        "tip_velocity_rmse_m_s": float(
            math.sqrt(np.mean([float(row["tip_velocity_error_m_s"]) ** 2 for row in stable]))
        ),
        "root_velocity_rmse_m_s": float(
            math.sqrt(np.mean([float(row["root_velocity_error_m_s"]) ** 2 for row in stable]))
        ),
        "mean_measurement_age_s": float(np.mean([row["measurement_age_s"] for row in stable])),
        "mean_estimator_compute_time_s": float(np.mean([row["compute_time_s"] for row in stable])),
        "p95_estimator_compute_time_s": float(
            np.quantile([row["compute_time_s"] for row in stable], 0.95)
        ),
        "mean_measured_fraction": float(np.mean([row["measured_fraction"] for row in stable])),
    }
    return tuple(adaptation_observations), metrics, rows


def _forced_dropout_mask(
    truth_time_s: np.ndarray,
    measurement_rate_hz: float,
    start_s: float | None,
    length_frames: int,
) -> np.ndarray:
    dt = 1.0 / measurement_rate_hz
    count = int(math.floor((truth_time_s[-1] - truth_time_s[0]) / dt + 1.0e-9)) + 1
    mask = np.zeros((count, 10), dtype=bool)
    if start_s is None or length_frames <= 0:
        return mask
    index = int(round((start_s - truth_time_s[0]) / dt))
    mask[max(0, index) : min(count, index + length_frames)] = True
    return mask


def exact_observations_at_physics_rate(rollout, workload: Workload) -> tuple[DistributedObservation, ...]:
    return observations_from_rollout(rollout, workload, ParameterEstimate())


def generate_extended_truth_rollout(
    workload: Workload,
    truth_ratio: tuple[float, float],
    *,
    horizon_s: float,
):
    simulation = replace(workload.simulation, horizon_s=horizon_s)
    truth_model = _truth_model(workload, *truth_ratio)
    simulator = WhipSimulator(truth_model, simulation, device="cuda")
    state = simulator.initial_state(workload.initial_xyz)
    controls = _two_second_controls(workload, simulation, simulator.device)
    rollout = simulator.rollout(state, controls, create_graph=False)
    torch.cuda.synchronize()
    return rollout


def _with_estimate(
    observation: DistributedObservation, estimate: ParameterEstimate
) -> DistributedObservation:
    return replace(observation, active_estimate=estimate)


def run_adaptation_replay(
    workload: Workload,
    observations: Sequence[DistributedObservation],
    settings: DistributedAdaptationSettings,
    truth_ratio: tuple[float, float],
    *,
    strike_count: int = 3,
) -> dict[str, object]:
    fitter = DistributedParameterFitter(workload.controller, settings, device="cuda")
    current = ParameterEstimate()
    fits = []
    trace = []
    estimates = []
    for strike in range(strike_count):
        monitor = OnlineAdaptationMonitor(fitter.predictor, settings)
        for original in observations:
            diagnostic = monitor.append(_with_estimate(original, current))
            if diagnostic is None:
                continue
            trace.append({"strike": strike + 1, **asdict(diagnostic)})
            if diagnostic.reason != "fit_candidate":
                continue
            fitting, validation = select_informative_segments(
                monitor.buffer.candidate_segments(), settings
            )
            if not fitting or not validation or not monitor.claim_trigger(diagnostic):
                continue
            result = fitter.fit(fitting, validation, current)
            fits.append(result)
            if result.accepted:
                current = result.candidate_estimate
        estimates.append(
            {
                "strike": strike + 1,
                "ei_ratio": current.ei_ratio,
                "cb_ratio": current.cb_ratio,
                "eta_error_l2": float(
                    np.linalg.norm(
                        current.eta - np.log(np.asarray(truth_ratio, dtype=np.float64))
                    )
                ),
            }
        )
    return {
        "truth_ei_ratio": truth_ratio[0],
        "truth_cb_ratio": truth_ratio[1],
        "estimates_by_strike": estimates,
        "final_ei_ratio": current.ei_ratio,
        "final_cb_ratio": current.cb_ratio,
        "final_eta_error_l2": estimates[-1]["eta_error_l2"],
        "fit_attempts": len(fits),
        "accepted_fits": sum(item.accepted for item in fits),
        "rejected_fits": sum(not item.accepted for item in fits),
        "fits": [asdict(item) for item in fits],
        "health_trace": trace,
    }


def phase_information(
    workload: Workload,
    observations: Sequence[DistributedObservation],
    settings: DistributedAdaptationSettings,
) -> list[dict[str, object]]:
    fitter = DistributedParameterFitter(workload.controller, settings, device="cuda")
    rows = []
    for phase, start_s in PHASE_STARTS.items():
        frames = [
            item
            for item in observations
            if start_s - 1.0e-9
            <= item.timestamp_s
            <= start_s + settings.segment_duration_s + 1.0e-9
        ]
        segment = _segment_from_observations(frames, 0.0, "sensing_phase")
        hypotheses = np.zeros((5, 2), dtype=np.float64)
        hypotheses[1, 0] = settings.finite_difference_log_step_ei
        hypotheses[2, 0] = -settings.finite_difference_log_step_ei
        hypotheses[3, 1] = settings.finite_difference_log_step_cb
        hypotheses[4, 1] = -settings.finite_difference_log_step_cb
        prediction, _velocity, _elapsed = fitter.predictor.predict((segment,), hypotheses)
        observed = segment.cable_positions_m[None]
        mask = segment.measurement_validity_mask[None]
        column_e = (
            _residual_vector(prediction[:, 1], observed, workload.controller.cable_length_m, mask)
            - _residual_vector(prediction[:, 2], observed, workload.controller.cable_length_m, mask)
        ) / (2.0 * settings.finite_difference_log_step_ei)
        column_c = (
            _residual_vector(prediction[:, 3], observed, workload.controller.cable_length_m, mask)
            - _residual_vector(prediction[:, 4], observed, workload.controller.cable_length_m, mask)
        ) / (2.0 * settings.finite_difference_log_step_cb)
        diagnostic = _information(np.stack((column_e, column_c), axis=1), settings)
        rows.append({"phase": phase, "start_time_s": start_s, **asdict(diagnostic)})
    return rows


def health_distribution(
    workload: Workload,
    observations: Sequence[DistributedObservation],
    settings: DistributedAdaptationSettings,
) -> dict[str, object]:
    diagnostic_settings = replace(
        settings,
        error_high_m2=1.0,
        error_low_m2=0.0,
        persistence_s=1000.0,
    )
    fitter = DistributedParameterFitter(
        workload.controller, diagnostic_settings, device="cuda"
    )
    monitor = OnlineAdaptationMonitor(fitter.predictor, diagnostic_settings)
    rows = []
    for observation in observations:
        diagnostic = monitor.append(observation)
        if diagnostic is not None:
            rows.append(asdict(diagnostic))
    instantaneous = np.asarray([item["instantaneous_error_m2"] for item in rows])
    ema = np.asarray([item["ema_error_m2"] for item in rows])
    return {
        "count": len(rows),
        "instantaneous_median_m2": float(np.median(instantaneous)),
        "instantaneous_p95_m2": float(np.quantile(instantaneous, 0.95)),
        "instantaneous_max_m2": float(np.max(instantaneous)),
        "ema_median_m2": float(np.median(ema)),
        "ema_p95_m2": float(np.quantile(ema, 0.95)),
        "ema_max_m2": float(np.max(ema)),
        "rows": rows,
    }


def calibrate_health_thresholds(
    workload: Workload,
    rollout,
    sensing: SyntheticSensingSettings,
    base_settings: DistributedAdaptationSettings,
    calibration_seeds: Sequence[int],
) -> tuple[DistributedAdaptationSettings, dict[str, object]]:
    ema_values = []
    runs = []
    for seed in calibration_seeds:
        observations, state_metrics, _rows = reconstruct_sensed_observations(
            rollout, workload, sensing, sensor_seed=seed
        )
        distribution = health_distribution(workload, observations, base_settings)
        ema_values.extend(item["ema_error_m2"] for item in distribution["rows"])
        runs.append({"seed": seed, "state_metrics": state_metrics, "health": distribution})
    values = np.asarray(ema_values, dtype=np.float64)
    # A conservative global threshold fitted only on matched calibration seeds.
    high = max(1.0e-14, float(np.max(values) * 1.25 + 1.0e-14))
    low = 0.40 * high
    return replace(base_settings, error_high_m2=high, error_low_m2=low), {
        "calibration_seeds": list(calibration_seeds),
        "matched_ema_median_m2": float(np.median(values)),
        "matched_ema_p95_m2": float(np.quantile(values, 0.95)),
        "matched_ema_max_m2": float(np.max(values)),
        "selected_error_high_m2": high,
        "selected_error_low_m2": low,
        "runs": runs,
    }


def run_zero_noise_stage(workload: Workload) -> dict[str, object]:
    settings = DistributedAdaptationSettings()
    cases = []
    for truth_ratio in ((1.0, 1.0), (0.8, 0.7)):
        rollout, _truth, _simulation = generate_truth_rollout(workload, *truth_ratio)
        exact = exact_observations_at_physics_rate(rollout, workload)
        exact_result = run_adaptation_replay(
            workload, exact, settings, truth_ratio
        )
        causal_sensing = SyntheticSensingSettings(position_output="latest_measurement")
        causal, metrics, _rows = reconstruct_sensed_observations(
            rollout, workload, causal_sensing, sensor_seed=0
        )
        causal_result = run_adaptation_replay(
            workload, causal, settings, truth_ratio
        )
        polynomial_sensing = replace(causal_sensing, position_output="polynomial")
        polynomial, polynomial_metrics, _rows = reconstruct_sensed_observations(
            rollout, workload, polynomial_sensing, sensor_seed=0
        )
        polynomial_result = run_adaptation_replay(
            workload, polynomial, settings, truth_ratio
        )
        improved_sensing = SyntheticSensingSettings(
            polynomial_degree=1,
            history_length=3,
            position_output="latest_measurement",
            project_inextensible_velocity=True,
        )
        improved, improved_metrics, _rows = reconstruct_sensed_observations(
            rollout, workload, improved_sensing, sensor_seed=0
        )
        improved_result = run_adaptation_replay(
            workload, improved, settings, truth_ratio
        )
        cases.append(
            {
                "truth_ratio": truth_ratio,
                "exact_position_exact_velocity": exact_result,
                "exact_position_causal_velocity": causal_result,
                "causal_polynomial_position_and_velocity": polynomial_result,
                "causal_first_order_three_sample": improved_result,
                "causal_state_metrics": metrics,
                "polynomial_state_metrics": polynomial_metrics,
                "improved_state_metrics": improved_metrics,
                "phase_information_exact": phase_information(workload, exact, settings),
                "phase_information_causal_velocity": phase_information(workload, causal, settings),
                "phase_information_polynomial": phase_information(workload, polynomial, settings),
                "phase_information_improved": phase_information(workload, improved, settings),
            }
        )
    return {"stage": "zero_noise", "cases": cases}


def run_noise_stage(workload: Workload) -> dict[str, object]:
    noise_levels = (0.0, 0.00025, 0.0005, 0.001, 0.002)
    truth_ratios = ((1.0, 1.0), (0.8, 1.0), (1.0, 0.7), (0.8, 0.7), (1.2, 0.7))
    base_settings = DistributedAdaptationSettings()
    matched_rollout, _truth, _simulation = generate_truth_rollout(workload, 1.0, 1.0)
    rows = []
    calibrations = []
    for noise in noise_levels:
        # The requested degree-2/seven-sample endpoint derivative is first
        # evaluated in the zero-noise stage.  It causes severe matched-model
        # physical bias.  The noise study therefore uses the simplest causal
        # correction supported by that ablation: a three-sample first-order
        # local fit, retaining the newest measured position.
        sensing = SyntheticSensingSettings(
            polynomial_degree=1,
            history_length=3,
            position_output="latest_measurement",
            position_noise_std_m=noise,
            project_inextensible_velocity=True,
        )
        calibrated, calibration = calibrate_health_thresholds(
            workload,
            matched_rollout,
            sensing,
            base_settings,
            calibration_seeds=(1001, 1002, 1003, 1004, 1005),
        )
        calibrations.append({"noise_std_m": noise, **calibration})
        for truth_ratio in truth_ratios:
            rollout, _truth, _simulation = generate_truth_rollout(workload, *truth_ratio)
            observations, state_metrics, _trace = reconstruct_sensed_observations(
                rollout, workload, sensing, sensor_seed=17
            )
            result = run_adaptation_replay(
                workload, observations, calibrated, truth_ratio
            )
            final = result["estimates_by_strike"][-1]
            final_estimate = ParameterEstimate(
                eta_e=math.log(float(final["ei_ratio"])),
                eta_c=math.log(float(final["cb_ratio"])),
                generation=1,
                source="sensing_study",
            )
            exact_evaluation = exact_observations_at_physics_rate(rollout, workload)
            prediction_fitter = DistributedParameterFitter(
                workload.controller, calibrated, device="cuda"
            )
            prediction_rows = _prediction_rows(
                prediction_fitter,
                exact_evaluation,
                final_estimate,
                np.log(np.asarray(truth_ratio, dtype=np.float64)),
                workload.problem.impact_direction,
            )
            rows.append(
                {
                    "noise_std_m": noise,
                    "truth_ratio": truth_ratio,
                    "health_thresholds": {
                        "error_high_m2": calibrated.error_high_m2,
                        "error_low_m2": calibrated.error_low_m2,
                    },
                    "state_metrics": state_metrics,
                    "adaptation": result,
                    "truth_evaluated_prediction_rows": prediction_rows,
                    "phase_information": phase_information(
                        workload, observations, calibrated
                    ),
                }
            )
            print(
                f"noise={1000*noise:.2f}mm truth={truth_ratio} "
                f"estimate=({result['final_ei_ratio']:.3f},{result['final_cb_ratio']:.3f}) "
                f"error={result['final_eta_error_l2']:.3f}"
            )
    return {"stage": "noise_sweep", "calibrations": calibrations, "rows": rows}


def run_fine_noise_boundary_stage(workload: Workload) -> dict[str, object]:
    """Resolve the detection boundary below the first requested 0.25 mm level.

    The requested sweep showed that 0.25 mm already suppresses every mismatch
    trigger after matched-model calibration.  This deliberately small follow-up
    brackets where that failure begins; it does not retune the estimator.
    """

    noise_levels = (0.00005, 0.00010, 0.00015, 0.00020)
    truth_ratios = ((1.0, 1.0), (0.8, 0.7))
    base_settings = DistributedAdaptationSettings()
    matched_rollout, _truth, _simulation = generate_truth_rollout(workload, 1.0, 1.0)
    rows = []
    calibrations = []
    for noise in noise_levels:
        sensing = SyntheticSensingSettings(
            polynomial_degree=1,
            history_length=3,
            position_output="latest_measurement",
            position_noise_std_m=noise,
            project_inextensible_velocity=True,
        )
        calibrated, calibration = calibrate_health_thresholds(
            workload,
            matched_rollout,
            sensing,
            base_settings,
            calibration_seeds=(1001, 1002, 1003, 1004, 1005),
        )
        calibrations.append({"noise_std_m": noise, **calibration})
        for truth_ratio in truth_ratios:
            rollout, _truth, _simulation = generate_truth_rollout(workload, *truth_ratio)
            observations, state_metrics, _trace = reconstruct_sensed_observations(
                rollout, workload, sensing, sensor_seed=17
            )
            result = run_adaptation_replay(workload, observations, calibrated, truth_ratio)
            rows.append(
                {
                    "noise_std_m": noise,
                    "truth_ratio": truth_ratio,
                    "health_thresholds": {
                        "error_high_m2": calibrated.error_high_m2,
                        "error_low_m2": calibrated.error_low_m2,
                    },
                    "state_metrics": state_metrics,
                    "adaptation": result,
                }
            )
            print(
                f"fine noise={1000*noise:.2f}mm truth={truth_ratio} "
                f"attempts={result['fit_attempts']} accepted={result['accepted_fits']} "
                f"estimate=({result['final_ei_ratio']:.3f},{result['final_cb_ratio']:.3f})"
            )
    return {
        "stage": "fine_noise_detection_boundary",
        "purpose": "diagnostic bracketing only; no estimator retuning",
        "calibrations": calibrations,
        "rows": rows,
    }


def run_full_sensed_control_stage(workload: Workload) -> dict[str, object]:
    """Small Mode-B diagnostic; exact truth is confined to each plant/evaluator."""

    base_settings = DistributedAdaptationSettings()
    rows = []
    adaptation_rows = []
    for noise in (0.0, 0.00005):
        sensing = SyntheticSensingSettings(
            polynomial_degree=1,
            history_length=3,
            position_output="latest_measurement",
            position_noise_std_m=noise,
            project_inextensible_velocity=True,
        )
        matched_rollout, _truth, _simulation = generate_truth_rollout(workload, 1.0, 1.0)
        calibrated, calibration = calibrate_health_thresholds(
            workload,
            matched_rollout,
            sensing,
            base_settings,
            calibration_seeds=(1001, 1002, 1003, 1004, 1005),
        )
        for truth_ratio in ((1.0, 1.0), (0.8, 0.7)):
            truth_rollout, _truth, _simulation = generate_truth_rollout(
                workload, *truth_ratio
            )
            sensed_observations, state_metrics, _trace = reconstruct_sensed_observations(
                truth_rollout, workload, sensing, sensor_seed=17
            )
            adaptation = run_adaptation_replay(
                workload, sensed_observations, calibrated, truth_ratio
            )
            adaptation_rows.append(
                {
                    "noise_std_m": noise,
                    "truth_ratio": truth_ratio,
                    "state_metrics": state_metrics,
                    "calibration": calibration,
                    "adaptation": adaptation,
                }
            )
            final = adaptation["estimates_by_strike"][-1]
            adapted_estimate = ParameterEstimate(
                eta_e=math.log(float(final["ei_ratio"])),
                eta_c=math.log(float(final["cb_ratio"])),
                generation=1,
                source="sensed_between_strike",
            )
            truth_model = _truth_model(workload, *truth_ratio)
            models = {
                "fixed_nominal": workload.controller,
                "adapted_after_strike_3": snapshot_for_estimate(
                    workload.controller, adapted_estimate
                ),
                "oracle": truth_model,
            }
            for seed in (11, 17, 29):
                for label, controller_model in models.items():
                    planner = WhipSimulator(
                        controller_model, workload.simulation, device="cuda"
                    )
                    plant = WhipSimulator(truth_model, workload.simulation, device="cuda")
                    initial = plant.initial_state(workload.initial_xyz)
                    provider = SensedControllerStateProvider(
                        workload, sensing, initial, sensor_seed=10000 + seed
                    )
                    execution = run_receding_horizon_mppi(
                        planner,
                        plant,
                        initial,
                        workload.problem,
                        replace(workload.mppi, seed=seed),
                        RecedingMppiSettings(
                            replan_interval_s=0.1,
                            timeout_s=1.2,
                            feedback_mode="full",
                        ),
                        workload.warm_knots,
                        controller_state_provider=provider,
                    )
                    row = _control_row(
                        execution,
                        f"ei{truth_ratio[0]:g}_cb{truth_ratio[1]:g}_noise{1000*noise:g}mm",
                        label,
                        seed,
                    )
                    row.update(
                        {
                            "noise_std_m": noise,
                            "truth_ei_ratio": truth_ratio[0],
                            "truth_cb_ratio": truth_ratio[1],
                            "feedback": "reconstructed_distributed_state",
                            "mean_measurement_age_s": float(
                                np.mean(provider.measurement_age_s)
                            ),
                            "mean_estimator_compute_time_s": float(
                                np.mean(provider.estimator_compute_time_s)
                            ),
                        }
                    )
                    rows.append(row)
                    print(
                        f"ModeB noise={1000*noise:.2f}mm truth={truth_ratio} "
                        f"{label} seed={seed} success={row['valid_strike']} "
                        f"error={1000*float(row['target_error_m']):.1f}mm"
                    )
    return {
        "stage": "full_sensed_control",
        "adaptation_rows": adaptation_rows,
        "control_rows": rows,
    }


def run_trigger_policy(
    workload: Workload,
    observations: Sequence[DistributedObservation],
    base_settings: DistributedAdaptationSettings,
    truth_ratio: tuple[float, float],
    policy: str,
) -> dict[str, object]:
    if policy == "continuous":
        settings = replace(
            base_settings,
            minimum_excitation=1.0e-12,
            minimum_information_eigenvalue=1.0e-14,
            maximum_information_condition=1.0e14,
        )
    elif policy == "error_only":
        settings = replace(
            base_settings,
            minimum_excitation=1.0e-12,
            minimum_information_eigenvalue=1.0e-14,
            maximum_information_condition=1.0e14,
        )
    elif policy == "error_excitation":
        settings = replace(
            base_settings,
            minimum_information_eigenvalue=1.0e-14,
            maximum_information_condition=1.0e14,
        )
    elif policy == "full":
        settings = base_settings
    else:
        raise ValueError(policy)
    fitter = DistributedParameterFitter(workload.controller, settings, device="cuda")
    monitor = OnlineAdaptationMonitor(fitter.predictor, settings)
    current = ParameterEstimate()
    attempts = accepted = rejected = short_rollouts = 0
    fitting_time_s = 0.0
    trigger_times = []
    for original in observations:
        diagnostic = monitor.append(_with_estimate(original, current))
        if diagnostic is None:
            continue
        should_attempt = (
            True if policy == "continuous" else diagnostic.reason == "fit_candidate"
        )
        if not should_attempt:
            continue
        fitting, validation = select_informative_segments(
            monitor.buffer.candidate_segments(), settings
        )
        if not fitting or not validation:
            continue
        if policy != "continuous" and not monitor.claim_trigger(diagnostic):
            continue
        attempts += 1
        trigger_times.append(diagnostic.timestamp_s)
        result = fitter.fit(fitting, validation, current)
        short_rollouts += result.timing.short_rollouts
        fitting_time_s += result.timing.total_s
        if result.accepted:
            accepted += 1
            current = result.candidate_estimate
        else:
            rejected += 1
    truth_eta = np.log(np.asarray(truth_ratio, dtype=np.float64))
    return {
        "policy": policy,
        "truth_ratio": truth_ratio,
        "fit_attempts": attempts,
        "accepted_fits": accepted,
        "rejected_fits": rejected,
        "short_dder_rollouts": short_rollouts,
        "fitting_time_s": fitting_time_s,
        "trigger_times_s": trigger_times,
        "first_trigger_latency_s": trigger_times[0] if trigger_times else None,
        "final_ei_ratio": current.ei_ratio,
        "final_cb_ratio": current.cb_ratio,
        "final_eta_error_l2": float(np.linalg.norm(current.eta - truth_eta)),
    }


def run_diagnostics_stage(workload: Workload) -> dict[str, object]:
    """Limited dropout/latency, cache, and trigger-component diagnostics."""

    base = DistributedAdaptationSettings()
    chosen_noise = 0.00005
    sensing_base = SyntheticSensingSettings(
        polynomial_degree=1,
        history_length=3,
        position_output="latest_measurement",
        position_noise_std_m=chosen_noise,
        project_inextensible_velocity=True,
    )
    matched_rollout, _truth, _simulation = generate_truth_rollout(workload, 1.0, 1.0)
    calibrated, calibration = calibrate_health_thresholds(
        workload,
        matched_rollout,
        sensing_base,
        base,
        calibration_seeds=(1001, 1002, 1003, 1004, 1005),
    )
    trigger_rows = []
    for truth_ratio in ((1.0, 1.0), (0.8, 0.7)):
        rollout, _truth, _simulation = generate_truth_rollout(workload, *truth_ratio)
        observations, _metrics, _trace = reconstruct_sensed_observations(
            rollout, workload, sensing_base, sensor_seed=17
        )
        for policy in ("continuous", "error_only", "error_excitation", "full"):
            trigger_rows.append(
                run_trigger_policy(
                    workload, observations, calibrated, truth_ratio, policy
                )
            )

    dropout_rows = []
    mismatch_rollout, _truth, _simulation = generate_truth_rollout(workload, 0.8, 0.7)
    for label, variant in (
        ("none", sensing_base),
        ("independent_1_percent", replace(sensing_base, independent_dropout_probability=0.01)),
        ("independent_5_percent", replace(sensing_base, independent_dropout_probability=0.05)),
        (
            "three_frame_all_marker_burst",
            replace(
                sensing_base,
                dropout_burst_start_s=0.28,
                dropout_burst_length_frames=3,
            ),
        ),
    ):
        observations, metrics, _trace = reconstruct_sensed_observations(
            mismatch_rollout, workload, variant, sensor_seed=17
        )
        dropout_rows.append(
            {
                "condition": label,
                "state_metrics": metrics,
                "adaptation": run_adaptation_replay(
                    workload, observations, calibrated, (0.8, 0.7)
                ),
            }
        )

    # Delayed between-strike request: newest FIFO is weak, while the optional
    # cache can retain the earlier reversal.  This ablates selection only.
    extended = generate_extended_truth_rollout(
        workload, (0.8, 0.7), horizon_s=3.0
    )
    extended_observations, _metrics, _trace = reconstruct_sensed_observations(
        extended, workload, sensing_base, sensor_seed=17
    )
    cache_rows = []
    for use_cache in (False, True):
        cache_settings = replace(
            calibrated,
            buffer_duration_s=0.50,
            use_informative_cache=use_cache,
        )
        buffer = RollingDistributedBuffer(cache_settings)
        for observation in extended_observations:
            buffer.append(observation)
        fitting, validation = select_informative_segments(
            buffer.candidate_segments(), cache_settings
        )
        row = {
            "use_informative_cache": use_cache,
            "candidate_segments": len(buffer.candidate_segments()),
            "fit_segments": len(fitting),
            "validation_segments": len(validation),
            "accepted": False,
        }
        if fitting and validation:
            fit = DistributedParameterFitter(
                workload.controller, cache_settings, device="cuda"
            ).fit(fitting, validation, ParameterEstimate())
            row.update(
                {
                    "accepted": fit.accepted,
                    "information": asdict(fit.information),
                    "ei_ratio": fit.candidate_estimate.ei_ratio,
                    "cb_ratio": fit.candidate_estimate.cb_ratio,
                    "validation_loss_before": fit.validation_loss_before,
                    "validation_loss_after": fit.validation_loss_after,
                }
            )
        cache_rows.append(row)

    latency_rows = []
    truth_model = _truth_model(workload, 0.8, 0.7)
    # Reuse the sensing-derived estimate; latency changes controller state age,
    # not the already completed between-strike fit.
    base_obs, _metrics, _trace = reconstruct_sensed_observations(
        mismatch_rollout, workload, sensing_base, sensor_seed=17
    )
    fitted = run_adaptation_replay(
        workload, base_obs, calibrated, (0.8, 0.7)
    )["estimates_by_strike"][-1]
    estimate = ParameterEstimate(
        eta_e=math.log(float(fitted["ei_ratio"])),
        eta_c=math.log(float(fitted["cb_ratio"])),
        generation=1,
        source="latency_diagnostic",
    )
    controller_model = snapshot_for_estimate(workload.controller, estimate)
    for latency_s in (0.0, 0.01, 0.02, 0.03):
        sensing = replace(sensing_base, latency_s=latency_s)
        for seed in (11, 17, 29):
            planner = WhipSimulator(controller_model, workload.simulation, device="cuda")
            plant = WhipSimulator(truth_model, workload.simulation, device="cuda")
            initial = plant.initial_state(workload.initial_xyz)
            provider = SensedControllerStateProvider(
                workload, sensing, initial, sensor_seed=10000 + seed
            )
            execution = run_receding_horizon_mppi(
                planner,
                plant,
                initial,
                workload.problem,
                replace(workload.mppi, seed=seed),
                RecedingMppiSettings(
                    replan_interval_s=0.1, timeout_s=1.2, feedback_mode="full"
                ),
                workload.warm_knots,
                controller_state_provider=provider,
            )
            row = _control_row(execution, f"latency_{1000*latency_s:g}ms", "adapted", seed)
            row.update(
                {
                    "latency_s": latency_s,
                    "mean_measurement_age_s": float(np.mean(provider.measurement_age_s)),
                }
            )
            latency_rows.append(row)
    return {
        "stage": "diagnostics",
        "selected_synthetic_noise_std_m": chosen_noise,
        "health_calibration": calibration,
        "trigger_rows": trigger_rows,
        "dropout_rows": dropout_rows,
        "cache_rows": cache_rows,
        "latency_rows": latency_rows,
    }


def run_representative_control_benchmark(workload: Workload) -> dict[str, object]:
    """Paired ten-seed development benchmark before any full-matrix claim."""

    truth_ratio = (0.8, 0.7)
    truth_model = _truth_model(workload, *truth_ratio)
    truth_rollout, _truth, _simulation = generate_truth_rollout(workload, *truth_ratio)
    matched_rollout, _truth, _simulation = generate_truth_rollout(workload, 1.0, 1.0)
    seeds = (11, 17, 29, 31, 43, 59, 71, 83, 97, 109)
    rows = []
    adaptation_rows = []
    for noise in (0.0, 0.00005):
        sensing = SyntheticSensingSettings(
            polynomial_degree=1,
            history_length=3,
            position_output="latest_measurement",
            position_noise_std_m=noise,
            project_inextensible_velocity=True,
        )
        calibrated, calibration = calibrate_health_thresholds(
            workload,
            matched_rollout,
            sensing,
            DistributedAdaptationSettings(),
            calibration_seeds=(1001, 1002, 1003, 1004, 1005),
        )
        observations, state_metrics, _trace = reconstruct_sensed_observations(
            truth_rollout, workload, sensing, sensor_seed=17
        )
        adaptation = run_adaptation_replay(
            workload, observations, calibrated, truth_ratio
        )
        final = adaptation["estimates_by_strike"][-1]
        estimate = ParameterEstimate(
            eta_e=math.log(float(final["ei_ratio"])),
            eta_c=math.log(float(final["cb_ratio"])),
            generation=1,
            source="representative_sensing_benchmark",
        )
        models = {
            "fixed_nominal": workload.controller,
            "adapted_after_strike_3": snapshot_for_estimate(workload.controller, estimate),
            "oracle": truth_model,
        }
        adaptation_rows.append(
            {
                "noise_std_m": noise,
                "calibration": calibration,
                "state_metrics": state_metrics,
                "adaptation": adaptation,
            }
        )
        for mode in ("mode_a_exact_controller_state", "mode_b_sensed_controller_state"):
            for seed in seeds:
                for label, controller_model in models.items():
                    planner = WhipSimulator(controller_model, workload.simulation, device="cuda")
                    plant = WhipSimulator(truth_model, workload.simulation, device="cuda")
                    initial = plant.initial_state(workload.initial_xyz)
                    provider = (
                        None
                        if mode == "mode_a_exact_controller_state"
                        else SensedControllerStateProvider(
                            workload, sensing, initial, sensor_seed=10000 + seed
                        )
                    )
                    execution = run_receding_horizon_mppi(
                        planner,
                        plant,
                        initial,
                        workload.problem,
                        replace(workload.mppi, seed=seed),
                        RecedingMppiSettings(
                            replan_interval_s=0.1, timeout_s=1.2, feedback_mode="full"
                        ),
                        workload.warm_knots,
                        controller_state_provider=provider,
                    )
                    row = _control_row(
                        execution,
                        f"representative_noise_{1000*noise:g}mm",
                        label,
                        seed,
                    )
                    row.update(
                        {
                            "mode": mode,
                            "noise_std_m": noise,
                            "sensor_seed": 10000 + seed,
                            "truth_ei_ratio": truth_ratio[0],
                            "truth_cb_ratio": truth_ratio[1],
                            "measurement_age_s": (
                                0.0
                                if provider is None
                                else float(np.mean(provider.measurement_age_s))
                            ),
                        }
                    )
                    rows.append(row)
                    print(
                        f"benchmark {mode} noise={1000*noise:.2f}mm "
                        f"{label} seed={seed} success={row['valid_strike']}"
                    )
    return {
        "stage": "representative_control_benchmark",
        "paired_seeds": seeds,
        "truth_ratio": truth_ratio,
        "adaptation_rows": adaptation_rows,
        "control_rows": rows,
    }


def authoritative_configuration(workload: Workload) -> dict[str, object]:
    return {
        "profile": workload.profile_payload,
        "frozen_task": {
            "initial_drone_position_m": workload.initial_xyz,
            "target_position_m": workload.problem.target_position_m,
            "target_radius_m": workload.problem.maximum_tip_error_m,
            "desired_impact_direction": workload.problem.impact_direction,
            "minimum_directed_speed_m_s": workload.problem.minimum_impact_speed_m_s,
            "impact_direction_half_angle_deg": workload.problem.maximum_impact_angle_deg,
            "tip_first_required": True,
            "non_tip_before_tip_is_failure": True,
            "maximum_acceleration_m_s2": workload.simulation.maximum_acceleration_m_s2,
            "maximum_drone_speed_m_s": workload.simulation.maximum_speed_m_s,
            "workspace_limit_enforced_in_mppi": workload.mppi.enforce_workspace_limit,
            "drone_keepout_radius_m": workload.problem.drone_keepout_radius_m,
            "ground_clearance_m": workload.mppi.ground_clearance_m,
            "maximum_altitude_m": workload.mppi.maximum_altitude_m,
            "cable_drone_clearance_m": workload.mppi.cable_drone_clearance_m,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument(
        "--stage",
        choices=(
            "zero-noise",
            "noise",
            "fine-noise",
            "mode-b",
            "diagnostics",
            "control-benchmark",
            "all",
        ),
        default="zero-noise",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workload = load_workload(args.profile)
    started = time.perf_counter()
    results = []
    if args.stage in {"zero-noise", "all"}:
        results.append(run_zero_noise_stage(workload))
    if args.stage in {"noise", "all"}:
        results.append(run_noise_stage(workload))
    if args.stage in {"fine-noise", "all"}:
        results.append(run_fine_noise_boundary_stage(workload))
    if args.stage in {"mode-b", "all"}:
        results.append(run_full_sensed_control_stage(workload))
    if args.stage in {"diagnostics", "all"}:
        results.append(run_diagnostics_stage(workload))
    if args.stage in {"control-benchmark", "all"}:
        results.append(run_representative_control_benchmark(workload))
    payload = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_configuration": authoritative_configuration(workload),
        "results": results,
        "elapsed_wall_s": time.perf_counter() - started,
    }
    output = args.output or DEFAULT_OUTPUT_DIRECTORY / f"{args.stage.replace('-', '_')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_serialize(payload), indent=2), encoding="utf-8")
    print(f"saved {output.resolve()}")


if __name__ == "__main__":
    main()
