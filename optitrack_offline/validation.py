"""Continuous attachment-driven validation of a fitted free-tip cable model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import uuid

import numpy as np
import torch

from cable_twin.shared.dder import (
    DderModel,
    DderParameters,
    DderState,
    START_PINNED_FREE_END,
)
from cable_twin.shared.observation_data import sha256_file

from .data import MotiveCableTake, load_motive_cable_csv
from .config import CableSpecification
from .fitting import (
    marker_positions_to_feasible_rod,
    marker_positions_to_rod,
    MODEL_SCHEMA,
)


RESULT_DIRECTORY = Path(__file__).resolve().parent / "validation_results"
RESULT_SCHEMA = "optitrack_free_tip_rollout_v1"


@dataclass(frozen=True, slots=True)
class ValidationRollout:
    """One held-out simulation with a continuous attachment-position input."""

    take: MotiveCableTake
    start_index: int
    predictions_m: np.ndarray
    per_frame_rmse_m: np.ndarray
    lead_time_s: np.ndarray
    overall_rmse_m: float
    free_tip_rmse_m: float
    observed_dynamic_marker_fraction: float
    history_frames: int
    observation_interval_s: float | None
    correction_mask: np.ndarray
    attachment_hold_mask: np.ndarray
    horizon_rmse_m: dict[str, float | None]
    marker_node_indices: tuple[int, ...]
    result_path: Path

    @property
    def frame_count(self) -> int:
        return int(self.predictions_m.shape[0])

    @property
    def node_count(self) -> int:
        """Number of simulated DER vertices in the fitted model."""

        return int(self.predictions_m.shape[1])

    @property
    def final_frame(self) -> int:
        return self.frame_count - 1

class _RuntimeRolloutStep:
    """Fixed-batch fitted-rod step with CUDA graph replay when available."""

    def __init__(
        self,
        model: DderModel,
        reference: torch.Tensor,
    ) -> None:
        self.model = model
        self.q = reference.clone()
        self.v = torch.zeros_like(reference)
        self.boundary = self.q[:, :1].clone()
        self.dt = torch.full(
            (reference.shape[0],),
            1.0 / 100.0,
            dtype=reference.dtype,
            device=reference.device,
        )
        self.constants = model.runtime_constants(self.q)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: DderState | None = None
        if reference.device.type == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.output = model.step_runtime(
                        DderState(self.q, self.v),
                        self.boundary,
                        self.dt,
                        self.constants,
                        pinned_endpoints=START_PINNED_FREE_END,
                    )
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = model.step_runtime(
                    DderState(self.q, self.v),
                    self.boundary,
                    self.dt,
                    self.constants,
                    pinned_endpoints=START_PINNED_FREE_END,
                )

    def __call__(
        self,
        state: DderState,
        boundary: torch.Tensor,
        dt: torch.Tensor,
    ) -> DderState:
        self.q.copy_(state.positions_m)
        self.v.copy_(state.velocities_m_s)
        self.boundary.copy_(boundary)
        self.dt.copy_(dt)
        if self.graph is None:
            return self.model.step_runtime(
                DderState(self.q, self.v),
                self.boundary,
                self.dt,
                self.constants,
                pinned_endpoints=START_PINNED_FREE_END,
            )
        self.graph.replay()
        assert self.output is not None
        return self.output


def _load_model_payload(path: str | Path) -> tuple[Path, dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Fit a cable model first: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != MODEL_SCHEMA:
        raise ValueError("The selected file is not a direct OptiTrack cable model.")
    return source, payload


def _artifact_cable(payload: dict[str, object]) -> CableSpecification:
    measured = payload.get("measured")
    if not isinstance(measured, dict):
        raise ValueError("Fitted cable artifact has no measured specification.")
    return CableSpecification(
        rest_lengths_m=tuple(
            float(value) for value in measured["marker_interval_lengths_m"]
        ),
        bare_cable_mass_kg=float(measured["bare_cable_mass_kg"]),
        moving_marker_masses_kg=tuple(
            float(value) for value in measured["moving_marker_masses_kg"]
        ),
        diameter_m=float(measured["diameter_m"]),
        rod_segments_per_marker_interval=int(
            measured["rod_segments_per_marker_interval"]
        ),
    )


def load_fitted_optitrack_model(path: str | Path) -> DderModel:
    source, payload = _load_model_payload(path)
    measured = payload.get("measured")
    optimized = payload.get("optimized")
    solver = payload.get("solver")
    if not all(isinstance(value, dict) for value in (measured, optimized, solver)):
        raise ValueError(f"Fitted cable artifact is incomplete: {source}")
    assert isinstance(measured, dict)
    assert isinstance(optimized, dict)
    assert isinstance(solver, dict)
    rest_lengths = tuple(float(value) for value in measured["rest_lengths_m"])
    vertex_masses = tuple(float(value) for value in measured["vertex_masses_kg"])
    node_count = int(measured["node_count"])
    marker_count = int(measured["marker_count"])
    marker_node_indices = tuple(
        int(value) for value in measured["marker_node_indices"]
    )
    if node_count != len(rest_lengths) + 1:
        raise ValueError("The fitted model has inconsistent DER edge lengths.")
    if (
        len(marker_node_indices) != marker_count
        or marker_node_indices[0] != 0
        or marker_node_indices[-1] != node_count - 1
        or any(
            following <= previous
            for previous, following in zip(
                marker_node_indices[:-1], marker_node_indices[1:]
            )
        )
    ):
        raise ValueError("The fitted model has an invalid marker observation map.")
    if len(vertex_masses) != node_count:
        raise ValueError("The fitted model has inconsistent vertex masses.")
    gravity = tuple(float(value) for value in solver["gravity_m_s2"])
    return DderModel(
        DderParameters(
            node_count=node_count,
            cable_length_m=float(measured["length_m"]),
            cable_mass_kg=float(measured["total_dynamic_mass_kg"]),
            cable_diameter_m=float(measured["diameter_m"]),
            bending_stiffness_n_m2=float(optimized["bending_stiffness_n_m2"]),
            bending_damping_n_m2_s=float(optimized["bending_damping_n_m2_s"]),
            torsional_stiffness_n_m2=0.0,
            external_drag_s_inv=float(optimized.get("external_drag_s_inv", 0.0)),
            gravity_camera_m_s2=gravity,  # type: ignore[arg-type]
            rest_lengths_m=rest_lengths,
            vertex_masses_kg=vertex_masses,
            substeps=int(solver["substeps"]),
            constraint_iterations=int(solver["constraint_iterations"]),
        )
    )


def _causal_velocity(
    positions_m: np.ndarray,
    timestamps_s: np.ndarray,
    current_index: int,
    history_frames: int,
) -> np.ndarray:
    """Estimate current velocity from only the current and preceding frames."""

    selected = np.arange(
        current_index - history_frames + 1,
        current_index + 1,
        dtype=np.int64,
    )
    relative_time = timestamps_s[selected] - timestamps_s[current_index]
    design = np.column_stack(
        (np.ones(history_frames), relative_time, np.square(relative_time))
    )
    targets = positions_m[selected].reshape(history_frames, -1)
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return coefficients[1].reshape(positions_m.shape[1:])


def _first_complete_history(take: MotiveCableTake, history_frames: int) -> int:
    if not take.has_attachment:
        raise ValueError(f"{take.source_path.name} has no rigid attachment pivot.")
    complete = np.all(take.observed, axis=1) & take.attachment_observed
    for current in range(history_frames - 1, take.frame_count - 1):
        if bool(np.all(complete[current - history_frames + 1 : current + 1])):
            return current
    raise ValueError(
        f"{take.source_path.name} has no {history_frames}-frame complete causal "
        "initialization followed by a future attachment trajectory."
    )


def _held_attachment_inputs(
    take: MotiveCableTake,
    start_index: int,
    quality: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return attachment positions with zero-order hold through bad observations."""

    dt = np.diff(take.timestamps_s[start_index:])
    maximum_dt_s = 1.5 / take.export_rate_hz
    invalid_dt = (dt <= 0.0) | (dt > maximum_dt_s)
    if bool(np.any(invalid_dt)):
        offset = int(np.flatnonzero(invalid_dt)[0])
        index = start_index + offset + 1
        raise ValueError(
            f"Continuous simulation has an interrupted timestamp at frame "
            f"{int(take.frame_numbers[index])}."
        )

    raw_positions = take.positions_m[start_index:, :1].copy()
    observed = take.observed[start_index:, :1].copy()
    observed[:, 0] &= take.attachment_observed[start_index:]
    observed[:, 0] &= take.attachment_error_m[start_index:] <= quality[
        "maximum_rigid_body_error_m"
    ]
    observed &= np.all(np.isfinite(raw_positions), axis=2)
    if not bool(observed[0, 0]):
        raise ValueError("Simulation must start from a trustworthy attachment position.")

    positions = np.empty_like(raw_positions)
    held = np.zeros(observed.shape, dtype=bool)
    positions[0] = raw_positions[0]
    for local in range(1, len(positions)):
        positions[local] = positions[local - 1]
        accepted = False
        if observed[local, 0]:
            # The input was explicitly held during every missing frame.  Test a
            # reacquired pivot against the current one-frame interval, not the
            # elapsed occlusion duration; otherwise a large catch-up jump would
            # be injected into the rod as an artificial boundary velocity.
            elapsed = dt[local - 1]
            displacement = np.linalg.norm(raw_positions[local, 0] - positions[local - 1, 0])
            if displacement / elapsed <= quality["maximum_attachment_speed_m_s"]:
                positions[local, 0] = raw_positions[local, 0]
                accepted = True
        held[local, 0] = not accepted
    for value in (positions, held):
        value.setflags(write=False)
    return positions, held


def _masked_error(
    prediction_m: np.ndarray,
    take: MotiveCableTake,
    start_index: int,
    marker_node_indices: tuple[int, ...],
    comparison_observed: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, float]:
    truth = take.positions_m[start_index:, 1:]
    observed = (
        take.observed[start_index:, 1:]
        if comparison_observed is None
        else comparison_observed
    )
    predicted_markers = prediction_m[:, marker_node_indices]
    squared = np.sum(np.square(predicted_markers[:, 1:] - truth), axis=2)
    count_per_frame = np.count_nonzero(observed, axis=1)
    per_frame = np.full(len(prediction_m), np.nan, dtype=np.float64)
    comparable = count_per_frame > 0
    per_frame[comparable] = np.sqrt(
        np.sum(np.where(observed, squared, 0.0), axis=1)[comparable]
        / count_per_frame[comparable]
    )
    observed_count = int(np.count_nonzero(observed))
    if observed_count == 0:
        raise ValueError("The selected take has no dynamic-marker observations for comparison.")
    overall = float(
        np.sqrt(np.sum(np.where(observed, squared, 0.0)) / observed_count)
    )
    tip_observed = observed[:, -1]
    if not bool(np.any(tip_observed)):
        raise ValueError("The selected take has no free-tip observations for comparison.")
    tip_rmse = float(np.sqrt(np.mean(squared[tip_observed, -1])))
    return per_frame, overall, observed_count / float(observed.size), tip_rmse


def _comparison_visibility(
    take: MotiveCableTake,
    start_index: int,
    cable: CableSpecification,
    quality: dict[str, float],
) -> np.ndarray:
    """Mask corrupt moving-marker truth without feeding it to simulation."""

    observed = take.observed[start_index:, 1:].copy()
    positions = take.positions_m[start_index:]
    timestamps = take.timestamps_s[start_index:]
    dt = np.diff(timestamps)
    for marker in range(1, take.marker_count):
        pair_observed = (
            take.observed[start_index:-1, marker]
            & take.observed[start_index + 1 :, marker]
        )
        displacement = np.linalg.vector_norm(
            np.diff(positions[:, marker], axis=0), axis=1
        )
        bad = pair_observed & (
            displacement / dt > quality["maximum_marker_speed_m_s"]
        )
        indices = np.flatnonzero(bad)
        observed[indices, marker - 1] = False
        observed[indices + 1, marker - 1] = False
    rest = np.asarray(cable.rest_lengths_m, dtype=np.float64)
    for edge in range(take.marker_count - 1):
        pair_observed = (
            take.observed[start_index:, edge]
            & take.observed[start_index:, edge + 1]
        )
        chord = np.linalg.vector_norm(
            positions[:, edge + 1] - positions[:, edge], axis=1
        )
        bad = pair_observed & (
            chord > rest[edge] + quality["maximum_chord_excess_m"]
        )
        if 0 < edge < take.marker_count:
            observed[bad, edge - 1] = False
        if 0 < edge + 1 < take.marker_count:
            observed[bad, edge] = False
    observed.setflags(write=False)
    return observed


def _scheduled_correction_indices(
    timestamps_s: np.ndarray,
    start_index: int,
    observation_interval_s: float | None,
) -> np.ndarray:
    """Return exact scheduled full-cable observation frames after initialization."""

    if observation_interval_s is None:
        return np.empty(0, dtype=np.int64)
    if observation_interval_s <= 0.0 or not math.isfinite(observation_interval_s):
        raise ValueError("Observation interval must be positive or attachment-only.")
    relative = timestamps_s[start_index:] - timestamps_s[start_index]
    duration = float(relative[-1])
    requested = np.arange(
        observation_interval_s,
        duration + 1.0e-12,
        observation_interval_s,
        dtype=np.float64,
    )
    if requested.size == 0:
        return np.empty(0, dtype=np.int64)
    local = np.searchsorted(relative, requested, side="left")
    local = np.clip(local, 1, len(relative) - 1)
    return np.unique(local.astype(np.int64))


def _periodic_correction_state(
    model: DderModel,
    state: DderState,
    take: MotiveCableTake,
    absolute_index: int,
    cable: CableSpecification,
    quality: dict[str, float],
    device: torch.device,
    attachment_velocity: torch.Tensor,
) -> DderState:
    """Assimilate a full position observation while retaining predicted velocity."""

    if not bool(np.all(take.observed[absolute_index])):
        raise ValueError(
            "Scheduled full-cable observation is incomplete at frame "
            f"{int(take.frame_numbers[absolute_index])}."
        )
    feasible, correction = marker_positions_to_feasible_rod(
        take.positions_m[absolute_index : absolute_index + 1], cable
    )
    correction_rmse = float(correction[0])
    if correction_rmse > quality["maximum_initialization_rmse_m"]:
        raise ValueError(
            "Scheduled full-cable observation requires "
            f"{1000.0 * correction_rmse:.3f} mm conditioning at frame "
            f"{int(take.frame_numbers[absolute_index])}."
        )
    positions = torch.as_tensor(feasible, dtype=torch.float64, device=device)
    velocity = model.project_velocities(
        positions,
        state.velocities_m_s,
        attachment_velocity,
        pinned_endpoints=START_PINNED_FREE_END,
    )
    return DderState(positions, velocity)


def _horizon_errors(
    per_frame_rmse_m: np.ndarray,
    correction_mask: np.ndarray,
    timestamps_s: np.ndarray,
    horizons_s: tuple[float, ...] = (0.1, 0.5, 1.0),
) -> dict[str, float | None]:
    """Aggregate error at fixed time since the last full-cable observation."""

    starts = np.flatnonzero(correction_mask)
    stops = np.r_[starts[1:], len(correction_mask)]
    result: dict[str, float | None] = {}
    for horizon in horizons_s:
        samples: list[float] = []
        for start, stop in zip(starts, stops):
            target = timestamps_s[start] + horizon
            index = int(np.searchsorted(timestamps_s, target, side="left"))
            if index >= stop:
                continue
            value = float(per_frame_rmse_m[index])
            if math.isfinite(value):
                samples.append(value)
        result[f"{horizon:g}_s"] = (
            None
            if not samples
            else float(np.sqrt(np.mean(np.square(samples))))
        )
    return result


def evaluate_continuous_validation_take(
    take_path: str | Path,
    model_path: str | Path,
    *,
    history_frames: int = 5,
    observation_interval_s: float | None = None,
    device: str = "cuda",
    result_path: str | Path | None = None,
) -> ValidationRollout:
    """Simulate from one causal initialization using only attachment motion."""

    if history_frames < 3:
        raise ValueError("Causal velocity history must contain at least three frames.")
    compute_device = torch.device(device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the validation configuration.")

    take = load_motive_cable_csv(take_path)
    resolved_model_path, payload = _load_model_payload(model_path)
    take_hash = sha256_file(take.source_path)
    if any(
        source.get("sha256") == take_hash
        and source.get("role") == "training"
        for source in payload.get("sources", [])
    ):
        raise ValueError(
            f"{take.source_path.name} was used to fit this model and "
            "is not held out. "
            "Choose a different take for simulation testing."
        )
    model = load_fitted_optitrack_model(resolved_model_path)
    cable = _artifact_cable(payload)
    quality_payload = payload.get("measurement_quality")
    required_quality = (
        "maximum_rigid_body_error_m",
        "maximum_attachment_speed_m_s",
        "maximum_marker_speed_m_s",
        "maximum_chord_excess_m",
        "maximum_initialization_rmse_m",
    )
    if not isinstance(quality_payload, dict) or any(
        key not in quality_payload for key in required_quality
    ):
        raise ValueError(
            "The fitted model lacks the measurement-quality contract; refit it "
            "before continuous validation."
        )
    quality = {key: float(quality_payload[key]) for key in required_quality}
    if take.marker_count != cable.marker_count:
        raise ValueError(
            f"{take.source_path.name} has {take.marker_count} markers but the fitted "
            f"model requires {cable.marker_count}."
        )
    start_index = _first_complete_history(take, history_frames)
    attachment_positions, attachment_hold_mask = _held_attachment_inputs(
        take,
        start_index,
        quality,
    )
    comparison_observed = _comparison_visibility(
        take,
        start_index,
        cable,
        quality,
    )

    feasible_initial, initial_correction = marker_positions_to_feasible_rod(
        take.positions_m[start_index : start_index + 1], cable
    )
    initialization_rmse_m = float(initial_correction[0])
    if initialization_rmse_m > quality["maximum_initialization_rmse_m"]:
        raise ValueError(
            "Continuous initialization requires "
            f"{1000.0 * initialization_rmse_m:.3f} mm marker conditioning, above "
            f"the fitted {1000.0 * quality['maximum_initialization_rmse_m']:.3f} mm limit."
        )
    positions = torch.tensor(
        feasible_initial,
        dtype=torch.float64,
        device=compute_device,
    )
    maximum_length_error = float(
        model.maximum_segment_error_m(positions).max().detach().cpu()
    )
    if maximum_length_error > 1.0e-8:
        raise RuntimeError(
            "Marker-conditioned continuous initialization violates cable length."
        )
    history_start = start_index - history_frames + 1
    history_rods, history_correction = marker_positions_to_feasible_rod(
        take.positions_m[history_start : start_index + 1], cable
    )
    if float(np.max(history_correction)) > quality["maximum_initialization_rmse_m"]:
        raise ValueError("Causal initialization history contains inconsistent marker geometry.")
    initial_velocity = torch.as_tensor(
        _causal_velocity(
            history_rods,
            take.timestamps_s[history_start : start_index + 1],
            history_frames - 1,
            history_frames,
        )[None],
        dtype=torch.float64,
        device=compute_device,
    )
    velocity = model.project_velocities(
        positions,
        initial_velocity,
        initial_velocity[:, :1],
        pinned_endpoints=START_PINNED_FREE_END,
    )

    attachment_sequence = torch.tensor(
        attachment_positions,
        dtype=torch.float64,
        device=compute_device,
    )
    timestamp_sequence = torch.tensor(
        take.timestamps_s[start_index:],
        dtype=torch.float64,
        device=compute_device,
    )
    maximum_dt_s = float(
        torch.max(timestamp_sequence[1:] - timestamp_sequence[:-1]).cpu()
    )
    if model.parameters.bending_stiffness_n_m2 > model.maximum_stable_bending_stiffness(
        maximum_dt_s,
        pinned_endpoints=START_PINNED_FREE_END,
    ):
        raise ValueError("The fitted EI is unstable at this take's sampling interval.")
    state = DderState(positions, velocity)
    predictions = [positions[0].clone()]
    correction_mask = np.zeros(len(timestamp_sequence), dtype=bool)
    correction_mask[0] = True
    scheduled_corrections = _scheduled_correction_indices(
        take.timestamps_s,
        start_index,
        observation_interval_s,
    )
    correction_set = set(int(value) for value in scheduled_corrections)
    step = _RuntimeRolloutStep(model, positions)
    for frame in range(1, len(timestamp_sequence)):
        state = step(
            state,
            attachment_sequence[frame : frame + 1],
            (timestamp_sequence[frame] - timestamp_sequence[frame - 1])[None],
        )
        absolute_index = start_index + frame
        correction_available = (
            bool(np.all(take.observed[absolute_index]))
            and bool(take.attachment_observed[absolute_index])
            and not bool(attachment_hold_mask[frame, 0])
            and bool(np.all(comparison_observed[frame]))
        )
        if frame in correction_set and correction_available:
            _, conditioning_rmse = marker_positions_to_feasible_rod(
                take.positions_m[absolute_index : absolute_index + 1], cable
            )
            correction_available = bool(
                float(np.max(conditioning_rmse))
                <= quality["maximum_initialization_rmse_m"]
            )
        if frame in correction_set and correction_available:
            attachment_velocity = (
                attachment_sequence[frame : frame + 1]
                - attachment_sequence[frame - 1 : frame]
            ) / (timestamp_sequence[frame] - timestamp_sequence[frame - 1])
            state = _periodic_correction_state(
                model,
                state,
                take,
                absolute_index,
                cable,
                quality,
                compute_device,
                attachment_velocity,
            )
            correction_mask[frame] = True
        predictions.append(state.positions_m[0].clone())
    prediction = torch.stack(predictions).detach().cpu().numpy()
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("Continuous attachment-driven simulation became non-finite.")
    per_frame, overall, observed_fraction, free_tip_rmse = _masked_error(
        prediction,
        take,
        start_index,
        cable.marker_node_indices,
        comparison_observed,
    )
    lead_time = take.timestamps_s[start_index:] - take.timestamps_s[start_index]
    horizon_rmse = _horizon_errors(
        per_frame,
        correction_mask,
        lead_time,
    )
    for value in (
        prediction,
        per_frame,
        lead_time,
        correction_mask,
        attachment_hold_mask,
    ):
        value.setflags(write=False)

    destination = (
        Path(result_path).expanduser().resolve()
        if result_path is not None
        else (
            RESULT_DIRECTORY
            / (
                f"{take.source_path.stem}_attachment_only.json"
                if observation_interval_s is None
                else (
                    f"{take.source_path.stem}_observe_every_"
                    f"{observation_interval_s:g}s.json"
                )
            )
        ).resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "schema": RESULT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "take": {
            "path": str(take.source_path),
            "sha256": take_hash,
            "name": take.take_name,
            "export_rate_hz": take.export_rate_hz,
        },
        "model": {
            "path": str(resolved_model_path),
            "sha256": sha256_file(resolved_model_path),
        },
        "protocol": {
            "history_frames": history_frames,
            "initial_frame_number": int(take.frame_numbers[start_index]),
            "simulated_frames": len(prediction),
            "simulated_duration_s": float(lead_time[-1]),
            "observation_mode": (
                "attachment_only"
                if observation_interval_s is None
                else "periodic_full_cable_position"
            ),
            "full_cable_observation_interval_s": observation_interval_s,
            "initial_state": (
                "marker-conditioned refined rods from the causal history and a "
                "past-only quadratic nodal velocity estimate"
            ),
            "der_node_count": cable.node_count,
            "marker_node_indices": list(cable.marker_node_indices),
            "initialization_projection_rmse_m": initialization_rmse_m,
            "boundary_input": (
                "recorded attachment rigid-body pivot position only; an unavailable "
                "or implausible pivot is held at its last trustworthy value"
            ),
            "attachment_occlusion_policy": "zero_order_hold_last_trustworthy_position",
            "attachment_held_frames": int(np.count_nonzero(attachment_hold_mask)),
            "future_dynamic_marker_observations_used_by_simulation": (
                bool(np.count_nonzero(correction_mask) > 1)
            ),
            "velocity_update_at_observation": "retain_and_project_predicted_velocity",
            "uses_future_measurements_for_velocity": False,
            "full_state_resets_after_initialization": 0,
            "periodic_observation_update": (
                "predict first, then apply a marker-conditioned full position "
                "update; predicted velocity is retained and projected onto the "
                "corrected rod"
            ),
            "state_correction_frame_numbers": [
                int(take.frame_numbers[start_index + index])
                for index in np.flatnonzero(correction_mask)[1:]
            ],
            "state_corrections_after_initialization": int(
                np.count_nonzero(correction_mask) - 1
            ),
            "measurement_quality_contract": quality,
        },
        "metrics": {
            "overall_visible_dynamic_marker_rmse_m": overall,
            "free_tip_rmse_m": free_tip_rmse,
            "per_frame_visible_dynamic_marker_rmse_m": [
                None if not math.isfinite(value) else float(value) for value in per_frame
            ],
            "lead_time_s": lead_time.tolist(),
            "observed_dynamic_marker_fraction": observed_fraction,
            "rmse_at_time_since_observation_m": horizon_rmse,
        },
    }
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return ValidationRollout(
        take=take,
        start_index=start_index,
        predictions_m=prediction,
        per_frame_rmse_m=per_frame,
        lead_time_s=lead_time,
        overall_rmse_m=overall,
        free_tip_rmse_m=free_tip_rmse,
        observed_dynamic_marker_fraction=observed_fraction,
        history_frames=history_frames,
        observation_interval_s=observation_interval_s,
        correction_mask=correction_mask,
        attachment_hold_mask=attachment_hold_mask,
        horizon_rmse_m=horizon_rmse,
        marker_node_indices=cable.marker_node_indices,
        result_path=destination,
    )
