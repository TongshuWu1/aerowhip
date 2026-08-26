"""OptiTrack-compatible cable observations and causal state reconstruction.

The production-facing contract intentionally contains positions only.  A
simulator may use its private truth trajectory to synthesize these
measurements, but neither the controller nor the physical adapter receives a
truth velocity or an unobserved cable state through this interface.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Literal, Protocol, Sequence

import numpy as np


def _finite_array(
    value: np.ndarray | Sequence[float], *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}.")
    array.setflags(write=False)
    return array


def _boolean_array(
    value: np.ndarray | Sequence[bool], *, name: str, shape: tuple[int, ...]
) -> np.ndarray:
    array = np.array(value, dtype=bool, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CableObservation:
    """One position-only observation matching the planned Motive payload.

    Node zero is represented by ``root_position_m``.  ``marker_positions_m``
    contains exactly c1...c10 in material order; c10 is the free tip.
    """

    sample_timestamp_s: float
    arrival_timestamp_s: float
    sequence_number: int
    root_position_m: np.ndarray
    marker_positions_m: np.ndarray
    marker_validity_mask: np.ndarray

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.sample_timestamp_s)
            or not math.isfinite(self.arrival_timestamp_s)
            or self.arrival_timestamp_s + 1.0e-12 < self.sample_timestamp_s
        ):
            raise ValueError(
                "Observation timestamps must be finite and arrival cannot precede sampling."
            )
        if self.sequence_number < 0:
            raise ValueError("Observation sequence number cannot be negative.")
        root = _finite_array(
            self.root_position_m, name="root_position_m", shape=(3,)
        )
        markers = _finite_array(
            self.marker_positions_m, name="marker_positions_m", shape=(10, 3)
        )
        validity = _boolean_array(
            self.marker_validity_mask,
            name="marker_validity_mask",
            shape=(10,),
        )
        object.__setattr__(self, "root_position_m", root)
        object.__setattr__(self, "marker_positions_m", markers)
        object.__setattr__(self, "marker_validity_mask", validity)

    @property
    def measurement_age_at_arrival_s(self) -> float:
        return self.arrival_timestamp_s - self.sample_timestamp_s


class CableObservationSource(Protocol):
    """Common source interface for simulation now and Motive later."""

    def observations_arrived_by(self, time_s: float) -> tuple[CableObservation, ...]:
        """Return each not-yet-delivered sample whose arrival time is <= time_s."""


@dataclass(frozen=True, slots=True)
class EstimatedCableState:
    """Causal distributed state produced only from :class:`CableObservation`."""

    sample_timestamp_s: float
    arrival_timestamp_s: float
    sequence_number: int
    cable_positions_m: np.ndarray
    cable_velocities_m_s: np.ndarray
    measurement_validity_mask: np.ndarray
    state_validity_mask: np.ndarray
    imputed_mask: np.ndarray
    history_counts: np.ndarray
    estimator_compute_time_s: float

    def __post_init__(self) -> None:
        positions = _finite_array(
            self.cable_positions_m, name="cable_positions_m", shape=(11, 3)
        )
        velocities = _finite_array(
            self.cable_velocities_m_s,
            name="cable_velocities_m_s",
            shape=(11, 3),
        )
        measurement = _boolean_array(
            self.measurement_validity_mask,
            name="measurement_validity_mask",
            shape=(11,),
        )
        state = _boolean_array(
            self.state_validity_mask, name="state_validity_mask", shape=(11,)
        )
        imputed = _boolean_array(
            self.imputed_mask, name="imputed_mask", shape=(11,)
        )
        counts = np.array(self.history_counts, dtype=np.int64, copy=True)
        if counts.shape != (11,) or np.any(counts < 0):
            raise ValueError("history_counts must be 11 non-negative integers.")
        counts.setflags(write=False)
        if (
            not math.isfinite(self.sample_timestamp_s)
            or not math.isfinite(self.arrival_timestamp_s)
            or not math.isfinite(self.estimator_compute_time_s)
            or self.estimator_compute_time_s < 0.0
        ):
            raise ValueError("Estimated-state timestamps and timing must be finite.")
        if np.any(imputed & measurement):
            raise ValueError("A genuinely measured node cannot also be marked imputed.")
        object.__setattr__(self, "cable_positions_m", positions)
        object.__setattr__(self, "cable_velocities_m_s", velocities)
        object.__setattr__(self, "measurement_validity_mask", measurement)
        object.__setattr__(self, "state_validity_mask", state)
        object.__setattr__(self, "imputed_mask", imputed)
        object.__setattr__(self, "history_counts", counts)

    @property
    def measurement_age_s(self) -> float:
        return self.arrival_timestamp_s - self.sample_timestamp_s


class CausalCableStateEstimator:
    """One-sided local-polynomial position and velocity estimator.

    A degree-two fit is used once three samples are available.  Two samples
    reduce to a first-order fit and one sample uses zero/previous velocity.
    Every fit uses timestamps and measurements already received; no centered
    filtering or future sample is possible through this API.
    """

    def __init__(
        self,
        *,
        polynomial_degree: int = 2,
        history_length: int = 7,
        position_output: Literal["polynomial", "latest_measurement"] = "polynomial",
        velocity_projection: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if polynomial_degree not in (1, 2):
            raise ValueError("Causal polynomial degree must be one or two.")
        if history_length < polynomial_degree + 1:
            raise ValueError("History must support the requested polynomial degree.")
        if position_output not in {"polynomial", "latest_measurement"}:
            raise ValueError("Unknown causal position-output mode.")
        self.polynomial_degree = polynomial_degree
        self.history_length = history_length
        self.position_output = position_output
        self.velocity_projection = velocity_projection
        self._history: list[deque[tuple[float, np.ndarray]]] = [
            deque(maxlen=history_length) for _ in range(11)
        ]
        self._last_velocity = np.zeros((11, 3), dtype=np.float64)
        self._last_sequence = -1
        self._last_arrival = -math.inf

    @staticmethod
    def _fit_history(
        history: deque[tuple[float, np.ndarray]],
        query_time_s: float,
        maximum_degree: int,
        fallback_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        if not history:
            return np.zeros(3), fallback_velocity.copy(), False
        times = np.asarray([item[0] for item in history], dtype=np.float64)
        values = np.stack([item[1] for item in history])
        degree = min(maximum_degree, len(history) - 1)
        if degree <= 0:
            dt = query_time_s - times[-1]
            return values[-1] + dt * fallback_velocity, fallback_velocity.copy(), True
        tau = times - query_time_s
        design = np.stack([tau**power for power in range(degree + 1)], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        position = coefficients[0]
        velocity = coefficients[1]
        return position, velocity, True

    def update(self, observation: CableObservation) -> EstimatedCableState:
        import time

        started = time.perf_counter()
        if observation.sequence_number <= self._last_sequence:
            raise ValueError("Cable observations must have increasing sequence numbers.")
        if observation.arrival_timestamp_s + 1.0e-12 < self._last_arrival:
            raise ValueError("Cable observations must be processed in arrival order.")
        self._last_sequence = observation.sequence_number
        self._last_arrival = observation.arrival_timestamp_s
        measured_positions = np.vstack(
            (observation.root_position_m[None], observation.marker_positions_m)
        )
        measured_mask = np.concatenate(
            (np.ones(1, dtype=bool), observation.marker_validity_mask)
        )
        for node in range(11):
            if measured_mask[node]:
                self._history[node].append(
                    (observation.sample_timestamp_s, measured_positions[node].copy())
                )
        positions = np.zeros((11, 3), dtype=np.float64)
        velocities = np.zeros_like(positions)
        state_validity = np.zeros(11, dtype=bool)
        counts = np.zeros(11, dtype=np.int64)
        for node, history in enumerate(self._history):
            counts[node] = len(history)
            position, velocity, valid = self._fit_history(
                history,
                observation.sample_timestamp_s,
                self.polynomial_degree,
                self._last_velocity[node],
            )
            positions[node] = position
            velocities[node] = velocity
            state_validity[node] = valid
            if measured_mask[node] and self.position_output == "latest_measurement":
                positions[node] = measured_positions[node]
        self._last_velocity[state_validity] = velocities[state_validity]
        if self.velocity_projection is not None and np.all(state_validity):
            projected = np.asarray(
                self.velocity_projection(positions, velocities), dtype=np.float64
            )
            if projected.shape != velocities.shape or not np.all(np.isfinite(projected)):
                raise ValueError("Velocity projection returned an invalid cable state.")
            velocities = projected
            self._last_velocity[:] = velocities
        return EstimatedCableState(
            sample_timestamp_s=observation.sample_timestamp_s,
            arrival_timestamp_s=observation.arrival_timestamp_s,
            sequence_number=observation.sequence_number,
            cable_positions_m=positions,
            cable_velocities_m_s=velocities,
            measurement_validity_mask=measured_mask,
            state_validity_mask=state_validity,
            imputed_mask=~measured_mask & state_validity,
            history_counts=counts,
            estimator_compute_time_s=time.perf_counter() - started,
        )


class DderInextensibilityVelocityProjector:
    """Apply only the known DDER chain velocity constraint to a sensed state.

    The operation depends on positions, vertex masses, and the attached-root
    boundary velocity.  It does not depend on EI, Cb, a simulator truth state,
    or a future observation.
    """

    def __init__(self, cable_snapshot) -> None:
        if cable_snapshot.node_count != 11:
            raise ValueError("The OptiTrack cable projector expects 11 nodes.")
        self._model = cable_snapshot.model

    def __call__(self, positions_m: np.ndarray, velocities_m_s: np.ndarray) -> np.ndarray:
        import torch
        from cable_twin.shared.dder import START_PINNED_FREE_END

        positions = torch.as_tensor(positions_m[None], dtype=torch.float64)
        velocities = torch.as_tensor(velocities_m_s[None], dtype=torch.float64)
        projected = self._model.project_velocities(
            positions,
            velocities,
            velocities[:, :1],
            pinned_endpoints=START_PINNED_FREE_END,
        )
        return projected[0].detach().cpu().numpy()


class SimulatedOptiTrackSource:
    """Position-only sampled observation source backed by a private truth path."""

    def __init__(
        self,
        truth_time_s: np.ndarray,
        root_positions_m: np.ndarray,
        cable_positions_m: np.ndarray,
        *,
        measurement_rate_hz: float = 100.0,
        position_noise_std_m: float = 0.0,
        latency_s: float = 0.0,
        random_seed: int = 0,
        independent_dropout_probability: float = 0.0,
        forced_invalid_mask: np.ndarray | None = None,
    ) -> None:
        time_values = np.asarray(truth_time_s, dtype=np.float64)
        root = np.asarray(root_positions_m, dtype=np.float64)
        cable = np.asarray(cable_positions_m, dtype=np.float64)
        if (
            time_values.ndim != 1
            or len(time_values) < 2
            or np.any(np.diff(time_values) <= 0.0)
            or root.shape != (len(time_values), 3)
            or cable.shape != (len(time_values), 11, 3)
            or not np.all(np.isfinite(root))
            or not np.all(np.isfinite(cable))
        ):
            raise ValueError("Synthetic OptiTrack truth paths have incompatible shapes.")
        if not math.isfinite(measurement_rate_hz) or measurement_rate_hz <= 0.0:
            raise ValueError("Measurement rate must be positive.")
        if not math.isfinite(position_noise_std_m) or position_noise_std_m < 0.0:
            raise ValueError("Position-noise standard deviation must be non-negative.")
        if not math.isfinite(latency_s) or latency_s < 0.0:
            raise ValueError("Synthetic sensing latency must be non-negative.")
        if not 0.0 <= independent_dropout_probability < 1.0:
            raise ValueError("Dropout probability must lie in [0, 1).")
        dt = 1.0 / measurement_rate_hz
        count = int(math.floor((time_values[-1] - time_values[0]) / dt + 1.0e-9)) + 1
        sample_times = time_values[0] + dt * np.arange(count, dtype=np.float64)
        sample_times = sample_times[sample_times <= time_values[-1] + 1.0e-9]
        root_samples = np.stack(
            [np.interp(sample_times, time_values, root[:, axis]) for axis in range(3)],
            axis=1,
        )
        marker_samples = np.empty((len(sample_times), 10, 3), dtype=np.float64)
        for marker in range(10):
            for axis in range(3):
                marker_samples[:, marker, axis] = np.interp(
                    sample_times, time_values, cable[:, marker + 1, axis]
                )
        rng = np.random.default_rng(random_seed)
        if position_noise_std_m > 0.0:
            root_samples += rng.normal(0.0, position_noise_std_m, root_samples.shape)
            marker_samples += rng.normal(
                0.0, position_noise_std_m, marker_samples.shape
            )
        validity = rng.random((len(sample_times), 10)) >= independent_dropout_probability
        if forced_invalid_mask is not None:
            forced = np.asarray(forced_invalid_mask, dtype=bool)
            if forced.shape != validity.shape:
                raise ValueError("forced_invalid_mask must match sampled frames x 10 markers.")
            validity &= ~forced
        self._observations = tuple(
            CableObservation(
                sample_timestamp_s=float(sample_time),
                arrival_timestamp_s=float(sample_time + latency_s),
                sequence_number=index,
                root_position_m=root_samples[index],
                marker_positions_m=marker_samples[index],
                marker_validity_mask=validity[index],
            )
            for index, sample_time in enumerate(sample_times)
        )
        self._next_index = 0

    @property
    def all_observations(self) -> tuple[CableObservation, ...]:
        return self._observations

    def observations_arrived_by(self, time_s: float) -> tuple[CableObservation, ...]:
        if not math.isfinite(time_s):
            raise ValueError("Controller time must be finite.")
        start = self._next_index
        while (
            self._next_index < len(self._observations)
            and self._observations[self._next_index].arrival_timestamp_s
            <= time_s + 1.0e-12
        ):
            self._next_index += 1
        return self._observations[start : self._next_index]


class StreamingSimulatedOptiTrackSource:
    """Online position sampler fed only successive simulator truth positions.

    Truth is confined to this source.  Consumers can retrieve only immutable
    position observations whose synthetic arrival time has passed.
    """

    def __init__(
        self,
        *,
        measurement_rate_hz: float = 100.0,
        position_noise_std_m: float = 0.0,
        latency_s: float = 0.0,
        independent_dropout_probability: float = 0.0,
        random_seed: int = 0,
    ) -> None:
        if measurement_rate_hz <= 0.0 or not math.isfinite(measurement_rate_hz):
            raise ValueError("Measurement rate must be positive.")
        if position_noise_std_m < 0.0 or not math.isfinite(position_noise_std_m):
            raise ValueError("Position noise must be non-negative.")
        if latency_s < 0.0 or not math.isfinite(latency_s):
            raise ValueError("Latency must be non-negative.")
        if not 0.0 <= independent_dropout_probability < 1.0:
            raise ValueError("Dropout probability must lie in [0, 1).")
        self._dt = 1.0 / measurement_rate_hz
        self._noise = position_noise_std_m
        self._latency = latency_s
        self._dropout = independent_dropout_probability
        self._rng = np.random.default_rng(random_seed)
        self._pending: deque[CableObservation] = deque()
        self._sequence = 0
        self._last_truth_time_s: float | None = None
        self._last_root: np.ndarray | None = None
        self._last_cable: np.ndarray | None = None
        self._next_sample_time_s: float | None = None

    def _queue(self, timestamp_s: float, root: np.ndarray, cable: np.ndarray) -> None:
        noisy_root = np.array(root, dtype=np.float64, copy=True)
        noisy_markers = np.array(cable[1:], dtype=np.float64, copy=True)
        if self._noise > 0.0:
            noisy_root += self._rng.normal(0.0, self._noise, 3)
            noisy_markers += self._rng.normal(0.0, self._noise, (10, 3))
        validity = self._rng.random(10) >= self._dropout
        self._pending.append(
            CableObservation(
                sample_timestamp_s=timestamp_s,
                arrival_timestamp_s=timestamp_s + self._latency,
                sequence_number=self._sequence,
                root_position_m=noisy_root,
                marker_positions_m=noisy_markers,
                marker_validity_mask=validity,
            )
        )
        self._sequence += 1

    def prime(
        self,
        time_s: float,
        root_position_m: np.ndarray,
        cable_positions_m: np.ndarray,
        *,
        prehistory_s: float = 0.10,
    ) -> None:
        if self._last_truth_time_s is not None:
            raise RuntimeError("Streaming source can only be primed once.")
        root = np.asarray(root_position_m, dtype=np.float64)
        cable = np.asarray(cable_positions_m, dtype=np.float64)
        if root.shape != (3,) or cable.shape != (11, 3):
            raise ValueError("Streaming truth positions have invalid shape.")
        count = max(1, int(round(prehistory_s / self._dt)))
        for offset in range(count, -1, -1):
            self._queue(time_s - offset * self._dt, root, cable)
        self._last_truth_time_s = time_s
        self._last_root = root.copy()
        self._last_cable = cable.copy()
        self._next_sample_time_s = time_s + self._dt

    def append_truth_positions(
        self,
        time_s: float,
        root_position_m: np.ndarray,
        cable_positions_m: np.ndarray,
    ) -> None:
        if self._last_truth_time_s is None:
            self.prime(time_s, root_position_m, cable_positions_m)
            return
        if time_s <= self._last_truth_time_s + 1.0e-12:
            if abs(time_s - self._last_truth_time_s) <= 1.0e-12:
                return
            raise ValueError("Streaming truth positions must advance in time.")
        root = np.asarray(root_position_m, dtype=np.float64)
        cable = np.asarray(cable_positions_m, dtype=np.float64)
        if root.shape != (3,) or cable.shape != (11, 3):
            raise ValueError("Streaming truth positions have invalid shape.")
        assert self._last_root is not None and self._last_cable is not None
        assert self._next_sample_time_s is not None
        duration = time_s - self._last_truth_time_s
        while self._next_sample_time_s <= time_s + 1.0e-12:
            fraction = np.clip(
                (self._next_sample_time_s - self._last_truth_time_s) / duration,
                0.0,
                1.0,
            )
            sampled_root = (1.0 - fraction) * self._last_root + fraction * root
            sampled_cable = (1.0 - fraction) * self._last_cable + fraction * cable
            self._queue(self._next_sample_time_s, sampled_root, sampled_cable)
            self._next_sample_time_s += self._dt
        self._last_truth_time_s = time_s
        self._last_root = root.copy()
        self._last_cable = cable.copy()

    def observations_arrived_by(self, time_s: float) -> tuple[CableObservation, ...]:
        ready = []
        while (
            self._pending
            and self._pending[0].arrival_timestamp_s <= time_s + 1.0e-12
        ):
            ready.append(self._pending.popleft())
        return tuple(ready)


def distributed_observation_from_estimate(
    estimate: EstimatedCableState,
    *,
    active_parameter_estimate,
    executed_action_m_s2: np.ndarray | Sequence[float],
    attachment_drop_m: float,
    contact: bool = False,
    safety_violation: bool = False,
):
    """Adapt an estimated state to the existing physical-adapter contract.

    The local import keeps the position-only measurement module independent of
    the parameter estimator while providing one audited bridge used by both
    simulation and the future Motive source.
    """

    from .distributed_adaptation import DistributedObservation

    if not math.isfinite(attachment_drop_m) or attachment_drop_m <= 0.0:
        raise ValueError("Attachment drop must be positive.")
    root = estimate.cable_positions_m[0]
    root_velocity = estimate.cable_velocities_m_s[0]
    drone_position = root + np.asarray((0.0, 0.0, attachment_drop_m))
    drone_state = np.concatenate((drone_position, root_velocity))
    return DistributedObservation(
        timestamp_s=estimate.sample_timestamp_s,
        attachment_position_m=root,
        attachment_velocity_m_s=root_velocity,
        cable_positions_m=estimate.cable_positions_m,
        cable_velocities_m_s=estimate.cable_velocities_m_s,
        drone_state=drone_state,
        executed_action_m_s2=np.asarray(executed_action_m_s2, dtype=np.float64),
        active_estimate=active_parameter_estimate,
        contact=contact,
        safety_violation=safety_violation,
        observation_valid=bool(np.all(estimate.state_validity_mask)),
        measurement_validity_mask=estimate.measurement_validity_mask,
    )
