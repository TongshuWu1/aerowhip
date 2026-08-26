"""Event-triggered distributed-state EI/Cb adaptation for DDER--MPPI.

The online controller and the estimator deliberately have different timing
contracts.  The controller reads one immutable :class:`ParameterEstimate` at
the beginning of an MPPI solve.  The estimator works on copied short measured
segments and may publish a new immutable estimate only after held-out
validation.  No fitting operation is on the controller's critical path.

Only the two homogeneous material parameters are adapted.  Geometry, masses,
gravity, time integration, damping discretization, and constraint projection
remain those of the supplied one-attached DDER model.  Online deployments may
choose any fixed discretization supported by the identified source artifact.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
from threading import Lock
import time
from typing import Iterable, Literal, Sequence

import numpy as np
import torch

from cable_twin.shared.dder import DderState, START_PINNED_FREE_END

from .model import CableModelSnapshot
from .reduced import reduce_cable_model
from .simulator import SimulationSettings, WhipSimulator


ADAPTATION_SCHEMA = "distributed_event_triggered_dder_adaptation_v1"
PARAMETER_NAMES = ("log_ei_ratio", "log_cb_ratio")
AdaptParameterMode = Literal["joint", "ei_only", "cb_only"]
SchedulingMode = Literal["replay", "between_strikes", "concurrent"]


def _readonly(
    value: np.ndarray | Sequence[float], *, name: str, ndim: int
) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D array.")
    array.setflags(write=False)
    return array


def _positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    valid = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}.")


@dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """One immutable controller-model version in nominal log coordinates."""

    eta_e: float = 0.0
    eta_c: float = 0.0
    generation: int = 0
    published_at_s: float = 0.0
    validation_loss: float = math.inf
    source: str = "nominal"

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.eta_e, self.eta_c, self.published_at_s)
        ):
            raise ValueError("Parameter-estimate values must be finite.")
        if self.generation < 0:
            raise ValueError("Parameter generation cannot be negative.")
        if not self.source:
            raise ValueError("Parameter source must be non-empty.")

    @property
    def eta(self) -> np.ndarray:
        result = np.asarray((self.eta_e, self.eta_c), dtype=np.float64)
        result.setflags(write=False)
        return result

    @property
    def ei_ratio(self) -> float:
        return math.exp(self.eta_e)

    @property
    def cb_ratio(self) -> float:
        return math.exp(self.eta_c)


class AtomicParameterStore:
    """Lock-protected whole-object publication; an MPPI solve never sees a mix."""

    def __init__(self, initial: ParameterEstimate | None = None) -> None:
        self._lock = Lock()
        self._value = initial or ParameterEstimate()

    def snapshot(self) -> ParameterEstimate:
        with self._lock:
            return self._value

    def publish_if_current(
        self, candidate: ParameterEstimate, *, expected_generation: int
    ) -> bool:
        with self._lock:
            if self._value.generation != expected_generation:
                return False
            if candidate.generation != expected_generation + 1:
                raise ValueError("Published generation must advance exactly once.")
            self._value = candidate
            return True


@dataclass(frozen=True, slots=True)
class DistributedObservation:
    """One distributed-state estimate consumed by adaptation.

    ``measurement_validity_mask`` records which node positions were genuinely
    observed at this frame.  Reconstructed values may still initialize and
    propagate DDER state, but invalid/imputed entries are excluded from the
    fitting residual and held-out score.
    """

    timestamp_s: float
    attachment_position_m: np.ndarray
    attachment_velocity_m_s: np.ndarray
    cable_positions_m: np.ndarray
    cable_velocities_m_s: np.ndarray
    drone_state: np.ndarray
    executed_action_m_s2: np.ndarray
    active_estimate: ParameterEstimate
    contact: bool = False
    safety_violation: bool = False
    observation_valid: bool = True
    measurement_validity_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s):
            raise ValueError("Observation timestamp must be finite.")
        attachment = _readonly(
            self.attachment_position_m, name="attachment_position_m", ndim=1
        )
        attachment_velocity = _readonly(
            self.attachment_velocity_m_s,
            name="attachment_velocity_m_s",
            ndim=1,
        )
        positions = _readonly(
            self.cable_positions_m, name="cable_positions_m", ndim=2
        )
        velocities = _readonly(
            self.cable_velocities_m_s, name="cable_velocities_m_s", ndim=2
        )
        drone = _readonly(self.drone_state, name="drone_state", ndim=1)
        action = _readonly(
            self.executed_action_m_s2, name="executed_action_m_s2", ndim=1
        )
        if attachment.shape != (3,) or attachment_velocity.shape != (3,):
            raise ValueError("Attachment state must contain XYZ vectors.")
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 3:
            raise ValueError("Cable positions must have shape Nx3 with N >= 3.")
        if velocities.shape != positions.shape:
            raise ValueError("Cable velocities must match cable positions.")
        if drone.size < 6:
            raise ValueError("Drone state must contain at least position and velocity.")
        if action.shape != (3,):
            raise ValueError("Executed action must be a three-vector.")
        validity = (
            np.ones(positions.shape[0], dtype=bool)
            if self.measurement_validity_mask is None
            else np.array(self.measurement_validity_mask, dtype=bool, copy=True)
        )
        if validity.shape != (positions.shape[0],):
            raise ValueError("Measurement validity must contain one value per node.")
        validity.setflags(write=False)
        object.__setattr__(self, "attachment_position_m", attachment)
        object.__setattr__(self, "attachment_velocity_m_s", attachment_velocity)
        object.__setattr__(self, "cable_positions_m", positions)
        object.__setattr__(self, "cable_velocities_m_s", velocities)
        object.__setattr__(self, "drone_state", drone)
        object.__setattr__(self, "executed_action_m_s2", action)
        object.__setattr__(self, "measurement_validity_mask", validity)


@dataclass(frozen=True, slots=True)
class AdaptationSegment:
    """Measured multiple-shooting segment initialized from its first frame."""

    time_s: np.ndarray
    attachment_positions_m: np.ndarray
    cable_positions_m: np.ndarray
    cable_velocities_m_s: np.ndarray
    excitation_score: float
    source: str
    start_time_s: float
    measurement_validity_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        times = _readonly(self.time_s, name="segment time_s", ndim=1)
        attachment = _readonly(
            self.attachment_positions_m,
            name="segment attachment_positions_m",
            ndim=2,
        )
        positions = _readonly(
            self.cable_positions_m, name="segment cable_positions_m", ndim=3
        )
        velocities = _readonly(
            self.cable_velocities_m_s, name="segment cable_velocities_m_s", ndim=3
        )
        if len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("A segment needs increasing timestamps and one transition.")
        if attachment.shape != (len(times), 3):
            raise ValueError("Segment attachment path must have shape Tx3.")
        if positions.shape[:1] != (len(times),) or positions.shape[-1] != 3:
            raise ValueError("Segment cable positions must have shape TxNx3.")
        if velocities.shape != positions.shape:
            raise ValueError("Segment velocities must match positions.")
        validity = (
            np.ones(positions.shape[:2], dtype=bool)
            if self.measurement_validity_mask is None
            else np.array(self.measurement_validity_mask, dtype=bool, copy=True)
        )
        if validity.shape != positions.shape[:2]:
            raise ValueError("Segment validity must have shape TxN.")
        validity.setflags(write=False)
        if not math.isfinite(self.excitation_score) or self.excitation_score < 0.0:
            raise ValueError("Segment excitation score must be finite and non-negative.")
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "attachment_positions_m", attachment)
        object.__setattr__(self, "cable_positions_m", positions)
        object.__setattr__(self, "cable_velocities_m_s", velocities)
        object.__setattr__(self, "measurement_validity_mask", validity)

    @property
    def frame_count(self) -> int:
        return len(self.time_s)

    @property
    def dynamic_measurement_coverage(self) -> float:
        # Initial state and prescribed root are not fitting residuals.
        mask = self.measurement_validity_mask[1:, 1:]
        return float(np.mean(mask)) if mask.size else 0.0


@dataclass(frozen=True, slots=True)
class DistributedAdaptationSettings:
    """Globally fixed causal monitoring and bounded fitting choices."""

    buffer_duration_s: float = 2.0
    cache_size: int = 8
    health_horizon_s: float = 0.10
    health_evaluation_interval_s: float = 0.10
    health_ema_alpha: float = 0.90
    # Exact-state simulation has an approximately 1e-14 m^2 matched numerical
    # floor.  A 20% EI mismatch produces 0.9--1.0e-8 m^2 and a 30% Cb
    # mismatch 2--4e-8 m^2 over the documented 0.1 s health horizon.  These
    # fixed thresholds sit safely between those populations; they must be
    # recalibrated, once globally, when measurement noise is introduced.
    error_high_m2: float = 1.0e-9
    error_low_m2: float = 2.5e-10
    persistence_s: float = 0.20
    cooldown_s: float = 1.0
    velocity_activity_scale_m_s: float = 1.0
    curvature_activity_scale: float = 1.0
    curvature_rate_activity_scale_s_inv: float = 10.0
    excitation_velocity_weight: float = 1.0
    excitation_curvature_weight: float = 0.10
    excitation_curvature_rate_weight: float = 0.10
    minimum_excitation: float = 0.01
    window_duration_s: float = 0.30
    segment_duration_s: float = 0.10
    fit_segment_count: int = 4
    validation_segment_count: int = 2
    finite_difference_log_step_ei: float = 0.03
    finite_difference_log_step_cb: float = 0.03
    minimum_information_eigenvalue: float = 1.0e-7
    maximum_information_condition: float = 1.0e7
    lm_damping: float = 1.0e-4
    maximum_log_step_ei: float = math.log(1.20)
    maximum_log_step_cb: float = math.log(1.30)
    ei_ratio_bounds: tuple[float, float] = (0.5, 2.0)
    cb_ratio_bounds: tuple[float, float] = (0.4, 2.5)
    line_search_steps: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    validation_minimum_improvement: float = 1.0e-10
    maximum_gn_iterations: int = 2
    parameter_mode: AdaptParameterMode = "joint"
    minimum_measurement_coverage: float = 0.80
    use_informative_cache: bool = True

    def __post_init__(self) -> None:
        for name in (
            "buffer_duration_s",
            "health_horizon_s",
            "health_evaluation_interval_s",
            "error_high_m2",
            "persistence_s",
            "cooldown_s",
            "velocity_activity_scale_m_s",
            "curvature_activity_scale",
            "curvature_rate_activity_scale_s_inv",
            "minimum_excitation",
            "window_duration_s",
            "segment_duration_s",
            "finite_difference_log_step_ei",
            "finite_difference_log_step_cb",
            "minimum_information_eigenvalue",
            "maximum_information_condition",
            "lm_damping",
            "maximum_log_step_ei",
            "maximum_log_step_cb",
        ):
            _positive(name, float(getattr(self, name)))
        _positive("error_low_m2", self.error_low_m2, allow_zero=True)
        _positive(
            "validation_minimum_improvement",
            self.validation_minimum_improvement,
            allow_zero=True,
        )
        if not 0.0 <= self.health_ema_alpha < 1.0:
            raise ValueError("health_ema_alpha must lie in [0, 1).")
        if self.error_low_m2 >= self.error_high_m2:
            raise ValueError("Health hysteresis requires error_low < error_high.")
        if self.cache_size < 1 or self.fit_segment_count < 1:
            raise ValueError("Cache and fit segment counts must be positive.")
        if self.validation_segment_count < 1:
            raise ValueError("Held-out validation requires at least one segment.")
        if self.maximum_gn_iterations not in (1, 2, 3):
            raise ValueError("maximum_gn_iterations must be 1, 2, or 3.")
        if self.segment_duration_s > self.window_duration_s:
            raise ValueError("Multiple-shooting segment cannot exceed its window.")
        for name, bounds in (
            ("EI", self.ei_ratio_bounds),
            ("Cb", self.cb_ratio_bounds),
        ):
            if (
                len(bounds) != 2
                or not all(math.isfinite(x) and x > 0.0 for x in bounds)
                or bounds[0] >= 1.0
                or bounds[1] <= 1.0
            ):
                raise ValueError(f"{name} bounds must be positive and contain one.")
        if (
            not self.line_search_steps
            or any(not math.isfinite(x) or x <= 0.0 for x in self.line_search_steps)
            or tuple(sorted(self.line_search_steps, reverse=True))
            != self.line_search_steps
        ):
            raise ValueError("Line-search steps must be positive and descending.")
        if self.parameter_mode not in {"joint", "ei_only", "cb_only"}:
            raise ValueError("Unknown adaptation parameter mode.")
        if not 0.0 < self.minimum_measurement_coverage <= 1.0:
            raise ValueError("Measurement coverage must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class ExcitationDiagnostic:
    relative_velocity: float
    curvature: float
    curvature_rate: float
    score: float


@dataclass(frozen=True, slots=True)
class HealthDiagnostic:
    timestamp_s: float
    instantaneous_error_m2: float
    ema_error_m2: float
    horizon_s: float
    excitation: ExcitationDiagnostic
    mismatch_persistent: bool
    hysteresis_armed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class InformationDiagnostic:
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition: float
    sensitivity_correlation: float
    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class FitTiming:
    sensitivity_forward_s: float
    information_s: float
    line_search_forward_s: float
    validation_s: float
    total_s: float
    short_rollouts: int


@dataclass(frozen=True, slots=True)
class DistributedFitResult:
    accepted: bool
    reason: str
    initial_estimate: ParameterEstimate
    candidate_estimate: ParameterEstimate
    information: InformationDiagnostic
    fit_loss_before: float
    fit_loss_after: float
    validation_loss_before: float
    validation_loss_after: float
    all_node_position_rmse_before_m: float
    all_node_position_rmse_after_m: float
    all_node_velocity_rmse_before_m_s: float
    all_node_velocity_rmse_after_m_s: float
    tip_velocity_rmse_before_m_s: float
    tip_velocity_rmse_after_m_s: float
    propagation_timing_error_before_s: float
    propagation_timing_error_after_s: float
    iterations: int
    selected_fit_segments: tuple[float, ...]
    selected_validation_segments: tuple[float, ...]
    timing: FitTiming
    history: tuple[dict[str, float], ...]


def curvature_binormals_numpy(positions_m: np.ndarray) -> np.ndarray:
    """DDER curvature binormals using the production denominator clamp."""

    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 3:
        raise ValueError("Curvature input must have shape Nx3 with N >= 3.")
    edges = np.diff(positions, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    tangents = edges / np.maximum(lengths[:, None], 1.0e-12)
    denominator = 1.0 + np.sum(tangents[:-1] * tangents[1:], axis=1)
    denominator = np.maximum(denominator, 1.0e-6)
    return 2.0 * np.cross(tangents[:-1], tangents[1:]) / denominator[:, None]


def excitation_diagnostic(
    current: DistributedObservation,
    previous: DistributedObservation | None,
    settings: DistributedAdaptationSettings,
) -> ExcitationDiagnostic:
    root_velocity = current.attachment_velocity_m_s
    relative = current.cable_velocities_m_s[1:] - root_velocity[None]
    velocity_score = float(np.mean(np.sum(relative * relative, axis=1))) / (
        settings.velocity_activity_scale_m_s**2
    )
    curvature = curvature_binormals_numpy(current.cable_positions_m)
    curvature_score = float(np.mean(np.sum(curvature * curvature, axis=1))) / (
        settings.curvature_activity_scale**2
    )
    rate_score = 0.0
    if previous is not None:
        dt = current.timestamp_s - previous.timestamp_s
        if dt > 0.0 and previous.cable_positions_m.shape == current.cable_positions_m.shape:
            old_curvature = curvature_binormals_numpy(previous.cable_positions_m)
            rate = (curvature - old_curvature) / dt
            rate_score = float(np.mean(np.sum(rate * rate, axis=1))) / (
                settings.curvature_rate_activity_scale_s_inv**2
            )
    score = (
        settings.excitation_velocity_weight * velocity_score
        + settings.excitation_curvature_weight * curvature_score
        + settings.excitation_curvature_rate_weight * rate_score
    )
    return ExcitationDiagnostic(velocity_score, curvature_score, rate_score, score)


class RollingDistributedBuffer:
    """Recent FIFO plus a bounded cache of informative measured segments."""

    def __init__(self, settings: DistributedAdaptationSettings) -> None:
        self.settings = settings
        self._recent: deque[DistributedObservation] = deque()
        self._cache: list[AdaptationSegment] = []

    @property
    def recent(self) -> tuple[DistributedObservation, ...]:
        return tuple(self._recent)

    @property
    def cache(self) -> tuple[AdaptationSegment, ...]:
        return tuple(self._cache)

    def start_new_recording(self) -> None:
        """End one discontinuous trial while retaining informative history.

        Separate strikes can restart from unrelated initial states.  Recent
        frames therefore cannot form one segment across that boundary, while
        the bounded informative cache is deliberately session-persistent.
        """

        self._recent.clear()

    def start_new_plant_regime(self) -> None:
        """Discard observations collected under a previous physical plant.

        A normal strike boundary preserves the informative cache because the
        cable physics are unchanged.  An intentional cable-parameter change is
        different: old segments must not be combined with motion from the new
        plant in one identification problem.
        """

        self._recent.clear()
        self._cache.clear()

    def append(self, observation: DistributedObservation) -> None:
        if self._recent and observation.timestamp_s <= self._recent[-1].timestamp_s:
            raise ValueError("Adaptation observations must arrive in time order.")
        if self._recent and observation.cable_positions_m.shape != self._recent[-1].cable_positions_m.shape:
            raise ValueError("Cable node count cannot change within one buffer.")
        self._recent.append(observation)
        cutoff = observation.timestamp_s - self.settings.buffer_duration_s
        while self._recent and self._recent[0].timestamp_s < cutoff:
            self._recent.popleft()
        self._cache_newest_segment()

    def _cache_newest_segment(self) -> None:
        if not self.settings.use_informative_cache:
            return
        if len(self._recent) < 2:
            return
        end = self._recent[-1]
        start_target = end.timestamp_s - self.settings.segment_duration_s
        frames = [obs for obs in self._recent if obs.timestamp_s >= start_target - 1.0e-9]
        if len(frames) < 2 or frames[0].timestamp_s > start_target + 0.5 * (
            frames[-1].timestamp_s - frames[-2].timestamp_s
        ):
            return
        if any(
            obs.contact or obs.safety_violation or not obs.observation_valid
            for obs in frames
        ):
            return
        previous: DistributedObservation | None = None
        diagnostics = []
        for obs in frames:
            diagnostics.append(excitation_diagnostic(obs, previous, self.settings))
            previous = obs
        score = float(np.mean([item.score for item in diagnostics]))
        candidate = _segment_from_observations(frames, score, "recent")
        # Avoid inserting a near-identical overlapping segment at every frame.
        if self._cache and abs(candidate.start_time_s - self._cache[-1].start_time_s) < 0.5 * self.settings.segment_duration_s:
            if candidate.excitation_score > self._cache[-1].excitation_score:
                self._cache[-1] = candidate
            return
        self._cache.append(candidate)
        if len(self._cache) > self.settings.cache_size:
            weakest = min(
                range(len(self._cache)),
                key=lambda index: self._cache[index].excitation_score,
            )
            self._cache.pop(weakest)

    def candidate_segments(self) -> tuple[AdaptationSegment, ...]:
        candidates = list(self._cache) if self.settings.use_informative_cache else []
        observations = list(self._recent)
        if len(observations) >= 2:
            window_start = observations[-1].timestamp_s - self.settings.window_duration_s
            window = [obs for obs in observations if obs.timestamp_s >= window_start - 1.0e-9]
            duration = self.settings.segment_duration_s
            if len(window) >= 2:
                next_start = window[0].timestamp_s
                while next_start + duration <= window[-1].timestamp_s + 1.0e-9:
                    frames = [
                        obs
                        for obs in window
                        if next_start - 1.0e-9 <= obs.timestamp_s <= next_start + duration + 1.0e-9
                    ]
                    if len(frames) >= 2 and not any(
                        obs.contact or obs.safety_violation or not obs.observation_valid
                        for obs in frames
                    ):
                        score = _mean_excitation(frames, self.settings)
                        candidates.append(_segment_from_observations(frames, score, "fifo"))
                    next_start += duration
        unique: dict[int, AdaptationSegment] = {}
        for segment in candidates:
            key = round(segment.start_time_s / max(self.settings.segment_duration_s, 1.0e-9))
            old = unique.get(key)
            if old is None or segment.excitation_score > old.excitation_score:
                unique[key] = segment
        return tuple(sorted(unique.values(), key=lambda segment: segment.start_time_s))

    @property
    def approximate_memory_bytes(self) -> int:
        total = 0
        for obs in self._recent:
            total += sum(
                array.nbytes
                for array in (
                    obs.attachment_position_m,
                    obs.attachment_velocity_m_s,
                    obs.cable_positions_m,
                    obs.cable_velocities_m_s,
                    obs.drone_state,
                    obs.executed_action_m_s2,
                    obs.measurement_validity_mask,
                )
            )
        for segment in self._cache:
            total += sum(
                array.nbytes
                for array in (
                    segment.time_s,
                    segment.attachment_positions_m,
                    segment.cable_positions_m,
                    segment.cable_velocities_m_s,
                    segment.measurement_validity_mask,
                )
            )
        return total


def _mean_excitation(
    observations: Sequence[DistributedObservation],
    settings: DistributedAdaptationSettings,
) -> float:
    previous: DistributedObservation | None = None
    values = []
    for observation in observations:
        values.append(excitation_diagnostic(observation, previous, settings).score)
        previous = observation
    return float(np.mean(values))


def _segment_from_observations(
    observations: Sequence[DistributedObservation], score: float, source: str
) -> AdaptationSegment:
    return AdaptationSegment(
        time_s=np.asarray([obs.timestamp_s for obs in observations]),
        attachment_positions_m=np.stack(
            [obs.attachment_position_m for obs in observations]
        ),
        cable_positions_m=np.stack([obs.cable_positions_m for obs in observations]),
        cable_velocities_m_s=np.stack(
            [obs.cable_velocities_m_s for obs in observations]
        ),
        measurement_validity_mask=np.stack(
            [obs.measurement_validity_mask for obs in observations]
        ),
        excitation_score=score,
        source=source,
        start_time_s=observations[0].timestamp_s,
    )


class BatchedMeasuredBoundaryPredictor:
    """Fast short DDER rollouts over segment and parameter-hypothesis batches."""

    def __init__(
        self,
        nominal_model: CableModelSnapshot,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        if nominal_model.node_count < 3:
            raise ValueError("The online adapter requires at least three DDER nodes.")
        self.nominal_model = nominal_model
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA adaptation was selected but is unavailable.")
        self.dtype = torch.float32 if self.device.type == "cuda" else torch.float64

    def predict(
        self,
        segments: Sequence[AdaptationSegment],
        eta_hypotheses: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if not segments:
            raise ValueError("At least one adaptation segment is required.")
        hypotheses = np.asarray(eta_hypotheses, dtype=np.float64)
        if hypotheses.ndim != 2 or hypotheses.shape[1] != 2 or not np.all(np.isfinite(hypotheses)):
            raise ValueError("Parameter hypotheses must have shape Px2 and be finite.")
        frame_count = segments[0].frame_count
        node_count = self.nominal_model.node_count
        if any(
            segment.frame_count != frame_count
            or segment.cable_positions_m.shape[1:] != (node_count, 3)
            or not np.allclose(np.diff(segment.time_s), np.diff(segment.time_s)[0], rtol=0.0, atol=1.0e-7)
            for segment in segments
        ):
            raise ValueError("Batched fitting requires equal, uniformly sampled segments.")
        segment_count = len(segments)
        hypothesis_count = len(hypotheses)
        batch_size = segment_count * hypothesis_count
        initial_positions = np.repeat(
            np.stack([segment.cable_positions_m[0] for segment in segments]),
            hypothesis_count,
            axis=0,
        )
        initial_velocities = np.repeat(
            np.stack([segment.cable_velocities_m_s[0] for segment in segments]),
            hypothesis_count,
            axis=0,
        )
        q = torch.as_tensor(initial_positions, dtype=self.dtype, device=self.device)
        v = torch.as_tensor(initial_velocities, dtype=self.dtype, device=self.device)
        state = DderState(q, v)
        expanded_eta = np.tile(hypotheses, (segment_count, 1))
        eta = torch.as_tensor(expanded_eta, dtype=self.dtype, device=self.device)
        constants = self.nominal_model.model.runtime_constants(q)
        constants = replace(
            constants,
            bending_stiffness_n_m2=(
                self.nominal_model.bending_stiffness_n_m2 * torch.exp(eta[:, 0])
            ),
            bending_damping_n_m2_s=(
                self.nominal_model.bending_damping_n_m2_s * torch.exp(eta[:, 1])
            ),
        )
        positions = [q]
        velocities = [v]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        for frame_index in range(1, frame_count):
            boundary = np.repeat(
                np.stack(
                    [
                        segment.attachment_positions_m[frame_index]
                        for segment in segments
                    ]
                ),
                hypothesis_count,
                axis=0,
            )
            dt_values = np.repeat(
                np.asarray(
                    [
                        segment.time_s[frame_index]
                        - segment.time_s[frame_index - 1]
                        for segment in segments
                    ],
                    dtype=np.float64,
                ),
                hypothesis_count,
            )
            state = self.nominal_model.model.step_runtime(
                state,
                torch.as_tensor(boundary[:, None], dtype=self.dtype, device=self.device),
                torch.as_tensor(dt_values, dtype=self.dtype, device=self.device),
                constants,
                iterative_damping=True,
                pinned_endpoints=START_PINNED_FREE_END,
            )
            positions.append(state.positions_m)
            velocities.append(state.velocities_m_s)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        position = torch.stack(positions, dim=1).reshape(
            segment_count, hypothesis_count, frame_count, node_count, 3
        )
        velocity = torch.stack(velocities, dim=1).reshape(
            segment_count, hypothesis_count, frame_count, node_count, 3
        )
        return (
            position.detach().cpu().numpy().astype(np.float64, copy=False),
            velocity.detach().cpu().numpy().astype(np.float64, copy=False),
            elapsed,
        )


def _observations(
    segments: Sequence[AdaptationSegment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([segment.cable_positions_m for segment in segments]),
        np.stack([segment.cable_velocities_m_s for segment in segments]),
        np.stack([segment.measurement_validity_mask for segment in segments]),
    )


def _residual_vector(
    predicted_positions: np.ndarray,
    observed_positions: np.ndarray,
    cable_length_m: float,
    measurement_validity_mask: np.ndarray | None = None,
) -> np.ndarray:
    # Vertex zero is prescribed by measured attachment motion and therefore is
    # not a dynamic residual.  Every other distributed material point is used.
    residual = (
        predicted_positions[:, 1:, 1:, :] - observed_positions[:, 1:, 1:, :]
    ) / cable_length_m
    if measurement_validity_mask is not None:
        mask = np.asarray(measurement_validity_mask, dtype=bool)[:, 1:, 1:]
        residual = residual[mask]
    return residual.reshape(-1)


def _loss_and_metrics(
    predicted_positions: np.ndarray,
    predicted_velocities: np.ndarray,
    observed_positions: np.ndarray,
    observed_velocities: np.ndarray,
    cable_length_m: float,
    measurement_validity_mask: np.ndarray | None = None,
) -> tuple[float, float, float, float]:
    position_error = predicted_positions[:, 1:, 1:] - observed_positions[:, 1:, 1:]
    velocity_error = predicted_velocities[:, 1:, 1:] - observed_velocities[:, 1:, 1:]
    tip_velocity_error = predicted_velocities[:, 1:, -1] - observed_velocities[:, 1:, -1]
    if measurement_validity_mask is None:
        position_values = position_error.reshape(-1, 3)
        velocity_values = velocity_error.reshape(-1, 3)
        tip_velocity_values = tip_velocity_error.reshape(-1, 3)
    else:
        mask = np.asarray(measurement_validity_mask, dtype=bool)[:, 1:, 1:]
        position_values = position_error[mask]
        velocity_values = velocity_error[mask]
        tip_mask = np.asarray(measurement_validity_mask, dtype=bool)[:, 1:, -1]
        tip_velocity_values = tip_velocity_error[tip_mask]
    if position_values.size == 0:
        return math.inf, math.inf, math.inf, math.inf
    loss = float(np.mean((position_values / cable_length_m) ** 2))
    return (
        loss,
        float(np.sqrt(np.mean(position_values**2))),
        float(np.sqrt(np.mean(velocity_values**2))),
        float(
            np.sqrt(np.mean(tip_velocity_values**2))
            if tip_velocity_values.size
            else math.nan
        ),
    )


def _propagation_timing_error(
    predicted_velocities: np.ndarray,
    observed_velocities: np.ndarray,
    segments: Sequence[AdaptationSegment],
) -> float:
    errors = []
    for index, segment in enumerate(segments):
        predicted_peak = int(
            np.argmax(np.linalg.norm(predicted_velocities[index, 1:, -1], axis=1))
            + 1
        )
        observed_peak = int(
            np.argmax(np.linalg.norm(observed_velocities[index, 1:, -1], axis=1))
            + 1
        )
        errors.append(abs(segment.time_s[predicted_peak] - segment.time_s[observed_peak]))
    return float(np.mean(errors))


def _information(jacobian: np.ndarray, settings: DistributedAdaptationSettings) -> InformationDiagnostic:
    matrix = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(matrix)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    condition = maximum / (minimum + np.finfo(np.float64).eps)
    norm_e = float(np.linalg.norm(jacobian[:, 0]))
    norm_c = float(np.linalg.norm(jacobian[:, 1]))
    correlation = float(
        np.dot(jacobian[:, 0], jacobian[:, 1])
        / (norm_e * norm_c + np.finfo(np.float64).eps)
    )
    accepted = (
        np.all(np.isfinite(matrix))
        and minimum > settings.minimum_information_eigenvalue
        and condition < settings.maximum_information_condition
    )
    if not np.all(np.isfinite(matrix)):
        reason = "non_finite_sensitivity"
    elif minimum <= settings.minimum_information_eigenvalue:
        reason = "minimum_information_eigenvalue"
    elif condition >= settings.maximum_information_condition:
        reason = "information_condition"
    else:
        reason = "accepted"
    return InformationDiagnostic(minimum, maximum, condition, correlation, accepted, reason)


def select_informative_segments(
    candidates: Sequence[AdaptationSegment],
    settings: DistributedAdaptationSettings,
) -> tuple[tuple[AdaptationSegment, ...], tuple[AdaptationSegment, ...]]:
    """Greedy excitation/diversity selection with a chronological holdout."""

    required = settings.fit_segment_count + settings.validation_segment_count
    eligible = [
        segment
        for segment in candidates
        if segment.excitation_score >= settings.minimum_excitation
        and segment.dynamic_measurement_coverage
        >= settings.minimum_measurement_coverage
    ]
    # Floating timestamps can place a boundary frame just outside one cached
    # interval.  The batched predictor needs a static shape, so retain the
    # frame-count group with the greatest usable excitation mass.
    if eligible:
        frame_counts = sorted({segment.frame_count for segment in eligible})
        best_frame_count = max(
            frame_counts,
            key=lambda count: (
                sum(
                    segment.excitation_score
                    for segment in eligible
                    if segment.frame_count == count
                ),
                sum(segment.frame_count == count for segment in eligible),
            ),
        )
        eligible = [
            segment for segment in eligible if segment.frame_count == best_frame_count
        ]
    if len(eligible) < required:
        return (), ()
    eligible.sort(key=lambda segment: segment.excitation_score, reverse=True)
    selected: list[AdaptationSegment] = []
    while eligible and len(selected) < required:
        if not selected:
            index = 0
        else:
            # Reward temporal separation to avoid six copies of one reversal.
            span = max(settings.buffer_duration_s, settings.segment_duration_s)
            values = []
            for segment in eligible:
                novelty = min(
                    abs(segment.start_time_s - old.start_time_s) / span
                    for old in selected
                )
                values.append(segment.excitation_score * (1.0 + novelty))
            index = int(np.argmax(values))
        selected.append(eligible.pop(index))
    selected.sort(key=lambda segment: segment.start_time_s)
    validation = tuple(selected[-settings.validation_segment_count :])
    fitting = tuple(selected[: -settings.validation_segment_count])
    return fitting, validation


class DistributedParameterFitter:
    """Two-variable finite-difference LM using short batched DDER rollouts."""

    def __init__(
        self,
        nominal_model: CableModelSnapshot,
        settings: DistributedAdaptationSettings = DistributedAdaptationSettings(),
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        self.nominal_model = nominal_model
        self.settings = settings
        self.predictor = BatchedMeasuredBoundaryPredictor(nominal_model, device=device)

    def fit(
        self,
        fitting_segments: Sequence[AdaptationSegment],
        validation_segments: Sequence[AdaptationSegment],
        current: ParameterEstimate,
    ) -> DistributedFitResult:
        if not fitting_segments or not validation_segments:
            raise ValueError("Fitting and held-out validation segments are required.")
        started = time.perf_counter()
        eta = np.asarray((current.eta_e, current.eta_c), dtype=np.float64)
        initial_eta = eta.copy()
        sensitivity_time = 0.0
        information_time = 0.0
        line_search_time = 0.0
        validation_time = 0.0
        short_rollouts = 0
        history: list[dict[str, float]] = []
        last_information = InformationDiagnostic(0.0, 0.0, math.inf, 0.0, False, "not_evaluated")
        (
            fit_observed_position,
            fit_observed_velocity,
            fit_validity,
        ) = _observations(fitting_segments)
        (
            val_observed_position,
            val_observed_velocity,
            val_validity,
        ) = _observations(validation_segments)

        initial_fit_position, initial_fit_velocity, elapsed = self.predictor.predict(
            fitting_segments, eta[None]
        )
        sensitivity_time += elapsed
        short_rollouts += len(fitting_segments)
        initial_fit_metrics = _loss_and_metrics(
            initial_fit_position[:, 0],
            initial_fit_velocity[:, 0],
            fit_observed_position,
            fit_observed_velocity,
            self.nominal_model.cable_length_m,
            fit_validity,
        )
        initial_val_position, initial_val_velocity, elapsed = self.predictor.predict(
            validation_segments, eta[None]
        )
        validation_time += elapsed
        short_rollouts += len(validation_segments)
        initial_val_metrics = _loss_and_metrics(
            initial_val_position[:, 0],
            initial_val_velocity[:, 0],
            val_observed_position,
            val_observed_velocity,
            self.nominal_model.cable_length_m,
            val_validity,
        )
        best_val = initial_val_metrics[0]
        best_fit = initial_fit_metrics[0]
        accepted_iterations = 0

        for iteration in range(self.settings.maximum_gn_iterations):
            hypotheses = np.repeat(eta[None], 5, axis=0)
            hypotheses[1, 0] += self.settings.finite_difference_log_step_ei
            hypotheses[2, 0] -= self.settings.finite_difference_log_step_ei
            hypotheses[3, 1] += self.settings.finite_difference_log_step_cb
            hypotheses[4, 1] -= self.settings.finite_difference_log_step_cb
            predictions, velocities, elapsed = self.predictor.predict(
                fitting_segments, hypotheses
            )
            sensitivity_time += elapsed
            short_rollouts += len(fitting_segments) * len(hypotheses)
            observed = fit_observed_position
            residual = _residual_vector(
                predictions[:, 0],
                observed,
                self.nominal_model.cable_length_m,
                fit_validity,
            )
            jacobian_e = (
                _residual_vector(
                    predictions[:, 1], observed, self.nominal_model.cable_length_m, fit_validity
                )
                - _residual_vector(
                    predictions[:, 2], observed, self.nominal_model.cable_length_m, fit_validity
                )
            ) / (2.0 * self.settings.finite_difference_log_step_ei)
            jacobian_c = (
                _residual_vector(
                    predictions[:, 3], observed, self.nominal_model.cable_length_m, fit_validity
                )
                - _residual_vector(
                    predictions[:, 4], observed, self.nominal_model.cable_length_m, fit_validity
                )
            ) / (2.0 * self.settings.finite_difference_log_step_cb)
            jacobian = np.stack((jacobian_e, jacobian_c), axis=1)
            if self.settings.parameter_mode == "ei_only":
                jacobian[:, 1] = 0.0
            elif self.settings.parameter_mode == "cb_only":
                jacobian[:, 0] = 0.0
            info_start = time.perf_counter()
            if self.settings.parameter_mode == "joint":
                last_information = _information(jacobian, self.settings)
            else:
                active = 0 if self.settings.parameter_mode == "ei_only" else 1
                scalar_information = float(np.dot(jacobian[:, active], jacobian[:, active]))
                accepted = (
                    math.isfinite(scalar_information)
                    and scalar_information > self.settings.minimum_information_eigenvalue
                )
                last_information = InformationDiagnostic(
                    scalar_information,
                    scalar_information,
                    1.0,
                    0.0,
                    accepted,
                    "accepted" if accepted else "minimum_information_eigenvalue",
                )
            information_time += time.perf_counter() - info_start
            if not last_information.accepted:
                break
            normal = jacobian.T @ jacobian + self.settings.lm_damping * np.eye(2)
            gradient = jacobian.T @ residual
            if self.settings.parameter_mode == "ei_only":
                normal[1] = (0.0, 1.0)
                normal[:, 1] = (0.0, 1.0)
                gradient[1] = 0.0
            elif self.settings.parameter_mode == "cb_only":
                normal[0] = (1.0, 0.0)
                normal[:, 0] = (1.0, 0.0)
                gradient[0] = 0.0
            delta = -np.linalg.solve(normal, gradient)
            delta[0] = np.clip(
                delta[0], -self.settings.maximum_log_step_ei, self.settings.maximum_log_step_ei
            )
            delta[1] = np.clip(
                delta[1], -self.settings.maximum_log_step_cb, self.settings.maximum_log_step_cb
            )
            lower = np.log(
                np.asarray(
                    (self.settings.ei_ratio_bounds[0], self.settings.cb_ratio_bounds[0])
                )
            )
            upper = np.log(
                np.asarray(
                    (self.settings.ei_ratio_bounds[1], self.settings.cb_ratio_bounds[1])
                )
            )
            line_candidates = np.stack(
                [np.clip(eta + alpha * delta, lower, upper) for alpha in self.settings.line_search_steps]
            )
            line_positions, line_velocities, elapsed = self.predictor.predict(
                validation_segments, line_candidates
            )
            line_search_time += elapsed
            short_rollouts += len(validation_segments) * len(line_candidates)
            candidate_metrics = [
                _loss_and_metrics(
                    line_positions[:, index],
                    line_velocities[:, index],
                    val_observed_position,
                    val_observed_velocity,
                    self.nominal_model.cable_length_m,
                    val_validity,
                )
                for index in range(len(line_candidates))
            ]
            selected_index = next(
                (
                    index
                    for index, metrics in enumerate(candidate_metrics)
                    if metrics[0]
                    < best_val - self.settings.validation_minimum_improvement
                ),
                None,
            )
            if selected_index is None:
                break
            eta = line_candidates[selected_index]
            best_val = candidate_metrics[selected_index][0]
            fit_position, fit_velocity, elapsed = self.predictor.predict(
                fitting_segments, eta[None]
            )
            sensitivity_time += elapsed
            short_rollouts += len(fitting_segments)
            fit_metrics = _loss_and_metrics(
                fit_position[:, 0],
                fit_velocity[:, 0],
                fit_observed_position,
                fit_observed_velocity,
                self.nominal_model.cable_length_m,
                fit_validity,
            )
            best_fit = fit_metrics[0]
            accepted_iterations += 1
            history.append(
                {
                    "iteration": float(iteration + 1),
                    "eta_e": float(eta[0]),
                    "eta_c": float(eta[1]),
                    "ei_ratio": float(math.exp(eta[0])),
                    "cb_ratio": float(math.exp(eta[1])),
                    "fit_loss": best_fit,
                    "validation_loss": best_val,
                    "step_norm": float(np.linalg.norm(delta)),
                    "line_search_alpha": float(self.settings.line_search_steps[selected_index]),
                }
            )

        accepted = accepted_iterations > 0 and best_val < initial_val_metrics[0] - self.settings.validation_minimum_improvement
        if not last_information.accepted:
            reason = f"information_rejected:{last_information.reason}"
        elif not accepted:
            reason = "held_out_validation_rejected"
        else:
            reason = "accepted"
        final_val_position, final_val_velocity, elapsed = self.predictor.predict(
            validation_segments, eta[None]
        )
        validation_time += elapsed
        short_rollouts += len(validation_segments)
        final_val_metrics = _loss_and_metrics(
            final_val_position[:, 0],
            final_val_velocity[:, 0],
            val_observed_position,
            val_observed_velocity,
            self.nominal_model.cable_length_m,
            val_validity,
        )
        initial_timing_error = _propagation_timing_error(
            initial_val_velocity[:, 0],
            val_observed_velocity,
            validation_segments,
        )
        final_timing_error = _propagation_timing_error(
            final_val_velocity[:, 0],
            val_observed_velocity,
            validation_segments,
        )
        candidate = ParameterEstimate(
            eta_e=float(eta[0] if accepted else initial_eta[0]),
            eta_c=float(eta[1] if accepted else initial_eta[1]),
            generation=current.generation + (1 if accepted else 0),
            published_at_s=time.time(),
            validation_loss=float(final_val_metrics[0] if accepted else initial_val_metrics[0]),
            source="distributed_lm" if accepted else current.source,
        )
        total = time.perf_counter() - started
        timing = FitTiming(
            sensitivity_time,
            information_time,
            line_search_time,
            validation_time,
            total,
            short_rollouts,
        )
        return DistributedFitResult(
            accepted=accepted,
            reason=reason,
            initial_estimate=current,
            candidate_estimate=candidate,
            information=last_information,
            fit_loss_before=initial_fit_metrics[0],
            fit_loss_after=best_fit,
            validation_loss_before=initial_val_metrics[0],
            validation_loss_after=final_val_metrics[0],
            all_node_position_rmse_before_m=initial_val_metrics[1],
            all_node_position_rmse_after_m=final_val_metrics[1],
            all_node_velocity_rmse_before_m_s=initial_val_metrics[2],
            all_node_velocity_rmse_after_m_s=final_val_metrics[2],
            tip_velocity_rmse_before_m_s=initial_val_metrics[3],
            tip_velocity_rmse_after_m_s=final_val_metrics[3],
            propagation_timing_error_before_s=initial_timing_error,
            propagation_timing_error_after_s=final_timing_error,
            iterations=accepted_iterations,
            selected_fit_segments=tuple(segment.start_time_s for segment in fitting_segments),
            selected_validation_segments=tuple(segment.start_time_s for segment in validation_segments),
            timing=timing,
            history=tuple(history),
        )


class OnlineAdaptationMonitor:
    """Causal low-cost monitor implementing persistence, hysteresis, and cooldown."""

    def __init__(
        self,
        predictor: BatchedMeasuredBoundaryPredictor,
        settings: DistributedAdaptationSettings,
    ) -> None:
        self.predictor = predictor
        self.settings = settings
        self.buffer = RollingDistributedBuffer(settings)
        self.ema_error_m2 = 0.0
        self._ema_initialized = False
        self._last_health_time_s = -math.inf
        self._high_since_s: float | None = None
        self._hysteresis_armed = True
        self._last_trigger_s = -math.inf
        self._trigger_in_flight = False
        self.diagnostics: list[HealthDiagnostic] = []

    def start_new_recording(self) -> None:
        """Prevent cross-strike segments without resetting trigger state."""

        self.buffer.start_new_recording()

    def start_new_plant_regime(self) -> None:
        """Rearm monitoring for a changed plant without resetting its model.

        Parameter publication and convergence history live outside this
        monitor.  Only health state and motion data tied to the former plant
        are cleared.  This lets a persistent controller estimate track a cable
        change while preventing mixed-regime fitting windows.
        """

        if self._trigger_in_flight:
            raise RuntimeError("Cannot change plant regime while a fit is in progress.")
        self.buffer.start_new_plant_regime()
        self.ema_error_m2 = 0.0
        self._ema_initialized = False
        self._last_health_time_s = -math.inf
        self._high_since_s = None
        self._hysteresis_armed = True
        self._last_trigger_s = -math.inf

    def append(self, observation: DistributedObservation) -> HealthDiagnostic | None:
        self.buffer.append(observation)
        if observation.timestamp_s - self._last_health_time_s < self.settings.health_evaluation_interval_s - 1.0e-9:
            return None
        recent = self.buffer.recent
        start_target = observation.timestamp_s - self.settings.health_horizon_s
        frames = [obs for obs in recent if obs.timestamp_s >= start_target - 1.0e-9]
        if len(frames) < 2 or frames[0].timestamp_s > start_target + 1.0e-6:
            return None
        if any(not frame.observation_valid for frame in frames):
            return None
        estimate = observation.active_estimate
        segment = _segment_from_observations(
            frames, _mean_excitation(frames, self.settings), "health"
        )
        prediction, _velocity, _elapsed = self.predictor.predict(
            (segment,), estimate.eta[None]
        )
        final_mask = segment.measurement_validity_mask[-1, 1:]
        if not np.any(final_mask):
            return None
        final_error = (
            prediction[0, 0, -1, 1:]
            - segment.cable_positions_m[-1, 1:]
        )
        error = float(np.mean(final_error[final_mask] ** 2))
        if not self._ema_initialized:
            self.ema_error_m2 = error
            self._ema_initialized = True
        else:
            alpha = self.settings.health_ema_alpha
            self.ema_error_m2 = alpha * self.ema_error_m2 + (1.0 - alpha) * error
        self._last_health_time_s = observation.timestamp_s
        if self.ema_error_m2 < self.settings.error_low_m2:
            self._hysteresis_armed = True
            self._high_since_s = None
        elif self.ema_error_m2 > self.settings.error_high_m2:
            if self._high_since_s is None:
                self._high_since_s = observation.timestamp_s
        else:
            self._high_since_s = None
        persistent = (
            self._high_since_s is not None
            and observation.timestamp_s - self._high_since_s >= self.settings.persistence_s
        )
        excitation = excitation_diagnostic(
            observation, recent[-2] if len(recent) > 1 else None, self.settings
        )
        if self.ema_error_m2 <= self.settings.error_high_m2:
            reason = "healthy"
        elif not persistent:
            reason = "persistence_pending"
        elif self._trigger_in_flight:
            reason = "fit_in_progress"
        elif not self._hysteresis_armed:
            reason = "hysteresis_disarmed"
        elif observation.timestamp_s - self._last_trigger_s < self.settings.cooldown_s:
            reason = "cooldown"
        elif excitation.score < self.settings.minimum_excitation:
            reason = "insufficient_excitation"
        else:
            reason = "fit_candidate"
        diagnostic = HealthDiagnostic(
            observation.timestamp_s,
            error,
            self.ema_error_m2,
            segment.time_s[-1] - segment.time_s[0],
            excitation,
            persistent,
            self._hysteresis_armed,
            reason,
        )
        self.diagnostics.append(diagnostic)
        return diagnostic

    def claim_trigger(self, diagnostic: HealthDiagnostic) -> bool:
        if diagnostic.reason != "fit_candidate" or self._trigger_in_flight:
            return False
        self._last_trigger_s = diagnostic.timestamp_s
        self._hysteresis_armed = False
        self._trigger_in_flight = True
        return True

    def complete_fit(self, *, published: bool) -> None:
        """Release one claimed trigger after rejection or atomic publication.

        Cooldown time and the informative cache intentionally survive both
        outcomes.  A rejected fit may therefore be retried with later data
        once cooldown expires, without requiring the still-mismatched model to
        cross the low hysteresis threshold first.  Publication additionally
        clears the health baseline accumulated under the old parameter
        generation; the first health prediction for the new generation starts
        a fresh EMA and persistence interval.
        """

        if not self._trigger_in_flight:
            raise RuntimeError("No claimed adaptation trigger is awaiting completion.")
        self._trigger_in_flight = False
        self._hysteresis_armed = True
        if published:
            self.ema_error_m2 = 0.0
            self._ema_initialized = False
            self._high_since_s = None


class AsynchronousDistributedAdapter:
    """Single-worker fitting queue whose submit/poll operations never wait."""

    def __init__(
        self,
        fitter: DistributedParameterFitter,
        store: AtomicParameterStore,
    ) -> None:
        self.fitter = fitter
        self.store = store
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cable-fit")
        self._future: Future[DistributedFitResult] | None = None
        self.completed: list[DistributedFitResult] = []

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def submit(
        self,
        fitting_segments: Sequence[AdaptationSegment],
        validation_segments: Sequence[AdaptationSegment],
    ) -> bool:
        if self.busy:
            return False
        current = self.store.snapshot()
        fit_copy = tuple(fitting_segments)
        validation_copy = tuple(validation_segments)
        self._future = self._executor.submit(
            self.fitter.fit, fit_copy, validation_copy, current
        )
        return True

    def poll(self) -> DistributedFitResult | None:
        if self._future is None or not self._future.done():
            return None
        result = self._future.result()
        self._future = None
        self.completed.append(result)
        if result.accepted:
            self.store.publish_if_current(
                result.candidate_estimate,
                expected_generation=result.initial_estimate.generation,
            )
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def snapshot_for_estimate(
    nominal: CableModelSnapshot, estimate: ParameterEstimate
) -> CableModelSnapshot:
    """Create a physically identical model snapshot with only EI/Cb changed."""

    return reduce_cable_model(
        nominal,
        node_count=nominal.node_count,
        substeps=nominal.model.parameters.substeps,
        constraint_iterations=nominal.model.parameters.constraint_iterations,
        bending_stiffness_scale=estimate.ei_ratio,
        bending_damping_scale=estimate.cb_ratio,
    )


@dataclass(frozen=True, slots=True)
class PublishedControllerRuntime:
    estimate: ParameterEstimate
    snapshot: CableModelSnapshot
    simulator: WhipSimulator
    rebuild_wall_time_s: float


class AtomicControllerRuntimeStore:
    """Atomic model+CUDA-runtime publication at an MPPI solve boundary.

    EI and Cb are tensors inside each captured simulator graph.  Consequently
    accepted estimates are materialized in a fresh simulator, optionally
    prewarmed outside the control loop, and then swapped as one object.
    """

    def __init__(
        self,
        nominal: CableModelSnapshot,
        simulation: SimulationSettings,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        self.nominal = nominal
        self.simulation = simulation
        self.device = torch.device(device)
        simulator = WhipSimulator(nominal, simulation, device=self.device)
        self._lock = Lock()
        self._runtime = PublishedControllerRuntime(
            ParameterEstimate(), nominal, simulator, 0.0
        )

    def snapshot(self) -> PublishedControllerRuntime:
        with self._lock:
            return self._runtime

    def prepare(
        self,
        estimate: ParameterEstimate,
        *,
        prewarm_batches: Iterable[tuple[int, int]] = (),
    ) -> PublishedControllerRuntime:
        started = time.perf_counter()
        snapshot = snapshot_for_estimate(self.nominal, estimate)
        simulator = WhipSimulator(snapshot, self.simulation, device=self.device)
        for batch_size, control_count in prewarm_batches:
            state = simulator.initial_state((0.0, 0.0, 1.5))
            controls = torch.zeros(
                (batch_size, control_count, 3),
                dtype=simulator.dtype,
                device=simulator.device,
            )
            simulator.rollout(state, controls, create_graph=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return PublishedControllerRuntime(
            estimate, snapshot, simulator, time.perf_counter() - started
        )

    def publish(self, runtime: PublishedControllerRuntime) -> bool:
        with self._lock:
            if runtime.estimate.generation <= self._runtime.estimate.generation:
                return False
            self._runtime = runtime
            return True


__all__ = [
    "ADAPTATION_SCHEMA",
    "AdaptationSegment",
    "AsynchronousDistributedAdapter",
    "AtomicControllerRuntimeStore",
    "AtomicParameterStore",
    "BatchedMeasuredBoundaryPredictor",
    "DistributedAdaptationSettings",
    "DistributedFitResult",
    "DistributedObservation",
    "DistributedParameterFitter",
    "ExcitationDiagnostic",
    "HealthDiagnostic",
    "InformationDiagnostic",
    "OnlineAdaptationMonitor",
    "ParameterEstimate",
    "PublishedControllerRuntime",
    "RollingDistributedBuffer",
    "curvature_binormals_numpy",
    "excitation_diagnostic",
    "select_informative_segments",
    "snapshot_for_estimate",
]
