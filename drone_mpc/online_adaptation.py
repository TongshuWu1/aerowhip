"""Between-strike EI/Cb adaptation for the accelerated online controller.

The control loop and fitting loop deliberately meet at only one atomic runtime
publication boundary.  A strike uses one immutable controller simulator.  Its
realized distributed trajectory is then processed after control has stopped;
an accepted fit is rebuilt and prewarmed before the next strike can use it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np

from .distributed_adaptation import (
    AtomicControllerRuntimeStore,
    DistributedAdaptationSettings,
    DistributedFitResult,
    DistributedObservation,
    DistributedParameterFitter,
    HealthDiagnostic,
    OnlineAdaptationMonitor,
    ParameterEstimate,
    PublishedControllerRuntime,
    select_informative_segments,
)
from .model import CableModelSnapshot
from .receding_mppi import RecedingMppiExecution
from .simulator import SimulationSettings


@dataclass(frozen=True, slots=True)
class BetweenStrikeAdaptationResult:
    """One completed between-strike monitor/fit/publication attempt."""

    triggered: bool
    fit_attempted: bool
    fit_result: DistributedFitResult | None
    published: bool
    reason: str
    observation_count: int
    candidate_segment_count: int
    fit_segment_count: int
    validation_segment_count: int
    monitoring_wall_time_s: float
    rebuild_wall_time_s: float
    health_diagnostic: HealthDiagnostic | None = None

    @property
    def accepted(self) -> bool:
        return bool(self.fit_result is not None and self.fit_result.accepted)


def observations_from_execution(
    execution: RecedingMppiExecution,
    simulation: SimulationSettings,
    estimate: ParameterEstimate,
) -> tuple[DistributedObservation, ...]:
    """Convert one realized strike into exact-state simulation observations.

    This is the simulation observation source.  A future Motive source should
    emit the same :class:`DistributedObservation` contract after causal state
    reconstruction; the fitter itself does not depend on simulator internals.
    """

    result = execution.result
    if result.frame_count < 2:
        return ()
    action_count = len(result.accelerations_m_s2)
    geometric_contact = bool(execution.cost_terms.get("geometric_tip_contact", 0.0))
    non_tip_contact = float(
        execution.cost_terms.get("non_tip_contact_violation", 0.0)
    ) > 0.0
    impact_frame = int(
        np.argmin(np.abs(np.asarray(result.time_s) - execution.impact_time_s))
    )
    observations: list[DistributedObservation] = []
    for frame, _timestamp in enumerate(result.time_s):
        if action_count:
            control_index = min(
                max((frame - 1) // simulation.steps_per_control, 0),
                action_count - 1,
            )
            action = result.accelerations_m_s2[control_index]
        else:
            action = np.zeros(3, dtype=np.float64)
        drone_state = np.concatenate(
            (result.drone_positions_m[frame], result.drone_velocities_m_s[frame])
        )
        observations.append(
            DistributedObservation(
                # Receding execution concatenates float32 segment clocks and
                # can accumulate sub-microsecond boundary jitter.  This exact
                # simulation source is sampled on the known physics clock, so
                # expose that canonical clock to the uniformly sampled fitter.
                timestamp_s=frame * simulation.simulation_dt_s,
                attachment_position_m=result.attachment_positions_m[frame],
                attachment_velocity_m_s=result.drone_velocities_m_s[frame],
                cable_positions_m=result.cable_positions_m[frame],
                cable_velocities_m_s=result.cable_velocities_m_s[frame],
                drone_state=drone_state,
                executed_action_m_s2=action,
                active_estimate=estimate,
                # Without a separately recorded non-tip contact timestamp, a
                # non-tip-first trial is conservatively excluded in full.  It
                # must never contaminate physical-parameter residuals.
                contact=non_tip_contact or (geometric_contact and frame >= impact_frame),
                safety_violation=(
                    execution.terminal_reason == "safety_violation"
                    and frame == result.frame_count - 1
                ),
                observation_valid=True,
            )
        )
    return tuple(observations)


class BetweenStrikeAdaptationSession:
    """Persistent model estimate and accelerated runtime across UI strikes."""

    def __init__(
        self,
        nominal: CableModelSnapshot,
        simulation: SimulationSettings,
        settings: DistributedAdaptationSettings = DistributedAdaptationSettings(),
        *,
        device: str = "cuda",
    ) -> None:
        self.nominal = nominal
        self.simulation = simulation
        self.settings = settings
        self.fitter = DistributedParameterFitter(nominal, settings, device=device)
        self.runtime_store = AtomicControllerRuntimeStore(
            nominal, simulation, device=device
        )
        self.runtime_store.snapshot().simulator.require_online_acceleration()
        self.monitor = OnlineAdaptationMonitor(self.fitter.predictor, self.settings)
        self._next_observation_time_s = 0.0
        self.plant_regime_count = 1
        self.history: list[BetweenStrikeAdaptationResult] = []

    @property
    def estimate(self) -> ParameterEstimate:
        return self.runtime_store.snapshot().estimate

    def runtime(self) -> PublishedControllerRuntime:
        runtime = self.runtime_store.snapshot()
        runtime.simulator.require_online_acceleration()
        return runtime

    def start_new_plant_regime(self) -> None:
        """Retain the learned model but isolate data from a changed plant."""

        self.monitor.start_new_plant_regime()
        self.plant_regime_count += 1

    def process_execution(
        self,
        execution: RecedingMppiExecution,
        *,
        prewarm_batch_size: int,
        prewarm_control_count: int,
    ) -> BetweenStrikeAdaptationResult:
        """Monitor, fit, validate, rebuild, and publish after one strike."""

        started = time.perf_counter()
        current = self.estimate
        local_observations = observations_from_execution(
            execution, self.simulation, current
        )
        # UI strikes are separate recordings whose local clocks restart at
        # zero.  Keep one monotonic session clock so health hysteresis,
        # cooldown, and the informative cache genuinely persist, but clear the
        # recent FIFO boundary so no fitting segment spans two initial states.
        self.monitor.start_new_recording()
        observations = tuple(
            replace(
                observation,
                timestamp_s=self._next_observation_time_s + observation.timestamp_s,
            )
            for observation in local_observations
        )
        if observations:
            self._next_observation_time_s = (
                observations[-1].timestamp_s + self.simulation.simulation_dt_s
            )
        trigger = None
        latest_diagnostic = None
        for observation in observations:
            diagnostic = self.monitor.append(observation)
            if diagnostic is not None:
                latest_diagnostic = diagnostic
                if diagnostic.reason == "fit_candidate":
                    trigger = diagnostic
        candidates = self.monitor.buffer.candidate_segments()
        fitting, validation = select_informative_segments(candidates, self.settings)
        monitoring_s = time.perf_counter() - started

        if trigger is None:
            reason = (
                latest_diagnostic.reason
                if latest_diagnostic is not None
                else "insufficient_health_history"
            )
            outcome = BetweenStrikeAdaptationResult(
                False,
                False,
                None,
                False,
                reason,
                len(observations),
                len(candidates),
                len(fitting),
                len(validation),
                monitoring_s,
                0.0,
                latest_diagnostic,
            )
            self.history.append(outcome)
            return outcome
        if not fitting or not validation:
            outcome = BetweenStrikeAdaptationResult(
                True,
                False,
                None,
                False,
                "waiting_for_informative_segments",
                len(observations),
                len(candidates),
                len(fitting),
                len(validation),
                monitoring_s,
                0.0,
                trigger,
            )
            self.history.append(outcome)
            return outcome
        if not self.monitor.claim_trigger(trigger):
            raise RuntimeError("A selected adaptation trigger could not be claimed.")

        try:
            fit_result = self.fitter.fit(fitting, validation, current)
        except Exception:
            self.monitor.complete_fit(published=False)
            raise
        if not fit_result.accepted:
            self.monitor.complete_fit(published=False)
            outcome = BetweenStrikeAdaptationResult(
                True,
                True,
                fit_result,
                False,
                fit_result.reason,
                len(observations),
                len(candidates),
                len(fitting),
                len(validation),
                monitoring_s,
                0.0,
                trigger,
            )
            self.history.append(outcome)
            return outcome

        candidate = fit_result.candidate_estimate
        try:
            runtime = self.runtime_store.prepare(
                candidate,
                prewarm_batches=(
                    (prewarm_batch_size, prewarm_control_count),
                    (1, prewarm_control_count),
                ),
            )
            runtime.simulator.require_online_acceleration()
            published = self.runtime_store.publish(runtime)
            if not published:
                raise RuntimeError(
                    "Accepted adaptation runtime was not atomically published."
                )
        except Exception:
            self.monitor.complete_fit(published=False)
            raise
        self.monitor.complete_fit(published=True)
        outcome = BetweenStrikeAdaptationResult(
            True,
            True,
            fit_result,
            True,
            "accepted_and_published",
            len(observations),
            len(candidates),
            len(fitting),
            len(validation),
            monitoring_s,
            runtime.rebuild_wall_time_s,
            trigger,
        )
        self.history.append(outcome)
        return outcome


__all__ = [
    "BetweenStrikeAdaptationResult",
    "BetweenStrikeAdaptationSession",
    "observations_from_execution",
]
