"""Full-state receding-horizon MPPI for nominal and mismatched DDER models.

The task objective and DDER equations remain defined by :mod:`drone_mpc.mppi`
and :mod:`drone_mpc.simulator`.  This module only closes the loop: it shifts a
previous MPPI solution, replans from the current distributed cable state, and
executes a short prefix in an independently constructed simulated plant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Callable, Literal

import numpy as np
import torch

from cable_twin.shared.dder import DderState

from .problem import MpcProblem
from .mppi import MppiPlan, MppiSettings, evaluate_mppi_rollout, optimize_mppi
from .simulator import (
    DroneCableState,
    SimulationSettings,
    SimulationResult,
    TensorRollout,
    WhipSimulator,
    tensor_rollout_to_result,
)


FeedbackMode = Literal["full", "endpoint"]
ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], bool]
StateTransform = Callable[[float, DroneCableState], DroneCableState]
ControllerStateProvider = Callable[[float, DroneCableState], DroneCableState]


@dataclass(frozen=True, slots=True)
class RecedingMppiSettings:
    """Execution settings outside the unchanged MPPI task objective."""

    replan_interval_s: float = 0.10
    timeout_s: float = 1.20
    feedback_mode: FeedbackMode = "full"

    def __post_init__(self) -> None:
        if not math.isfinite(self.replan_interval_s) or self.replan_interval_s <= 0.0:
            raise ValueError("Replan interval must be positive.")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("Execution timeout must be positive.")
        if self.feedback_mode not in {"full", "endpoint"}:
            raise ValueError("Feedback mode must be 'full' or 'endpoint'.")


@dataclass(frozen=True, slots=True)
class RecedingMppiUpdate:
    """One prediction, optimization, and executed prefix."""

    index: int
    start_time_s: float
    planning_wall_time_s: float
    executed_control_count: int
    rollout_count: int
    nominal_knots_m_s2: np.ndarray
    optimized_knots_m_s2: np.ndarray
    executed_controls_m_s2: np.ndarray
    knot_correction_l2_m_s2: float
    prefix_correction_l2_m_s2: float
    predicted_cost: float
    predicted_feasible: bool
    prediction: SimulationResult


@dataclass(frozen=True, slots=True)
class RecedingMppiExecution:
    """Closed-loop realization and all predictions made along it."""

    result: SimulationResult
    updates: tuple[RecedingMppiUpdate, ...]
    cost: float
    cost_terms: dict[str, float]
    impact_time_s: float
    feasible: bool
    terminal_reason: str
    total_rollouts: int
    total_planning_wall_time_s: float
    feedback_mode: FeedbackMode
    controller_model_sha256: str = ""
    plant_model_sha256: str = ""


@dataclass(frozen=True, slots=True)
class RecedingMppiLiveUpdate:
    """One completed MPC update, ready for thread-safe live visualization."""

    update: RecedingMppiUpdate
    realized: SimulationResult
    realized_cost: float
    realized_cost_terms: dict[str, float]
    realized_impact_time_s: float
    terminal_reason: str | None


LiveUpdateCallback = Callable[[RecedingMppiLiveUpdate], None]


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def _bound_numpy_vectors(values: np.ndarray, maximum_norm: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    norms = np.linalg.norm(result, axis=-1, keepdims=True)
    result *= np.minimum(1.0, maximum_norm / np.maximum(norms, 1.0e-12))
    return result


def shift_control_knots(
    knots_m_s2: np.ndarray,
    shift_s: float,
    horizon_s: float,
    maximum_acceleration_m_s2: float,
) -> np.ndarray:
    """Shift a fixed-horizon knot trajectory and hold its final tail value."""

    knots = np.asarray(knots_m_s2, dtype=np.float32)
    if knots.ndim != 2 or knots.shape[0] < 2 or knots.shape[1] != 3:
        raise ValueError("Control knots must have shape Mx3 with M >= 2.")
    if not math.isfinite(shift_s) or shift_s < 0.0:
        raise ValueError("Control shift must be finite and non-negative.")
    if not math.isfinite(horizon_s) or horizon_s <= 0.0:
        raise ValueError("Control horizon must be positive.")
    knot_times = np.linspace(0.0, horizon_s, knots.shape[0])
    sample_times = np.minimum(knot_times + shift_s, horizon_s)
    shifted = np.stack(
        [np.interp(sample_times, knot_times, knots[:, axis]) for axis in range(3)],
        axis=1,
    )
    return _bound_numpy_vectors(shifted, maximum_acceleration_m_s2)


def endpoint_conditioned_state(
    predicted_state: DroneCableState,
    observed_state: DroneCableState,
    attachment_drop_m: float,
) -> DroneCableState:
    """Condition a model-predicted cable state using only observed tip state.

    The controller retains its own DDER-predicted interior.  A smooth
    root-to-tip correction injects the observed tip position and velocity,
    while the known drone state fixes the attachment.  No true interior node
    is read by this function.
    """

    if predicted_state.batch_size != 1 or observed_state.batch_size != 1:
        raise ValueError("Endpoint conditioning expects single states.")
    predicted_positions = predicted_state.cable.positions_m
    predicted_velocities = predicted_state.cable.velocities_m_s
    observed_tip_position = observed_state.cable.positions_m[:, -1]
    observed_tip_velocity = observed_state.cable.velocities_m_s[:, -1]
    node_count = predicted_positions.shape[1]
    coordinate = torch.linspace(
        0.0,
        1.0,
        node_count,
        dtype=predicted_positions.dtype,
        device=predicted_positions.device,
    )
    # Smoothstep is zero with zero slope at the known root and equals one at
    # the observed free tip.
    weight = coordinate.square() * (3.0 - 2.0 * coordinate)
    position_delta = observed_tip_position - predicted_positions[:, -1]
    velocity_delta = observed_tip_velocity - predicted_velocities[:, -1]
    positions = predicted_positions + weight[None, :, None] * position_delta[:, None]
    velocities = predicted_velocities + weight[None, :, None] * velocity_delta[:, None]
    attachment = observed_state.drone_position_m + torch.as_tensor(
        ((0.0, 0.0, -attachment_drop_m),),
        dtype=positions.dtype,
        device=positions.device,
    )
    positions = positions.clone()
    velocities = velocities.clone()
    positions[:, 0] = attachment
    velocities[:, 0] = observed_state.drone_velocity_m_s
    return DroneCableState(
        observed_state.drone_position_m,
        observed_state.drone_velocity_m_s,
        DderState(positions, velocities),
    )


def perturb_cable_state(
    state: DroneCableState,
    *,
    position_delta_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    velocity_delta_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0),
    profile: Literal["sway", "interior", "distal"] = "sway",
) -> DroneCableState:
    """Apply a controlled distributed state perturbation without changing DDER."""

    positions = state.cable.positions_m
    node_count = positions.shape[1]
    coordinate = torch.linspace(
        0.0,
        1.0,
        node_count,
        dtype=positions.dtype,
        device=positions.device,
    )
    if profile == "sway":
        weight = coordinate.square() * (3.0 - 2.0 * coordinate)
    elif profile == "interior":
        weight = torch.sin(math.pi * coordinate)
    elif profile == "distal":
        weight = coordinate.square() * torch.sin(math.pi * coordinate)
    else:
        raise ValueError(f"Unknown cable perturbation profile: {profile}")
    position_delta = torch.as_tensor(
        position_delta_m, dtype=positions.dtype, device=positions.device
    )
    velocity_delta = torch.as_tensor(
        velocity_delta_m_s, dtype=positions.dtype, device=positions.device
    )
    perturbed_positions = (
        state.cable.positions_m + weight[None, :, None] * position_delta[None, None]
    )
    perturbed_velocities = (
        state.cable.velocities_m_s
        + weight[None, :, None] * velocity_delta[None, None]
    )
    return DroneCableState(
        state.drone_position_m,
        state.drone_velocity_m_s,
        DderState(perturbed_positions, perturbed_velocities),
    )


def _initial_rollout(
    state: DroneCableState,
    simulator: WhipSimulator,
) -> TensorRollout:
    attachment = state.drone_position_m + torch.as_tensor(
        ((0.0, 0.0, -simulator.settings.attachment_drop_m),),
        dtype=simulator.dtype,
        device=simulator.device,
    )
    return TensorRollout(
        time_s=torch.zeros((1,), dtype=simulator.dtype, device=simulator.device),
        drone_positions_m=state.drone_position_m[:, None],
        drone_velocities_m_s=state.drone_velocity_m_s[:, None],
        attachment_positions_m=attachment[:, None],
        cable_positions_m=state.cable.positions_m[:, None],
        cable_velocities_m_s=state.cable.velocities_m_s[:, None],
        accelerations_m_s2=torch.empty(
            (1, 0, 3), dtype=simulator.dtype, device=simulator.device
        ),
    )


def _replace_final_state(
    rollout: TensorRollout,
    state: DroneCableState,
    simulator: WhipSimulator,
) -> TensorRollout:
    drone_positions = rollout.drone_positions_m.clone()
    drone_velocities = rollout.drone_velocities_m_s.clone()
    cable_positions = rollout.cable_positions_m.clone()
    cable_velocities = rollout.cable_velocities_m_s.clone()
    attachments = rollout.attachment_positions_m.clone()
    drone_positions[:, -1] = state.drone_position_m
    drone_velocities[:, -1] = state.drone_velocity_m_s
    cable_positions[:, -1] = state.cable.positions_m
    cable_velocities[:, -1] = state.cable.velocities_m_s
    attachments[:, -1] = state.drone_position_m + torch.as_tensor(
        ((0.0, 0.0, -simulator.settings.attachment_drop_m),),
        dtype=simulator.dtype,
        device=simulator.device,
    )
    return TensorRollout(
        rollout.time_s,
        drone_positions,
        drone_velocities,
        attachments,
        cable_positions,
        cable_velocities,
        rollout.accelerations_m_s2,
    )


def _append_rollout(
    accumulated: TensorRollout,
    segment: TensorRollout,
    start_time_s: float,
) -> TensorRollout:
    return TensorRollout(
        time_s=torch.cat((accumulated.time_s, segment.time_s[1:] + start_time_s)),
        drone_positions_m=torch.cat(
            (accumulated.drone_positions_m, segment.drone_positions_m[:, 1:]), dim=1
        ),
        drone_velocities_m_s=torch.cat(
            (accumulated.drone_velocities_m_s, segment.drone_velocities_m_s[:, 1:]),
            dim=1,
        ),
        attachment_positions_m=torch.cat(
            (
                accumulated.attachment_positions_m,
                segment.attachment_positions_m[:, 1:],
            ),
            dim=1,
        ),
        cable_positions_m=torch.cat(
            (accumulated.cable_positions_m, segment.cable_positions_m[:, 1:]), dim=1
        ),
        cable_velocities_m_s=torch.cat(
            (accumulated.cable_velocities_m_s, segment.cable_velocities_m_s[:, 1:]),
            dim=1,
        ),
        accelerations_m_s2=torch.cat(
            (accumulated.accelerations_m_s2, segment.accelerations_m_s2), dim=1
        ),
    )


def _terminal_reason(terms: dict[str, float], elapsed_s: float, timeout_s: float) -> str | None:
    if bool(terms["geometric_tip_contact"]):
        return "valid_strike" if bool(terms["feasible"]) else "invalid_tip_contact"
    if float(terms["non_tip_contact_violation"]) > 0.0:
        return "non_tip_first"
    safety_names = (
        "workspace_violation",
        "keepout_violation",
        "speed_limit_violation",
        "ground_violation",
        "altitude_violation",
        "cable_drone_violation",
        "actuator_violation",
    )
    if any(float(terms[name]) > 0.0 for name in safety_names):
        return "safety_violation"
    if elapsed_s + 1.0e-9 >= timeout_s:
        return "timeout"
    return None


def run_receding_horizon_mppi(
    planner: WhipSimulator,
    plant: WhipSimulator,
    initial_state: DroneCableState,
    problem: MpcProblem,
    mppi_settings: MppiSettings,
    execution_settings: RecedingMppiSettings,
    initial_warm_start_knots_m_s2: np.ndarray,
    *,
    state_transform: StateTransform | None = None,
    controller_state_provider: ControllerStateProvider | None = None,
    progress: ProgressCallback | None = None,
    live_update: LiveUpdateCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> RecedingMppiExecution:
    """Execute shifted-warm-start MPPI from the current distributed state."""

    report = progress if progress is not None else (lambda _message: None)
    if planner.settings != plant.settings:
        raise ValueError("Planner and plant simulation settings must match.")
    planner_coordinates = np.asarray(
        planner.snapshot.rod_material_coordinates_m, dtype=np.float64
    )
    plant_coordinates = np.asarray(
        plant.snapshot.rod_material_coordinates_m, dtype=np.float64
    )
    if (
        planner.snapshot.node_count != plant.snapshot.node_count
        or planner_coordinates.shape != plant_coordinates.shape
        or not np.allclose(
            planner_coordinates, plant_coordinates, rtol=0.0, atol=1.0e-12
        )
    ):
        raise ValueError(
            "Controller and plant models must use the same material grid for "
            "full-state feedback."
        )
    ratio = execution_settings.replan_interval_s / plant.settings.control_interval_s
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Replan interval must be a multiple of the control interval.")
    controls_per_replan = max(1, int(round(ratio)))
    initial_controller_state = (
        initial_state
        if controller_state_provider is None
        else controller_state_provider(0.0, initial_state)
    )
    reference_state = initial_controller_state
    plant_state = initial_state
    observer_state = initial_controller_state
    accumulated = _initial_rollout(initial_state, plant)
    warm_knots = _bound_numpy_vectors(
        initial_warm_start_knots_m_s2,
        planner.settings.maximum_acceleration_m_s2,
    )
    if warm_knots.shape != (mppi_settings.knot_count, 3):
        raise ValueError(
            "Initial warm start shape does not match the MPPI knot count: "
            f"{warm_knots.shape} != {(mppi_settings.knot_count, 3)}"
        )
    elapsed = 0.0
    updates: list[RecedingMppiUpdate] = []
    terminal_reason: str | None = None
    last_terms: dict[str, float] | None = None
    last_cost = math.inf
    last_impact_frame = 0

    while terminal_reason is None:
        if cancelled is not None and cancelled():
            raise RuntimeError("Receding-horizon MPPI stopped by user.")
        if state_transform is not None:
            transformed = state_transform(elapsed, plant_state)
            if transformed is not plant_state:
                plant_state = transformed
                accumulated = _replace_final_state(accumulated, plant_state, plant)
        if execution_settings.feedback_mode == "full":
            controller_state = (
                plant_state
                if controller_state_provider is None
                else controller_state_provider(elapsed, plant_state)
            )
        else:
            controller_state = endpoint_conditioned_state(
                observer_state,
                plant_state,
                planner.settings.attachment_drop_m,
            )

        update_index = len(updates)
        settings_this_update = replace(
            mppi_settings, seed=mppi_settings.seed + update_index
        )
        report(
            f"Replan {update_index + 1}: t={elapsed:.3f}s, "
            f"feedback={execution_settings.feedback_mode}, "
            f"warm-start=shifted"
        )
        if planner.device.type == "cuda":
            torch.cuda.synchronize(planner.device)
        started = time.perf_counter()
        plan: MppiPlan = optimize_mppi(
            planner,
            controller_state,
            problem,
            settings_this_update,
            warm_start_knots_m_s2=warm_knots,
            objective_reference_state=reference_state,
            progress=None,
            cancelled=cancelled,
        )
        if planner.device.type == "cuda":
            torch.cuda.synchronize(planner.device)
        planning_wall = time.perf_counter() - started
        remaining_controls = max(
            1,
            int(
                math.ceil(
                    (execution_settings.timeout_s - elapsed)
                    / plant.settings.control_interval_s
                    - 1.0e-9
                )
            ),
        )
        apply_count = min(controls_per_replan, remaining_controls)
        controls_array = np.array(
            plan.controls_m_s2[:apply_count], dtype=np.float32, copy=True
        )
        nominal_controls = np.stack(
            [
                np.interp(
                    np.linspace(0.0, planner.settings.horizon_s, planner.settings.control_count)[
                        :apply_count
                    ],
                    np.linspace(0.0, planner.settings.horizon_s, warm_knots.shape[0]),
                    warm_knots[:, axis],
                )
                for axis in range(3)
            ],
            axis=1,
        )
        controls = torch.as_tensor(
            controls_array, dtype=plant.dtype, device=plant.device
        )[None]
        with torch.no_grad():
            plant_segment = plant.rollout(plant_state, controls, create_graph=False)
            observer_segment = planner.rollout(
                controller_state, controls.to(planner.device), create_graph=False
            )
        ingest_rollout = (
            None
            if controller_state_provider is None
            else getattr(controller_state_provider, "ingest_plant_rollout", None)
        )
        if ingest_rollout is not None:
            ingest_rollout(elapsed, plant_segment)
        accumulated = _append_rollout(accumulated, plant_segment, elapsed)
        plant_state = plant_segment.final_state()
        observer_state = observer_segment.final_state()
        segment_duration = apply_count * plant.settings.control_interval_s
        update = RecedingMppiUpdate(
            index=update_index,
            start_time_s=elapsed,
            planning_wall_time_s=planning_wall,
            executed_control_count=apply_count,
            rollout_count=settings_this_update.samples * settings_this_update.iterations,
            nominal_knots_m_s2=_readonly(warm_knots),
            optimized_knots_m_s2=_readonly(plan.control_knots_m_s2),
            executed_controls_m_s2=_readonly(controls_array),
            knot_correction_l2_m_s2=float(
                np.linalg.norm(plan.control_knots_m_s2 - warm_knots)
            ),
            prefix_correction_l2_m_s2=float(
                np.linalg.norm(controls_array - nominal_controls)
            ),
            predicted_cost=plan.cost,
            predicted_feasible=plan.feasible,
            prediction=plan.prediction,
        )
        updates.append(update)
        elapsed += segment_duration
        warm_knots = shift_control_knots(
            plan.control_knots_m_s2,
            segment_duration,
            planner.settings.horizon_s,
            planner.settings.maximum_acceleration_m_s2,
        )
        with torch.no_grad():
            cost, diagnostics, impact_frames = evaluate_mppi_rollout(
                accumulated,
                reference_state,
                problem,
                plant,
                mppi_settings,
            )
        last_cost = float(cost[0].detach().cpu())
        last_terms = {
            name: float(value[0].detach().cpu())
            for name, value in diagnostics.items()
        }
        last_impact_frame = int(impact_frames[0].detach().cpu())
        terminal_reason = _terminal_reason(
            last_terms, elapsed, execution_settings.timeout_s
        )
        if live_update is not None:
            live_result = tensor_rollout_to_result(
                accumulated,
                batch_index=0,
                target_position_m=torch.as_tensor(
                    problem.target_position_m, dtype=plant.dtype
                ),
                impact_direction=torch.as_tensor(
                    problem.impact_direction, dtype=plant.dtype
                ),
                model_sha256=plant.snapshot.sha256,
            )
            live_update(
                RecedingMppiLiveUpdate(
                    update=update,
                    realized=live_result,
                    realized_cost=last_cost,
                    realized_cost_terms=dict(last_terms),
                    realized_impact_time_s=last_terms["impact_time_s"],
                    terminal_reason=terminal_reason,
                )
            )
        report(
            f"  planned in {planning_wall:.3f}s; execute={segment_duration:.3f}s; "
            f"realized error={1000.0 * last_terms['position_error_m']:.1f}mm; "
            f"terminal={terminal_reason or 'continue'}"
        )

    assert last_terms is not None
    target = torch.as_tensor(problem.target_position_m, dtype=plant.dtype)
    direction = torch.as_tensor(problem.impact_direction, dtype=plant.dtype)
    result = tensor_rollout_to_result(
        accumulated,
        batch_index=0,
        target_position_m=target,
        impact_direction=direction,
        model_sha256=plant.snapshot.sha256,
    )
    return RecedingMppiExecution(
        result=result,
        updates=tuple(updates),
        cost=last_cost,
        cost_terms=last_terms,
        impact_time_s=last_terms["impact_time_s"],
        feasible=bool(last_terms["feasible"]),
        terminal_reason=terminal_reason,
        total_rollouts=sum(update.rollout_count for update in updates),
        total_planning_wall_time_s=sum(
            update.planning_wall_time_s for update in updates
        ),
        feedback_mode=execution_settings.feedback_mode,
        controller_model_sha256=planner.snapshot.sha256,
        plant_model_sha256=plant.snapshot.sha256,
    )


def replay_open_loop(
    plant: WhipSimulator,
    initial_state: DroneCableState,
    problem: MpcProblem,
    mppi_settings: MppiSettings,
    controls_m_s2: np.ndarray,
    *,
    observation_interval_s: float,
    timeout_s: float,
    state_transform: StateTransform | None = None,
    cancelled: CancellationCallback | None = None,
) -> RecedingMppiExecution:
    """Replay fixed controls with the same disturbance and terminal handling."""

    ratio = observation_interval_s / plant.settings.control_interval_s
    if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("Observation interval must be a multiple of control interval.")
    controls_per_block = max(1, int(round(ratio)))
    controls = _bound_numpy_vectors(
        controls_m_s2, plant.settings.maximum_acceleration_m_s2
    )
    maximum_controls = min(
        len(controls),
        int(math.ceil(timeout_s / plant.settings.control_interval_s - 1.0e-9)),
    )
    reference_state = initial_state
    plant_state = initial_state
    accumulated = _initial_rollout(initial_state, plant)
    elapsed = 0.0
    terminal_reason: str | None = None
    last_terms: dict[str, float] | None = None
    last_cost = math.inf
    last_impact_frame = 0
    control_index = 0
    while terminal_reason is None and control_index < maximum_controls:
        if cancelled is not None and cancelled():
            raise RuntimeError("Open-loop replay stopped by user.")
        if state_transform is not None:
            transformed = state_transform(elapsed, plant_state)
            if transformed is not plant_state:
                plant_state = transformed
                accumulated = _replace_final_state(accumulated, plant_state, plant)
        stop = min(control_index + controls_per_block, maximum_controls)
        segment_controls = torch.as_tensor(
            controls[control_index:stop], dtype=plant.dtype, device=plant.device
        )[None]
        with torch.no_grad():
            segment = plant.rollout(
                plant_state, segment_controls, create_graph=False
            )
        accumulated = _append_rollout(accumulated, segment, elapsed)
        plant_state = segment.final_state()
        elapsed += (stop - control_index) * plant.settings.control_interval_s
        control_index = stop
        with torch.no_grad():
            cost, diagnostics, impact_frames = evaluate_mppi_rollout(
                accumulated,
                reference_state,
                problem,
                plant,
                mppi_settings,
            )
        last_cost = float(cost[0].detach().cpu())
        last_terms = {
            name: float(value[0].detach().cpu())
            for name, value in diagnostics.items()
        }
        last_impact_frame = int(impact_frames[0].detach().cpu())
        terminal_reason = _terminal_reason(last_terms, elapsed, timeout_s)
    if terminal_reason is None:
        terminal_reason = "timeout"
    assert last_terms is not None
    result = tensor_rollout_to_result(
        accumulated,
        batch_index=0,
        target_position_m=torch.as_tensor(problem.target_position_m),
        impact_direction=torch.as_tensor(problem.impact_direction),
        model_sha256=plant.snapshot.sha256,
    )
    return RecedingMppiExecution(
        result=result,
        updates=(),
        cost=last_cost,
        cost_terms=last_terms,
        impact_time_s=last_terms["impact_time_s"],
        feasible=bool(last_terms["feasible"]),
        terminal_reason=terminal_reason,
        total_rollouts=0,
        total_planning_wall_time_s=0.0,
        feedback_mode="full",
        controller_model_sha256=plant.snapshot.sha256,
        plant_model_sha256=plant.snapshot.sha256,
    )


def save_receding_mppi_execution(
    path: str | Path,
    execution: RecedingMppiExecution,
    problem: MpcProblem,
    mppi_settings: MppiSettings,
    execution_settings: RecedingMppiSettings,
    simulation_settings: SimulationSettings,
    *,
    warm_start_source: str | Path | None = None,
    model_provenance: dict[str, object] | None = None,
) -> Path:
    """Save the realized closed loop and every prediction made while replanning."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = execution.result
    arrays: dict[str, np.ndarray] = {
        "simulation_node_count": np.asarray(
            result.cable_positions_m.shape[1], dtype=np.int32
        ),
        "controller_model_sha256": np.asarray(
            execution.controller_model_sha256 or result.model_sha256
        ),
        "plant_model_sha256": np.asarray(
            execution.plant_model_sha256 or result.model_sha256
        ),
        "time_s": result.time_s,
        "drone_positions_m": result.drone_positions_m,
        "drone_velocities_m_s": result.drone_velocities_m_s,
        "attachment_positions_m": result.attachment_positions_m,
        "cable_positions_m": result.cable_positions_m,
        "cable_velocities_m_s": result.cable_velocities_m_s,
        "accelerations_m_s2": result.accelerations_m_s2,
        "target_position_m": result.target_position_m,
        "impact_direction": result.impact_direction,
    }
    if execution.updates:
        arrays.update(
            {
                "prediction_start_times_s": np.asarray(
                    [update.start_time_s for update in execution.updates],
                    dtype=np.float32,
                ),
                "replanning_wall_times_s": np.asarray(
                    [update.planning_wall_time_s for update in execution.updates],
                    dtype=np.float32,
                ),
                "nominal_knots_m_s2": np.stack(
                    [update.nominal_knots_m_s2 for update in execution.updates]
                ),
                "optimized_knots_m_s2": np.stack(
                    [update.optimized_knots_m_s2 for update in execution.updates]
                ),
                "executed_controls_by_update_m_s2": np.concatenate(
                    [update.executed_controls_m_s2 for update in execution.updates],
                    axis=0,
                ),
                "executed_control_counts": np.asarray(
                    [update.executed_control_count for update in execution.updates],
                    dtype=np.int32,
                ),
                "predicted_drone_positions_m": np.stack(
                    [update.prediction.drone_positions_m for update in execution.updates]
                ),
                "predicted_cable_positions_m": np.stack(
                    [update.prediction.cable_positions_m for update in execution.updates]
                ),
                "predicted_cable_velocities_m_s": np.stack(
                    [update.prediction.cable_velocities_m_s for update in execution.updates]
                ),
                "predicted_accelerations_m_s2": np.stack(
                    [update.prediction.accelerations_m_s2 for update in execution.updates]
                ),
            }
        )
    np.savez_compressed(output, **arrays)
    metadata = {
        "schema": "receding_horizon_dder_mppi_execution_v2",
        "model_sha256": result.model_sha256,
        "controller_model_sha256": (
            execution.controller_model_sha256 or result.model_sha256
        ),
        "plant_model_sha256": execution.plant_model_sha256 or result.model_sha256,
        "model_provenance": {} if model_provenance is None else model_provenance,
        "simulation_node_count": int(result.cable_positions_m.shape[1]),
        "warm_start_source": (
            None
            if warm_start_source is None
            else str(Path(warm_start_source).expanduser().resolve())
        ),
        "feedback_mode": execution.feedback_mode,
        "feasible": execution.feasible,
        "terminal_reason": execution.terminal_reason,
        "cost": execution.cost,
        "impact_time_s": execution.impact_time_s,
        "total_rollouts": execution.total_rollouts,
        "total_planning_wall_time_s": execution.total_planning_wall_time_s,
        "problem": asdict(problem),
        "mppi_settings": asdict(mppi_settings),
        "execution_settings": asdict(execution_settings),
        "simulation_settings": asdict(simulation_settings),
        "cost_terms": execution.cost_terms,
        "updates": [
            {
                "index": update.index,
                "start_time_s": update.start_time_s,
                "planning_wall_time_s": update.planning_wall_time_s,
                "executed_control_count": update.executed_control_count,
                "rollout_count": update.rollout_count,
                "knot_correction_l2_m_s2": update.knot_correction_l2_m_s2,
                "prefix_correction_l2_m_s2": update.prefix_correction_l2_m_s2,
                "predicted_cost": update.predicted_cost,
                "predicted_feasible": update.predicted_feasible,
            }
            for update in execution.updates
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output
