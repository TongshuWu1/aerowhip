"""Deterministic full-plant versus reduced-controller whip benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Callable, Iterable

import numpy as np
import torch

from .model import CableModelSnapshot
from .mpc import CostWeights, MpcPlan, MpcProblem, OptimizerSettings, optimize_controls
from .realtime import RealtimeSettings
from .reduced import stable_controller_model, transfer_dder_state
from .simulator import DroneCableState, SimulationSettings, WhipSimulator


BENCHMARK_SCHEMA = "drone_whip_full_plant_reduced_mpc_v3"
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    controller_ei_scale: float = 1.0
    controller_cb_scale: float = 1.0
    controller_node_count: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Benchmark case name cannot be empty.")
        if (
            not math.isfinite(self.controller_ei_scale)
            or self.controller_ei_scale <= 0.0
            or not math.isfinite(self.controller_cb_scale)
            or self.controller_cb_scale <= 0.0
        ):
            raise ValueError("Benchmark EI and Cb scales must be finite and positive.")
        if self.controller_node_count is not None and self.controller_node_count < 6:
            raise ValueError("Benchmark controller node count must be at least 6.")


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    case: str
    controller_ei_scale: float
    controller_cb_scale: float
    actual_hit_feasible: bool
    scheduled_impact_time_s: float
    impact_time_s: float
    tip_error_m: float
    directional_speed_m_s: float
    direction_error_deg: float
    maximum_drone_excursion_m: float
    minimum_drone_clearance_m: float
    maximum_drone_speed_m_s: float
    drone_speed_at_impact_m_s: float
    maximum_forward_stroke_m: float
    recoil_stroke_m: float
    directional_tip_energy_j: float
    replan_count: int
    infeasible_replans_rejected: int
    controller_deadline_misses: int
    initial_controller_solve_time_s: float
    maximum_controller_solve_time_s: float
    mean_controller_solve_time_s: float
    wall_time_s: float
    plant_model_sha256: str
    controller_model_sha256: str
    plant_node_count: int
    controller_node_count: int
    controller_substeps: int
    source_model_provisional: bool
    casting_forward_elevation_deg: float
    casting_recoil_deflection_deg: float
    casting_forward_excursion_m: float
    casting_recoil_excursion_m: float
    casting_reversal_fraction: float
    casting_motion_fraction: float


def default_mismatch_cases() -> tuple[BenchmarkCase, ...]:
    """Small, controlled local sensitivity study around the identified prior."""

    return (
        BenchmarkCase("matched", 1.0, 1.0),
        BenchmarkCase("EI -30%", 0.7, 1.0),
        BenchmarkCase("EI +30%", 1.3, 1.0),
        BenchmarkCase("Cb -30%", 1.0, 0.7),
        BenchmarkCase("Cb +30%", 1.0, 1.3),
    )


def default_resolution_cases(source_node_count: int) -> tuple[BenchmarkCase, ...]:
    """Controller grids used to select the smallest adequate online model."""

    if source_node_count < 6:
        raise ValueError("The source cable model must contain at least six nodes.")
    counts = tuple(
        dict.fromkeys(
            value for value in (7, 11, 15, source_node_count)
            if 6 <= value <= source_node_count
        )
    )
    return tuple(
        BenchmarkCase(f"{count}-node controller", controller_node_count=count)
        for count in counts
    )


def _controller_state(
    plant_state: DroneCableState,
    plant_snapshot: CableModelSnapshot,
    controller_snapshot: CableModelSnapshot,
) -> DroneCableState:
    return DroneCableState(
        plant_state.drone_position_m,
        plant_state.drone_velocity_m_s,
        transfer_dder_state(
            plant_state.cable,
            plant_snapshot,
            controller_snapshot,
        ),
    )


def _control_at(
    controls_m_s2: np.ndarray,
    epoch_s: float,
    time_s: float,
    control_interval_s: float,
) -> np.ndarray:
    ratio = (time_s - epoch_s) / control_interval_s
    index = int(math.floor(ratio + 1.0e-7))
    if 0 <= index < len(controls_m_s2):
        return np.asarray(controls_m_s2[index], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


def _snap_physics_time(value_s: float, physics_dt_s: float) -> float:
    return round(float(value_s) / physics_dt_s) * physics_dt_s


def _warm_start(
    controls_m_s2: np.ndarray,
    epoch_s: float,
    start_s: float,
    count: int,
    control_interval_s: float,
) -> np.ndarray:
    return np.stack(
        [
            _control_at(
                controls_m_s2,
                epoch_s,
                start_s + index * control_interval_s,
                control_interval_s,
            )
            for index in range(count)
        ],
        axis=0,
    )


def _impact_metrics(
    state: DroneCableState,
    problem: MpcProblem,
    *,
    maximum_excursion_m: float,
    minimum_clearance_m: float,
    maximum_drone_speed_m_s: float,
    maximum_speed_m_s: float,
    maximum_forward_stroke_m: float,
    recoil_stroke_m: float,
) -> tuple[bool, float, float, float]:
    tip = state.cable.positions_m[0, -1].detach().cpu().numpy()
    velocity = state.cable.velocities_m_s[0, -1].detach().cpu().numpy()
    target = np.asarray(problem.target_position_m, dtype=np.float64)
    direction = np.asarray(problem.impact_direction, dtype=np.float64)
    error = float(np.linalg.norm(tip - target))
    directed_speed = float(np.dot(velocity, direction))
    speed = float(np.linalg.norm(velocity))
    angle = 90.0 if speed <= 1.0e-9 else math.degrees(
        math.acos(np.clip(directed_speed / speed, -1.0, 1.0))
    )
    lateral_speed = float(np.linalg.norm(velocity - directed_speed * direction))
    inside_cone = lateral_speed <= (
        math.tan(math.radians(problem.maximum_impact_angle_deg)) * directed_speed
    )
    feasible = bool(
        error <= problem.maximum_tip_error_m
        and directed_speed >= problem.minimum_impact_speed_m_s
        and inside_cone
        and maximum_excursion_m <= problem.maximum_drone_excursion_m
        and minimum_clearance_m >= problem.drone_keepout_radius_m
        and maximum_drone_speed_m_s <= maximum_speed_m_s
        and maximum_forward_stroke_m >= problem.minimum_forward_stroke_m
        and recoil_stroke_m >= problem.minimum_recoil_stroke_m
    )
    return feasible, error, directed_speed, angle


def run_benchmark_case(
    source_snapshot: CableModelSnapshot,
    settings: RealtimeSettings,
    problem: MpcProblem,
    initial_drone_position_m: tuple[float, float, float],
    case: BenchmarkCase,
    *,
    device: str = "cuda",
    progress: ProgressCallback | None = None,
) -> BenchmarkResult:
    """Run one logical-time closed loop with oracle full-state observations.

    The plant is the full fitted DER.  MPC always rolls out a separate reduced
    DER and receives only a deterministic material-coordinate projection of the
    current plant state at each replan.  Controller computation does not pause
    logical simulation time; its measured wall time is reported against the
    requested replan deadline.
    """

    report = progress if progress is not None else (lambda _text: None)
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError("The canonical whip benchmark requires CUDA.")
    workspace_center = np.asarray(initial_drone_position_m, dtype=np.float64)
    active_problem = replace(
        problem,
        drone_workspace_center_m=tuple(float(value) for value in workspace_center),
    )
    controller_node_count = (
        settings.reduced_node_count
        if case.controller_node_count is None
        else case.controller_node_count
    )
    controller_snapshot = stable_controller_model(
        source_snapshot,
        simulation_dt_s=settings.physics_dt_s,
        node_count=controller_node_count,
        constraint_iterations=4,
        bending_stiffness_scale=case.controller_ei_scale,
        bending_damping_scale=case.controller_cb_scale,
    )
    plant = WhipSimulator(
        source_snapshot,
        SimulationSettings(
            horizon_s=settings.physics_dt_s,
            simulation_dt_s=settings.physics_dt_s,
            control_interval_s=settings.physics_dt_s,
            attachment_drop_m=settings.attachment_drop_m,
            maximum_acceleration_m_s2=settings.maximum_acceleration_m_s2,
            maximum_speed_m_s=settings.maximum_speed_m_s,
        ),
        device=torch_device,
    )
    planner = WhipSimulator(
        controller_snapshot,
        SimulationSettings(
            horizon_s=settings.horizon_s,
            simulation_dt_s=settings.physics_dt_s,
            control_interval_s=settings.control_interval_s,
            attachment_drop_m=settings.attachment_drop_m,
            maximum_acceleration_m_s2=settings.maximum_acceleration_m_s2,
            maximum_speed_m_s=settings.maximum_speed_m_s,
        ),
        device=torch_device,
    )
    initial_optimizer = OptimizerSettings(
        iterations=settings.initial_ipopt_iterations,
        tolerance=settings.ipopt_tolerance,
        acceptable_tolerance=settings.ipopt_acceptable_tolerance,
        maximum_wall_time_s=settings.initial_ipopt_max_wall_time_s,
        finite_difference_step=settings.ipopt_finite_difference_step,
        replan_interval_s=settings.replan_interval_s,
    )
    update_optimizer = replace(
        initial_optimizer,
        iterations=settings.ipopt_iterations,
        maximum_wall_time_s=settings.ipopt_max_wall_time_s,
    )
    weights = CostWeights()
    state = plant.initial_state(initial_drone_position_m)
    simulation_time_s = 0.0
    initial_controller_state = _controller_state(
        state, source_snapshot, controller_snapshot
    )
    initial_solve_started = time.perf_counter()
    initial_plan = optimize_controls(
        planner,
        initial_controller_state,
        active_problem,
        initial_optimizer,
        weights=weights,
        previous_acceleration_m_s2=np.zeros(3, dtype=np.float64),
        optimize_impact_time=True,
    )
    torch.cuda.synchronize(torch_device)
    initial_solve_time_s = time.perf_counter() - initial_solve_started
    if not initial_plan.feasible:
        raise RuntimeError(
            f"{case.name}: startup MPC found no feasible whip "
            f"(violation={initial_plan.constraint_violation:.4g})."
        )
    active_controls = initial_plan.controls_m_s2.copy()
    active_action = initial_plan.casting_action
    active_epoch_s = 0.0
    impact_time_s = _snap_physics_time(
        initial_plan.impact_time_s, settings.physics_dt_s
    )
    pending_controls: np.ndarray | None = None
    pending_action = None
    pending_epoch_s = math.inf
    pending_impact_time_s = math.inf
    online_solve_times: list[float] = []
    infeasible_replans = 0
    deadline_misses = 0
    replan_count = 0
    maximum_excursion_m = 0.0
    maximum_drone_speed_m_s = 0.0
    maximum_forward_stroke_m = 0.0
    recoil_stroke_m = 0.0
    horizontal_forward = (
        np.asarray(active_problem.target_position_m, dtype=np.float64)
        - workspace_center
    )
    horizontal_forward[2] = 0.0
    if np.linalg.norm(horizontal_forward) <= 1.0e-9:
        horizontal_forward = np.asarray(
            active_problem.impact_direction, dtype=np.float64
        ).copy()
        horizontal_forward[2] = 0.0
    if np.linalg.norm(horizontal_forward) <= 1.0e-9:
        horizontal_forward = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    horizontal_forward /= np.linalg.norm(horizontal_forward)
    minimum_clearance_m = float(
        np.linalg.norm(workspace_center - np.asarray(active_problem.target_position_m))
    )
    started = time.perf_counter()
    report(
        f"{case.name}: initial solve accepted "
        f"solve={initial_solve_time_s:.3f}s impact={impact_time_s:.3f}s"
    )

    while simulation_time_s < settings.horizon_s - 0.5 * settings.physics_dt_s:
        if simulation_time_s + 1.0e-9 >= pending_epoch_s:
            assert pending_controls is not None
            active_controls = pending_controls
            assert pending_action is not None
            active_action = pending_action
            active_epoch_s = pending_epoch_s
            impact_time_s = pending_impact_time_s
            pending_controls = None
            pending_action = None
            pending_epoch_s = math.inf
            pending_impact_time_s = math.inf
        on_replan = math.isclose(
            simulation_time_s / settings.replan_interval_s,
            round(simulation_time_s / settings.replan_interval_s),
            abs_tol=1.0e-7,
        )
        activation_time_s = _snap_physics_time(
            simulation_time_s + settings.replan_interval_s,
            settings.physics_dt_s,
        )
        if (
            on_replan
            and activation_time_s
            <= impact_time_s - settings.control_interval_s
            and activation_time_s < settings.horizon_s - 0.5 * settings.physics_dt_s
        ):
            remaining = int(
                math.floor(
                    (settings.horizon_s - activation_time_s)
                    / settings.control_interval_s
                    + 1.0e-7
                )
            )
            if remaining > 0:
                controller_state = _controller_state(
                    state, source_snapshot, controller_snapshot
                )
                delay_count = int(
                    round(
                        settings.replan_interval_s
                        / settings.control_interval_s
                    )
                )
                delay_controls = _warm_start(
                    active_controls,
                    active_epoch_s,
                    simulation_time_s,
                    delay_count,
                    settings.control_interval_s,
                )
                with torch.no_grad():
                    predicted_state = planner.rollout(
                        controller_state,
                        torch.as_tensor(
                            delay_controls,
                            dtype=planner.dtype,
                            device=torch_device,
                        )[None],
                        create_graph=False,
                    ).final_state()
                previous = _control_at(
                    active_controls,
                    active_epoch_s,
                    activation_time_s - 1.0e-7,
                    settings.control_interval_s,
                )
                solve_started = time.perf_counter()
                plan: MpcPlan = optimize_controls(
                    planner,
                    predicted_state,
                    active_problem,
                    update_optimizer,
                    control_count=remaining,
                    warm_start_action=active_action,
                    weights=weights,
                    previous_acceleration_m_s2=previous,
                    optimize_impact_time=True,
                )
                torch.cuda.synchronize(torch_device)
                solve_time = time.perf_counter() - solve_started
                online_solve_times.append(solve_time)
                late = solve_time > settings.replan_interval_s
                deadline_misses += int(late)
                replan_count += 1
                if plan.feasible and not late:
                    pending_controls = plan.controls_m_s2.copy()
                    pending_action = plan.casting_action
                    pending_epoch_s = activation_time_s
                    pending_impact_time_s = _snap_physics_time(
                        activation_time_s + plan.impact_time_s,
                        settings.physics_dt_s,
                    )
                else:
                    infeasible_replans += 1
                report(
                    f"{case.name}: replan {replan_count} "
                    f"{'accepted' if plan.feasible and not late else 'rejected'} "
                    f"solve={solve_time:.3f}s activation={activation_time_s:.3f}s "
                    f"impact={pending_impact_time_s if pending_controls is not None else impact_time_s:.3f}s"
                )

        acceleration = _control_at(
            active_controls,
            active_epoch_s,
            simulation_time_s,
            settings.control_interval_s,
        )
        control = torch.as_tensor(
            acceleration,
            dtype=plant.dtype,
            device=torch_device,
        ).reshape(1, 1, 3)
        with torch.no_grad():
            state = plant.rollout(state, control, create_graph=False).final_state()
        simulation_time_s += settings.physics_dt_s
        drone = state.drone_position_m[0].detach().cpu().numpy()
        drone_velocity = state.drone_velocity_m_s[0].detach().cpu().numpy()
        maximum_excursion_m = max(
            maximum_excursion_m,
            float(np.linalg.norm(drone - workspace_center)),
        )
        minimum_clearance_m = min(
            minimum_clearance_m,
            float(np.linalg.norm(drone - np.asarray(active_problem.target_position_m))),
        )
        maximum_drone_speed_m_s = max(
            maximum_drone_speed_m_s,
            float(np.linalg.norm(drone_velocity)),
        )
        forward_position = float(
            np.dot(drone - workspace_center, horizontal_forward)
        )
        maximum_forward_stroke_m = max(
            maximum_forward_stroke_m,
            forward_position,
        )
        recoil_stroke_m = max(
            0.0,
            maximum_forward_stroke_m - forward_position,
        )
        if simulation_time_s + 1.0e-9 >= impact_time_s:
            break

    feasible, error, speed, angle = _impact_metrics(
        state,
        active_problem,
        maximum_excursion_m=maximum_excursion_m,
        minimum_clearance_m=minimum_clearance_m,
        maximum_drone_speed_m_s=maximum_drone_speed_m_s,
        maximum_speed_m_s=settings.maximum_speed_m_s,
        maximum_forward_stroke_m=maximum_forward_stroke_m,
        recoil_stroke_m=recoil_stroke_m,
    )
    tip_mass_kg = float(source_snapshot.model.parameters.vertex_masses_kg[-1])
    directional_tip_energy_j = 0.5 * tip_mass_kg * max(0.0, speed) ** 2
    return BenchmarkResult(
        case=case.name,
        controller_ei_scale=case.controller_ei_scale,
        controller_cb_scale=case.controller_cb_scale,
        actual_hit_feasible=feasible,
        scheduled_impact_time_s=impact_time_s,
        impact_time_s=simulation_time_s,
        tip_error_m=error,
        directional_speed_m_s=speed,
        direction_error_deg=angle,
        maximum_drone_excursion_m=maximum_excursion_m,
        minimum_drone_clearance_m=minimum_clearance_m,
        maximum_drone_speed_m_s=maximum_drone_speed_m_s,
        drone_speed_at_impact_m_s=float(
            np.linalg.norm(state.drone_velocity_m_s[0].detach().cpu().numpy())
        ),
        maximum_forward_stroke_m=maximum_forward_stroke_m,
        recoil_stroke_m=recoil_stroke_m,
        directional_tip_energy_j=directional_tip_energy_j,
        replan_count=replan_count,
        infeasible_replans_rejected=infeasible_replans,
        controller_deadline_misses=deadline_misses,
        initial_controller_solve_time_s=initial_solve_time_s,
        maximum_controller_solve_time_s=max(online_solve_times, default=0.0),
        mean_controller_solve_time_s=(
            float(np.mean(online_solve_times)) if online_solve_times else 0.0
        ),
        wall_time_s=time.perf_counter() - started,
        plant_model_sha256=source_snapshot.sha256,
        controller_model_sha256=controller_snapshot.sha256,
        plant_node_count=source_snapshot.node_count,
        controller_node_count=controller_snapshot.node_count,
        controller_substeps=controller_snapshot.model.parameters.substeps,
        source_model_provisional=source_snapshot.provisional,
        casting_forward_elevation_deg=active_action.forward_elevation_deg,
        casting_recoil_deflection_deg=active_action.recoil_deflection_deg,
        casting_forward_excursion_m=active_action.forward_excursion_m,
        casting_recoil_excursion_m=active_action.recoil_excursion_m,
        casting_reversal_fraction=active_action.reversal_fraction,
        casting_motion_fraction=active_action.motion_fraction,
    )


def run_mismatch_benchmark(
    source_snapshot: CableModelSnapshot,
    settings: RealtimeSettings,
    problem: MpcProblem,
    initial_drone_position_m: tuple[float, float, float],
    *,
    cases: Iterable[BenchmarkCase] | None = None,
    device: str = "cuda",
    progress: ProgressCallback | None = None,
) -> tuple[BenchmarkResult, ...]:
    selected = tuple(default_mismatch_cases() if cases is None else cases)
    if not selected:
        raise ValueError("At least one benchmark case is required.")
    return tuple(
        run_benchmark_case(
            source_snapshot,
            settings,
            problem,
            initial_drone_position_m,
            case,
            device=device,
            progress=progress,
        )
        for case in selected
    )


def save_benchmark_results(
    path: str | Path,
    results: Iterable[BenchmarkResult],
    settings: RealtimeSettings,
    problem: MpcProblem,
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BENCHMARK_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "plant": "full fitted DER",
            "controller": "reduced DER",
            "control_parameterization": (
                "target-aligned two-sweep casting primitive; IPOPT optimizes "
                "sweep angle, curvature, wind-up/cast excursions, reversal "
                "time, and motion duration"
            ),
            "state_feedback": "oracle full plant state projected at every replan",
            "timing": "fixed logical simulation time; wall solve time reported separately",
            "replan_activation": (
                "controller state predicted one replan interval forward; "
                "late or infeasible replans rejected"
            ),
        },
        "settings": asdict(settings),
        "problem": asdict(problem),
        "results": [asdict(result) for result in results],
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination
