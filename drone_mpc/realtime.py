"""Continuous matched-model plant and asynchronous receding-horizon MPC."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from collections import deque
import math
import threading
import time

import numpy as np
import torch

from cable_twin.shared.dder import DderState

from .model import CableModelSnapshot
from .mpc import (
    CastingPhaseSchedule,
    CostWeights,
    MpcPlan,
    MpcProblem,
    OptimizerSettings,
    optimize_controls,
)
from .reduced import stable_controller_model, transfer_dder_state
from .simulator import DroneCableState, SimulationSettings, WhipSimulator


@dataclass(frozen=True, slots=True)
class RealtimeSettings:
    physics_dt_s: float = 0.01
    horizon_s: float = 0.40
    control_interval_s: float = 0.02
    replan_interval_s: float = 0.10
    mission_duration_s: float = 4.0
    injection_steps: int = 10
    matched_model: bool = True
    reduced_node_count: int = 7
    initial_ipopt_iterations: int = 40
    ipopt_iterations: int = 8
    ipopt_tolerance: float = 1.0e-4
    ipopt_acceptable_tolerance: float = 1.0e-3
    initial_ipopt_max_wall_time_s: float = 3.0
    ipopt_max_wall_time_s: float = 0.09
    ipopt_finite_difference_step: float = 1.0e-3
    attachment_drop_m: float = 0.10
    maximum_acceleration_m_s2: float = 20.0
    maximum_speed_m_s: float = 3.0
    controller_ei_scale: float = 1.0
    controller_cb_scale: float = 1.0
    acceleration_effort_weight: float = 1.0
    acceleration_smoothness_weight: float = 1.0

    def __post_init__(self) -> None:
        positive = (
            self.physics_dt_s,
            self.horizon_s,
            self.control_interval_s,
            self.replan_interval_s,
            self.mission_duration_s,
            self.attachment_drop_m,
            self.maximum_acceleration_m_s2,
            self.maximum_speed_m_s,
            self.controller_ei_scale,
            self.controller_cb_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Real-time MPC times and limits must be positive.")
        if not math.isclose(
            self.control_interval_s / self.physics_dt_s,
            round(self.control_interval_s / self.physics_dt_s),
            abs_tol=1.0e-9,
        ):
            raise ValueError("Control interval must be a multiple of physics dt.")
        if not math.isclose(
            self.horizon_s / self.control_interval_s,
            round(self.horizon_s / self.control_interval_s),
            abs_tol=1.0e-9,
        ):
            raise ValueError("Horizon must be a multiple of control interval.")
        if not math.isclose(
            self.replan_interval_s / self.control_interval_s,
            round(self.replan_interval_s / self.control_interval_s),
            abs_tol=1.0e-9,
        ):
            raise ValueError("Replan interval must be a multiple of control interval.")
        if self.replan_interval_s > self.horizon_s:
            raise ValueError("Applied controls per replan cannot exceed the horizon.")
        if not 1 <= self.injection_steps < self.horizon_steps:
            raise ValueError(
                "Injection steps must be at least one and smaller than the horizon."
            )
        if self.mission_duration_s < self.horizon_s:
            raise ValueError("Mission duration must be at least one MPC horizon.")
        if self.reduced_node_count < 3:
            raise ValueError("Reduced controller model requires at least three nodes.")
        if (
            not 1 <= self.initial_ipopt_iterations <= 500
            or not 1 <= self.ipopt_iterations <= 500
        ):
            raise ValueError("IPOPT iteration counts must be between 1 and 500.")
        if (
            not math.isfinite(self.ipopt_tolerance)
            or self.ipopt_tolerance <= 0.0
            or not math.isfinite(self.ipopt_acceptable_tolerance)
            or self.ipopt_acceptable_tolerance < self.ipopt_tolerance
            or not math.isfinite(self.initial_ipopt_max_wall_time_s)
            or self.initial_ipopt_max_wall_time_s <= 0.0
            or not math.isfinite(self.ipopt_max_wall_time_s)
            or self.ipopt_max_wall_time_s <= 0.0
            or not math.isfinite(self.ipopt_finite_difference_step)
            or not 1.0e-6 <= self.ipopt_finite_difference_step <= 0.1
        ):
            raise ValueError("IPOPT tolerances, timing, or derivative step are invalid.")
        regularization = (
            self.acceleration_effort_weight,
            self.acceleration_smoothness_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in regularization):
            raise ValueError("MPC regularization weights must be finite and non-negative.")
        if sum(regularization) <= 0.0:
            raise ValueError("At least one MPC regularization weight must be positive.")

    @property
    def horizon_steps(self) -> int:
        return int(round(self.horizon_s / self.control_interval_s))

    @property
    def apply_steps(self) -> int:
        return int(round(self.replan_interval_s / self.control_interval_s))


@dataclass(frozen=True, slots=True)
class LiveFrame:
    simulation_time_s: float
    wall_elapsed_s: float
    drone_position_m: np.ndarray
    drone_velocity_m_s: np.ndarray
    attachment_position_m: np.ndarray
    cable_positions_m: np.ndarray
    cable_velocities_m_s: np.ndarray
    commanded_acceleration_m_s2: np.ndarray
    target_position_m: np.ndarray
    impact_direction: np.ndarray
    controller_solve_time_s: float | None
    controller_rate_hz: float | None
    controller_deadline_missed: bool
    impact_time_s: float
    mission_duration_s: float
    casting_phase: str
    mission_complete: bool
    maximum_drone_excursion_m: float
    drone_excursion_limit_m: float
    impact_tip_error_m: float | None
    impact_directional_speed_m_s: float | None
    impact_direction_error_deg: float | None
    impact_drone_clearance_m: float | None
    actual_hit_feasible: bool | None
    minimum_drone_clearance_m: float
    maximum_drone_speed_m_s: float
    maximum_forward_stroke_m: float
    recoil_stroke_m: float
    impact_directional_tip_energy_j: float | None
    plan: MpcPlan | None


def _clone_state(state: DroneCableState) -> DroneCableState:
    return DroneCableState(
        state.drone_position_m.clone(),
        state.drone_velocity_m_s.clone(),
        DderState(
            state.cable.positions_m.clone(),
            state.cable.velocities_m_s.clone(),
        ),
    )


class RealtimeMpcSession:
    """Run the plant continuously while an independent worker replans."""

    def __init__(
        self,
        source_snapshot: CableModelSnapshot,
        settings: RealtimeSettings,
        problem: MpcProblem,
        initial_drone_position_m: tuple[float, float, float],
        *,
        device: str = "cuda",
    ) -> None:
        self.settings = settings
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Real-time MPC requires CUDA.")
        self.plant_snapshot = source_snapshot
        if settings.matched_model:
            self.controller_snapshot = source_snapshot
        else:
            self.controller_snapshot = stable_controller_model(
                source_snapshot,
                simulation_dt_s=settings.physics_dt_s,
                node_count=settings.reduced_node_count,
                constraint_iterations=4,
                bending_stiffness_scale=settings.controller_ei_scale,
                bending_damping_scale=settings.controller_cb_scale,
            )
        # ``snapshot`` remains the public controller-model handle used by the
        # viewer/archive.  In the canonical baseline it is exactly the plant
        # model; the mismatch path is retained only for controlled ablations.
        self.snapshot = self.controller_snapshot
        plant_settings = SimulationSettings(
            horizon_s=settings.physics_dt_s,
            simulation_dt_s=settings.physics_dt_s,
            control_interval_s=settings.physics_dt_s,
            attachment_drop_m=settings.attachment_drop_m,
            maximum_acceleration_m_s2=settings.maximum_acceleration_m_s2,
            maximum_speed_m_s=settings.maximum_speed_m_s,
        )
        planning_settings = SimulationSettings(
            horizon_s=settings.horizon_s,
            simulation_dt_s=settings.physics_dt_s,
            control_interval_s=settings.control_interval_s,
            attachment_drop_m=settings.attachment_drop_m,
            maximum_acceleration_m_s2=settings.maximum_acceleration_m_s2,
            maximum_speed_m_s=settings.maximum_speed_m_s,
        )
        self.plant = WhipSimulator(
            self.plant_snapshot, plant_settings, device=self.device
        )
        self.planner = WhipSimulator(
            self.controller_snapshot, planning_settings, device=self.device
        )
        self.optimizer = OptimizerSettings(
            iterations=settings.ipopt_iterations,
            tolerance=settings.ipopt_tolerance,
            acceptable_tolerance=settings.ipopt_acceptable_tolerance,
            maximum_wall_time_s=settings.ipopt_max_wall_time_s,
            finite_difference_step=settings.ipopt_finite_difference_step,
            replan_interval_s=settings.replan_interval_s,
        )
        self.initial_optimizer = OptimizerSettings(
            iterations=settings.initial_ipopt_iterations,
            tolerance=settings.ipopt_tolerance,
            acceptable_tolerance=settings.ipopt_acceptable_tolerance,
            maximum_wall_time_s=settings.initial_ipopt_max_wall_time_s,
            finite_difference_step=settings.ipopt_finite_difference_step,
            replan_interval_s=settings.replan_interval_s,
        )
        self.cost_weights = CostWeights(
            acceleration_effort=settings.acceleration_effort_weight,
            acceleration_smoothness=settings.acceleration_smoothness_weight,
        )
        self._state = self.plant.initial_state(initial_drone_position_m)
        self._workspace_center_m = tuple(float(v) for v in initial_drone_position_m)
        self._problem = replace(
            problem,
            drone_workspace_center_m=self._workspace_center_m,
        )
        self._simulation_time_s = 0.0
        self._active_controls: np.ndarray | None = None
        self._active_epoch_s = 0.0
        self._pending_controls: np.ndarray | None = None
        self._pending_epoch_s = math.inf
        self._pending_impact_time_s: float | None = None
        self._latest_plan: MpcPlan | None = None
        self._latest_frame: LiveFrame | None = None
        self._history: deque[LiveFrame] = deque(maxlen=30000)
        self._last_solve_s: float | None = None
        self._last_controller_rate_hz: float | None = None
        self._deadline_missed = False
        self._impact_time_s = math.inf
        self._mission_complete = False
        self._maximum_drone_excursion_m = 0.0
        self._impact_tip_error_m: float | None = None
        self._impact_directional_speed_m_s: float | None = None
        self._impact_direction_error_deg: float | None = None
        self._impact_drone_clearance_m: float | None = None
        self._actual_hit_feasible: bool | None = None
        self._minimum_drone_clearance_m = float(
            np.linalg.norm(
                np.asarray(initial_drone_position_m, dtype=np.float64)
                - np.asarray(self._problem.target_position_m, dtype=np.float64)
            )
        )
        self._maximum_drone_speed_m_s = 0.0
        horizontal_forward = (
            np.asarray(self._problem.target_position_m, dtype=np.float64)
            - np.asarray(self._workspace_center_m, dtype=np.float64)
        )
        horizontal_forward[2] = 0.0
        if np.linalg.norm(horizontal_forward) <= 1.0e-9:
            horizontal_forward = np.asarray(
                self._problem.impact_direction, dtype=np.float64
            ).copy()
            horizontal_forward[2] = 0.0
        if np.linalg.norm(horizontal_forward) <= 1.0e-9:
            horizontal_forward = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        self._horizontal_forward = horizontal_forward / np.linalg.norm(
            horizontal_forward
        )
        self._maximum_forward_stroke_m = 0.0
        self._recoil_stroke_m = 0.0
        self._impact_directional_tip_energy_j: float | None = None
        self._error: str | None = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._plant_thread: threading.Thread | None = None
        self._controller_thread: threading.Thread | None = None
        self._started_wall_s = 0.0
        self._plant_stream = torch.cuda.Stream(priority=-1)
        self._planner_stream = torch.cuda.Stream(priority=0)

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def running(self) -> bool:
        return self._plant_thread is not None and self._plant_thread.is_alive()

    def latest_frame(self) -> LiveFrame | None:
        with self._lock:
            return self._latest_frame

    def history(self) -> tuple[LiveFrame, ...]:
        with self._lock:
            return tuple(self._history)

    def _controller_state(self, plant_state: DroneCableState) -> DroneCableState:
        if self.settings.matched_model:
            return plant_state
        return DroneCableState(
            plant_state.drone_position_m,
            plant_state.drone_velocity_m_s,
            transfer_dder_state(
                plant_state.cable,
                self.plant_snapshot,
                self.controller_snapshot,
            ),
        )

    def _snap_physics_time(self, value_s: float) -> float:
        """Represent a DDER event on the plant's exact logical-time grid."""

        return (
            round(float(value_s) / self.settings.physics_dt_s)
            * self.settings.physics_dt_s
        )

    def _phase_schedule(self, elapsed_s: float) -> CastingPhaseSchedule:
        return CastingPhaseSchedule(
            elapsed_s=self._snap_physics_time(elapsed_s),
            injection_end_s=(
                self.settings.injection_steps
                * self.settings.control_interval_s
            ),
            motion_end_s=self.settings.horizon_s,
            lock_to_target_plane=True,
        )

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._warm_runtime_graphs()
        self._initialize_plan()
        self._started_wall_s = time.perf_counter()
        self._plant_thread = threading.Thread(
            target=self._plant_loop,
            name="drone-cable-plant",
            daemon=True,
        )
        self._controller_thread = threading.Thread(
            target=self._controller_loop,
            name="drone-cable-mpc",
            daemon=True,
        )
        self._plant_thread.start()
        self._controller_thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout_s: float = 2.0) -> None:
        for thread in (self._controller_thread, self._plant_thread):
            if thread is not None:
                thread.join(timeout=timeout_s)

    def _warm_runtime_graphs(self) -> None:
        zero_plant = torch.zeros((1, 1, 3), dtype=self.plant.dtype, device=self.device)
        # The online phase schedule leaves at most three casting variables and
        # one impact-time variable active.  The finite-difference Jacobian
        # evaluates lower/upper perturbations as one eight-rollout CUDA batch.
        zero_plan = torch.zeros(
            (8, self.planner.settings.control_count, 3),
            dtype=self.planner.dtype,
            device=self.device,
        )
        delay_control_count = int(
            round(self.settings.replan_interval_s / self.settings.control_interval_s)
        )
        zero_delay = torch.zeros(
            (1, delay_control_count, 3),
            dtype=self.planner.dtype,
            device=self.device,
        )
        state = _clone_state(self._state)
        controller_state = self._controller_state(state)
        with torch.no_grad(), torch.cuda.stream(self._plant_stream):
            self.plant.rollout(state, zero_plant, create_graph=False)
        with torch.no_grad(), torch.cuda.stream(self._planner_stream):
            self.planner.rollout(controller_state, zero_delay, create_graph=False)
            self.planner.rollout(controller_state, zero_plan, create_graph=False)
        self._plant_stream.synchronize()
        self._planner_stream.synchronize()

    def _initialize_plan(self) -> None:
        started = time.perf_counter()
        with self._lock:
            state = self._controller_state(_clone_state(self._state))
            problem = self._problem
        with torch.cuda.stream(self._planner_stream):
            plan = optimize_controls(
                self.planner,
                state,
                problem,
                self.initial_optimizer,
                previous_acceleration_m_s2=np.zeros(3, dtype=np.float64),
                cancelled=self._stop.is_set,
                weights=self.cost_weights,
                optimize_impact_time=True,
                truncate_at_impact=False,
                phase_schedule=self._phase_schedule(0.0),
            )
        self._planner_stream.synchronize()
        if plan.cost_terms.get("safety_violation", math.inf) > 1.0e-12:
            raise RuntimeError(
                "No safe first MPC action was found for the requested workspace "
                f"(safety violation={plan.cost_terms['safety_violation']:.4g}, "
                f"tip error={plan.cost_terms['position_error_m']:.3f} m, "
                f"directed speed={plan.cost_terms['directional_speed_m_s']:.3f} m/s, "
                f"direction error={plan.cost_terms['direction_error_deg']:.1f} deg)."
            )
        with self._lock:
            self._active_controls = plan.controls_m_s2.copy()
            self._active_epoch_s = 0.0
            self._impact_time_s = self._snap_physics_time(plan.impact_time_s)
            self._latest_plan = plan
            self._last_solve_s = time.perf_counter() - started

    def _promote_pending_locked(self) -> None:
        if (
            self._pending_controls is not None
            and self._simulation_time_s + 1.0e-9 >= self._pending_epoch_s
        ):
            self._active_controls = self._pending_controls
            self._active_epoch_s = self._pending_epoch_s
            if self._pending_impact_time_s is not None:
                self._impact_time_s = self._pending_impact_time_s
            self._pending_controls = None
            self._pending_epoch_s = math.inf
            self._pending_impact_time_s = None

    def _control_at_locked(self, simulation_time_s: float) -> np.ndarray:
        if (
            self._pending_controls is not None
            and simulation_time_s + 1.0e-9 >= self._pending_epoch_s
        ):
            controls = self._pending_controls
            epoch = self._pending_epoch_s
        else:
            controls = self._active_controls
            epoch = self._active_epoch_s
        if controls is None or simulation_time_s < epoch:
            return np.zeros(3, dtype=np.float64)
        # Simulation time is accumulated in physics-dt increments.  Snap ratios
        # that are numerically just below an integer to that integer; otherwise
        # one nominal 0.20 s control can be executed for 11 rather than 10
        # physics steps and the plant no longer matches its MPC rollout.
        ratio = (simulation_time_s - epoch) / self.settings.control_interval_s
        index = int(math.floor(ratio + 1.0e-7))
        if not 0 <= index < len(controls):
            return np.zeros(3, dtype=np.float64)
        return np.asarray(controls[index], dtype=np.float64)

    def _control_sequence_locked(
        self,
        start_time_s: float,
        count: int,
    ) -> np.ndarray:
        return np.stack(
            [
                self._control_at_locked(
                    start_time_s + index * self.settings.control_interval_s
                )
                for index in range(count)
            ],
            axis=0,
        )

    def _plant_loop(self) -> None:
        next_wall_s = time.perf_counter()
        dt = self.settings.physics_dt_s
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                if now < next_wall_s:
                    self._stop.wait(next_wall_s - now)
                    continue
                with self._lock:
                    self._promote_pending_locked()
                    problem = self._problem
                    if not self._mission_complete and (
                        self._current_hit_locked(problem)
                        or self._simulation_time_s + 1.0e-9
                        >= self.settings.mission_duration_s
                    ):
                        self._record_impact_locked(problem)
                        self._mission_complete = True
                        self._impact_time_s = self._simulation_time_s
                    if self._mission_complete:
                        position = self._state.drone_position_m[0].detach().cpu().numpy()
                        velocity = self._state.drone_velocity_m_s[0].detach().cpu().numpy()
                        displacement = position - np.asarray(self._workspace_center_m)
                        # After the selected impact event, hand over to a bounded
                        # critically damped return-to-start controller rather
                        # than allowing residual velocity to carry the drone
                        # beyond the maneuver workspace.
                        acceleration = -8.0 * displacement - 4.0 * velocity
                        norm = float(np.linalg.norm(acceleration))
                        if norm > self.settings.maximum_acceleration_m_s2:
                            acceleration *= self.settings.maximum_acceleration_m_s2 / norm
                    else:
                        acceleration = self._control_at_locked(self._simulation_time_s)
                    state = self._state
                control = torch.tensor(
                    acceleration,
                    dtype=self.plant.dtype,
                    device=self.device,
                ).reshape(1, 1, 3)
                with torch.no_grad(), torch.cuda.stream(self._plant_stream):
                    rollout = self.plant.rollout(state, control, create_graph=False)
                self._plant_stream.synchronize()
                next_state = rollout.final_state()
                with self._lock:
                    self._state = next_state
                    self._simulation_time_s += dt
                    self._latest_frame = self._frame_locked(acceleration, problem)
                    self._history.append(self._latest_frame)
                next_wall_s += dt
                if time.perf_counter() - next_wall_s > 2.0 * dt:
                    next_wall_s = time.perf_counter()
        except Exception as error:
            with self._lock:
                self._error = f"Plant failed: {error}"
            self._stop.set()

    def _frame_locked(self, acceleration: np.ndarray, problem: MpcProblem) -> LiveFrame:
        def array(value: torch.Tensor) -> np.ndarray:
            return value[0].detach().cpu().numpy().copy()

        drone = array(self._state.drone_position_m)
        drone_velocity = array(self._state.drone_velocity_m_s)
        current_excursion = float(
            np.linalg.norm(drone - np.asarray(self._workspace_center_m))
        )
        forward_position = float(
            np.dot(
                drone - np.asarray(self._workspace_center_m),
                self._horizontal_forward,
            )
        )
        if not self._mission_complete:
            self._maximum_drone_excursion_m = max(
                self._maximum_drone_excursion_m,
                current_excursion,
            )
            self._minimum_drone_clearance_m = min(
                self._minimum_drone_clearance_m,
                float(np.linalg.norm(drone - np.asarray(problem.target_position_m))),
            )
            self._maximum_drone_speed_m_s = max(
                self._maximum_drone_speed_m_s,
                float(np.linalg.norm(drone_velocity)),
            )
            self._maximum_forward_stroke_m = max(
                self._maximum_forward_stroke_m,
                forward_position,
            )
            self._recoil_stroke_m = max(
                0.0,
                self._maximum_forward_stroke_m - forward_position,
            )
        attachment = drone + np.array((0.0, 0.0, -self.settings.attachment_drop_m))
        injection_end_s = (
            self.settings.injection_steps * self.settings.control_interval_s
        )
        if self._mission_complete:
            casting_phase = "return"
        elif self._simulation_time_s < injection_end_s - 1.0e-9:
            casting_phase = "injection"
        elif self._simulation_time_s < self.settings.horizon_s - 1.0e-9:
            casting_phase = "release/recoil"
        else:
            casting_phase = "coast/strike"
        return LiveFrame(
            simulation_time_s=self._simulation_time_s,
            wall_elapsed_s=time.perf_counter() - self._started_wall_s,
            drone_position_m=drone,
            drone_velocity_m_s=drone_velocity,
            attachment_position_m=attachment,
            cable_positions_m=array(self._state.cable.positions_m),
            cable_velocities_m_s=array(self._state.cable.velocities_m_s),
            commanded_acceleration_m_s2=np.asarray(acceleration).copy(),
            target_position_m=np.asarray(problem.target_position_m).copy(),
            impact_direction=np.asarray(problem.impact_direction).copy(),
            controller_solve_time_s=self._last_solve_s,
            controller_rate_hz=self._last_controller_rate_hz,
            controller_deadline_missed=self._deadline_missed,
            impact_time_s=self._impact_time_s,
            mission_duration_s=self.settings.mission_duration_s,
            casting_phase=casting_phase,
            mission_complete=self._mission_complete,
            maximum_drone_excursion_m=self._maximum_drone_excursion_m,
            drone_excursion_limit_m=problem.maximum_drone_excursion_m,
            impact_tip_error_m=self._impact_tip_error_m,
            impact_directional_speed_m_s=self._impact_directional_speed_m_s,
            impact_direction_error_deg=self._impact_direction_error_deg,
            impact_drone_clearance_m=self._impact_drone_clearance_m,
            actual_hit_feasible=self._actual_hit_feasible,
            minimum_drone_clearance_m=self._minimum_drone_clearance_m,
            maximum_drone_speed_m_s=self._maximum_drone_speed_m_s,
            maximum_forward_stroke_m=self._maximum_forward_stroke_m,
            recoil_stroke_m=self._recoil_stroke_m,
            impact_directional_tip_energy_j=(
                self._impact_directional_tip_energy_j
            ),
            plan=self._latest_plan,
        )

    def _current_hit_locked(self, problem: MpcProblem) -> bool:
        tip = self._state.cable.positions_m[0, -1].detach().cpu().numpy()
        tip_velocity = self._state.cable.velocities_m_s[0, -1].detach().cpu().numpy()
        target = np.asarray(problem.target_position_m)
        direction = np.asarray(problem.impact_direction)
        directional_speed = float(np.dot(tip_velocity, direction))
        lateral_speed = float(
            np.linalg.norm(tip_velocity - directional_speed * direction)
        )
        within_direction_cone = lateral_speed <= (
            math.tan(math.radians(problem.maximum_impact_angle_deg))
            * max(directional_speed, 0.0)
        )
        return bool(
            np.linalg.norm(tip - target) <= problem.maximum_tip_error_m
            and directional_speed >= problem.minimum_impact_speed_m_s
            and within_direction_cone
            and self._maximum_drone_excursion_m
            <= problem.maximum_drone_excursion_m
            and self._minimum_drone_clearance_m
            >= problem.drone_keepout_radius_m
            and self._maximum_drone_speed_m_s <= self.settings.maximum_speed_m_s
            and self._maximum_forward_stroke_m
            >= problem.minimum_forward_stroke_m
            and self._recoil_stroke_m >= problem.minimum_recoil_stroke_m
        )

    def _record_impact_locked(self, problem: MpcProblem) -> None:
        drone = self._state.drone_position_m[0].detach().cpu().numpy()
        tip = self._state.cable.positions_m[0, -1].detach().cpu().numpy()
        tip_velocity = self._state.cable.velocities_m_s[0, -1].detach().cpu().numpy()
        target = np.asarray(problem.target_position_m)
        direction = np.asarray(problem.impact_direction)
        self._impact_tip_error_m = float(np.linalg.norm(tip - target))
        self._impact_directional_speed_m_s = float(np.dot(tip_velocity, direction))
        tip_mass_kg = float(
            self.plant_snapshot.model.parameters.vertex_masses_kg[-1]
        )
        self._impact_directional_tip_energy_j = (
            0.5
            * tip_mass_kg
            * max(0.0, self._impact_directional_speed_m_s) ** 2
        )
        tip_speed = float(np.linalg.norm(tip_velocity))
        if tip_speed <= 1.0e-9:
            self._impact_direction_error_deg = 90.0
        else:
            cosine = float(
                np.clip(self._impact_directional_speed_m_s / tip_speed, -1.0, 1.0)
            )
            self._impact_direction_error_deg = math.degrees(math.acos(cosine))
        self._impact_drone_clearance_m = float(np.linalg.norm(drone - target))
        lateral_speed = float(
            np.linalg.norm(
                tip_velocity - self._impact_directional_speed_m_s * direction
            )
        )
        within_direction_cone = lateral_speed <= (
            math.tan(math.radians(problem.maximum_impact_angle_deg))
            * self._impact_directional_speed_m_s
        )
        self._actual_hit_feasible = bool(
            self._impact_tip_error_m <= problem.maximum_tip_error_m
            and self._impact_directional_speed_m_s
            >= problem.minimum_impact_speed_m_s
            and within_direction_cone
            and self._maximum_drone_excursion_m
            <= problem.maximum_drone_excursion_m
            and self._minimum_drone_clearance_m
            >= problem.drone_keepout_radius_m
            and self._maximum_drone_speed_m_s <= self.settings.maximum_speed_m_s
            and self._maximum_forward_stroke_m
            >= problem.minimum_forward_stroke_m
            and self._recoil_stroke_m >= problem.minimum_recoil_stroke_m
        )

    def _controller_loop(self) -> None:
        next_replan_wall_s = time.perf_counter() + self.settings.replan_interval_s
        previous_complete_wall_s: float | None = None
        try:
            while not self._stop.is_set():
                now = time.perf_counter()
                if now < next_replan_wall_s:
                    self._stop.wait(next_replan_wall_s - now)
                    continue
                solve_started_wall_s = time.perf_counter()
                with self._lock:
                    snapshot_state = self._controller_state(
                        _clone_state(self._state)
                    )
                    snapshot_time = self._simulation_time_s
                    if self._mission_complete:
                        return
                    problem = self._problem
                    delay_control_count = self.settings.apply_steps
                    delay_controls = self._control_sequence_locked(
                        snapshot_time,
                        delay_control_count,
                    )
                    activation_time = self._snap_physics_time(
                        snapshot_time + self.settings.replan_interval_s
                    )
                    if activation_time >= self.settings.horizon_s - 1.0e-9:
                        return
                    warm_action = (
                        None
                        if self._latest_plan is None
                        else self._latest_plan.casting_action
                    )
                    previous_acceleration = self._control_at_locked(
                        activation_time - 1.0e-6
                    )
                predicted_state = snapshot_state
                if delay_control_count:
                    delay_tensor = torch.tensor(
                        delay_controls,
                        dtype=self.planner.dtype,
                        device=self.device,
                    )[None]
                    with torch.no_grad(), torch.cuda.stream(self._planner_stream):
                        predicted = self.planner.rollout(
                            snapshot_state,
                            delay_tensor,
                            create_graph=False,
                        )
                    self._planner_stream.synchronize()
                    predicted_state = predicted.final_state()
                with torch.cuda.stream(self._planner_stream):
                    plan = optimize_controls(
                        self.planner,
                        predicted_state,
                        problem,
                        self.optimizer,
                        control_count=self.settings.horizon_steps,
                        warm_start_action=warm_action,
                        previous_acceleration_m_s2=previous_acceleration,
                        cancelled=self._stop.is_set,
                        weights=self.cost_weights,
                        optimize_impact_time=True,
                        truncate_at_impact=False,
                        phase_schedule=self._phase_schedule(activation_time),
                    )
                self._planner_stream.synchronize()
                completed_wall_s = time.perf_counter()
                solve_time = completed_wall_s - solve_started_wall_s
                with self._lock:
                    late = self._simulation_time_s > activation_time + 1.0e-9
                    self._last_solve_s = solve_time
                    if previous_complete_wall_s is not None:
                        self._last_controller_rate_hz = 1.0 / max(
                            completed_wall_s - previous_complete_wall_s,
                            1.0e-9,
                        )
                    previous_complete_wall_s = completed_wall_s
                    self._deadline_missed = late
                    safe = plan.cost_terms.get("safety_violation", math.inf) <= 1.0e-12
                    # Short horizons often do not contain the eventual target
                    # crossing.  A safe best-progress plan is therefore valid
                    # even when its predicted hit is not yet feasible.
                    if not late and safe:
                        self._pending_controls = plan.controls_m_s2.copy()
                        self._pending_epoch_s = activation_time
                        self._pending_impact_time_s = self._snap_physics_time(
                            activation_time + plan.impact_time_s
                        )
                        self._latest_plan = plan
                next_replan_wall_s += self.settings.replan_interval_s
                if completed_wall_s > next_replan_wall_s:
                    next_replan_wall_s = completed_wall_s
        except Exception as error:
            if not self._stop.is_set():
                with self._lock:
                    self._error = f"Controller failed: {error}"
                self._stop.set()
