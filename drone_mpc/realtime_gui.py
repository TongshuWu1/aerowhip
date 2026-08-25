"""Standalone live drone-whip MPC user interface."""

from __future__ import annotations

import queue
from pathlib import Path
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from optitrack_offline.config import DEFAULT_MODEL_PATH

from .live_viewer import LiveDroneCanvas
from .model import CableModelSnapshot, load_cable_model
from .mpc import MpcCancelled, MpcProblem
from .realtime import RealtimeMpcSession, RealtimeSettings


class RealtimeMpcGui:
    TICK_MS = 33

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cable Twin – Online Receding-Horizon MPC")
        self.root.geometry("1600x960")
        self.root.minsize(1220, 760)
        self.root.configure(background="#ffffff")
        self._configure_style()
        self.snapshot: CableModelSnapshot | None = None
        self.session_source_snapshot: CableModelSnapshot | None = None
        self.session: RealtimeMpcSession | None = None
        self.active_problem: MpcProblem | None = None
        self.workspace_center_m: tuple[float, float, float] | None = None
        self.starting = False
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.last_simulation_time_s = -1.0

        self.model_path_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH.resolve()))
        self.model_status_var = tk.StringVar(value="No model loaded")
        self.status_var = tk.StringVar(value="Load the cable model, then start online MPC")
        self.initial_drone_var = tk.StringVar(value="0.0, 0.0, 1.5")
        self.target_var = tk.StringVar(value="0.48, 0.0, 1.30")
        self.direction_var = tk.StringVar(value="1.0, 0.0, 0.0")
        self.minimum_speed_var = tk.StringVar(value="1.5")
        self.hit_tolerance_var = tk.StringVar(value="0.05")
        self.planning_margin_var = tk.StringVar(value="0.003")
        self.impact_angle_var = tk.StringVar(value="35.0")
        self.planning_angle_margin_var = tk.StringVar(value="2.0")
        self.keepout_var = tk.StringVar(value="0.30")
        self.excursion_var = tk.StringVar(value="0.15")
        self.mission_duration_var = tk.StringVar(value="4.0")
        self.physics_rate_var = tk.StringVar(value="100")
        self.control_rate_var = tk.StringVar(value="50")
        self.horizon_steps_var = tk.StringVar(value="20")
        self.injection_steps_var = tk.StringVar(value="10")
        self.apply_steps_var = tk.StringVar(value="5")
        self.initial_iterations_var = tk.StringVar(value="40")
        self.update_iterations_var = tk.StringVar(value="8")
        self.ipopt_tolerance_var = tk.StringVar(value="1e-4")
        self.ipopt_acceptable_tolerance_var = tk.StringVar(value="1e-3")
        self.initial_solve_time_var = tk.StringVar(value="3.0")
        self.update_solve_time_var = tk.StringVar(value="0.09")
        self.finite_difference_step_var = tk.StringVar(value="1e-3")
        self.max_acceleration_var = tk.StringVar(value="20.0")
        self.max_speed_var = tk.StringVar(value="3.0")
        self.effort_weight_var = tk.StringVar(value="1.0")
        self.smoothness_weight_var = tk.StringVar(value="1.0")
        self.timing_summary_var = tk.StringVar()

        for variable in (
            self.physics_rate_var,
            self.control_rate_var,
            self.horizon_steps_var,
            self.injection_steps_var,
            self.apply_steps_var,
        ):
            variable.trace_add("write", self._update_timing_summary)

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.TICK_MS, self._tick)
        self.load_model(silent=True)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        text = "#111111"
        border = "#cfcfcf"
        style.configure(".", background="#ffffff", foreground=text)
        style.configure("TFrame", background="#ffffff")
        style.configure(
            "TLabelframe",
            background="#ffffff",
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.configure(
            "TLabelframe.Label",
            background="#ffffff",
            foreground=text,
            font=("Segoe UI Semibold", 10),
        )
        style.configure("TLabel", background="#ffffff", foreground=text)
        style.configure(
            "TButton",
            background="#f2f2f2",
            foreground=text,
            padding=7,
            bordercolor=border,
        )
        style.map(
            "TButton",
            background=[("active", "#e6eef5"), ("disabled", "#f5f5f5")],
            foreground=[("disabled", "#999999")],
        )
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground=text,
            insertcolor=text,
            bordercolor=border,
        )

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="Online two-phase whip MPC",
            font=("Segoe UI Semibold", 24),
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(
                "Matched fitted DDER model, full-state feedback, and fixed-horizon "
                "replanning. Solve N steps, apply M steps, then observe and solve again."
            ),
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(0, 10))

        model = ttk.LabelFrame(outer, text="Cable model", padding=10)
        model.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(model, textvariable=self.model_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(model, text="Browse", command=self.browse_model).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Button(model, text="Load", command=self.load_model).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        ttk.Label(model, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=(12, 0))

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(body, width=440)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls.pack_propagate(False)
        view = ttk.Frame(body)
        view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(controls)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        task = ttk.Frame(notebook, padding=10)
        controller = ttk.Frame(notebook, padding=10)
        notebook.add(task, text="Task")
        notebook.add(controller, text="MPC")

        ttk.Label(
            task,
            text="Hit condition and flight envelope",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor=tk.W, pady=(0, 5))
        self._entry(task, "Initial drone XYZ", self.initial_drone_var, "m")
        self._entry(task, "Free-tip target XYZ", self.target_var, "m")
        self._entry(task, "Impact direction", self.direction_var, "unit")
        self._entry(task, "Min. directed tip speed", self.minimum_speed_var, "m/s")
        self._entry(task, "Tip hit tolerance", self.hit_tolerance_var, "m")
        self._entry(task, "MPC robustness margin", self.planning_margin_var, "m")
        self._entry(task, "Direction half-angle", self.impact_angle_var, "deg")
        self._entry(
            task,
            "MPC direction margin",
            self.planning_angle_margin_var,
            "deg",
        )
        self._entry(task, "Drone target keepout", self.keepout_var, "m")
        self._entry(task, "Maximum drone excursion", self.excursion_var, "m")
        self._entry(task, "Mission timeout", self.mission_duration_var, "s")
        ttk.Label(
            task,
            text=(
                "Forward injection and recoil are structural phases of every candidate; "
                "their amplitudes are optimized while the declared phase switch persists "
                "across replans."
            ),
            foreground="#555555",
            wraplength=385,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))

        ttk.Label(
            controller,
            text="Receding horizon",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor=tk.W, pady=(0, 5))
        self._entry(controller, "Physics rate", self.physics_rate_var, "Hz")
        self._entry(controller, "Control rate", self.control_rate_var, "Hz")
        self._entry(controller, "N prediction horizon", self.horizon_steps_var, "steps")
        self._entry(controller, "Injection phase", self.injection_steps_var, "steps")
        self._entry(controller, "M steps applied", self.apply_steps_var, "steps")
        ttk.Label(
            controller,
            textvariable=self.timing_summary_var,
            foreground="#555555",
            wraplength=385,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(2, 9))
        ttk.Separator(controller).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            controller,
            text="Constrained IPOPT optimizer",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor=tk.W, pady=(0, 5))
        self._entry(controller, "Initial IPOPT iterations", self.initial_iterations_var, "")
        self._entry(controller, "IPOPT iterations / replan", self.update_iterations_var, "")
        self._entry(controller, "Desired tolerance", self.ipopt_tolerance_var, "")
        self._entry(
            controller,
            "Acceptable tolerance",
            self.ipopt_acceptable_tolerance_var,
            "",
        )
        self._entry(controller, "Initial solve limit", self.initial_solve_time_var, "s")
        self._entry(controller, "Replan solve limit", self.update_solve_time_var, "s")
        self._entry(
            controller,
            "Derivative step",
            self.finite_difference_step_var,
            "normalized",
        )
        ttk.Separator(controller).pack(fill=tk.X, pady=8)
        ttk.Label(
            controller,
            text="Vehicle and regularization",
            font=("Segoe UI Semibold", 10),
        ).pack(anchor=tk.W, pady=(0, 5))
        self._entry(controller, "Maximum acceleration", self.max_acceleration_var, "m/s²")
        self._entry(controller, "Maximum speed", self.max_speed_var, "m/s")
        self._entry(controller, "Acceleration effort weight", self.effort_weight_var, "")
        self._entry(controller, "Acceleration change weight", self.smoothness_weight_var, "")
        ttk.Label(
            controller,
            text="Matched-model baseline: planner model = simulated plant model.",
            foreground="#1d5f2c",
            wraplength=385,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))

        actions = ttk.LabelFrame(controls, text="Execution", padding=10)
        actions.pack(fill=tk.X)
        self.start_button = ttk.Button(
            actions,
            text="Start real-time MPC",
            command=self.start,
        )
        self.start_button.pack(fill=tk.X)
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self.stop,
            state=tk.DISABLED,
        )
        self.stop_button.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(actions, text="Save live recording", command=self.save).pack(
            fill=tk.X, pady=(6, 0)
        )
        ttk.Button(actions, text="Reset view", command=lambda: self.canvas.reset_view()).pack(
            fill=tk.X, pady=(6, 0)
        )

        self.canvas = LiveDroneCanvas(view)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        ttk.Label(view, textvariable=self.status_var).pack(anchor=tk.W, pady=(5, 0))
        self._update_timing_summary()

    def _update_timing_summary(self, *_args: object) -> None:
        try:
            physics_rate = float(self.physics_rate_var.get())
            control_rate = float(self.control_rate_var.get())
            horizon_steps = int(self.horizon_steps_var.get())
            injection_steps = int(self.injection_steps_var.get())
            apply_steps = int(self.apply_steps_var.get())
            if min(
                physics_rate,
                control_rate,
                horizon_steps,
                injection_steps,
                apply_steps,
            ) <= 0 or injection_steps >= horizon_steps:
                raise ValueError
            self.timing_summary_var.set(
                f"Horizon = {horizon_steps / control_rate:.3f} s; "
                f"inject for {injection_steps / control_rate:.3f} s, then "
                f"release/recoil for {(horizon_steps - injection_steps) / control_rate:.3f} s. "
                f"new plan every {apply_steps / control_rate:.3f} s. "
                f"The solver must finish inside that {1000.0 * apply_steps / control_rate:.0f} ms deadline."
            )
        except (TypeError, ValueError):
            self.timing_summary_var.set("Enter valid positive rates and step counts.")

    @staticmethod
    def _entry(parent: ttk.Frame, label: str, variable: tk.StringVar, unit: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=16).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Label(row, text=unit, width=7).pack(side=tk.LEFT, padx=(5, 0))

    @staticmethod
    def _vector(text: str, name: str) -> tuple[float, float, float]:
        tokens = text.replace(",", " ").split()
        if len(tokens) != 3:
            raise ValueError(f"{name} requires three numbers.")
        values = tuple(float(value) for value in tokens)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite.")
        return values  # type: ignore[return-value]

    def _problem(self) -> MpcProblem:
        return MpcProblem(
            target_position_m=self._vector(self.target_var.get(), "Target XYZ"),
            impact_direction=self._vector(self.direction_var.get(), "Impact direction"),
            minimum_impact_speed_m_s=float(self.minimum_speed_var.get()),
            drone_keepout_radius_m=float(self.keepout_var.get()),
            maximum_drone_excursion_m=float(self.excursion_var.get()),
            maximum_tip_error_m=float(self.hit_tolerance_var.get()),
            planning_tip_error_margin_m=float(self.planning_margin_var.get()),
            maximum_impact_angle_deg=float(self.impact_angle_var.get()),
            planning_impact_angle_margin_deg=float(
                self.planning_angle_margin_var.get()
            ),
        )

    def _settings(self) -> RealtimeSettings:
        physics_rate_hz = float(self.physics_rate_var.get())
        control_rate_hz = float(self.control_rate_var.get())
        horizon_steps = int(self.horizon_steps_var.get())
        injection_steps = int(self.injection_steps_var.get())
        apply_steps = int(self.apply_steps_var.get())
        return RealtimeSettings(
            physics_dt_s=1.0 / physics_rate_hz,
            control_interval_s=1.0 / control_rate_hz,
            horizon_s=horizon_steps / control_rate_hz,
            replan_interval_s=apply_steps / control_rate_hz,
            mission_duration_s=float(self.mission_duration_var.get()),
            injection_steps=injection_steps,
            matched_model=True,
            initial_ipopt_iterations=int(self.initial_iterations_var.get()),
            ipopt_iterations=int(self.update_iterations_var.get()),
            ipopt_tolerance=float(self.ipopt_tolerance_var.get()),
            ipopt_acceptable_tolerance=float(
                self.ipopt_acceptable_tolerance_var.get()
            ),
            initial_ipopt_max_wall_time_s=float(self.initial_solve_time_var.get()),
            ipopt_max_wall_time_s=float(self.update_solve_time_var.get()),
            ipopt_finite_difference_step=float(self.finite_difference_step_var.get()),
            maximum_acceleration_m_s2=float(self.max_acceleration_var.get()),
            maximum_speed_m_s=float(self.max_speed_var.get()),
            acceleration_effort_weight=float(self.effort_weight_var.get()),
            acceleration_smoothness_weight=float(self.smoothness_weight_var.get()),
        )

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select cable model",
            filetypes=(("Cable model", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.model_path_var.set(path)

    def load_model(self, *, silent: bool = False) -> None:
        if self.session is not None and self.session.running:
            if not silent:
                messagebox.showinfo("Cable model", "Stop the live session first.", parent=self.root)
            return
        try:
            snapshot = load_cable_model(self.model_path_var.get())
        except Exception as error:
            self.snapshot = None
            self.model_status_var.set("Incompatible model")
            if not silent:
                messagebox.showerror("Cable model", str(error), parent=self.root)
            return
        self.snapshot = snapshot
        prefix = "PROVISIONAL  " if snapshot.provisional else ""
        self.model_status_var.set(
            f"{prefix}N={snapshot.node_count}  EI={snapshot.bending_stiffness_n_m2:.3g}  "
            f"Cb={snapshot.bending_damping_n_m2_s:.3g}"
        )

    def start(self) -> None:
        if self.starting or (self.session is not None and self.session.running):
            return
        if self.snapshot is None:
            messagebox.showerror("Live MPC", "Load a compatible model first.", parent=self.root)
            return
        try:
            problem = self._problem()
            initial_drone = self._vector(
                self.initial_drone_var.get(), "Initial drone XYZ"
            )
            session = RealtimeMpcSession(
                self.snapshot,
                self._settings(),
                problem,
                initial_drone,
            )
        except Exception as error:
            messagebox.showerror("Live MPC", str(error), parent=self.root)
            return
        self.session = session
        self.session_source_snapshot = self.snapshot
        self.active_problem = problem
        self.workspace_center_m = initial_drone
        self.starting = True
        self.last_simulation_time_s = -1.0
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.canvas.clear()
        self.status_var.set(
            "Warming CUDA graphs and solving the first short MPC horizon..."
        )

        def launch() -> None:
            try:
                session.start()
            except MpcCancelled:
                self.events.put(("stopped", "Initial MPC solve stopped"))
            except Exception as error:
                self.events.put(("error", str(error)))
            else:
                self.events.put(("started", "Live MPC started"))

        threading.Thread(target=launch, name="start-live-mpc", daemon=True).start()

    def stop(self) -> None:
        if self.session is not None:
            self.session.stop()
            self.session.join(timeout_s=0.5)
        self.starting = False
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopped")

    def save(self) -> None:
        if self.session is None or not self.session.history():
            messagebox.showinfo("Save", "Start a live session first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save live drone-whip recording",
            defaultextension=".npz",
            filetypes=(("NumPy archive", "*.npz"),),
        )
        if not path:
            return
        frames = self.session.history()
        settings = self.session.settings
        controller_model = self.session.snapshot
        source_model = self.session_source_snapshot
        if source_model is None:
            raise RuntimeError("The source cable model is no longer available.")
        problem = self.active_problem or self._problem()

        def optional(values: list[float | None]) -> np.ndarray:
            return np.asarray(
                [np.nan if value is None else value for value in values],
                dtype=np.float64,
            )

        plans = [frame.plan for frame in frames]
        final_plan = next((plan for plan in reversed(plans) if plan is not None), None)
        final_terms = {} if final_plan is None else final_plan.cost_terms
        model_parameters = controller_model.model.parameters
        np.savez_compressed(
            path,
            archive_schema=np.asarray("drone_whip_online_mpc_v1"),
            simulation_time_s=np.asarray([frame.simulation_time_s for frame in frames]),
            wall_elapsed_s=np.asarray([frame.wall_elapsed_s for frame in frames]),
            drone_positions_m=np.stack([frame.drone_position_m for frame in frames]),
            drone_velocities_m_s=np.stack([frame.drone_velocity_m_s for frame in frames]),
            cable_positions_m=np.stack([frame.cable_positions_m for frame in frames]),
            cable_velocities_m_s=np.stack([frame.cable_velocities_m_s for frame in frames]),
            commanded_accelerations_m_s2=np.stack(
                [frame.commanded_acceleration_m_s2 for frame in frames]
            ),
            controller_solve_time_s=optional(
                [frame.controller_solve_time_s for frame in frames]
            ),
            controller_rate_hz=optional(
                [frame.controller_rate_hz for frame in frames]
            ),
            controller_deadline_missed=np.asarray(
                [frame.controller_deadline_missed for frame in frames],
                dtype=np.bool_,
            ),
            target_positions_m=np.stack([frame.target_position_m for frame in frames]),
            impact_directions=np.stack([frame.impact_direction for frame in frames]),
            target_position_m=frames[-1].target_position_m,
            impact_direction=frames[-1].impact_direction,
            scheduled_impact_time_s=np.asarray(
                [frame.impact_time_s for frame in frames], dtype=np.float64
            ),
            impact_time_s=np.asarray(frames[-1].impact_time_s, dtype=np.float64),
            mission_complete=np.asarray(
                [frame.mission_complete for frame in frames], dtype=np.bool_
            ),
            casting_phase=np.asarray([frame.casting_phase for frame in frames]),
            maximum_drone_excursion_m=np.asarray(
                frames[-1].maximum_drone_excursion_m
            ),
            maximum_drone_excursion_history_m=np.asarray(
                [frame.maximum_drone_excursion_m for frame in frames],
                dtype=np.float64,
            ),
            drone_excursion_limit_m=np.asarray(frames[-1].drone_excursion_limit_m),
            minimum_drone_clearance_history_m=np.asarray(
                [frame.minimum_drone_clearance_m for frame in frames],
                dtype=np.float64,
            ),
            maximum_drone_speed_history_m_s=np.asarray(
                [frame.maximum_drone_speed_m_s for frame in frames],
                dtype=np.float64,
            ),
            maximum_forward_stroke_history_m=np.asarray(
                [frame.maximum_forward_stroke_m for frame in frames],
                dtype=np.float64,
            ),
            recoil_stroke_history_m=np.asarray(
                [frame.recoil_stroke_m for frame in frames],
                dtype=np.float64,
            ),
            impact_directional_tip_energy_j=np.asarray(
                np.nan
                if frames[-1].impact_directional_tip_energy_j is None
                else frames[-1].impact_directional_tip_energy_j
            ),
            impact_tip_error_m=np.asarray(
                np.nan
                if frames[-1].impact_tip_error_m is None
                else frames[-1].impact_tip_error_m
            ),
            impact_directional_speed_m_s=np.asarray(
                np.nan
                if frames[-1].impact_directional_speed_m_s is None
                else frames[-1].impact_directional_speed_m_s
            ),
            impact_direction_error_deg=np.asarray(
                np.nan
                if frames[-1].impact_direction_error_deg is None
                else frames[-1].impact_direction_error_deg
            ),
            impact_drone_clearance_m=np.asarray(
                np.nan
                if frames[-1].impact_drone_clearance_m is None
                else frames[-1].impact_drone_clearance_m
            ),
            actual_hit_feasible=np.asarray(
                -1
                if frames[-1].actual_hit_feasible is None
                else int(frames[-1].actual_hit_feasible),
                dtype=np.int8,
            ),
            planned_feasible=np.asarray(
                [-1 if plan is None else int(plan.feasible) for plan in plans],
                dtype=np.int8,
            ),
            planned_constraint_violation=optional(
                [None if plan is None else plan.constraint_violation for plan in plans]
            ),
            planned_cost=optional(
                [None if plan is None else plan.cost for plan in plans]
            ),
            planned_impact_time_s=optional(
                [None if plan is None else plan.impact_time_s for plan in plans]
            ),
            planned_forward_elevation_deg=optional(
                [
                    None if plan is None else plan.casting_action.forward_elevation_deg
                    for plan in plans
                ]
            ),
            planned_recoil_deflection_deg=optional(
                [
                    None if plan is None else plan.casting_action.recoil_deflection_deg
                    for plan in plans
                ]
            ),
            planned_forward_excursion_m=optional(
                [
                    None
                    if plan is None
                    else plan.casting_action.forward_excursion_m
                    for plan in plans
                ]
            ),
            planned_recoil_excursion_m=optional(
                [
                    None if plan is None else plan.casting_action.recoil_excursion_m
                    for plan in plans
                ]
            ),
            planned_cast_reversal_fraction=optional(
                [
                    None if plan is None else plan.casting_action.reversal_fraction
                    for plan in plans
                ]
            ),
            planned_cast_motion_fraction=optional(
                [
                    None if plan is None else plan.casting_action.motion_fraction
                    for plan in plans
                ]
            ),
            final_plan_term_names=np.asarray(tuple(final_terms.keys())),
            final_plan_term_values=np.asarray(tuple(final_terms.values()), dtype=np.float64),
            control_parameterization=np.asarray(
                "phase_locked_target_plane_forward_recoil_primitive_v4"
            ),
            problem_target_position_m=np.asarray(problem.target_position_m),
            problem_impact_direction=np.asarray(problem.impact_direction),
            problem_minimum_impact_speed_m_s=np.asarray(
                problem.minimum_impact_speed_m_s
            ),
            problem_minimum_forward_stroke_m=np.asarray(
                problem.minimum_forward_stroke_m
            ),
            problem_minimum_recoil_stroke_m=np.asarray(
                problem.minimum_recoil_stroke_m
            ),
            problem_maximum_tip_error_m=np.asarray(problem.maximum_tip_error_m),
            problem_planning_tip_error_margin_m=np.asarray(
                problem.planning_tip_error_margin_m
            ),
            problem_planning_tip_error_limit_m=np.asarray(
                problem.planning_tip_error_limit_m
            ),
            problem_maximum_impact_angle_deg=np.asarray(
                problem.maximum_impact_angle_deg
            ),
            problem_planning_impact_angle_margin_deg=np.asarray(
                problem.planning_impact_angle_margin_deg
            ),
            problem_planning_impact_angle_limit_deg=np.asarray(
                problem.planning_impact_angle_limit_deg
            ),
            problem_drone_keepout_radius_m=np.asarray(
                problem.drone_keepout_radius_m
            ),
            problem_maximum_drone_excursion_m=np.asarray(
                problem.maximum_drone_excursion_m
            ),
            problem_drone_workspace_center_m=np.asarray(
                self.workspace_center_m, dtype=np.float64
            ),
            physics_dt_s=np.asarray(settings.physics_dt_s),
            horizon_s=np.asarray(settings.horizon_s),
            horizon_steps=np.asarray(settings.horizon_steps),
            injection_steps=np.asarray(settings.injection_steps),
            control_interval_s=np.asarray(settings.control_interval_s),
            replan_interval_s=np.asarray(settings.replan_interval_s),
            apply_steps=np.asarray(settings.apply_steps),
            mission_duration_s=np.asarray(settings.mission_duration_s),
            matched_model=np.asarray(settings.matched_model),
            reduced_node_count=np.asarray(controller_model.node_count),
            optimizer=np.asarray("IPOPT"),
            initial_ipopt_iterations=np.asarray(settings.initial_ipopt_iterations),
            ipopt_iterations=np.asarray(settings.ipopt_iterations),
            ipopt_tolerance=np.asarray(settings.ipopt_tolerance),
            ipopt_acceptable_tolerance=np.asarray(
                settings.ipopt_acceptable_tolerance
            ),
            initial_ipopt_max_wall_time_s=np.asarray(
                settings.initial_ipopt_max_wall_time_s
            ),
            ipopt_max_wall_time_s=np.asarray(settings.ipopt_max_wall_time_s),
            ipopt_finite_difference_step=np.asarray(
                settings.ipopt_finite_difference_step
            ),
            attachment_drop_m=np.asarray(settings.attachment_drop_m),
            maximum_acceleration_m_s2=np.asarray(settings.maximum_acceleration_m_s2),
            maximum_speed_m_s=np.asarray(settings.maximum_speed_m_s),
            controller_ei_scale=np.asarray(settings.controller_ei_scale),
            controller_cb_scale=np.asarray(settings.controller_cb_scale),
            regularization_acceleration_effort=np.asarray(
                self.session.cost_weights.acceleration_effort
            ),
            regularization_acceleration_smoothness=np.asarray(
                self.session.cost_weights.acceleration_smoothness
            ),
            source_model_path=np.asarray(str(source_model.source_path)),
            source_model_sha256=np.asarray(source_model.sha256),
            controller_model_sha256=np.asarray(controller_model.sha256),
            model_sha256=np.asarray(controller_model.sha256),
            source_model_provisional=np.asarray(source_model.provisional),
            model_provenance_note=np.asarray(controller_model.provenance_note),
            plant_model_node_count=np.asarray(source_model.node_count),
            controller_model_node_count=np.asarray(controller_model.node_count),
            cable_length_m=np.asarray(controller_model.cable_length_m),
            cable_mass_kg=np.asarray(model_parameters.cable_mass_kg),
            cable_diameter_m=np.asarray(model_parameters.cable_diameter_m),
            bending_stiffness_n_m2=np.asarray(
                controller_model.bending_stiffness_n_m2
            ),
            bending_damping_n_m2_s=np.asarray(
                controller_model.bending_damping_n_m2_s
            ),
            plant_bending_stiffness_n_m2=np.asarray(
                source_model.bending_stiffness_n_m2
            ),
            plant_bending_damping_n_m2_s=np.asarray(
                source_model.bending_damping_n_m2_s
            ),
        )
        self.status_var.set(f"Saved {Path(path).resolve()}")

    def _tick(self) -> None:
        try:
            while True:
                kind, text = self.events.get_nowait()
                if kind == "error":
                    self.starting = False
                    self.start_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    self.status_var.set(text)
                    messagebox.showerror("Live MPC", text, parent=self.root)
                elif kind == "started":
                    self.starting = False
                    self.status_var.set(text)
                elif kind == "stopped":
                    self.starting = False
                    self.start_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    self.status_var.set(text)
        except queue.Empty:
            pass
        if self.session is not None:
            if self.session.error is not None:
                self.status_var.set(self.session.error)
            frame = self.session.latest_frame()
            if frame is not None and frame.simulation_time_s > self.last_simulation_time_s:
                self.last_simulation_time_s = frame.simulation_time_s
                self.canvas.set_live_frame(frame)
                if frame.mission_complete and frame.impact_tip_error_m is not None:
                    target_error = frame.impact_tip_error_m
                    impact_speed = frame.impact_directional_speed_m_s
                    direction_error = frame.impact_direction_error_deg
                else:
                    target_error = float(
                        np.linalg.norm(
                            frame.cable_positions_m[-1] - frame.target_position_m
                        )
                    )
                    tip_velocity = frame.cable_velocities_m_s[-1]
                    impact_speed = float(
                        np.dot(tip_velocity, frame.impact_direction)
                    )
                    tip_speed = float(np.linalg.norm(tip_velocity))
                    direction_error = float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    impact_speed / max(tip_speed, 1.0e-9),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                solve = "warming" if frame.controller_solve_time_s is None else (
                    f"{1000.0 * frame.controller_solve_time_s:.0f}ms"
                )
                deadline = (
                    " MISSED M-STEP DEADLINE"
                    if frame.controller_deadline_missed
                    else ""
                )
                if frame.mission_complete:
                    outcome = "IMPACT" if frame.actual_hit_feasible is None else (
                        "HIT" if frame.actual_hit_feasible else "MISS"
                    )
                    phase = f"{outcome} / returning"
                else:
                    phase = "executing"
                if frame.plan is None:
                    plan_text = "plan warming"
                elif frame.plan.feasible:
                    plan_text = (
                        f"predicted HIT in {frame.plan.impact_time_s:.2f}s"
                    )
                else:
                    plan_text = (
                        "best safe progress "
                        f"(violation={frame.plan.constraint_violation:.3g})"
                    )
                impact_text = "" if impact_speed is None else (
                    f"  world directed speed={impact_speed:.2f}m/s"
                )
                angle_text = "" if direction_error is None else (
                    f"  angle={direction_error:.1f}deg"
                )
                realtime_factor = frame.simulation_time_s / max(
                    frame.wall_elapsed_s, 1.0e-9
                )
                self.status_var.set(
                    f"{phase}  phase={frame.casting_phase}  {plan_text}  "
                    f"t={frame.simulation_time_s:.2f}s  "
                    f"RTF={realtime_factor:.2f}x  "
                    f"MPC={solve}{deadline}  tip error={1000.0 * target_error:.0f}mm"
                    f"{impact_text}{angle_text}  "
                    f"excursion={frame.maximum_drone_excursion_m:.2f}/"
                    f"{frame.drone_excursion_limit_m:.2f}m  "
                    f"forward/recoil={frame.maximum_forward_stroke_m:.2f}/"
                    f"{frame.recoil_stroke_m:.2f}m  "
                    f"min clearance={frame.minimum_drone_clearance_m:.2f}m  "
                    f"max drone speed={frame.maximum_drone_speed_m_s:.2f}/"
                    f"{self.session.settings.maximum_speed_m_s:.2f}m/s"
                )
        self.root.after(self.TICK_MS, self._tick)

    def close(self) -> None:
        if self.session is not None:
            self.session.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    RealtimeMpcGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
