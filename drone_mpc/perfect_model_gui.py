"""Research UI for matched-model, long-horizon whip optimization."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from optitrack_offline.config import DEFAULT_MODEL_PATH

from .model import CableModelSnapshot, load_cable_model
from .mpc import MpcProblem
from .perfect_model import (
    PerfectMpcResult,
    PerfectMpcSettings,
    save_perfect_mpc_result,
    solve_perfect_model_mpc,
)
from .trajectory_gui import WhipTrajectoryCanvas


DEFAULT_OUTPUT_PATH = Path("data/drone_mpc/perfect_model_optimizer.npz")


class EnergyPlotCanvas(tk.Canvas):
    """Small dependency-free plot of the physically interpretable energy traces."""

    COLORS = {
        "internal": "#1769aa",
        "tip": "#c43c9c",
        "bending": "#2d8a57",
    }

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(
            parent,
            height=190,
            background="#ffffff",
            highlightbackground="#cfcfcf",
            highlightthickness=1,
        )
        self.result: PerfectMpcResult | None = None
        self.frame_index = 0
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_result(self, result: PerfectMpcResult) -> None:
        self.result = result
        self.frame_index = 0
        self.redraw()

    def set_frame(self, frame_index: int) -> None:
        self.frame_index = frame_index
        self.redraw()

    def _x(self, time_s: float, left: float, right: float, duration: float) -> float:
        return left + (right - left) * time_s / max(duration, 1.0e-12)

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        left, right, top, bottom = 58.0, width - 18.0, 28.0, height - 30.0
        self.create_text(
            12,
            8,
            anchor=tk.NW,
            text="Cable energy during the discovered maneuver",
            fill="#111111",
            font=("Segoe UI Semibold", 10),
        )
        if self.result is None or width < 100 or height < 80:
            return
        energy = self.result.energy
        traces = {
            "internal": 1000.0 * energy.internal_motion_energy_j,
            "tip": 1000.0 * energy.directed_tip_energy_j,
            "bending": 1000.0 * energy.bending_energy_j,
        }
        maximum = max(max(float(np.max(values)), 0.0) for values in traces.values())
        maximum = max(maximum, 1.0e-3)
        duration = float(energy.time_s[-1])
        for fraction in (0.0, 0.5, 1.0):
            y = bottom - fraction * (bottom - top)
            self.create_line(left, y, right, y, fill="#e4e4e4")
            self.create_text(
                left - 7,
                y,
                anchor=tk.E,
                text=f"{fraction * maximum:.2f}",
                fill="#555555",
                font=("Segoe UI", 8),
            )
        self.create_text(8, 0.5 * (top + bottom), text="mJ", fill="#555555")
        self.create_line(left, bottom, right, bottom, fill="#777777")
        self.create_line(left, top, left, bottom, fill="#777777")

        for name, values in traces.items():
            points: list[float] = []
            for time_s, value in zip(energy.time_s, values, strict=True):
                points.extend(
                    (
                        self._x(float(time_s), left, right, duration),
                        bottom - (bottom - top) * float(value) / maximum,
                    )
                )
            if len(points) >= 4:
                self.create_line(*points, fill=self.COLORS[name], width=2)

        phase_lines = (
            (energy.injection_end_s, "peak forward", "#d17a00"),
            (energy.recoil_end_s, "selected impact", "#555555"),
            (
                float(self.result.prediction.time_s[self.result.impact_frame]),
                "impact",
                "#c62828",
            ),
        )
        for time_s, label, color in phase_lines:
            x = self._x(time_s, left, right, duration)
            self.create_line(x, top, x, bottom, fill=color, dash=(4, 3), width=2)
            self.create_text(
                x + 3,
                top + 2,
                anchor=tk.NW,
                text=label,
                fill=color,
                font=("Segoe UI", 8),
            )
        frame = int(np.clip(self.frame_index, 0, len(energy.time_s) - 1))
        cursor = self._x(float(energy.time_s[frame]), left, right, duration)
        self.create_line(cursor, top, cursor, bottom, fill="#111111", width=1)

        legend_x = left
        for name, label in (
            ("internal", "relative kinetic + bending"),
            ("tip", "directed tip KE"),
            ("bending", "bending"),
        ):
            self.create_line(
                legend_x, height - 12, legend_x + 18, height - 12,
                fill=self.COLORS[name], width=3,
            )
            self.create_text(
                legend_x + 23,
                height - 12,
                anchor=tk.W,
                text=label,
                fill="#333333",
                font=("Segoe UI", 8),
            )
            legend_x += 175


class PerfectModelMpcGui:
    TICK_MS = 10

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cable Twin – Perfect-model optimizer verification")
        self.root.geometry("1580x980")
        self.root.minsize(1220, 780)
        self.root.configure(background="#ffffff")
        self._configure_style()
        self.snapshot: CableModelSnapshot | None = None
        self.result: PerfectMpcResult | None = None
        self.active_settings: PerfectMpcSettings | None = None
        self.active_problem: MpcProblem | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.optimizing = False
        self.playing = False

        self.model_path_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH.resolve()))
        self.model_status_var = tk.StringVar(value="No model loaded")
        self.initial_var = tk.StringVar(value="0.0, 0.0, 1.5")
        self.target_var = tk.StringVar(value="1.00, 0.0, 1.40")
        self.direction_var = tk.StringVar(value="1.0, 0.0, 0.0")
        self.minimum_speed_var = tk.StringVar(value="3.5")
        self.hit_tolerance_var = tk.StringVar(value="0.05")
        self.angle_var = tk.StringVar(value="35.0")
        self.excursion_var = tk.StringVar(value="1.20")
        self.minimum_forward_var = tk.StringVar(value="0.0")
        self.minimum_recoil_var = tk.StringVar(value="0.0")
        self.horizon_var = tk.StringVar(value="3.0")
        self.physics_rate_var = tk.StringVar(value="100")
        self.control_rate_var = tk.StringVar(value="50")
        self.acceleration_var = tk.StringVar(value="20.0")
        self.maximum_speed_var = tk.StringVar(value="3.0")
        self.solver_var = tk.StringVar(value="MPPI")
        self.iterations_var = tk.StringVar(value="120")
        self.initial_samples_var = tk.StringVar(value="256")
        self.wall_time_var = tk.StringVar(value="90.0")
        self.mppi_iterations_var = tk.StringVar(value="30")
        self.mppi_samples_var = tk.StringVar(value="256")
        self.mppi_batch_var = tk.StringVar(value="128")
        self.mppi_knots_var = tk.StringVar(value="16")
        self.mppi_temperature_var = tk.StringVar(value="1.0")
        self.mppi_noise_var = tk.StringVar(value="8.0")
        self.mppi_noise_decay_var = tk.StringVar(value="0.94")
        self.mppi_stage_var = tk.StringVar(value="full")
        self.mppi_position_sigma_var = tk.StringVar(value="0.25")
        self.mppi_velocity_sigma_var = tk.StringVar(value="0.15")
        self.mppi_position_weight_var = tk.StringVar(value="10.0")
        self.mppi_speed_weight_var = tk.StringVar(value="8.0")
        self.mppi_direction_weight_var = tk.StringVar(value="10.0")
        self.mppi_success_cost_var = tk.StringVar(value="200.0")
        self.mppi_displacement_weight_var = tk.StringVar(value="2.0")
        self.mppi_safety_weight_var = tk.StringVar(value="150.0")
        self.status_var = tk.StringVar(
            value="Load the cable model, then run the matched-model verification"
        )
        self.timeline_var = tk.DoubleVar(value=0.0)

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.TICK_MS, self._tick)
        self.load_model(silent=True)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#ffffff", foreground="#111111")
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff", foreground="#111111")
        style.configure("TLabelframe", background="#ffffff", foreground="#111111")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#111111")
        style.configure("TButton", background="#f1f1f1", foreground="#111111", padding=7)
        style.configure("TEntry", fieldbackground="#ffffff", foreground="#111111")

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="Perfect-model whip verification",
            font=("Segoe UI Semibold", 24),
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(
                "Low-frequency MPPI or IPOPT, followed by an independent replay with "
                "the identical full DDER cable. No learning, adaptation, estimator, or mismatch."
            ),
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(0, 10))

        model = ttk.LabelFrame(outer, text="Frozen cable model", padding=10)
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
        ttk.Label(model, textvariable=self.model_status_var).pack(
            side=tk.LEFT, padx=(12, 0)
        )

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        controls_shell = ttk.Frame(body, width=450)
        controls_shell.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls_shell.pack_propagate(False)
        controls_canvas = tk.Canvas(
            controls_shell,
            background="#ffffff",
            highlightthickness=0,
            borderwidth=0,
        )
        controls_scrollbar = ttk.Scrollbar(
            controls_shell, orient=tk.VERTICAL, command=controls_canvas.yview
        )
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set)
        controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controls_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controls = ttk.Frame(controls_canvas)
        controls_window = controls_canvas.create_window(
            (0, 0), window=controls, anchor=tk.NW
        )
        controls.bind(
            "<Configure>",
            lambda _event: controls_canvas.configure(
                scrollregion=controls_canvas.bbox("all")
            ),
        )
        controls_canvas.bind(
            "<Configure>",
            lambda event: controls_canvas.itemconfigure(
                controls_window, width=event.width
            ),
        )
        controls_canvas.bind(
            "<Enter>",
            lambda _event: controls_canvas.bind_all(
                "<MouseWheel>",
                lambda wheel: controls_canvas.yview_scroll(
                    int(-wheel.delta / 120), "units"
                ),
            ),
        )
        controls_canvas.bind(
            "<Leave>", lambda _event: controls_canvas.unbind_all("<MouseWheel>")
        )
        view = ttk.Frame(body)
        view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        task = ttk.LabelFrame(controls, text="Task and physical constraints", padding=10)
        task.pack(fill=tk.X)
        self._entry(task, "Initial drone XYZ", self.initial_var, "m")
        self._entry(task, "Free-tip target XYZ", self.target_var, "m")
        self._entry(task, "Impact direction", self.direction_var, "unit")
        self._entry(task, "Minimum directed speed", self.minimum_speed_var, "m/s")
        self._entry(task, "Tip hit tolerance", self.hit_tolerance_var, "m")
        self._entry(task, "Direction half-angle", self.angle_var, "deg")
        self._entry(task, "Workspace radius", self.excursion_var, "m")
        self._entry(task, "IPOPT minimum forward", self.minimum_forward_var, "m")
        self._entry(task, "IPOPT minimum recoil", self.minimum_recoil_var, "m")

        solver = ttk.LabelFrame(controls, text="Long-horizon optimizer", padding=10)
        solver.pack(fill=tk.X, pady=(8, 0))
        row = ttk.Frame(solver)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Solver", width=25).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.solver_var,
            values=("MPPI", "IPOPT"),
            state="readonly",
            width=11,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row, text="", width=6).pack(side=tk.LEFT, padx=(5, 0))
        self._entry(solver, "Prediction horizon", self.horizon_var, "s")
        self._entry(solver, "Physics rate", self.physics_rate_var, "Hz")
        self._entry(solver, "Control rate", self.control_rate_var, "Hz")
        self._entry(solver, "Maximum acceleration", self.acceleration_var, "m/s²")
        self._entry(solver, "Maximum drone speed", self.maximum_speed_var, "m/s")
        self._entry(solver, "MPPI iterations", self.mppi_iterations_var, "")
        self._entry(solver, "MPPI samples", self.mppi_samples_var, "")
        self._entry(solver, "Rollout batch size", self.mppi_batch_var, "")
        self._entry(solver, "Acceleration knots", self.mppi_knots_var, "")
        self._entry(solver, "Temperature", self.mppi_temperature_var, "")
        self._entry(solver, "Noise sigma", self.mppi_noise_var, "m/s²")
        self._entry(solver, "Noise decay", self.mppi_noise_decay_var, "")
        row = ttk.Frame(solver)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Diagnostic objective", width=25).pack(side=tk.LEFT)
        ttk.Combobox(
            row,
            textvariable=self.mppi_stage_var,
            values=("position", "speed", "full"),
            state="readonly",
            width=11,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row, text="", width=6).pack(side=tk.LEFT, padx=(5, 0))
        self._entry(solver, "Position sigma", self.mppi_position_sigma_var, "m")
        self._entry(solver, "Velocity gate sigma", self.mppi_velocity_sigma_var, "m")
        self._entry(solver, "Position weight", self.mppi_position_weight_var, "")
        self._entry(solver, "Speed weight", self.mppi_speed_weight_var, "")
        self._entry(solver, "Direction weight", self.mppi_direction_weight_var, "")
        self._entry(solver, "Success magnitude", self.mppi_success_cost_var, "")
        self._entry(solver, "Drone displacement", self.mppi_displacement_weight_var, "")
        self._entry(solver, "Safety weight", self.mppi_safety_weight_var, "")
        self._entry(solver, "IPOPT iterations", self.iterations_var, "")
        self._entry(solver, "IPOPT seed samples", self.initial_samples_var, "")
        self._entry(solver, "IPOPT wall-time", self.wall_time_var, "s")

        actions = ttk.LabelFrame(controls, text="Run", padding=10)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.optimize_button = ttk.Button(
            actions, text="Solve and independently replay", command=self.optimize
        )
        self.optimize_button.pack(fill=tk.X)
        self.stop_button = ttk.Button(
            actions, text="Stop", command=self.stop, state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X, pady=(6, 0))
        self.play_button = ttk.Button(
            actions, text="Play replay", command=self.toggle_play, state=tk.DISABLED
        )
        self.play_button.pack(fill=tk.X, pady=(6, 0))
        self.save_button = ttk.Button(
            actions, text="Save as...", command=self.save_as, state=tk.DISABLED
        )
        self.save_button.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            actions, text="Reset 3D view", command=lambda: self.canvas.reset_view()
        ).pack(fill=tk.X, pady=(6, 0))

        log_frame = ttk.LabelFrame(controls, text="Verification log", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log = tk.Text(
            log_frame,
            height=10,
            background="#ffffff",
            foreground="#222222",
            insertbackground="#111111",
            highlightbackground="#cfcfcf",
            highlightthickness=1,
            borderwidth=0,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.log.pack(fill=tk.BOTH, expand=True)

        self.canvas = WhipTrajectoryCanvas(view)
        self.canvas.banner_text = (
            "PERFECT-MODEL MPPI  |  INDEPENDENT FULL-DDER REPLAY  |  WORLD Z UP"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.energy_plot = EnergyPlotCanvas(view)
        self.energy_plot.pack(fill=tk.X, pady=(6, 0))
        self.timeline = ttk.Scale(
            view,
            from_=0,
            to=1,
            variable=self.timeline_var,
            command=self._timeline_changed,
        )
        self.timeline.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(view, textvariable=self.status_var).pack(anchor=tk.W, pady=(4, 0))

    @staticmethod
    def _entry(parent: ttk.Frame, label: str, variable: tk.StringVar, unit: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=25).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=13).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Label(row, text=unit, width=6).pack(side=tk.LEFT, padx=(5, 0))

    @staticmethod
    def _vector(text: str, name: str) -> tuple[float, float, float]:
        values = tuple(float(item) for item in text.replace(",", " ").split())
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} requires three finite numbers.")
        return values  # type: ignore[return-value]

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select cable model",
            filetypes=(("Cable model", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.model_path_var.set(path)

    def load_model(self, *, silent: bool = False) -> None:
        if self.optimizing:
            return
        try:
            self.snapshot = load_cable_model(self.model_path_var.get())
        except Exception as error:
            self.snapshot = None
            self.model_status_var.set("Incompatible model")
            if not silent:
                messagebox.showerror("Cable model", str(error), parent=self.root)
            return
        assert self.snapshot is not None
        label = "PROVISIONAL  " if self.snapshot.provisional else ""
        self.model_status_var.set(
            f"{label}{self.snapshot.node_count} nodes  "
            f"EI={self.snapshot.bending_stiffness_n_m2:.3g}  "
            f"Cb={self.snapshot.bending_damping_n_m2_s:.3g}"
        )

    def _append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def optimize(self) -> None:
        if self.optimizing:
            return
        if self.snapshot is None:
            messagebox.showerror(
                "Perfect-model MPC", "Load a cable model first.", parent=self.root
            )
            return
        try:
            initial = self._vector(self.initial_var.get(), "Initial drone XYZ")
            problem = MpcProblem(
                target_position_m=self._vector(self.target_var.get(), "Target XYZ"),
                impact_direction=self._vector(
                    self.direction_var.get(), "Impact direction"
                ),
                minimum_impact_speed_m_s=float(self.minimum_speed_var.get()),
                maximum_tip_error_m=float(self.hit_tolerance_var.get()),
                maximum_impact_angle_deg=float(self.angle_var.get()),
                maximum_drone_excursion_m=float(self.excursion_var.get()),
                minimum_forward_stroke_m=float(self.minimum_forward_var.get()),
                minimum_recoil_stroke_m=float(self.minimum_recoil_var.get()),
                drone_keepout_radius_m=0.30,
                drone_workspace_center_m=initial,
            )
            settings = PerfectMpcSettings(
                horizon_s=float(self.horizon_var.get()),
                physics_dt_s=1.0 / float(self.physics_rate_var.get()),
                control_interval_s=1.0 / float(self.control_rate_var.get()),
                maximum_acceleration_m_s2=float(self.acceleration_var.get()),
                maximum_speed_m_s=float(self.maximum_speed_var.get()),
                solver=self.solver_var.get().strip().lower(),
                mppi_iterations=int(self.mppi_iterations_var.get()),
                mppi_samples=int(self.mppi_samples_var.get()),
                mppi_rollout_batch_size=int(self.mppi_batch_var.get()),
                mppi_knot_count=int(self.mppi_knots_var.get()),
                mppi_temperature=float(self.mppi_temperature_var.get()),
                mppi_noise_sigma_m_s2=float(self.mppi_noise_var.get()),
                mppi_noise_decay=float(self.mppi_noise_decay_var.get()),
                mppi_objective_stage=self.mppi_stage_var.get(),
                mppi_position_sigma_m=float(self.mppi_position_sigma_var.get()),
                mppi_velocity_gate_sigma_m=float(
                    self.mppi_velocity_sigma_var.get()
                ),
                mppi_position_weight=float(self.mppi_position_weight_var.get()),
                mppi_speed_weight=float(self.mppi_speed_weight_var.get()),
                mppi_direction_weight=float(self.mppi_direction_weight_var.get()),
                mppi_success_cost=float(self.mppi_success_cost_var.get()),
                mppi_drone_displacement_weight=float(
                    self.mppi_displacement_weight_var.get()
                ),
                mppi_safety_weight=float(self.mppi_safety_weight_var.get()),
                ipopt_iterations=int(self.iterations_var.get()),
                initial_samples=int(self.initial_samples_var.get()),
                ipopt_max_wall_time_s=float(self.wall_time_var.get()),
            )
        except Exception as error:
            messagebox.showerror("Perfect-model MPC", str(error), parent=self.root)
            return
        self.active_problem = problem
        self.active_settings = settings
        self.cancel_event.clear()
        self.optimizing = True
        self.playing = False
        self.result = None
        self.canvas.result = None
        self.canvas.redraw()
        self.energy_plot.result = None
        self.energy_plot.redraw()
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        self.optimize_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.play_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        solver_label = settings.solver.upper()
        self.status_var.set(
            f"{solver_label} is solving the long-horizon matched-model problem..."
        )
        self.canvas.banner_text = (
            f"PERFECT-MODEL {solver_label}  |  INDEPENDENT FULL-DDER REPLAY  |  WORLD Z UP"
        )
        snapshot = self.snapshot

        def worker() -> None:
            try:
                result = solve_perfect_model_mpc(
                    snapshot,
                    problem,
                    initial,
                    settings,
                    progress=lambda message: self.events.put(("log", message)),
                    cancelled=self.cancel_event.is_set,
                )
                output = save_perfect_mpc_result(
                    DEFAULT_OUTPUT_PATH, result, settings, problem
                )
            except Exception as error:
                self.events.put(("error", error))
            else:
                self.events.put(("complete", (result, output)))

        threading.Thread(
            target=worker,
            name=f"perfect-model-{settings.solver}",
            daemon=True,
        ).start()

    def stop(self) -> None:
        if self.optimizing:
            self.cancel_event.set()
            self.status_var.set("Stopping after the current DDER rollout...")

    def toggle_play(self) -> None:
        if self.result is None:
            return
        self.playing = not self.playing
        if self.playing and self.timeline_var.get() >= self.result.prediction.frame_count - 1:
            self.timeline_var.set(0.0)
        self.play_button.configure(text="Pause" if self.playing else "Play replay")

    def _timeline_changed(self, _value: str) -> None:
        if self.result is None:
            return
        frame = int(round(self.timeline_var.get()))
        self.canvas.set_trajectory_frame(frame)
        self.energy_plot.set_frame(frame)

    def save_as(self) -> None:
        if self.result is None or self.active_settings is None or self.active_problem is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save perfect-model MPC verification",
            defaultextension=".npz",
            filetypes=(("NumPy archive", "*.npz"),),
        )
        if path:
            output = save_perfect_mpc_result(
                path, self.result, self.active_settings, self.active_problem
            )
            self.status_var.set(f"Saved {output}")

    def _tick(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "error":
                    self.optimizing = False
                    self.optimize_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    if self.cancel_event.is_set():
                        self.status_var.set("Optimization stopped")
                    else:
                        self.status_var.set("Optimization failed")
                        messagebox.showerror(
                            "Perfect-model MPC", str(payload), parent=self.root
                        )
                elif kind == "complete":
                    result, output = payload  # type: ignore[misc]
                    self.result = result
                    self.optimizing = False
                    self.canvas.set_result(result)
                    self.energy_plot.set_result(result)
                    self.timeline.configure(to=result.prediction.frame_count - 1)
                    self.timeline_var.set(0.0)
                    self.optimize_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    self.play_button.configure(state=tk.NORMAL, text="Play replay")
                    self.save_button.configure(state=tk.NORMAL)
                    outcome = "FEASIBLE HIT" if result.feasible else "INFEASIBLE"
                    self.status_var.set(
                        f"{outcome}: {1000.0 * result.terms['position_error_m']:.1f} mm, "
                        f"{result.terms['directional_speed_m_s']:.2f} m/s; saved {output}"
                    )
        except queue.Empty:
            pass
        if self.playing and self.result is not None:
            next_frame = int(round(self.timeline_var.get())) + 1
            if next_frame >= self.result.prediction.frame_count:
                self.playing = False
                self.play_button.configure(text="Play replay")
            else:
                self.timeline_var.set(float(next_frame))
                self.canvas.set_trajectory_frame(next_frame)
                self.energy_plot.set_frame(next_frame)
        self.root.after(self.TICK_MS, self._tick)

    def close(self) -> None:
        self.cancel_event.set()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PerfectModelMpcGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
