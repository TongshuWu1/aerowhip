"""Compact research UI for continuous free-tip figure-eight tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch

from optitrack_offline.config import DEFAULT_MODEL_PATH

from .figure8_canvas import Figure8TrackingCanvas, TrackingErrorCanvas
from .history_observer import (
    DderHistoryObserver,
    EndpointHistoryObservation,
    HistoryObserverSettings,
)
from .figure8_tracking import (
    Figure8Execution,
    Figure8ExecutionSettings,
    Figure8LiveUpdate,
    Figure8Reference,
    ObservationMode,
    TrackingCostSettings,
    run_figure8_tracking,
    save_figure8_execution,
)
from .model import load_cable_model
from .mppi import MppiSettings
from .reduced import build_controller_and_truth_models
from .simulator import SimulationSettings, WhipSimulator


DEFAULT_OUTPUT_DIRECTORY = Path("data/drone_mpc/figure8_tracking")


@dataclass(frozen=True, slots=True)
class _RunConfiguration:
    model_path: Path
    initial_drone_position_m: tuple[float, float, float]
    reference: Figure8Reference
    simulation: SimulationSettings
    mppi: MppiSettings
    cost: TrackingCostSettings
    execution: Figure8ExecutionSettings
    history_observer: HistoryObserverSettings


class Figure8TrackingApp:
    TICK_MS = 35

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DDER–MPPI free-tip figure-8 tracking")
        self.root.geometry("1450x900")
        self.root.minsize(1180, 720)
        self._configure_style()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.observation_mode_lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.execution: Figure8Execution | None = None
        self.active_configuration: _RunConfiguration | None = None
        self.last_update: Figure8LiveUpdate | None = None

        self.model_path_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH))
        self.initial_drone_var = tk.StringVar(value="0, 0, 1.45")
        self.center_var = tk.StringVar(value="auto")
        self.amplitude_x_var = tk.StringVar(value="0.7")
        self.amplitude_y_var = tk.StringVar(value="0.5")
        self.tracking_weight_var = tk.StringVar(value="5000")
        self.path_progress_reward_weight_var = tk.StringVar(value="5")
        self.control_effort_weight_var = tk.StringVar(value="0.0002")
        self.control_smoothness_weight_var = tk.StringVar(value="0.05")
        self.tip_motion_smoothness_weight_var = tk.StringVar(value="0.5")
        self.mode_var = tk.StringVar(value="full")
        self._live_observation_mode: ObservationMode = "full"
        self.horizon_var = tk.StringVar(value="1.0")
        self.samples_var = tk.StringVar(value="1024")
        self.iterations_var = tk.StringVar(value="2")
        self.knots_var = tk.StringVar(value="11")
        self.physics_rate_var = tk.StringVar(value="50")
        self.control_rate_var = tk.StringVar(value="50")
        self.replan_rate_var = tk.StringVar(value="10")
        self.seed_var = tk.StringVar(value="17")
        self.periodic_bootstrap_var = tk.BooleanVar(value=True)
        self.observer_history_var = tk.StringVar(value="0.30")
        self.observer_modes_var = tk.StringVar(value="4")
        self.observer_position_scale_var = tk.StringVar(value="0.025")
        self.observer_velocity_scale_var = tk.StringVar(value="0.25")
        self.observer_prior_weight_var = tk.StringVar(value="0.0001")
        self.observer_lm_damping_var = tk.StringVar(value="0.0001")
        self.observer_fd_step_var = tk.StringVar(value="0.04")
        self.current_error_var = tk.StringVar(value="Current error: —")
        self.rmse_var = tk.StringVar(value="RMSE since reset: —")
        self.timing_var = tk.StringVar(value="MPPI update: —")
        self.status_var = tk.StringVar(
            value="Configure the reference and select an observation mode, then Play."
        )
        self.summary_var = tk.StringVar(value="No completed run.")
        self.config_widgets: list[tk.Widget] = []
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.TICK_MS, self._tick)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#ffffff", foreground="#111111")
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff", foreground="#111111")
        style.configure("TLabelframe", background="#ffffff")
        style.configure("TLabelframe.Label", background="#ffffff")
        style.configure("TButton", padding=7)
        style.configure("Primary.TButton", padding=8, font=("Segoe UI Semibold", 10))

    def _entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        unit: str = "",
    ) -> ttk.Entry:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable, width=18)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if unit:
            ttk.Label(row, text=unit, width=8).pack(side=tk.LEFT, padx=(6, 0))
        self.config_widgets.append(entry)
        return entry

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            header,
            text="Free-tip figure-8 path following",
            font=("Segoe UI Semibold", 18),
        ).pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="Matched EI/Cb · 11-node DDER · no adaptation",
            foreground="#555555",
        ).pack(side=tk.RIGHT)

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(body, width=420)
        view = ttk.Frame(body)
        body.add(controls, weight=0)
        body.add(view, weight=1)

        config_tabs = ttk.Notebook(controls)
        config_tabs.pack(fill=tk.X, pady=(0, 8))
        task_tab = ttk.Frame(config_tabs, padding=8)
        objective_tab = ttk.Frame(config_tabs, padding=8)
        mppi_tab = ttk.Frame(config_tabs, padding=8)
        observer_tab = ttk.Frame(config_tabs, padding=8)
        config_tabs.add(task_tab, text="Task")
        config_tabs.add(objective_tab, text="Objective")
        config_tabs.add(mppi_tab, text="MPPI")
        config_tabs.add(observer_tab, text="Observer")

        model = ttk.LabelFrame(task_tab, text="Model and initial condition", padding=8)
        model.pack(fill=tk.X)
        row = ttk.Frame(model)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Cable model", width=24).pack(side=tk.LEFT)
        model_entry = ttk.Entry(row, textvariable=self.model_path_var)
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse = ttk.Button(row, text="…", width=3, command=self.browse_model)
        browse.pack(side=tk.LEFT, padx=(5, 0))
        self.config_widgets.extend((model_entry, browse))
        self._entry(model, "Initial drone XYZ", self.initial_drone_var, "m")

        reference = ttk.LabelFrame(task_tab, text="Figure-8 reference", padding=8)
        reference.pack(fill=tk.X, pady=(8, 0))
        self._entry(reference, "Center XYZ", self.center_var, "m/auto")
        self._entry(reference, "X amplitude", self.amplitude_x_var, "m")
        self._entry(reference, "Y amplitude", self.amplitude_y_var, "m")
        ttk.Label(
            reference,
            text=(
                "The path is flat in X-Y. It has no reference speed or loop "
                "frequency: MPPI chooses how quickly the tip advances."
            ),
            foreground="#555555",
            wraplength=370,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(5, 0))

        objective = ttk.LabelFrame(objective_tab, text="Tracking objective", padding=8)
        objective.pack(fill=tk.X, pady=(8, 0))
        self._entry(objective, "Contour tracking", self.tracking_weight_var)
        self._entry(
            objective,
            "Geometric progress reward",
            self.path_progress_reward_weight_var,
        )
        self._entry(
            objective,
            "Control effort",
            self.control_effort_weight_var,
        )
        self._entry(
            objective,
            "Command smoothness",
            self.control_smoothness_weight_var,
        )
        self._entry(
            objective,
            "Free-tip motion smoothness",
            self.tip_motion_smoothness_weight_var,
        )
        ttk.Label(
            objective,
            text=(
                "MPCC-style objective: contour error minus geometric path progress, "
                "with free-tip acceleration, effort, and command smoothness penalties. "
                "No reference speed or swing maneuver is prescribed."
            ),
            foreground="#555555",
            wraplength=370,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(5, 0))

        comparison = ttk.LabelFrame(task_tab, text="Observation comparison", padding=8)
        comparison.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(comparison, text="Controller observation", width=24).pack(side=tk.LEFT)
        mode = ttk.Combobox(
            comparison,
            textvariable=self.mode_var,
            values=("full", "endpoint", "history"),
            state="readonly",
            width=16,
        )
        mode.pack(side=tk.LEFT, fill=tk.X, expand=True)
        mode.bind("<<ComboboxSelected>>", self._on_observation_mode_changed)
        self.mode_selector = mode
        ttk.Label(
            comparison,
            text=(
                "full: distributed truth. endpoint: instantaneous root/tip state. "
                "history: root/tip-position history corrected through known DDER. "
                "This can be switched during flight and applies at the next replan."
            ),
            foreground="#555555",
            wraplength=370,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(7, 0))

        compute = ttk.LabelFrame(mppi_tab, text="MPPI compute", padding=8)
        compute.pack(fill=tk.X, pady=(8, 0))
        self._entry(compute, "Prediction horizon", self.horizon_var, "s")
        self._entry(compute, "Samples", self.samples_var)
        self._entry(compute, "Iterations", self.iterations_var)
        self._entry(compute, "Acceleration knots", self.knots_var)
        ttk.Label(
            compute,
            text=(
                "Keep acceleration-knot spacing near 0.1 s when changing the horizon "
                "(1 s → 11 knots, 2 s → 21 knots). Longer horizons also increase "
                "the MPPI search dimension."
            ),
            foreground="#555555",
            wraplength=370,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(3, 5))
        self._entry(compute, "Physics rate", self.physics_rate_var, "Hz")
        self._entry(compute, "Control rate", self.control_rate_var, "Hz")
        self._entry(compute, "Replanning rate", self.replan_rate_var, "Hz")
        self._entry(compute, "Seed", self.seed_var)
        bootstrap = ttk.Checkbutton(
            compute,
            text="Periodic out-and-back bootstrap on first solve",
            variable=self.periodic_bootstrap_var,
        )
        bootstrap.pack(anchor=tk.W, pady=(5, 0))
        self.config_widgets.append(bootstrap)

        observer = ttk.LabelFrame(
            observer_tab, text="DDER moving-history observer", padding=8
        )
        observer.pack(fill=tk.X)
        self._entry(observer, "History duration", self.observer_history_var, "s")
        self._entry(observer, "Spatial modes", self.observer_modes_var)
        self._entry(
            observer,
            "Position correction scale",
            self.observer_position_scale_var,
            "m",
        )
        self._entry(
            observer,
            "Velocity correction scale",
            self.observer_velocity_scale_var,
            "m/s",
        )
        self._entry(observer, "Arrival/prior weight", self.observer_prior_weight_var)
        self._entry(observer, "LM damping", self.observer_lm_damping_var)
        self._entry(observer, "Finite-difference step", self.observer_fd_step_var)
        ttk.Label(
            observer,
            text=(
                "Uses only causal attachment motion and free-tip position history. "
                "EI/Cb remain fixed and identical to the plant."
            ),
            foreground="#555555",
            wraplength=370,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(5, 0))

        actions = ttk.LabelFrame(controls, text="Simulation", padding=8)
        actions.pack(fill=tk.X, pady=(8, 0))
        row = ttk.Frame(actions)
        row.pack(fill=tk.X)
        self.play_button = ttk.Button(
            row, text="Play", command=self.play, style="Primary.TButton"
        )
        self.play_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.pause_button = ttk.Button(
            row, text="Pause", command=self.pause, state=tk.DISABLED
        )
        self.pause_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.stop_button = ttk.Button(
            row, text="Stop", command=self.stop, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        self.reset_button = ttk.Button(row, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        ttk.Label(actions, textvariable=self.current_error_var).pack(anchor=tk.W, pady=(8, 0))
        ttk.Label(actions, textvariable=self.rmse_var).pack(anchor=tk.W)
        ttk.Label(actions, textvariable=self.timing_var).pack(anchor=tk.W)

        summary = ttk.LabelFrame(controls, text="Run summary", padding=8)
        summary.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ttk.Label(
            summary,
            textvariable=self.summary_var,
            wraplength=370,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        ttk.Label(
            controls,
            textvariable=self.status_var,
            foreground="#444444",
            wraplength=400,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 0))

        self.scene = Figure8TrackingCanvas(view)
        self.scene.pack(fill=tk.BOTH, expand=True)
        self.error_plot = TrackingErrorCanvas(view)
        self.error_plot.pack(fill=tk.X, pady=(8, 0))

    @staticmethod
    def _vector(text: str, name: str) -> tuple[float, float, float]:
        values = tuple(float(value.strip()) for value in text.split(","))
        if len(values) != 3 or any(not math.isfinite(value) for value in values):
            raise ValueError(f"{name} must contain three finite comma-separated values.")
        return values  # type: ignore[return-value]

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select fitted one-attachment cable model",
            filetypes=(("Cable model", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.model_path_var.set(path)

    def _on_observation_mode_changed(self, _event: tk.Event | None = None) -> None:
        selected = self.mode_var.get()
        if selected not in {"full", "endpoint", "history"}:
            return
        with self.observation_mode_lock:
            self._live_observation_mode = selected  # type: ignore[assignment]
        if self.worker is not None and self.worker.is_alive():
            self.status_var.set(
                f"Observation switch requested: {selected}. "
                "It will apply at the next MPPI replan."
            )
        else:
            self.status_var.set(f"Next run will start with {selected} observation.")

    def _current_observation_mode(self) -> ObservationMode:
        with self.observation_mode_lock:
            return self._live_observation_mode

    def _configuration(self) -> tuple[_RunConfiguration, WhipSimulator, WhipSimulator, object]:
        if not torch.cuda.is_available():
            raise RuntimeError("The accelerated figure-eight simulator requires CUDA.")
        model_path = Path(self.model_path_var.get()).expanduser().resolve()
        source = load_cable_model(model_path)
        initial_drone = self._vector(self.initial_drone_var.get(), "Initial drone XYZ")
        physics_rate = float(self.physics_rate_var.get())
        control_rate = float(self.control_rate_var.get())
        replan_rate = float(self.replan_rate_var.get())
        horizon = float(self.horizon_var.get())
        simulation = SimulationSettings(
            horizon_s=horizon,
            simulation_dt_s=1.0 / physics_rate,
            control_interval_s=1.0 / control_rate,
            attachment_drop_m=0.10,
            maximum_acceleration_m_s2=6.0,
            maximum_speed_m_s=3.0,
        )
        controller_model, plant_model = build_controller_and_truth_models(
            source,
            simulation_dt_s=simulation.simulation_dt_s,
            node_count=11,
            truth_bending_stiffness_scale=1.0,
            truth_bending_damping_scale=1.0,
        )
        if controller_model.sha256 != plant_model.sha256:
            raise RuntimeError("Matched-physics construction produced different models.")
        planner = WhipSimulator(controller_model, simulation, device="cuda")
        plant = WhipSimulator(plant_model, simulation, device="cuda")
        planner.require_online_acceleration()
        plant.require_online_acceleration()
        initial_state = plant.initial_state(initial_drone)
        center_text = self.center_var.get().strip().lower()
        center = (
            tuple(
                float(value)
                for value in initial_state.cable.positions_m[0, -1]
                .detach()
                .cpu()
                .numpy()
            )
            if center_text in {"", "auto"}
            else self._vector(self.center_var.get(), "Figure-eight center XYZ")
        )
        reference = Figure8Reference(
            center_position_m=center,  # type: ignore[arg-type]
            amplitude_x_m=float(self.amplitude_x_var.get()),
            amplitude_y_m=float(self.amplitude_y_var.get()),
        )
        samples = int(self.samples_var.get())
        mppi = MppiSettings(
            samples=samples,
            rollout_batch_size=samples,
            iterations=int(self.iterations_var.get()),
            knot_count=int(self.knots_var.get()),
            temperature=1.0,
            acceleration_noise_sigma_m_s2=3.0,
            noise_decay=0.92,
            seed=int(self.seed_var.get()),
            gradient_guidance_fraction=0.0,
        )
        execution = Figure8ExecutionSettings(
            replan_interval_s=1.0 / replan_rate,
            observation_mode=self.mode_var.get(),  # type: ignore[arg-type]
            duration_s=None,
            periodic_swing_bootstrap=bool(self.periodic_bootstrap_var.get()),
        )
        history_observer = HistoryObserverSettings(
            history_duration_s=float(self.observer_history_var.get()),
            spatial_mode_count=int(self.observer_modes_var.get()),
            position_correction_scale_m=float(
                self.observer_position_scale_var.get()
            ),
            velocity_correction_scale_m_s=float(
                self.observer_velocity_scale_var.get()
            ),
            finite_difference_step=float(self.observer_fd_step_var.get()),
            prior_weight=float(self.observer_prior_weight_var.get()),
            lm_damping=float(self.observer_lm_damping_var.get()),
        )
        cost = TrackingCostSettings(
            tracking_weight=float(self.tracking_weight_var.get()),
            path_progress_reward_weight=float(
                self.path_progress_reward_weight_var.get()
            ),
            control_effort_weight=float(self.control_effort_weight_var.get()),
            control_smoothness_weight=float(
                self.control_smoothness_weight_var.get()
            ),
            tip_motion_smoothness_weight=float(
                self.tip_motion_smoothness_weight_var.get()
            ),
        )
        configuration = _RunConfiguration(
            model_path=model_path,
            initial_drone_position_m=initial_drone,
            reference=reference,
            simulation=simulation,
            mppi=mppi,
            cost=cost,
            execution=execution,
            history_observer=history_observer,
        )
        return configuration, planner, plant, initial_state

    def _set_running(self, running: bool) -> None:
        for widget in self.config_widgets:
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="disabled" if running else "readonly")
                else:
                    widget.configure(state=tk.DISABLED if running else tk.NORMAL)
            except tk.TclError:
                pass
        self.play_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.pause_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def play(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.pause_event.clear()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Running.")
            return
        try:
            configuration, planner, plant, initial_state = self._configuration()
        except Exception as error:
            messagebox.showerror("Figure-8 configuration", str(error), parent=self.root)
            return
        self.stop_event.clear()
        self.pause_event.clear()
        with self.observation_mode_lock:
            self._live_observation_mode = configuration.execution.observation_mode
        self.execution = None
        self.active_configuration = configuration
        self.last_update = None
        self.error_plot.clear()
        self.summary_var.set("Running…")
        self._set_running(True)
        initial_cable = initial_state.cable.positions_m[0].detach().cpu().numpy()
        initial_drone = initial_state.drone_position_m[0].detach().cpu().numpy()
        self.scene.configure_scene(
            initial_cable,
            initial_drone,
            configuration.reference,
            configuration.execution.observation_mode,
        )
        self.status_var.set(
            f"Running {configuration.execution.observation_mode} observation; "
            "controller and plant use identical EI/Cb."
        )
        initial_cable_position = (
            initial_state.cable.positions_m[0].detach().cpu().numpy()
        )
        initial_cable_velocity = (
            initial_state.cable.velocities_m_s[0].detach().cpu().numpy()
        )
        history_observer = DderHistoryObserver(
            planner.snapshot,
            initial_state.cable,
            EndpointHistoryObservation(
                timestamp_s=0.0,
                root_position_m=initial_cable_position[0],
                root_velocity_m_s=initial_cable_velocity[0],
                tip_position_m=initial_cable_position[-1],
            ),
            configuration.history_observer,
            device=planner.device,
        )

        def wait_if_paused() -> None:
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.04)

        def worker() -> None:
            try:
                self.events.put(("status", "Prewarming fixed-history DDER observer…"))
                history_observer.prewarm(planner.settings.simulation_dt_s)
                execution = run_figure8_tracking(
                    planner,
                    plant,
                    initial_state,
                    configuration.reference,
                    configuration.mppi,
                    configuration.cost,
                    configuration.execution,
                    live_update=lambda update: self.events.put(("live", update)),
                    progress=lambda message: self.events.put(("status", message)),
                    cancelled=self.stop_event.is_set,
                    wait_if_paused=wait_if_paused,
                    observation_mode_provider=self._current_observation_mode,
                    history_observer=history_observer,
                )
                paths = save_figure8_execution(
                    DEFAULT_OUTPUT_DIRECTORY,
                    execution,
                    configuration.simulation,
                    configuration.mppi,
                    configuration.cost,
                    configuration.execution,
                )
                self.events.put(("done", (execution, paths)))
            except Exception as error:
                self.events.put(("error", error))

        self.worker = threading.Thread(target=worker, daemon=True)
        self.worker.start()

    def pause(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Running.")
        else:
            self.pause_event.set()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Paused after the current MPPI update.")

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.status_var.set("Stopping after the current MPPI update…")

    def reset(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.stop()
            self.status_var.set("Stopping; press Reset again when the run summary appears.")
            return
        self.execution = None
        self.last_update = None
        self.error_plot.clear()
        self.scene.update_data = None
        self.scene.redraw()
        self.current_error_var.set("Current error: —")
        self.rmse_var.set("RMSE since reset: —")
        self.timing_var.set("MPPI update: —")
        self.summary_var.set("No completed run.")
        self.status_var.set("Reset to the same configured initial condition.")

    def _handle_live(self, update: Figure8LiveUpdate) -> None:
        self.last_update = update
        self.scene.set_live_update(update)
        self.error_plot.set_data(update.recent_time_s, update.recent_tracking_errors_m)
        self.current_error_var.set(f"Current error: {update.tracking_error_m:.4f} m")
        self.rmse_var.set(f"RMSE since reset: {update.rmse_m:.4f} m")
        observer_timing = (
            f" · observer {update.observer_wall_time_s:.3f} s"
            if update.observation_mode == "history"
            else ""
        )
        self.timing_var.set(
            f"MPPI update: {update.planning_wall_time_s:.3f} s{observer_timing}"
        )
        observer_summary = ""
        if update.observation_mode == "history":
            observer_summary = (
                "\nObserver: "
                + (
                    f"history RMSE {update.observer_history_rmse_m:.4f} m; "
                    f"{'corrected' if update.observer_accepted else 'prior retained'}"
                    if update.observer_ready
                    else "collecting history"
                )
            )
        self.summary_var.set(
            f"Mode: {update.observation_mode}\n"
            f"Path progress: {update.path_progress_cycles:.3f} cycles\n"
            f"Tip RMSE: {update.rmse_m:.4f} m\n"
            f"Mean / max / p95: {update.mean_error_m:.4f} / "
            f"{update.maximum_error_m:.4f} / {update.p95_error_m:.4f} m\n"
            f"MPPI mean / p95: {update.mean_planning_wall_time_s:.3f} / "
            f"{update.p95_planning_wall_time_s:.3f} s\n"
            f"Drone horizontal displacement current / max: "
            f"{update.drone_horizontal_displacement_from_start_m:.3f} / "
            f"{update.maximum_drone_horizontal_displacement_from_start_m:.3f} m\n"
            f"Drone vertical displacement: "
            f"{update.drone_vertical_displacement_m:+.3f} m"
            f"{observer_summary}"
        )

    def _handle_done(self, payload: object) -> None:
        execution, paths = payload  # type: ignore[misc]
        assert isinstance(execution, Figure8Execution)
        self.execution = execution
        self._set_running(False)
        self.pause_button.configure(text="Pause")
        summary = execution.summary
        self.summary_var.set(
            f"Mode: {execution.observation_mode}\n"
            f"Completed path: {summary['completed_path_cycles']:.3f} cycles\n"
            f"Tip RMSE: {summary['tip_position_rmse_m']:.4f} m\n"
            f"Mean / max / p95: {summary['tip_position_mean_error_m']:.4f} / "
            f"{summary['tip_position_maximum_error_m']:.4f} / "
            f"{summary['tip_position_p95_error_m']:.4f} m\n"
            f"Normal-speed RMS / p95: "
            f"{summary['tip_normal_velocity_rms_m_s']:.3f} / "
            f"{summary['tip_normal_velocity_p95_m_s']:.3f} m/s\n"
            f"Tangent acceleration / deceleration RMS: "
            f"{summary['tip_tangent_acceleration_rms_m_s2']:.3f} / "
            f"{summary['tip_tangent_deceleration_rms_m_s2']:.3f} m/s²\n"
            f"Command-change RMS: {summary['command_change_rms_m_s2']:.3f} m/s²\n"
            f"MPPI mean / p95: {summary['mppi_mean_update_time_s']:.3f} / "
            f"{summary['mppi_p95_update_time_s']:.3f} s\n"
            f"Drone horizontal displacement RMS / max: "
            f"{summary['drone_horizontal_rms_displacement_from_start_m']:.3f} / "
            f"{summary['drone_horizontal_maximum_displacement_from_start_m']:.3f} m\n"
            f"Drone vertical min / max: "
            f"{summary['drone_minimum_vertical_displacement_m']:+.3f} / "
            f"{summary['drone_maximum_vertical_displacement_m']:+.3f} m\n"
            f"Tip max / motion amplification: "
            f"{summary['tip_maximum_displacement_from_start_m']:.3f} m / "
            f"{summary['tip_to_drone_motion_ratio']:.2f}×"
        )
        self.status_var.set(f"Stopped. Saved {paths[0].name}, NPZ truth, and JSON metadata.")

    def _tick(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "live":
                    self._handle_live(payload)  # type: ignore[arg-type]
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "done":
                    self._handle_done(payload)
                elif kind == "error":
                    self._set_running(False)
                    self.summary_var.set("Run failed.")
                    self.status_var.set(str(payload))
                    messagebox.showerror("Figure-8 simulator", str(payload), parent=self.root)
        except queue.Empty:
            pass
        self.root.after(self.TICK_MS, self._tick)

    def close(self) -> None:
        self.stop_event.set()
        self.pause_event.clear()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    Figure8TrackingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
