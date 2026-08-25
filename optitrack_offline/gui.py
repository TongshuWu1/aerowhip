"""Single Tkinter application for OptiTrack take review and cable fitting."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import queue
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np

from .config import (
    CableSpecification,
    DEFAULT_CONFIG_PATH,
    OptitrackFitConfig,
    load_config,
    save_config,
)
from .data import MotiveCableTake, load_motive_cable_csv
from .fitting import (
    MODEL_SCHEMA,
    FitCancelled,
    OptitrackFitResult,
    TakeWindowAudit,
    audit_optitrack_takes,
    fit_optitrack_takes,
)
from .validation import (
    ValidationRollout,
    evaluate_continuous_validation_take,
)
from .viewer import CSV_DIRECTORY, CableCanvas


ROLE_PATH = Path(__file__).resolve().with_name("take_roles.json")
ROLE_SCHEMA = "optitrack_take_roles_v3"
ROLES = ("training", "validation", "unused")


class ValidationViewer:
    """Synchronized views of one attachment-driven validation simulation."""

    TICK_MS = 8

    def __init__(self, parent: tk.Tk, result: ValidationRollout) -> None:
        self.result = result
        self.frame_index = 0
        self.playing = False
        self._setting_frame = False
        self._play_wall_s = 0.0
        self._play_source_s = 0.0
        self.window = tk.Toplevel(parent)
        self.window.title("Free-tip validation")
        self.window.geometry("1500x820")
        self.window.minsize(1100, 650)
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="One-attachment free-tip validation",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(
                "The recorded attachment-pivot position is supplied continuously. "
                + (
                    "The moving cable is initialized once and never corrected. "
                    if result.observation_interval_s is None
                    else (
                        "The full cable position is corrected every "
                        f"{result.observation_interval_s:g} s. "
                    )
                )
                + f"The fitted model uses {result.node_count} DER nodes and "
                f"{result.take.moving_marker_count} moving markers."
            ),
        ).pack(anchor=tk.W, pady=(1, 8))
        viewports = ttk.Frame(outer)
        viewports.pack(fill=tk.BOTH, expand=True)
        viewports.columnconfigure(0, weight=1)
        viewports.columnconfigure(1, weight=1)
        viewports.rowconfigure(0, weight=1)
        prediction_frame = ttk.LabelFrame(
            viewports,
            text="Prediction with measured reference",
            padding=3,
        )
        prediction_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4))
        actual_frame = ttk.LabelFrame(
            viewports,
            text="Actual OptiTrack measurement",
            padding=3,
        )
        actual_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=(4, 0))
        self.prediction_canvas = CableCanvas(prediction_frame)
        self.prediction_canvas.pack(fill=tk.BOTH, expand=True)
        self.actual_canvas = CableCanvas(actual_frame)
        self.actual_canvas.pack(fill=tk.BOTH, expand=True)
        for canvas in (self.prediction_canvas, self.actual_canvas):
            canvas.set_take(result.take)
        self.prediction_canvas.bind(
            "<B1-Motion>",
            lambda _event: self._sync_view(
                self.prediction_canvas,
                self.actual_canvas,
            ),
            add="+",
        )
        self.actual_canvas.bind(
            "<B1-Motion>",
            lambda _event: self._sync_view(
                self.actual_canvas,
                self.prediction_canvas,
            ),
            add="+",
        )
        self.prediction_canvas.bind(
            "<MouseWheel>",
            lambda _event: self._sync_view(
                self.prediction_canvas,
                self.actual_canvas,
            ),
            add="+",
        )
        self.prediction_canvas.bind(
            "<B3-Motion>",
            lambda _event: self._sync_view(
                self.prediction_canvas,
                self.actual_canvas,
            ),
            add="+",
        )
        self.actual_canvas.bind(
            "<B3-Motion>",
            lambda _event: self._sync_view(
                self.actual_canvas,
                self.prediction_canvas,
            ),
            add="+",
        )
        self.actual_canvas.bind(
            "<MouseWheel>",
            lambda _event: self._sync_view(
                self.actual_canvas,
                self.prediction_canvas,
            ),
            add="+",
        )

        controls = ttk.Frame(outer, padding=(0, 8, 0, 0))
        controls.pack(fill=tk.X)
        self.play_button = ttk.Button(
            controls,
            text="Play",
            command=self.toggle_play,
        )
        self.play_button.pack(side=tk.LEFT)
        ttk.Button(controls, text="Reset", command=self.reset_sequence).pack(
            side=tk.LEFT,
            padx=(7, 0),
        )
        ttk.Button(controls, text="Reset views", command=self.reset_views).pack(
            side=tk.LEFT,
            padx=(7, 0),
        )
        ttk.Label(
            controls,
            text=(
                "Attachment-only after initialization"
                if result.observation_interval_s is None
                else f"Full cable position every {result.observation_interval_s:g} s"
            ),
            style="Value.TLabel",
        ).pack(side=tk.LEFT, padx=(14, 0))

        lead = ttk.Frame(outer)
        lead.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(lead, text="Sequence frame").pack(side=tk.LEFT)
        self.frame_scale = ttk.Scale(
            lead,
            from_=0,
            to=result.final_frame,
            command=self._select_frame,
        )
        self.frame_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 7))
        self.status = ttk.Label(lead, text="")
        self.status.pack(side=tk.RIGHT)
        self.summary_var = tk.StringVar()
        self.summary = ttk.Label(
            outer,
            textvariable=self.summary_var,
            style="Value.TLabel",
        )
        self.summary.pack(anchor=tk.W, pady=(7, 0))
        self._show()
        self.window.after(self.TICK_MS, self._tick)

    @staticmethod
    def _sync_view(source: CableCanvas, target: CableCanvas) -> None:
        target.azimuth_deg = source.azimuth_deg
        target.elevation_deg = source.elevation_deg
        target.zoom = source.zoom
        target.pan_offset_m = source.pan_offset_m.copy()
        target.redraw()

    def reset_views(self) -> None:
        self.prediction_canvas.reset_view()
        self._sync_view(self.prediction_canvas, self.actual_canvas)

    def _select_frame(self, value: str) -> None:
        if self._setting_frame:
            return
        self.frame_index = int(
            np.clip(round(float(value)), 0, self.result.final_frame)
        )
        if self.playing:
            self._start_clock()
        self._show()

    def _set_frame(self, value: int) -> None:
        self.frame_index = int(np.clip(value, 0, self.result.final_frame))
        self._setting_frame = True
        self.frame_scale.set(self.frame_index)
        self._setting_frame = False
        self._show()

    def _start_clock(self) -> None:
        self._play_wall_s = time.perf_counter()
        self._play_source_s = float(
            self.result.take.timestamps_s[
                self.result.start_index + self.frame_index
            ]
        )

    def _show(self) -> None:
        truth_index = self.result.start_index + self.frame_index
        prediction = self.result.predictions_m[self.frame_index]
        truth = self.result.take.positions_m[truth_index]
        self.prediction_canvas.frame_index = truth_index
        self.actual_canvas.frame_index = truth_index
        self.prediction_canvas.set_rollout_display(
            "prediction",
            prediction,
            reference_points_m=truth,
            show_correspondence=True,
            marker_node_indices=self.result.marker_node_indices,
        )
        self.actual_canvas.set_rollout_display(
            "actual",
            truth,
            observed=self.result.take.observed[truth_index],
        )
        lead_time = float(self.result.lead_time_s[self.frame_index])
        rmse = float(self.result.per_frame_rmse_m[self.frame_index])
        error_text = (
            f"{1000.0 * rmse:.3f} mm"
            if np.isfinite(rmse)
            else "no moving-marker observation"
        )
        tip_node = self.result.marker_node_indices[-1]
        tip_error = float("nan")
        if bool(self.result.take.observed[truth_index, -1]):
            tip_error = float(
                np.linalg.norm(prediction[tip_node] - truth[-1])
            )
        tip_error_text = (
            f"{1000.0 * tip_error:.3f} mm"
            if np.isfinite(tip_error)
            else "not observed"
        )
        correction_text = (
            "  FULL OBSERVATION"
            if bool(self.result.correction_mask[self.frame_index])
            else ""
        )
        hold_text = (
            "  HELD ATTACHMENT"
            if bool(np.any(self.result.attachment_hold_mask[self.frame_index]))
            else ""
        )
        self.status.configure(
            text=f"frame={self.frame_index}/{self.result.final_frame}  "
            f"t={lead_time:.3f} s  cable={error_text}  tip={tip_error_text}"
            f"{correction_text}{hold_text}"
        )
        horizon_text = ", ".join(
            f"{name}: "
            + ("n/a" if value is None else f"{1000.0 * value:.1f} mm")
            for name, value in self.result.horizon_rmse_m.items()
        )
        self.summary_var.set(
            f"Model: {self.result.node_count} DER nodes / "
            f"{self.result.take.moving_marker_count} moving markers  |  "
            f"Whole-cable RMSE: {1000.0 * self.result.overall_rmse_m:.3f} mm  |  "
            f"Free-tip RMSE: {1000.0 * self.result.free_tip_rmse_m:.3f} mm  |  "
            f"time since observation: {horizon_text}  |  "
            "Prediction: magenta  Actual: green  Correspondence error: dotted"
        )

    def reset_sequence(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        self._set_frame(0)

    def toggle_play(self) -> None:
        if self.playing:
            self.playing = False
            self.play_button.configure(text="Resume")
            return
        if self.frame_index >= self.result.final_frame:
            self._set_frame(0)
        self.playing = True
        self.play_button.configure(text="Pause")
        self._start_clock()

    def _tick(self) -> None:
        if not self.window.winfo_exists():
            return
        if self.playing:
            start = self.result.start_index
            stop = start + self.result.final_frame
            target_time = self._play_source_s + (
                time.perf_counter() - self._play_wall_s
            )
            absolute_index = int(
                np.searchsorted(
                    self.result.take.timestamps_s,
                    target_time,
                    side="right",
                )
                - 1
            )
            absolute_index = int(np.clip(absolute_index, start, stop))
            next_lead = absolute_index - start
            if target_time >= self.result.take.timestamps_s[stop]:
                self.playing = False
                self.play_button.configure(text="Replay")
                next_lead = self.result.final_frame
            if next_lead != self.frame_index:
                self._set_frame(next_lead)
        self.window.after(self.TICK_MS, self._tick)


class OptitrackOfflineGui:
    TICK_MS = 16

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.config = load_config()
        self.roles = self._load_roles()
        self.takes: dict[Path, MotiveCableTake] = {}
        self.take_errors: dict[Path, str] = {}
        self.take_audits: dict[Path, TakeWindowAudit] = {}
        self.take: MotiveCableTake | None = None
        self.frame_index = 0
        self.playing = False
        self._play_wall_s = 0.0
        self._play_source_s = 0.0
        self._setting_slider = False
        self._fit_thread: threading.Thread | None = None
        self._fit_cancel = threading.Event()
        self._active_operation: str | None = None
        self._fit_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.validation_mode_var = tk.StringVar(value="Attachment-only")

        self.root.title("Cable Twin - Offline Fitting")
        self.root.geometry("1500x900")
        self.root.minsize(1120, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._build()
        self.refresh_takes()
        self._load_existing_model()
        self.root.after(self.TICK_MS, self._tick)

    def _style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        background = "#ffffff"
        panel = "#ffffff"
        text = "#111111"
        muted = "#555555"
        border = "#cfcfcf"
        self.root.configure(background=background)
        style.configure(".", background=background, foreground=text, font=("Segoe UI", 10))
        style.configure("TFrame", background=background)
        style.configure("Panel.TFrame", background=panel)
        style.configure("TLabel", background=background, foreground=text)
        style.configure("Panel.TLabel", background=panel, foreground=text)
        style.configure("Muted.TLabel", background=background, foreground=muted)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 20), foreground=text)
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 10), foreground=text)
        style.configure("Value.TLabel", foreground="#005a8c", font=("Segoe UI Semibold", 10))
        style.configure(
            "TButton",
            padding=(10, 6),
            background="#f2f2f2",
            foreground=text,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.map(
            "TButton",
            background=[("active", "#e6eef5"), ("disabled", "#f5f5f5")],
            foreground=[("disabled", "#999999")],
        )
        style.configure(
            "Primary.TButton",
            background="#235f91",
            foreground="#ffffff",
            padding=(12, 7),
            bordercolor="#235f91",
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#194b73"), ("disabled", "#aebdca")],
            foreground=[("disabled", "#f3f3f3")],
        )
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground=text,
            insertcolor=text,
            bordercolor=border,
        )
        style.configure(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground=text,
            rowheight=27,
            bordercolor=border,
        )
        style.configure(
            "Treeview.Heading",
            background="#ededed",
            foreground=text,
            font=("Segoe UI Semibold", 9),
            bordercolor=border,
        )
        style.map(
            "Treeview",
            background=[("selected", "#dbeaf4")],
            foreground=[("selected", text)],
        )
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

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="Cable model identification", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(
                "Review OptiTrack takes, define the training and held-out sets, "
                "then estimate EI and Cb for the one-attached, free-tip cable."
            ),
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(1, 10))

        upper = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        upper.pack(fill=tk.BOTH, expand=True)
        take_panel = ttk.Frame(upper, padding=10)
        view_panel = ttk.Frame(upper, padding=(10, 0, 0, 0))
        upper.add(take_panel, weight=1)
        upper.add(view_panel, weight=3)
        self._build_take_panel(take_panel)
        self._build_view_panel(view_panel)

        controls = ttk.Frame(outer, padding=(0, 10, 0, 0))
        controls.pack(fill=tk.X)
        controls.columnconfigure(0, weight=3)
        controls.columnconfigure(1, weight=2)
        controls.columnconfigure(2, weight=2)
        self._build_cable_panel(controls)
        self._build_fit_panel(controls)
        self._build_result_panel(controls)

        ttk.Label(outer, text="Run log", style="Section.TLabel").pack(anchor=tk.W, pady=(10, 3))
        self.log = scrolledtext.ScrolledText(
            outer,
            height=4,
            background="#ffffff",
            foreground="#111111",
            insertbackground="#111111",
            borderwidth=1,
            relief=tk.SOLID,
            font=("Cascadia Mono", 9),
            state=tk.DISABLED,
        )
        self.log.pack(fill=tk.X)

    def _build_take_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Takes and data split", style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="Select one or more rows, then assign their role.",
            style="Panel.TLabel",
        ).pack(anchor=tk.W, pady=(1, 7))
        tree_frame = ttk.Frame(parent, style="Panel.TFrame")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        columns = (
            "role",
            "rate",
            "frames",
            "duration",
            "markers",
            "complete",
            "windows",
        )
        self.take_tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="tree headings",
            selectmode="extended",
            height=10,
        )
        self.take_tree.heading("#0", text="CSV")
        self.take_tree.heading("role", text="Role")
        self.take_tree.heading("rate", text="Rate")
        self.take_tree.heading("frames", text="Frames")
        self.take_tree.heading("duration", text="Time")
        self.take_tree.heading("markers", text="Markers")
        self.take_tree.heading("complete", text="Complete")
        self.take_tree.heading("windows", text="Clean windows")
        self.take_tree.column("#0", width=215, minwidth=130)
        self.take_tree.column("role", width=82, anchor=tk.CENTER)
        self.take_tree.column("rate", width=62, anchor=tk.E)
        self.take_tree.column("frames", width=62, anchor=tk.E)
        self.take_tree.column("duration", width=62, anchor=tk.E)
        self.take_tree.column("markers", width=58, anchor=tk.CENTER)
        self.take_tree.column("complete", width=67, anchor=tk.E)
        self.take_tree.column("windows", width=92, anchor=tk.E)
        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.take_tree.yview)
        self.take_tree.configure(yscrollcommand=scroll.set)
        self.take_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.take_tree.bind("<<TreeviewSelect>>", self._view_selected_take)
        self.take_tree.tag_configure("training", foreground="#006b3c")
        self.take_tree.tag_configure("validation", foreground="#005a8c")
        self.take_tree.tag_configure("unused", foreground="#666666")
        self.take_tree.tag_configure("invalid", foreground="#b42318")

        files = ttk.Frame(parent, style="Panel.TFrame")
        files.pack(fill=tk.X, pady=(8, 4))
        self.add_button = ttk.Button(files, text="Add CSVs", command=self.add_csvs)
        self.add_button.pack(side=tk.LEFT)
        self.refresh_button = ttk.Button(files, text="Refresh", command=self.refresh_takes)
        self.refresh_button.pack(side=tk.LEFT, padx=(6, 0))
        self.delete_button = ttk.Button(files, text="Delete selected", command=self.delete_takes)
        self.delete_button.pack(side=tk.LEFT, padx=(6, 0))

        roles = ttk.Frame(parent, style="Panel.TFrame")
        roles.pack(fill=tk.X)
        self.training_button = ttk.Button(roles, text="Use for training", command=lambda: self.set_role("training"))
        self.validation_button = ttk.Button(roles, text="Hold out", command=lambda: self.set_role("validation"))
        self.unused_button = ttk.Button(roles, text="Exclude", command=lambda: self.set_role("unused"))
        self.training_button.pack(side=tk.LEFT)
        self.validation_button.pack(side=tk.LEFT, padx=(6, 0))
        self.unused_button.pack(side=tk.LEFT, padx=(6, 0))
        self.audit_button = ttk.Button(
            parent,
            text="Check usable windows",
            command=self.start_take_audit,
        )
        self.audit_button.pack(fill=tk.X, pady=(8, 0))
        self.split_summary = ttk.Label(
            parent,
            text="Training 0  |  Validation 0",
            style="Panel.TLabel",
        )
        self.split_summary.pack(anchor=tk.W, pady=(8, 0))

    def _build_view_panel(self, parent: ttk.Frame) -> None:
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, pady=(0, 7))
        self.play_button = ttk.Button(toolbar, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Reset view", command=self.canvas_reset).pack(side=tk.LEFT, padx=(7, 0))
        self.file_label = ttk.Label(toolbar, text="Select a take")
        self.file_label.pack(side=tk.RIGHT)
        self.canvas = CableCanvas(parent)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.slider = ttk.Scale(parent, from_=0, to=1, command=self._scrub, state=tk.DISABLED)
        self.slider.pack(fill=tk.X, pady=(7, 0))
        self.view_status = ttk.Label(parent, text="No take selected")
        self.view_status.pack(anchor=tk.W, pady=(4, 0))

    def _build_cable_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Measured cable", padding=9)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        panel.columnconfigure(1, weight=1)
        cable = self.config.cable
        self.lengths_var = tk.StringVar(
            value=", ".join(f"{100.0 * value:g}" for value in cable.rest_lengths_m)
        )
        self.bare_mass_var = tk.StringVar(value=f"{1000.0 * cable.bare_cable_mass_kg:g}")
        self.marker_mass_var = tk.StringVar(
            value=", ".join(
                f"{1000.0 * value:g}"
                for value in cable.moving_marker_masses_kg
            )
        )
        self.diameter_var = tk.StringVar(value=f"{1000.0 * cable.diameter_m:g}")
        self.rod_node_count_var = tk.StringVar(value=str(cable.node_count))
        self.cable_summary = tk.StringVar()
        rows = (
            ("Segment lengths", self.lengths_var, "cm"),
            ("Bare cable mass", self.bare_mass_var, "g"),
            ("Moving marker masses", self.marker_mass_var, "g each"),
            ("Diameter", self.diameter_var, "mm"),
            ("Simulated DER nodes", self.rod_node_count_var, "nodes"),
        )
        self.cable_inputs: list[ttk.Entry] = []
        for row, (label, variable, unit) in enumerate(rows):
            ttk.Label(panel, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky=tk.W, pady=2)
            entry = ttk.Entry(panel, textvariable=variable)
            entry.grid(row=row, column=1, sticky=tk.EW, padx=(8, 5), pady=2)
            self.cable_inputs.append(entry)
            ttk.Label(panel, text=unit, style="Panel.TLabel").grid(row=row, column=2, sticky=tk.W, pady=2)
        ttk.Label(panel, textvariable=self.cable_summary, style="Panel.TLabel").grid(
            row=len(rows), column=0, columnspan=3, sticky=tk.W, pady=(6, 4)
        )
        self.save_config_button = ttk.Button(
            panel,
            text="Save configuration",
            command=self.save_cable,
        )
        self.save_config_button.grid(
            row=len(rows) + 1, column=0, columnspan=3, sticky=tk.W
        )
        self._update_cable_summary(cable)

    def _build_fit_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Identification", padding=9)
        panel.grid(row=0, column=1, sticky="nsew", padx=5)
        self.primary_setting_inputs: list[ttk.Entry] = []
        self.optimizer_iterations_var = tk.StringVar(
            value=str(self.config.fit.optimizer_iterations)
        )
        ttk.Label(
            panel,
            text=(
                "Fixed 1 s rollouts; strict measurement preflight; "
                "every clean window is used"
            ),
            style="Panel.TLabel",
            wraplength=310,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(panel, text="Adam updates", style="Panel.TLabel").grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        optimizer_iterations_entry = ttk.Entry(
            panel,
            textvariable=self.optimizer_iterations_var,
            width=8,
        )
        optimizer_iterations_entry.grid(
            row=1, column=1, sticky=tk.E, padx=(8, 0), pady=(8, 0)
        )
        self.primary_setting_inputs.append(optimizer_iterations_entry)
        self.fit_button = ttk.Button(
            panel,
            text="Fit EI + Cb",
            command=self.start_fit,
            style="Primary.TButton",
        )
        self.fit_button.grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        self.stop_fit_button = ttk.Button(
            panel,
            text="Stop fitting",
            command=self.stop_fit,
            state=tk.DISABLED,
        )
        self.stop_fit_button.grid(row=2, column=1, sticky=tk.E, pady=(8, 0))
        ttk.Label(panel, text="Validation observations", style="Panel.TLabel").grid(
            row=3, column=0, sticky=tk.W, pady=(10, 2)
        )
        self.validation_mode_combo = ttk.Combobox(
            panel,
            textvariable=self.validation_mode_var,
            values=(
                "Attachment-only",
                "Every 1 s",
                "Every 5 s",
                "Every 10 s",
            ),
            state="readonly",
            width=15,
        )
        self.validation_mode_combo.grid(
            row=3, column=1, sticky=tk.E, padx=(8, 0), pady=(10, 2)
        )
        self.validation_button_run = ttk.Button(
            panel,
            text="Validate selected take",
            command=self.start_validation,
        )
        self.validation_button_run.grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(6, 0)
        )
        self.fit_status = tk.StringVar(value="Ready")
        ttk.Label(panel, textvariable=self.fit_status, style="Panel.TLabel", wraplength=270).grid(
            row=5, column=0, columnspan=2, sticky=tk.W, pady=(8, 0)
        )

    def _build_result_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(parent, text="Latest model", padding=9)
        panel.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        self.ei_var = tk.StringVar(value="-")
        self.cb_var = tk.StringVar(value="-")
        self.train_error_var = tk.StringVar(value="-")
        self.validation_error_var = tk.StringVar(value="-")
        for row, (label, variable) in enumerate(
            (
                ("EI", self.ei_var),
                ("Cb", self.cb_var),
                ("Training error", self.train_error_var),
                ("Fit-window validation", self.validation_error_var),
            )
        ):
            ttk.Label(panel, text=label, style="Panel.TLabel").grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Label(panel, textvariable=variable, style="Value.TLabel").grid(row=row, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        self.model_file_var = tk.StringVar(value="No fitted model")
        ttk.Label(panel, textvariable=self.model_file_var, style="Panel.TLabel", wraplength=285).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(7, 0)
        )

    def _load_roles(self) -> dict[str, str]:
        if not ROLE_PATH.is_file():
            return {}
        try:
            payload = json.loads(ROLE_PATH.read_text(encoding="utf-8"))
            if payload.get("schema") not in {
                "optitrack_take_roles_v1",
                "optitrack_take_roles_v2",
                ROLE_SCHEMA,
            }:
                return {}
            return {
                str(name): (
                    str(role)
                    if str(role) in ROLES
                    else "unused"
                )
                for name, role in payload.get("roles", {}).items()
            }
        except (OSError, json.JSONDecodeError, TypeError):
            return {}

    def _save_roles(self) -> None:
        payload = {"schema": ROLE_SCHEMA, "roles": self.roles}
        temporary = ROLE_PATH.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(ROLE_PATH)

    def add_csvs(self) -> None:
        selected = filedialog.askopenfilenames(
            title="Add Motive CSV takes",
            initialdir=CSV_DIRECTORY,
            filetypes=(("Motive CSV", "*.csv"),),
        )
        if not selected:
            return
        CSV_DIRECTORY.mkdir(parents=True, exist_ok=True)
        added: list[Path] = []
        try:
            for value in selected:
                source = Path(value).resolve()
                load_motive_cable_csv(source)
                destination = (CSV_DIRECTORY / source.name).resolve()
                if source != destination:
                    if destination.exists():
                        raise FileExistsError(f"A take named {destination.name} already exists.")
                    shutil.copy2(source, destination)
                added.append(destination)
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot add CSV", str(error), parent=self.root)
        self.refresh_takes(select=added[0] if added else None)

    def delete_takes(self) -> None:
        selection = self.take_tree.selection()
        if not selection:
            messagebox.showinfo(
                "Delete takes",
                "Select one or more takes first.",
                parent=self.root,
            )
            return
        directory = CSV_DIRECTORY.resolve()
        paths = [Path(iid).resolve() for iid in selection]
        if any(path.parent != directory for path in paths):
            messagebox.showerror(
                "Cannot delete take",
                "Only CSV files inside the managed OptiTrack take folder can be deleted.",
                parent=self.root,
            )
            return
        names = "\n".join(f"  {path.name}" for path in paths)
        if not messagebox.askyesno(
            "Delete takes permanently?",
            f"Delete {len(paths)} selected take(s) from optitrack_offline/csv?\n\n"
            f"{names}\n\nThis cannot be undone.",
            icon=messagebox.WARNING,
            parent=self.root,
        ):
            return
        deleted: list[Path] = []
        failure: OSError | None = None
        for path in paths:
            try:
                path.unlink()
            except OSError as error:
                failure = error
                break
            deleted.append(path)
        for path in deleted:
            self.roles.pop(path.name, None)
        if deleted:
            self._save_roles()
        if self.take is not None and self.take.source_path.resolve() in deleted:
            self._clear_view()
        self.refresh_takes()
        if failure is not None:
            messagebox.showerror("Cannot delete take", str(failure), parent=self.root)

    def _clear_view(self) -> None:
        self.take = None
        self.frame_index = 0
        self.playing = False
        self.canvas.take = None
        self.canvas.redraw()
        self.play_button.configure(text="Play", state=tk.DISABLED)
        self.slider.configure(from_=0, to=1, state=tk.DISABLED)
        self.file_label.configure(text="Select a take")
        self.view_status.configure(text="No take selected")

    def refresh_takes(
        self,
        select: Path | None = None,
        *,
        preserve_audits: bool = False,
    ) -> None:
        previous = set(self.take_tree.selection()) if hasattr(self, "take_tree") else set()
        if not preserve_audits:
            self.take_audits.clear()
        self.take_tree.delete(*self.take_tree.get_children())
        self.takes.clear()
        self.take_errors.clear()
        CSV_DIRECTORY.mkdir(parents=True, exist_ok=True)
        paths = sorted(CSV_DIRECTORY.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        expected = self.config.cable.marker_count
        for path in paths:
            resolved = path.resolve()
            role = self.roles.get(path.name, "unused")
            audit = self.take_audits.get(resolved)
            window_text = (
                "-"
                if audit is None
                else (
                    "Rejected"
                    if audit.error is not None
                    else f"{audit.accepted_window_count}/{audit.candidate_window_count}"
                )
            )
            try:
                take = load_motive_cable_csv(resolved)
                self.takes[resolved] = take
                complete = 100.0 * float(np.mean(np.all(take.observed, axis=1)))
                marker_text = f"1P+{take.moving_marker_count}M"
                tag = (
                    role
                    if take.marker_count == expected and take.has_attachment
                    else "invalid"
                )
                values = (
                    role.title(),
                    f"{take.export_rate_hz:g}",
                    take.frame_count,
                    f"{take.duration_s:.1f}s",
                    marker_text,
                    f"{complete:.0f}%",
                    window_text,
                )
            except (OSError, ValueError) as error:
                self.take_errors[resolved] = str(error)
                tag = "invalid"
                values = (
                    f"{role.title()} / INVALID",
                    "-",
                    "-",
                    "-",
                    "Wrong CSV",
                    "Invalid",
                    "Rejected",
                )
            self.take_tree.insert("", tk.END, iid=str(resolved), text=path.name, values=values, tags=(tag,))
        children = set(self.take_tree.get_children())
        target = str(select.resolve()) if select is not None else None
        if target in children:
            self.take_tree.selection_set(target)
            self.take_tree.see(target)
        elif previous & children:
            self.take_tree.selection_set(tuple(previous & children))
        elif children:
            first = self.take_tree.get_children()[0]
            self.take_tree.selection_set(first)
        self._update_split_summary()

    def set_role(self, role: str) -> None:
        selection = self.take_tree.selection()
        if not selection:
            return
        for iid in selection:
            path = Path(iid)
            self.roles[path.name] = role
            values = list(self.take_tree.item(iid, "values"))
            tag = role
            take = self.takes.get(path)
            if (
                take is None
                or take.marker_count != self.config.cable.marker_count
                or not take.has_attachment
            ):
                tag = "invalid"
            values[0] = (
                role.title()
                if tag != "invalid"
                else f"{role.title()} / INVALID"
            )
            self.take_tree.item(iid, values=values)
            self.take_tree.item(iid, tags=(tag,))
        self._save_roles()
        self.take_audits.clear()
        for iid in self.take_tree.get_children():
            values = list(self.take_tree.item(iid, "values"))
            values[6] = "-"
            self.take_tree.item(iid, values=values)
        self._update_split_summary()

    def _update_split_summary(self) -> None:
        training = sum(
            self.roles.get(Path(iid).name, "unused") == "training"
            for iid in self.take_tree.get_children()
        )
        validation = sum(
            self.roles.get(Path(iid).name, "unused") == "validation"
            for iid in self.take_tree.get_children()
        )
        self.split_summary.configure(
            text=f"Training {training}  |  Validation {validation}"
        )

    def _view_selected_take(self, _event: object = None) -> None:
        selection = self.take_tree.selection()
        if not selection:
            return
        path = Path(selection[0])
        if path in self.take_errors:
            error = self.take_errors[path]
            self._clear_view()
            self.file_label.configure(text=path.name)
            self.view_status.configure(text="INVALID ONE-ATTACHMENT CSV: " + error)
            return
        take = self.takes.get(path)
        if take is None:
            return
        self.take = take
        self.frame_index = 0
        self.playing = False
        self.canvas.set_take(take)
        static = take.max_marker_displacement_m <= 1.0e-9
        self.play_button.configure(text="Play", state=tk.DISABLED if static else tk.NORMAL)
        self.slider.configure(to=max(take.frame_count - 1, 1), state=tk.NORMAL)
        self.file_label.configure(text=path.name)
        self.set_frame(0)
        if static:
            self.view_status.configure(text="Static CSV: the marker coordinates do not change.")

    def set_frame(self, frame_index: int) -> None:
        if self.take is None:
            return
        self.frame_index = int(np.clip(frame_index, 0, self.take.frame_count - 1))
        self.canvas.set_frame(self.frame_index)
        self._setting_slider = True
        self.slider.set(self.frame_index)
        self._setting_slider = False
        observed = self.take.observed[self.frame_index]
        attachment = int(self.take.attachment_observed[self.frame_index])
        elapsed = self.take.timestamps_s[self.frame_index] - self.take.timestamps_s[0]
        self.view_status.configure(
            text=(
                f"frame {self.take.frame_numbers[self.frame_index]}  |  "
                f"t={elapsed:.3f}s  |  vertices={int(np.sum(observed))}/"
                f"{self.take.marker_count}  |  attachment={attachment}/1"
            )
        )

    def _scrub(self, value: str) -> None:
        if self.take is None or self._setting_slider:
            return
        self.set_frame(int(round(float(value))))
        if self.playing:
            self._start_clock()

    def toggle_play(self) -> None:
        if self.take is None:
            return
        if self.playing:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        if self.frame_index >= self.take.frame_count - 1:
            self.set_frame(0)
        self.playing = True
        self.play_button.configure(text="Pause")
        self._start_clock()

    def _start_clock(self) -> None:
        assert self.take is not None
        self._play_wall_s = time.perf_counter()
        self._play_source_s = float(self.take.timestamps_s[self.frame_index])

    def canvas_reset(self) -> None:
        self.canvas.reset_view()

    def _read_config(self) -> OptitrackFitConfig:
        length_tokens = self.lengths_var.get().replace(",", " ").split()
        rest_lengths = tuple(float(value) * 0.01 for value in length_tokens)
        marker_count = len(rest_lengths) + 1
        marker_intervals = marker_count - 1
        if marker_intervals != 10:
            raise ValueError(
                "Enter exactly ten segment lengths: attachment-to-c1 through "
                "c9-to-free-tip-c10."
            )
        node_count = int(self.rod_node_count_var.get())
        if (
            node_count < 1 + 2 * marker_intervals
            or (node_count - 1) % marker_intervals != 0
        ):
            valid_counts = ", ".join(
                str(1 + subdivision * marker_intervals)
                for subdivision in range(2, 9)
            )
            raise ValueError(
                f"With {marker_count} measured marker sites, Simulated DER nodes "
                f"must be one of: {valid_counts}."
            )
        rod_subdivision = (node_count - 1) // marker_intervals
        marker_masses = tuple(
            float(value) * 0.001
            for value in self.marker_mass_var.get().replace(",", " ").split()
        )
        cable = CableSpecification(
            rest_lengths_m=rest_lengths,
            bare_cable_mass_kg=float(self.bare_mass_var.get()) * 0.001,
            moving_marker_masses_kg=marker_masses,
            diameter_m=float(self.diameter_var.get()) * 0.001,
            rod_segments_per_marker_interval=rod_subdivision,
        )
        fit = replace(
            self.config.fit,
            optimizer_iterations=int(self.optimizer_iterations_var.get()),
            window_frames=100,
            window_stride=100,
        )
        validation = self.config.validation
        return OptitrackFitConfig(
            cable=cable,
            fit=fit,
            model_path=self.config.model_path,
            validation=validation,
        )

    def _update_cable_summary(self, cable: CableSpecification) -> None:
        self.cable_summary.set(
            f"{cable.marker_count} measured points -> {cable.node_count} DER nodes  |  "
            f"L={cable.length_m:.3f} m  |  "
            f"dynamic mass={1000.0 * cable.total_dynamic_mass_kg:.2f} g"
        )

    def save_cable(self) -> bool:
        try:
            config = self._read_config()
            save_config(config, DEFAULT_CONFIG_PATH)
        except (OSError, ValueError) as error:
            messagebox.showerror("Invalid configuration", str(error), parent=self.root)
            return False
        self.config = config
        self.rod_node_count_var.set(str(config.cable.node_count))
        self.optimizer_iterations_var.set(str(config.fit.optimizer_iterations))
        self._update_cable_summary(config.cable)
        self.refresh_takes()
        self.fit_status.set("Configuration saved")
        return True

    def start_take_audit(self) -> None:
        """Check every take and retain only scientifically valid fit windows."""

        if self._fit_thread is not None and self._fit_thread.is_alive():
            return
        if not self.save_cable():
            return
        paths = [Path(iid).resolve() for iid in self.take_tree.get_children()]
        take_roles = [
            (path, self.roles.get(path.name, "unused"))
            for path in paths
            if self.roles.get(path.name, "unused") in {"training", "validation"}
        ]
        if not take_roles:
            messagebox.showinfo(
                "Check takes",
                "Mark at least one take as Training or Validation first.",
                parent=self.root,
            )
            return
        self._fit_cancel.clear()
        self._active_operation = "audit"
        self._set_fitting(True)
        self._clear_log()
        self.fit_status.set("Checking all takes...")
        self._append_log(
            "Strict preflight: measurement integrity, one-second continuity, "
            "physical feasibility, and marker-conditioned initialization."
        )
        self._append_log(
            "No fitted residual or validation result is used to select windows."
        )

        def worker() -> None:
            try:
                audits = audit_optitrack_takes(
                    take_roles,
                    self.config,
                    progress=lambda text: self._fit_events.put(("log", text)),
                )
            except Exception as error:
                self._fit_events.put(("audit_error", str(error)))
            else:
                self._fit_events.put(("audit_result", audits))

        self._fit_thread = threading.Thread(
            target=worker,
            name="optitrack-take-audit",
            daemon=True,
        )
        self._fit_thread.start()

    def start_fit(self) -> None:
        if self._fit_thread is not None and self._fit_thread.is_alive():
            return
        if not self.save_cable():
            return
        paths = [Path(iid) for iid in self.take_tree.get_children()]
        training = [path for path in paths if self.roles.get(path.name, "unused") == "training"]
        validation = [path for path in paths if self.roles.get(path.name, "unused") == "validation"]
        if not training:
            messagebox.showinfo(
                "Fit EI + Cb",
                "Mark at least one take as Training.",
                parent=self.root,
            )
            return
        incompatible = [
            path.name
            for path in (*training, *validation)
            if path not in self.takes
            or self.takes[path].marker_count != self.config.cable.marker_count
            or not self.takes[path].has_attachment
        ]
        if incompatible:
            messagebox.showerror(
                "Selected takes are incompatible",
                "The one-attachment model requires one rigid pivot plus c1...c10. "
                "Mark these legacy/incompatible takes as Unused:\n\n"
                + "\n".join(incompatible),
                parent=self.root,
            )
            return
        self._fit_cancel.clear()
        self._active_operation = "fit"
        self._set_fitting(True)
        self._clear_log()
        self._append_log(
            f"Starting EI + Cb fit with {len(training)} "
            f"training and {len(validation)} validation take(s)."
        )
        self._append_log(
            f"fit horizon={self.config.fit.window_frames} frames, "
            f"stride={self.config.fit.window_stride}, "
            f"Adam updates={self.config.fit.optimizer_iterations}, "
            f"device={self.config.fit.device}"
        )
        self.fit_status.set(f"Fitting on {self.config.fit.device.upper()}...")

        def worker() -> None:
            try:
                result = fit_optitrack_takes(
                    training,
                    validation,
                    self.config,
                    progress=lambda text: self._fit_events.put(("log", text)),
                    cancelled=self._fit_cancel.is_set,
                )
            except FitCancelled as error:
                self._fit_events.put(("cancelled", str(error)))
            except Exception as error:
                self._fit_events.put(("error", str(error)))
            else:
                self._fit_events.put(("result", result))

        self._fit_thread = threading.Thread(target=worker, name="optitrack-fit", daemon=True)
        self._fit_thread.start()

    def start_validation(self) -> None:
        if self._fit_thread is not None and self._fit_thread.is_alive():
            return
        if not self.save_cable():
            return
        selection = self.take_tree.selection()
        if len(selection) != 1:
            messagebox.showinfo(
                "Test validation",
                "Select exactly one Validation take.",
                parent=self.root,
            )
            return
        path = Path(selection[0])
        if self.roles.get(path.name, "unused") != "validation":
            messagebox.showinfo(
                "Test validation",
                "Mark the selected take as Validation first.",
                parent=self.root,
            )
            return
        if not self.config.model_path.is_file():
            messagebox.showinfo(
                "Test validation",
                "Fit and save a cable model first.",
                parent=self.root,
            )
            return
        take = self.takes.get(path)
        if take is None:
            messagebox.showerror(
                "Test validation",
                "The selected take is not a valid one-attachment CSV.\n\n"
                + self.take_errors.get(path, "The take could not be loaded."),
                parent=self.root,
            )
            return
        self._fit_cancel.clear()
        self._active_operation = "validation"
        self._set_fitting(True)
        protocols = {
            "Attachment-only": None,
            "Every 1 s": 1.0,
            "Every 5 s": 5.0,
            "Every 10 s": 10.0,
        }
        mode = self.validation_mode_var.get()
        if mode not in protocols:
            raise ValueError(f"Unknown validation mode: {mode}")
        observation_interval_s = protocols[mode]
        self.fit_status.set(f"Validating: {mode}...")
        self._append_log(
            f"Validating {path.name}: {mode}. The attachment pivot is always supplied; "
            + (
                "the moving cable is never corrected after initialization."
                if observation_interval_s is None
                else (
                    "one full cable position observation is supplied every "
                    f"{observation_interval_s:g} s."
                )
            )
        )

        def worker() -> None:
            try:
                result = evaluate_continuous_validation_take(
                    path,
                    self.config.model_path,
                    history_frames=self.config.validation.history_frames,
                    observation_interval_s=observation_interval_s,
                    device=self.config.fit.device,
                )
            except Exception as error:
                self._fit_events.put(("validation_error", str(error)))
            else:
                self._fit_events.put(("validation_result", result))

        self._fit_thread = threading.Thread(
            target=worker,
            name="optitrack-validation",
            daemon=True,
        )
        self._fit_thread.start()

    def stop_fit(self) -> None:
        if self._active_operation != "fit":
            return
        if self._fit_thread is None or not self._fit_thread.is_alive():
            return
        self._fit_cancel.set()
        self.stop_fit_button.configure(state=tk.DISABLED)
        self.fit_status.set("Stopping safely...")
        self._append_log(
            "Stop requested; waiting for the current CUDA operation to finish."
        )

    def _set_fitting(self, fitting: bool) -> None:
        state = tk.DISABLED if fitting else tk.NORMAL
        for button in (
            self.add_button,
            self.refresh_button,
            self.delete_button,
            self.training_button,
            self.validation_button,
            self.unused_button,
            self.audit_button,
            self.fit_button,
            self.validation_button_run,
            self.save_config_button,
        ):
            button.configure(state=state)
        self.stop_fit_button.configure(
            state=(
                tk.NORMAL
                if fitting
                and self._active_operation == "fit"
                and not self._fit_cancel.is_set()
                else tk.DISABLED
            )
        )
        self.validation_mode_combo.configure(
            state="disabled" if fitting else "readonly"
        )
        for entry in (*self.primary_setting_inputs, *self.cable_inputs):
            entry.configure(state=state)

    def _append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text.rstrip() + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _clear_log(self) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _show_result(self, result: OptitrackFitResult) -> None:
        self.ei_var.set(f"{result.bending_stiffness_n_m2:.6g} N m^2")
        self.cb_var.set(f"{result.bending_damping_n_m2_s:.6g} N m^2 s")
        self.train_error_var.set(
            f"{1000.0 * result.train_rmse_m:.3f} mm "
            f"({result.train_window_count} windows)"
        )
        self.validation_error_var.set(
            "Not selected"
            if result.validation_rmse_m is None
            else (
                f"{1000.0 * result.validation_rmse_m:.3f} mm "
                f"({result.validation_window_count} windows)"
            )
        )
        self.model_file_var.set(f"Saved: {result.model_path.name}")
        self.fit_status.set(f"Completed in {result.elapsed_s:.1f} s")

    def _load_existing_model(self) -> None:
        path = self.config.model_path
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != MODEL_SCHEMA:
                self.model_file_var.set("Legacy model: refit required")
                return
            optimized, fit = payload["optimized"], payload["fit"]
            result = OptitrackFitResult(
                bending_stiffness_n_m2=float(optimized["bending_stiffness_n_m2"]),
                bending_damping_n_m2_s=float(optimized["bending_damping_n_m2_s"]),
                train_rmse_m=float(fit["training_rollout_rmse_m"]),
                validation_rmse_m=(
                    None
                    if fit["validation_rollout_rmse_m"] is None
                    else float(fit["validation_rollout_rmse_m"])
                ),
                train_window_count=int(fit["training_window_count"]),
                validation_window_count=int(fit["validation_window_count"]),
                elapsed_s=float(fit["elapsed_s"]),
                model_path=path,
            )
            self._show_result(result)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

    def _tick(self) -> None:
        if self.playing and self.take is not None:
            target = self._play_source_s + time.perf_counter() - self._play_wall_s
            index = int(np.searchsorted(self.take.timestamps_s, target, side="right") - 1)
            index = int(np.clip(index, self.frame_index, self.take.frame_count - 1))
            if index != self.frame_index:
                self.set_frame(index)
            if index >= self.take.frame_count - 1:
                self.playing = False
                self.play_button.configure(text="Play")
        while True:
            try:
                kind, payload = self._fit_events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "error":
                self._active_operation = None
                self._set_fitting(False)
                self.fit_status.set("Fit failed")
                self._append_log("FAILED: " + str(payload))
                messagebox.showerror("OptiTrack fit failed", str(payload), parent=self.root)
            elif kind == "result":
                self._active_operation = None
                self._set_fitting(False)
                self._show_result(payload)  # type: ignore[arg-type]
            elif kind == "cancelled":
                self._active_operation = None
                self._set_fitting(False)
                self.fit_status.set("Fit stopped")
                self._append_log(str(payload))
            elif kind == "audit_error":
                self._active_operation = None
                self._set_fitting(False)
                self.fit_status.set("Take check failed")
                self._append_log("CHECK FAILED: " + str(payload))
                messagebox.showerror("Take check failed", str(payload), parent=self.root)
            elif kind == "audit_result":
                self._active_operation = None
                self._set_fitting(False)
                audits = payload
                if not isinstance(audits, tuple) or not all(
                    isinstance(item, TakeWindowAudit) for item in audits
                ):
                    raise TypeError("Take checker returned an invalid result.")
                self.take_audits = {
                    item.source_path.resolve(): item for item in audits
                }
                excluded: list[str] = []
                usable_windows = 0
                for item in audits:
                    role = self.roles.get(item.source_path.name, "unused")
                    usable_windows += item.accepted_window_count
                    insufficient = item.error is not None or (
                        role == "training"
                        and item.accepted_window_count < 1
                    ) or (
                        role == "validation"
                        and item.accepted_window_count < 1
                    )
                    if role in {"training", "validation"} and insufficient:
                        self.roles[item.source_path.name] = "unused"
                        excluded.append(item.source_path.name)
                if excluded:
                    self._save_roles()
                self.refresh_takes(preserve_audits=True)
                self.fit_status.set(
                    f"Take check complete: {usable_windows} clean windows"
                )
                if excluded:
                    self._append_log(
                        "Automatically marked Unused: " + ", ".join(excluded)
                    )
                else:
                    self._append_log("All selected Training/Validation takes passed.")
            elif kind == "validation_error":
                self._active_operation = None
                self._set_fitting(False)
                self.fit_status.set("Validation failed")
                self._append_log("VALIDATION FAILED: " + str(payload))
                messagebox.showerror("Validation failed", str(payload), parent=self.root)
            elif kind == "validation_result":
                self._active_operation = None
                self._set_fitting(False)
                result = payload
                if not isinstance(result, ValidationRollout):
                    raise TypeError("Validation worker returned an invalid result.")
                self.fit_status.set(
                    f"{result.node_count}-node cable RMSE "
                    f"{1000.0 * result.overall_rmse_m:.3f} mm; tip "
                    f"{1000.0 * result.free_tip_rmse_m:.3f} mm"
                )
                mode = (
                    "attachment-only"
                    if result.observation_interval_s is None
                    else f"full observation every {result.observation_interval_s:g}s"
                )
                horizon_text = ", ".join(
                    f"{name}="
                    + ("n/a" if value is None else f"{1000.0 * value:.2f}mm")
                    for name, value in result.horizon_rmse_m.items()
                )
                self._append_log(
                    f"validation ({mode}): DER nodes={result.node_count}, "
                    f"frames={result.frame_count}, duration={result.lead_time_s[-1]:.3f}s, "
                    f"moving-marker coverage="
                    f"{100.0 * result.observed_dynamic_marker_fraction:.1f}%, "
                    f"attachment holds="
                    f"{int(np.count_nonzero(result.attachment_hold_mask))}, "
                    f"whole-cable error={1000.0 * result.overall_rmse_m:.3f}mm, "
                    f"free-tip error={1000.0 * result.free_tip_rmse_m:.3f}mm; "
                    f"time-since-observation error: {horizon_text}"
                )
                self._append_log(f"saved metrics: {result.result_path}")
                ValidationViewer(self.root, result)
        self.root.after(self.TICK_MS, self._tick)

    def close(self) -> None:
        if self._fit_thread is not None and self._fit_thread.is_alive():
            messagebox.showinfo(
                "Operation active",
                (
                    "Press Stop fitting and wait for the fit to stop before closing."
                    if self._active_operation == "fit"
                    else "Wait for the current operation to finish before closing."
                ),
                parent=self.root,
            )
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    OptitrackOfflineGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
