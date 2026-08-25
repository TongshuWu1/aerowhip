"""Desktop viewer for nominal-controller versus hidden-cable evaluation runs.

The GUI is deliberately a presentation layer.  The causal experiment lives in
``drone_mpc.adaptation_testbed`` and is imported only by the worker thread.  In
particular, this module never constructs a controller observation from the
hidden cable curve shown in the viewport.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import importlib
import math
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Sequence

import numpy as np

from optitrack_offline.config import DEFAULT_MODEL_PATH
from optitrack_offline.viewer import CableCanvas


REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = (
    REPOSITORY_DIRECTORY / "data" / "drone_mpc" / "sac_policy_goals_demo12.pt"
)
DEFAULT_TRIAL_DIRECTORY = REPOSITORY_DIRECTORY / "data" / "drone_mpc" / "testbed"


def _vector(text: str, name: str) -> tuple[float, float, float]:
    values = text.replace(",", " ").split()
    if len(values) != 3:
        raise ValueError(f"{name} requires three numbers.")
    result = tuple(float(value) for value in values)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    return result  # type: ignore[return-value]


def _positive_float(text: str, name: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _read_array(frame: object, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(getattr(frame, name), dtype=np.float64)
    if value.shape[-len(shape_tail) :] != shape_tail or not np.all(np.isfinite(value)):
        raise ValueError(f"Testbed frame field {name} has an invalid shape or value.")
    return value


@dataclass(frozen=True, slots=True)
class _DisplayFrame:
    time_s: float
    hidden_cable_positions_m: np.ndarray
    belief_cable_positions_m: np.ndarray
    drone_position_m: np.ndarray
    attachment_position_m: np.ndarray
    measured_marker_positions_m: np.ndarray
    measured_marker_valid: np.ndarray
    measured_tip_position_m: np.ndarray
    target_position_m: np.ndarray
    impact_direction: np.ndarray
    tip_innovation_norm_m: float
    marker_innovation_rms_m: float
    tip_target_distance_m: float
    commanded_acceleration_m_s2: np.ndarray
    action_updated: bool
    done: bool
    success: bool
    unsafe: bool

    @classmethod
    def from_backend(cls, frame: object) -> "_DisplayFrame":
        hidden = _read_array(frame, "hidden_cable_positions_m", (3,))
        belief = _read_array(frame, "belief_cable_positions_m", (3,))
        markers = _read_array(frame, "measured_marker_positions_m", (3,))
        if hidden.ndim != 2 or belief.ndim != 2 or markers.ndim != 2:
            raise ValueError("Testbed cable curves must have shape nodes x 3.")
        marker_valid = np.asarray(getattr(frame, "measured_marker_valid"), dtype=bool)
        if marker_valid.shape != (len(markers),):
            raise ValueError("Measured marker validity must have one value per marker.")

        def vector(name: str) -> np.ndarray:
            value = _read_array(frame, name, (3,))
            if value.shape != (3,):
                raise ValueError(f"Testbed frame field {name} must have shape 3.")
            return value

        time_s = float(getattr(frame, "time_s"))
        innovation = float(getattr(frame, "innovation_norm_m"))
        marker_innovation = float(getattr(frame, "marker_innovation_rms_m"))
        distance = float(getattr(frame, "tip_target_distance_m"))
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (time_s, innovation, marker_innovation, distance)
        ):
            raise ValueError("Testbed time and error metrics must be finite and non-negative.")
        return cls(
            time_s=time_s,
            hidden_cable_positions_m=hidden,
            belief_cable_positions_m=belief,
            drone_position_m=vector("drone_position_m"),
            attachment_position_m=vector("attachment_position_m"),
            measured_marker_positions_m=markers,
            measured_marker_valid=marker_valid,
            measured_tip_position_m=vector("measured_tip_position_m"),
            target_position_m=vector("target_position_m"),
            impact_direction=vector("impact_direction"),
            tip_innovation_norm_m=innovation,
            marker_innovation_rms_m=marker_innovation,
            tip_target_distance_m=distance,
            commanded_acceleration_m_s2=vector("commanded_acceleration_m_s2"),
            action_updated=bool(getattr(frame, "action_updated")),
            done=bool(getattr(frame, "done", False)),
            success=bool(getattr(frame, "success", False)),
            unsafe=bool(getattr(frame, "unsafe")),
        )


class TimeSeriesPlot(tk.Canvas):
    """One dependency-free metric plot for short testbed episodes."""

    def __init__(self, parent: tk.Misc, *, title: str, color: str, unit: str) -> None:
        super().__init__(
            parent,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#c9ced3",
        )
        self.title = title
        self.color = color
        self.unit = unit
        self.samples: list[tuple[float, float]] = []
        self.bind("<Configure>", lambda _event: self.redraw())

    def clear(self) -> None:
        self.samples.clear()
        self.redraw()

    def set_samples(self, values: Sequence[tuple[float, float]]) -> None:
        self.samples = [
            (float(time_s), float(value))
            for time_s, value in values
            if math.isfinite(time_s) and math.isfinite(value)
        ]
        self.redraw()

    @staticmethod
    def _nice_ceiling(value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            return 1.0
        exponent = 10.0 ** math.floor(math.log10(value))
        normalized = value / exponent
        multiple = 1.0 if normalized <= 1.0 else 2.0 if normalized <= 2.0 else 5.0 if normalized <= 5.0 else 10.0
        return multiple * exponent

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 120)
        height = max(self.winfo_height(), 90)
        left, right, top, bottom = 58.0, width - 15.0, 33.0, height - 29.0
        plot_width = max(1.0, right - left)
        plot_height = max(1.0, bottom - top)
        self.create_text(
            12,
            9,
            anchor=tk.NW,
            text=self.title,
            fill="#1f2933",
            font=("Segoe UI Semibold", 10),
        )
        if not self.samples:
            self.create_text(
                0.5 * (left + right),
                0.5 * (top + bottom),
                text="No trial data",
                fill="#7b858e",
                font=("Segoe UI", 9),
            )
            return
        x_max = max(1.0, max((sample[0] for sample in self.samples), default=1.0))
        y_max = self._nice_ceiling(max((sample[1] for sample in self.samples), default=0.001))
        for tick in range(5):
            fraction = tick / 4.0
            y = bottom - fraction * plot_height
            self.create_line(left, y, right, y, fill="#e5e7eb")
            self.create_text(
                left - 7,
                y,
                anchor=tk.E,
                text=f"{fraction * y_max:g} {self.unit}",
                fill="#66727d",
                font=("Segoe UI", 8),
            )
        for tick in range(5):
            fraction = tick / 4.0
            x = left + fraction * plot_width
            self.create_line(x, top, x, bottom, fill="#eef0f2")
            self.create_text(
                x,
                bottom + 7,
                anchor=tk.N,
                text=f"{fraction * x_max:.1f}s",
                fill="#66727d",
                font=("Segoe UI", 8),
            )
        stride = max(1, math.ceil(len(self.samples) / max(int(plot_width), 1)))
        displayed = self.samples[::stride]
        if displayed[-1] != self.samples[-1]:
            displayed = [*displayed, self.samples[-1]]
        line: list[float] = []
        for time_s, value in displayed:
            line.extend(
                (
                    left + min(max(time_s / x_max, 0.0), 1.0) * plot_width,
                    bottom - min(max(value / y_max, 0.0), 1.0) * plot_height,
                )
            )
        if len(line) >= 4:
            self.create_line(*line, fill=self.color, width=2)
        else:
            x, y = line
        self.create_oval(x - 2, y - 2, x + 2, y + 2, fill=self.color, outline="")


class VerticalScrolledFrame(ttk.Frame):
    """A small scrollbar-backed frame for setup controls on shorter displays."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(
            self,
            background="#ffffff",
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.content = ttk.Frame(self.canvas)
        self._window = self.canvas.create_window(
            (0, 0), window=self.content, anchor=tk.NW
        )
        self.content.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._fit_content_width)
        self.canvas.bind("<MouseWheel>", self._scroll)

    def _update_scroll_region(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _fit_content_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window, width=max(int(event.width), 1))

    def _scroll(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")


class AdaptationTestbedCanvas(CableCanvas):
    """OptiTrack truth and measurement-corrected prediction viewport."""

    def __init__(self, parent: tk.Misc) -> None:
        self.frame: _DisplayFrame | None = None
        self._view_initialized = False
        super().__init__(parent)
        self.configure(
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#c9ced3",
        )

    def clear(self) -> None:
        self.frame = None
        self._view_initialized = False
        self.redraw()

    def set_testbed_frame(self, frame: _DisplayFrame) -> None:
        self.frame = frame
        if not self._view_initialized:
            points = np.concatenate(
                (
                    frame.hidden_cable_positions_m,
                    frame.belief_cable_positions_m,
                    frame.drone_position_m[None],
                    frame.target_position_m[None],
                ),
                axis=0,
            )
            lower = np.min(points, axis=0)
            upper = np.max(points, axis=0)
            lower[2] = min(lower[2], 0.0)
            span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.4)
            self.grid_step_m = self._nice_grid_step(span / 12.0)
            center_xy = 0.5 * (lower[:2] + upper[:2])
            half = max(0.85, 0.72 * span)
            self.floor_bounds_m = (
                math.floor((center_xy[0] - half) / self.grid_step_m) * self.grid_step_m,
                math.ceil((center_xy[0] + half) / self.grid_step_m) * self.grid_step_m,
                math.floor((center_xy[1] - half) / self.grid_step_m) * self.grid_step_m,
                math.ceil((center_xy[1] + half) / self.grid_step_m) * self.grid_step_m,
            )
            self.center_m = 0.5 * (lower + upper)
            self.radius_m = max(float(np.linalg.norm(upper - lower) * 0.65), 0.75)
            self._view_initialized = True
            self.reset_view()
        else:
            self.redraw()

    def _curve(self, values: np.ndarray, color: str, width: int, dash: tuple[int, int] | None) -> None:
        screen, _depth = self._project(values)
        options: dict[str, object] = {"fill": color, "width": width}
        if dash is not None:
            options["dash"] = dash
        self.create_line(*screen.reshape(-1), **options)

    def _marker(self, value: np.ndarray, *, radius: int, fill: str, outline: str) -> None:
        screen, _depth = self._project(value[None])
        x, y = screen[0]
        self.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=fill,
            outline=outline,
            width=2,
        )

    def _draw_floor(self) -> None:
        """Draw a neutral laboratory reference plane for this light viewport."""

        xmin, xmax, ymin, ymax = self.floor_bounds_m
        corners = np.asarray(
            (
                (xmin, ymin, 0.0),
                (xmax, ymin, 0.0),
                (xmax, ymax, 0.0),
                (xmin, ymax, 0.0),
            ),
            dtype=np.float64,
        )
        screen, _depth = self._project(corners)
        self.create_polygon(
            *screen.reshape(-1),
            fill="#f7f8f9",
            outline="#b9c0c6",
            width=1,
            tags=("environment",),
        )
        major_step = 5.0 * self.grid_step_m
        tolerance = self.grid_step_m * 1.0e-5
        x_values = np.arange(xmin, xmax + 0.5 * self.grid_step_m, self.grid_step_m)
        y_values = np.arange(ymin, ymax + 0.5 * self.grid_step_m, self.grid_step_m)
        for x in x_values:
            projected, _depth = self._project(
                np.asarray(((x, ymin, 0.0), (x, ymax, 0.0)))
            )
            major = abs(x / major_step - round(x / major_step)) < tolerance
            self.create_line(
                *projected.reshape(-1),
                fill="#cbd1d6" if major else "#e4e7ea",
                width=1,
                tags=("environment",),
            )
        for y in y_values:
            projected, _depth = self._project(
                np.asarray(((xmin, y, 0.0), (xmax, y, 0.0)))
            )
            major = abs(y / major_step - round(y / major_step)) < tolerance
            self.create_line(
                *projected.reshape(-1),
                fill="#cbd1d6" if major else "#e4e7ea",
                width=1,
                tags=("environment",),
            )
        self.create_text(
            *screen[0],
            text=f"  Z = 0 m · grid {self.grid_step_m:g} m",
            anchor=tk.NW,
            fill="#6b747c",
            font=("Segoe UI", 9),
            tags=("environment",),
        )
        self.tag_lower("environment")

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        self.create_text(
            18,
            14,
            anchor=tk.NW,
            text="3D experiment view · world Z up",
            fill="#374151",
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            width - 18,
            14,
            anchor=tk.NE,
            text="Drag rotate · Right-drag pan · Wheel zoom",
            fill="#6b7280",
            font=("Segoe UI", 9),
        )
        if self.frame is None:
            self.create_text(
                width * 0.5,
                max(self.winfo_height(), 1) * 0.5,
                text="Configure the models and run an evaluation",
                fill="#4b5563",
                font=("Segoe UI", 13),
            )
            return
        self._draw_floor()
        self._draw_axes()
        frame = self.frame
        self._curve(frame.hidden_cable_positions_m, "#6b7280", 2, None)
        self._curve(frame.belief_cable_positions_m, "#b26a00", 3, (7, 4))

        target, _depth = self._project(frame.target_position_m[None])
        tx, ty = target[0]
        self.create_oval(
            tx - 10, ty - 10, tx + 10, ty + 10, outline="#b42318", width=3
        )
        arrow_start = frame.target_position_m - 0.18 * frame.impact_direction
        arrow, _depth = self._project(np.stack((arrow_start, frame.target_position_m)))
        self.create_line(*arrow.reshape(-1), fill="#b42318", width=2, arrow=tk.LAST)

        link, _depth = self._project(
            np.stack((frame.drone_position_m, frame.attachment_position_m))
        )
        self.create_line(*link.reshape(-1), fill="#1f5f8b", width=2)
        x, y = link[0]
        radius = 11
        self.create_polygon(
            x,
            y - radius,
            x + radius,
            y,
            x,
            y + radius,
            x - radius,
            y,
            fill="#1f5f8b",
            outline="#143e5a",
            width=2,
        )
        self._marker(
            frame.attachment_position_m,
            radius=5,
            fill="#79a8c5",
            outline="#1f5f8b",
        )
        marker_count = len(frame.measured_marker_positions_m)
        for marker_index, (marker, valid) in enumerate(
            zip(frame.measured_marker_positions_m, frame.measured_marker_valid)
        ):
            self._marker(
                marker,
                radius=6 if marker_index == marker_count - 1 else 4,
                fill="#76528b" if valid else "#c0c5ca",
                outline="#4e365d" if valid else "#8b9298",
            )

        acceleration_norm = float(np.linalg.norm(frame.commanded_acceleration_m_s2))
        outcome = (
            "HIT"
            if frame.success
            else "UNSAFE"
            if frame.unsafe
            else "COMPLETE"
            if frame.done
            else "RUNNING"
        )
        outcome_color = (
            "#18794e" if frame.success else "#b42318" if frame.unsafe else "#1f2933"
        )
        self.create_text(
            18,
            39,
            anchor=tk.NW,
            text=(
                f"t = {frame.time_s:.2f} s  ·  {outcome}  ·  "
                f"|a_cmd| = {acceleration_norm:.2f} m/s²"
            ),
            fill=outcome_color,
            font=("Segoe UI Semibold", 10),
        )


class AdaptationTestbedGui:
    POLL_MS = 33

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cable Twin - Online Testbed (Simulation)")
        self.root.geometry("1500x920")
        self.root.minsize(1180, 760)
        self.root.configure(background="#ffffff")
        self._configure_style()

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.backend_module: Any | None = None
        self.episode: Any | None = None
        self.frames: list[_DisplayFrame] = []
        self.frame_index = 0
        self.playing = False
        self.paused = False
        self.computing = False
        self.close_requested = False
        self._playback_anchor_wall_s = 0.0
        self._playback_anchor_episode_s = 0.0
        self._frame_times_s = np.empty((0,), dtype=np.float64)
        self._timeline_update = False
        self.experiment_inputs: list[tk.Widget] = []
        self._active_run_summary = ""
        self._active_run_file_stem = "hidden_model_evaluation_trial"

        default_model = str(DEFAULT_MODEL_PATH.resolve())
        self.nominal_model_var = tk.StringVar(value=default_model)
        self.policy_var = tk.StringVar(value=str(DEFAULT_POLICY_PATH.resolve()))
        self.hidden_model_var = tk.StringVar(value=default_model)
        self.hidden_ei_scale_var = tk.StringVar(value="1.00")
        self.hidden_cb_scale_var = tk.StringVar(value="1.00")
        self.initial_drone_var = tk.StringVar(value="0.0, 0.0, 0.0")
        self.target_var = tk.StringVar(value="0.8, 0.0, -0.1")
        self.direction_display_var = tk.StringVar(value="radial horizontal (automatic)")
        self.seed_var = tk.StringVar(value="42")
        self.controller_status_var = tk.StringVar(
            value=(
                "Full controller-grid marker feedback at policy-control boundaries "
                "(perfect association)"
            )
        )
        self.truth_status_var = tk.StringVar(value="Hidden from controller; visible only for scoring")
        self.status_var = tk.StringVar(value="Ready")
        self.time_var = tk.StringVar(value="0.00 s")
        self.innovation_var = tk.StringVar(value="--")
        self.distance_var = tk.StringVar(value="--")
        self.outcome_var = tk.StringVar(value="Not run")
        self.active_run_var = tk.StringVar(
            value="No active run — configure the experiment, then Run."
        )
        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline_text_var = tk.StringVar(value="frame 0 / 0")

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.POLL_MS, self._tick)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            ".",
            background="#ffffff",
            foreground="#1f2328",
            font=("Segoe UI", 10),
        )
        style.configure("TFrame", background="#ffffff")
        style.configure("Panel.TFrame", background="#f5f6f7")
        style.configure("Notice.TFrame", background="#f5f6f7")
        style.configure("Card.TFrame", background="#ffffff", relief=tk.SOLID, borderwidth=1)
        style.configure("TLabel", background="#ffffff", foreground="#1f2328")
        style.configure("Panel.TLabel", background="#f5f6f7", foreground="#1f2328")
        style.configure("Notice.TLabel", background="#f5f6f7", foreground="#4b535b")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 20))
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 11))
        style.configure(
            "PanelSection.TLabel",
            background="#f5f6f7",
            foreground="#1f2328",
            font=("Segoe UI Semibold", 11),
        )
        style.configure("Muted.TLabel", foreground="#5f6871")
        style.configure("Field.TLabel", foreground="#30363d")
        style.configure(
            "TLabelframe",
            background="#ffffff",
            foreground="#1f2328",
            bordercolor="#c9ced3",
            lightcolor="#c9ced3",
            darkcolor="#c9ced3",
            relief=tk.SOLID,
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background="#ffffff",
            foreground="#30363d",
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "TButton",
            background="#f4f5f6",
            foreground="#1f2328",
            bordercolor="#aeb5bb",
            padding=(9, 6),
        )
        style.map(
            "TButton",
            background=[("active", "#e9ecef"), ("disabled", "#f2f3f4")],
            foreground=[("disabled", "#9aa1a7")],
        )
        style.configure(
            "Primary.TButton",
            background="#245f85",
            foreground="#ffffff",
            bordercolor="#245f85",
            padding=(9, 7),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#174866"), ("disabled", "#a8bdca")],
            foreground=[("disabled", "#eef2f4")],
        )
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground="#111418",
            bordercolor="#aeb5bb",
            insertcolor="#111418",
            padding=4,
        )
        style.configure("TNotebook", background="#f5f6f7", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 6))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")])
        style.configure(
            "Value.TLabel",
            font=("Segoe UI Semibold", 12),
            foreground="#111418",
        )
        style.configure("Status.TLabel", foreground="#4b535b")

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="Online testbed (simulation)",
            style="Title.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="Evaluate a frozen nominal controller against a separately configured cable plant.",
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(1, 7))

        notice = ttk.Frame(outer, padding=(9, 6), style="Notice.TFrame")
        notice.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            notice,
            text=(
                "Model boundary: acceleration-tracked point mass; cable reaction is "
                "not force-coupled."
            ),
            style="Notice.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            notice,
            text=(
                "Adaptation status: disabled. The nominal DDER and SAC policy remain "
                "frozen; hidden state is used only for display and scoring."
            ),
            style="Notice.TLabel",
        ).pack(anchor=tk.W, pady=(2, 0))

        status_bar = ttk.Frame(outer, padding=(0, 5, 0, 0))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(
            status_bar, textvariable=self.status_var, style="Status.TLabel"
        ).pack(anchor=tk.W)

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)
        setup_panel = ttk.Frame(body, padding=10, style="Panel.TFrame")
        workspace = ttk.Frame(body, padding=(12, 0, 0, 0))
        body.add(setup_panel, weight=1)
        body.add(workspace, weight=3)
        setup_panel.columnconfigure(0, weight=1)
        setup_panel.rowconfigure(1, weight=1)

        ttk.Label(
            setup_panel, text="Experiment setup", style="PanelSection.TLabel"
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 6))
        notebook = ttk.Notebook(setup_panel)
        notebook.grid(row=1, column=0, sticky=tk.NSEW)
        setup_scroller = VerticalScrolledFrame(notebook)
        setup = setup_scroller.content
        provenance = ttk.Frame(notebook, padding=8)
        notebook.add(setup_scroller, text="Setup")
        notebook.add(provenance, text="Provenance")

        controller = ttk.LabelFrame(
            setup, text="Controller-visible inputs", padding=8
        )
        controller.pack(fill=tk.X)
        self.experiment_inputs.extend(
            self._path_row(
                controller,
                "Nominal cable",
                self.nominal_model_var,
                self.browse_nominal,
                "JSON",
            )
        )
        self.experiment_inputs.extend(
            self._path_row(
                controller,
                "Frozen SAC policy",
                self.policy_var,
                self.browse_policy,
                "PT",
            )
        )
        ttk.Label(
            controller,
            textvariable=self.controller_status_var,
            style="Muted.TLabel",
            justify=tk.LEFT,
            wraplength=330,
        ).pack(anchor=tk.W, pady=(5, 0))

        truth = ttk.LabelFrame(setup, text="Hidden plant · evaluation only", padding=8)
        truth.pack(fill=tk.X, pady=(8, 0))
        self.experiment_inputs.extend(
            self._path_row(
                truth,
                "Hidden cable",
                self.hidden_model_var,
                self.browse_hidden,
                "JSON",
            )
        )
        scale_row = ttk.Frame(truth)
        scale_row.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(scale_row, text="Parameter scale", width=15).pack(side=tk.LEFT)
        ttk.Label(scale_row, text="EI ×").pack(side=tk.LEFT)
        hidden_ei_entry = ttk.Entry(
            scale_row, textvariable=self.hidden_ei_scale_var, width=8
        )
        hidden_ei_entry.pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(scale_row, text="Cb ×").pack(side=tk.LEFT)
        hidden_cb_entry = ttk.Entry(
            scale_row, textvariable=self.hidden_cb_scale_var, width=8
        )
        hidden_cb_entry.pack(side=tk.LEFT, padx=(4, 0))
        self.experiment_inputs.extend((hidden_ei_entry, hidden_cb_entry))
        ttk.Label(
            truth,
            textvariable=self.truth_status_var,
            style="Muted.TLabel",
            justify=tk.LEFT,
            wraplength=330,
        ).pack(anchor=tk.W, pady=(5, 0))

        task = ttk.LabelFrame(setup, text="Trial definition", padding=8)
        task.pack(fill=tk.X, pady=(8, 0))
        self.experiment_inputs.append(
            self._compact_entry(task, "Initial drone XYZ", self.initial_drone_var, "m")
        )
        self.experiment_inputs.append(
            self._compact_entry(task, "Target XYZ", self.target_var, "m")
        )
        direction = ttk.Frame(task)
        direction.pack(fill=tk.X, pady=2)
        ttk.Label(direction, text="Impact direction", width=15).pack(side=tk.LEFT)
        ttk.Label(
            direction,
            textvariable=self.direction_display_var,
            style="Muted.TLabel",
        ).pack(side=tk.LEFT, padx=(5, 0))
        self.experiment_inputs.append(
            self._compact_entry(task, "Seed", self.seed_var, "")
        )
        actions = ttk.LabelFrame(setup_panel, text="Execution", padding=8)
        actions.grid(row=3, column=0, sticky=tk.EW, pady=(7, 0))
        self.run_button = ttk.Button(
            actions,
            text="Run evaluation",
            command=self.run,
            style="Primary.TButton",
        )
        self.run_button.pack(fill=tk.X)
        self.stop_button = ttk.Button(
            actions, text="Stop safely", command=self.stop, state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X, pady=(5, 0))
        playback_actions = ttk.Frame(actions)
        playback_actions.pack(fill=tk.X, pady=(5, 0))
        self.replay_button = ttk.Button(
            playback_actions, text="Replay", command=self.replay, state=tk.DISABLED
        )
        self.replay_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.pause_button = ttk.Button(
            playback_actions, text="Pause", command=self.pause, state=tk.DISABLED
        )
        self.pause_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        record_actions = ttk.Frame(actions)
        record_actions.pack(fill=tk.X, pady=(5, 0))
        self.reset_button = ttk.Button(
            record_actions, text="Clear display", command=self.reset
        )
        self.reset_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.save_button = ttk.Button(
            record_actions, text="Save trial", command=self.save, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        active_run = ttk.LabelFrame(
            setup_panel, text="Run record", padding=(8, 6)
        )
        active_run.grid(row=2, column=0, sticky=tk.EW, pady=(7, 0))
        ttk.Label(
            active_run,
            textvariable=self.active_run_var,
            style="Muted.TLabel",
            justify=tk.LEFT,
            wraplength=330,
        ).pack(anchor=tk.W)

        ttk.Label(
            provenance,
            text="Immutable metadata from the completed trial archive.",
            style="Muted.TLabel",
            wraplength=330,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 6))
        provenance_scroll = ttk.Scrollbar(provenance, orient=tk.VERTICAL)
        provenance_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.provenance_text = tk.Text(
            provenance,
            background="#ffffff",
            foreground="#1f2328",
            selectbackground="#cfe3f1",
            borderwidth=1,
            relief=tk.SOLID,
            highlightthickness=0,
            font=("Cascadia Mono", 8),
            wrap=tk.WORD,
            state=tk.DISABLED,
            yscrollcommand=provenance_scroll.set,
        )
        self.provenance_text.pack(fill=tk.BOTH, expand=True)
        provenance_scroll.configure(command=self.provenance_text.yview)
        self._set_provenance_text(
            "No completed trial.\n\nFull paths, hashes, rates, settings and the "
            "measurement contract are retained in each saved NPZ archive."
        )

        workspace_header = ttk.Frame(workspace)
        workspace_header.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(
            workspace_header, text="Experiment", style="Section.TLabel"
        ).pack(side=tk.LEFT)
        ttk.Button(
            workspace_header,
            text="Reset camera",
            command=lambda: self.canvas.reset_view(),
        ).pack(side=tk.RIGHT)

        legend = ttk.Frame(workspace)
        legend.pack(fill=tk.X, pady=(0, 5))
        for text_value, color in (
            ("● Hidden plant (evaluation only)", "#6b7280"),
            ("— Nominal prediction", "#b26a00"),
            ("● OptiTrack markers", "#76528b"),
            ("○ Target", "#b42318"),
        ):
            ttk.Label(legend, text=text_value, foreground=color).pack(
                side=tk.LEFT, padx=(0, 16)
            )

        self.canvas = AdaptationTestbedCanvas(workspace)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        timeline = ttk.Frame(workspace)
        timeline.pack(fill=tk.X, pady=(5, 0))
        self.step_back_button = ttk.Button(
            timeline,
            text="◀ Step",
            command=lambda: self.step_frame(-1),
            state=tk.DISABLED,
        )
        self.step_back_button.pack(side=tk.LEFT)
        self.timeline_scale = ttk.Scale(
            timeline,
            from_=0.0,
            to=0.0,
            variable=self.timeline_var,
            command=self.scrub,
            state=tk.DISABLED,
        )
        self.timeline_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=7)
        self.step_forward_button = ttk.Button(
            timeline,
            text="Step ▶",
            command=lambda: self.step_frame(1),
            state=tk.DISABLED,
        )
        self.step_forward_button.pack(side=tk.LEFT)
        ttk.Label(
            timeline,
            textvariable=self.timeline_text_var,
            style="Muted.TLabel",
            width=18,
            anchor=tk.E,
        ).pack(side=tk.LEFT, padx=(8, 0))

        cards = ttk.Frame(workspace)
        cards.pack(fill=tk.X, pady=(6, 5))
        for column, (title, variable) in enumerate((
            ("Time", self.time_var),
            ("Pre-correction innovation", self.innovation_var),
            ("Target distance", self.distance_var),
            ("Outcome", self.outcome_var),
        )):
            cards.columnconfigure(column, weight=1, uniform="metric")
            card = ttk.Frame(cards, padding=(8, 5), style="Card.TFrame")
            card.grid(
                row=0,
                column=column,
                sticky=tk.EW,
                padx=(0 if column == 0 else 5, 0),
            )
            ttk.Label(card, text=title, style="Muted.TLabel").pack(anchor=tk.W)
            ttk.Label(card, textvariable=variable, style="Value.TLabel").pack(anchor=tk.W)

        plots = ttk.Frame(workspace)
        plots.pack(fill=tk.X)
        self.innovation_plot = TimeSeriesPlot(
            plots,
            title="Model innovation at control updates",
            color="#b26a00",
            unit="mm",
        )
        self.innovation_plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.distance_plot = TimeSeriesPlot(
            plots,
            title="Hidden free-tip distance to target",
            color="#76528b",
            unit="mm",
        )
        self.distance_plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        plots.configure(height=165)
        self.innovation_plot.configure(height=165)
        self.distance_plot.configure(height=165)

    @staticmethod
    def _path_row(
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        command: object,
        kind: str,
    ) -> tuple[tk.Widget, tk.Widget]:
        field = ttk.Frame(parent)
        field.pack(fill=tk.X, pady=2)
        ttk.Label(field, text=label, style="Field.TLabel").pack(anchor=tk.W)
        row = ttk.Frame(field)
        row.pack(fill=tk.X, pady=(2, 0))
        entry = ttk.Entry(row, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse = ttk.Button(row, text="Browse", command=command)
        browse.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(row, text=kind, style="Muted.TLabel", width=5).pack(
            side=tk.LEFT, padx=(5, 0)
        )
        return entry, browse

    @staticmethod
    def _compact_entry(
        parent: ttk.Frame, label: str, variable: tk.StringVar, unit: str
    ) -> tk.Widget:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=15).pack(side=tk.LEFT)
        entry = ttk.Entry(
            row,
            textvariable=variable,
            width=18 if "XYZ" in label or "direction" in label else 8,
        )
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 3))
        ttk.Label(row, text=unit).pack(side=tk.LEFT)
        return entry

    def _set_provenance_text(self, value: str) -> None:
        self.provenance_text.configure(state=tk.NORMAL)
        self.provenance_text.delete("1.0", tk.END)
        self.provenance_text.insert("1.0", value.rstrip() + "\n")
        self.provenance_text.configure(state=tk.DISABLED)

    def _update_provenance(self, episode: object) -> None:
        settings = getattr(episode, "settings")
        lines = (
            f"schema\n{getattr(episode, 'schema')}",
            f"\ntermination\n{getattr(episode, 'termination_reason')}",
            f"\nnominal model\n{getattr(episode, 'nominal_source_path')}",
            f"nominal source sha256\n{getattr(episode, 'nominal_source_sha256')}",
            f"nominal controller sha256\n{getattr(episode, 'nominal_controller_sha256')}",
            f"\npolicy\n{getattr(episode, 'policy_path')}",
            f"policy sha256\n{getattr(episode, 'policy_sha256')}",
            f"\nhidden model\n{getattr(episode, 'hidden_source_path')}",
            f"hidden source sha256\n{getattr(episode, 'hidden_source_sha256')}",
            f"hidden plant sha256\n{getattr(episode, 'hidden_plant_sha256')}",
            (
                "hidden parameter scale\n"
                f"EI ×{float(getattr(settings, 'hidden_ei_scale')):g}; "
                f"Cb ×{float(getattr(settings, 'hidden_cb_scale')):g}"
            ),
            (
                "\nrates\n"
                f"physics {float(getattr(episode, 'physics_rate_hz')):g} Hz; "
                f"control {float(getattr(episode, 'control_rate_hz')):g} Hz"
            ),
            (
                "marker contract\n"
                f"{len(getattr(episode, 'marker_material_coordinates_m'))} ordered "
                "controller-grid points; perfect association baseline"
            ),
            f"\naction semantics\n{getattr(episode, 'action_semantics')}",
            (
                "\ncausal boundary\nHidden state is never passed to the controller. "
                "The online adapter is disabled."
            ),
        )
        self._set_provenance_text("\n".join(lines))

    def browse_nominal(self) -> None:
        self._browse_json(self.nominal_model_var)

    def browse_hidden(self) -> None:
        self._browse_json(self.hidden_model_var)

    def _browse_json(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select cable model",
            initialdir=str(Path(variable.get()).expanduser().parent),
            filetypes=(("Cable model", "*.json"), ("All files", "*.*")),
        )
        if selected:
            variable.set(selected)

    def browse_policy(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select frozen SAC policy",
            initialdir=str(Path(self.policy_var.get()).expanduser().parent),
            filetypes=(("SAC policy", "*.pt"), ("All files", "*.*")),
        )
        if selected:
            self.policy_var.set(selected)

    def _arguments(self) -> dict[str, object]:
        nominal = Path(self.nominal_model_var.get()).expanduser().resolve()
        hidden = Path(self.hidden_model_var.get()).expanduser().resolve()
        policy = Path(self.policy_var.get()).expanduser().resolve()
        for name, path in (("Nominal cable", nominal), ("Hidden cable", hidden), ("SAC policy", policy)):
            if not path.is_file():
                raise FileNotFoundError(f"{name} was not found: {path}")
        seed = int(self.seed_var.get())
        if seed < 0:
            raise ValueError("Seed must be non-negative.")
        initial = _vector(self.initial_drone_var.get(), "Drone XYZ")
        target = _vector(self.target_var.get(), "Target XYZ")
        direction = np.asarray(target, dtype=np.float64) - np.asarray(
            initial, dtype=np.float64
        )
        direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            raise ValueError(
                "Target must be horizontally separated from the initial drone."
            )
        direction /= norm
        self.direction_display_var.set(
            ", ".join(f"{value:+.3f}" for value in direction)
        )
        return {
            "nominal_model_path": nominal,
            "hidden_model_path": hidden,
            "policy_path": policy,
            "initial_drone_position_m": initial,
            "target_position_m": target,
            "impact_direction": tuple(float(value) for value in direction),
            "hidden_ei_scale": _positive_float(self.hidden_ei_scale_var.get(), "Hidden EI scale"),
            "hidden_cb_scale": _positive_float(self.hidden_cb_scale_var.get(), "Hidden Cb scale"),
            "seed": seed,
        }

    @staticmethod
    def _summarize_arguments(arguments: dict[str, object]) -> str:
        initial = tuple(float(value) for value in arguments["initial_drone_position_m"])
        target = tuple(float(value) for value in arguments["target_position_m"])
        return (
            f"seed {int(arguments['seed'])}  |  "
            f"start ({initial[0]:+.2f}, {initial[1]:+.2f}, {initial[2]:+.2f}) m  →  "
            f"target ({target[0]:+.2f}, {target[1]:+.2f}, {target[2]:+.2f}) m  |  "
            f"hidden EI ×{float(arguments['hidden_ei_scale']):g}, "
            f"Cb ×{float(arguments['hidden_cb_scale']):g}  |  "
            f"nominal {Path(arguments['nominal_model_path']).name}  |  "
            f"hidden {Path(arguments['hidden_model_path']).name}  |  "
            f"policy {Path(arguments['policy_path']).name}"
        )

    def _set_running_ui(self, running: bool) -> None:
        input_state = tk.DISABLED if running else tk.NORMAL
        for widget in self.experiment_inputs:
            widget.configure(state=input_state)
        self.run_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.reset_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        has_episode = self.episode is not None and bool(self.frames)
        result_state = tk.NORMAL if has_episode and not running else tk.DISABLED
        self.replay_button.configure(state=result_state)
        self.save_button.configure(state=result_state)
        self.pause_button.configure(state=tk.DISABLED)
        timeline_state = tk.NORMAL if has_episode and not running else tk.DISABLED
        self.timeline_scale.configure(state=timeline_state)
        self.step_back_button.configure(state=timeline_state)
        self.step_forward_button.configure(state=timeline_state)

    def _set_plot_samples(self) -> None:
        self.innovation_plot.set_samples(
            [
                (frame.time_s, 1000.0 * frame.marker_innovation_rms_m)
                for frame in self.frames
                if frame.action_updated
            ]
        )
        self.distance_plot.set_samples(
            [
                (frame.time_s, 1000.0 * frame.tip_target_distance_m)
                for frame in self.frames
            ]
        )

    def _configure_timeline(self) -> None:
        last_index = max(len(self.frames) - 1, 0)
        self.timeline_scale.configure(to=float(last_index))
        self._timeline_update = True
        try:
            self.timeline_var.set(float(min(self.frame_index, last_index)))
        finally:
            self._timeline_update = False
        self.timeline_text_var.set(
            f"frame {self.frame_index + 1} / {len(self.frames)}"
            if self.frames
            else "frame 0 / 0"
        )

    def _start_playback_clock(self) -> None:
        self._playback_anchor_wall_s = time.perf_counter()
        self._playback_anchor_episode_s = self.frames[self.frame_index].time_s

    def run(self) -> None:
        if self.computing:
            return
        try:
            arguments = self._arguments()
        except Exception as error:
            messagebox.showerror("Cannot start evaluation", str(error), parent=self.root)
            return
        self.reset(clear_episode=True)
        self.computing = True
        self.stop_event.clear()
        self._active_run_summary = self._summarize_arguments(arguments)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ei_tag = f"{float(arguments['hidden_ei_scale']):.3f}".replace(".", "p")
        cb_tag = f"{float(arguments['hidden_cb_scale']):.3f}".replace(".", "p")
        self._active_run_file_stem = (
            f"hidden_model_eval_{timestamp}_seed{int(arguments['seed'])}"
            f"_ei{ei_tag}_cb{cb_tag}"
        )
        self.active_run_var.set(f"RUNNING  |  {self._active_run_summary}")
        self.status_var.set("Loading the policy and both physical models on the worker…")
        self._set_running_ui(True)
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(arguments,),
            name="adaptation-testbed-worker",
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, arguments: dict[str, object]) -> None:
        try:
            backend = importlib.import_module("drone_mpc.adaptation_testbed")
            settings = backend.TestbedSettings(
                initial_drone_position_m=arguments["initial_drone_position_m"],
                target_position_m=arguments["target_position_m"],
                impact_direction=arguments["impact_direction"],
                hidden_ei_scale=arguments["hidden_ei_scale"],
                hidden_cb_scale=arguments["hidden_cb_scale"],
                seed=arguments["seed"],
            )

            def frame_callback(frame: object) -> None:
                self.events.put(("frame", _DisplayFrame.from_backend(frame)))

            session = backend.load_adaptation_testbed(
                arguments["nominal_model_path"],
                arguments["policy_path"],
                hidden_model_path=arguments["hidden_model_path"],
                settings=settings,
                device="cuda",
            )
            episode = session.run_episode(
                frame_callback=frame_callback,
                cancelled=self.stop_event.is_set,
            )
            self.events.put(("complete", (backend, episode)))
        except BaseException as error:
            self.events.put(("stopped", None) if self.stop_event.is_set() else ("error", error))

    def stop(self) -> None:
        if not self.computing:
            return
        self.stop_event.set()
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("Stopping after the active physics step…")

    def pause(self) -> None:
        if not self.frames:
            return
        if self.playing:
            self.playing = False
            self.paused = True
            self.pause_button.configure(text="Resume")
            self.status_var.set("Playback paused")
            return
        if not self.paused:
            return
        if self.frame_index >= len(self.frames) - 1:
            self.frame_index = 0
            self._display(0)
        self.paused = False
        self.playing = True
        self._start_playback_clock()
        self.pause_button.configure(text="Pause")
        self.status_var.set("Playback resumed")

    def reset(self, *, clear_episode: bool = False) -> None:
        if self.computing:
            return
        self.playing = False
        self.paused = False
        self.frame_index = 0
        self._frame_times_s = np.empty((0,), dtype=np.float64)
        self.pause_button.configure(text="Pause", state=tk.DISABLED)
        self.canvas.clear()
        self.innovation_plot.clear()
        self.distance_plot.clear()
        self.time_var.set("0.00 s")
        self.innovation_var.set("--")
        self.distance_var.set("--")
        self.outcome_var.set("Not run")
        if clear_episode:
            self.episode = None
            self.frames = []
            self.backend_module = None
            self._active_run_summary = ""
            self.active_run_var.set(
                "No active run — configure the experiment, then Run."
            )
            self._set_provenance_text(
                "No completed trial.\n\nFull paths, hashes, rates, settings and "
                "the measurement contract are retained in each saved NPZ archive."
            )
            self.replay_button.configure(state=tk.DISABLED)
            self.save_button.configure(state=tk.DISABLED)
        elif self.frames:
            self._frame_times_s = np.asarray(
                [frame.time_s for frame in self.frames], dtype=np.float64
            )
            self._set_plot_samples()
        self._configure_timeline()
        self.status_var.set("Ready" if clear_episode else "Display reset; press Replay")

    def replay(self) -> None:
        if not self.frames or self.computing:
            return
        if len(self._frame_times_s) != len(self.frames):
            self._frame_times_s = np.asarray(
                [frame.time_s for frame in self.frames], dtype=np.float64
            )
        self.frame_index = 0
        self.playing = True
        self.paused = False
        self.pause_button.configure(text="Pause", state=tk.NORMAL)
        self._display(0)
        self._start_playback_clock()
        self.status_var.set("Replaying the completed causal trial")

    def scrub(self, value: str) -> None:
        if self._timeline_update or self.computing or not self.frames:
            return
        index = int(np.clip(round(float(value)), 0, len(self.frames) - 1))
        self.playing = False
        self.paused = True
        self.pause_button.configure(text="Resume", state=tk.NORMAL)
        self._display(index)
        self.status_var.set("Playback positioned manually")

    def step_frame(self, delta: int) -> None:
        if self.computing or not self.frames:
            return
        self.scrub(str(self.frame_index + int(delta)))

    def save(self) -> None:
        if self.episode is None or self.backend_module is None:
            return
        DEFAULT_TRIAL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        selected = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save hidden-model evaluation trial",
            initialdir=str(DEFAULT_TRIAL_DIRECTORY),
            initialfile=self._active_run_file_stem + ".npz",
            defaultextension=".npz",
            filetypes=(("Testbed trial", "*.npz"),),
        )
        if not selected:
            return
        try:
            path = self.backend_module.save_episode(self.episode, Path(selected))
        except Exception as error:
            messagebox.showerror("Could not save trial", str(error), parent=self.root)
            return
        self.status_var.set(f"Saved {path}")

    def _display(self, frame_index: int) -> None:
        if not self.frames:
            return
        self.frame_index = int(np.clip(frame_index, 0, len(self.frames) - 1))
        frame = self.frames[self.frame_index]
        self.canvas.set_testbed_frame(frame)
        self.time_var.set(f"{frame.time_s:.2f} s")
        self.innovation_var.set(f"{1000.0 * frame.marker_innovation_rms_m:.1f} mm")
        self.distance_var.set(f"{1000.0 * frame.tip_target_distance_m:.1f} mm")
        self.outcome_var.set(
            "Hit"
            if frame.success
            else "Unsafe"
            if frame.unsafe
            else "Complete"
            if frame.done
            else "Running"
        )
        self._timeline_update = True
        try:
            self.timeline_var.set(float(self.frame_index))
        finally:
            self._timeline_update = False
        self.timeline_text_var.set(
            f"frame {self.frame_index + 1} / {len(self.frames)}"
        )

    def _finish_playback(self) -> None:
        self.playing = False
        self.paused = False
        self.pause_button.configure(text="Pause", state=tk.DISABLED)
        final = self.frames[-1]
        self.status_var.set(
            "Trial complete — hit"
            if final.success
            else "Trial terminated — unsafe"
            if final.unsafe
            else "Trial complete — target missed"
        )

    def _tick(self) -> None:
        latest_stream_index: int | None = None
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "frame":
                    # Worker frames are buffered for post-run deterministic
                    # playback. They are never fed back into the backend. The
                    # renderer coalesces a burst to its newest frame below.
                    self.frames.append(payload)  # type: ignore[arg-type]
                    latest_stream_index = len(self.frames) - 1
                elif kind == "complete":
                    backend, episode = payload  # type: ignore[misc]
                    self.backend_module = backend
                    self.episode = episode
                    backend_frames = [
                        _DisplayFrame.from_backend(frame)
                        for frame in getattr(episode, "frames")
                    ]
                    if backend_frames:
                        self.frames = backend_frames
                    if not self.frames:
                        raise RuntimeError("The testbed returned no frames.")
                    self._frame_times_s = np.asarray(
                        [frame.time_s for frame in self.frames], dtype=np.float64
                    )
                    self._configure_timeline()
                    self._set_plot_samples()
                    self.controller_status_var.set(
                        f"{len(getattr(episode, 'marker_material_coordinates_m'))}-point grid  |  "
                        "nominal "
                        f"{str(getattr(episode, 'nominal_source_sha256'))[:10]}  |  "
                        "policy "
                        f"{str(getattr(episode, 'policy_sha256'))[:10]}  |  "
                        f"{float(getattr(episode, 'control_rate_hz')):g} Hz"
                    )
                    self.truth_status_var.set(
                        "hidden source "
                        f"{str(getattr(episode, 'hidden_source_sha256'))[:10]}  |  "
                        f"EI ×{float(getattr(episode.settings, 'hidden_ei_scale')):g}, "
                        f"Cb ×{float(getattr(episode.settings, 'hidden_cb_scale')):g}  |  "
                        "never passed to controller"
                    )
                    self._update_provenance(episode)
                    self.computing = False
                    termination = str(getattr(episode, "termination_reason")).upper()
                    self.active_run_var.set(
                        f"{termination}  |  {self._active_run_summary}"
                    )
                    self._set_running_ui(False)
                    latest_stream_index = None
                    if self.close_requested:
                        self.root.destroy()
                        return
                    if bool(getattr(episode, "cancelled")):
                        self.playing = False
                        self.pause_button.configure(state=tk.DISABLED)
                        self._display(len(self.frames) - 1)
                        self.status_var.set(
                            f"Trial stopped after {len(self.frames)} frames; "
                            "the partial causal record can be replayed or saved"
                        )
                    else:
                        self.replay()
                elif kind == "stopped":
                    self.computing = False
                    self.active_run_var.set(
                        f"STOPPED  |  {self._active_run_summary}"
                    )
                    self._set_running_ui(False)
                    self.status_var.set("Trial stopped; no incomplete result was saved")
                    if self.close_requested:
                        self.root.destroy()
                        return
                elif kind == "error":
                    self.computing = False
                    self.active_run_var.set(
                        f"FAILED  |  {self._active_run_summary}"
                    )
                    self._set_running_ui(False)
                    self.status_var.set("Evaluation failed")
                    messagebox.showerror("Evaluation failed", str(payload), parent=self.root)
                    if self.close_requested:
                        self.root.destroy()
                        return
        except queue.Empty:
            pass
        except Exception as error:
            self.computing = False
            self.active_run_var.set(
                f"DISPLAY FAILED  |  {self._active_run_summary}"
            )
            self._set_running_ui(False)
            self.status_var.set("Testbed display failed")
            messagebox.showerror("Testbed display failed", str(error), parent=self.root)

        if latest_stream_index is not None and self.computing:
            self._display(latest_stream_index)
            self.status_var.set(
                f"Computing causal trial… {len(self.frames)} frames buffered"
            )

        if self.playing and not self.paused and self.frames:
            target_time_s = self._playback_anchor_episode_s + (
                time.perf_counter() - self._playback_anchor_wall_s
            )
            if target_time_s >= self.frames[-1].time_s:
                self._display(len(self.frames) - 1)
                self._finish_playback()
            else:
                target_index = int(
                    np.searchsorted(self._frame_times_s, target_time_s, side="right")
                    - 1
                )
                target_index = int(np.clip(target_index, 0, len(self.frames) - 1))
                if target_index != self.frame_index:
                    self._display(target_index)
        self.root.after(self.POLL_MS, self._tick)

    def close(self) -> None:
        if self.computing:
            if not messagebox.askyesno(
                "Close evaluation testbed",
                "Stop the active CUDA trial and close safely?",
                parent=self.root,
            ):
                return
            self.close_requested = True
            self.stop()
            return
        self.root.destroy()


def main() -> None:
    # ``run_online.py`` is the canonical third-stage launcher.  It originally
    # targeted this adaptation testbed; online adaptation is now deliberately
    # deferred while the matched-model MPC baseline is established.
    from .realtime_gui import main as run_realtime_mpc

    run_realtime_mpc()


if __name__ == "__main__":
    main()
