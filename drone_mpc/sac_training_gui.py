"""Small desktop dashboard for the long-running SAC training process."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import uuid

import numpy as np

from optitrack_offline.config import DEFAULT_MODEL_PATH
from optitrack_offline.viewer import CableCanvas


REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
TRAINING_MODULE = "research_tools.sac_training_worker"
EVENT_PREFIX = "SAC_EVENT_JSON="


# Shared neutral palette for the research-facing desktop tools.  Color is
# reserved for actions and measured outcomes rather than decorative chrome.
WINDOW_BACKGROUND = "#ffffff"
PANEL_BACKGROUND = "#f4f5f6"
CARD_BACKGROUND = "#f7f7f7"
PLOT_BACKGROUND = "#ffffff"
TEXT_COLOR = "#17191b"
MUTED_TEXT_COLOR = "#5f666d"
BORDER_COLOR = "#c9ced3"
GRID_COLOR = "#e4e7e9"
BLUE = "#245f91"
GREEN = "#2f7d4a"
RED = "#ad3b36"
DARK_GRAY = "#626b73"


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer_text(value: int) -> str:
    return f"{int(value):,}"


def _duration_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


class MetricPlot(tk.Canvas):
    """Dependency-free line plot for a few scalar training metrics."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        title: str,
        series: tuple[tuple[str, str], ...],
        percent: bool = False,
        unit: str = "",
        x_label: str = "Transitions",
    ) -> None:
        super().__init__(
            parent,
            background=PLOT_BACKGROUND,
            height=175,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
        )
        self.title = title
        self.series = series
        self.percent = percent
        self.unit = unit
        self.x_label = x_label
        self.samples: list[tuple[float, dict[str, float | None]]] = []
        self.x_limit: float | None = None
        self.bind("<Configure>", lambda _event: self.redraw())

    def clear(self, *, x_limit: float | None) -> None:
        self.samples.clear()
        self.x_limit = x_limit
        self.redraw()

    def add_sample(self, x_value: float, values: dict[str, float | None]) -> None:
        sample = (float(x_value), dict(values))
        if self.samples and math.isclose(self.samples[-1][0], sample[0]):
            self.samples[-1] = sample
        else:
            self.samples.append(sample)
        self.redraw()

    @staticmethod
    def _nice_ceiling(value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            return 1.0
        exponent = 10.0 ** math.floor(math.log10(value))
        normalized = value / exponent
        multiplier = 1.0 if normalized <= 1.0 else 2.0 if normalized <= 2.0 else 5.0 if normalized <= 5.0 else 10.0
        return multiplier * exponent

    def redraw(self) -> None:
        self.delete("all")
        width = max(100, self.winfo_width())
        height = max(80, self.winfo_height())
        left, right, top, bottom = 58.0, width - 16.0, 40.0, height - 44.0
        plot_width = max(1.0, right - left)
        plot_height = max(1.0, bottom - top)
        self.create_text(
            14,
            12,
            anchor=tk.NW,
            text=self.title,
            fill=TEXT_COLOR,
            font=("Segoe UI Semibold", 11),
        )
        legend_x = right
        for name, color in reversed(self.series):
            self.create_text(
                legend_x,
                15,
                anchor=tk.NE,
                text=name,
                fill=color,
                font=("Segoe UI", 9),
            )
            legend_x -= 14 + 7 * len(name)

        finite_values = [
            value
            for _, values in self.samples
            for value in values.values()
            if value is not None and math.isfinite(value)
        ]
        if self.percent:
            y_min, y_max = 0.0, 1.0
        else:
            raw_min = min(finite_values, default=0.0)
            raw_max = max(finite_values, default=1.0)
            y_min = -self._nice_ceiling(abs(raw_min)) if raw_min < 0.0 else 0.0
            y_max = max(self._nice_ceiling(raw_max), 1.0)
        y_range = max(y_max - y_min, 1.0e-12)
        maximum_sample_x = max((sample[0] for sample in self.samples), default=1.0)
        x_max = max(1.0, self.x_limit or maximum_sample_x)

        for tick in range(5):
            fraction = tick / 4.0
            y = bottom - fraction * plot_height
            self.create_line(left, y, right, y, fill=GRID_COLOR)
            value = y_min + fraction * y_range
            label = f"{100.0 * value:.0f}%" if self.percent else f"{value:g}{self.unit}"
            self.create_text(
                left - 7,
                y,
                anchor=tk.E,
                text=label,
                fill=MUTED_TEXT_COLOR,
                font=("Segoe UI", 8),
            )
        for tick in range(5):
            fraction = tick / 4.0
            x = left + fraction * plot_width
            self.create_line(x, top, x, bottom, fill=GRID_COLOR)
            transition = fraction * x_max
            label = (
                f"{transition / 1.0e6:.1f}M"
                if transition >= 1.0e6
                else f"{transition / 1.0e3:.0f}k"
                if transition >= 1.0e3
                else f"{transition:.0f}"
            )
            self.create_text(
                x,
                bottom + 8,
                anchor=tk.N,
                text=label,
                fill=MUTED_TEXT_COLOR,
                font=("Segoe UI", 8),
            )
        self.create_text(
            left + 0.5 * plot_width,
            height - 4,
            anchor=tk.S,
            text=self.x_label,
            fill=MUTED_TEXT_COLOR,
            font=("Segoe UI", 8),
        )

        # At most one point per horizontal pixel is useful to Tk's renderer.
        stride = max(1, math.ceil(len(self.samples) / max(int(plot_width), 1)))
        displayed = self.samples[::stride]
        if self.samples and (not displayed or displayed[-1] is not self.samples[-1]):
            displayed = [*displayed, self.samples[-1]]
        for name, color in self.series:
            segment: list[float] = []
            for x_value, values in displayed:
                value = values.get(name)
                if value is None or not math.isfinite(value):
                    if len(segment) >= 4:
                        self.create_line(*segment, fill=color, width=2, smooth=False)
                    segment = []
                    continue
                x = left + min(max(x_value / x_max, 0.0), 1.0) * plot_width
                y_fraction = (value - y_min) / y_range
                y = bottom - min(max(y_fraction, 0.0), 1.0) * plot_height
                segment.extend((x, y))
            if len(segment) >= 4:
                self.create_line(*segment, fill=color, width=2, smooth=False)
            elif len(segment) == 2:
                x, y = segment
                self.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")


class ValidationPreviewCanvas(CableCanvas):
    """Real-time playback of one deterministic checkpoint-validation episode."""

    def __init__(self, parent: tk.Misc) -> None:
        self.time_s = np.empty(0, dtype=np.float64)
        self.drone_positions_m = np.empty((0, 3), dtype=np.float64)
        self.cable_positions_m = np.empty((0, 0, 3), dtype=np.float64)
        self.target_position_m = np.zeros(3, dtype=np.float64)
        self.desired_impact_direction = np.array((1.0, 0.0, 0.0))
        self.checkpoint_transitions = 0
        self.completed_training_episodes = 0
        self.validation_seed = 0
        self.success = False
        self.unsafe = False
        self.minimum_tip_error_m = math.nan
        self.playback_started_s: float | None = None
        super().__init__(parent)

    @property
    def has_preview(self) -> bool:
        return len(self.time_s) > 0

    def clear_preview(self) -> None:
        self.time_s = np.empty(0, dtype=np.float64)
        self.drone_positions_m = np.empty((0, 3), dtype=np.float64)
        self.cable_positions_m = np.empty((0, 0, 3), dtype=np.float64)
        self.playback_started_s = None
        self.frame_index = 0
        self.redraw()

    def set_preview(self, payload: dict[str, object]) -> None:
        time_s = np.asarray(payload.get("time_s"), dtype=np.float64)
        drone = np.asarray(payload.get("drone_positions_m"), dtype=np.float64)
        cable = np.asarray(payload.get("cable_positions_m"), dtype=np.float64)
        target = np.asarray(payload.get("target_position_m"), dtype=np.float64)
        impact_direction = np.asarray(
            payload.get("desired_impact_direction"), dtype=np.float64
        )
        if (
            time_s.ndim != 1
            or len(time_s) < 2
            or drone.shape != (len(time_s), 3)
            or cable.ndim != 3
            or cable.shape[0] != len(time_s)
            or cable.shape[1] < 3
            or cable.shape[2] != 3
            or target.shape != (3,)
            or impact_direction.shape != (3,)
            or not np.all(np.isfinite(time_s))
            or not np.all(np.diff(time_s) > 0.0)
            or not np.all(np.isfinite(drone))
            or not np.all(np.isfinite(cable))
            or not np.all(np.isfinite(target))
            or not np.all(np.isfinite(impact_direction))
            or np.linalg.norm(impact_direction) < 1.0e-9
        ):
            raise ValueError("Validation preview trajectory is malformed.")
        self.time_s = time_s
        self.drone_positions_m = drone
        self.cable_positions_m = cable
        self.target_position_m = target
        self.desired_impact_direction = (
            impact_direction / np.linalg.norm(impact_direction)
        )
        self.checkpoint_transitions = int(
            _finite_float(payload.get("checkpoint_transitions")) or 0
        )
        self.completed_training_episodes = int(
            _finite_float(payload.get("completed_training_episodes")) or 0
        )
        self.validation_seed = int(
            _finite_float(payload.get("validation_seed")) or 0
        )
        self.success = bool(payload.get("success"))
        self.unsafe = bool(payload.get("unsafe"))
        error = _finite_float(payload.get("minimum_tip_error_m"))
        self.minimum_tip_error_m = math.nan if error is None else error

        points = np.concatenate(
            (cable.reshape(-1, 3), drone, target[None]), axis=0
        )
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        self.center_m = 0.5 * (lower + upper)
        self.radius_m = max(float(np.linalg.norm(upper - lower) * 0.58), 0.65)
        horizontal_span = max(
            float(upper[0] - lower[0]),
            float(upper[1] - lower[1]),
            1.0,
        )
        self.grid_step_m = self._nice_grid_step(horizontal_span / 10.0)
        horizontal_center = 0.5 * (lower[:2] + upper[:2])
        half = max(0.65 * horizontal_span, 0.6)
        self.floor_bounds_m = (
            math.floor((horizontal_center[0] - half) / self.grid_step_m)
            * self.grid_step_m,
            math.ceil((horizontal_center[0] + half) / self.grid_step_m)
            * self.grid_step_m,
            math.floor((horizontal_center[1] - half) / self.grid_step_m)
            * self.grid_step_m,
            math.ceil((horizontal_center[1] + half) / self.grid_step_m)
            * self.grid_step_m,
        )
        self.frame_index = 0
        self.playback_started_s = time.perf_counter()
        self.reset_view()

    def advance(self, now_s: float) -> None:
        if not self.has_preview or self.playback_started_s is None:
            return
        elapsed = min(max(now_s - self.playback_started_s, 0.0), self.time_s[-1])
        frame = int(np.searchsorted(self.time_s, elapsed, side="right") - 1)
        frame = int(np.clip(frame, 0, len(self.time_s) - 1))
        if frame != self.frame_index:
            self.frame_index = frame
            self.redraw()

    def _draw_path(self, points_m: np.ndarray, color: str, width: int) -> None:
        if len(points_m) < 2:
            return
        screen, _depth = self._project(points_m)
        self.create_line(
            *screen.reshape(-1), fill=color, width=width, dash=(5, 3)
        )

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        self.create_text(
            14,
            12,
            anchor=tk.NW,
            text="CURRENT POLICY  |  FIXED DETERMINISTIC VALIDATION",
            fill=TEXT_COLOR,
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            width - 14,
            12,
            anchor=tk.NE,
            text="Drag: rotate   Right-drag: pan   Wheel: zoom",
            fill=MUTED_TEXT_COLOR,
            font=("Segoe UI", 8),
        )
        if not self.has_preview:
            self.create_text(
                0.5 * width,
                0.5 * height,
                text="First playback appears after checkpoint validation",
                fill=MUTED_TEXT_COLOR,
                font=("Segoe UI Semibold", 11),
            )
            return

        self._draw_floor()
        self._draw_axes()
        frame = self.frame_index
        self._draw_path(self.drone_positions_m[: frame + 1], "#c87500", 2)
        self._draw_path(self.cable_positions_m[: frame + 1, -1], "#7a3db8", 3)

        target_screen, _ = self._project(self.target_position_m[None])
        tx, ty = target_screen[0]
        self.create_oval(
            tx - 9, ty - 9, tx + 9, ty + 9, outline=RED, width=3
        )
        self.create_line(tx - 12, ty, tx + 12, ty, fill=RED, width=1)
        self.create_line(tx, ty - 12, tx, ty + 12, fill=RED, width=1)
        impact_axis, _ = self._project(
            np.stack(
                (
                    self.target_position_m - 0.18 * self.desired_impact_direction,
                    self.target_position_m + 0.08 * self.desired_impact_direction,
                )
            )
        )
        self.create_line(
            *impact_axis.reshape(-1), fill=RED, width=3, arrow=tk.LAST
        )

        cable = self.cable_positions_m[frame]
        cable_screen, depth = self._project(cable)
        self.create_line(*cable_screen.reshape(-1), fill=BLUE, width=4)
        for index in np.argsort(depth):
            x, y = cable_screen[index]
            endpoint = index in (0, len(cable) - 1)
            radius = 6 if endpoint else 3
            color = "#c87500" if index == 0 else (
                "#7a3db8" if index == len(cable) - 1 else BLUE
            )
            self.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline="#ffffff",
                width=1,
            )

        pair, _ = self._project(
            np.stack((self.drone_positions_m[frame], cable[0]))
        )
        self.create_line(*pair.reshape(-1), fill="#c87500", width=3)
        x, y = pair[0]
        self.create_polygon(
            x,
            y - 10,
            x + 10,
            y,
            x,
            y + 10,
            x - 10,
            y,
            fill="#e59a31",
            outline="#7a4b08",
            width=1,
        )

        final_frame = frame + 1 == len(self.time_s)
        if final_frame:
            outcome = "HIT" if self.success else ("UNSAFE" if self.unsafe else "MISS")
            outcome_color = GREEN if self.success else RED if self.unsafe else DARK_GRAY
        else:
            outcome = "PLAYING"
            outcome_color = BLUE
        error_text = (
            "n/a"
            if not math.isfinite(self.minimum_tip_error_m)
            else f"{1000.0 * self.minimum_tip_error_m:.1f} mm"
        )
        self.create_text(
            14,
            35,
            anchor=tk.NW,
            text=(
                f"{outcome}  |  t={self.time_s[frame]:.2f}s  |  "
                f"STRIKE ATTEMPT  |  minimum error={error_text}"
            ),
            fill=outcome_color,
            font=("Segoe UI Semibold", 9),
        )
        self.create_text(
            14,
            height - 10,
            anchor=tk.SW,
            text=(
                f"checkpoint {_integer_text(self.checkpoint_transitions)} transitions  |  "
                f"{_integer_text(self.completed_training_episodes)} training episodes  |  "
                f"validation seed {self.validation_seed}"
            ),
            fill=MUTED_TEXT_COLOR,
            font=("Segoe UI", 8),
        )


class SacTrainingGui:
    POLL_MS = 40

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stop_file: Path | None = None
        self.running = False
        self.stopping = False
        self.close_requested = False
        self.started_wall_s = 0.0
        self.last_event: dict[str, object] = {}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        default_output = (
            REPOSITORY_DIRECTORY
            / "data"
            / "drone_mpc"
            / "training_runs"
            / f"sac_{timestamp}.pt"
        )
        self.model_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH.resolve()))
        self.output_var = tk.StringVar(value=str(default_output.resolve()))
        self.seed_var = tk.StringVar(value="42")
        self.endless_var = tk.BooleanVar(value=False)
        self.transitions_var = tk.StringVar(value="500000")
        self.environments_var = tk.StringVar(value="128")
        self.batch_var = tk.StringVar(value="512")
        self.warmup_var = tk.StringVar(value="100000")
        self.checkpoint_episodes_var = tk.StringVar(value="128")
        self.her_goals_var = tk.StringVar(value="2")
        self.her_min_speed_var = tk.StringVar(value="0.25")
        self.progress_reward_var = tk.StringVar(value="5.0")
        self.success_reward_var = tk.StringVar(value="100.0")
        self.target_radius_var = tk.StringVar(value="0.05")
        self.target_distance_var = tk.StringVar(value="0.80")
        self.target_height_var = tk.StringVar(value="-0.10")
        self.target_azimuth_var = tk.StringVar(value="0.0")
        self.impact_speed_var = tk.StringVar(value="1.5")
        self.impact_azimuth_var = tk.StringVar(value="0.0")
        self.impact_elevation_var = tk.StringVar(value="0.0")
        self.impact_angle_var = tk.StringVar(value="35.0")

        self.state_var = tk.StringVar(value="Ready")
        self.progress_card_var = tk.StringVar(value="0 / 500,000")
        self.performance_var = tk.StringVar(value="0 trans/s")
        self.validation_var = tk.StringVar(value="Not evaluated")
        self.best_var = tk.StringVar(value="No checkpoint")
        self.summary_var = tk.StringVar()
        self.progress_var = tk.DoubleVar(value=0.0)

        self.input_widgets: list[tk.Widget] = []
        self.root.title("Cable Twin - SAC Training")
        self.root.geometry("1460x900")
        self.root.minsize(1120, 740)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()
        self._build()
        self._sync_mode()
        self._update_summary()
        self.root.after(self.POLL_MS, self._tick)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(background=WINDOW_BACKGROUND)
        style.configure(
            ".",
            background=WINDOW_BACKGROUND,
            foreground=TEXT_COLOR,
            font=("Segoe UI", 10),
        )
        style.configure("TFrame", background=WINDOW_BACKGROUND)
        style.configure("Panel.TFrame", background=PANEL_BACKGROUND)
        style.configure(
            "Card.TFrame",
            background=CARD_BACKGROUND,
            relief="solid",
            borderwidth=1,
        )
        style.configure("TLabel", background=WINDOW_BACKGROUND, foreground=TEXT_COLOR)
        style.configure(
            "Panel.TLabel",
            background=PANEL_BACKGROUND,
            foreground=TEXT_COLOR,
        )
        style.configure(
            "Card.TLabel",
            background=CARD_BACKGROUND,
            foreground=MUTED_TEXT_COLOR,
        )
        style.configure(
            "CardValue.TLabel",
            background=CARD_BACKGROUND,
            foreground=TEXT_COLOR,
            font=("Segoe UI Semibold", 12),
        )
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 20), foreground=TEXT_COLOR)
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 11), foreground=TEXT_COLOR)
        style.configure("Muted.TLabel", foreground=MUTED_TEXT_COLOR)
        style.configure(
            "TLabelframe",
            background=PANEL_BACKGROUND,
            bordercolor=BORDER_COLOR,
            lightcolor=BORDER_COLOR,
            darkcolor=BORDER_COLOR,
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "TLabelframe.Label",
            background=WINDOW_BACKGROUND,
            foreground=TEXT_COLOR,
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "TButton",
            background="#f2f3f4",
            foreground=TEXT_COLOR,
            bordercolor="#aeb4b9",
            padding=(10, 6),
        )
        style.map(
            "TButton",
            background=[("active", "#e5e8ea"), ("disabled", "#f4f4f4")],
            foreground=[("disabled", "#9aa0a5")],
        )
        style.configure(
            "Primary.TButton",
            background=BLUE,
            foreground="#ffffff",
            bordercolor=BLUE,
            padding=(10, 7),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#1c4d77"), ("disabled", "#aab8c4")],
            foreground=[("disabled", "#f1f3f5")],
        )
        style.configure(
            "Danger.TButton",
            background="#ffffff",
            foreground=RED,
            bordercolor=RED,
            padding=(10, 7),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#f8eae9"), ("disabled", "#f4f4f4")],
            foreground=[("disabled", "#a5a5a5")],
        )
        style.configure(
            "TCheckbutton",
            background=PANEL_BACKGROUND,
            foreground=TEXT_COLOR,
        )
        style.map("TCheckbutton", background=[("active", PANEL_BACKGROUND)])
        style.configure(
            "TEntry",
            fieldbackground="#ffffff",
            foreground=TEXT_COLOR,
            insertcolor=TEXT_COLOR,
            bordercolor="#aeb4b9",
            lightcolor="#aeb4b9",
            darkcolor="#aeb4b9",
        )
        style.configure("TSeparator", background=BORDER_COLOR)
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#e4e7e9",
            background=GREEN,
            bordercolor="#e4e7e9",
            lightcolor=GREEN,
            darkcolor=GREEN,
        )

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=(14, 10))
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="SAC policy training", style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(
                "Nominal goal-conditioned cable-whip policy  |  "
                "CUDA training with fixed deterministic checkpoint validation"
            ),
            style="Muted.TLabel",
        ).pack(anchor=tk.W, pady=(1, 9))
        ttk.Separator(outer).pack(fill=tk.X, pady=(0, 11))

        paths = ttk.LabelFrame(outer, text="Experiment inputs and outputs", padding=10)
        paths.pack(fill=tk.X)
        paths.columnconfigure(1, weight=1)
        ttk.Label(paths, text="Cable model", style="Panel.TLabel").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_entry = ttk.Entry(paths, textvariable=self.model_var)
        self.model_entry.grid(row=0, column=1, sticky=tk.EW, padx=(9, 7), pady=2)
        self.model_button = ttk.Button(paths, text="Browse", command=self.browse_model)
        self.model_button.grid(row=0, column=2, pady=2)
        ttk.Label(paths, text="New policy", style="Panel.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.output_entry = ttk.Entry(paths, textvariable=self.output_var)
        self.output_entry.grid(row=1, column=1, sticky=tk.EW, padx=(9, 7), pady=2)
        self.output_button = ttk.Button(paths, text="Browse", command=self.browse_output)
        self.output_button.grid(row=1, column=2, pady=2)

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        controls = ttk.Frame(body, padding=12, style="Panel.TFrame")
        plots = ttk.Frame(body, padding=(10, 0, 0, 0))
        body.add(controls, weight=1)
        body.add(plots, weight=3)

        ttk.Label(controls, text="Training configuration", style="Section.TLabel").pack(anchor=tk.W)
        ttk.Label(
            controls,
            text=(
                "Goal-conditioned SAC with future achieved-goal relabeling. The policy "
                "discovers abrupt wind-up and release from physical target contact."
            ),
            style="Panel.TLabel",
            wraplength=680,
        ).pack(anchor=tk.W, pady=(1, 8))

        parameters = ttk.Frame(controls, style="Panel.TFrame")
        parameters.pack(fill=tk.X)
        parameters.columnconfigure(0, weight=1, uniform="parameter")
        parameters.columnconfigure(1, weight=1, uniform="parameter")
        self._entry(
            parameters,
            "Random seed",
            self.seed_var,
            "",
            row_index=0,
            column=0,
        )
        self.transitions_entry = self._entry(
            parameters,
            "Transition limit",
            self.transitions_var,
            "",
            row_index=0,
            column=1,
        )
        self._entry(
            parameters,
            "Parallel environments",
            self.environments_var,
            "",
            row_index=1,
            column=0,
        )
        self._entry(
            parameters,
            "SAC batch",
            self.batch_var,
            "",
            row_index=1,
            column=1,
        )
        self._entry(
            parameters,
            "Random warmup",
            self.warmup_var,
            "transitions",
            row_index=2,
            column=0,
        )
        self._entry(
            parameters,
            "Checkpoint validation",
            self.checkpoint_episodes_var,
            "episodes",
            row_index=2,
            column=1,
        )
        self._entry(
            parameters,
            "HER goals / episode",
            self.her_goals_var,
            "future goals",
            row_index=3,
            column=0,
        )
        self._entry(
            parameters,
            "HER minimum speed",
            self.her_min_speed_var,
            "m/s",
            row_index=3,
            column=1,
        )
        self._entry(
            parameters,
            "Progress potential",
            self.progress_reward_var,
            "weight",
            row_index=4,
            column=0,
        )
        self._entry(
            parameters,
            "Valid hit",
            self.success_reward_var,
            "bonus",
            row_index=4,
            column=1,
        )
        self._entry(
            parameters,
            "Physical target radius",
            self.target_radius_var,
            "m",
            row_index=5,
            column=0,
        )
        self._entry(
            parameters,
            "Target distance",
            self.target_distance_var,
            "m",
            row_index=5,
            column=1,
        )
        self._entry(
            parameters,
            "Target height",
            self.target_height_var,
            "m rel.",
            row_index=6,
            column=0,
        )
        self._entry(
            parameters,
            "Target azimuth",
            self.target_azimuth_var,
            "deg",
            row_index=6,
            column=1,
        )
        self._entry(
            parameters,
            "Impact speed",
            self.impact_speed_var,
            "m/s min",
            row_index=7,
            column=0,
        )
        self._entry(
            parameters,
            "Impact yaw offset",
            self.impact_azimuth_var,
            "deg",
            row_index=7,
            column=1,
        )
        self._entry(
            parameters,
            "Impact elevation",
            self.impact_elevation_var,
            "deg",
            row_index=8,
            column=0,
        )
        self._entry(
            parameters,
            "Impact cone",
            self.impact_angle_var,
            "deg",
            row_index=8,
            column=1,
        )
        self.endless_check = ttk.Checkbutton(
            controls,
            text="Train until I press Stop",
            variable=self.endless_var,
            command=self._sync_mode,
        )
        self.endless_check.pack(anchor=tk.W, pady=(7, 1))
        self.input_widgets.append(self.endless_check)

        actions = ttk.LabelFrame(controls, text="Execution", padding=9)
        actions.pack(fill=tk.X, pady=(12, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)
        actions.columnconfigure(2, weight=1)
        self.start_button = ttk.Button(
            actions,
            text="Start training",
            command=self.start_training,
            style="Primary.TButton",
        )
        self.start_button.grid(row=0, column=0, sticky=tk.EW)
        self.stop_button = ttk.Button(
            actions,
            text="Stop safely",
            command=self.stop_training,
            state=tk.DISABLED,
            style="Danger.TButton",
        )
        self.stop_button.grid(row=0, column=1, sticky=tk.EW, padx=6)
        self.folder_button = ttk.Button(actions, text="Open output folder", command=self.open_output_folder)
        self.folder_button.grid(row=0, column=2, sticky=tk.EW)
        self.progress = ttk.Progressbar(actions, variable=self.progress_var, maximum=100.0)
        self.progress.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=(9, 0))

        experiment = ttk.LabelFrame(controls, text="Active experiment", padding=9)
        experiment.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Label(
            experiment,
            textvariable=self.summary_var,
            style="Panel.TLabel",
            justify=tk.LEFT,
            wraplength=680,
        ).pack(anchor=tk.NW)

        plots.rowconfigure(0, weight=2)
        plots.rowconfigure(1, weight=1)
        plots.rowconfigure(2, weight=1)
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        self.validation_preview = ValidationPreviewCanvas(plots)
        self.validation_preview.grid(
            row=0, column=0, sticky=tk.NSEW, padx=(0, 5), pady=(0, 5)
        )
        self.success_plot = MetricPlot(
            plots,
            title="Success rate by episode",
            series=(("Training", BLUE), ("Validation", GREEN)),
            percent=True,
            x_label="Completed episodes",
        )
        self.success_plot.grid(
            row=0, column=1, sticky=tk.NSEW, padx=(5, 0), pady=(0, 5)
        )
        self.reward_plot = MetricPlot(
            plots,
            title="Mean reward per completed episode",
            series=(("Training", BLUE), ("Validation", GREEN)),
            x_label="Completed episodes",
        )
        self.reward_plot.grid(
            row=1, column=0, sticky=tk.NSEW, padx=(0, 5), pady=5
        )
        self.error_plot = MetricPlot(
            plots,
            title="Mean minimum free-tip error",
            series=(("Error", RED),),
            unit="mm",
        )
        self.error_plot.grid(
            row=1, column=1, sticky=tk.NSEW, padx=(5, 0), pady=5
        )
        self.speed_plot = MetricPlot(
            plots,
            title="Mean directed tip speed near target",
            series=(("Speed", GREEN),),
            unit=" m/s",
        )
        self.speed_plot.grid(
            row=2, column=0, sticky=tk.NSEW, padx=(0, 5), pady=(5, 0)
        )
        self.unsafe_plot = MetricPlot(
            plots,
            title="Unsafe episode rate",
            series=(("Unsafe", RED),),
            percent=True,
        )
        self.unsafe_plot.grid(
            row=2, column=1, sticky=tk.NSEW, padx=(5, 0), pady=(5, 0)
        )

        cards = ttk.Frame(outer)
        cards.pack(fill=tk.X, pady=(8, 0))
        for column in range(5):
            cards.columnconfigure(column, weight=1, uniform="card")
        for column, (name, variable) in enumerate(
            (
                ("STATE", self.state_var),
                ("TRANSITIONS", self.progress_card_var),
                ("THROUGHPUT", self.performance_var),
                ("VALIDATION", self.validation_var),
                ("BEST POLICY", self.best_var),
            )
        ):
            card = ttk.Frame(cards, padding=8, style="Card.TFrame")
            card.grid(row=0, column=column, sticky=tk.EW, padx=(0 if column == 0 else 4, 0))
            ttk.Label(card, text=name, style="Card.TLabel").pack(anchor=tk.W)
            ttk.Label(card, textvariable=variable, style="CardValue.TLabel", wraplength=260).pack(anchor=tk.W)

        ttk.Label(outer, text="Training log", style="Section.TLabel").pack(anchor=tk.W, pady=(8, 3))
        self.log = scrolledtext.ScrolledText(
            outer,
            height=4,
            background="#ffffff",
            foreground=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            selectbackground="#d7e6f2",
            selectforeground=TEXT_COLOR,
            borderwidth=1,
            relief=tk.SOLID,
            highlightthickness=0,
            font=("Cascadia Mono", 9),
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.log.pack(fill=tk.X)

        self.input_widgets.extend(
            (self.model_entry, self.model_button, self.output_entry, self.output_button)
        )
        for variable in (
            self.model_var,
            self.output_var,
            self.seed_var,
            self.transitions_var,
            self.environments_var,
            self.batch_var,
            self.warmup_var,
            self.checkpoint_episodes_var,
            self.her_goals_var,
            self.her_min_speed_var,
            self.progress_reward_var,
            self.success_reward_var,
            self.target_radius_var,
            self.target_distance_var,
            self.target_height_var,
            self.target_azimuth_var,
            self.impact_speed_var,
            self.impact_azimuth_var,
            self.impact_elevation_var,
            self.impact_angle_var,
        ):
            variable.trace_add("write", lambda *_args: self._update_summary())

    def _entry(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        unit: str,
        *,
        row_index: int | None = None,
        column: int = 0,
    ) -> ttk.Entry:
        row_frame = ttk.Frame(parent, style="Panel.TFrame")
        if row_index is None:
            row_frame.pack(fill=tk.X, pady=3)
        else:
            row_frame.grid(
                row=row_index,
                column=column,
                sticky=tk.EW,
                padx=(0 if column == 0 else 6, 6 if column == 0 else 0),
                pady=3,
            )
        ttk.Label(row_frame, text=label, width=19, style="Panel.TLabel").pack(side=tk.LEFT)
        entry = ttk.Entry(row_frame, textvariable=variable, width=13)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row_frame, text=unit, width=10, style="Panel.TLabel").pack(side=tk.LEFT, padx=(5, 0))
        self.input_widgets.append(entry)
        return entry

    def _update_summary(self) -> None:
        if not hasattr(self, "summary_var"):
            return
        mode = "continuous until safely stopped" if self.endless_var.get() else f"finite: {self.transitions_var.get().strip() or '?'} transitions"
        progress_reward = self.progress_reward_var.get().strip() or "?"
        hit_reward = self.success_reward_var.get().strip() or "?"
        her_goals = self.her_goals_var.get().strip() or "?"
        impact_speed = self.impact_speed_var.get().strip() or "?"
        impact_yaw = self.impact_azimuth_var.get().strip() or "?"
        impact_elevation = self.impact_elevation_var.get().strip() or "?"
        impact_angle = self.impact_angle_var.get().strip() or "?"
        target_distance = self.target_distance_var.get().strip() or "?"
        target_height = self.target_height_var.get().strip() or "?"
        target_azimuth = self.target_azimuth_var.get().strip() or "?"
        target_radius = self.target_radius_var.get().strip() or "?"
        self.summary_var.set(
            f"{self._model_note()}\n"
            "Goal-conditioned SAC + future HER  |  twin Q critics  |  no demonstrations\n"
            "15-node constrained DER  |  full cable state  |  abrupt 3-D acceleration\n"
            "4.0 s episodes  |  100 Hz cable physics, 50 Hz policy  |  20 m/s^2, 3 m/s limits\n"
            f"Reward: physical hit +{hit_reward}; bounded progress potential {progress_reward}; "
            "no energy or action-change reward\n"
            f"Fixed baseline target: r={target_distance} m, z={target_height} m, "
            f"azimuth={target_azimuth} deg, physical radius={target_radius} m\n"
            f"Impact vector: {impact_speed} m/s min, yaw offset {impact_yaw} deg, "
            f"elevation {impact_elevation} deg, cone {impact_angle} deg\n"
            f"HER: {her_goals} future achieved goals per completed episode\n"
            f"Run: {mode}  |  validation seed: +1000  |  final-test seed: +10000"
        )
        if not self.running:
            limit = self.transitions_var.get().strip()
            self.progress_card_var.set("0 / endless" if self.endless_var.get() else f"0 / {limit or '?'}")

    def _model_note(self) -> str:
        source = Path(self.model_var.get()).expanduser()
        if not source.is_file():
            return f"Model: {source.name or 'not selected'} (missing)"
        try:
            schema = json.loads(source.read_text(encoding="utf-8")).get("schema")
        except (OSError, ValueError, json.JSONDecodeError):
            return f"Model: {source.name} (unreadable)"
        if schema == "optitrack_twist_aware_rod_v5":
            return f"Model: {source.name} (PROVISIONAL two-holder transfer)"
        if schema == "optitrack_one_attached_free_rod_v1":
            return f"Model: {source.name} (one attachment / free tip)"
        return f"Model: {source.name} (incompatible schema)"

    def _sync_mode(self) -> None:
        if not self.running:
            self.transitions_entry.configure(state=tk.DISABLED if self.endless_var.get() else tk.NORMAL)
        self._update_summary()

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select cable model",
            filetypes=(("Cable model", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.model_var.set(path)

    def browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="New best-policy file",
            initialdir=str(Path(self.output_var.get()).expanduser().parent),
            initialfile=Path(self.output_var.get()).name,
            defaultextension=".pt",
            filetypes=(("PyTorch policy", "*.pt"),),
        )
        if path:
            self.output_var.set(path)

    def open_output_folder(self) -> None:
        folder = Path(self.output_var.get()).expanduser().resolve().parent
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(folder)  # type: ignore[attr-defined]
        except OSError as error:
            messagebox.showerror("Output folder", str(error), parent=self.root)

    def _read_positive_int(self, variable: tk.StringVar, name: str) -> int:
        try:
            value = int(variable.get())
        except ValueError as error:
            raise ValueError(f"{name} must be an integer.") from error
        if value < 1:
            raise ValueError(f"{name} must be positive.")
        return value

    @staticmethod
    def _read_nonnegative_float(variable: tk.StringVar, name: str) -> float:
        try:
            value = float(variable.get())
        except ValueError as error:
            raise ValueError(f"{name} must be a number.") from error
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
        return value

    @staticmethod
    def _read_finite_float(variable: tk.StringVar, name: str) -> float:
        try:
            value = float(variable.get())
        except ValueError as error:
            raise ValueError(f"{name} must be a number.") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        return value

    @staticmethod
    def _output_artifacts(output: Path) -> tuple[Path, ...]:
        return (
            output,
            output.with_name(f"{output.stem}.latest{output.suffix}"),
            output.with_name(f"{output.stem}.final{output.suffix}"),
            output.with_suffix(".json"),
        )

    def _build_command(self) -> tuple[list[str], Path, int | None]:
        model = Path(self.model_var.get()).expanduser().resolve()
        output = Path(self.output_var.get()).expanduser().resolve()
        if not model.is_file():
            raise ValueError(f"Cable model does not exist: {model}")
        if output.suffix.lower() != ".pt":
            raise ValueError("The policy output must use the .pt extension.")
        try:
            seed = int(self.seed_var.get())
        except ValueError as error:
            raise ValueError("Random seed must be an integer.") from error
        # Zero is a valid deterministic seed.
        if seed < 0:
            raise ValueError("Random seed must be non-negative.")
        transitions = self._read_positive_int(self.transitions_var, "Transition limit")
        environments = self._read_positive_int(self.environments_var, "Parallel environments")
        batch = self._read_positive_int(self.batch_var, "SAC batch")
        try:
            warmup = int(self.warmup_var.get())
        except ValueError as error:
            raise ValueError("Random warmup must be an integer.") from error
        checkpoint_episodes = self._read_positive_int(
            self.checkpoint_episodes_var, "Checkpoint validation episodes"
        )
        her_goals = self._read_positive_int(
            self.her_goals_var, "HER goals per episode"
        )
        her_min_speed = self._read_nonnegative_float(
            self.her_min_speed_var, "HER minimum achieved speed"
        )
        progress_reward = self._read_nonnegative_float(
            self.progress_reward_var, "Progress potential"
        )
        success_reward = self._read_nonnegative_float(
            self.success_reward_var, "Valid-hit reward"
        )
        if success_reward <= 0.0:
            raise ValueError("Valid-hit reward must be positive.")
        target_radius = self._read_nonnegative_float(
            self.target_radius_var, "Physical target radius"
        )
        target_distance = self._read_nonnegative_float(
            self.target_distance_var, "Target distance"
        )
        target_height = self._read_finite_float(
            self.target_height_var, "Target height"
        )
        target_azimuth = self._read_finite_float(
            self.target_azimuth_var, "Target azimuth"
        )
        impact_speed = self._read_nonnegative_float(
            self.impact_speed_var, "Impact speed"
        )
        if impact_speed <= 0.0:
            raise ValueError("Impact speed must be positive.")
        impact_elevation = self._read_finite_float(
            self.impact_elevation_var, "Impact elevation"
        )
        impact_azimuth = self._read_finite_float(
            self.impact_azimuth_var, "Impact yaw offset"
        )
        if not -89.0 <= impact_elevation <= 89.0:
            raise ValueError("Impact elevation must be between -89 and 89 degrees.")
        impact_angle = self._read_nonnegative_float(
            self.impact_angle_var, "Impact cone"
        )
        if not 0.0 < impact_angle < 90.0:
            raise ValueError("Impact cone must be between 0 and 90 degrees.")
        if her_min_speed <= 0.0:
            raise ValueError("HER minimum achieved speed must be positive.")
        if target_radius <= 0.0:
            raise ValueError("Physical target radius must be positive.")
        if target_distance <= 0.0:
            raise ValueError("Target distance must be positive.")
        if not -180.0 <= target_azimuth <= 180.0:
            raise ValueError("Target azimuth must be in [-180, 180].")
        if not -180.0 <= impact_azimuth <= 180.0:
            raise ValueError("Impact yaw offset must be in [-180, 180].")
        if warmup < 0:
            raise ValueError("Random warmup cannot be negative.")
        if warmup > transitions:
            raise ValueError("Random warmup cannot exceed the transition limit.")
        if environments > 250_000 or batch > 250_000:
            raise ValueError("Parallel environments and SAC batch must fit the 250,000-transition replay.")

        collisions = [path for path in self._output_artifacts(output) if path.exists()]
        if collisions:
            raise FileExistsError(
                "Choose a new policy name. These run artifacts already exist:\n"
                + "\n".join(str(path) for path in collisions)
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        stop_file = output.parent / f".{output.stem}.{uuid.uuid4().hex}.stop"
        command = [
            sys.executable,
            "-m",
            TRAINING_MODULE,
            "--model",
            str(model),
            "--output",
            str(output),
            "--transitions",
            str(transitions),
            "--environments",
            str(environments),
            "--batch-size",
            str(batch),
            "--warmup",
            str(warmup),
            "--checkpoint-evaluation-episodes",
            str(checkpoint_episodes),
            "--her-goals-per-episode",
            str(her_goals),
            "--her-minimum-achieved-speed",
            str(her_min_speed),
            "--seed",
            str(seed),
            "--stop-file",
            str(stop_file),
            "--progress-json",
            "--validation-preview",
            "--progress-reward",
            str(progress_reward),
            "--success-reward",
            str(success_reward),
            "--hit-tolerance",
            str(target_radius),
            "--target-distance-min",
            str(target_distance),
            "--target-distance-max",
            str(target_distance),
            "--target-height-min",
            str(target_height),
            "--target-height-max",
            str(target_height),
            "--target-azimuth-min",
            str(target_azimuth),
            "--target-azimuth-max",
            str(target_azimuth),
            "--impact-speed",
            str(impact_speed),
            "--impact-azimuth-offset-min",
            str(impact_azimuth),
            "--impact-azimuth-offset-max",
            str(impact_azimuth),
            "--impact-elevation",
            str(impact_elevation),
            "--impact-angle",
            str(impact_angle),
        ]
        if self.endless_var.get():
            command.append("--endless")
        return command, stop_file, None if self.endless_var.get() else transitions

    def _set_running(self, running: bool) -> None:
        self.running = running
        for widget in self.input_widgets:
            widget.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_button.configure(state=tk.NORMAL if running and not self.stopping else tk.DISABLED)
        if not running:
            self._sync_mode()

    def _clear_run_display(self, limit: int | None) -> None:
        self.last_event = {}
        self.progress_var.set(0.0)
        self.state_var.set("Starting")
        self.progress_card_var.set("0 / endless" if limit is None else f"0 / {_integer_text(limit)}")
        self.performance_var.set("Starting CUDA")
        self.validation_var.set("Not evaluated")
        self.best_var.set("No checkpoint")
        self.validation_preview.clear_preview()
        # Episode count is not known in advance, even for a finite transition run.
        self.success_plot.clear(x_limit=None)
        self.reward_plot.clear(x_limit=None)
        self.error_plot.clear(x_limit=None if limit is None else float(limit))
        self.speed_plot.clear(x_limit=None if limit is None else float(limit))
        self.unsafe_plot.clear(x_limit=None if limit is None else float(limit))
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def start_training(self) -> None:
        if self.running:
            return
        try:
            command, stop_file, transition_limit = self._build_command()
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot start SAC training", str(error), parent=self.root)
            return
        self.stop_file = stop_file
        self.stopping = False
        self.close_requested = False
        self.started_wall_s = time.perf_counter()
        self._clear_run_display(transition_limit)
        self._set_running(True)
        if transition_limit is None:
            self.progress.configure(mode="indeterminate")
            self.progress.start(18)
        else:
            self.progress.configure(mode="determinate")
        self._append_log(
            "Starting continuous SAC training; press Stop safely to finish."
            if transition_limit is None
            else f"Starting finite SAC training for {_integer_text(transition_limit)} transitions."
        )
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            process = subprocess.Popen(
                command,
                cwd=str(REPOSITORY_DIRECTORY),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creation_flags,
            )
        except Exception as error:
            self.progress.stop()
            self._set_running(False)
            self.state_var.set("Launch failed")
            messagebox.showerror("SAC training", str(error), parent=self.root)
            return
        self.process = process

        def read_output(active_process: subprocess.Popen[str]) -> None:
            stream = active_process.stdout
            if stream is not None:
                for raw_line in stream:
                    line = raw_line.rstrip("\r\n")
                    if line.startswith(EVENT_PREFIX):
                        try:
                            payload = json.loads(line[len(EVENT_PREFIX) :])
                        except (TypeError, ValueError, json.JSONDecodeError):
                            self.events.put(("log", line))
                        else:
                            self.events.put(("event", payload))
                    elif line:
                        self.events.put(("log", line))
            self.events.put(("exit", (active_process, active_process.wait())))

        threading.Thread(
            target=read_output,
            args=(process,),
            name="sac-training-output",
            daemon=True,
        ).start()

    def stop_training(self) -> None:
        if not self.running or self.stopping:
            return
        if self.stop_file is None:
            return
        try:
            self.stop_file.write_text("stop\n", encoding="utf-8")
        except OSError as error:
            messagebox.showerror("Stop SAC training", str(error), parent=self.root)
            return
        self.stopping = True
        self.stop_button.configure(state=tk.DISABLED)
        self.state_var.set("Stopping safely")
        self._append_log("Stop requested; waiting for the current CUDA/evaluation operation to finish.")

    def _append_log(self, text: str) -> None:
        message = text.strip()
        if not message:
            return
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        try:
            line_count = int(self.log.index("end-1c").split(".")[0])
            if line_count > 400:
                self.log.delete("1.0", f"{line_count - 400}.0")
        except (ValueError, tk.TclError):
            pass
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _handle_training_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._append_log("Ignored malformed SAC progress event.")
            return
        self.last_event = payload
        status = str(
            payload.get("run_status") or payload.get("status") or "running"
        ).replace("_", " ")
        self.state_var.set(status.capitalize())
        transitions_value = _finite_float(payload.get("transitions"))
        transitions = int(transitions_value) if transitions_value is not None else 0
        limit_value = _finite_float(payload.get("transition_limit"))
        limit = int(limit_value) if limit_value is not None else None
        endless = bool(payload.get("endless", limit is None))
        episodes_value = _finite_float(payload.get("completed_episodes"))
        episodes = int(episodes_value) if episodes_value is not None else 0
        recent_training = _finite_float(
            payload.get("recent_training_success_rate")
        )
        recent_reward = _finite_float(
            payload.get("recent_training_mean_episode_reward")
        )
        episode_text = f"{_integer_text(episodes)} episodes"
        if recent_training is not None:
            episode_text += f"; {100.0 * recent_training:.1f}% recent"
        if recent_reward is not None:
            episode_text += f"; R={recent_reward:.2f}"
        validation_success = _finite_float(payload.get("validation_success_rate"))
        validation_reward = _finite_float(
            payload.get("validation_mean_episode_reward")
        )
        if episodes_value is not None:
            self.success_plot.add_sample(
                episodes_value,
                {"Training": recent_training, "Validation": validation_success},
            )
            self.reward_plot.add_sample(
                episodes_value,
                {"Training": recent_reward, "Validation": validation_reward},
            )
        self.progress_card_var.set(
            f"{_integer_text(transitions)} / endless\n{episode_text}"
            if endless
            else f"{_integer_text(transitions)} / {_integer_text(limit or 0)}\n{episode_text}"
        )
        if not endless and limit:
            self.progress_var.set(100.0 * min(max(transitions / limit, 0.0), 1.0))
        elapsed = _finite_float(payload.get("elapsed_s"))
        throughput = _finite_float(payload.get("transitions_per_s"))
        if elapsed is not None or throughput is not None:
            self.performance_var.set(
                f"{(throughput or 0.0):,.0f} trans/s\n{_duration_text(elapsed or 0.0)}"
            )

        overall = validation_success
        within = _finite_float(payload.get("validation_within_reach_success_rate"))
        beyond = _finite_float(payload.get("validation_beyond_reach_success_rate"))
        error_m = _finite_float(
            payload.get("validation_mean_minimum_error_m")
            if "validation_mean_minimum_error_m" in payload
            else payload.get("validation_mean_error_m")
        )
        speed_m_s = _finite_float(
            payload.get("validation_mean_directional_speed_m_s")
        )
        unsafe_rate = _finite_float(payload.get("validation_unsafe_rate"))
        if transitions_value is not None and overall is not None:
            self.error_plot.add_sample(
                transitions_value,
                {"Error": None if error_m is None else 1000.0 * error_m},
            )
            self.speed_plot.add_sample(
                transitions_value,
                {"Speed": speed_m_s},
            )
            self.unsafe_plot.add_sample(
                transitions_value,
                {"Unsafe": unsafe_rate},
            )
            error_text = "n/a" if error_m is None else f"{1000.0 * error_m:.1f} mm"
            speed_text = "n/a" if speed_m_s is None else f"{speed_m_s:.2f} m/s"
            unsafe_text = (
                "n/a" if unsafe_rate is None else f"{100.0 * unsafe_rate:.1f}% unsafe"
            )
            reward_text = (
                "n/a" if validation_reward is None else f"R={validation_reward:.2f}"
            )
            self.validation_var.set(
                f"{100.0 * overall:.1f}% success  |  {reward_text}  |  {error_text}\n"
                f"{speed_text}  |  {unsafe_text}"
            )

        best = _finite_float(
            payload.get("best_validation_success_rate")
            if "best_validation_success_rate" in payload
            else payload.get("best_success_rate")
        )
        best_transition_value = _finite_float(payload.get("best_transitions"))
        if best is not None:
            best_transition = int(best_transition_value or 0.0)
            self.best_var.set(f"{100.0 * best:.1f}%\n@ {_integer_text(best_transition)}")
        if bool(payload.get("best_checkpoint_updated")):
            self._append_log(
                f"Best policy updated at {_integer_text(int(best_transition_value or transitions))} transitions."
            )
        message = payload.get("message")
        if isinstance(message, str) and message.strip() and status.lower() != "training":
            self._append_log(message)

    def _handle_validation_preview(self, payload: object) -> None:
        if not isinstance(payload, dict):
            self._append_log("Ignored malformed validation preview event.")
            return
        try:
            self.validation_preview.set_preview(payload)
        except (TypeError, ValueError) as error:
            self._append_log(f"Ignored malformed validation preview: {error}")

    def _handle_exit(self, payload: object) -> None:
        if not isinstance(payload, tuple) or len(payload) != 2:
            return
        process, return_code = payload
        if process is not self.process:
            return
        self.progress.stop()
        self.process = None
        if self.stop_file is not None and self.stop_file.is_file():
            try:
                self.stop_file.unlink()
            except OSError:
                pass
        self.stop_file = None
        stopped = self.stopping
        self.stopping = False
        self._set_running(False)
        if int(return_code) == 0:
            self.progress_var.set(100.0 if not self.endless_var.get() and not stopped else self.progress_var.get())
            self.state_var.set("Stopped safely" if stopped else "Complete")
            self._append_log("Training stopped safely." if stopped else "Training completed successfully.")
        else:
            self.state_var.set(f"Failed (exit {int(return_code)})")
            self._append_log(f"Training process exited with code {int(return_code)}.")
            if not self.close_requested:
                messagebox.showerror(
                    "SAC training failed",
                    "The training process failed. See the concise log for the reported error.",
                    parent=self.root,
                )
        if self.close_requested:
            self.root.destroy()

    def _tick(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "event":
                if (
                    isinstance(payload, dict)
                    and payload.get("event_type") == "validation_preview"
                ):
                    self._handle_validation_preview(payload)
                else:
                    self._handle_training_event(payload)
            elif kind == "exit":
                self._handle_exit(payload)
        if self.running and not self.last_event:
            elapsed = time.perf_counter() - self.started_wall_s
            self.performance_var.set(f"Starting CUDA\n{_duration_text(elapsed)}")
        self.validation_preview.advance(time.perf_counter())
        if self.root.winfo_exists():
            self.root.after(self.POLL_MS, self._tick)

    def close(self) -> None:
        if self.running:
            self.close_requested = True
            self.stop_training()
            self.state_var.set("Stopping before close")
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    SacTrainingGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
