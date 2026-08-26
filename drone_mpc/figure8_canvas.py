"""Dependency-free live views for the figure-eight tracking experiment."""

from __future__ import annotations

import math
import tkinter as tk

import numpy as np

from optitrack_offline.viewer import CableCanvas

from .figure8_tracking import Figure8LiveUpdate, Figure8Reference


def causal_rolling_mean(
    time_s: np.ndarray,
    values: np.ndarray,
    window_s: float,
) -> np.ndarray:
    """Return a trailing time-window mean without using future samples."""

    times = np.asarray(time_s, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    if times.shape != samples.shape:
        raise ValueError("Rolling-mean timestamps and values must have the same shape.")
    if times.ndim != 1:
        raise ValueError("Rolling-mean inputs must be one-dimensional.")
    if window_s <= 0.0 or not math.isfinite(window_s):
        raise ValueError("Rolling-mean window must be finite and positive.")
    if len(times) == 0:
        return samples.copy()
    if np.any(np.diff(times) < 0.0):
        raise ValueError("Rolling-mean timestamps must be nondecreasing.")
    starts = np.searchsorted(times, times - window_s, side="left")
    prefix = np.concatenate(([0.0], np.cumsum(samples, dtype=np.float64)))
    ends = np.arange(1, len(samples) + 1)
    return (prefix[ends] - prefix[starts]) / (ends - starts)


class Figure8TrackingCanvas(CableCanvas):
    """Interactive orthographic 3-D view of the live tracking state."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.update_data: Figure8LiveUpdate | None = None
        self.reference_path_m: np.ndarray | None = None
        self.observation_mode = "full"

    def configure_scene(
        self,
        cable_positions_m: np.ndarray,
        drone_position_m: np.ndarray,
        reference: Figure8Reference,
        observation_mode: str,
    ) -> None:
        self.reference_path_m = reference.path_numpy()
        self.observation_mode = observation_mode
        points = np.concatenate(
            (
                np.asarray(cable_positions_m),
                np.asarray(drone_position_m)[None],
                self.reference_path_m,
            ),
            axis=0,
        )
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        view_lower = lower.copy()
        view_lower[2] = min(view_lower[2], 0.0)
        span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 0.8)
        self.grid_step_m = self._nice_grid_step(span / 10.0)
        center_xy = 0.5 * (lower[:2] + upper[:2])
        half = max(0.45, 0.70 * span)
        self.floor_bounds_m = (
            math.floor((center_xy[0] - half) / self.grid_step_m) * self.grid_step_m,
            math.ceil((center_xy[0] + half) / self.grid_step_m) * self.grid_step_m,
            math.floor((center_xy[1] - half) / self.grid_step_m) * self.grid_step_m,
            math.ceil((center_xy[1] + half) / self.grid_step_m) * self.grid_step_m,
        )
        self.center_m = 0.5 * (view_lower + upper)
        self.radius_m = max(float(np.linalg.norm(upper - view_lower) * 0.62), 0.55)
        self.update_data = None
        self.reset_view()

    def set_live_update(self, update: Figure8LiveUpdate) -> None:
        self.update_data = update
        self.observation_mode = update.observation_mode
        self.redraw()

    def _draw_path(
        self,
        values: np.ndarray,
        color: str,
        width: int,
        dash: tuple[int, int] | None = None,
    ) -> None:
        if len(values) < 2:
            return
        screen, _ = self._project(values)
        options: dict[str, object] = {"fill": color, "width": width}
        if dash is not None:
            options["dash"] = dash
        self.create_line(*screen.reshape(-1), **options)

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        self.create_text(
            16,
            14,
            anchor=tk.NW,
            text=(
                "FREE-TIP FIGURE-8 PATH FOLLOWING  |  "
                f"{self.observation_mode.upper()} OBSERVATION  |  WORLD Z UP"
            ),
            fill="#333333",
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            width - 16,
            14,
            anchor=tk.NE,
            text="Drag: rotate    Right-drag: pan    Wheel: zoom",
            fill="#777777",
            font=("Segoe UI", 9),
        )
        if self.reference_path_m is None:
            self.create_text(
                width * 0.5,
                max(self.winfo_height(), 1) * 0.5,
                text="Load the cable model",
                fill="#777777",
                font=("Segoe UI Semibold", 15),
            )
            return
        self._draw_floor()
        self._draw_axes()
        self._draw_path(self.reference_path_m, "#2f7d45", 2, (6, 4))
        update = self.update_data
        if update is None:
            return
        self._draw_path(
            update.recent_actual_tip_positions_m, "#9b2c83", 3
        )
        cable_screen, depth = self._project(update.cable_positions_m)
        self.create_line(*cable_screen.reshape(-1), fill="#267f9b", width=4)
        for index in np.argsort(depth):
            x, y = cable_screen[index]
            radius = 7 if index in (0, len(cable_screen) - 1) else 3
            color = (
                "#e39a36"
                if index == 0
                else "#9b2c83"
                if index == len(cable_screen) - 1
                else "#267f9b"
            )
            self.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline="#ffffff",
            )
        drone_and_root = np.stack(
            (update.drone_position_m, update.cable_positions_m[0])
        )
        pair, _ = self._project(drone_and_root)
        self.create_line(*pair.reshape(-1), fill="#d28a28", width=3)
        x, y = pair[0]
        self.create_polygon(
            x,
            y - 11,
            x + 11,
            y,
            x,
            y + 11,
            x - 11,
            y,
            fill="#e39a36",
            outline="#6b4614",
            width=2,
        )
        error_pair = np.stack(
            (
                update.cable_positions_m[-1],
                update.reference_tip_position_m,
            )
        )
        error_screen, _ = self._project(error_pair)
        self.create_line(
            *error_screen.reshape(-1), fill="#c23b32", width=2, dash=(4, 3)
        )
        rx, ry = error_screen[1]
        self.create_oval(
            rx - 7,
            ry - 7,
            rx + 7,
            ry + 7,
            outline="#2f7d45",
            width=3,
        )
        self.create_text(
            16,
            42,
            anchor=tk.NW,
            text=(
                f"t={update.time_s:.2f}s   progress={update.path_progress_cycles:.3f} cycles   "
                f"error={update.tracking_error_m:.4f}m   "
                f"RMSE={update.rmse_m:.4f}m   MPPI={update.planning_wall_time_s:.3f}s"
            ),
            fill="#222222",
            font=("Segoe UI Semibold", 10),
        )


class TrackingErrorCanvas(tk.Canvas):
    """Small scrolling causal rolling-average plot without matplotlib."""

    ROLLING_WINDOW_S = 0.5

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(
            parent,
            height=180,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#cfcfcf",
        )
        self.time_s = np.empty(0)
        self.error_m = np.empty(0)
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_data(self, time_s: np.ndarray, error_m: np.ndarray) -> None:
        self.time_s = np.asarray(time_s, dtype=np.float64)
        self.error_m = causal_rolling_mean(
            self.time_s,
            np.asarray(error_m, dtype=np.float64),
            self.ROLLING_WINDOW_S,
        )
        self.redraw()

    def clear(self) -> None:
        self.time_s = np.empty(0)
        self.error_m = np.empty(0)
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        left, right, top, bottom = 52.0, width - 14.0, 18.0, height - 30.0
        self.create_text(
            10,
            8,
            anchor=tk.NW,
            text=f"Live free-tip tracking error · {self.ROLLING_WINDOW_S:.1f} s rolling mean",
            fill="#222222",
            font=("Segoe UI Semibold", 10),
        )
        self.create_line(left, top, left, bottom, fill="#777777")
        self.create_line(left, bottom, right, bottom, fill="#777777")
        if len(self.time_s) < 2:
            self.create_text(
                (left + right) * 0.5,
                (top + bottom) * 0.5,
                text="Play to collect tracking error",
                fill="#777777",
            )
            return
        time_min = float(self.time_s[0])
        time_max = max(float(self.time_s[-1]), time_min + 1.0e-6)
        error_max = max(float(np.max(self.error_m)) * 1.12, 0.005)
        x = left + (self.time_s - time_min) / (time_max - time_min) * (right - left)
        y = bottom - self.error_m / error_max * (bottom - top)
        points = np.stack((x, y), axis=1)
        self.create_line(*points.reshape(-1), fill="#9b2c83", width=2)
        self.create_text(
            left - 6,
            top,
            anchor=tk.E,
            text=f"{error_max:.3f} m",
            fill="#555555",
            font=("Segoe UI", 8),
        )
        self.create_text(
            left - 6,
            bottom,
            anchor=tk.E,
            text="0",
            fill="#555555",
            font=("Segoe UI", 8),
        )
        self.create_text(
            left,
            bottom + 7,
            anchor=tk.NW,
            text=f"{time_min:.1f}s",
            fill="#555555",
            font=("Segoe UI", 8),
        )
        self.create_text(
            right,
            bottom + 7,
            anchor=tk.NE,
            text=f"{time_max:.1f}s",
            fill="#555555",
            font=("Segoe UI", 8),
        )
