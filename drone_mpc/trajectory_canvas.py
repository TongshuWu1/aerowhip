"""Reusable trajectory viewport without legacy optimizer dependencies."""

from __future__ import annotations

import math
from typing import Protocol
import tkinter as tk

import numpy as np

from optitrack_offline.viewer import CableCanvas


class _TrajectoryResult(Protocol):
    prediction: object
    feasible: bool
    impact_frame: int
    terms: dict[str, float]
    speed_amplification: float


class WhipTrajectoryCanvas(CableCanvas):
    """3-D cable/drone playback used by both current and legacy applications."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.banner_text = (
            "OFFLINE WHIP TRAJECTORY  |  EXACT FULL DER PLAYBACK  |  WORLD Z UP"
        )
        self.result: _TrajectoryResult | None = None
        self.frame_index = 0

    def set_result(self, result: _TrajectoryResult) -> None:
        self.result = result
        self.frame_index = 0
        prediction = result.prediction
        points = np.concatenate(
            (
                prediction.cable_positions_m.reshape(-1, 3),
                prediction.drone_positions_m,
                prediction.target_position_m[None],
            ),
            axis=0,
        )
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        lower[2] = min(lower[2], 0.0)
        span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.2)
        self.grid_step_m = self._nice_grid_step(span / 10.0)
        center_xy = 0.5 * (lower[:2] + upper[:2])
        half = max(0.7, 0.65 * span)
        self.floor_bounds_m = (
            math.floor((center_xy[0] - half) / self.grid_step_m) * self.grid_step_m,
            math.ceil((center_xy[0] + half) / self.grid_step_m) * self.grid_step_m,
            math.floor((center_xy[1] - half) / self.grid_step_m) * self.grid_step_m,
            math.ceil((center_xy[1] + half) / self.grid_step_m) * self.grid_step_m,
        )
        self.center_m = 0.5 * (lower + upper)
        self.radius_m = max(float(np.linalg.norm(upper - lower) * 0.6), 0.7)
        self.reset_view()

    def set_trajectory_frame(self, frame_index: int) -> None:
        if self.result is None:
            return
        self.frame_index = int(
            np.clip(frame_index, 0, self.result.prediction.frame_count - 1)
        )
        self.redraw()

    def _draw_path(self, values: np.ndarray, color: str, width: int) -> None:
        if len(values) < 2:
            return
        screen, _ = self._project(values)
        self.create_line(*screen.reshape(-1), fill=color, width=width, dash=(5, 3))

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        self.create_text(
            18,
            16,
            anchor=tk.NW,
            text=self.banner_text,
            fill="#a9b9c8",
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            width - 18,
            16,
            anchor=tk.NE,
            text="Drag: rotate    Right-drag: pan    Wheel: zoom",
            fill="#72879a",
            font=("Segoe UI", 9),
        )
        if self.result is None:
            self.create_text(
                width * 0.5,
                max(self.winfo_height(), 1) * 0.5,
                text="Run or load one trajectory",
                fill="#d8e3ec",
                font=("Segoe UI Semibold", 16),
            )
            return
        self._draw_floor()
        self._draw_axes()
        result = self.result
        prediction = result.prediction
        frame = self.frame_index
        self._draw_path(prediction.drone_positions_m[: frame + 1], "#ffad55", 2)
        self._draw_path(prediction.free_tip_positions_m[: frame + 1], "#ee67d4", 3)

        target_screen, _ = self._project(prediction.target_position_m[None])
        tx, ty = target_screen[0]
        self.create_oval(
            tx - 11, ty - 11, tx + 11, ty + 11, outline="#ff5e67", width=3
        )
        arrow_start = prediction.target_position_m - 0.18 * prediction.impact_direction
        arrow, _ = self._project(np.stack((arrow_start, prediction.target_position_m)))
        self.create_line(*arrow.reshape(-1), fill="#ff7c82", width=3, arrow=tk.LAST)

        cable = prediction.cable_positions_m[frame]
        cable_screen, depth = self._project(cable)
        self.create_line(*cable_screen.reshape(-1), fill="#55d7e8", width=4)
        for index in np.argsort(depth):
            x, y = cable_screen[index]
            radius = 7 if index in (0, len(cable) - 1) else 3
            color = (
                "#ffd067"
                if index == 0
                else "#ee67d4"
                if index == len(cable) - 1
                else "#55d7e8"
            )
            self.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline="#f0f7fb",
            )
        pair, _ = self._project(
            np.stack(
                (
                    prediction.drone_positions_m[frame],
                    prediction.attachment_positions_m[frame],
                )
            )
        )
        self.create_line(*pair.reshape(-1), fill="#e8a854", width=3)
        x, y = pair[0]
        self.create_polygon(
            x,
            y - 12,
            x + 12,
            y,
            x,
            y + 12,
            x - 12,
            y,
            fill="#ffad55",
            outline="#fff0d8",
            width=2,
        )
        outcome = "FEASIBLE HIT" if result.feasible else "BEST INFEASIBLE TRAJECTORY"
        color = "#66df91" if result.feasible else "#ff7c82"
        self.create_text(
            18,
            42,
            anchor=tk.NW,
            text=(
                f"t={prediction.time_s[frame]:.2f}s  "
                f"impact={prediction.time_s[result.impact_frame]:.2f}s  {outcome}"
            ),
            fill=color,
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            18,
            62,
            anchor=tk.NW,
            text=(
                f"tip error={1000.0 * result.terms['position_error_m']:.1f}mm   "
                f"directed speed={result.terms['directional_speed_m_s']:.2f}m/s   "
                f"direction error={result.terms['direction_error_deg']:.1f}deg   "
                f"speed amplification={result.speed_amplification:.2f}x"
            ),
            fill="#d8e3ec",
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            18,
            82,
            anchor=tk.NW,
            text=(
                f"measured max displacement="
                f"{result.terms['maximum_drone_excursion_m']:.3f}m   "
                f"forward/recoil={result.terms.get('maximum_forward_stroke_m', 0.0):.3f}/"
                f"{result.terms.get('recoil_stroke_m', 0.0):.3f}m   "
                f"drone max speed={result.terms['maximum_drone_speed_m_s']:.2f}m/s"
            ),
            fill="#9fb2c2",
            font=("Segoe UI Semibold", 10),
        )


__all__ = ["WhipTrajectoryCanvas"]
