"""Live 3D canvas for continuous drone–cable control."""

from __future__ import annotations

import math
import tkinter as tk

import numpy as np

from optitrack_offline.viewer import CableCanvas

from .realtime import LiveFrame


class LiveDroneCanvas(CableCanvas):
    def __init__(self, parent: tk.Misc) -> None:
        self.frame: LiveFrame | None = None
        self.drone_path: list[np.ndarray] = []
        self.tip_path: list[np.ndarray] = []
        self._view_initialized = False
        super().__init__(parent)

    def clear(self) -> None:
        self.frame = None
        self.drone_path.clear()
        self.tip_path.clear()
        self._view_initialized = False
        self.redraw()

    def set_live_frame(self, frame: LiveFrame) -> None:
        self.frame = frame
        self.drone_path.append(frame.drone_position_m.copy())
        self.tip_path.append(frame.cable_positions_m[-1].copy())
        if not self._view_initialized:
            points = np.concatenate(
                (
                    frame.cable_positions_m,
                    frame.drone_position_m[None],
                    frame.target_position_m[None],
                ),
                axis=0,
            )
            lower = np.min(points, axis=0)
            upper = np.max(points, axis=0)
            lower[2] = min(lower[2], 0.0)
            span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.5)
            self.grid_step_m = self._nice_grid_step(span / 12.0)
            center_xy = 0.5 * (lower[:2] + upper[:2])
            half = max(0.9, 0.75 * span)
            self.floor_bounds_m = (
                math.floor((center_xy[0] - half) / self.grid_step_m) * self.grid_step_m,
                math.ceil((center_xy[0] + half) / self.grid_step_m) * self.grid_step_m,
                math.floor((center_xy[1] - half) / self.grid_step_m) * self.grid_step_m,
                math.ceil((center_xy[1] + half) / self.grid_step_m) * self.grid_step_m,
            )
            self.center_m = 0.5 * (lower + upper)
            self.radius_m = max(float(np.linalg.norm(upper - lower) * 0.7), 0.8)
            self._view_initialized = True
            self.reset_view()
        else:
            self.redraw()

    def _draw_path(self, points: list[np.ndarray] | np.ndarray, color: str, width: int) -> None:
        values = np.asarray(points)
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
            text=(
                "ONLINE TWO-PHASE MPC  |  MATCHED DDER MODEL  |  "
                "WORLD Z UP"
            ),
            fill="#222222",
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            width - 18,
            16,
            anchor=tk.NE,
            text="Drag: rotate    Right-drag: pan    Wheel: zoom",
            fill="#666666",
            font=("Segoe UI", 9),
        )
        if self.frame is None:
            self.create_text(
                width * 0.5,
                max(self.winfo_height(), 1) * 0.5,
                text="Start real-time MPC",
                fill="#555555",
                font=("Segoe UI Semibold", 16),
            )
            return
        self._draw_floor()
        self._draw_axes()
        frame = self.frame
        self._draw_path(self.drone_path, "#c87500", 2)
        self._draw_path(self.tip_path, "#7a3db8", 3)
        if frame.plan is not None:
            self._draw_path(frame.plan.prediction.drone_positions_m, "#9b7a4c", 1)
            self._draw_path(frame.plan.prediction.free_tip_positions_m, "#8e6aa3", 2)

        target_screen, _ = self._project(frame.target_position_m[None])
        tx, ty = target_screen[0]
        radius = 11
        self.create_oval(
            tx - radius,
            ty - radius,
            tx + radius,
            ty + radius,
            outline="#b42318",
            width=3,
        )
        arrow_start = frame.target_position_m - 0.18 * frame.impact_direction
        arrow_screen, _ = self._project(np.stack((arrow_start, frame.target_position_m)))
        self.create_line(
            *arrow_screen.reshape(-1),
            fill="#b42318",
            width=3,
            arrow=tk.LAST,
        )

        cable_screen, cable_depth = self._project(frame.cable_positions_m)
        self.create_line(*cable_screen.reshape(-1), fill="#0072b2", width=4)
        for index in np.argsort(cable_depth):
            x, y = cable_screen[index]
            endpoint = index in (0, len(cable_screen) - 1)
            point_radius = 7 if endpoint else 3
            color = "#c87500" if index == 0 else (
                "#7a3db8" if index == len(cable_screen) - 1 else "#0072b2"
            )
            self.create_oval(
                x - point_radius,
                y - point_radius,
                x + point_radius,
                y + point_radius,
                fill=color,
                outline="#333333",
                width=1,
            )

        pair, _ = self._project(
            np.stack((frame.drone_position_m, frame.attachment_position_m))
        )
        self.create_line(*pair.reshape(-1), fill="#c87500", width=3)
        x, y = pair[0]
        r = 12
        self.create_polygon(
            x,
            y - r,
            x + r,
            y,
            x,
            y + r,
            x - r,
            y,
            fill="#c87500",
            outline="#333333",
            width=2,
        )
        solve = "warming" if frame.controller_solve_time_s is None else (
            f"{1000.0 * frame.controller_solve_time_s:.0f}ms"
        )
        tip_velocity = frame.cable_velocities_m_s[-1]
        directional_speed = float(np.dot(tip_velocity, frame.impact_direction))
        tip_speed = float(np.linalg.norm(tip_velocity))
        direction_error_deg = float(
            np.degrees(
                np.arccos(
                    np.clip(directional_speed / max(tip_speed, 1.0e-9), -1.0, 1.0)
                )
            )
        )
        tip_error_m = float(
            np.linalg.norm(frame.cable_positions_m[-1] - frame.target_position_m)
        )
        directional_energy_j = (
            None
            if frame.plan is None
            else frame.plan.cost_terms.get("directional_tip_energy_j")
        )
        if frame.mission_complete:
            if frame.impact_tip_error_m is not None:
                tip_error_m = frame.impact_tip_error_m
            if frame.impact_directional_speed_m_s is not None:
                directional_speed = frame.impact_directional_speed_m_s
            if frame.impact_direction_error_deg is not None:
                direction_error_deg = frame.impact_direction_error_deg
            if frame.impact_directional_tip_energy_j is not None:
                directional_energy_j = frame.impact_directional_tip_energy_j
            outcome = "IMPACT" if frame.actual_hit_feasible is None else (
                "HIT" if frame.actual_hit_feasible else "MISS"
            )
            outcome_color = "#222222" if frame.actual_hit_feasible is None else (
                "#18794e" if frame.actual_hit_feasible else "#b42318"
            )
        elif frame.plan is None:
            outcome = "PLAN WARMING"
            outcome_color = "#222222"
        elif frame.plan.feasible:
            outcome = f"PREDICTED HIT IN {frame.plan.impact_time_s:.2f}s"
            outcome_color = "#18794e"
        else:
            outcome = (
                "BEST SAFE PROGRESS  "
                f"violation={frame.plan.constraint_violation:.3g}"
            )
            outcome_color = "#9a6700"
        self.create_text(
            18,
            42,
            anchor=tk.NW,
            text=(
                f"t={frame.simulation_time_s:.2f}/{frame.mission_duration_s:.2f}s   "
                f"phase={frame.casting_phase}   MPC={solve}   {outcome}"
            ),
            fill=outcome_color,
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            18,
            62,
            anchor=tk.NW,
            text=(
                f"tip error={1000.0 * tip_error_m:.0f}mm   "
                f"world directed speed={directional_speed:.2f}m/s   "
                f"direction error={direction_error_deg:.1f}deg   "
                f"directed tip energy="
                f"{'--' if directional_energy_j is None else f'{1000.0 * directional_energy_j:.2f}mJ'}"
            ),
            fill="#222222",
            font=("Segoe UI Semibold", 10),
        )
        self.create_text(
            18,
            82,
            anchor=tk.NW,
            text=(
                f"drone excursion={frame.maximum_drone_excursion_m:.2f}/"
                f"{frame.drone_excursion_limit_m:.2f}m   "
                f"forward/recoil={frame.maximum_forward_stroke_m:.2f}/"
                f"{frame.recoil_stroke_m:.2f}m   "
                f"minimum target clearance={frame.minimum_drone_clearance_m:.2f}m   "
                f"maximum drone speed={frame.maximum_drone_speed_m_s:.2f}m/s"
            ),
            fill="#555555",
            font=("Segoe UI Semibold", 10),
        )
