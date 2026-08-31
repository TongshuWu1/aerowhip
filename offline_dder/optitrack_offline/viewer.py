"""Small 3D playback UI for labeled OptiTrack cable markers."""

from __future__ import annotations

import math
from pathlib import Path
import tkinter as tk

import numpy as np

from .data import MotiveCableTake


CSV_DIRECTORY = Path(__file__).resolve().parent / "csv"


class CableCanvas(tk.Canvas):
    """Dependency-free orthographic 3D view for a small marker chain."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(
            parent,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#cfcfcf",
            cursor="fleur",
        )
        self.take: MotiveCableTake | None = None
        self.frame_index = 0
        self.center_m = np.zeros(3, dtype=np.float64)
        self.radius_m = 0.25
        self.floor_bounds_m = (-0.5, 0.5, -0.5, 0.5)
        self.grid_step_m = 0.1
        self.azimuth_deg = -45.0
        self.elevation_deg = 24.0
        self.zoom = 1.0
        self.pan_offset_m = np.zeros(3, dtype=np.float64)
        self.rollout_mode: str | None = None
        self.rollout_curve_m: np.ndarray | None = None
        self.rollout_observed: np.ndarray | None = None
        self.rollout_marker_node_indices: tuple[int, ...] | None = None
        self.prediction_trajectory_m: np.ndarray | None = None
        self.actual_trajectory_m: np.ndarray | None = None
        self.reference_points_m: np.ndarray | None = None
        self.show_correspondence = False
        self._last_mouse: tuple[int, int] | None = None
        self._last_pan_mouse: tuple[int, int] | None = None
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<ButtonPress-1>", self._start_rotate)
        self.bind("<B1-Motion>", self._rotate)
        self.bind("<ButtonRelease-1>", lambda _event: self._stop_rotate())
        self.bind("<ButtonPress-3>", self._start_pan)
        self.bind("<B3-Motion>", self._pan)
        self.bind("<ButtonRelease-3>", lambda _event: self._stop_pan())
        self.bind("<MouseWheel>", self._wheel)

    def set_take(self, take: MotiveCableTake) -> None:
        self.take = take
        self.rollout_mode = None
        self.rollout_curve_m = None
        self.rollout_observed = None
        self.rollout_marker_node_indices = None
        self.prediction_trajectory_m = None
        self.actual_trajectory_m = None
        self.reference_points_m = None
        self.show_correspondence = False
        self.frame_index = 0
        observed_positions = take.positions_m[take.observed]
        if len(observed_positions) == 0:
            raise ValueError("The cable take contains no observed markers.")
        lower = np.min(observed_positions, axis=0)
        upper = np.max(observed_positions, axis=0)
        horizontal_span = max(float(upper[0] - lower[0]), float(upper[1] - lower[1]), 1.0)
        self.grid_step_m = self._nice_grid_step(horizontal_span / 10.0)
        horizontal_center = 0.5 * (lower[:2] + upper[:2])
        half_floor = 0.65 * horizontal_span
        xmin = math.floor((horizontal_center[0] - half_floor) / self.grid_step_m) * self.grid_step_m
        xmax = math.ceil((horizontal_center[0] + half_floor) / self.grid_step_m) * self.grid_step_m
        ymin = math.floor((horizontal_center[1] - half_floor) / self.grid_step_m) * self.grid_step_m
        ymax = math.ceil((horizontal_center[1] + half_floor) / self.grid_step_m) * self.grid_step_m
        self.floor_bounds_m = (xmin, xmax, ymin, ymax)

        view_lower = lower.copy()
        view_lower[2] = min(float(view_lower[2]), 0.0)
        self.center_m = 0.5 * (view_lower + upper)
        self.radius_m = max(float(np.linalg.norm(upper - view_lower) * 0.55), 0.25)
        self.reset_view()

    def set_rollout_display(
        self,
        mode: str,
        curve_m: np.ndarray,
        *,
        prediction_trajectory_m: np.ndarray | None = None,
        actual_trajectory_m: np.ndarray | None = None,
        reference_points_m: np.ndarray | None = None,
        show_correspondence: bool = False,
        observed: np.ndarray | None = None,
        marker_node_indices: tuple[int, ...] | None = None,
    ) -> None:
        if self.take is None:
            raise ValueError("Set a take before drawing a rollout.")
        if mode not in {"prediction", "actual"}:
            raise ValueError("Rollout display mode must be prediction or actual.")
        curve = np.asarray(curve_m, dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 3:
            raise ValueError("Rollout curve must have shape Nx3.")
        mapping = (
            tuple(range(self.take.marker_count))
            if marker_node_indices is None
            else tuple(int(value) for value in marker_node_indices)
        )
        if (
            len(mapping) != self.take.marker_count
            or mapping[0] != 0
            or mapping[-1] != len(curve) - 1
            or any(b <= a for a, b in zip(mapping[:-1], mapping[1:]))
        ):
            raise ValueError("marker_node_indices is not a valid observation map.")
        inferred_observed = np.all(np.isfinite(curve), axis=1)
        curve_observed = (
            inferred_observed
            if observed is None
            else np.asarray(observed, dtype=bool)
        )
        if curve_observed.shape != (len(curve),):
            raise ValueError(f"observed must have shape ({len(curve)},).")
        if not np.array_equal(inferred_observed, curve_observed):
            raise ValueError("Each displayed marker must contain either XYZ or no value.")

        def trajectory(value: np.ndarray | None, name: str) -> np.ndarray | None:
            if value is None:
                return None
            result = np.asarray(value, dtype=np.float64)
            if result.ndim != 3 or result.shape[1:] != curve.shape:
                raise ValueError(f"{name} must have shape Tx{len(curve)}x3.")
            return result

        self.rollout_mode = mode
        self.rollout_curve_m = curve
        self.rollout_observed = curve_observed
        self.rollout_marker_node_indices = mapping
        self.prediction_trajectory_m = trajectory(
            prediction_trajectory_m,
            "prediction_trajectory_m",
        )
        self.actual_trajectory_m = trajectory(
            actual_trajectory_m,
            "actual_trajectory_m",
        )
        if reference_points_m is None:
            self.reference_points_m = None
        else:
            reference = np.asarray(reference_points_m, dtype=np.float64)
            expected_reference = (self.take.marker_count, 3)
            if reference.shape != expected_reference:
                raise ValueError(
                    f"reference_points_m must have shape {expected_reference}."
                )
            self.reference_points_m = reference
        self.show_correspondence = bool(show_correspondence)
        self.redraw()

    @staticmethod
    def _nice_grid_step(target_m: float) -> float:
        exponent = 10.0 ** math.floor(math.log10(max(target_m, 1.0e-6)))
        for multiplier in (1.0, 2.0, 5.0, 10.0):
            candidate = multiplier * exponent
            if candidate >= target_m:
                return candidate
        raise RuntimeError("Unable to choose a grid spacing.")

    def set_frame(self, frame_index: int) -> None:
        if self.take is None:
            return
        self.frame_index = int(np.clip(frame_index, 0, self.take.frame_count - 1))
        self.redraw()

    def reset_view(self) -> None:
        self.azimuth_deg = -45.0
        self.elevation_deg = 24.0
        self.zoom = 1.0
        self.pan_offset_m = np.zeros(3, dtype=np.float64)
        self.redraw()

    def _start_rotate(self, event: tk.Event) -> None:
        self._last_mouse = (event.x, event.y)

    def _stop_rotate(self) -> None:
        self._last_mouse = None

    def _start_pan(self, event: tk.Event) -> None:
        self._last_pan_mouse = (event.x, event.y)

    def _stop_pan(self) -> None:
        self._last_pan_mouse = None

    def _rotate(self, event: tk.Event) -> None:
        if self._last_mouse is None:
            return
        dx, dy = event.x - self._last_mouse[0], event.y - self._last_mouse[1]
        self._last_mouse = (event.x, event.y)
        self.azimuth_deg += 0.45 * dx
        self.elevation_deg = float(np.clip(self.elevation_deg + 0.35 * dy, -85.0, 85.0))
        self.redraw()

    def _wheel(self, event: tk.Event) -> None:
        factor = 1.12 if event.delta > 0 else 1.0 / 1.12
        self.zoom = float(np.clip(self.zoom * factor, 0.25, 8.0))
        self.redraw()

    def _pan(self, event: tk.Event) -> None:
        if self._last_pan_mouse is None:
            return
        dx = event.x - self._last_pan_mouse[0]
        dy = event.y - self._last_pan_mouse[1]
        self._last_pan_mouse = (event.x, event.y)
        right, up, _camera = self._camera_basis()
        scale = self._pixels_per_metre()
        self.pan_offset_m = (
            self.pan_offset_m - (dx / scale) * right + (dy / scale) * up
        )
        self.redraw()

    def _camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        azimuth = math.radians(self.azimuth_deg)
        elevation = math.radians(self.elevation_deg)
        camera = np.asarray(
            (
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ),
            dtype=np.float64,
        )
        right = np.asarray((-math.sin(azimuth), math.cos(azimuth), 0.0))
        up = np.cross(camera, right)
        return right, up, camera

    def _project(self, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        width, height = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        right, up, camera = self._camera_basis()
        centered = (
            np.asarray(points_m, dtype=np.float64)
            - self.center_m
            - self.pan_offset_m
        )
        scale = self._pixels_per_metre()
        screen = np.empty((len(centered), 2), dtype=np.float64)
        screen[:, 0] = width * 0.5 + centered @ right * scale
        screen[:, 1] = height * 0.5 - centered @ up * scale
        depth = centered @ camera
        return screen, depth

    def _pixels_per_metre(self) -> float:
        width, height = max(self.winfo_width(), 1), max(self.winfo_height(), 1)
        return 0.40 * min(width, height) * self.zoom / self.radius_m

    def _draw_axes(self) -> None:
        length = self.radius_m * 0.38
        points = np.asarray(
            [
                self.center_m,
                self.center_m + (length, 0.0, 0.0),
                self.center_m + (0.0, length, 0.0),
                self.center_m + (0.0, 0.0, length),
            ]
        )
        screen, _ = self._project(points)
        origin = screen[0]
        for target, color, label in zip(
            screen[1:], ("#b42318", "#2f7d45", "#2f62a2"), ("X", "Y", "Z")
        ):
            self.create_line(*origin, *target, fill=color, width=2, arrow=tk.LAST)
            self.create_text(*target, text=label, fill=color, font=("Segoe UI Semibold", 10))

    def _draw_floor(self) -> None:
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
        screen, _ = self._project(corners)
        self.create_polygon(
            *screen.reshape(-1),
            fill="#fafafa",
            outline="#b8b8b8",
            width=1,
            tags=("environment",),
        )

        major_step = 5.0 * self.grid_step_m
        tolerance = self.grid_step_m * 1.0e-5
        x_values = np.arange(xmin, xmax + 0.5 * self.grid_step_m, self.grid_step_m)
        y_values = np.arange(ymin, ymax + 0.5 * self.grid_step_m, self.grid_step_m)
        for x in x_values:
            points = np.asarray(((x, ymin, 0.0), (x, ymax, 0.0)))
            projected, _ = self._project(points)
            major = abs(x / major_step - round(x / major_step)) < tolerance
            self.create_line(
                *projected.reshape(-1),
                fill="#c7c7c7" if major else "#e5e5e5",
                width=1,
                tags=("environment",),
            )
        for y in y_values:
            points = np.asarray(((xmin, y, 0.0), (xmax, y, 0.0)))
            projected, _ = self._project(points)
            major = abs(y / major_step - round(y / major_step)) < tolerance
            self.create_line(
                *projected.reshape(-1),
                fill="#c7c7c7" if major else "#e5e5e5",
                width=1,
                tags=("environment",),
            )
        self.create_text(
            *screen[0],
            text=f"  Z = 0 m   grid {self.grid_step_m:g} m",
            anchor=tk.NW,
            fill="#666666",
            font=("Segoe UI", 9),
            tags=("environment",),
        )
        self.tag_lower("environment")

    def _draw_marker_trajectories(
        self,
        trajectory_m: np.ndarray | None,
        *,
        color: str,
        dash: tuple[int, int] | None,
        marker_indices: range | tuple[int, ...] | None = None,
    ) -> None:
        if trajectory_m is None or len(trajectory_m) < 2:
            return
        indices = (
            range(trajectory_m.shape[1])
            if marker_indices is None
            else marker_indices
        )
        for marker in indices:
            screen, _depth = self._project(trajectory_m[:, marker])
            options: dict[str, object] = {
                "fill": color,
                "width": 2,
            }
            if dash is not None:
                options["dash"] = dash
            self.create_line(*screen.reshape(-1), **options)

    def redraw(self) -> None:
        self.delete("all")
        width = max(self.winfo_width(), 1)
        self.create_text(
            18,
            16,
            anchor=tk.NW,
            text=(
                (
                    "PREDICTED ROD / ACTUAL MARKERS / ERROR LINKS"
                    if self.reference_points_m is not None
                    else "KNOWN ATTACHMENT / PREDICTED FREE CABLE  |  PROJECT Z UP"
                )
                if self.rollout_mode == "prediction"
                else (
                    "ACTUAL OPTITRACK CABLE / ATTACHMENT  |  PROJECT Z UP"
                    if self.rollout_mode == "actual"
                    else "MEASURED OPTITRACK MARKERS  |  PROJECT Z UP"
                )
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
        if self.take is None:
            self.create_text(
                width * 0.5,
                max(self.winfo_height(), 1) * 0.5,
                text="Open a Motive CSV",
                fill="#555555",
                font=("Segoe UI Semibold", 18),
            )
            return

        self._draw_floor()
        self._draw_axes()
        self._draw_marker_trajectories(
            self.prediction_trajectory_m,
            color="#7a3db8",
            dash=(5, 3),
            marker_indices=(
                range(1, self.take.marker_count)
                if self.rollout_marker_node_indices is None
                else self.rollout_marker_node_indices[1:]
            ),
        )
        self._draw_marker_trajectories(
            self.prediction_trajectory_m,
            color="#c87500",
            dash=None,
            marker_indices=(
                (0,)
                if self.rollout_marker_node_indices is None
                else (self.rollout_marker_node_indices[0],)
            ),
        )
        self._draw_marker_trajectories(
            self.actual_trajectory_m,
            color="#198754",
            dash=None,
            marker_indices=range(1, self.take.marker_count),
        )
        points = (
            self.rollout_curve_m
            if self.rollout_curve_m is not None
            else self.take.positions_m[self.frame_index]
        )
        observed = (
            self.rollout_observed
            if self.rollout_curve_m is not None
            else self.take.observed[self.frame_index]
        )
        assert observed is not None
        marker_node_indices = (
            tuple(range(self.take.marker_count))
            if self.rollout_marker_node_indices is None
            else self.rollout_marker_node_indices
        )
        if self.reference_points_m is not None:
            reference_valid = np.all(np.isfinite(self.reference_points_m), axis=1)
            if self.show_correspondence:
                for marker in range(1, self.take.marker_count):
                    node = marker_node_indices[marker]
                    if not (
                        observed[node]
                        and reference_valid[marker]
                        and np.all(np.isfinite(points[node]))
                    ):
                        continue
                    pair_screen, _ = self._project(
                        np.stack((points[node], self.reference_points_m[marker]))
                    )
                    self.create_line(
                        *pair_screen.reshape(-1),
                        fill="#888888",
                        width=2,
                        dash=(3, 4),
                    )
            if np.any(reference_valid):
                reference_screen, reference_depth = self._project(
                    self.reference_points_m[reference_valid]
                )
                reference_indices = np.flatnonzero(reference_valid)
                for row in np.argsort(reference_depth):
                    index = int(reference_indices[row])
                    x, y = reference_screen[row]
                    radius = 6 if index in (0, self.take.marker_count - 1) else 5
                    self.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        fill="#198754",
                        outline="#ffffff",
                        width=1,
                    )
        cable_color = "#7a3db8" if self.rollout_mode == "prediction" else "#198754"
        valid_points = points[observed]
        valid_indices = np.flatnonzero(observed)
        if len(valid_points):
            screen, depth = self._project(valid_points)
            projected = {int(index): screen[row] for row, index in enumerate(valid_indices)}
            for index in range(len(points) - 1):
                if observed[index] and observed[index + 1]:
                    self.create_line(
                        *projected[index],
                        *projected[index + 1],
                        fill=cable_color,
                        width=4,
                    )
            for row in np.argsort(depth):
                index = int(valid_indices[row])
                x, y = screen[row]
                attachment = index == 0
                free_tip = index == len(points) - 1
                measured_node = index in marker_node_indices
                radius = 8 if attachment else (7 if free_tip else (6 if measured_node else 3))
                color = (
                    "#c87500"
                    if attachment
                    else ("#7a3db8" if self.rollout_mode == "prediction" else "#198754")
                )
                self.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=color,
                    outline="#333333",
                    width=2,
                )
                if measured_node:
                    marker = marker_node_indices.index(index)
                    self.create_text(
                        x + 10,
                        y - 11,
                        anchor=tk.SW,
                        text=self.take.marker_names[marker],
                        fill="#111111",
                        font=("Segoe UI Semibold", 10),
                    )
        missing = [
            self.take.marker_names[index]
            for index in range(self.take.marker_count)
            if not observed[marker_node_indices[index]]
        ]
        if missing:
            self.create_text(
                18,
                42,
                anchor=tk.NW,
                text="Missing: " + ", ".join(missing),
                fill="#b42318",
                font=("Segoe UI Semibold", 10),
            )
