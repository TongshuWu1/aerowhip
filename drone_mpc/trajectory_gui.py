"""Small UI for one offline, full-model-verified whip trajectory."""

from __future__ import annotations

import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from optitrack_offline.config import DEFAULT_MODEL_PATH
from optitrack_offline.viewer import CableCanvas

from .model import CableModelSnapshot, load_cable_model
from .mpc import MpcProblem
from .oracle import (
    ReachabilityResult,
    ReachabilitySettings,
    optimize_reachability,
    save_reachability_result,
)


DEFAULT_OUTPUT_PATH = Path("data/drone_mpc/whip_trajectory.npz")


class WhipTrajectoryCanvas(CableCanvas):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.banner_text = (
            "OFFLINE WHIP TRAJECTORY  |  EXACT FULL DER PLAYBACK  |  WORLD Z UP"
        )
        self.result: ReachabilityResult | None = None
        self.frame_index = 0

    def set_result(self, result: ReachabilityResult) -> None:
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
        self.create_line(
            *screen.reshape(-1), fill=color, width=width, dash=(5, 3)
        )

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
                text="Optimize one trajectory",
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
            color = "#ffd067" if index == 0 else (
                "#ee67d4" if index == len(cable) - 1 else "#55d7e8"
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
                f"t={prediction.time_s[frame]:.2f}s  impact={prediction.time_s[result.impact_frame]:.2f}s  "
                f"{outcome}"
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
                f"measured max displacement={result.terms['maximum_drone_excursion_m']:.3f}m   "
                f"forward/recoil={result.terms['maximum_forward_stroke_m']:.3f}/"
                f"{result.terms['recoil_stroke_m']:.3f}m   "
                f"drone max speed={result.terms['maximum_drone_speed_m_s']:.2f}m/s"
            ),
            fill="#9fb2c2",
            font=("Segoe UI Semibold", 10),
        )


class WhipTrajectoryGui:
    TICK_MS = 20

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cable Twin – Offline Whip Trajectory")
        self.root.geometry("1500x900")
        self.root.minsize(1180, 720)
        self.root.configure(background="#0d1721")
        self._configure_style()
        self.snapshot: CableModelSnapshot | None = None
        self.result: ReachabilityResult | None = None
        self.active_settings: ReachabilitySettings | None = None
        self.active_problem: MpcProblem | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.optimizing = False
        self.playing = False

        self.model_path_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH.resolve()))
        self.model_status_var = tk.StringVar(value="No model loaded")
        self.initial_var = tk.StringVar(value="0.0, 0.0, 1.5")
        self.target_var = tk.StringVar(value="0.48, 0.0, 0.90")
        self.direction_var = tk.StringVar(value="1.0, 0.0, 0.0")
        self.minimum_speed_var = tk.StringVar(value="1.0")
        self.hit_tolerance_var = tk.StringVar(value="0.05")
        self.angle_var = tk.StringVar(value="35.0")
        self.excursion_var = tk.StringVar(value="0.15")
        self.horizon_var = tk.StringVar(value="4.0")
        self.acceleration_var = tk.StringVar(value="20.0")
        self.search_iterations_var = tk.StringVar(value="40")
        self.status_var = tk.StringVar(value="Load the cable model, then optimize")
        self.timeline_var = tk.DoubleVar(value=0.0)

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.TICK_MS, self._tick)
        self.load_model(silent=True)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#132331", foreground="#dce8f1")
        style.configure("TFrame", background="#132331")
        style.configure("TLabelframe", background="#132331", foreground="#9fc0d8")
        style.configure("TLabelframe.Label", background="#132331", foreground="#9fc0d8")
        style.configure("TLabel", background="#132331", foreground="#dce8f1")
        style.configure("TButton", background="#1d3446", foreground="#eaf2f8", padding=7)
        style.configure("TEntry", fieldbackground="#f6f7f8", foreground="#17212a")

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(outer, text="Offline whip trajectory", font=("Segoe UI Semibold", 24)).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text="Reduced-model search followed by mandatory exact full-DER refinement and verification.",
            foreground="#91a9ba",
        ).pack(anchor=tk.W, pady=(0, 10))
        model = ttk.LabelFrame(outer, text="Frozen cable model", padding=10)
        model.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(model, textvariable=self.model_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model, text="Browse", command=self.browse_model).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(model, text="Load", command=self.load_model).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(model, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=(12, 0))

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(body, width=390)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls.pack_propagate(False)
        view = ttk.Frame(body)
        view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        task = ttk.LabelFrame(controls, text="One fixed development task", padding=10)
        task.pack(fill=tk.X)
        self._entry(task, "Initial drone XYZ", self.initial_var, "m")
        self._entry(task, "Free-tip target XYZ", self.target_var, "m")
        self._entry(task, "Impact direction", self.direction_var, "unit")
        self._entry(task, "Minimum directed speed", self.minimum_speed_var, "m/s")
        self._entry(task, "Tip hit tolerance", self.hit_tolerance_var, "m")
        self._entry(task, "Direction half-angle", self.angle_var, "deg")
        self._entry(task, "Maximum drone excursion", self.excursion_var, "m")
        self._entry(task, "Trajectory horizon", self.horizon_var, "s")
        self._entry(task, "Maximum acceleration", self.acceleration_var, "m/s²")
        self._entry(task, "Search iterations", self.search_iterations_var, "")

        actions = ttk.LabelFrame(controls, text="Run", padding=10)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.optimize_button = ttk.Button(actions, text="Optimize trajectory", command=self.optimize)
        self.optimize_button.pack(fill=tk.X)
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=(6, 0))
        self.play_button = ttk.Button(actions, text="Play", command=self.toggle_play, state=tk.DISABLED)
        self.play_button.pack(fill=tk.X, pady=(6, 0))
        self.save_button = ttk.Button(actions, text="Save as...", command=self.save_as, state=tk.DISABLED)
        self.save_button.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(actions, text="Reset view", command=lambda: self.canvas.reset_view()).pack(fill=tk.X, pady=(6, 0))

        log_frame = ttk.LabelFrame(controls, text="Optimizer", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log = tk.Text(
            log_frame,
            height=12,
            background="#09111a",
            foreground="#b9cad7",
            insertbackground="#ffffff",
            borderwidth=0,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.log.pack(fill=tk.BOTH, expand=True)

        self.canvas = WhipTrajectoryCanvas(view)
        self.canvas.pack(fill=tk.BOTH, expand=True)
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
        row.pack(fill=tk.X, pady=3)
        ttk.Label(row, text=label, width=24).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=15).pack(side=tk.LEFT, fill=tk.X, expand=True)
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
            messagebox.showerror("Whip trajectory", "Load a cable model first.", parent=self.root)
            return
        try:
            initial = self._vector(self.initial_var.get(), "Initial drone XYZ")
            problem = MpcProblem(
                target_position_m=self._vector(self.target_var.get(), "Target XYZ"),
                impact_direction=self._vector(self.direction_var.get(), "Impact direction"),
                minimum_impact_speed_m_s=float(self.minimum_speed_var.get()),
                maximum_tip_error_m=float(self.hit_tolerance_var.get()),
                maximum_impact_angle_deg=float(self.angle_var.get()),
                maximum_drone_excursion_m=float(self.excursion_var.get()),
                drone_keepout_radius_m=0.30,
                minimum_forward_stroke_m=0.05,
                minimum_recoil_stroke_m=0.05,
            )
            settings = ReachabilitySettings(
                horizon_s=float(self.horizon_var.get()),
                iterations=int(self.search_iterations_var.get()),
                maximum_acceleration_m_s2=float(self.acceleration_var.get()),
            )
        except Exception as error:
            messagebox.showerror("Whip trajectory", str(error), parent=self.root)
            return
        self.active_problem = problem
        self.active_settings = settings
        self.cancel_event.clear()
        self.optimizing = True
        self.playing = False
        self.result = None
        self.canvas.result = None
        self.canvas.redraw()
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        self.optimize_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.play_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.status_var.set("Searching smooth forward/recoil motions...")
        snapshot = self.snapshot

        def worker() -> None:
            try:
                result = optimize_reachability(
                    snapshot,
                    problem,
                    initial,
                    settings,
                    progress=lambda message: self.events.put(("log", message)),
                    cancelled=self.cancel_event.is_set,
                )
                output = save_reachability_result(
                    DEFAULT_OUTPUT_PATH, result, settings, problem
                )
            except Exception as error:
                self.events.put(("error", error))
            else:
                self.events.put(("complete", (result, output)))

        threading.Thread(target=worker, name="offline-whip-optimizer", daemon=True).start()

    def stop(self) -> None:
        if self.optimizing:
            self.cancel_event.set()
            self.status_var.set("Stopping after the current rollout...")

    def toggle_play(self) -> None:
        if self.result is None:
            return
        self.playing = not self.playing
        if self.playing and self.timeline_var.get() >= self.result.prediction.frame_count - 1:
            self.timeline_var.set(0.0)
        self.play_button.configure(text="Pause" if self.playing else "Play")

    def _timeline_changed(self, _value: str) -> None:
        if self.result is not None:
            self.canvas.set_trajectory_frame(int(round(self.timeline_var.get())))

    def save_as(self) -> None:
        if self.result is None or self.active_settings is None or self.active_problem is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save verified whip trajectory",
            defaultextension=".npz",
            filetypes=(("NumPy archive", "*.npz"),),
        )
        if path:
            save_reachability_result(path, self.result, self.active_settings, self.active_problem)
            self.status_var.set(f"Saved {Path(path).resolve()}")

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
                        messagebox.showerror("Whip trajectory", str(payload), parent=self.root)
                elif kind == "complete":
                    result, output = payload  # type: ignore[misc]
                    self.result = result
                    self.optimizing = False
                    self.canvas.set_result(result)
                    self.timeline.configure(to=result.prediction.frame_count - 1)
                    self.timeline_var.set(0.0)
                    self.optimize_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    self.play_button.configure(state=tk.NORMAL, text="Play")
                    self.save_button.configure(state=tk.NORMAL)
                    outcome = "Feasible hit" if result.feasible else "Best trajectory is infeasible"
                    self.status_var.set(f"{outcome}; saved {output}")
        except queue.Empty:
            pass
        if self.playing and self.result is not None:
            next_frame = int(round(self.timeline_var.get())) + 1
            if next_frame >= self.result.prediction.frame_count:
                self.playing = False
                self.play_button.configure(text="Play")
            else:
                self.timeline_var.set(float(next_frame))
                self.canvas.set_trajectory_frame(next_frame)
        self.root.after(self.TICK_MS, self._tick)

    def close(self) -> None:
        self.cancel_event.set()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    WhipTrajectoryGui(root)
    root.mainloop()
