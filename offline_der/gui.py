"""Dedicated desktop UI for offline DDER extraction and identification.

The window deliberately launches the command-line program in a child process.
This keeps the UI responsive, gives cancellation real process semantics, and
ensures the graphical and command-line workflows execute identical code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFLINE_CLI = PROJECT_ROOT / "run_offline_der.py"
LIVE_APP = PROJECT_ROOT / "run_cable_twin.py"
DEFAULT_CONFIG = PROJECT_ROOT / "cable_twin" / "config.toml"
DEFAULT_REFERENCE_DIRECTORY = PROJECT_ROOT / "data" / "dder_references"
DEFAULT_MODEL = PROJECT_ROOT / "data" / "models" / "cable_dder_v1.json"

BACKGROUND = "#0d141c"
SURFACE = "#151f2b"
SURFACE_ALT = "#1c2937"
TEXT = "#eef4f8"
MUTED = "#8fa3b5"
ACCENT = "#25c2a0"
ACCENT_ACTIVE = "#3bd6b5"
WARNING = "#f4b860"
ERROR = "#ff6b6b"
PLOT = "#67d8ff"


@dataclass(frozen=True, slots=True)
class ArtifactView:
    kind: str
    title: str
    subtitle: str
    metrics: tuple[tuple[str, str], ...]
    series: np.ndarray
    plot_title: str
    plot_unit: str
    log_scale: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_ranges(sequence_ids: np.ndarray) -> Iterable[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(sequence_ids) != 0) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(sequence_ids)]))
    return zip(starts.tolist(), stops.tolist())


def _reference_view(path: Path) -> ArtifactView:
    sidecar = Path(f"{path}.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing reference sidecar: {sidecar}")
    with sidecar.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported DDER reference schema: {metadata.get('schema_version')}")
    if metadata.get("artifact_sha256") != _sha256_file(path):
        raise ValueError("Reference artifact hash does not match its sidecar")
    with np.load(path, allow_pickle=False) as values:
        centerlines = values["centerlines_m_f32"].astype(np.float64)
        velocities = values["velocities_m_s_f32"].astype(np.float64)
        observed = values["node_observed_bool"]
        sequence_ids = values["sequence_id_i32"]
        timestamps = values["timestamps_ns_i64"].astype(np.float64) * 1.0e-9
        depth_fraction = values["depth_valid_fraction_f32"]
        missing_arc = values["maximum_missing_arc_m_f32"]
        cable_id = int(values["cable_id"])
        cable_length = float(values["cable_length_m"])

    ranges = list(_sequence_ranges(sequence_ids))
    excitation = []
    duration_s = 0.0
    for start, stop in ranges:
        sequence = centerlines[start:stop]
        excitation.append(
            (sequence - np.mean(sequence, axis=0, keepdims=True)).reshape(-1) ** 2
        )
        duration_s += float(timestamps[stop - 1] - timestamps[start])
    shape_excitation_m = math.sqrt(float(np.mean(np.concatenate(excitation))))
    motion_rms_m_s = math.sqrt(float(np.mean(velocities.reshape(-1) ** 2)))
    segment_target = cable_length / (centerlines.shape[1] - 1)
    segment_lengths = np.linalg.norm(np.diff(centerlines, axis=1), axis=2)
    length_error_mm = float(np.max(np.abs(segment_lengths - segment_target))) * 1000.0
    speed_series = np.sqrt(np.mean(velocities**2, axis=(1, 2)))
    source = metadata.get("source_svo") or "source recording unavailable"
    metrics = (
        ("Retained frames", f"{centerlines.shape[0]:,}"),
        ("Sequences", str(len(ranges))),
        ("Cable / nodes", f"{cable_id} / {centerlines.shape[1]}"),
        ("Cable length", f"{cable_length:.3f} m"),
        ("Observed nodes", f"{100.0 * float(np.mean(observed)):.1f}%"),
        ("Mean valid depth", f"{100.0 * float(np.mean(depth_fraction)):.1f}%"),
        ("Motion RMS", f"{motion_rms_m_s:.4f} m/s"),
        ("Shape excitation", f"{shape_excitation_m * 1000.0:.2f} mm"),
        ("Max segment error", f"{length_error_mm:.3f} mm"),
        ("Max missing arc", f"{float(np.max(missing_arc)) * 1000.0:.1f} mm"),
        ("Retained duration", f"{duration_s:.2f} s"),
        ("Schema", str(metadata.get("schema_version", "unknown"))),
    )
    return ArtifactView(
        kind="REFERENCE",
        title=path.name,
        subtitle=str(source),
        metrics=metrics,
        series=np.asarray(speed_series, dtype=np.float64),
        plot_title="Reference motion across retained frames",
        plot_unit="RMS velocity (m/s)",
    )


def _model_view(path: Path) -> ArtifactView:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    parameters = payload.get("parameters")
    identification = payload.get("identification")
    if (
        payload.get("model_family") != "differentiable_discrete_elastic_rod"
        or not isinstance(parameters, dict)
        or not isinstance(identification, dict)
    ):
        raise ValueError("The selected JSON is not a DDER deployment model")
    history = identification.get("history", [])
    series = np.asarray(
        [float(item["evaluation_loss"]) for item in history if "evaluation_loss" in item],
        dtype=np.float64,
    )
    if series.size == 0:
        series = np.asarray(
            [float(identification.get("final_covariance_weighted_loss", 0.0))],
            dtype=np.float64,
        )
    checkpoint = identification.get("selected_checkpoint", {})
    gravity = parameters.get("gravity_camera_m_s2", (math.nan, math.nan, math.nan))
    metrics = (
        ("Bending stiffness EI", f"{float(parameters['bending_stiffness_n_m2']):.6g} N m²"),
        ("Effective damping", f"{float(parameters['velocity_damping_s_inv']):.4g} s⁻¹"),
        ("Observed RMSE", f"{float(identification['final_observed_rmse_m']) * 1000.0:.3f} mm"),
        ("Selected epoch", str(checkpoint.get("epoch", "coarse"))),
        ("References", str(identification.get("reference_count", "unknown"))),
        ("Training windows", f"{int(identification.get('training_window_count', 0)):,}"),
        ("Motion RMS", f"{float(identification.get('motion_rms_m_s', math.nan)):.4f} m/s"),
        ("Shape excitation", f"{float(identification.get('shape_excitation_rms_m', math.nan)) * 1000.0:.2f} mm"),
        ("Linear density", f"{float(parameters['linear_density_kg_m']):.5f} kg/m"),
        ("Cable / nodes", f"{float(parameters['cable_length_m']):.3f} m / {int(parameters['node_count'])}"),
        ("Gravity in camera", f"[{float(gravity[0]):.3f}, {float(gravity[1]):.3f}, {float(gravity[2]):.3f}]"),
        ("DDER solver", f"{int(parameters['substeps'])} substeps / {int(parameters['constraint_iterations'])} constraints"),
    )
    return ArtifactView(
        kind="DDER MODEL",
        title=path.name,
        subtitle=str(payload.get("equations", "equation version unavailable")),
        metrics=metrics,
        series=series,
        plot_title="Full-dataset loss during refinement",
        plot_unit="Weighted rollout loss",
        log_scale=True,
    )


def load_artifact_view(path: Path) -> ArtifactView:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".npz":
        return _reference_view(source)
    if source.suffix.lower() == ".json":
        return _model_view(source)
    raise ValueError("Select a DDER reference .npz or deployment model .json")


class SeriesCanvas(tk.Canvas):
    """Small dependency-free line plot for the two offline diagnostics."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(
            master,
            background=SURFACE_ALT,
            highlightthickness=0,
            height=190,
        )
        self._values = np.empty(0, dtype=np.float64)
        self._title = "No artifact selected"
        self._unit = ""
        self._log_scale = False
        self.bind("<Configure>", lambda _event: self._draw())

    def set_series(
        self,
        values: np.ndarray,
        *,
        title: str,
        unit: str,
        log_scale: bool,
    ) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._title = title
        self._unit = unit
        self._log_scale = log_scale
        self._draw()

    def _draw(self) -> None:
        self.delete("all")
        width = max(10, self.winfo_width())
        height = max(10, self.winfo_height())
        self.create_text(
            16,
            13,
            anchor="nw",
            text=self._title,
            fill=TEXT,
            font=("Segoe UI Semibold", 10),
        )
        left, right, top, bottom = 58, width - 18, 44, height - 30
        if right <= left or bottom <= top:
            return
        for fraction in (0.0, 0.5, 1.0):
            y = top + fraction * (bottom - top)
            self.create_line(left, y, right, y, fill="#304052", width=1)
        values = self._values[np.isfinite(self._values)]
        if values.size == 0:
            self.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="No diagnostic samples",
                fill=MUTED,
                font=("Segoe UI", 10),
            )
            return
        display = values.copy()
        if self._log_scale:
            positive = display[display > 0.0]
            floor = float(np.min(positive)) if positive.size else 1.0e-12
            display = np.log10(np.maximum(display, floor))
        low, high = float(np.min(display)), float(np.max(display))
        if math.isclose(low, high):
            low -= 0.5
            high += 0.5
        count = len(display)
        xs = np.linspace(left, right, count) if count > 1 else np.asarray([(left + right) / 2])
        ys = bottom - (display - low) / (high - low) * (bottom - top)
        if count > 1:
            points = [coordinate for xy in zip(xs.tolist(), ys.tolist()) for coordinate in xy]
            self.create_line(*points, fill=PLOT, width=2, smooth=count < 200)
        else:
            self.create_oval(xs[0] - 3, ys[0] - 3, xs[0] + 3, ys[0] + 3, fill=PLOT, outline="")
        actual_low = float(np.min(values))
        actual_high = float(np.max(values))
        self.create_text(left - 7, top, anchor="e", text=f"{actual_high:.3g}", fill=MUTED, font=("Consolas", 8))
        self.create_text(left - 7, bottom, anchor="e", text=f"{actual_low:.3g}", fill=MUTED, font=("Consolas", 8))
        suffix = " · log scale" if self._log_scale else ""
        self.create_text(left, height - 8, anchor="sw", text=f"{self._unit}{suffix}", fill=MUTED, font=("Segoe UI", 8))
        self.create_text(right, height - 8, anchor="se", text=f"{count} samples", fill=MUTED, font=("Segoe UI", 8))


class ScrollableTab(ttk.Frame):
    """Notebook page whose form remains usable on smaller displays."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, style="Card.TFrame")
        self.canvas = tk.Canvas(
            self,
            background=SURFACE,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.content = ttk.Frame(self.canvas, style="Card.TFrame", padding=16)
        self._window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(self._window, width=event.width),
        )
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.content.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _bind_wheel(self, _event: tk.Event[Any]) -> None:
        self.canvas.bind_all("<MouseWheel>", self._mouse_wheel)

    def _unbind_wheel(self, _event: tk.Event[Any]) -> None:
        self.canvas.unbind_all("<MouseWheel>")

    def _mouse_wheel(self, event: tk.Event[Any]) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


class OfflineDderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cable Twin · Offline DDER Lab")
        self.root.geometry("1420x900")
        self.root.minsize(1160, 760)
        self.root.configure(background=BACKGROUND)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self._events: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._running = False
        self._cancel_requested = False
        self._viewer_lock = threading.Lock()
        self._viewer_process: subprocess.Popen[str] | None = None
        self._viewer_starting = False
        self._viewer_running = False
        self._viewer_recording = False
        self._viewer_quit_requested = False
        self._close_after_viewer = False
        self._artifact_paths: dict[str, Path] = {}
        self._next_artifact_id = 0

        self.svo_paths: list[Path] = []
        self.reference_paths: list[Path] = []
        self._build_style()
        self._build_variables()
        self._build_layout()
        self._set_running(False)
        self._set_viewer_state(running=False, recording=False)
        self.root.after(100, self._drain_events)
        self.root.after(350, self._launch_viewer)

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=BACKGROUND, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=SURFACE)
        style.configure("Card.TLabel", background=SURFACE, foreground=TEXT)
        style.configure("Muted.Card.TLabel", background=SURFACE, foreground=MUTED)
        style.configure("Header.TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI Semibold", 19))
        style.configure("Subheader.TLabel", background=BACKGROUND, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Section.Card.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 12))
        style.configure("MetricName.Card.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 8))
        style.configure("MetricValue.Card.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 10))
        style.configure("TButton", background=SURFACE_ALT, foreground=TEXT, padding=(11, 7), borderwidth=0)
        style.map("TButton", background=[("active", "#293a4c"), ("disabled", "#17212c")], foreground=[("disabled", "#607080")])
        style.configure("Accent.TButton", background=ACCENT, foreground="#071410", padding=(13, 8), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", ACCENT_ACTIVE), ("disabled", "#24584f")], foreground=[("disabled", "#789a93")])
        style.configure("Danger.TButton", background="#4a2730", foreground="#ffd8dc")
        style.map("Danger.TButton", background=[("active", "#67313d")])
        style.configure("TEntry", fieldbackground="#0e1721", foreground=TEXT, insertcolor=TEXT, bordercolor="#334457", padding=6)
        style.configure("TCombobox", fieldbackground="#0e1721", foreground=TEXT, arrowcolor=TEXT, bordercolor="#334457", padding=5)
        style.map("TCombobox", fieldbackground=[("readonly", "#0e1721")], selectbackground=[("readonly", "#0e1721")], selectforeground=[("readonly", TEXT)])
        style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED, padding=(18, 9), font=("Segoe UI Semibold", 10))
        style.map("TNotebook.Tab", background=[("selected", SURFACE_ALT)], foreground=[("selected", TEXT)])
        style.configure("Treeview", background="#101923", fieldbackground="#101923", foreground=TEXT, rowheight=27, borderwidth=0)
        style.configure("Treeview.Heading", background=SURFACE_ALT, foreground=MUTED, relief="flat", font=("Segoe UI Semibold", 9))
        style.map("Treeview", background=[("selected", "#214f52")], foreground=[("selected", TEXT)])
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=SURFACE_ALT, borderwidth=0)
        style.configure("TCheckbutton", background=SURFACE, foreground=TEXT)

    def _build_variables(self) -> None:
        self.config_path = tk.StringVar(value=str(DEFAULT_CONFIG))
        self.reference_output_directory = tk.StringVar(value=str(DEFAULT_REFERENCE_DIRECTORY))
        self.cable_choice = tk.StringVar(value="Both cables")
        self.nodes = tk.StringVar(value="24")
        self.minimum_depth = tk.StringVar(value="0.65")
        self.maximum_missing_arc = tk.StringVar(value="0.075")
        self.projection_iterations = tk.StringVar(value="20")
        self.projection_tolerance = tk.StringVar(value="0.0002")
        self.continuity_sigma = tk.StringVar(value="0.025")
        self.minimum_sequence = tk.StringVar(value="8")

        self.model_output = tk.StringVar(value=str(DEFAULT_MODEL))
        self.cable_mass_g = tk.StringVar(value="")
        self.cable_diameter_mm = tk.StringVar(value="9.0")
        self.cable_length_text = tk.StringVar(value="Add a reference to determine length")
        self.linear_density_text = tk.StringVar(value="—")
        self.gravity_x = tk.StringVar(value="0.0")
        self.gravity_y = tk.StringVar(value="-9.80665")
        self.gravity_z = tk.StringVar(value="0.0")
        self.epochs = tk.StringVar(value="80")
        self.rollout_steps = tk.StringVar(value="5")
        self.batch_size = tk.StringVar(value="32")
        self.learning_rate = tk.StringVar(value="0.01")
        self.sigma_floor = tk.StringVar(value="0.003")
        self.minimum_excitation = tk.StringVar(value="0.005")
        self.substeps = tk.StringVar(value="2")
        self.constraint_iterations = tk.StringVar(value="8")
        self.random_seed = tk.StringVar(value="1729")
        self.status_text = tk.StringVar(value="Ready")
        self.viewer_status_text = tk.StringVar(value="Viewer not started")
        self.recording_status_text = tk.StringVar(value="Not recording")
        self.latest_recording_text = tk.StringVar(
            value="A completed SVO will be added automatically to reference extraction."
        )
        self.cable_mass_g.trace_add("write", lambda *_args: self._update_density_preview())

    def _build_layout(self) -> None:
        header = ttk.Frame(self.root, padding=(22, 16, 22, 12))
        header.pack(fill="x")
        ttk.Label(header, text="Offline DDER Lab", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Acquire a visible free-cable motion, extract reference states, then identify the DDER model",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        main = ttk.Panedwindow(self.root, orient="horizontal")
        main.pack(fill="both", expand=True, padx=22)
        workflow = ttk.Frame(main)
        dashboard = ttk.Frame(main)
        main.add(workflow, weight=3)
        main.add(dashboard, weight=2)

        self.notebook = ttk.Notebook(workflow)
        self.notebook.pack(fill="both", expand=True, padx=(0, 9))
        capture_tab = ScrollableTab(self.notebook)
        extraction_tab = ScrollableTab(self.notebook)
        fitting_tab = ScrollableTab(self.notebook)
        self.notebook.add(capture_tab, text="1  CAPTURE MOTION")
        self.notebook.add(extraction_tab, text="2  EXTRACT REFERENCES")
        self.notebook.add(fitting_tab, text="3  FIT DDER MODEL")
        self._build_capture_tab(capture_tab.content)
        self._build_extraction_tab(extraction_tab.content)
        self._build_fitting_tab(fitting_tab.content)
        self._build_dashboard(dashboard)

        bottom = ttk.Frame(self.root, padding=(22, 10, 22, 14))
        bottom.pack(fill="x")
        log_header = ttk.Frame(bottom)
        log_header.pack(fill="x", pady=(0, 5))
        ttk.Label(log_header, text="PROCESS LOG", style="Subheader.TLabel").pack(side="left")
        ttk.Button(log_header, text="Clear", command=self._clear_log).pack(side="right")
        self.log = scrolledtext.ScrolledText(
            bottom,
            height=7,
            background="#080d13",
            foreground="#cbd8e3",
            insertbackground=TEXT,
            selectbackground="#24515a",
            borderwidth=0,
            font=("Cascadia Mono", 9),
            padx=10,
            pady=8,
            state="disabled",
        )
        self.log.pack(fill="x")
        self.log.tag_configure("error", foreground=ERROR)
        self.log.tag_configure("success", foreground=ACCENT_ACTIVE)
        status = ttk.Frame(bottom)
        status.pack(fill="x", pady=(7, 0))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=180)
        self.progress.pack(side="left")
        ttk.Label(status, textvariable=self.status_text, style="Subheader.TLabel").pack(side="left", padx=10)

    def _section(self, parent: tk.Misc, title: str, description: str = "") -> ttk.Frame:
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="x", pady=(0, 14))
        ttk.Label(frame, text=title, style="Section.Card.TLabel").pack(anchor="w")
        if description:
            ttk.Label(frame, text=description, style="Muted.Card.TLabel", wraplength=720).pack(anchor="w", pady=(2, 8))
        return frame

    def _build_capture_tab(self, parent: ttk.Frame) -> None:
        live = self._section(
            parent,
            "Live 3D acquisition",
            "This launches the canonical cable-twin viewer as a separate process. "
            "It shows the registered RGB-D point cloud, PIDNet segmentation, graph "
            "edges, endpoints, and route hypotheses while you move the cable.",
        )
        state = ttk.Frame(live, style="Card.TFrame")
        state.pack(fill="x", pady=(2, 9))
        viewer = ttk.Frame(state, style="Card.TFrame")
        viewer.pack(side="left", fill="x", expand=True)
        ttk.Label(viewer, text="VIEWER", style="MetricName.Card.TLabel").pack(anchor="w")
        ttk.Label(viewer, textvariable=self.viewer_status_text, style="MetricValue.Card.TLabel").pack(anchor="w")
        recording = ttk.Frame(state, style="Card.TFrame")
        recording.pack(side="left", fill="x", expand=True, padx=(14, 0))
        ttk.Label(recording, text="LOSSLESS SOURCE", style="MetricName.Card.TLabel").pack(anchor="w")
        ttk.Label(recording, textvariable=self.recording_status_text, style="MetricValue.Card.TLabel").pack(anchor="w")

        controls = ttk.Frame(live, style="Card.TFrame")
        controls.pack(fill="x")
        self.viewer_launch_button = ttk.Button(
            controls,
            text="Launch 3D + segmentation viewer",
            style="Accent.TButton",
            command=self._launch_viewer,
        )
        self.viewer_launch_button.pack(side="left")
        self.viewer_record_button = ttk.Button(
            controls,
            text="Start recording",
            command=self._start_viewer_recording,
        )
        self.viewer_record_button.pack(side="left", padx=(8, 0))
        self.viewer_stop_button = ttk.Button(
            controls,
            text="Stop and save",
            command=self._stop_viewer_recording,
        )
        self.viewer_stop_button.pack(side="left", padx=(8, 0))
        self.viewer_close_button = ttk.Button(
            controls,
            text="Close viewer safely",
            command=self._close_viewer,
        )
        self.viewer_close_button.pack(side="right")

        experiment = self._section(
            parent,
            "Reference experiment",
            "Keep the full cable visible and free of the cube, table, frame, and the "
            "other cable. Record several smooth endpoint motions followed by free "
            "transients. The 3D window remains interactive during capture.",
        )
        ttk.Label(
            experiment,
            text="Viewer controls: left-drag orbit | right-drag pan | wheel zoom | Home refit | D depth | Q close",
            style="Muted.Card.TLabel",
            wraplength=720,
        ).pack(anchor="w")
        ttk.Label(
            experiment,
            text=(
                "The current window visualizes measured perception evidence. It does not "
                "yet display a PF posterior; that appears when the learned DDER transition "
                "is integrated into the online tracker."
            ),
            style="Muted.Card.TLabel",
            wraplength=720,
        ).pack(anchor="w", pady=(7, 0))

        result = self._section(parent, "Latest completed recording")
        ttk.Label(
            result,
            textvariable=self.latest_recording_text,
            style="MetricValue.Card.TLabel",
            wraplength=720,
        ).pack(anchor="w")
        ttk.Button(
            result,
            text="Continue to reference extraction",
            command=lambda: self.notebook.select(1),
        ).pack(anchor="w", pady=(9, 0))

    def _build_extraction_tab(self, parent: ttk.Frame) -> None:
        recordings = self._section(parent, "Source recordings", "Use lossless SVO/SVO2 recordings made by the canonical 1080p camera path.")
        self.svo_tree = ttk.Treeview(recordings, columns=("name", "path"), show="headings", height=5, selectmode="extended")
        self.svo_tree.heading("name", text="RECORDING")
        self.svo_tree.heading("path", text="PATH")
        self.svo_tree.column("name", width=210, stretch=False)
        self.svo_tree.column("path", width=460)
        self.svo_tree.pack(fill="x")
        controls = ttk.Frame(recordings, style="Card.TFrame")
        controls.pack(fill="x", pady=(7, 0))
        ttk.Button(controls, text="Add recordings", command=self._add_svos).pack(side="left")
        ttk.Button(controls, text="Remove selected", command=self._remove_svos).pack(side="left", padx=6)
        ttk.Button(controls, text="Clear", command=self._clear_svos).pack(side="left")

        destination = self._section(parent, "Extraction output")
        self._path_row(destination, "Output directory", self.reference_output_directory, self._choose_reference_directory)
        self._path_row(destination, "Runtime config", self.config_path, self._choose_config)
        cable_row = ttk.Frame(destination, style="Card.TFrame")
        cable_row.pack(fill="x", pady=(7, 0))
        ttk.Label(cable_row, text="Cable selection", style="Muted.Card.TLabel", width=18).pack(side="left")
        ttk.Combobox(cable_row, textvariable=self.cable_choice, values=("Both cables", "Cable 0", "Cable 1"), state="readonly", width=20).pack(side="left")

        advanced = self._section(parent, "Reference quality settings", "These gates reject incomplete or poorly reconstructed cable trajectories.")
        grid = ttk.Frame(advanced, style="Card.TFrame")
        grid.pack(fill="x")
        self._field(grid, 0, 0, "Centreline nodes", self.nodes)
        self._field(grid, 0, 1, "Minimum valid depth", self.minimum_depth)
        self._field(grid, 0, 2, "Maximum missing arc (m)", self.maximum_missing_arc)
        self._field(grid, 1, 0, "Minimum sequence frames", self.minimum_sequence)
        self._field(grid, 1, 1, "Projection iterations", self.projection_iterations)
        self._field(grid, 1, 2, "Projection tolerance (m)", self.projection_tolerance)
        self._field(grid, 2, 0, "Route continuity σ (m)", self.continuity_sigma)

        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.pack(fill="x", side="bottom")
        self.extract_button = ttk.Button(actions, text="Extract reference trajectories", style="Accent.TButton", command=self._start_extract)
        self.extract_button.pack(side="left")
        self.extract_cancel_button = ttk.Button(actions, text="Cancel", style="Danger.TButton", command=self._cancel)
        self.extract_cancel_button.pack(side="left", padx=8)

    def _build_fitting_tab(self, parent: ttk.Frame) -> None:
        references = self._section(parent, "Reference trajectories", "Combine recordings only when cable material, diameter, length, and mass per metre are identical.")
        self.reference_tree = ttk.Treeview(references, columns=("name", "path"), show="headings", height=4, selectmode="extended")
        self.reference_tree.heading("name", text="REFERENCE")
        self.reference_tree.heading("path", text="PATH")
        self.reference_tree.column("name", width=230, stretch=False)
        self.reference_tree.column("path", width=440)
        self.reference_tree.pack(fill="x")
        controls = ttk.Frame(references, style="Card.TFrame")
        controls.pack(fill="x", pady=(7, 0))
        ttk.Button(controls, text="Add references", command=self._add_references).pack(side="left")
        ttk.Button(controls, text="Remove selected", command=self._remove_references).pack(side="left", padx=6)
        ttk.Button(controls, text="Clear", command=self._clear_references).pack(side="left")

        physical = self._section(parent, "Measured physical inputs", "Mass and camera-frame gravity are measured inputs; the optimizer fits only EI and effective damping.")
        grid = ttk.Frame(physical, style="Card.TFrame")
        grid.pack(fill="x")
        self._field(grid, 0, 0, "Cable mass (g)", self.cable_mass_g)
        self._field(grid, 0, 1, "Cable diameter (mm)", self.cable_diameter_mm)
        preview = ttk.Frame(grid, style="Card.TFrame")
        preview.grid(row=0, column=2, sticky="ew", padx=(8, 0), pady=3)
        ttk.Label(preview, text="REFERENCE LENGTH", style="MetricName.Card.TLabel").pack(anchor="w")
        ttk.Label(preview, textvariable=self.cable_length_text, style="MetricValue.Card.TLabel").pack(anchor="w")
        density = ttk.Frame(grid, style="Card.TFrame")
        density.grid(row=1, column=2, sticky="ew", padx=(8, 0), pady=3)
        ttk.Label(density, text="COMPUTED LINEAR DENSITY", style="MetricName.Card.TLabel").pack(anchor="w")
        ttk.Label(density, textvariable=self.linear_density_text, style="MetricValue.Card.TLabel").pack(anchor="w")
        self._field(grid, 1, 0, "Gravity X (m/s²)", self.gravity_x)
        self._field(grid, 1, 1, "Gravity Y (m/s²)", self.gravity_y)
        self._field(grid, 2, 0, "Gravity Z (m/s²)", self.gravity_z)
        ttk.Button(grid, text="Level-camera gravity", command=self._set_level_gravity).grid(row=2, column=1, sticky="w", padx=8, pady=4)
        for column in range(3):
            grid.columnconfigure(column, weight=1)

        destination = self._section(parent, "Deployment model")
        self._path_row(destination, "Output JSON", self.model_output, self._choose_model_output)

        numerical = self._section(parent, "Identification settings", "The bounded coarse physical search and best-checkpoint selection remain enabled.")
        grid = ttk.Frame(numerical, style="Card.TFrame")
        grid.pack(fill="x")
        self._field(grid, 0, 0, "Refinement epochs", self.epochs)
        self._field(grid, 0, 1, "Rollout frames", self.rollout_steps)
        self._field(grid, 0, 2, "Batch size", self.batch_size)
        self._field(grid, 1, 0, "Learning rate", self.learning_rate)
        self._field(grid, 1, 1, "Observation σ floor (m)", self.sigma_floor)
        self._field(grid, 1, 2, "Minimum excitation (m)", self.minimum_excitation)
        self._field(grid, 2, 0, "DDER substeps", self.substeps)
        self._field(grid, 2, 1, "Constraint iterations", self.constraint_iterations)
        self._field(grid, 2, 2, "Random seed", self.random_seed)

        actions = ttk.Frame(parent, style="Card.TFrame")
        actions.pack(fill="x", side="bottom")
        self.fit_button = ttk.Button(actions, text="Identify and save DDER model", style="Accent.TButton", command=self._start_fit)
        self.fit_button.pack(side="left")
        self.fit_cancel_button = ttk.Button(actions, text="Cancel", style="Danger.TButton", command=self._cancel)
        self.fit_cancel_button.pack(side="left", padx=8)

    def _build_dashboard(self, parent: ttk.Frame) -> None:
        card = ttk.Frame(parent, style="Card.TFrame", padding=15)
        card.pack(fill="both", expand=True, padx=(9, 0))
        heading = ttk.Frame(card, style="Card.TFrame")
        heading.pack(fill="x")
        ttk.Label(heading, text="Artifact review", style="Section.Card.TLabel").pack(side="left")
        ttk.Button(heading, text="Open artifact", command=self._choose_artifact).pack(side="right")
        ttk.Button(heading, text="Open folder", command=self._open_artifact_folder).pack(side="right", padx=6)
        self.artifact_tree = ttk.Treeview(card, columns=("kind", "name"), show="headings", height=4, selectmode="browse")
        self.artifact_tree.heading("kind", text="TYPE")
        self.artifact_tree.heading("name", text="ARTIFACT")
        self.artifact_tree.column("kind", width=100, stretch=False)
        self.artifact_tree.column("name", width=330)
        self.artifact_tree.pack(fill="x", pady=(10, 13))
        self.artifact_tree.bind("<<TreeviewSelect>>", self._artifact_selected)

        self.artifact_kind = ttk.Label(card, text="NO ARTIFACT", style="Muted.Card.TLabel")
        self.artifact_kind.pack(anchor="w")
        self.artifact_title = ttk.Label(card, text="Run extraction or open an existing artifact", style="Section.Card.TLabel", wraplength=500)
        self.artifact_title.pack(anchor="w", pady=(2, 0))
        self.artifact_subtitle = ttk.Label(card, text="Reference quality and fitted model information will appear here.", style="Muted.Card.TLabel", wraplength=500)
        self.artifact_subtitle.pack(anchor="w", pady=(2, 12))
        self.metrics_frame = ttk.Frame(card, style="Card.TFrame")
        self.metrics_frame.pack(fill="x")
        self.plot = SeriesCanvas(card)
        self.plot.pack(fill="both", expand=True, pady=(14, 0))

    def _path_row(self, parent: tk.Misc, label: str, variable: tk.StringVar, command: Any) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="Muted.Card.TLabel", width=18).pack(side="left")
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", command=command).pack(side="left", padx=(7, 0))

    def _field(self, parent: tk.Misc, row: int, column: int, label: str, variable: tk.StringVar) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row, column=column, sticky="ew", padx=(0 if column == 0 else 8, 0), pady=3)
        ttk.Label(frame, text=label.upper(), style="MetricName.Card.TLabel").pack(anchor="w")
        ttk.Entry(frame, textvariable=variable, width=18).pack(fill="x", pady=(2, 0))
        parent.columnconfigure(column, weight=1)

    def _viewer_is_active(self) -> bool:
        return self._viewer_starting or self._viewer_running

    def _launch_viewer(self) -> None:
        if self._running:
            messagebox.showwarning(
                "Offline process active",
                "Wait for extraction or fitting to finish before opening the live camera.",
                parent=self.root,
            )
            return
        if self._viewer_is_active():
            self.viewer_status_text.set("Live viewer already running")
            return
        config = Path(self.config_path.get()).expanduser().resolve()
        if not config.is_file():
            messagebox.showerror("Cannot launch viewer", f"Configuration not found: {config}", parent=self.root)
            return

        command = [
            sys.executable,
            "-u",
            str(LIVE_APP),
            "--config",
            str(config),
            "--control-stdin",
        ]
        self._viewer_starting = True
        self._viewer_quit_requested = False
        self.viewer_status_text.set("Starting live camera and PIDNet...")
        self.recording_status_text.set("Not recording")
        self._refresh_viewer_buttons()
        self._append_log(f"\n> {subprocess.list2cmdline(command)}\n")
        threading.Thread(
            target=self._run_viewer_process,
            args=(command,),
            name="offline-der-live-viewer",
            daemon=True,
        ).start()

    def _run_viewer_process(self, command: list[str]) -> None:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creationflags,
            )
            with self._viewer_lock:
                self._viewer_process = process
            self._events.put(("viewer_started",))
            assert process.stdout is not None
            for line in process.stdout:
                self._events.put(("viewer_line", line.rstrip("\r\n")))
            self._events.put(("viewer_done", process.wait()))
        except Exception as error:
            self._events.put(("viewer_line", f"Live viewer worker failed: {error}"))
            self._events.put(("viewer_done", -1))
        finally:
            if process is not None and process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            with self._viewer_lock:
                self._viewer_process = None

    def _send_viewer_command(self, command: str) -> bool:
        with self._viewer_lock:
            process = self._viewer_process
            if process is None or process.poll() is not None or process.stdin is None:
                process = None
            else:
                try:
                    process.stdin.write(command + "\n")
                    process.stdin.flush()
                    return True
                except (BrokenPipeError, OSError) as error:
                    self._events.put(("viewer_line", f"Viewer control failed: {error}"))
        return False

    def _start_viewer_recording(self) -> None:
        if self._send_viewer_command("record_start"):
            self.recording_status_text.set("Starting lossless SVO...")
            self.viewer_record_button.configure(state="disabled")

    def _stop_viewer_recording(self) -> None:
        if self._send_viewer_command("record_stop"):
            self.recording_status_text.set("Finalizing SVO and manifest...")
            self.viewer_stop_button.configure(state="disabled")

    def _close_viewer(self) -> None:
        if not self._viewer_is_active():
            return
        self._viewer_quit_requested = True
        self.viewer_status_text.set("Closing safely...")
        if self._viewer_recording:
            self.recording_status_text.set("Finalizing SVO and manifest...")
        self._refresh_viewer_buttons(closing=True)
        if self._viewer_running:
            self._send_viewer_command("quit")

    def _set_viewer_state(self, *, running: bool, recording: bool) -> None:
        self._viewer_starting = False
        self._viewer_running = running
        self._viewer_recording = recording if running else False
        if running:
            self.viewer_status_text.set("Live 3D + segmentation viewer running")
            self.recording_status_text.set("Recording" if recording else "Not recording")
        elif not self._close_after_viewer:
            self.viewer_status_text.set("Viewer not running")
            self.recording_status_text.set("Not recording")
        self._refresh_viewer_buttons()

    def _refresh_viewer_buttons(self, *, closing: bool = False) -> None:
        if not hasattr(self, "viewer_launch_button"):
            return
        busy = self._running or closing
        active = self._viewer_running and not busy
        self.viewer_launch_button.configure(
            state="normal" if not busy and not self._viewer_is_active() else "disabled"
        )
        self.viewer_record_button.configure(
            state="normal" if active and not self._viewer_recording else "disabled"
        )
        self.viewer_stop_button.configure(
            state="normal" if active and self._viewer_recording else "disabled"
        )
        self.viewer_close_button.configure(
            state="normal" if self._viewer_is_active() and not closing else "disabled"
        )

    def _add_svo_path(self, value: Path) -> None:
        path = Path(value).expanduser().resolve()
        if path in self.svo_paths:
            return
        self.svo_paths.append(path)
        manifest = Path(f"{path}.json")
        status = str(path) if manifest.is_file() else f"{path}  [manifest missing]"
        self.svo_tree.insert("", "end", values=(path.name, status))

    def _add_svos(self) -> None:
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Add lossless ZED recordings",
            filetypes=(("ZED recordings", "*.svo *.svo2"), ("All files", "*.*")),
        )
        for value in selected:
            self._add_svo_path(Path(value))

    def _remove_svos(self) -> None:
        indices = sorted((self.svo_tree.index(item) for item in self.svo_tree.selection()), reverse=True)
        for index in indices:
            self.svo_paths.pop(index)
            self.svo_tree.delete(self.svo_tree.get_children()[index])

    def _clear_svos(self) -> None:
        self.svo_paths.clear()
        self.svo_tree.delete(*self.svo_tree.get_children())

    def _add_references(self) -> None:
        selected = filedialog.askopenfilenames(
            parent=self.root,
            title="Add DDER reference trajectories",
            filetypes=(("DDER references", "*.npz"),),
        )
        for value in selected:
            path = Path(value).resolve()
            if path not in self.reference_paths:
                self.reference_paths.append(path)
                self.reference_tree.insert("", "end", values=(path.name, str(path)))
                self._add_artifact(path, "Reference")
        self._update_density_preview()

    def _remove_references(self) -> None:
        indices = sorted((self.reference_tree.index(item) for item in self.reference_tree.selection()), reverse=True)
        for index in indices:
            self.reference_paths.pop(index)
            self.reference_tree.delete(self.reference_tree.get_children()[index])
        self._update_density_preview()

    def _clear_references(self) -> None:
        self.reference_paths.clear()
        self.reference_tree.delete(*self.reference_tree.get_children())
        self._update_density_preview()

    def _choose_reference_directory(self) -> None:
        value = filedialog.askdirectory(parent=self.root, initialdir=self.reference_output_directory.get())
        if value:
            self.reference_output_directory.set(value)

    def _choose_config(self) -> None:
        value = filedialog.askopenfilename(parent=self.root, filetypes=(("TOML configuration", "*.toml"),))
        if value:
            self.config_path.set(value)

    def _choose_model_output(self) -> None:
        value = filedialog.asksaveasfilename(
            parent=self.root,
            initialdir=str(Path(self.model_output.get()).parent),
            initialfile=Path(self.model_output.get()).name,
            defaultextension=".json",
            filetypes=(("DDER model", "*.json"),),
        )
        if value:
            self.model_output.set(value)

    def _choose_artifact(self) -> None:
        value = filedialog.askopenfilename(
            parent=self.root,
            title="Open offline DDER artifact",
            filetypes=(("DDER artifacts", "*.npz *.json"), ("DDER reference", "*.npz"), ("DDER model", "*.json")),
        )
        if value:
            path = Path(value).resolve()
            self._add_artifact(path, "Reference" if path.suffix.lower() == ".npz" else "Model")

    def _set_level_gravity(self) -> None:
        self.gravity_x.set("0.0")
        self.gravity_y.set("-9.80665")
        self.gravity_z.set("0.0")

    def _reference_length(self) -> float | None:
        if not self.reference_paths:
            return None
        lengths = []
        for path in self.reference_paths:
            with np.load(path, allow_pickle=False) as values:
                lengths.append(float(values["cable_length_m"]))
        if not all(math.isclose(length, lengths[0], abs_tol=1.0e-9) for length in lengths):
            raise ValueError("Selected references have different cable lengths")
        return lengths[0]

    def _update_density_preview(self) -> None:
        try:
            length = self._reference_length()
            if length is None:
                self.cable_length_text.set("Add a reference to determine length")
                self.linear_density_text.set("—")
                return
            self.cable_length_text.set(f"{length:.3f} m")
            mass_g = float(self.cable_mass_g.get())
            if not math.isfinite(mass_g) or mass_g <= 0.0:
                self.linear_density_text.set("Enter measured mass")
                return
            density = mass_g * 1.0e-3 / length
            self.linear_density_text.set(f"{density:.6f} kg/m")
        except (OSError, KeyError, ValueError) as error:
            self.cable_length_text.set("Reference mismatch")
            self.linear_density_text.set(str(error))

    @staticmethod
    def _positive_float(variable: tk.StringVar, name: str) -> float:
        value = float(variable.get())
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _positive_int(variable: tk.StringVar, name: str) -> int:
        value = int(variable.get())
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _finite_float(variable: tk.StringVar, name: str) -> float:
        value = float(variable.get())
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    def _start_extract(self) -> None:
        if self._viewer_is_active():
            messagebox.showwarning(
                "Close live acquisition first",
                "Stop and save the recording, then close the live viewer safely before extraction.",
                parent=self.root,
            )
            self.notebook.select(0)
            return
        try:
            if not self.svo_paths:
                raise ValueError("Add at least one SVO recording")
            for path in self.svo_paths:
                if not path.is_file():
                    raise FileNotFoundError(path)
                if not Path(f"{path}.json").is_file():
                    raise FileNotFoundError(f"Missing recording manifest: {path}.json")
            config = Path(self.config_path.get()).expanduser().resolve()
            if not config.is_file():
                raise FileNotFoundError(config)
            output = Path(self.reference_output_directory.get()).expanduser().resolve()
            node_count = self._positive_int(self.nodes, "Centreline nodes")
            minimum_depth = self._positive_float(self.minimum_depth, "Minimum valid depth")
            if minimum_depth > 1.0:
                raise ValueError("Minimum valid depth must not exceed 1")
            maximum_missing = self._positive_float(self.maximum_missing_arc, "Maximum missing arc")
            projection_iterations = self._positive_int(self.projection_iterations, "Projection iterations")
            projection_tolerance = self._positive_float(self.projection_tolerance, "Projection tolerance")
            continuity_sigma = self._positive_float(self.continuity_sigma, "Route continuity sigma")
            minimum_sequence = self._positive_int(self.minimum_sequence, "Minimum sequence frames")
            cable = {"Both cables": "both", "Cable 0": "0", "Cable 1": "1"}[self.cable_choice.get()]
        except (OSError, KeyError, ValueError) as error:
            messagebox.showerror("Cannot start extraction", str(error), parent=self.root)
            return
        command = [
            sys.executable,
            "-u",
            str(OFFLINE_CLI),
            "extract",
            "--svo",
            *[str(path) for path in self.svo_paths],
            "--output-directory",
            str(output),
            "--config",
            str(config),
            "--cable",
            cable,
            "--nodes",
            str(node_count),
            "--minimum-depth-valid-fraction",
            str(minimum_depth),
            "--maximum-missing-arc-m",
            str(maximum_missing),
            "--projection-iterations",
            str(projection_iterations),
            "--projection-tolerance-m",
            str(projection_tolerance),
            "--route-continuity-sigma-m",
            str(continuity_sigma),
            "--minimum-sequence-frames",
            str(minimum_sequence),
        ]
        self._launch(command, "Extracting reference trajectories")

    def _start_fit(self) -> None:
        if self._viewer_is_active():
            messagebox.showwarning(
                "Close live acquisition first",
                "Close the live viewer safely before DDER identification so CUDA resources are dedicated to fitting.",
                parent=self.root,
            )
            self.notebook.select(0)
            return
        try:
            if not self.reference_paths:
                raise ValueError("Add at least one DDER reference")
            for path in self.reference_paths:
                if not path.is_file() or not Path(f"{path}.json").is_file():
                    raise FileNotFoundError(f"Reference or sidecar is missing: {path}")
            length = self._reference_length()
            if length is None:
                raise ValueError("Reference length is unavailable")
            mass_g = self._positive_float(self.cable_mass_g, "Cable mass")
            density = mass_g * 1.0e-3 / length
            diameter_mm = self._positive_float(self.cable_diameter_mm, "Cable diameter")
            gravity = (
                self._finite_float(self.gravity_x, "Gravity X"),
                self._finite_float(self.gravity_y, "Gravity Y"),
                self._finite_float(self.gravity_z, "Gravity Z"),
            )
            if math.sqrt(sum(value * value for value in gravity)) < 9.0:
                raise ValueError("Camera-frame gravity magnitude is implausibly small")
            output = Path(self.model_output.get()).expanduser().resolve()
            if output.suffix.lower() != ".json":
                raise ValueError("The deployment model must use the .json suffix")
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}")
            epochs = self._positive_int(self.epochs, "Refinement epochs")
            rollout = self._positive_int(self.rollout_steps, "Rollout frames")
            batch = self._positive_int(self.batch_size, "Batch size")
            learning_rate = self._positive_float(self.learning_rate, "Learning rate")
            sigma_floor = self._positive_float(self.sigma_floor, "Observation sigma floor")
            minimum_excitation = self._positive_float(self.minimum_excitation, "Minimum excitation")
            substeps = self._positive_int(self.substeps, "DDER substeps")
            constraints = self._positive_int(self.constraint_iterations, "Constraint iterations")
            seed = int(self.random_seed.get())
        except (OSError, ValueError) as error:
            messagebox.showerror("Cannot start identification", str(error), parent=self.root)
            return
        command = [
            sys.executable,
            "-u",
            str(OFFLINE_CLI),
            "fit",
            "--reference",
            *[str(path) for path in self.reference_paths],
            "--output",
            str(output),
            "--linear-density-kg-m",
            f"{density:.12g}",
            "--gravity-camera-m-s2",
            *(f"{value:.12g}" for value in gravity),
            "--cable-radius-m",
            f"{diameter_mm * 0.0005:.12g}",
            "--epochs",
            str(epochs),
            "--rollout-steps",
            str(rollout),
            "--batch-size",
            str(batch),
            "--learning-rate",
            str(learning_rate),
            "--observation-sigma-floor-m",
            str(sigma_floor),
            "--minimum-shape-excitation-rms-m",
            str(minimum_excitation),
            "--substeps",
            str(substeps),
            "--constraint-iterations",
            str(constraints),
            "--seed",
            str(seed),
        ]
        self._launch(command, "Identifying DDER parameters")

    def _launch(self, command: list[str], status: str) -> None:
        if self._viewer_is_active():
            messagebox.showwarning(
                "Live viewer active",
                "Close the live viewer safely before starting offline CUDA work.",
                parent=self.root,
            )
            return
        with self._process_lock:
            if self._process is not None and self._process.poll() is None:
                messagebox.showwarning("Offline process active", "Cancel or wait for the current process.", parent=self.root)
                return
        self._cancel_requested = False
        self._set_running(True)
        self.status_text.set(status)
        self._append_log(f"\n> {subprocess.list2cmdline(command)}\n")
        thread = threading.Thread(target=self._run_process, args=(command,), daemon=True)
        thread.start()

    def _run_process(self, command: list[str]) -> None:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creationflags,
            )
            with self._process_lock:
                self._process = process
                cancel_immediately = self._cancel_requested
            if cancel_immediately:
                process.terminate()
            assert process.stdout is not None
            for line in process.stdout:
                self._events.put(("line", line.rstrip("\r\n")))
            return_code = process.wait()
            self._events.put(("done", return_code, self._cancel_requested))
        except Exception as error:  # subprocess boundary: report exact failure in UI
            self._events.put(("line", f"UI worker failed: {error}"))
            self._events.put(("done", -1, self._cancel_requested))
        finally:
            with self._process_lock:
                self._process = None

    def _cancel(self) -> None:
        with self._process_lock:
            if not self._running:
                return
            self._cancel_requested = True
            process = self._process
            if process is None or process.poll() is not None:
                process = None
            else:
                process.terminate()
        self.status_text.set("Cancelling…")
        self._append_log("Cancellation requested.\n", "error")

    def _drain_events(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                if event[0] == "line":
                    self._handle_process_line(str(event[1]))
                elif event[0] == "done":
                    self._process_done(int(event[1]), bool(event[2]))
                elif event[0] == "viewer_started":
                    self._set_viewer_state(running=True, recording=False)
                    if self._viewer_quit_requested:
                        self._close_viewer()
                elif event[0] == "viewer_line":
                    self._handle_viewer_line(str(event[1]))
                elif event[0] == "viewer_done":
                    self._viewer_done(int(event[1]))
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _handle_viewer_line(self, line: str) -> None:
        if not line.startswith("frame="):
            tag = "error" if "error" in line.lower() or "traceback" in line.lower() else None
            self._append_log(f"[viewer] {line}\n", tag)
        if line.startswith("PIDNet ready"):
            self.viewer_status_text.set("Live 3D + segmentation viewer running")
        elif line.startswith("Recording lossless SVO: "):
            path = Path(line.removeprefix("Recording lossless SVO: ")).expanduser().resolve()
            self._viewer_recording = True
            self.recording_status_text.set(f"Recording {path.name}")
            self.latest_recording_text.set(f"Recording in progress: {path}")
            self._refresh_viewer_buttons()
        elif line.startswith("Saved lossless SVO and manifest: "):
            path_text = line.removeprefix("Saved lossless SVO and manifest: ").strip()
            if path_text and path_text != "None":
                path = Path(path_text).expanduser().resolve()
                self._add_svo_path(path)
                self.latest_recording_text.set(str(path))
            self._viewer_recording = False
            self.recording_status_text.set("Saved and ready for extraction")
            self._refresh_viewer_buttons()

    def _viewer_done(self, return_code: int) -> None:
        closing_application = self._close_after_viewer
        self._set_viewer_state(running=False, recording=False)
        self._viewer_quit_requested = False
        if closing_application:
            self.root.after(10, self.root.destroy)
            return
        if return_code == 0:
            self.viewer_status_text.set("Viewer closed")
        else:
            self.viewer_status_text.set(f"Viewer failed (exit code {return_code})")
            self._append_log(f"Live viewer exited with code {return_code}.\n", "error")

    def _handle_process_line(self, line: str) -> None:
        tag = "error" if "error" in line.lower() or "traceback" in line.lower() else None
        self._append_log(line + "\n", tag)
        frame_match = re.search(r"extract frame=(\d+)", line)
        if frame_match:
            self.status_text.set(f"Extracting · source frame {frame_match.group(1)}")
        saved_reference = re.match(r"saved (.+?\.npz) frames=", line)
        if saved_reference:
            self._add_artifact(Path(saved_reference.group(1)), "Reference")
        saved_model = re.match(r"saved deployment DDER model: (.+\.json)$", line)
        if saved_model:
            self._add_artifact(Path(saved_model.group(1)), "Model")

    def _process_done(self, return_code: int, cancelled: bool) -> None:
        self._set_running(False)
        if cancelled:
            self.status_text.set("Cancelled")
            self._append_log("Process cancelled.\n", "error")
        elif return_code == 0:
            self.status_text.set("Completed successfully")
            self._append_log("Process completed successfully.\n", "success")
        else:
            self.status_text.set(f"Failed · exit code {return_code}")
            self._append_log(f"Process failed with exit code {return_code}.\n", "error")

    def _set_running(self, running: bool) -> None:
        self._running = running
        state = "disabled" if running else "normal"
        cancel_state = "normal" if running else "disabled"
        if hasattr(self, "extract_button"):
            self.extract_button.configure(state=state)
            self.fit_button.configure(state=state)
            self.extract_cancel_button.configure(state=cancel_state)
            self.fit_cancel_button.configure(state=cancel_state)
        if running:
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(value=0)
        self._refresh_viewer_buttons()

    def _add_artifact(self, path: Path, kind: str) -> None:
        source = Path(path).expanduser().resolve()
        for item_id, existing in self._artifact_paths.items():
            if existing == source:
                self.artifact_tree.selection_set(item_id)
                self.artifact_tree.see(item_id)
                self._show_artifact(source)
                return
        item_id = f"artifact_{self._next_artifact_id}"
        self._next_artifact_id += 1
        self._artifact_paths[item_id] = source
        self.artifact_tree.insert("", "end", iid=item_id, values=(kind.upper(), source.name))
        self.artifact_tree.selection_set(item_id)
        self.artifact_tree.see(item_id)
        self._show_artifact(source)

    def _artifact_selected(self, _event: tk.Event[Any]) -> None:
        selected = self.artifact_tree.selection()
        if selected:
            self._show_artifact(self._artifact_paths[selected[0]])

    def _show_artifact(self, path: Path) -> None:
        try:
            view = load_artifact_view(path)
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            self.artifact_kind.configure(text="ARTIFACT ERROR")
            self.artifact_title.configure(text=path.name)
            self.artifact_subtitle.configure(text=str(error))
            self._render_metrics(())
            self.plot.set_series(np.empty(0), title="Artifact could not be read", unit="", log_scale=False)
            return
        self.artifact_kind.configure(text=view.kind)
        self.artifact_title.configure(text=view.title)
        self.artifact_subtitle.configure(text=view.subtitle)
        self._render_metrics(view.metrics)
        self.plot.set_series(view.series, title=view.plot_title, unit=view.plot_unit, log_scale=view.log_scale)

    def _render_metrics(self, metrics: tuple[tuple[str, str], ...]) -> None:
        for child in self.metrics_frame.winfo_children():
            child.destroy()
        for index, (name, value) in enumerate(metrics):
            card = ttk.Frame(self.metrics_frame, style="Card.TFrame")
            card.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0 if index % 2 == 0 else 10, 0), pady=3)
            ttk.Label(card, text=name.upper(), style="MetricName.Card.TLabel").pack(anchor="w")
            ttk.Label(card, text=value, style="MetricValue.Card.TLabel", wraplength=220).pack(anchor="w")
        self.metrics_frame.columnconfigure(0, weight=1)
        self.metrics_frame.columnconfigure(1, weight=1)

    def _open_artifact_folder(self) -> None:
        selected = self.artifact_tree.selection()
        if not selected:
            return
        directory = self._artifact_paths[selected[0]].parent
        if os.name == "nt":
            os.startfile(directory)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(directory)])

    def _append_log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _close(self) -> None:
        with self._process_lock:
            offline_active = self._running
        viewer_active = self._viewer_is_active()
        if offline_active or viewer_active:
            activities = []
            if viewer_active:
                activities.append("the live viewer")
            if offline_active:
                activities.append("the offline extraction/fit")
            close = messagebox.askyesno(
                "Close Offline DDER Lab?",
                f"Stop {' and '.join(activities)} and close? "
                "Any active SVO recording will be finalized first.",
                parent=self.root,
            )
            if not close:
                return
        if offline_active:
            self._cancel()
        if viewer_active:
            self._close_after_viewer = True
            self._close_viewer()
            return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    OfflineDderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
