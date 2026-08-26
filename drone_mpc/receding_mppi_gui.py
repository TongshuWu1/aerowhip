"""Research UI for full-state receding-horizon DDER-MPPI.

The window deliberately separates the strike definition from controller
compute settings.  The validated MPPI task weights and safety logic are shown
read-only; this UI is for running the controller, not retuning its objective.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import secrets
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch

from optitrack_offline.config import DEFAULT_MODEL_PATH

from .model import CableModelSnapshot, load_cable_model
from .mpc import MpcProblem
from .mppi import MppiSettings
from .online_adaptation import (
    BetweenStrikeAdaptationResult,
    BetweenStrikeAdaptationSession,
)
from .reduced import build_controller_and_truth_models
from .receding_mppi import (
    RecedingMppiExecution,
    RecedingMppiLiveUpdate,
    RecedingMppiSettings,
    run_receding_horizon_mppi,
    save_receding_mppi_execution,
)
from .simulator import SimulationResult, SimulationSettings, WhipSimulator
from .trajectory_gui import WhipTrajectoryCanvas


DEFAULT_WARM_START_PATH = Path(
    "data/drone_mpc/ablations/mppi_discovery_pilot/"
    "initialization_forward_recoil__T2__k11__vg0_12__pw0__"
    "pilot__i6__n128__b128__seed17.npz"
)
DEFAULT_OUTPUT_PATH = Path("data/drone_mpc/receding_mppi/latest_execution.npz")
DEFAULT_SETTINGS_PROFILE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "drone_mpc" / "settings_profiles"
)
MAXIMUM_RUN_SEED = 2**31 - 1
SETTINGS_PROFILE_SCHEMA = "receding_horizon_dder_mppi_settings_v1"

SETTINGS_PROFILE_FIELDS = {
    "inputs": ("model_path", "warm_start_path"),
    "task": (
        "initial_drone_xyz",
        "target_xyz",
        "impact_direction",
        "minimum_directed_speed_m_s",
        "target_radius_m",
        "direction_half_angle_deg",
    ),
    "controller": (
        "horizon_s",
        "physics_rate_hz",
        "control_rate_hz",
        "simulation_nodes",
        "replanning_rate_hz",
        "execution_timeout_s",
        "maximum_acceleration_m_s2",
        "maximum_speed_m_s",
        "samples",
        "iterations",
        "acceleration_knots",
        "noise_sigma_m_s2",
        "noise_decay",
        "temperature",
        "seed",
    ),
    "plant_truth": ("ei_scale", "cb_scale"),
    "adaptation": ("enabled",),
}


FIXED_OBJECTIVE = {
    "objective_stage": "full",
    "position_sigma_m": 0.18,
    "velocity_gate_sigma_m": 0.12,
    "position_weight": 40.0,
    "speed_weight": 25.0,
    "direction_weight": 20.0,
    "success_cost": 600.0,
    "drone_displacement_weight": 2.0,
    "safety_weight": 180.0,
    "control_effort_weight": 1.0e-5,
    "control_smoothness_weight": 1.0e-5,
}


def normalize_settings_profile(payload: object) -> dict[str, dict[str, str]]:
    """Validate a complete profile before any Tk variables are changed."""

    if not isinstance(payload, dict):
        raise ValueError("Settings profile must be a JSON object.")
    if payload.get("schema") != SETTINGS_PROFILE_SCHEMA:
        raise ValueError(
            "Unsupported settings profile schema. Expected "
            f"'{SETTINGS_PROFILE_SCHEMA}'."
        )
    if payload.get("fixed_objective") != FIXED_OBJECTIVE:
        raise ValueError(
            "The profile's fixed strike objective does not match this controller "
            "version. Refusing to load a scientifically different experiment."
        )
    normalized: dict[str, dict[str, str]] = {}
    for section, fields in SETTINGS_PROFILE_FIELDS.items():
        values = payload.get(section)
        if section == "adaptation" and values is None:
            # Profiles saved before online adaptation remain valid and adopt
            # the new safe default: between-strike adaptation enabled.
            values = {"enabled": "true"}
        if not isinstance(values, dict):
            raise ValueError(f"Settings profile is missing section '{section}'.")
        normalized[section] = {}
        for field in fields:
            value = values.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Settings profile field '{section}.{field}' must be a "
                    "non-empty string."
                )
            normalized[section][field] = value
    return normalized


def resolve_run_seed(requested_seed: int) -> int:
    """Resolve UI seed zero to a fresh, saved, reproducible run seed."""

    if requested_seed < 0:
        raise ValueError("Random seed must be zero or a positive integer.")
    if requested_seed == 0:
        return secrets.randbelow(MAXIMUM_RUN_SEED) + 1
    return requested_seed


@dataclass(frozen=True, slots=True)
class ExecutionView:
    """Adapter for the shared 3-D trajectory canvas."""

    prediction: SimulationResult
    feasible: bool
    impact_frame: int
    terms: dict[str, float]
    speed_amplification: float


@dataclass(frozen=True, slots=True)
class TruthModelSettings:
    """Hidden simulated-plant parameters relative to the fitted model."""

    bending_stiffness_scale: float = 1.0
    bending_damping_scale: float = 1.0

    def __post_init__(self) -> None:
        values = (self.bending_stiffness_scale, self.bending_damping_scale)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Truth EI and Cb scales must be finite and positive.")


def model_pair_provenance(
    source: CableModelSnapshot,
    controller: CableModelSnapshot,
    plant: CableModelSnapshot,
    truth_settings: TruthModelSettings,
) -> dict[str, object]:
    """Return explicit, serializable provenance for a mismatch experiment."""

    def describe(snapshot: CableModelSnapshot) -> dict[str, object]:
        return {
            "sha256": snapshot.sha256,
            "node_count": snapshot.node_count,
            "bending_stiffness_n_m2": snapshot.bending_stiffness_n_m2,
            "bending_damping_n_m2_s": snapshot.bending_damping_n_m2_s,
            "substeps": snapshot.model.parameters.substeps,
            "constraint_iterations": snapshot.model.parameters.constraint_iterations,
        }

    return {
        "source_fitted_model_path": str(source.source_path),
        "source_fitted_model_sha256": source.sha256,
        "controller": describe(controller),
        "plant_truth": describe(plant),
        "truth_scales": {
            "bending_stiffness": truth_settings.bending_stiffness_scale,
            "bending_damping": truth_settings.bending_damping_scale,
        },
        "matched": controller.sha256 == plant.sha256,
        "controlled_mismatch": "EI and Cb only",
    }


def _resample_knots(knots: np.ndarray, knot_count: int) -> np.ndarray:
    values = np.asarray(knots, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 3:
        raise ValueError("Warm-start controls must have shape Mx3 with M >= 2.")
    if knot_count < 2:
        raise ValueError("MPPI requires at least two acceleration knots.")
    if len(values) == knot_count:
        return values.copy()
    old_coordinate = np.linspace(0.0, 1.0, len(values))
    new_coordinate = np.linspace(0.0, 1.0, knot_count)
    return np.stack(
        [
            np.interp(new_coordinate, old_coordinate, values[:, axis])
            for axis in range(3)
        ],
        axis=1,
    ).astype(np.float32)


def load_warm_start_knots(path: str | Path, knot_count: int) -> np.ndarray:
    """Load saved knots from a one-shot plan or receding-horizon execution."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Warm-start trajectory does not exist: {source}")
    knots: np.ndarray | None = None
    metadata_path = source if source.suffix.lower() == ".json" else source.with_suffix(".json")
    if metadata_path.is_file():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        parameterization = payload.get("control_parameterization")
        if isinstance(parameterization, dict) and "knots_m_s2" in parameterization:
            knots = np.asarray(parameterization["knots_m_s2"], dtype=np.float32)
    if knots is None and source.suffix.lower() == ".npz":
        with np.load(source) as archive:
            if "optimized_knots_m_s2" in archive:
                values = np.asarray(archive["optimized_knots_m_s2"], dtype=np.float32)
                knots = values[-1] if values.ndim == 3 else values
            elif "control_knots_m_s2" in archive:
                knots = np.asarray(archive["control_knots_m_s2"], dtype=np.float32)
            elif "nominal_knots_m_s2" in archive:
                values = np.asarray(archive["nominal_knots_m_s2"], dtype=np.float32)
                knots = values[-1] if values.ndim == 3 else values
    if knots is None:
        raise ValueError(
            "No acceleration knots were found. Select a saved MPPI .npz file "
            "or its JSON sidecar."
        )
    if not np.all(np.isfinite(knots)):
        raise ValueError("Warm-start knots contain NaN or infinity.")
    return _resample_knots(knots, knot_count)


def simulation_view(
    result: SimulationResult,
    terms_source: dict[str, float],
    *,
    feasible: bool,
    impact_time_s: float,
) -> ExecutionView:
    """Build the common viewport adapter for live or completed execution."""

    impact_frame = int(np.argmin(np.abs(result.time_s - impact_time_s)))
    terms = dict(terms_source)
    initial = result.drone_positions_m[0]
    target_delta = result.target_position_m - initial
    target_delta = target_delta.copy()
    target_delta[2] = 0.0
    norm = float(np.linalg.norm(target_delta))
    if norm <= 1.0e-9:
        forward = result.impact_direction.copy()
        forward[2] = 0.0
        norm = float(np.linalg.norm(forward))
    else:
        forward = target_delta
    if norm <= 1.0e-9:
        forward = np.asarray((1.0, 0.0, 0.0))
    else:
        forward = forward / norm
    projection = (result.drone_positions_m - initial) @ forward
    peak_index = int(np.argmax(projection[: impact_frame + 1]))
    maximum_forward = float(projection[peak_index])
    recoil = float(
        maximum_forward - np.min(projection[peak_index : impact_frame + 1])
    )
    terms.setdefault("maximum_forward_stroke_m", maximum_forward)
    terms.setdefault("recoil_stroke_m", recoil)
    maximum_drone_speed = max(float(terms["maximum_drone_speed_m_s"]), 1.0e-9)
    amplification = float(terms["tip_speed_m_s"]) / maximum_drone_speed
    return ExecutionView(
        prediction=result,
        feasible=feasible,
        impact_frame=impact_frame,
        terms=terms,
        speed_amplification=amplification,
    )


def execution_view(execution: RecedingMppiExecution) -> ExecutionView:
    return simulation_view(
        execution.result,
        execution.cost_terms,
        feasible=execution.feasible,
        impact_time_s=execution.impact_time_s,
    )


def live_execution_view(update: RecedingMppiLiveUpdate) -> ExecutionView:
    return simulation_view(
        update.realized,
        update.realized_cost_terms,
        feasible=bool(update.realized_cost_terms["feasible"]),
        impact_time_s=update.realized_impact_time_s,
    )


class RecedingMppiGui:
    TICK_MS = 20

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Cable Twin – Receding-horizon DDER-MPPI")
        self.root.geometry("1580x960")
        self.root.minsize(1220, 760)
        self.root.configure(background="#ffffff")
        self._configure_style()

        self.snapshot: CableModelSnapshot | None = None
        self.execution: RecedingMppiExecution | None = None
        self.view_result: ExecutionView | None = None
        self.live_view_result: ExecutionView | None = None
        self.live_target_frame = 0
        self.pending_completion: (
            tuple[RecedingMppiExecution, Path, dict[str, object]] | None
        ) = None
        self.active_problem: MpcProblem | None = None
        self.active_mppi: MppiSettings | None = None
        self.active_execution_settings: RecedingMppiSettings | None = None
        self.active_simulation: SimulationSettings | None = None
        self.active_model_provenance: dict[str, object] = {}
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.cancel_event = threading.Event()
        self.running = False
        self.playing = False
        self.replay_wall_start_s: float | None = None
        self.replay_sim_start_s = 0.0

        self.model_path_var = tk.StringVar(value=str(DEFAULT_MODEL_PATH.resolve()))
        self.model_status_var = tk.StringVar(value="No model loaded")
        self.profile_status_var = tk.StringVar(value="No settings profile loaded")
        self.warm_start_var = tk.StringVar(value=str(DEFAULT_WARM_START_PATH.resolve()))
        self.warm_status_var = tk.StringVar(value="Warm start not checked")

        self.initial_var = tk.StringVar(value="0.0, 0.0, 1.5")
        self.target_var = tk.StringVar(value="1.0, 0.0, 1.4")
        self.direction_var = tk.StringVar(value="1.0, 0.0, 0.0")
        self.minimum_speed_var = tk.StringVar(value="3.5")
        self.tip_radius_var = tk.StringVar(value="0.05")
        self.angle_var = tk.StringVar(value="35.0")

        self.horizon_var = tk.StringVar(value="2.0")
        self.physics_rate_var = tk.StringVar(value="100")
        self.control_rate_var = tk.StringVar(value="50")
        # The validated maximum-throughput topology is the default.  Other
        # fitted resolutions remain available through captured CUDA rollout.
        self.simulation_nodes_var = tk.StringVar(value="11")
        self.replan_rate_var = tk.StringVar(value="10")
        self.timeout_var = tk.StringVar(value="1.20")
        self.maximum_acceleration_var = tk.StringVar(value="20.0")
        self.maximum_speed_var = tk.StringVar(value="3.0")

        self.samples_var = tk.StringVar(value="512")
        self.iterations_var = tk.StringVar(value="1")
        self.knots_var = tk.StringVar(value="11")
        self.noise_var = tk.StringVar(value="2.5")
        self.noise_decay_var = tk.StringVar(value="0.92")
        self.temperature_var = tk.StringVar(value="1.0")
        self.seed_var = tk.StringVar(value="17")
        self.truth_ei_scale_var = tk.StringVar(value="1.0")
        self.truth_cb_scale_var = tk.StringVar(value="1.0")
        self.truth_summary_var = tk.StringVar(value="Load a fitted model first.")
        self.adaptation_enabled_var = tk.StringVar(value="true")
        self.adaptation_status_var = tk.StringVar(
            value="Nominal EI/Cb · generation 0 · no completed strike"
        )
        self.preset_var = tk.StringVar(value="Balanced GPU: 512 samples × 1")
        self.controller_summary_var = tk.StringVar(value="")
        self.gpu_summary_var = tk.StringVar(value=self._gpu_summary())
        self.adaptation_session: BetweenStrikeAdaptationSession | None = None
        self.adaptation_session_key: tuple[object, ...] | None = None

        self.status_var = tk.StringVar(
            value="Load the cable model and verify the warm start, then run MPC."
        )
        self.outcome_var = tk.StringVar(value="No execution")
        self.quality_var = tk.StringVar(value="")
        self.compute_var = tk.StringVar(value="")
        self.timeline_var = tk.DoubleVar(value=0.0)

        self._build()
        for variable in (
            self.horizon_var,
            self.physics_rate_var,
            self.control_rate_var,
            self.simulation_nodes_var,
            self.replan_rate_var,
            self.samples_var,
            self.iterations_var,
            self.knots_var,
        ):
            variable.trace_add("write", lambda *_args: self._update_controller_summary())
        for variable in (self.truth_ei_scale_var, self.truth_cb_scale_var):
            variable.trace_add("write", lambda *_args: self._update_truth_summary())
        self._update_controller_summary()
        self._update_truth_summary()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(self.TICK_MS, self._tick)
        self.load_model(silent=True)
        self.check_warm_start(silent=True)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#ffffff", foreground="#111111")
        style.configure("TFrame", background="#ffffff")
        style.configure("TLabel", background="#ffffff", foreground="#111111")
        style.configure("TLabelframe", background="#ffffff", foreground="#111111")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#111111")
        style.configure("TNotebook", background="#ffffff", borderwidth=1)
        style.configure("TNotebook.Tab", padding=(12, 7))
        style.configure("TButton", padding=7)
        style.configure("Primary.TButton", padding=9, font=("Segoe UI Semibold", 10))
        style.configure("Result.TLabel", font=("Segoe UI Semibold", 11))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)
        ttk.Label(
            outer,
            text="Receding-horizon DDER-MPPI",
            font=("Segoe UI Semibold", 24),
        ).pack(anchor=tk.W)
        ttk.Label(
            outer,
            text=(
                "Full cable-state prediction, shifted warm start, short-prefix execution, "
                "and replanning. Controller compute is separated from the fixed strike objective."
            ),
            foreground="#555555",
        ).pack(anchor=tk.W, pady=(0, 10))

        model = ttk.LabelFrame(outer, text="Identified nominal cable model", padding=9)
        model.pack(fill=tk.X, pady=(0, 9))
        ttk.Entry(model, textvariable=self.model_path_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(model, text="Browse", command=self.browse_model).pack(
            side=tk.LEFT, padx=(7, 0)
        )
        ttk.Button(model, text="Load", command=self.load_model).pack(
            side=tk.LEFT, padx=(7, 0)
        )
        ttk.Label(model, textvariable=self.model_status_var).pack(
            side=tk.LEFT, padx=(12, 0)
        )

        profile = ttk.LabelFrame(outer, text="Settings profile", padding=7)
        profile.pack(fill=tk.X, pady=(0, 9))
        ttk.Button(
            profile,
            text="Load profile...",
            command=self.load_settings_profile,
        ).pack(side=tk.LEFT)
        ttk.Button(
            profile,
            text="Save profile...",
            command=self.save_settings_profile,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(profile, textvariable=self.profile_status_var).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(body, width=485)
        controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        controls.pack_propagate(False)
        view = ttk.Frame(body)
        view.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(controls)
        notebook.pack(fill=tk.BOTH, expand=True)
        task_tab = ttk.Frame(notebook, padding=10)
        controller_container = ttk.Frame(notebook)
        truth_tab = ttk.Frame(notebook, padding=10)
        adaptation_tab = ttk.Frame(notebook, padding=10)
        notebook.add(task_tab, text="Task")
        notebook.add(controller_container, text="Controller")
        notebook.add(truth_tab, text="Plant truth")
        notebook.add(adaptation_tab, text="Adaptation")
        self._build_task_tab(task_tab)
        self._build_truth_tab(truth_tab)
        self._build_adaptation_tab(adaptation_tab)

        controller_canvas = tk.Canvas(
            controller_container,
            background="#ffffff",
            highlightthickness=0,
            borderwidth=0,
        )
        controller_scrollbar = ttk.Scrollbar(
            controller_container, orient=tk.VERTICAL, command=controller_canvas.yview
        )
        controller_canvas.configure(yscrollcommand=controller_scrollbar.set)
        controller_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        controller_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        controller_tab = ttk.Frame(controller_canvas, padding=10)
        controller_window = controller_canvas.create_window(
            (0, 0), window=controller_tab, anchor=tk.NW
        )
        controller_tab.bind(
            "<Configure>",
            lambda _event: controller_canvas.configure(
                scrollregion=controller_canvas.bbox("all")
            ),
        )
        controller_canvas.bind(
            "<Configure>",
            lambda event: controller_canvas.itemconfigure(
                controller_window, width=event.width
            ),
        )
        self._build_controller_tab(controller_tab)

        actions = ttk.LabelFrame(controls, text="Execution", padding=9)
        actions.pack(fill=tk.X, pady=(8, 0))
        self.run_button = ttk.Button(
            actions,
            text="Run receding-horizon MPPI",
            command=self.run,
            style="Primary.TButton",
        )
        self.run_button.pack(fill=tk.X)
        row = ttk.Frame(actions)
        row.pack(fill=tk.X, pady=(6, 0))
        self.stop_button = ttk.Button(
            row, text="Stop", command=self.stop, state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.play_button = ttk.Button(
            row,
            text="Replay real time (1×)",
            command=self.toggle_play,
            state=tk.DISABLED,
        )
        self.play_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.save_button = ttk.Button(
            row, text="Save as...", command=self.save_as, state=tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

        log_frame = ttk.LabelFrame(controls, text="MPC log", padding=6)
        log_frame.pack(fill=tk.BOTH, pady=(8, 0))
        self.log = tk.Text(
            log_frame,
            height=8,
            background="#ffffff",
            foreground="#222222",
            insertbackground="#111111",
            highlightbackground="#cfcfcf",
            highlightthickness=1,
            borderwidth=0,
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.log.pack(fill=tk.BOTH)

        self.canvas = WhipTrajectoryCanvas(view)
        self.canvas.banner_text = (
            "RECEDING DDER-MPPI  |  REALIZED CLOSED LOOP  |  WORLD Z UP"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        metrics = ttk.LabelFrame(view, text="Execution result", padding=8)
        metrics.pack(fill=tk.X, pady=(7, 0))
        ttk.Label(metrics, textvariable=self.outcome_var, style="Result.TLabel").pack(
            anchor=tk.W
        )
        ttk.Label(metrics, textvariable=self.quality_var).pack(anchor=tk.W)
        ttk.Label(metrics, textvariable=self.compute_var).pack(anchor=tk.W)
        self.timeline = ttk.Scale(
            view,
            from_=0,
            to=1,
            variable=self.timeline_var,
            command=self._timeline_changed,
        )
        self.timeline.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(view, textvariable=self.status_var).pack(anchor=tk.W, pady=(4, 0))

    def _build_task_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Strike definition",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text="These fields define the goal. They do not retune MPPI.",
            foreground="#666666",
        ).pack(anchor=tk.W, pady=(0, 8))
        self._entry(parent, "Initial drone XYZ", self.initial_var, "m")
        self._entry(parent, "Target XYZ", self.target_var, "m")
        self._entry(parent, "Impact direction", self.direction_var, "unit")
        self._entry(parent, "Minimum directed speed", self.minimum_speed_var, "m/s")
        self._entry(parent, "Target sphere radius", self.tip_radius_var, "m")
        self._entry(parent, "Direction half-angle", self.angle_var, "deg")
        note = ttk.LabelFrame(parent, text="Success", padding=8)
        note.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(
            note,
            text=(
                "First geometric tip entry into the target sphere, sufficient directed "
                "speed, impact inside the direction cone, tip-first ordering, and no "
                "safety violation."
            ),
            wraplength=410,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)

    def _build_controller_tab(self, parent: ttk.Frame) -> None:
        preset = ttk.LabelFrame(parent, text="Compute preset", padding=8)
        preset.pack(fill=tk.X)
        ttk.Combobox(
            preset,
            textvariable=self.preset_var,
            values=(
                "Low latency: 128 samples × 1",
                "Balanced GPU: 512 samples × 1",
                "GPU saturation: 2048 samples × 1",
                "Higher refinement: 2048 samples × 2",
            ),
            state="readonly",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(preset, text="Apply", command=self.apply_preset).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        prediction = ttk.LabelFrame(parent, text="Prediction discretization", padding=8)
        prediction.pack(fill=tk.X, pady=(8, 0))
        self._entry(prediction, "Prediction horizon", self.horizon_var, "s")
        self._entry(prediction, "Physics rate", self.physics_rate_var, "Hz")
        self._entry(prediction, "Acceleration command rate", self.control_rate_var, "Hz")
        self._entry(prediction, "DDER simulation nodes", self.simulation_nodes_var, "nodes")
        self._entry(prediction, "Acceleration knots", self.knots_var, "")

        budget = ttk.LabelFrame(parent, text="MPPI compute budget", padding=8)
        budget.pack(fill=tk.X, pady=(8, 0))
        self._entry(budget, "Samples / update", self.samples_var, "")
        self._entry(budget, "Iterations / update", self.iterations_var, "")
        ttk.Label(
            budget,
            textvariable=self.gpu_summary_var,
            foreground="#555555",
            wraplength=410,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 4))
        self._entry(budget, "Noise sigma", self.noise_var, "m/s²")
        self._entry(budget, "Noise decay", self.noise_decay_var, "")
        self._entry(budget, "Temperature", self.temperature_var, "")
        self._entry(budget, "Random seed (0 = new)", self.seed_var, "")

        closed_loop = ttk.LabelFrame(parent, text="Closed-loop execution", padding=8)
        closed_loop.pack(fill=tk.X, pady=(8, 0))
        self._entry(closed_loop, "Replanning rate", self.replan_rate_var, "Hz")
        self._entry(closed_loop, "Execution timeout", self.timeout_var, "s")
        self._entry(
            closed_loop, "Maximum acceleration", self.maximum_acceleration_var, "m/s²"
        )
        self._entry(closed_loop, "Maximum drone speed", self.maximum_speed_var, "m/s")

        warm = ttk.LabelFrame(parent, text="First-solve warm start", padding=8)
        warm.pack(fill=tk.X, pady=(8, 0))
        ttk.Entry(warm, textvariable=self.warm_start_var).pack(fill=tk.X)
        row = ttk.Frame(warm)
        row.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(row, text="Browse", command=self.browse_warm_start).pack(side=tk.LEFT)
        ttk.Button(row, text="Verify", command=self.check_warm_start).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Label(row, textvariable=self.warm_status_var).pack(side=tk.LEFT, padx=(8, 0))

        ttk.Label(
            parent,
            textvariable=self.controller_summary_var,
            foreground="#333333",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))

    def _build_truth_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Hidden simulated plant",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text=(
                "The MPPI controller always predicts with the fitted cable model. "
                "These ratios change only the plant used to execute the controls."
            ),
            foreground="#555555",
            wraplength=410,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 10))

        parameters = ttk.LabelFrame(parent, text="Physical mismatch", padding=8)
        parameters.pack(fill=tk.X)
        self._entry(
            parameters,
            "Truth EI / fitted EI",
            self.truth_ei_scale_var,
            "ratio",
        )
        self._entry(
            parameters,
            "Truth Cb / fitted Cb",
            self.truth_cb_scale_var,
            "ratio",
        )
        ttk.Label(
            parameters,
            text="Use 1.0 and 1.0 for the matched-model baseline.",
            foreground="#666666",
        ).pack(anchor=tk.W, pady=(5, 0))

        summary = ttk.LabelFrame(parent, text="Resolved experiment", padding=8)
        summary.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(
            summary,
            textvariable=self.truth_summary_var,
            wraplength=410,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text=(
                "Node grid, cable mass, geometry, gravity, drag, constraints, time "
                "step, and solver substeps are held common. The controller is not "
                "given the truth ratios."
            ),
            foreground="#555555",
            wraplength=410,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(10, 0))

    def _build_adaptation_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Between-strike physical adaptation",
            font=("Segoe UI Semibold", 12),
        ).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text=(
                "The active EI/Cb model is frozen during every MPPI strike. After "
                "the strike, distributed cable motion is checked; an informative "
                "mismatch is fitted and held-out validated, then a fresh accelerated "
                "runtime is prewarmed and atomically published for the next strike."
            ),
            foreground="#555555",
            wraplength=410,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(0, 10))
        ttk.Checkbutton(
            parent,
            text="Enable validated between-strike EI/Cb adaptation",
            variable=self.adaptation_enabled_var,
            onvalue="true",
            offvalue="false",
        ).pack(anchor=tk.W)
        ttk.Label(
            parent,
            text=(
                "Current simulation mode uses the exact distributed plant state. "
                "The physical OptiTrack path will use the same adapter only after "
                "the causal state estimator is validated. Fitting never runs inside "
                "the 10 Hz planning path."
            ),
            foreground="#666666",
            wraplength=410,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(7, 10))
        status = ttk.LabelFrame(parent, text="Published controller model", padding=8)
        status.pack(fill=tk.X)
        ttk.Label(
            status,
            textvariable=self.adaptation_status_var,
            wraplength=390,
            justify=tk.LEFT,
        ).pack(anchor=tk.W)
        ttk.Button(
            parent,
            text="Reset adapted model to fitted EI/Cb",
            command=self.reset_adaptation,
        ).pack(anchor=tk.W, pady=(10, 0))

    @staticmethod
    def _adaptation_enabled(value: str) -> bool:
        normalized = value.strip().lower()
        if normalized not in {"true", "false"}:
            raise ValueError("Adaptation enabled must be 'true' or 'false'.")
        return normalized == "true"

    def reset_adaptation(self) -> None:
        if self.running:
            messagebox.showerror(
                "Online adaptation",
                "Stop the current strike before resetting the physical model.",
                parent=self.root,
            )
            return
        self.adaptation_session = None
        self.adaptation_session_key = None
        self.adaptation_status_var.set(
            "Nominal EI/Cb · generation 0 · no completed strike"
        )
        self.status_var.set("Adapted controller model reset to the fitted EI/Cb.")

    @staticmethod
    def _entry(
        parent: ttk.Frame, label: str, variable: tk.StringVar, unit: str
    ) -> ttk.Entry:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=25).pack(side=tk.LEFT)
        entry = ttk.Entry(row, textvariable=variable, width=14)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(row, text=unit, width=6).pack(side=tk.LEFT, padx=(5, 0))
        return entry

    @staticmethod
    def _vector(text: str, name: str) -> tuple[float, float, float]:
        values = tuple(float(item) for item in text.replace(",", " ").split())
        if len(values) != 3 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} requires three finite numbers.")
        return values  # type: ignore[return-value]

    def _update_controller_summary(self) -> None:
        try:
            horizon = float(self.horizon_var.get())
            physics_rate = float(self.physics_rate_var.get())
            control_rate = float(self.control_rate_var.get())
            replan_rate = float(self.replan_rate_var.get())
            samples = int(self.samples_var.get())
            iterations = int(self.iterations_var.get())
            knots = int(self.knots_var.get())
            simulation_nodes = int(self.simulation_nodes_var.get())
            acceleration_tier = (
                "maximum fused 11-node CUDA"
                if simulation_nodes == 11
                else "captured arbitrary-node CUDA"
            )
            physics_steps = round(horizon * physics_rate)
            controls = round(horizon * control_rate)
            apply_controls = control_rate / replan_rate
            self.controller_summary_var.set(
                f"Per update: {samples * iterations} DDER rollouts; all {samples} "
                f"samples run in one CUDA batch per iteration. Horizon: "
                f"{physics_steps} physics steps, "
                f"{controls} controls, {knots} knots, {simulation_nodes} DDER nodes. "
                f"Runtime: {acceleration_tier}. "
                f"Execute {apply_controls:g} "
                "control interval(s) before replanning."
            )
        except (ValueError, ZeroDivisionError):
            self.controller_summary_var.set("Enter valid controller settings.")

    def _update_truth_summary(self) -> None:
        if self.snapshot is None:
            self.truth_summary_var.set("Load a fitted model first.")
            return
        try:
            settings = TruthModelSettings(
                bending_stiffness_scale=float(self.truth_ei_scale_var.get()),
                bending_damping_scale=float(self.truth_cb_scale_var.get()),
            )
        except ValueError:
            self.truth_summary_var.set("Enter positive finite EI and Cb ratios.")
            return
        fitted_ei = self.snapshot.bending_stiffness_n_m2
        fitted_cb = self.snapshot.bending_damping_n_m2_s
        matched = (
            settings.bending_stiffness_scale == 1.0
            and settings.bending_damping_scale == 1.0
        )
        mode = "MATCHED MODEL" if matched else "PARAMETER MISMATCH"
        self.truth_summary_var.set(
            f"{mode}\n"
            f"Controller: EI={fitted_ei:.6g} N m^2, "
            f"Cb={fitted_cb:.6g} N m^2 s\n"
            f"Truth plant: EI={fitted_ei * settings.bending_stiffness_scale:.6g} "
            f"N m^2, Cb={fitted_cb * settings.bending_damping_scale:.6g} N m^2 s"
        )

    def apply_preset(self) -> None:
        label = self.preset_var.get()
        if label.startswith("Low"):
            samples, iterations = 128, 1
        elif label.startswith("GPU"):
            samples, iterations = 2048, 1
        elif label.startswith("Higher"):
            samples, iterations = 2048, 2
        else:
            samples, iterations = 512, 1
        self.samples_var.set(str(samples))
        self.iterations_var.set(str(iterations))

    @staticmethod
    def _gpu_summary() -> str:
        if not torch.cuda.is_available():
            return "CUDA is unavailable; the online MPPI controller requires a CUDA GPU."
        properties = torch.cuda.get_device_properties(0)
        total_gib = properties.total_memory / (1024.0**3)
        return (
            f"{properties.name} · {total_gib:.1f} GiB. Online runs require captured "
            "full-horizon CUDA and fused CUDA cost evaluation. Eleven nodes also "
            "use the validated fused damping/projection kernels; other resolutions "
            "retain captured arbitrary-node DDER mechanics."
        )

    def _profile_variable_map(self) -> dict[str, dict[str, tk.StringVar]]:
        return {
            "inputs": {
                "model_path": self.model_path_var,
                "warm_start_path": self.warm_start_var,
            },
            "task": {
                "initial_drone_xyz": self.initial_var,
                "target_xyz": self.target_var,
                "impact_direction": self.direction_var,
                "minimum_directed_speed_m_s": self.minimum_speed_var,
                "target_radius_m": self.tip_radius_var,
                "direction_half_angle_deg": self.angle_var,
            },
            "controller": {
                "horizon_s": self.horizon_var,
                "physics_rate_hz": self.physics_rate_var,
                "control_rate_hz": self.control_rate_var,
                "simulation_nodes": self.simulation_nodes_var,
                "replanning_rate_hz": self.replan_rate_var,
                "execution_timeout_s": self.timeout_var,
                "maximum_acceleration_m_s2": self.maximum_acceleration_var,
                "maximum_speed_m_s": self.maximum_speed_var,
                "samples": self.samples_var,
                "iterations": self.iterations_var,
                "acceleration_knots": self.knots_var,
                "noise_sigma_m_s2": self.noise_var,
                "noise_decay": self.noise_decay_var,
                "temperature": self.temperature_var,
                "seed": self.seed_var,
            },
            "plant_truth": {
                "ei_scale": self.truth_ei_scale_var,
                "cb_scale": self.truth_cb_scale_var,
            },
            "adaptation": {
                "enabled": self.adaptation_enabled_var,
            },
        }

    def save_settings_profile(self) -> None:
        if self.running:
            messagebox.showerror(
                "Settings profile",
                "Stop the current execution before saving a profile.",
                parent=self.root,
            )
            return
        try:
            # Apply the same semantic checks used by Run before creating a
            # profile that appears reproducible but cannot execute.
            self._build_run_configuration()
        except Exception as error:
            messagebox.showerror(
                "Settings profile", str(error), parent=self.root
            )
            return
        DEFAULT_SETTINGS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save DDER-MPPI settings profile",
            initialdir=str(DEFAULT_SETTINGS_PROFILE_DIR),
            initialfile="mppi_settings.json",
            defaultextension=".json",
            filetypes=(("Settings profile", "*.json"),),
        )
        if not path:
            return
        variables = self._profile_variable_map()
        payload: dict[str, object] = {
            "schema": SETTINGS_PROFILE_SCHEMA,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "fixed_objective": dict(FIXED_OBJECTIVE),
        }
        for section, fields in variables.items():
            payload[section] = {
                field: variable.get() for field, variable in fields.items()
            }
        # Validate the serialized representation before touching the target.
        normalize_settings_profile(payload)
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(output)
        self.profile_status_var.set(f"Saved {output.name}")

    def load_settings_profile(self) -> None:
        if self.running:
            messagebox.showerror(
                "Settings profile",
                "Stop the current execution before loading a profile.",
                parent=self.root,
            )
            return
        DEFAULT_SETTINGS_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load DDER-MPPI settings profile",
            initialdir=str(DEFAULT_SETTINGS_PROFILE_DIR),
            filetypes=(("Settings profile", "*.json"), ("All files", "*.*")),
        )
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            normalized = normalize_settings_profile(payload)
        except Exception as error:
            messagebox.showerror(
                "Settings profile", str(error), parent=self.root
            )
            return

        # Preserve every old value so the semantic validation below is
        # transactional at the UI level.
        variables = self._profile_variable_map()
        previous = {
            section: {
                field: variable.get() for field, variable in fields.items()
            }
            for section, fields in variables.items()
        }
        for section, fields in normalized.items():
            for field, value in fields.items():
                variables[section][field].set(value)
        requested_nodes = normalized["controller"]["simulation_nodes"]
        try:
            self.load_model(silent=True)
            if self.snapshot is None:
                raise ValueError(
                    "The cable model referenced by the profile could not be loaded."
                )
            if self.simulation_nodes_var.get() != requested_nodes:
                raise ValueError(
                    "The profile's DDER node count is incompatible with its cable model."
                )
            self._build_run_configuration()
        except Exception as error:
            for section, fields in previous.items():
                for field, value in fields.items():
                    variables[section][field].set(value)
            self.load_model(silent=True)
            self.check_warm_start(silent=True)
            self._update_controller_summary()
            self._update_truth_summary()
            messagebox.showerror(
                "Settings profile", str(error), parent=self.root
            )
            return
        self.check_warm_start(silent=True)
        self._update_controller_summary()
        self._update_truth_summary()
        self.profile_status_var.set(f"Loaded {Path(path).name}")

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select cable model",
            filetypes=(("Cable model", "*.json"), ("All files", "*.*")),
        )
        if path:
            self.model_path_var.set(path)

    def load_model(self, *, silent: bool = False) -> None:
        if self.running:
            return
        try:
            self.snapshot = load_cable_model(self.model_path_var.get())
        except Exception as error:
            self.snapshot = None
            self.model_status_var.set("Incompatible model")
            if not silent:
                messagebox.showerror("Cable model", str(error), parent=self.root)
            return
        assert self.snapshot is not None
        try:
            requested_nodes = int(self.simulation_nodes_var.get())
        except ValueError:
            requested_nodes = self.snapshot.node_count
        if not 6 <= requested_nodes <= self.snapshot.node_count:
            self.simulation_nodes_var.set(str(self.snapshot.node_count))
        provisional = "PROVISIONAL  " if self.snapshot.provisional else ""
        self.model_status_var.set(
            f"{provisional}{self.snapshot.node_count} nodes  "
            f"EI={self.snapshot.bending_stiffness_n_m2:.3g}  "
            f"Cb={self.snapshot.bending_damping_n_m2_s:.3g}"
        )
        self._update_truth_summary()

    def browse_warm_start(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select MPPI warm start",
            filetypes=(
                ("Saved MPPI result", "*.npz"),
                ("Result metadata", "*.json"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.warm_start_var.set(path)
            self.check_warm_start()

    def check_warm_start(self, *, silent: bool = False) -> None:
        try:
            knots = load_warm_start_knots(
                self.warm_start_var.get(), int(self.knots_var.get())
            )
        except Exception as error:
            self.warm_status_var.set("Invalid")
            if not silent:
                messagebox.showerror("Warm start", str(error), parent=self.root)
            return
        peak = float(np.max(np.linalg.vector_norm(knots, axis=1)))
        self.warm_status_var.set(f"{len(knots)} knots, peak {peak:.1f} m/s²")

    def _build_run_configuration(self):
        initial = self._vector(self.initial_var.get(), "Initial drone XYZ")
        if self.snapshot is None:
            raise ValueError("Load a cable model first.")
        simulation_node_count = int(self.simulation_nodes_var.get())
        if not 6 <= simulation_node_count <= self.snapshot.node_count:
            raise ValueError(
                "DDER simulation nodes must be between 6 and the fitted model "
                f"resolution ({self.snapshot.node_count})."
            )
        problem = MpcProblem(
            target_position_m=self._vector(self.target_var.get(), "Target XYZ"),
            impact_direction=self._vector(self.direction_var.get(), "Impact direction"),
            minimum_impact_speed_m_s=float(self.minimum_speed_var.get()),
            maximum_tip_error_m=float(self.tip_radius_var.get()),
            maximum_impact_angle_deg=float(self.angle_var.get()),
            # The public MPPI objective does not use an excursion radius.  The
            # shared legacy problem schema still requires a positive value;
            # ``enforce_workspace_limit=False`` below makes it diagnostic only.
            maximum_drone_excursion_m=1.0,
            minimum_forward_stroke_m=0.0,
            minimum_recoil_stroke_m=0.0,
            drone_keepout_radius_m=0.30,
            drone_workspace_center_m=initial,
        )
        horizon = float(self.horizon_var.get())
        physics_rate = float(self.physics_rate_var.get())
        control_rate = float(self.control_rate_var.get())
        replan_rate = float(self.replan_rate_var.get())
        simulation = SimulationSettings(
            horizon_s=horizon,
            simulation_dt_s=1.0 / physics_rate,
            control_interval_s=1.0 / control_rate,
            attachment_drop_m=0.10,
            maximum_acceleration_m_s2=float(self.maximum_acceleration_var.get()),
            maximum_speed_m_s=float(self.maximum_speed_var.get()),
        )
        replan_interval = 1.0 / replan_rate
        ratio = replan_interval / simulation.control_interval_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                "Replanning rate must execute an integer number of control intervals. "
                f"At {control_rate:g} Hz control, choose a rate such as "
                f"{control_rate:g}, {control_rate / 2:g}, {control_rate / 5:g}, or "
                f"{control_rate / 10:g} Hz."
            )
        samples = int(self.samples_var.get())
        seed = resolve_run_seed(int(self.seed_var.get()))
        mppi = MppiSettings(
            iterations=int(self.iterations_var.get()),
            samples=samples,
            rollout_batch_size=samples,
            knot_count=int(self.knots_var.get()),
            temperature=float(self.temperature_var.get()),
            acceleration_noise_sigma_m_s2=float(self.noise_var.get()),
            noise_decay=float(self.noise_decay_var.get()),
            seed=seed,
            objective_stage=str(FIXED_OBJECTIVE["objective_stage"]),
            position_sigma_m=float(FIXED_OBJECTIVE["position_sigma_m"]),
            velocity_gate_sigma_m=float(FIXED_OBJECTIVE["velocity_gate_sigma_m"]),
            position_weight=float(FIXED_OBJECTIVE["position_weight"]),
            speed_weight=float(FIXED_OBJECTIVE["speed_weight"]),
            predictive_speed_weight=0.0,
            direction_weight=float(FIXED_OBJECTIVE["direction_weight"]),
            success_cost=float(FIXED_OBJECTIVE["success_cost"]),
            drone_displacement_weight=float(
                FIXED_OBJECTIVE["drone_displacement_weight"]
            ),
            safety_weight=float(FIXED_OBJECTIVE["safety_weight"]),
            enforce_workspace_limit=False,
            control_effort_weight=float(FIXED_OBJECTIVE["control_effort_weight"]),
            control_smoothness_weight=float(
                FIXED_OBJECTIVE["control_smoothness_weight"]
            ),
            gradient_guidance_fraction=0.0,
        )
        execution_settings = RecedingMppiSettings(
            replan_interval_s=replan_interval,
            timeout_s=float(self.timeout_var.get()),
            feedback_mode="full",
        )
        warm = load_warm_start_knots(self.warm_start_var.get(), mppi.knot_count)
        truth_settings = TruthModelSettings(
            bending_stiffness_scale=float(self.truth_ei_scale_var.get()),
            bending_damping_scale=float(self.truth_cb_scale_var.get()),
        )
        # Resolve both snapshots here as validation so an unstable hidden truth
        # model is rejected by Run/Profile before a worker thread is launched.
        controller_snapshot, _plant_snapshot = build_controller_and_truth_models(
            self.snapshot,
            simulation_dt_s=simulation.simulation_dt_s,
            node_count=simulation_node_count,
            truth_bending_stiffness_scale=truth_settings.bending_stiffness_scale,
            truth_bending_damping_scale=truth_settings.bending_damping_scale,
        )
        # The normal UI has no slow-path selector.  Reference propagation is
        # retained only for numerical tests and differentiable fitting.
        WhipSimulator(
            controller_snapshot, simulation, device="cuda"
        ).require_online_acceleration()
        adaptation_enabled = self._adaptation_enabled(
            self.adaptation_enabled_var.get()
        )
        return (
            initial,
            problem,
            simulation,
            mppi,
            execution_settings,
            warm,
            simulation_node_count,
            truth_settings,
            adaptation_enabled,
        )

    def _append_log(self, message: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def run(self) -> None:
        if self.running:
            return
        if self.snapshot is None:
            messagebox.showerror("Online MPC", "Load a cable model first.", parent=self.root)
            return
        try:
            (
                initial,
                problem,
                simulation,
                mppi,
                execution_settings,
                warm,
                simulation_node_count,
                truth_settings,
                adaptation_enabled,
            ) = self._build_run_configuration()
        except Exception as error:
            messagebox.showerror("Online MPC settings", str(error), parent=self.root)
            return

        self.active_problem = problem
        self.active_simulation = simulation
        self.active_model_provenance = {}
        self.active_mppi = mppi
        self.active_execution_settings = execution_settings
        self.cancel_event.clear()
        self.running = True
        self.playing = False
        self.execution = None
        self.view_result = None
        self.live_view_result = None
        self.live_target_frame = 0
        self.pending_completion = None
        self.canvas.result = None
        self.canvas.redraw()
        self.timeline_var.set(0.0)
        self.outcome_var.set("Controller running")
        self.quality_var.set("")
        self.compute_var.set("")
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        requested_seed = int(self.seed_var.get())
        self._append_log(
            f"MPPI run seed: {mppi.seed}"
            + (" (randomized because UI seed is 0)" if requested_seed == 0 else "")
        )
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.play_button.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self.status_var.set(
            "Warming DDER CUDA graphs, then replanning from the current cable state..."
        )
        snapshot = self.snapshot
        warm_source = self.warm_start_var.get()

        def worker() -> None:
            try:
                nominal_controller, plant_snapshot = (
                    build_controller_and_truth_models(
                        snapshot,
                        simulation_dt_s=simulation.simulation_dt_s,
                        node_count=simulation_node_count,
                        truth_bending_stiffness_scale=(
                            truth_settings.bending_stiffness_scale
                        ),
                        truth_bending_damping_scale=(
                            truth_settings.bending_damping_scale
                        ),
                    )
                )
                session_key = (
                    nominal_controller.sha256,
                    simulation,
                    truth_settings,
                )
                if adaptation_enabled:
                    if (
                        self.adaptation_session is None
                        or self.adaptation_session_key != session_key
                    ):
                        self.adaptation_session = BetweenStrikeAdaptationSession(
                            nominal_controller,
                            simulation,
                            device="cuda",
                        )
                        self.adaptation_session_key = session_key
                        self.events.put(
                            (
                                "log",
                                "Adaptation session initialized at fitted EI/Cb; "
                                "future accepted updates apply to the next strike.",
                            )
                        )
                    runtime = self.adaptation_session.runtime()
                    controller_snapshot = runtime.snapshot
                    planner = runtime.simulator
                    active_estimate = runtime.estimate
                else:
                    controller_snapshot = nominal_controller
                    planner = WhipSimulator(
                        controller_snapshot, simulation, device="cuda"
                    )
                    active_estimate = None
                planner_tier = planner.require_online_acceleration()
                provenance = model_pair_provenance(
                    snapshot,
                    controller_snapshot,
                    plant_snapshot,
                    truth_settings,
                )
                provenance["runtime_acceleration"] = asdict(planner_tier)
                provenance["adaptation_enabled"] = adaptation_enabled
                if active_estimate is not None:
                    provenance["controller_estimate_at_strike_start"] = {
                        "generation": active_estimate.generation,
                        "ei_ratio": active_estimate.ei_ratio,
                        "cb_ratio": active_estimate.cb_ratio,
                        "source": active_estimate.source,
                    }
                self.events.put(
                    (
                        "log",
                        f"DDER resolution: fitted={snapshot.node_count} nodes, "
                        f"simulation={controller_snapshot.node_count} nodes, "
                        f"common substeps={controller_snapshot.model.parameters.substeps}",
                    )
                )
                self.events.put(
                    (
                        "log",
                        f"Accelerated runtime: {planner_tier.tier}",
                    )
                )
                self.events.put(
                    (
                        "log",
                        "Controller nominal: "
                        f"EI={controller_snapshot.bending_stiffness_n_m2:.6g} N m^2, "
                        f"Cb={controller_snapshot.bending_damping_n_m2_s:.6g} N m^2 s; "
                        "plant truth: "
                        f"EI={plant_snapshot.bending_stiffness_n_m2:.6g} N m^2 "
                        f"({truth_settings.bending_stiffness_scale:g}x), "
                        f"Cb={plant_snapshot.bending_damping_n_m2_s:.6g} N m^2 s "
                        f"({truth_settings.bending_damping_scale:g}x)",
                    )
                )
                plant = WhipSimulator(
                    plant_snapshot, simulation, device="cuda"
                )
                plant.require_online_acceleration()
                state = plant.initial_state(initial)
                execution = run_receding_horizon_mppi(
                    planner,
                    plant,
                    state,
                    problem,
                    mppi,
                    execution_settings,
                    warm,
                    progress=lambda message: self.events.put(("log", message)),
                    live_update=lambda update: self.events.put(
                        ("live_update", update)
                    ),
                    cancelled=self.cancel_event.is_set,
                )
                if adaptation_enabled:
                    assert self.adaptation_session is not None
                    self.events.put(
                        (
                            "adaptation_working",
                            "Strike complete. Monitoring distributed motion and "
                            "fitting only if mismatch and information gates pass...",
                        )
                    )
                    adaptation_result = self.adaptation_session.process_execution(
                        execution,
                        prewarm_batch_size=mppi.samples,
                        prewarm_control_count=simulation.control_count,
                    )
                    estimate_after = self.adaptation_session.estimate
                    provenance["adaptation_after_strike"] = {
                        "triggered": adaptation_result.triggered,
                        "fit_attempted": adaptation_result.fit_attempted,
                        "accepted": adaptation_result.accepted,
                        "published": adaptation_result.published,
                        "reason": adaptation_result.reason,
                        "generation": estimate_after.generation,
                        "ei_ratio": estimate_after.ei_ratio,
                        "cb_ratio": estimate_after.cb_ratio,
                        "monitoring_wall_time_s": (
                            adaptation_result.monitoring_wall_time_s
                        ),
                        "rebuild_wall_time_s": adaptation_result.rebuild_wall_time_s,
                    }
                    self.events.put(
                        (
                            "adaptation_result",
                            (adaptation_result, estimate_after),
                        )
                    )
                output = save_receding_mppi_execution(
                    DEFAULT_OUTPUT_PATH,
                    execution,
                    problem,
                    mppi,
                    execution_settings,
                    simulation,
                    warm_start_source=warm_source,
                    model_provenance=provenance,
                )
            except Exception as error:
                self.events.put(("error", error))
            else:
                self.events.put(("complete", (execution, output, provenance)))

        threading.Thread(
            target=worker,
            name="receding-dder-mppi",
            daemon=True,
        ).start()

    def stop(self) -> None:
        if self.running:
            self.cancel_event.set()
            self.status_var.set("Stopping after the current DDER rollout...")

    def toggle_play(self) -> None:
        if self.view_result is None:
            return
        if self.playing:
            self.playing = False
            self.replay_wall_start_s = None
            self.play_button.configure(text="Replay real time (1×)")
            return
        if self.timeline_var.get() >= self.view_result.prediction.frame_count - 1:
            self.timeline_var.set(0.0)
            self.canvas.set_trajectory_frame(0)
        frame = int(round(self.timeline_var.get()))
        frame = int(np.clip(frame, 0, self.view_result.prediction.frame_count - 1))
        self.replay_sim_start_s = float(self.view_result.prediction.time_s[frame])
        self.replay_wall_start_s = time.perf_counter()
        self.playing = True
        self.play_button.configure(text="Pause replay")

    def _timeline_changed(self, _value: str) -> None:
        if self.view_result is None:
            return
        self.canvas.set_trajectory_frame(int(round(self.timeline_var.get())))

    def save_as(self) -> None:
        if not all(
            value is not None
            for value in (
                self.execution,
                self.active_problem,
                self.active_mppi,
                self.active_execution_settings,
                self.active_simulation,
            )
        ):
            return
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save receding-horizon execution",
            defaultextension=".npz",
            filetypes=(("NumPy archive", "*.npz"),),
        )
        if not path:
            return
        output = save_receding_mppi_execution(
            path,
            self.execution,  # type: ignore[arg-type]
            self.active_problem,  # type: ignore[arg-type]
            self.active_mppi,  # type: ignore[arg-type]
            self.active_execution_settings,  # type: ignore[arg-type]
            self.active_simulation,  # type: ignore[arg-type]
            warm_start_source=self.warm_start_var.get(),
            model_provenance=self.active_model_provenance,
        )
        self.status_var.set(f"Saved {output}")

    def _show_complete(
        self,
        execution: RecedingMppiExecution,
        output: Path,
        provenance: dict[str, object],
    ) -> None:
        self.execution = execution
        self.active_model_provenance = provenance
        self.view_result = execution_view(execution)
        if self.canvas.result is None:
            self.canvas.set_result(self.view_result)  # type: ignore[arg-type]
        else:
            self.canvas.result = self.view_result  # type: ignore[assignment]
        self.timeline.configure(to=self.view_result.prediction.frame_count - 1)
        final_frame = self.view_result.prediction.frame_count - 1
        self.timeline_var.set(float(final_frame))
        self.canvas.set_trajectory_frame(final_frame)
        terms = execution.cost_terms
        planning = [update.planning_wall_time_s for update in execution.updates]
        median_plan = float(np.median(planning)) if planning else 0.0
        assert self.active_execution_settings is not None
        deadline = self.active_execution_settings.replan_interval_s
        deadline_rate = 100.0 * sum(value <= deadline for value in planning) / max(
            len(planning), 1
        )
        outcome = "VALID STRIKE" if execution.feasible else execution.terminal_reason.upper()
        model_mode = (
            "matched"
            if bool(provenance.get("matched", False))
            else "EI/Cb mismatch"
        )
        self.outcome_var.set(
            f"{outcome}  |  feedback={execution.feedback_mode}  |  plant={model_mode}"
        )
        self.quality_var.set(
            f"error={1000.0 * terms['position_error_m']:.1f} mm    "
            f"directed speed={terms['directional_speed_m_s']:.2f} m/s    "
            f"direction error={terms['direction_error_deg']:.1f}°    "
            f"impact={execution.impact_time_s:.2f} s"
        )
        self.compute_var.set(
            f"updates={len(execution.updates)}    rollouts={execution.total_rollouts}    "
            f"median update={median_plan:.2f} s    deadlines met={deadline_rate:.0f}%    "
            f"planning total={execution.total_planning_wall_time_s:.1f} s"
        )
        self.status_var.set(f"Saved complete execution and predictions to {output}")

    def _show_live_update(self, update: RecedingMppiLiveUpdate) -> None:
        """Append one realized MPC block without resetting the camera."""

        view = live_execution_view(update)
        previous_frame = self.canvas.frame_index if self.canvas.result is not None else 0
        if self.canvas.result is None:
            self.canvas.set_result(view)  # type: ignore[arg-type]
            previous_frame = 0
        else:
            self.canvas.result = view  # type: ignore[assignment]
        self.live_view_result = view
        self.live_target_frame = view.prediction.frame_count - 1
        self.timeline.configure(to=max(self.live_target_frame, 1))
        self.timeline_var.set(float(min(previous_frame, self.live_target_frame)))
        self.canvas.set_trajectory_frame(min(previous_frame, self.live_target_frame))

        terms = update.realized_cost_terms
        step = update.update.index + 1
        terminal = update.terminal_reason or "replanning"
        self.outcome_var.set(
            f"LIVE MPC STEP {step}  |  executed to t={view.prediction.time_s[-1]:.2f}s"
        )
        self.quality_var.set(
            f"current closest error={1000.0 * terms['position_error_m']:.1f} mm    "
            f"directed speed={terms['directional_speed_m_s']:.2f} m/s    "
            f"direction error={terms['direction_error_deg']:.1f}°"
        )
        self.compute_var.set(
            f"latest planning={update.update.planning_wall_time_s:.2f} s    "
            f"executed controls={update.update.executed_control_count}    "
            f"sampled rollouts={update.update.rollout_count}    state={terminal}"
        )
        self.status_var.set(
            "Rendering the newly executed physics frames step by step; "
            "the worker is preparing the next MPC update."
        )

    def _show_adaptation_result(
        self,
        result: BetweenStrikeAdaptationResult,
        estimate,
    ) -> None:
        if result.accepted and result.published:
            state = "ACCEPTED + PUBLISHED"
        elif result.fit_attempted:
            state = "FIT REJECTED"
        elif result.triggered:
            state = "WAITING FOR INFORMATIVE DATA"
        else:
            state = "NO FIT NEEDED"
        self.adaptation_status_var.set(
            f"{state}\n"
            f"generation {estimate.generation} · EI/fitted={estimate.ei_ratio:.4f} · "
            f"Cb/fitted={estimate.cb_ratio:.4f}\n"
            f"reason={result.reason} · candidates={result.candidate_segment_count} · "
            f"fit/validation={result.fit_segment_count}/{result.validation_segment_count}"
        )
        self._append_log(
            f"Between-strike adaptation: {state.lower()}, {result.reason}; "
            f"EI ratio={estimate.ei_ratio:.4f}, Cb ratio={estimate.cb_ratio:.4f}, "
            f"generation={estimate.generation}."
        )
        if result.fit_result is not None:
            fit = result.fit_result
            self._append_log(
                "  held-out all-node position RMSE: "
                f"{1000.0 * fit.all_node_position_rmse_before_m:.3f} -> "
                f"{1000.0 * fit.all_node_position_rmse_after_m:.3f} mm; "
                f"fit={fit.timing.total_s:.3f}s, runtime rebuild="
                f"{result.rebuild_wall_time_s:.3f}s"
            )

    def _complete_after_live_render(self) -> None:
        if self.pending_completion is None:
            return
        execution, output, provenance = self.pending_completion
        self.pending_completion = None
        self.running = False
        self._show_complete(execution, output, provenance)
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.play_button.configure(
            state=tk.NORMAL,
            text="Replay real time (1×)",
        )
        self.save_button.configure(state=tk.NORMAL)

    def _tick(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "adaptation_working":
                    self.status_var.set(str(payload))
                    self._append_log(str(payload))
                elif kind == "adaptation_result":
                    adaptation_result, estimate = payload  # type: ignore[misc]
                    self._show_adaptation_result(adaptation_result, estimate)
                elif kind == "live_update":
                    self._show_live_update(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self.running = False
                    self.pending_completion = None
                    self.run_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    if self.cancel_event.is_set():
                        self.outcome_var.set("Stopped")
                        self.status_var.set("Execution stopped")
                    else:
                        self.outcome_var.set("Execution failed")
                        self.status_var.set("Execution failed")
                        messagebox.showerror("Receding DDER-MPPI", str(payload), parent=self.root)
                elif kind == "complete":
                    execution, output, provenance = payload  # type: ignore[misc]
                    self.pending_completion = (execution, output, provenance)
                    self.status_var.set(
                        "MPC execution is complete; finishing the step-by-step live render."
                    )
        except queue.Empty:
            pass

        if self.running and self.live_view_result is not None:
            frame = self.canvas.frame_index
            if frame < self.live_target_frame:
                next_frame = frame + 1
                self.timeline_var.set(float(next_frame))
                self.canvas.set_trajectory_frame(next_frame)
            elif self.pending_completion is not None:
                self._complete_after_live_render()
        elif self.playing and self.view_result is not None:
            time_values = self.view_result.prediction.time_s
            assert self.replay_wall_start_s is not None
            replay_time = self.replay_sim_start_s + (
                time.perf_counter() - self.replay_wall_start_s
            )
            if replay_time >= float(time_values[-1]):
                final_frame = self.view_result.prediction.frame_count - 1
                self.timeline_var.set(float(final_frame))
                self.canvas.set_trajectory_frame(final_frame)
                self.playing = False
                self.replay_wall_start_s = None
                self.play_button.configure(text="Replay real time (1×)")
            else:
                next_frame = int(np.searchsorted(time_values, replay_time, side="right") - 1)
                next_frame = max(next_frame, 0)
                if next_frame != self.canvas.frame_index:
                    self.timeline_var.set(float(next_frame))
                    self.canvas.set_trajectory_frame(next_frame)
        self.root.after(self.TICK_MS, self._tick)

    def close(self) -> None:
        self.cancel_event.set()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    RecedingMppiGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
