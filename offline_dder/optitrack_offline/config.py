"""Measured cable and optimizer configuration for OptiTrack identification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import uuid


PACKAGE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIRECTORY / "config.json"
DEFAULT_MODEL_PATH = PACKAGE_DIRECTORY / "models" / "cable_model.json"
CONFIG_SCHEMA = "optitrack_one_attachment_fit_config_v1"
FIXED_CB_MAX_N_M2_S = 1.5e-5


@dataclass(frozen=True, slots=True)
class CableSpecification:
    rest_lengths_m: tuple[float, ...]
    bare_cable_mass_kg: float
    moving_marker_masses_kg: tuple[float, ...]
    diameter_m: float
    rod_segments_per_marker_interval: int = 2

    def __post_init__(self) -> None:
        rest_lengths = tuple(float(value) for value in self.rest_lengths_m)
        if len(rest_lengths) < 5:
            raise ValueError("The instrumented cable requires at least six markers.")
        values = rest_lengths + (
            float(self.bare_cable_mass_kg),
            float(self.diameter_m),
        ) + tuple(float(value) for value in self.moving_marker_masses_kg)
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Cable dimensions and masses must be finite and positive.")
        if not 1 <= self.rod_segments_per_marker_interval <= 8:
            raise ValueError(
                "DER segments per measured marker interval must be between 1 and 8."
            )
        marker_masses = tuple(float(value) for value in self.moving_marker_masses_kg)
        if len(marker_masses) != len(rest_lengths):
            raise ValueError(
                "Moving marker masses must contain one value for c1 through the free-tip marker."
            )
        object.__setattr__(self, "rest_lengths_m", rest_lengths)
        object.__setattr__(self, "moving_marker_masses_kg", marker_masses)

    @property
    def marker_count(self) -> int:
        return len(self.rest_lengths_m) + 1

    @property
    def length_m(self) -> float:
        return sum(self.rest_lengths_m)

    @property
    def moving_marker_count(self) -> int:
        return self.marker_count - 1

    @property
    def total_dynamic_mass_kg(self) -> float:
        """Cable plus all moving marker masses; the attachment body is prescribed."""

        return self.bare_cable_mass_kg + sum(self.moving_marker_masses_kg)

    @property
    def node_count(self) -> int:
        return (
            (self.marker_count - 1) * self.rod_segments_per_marker_interval + 1
        )

    @property
    def marker_node_indices(self) -> tuple[int, ...]:
        subdivision = self.rod_segments_per_marker_interval
        return tuple(range(0, self.node_count, subdivision))

    @property
    def marker_material_coordinates_m(self) -> tuple[float, ...]:
        coordinates = [0.0]
        for length in self.rest_lengths_m:
            coordinates.append(coordinates[-1] + length)
        return tuple(coordinates)

    @property
    def rod_rest_lengths_m(self) -> tuple[float, ...]:
        subdivision = self.rod_segments_per_marker_interval
        return tuple(
            length / subdivision
            for length in self.rest_lengths_m
            for _ in range(subdivision)
        )

    @property
    def rod_material_coordinates_m(self) -> tuple[float, ...]:
        coordinates = [0.0]
        for length in self.rod_rest_lengths_m:
            coordinates.append(coordinates[-1] + length)
        return tuple(coordinates)

    @property
    def vertex_masses_kg(self) -> tuple[float, ...]:
        density = self.bare_cable_mass_kg / self.length_m
        rest_lengths = self.rod_rest_lengths_m
        cable_mass = [0.0] * self.node_count
        cable_mass[0] = 0.5 * density * rest_lengths[0]
        cable_mass[-1] = 0.5 * density * rest_lengths[-1]
        for index in range(1, self.node_count - 1):
            cable_mass[index] = 0.5 * density * (
                rest_lengths[index - 1] + rest_lengths[index]
            )
        for index, marker_mass in zip(
            self.marker_node_indices[1:], self.moving_marker_masses_kg
        ):
            cable_mass[index] += marker_mass
        return tuple(cable_mass)


@dataclass(frozen=True, slots=True)
class FitSettings:
    device: str
    optimizer_iterations: int
    window_frames: int
    window_stride: int
    velocity_window_frames: int
    robust_scale_m: float
    ei_min_n_m2: float
    ei_max_n_m2: float
    cb_min_n_m2_s: float
    cb_max_n_m2_s: float
    substeps: int
    constraint_iterations: int
    maximum_rigid_body_error_m: float = 0.002
    maximum_attachment_speed_m_s: float = 3.0
    maximum_marker_speed_m_s: float = 3.0
    maximum_chord_excess_m: float = 0.003
    maximum_initialization_rmse_m: float = 0.002
    initializer_candidates: int = 24
    initializer_windows: int = 12
    optimizer_batch_windows: int = 64
    optimizer_learning_rate: float = 0.02
    optimizer_seed: int = 42
    optimizer_gradient_clip: float = 10.0

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Fit device must be cpu or cuda.")
        if not 1 <= self.optimizer_iterations <= 1000:
            raise ValueError("Optimizer iterations must be between 1 and 1000.")
        if not 2 <= self.initializer_candidates <= 512:
            raise ValueError("Initializer candidates must be between 2 and 512.")
        if not 1 <= self.initializer_windows <= 512:
            raise ValueError("Initializer windows must be between 1 and 512.")
        if not 1 <= self.optimizer_batch_windows <= 512:
            raise ValueError("Optimizer batch size must be between 1 and 512 windows.")
        if self.optimizer_seed < 0:
            raise ValueError("Optimizer seed must be non-negative.")
        if self.window_frames < 3 or self.window_stride < 1:
            raise ValueError("Fit windows require at least three frames and positive stride.")
        if self.velocity_window_frames < 3 or self.velocity_window_frames % 2 == 0:
            raise ValueError("Velocity window must be odd and at least three frames.")
        positive = (
            self.robust_scale_m,
            self.ei_min_n_m2,
            self.ei_max_n_m2,
            self.cb_min_n_m2_s,
            self.cb_max_n_m2_s,
            self.maximum_rigid_body_error_m,
            self.maximum_attachment_speed_m_s,
            self.maximum_marker_speed_m_s,
            self.maximum_chord_excess_m,
            self.maximum_initialization_rmse_m,
            self.optimizer_learning_rate,
            self.optimizer_gradient_clip,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Fit scales and parameter bounds must be finite and positive.")
        if self.ei_min_n_m2 >= self.ei_max_n_m2:
            raise ValueError("EI bounds must increase.")
        if self.cb_min_n_m2_s >= self.cb_max_n_m2_s:
            raise ValueError("Cb bounds must increase.")
        if not math.isclose(
            self.cb_max_n_m2_s,
            FIXED_CB_MAX_N_M2_S,
            rel_tol=0.0,
            abs_tol=1.0e-16,
        ):
            raise ValueError(
                f"Cb upper bound is fixed at {FIXED_CB_MAX_N_M2_S:g} N m^2 s."
            )
        if min(self.substeps, self.constraint_iterations) < 1:
            raise ValueError("Rod solver counts must be positive.")


@dataclass(frozen=True, slots=True)
class ValidationSettings:
    history_frames: int = 5

    def __post_init__(self) -> None:
        if self.history_frames < 3:
            raise ValueError("Validation history must contain at least three frames.")


@dataclass(frozen=True, slots=True)
class OptitrackFitConfig:
    cable: CableSpecification
    fit: FitSettings
    model_path: Path = DEFAULT_MODEL_PATH
    validation: ValidationSettings = field(default_factory=ValidationSettings)


DEFAULT_CONFIG = OptitrackFitConfig(
    cable=CableSpecification(
        rest_lengths_m=(0.060, 0.085, 0.100, 0.100, 0.098, 0.100, 0.100, 0.100, 0.100, 0.100),
        bare_cable_mass_kg=0.007,
        moving_marker_masses_kg=((0.017 - 0.007) / 11.0,) * 10,
        diameter_m=0.0035,
        rod_segments_per_marker_interval=2,
    ),
    fit=FitSettings(
        device="cuda",
        optimizer_iterations=20,
        window_frames=100,
        window_stride=100,
        velocity_window_frames=5,
        robust_scale_m=0.002,
        ei_min_n_m2=1.0e-8,
        ei_max_n_m2=4.0e-4,
        cb_min_n_m2_s=1.0e-10,
        cb_max_n_m2_s=FIXED_CB_MAX_N_M2_S,
        substeps=3,
        constraint_iterations=4,
    ),
    validation=ValidationSettings(),
)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_config(config: OptitrackFitConfig, path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    destination = Path(path).expanduser().resolve()
    model_path = config.model_path.expanduser().resolve()
    try:
        stored_model_path = str(model_path.relative_to(destination.parent))
    except ValueError:
        stored_model_path = str(model_path)
    _atomic_json(
        destination,
        {
            "schema": CONFIG_SCHEMA,
            "cable": asdict(config.cable),
            "fit": asdict(config.fit),
            "validation": asdict(config.validation),
            "model_path": stored_model_path,
        },
    )


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> OptitrackFitConfig:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        save_config(DEFAULT_CONFIG, source)
        return DEFAULT_CONFIG
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(
            "OptiTrack configuration predates the one-attachment/free-tip "
            "measurement contract. Recreate it with the current application."
        )
    cable = payload["cable"]
    fit = payload["fit"]
    validation = payload.get("validation", {})
    model_path = Path(payload["model_path"]).expanduser()
    if not model_path.is_absolute():
        model_path = source.parent / model_path
    return OptitrackFitConfig(
        cable=CableSpecification(
            rest_lengths_m=tuple(float(value) for value in cable["rest_lengths_m"]),
            bare_cable_mass_kg=float(cable["bare_cable_mass_kg"]),
            moving_marker_masses_kg=tuple(
                float(value) for value in cable["moving_marker_masses_kg"]
            ),
            diameter_m=float(cable["diameter_m"]),
            rod_segments_per_marker_interval=int(cable["rod_segments_per_marker_interval"]),
        ),
        fit=FitSettings(**fit),
        model_path=model_path.resolve(),
        validation=ValidationSettings(
            history_frames=int(validation.get("history_frames", 5)),
        ),
    )
