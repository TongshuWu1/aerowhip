from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True, slots=True)
class CableSettings:
    length_m: float
    mass_kg: float
    diameter_m: float
    node_count: int


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    device: str
    optimizer_iterations: int
    maximum_dt_s: float
    window_frames: int
    window_stride: int
    velocity_window_frames: int
    robust_scale_m: float
    ei_min_n_m2: float
    ei_max_n_m2: float
    cb_min_n_m2_s: float
    cb_max_n_m2_s: float


@dataclass(frozen=True, slots=True)
class SolverSettings:
    substeps: int
    constraint_iterations: int


@dataclass(frozen=True, slots=True)
class OfflineSettings:
    recording_directory: Path
    observation_directory: Path
    model_directory: Path
    cable: CableSettings
    optimization: OptimizationSettings
    solver: SolverSettings


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> OfflineSettings:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        values = tomllib.load(stream)
    paths, cable = values["paths"], values["cable"]
    fit, solver = values["optimization"], values["solver"]
    settings = OfflineSettings(
        recording_directory=_project_path(paths["recording_directory"]),
        observation_directory=_project_path(paths["observation_directory"]),
        model_directory=_project_path(paths["model_directory"]),
        cable=CableSettings(
            length_m=float(cable["length_m"]),
            mass_kg=float(cable["mass_kg"]),
            diameter_m=float(cable["diameter_m"]),
            node_count=int(cable["node_count"]),
        ),
        optimization=OptimizationSettings(
            device=str(fit["device"]),
            optimizer_iterations=int(fit["optimizer_iterations"]),
            maximum_dt_s=float(fit["maximum_dt_s"]),
            window_frames=int(fit["window_frames"]),
            window_stride=int(fit["window_stride"]),
            velocity_window_frames=int(fit["velocity_window_frames"]),
            robust_scale_m=float(fit["robust_scale_m"]),
            ei_min_n_m2=float(fit["ei_min_n_m2"]),
            ei_max_n_m2=float(fit["ei_max_n_m2"]),
            cb_min_n_m2_s=float(fit["cb_min_n_m2_s"]),
            cb_max_n_m2_s=float(fit["cb_max_n_m2_s"]),
        ),
        solver=SolverSettings(
            substeps=int(solver["substeps"]),
            constraint_iterations=int(solver["constraint_iterations"]),
        ),
    )
    _validate(settings)
    return settings


def _validate(settings: OfflineSettings) -> None:
    cable = settings.cable
    if cable.node_count < 6:
        raise ValueError("Cable node_count must be at least six.")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (cable.length_m, cable.mass_kg, cable.diameter_m)
    ):
        raise ValueError("Cable length, mass, and diameter must be finite and positive.")
    fit = settings.optimization
    positive = (
        fit.maximum_dt_s,
        fit.robust_scale_m,
        fit.ei_min_n_m2,
        fit.ei_max_n_m2,
        fit.cb_min_n_m2_s,
        fit.cb_max_n_m2_s,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("Optimization scales and bounds must be finite and positive.")
    if fit.optimizer_iterations < 3:
        raise ValueError("Optimizer iterations must be at least 3.")
    if fit.window_frames < 3 or fit.window_stride < 1:
        raise ValueError("Trajectory windows require at least three frames and positive stride.")
    if fit.velocity_window_frames < 3 or fit.velocity_window_frames % 2 == 0:
        raise ValueError("Velocity fit window must be odd and at least three frames.")
    if fit.ei_min_n_m2 >= fit.ei_max_n_m2:
        raise ValueError("EI bounds must be increasing.")
    if fit.cb_min_n_m2_s >= fit.cb_max_n_m2_s:
        raise ValueError("Cb bounds must be increasing.")
    if min(settings.solver.substeps, settings.solver.constraint_iterations) < 1:
        raise ValueError("Constrained-rod solver counts must be positive.")


def save_cable_settings(
    path: str | Path,
    *,
    length_m: float,
    mass_kg: float,
    diameter_m: float,
) -> None:
    """Update only the measured cable values in the small TOML file."""

    values = (float(length_m), float(mass_kg), float(diameter_m))
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Cable values must be finite and positive.")
    config_path = Path(path).expanduser().resolve()
    lines = config_path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "length_m": values[0],
        "mass_kg": values[1],
        "diameter_m": values[2],
    }
    section = ""
    updated: set[str] = set()
    output: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if section == "cable" and key in replacements:
            output.append(f"{key} = {replacements[key]:.9g}")
            updated.add(key)
        else:
            output.append(line)
    if updated != set(replacements):
        raise ValueError("Cable section is incomplete in cable_twin/offline/config.toml.")
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
    temporary.replace(config_path)
