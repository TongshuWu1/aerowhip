from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True, slots=True)
class OnlineCableSettings:
    length_m: float
    diameter_m: float
    node_count: int
    solver_substeps: int
    constraint_iterations: int


@dataclass(frozen=True, slots=True)
class ParticleFilterSettings:
    device: str
    particle_count: int
    random_seed: int
    maximum_dt_s: float
    maximum_gap_s: float
    ess_resample_fraction: float
    process_acceleration_sigma_m_s2: float = 0.50
    endpoint_acceleration_sigma_m_s2: float = 1.00
    metric_curve_noise_m: float = 0.003
    metric_endpoint_noise_m: float = 0.003
    body_pixel_sigma_px: float = 4.0
    endpoint_pixel_sigma_px: float = 6.0
    student_t_degrees_of_freedom: float = 4.0


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> ParticleFilterSettings:
    with Path(path).expanduser().resolve().open("rb") as stream:
        values = tomllib.load(stream)["filter"]
    settings = ParticleFilterSettings(
        device=str(values["device"]),
        particle_count=int(values["particle_count"]),
        random_seed=int(values["random_seed"]),
        maximum_dt_s=float(values["maximum_dt_s"]),
        maximum_gap_s=float(values["maximum_gap_s"]),
        ess_resample_fraction=float(values["ess_resample_fraction"]),
        process_acceleration_sigma_m_s2=float(
            values["process_acceleration_sigma_m_s2"]
        ),
        endpoint_acceleration_sigma_m_s2=float(
            values["endpoint_acceleration_sigma_m_s2"]
        ),
        metric_curve_noise_m=float(values["metric_curve_noise_m"]),
        metric_endpoint_noise_m=float(values["metric_endpoint_noise_m"]),
        body_pixel_sigma_px=float(values["body_pixel_sigma_px"]),
        endpoint_pixel_sigma_px=float(values["endpoint_pixel_sigma_px"]),
        student_t_degrees_of_freedom=float(
            values["student_t_degrees_of_freedom"]
        ),
    )
    positive = (
        settings.maximum_dt_s,
        settings.maximum_gap_s,
        settings.process_acceleration_sigma_m_s2,
        settings.endpoint_acceleration_sigma_m_s2,
        settings.metric_curve_noise_m,
        settings.metric_endpoint_noise_m,
        settings.body_pixel_sigma_px,
        settings.endpoint_pixel_sigma_px,
        settings.student_t_degrees_of_freedom,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("PF time and noise scales must be finite and positive.")
    if settings.particle_count < 16:
        raise ValueError("Use at least 16 cable particles.")
    if not 0.0 < settings.ess_resample_fraction <= 1.0:
        raise ValueError("ESS resampling fraction must be in (0, 1].")
    if settings.maximum_gap_s < settings.maximum_dt_s:
        raise ValueError("PF maximum gap must not be shorter than its integration step.")
    return settings


def load_cable_settings(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> OnlineCableSettings:
    with Path(path).expanduser().resolve().open("rb") as stream:
        values = tomllib.load(stream)["cable"]
    settings = OnlineCableSettings(
        length_m=float(values["length_m"]),
        diameter_m=float(values["diameter_m"]),
        node_count=int(values["node_count"]),
        solver_substeps=int(values["solver_substeps"]),
        constraint_iterations=int(values["constraint_iterations"]),
    )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (settings.length_m, settings.diameter_m)
    ):
        raise ValueError("Online cable dimensions must be finite and positive.")
    if settings.diameter_m >= settings.length_m:
        raise ValueError("Online cable diameter must be smaller than its length.")
    if settings.node_count < 6:
        raise ValueError("Online constrained rod requires at least six nodes.")
    if min(settings.solver_substeps, settings.constraint_iterations) < 1:
        raise ValueError("Online cable solver counts must be positive.")
    return settings
