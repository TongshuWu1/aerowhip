from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True, slots=True)
class ParticleFilterSettings:
    device: str
    particle_count: int
    random_seed: int
    maximum_dt_s: float
    maximum_gap_s: float
    ess_resample_fraction: float


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
    )
    positive = (
        settings.maximum_dt_s,
        settings.maximum_gap_s,
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
