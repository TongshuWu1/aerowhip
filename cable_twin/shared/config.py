from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.toml")


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True, slots=True)
class RouteSettings:
    endpoint_min_area_px: int
    dense_samples: int
    maximum_segments: int
    minimum_segment_length_px: float


@dataclass(frozen=True, slots=True)
class DepthSettings:
    minimum_m: float
    maximum_m: float
    neighborhood_radius_px: int
    minimum_support: int
    maximum_cluster_span_m: float
    viewer_stride_px: int


@dataclass(frozen=True, slots=True)
class AppSettings:
    pidnet_runtime_config: Path
    cable_identity: int
    route: RouteSettings
    depth: DepthSettings
    viewer_width_px: int
    viewer_height_px: int


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> AppSettings:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as stream:
        values = tomllib.load(stream)
    route = values["route"]
    settings = AppSettings(
        pidnet_runtime_config=_project_path(values["pidnet"]["runtime_config"]),
        cable_identity=int(values["cable"]["identity"]),
        route=RouteSettings(
            endpoint_min_area_px=int(route["endpoint_min_area_px"]),
            dense_samples=int(route["dense_samples"]),
            maximum_segments=int(route["maximum_segments"]),
            minimum_segment_length_px=float(route["minimum_segment_length_px"]),
        ),
        depth=DepthSettings(
            minimum_m=float(values["depth"]["minimum_m"]),
            maximum_m=float(values["depth"]["maximum_m"]),
            neighborhood_radius_px=int(values["depth"]["neighborhood_radius_px"]),
            minimum_support=int(values["depth"]["minimum_support"]),
            maximum_cluster_span_m=float(values["depth"]["maximum_cluster_span_m"]),
            viewer_stride_px=int(values["depth"]["viewer_stride_px"]),
        ),
        viewer_width_px=int(values["viewer"]["width_px"]),
        viewer_height_px=int(values["viewer"]["height_px"]),
    )
    if settings.cable_identity not in (1, 2):
        raise ValueError("cable.identity must be 1 or 2.")
    if settings.route.endpoint_min_area_px < 1:
        raise ValueError("route.endpoint_min_area_px must be positive.")
    if settings.route.dense_samples < 24:
        raise ValueError("route.dense_samples must be at least 24.")
    if settings.route.maximum_segments < 1:
        raise ValueError("route.maximum_segments must be positive.")
    if settings.route.minimum_segment_length_px <= 0.0:
        raise ValueError("route.minimum_segment_length_px must be positive.")
    if not 0.0 < settings.depth.minimum_m < settings.depth.maximum_m:
        raise ValueError("depth range must satisfy 0 < minimum_m < maximum_m.")
    if settings.depth.neighborhood_radius_px < 1 or settings.depth.minimum_support < 2:
        raise ValueError("depth neighbourhood and minimum support are invalid.")
    if settings.depth.maximum_cluster_span_m <= 0.0:
        raise ValueError("depth.maximum_cluster_span_m must be positive.")
    if not 1 <= settings.depth.viewer_stride_px <= 16:
        raise ValueError("depth.viewer_stride_px must be between 1 and 16.")
    if settings.viewer_width_px < 900 or settings.viewer_height_px < 500:
        raise ValueError("Viewer dimensions are too small.")
    if not settings.pidnet_runtime_config.is_file():
        raise FileNotFoundError(
            f"PIDNet runtime configuration not found: {settings.pidnet_runtime_config}"
        )
    return settings
