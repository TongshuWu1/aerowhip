"""Strict configuration for the camera/perception runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "cable_twin" / "config.toml"


def _mapping(root: dict, name: str) -> dict:
    value = root.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration requires [{name}]")
    return value


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True, slots=True)
class CameraSettings:
    resolution: str
    fps: int
    depth_mode: str
    confidence: int
    texture_confidence: int
    fill: bool
    disable_self_calibration: bool

    def __post_init__(self) -> None:
        if self.resolution != "HD1080":
            raise ValueError("The canonical runtime resolution is HD1080")
        if self.fps != 30:
            raise ValueError("The canonical runtime frame rate is 30 FPS")
        if self.depth_mode not in {"NEURAL", "NEURAL_PLUS", "NEURAL_LIGHT"}:
            raise ValueError("Unsupported ZED depth mode")
        if not self.disable_self_calibration:
            raise ValueError(
                "Canonical live/SVO calibration requires self-calibration disabled"
            )
        for name, value in (
            ("confidence", self.confidence),
            ("texture_confidence", self.texture_confidence),
        ):
            if not 0 <= value <= 100:
                raise ValueError(f"camera.{name} must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class SvoSettings:
    recording_directory: Path
    compression: str
    playback_realtime: bool

    def __post_init__(self) -> None:
        if self.compression != "LOSSLESS":
            raise ValueError("The canonical SVO compression is LOSSLESS")


@dataclass(frozen=True, slots=True)
class PidNetSettings:
    runtime_config: Path


@dataclass(frozen=True, slots=True)
class ViewerSettings:
    enabled: bool
    window_name: str
    width_px: int
    height_px: int
    depth_min_m: float
    depth_max_m: float
    overlay_alpha: float
    point_cloud_stride: int
    point_size_px: float
    inset_width_px: int
    render_fps: float

    def __post_init__(self) -> None:
        if not self.window_name.strip():
            raise ValueError("viewer.window_name must be non-empty")
        if self.width_px < 640:
            raise ValueError("viewer.width_px must be at least 640")
        if self.height_px < 480:
            raise ValueError("viewer.height_px must be at least 480")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("viewer depth range must satisfy 0 < min < max")
        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ValueError("viewer.overlay_alpha must be in [0, 1]")
        if not 1 <= self.point_cloud_stride <= 16:
            raise ValueError("viewer.point_cloud_stride must be in [1, 16]")
        if not 1.0 <= self.point_size_px <= 10.0:
            raise ValueError("viewer.point_size_px must be in [1, 10]")
        if not 160 <= self.inset_width_px < self.width_px:
            raise ValueError(
                "viewer.inset_width_px must be at least 160 and less than width_px"
            )
        if not 1.0 <= self.render_fps <= 120.0:
            raise ValueError("viewer.render_fps must be in [1, 120]")


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    camera: CameraSettings
    svo: SvoSettings
    pidnet: PidNetSettings
    viewer: ViewerSettings
    status_period_s: float


def load_settings(path: Path = DEFAULT_CONFIG_PATH) -> RuntimeSettings:
    with Path(path).open("rb") as stream:
        root = tomllib.load(stream)
    camera = _mapping(root, "camera")
    svo = _mapping(root, "svo")
    pidnet = _mapping(root, "pidnet")
    viewer = _mapping(root, "viewer")
    runtime = _mapping(root, "runtime")
    settings = RuntimeSettings(
        camera=CameraSettings(
            resolution=str(camera["resolution"]).upper(),
            fps=int(camera["fps"]),
            depth_mode=str(camera["depth_mode"]).upper(),
            confidence=int(camera["confidence"]),
            texture_confidence=int(camera["texture_confidence"]),
            fill=bool(camera["fill"]),
            disable_self_calibration=bool(camera["disable_self_calibration"]),
        ),
        svo=SvoSettings(
            recording_directory=project_path(svo["recording_directory"]),
            compression=str(svo["compression"]).upper(),
            playback_realtime=bool(svo["playback_realtime"]),
        ),
        pidnet=PidNetSettings(
            runtime_config=project_path(pidnet["runtime_config"]),
        ),
        viewer=ViewerSettings(
            enabled=bool(viewer["enabled"]),
            window_name=str(viewer["window_name"]),
            width_px=int(viewer["width_px"]),
            height_px=int(viewer["height_px"]),
            depth_min_m=float(viewer["depth_min_m"]),
            depth_max_m=float(viewer["depth_max_m"]),
            overlay_alpha=float(viewer["overlay_alpha"]),
            point_cloud_stride=int(viewer["point_cloud_stride"]),
            point_size_px=float(viewer["point_size_px"]),
            inset_width_px=int(viewer["inset_width_px"]),
            render_fps=float(viewer["render_fps"]),
        ),
        status_period_s=float(runtime["status_period_s"]),
    )
    if settings.status_period_s <= 0.0:
        raise ValueError("runtime.status_period_s must be positive")
    if not settings.pidnet.runtime_config.is_file():
        raise FileNotFoundError(
            f"PIDNet runtime configuration not found: {settings.pidnet.runtime_config}"
        )
    return settings
