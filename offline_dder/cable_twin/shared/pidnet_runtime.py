from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time
import tomllib

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIDNET_SOURCE = PROJECT_ROOT / "NN_collection_training" / "source"
CANONICAL_RUNTIME_CONFIG = (PIDNET_SOURCE / "config.toml").resolve()
CANONICAL_RUNTIME_CHECKPOINT = (
    PROJECT_ROOT / "data" / "models" / "pidnet_two_cable_best.pt"
).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PidnetResult:
    masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    body_component_count: int
    inference_ms: float
    postprocess_ms: float


class PidnetRuntime:
    """Canonical three-channel PIDNet for the rectified left image."""

    def __init__(self, runtime_config_path: str | Path) -> None:
        runtime_path = Path(runtime_config_path).expanduser().resolve()
        if not runtime_path.is_file():
            raise FileNotFoundError(f"PIDNet runtime configuration not found: {runtime_path}")
        if runtime_path != CANONICAL_RUNTIME_CONFIG:
            raise ValueError(
                "Phase 1 must use the PIDNet runtime configuration written by the "
                f"collection GUI: {CANONICAL_RUNTIME_CONFIG}"
            )
        with runtime_path.open("rb") as stream:
            values = tomllib.load(stream)
        pidnet_values = values.get("pidnet")
        if not isinstance(pidnet_values, dict):
            raise ValueError("PIDNet runtime configuration is missing [pidnet].")
        checkpoint = Path(str(pidnet_values["checkpoint"])).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = PROJECT_ROOT / checkpoint
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PIDNet checkpoint not found: {checkpoint}")
        if checkpoint != CANONICAL_RUNTIME_CHECKPOINT:
            raise ValueError(
                "The shared PIDNet config must point to the canonical runtime "
                f"checkpoint: {CANONICAL_RUNTIME_CHECKPOINT}"
            )
        configured_checkpoint_sha256 = str(
            pidnet_values.get("checkpoint_sha256", "")
        ).strip().upper()
        actual_checkpoint_sha256 = _sha256(checkpoint).upper()
        if configured_checkpoint_sha256 != actual_checkpoint_sha256:
            raise ValueError(
                "PIDNet runtime checkpoint does not match the model calibrated by the "
                "collection GUI. Run Calibrate + Save Runtime again."
            )

        source_text = str(PIDNET_SOURCE)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        try:
            from cable_detection import PidNetMaskConfig, postprocess_pidnet_masks
            from cable_pidnet import PidNetSegmenter
            from pidnet_preprocessing import parse_image_size
        except ImportError as error:
            raise RuntimeError(
                "PIDNet dependencies are unavailable. Run Phase 1 in the CUDA Python "
                "environment used by NN_collection_training."
            ) from error

        self.runtime_path = runtime_path
        self.checkpoint_path = checkpoint
        self.mask_config = PidNetMaskConfig.from_mapping(values)
        self._postprocess = postprocess_pidnet_masks
        self.segmenter = PidNetSegmenter(
            checkpoint,
            device=str(pidnet_values.get("device", "cuda")),
            amp=bool(pidnet_values.get("amp", True)),
            channels_last=bool(pidnet_values.get("channels_last", True)),
        )
        training_size = parse_image_size(self.segmenter.config.get("imgsz", ""))
        if training_size != (1920, 1080):
            raise ValueError(
                "The PIDNet model was not trained for the required HD1080 input: "
                f"checkpoint imgsz={training_size[0]}x{training_size[1]}."
            )
        self.identity = {
            "runtime_config": str(runtime_path),
            "runtime_config_sha256": _sha256(runtime_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": actual_checkpoint_sha256,
            "thresholds": [float(value) for value in self.mask_config.thresholds],
            "mask_cleanup": {
                "min_area_px": int(self.mask_config.min_area_px),
                "open_kernel": int(self.mask_config.open_kernel),
                "close_kernel": int(self.mask_config.close_kernel),
            },
            "training_image_size_px": list(training_size),
        }

    def warm_up(self, height_px: int = 1080, width_px: int = 1920) -> float:
        """Materialize CUDA kernels and cuDNN plans before opening the camera."""

        dummy = np.zeros((1, int(height_px), int(width_px), 3), dtype=np.uint8)
        start = time.perf_counter()
        self.segmenter.mask_channels_batch(
            dummy,
            channels=(0, 1, 2),
            thresholds=self.mask_config.thresholds,
        )
        return float((time.perf_counter() - start) * 1000.0)

    def infer(self, left_bgr: np.ndarray) -> PidnetResult:
        """Segment the rectified left view once for either observation path."""

        image = np.asarray(left_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("PIDNet input must be an HxWx3 BGR image.")
        start = time.perf_counter()
        raw = self.segmenter.mask_channels_batch(
            np.array(image[None], dtype=np.uint8, order="C", copy=True),
            channels=(0, 1, 2),
            thresholds=self.mask_config.thresholds,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        masks, components = self._postprocess(raw[0], self.mask_config)
        postprocess_ms = (time.perf_counter() - start) * 1000.0
        return PidnetResult(
            masks=tuple(masks),
            body_component_count=int(components),
            inference_ms=float(inference_ms),
            postprocess_ms=float(postprocess_ms),
        )
