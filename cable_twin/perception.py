"""One-pass compatibility adapter for the preserved PIDNet runtime.

This module is the only new-runtime boundary that knows how the preserved
``NN_collection_training/source`` directory is laid out.  The model,
preprocessing, checkpoint schema, thresholds, and mask cleanup remain owned by
that preserved implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import sys
import time
import tomllib
import warnings

import numpy as np
import torch

from .frames import RgbdFrame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NN_SOURCE_DIR = PROJECT_ROOT / "NN_collection_training" / "source"
DEFAULT_RUNTIME_CONFIG_PATH = NN_SOURCE_DIR / "config.toml"
HD1080_WIDTH_PX = 1920
HD1080_HEIGHT_PX = 1080


# The preserved files intentionally use sibling modules as top-level imports.
# Keep that legacy import accommodation local to this compatibility boundary,
# and remove the path entry again once the preserved modules are loaded.
_nn_source_text = str(NN_SOURCE_DIR)
_inserted_nn_source = _nn_source_text not in sys.path
if _inserted_nn_source:
    sys.path.insert(0, _nn_source_text)
try:
    from cable_detection import PidNetMaskConfig, postprocess_pidnet_masks
    from cable_pidnet import PidNetSegmenter, probability_to_logit_threshold
finally:
    if _inserted_nn_source:
        sys.path.remove(_nn_source_text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _required_bool(values: dict, key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"pidnet.{key} must be a boolean")
    return value


def _validate_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class PerceptionTiming:
    """Wall-clock stage timings for one synchronized perception result."""

    inference_ms: float
    mask_transfer_ms: float
    postprocess_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        for name, value in (
            ("inference_ms", self.inference_ms),
            ("mask_transfer_ms", self.mask_transfer_ms),
            ("postprocess_ms", self.postprocess_ms),
            ("total_ms", self.total_ms),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    """PIDNet evidence aligned one-to-one with an immutable RGB-D frame.

    ``logits_cuda`` contains the three unactivated network channels in the
    canonical order: shared cable body, cable-1 endpoint set, cable-2 endpoint
    set.  ``masks_u8`` follows the same order on CPU; the body mask has the
    canonical morphology/component cleanup while endpoint masks are exactly
    thresholded.
    """

    rgbd: RgbdFrame
    logits_cuda: torch.Tensor
    masks_u8: tuple[np.ndarray, np.ndarray, np.ndarray]
    body_component_count: int
    timing: PerceptionTiming
    checkpoint_sha256: str
    runtime_config_sha256: str

    def __post_init__(self) -> None:
        height, width = self.rgbd.bgr_u8.shape[:2]
        if not isinstance(self.logits_cuda, torch.Tensor):
            raise TypeError("logits_cuda must be a torch.Tensor")
        if self.logits_cuda.device.type != "cuda":
            raise ValueError("logits_cuda must remain on CUDA")
        if not self.logits_cuda.is_floating_point():
            raise ValueError("logits_cuda must contain floating-point logits")
        if self.logits_cuda.ndim != 3 or tuple(self.logits_cuda.shape) != (3, height, width):
            raise ValueError(
                "logits_cuda must have shape "
                f"(3, {height}, {width}); got {tuple(self.logits_cuda.shape)}"
            )
        if self.logits_cuda.requires_grad:
            raise ValueError("logits_cuda must be produced under inference mode")
        if len(self.masks_u8) != 3:
            raise ValueError("masks_u8 must contain exactly three channel masks")
        for channel, mask in enumerate(self.masks_u8):
            if not isinstance(mask, np.ndarray):
                raise TypeError(f"masks_u8[{channel}] must be a NumPy array")
            if mask.dtype != np.uint8 or mask.shape != (height, width):
                raise ValueError(
                    f"masks_u8[{channel}] must be uint8 with shape {(height, width)}"
                )
            if not mask.flags.c_contiguous:
                raise ValueError(f"masks_u8[{channel}] must be C-contiguous")
            mask.setflags(write=False)
        if self.body_component_count < 0:
            raise ValueError("body_component_count must be nonnegative")
        _validate_sha256("checkpoint_sha256", self.checkpoint_sha256)
        _validate_sha256("runtime_config_sha256", self.runtime_config_sha256)

    @property
    def body_mask_u8(self) -> np.ndarray:
        return self.masks_u8[0]

    @property
    def endpoint1_mask_u8(self) -> np.ndarray:
        return self.masks_u8[1]

    @property
    def endpoint2_mask_u8(self) -> np.ndarray:
        return self.masks_u8[2]


class PerceptionRuntime:
    """Load and run the canonical preserved PIDNet configuration once/frame."""

    def __init__(
        self,
        runtime_config_path: str | Path = DEFAULT_RUNTIME_CONFIG_PATH,
    ) -> None:
        self.runtime_config_path = _project_path(runtime_config_path)
        if not self.runtime_config_path.is_file():
            raise FileNotFoundError(
                f"PIDNet runtime configuration not found: {self.runtime_config_path}"
            )
        self.runtime_config_sha256 = _sha256(self.runtime_config_path)
        with self.runtime_config_path.open("rb") as stream:
            runtime_values = tomllib.load(stream)
        pidnet_values = runtime_values.get("pidnet")
        if not isinstance(pidnet_values, dict):
            raise ValueError("PIDNet runtime configuration requires [pidnet]")

        checkpoint_value = pidnet_values.get("checkpoint")
        if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
            raise ValueError("pidnet.checkpoint must be a non-empty path")
        self.checkpoint_path = _project_path(checkpoint_value)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"PIDNet checkpoint not found: {self.checkpoint_path}")
        self.checkpoint_sha256 = _sha256(self.checkpoint_path)

        self.mask_config = PidNetMaskConfig.from_mapping(runtime_values)
        self._segmenter = PidNetSegmenter(
            self.checkpoint_path,
            device=str(pidnet_values.get("device", "")),
            amp=_required_bool(pidnet_values, "amp"),
            channels_last=_required_bool(pidnet_values, "channels_last"),
        )
        if self._segmenter.device.type != "cuda":
            raise RuntimeError("The canonical PIDNet runtime must execute on CUDA")
        self._threshold_cache: dict[torch.dtype, torch.Tensor] = {}
        self._inference_started = torch.cuda.Event(enable_timing=True)
        self._inference_finished = torch.cuda.Event(enable_timing=True)

    @property
    def thresholds(self) -> tuple[float, float, float]:
        return self.mask_config.thresholds

    @property
    def device(self) -> torch.device:
        """CUDA device used by the preserved PIDNet runtime."""

        return self._segmenter.device

    @property
    def checkpoint_schema(self) -> dict[str, object]:
        """Copy of the architecture metadata stored in the checkpoint."""

        return dict(self._segmenter.config)

    def _threshold_logits(self, dtype: torch.dtype) -> torch.Tensor:
        cached = self._threshold_cache.get(dtype)
        if cached is None:
            cached = torch.tensor(
                [probability_to_logit_threshold(value) for value in self.thresholds],
                dtype=dtype,
                device=self._segmenter.device,
            ).view(3, 1, 1)
            self._threshold_cache[dtype] = cached
        return cached

    @staticmethod
    def _require_hd1080(bgr_u8: np.ndarray) -> None:
        expected = (HD1080_HEIGHT_PX, HD1080_WIDTH_PX, 3)
        if bgr_u8.shape != expected:
            raise ValueError(
                f"PIDNet runtime requires an HD1080 BGR frame with shape {expected}; "
                f"got {bgr_u8.shape}. Do not resize registered RGB-D inside perception."
            )

    @torch.inference_mode()
    def _infer_arrays(
        self,
        bgr_u8: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        tuple[np.ndarray, np.ndarray, np.ndarray],
        int,
        PerceptionTiming,
    ]:
        self._require_hd1080(bgr_u8)
        total_started = time.perf_counter_ns()
        self._inference_started.record()
        # The preserved runtime has no public CUDA-logit method.  This is the
        # single protected compatibility call; it performs the authoritative
        # BGR->RGB conversion, ImageNet normalization, AMP, and model forward.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable.*",
                category=UserWarning,
            )
            batched_logits = self._segmenter._forward_logits(bgr_u8)
        self._inference_finished.record()

        expected_logits_shape = (1, 3, HD1080_HEIGHT_PX, HD1080_WIDTH_PX)
        if (
            batched_logits.device.type != "cuda"
            or tuple(batched_logits.shape) != expected_logits_shape
        ):
            raise RuntimeError(
                f"PIDNet must return CUDA logits with shape {expected_logits_shape}; "
                f"got shape={tuple(batched_logits.shape)} device={batched_logits.device}"
            )
        logits_cuda = batched_logits[0]
        transfer_started = time.perf_counter_ns()
        raw_masks_cpu = (
            (logits_cuda >= self._threshold_logits(logits_cuda.dtype))
            .to(torch.uint8)
            .mul_(255)
            .cpu()
            .numpy()
        )
        transfer_finished = time.perf_counter_ns()
        inference_ms = float(
            self._inference_started.elapsed_time(self._inference_finished)
        )
        transfer_wall_ms = (transfer_finished - transfer_started) / 1_000_000.0

        postprocess_started = time.perf_counter_ns()
        raw_masks = tuple(
            np.ascontiguousarray(raw_masks_cpu[channel], dtype=np.uint8)
            for channel in range(3)
        )
        masks, body_component_count = postprocess_pidnet_masks(
            raw_masks,
            self.mask_config,
        )
        postprocess_finished = time.perf_counter_ns()
        timing = PerceptionTiming(
            inference_ms=inference_ms,
            mask_transfer_ms=max(0.0, transfer_wall_ms - inference_ms),
            postprocess_ms=(postprocess_finished - postprocess_started) / 1_000_000.0,
            total_ms=(postprocess_finished - total_started) / 1_000_000.0,
        )
        return logits_cuda, masks, int(body_component_count), timing

    def infer(self, rgbd: RgbdFrame) -> PerceptionFrame:
        if not isinstance(rgbd, RgbdFrame):
            raise TypeError("rgbd must be an RgbdFrame")
        logits, masks, body_component_count, timing = self._infer_arrays(rgbd.bgr_u8)
        return PerceptionFrame(
            rgbd=rgbd,
            logits_cuda=logits,
            masks_u8=masks,
            body_component_count=body_component_count,
            timing=timing,
            checkpoint_sha256=self.checkpoint_sha256,
            runtime_config_sha256=self.runtime_config_sha256,
        )

    def warmup_1080p(self) -> PerceptionTiming:
        """Warm the complete one-pass CUDA/mask path at production resolution."""

        blank_bgr = np.zeros(
            (HD1080_HEIGHT_PX, HD1080_WIDTH_PX, 3),
            dtype=np.uint8,
        )
        _logits, _masks, _component_count, timing = self._infer_arrays(blank_bgr)
        return timing


__all__ = [
    "DEFAULT_RUNTIME_CONFIG_PATH",
    "HD1080_HEIGHT_PX",
    "HD1080_WIDTH_PX",
    "PerceptionFrame",
    "PerceptionTiming",
    "PerceptionRuntime",
]
