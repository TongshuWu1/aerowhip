"""Canonical PIDNet thresholding and mask cleanup shared by every runtime."""

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=16)
def _ellipse_kernel(size):
    value = int(size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (value, value))


@dataclass(frozen=True)
class PidNetMaskConfig:
    """Complete runtime definition of the three thresholded PIDNet masks."""

    cable_threshold: float
    endpoint_thresholds: tuple[float, float]
    min_area_px: int
    open_kernel: int
    close_kernel: int

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, dict):
            raise ValueError("PIDNet runtime configuration must be a mapping.")
        pidnet = values.get("pidnet")
        detector = values.get("detector")
        if not isinstance(pidnet, dict) or not isinstance(detector, dict):
            raise ValueError("PIDNet runtime configuration requires [pidnet] and [detector].")
        endpoint_thresholds = tuple(
            float(value) for value in pidnet.get("endpoint_thresholds", ())
        )
        if len(endpoint_thresholds) != 2:
            raise ValueError("pidnet.endpoint_thresholds must contain exactly two values.")
        config = cls(
            cable_threshold=float(pidnet["threshold"]),
            endpoint_thresholds=endpoint_thresholds,
            min_area_px=int(detector["min_area_px"]),
            open_kernel=int(detector["open_kernel"]),
            close_kernel=int(detector["close_kernel"]),
        )
        config.validate()
        return config

    @property
    def thresholds(self):
        return (self.cable_threshold, *self.endpoint_thresholds)

    def validate(self):
        if len(self.endpoint_thresholds) != 2:
            raise ValueError("Exactly two endpoint thresholds are required.")
        for name, threshold in zip(
            ("cable", "endpoint1", "endpoint2"),
            self.thresholds,
        ):
            if not 0.0 < float(threshold) < 1.0:
                raise ValueError(f"PIDNet {name} threshold must be strictly between 0 and 1.")
        if self.min_area_px < 0:
            raise ValueError("detector.min_area_px cannot be negative.")
        for name, kernel in (
            ("open_kernel", self.open_kernel),
            ("close_kernel", self.close_kernel),
        ):
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError(f"detector.{name} must be a positive odd integer.")


def remove_small_components(mask, min_area=80):
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask), 0

    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            components.append((area, label))

    if not components:
        return np.zeros_like(mask), 0

    cleaned = np.zeros_like(mask)
    for _area, label in components:
        cleaned[labels == label] = 255
    return cleaned, len(components)


def clean_binary_mask(mask, min_area_px, open_kernel, close_kernel):
    """Apply canonical cleanup inside an exactly equivalent padded foreground ROI."""

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError(f"Binary mask must be two-dimensional; got {binary.shape}.")
    open_kernel = int(open_kernel)
    close_kernel = int(close_kernel)
    foreground_x, foreground_y, foreground_width, foreground_height = (
        cv2.boundingRect(binary)
    )
    if foreground_width <= 0 or foreground_height <= 0:
        return np.zeros(binary.shape, dtype=np.uint8), 0
    margin = open_kernel // 2 + close_kernel // 2 + 1
    x0 = max(0, int(foreground_x) - margin)
    x1 = min(binary.shape[1], int(foreground_x + foreground_width) + margin)
    y0 = max(0, int(foreground_y) - margin)
    y1 = min(binary.shape[0], int(foreground_y + foreground_height) + margin)
    cleaned = np.ascontiguousarray(binary[y0:y1, x0:x1])
    if open_kernel > 1:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            _ellipse_kernel(open_kernel),
            iterations=1,
        )
    if close_kernel > 1:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            _ellipse_kernel(close_kernel),
            iterations=1,
        )
    cleaned, component_count = remove_small_components(
        cleaned,
        min_area=int(min_area_px),
    )
    output = np.zeros(binary.shape, dtype=np.uint8)
    output[y0:y1, x0:x1] = cleaned
    return output, component_count


def postprocess_pidnet_masks(raw_masks, config):
    """Clean the body channel and normalize both endpoint channels to uint8."""

    if not isinstance(config, PidNetMaskConfig):
        raise TypeError("config must be a PidNetMaskConfig.")
    if len(raw_masks) != 3:
        raise ValueError(f"PIDNet must provide exactly three masks; got {len(raw_masks)}.")
    shapes = tuple(np.asarray(mask).shape for mask in raw_masks)
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(f"PIDNet mask shapes do not match: {shapes}.")
    body, component_count = clean_binary_mask(
        raw_masks[0],
        config.min_area_px,
        config.open_kernel,
        config.close_kernel,
    )
    endpoints = tuple(
        np.ascontiguousarray(np.asarray(mask, dtype=np.uint8))
        for mask in raw_masks[1:]
    )
    return (np.ascontiguousarray(body, dtype=np.uint8), *endpoints), int(component_count)


def pidnet_masks_from_probability(probability, config):
    """Threshold and postprocess HxWx3 probabilities using the runtime contract."""

    if not isinstance(config, PidNetMaskConfig):
        raise TypeError("config must be a PidNetMaskConfig.")
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 3 or probability.shape[2] != 3:
        raise ValueError(f"PIDNet probability must have shape HxWx3; got {probability.shape}.")
    raw_masks = tuple(
        np.ascontiguousarray(
            np.where(probability[:, :, channel] >= threshold, 255, 0),
            dtype=np.uint8,
        )
        for channel, threshold in enumerate(config.thresholds)
    )
    return postprocess_pidnet_masks(raw_masks, config)
