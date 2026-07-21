"""Image-mask utilities required by the PIDNet collection and training tools.

This module intentionally contains no point-cloud or particle-filter code.  It
preserves the mask cleanup and resize behavior used by the previous project.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CableDetection2D:
    mask: np.ndarray
    component_count: int
    component_rejected_mask: np.ndarray | None = None
    morphology_rejected_mask: np.ndarray | None = None


def odd_kernel_size(value):
    value = int(max(0, value))
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value + 1


def remove_small_components(mask, min_area=80, keep_largest_component=False, max_components=0):
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

    components.sort(reverse=True)
    if keep_largest_component:
        components = components[:1]
    elif int(max_components) > 0:
        components = components[: int(max_components)]

    cleaned = np.zeros_like(mask)
    for _area, label in components:
        cleaned[labels == label] = 255
    return cleaned, len(components)


def resize_optional_mask(mask, output_shape):
    if mask is None:
        return np.zeros(tuple(output_shape[:2]), dtype=np.uint8)
    values = np.asarray(mask, dtype=np.uint8)
    output_h, output_w = int(output_shape[0]), int(output_shape[1])
    if values.shape[:2] != (output_h, output_w):
        values = cv2.resize(values, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(np.where(values > 0, 255, 0), dtype=np.uint8)


class CableMaskDetector:
    """Shared morphological cleanup for neural cable masks."""

    def __init__(self, min_area=80, open_kernel=3, close_kernel=5):
        self.min_area = int(min_area)
        self.open_kernel = odd_kernel_size(open_kernel)
        self.close_kernel = odd_kernel_size(close_kernel)

    def clean_mask(self, raw_mask):
        raw = np.where(np.asarray(raw_mask, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
        mask = raw.copy()
        if self.open_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_kernel, self.open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if self.close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        morphology_rejected = np.where((raw > 0) & (mask == 0), 255, 0).astype(np.uint8)
        cleaned, component_count = remove_small_components(
            mask,
            min_area=self.min_area,
            keep_largest_component=False,
            max_components=0,
        )
        component_rejected = np.where((mask > 0) & (cleaned == 0), 255, 0).astype(np.uint8)
        return cleaned, component_count, component_rejected, morphology_rejected


def resize_detection(detection, output_shape):
    output_h, output_w = [int(value) for value in output_shape[:2]]
    input_h, input_w = detection.mask.shape[:2]
    if input_h == output_h and input_w == output_w:
        return detection

    mask = cv2.resize(detection.mask, (output_w, output_h), interpolation=cv2.INTER_NEAREST)
    return CableDetection2D(
        mask=np.ascontiguousarray(mask, dtype=np.uint8),
        component_count=int(detection.component_count),
        component_rejected_mask=resize_optional_mask(
            detection.component_rejected_mask,
            (output_h, output_w),
        ),
        morphology_rejected_mask=resize_optional_mask(
            detection.morphology_rejected_mask,
            (output_h, output_w),
        ),
    )
