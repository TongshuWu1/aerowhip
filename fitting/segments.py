"""Auditable manual segment annotations layered over immutable data."""

from __future__ import annotations

import numpy as np

from .dataset import ProcessedTake


def manual_use_mask(take: ProcessedTake) -> np.ndarray:
    time = take.arrays["time_s"]
    if not take.segments:
        return np.ones(len(time), dtype=bool)
    use_segments = [segment for segment in take.segments if bool(segment.get("use", True))]
    # Explicit Use intervals establish a selected domain.  If the user has only
    # drawn Exclude intervals, the unannotated remainder stays usable.
    mask = np.zeros(len(time), dtype=bool) if use_segments else np.ones(len(time), dtype=bool)
    for segment in take.segments:
        inside = (time >= float(segment["start_s"])) & (time <= float(segment["end_s"]))
        if bool(segment.get("use", True)):
            mask |= inside
    for segment in take.segments:
        if not bool(segment.get("use", True)):
            inside = (time >= float(segment["start_s"])) & (time <= float(segment["end_s"]))
            mask &= ~inside
    return mask


def effective_use_mask(take: ProcessedTake) -> np.ndarray:
    return manual_use_mask(take) & take.arrays["auto_frame_valid"].astype(bool)
