"""Causal initialization boundary used by PhysicalEpisodes."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PredictionWindow:
    take_id: str
    role: str
    start_index: int
    stop_index: int
    history_start_index: int
    start_s: float
    end_s: float
    valid_marker_fraction: float
