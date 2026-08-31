"""Continuous physical-episode construction for long-horizon identification.

An episode is a maximally long command-propagatable piece of one Motive take.
Observation quality changes loss masks; it never creates a simulator reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .config import FitConfiguration
from .dataset import ProcessedTake
from .segments import manual_use_mask
from .windows import PredictionWindow


@dataclass(frozen=True, slots=True)
class PhysicalEpisode:
    take_id: str
    role: str
    episode_index: int
    start_index: int
    end_index: int
    history_start_index: int
    residual_eligible_start_index: int | None
    time_s: np.ndarray
    command_position_m: np.ndarray
    command_velocity_mps: np.ndarray
    command_acceleration_mps2: np.ndarray
    command_orientation_xyzw: np.ndarray
    command_angular_velocity: np.ndarray
    uav_position_m: np.ndarray
    uav_orientation_xyzw: np.ndarray
    cable_marker_positions_m: np.ndarray
    uav_observation_valid: np.ndarray
    cable_observation_valid: np.ndarray
    metadata: dict[str, object]

    @property
    def step_count(self) -> int:
        return int(self.end_index - self.start_index)

    @property
    def state_count(self) -> int:
        return self.step_count + 1

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def episode_id(self) -> str:
        return f"{self.take_id}__episode_{self.episode_index:03d}"

    def initialization_window(self) -> PredictionWindow:
        """Compatibility view for the accepted causal state initializer."""

        return PredictionWindow(
            take_id=self.take_id,
            role=self.role,
            start_index=self.start_index,
            stop_index=self.end_index,
            history_start_index=self.history_start_index,
            start_s=float(self.time_s[0]),
            end_s=float(self.time_s[-1]),
            valid_marker_fraction=float(np.mean(self.cable_observation_valid)),
        )


@dataclass(frozen=True, slots=True)
class EpisodeAudit:
    take_id: str
    candidate_start_index: int
    candidate_end_index: int
    accepted: bool
    rejection_reason: str | None
    accepted_start_index: int | None
    duration_s: float


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if index != previous + 1:
            runs.append((start, previous))
            start = index
        previous = index
    runs.append((start, previous))
    return runs


def _break_transition_indices(
    time_s: np.ndarray, break_times_s: Iterable[float]
) -> set[int]:
    """Return transition indices k for breaks between state k and k+1."""

    result: set[int] = set()
    for raw in break_times_s:
        value = float(raw)
        index = int(np.searchsorted(time_s, value, side="left"))
        if 0 < index < len(time_s):
            result.add(index - 1)
    return result


def _initialization_is_valid(
    take: ProcessedTake, start: int, history_frames: int
) -> bool:
    history_start = start - history_frames + 1
    if history_start < 0:
        return False
    arrays = take.arrays
    history = slice(history_start, start + 1)
    return bool(
        np.all(arrays["uav_valid"][history])
        and np.all(arrays["cable_marker_valid"][history])
        and np.all(arrays["auto_frame_valid"][history])
    )


def _residual_eligible_start(
    take: ProcessedTake,
    start: int,
    end_command: int,
    config: FitConfiguration,
) -> int | None:
    history_samples = int(config.uav_residual_ablation["history_samples"])
    derivative_frames = int(config.initialization_history_frames)
    arrays = take.arrays
    for candidate in range(start, end_command + 1):
        feature_start = candidate - history_samples
        earliest = feature_start - derivative_frames + 1
        if earliest < 0:
            continue
        if not bool(np.all(arrays["uav_valid"][earliest : candidate + 1])):
            continue
        if not bool(np.all(arrays["auto_frame_valid"][earliest : candidate + 1])):
            continue
        if not bool(np.all(arrays["command_valid"][feature_start : candidate + 1])):
            continue
        return candidate
    return None


def build_physical_episodes(
    take: ProcessedTake,
    config: FitConfiguration,
) -> tuple[list[PhysicalEpisode], list[EpisodeAudit]]:
    """Build maximal propagatable episodes without measurement-driven breaks."""

    arrays = take.arrays
    time = np.asarray(arrays["time_s"], dtype=np.float64)
    if len(time) < 2:
        return [], []
    finite_time = np.isfinite(time[:-1]) & np.isfinite(time[1:])
    monotonic = np.diff(time) > 0.0
    command_finite = (
        np.isfinite(arrays["command_position_m"][:-1]).all(axis=1)
        & np.isfinite(arrays["command_velocity_mps"][:-1]).all(axis=1)
        & np.isfinite(arrays["command_acceleration_mps2"][:-1]).all(axis=1)
        & np.isfinite(arrays["command_orientation_xyzw"][:-1]).all(axis=1)
        & np.isfinite(arrays["command_angular_velocity"][:-1]).all(axis=1)
    )
    propagation_valid = (
        finite_time
        & monotonic
        & arrays["command_valid"][:-1].astype(bool)
        & command_finite
    )
    decision = take.metadata.get("dataset_decision", {})
    manifest_breaks = []
    if isinstance(decision, dict):
        manifest_breaks = list(decision.get("episode_breaks_s", []))
    # ProcessedTake carries the current manifest decision in fields rather than
    # metadata, so accept the explicit list attached by load_dataset as well.
    explicit_breaks = getattr(take, "episode_breaks_s", ())
    for transition in _break_transition_indices(
        time, tuple(manifest_breaks) + tuple(explicit_breaks)
    ):
        propagation_valid[transition] = False

    manual = manual_use_mask(take)
    uav_obs = (
        manual
        & arrays["uav_valid"].astype(bool)
        & ~arrays.get("quality_uav_jump", np.zeros(len(time), dtype=bool)).astype(bool)
    )
    cable_obs = manual[:, None] & arrays["cable_marker_valid"].astype(bool)
    cable_bad_frame = (
        arrays.get("quality_marker_jump", np.zeros(len(time), dtype=bool)).astype(bool)
        | arrays.get("quality_geometry_invalid", np.zeros(len(time), dtype=bool)).astype(bool)
    )
    cable_obs[cable_bad_frame] = False

    episodes: list[PhysicalEpisode] = []
    audit: list[EpisodeAudit] = []
    history_frames = int(config.initialization_history_frames)
    for candidate_start, end_command in _contiguous_true_runs(propagation_valid):
        start = next(
            (
                index
                for index in range(candidate_start, end_command + 1)
                if _initialization_is_valid(take, index, history_frames)
            ),
            None,
        )
        if start is None:
            audit.append(
                EpisodeAudit(
                    take.take_id,
                    candidate_start,
                    end_command + 1,
                    False,
                    "no_complete_causal_initialization",
                    None,
                    float(time[end_command + 1] - time[candidate_start]),
                )
            )
            continue
        end = end_command + 1
        if end <= start:
            audit.append(
                EpisodeAudit(
                    take.take_id,
                    candidate_start,
                    end,
                    False,
                    "no_propagation_step_after_initialization",
                    start,
                    0.0,
                )
            )
            continue
        residual_start = _residual_eligible_start(take, start, end_command, config)
        state_slice = slice(start, end + 1)
        command_slice = slice(start, end)
        episode_index = len(episodes)
        episode = PhysicalEpisode(
            take_id=take.take_id,
            role=take.role,
            episode_index=episode_index,
            start_index=start,
            end_index=end,
            history_start_index=start - history_frames + 1,
            residual_eligible_start_index=residual_start,
            time_s=np.array(time[state_slice], copy=True),
            command_position_m=np.array(arrays["command_position_m"][command_slice], copy=True),
            command_velocity_mps=np.array(arrays["command_velocity_mps"][command_slice], copy=True),
            command_acceleration_mps2=np.array(arrays["command_acceleration_mps2"][command_slice], copy=True),
            command_orientation_xyzw=np.array(arrays["command_orientation_xyzw"][command_slice], copy=True),
            command_angular_velocity=np.array(arrays["command_angular_velocity"][command_slice], copy=True),
            uav_position_m=np.array(arrays["uav_position_m"][state_slice], copy=True),
            uav_orientation_xyzw=np.array(arrays["uav_orientation_xyzw"][state_slice], copy=True),
            cable_marker_positions_m=np.array(arrays["cable_marker_positions_m"][state_slice], copy=True),
            uav_observation_valid=np.array(uav_obs[state_slice], copy=True),
            cable_observation_valid=np.array(cable_obs[state_slice], copy=True),
            metadata={
                "propagation_rule": "contiguous_finite_monotonic_time_and_complete_command",
                "observation_exclusion_does_not_break_episode": True,
                "candidate_start_index": candidate_start,
                "initialization_shift_frames": start - candidate_start,
                "explicit_measurement_reset_inside_episode": False,
            },
        )
        episodes.append(episode)
        audit.append(
            EpisodeAudit(
                take.take_id,
                candidate_start,
                end,
                True,
                None,
                start,
                episode.duration_s,
            )
        )
    return episodes, audit


def episode_manifest(
    takes: Iterable[ProcessedTake], config: FitConfiguration
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "physical_episode_manifest_v1",
        "methodology": "continuous_physical_episode_v1",
        "takes": {},
    }
    take_payload = payload["takes"]
    assert isinstance(take_payload, dict)
    for take in takes:
        episodes, audit = build_physical_episodes(take, config)
        take_payload[take.take_id] = {
            "role": take.role,
            "episode_count": len(episodes),
            "total_duration_s": float(sum(item.duration_s for item in episodes)),
            "episodes": [
                {
                    "episode_id": item.episode_id,
                    "start_index": item.start_index,
                    "end_index": item.end_index,
                    "start_s": float(item.time_s[0]),
                    "end_s": float(item.time_s[-1]),
                    "duration_s": item.duration_s,
                    "step_count": item.step_count,
                    "residual_eligible_start_index": item.residual_eligible_start_index,
                    "uav_observation_count": int(np.count_nonzero(item.uav_observation_valid)),
                    "cable_observation_count": int(np.count_nonzero(item.cable_observation_valid)),
                }
                for item in episodes
            ],
            "rejected_fragments": [
                {
                    "candidate_start_index": item.candidate_start_index,
                    "candidate_end_index": item.candidate_end_index,
                    "reason": item.rejection_reason,
                }
                for item in audit
                if not item.accepted
            ],
        }
    return payload
