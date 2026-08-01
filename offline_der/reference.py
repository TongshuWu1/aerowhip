"""Build inspectable fixed-length DDER reference trajectories from observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from cable_twin.cable_observation import CableObservationFrame, RouteHypothesis
from cable_twin.der import project_inextensible
from cable_twin.recording import sha256_file


REFERENCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReferenceExtractionSettings:
    node_count: int = 24
    minimum_depth_valid_fraction: float = 0.65
    maximum_missing_arc_m: float = 0.075
    projection_iterations: int = 20
    projection_tolerance_m: float = 2.0e-4
    route_continuity_sigma_m: float = 0.025
    route_score_weight: float = 1.0
    minimum_sequence_frames: int = 8

    def __post_init__(self) -> None:
        if self.node_count < 4:
            raise ValueError("reference reconstruction requires at least four nodes")
        if not 0.0 < self.minimum_depth_valid_fraction <= 1.0:
            raise ValueError("minimum_depth_valid_fraction must be in (0, 1]")
        for name in (
            "maximum_missing_arc_m",
            "projection_tolerance_m",
            "route_continuity_sigma_m",
            "route_score_weight",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.projection_iterations <= 0 or self.minimum_sequence_frames < 2:
            raise ValueError("reference solver/frame counts are invalid")


@dataclass(frozen=True, slots=True)
class _RouteCandidate:
    positions_m_f32: np.ndarray
    covariance_m2_f32: np.ndarray
    node_observed_bool: np.ndarray
    route_score: float
    depth_valid_fraction: float
    maximum_missing_arc_m: float


@dataclass(frozen=True, slots=True)
class _FrameCandidates:
    sequence_index: int
    source_position: int
    timestamp_ns: int
    candidates: tuple[_RouteCandidate, ...]


@dataclass(frozen=True, slots=True)
class DderReferenceDataset:
    cable_id: int
    cable_length_m: float
    timestamps_ns_i64: np.ndarray
    source_sequence_i64: np.ndarray
    source_position_i64: np.ndarray
    sequence_id_i32: np.ndarray
    centerlines_m_f32: np.ndarray
    velocities_m_s_f32: np.ndarray
    node_observed_bool: np.ndarray
    covariance_m2_f32: np.ndarray
    route_score_f32: np.ndarray
    maximum_missing_arc_m_f32: np.ndarray
    depth_valid_fraction_f32: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.cable_id not in (0, 1):
            raise ValueError("cable_id must be 0 or 1")
        if not math.isfinite(self.cable_length_m) or self.cable_length_m <= 0.0:
            raise ValueError("cable_length_m must be finite and positive")
        frame_count = len(self.timestamps_ns_i64)
        node_count = self.centerlines_m_f32.shape[1]
        expected_vectors = {
            "timestamps_ns_i64": (self.timestamps_ns_i64, np.int64),
            "source_sequence_i64": (self.source_sequence_i64, np.int64),
            "source_position_i64": (self.source_position_i64, np.int64),
            "sequence_id_i32": (self.sequence_id_i32, np.int32),
            "route_score_f32": (self.route_score_f32, np.float32),
            "maximum_missing_arc_m_f32": (
                self.maximum_missing_arc_m_f32,
                np.float32,
            ),
            "depth_valid_fraction_f32": (
                self.depth_valid_fraction_f32,
                np.float32,
            ),
        }
        for name, (value, dtype) in expected_vectors.items():
            if value.dtype != dtype or value.shape != (frame_count,):
                raise ValueError(f"{name} must be {dtype.__name__} length T")
        if (
            self.centerlines_m_f32.dtype != np.float32
            or self.centerlines_m_f32.ndim != 3
            or self.centerlines_m_f32.shape != (frame_count, node_count, 3)
        ):
            raise ValueError("centerlines_m_f32 must have shape TxNx3")
        if self.velocities_m_s_f32.shape != self.centerlines_m_f32.shape:
            raise ValueError("velocities must match centerlines")
        if self.velocities_m_s_f32.dtype != np.float32:
            raise ValueError("velocities must be float32")
        if self.node_observed_bool.shape != (frame_count, node_count):
            raise ValueError("node_observed_bool must have shape TxN")
        if self.node_observed_bool.dtype != np.bool_:
            raise ValueError("node_observed_bool must be bool")
        if self.covariance_m2_f32.shape != (frame_count, node_count, 3, 3):
            raise ValueError("covariance must have shape TxNx3x3")
        if self.covariance_m2_f32.dtype != np.float32:
            raise ValueError("covariance must be float32")
        if frame_count < 2 or node_count < 4:
            raise ValueError("reference dataset is too short")
        if not np.isfinite(self.centerlines_m_f32).all():
            raise ValueError("reference centerlines must be finite")
        if not np.isfinite(self.velocities_m_s_f32).all():
            raise ValueError("reference velocities must be finite")
        if not np.isfinite(self.covariance_m2_f32).all():
            raise ValueError("reference covariance must be finite")
        if not np.allclose(
            self.covariance_m2_f32,
            np.swapaxes(self.covariance_m2_f32, -1, -2),
            rtol=1.0e-5,
            atol=1.0e-8,
        ):
            raise ValueError("reference covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(self.covariance_m2_f32) < -1.0e-9):
            raise ValueError("reference covariance must be positive semidefinite")
        if np.any(np.diff(self.timestamps_ns_i64) <= 0):
            raise ValueError("reference timestamps must be strictly increasing")
        segment_lengths = np.linalg.norm(
            np.diff(self.centerlines_m_f32, axis=1),
            axis=2,
        )
        segment_target = self.cable_length_m / (node_count - 1)
        if float(np.max(np.abs(segment_lengths - segment_target))) > 5.0e-4:
            raise ValueError("reference centerline violates fixed cable length")

    @property
    def node_count(self) -> int:
        return int(self.centerlines_m_f32.shape[1])

    @property
    def frame_count(self) -> int:
        return int(self.centerlines_m_f32.shape[0])


def _route_candidate(
    route: RouteHypothesis,
    settings: ReferenceExtractionSettings,
) -> _RouteCandidate | None:
    pixels = route.pixels_xy_f32.astype(np.float64, copy=False)
    segment_pixels = np.linalg.norm(np.diff(pixels, axis=0), axis=1)
    valid = route.depth_valid_bool
    consecutive_depth = valid[:-1] & valid[1:]
    segment_metric = np.zeros_like(segment_pixels)
    segment_metric[consecutive_depth] = np.linalg.norm(
        np.diff(route.xyz_camera_m_f32, axis=0)[consecutive_depth],
        axis=1,
    )
    missing = ~consecutive_depth
    if np.any(missing):
        missing_metric_length = max(
            0.0,
            route.estimated_length_m - float(np.sum(segment_metric)),
        )
        missing_pixels = float(np.sum(segment_pixels[missing]))
        if missing_metric_length <= 0.0 or missing_pixels <= 0.0:
            return None
        segment_metric[missing] = (
            missing_metric_length * segment_pixels[missing] / missing_pixels
        )
    total_metric = float(np.sum(segment_metric))
    if not math.isfinite(total_metric) or total_metric <= 0.0:
        return None
    measured_arc = np.concatenate(([0.0], np.cumsum(segment_metric)))
    arc_m = route.target_length_m * measured_arc / total_metric
    valid_indices = np.flatnonzero(valid)
    if (
        len(valid_indices) < 2
        or valid_indices[0] != 0
        or valid_indices[-1] != len(valid) - 1
    ):
        return None
    valid_fraction = float(np.mean(valid))
    valid_arc = arc_m[valid]
    maximum_gap = float(np.max(np.diff(valid_arc)))
    if (
        valid_fraction < settings.minimum_depth_valid_fraction
        or maximum_gap > settings.maximum_missing_arc_m
    ):
        return None
    endpoint_distance = float(
        np.linalg.norm(route.xyz_camera_m_f32[-1] - route.xyz_camera_m_f32[0])
    )
    if endpoint_distance > route.target_length_m + settings.projection_tolerance_m:
        return None

    targets = np.linspace(0.0, route.target_length_m, settings.node_count)
    valid_xyz = route.xyz_camera_m_f32[valid].astype(np.float64, copy=False)
    positions = np.column_stack(
        [np.interp(targets, valid_arc, valid_xyz[:, axis]) for axis in range(3)]
    ).astype(np.float32)
    valid_covariance = route.covariance_m2_f32[valid].astype(np.float64, copy=False)
    covariance = np.empty((settings.node_count, 3, 3), dtype=np.float32)
    for row in range(3):
        for column in range(3):
            covariance[:, row, column] = np.interp(
                targets,
                valid_arc,
                valid_covariance[:, row, column],
            )
    nearest_distance = np.min(
        np.abs(targets[:, None] - valid_arc[None, :]),
        axis=1,
    )
    observed = nearest_distance <= 0.75 * route.target_length_m / (
        settings.node_count - 1
    )
    return _RouteCandidate(
        positions_m_f32=np.ascontiguousarray(positions),
        covariance_m2_f32=np.ascontiguousarray(covariance),
        node_observed_bool=np.ascontiguousarray(observed, dtype=np.bool_),
        route_score=float(route.score),
        depth_valid_fraction=valid_fraction,
        maximum_missing_arc_m=maximum_gap,
    )


class ReferenceAccumulator:
    """Collect route candidates once, then solve temporal route identity offline."""

    def __init__(
        self,
        settings: ReferenceExtractionSettings,
        cable_lengths_m: tuple[float, float],
        *,
        device: torch.device,
    ) -> None:
        self.settings = settings
        self.cable_lengths_m = cable_lengths_m
        self.device = device
        self._frames: tuple[list[_FrameCandidates], list[_FrameCandidates]] = ([], [])
        self.total_observation_frames = 0

    def add(self, observation: CableObservationFrame) -> None:
        self.total_observation_frames += 1
        for cable_id in (0, 1):
            candidates = tuple(
                candidate
                for route in observation.routes_by_cable[cable_id]
                if (candidate := _route_candidate(route, self.settings)) is not None
            )
            self._frames[cable_id].append(
                _FrameCandidates(
                    sequence_index=observation.key.sequence_index,
                    source_position=observation.key.source_position,
                    timestamp_ns=observation.key.timestamp_ns,
                    candidates=candidates,
                )
            )

    def _project_candidates(self, cable_id: int) -> list[_FrameCandidates]:
        frames = self._frames[cable_id]
        flat = [candidate for frame in frames for candidate in frame.candidates]
        if not flat:
            return frames
        segment_length = self.cable_lengths_m[cable_id] / (
            self.settings.node_count - 1
        )
        projected: list[_RouteCandidate | None] = []
        batch_size = 256
        for start in range(0, len(flat), batch_size):
            batch = flat[start : start + batch_size]
            original = np.stack([value.positions_m_f32 for value in batch])
            tensor = torch.as_tensor(original, device=self.device)
            with torch.no_grad():
                result = project_inextensible(
                    tensor,
                    segment_length,
                    iterations=self.settings.projection_iterations,
                )
                lengths = torch.linalg.vector_norm(
                    result[:, 1:] - result[:, :-1], dim=-1
                )
                errors = torch.amax(torch.abs(lengths - segment_length), dim=1)
            result_cpu = result.detach().cpu().numpy()
            errors_cpu = errors.detach().cpu().numpy()
            for source, positions, error in zip(batch, result_cpu, errors_cpu):
                if float(error) > self.settings.projection_tolerance_m:
                    projected.append(None)
                    continue
                displacement = positions - source.positions_m_f32
                covariance = source.covariance_m2_f32.copy()
                displacement_variance = np.sum(displacement * displacement, axis=1)
                covariance[:, range(3), range(3)] += displacement_variance[:, None]
                projected.append(
                    replace(
                        source,
                        positions_m_f32=np.ascontiguousarray(positions, dtype=np.float32),
                        covariance_m2_f32=np.ascontiguousarray(covariance),
                    )
                )
        iterator = iter(projected)
        output = []
        for frame in frames:
            frame_candidates = [next(iterator) for _ in frame.candidates]
            output.append(
                replace(
                    frame,
                    candidates=tuple(
                        candidate
                        for candidate in frame_candidates
                        if candidate is not None
                    ),
                )
            )
        return output

    def _select_contiguous_segment(
        self,
        frames: list[_FrameCandidates],
    ) -> list[tuple[_FrameCandidates, _RouteCandidate]]:
        costs: list[np.ndarray] = []
        parents: list[np.ndarray] = []
        first_cost = self.settings.route_score_weight * np.asarray(
            [candidate.route_score for candidate in frames[0].candidates],
            dtype=np.float64,
        )
        costs.append(first_cost)
        parents.append(np.full(len(first_cost), -1, dtype=np.int32))
        sigma2 = self.settings.route_continuity_sigma_m**2
        for frame_index in range(1, len(frames)):
            previous = frames[frame_index - 1].candidates
            current = frames[frame_index].candidates
            previous_positions = np.stack(
                [candidate.positions_m_f32 for candidate in previous]
            )
            current_positions = np.stack(
                [candidate.positions_m_f32 for candidate in current]
            )
            difference = current_positions[:, None] - previous_positions[None, :]
            transition = np.mean(difference * difference, axis=(2, 3)) / sigma2
            total = transition + costs[-1][None, :]
            parent = np.argmin(total, axis=1).astype(np.int32)
            emission = self.settings.route_score_weight * np.asarray(
                [candidate.route_score for candidate in current],
                dtype=np.float64,
            )
            costs.append(emission + total[np.arange(len(current)), parent])
            parents.append(parent)
        selected_indices = [int(np.argmin(costs[-1]))]
        for frame_index in range(len(frames) - 1, 0, -1):
            selected_indices.append(
                int(parents[frame_index][selected_indices[-1]])
            )
        selected_indices.reverse()
        return [
            (frame, frame.candidates[index])
            for frame, index in zip(frames, selected_indices)
        ]

    def finish(
        self,
        cable_id: int,
        *,
        metadata: Mapping[str, Any],
    ) -> DderReferenceDataset:
        if cable_id not in (0, 1):
            raise ValueError("cable_id must be 0 or 1")
        frames = self._project_candidates(cable_id)
        segments: list[list[_FrameCandidates]] = []
        active: list[_FrameCandidates] = []
        previous_sequence = -2
        for frame in frames:
            if not frame.candidates or frame.sequence_index != previous_sequence + 1:
                if len(active) >= self.settings.minimum_sequence_frames:
                    segments.append(active)
                active = []
            if frame.candidates:
                active.append(frame)
            previous_sequence = frame.sequence_index
        if len(active) >= self.settings.minimum_sequence_frames:
            segments.append(active)
        if not segments:
            raise ValueError(
                f"Cable {cable_id} has no fully observed sequence with at least "
                f"{self.settings.minimum_sequence_frames} consecutive frames"
            )

        selected = [self._select_contiguous_segment(segment) for segment in segments]
        flat = [item for segment in selected for item in segment]
        sequence_ids = np.concatenate(
            [np.full(len(segment), index, dtype=np.int32) for index, segment in enumerate(selected)]
        )
        timestamps = np.asarray([frame.timestamp_ns for frame, _ in flat], dtype=np.int64)
        centerlines = np.stack([candidate.positions_m_f32 for _, candidate in flat])
        velocities = np.empty_like(centerlines)
        offset = 0
        for segment in selected:
            count = len(segment)
            segment_positions = centerlines[offset : offset + count]
            segment_times = timestamps[offset : offset + count].astype(np.float64) * 1.0e-9
            velocities[offset] = (segment_positions[1] - segment_positions[0]) / (
                segment_times[1] - segment_times[0]
            )
            velocities[offset + count - 1] = (
                segment_positions[-1] - segment_positions[-2]
            ) / (segment_times[-1] - segment_times[-2])
            if count > 2:
                time_span = (segment_times[2:] - segment_times[:-2])[:, None, None]
                velocities[offset + 1 : offset + count - 1] = (
                    segment_positions[2:] - segment_positions[:-2]
                ) / time_span
            offset += count

        output_metadata = dict(metadata)
        output_metadata.update(
            {
                "reference_schema_version": REFERENCE_SCHEMA_VERSION,
                "extraction_settings": asdict(self.settings),
                "input_observation_frames": self.total_observation_frames,
                "retained_frames": len(flat),
                "retained_sequences": len(selected),
            }
        )
        return DderReferenceDataset(
            cable_id=cable_id,
            cable_length_m=self.cable_lengths_m[cable_id],
            timestamps_ns_i64=timestamps,
            source_sequence_i64=np.asarray(
                [frame.sequence_index for frame, _ in flat], dtype=np.int64
            ),
            source_position_i64=np.asarray(
                [frame.source_position for frame, _ in flat], dtype=np.int64
            ),
            sequence_id_i32=sequence_ids,
            centerlines_m_f32=np.ascontiguousarray(centerlines, dtype=np.float32),
            velocities_m_s_f32=np.ascontiguousarray(velocities, dtype=np.float32),
            node_observed_bool=np.ascontiguousarray(
                np.stack([candidate.node_observed_bool for _, candidate in flat]),
                dtype=np.bool_,
            ),
            covariance_m2_f32=np.ascontiguousarray(
                np.stack([candidate.covariance_m2_f32 for _, candidate in flat]),
                dtype=np.float32,
            ),
            route_score_f32=np.asarray(
                [candidate.route_score for _, candidate in flat], dtype=np.float32
            ),
            maximum_missing_arc_m_f32=np.asarray(
                [candidate.maximum_missing_arc_m for _, candidate in flat],
                dtype=np.float32,
            ),
            depth_valid_fraction_f32=np.asarray(
                [candidate.depth_valid_fraction for _, candidate in flat],
                dtype=np.float32,
            ),
            metadata=output_metadata,
        )


def save_reference_dataset(path: Path, dataset: DderReferenceDataset) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        raise ValueError("DDER reference artifact must use the .npz suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(f"{destination}.json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError(destination if destination.exists() else sidecar)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        cable_id=np.asarray(dataset.cable_id, dtype=np.int32),
        cable_length_m=np.asarray(dataset.cable_length_m, dtype=np.float64),
        timestamps_ns_i64=dataset.timestamps_ns_i64,
        source_sequence_i64=dataset.source_sequence_i64,
        source_position_i64=dataset.source_position_i64,
        sequence_id_i32=dataset.sequence_id_i32,
        centerlines_m_f32=dataset.centerlines_m_f32,
        velocities_m_s_f32=dataset.velocities_m_s_f32,
        node_observed_bool=dataset.node_observed_bool,
        covariance_m2_f32=dataset.covariance_m2_f32,
        route_score_f32=dataset.route_score_f32,
        maximum_missing_arc_m_f32=dataset.maximum_missing_arc_m_f32,
        depth_valid_fraction_f32=dataset.depth_valid_fraction_f32,
    )
    os.replace(temporary, destination)
    payload = dict(dataset.metadata)
    payload.update(
        {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "artifact": destination.name,
            "artifact_sha256": sha256_file(destination),
            "cable_id": dataset.cable_id,
            "cable_length_m": dataset.cable_length_m,
            "node_count": dataset.node_count,
            "frame_count": dataset.frame_count,
        }
    )
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    with temporary_sidecar.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_sidecar, sidecar)


def load_reference_dataset(path: Path) -> DderReferenceDataset:
    source = Path(path).expanduser().resolve()
    sidecar = Path(f"{source}.json")
    with sidecar.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported DDER reference schema: {sidecar}")
    if metadata.get("artifact_sha256") != sha256_file(source):
        raise ValueError(f"DDER reference hash mismatch: {source}")
    with np.load(source, allow_pickle=False) as values:
        return DderReferenceDataset(
            cable_id=int(values["cable_id"]),
            cable_length_m=float(values["cable_length_m"]),
            timestamps_ns_i64=values["timestamps_ns_i64"],
            source_sequence_i64=values["source_sequence_i64"],
            source_position_i64=values["source_position_i64"],
            sequence_id_i32=values["sequence_id_i32"],
            centerlines_m_f32=values["centerlines_m_f32"],
            velocities_m_s_f32=values["velocities_m_s_f32"],
            node_observed_bool=values["node_observed_bool"],
            covariance_m2_f32=values["covariance_m2_f32"],
            route_score_f32=values["route_score_f32"],
            maximum_missing_arc_m_f32=values["maximum_missing_arc_m_f32"],
            depth_valid_fraction_f32=values["depth_valid_fraction_f32"],
            metadata=metadata,
        )


__all__ = [
    "DderReferenceDataset",
    "REFERENCE_SCHEMA_VERSION",
    "ReferenceAccumulator",
    "ReferenceExtractionSettings",
    "load_reference_dataset",
    "save_reference_dataset",
]
