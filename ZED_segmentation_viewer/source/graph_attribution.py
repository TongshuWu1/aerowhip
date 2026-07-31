"""Diagnostic association of observed skeleton edges with predicted cables."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from observation import GraphEdgeObservation


@dataclass(frozen=True)
class GraphAttributionConfig:
    """Calibrated costs for ordered edge-to-cable interval association."""

    curve_samples_per_segment: int = 4
    distance_sigma_m: float = 0.012
    length_sigma_m: float = 0.025
    length_weight: float = 0.25
    huber_delta: float = 1.5
    unassigned_energy: float = 3.0
    minimum_identity_probability: float = 0.60
    temporal_identity_weight: float = 1.0
    temporal_match_sigma_m: float = 0.020
    temporal_match_gate_m: float = 0.050
    temporal_history_frames: int = 5
    temporal_history_decay: float = 0.75
    temporal_probability_floor: float = 0.05


@dataclass(frozen=True)
class GraphEdgeAttribution:
    """Soft cable identity and arc interval for one directly observed edge."""

    component_label: int
    edge_id: int
    cable_probabilities: np.ndarray
    unassigned_probability: float
    selected_cable: int
    confidence: float
    arc_start_m: float
    arc_end_m: float
    forward: bool
    mean_residual_m: float
    endpoint_anchored: bool
    temporal_prior_applied: bool
    temporal_match_distance_m: float


@dataclass(frozen=True)
class GraphAttributionDiagnostics:
    """Human-readable summary of ordered graph-edge attribution."""

    edges: tuple[GraphEdgeAttribution, ...]
    prediction_spread_m: np.ndarray
    initialized: np.ndarray
    attributed_count: int
    ambiguous_count: int
    unassigned_count: int
    temporally_matched_count: int
    processing_ms: float
    gpu_ms: float


@dataclass(frozen=True)
class GraphAttributionBatch:
    """Device-resident ordered edge measurements used by the particle filter.

    The association is evaluated against the predicted prior, before the
    current measurement changes particle positions or weights.  This avoids
    assigning an edge to a cable merely because a posterior particle was
    already pulled toward that edge.
    """

    graph_edges: tuple[GraphEdgeObservation, ...]
    observed_xyz: torch.Tensor
    edge_lengths_m: torch.Tensor
    edge_valid: torch.Tensor
    best_energy: torch.Tensor
    best_residual_m: torch.Tensor
    best_arc_start_m: torch.Tensor
    best_arc_end_m: torch.Tensor
    best_model_xyz: torch.Tensor
    probabilities: torch.Tensor
    selected_cable: torch.Tensor
    selected_confidence: torch.Tensor
    endpoint_anchored: torch.Tensor
    temporal_prior_applied: torch.Tensor
    temporal_match_distance_m: torch.Tensor
    prediction_spread_m: torch.Tensor
    initialized: np.ndarray
    processing_ms: float
    gpu_start: torch.cuda.Event | None
    gpu_end: torch.cuda.Event | None

    @property
    def edge_count(self) -> int:
        return int(self.observed_xyz.shape[0])

    @property
    def observed_count(self) -> int:
        return int(self.observed_xyz.shape[1])

    def selected_mask(self) -> torch.Tensor:
        """Return ``(2, edges)`` cable-selection flags on the PF device."""

        cable_indices = torch.arange(
            2,
            device=self.selected_cable.device,
            dtype=self.selected_cable.dtype,
        )[:, None]
        return self.selected_cable[None, :] == cable_indices


@dataclass(frozen=True)
class _CandidateGrid:
    """Cached exhaustive ordered interval candidates for one tensor shape."""

    starts_np: np.ndarray
    ends_np: np.ndarray
    lower: torch.Tensor
    upper: torch.Tensor
    alpha: torch.Tensor
    start_fraction: torch.Tensor
    end_fraction: torch.Tensor
    length_fraction: torch.Tensor


def _interval_end_constraints(
    edge: GraphEdgeObservation,
    cable_index: int,
    final_index: int,
) -> tuple[int | None, int | None, bool]:
    """Return raw-edge start/end arc constraints and whether identity is allowed."""

    cable_anchors = tuple(
        anchor
        for anchor in edge.endpoint_anchors
        if int(anchor.cable_index) == int(cable_index)
    )
    anchored_cables = {
        int(anchor.cable_index) for anchor in edge.endpoint_anchors
    }
    if anchored_cables and not cable_anchors:
        return None, None, False

    start_constraint = None
    end_constraint = None
    for anchor in cable_anchors:
        arc_index = 0 if int(anchor.endpoint_index) == 0 else final_index
        if int(anchor.edge_end) == 0:
            if start_constraint is not None and start_constraint != arc_index:
                return None, None, False
            start_constraint = arc_index
        else:
            if end_constraint is not None and end_constraint != arc_index:
                return None, None, False
            end_constraint = arc_index
    return start_constraint, end_constraint, bool(cable_anchors)


def _huber(values: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = torch.abs(values)
    return torch.where(
        absolute <= delta,
        0.5 * absolute.square(),
        delta * (absolute - 0.5 * delta),
    )


class GraphEdgeAttributor:
    """Batched exhaustive ordered-interval association on the PF device."""

    def __init__(
        self,
        config: GraphAttributionConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = torch.device(device)
        self.dtype = torch.float32
        self._candidate_grids: dict[tuple[int, int], _CandidateGrid] = {}
        self._curve_sample_fractions: dict[int, torch.Tensor] = {}
        self._temporal_history: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []

    def clear_temporal_history(self) -> None:
        self._temporal_history.clear()

    def _temporal_identity_prior(
        self,
        observed: torch.Tensor,
        edge_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match current fragments to recent observed fragments on the device.

        Directed current-to-history distance deliberately supports graph-edge
        splitting: a newly visible fragment may be a subset of an older edge.
        The returned probability is a soft prior, never a hard identity lock.
        """

        edge_count = int(observed.shape[0])
        uniform = torch.full(
            (edge_count, 3),
            1.0 / 3.0,
            device=self.device,
            dtype=self.dtype,
        )
        no_match = torch.full(
            (edge_count,),
            torch.inf,
            device=self.device,
            dtype=self.dtype,
        )
        if not self._temporal_history:
            return uniform, torch.zeros_like(edge_valid), no_match

        history_xyz = []
        history_probability = []
        history_valid = []
        history_age = []
        for age, (xyz, probability, valid) in enumerate(
            reversed(self._temporal_history),
            start=1,
        ):
            if xyz.numel() == 0 or xyz.shape[1] != observed.shape[1]:
                continue
            history_xyz.append(xyz)
            history_probability.append(probability)
            history_valid.append(valid)
            history_age.append(
                torch.full(
                    (xyz.shape[0],),
                    age,
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        if not history_xyz:
            return uniform, torch.zeros_like(edge_valid), no_match

        previous_xyz = torch.cat(history_xyz, dim=0)
        previous_probability = torch.cat(history_probability, dim=0)
        previous_valid = torch.cat(history_valid, dim=0)
        previous_age = torch.cat(history_age, dim=0)
        pairwise = torch.cdist(
            observed[:, None, :, :],
            previous_xyz[None, :, :, :],
        )
        directed_distance = pairwise.amin(dim=-1).mean(dim=-1)
        candidate_valid = edge_valid[:, None] & previous_valid[None, :]
        match_cost = directed_distance + (
            previous_age[None, :] - 1.0
        ) * (0.25 * self.config.temporal_match_sigma_m)
        match_cost = torch.where(
            candidate_valid,
            match_cost,
            torch.full_like(match_cost, torch.inf),
        )
        best_cost, best_index = torch.min(match_cost, dim=1)
        best_distance = torch.gather(
            directed_distance,
            1,
            best_index[:, None],
        )[:, 0]
        best_age = previous_age[best_index]
        matched = (
            torch.isfinite(best_cost)
            & (best_distance <= self.config.temporal_match_gate_m)
        )
        floor = float(np.clip(self.config.temporal_probability_floor, 0.0, 1.0 / 3.0))
        previous_prior = (
            (1.0 - 3.0 * floor) * previous_probability[best_index] + floor
        )
        distance_strength = torch.exp(
            -0.5
            * (
                best_distance / self.config.temporal_match_sigma_m
            ).square()
        )
        age_strength = self.config.temporal_history_decay ** (
            best_age - 1.0
        )
        strength = (distance_strength * age_strength).clamp(0.0, 1.0)
        prior = (
            strength[:, None] * previous_prior
            + (1.0 - strength[:, None]) * uniform
        )
        prior = torch.where(matched[:, None], prior, uniform)
        best_distance = torch.where(matched, best_distance, no_match)
        return prior, matched, best_distance

    def _remember_temporal_identity(
        self,
        observed: torch.Tensor,
        probability: torch.Tensor,
        edge_valid: torch.Tensor,
    ) -> None:
        self._temporal_history.append(
            (
                observed.detach().clone(),
                probability.detach().clone(),
                edge_valid.detach().clone(),
            )
        )
        maximum = max(1, int(self.config.temporal_history_frames))
        if len(self._temporal_history) > maximum:
            del self._temporal_history[:-maximum]

    def _candidate_grid(
        self,
        dense_count: int,
        observed_count: int,
    ) -> _CandidateGrid:
        key = (int(dense_count), int(observed_count))
        cached = self._candidate_grids.get(key)
        if cached is not None:
            return cached

        indices = np.arange(dense_count, dtype=np.int64)
        starts, ends = np.meshgrid(indices, indices, indexing="ij")
        starts = starts.reshape(-1)
        ends = ends.reshape(-1)
        nonzero = starts != ends
        starts = np.ascontiguousarray(starts[nonzero])
        ends = np.ascontiguousarray(ends[nonzero])

        sample_fraction = torch.linspace(
            0.0,
            1.0,
            observed_count,
            device=self.device,
            dtype=self.dtype,
        )
        starts_tensor = torch.as_tensor(
            starts,
            device=self.device,
            dtype=self.dtype,
        )
        ends_tensor = torch.as_tensor(
            ends,
            device=self.device,
            dtype=self.dtype,
        )
        target_index = (
            starts_tensor[:, None]
            + sample_fraction[None, :]
            * (ends_tensor - starts_tensor)[:, None]
        )
        lower = torch.floor(target_index).to(torch.int64)
        upper = torch.ceil(target_index).to(torch.int64).clamp_max(
            dense_count - 1
        )
        candidate = _CandidateGrid(
            starts_np=starts,
            ends_np=ends,
            lower=lower,
            upper=upper,
            alpha=target_index - lower.to(self.dtype),
            start_fraction=starts_tensor / float(dense_count - 1),
            end_fraction=ends_tensor / float(dense_count - 1),
            length_fraction=torch.abs(ends_tensor - starts_tensor)
            / float(dense_count - 1),
        )
        self._candidate_grids[key] = candidate
        return candidate

    def _resample_predicted_curves(
        self,
        predicted_curves: torch.Tensor,
        dense_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample each predicted polyline at uniform geometric arc length."""

        curves = torch.as_tensor(
            predicted_curves,
            device=self.device,
            dtype=self.dtype,
        )
        if curves.shape[0] != 2 or curves.ndim != 3 or curves.shape[-1] != 3:
            raise ValueError("predicted_curves must have shape (2, nodes, 3)")
        finite = torch.all(torch.isfinite(curves), dim=(1, 2))
        safe_curves = torch.nan_to_num(curves)
        segment_lengths = torch.linalg.vector_norm(
            safe_curves[:, 1:] - safe_curves[:, :-1],
            dim=-1,
        )
        cumulative = torch.cat(
            (
                torch.zeros(
                    (2, 1),
                    device=self.device,
                    dtype=self.dtype,
                ),
                torch.cumsum(segment_lengths, dim=1),
            ),
            dim=1,
        )
        total_length = cumulative[:, -1]
        curve_valid = finite & torch.isfinite(total_length) & (total_length > 1e-9)
        sample_fraction = self._curve_sample_fractions.get(int(dense_count))
        if sample_fraction is None:
            sample_fraction = torch.linspace(
                0.0,
                1.0,
                dense_count,
                device=self.device,
                dtype=self.dtype,
            )
            self._curve_sample_fractions[int(dense_count)] = sample_fraction
        target_arc = (
            total_length[:, None]
            * sample_fraction[None, :]
        )
        segment_index = torch.searchsorted(
            cumulative.contiguous(),
            target_arc.contiguous(),
            right=True,
        ) - 1
        segment_index = segment_index.clamp(0, curves.shape[1] - 2)
        gather_index = segment_index[..., None].expand(-1, -1, 3)
        point_a = torch.gather(safe_curves, 1, gather_index)
        point_b = torch.gather(safe_curves, 1, gather_index + 1)
        arc_a = torch.gather(cumulative, 1, segment_index)
        link_length = torch.gather(segment_lengths, 1, segment_index)
        fraction = (
            (target_arc - arc_a) / link_length.clamp_min(1e-9)
        ).clamp(0.0, 1.0)
        sampled = point_a + fraction[..., None] * (point_b - point_a)
        return sampled, total_length, curve_valid

    @torch.inference_mode()
    def attribute_batch(
        self,
        graph_edges: tuple[GraphEdgeObservation, ...],
        predicted_curves: torch.Tensor,
        prediction_spread_m: torch.Tensor,
        initialized: np.ndarray,
        use_temporal_identity: bool = True,
    ) -> GraphAttributionBatch:
        """Associate all observed edges and retain the result on the device."""

        started = time.perf_counter()
        initialized_np = np.asarray(initialized, dtype=bool).reshape(2)
        spread = torch.as_tensor(
            prediction_spread_m,
            device=self.device,
            dtype=self.dtype,
        ).reshape(2)
        if not graph_edges:
            empty_float = torch.empty(
                (0,),
                device=self.device,
                dtype=self.dtype,
            )
            empty_long = torch.empty(
                (0,),
                device=self.device,
                dtype=torch.int64,
            )
            empty_probability = empty_float.reshape(0, 3)
            if use_temporal_identity:
                self._remember_temporal_identity(
                    empty_float.reshape(0, 0, 3),
                    empty_probability,
                    empty_long.to(torch.bool),
                )
            else:
                self.clear_temporal_history()
            return GraphAttributionBatch(
                graph_edges=(),
                observed_xyz=empty_float.reshape(0, 0, 3),
                edge_lengths_m=empty_float,
                edge_valid=empty_long.to(torch.bool),
                best_energy=empty_float.reshape(2, 0),
                best_residual_m=empty_float.reshape(2, 0),
                best_arc_start_m=empty_float.reshape(2, 0),
                best_arc_end_m=empty_float.reshape(2, 0),
                best_model_xyz=empty_float.reshape(2, 0, 0, 3),
                probabilities=empty_probability,
                selected_cable=empty_long,
                selected_confidence=empty_float,
                endpoint_anchored=empty_long.to(torch.bool),
                temporal_prior_applied=empty_long.to(torch.bool),
                temporal_match_distance_m=empty_float,
                prediction_spread_m=spread,
                initialized=np.ascontiguousarray(initialized_np),
                processing_ms=float(
                    (time.perf_counter() - started) * 1000.0
                ),
                gpu_start=None,
                gpu_end=None,
            )

        observed_count = max(2, int(len(graph_edges[0].xyz)))
        observed_np = np.zeros(
            (len(graph_edges), observed_count, 3),
            dtype=np.float32,
        )
        edge_valid_np = np.zeros(len(graph_edges), dtype=bool)
        edge_lengths_np = np.zeros(len(graph_edges), dtype=np.float32)
        for edge_index, edge in enumerate(graph_edges):
            sampled = np.asarray(edge.xyz, dtype=np.float32)
            length_m = float(edge.length_m)
            if (
                sampled.shape != (observed_count, 3)
                or not np.all(np.isfinite(sampled))
                or not np.isfinite(length_m)
                or length_m <= 1e-9
            ):
                continue
            observed_np[edge_index] = sampled
            edge_lengths_np[edge_index] = length_m
            edge_valid_np[edge_index] = True

        node_count = int(predicted_curves.shape[1])
        dense_count = (
            max(1, node_count - 1) * self.config.curve_samples_per_segment + 1
        )
        grid = self._candidate_grid(dense_count, observed_count)
        candidate_count = len(grid.starts_np)
        valid_np = np.broadcast_to(
            initialized_np[:, None, None],
            (2, len(graph_edges), candidate_count),
        ).copy()
        valid_np &= edge_valid_np[None, :, None]
        anchored_identity_np = np.full(
            len(graph_edges),
            -1,
            dtype=np.int64,
        )
        endpoint_anchored_np = np.zeros(len(graph_edges), dtype=bool)
        for edge_index, edge in enumerate(graph_edges):
            anchored_cables = {
                int(anchor.cable_index) for anchor in edge.endpoint_anchors
            }
            endpoint_anchored_np[edge_index] = bool(anchored_cables)
            if len(anchored_cables) == 1:
                anchored_identity_np[edge_index] = next(iter(anchored_cables))
            for cable_index in range(2):
                start_constraint, end_constraint, identity_allowed = (
                    _interval_end_constraints(
                        edge,
                        cable_index,
                        dense_count - 1,
                    )
                )
                if edge.endpoint_anchors and not identity_allowed:
                    valid_np[cable_index, edge_index] = False
                    continue
                if start_constraint is not None:
                    valid_np[cable_index, edge_index] &= (
                        grid.starts_np == start_constraint
                    )
                if end_constraint is not None:
                    valid_np[cable_index, edge_index] &= (
                        grid.ends_np == end_constraint
                    )

        gpu_start = gpu_end = None
        if self.device.type == "cuda":
            gpu_start = torch.cuda.Event(enable_timing=True)
            gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()

        dense_curves, curve_lengths, curve_valid = (
            self._resample_predicted_curves(predicted_curves, dense_count)
        )
        point_a = dense_curves[:, grid.lower]
        point_b = dense_curves[:, grid.upper]
        candidate_curves = point_a + grid.alpha[None, :, :, None] * (
            point_b - point_a
        )
        observed = torch.as_tensor(
            observed_np,
            device=self.device,
            dtype=self.dtype,
        )
        edge_lengths = torch.as_tensor(
            edge_lengths_np,
            device=self.device,
            dtype=self.dtype,
        )
        residual = torch.linalg.vector_norm(
            candidate_curves[:, None] - observed[None, :, None],
            dim=-1,
        )
        distance_sigma = torch.sqrt(
            self.config.distance_sigma_m**2 + spread.square()
        ).clamp_min(1e-6)
        distance_energy = _huber(
            residual / distance_sigma[:, None, None, None],
            self.config.huber_delta,
        ).mean(dim=-1)
        interval_length = (
            curve_lengths[:, None, None]
            * grid.length_fraction[None, None, :]
        )
        length_energy = _huber(
            (
                interval_length
                - edge_lengths[None, :, None]
            )
            / self.config.length_sigma_m,
            self.config.huber_delta,
        )
        energy = distance_energy + self.config.length_weight * length_energy
        valid = torch.as_tensor(
            valid_np,
            device=self.device,
            dtype=torch.bool,
        )
        valid &= curve_valid[:, None, None]
        energy = torch.where(
            valid,
            energy,
            torch.full_like(energy, torch.inf),
        )
        best_energy, best_index = torch.min(energy, dim=-1)
        mean_residual = residual.mean(dim=-1)
        best_residual = torch.gather(
            mean_residual,
            dim=-1,
            index=best_index[..., None],
        )[..., 0]
        best_arc_start = (
            curve_lengths[:, None] * grid.start_fraction[best_index]
        )
        best_arc_end = (
            curve_lengths[:, None] * grid.end_fraction[best_index]
        )
        best_model_xyz = torch.gather(
            candidate_curves[:, None].expand(
                -1,
                len(graph_edges),
                -1,
                -1,
                -1,
            ),
            dim=2,
            index=best_index[:, :, None, None, None].expand(
                -1,
                -1,
                1,
                observed_count,
                3,
            ),
        )[:, :, 0]
        probability_energy = torch.cat(
            (
                best_energy.transpose(0, 1),
                torch.full(
                    (len(graph_edges), 1),
                    self.config.unassigned_energy,
                    device=self.device,
                    dtype=self.dtype,
                ),
            ),
            dim=1,
        )
        edge_valid = torch.as_tensor(
            edge_valid_np,
            device=self.device,
            dtype=torch.bool,
        )
        if use_temporal_identity:
            temporal_prior, temporal_prior_applied, temporal_match_distance = (
                self._temporal_identity_prior(observed, edge_valid)
            )
            probability_energy = probability_energy - (
                self.config.temporal_identity_weight
                * torch.log(temporal_prior.clamp_min(1e-6))
            )
        else:
            self.clear_temporal_history()
            temporal_prior_applied = torch.zeros_like(edge_valid)
            temporal_match_distance = torch.full(
                edge_valid.shape,
                torch.inf,
                device=self.device,
                dtype=self.dtype,
            )
        probability = torch.softmax(-probability_energy, dim=1)
        anchored_identity = torch.as_tensor(
            anchored_identity_np,
            device=self.device,
            dtype=torch.int64,
        )
        anchored_mask = anchored_identity >= 0
        anchored_probability = torch.zeros_like(probability)
        anchored_probability.scatter_(
            1,
            anchored_identity.clamp_min(0)[:, None],
            1.0,
        )
        probability = torch.where(
            anchored_mask[:, None],
            anchored_probability,
            probability,
        )
        cable_probability = probability[:, :2]
        best_cable = torch.argmax(cable_probability, dim=1)
        selected_confidence = torch.gather(
            cable_probability,
            1,
            best_cable[:, None],
        )[:, 0]
        selected_fit_energy = torch.gather(
            best_energy.transpose(0, 1),
            1,
            best_cable[:, None],
        )[:, 0]
        selected_valid = (
            torch.isfinite(selected_fit_energy)
            & (
                selected_confidence
                >= self.config.minimum_identity_probability
            )
            & (selected_confidence > probability[:, 2])
        )
        selected_cable = torch.where(
            selected_valid,
            best_cable,
            torch.full_like(best_cable, -1),
        )
        if use_temporal_identity:
            self._remember_temporal_identity(
                observed,
                probability,
                edge_valid,
            )

        if gpu_end is not None:
            gpu_end.record()
        return GraphAttributionBatch(
            graph_edges=graph_edges,
            observed_xyz=observed,
            edge_lengths_m=edge_lengths,
            edge_valid=edge_valid,
            best_energy=best_energy,
            best_residual_m=best_residual,
            best_arc_start_m=best_arc_start,
            best_arc_end_m=best_arc_end,
            best_model_xyz=best_model_xyz,
            probabilities=probability,
            selected_cable=selected_cable,
            selected_confidence=selected_confidence,
            endpoint_anchored=torch.as_tensor(
                endpoint_anchored_np,
                device=self.device,
                dtype=torch.bool,
            ),
            temporal_prior_applied=temporal_prior_applied,
            temporal_match_distance_m=temporal_match_distance,
            prediction_spread_m=spread,
            initialized=np.ascontiguousarray(initialized_np),
            processing_ms=float(
                (time.perf_counter() - started) * 1000.0
            ),
            gpu_start=gpu_start,
            gpu_end=gpu_end,
        )

    def diagnostics(
        self,
        batch: GraphAttributionBatch,
    ) -> GraphAttributionDiagnostics:
        """Convert an already-computed device batch into viewer diagnostics."""

        if batch.edge_count == 0:
            return GraphAttributionDiagnostics(
                edges=(),
                prediction_spread_m=np.ascontiguousarray(
                    batch.prediction_spread_m.cpu().numpy(),
                    dtype=np.float32,
                ),
                initialized=np.ascontiguousarray(batch.initialized),
                attributed_count=0,
                ambiguous_count=0,
                unassigned_count=0,
                temporally_matched_count=0,
                processing_ms=float(batch.processing_ms),
                gpu_ms=0.0,
            )

        fit_values = torch.stack(
            (
                batch.best_energy,
                batch.best_residual_m,
                batch.best_arc_start_m,
                batch.best_arc_end_m,
            ),
            dim=-1,
        ).permute(1, 0, 2)
        result = torch.cat(
            (
                fit_values.reshape(batch.edge_count, 8),
                batch.probabilities,
                batch.selected_cable[:, None].to(self.dtype),
                batch.selected_confidence[:, None],
                batch.temporal_prior_applied[:, None].to(self.dtype),
                batch.temporal_match_distance_m[:, None],
            ),
            dim=1,
        ).cpu().numpy()
        spread_np = np.ascontiguousarray(
            batch.prediction_spread_m.cpu().numpy(),
            dtype=np.float32,
        )
        gpu_ms = (
            float(batch.gpu_start.elapsed_time(batch.gpu_end))
            if batch.gpu_start is not None and batch.gpu_end is not None
            else 0.0
        )
        attributions = []
        for edge_index, edge in enumerate(batch.graph_edges):
            fit = result[edge_index, :8].reshape(2, 4)
            probability_np = np.asarray(
                result[edge_index, 8:11],
                dtype=np.float64,
            )
            cable_probabilities = probability_np[:2]
            selected_cable = int(round(float(result[edge_index, 11])))
            confidence = float(result[edge_index, 12])
            temporal_prior_applied = bool(round(float(result[edge_index, 13])))
            temporal_match_distance_m = float(result[edge_index, 14])
            if selected_cable >= 0:
                residual_m = float(fit[selected_cable, 1])
                arc_start_m = float(fit[selected_cable, 2])
                arc_end_m = float(fit[selected_cable, 3])
            else:
                residual_m = float("nan")
                arc_start_m = float("nan")
                arc_end_m = float("nan")
            attributions.append(
                GraphEdgeAttribution(
                    component_label=int(edge.component_label),
                    edge_id=int(edge.edge_id),
                    cable_probabilities=np.ascontiguousarray(
                        cable_probabilities,
                        dtype=np.float32,
                    ),
                    unassigned_probability=float(probability_np[2]),
                    selected_cable=selected_cable,
                    confidence=confidence,
                    arc_start_m=arc_start_m,
                    arc_end_m=arc_end_m,
                    forward=bool(
                        selected_cable >= 0 and arc_end_m >= arc_start_m
                    ),
                    mean_residual_m=residual_m,
                    endpoint_anchored=bool(
                        batch.endpoint_anchored[edge_index].item()
                    ),
                    temporal_prior_applied=temporal_prior_applied,
                    temporal_match_distance_m=temporal_match_distance_m,
                )
            )

        attributed_count = sum(
            edge.selected_cable >= 0 for edge in attributions
        )
        unassigned_count = sum(
            edge.unassigned_probability
            >= max(float(edge.cable_probabilities.max()), 0.0)
            for edge in attributions
        )
        ambiguous_count = (
            len(attributions) - attributed_count - unassigned_count
        )
        temporally_matched_count = sum(
            edge.temporal_prior_applied for edge in attributions
        )
        return GraphAttributionDiagnostics(
            edges=tuple(attributions),
            prediction_spread_m=spread_np,
            initialized=np.ascontiguousarray(batch.initialized),
            attributed_count=int(attributed_count),
            ambiguous_count=int(max(0, ambiguous_count)),
            unassigned_count=int(unassigned_count),
            temporally_matched_count=int(temporally_matched_count),
            processing_ms=float(batch.processing_ms),
            gpu_ms=gpu_ms,
        )
