"""Batched fixed-length particle hypotheses for two endpoint-identified cables."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from cuda_constraints import FusedCudaConstraints
from graph_attribution import (
    GraphAttributionBatch,
    GraphAttributionConfig,
    GraphAttributionDiagnostics,
    GraphEdgeAttributor,
)
from observation import FrameObservation


VISIBILITY_UNKNOWN = 0
VISIBILITY_SUPPORTED = 1
VISIBILITY_MISSING = 2


@dataclass(frozen=True)
class ParticleFeatures:
    connected_trace: bool = True
    smoothness: bool = True
    fixed_length: bool = True
    temporal_prediction: bool = True
    endpoint_motion_transport: bool = True
    ess_resampling: bool = True
    global_particles: bool = True
    local_node_motion: bool = True
    single_endpoint_updates: bool = True
    prediction_without_measurement: bool = True
    posterior_uncertainty: bool = True
    fused_constraint_kernels: bool = True
    cuda_graph_replay: bool = True
    graph_edge_attribution: bool = True
    temporal_edge_identity: bool = True
    visible_edge_scoring: bool = True
    visible_edge_transport: bool = True
    visible_edge_exploration: bool = True

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ParticleFeatures":
        values = values or {}
        defaults = cls()
        return cls(**{
            field: bool(values.get(field, getattr(defaults, field)))
            for field in cls.__dataclass_fields__
        })

    def summary(self) -> str:
        enabled = [
            name.replace("_", " ")
            for name in self.__dataclass_fields__
            if bool(getattr(self, name))
        ]
        return ", ".join(enabled) if enabled else "none"


@dataclass(frozen=True)
class ParticleFilterConfig:
    device: str = "cuda"
    particle_count: int = 1024
    node_count: int = 13
    cable_lengths_m: tuple[float, float] = (0.515, 0.515)
    cable_diameter_m: float = 0.009
    dense_samples_per_segment: int = 4
    deformation_modes: int = 5
    initial_noise_m: float = 0.010
    global_noise_m: float = 0.018
    global_fraction: float = 0.10
    ess_resample_fraction: float = 0.50
    acceleration_noise_mps2: float = 0.45
    velocity_damping_per_second: float = 1.25
    endpoint_velocity_smoothing: float = 0.50
    local_velocity_gain: float = 0.65
    local_support_radius_nodes: float = 1.0
    maximum_node_velocity_mps: float = 2.50
    projection_iterations: int = 8
    velocity_projection_iterations: int = 2
    endpoint_tolerance_m: float = 0.003
    trace_sigma_m: float = 0.008
    huber_delta: float = 1.5
    smoothness_weight: float = 2.0
    support_distance_m: float = 0.015
    maximum_unsupported_length_m: float = 0.080
    unsupported_weight: float = 1.0
    attribution_curve_samples_per_segment: int = 4
    attribution_distance_sigma_m: float = 0.012
    attribution_length_sigma_m: float = 0.025
    attribution_length_weight: float = 0.25
    attribution_unassigned_energy: float = 3.0
    attribution_minimum_identity_probability: float = 0.60
    attribution_temporal_identity_weight: float = 1.0
    attribution_temporal_match_sigma_m: float = 0.020
    attribution_temporal_match_gate_m: float = 0.050
    attribution_temporal_history_frames: int = 5
    attribution_temporal_history_decay: float = 0.75
    attribution_temporal_probability_floor: float = 0.05
    top_particle_count: int = 12
    random_seed: int = 7
    profile_stages: bool = True

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ParticleFilterConfig":
        values = values or {}
        lengths = tuple(float(item) for item in values.get("cable_lengths_m", (0.515, 0.515)))
        if len(lengths) != 2 or any(not np.isfinite(item) or item <= 0.0 for item in lengths):
            raise ValueError("particle_filter.cable_lengths_m must contain two positive lengths")
        cable_diameter_m = float(values.get("cable_diameter_m", 0.009))
        if not np.isfinite(cable_diameter_m) or cable_diameter_m <= 0.0:
            raise ValueError("particle_filter.cable_diameter_m must be positive")
        if cable_diameter_m >= min(lengths):
            raise ValueError(
                "particle_filter.cable_diameter_m must be smaller than both cable lengths"
            )
        return cls(
            device=str(values.get("device", "cuda")),
            particle_count=max(32, int(values.get("particle_count", 1024))),
            node_count=max(4, int(values.get("node_count", 13))),
            cable_lengths_m=(lengths[0], lengths[1]),
            cable_diameter_m=cable_diameter_m,
            dense_samples_per_segment=max(1, int(values.get("dense_samples_per_segment", 4))),
            deformation_modes=max(1, int(values.get("deformation_modes", 5))),
            initial_noise_m=max(0.0, float(values.get("initial_noise_m", 0.010))),
            global_noise_m=max(0.0, float(values.get("global_noise_m", 0.018))),
            global_fraction=float(np.clip(values.get("global_fraction", 0.10), 0.0, 1.0)),
            ess_resample_fraction=float(
                np.clip(values.get("ess_resample_fraction", 0.50), 0.01, 1.0)
            ),
            acceleration_noise_mps2=max(
                0.0, float(values.get("acceleration_noise_mps2", 0.45))
            ),
            velocity_damping_per_second=max(
                0.0, float(values.get("velocity_damping_per_second", 1.25))
            ),
            endpoint_velocity_smoothing=float(
                np.clip(values.get("endpoint_velocity_smoothing", 0.50), 0.0, 0.999)
            ),
            local_velocity_gain=float(
                np.clip(values.get("local_velocity_gain", 0.65), 0.0, 1.0)
            ),
            local_support_radius_nodes=max(
                0.1, float(values.get("local_support_radius_nodes", 1.0))
            ),
            maximum_node_velocity_mps=max(
                1e-3, float(values.get("maximum_node_velocity_mps", 2.50))
            ),
            projection_iterations=max(1, int(values.get("projection_iterations", 8))),
            velocity_projection_iterations=max(
                1, int(values.get("velocity_projection_iterations", 2))
            ),
            endpoint_tolerance_m=max(1e-6, float(values.get("endpoint_tolerance_m", 0.003))),
            trace_sigma_m=max(1e-6, float(values.get("trace_sigma_m", 0.008))),
            huber_delta=max(1e-3, float(values.get("huber_delta", 1.5))),
            smoothness_weight=max(
                0.0, float(values.get("smoothness_weight", 2.0))
            ),
            support_distance_m=max(1e-6, float(values.get("support_distance_m", 0.015))),
            maximum_unsupported_length_m=max(
                0.0, float(values.get("maximum_unsupported_length_m", 0.080))
            ),
            unsupported_weight=max(0.0, float(values.get("unsupported_weight", 1.0))),
            attribution_curve_samples_per_segment=max(
                1,
                int(values.get("attribution_curve_samples_per_segment", 4)),
            ),
            attribution_distance_sigma_m=max(
                1e-6,
                float(values.get("attribution_distance_sigma_m", 0.012)),
            ),
            attribution_length_sigma_m=max(
                1e-6,
                float(values.get("attribution_length_sigma_m", 0.025)),
            ),
            attribution_length_weight=max(
                0.0,
                float(values.get("attribution_length_weight", 0.25)),
            ),
            attribution_unassigned_energy=max(
                0.0,
                float(values.get("attribution_unassigned_energy", 3.0)),
            ),
            attribution_minimum_identity_probability=float(
                np.clip(
                    values.get(
                        "attribution_minimum_identity_probability",
                        0.60,
                    ),
                    0.0,
                    1.0,
                )
            ),
            attribution_temporal_identity_weight=max(
                0.0,
                float(values.get("attribution_temporal_identity_weight", 1.0)),
            ),
            attribution_temporal_match_sigma_m=max(
                1e-6,
                float(values.get("attribution_temporal_match_sigma_m", 0.020)),
            ),
            attribution_temporal_match_gate_m=max(
                1e-6,
                float(values.get("attribution_temporal_match_gate_m", 0.050)),
            ),
            attribution_temporal_history_frames=max(
                1,
                int(values.get("attribution_temporal_history_frames", 5)),
            ),
            attribution_temporal_history_decay=float(
                np.clip(
                    values.get("attribution_temporal_history_decay", 0.75),
                    0.0,
                    1.0,
                )
            ),
            attribution_temporal_probability_floor=float(
                np.clip(
                    values.get("attribution_temporal_probability_floor", 0.05),
                    0.0,
                    1.0 / 3.0,
                )
            ),
            top_particle_count=max(1, int(values.get("top_particle_count", 12))),
            random_seed=int(values.get("random_seed", 7)),
            profile_stages=bool(values.get("profile_stages", True)),
        )


@dataclass(frozen=True)
class CableFilterDiagnostics:
    initialized: bool
    measurement_valid: bool
    status: str
    selected_weight: float
    effective_sample_size: float
    resampled: bool
    trace_mean_mm: float
    trace_rms_mm: float
    trace_p95_mm: float
    trace_max_mm: float
    support_fraction: float
    longest_unsupported_mm: float
    maximum_link_error_mm: float
    endpoint_error_mm: float
    observed_route_length_m: float
    route_candidate_count: int
    selected_route_index: int
    route_length_error_mm: float
    turn_rms_degrees: float
    tracking_state: str
    endpoint_visible_count: int
    visible_fraction: float
    missing_fraction: float
    unknown_fraction: float
    maximum_node_uncertainty_mm: float
    complete_observation_age_s: float
    predicted_node_speed_mps: float
    local_velocity_innovation_speed_mps: float
    corrected_node_speed_mps: float
    maximum_corrected_node_speed_mps: float
    local_node_motion_applied: bool
    locally_supported_node_fraction: float
    attributed_edge_count: int
    measurement_source: str


@dataclass(frozen=True)
class CableFilterOutput:
    curve: np.ndarray
    dense_curve: np.ndarray
    dense_visibility: np.ndarray
    node_visibility: np.ndarray
    node_covariance: np.ndarray
    top_particles: np.ndarray
    diagnostics: CableFilterDiagnostics


@dataclass(frozen=True)
class ParticleFilterFrame:
    cables: tuple[CableFilterOutput, CableFilterOutput]
    cable_radius_m: float
    initialized: tuple[bool, bool]
    measurement_used: tuple[bool, bool]
    processing_ms: float
    gpu_ms: float
    active_features: str
    profile: "ParticleFilterProfile | None" = None
    diagnostics_refreshed: bool = True
    graph_attribution: GraphAttributionDiagnostics | None = None


@dataclass(frozen=True)
class ParticleFilterProfile:
    """One-frame PF stage timings without additional CUDA synchronization."""

    route_input_cpu_ms: float
    inputs_gpu_ms: float
    graph_attribution_gpu_ms: float
    prediction_gpu_ms: float
    prediction_constraint_gpu_ms: float
    proposal_gpu_ms: float
    measurement_constraint_gpu_ms: float
    local_motion_gpu_ms: float
    route_score_gpu_ms: float
    visible_edge_score_gpu_ms: float
    regularization_gpu_ms: float
    weight_update_gpu_ms: float
    posterior_gpu_ms: float
    estimate_constraint_gpu_ms: float
    diagnostics_gpu_ms: float
    readback_cpu_ms: float
    graph_attribution_wall_ms: float


@dataclass(frozen=True)
class _PendingCableOutput:
    """Current estimate plus an optional fresh diagnostic payload."""

    estimate_payload: torch.Tensor
    state_payload: torch.Tensor
    diagnostic_float_payload: torch.Tensor | None
    diagnostic_visibility_payload: torch.Tensor | None
    endpoint_visible_count: int
    route_candidate_count: int
    dense_count: int
    top_count: int


@dataclass(frozen=True)
class _CachedCableDiagnostics:
    """Viewer-only values retained between diagnostic refresh frames."""

    dense_visibility: np.ndarray
    node_visibility: np.ndarray
    node_covariance: np.ndarray
    top_particles: np.ndarray
    diagnostics: CableFilterDiagnostics


@dataclass(frozen=True)
class _ConstraintGraphEntry:
    graph: torch.cuda.CUDAGraph
    particles: torch.Tensor
    endpoints: torch.Tensor
    output: torch.Tensor


@dataclass(frozen=True)
class _VelocityGraphEntry:
    graph: torch.cuda.CUDAGraph
    velocities: torch.Tensor
    particles: torch.Tensor
    output: torch.Tensor


def _empty_output(status: str, measurement_valid: bool = False) -> CableFilterOutput:
    diagnostics = CableFilterDiagnostics(
        initialized=False,
        measurement_valid=measurement_valid,
        status=str(status),
        selected_weight=0.0,
        effective_sample_size=0.0,
        resampled=False,
        trace_mean_mm=float("nan"),
        trace_rms_mm=float("nan"),
        trace_p95_mm=float("nan"),
        trace_max_mm=float("nan"),
        support_fraction=0.0,
        longest_unsupported_mm=float("nan"),
        maximum_link_error_mm=float("nan"),
        endpoint_error_mm=float("nan"),
        observed_route_length_m=float("nan"),
        route_candidate_count=0,
        selected_route_index=-1,
        route_length_error_mm=float("nan"),
        turn_rms_degrees=float("nan"),
        tracking_state="LOST",
        endpoint_visible_count=0,
        visible_fraction=0.0,
        missing_fraction=0.0,
        unknown_fraction=1.0,
        maximum_node_uncertainty_mm=float("nan"),
        complete_observation_age_s=float("inf"),
        predicted_node_speed_mps=float("nan"),
        local_velocity_innovation_speed_mps=float("nan"),
        corrected_node_speed_mps=float("nan"),
        maximum_corrected_node_speed_mps=float("nan"),
        local_node_motion_applied=False,
        locally_supported_node_fraction=0.0,
        attributed_edge_count=0,
        measurement_source="none",
    )
    return CableFilterOutput(
        curve=np.empty((0, 3), dtype=np.float32),
        dense_curve=np.empty((0, 3), dtype=np.float32),
        dense_visibility=np.empty(0, dtype=np.uint8),
        node_visibility=np.empty(0, dtype=np.uint8),
        node_covariance=np.empty((0, 3, 3), dtype=np.float32),
        top_particles=np.empty((0, 0, 3), dtype=np.float32),
        diagnostics=diagnostics,
    )


def _resample_curve(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return np.empty((0, 3), dtype=np.float32)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 1e-9))
    points = points[keep]
    if len(points) < 2:
        return np.repeat(points[:1], count, axis=0).astype(np.float32)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    target = np.linspace(0.0, float(cumulative[-1]), int(count), dtype=np.float64)
    result = np.empty((count, 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(target, cumulative, points[:, axis])
    return np.ascontiguousarray(result, dtype=np.float32)


def _dense_samples(
    particles: torch.Tensor,
    samples_per_segment: int,
    alpha: torch.Tensor | None = None,
) -> torch.Tensor:
    if alpha is None:
        alpha = torch.arange(
            int(samples_per_segment), device=particles.device, dtype=particles.dtype
        ) / float(samples_per_segment)
    starts = particles[:, :, :-1, :, None]
    differences = (particles[:, :, 1:, :] - particles[:, :, :-1, :])[:, :, :, :, None]
    samples = starts + differences * alpha.view(1, 1, 1, 1, -1)
    samples = samples.permute(0, 1, 2, 4, 3).reshape(
        particles.shape[0], particles.shape[1], -1, 3
    )
    return torch.cat((samples, particles[:, :, -1:, :]), dim=2)


def _clamp_vector_norm(vectors: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    """Limit vector magnitude without changing direction."""

    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    scale = torch.clamp(float(maximum_norm) / norms.clamp_min(1e-8), max=1.0)
    return vectors * scale


def _unordered_graph_residual(
    dense: torch.Tensor,
    route_dense: torch.Tensor,
    route_valid: torch.Tensor,
    particle_chunk: int = 128,
) -> torch.Tensor:
    """Ablation score: nearest distance to the unordered union of graph trails."""

    output = torch.full(
        dense.shape[:-1],
        torch.inf,
        device=dense.device,
        dtype=dense.dtype,
    )
    for cable_index in range(dense.shape[0]):
        graph_points = route_dense[cable_index, route_valid[cable_index]].reshape(-1, 3)
        if len(graph_points) == 0:
            continue
        for start in range(0, dense.shape[1], particle_chunk):
            stop = min(dense.shape[1], start + particle_chunk)
            distances = torch.cdist(
                dense[cable_index, start:stop],
                graph_points[None, :, :].expand(stop - start, -1, -1),
            )
            output[cable_index, start:stop] = distances.amin(dim=-1)
    return output


def _constrain_fixed_links(
    particles: torch.Tensor,
    endpoints: torch.Tensor,
    endpoint_visible: tuple[tuple[bool, bool], tuple[bool, bool]],
    segment_lengths: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    output = particles.clone()
    epsilon = torch.finfo(output.dtype).eps
    for cable_index in range(output.shape[0]):
        segment = segment_lengths[cable_index]
        start_visible = endpoint_visible[cable_index][0]
        end_visible = endpoint_visible[cable_index][1]
        cable = output[cable_index]
        if start_visible and end_visible:
            for _ in range(int(iterations)):
                cable[:, -1, :] = endpoints[cable_index, 1]
                for index in range(cable.shape[1] - 2, -1, -1):
                    direction = cable[:, index, :] - cable[:, index + 1, :]
                    direction = direction / torch.linalg.vector_norm(
                        direction, dim=-1, keepdim=True
                    ).clamp_min(epsilon)
                    cable[:, index, :] = cable[:, index + 1, :] + segment * direction
                cable[:, 0, :] = endpoints[cable_index, 0]
                for index in range(cable.shape[1] - 1):
                    direction = cable[:, index + 1, :] - cable[:, index, :]
                    direction = direction / torch.linalg.vector_norm(
                        direction, dim=-1, keepdim=True
                    ).clamp_min(epsilon)
                    cable[:, index + 1, :] = cable[:, index, :] + segment * direction
        elif start_visible:
            cable[:, 0, :] = endpoints[cable_index, 0]
            for index in range(cable.shape[1] - 1):
                direction = cable[:, index + 1, :] - cable[:, index, :]
                direction = direction / torch.linalg.vector_norm(
                    direction, dim=-1, keepdim=True
                ).clamp_min(epsilon)
                cable[:, index + 1, :] = cable[:, index, :] + segment * direction
        elif end_visible:
            cable[:, -1, :] = endpoints[cable_index, 1]
            for index in range(cable.shape[1] - 2, -1, -1):
                direction = cable[:, index, :] - cable[:, index + 1, :]
                direction = direction / torch.linalg.vector_norm(
                    direction, dim=-1, keepdim=True
                ).clamp_min(epsilon)
                cable[:, index, :] = cable[:, index + 1, :] + segment * direction
        else:
            original_center = cable.mean(dim=1, keepdim=True)
            for index in range(cable.shape[1] - 1):
                direction = cable[:, index + 1, :] - cable[:, index, :]
                direction = direction / torch.linalg.vector_norm(
                    direction, dim=-1, keepdim=True
                ).clamp_min(epsilon)
                cable[:, index + 1, :] = cable[:, index, :] + segment * direction
            cable += original_center - cable.mean(dim=1, keepdim=True)
    return output


def _project_link_velocities(
    velocities: torch.Tensor,
    particles: torch.Tensor,
    endpoint_visible: tuple[tuple[bool, bool], tuple[bool, bool]],
    iterations: int,
) -> torch.Tensor:
    """Project velocity onto the fixed-link constraint tangent space."""

    output = velocities.clone()
    epsilon = torch.finfo(output.dtype).eps
    for cable_index in range(output.shape[0]):
        cable_velocity = output[cable_index]
        cable = particles[cable_index]
        for _ in range(int(iterations)):
            for index in range(cable.shape[1] - 1):
                direction = cable[:, index + 1, :] - cable[:, index, :]
                unit = direction / torch.linalg.vector_norm(
                    direction, dim=-1, keepdim=True
                ).clamp_min(epsilon)
                radial = torch.sum(
                    (cable_velocity[:, index + 1, :] - cable_velocity[:, index, :])
                    * unit,
                    dim=-1,
                    keepdim=True,
                )
                first_fixed = index == 0 and endpoint_visible[cable_index][0]
                second_fixed = (
                    index + 1 == cable.shape[1] - 1
                    and endpoint_visible[cable_index][1]
                )
                if first_fixed and not second_fixed:
                    cable_velocity[:, index + 1, :] -= radial * unit
                elif second_fixed and not first_fixed:
                    cable_velocity[:, index, :] += radial * unit
                elif not first_fixed and not second_fixed:
                    cable_velocity[:, index, :] += 0.5 * radial * unit
                    cable_velocity[:, index + 1, :] -= 0.5 * radial * unit
    return output


class BatchedCableParticleFilter:
    """Two independent filters evaluated together with a cable batch dimension."""

    def __init__(
        self,
        config: ParticleFilterConfig,
        features: ParticleFeatures,
    ):
        self.config = config
        self.features = features
        graph_attribution_config = GraphAttributionConfig(
            curve_samples_per_segment=config.attribution_curve_samples_per_segment,
            distance_sigma_m=config.attribution_distance_sigma_m,
            length_sigma_m=config.attribution_length_sigma_m,
            length_weight=config.attribution_length_weight,
            huber_delta=config.huber_delta,
            unassigned_energy=config.attribution_unassigned_energy,
            minimum_identity_probability=(
                config.attribution_minimum_identity_probability
            ),
            temporal_identity_weight=config.attribution_temporal_identity_weight,
            temporal_match_sigma_m=config.attribution_temporal_match_sigma_m,
            temporal_match_gate_m=config.attribution_temporal_match_gate_m,
            temporal_history_frames=config.attribution_temporal_history_frames,
            temporal_history_decay=config.attribution_temporal_history_decay,
            temporal_probability_floor=(
                config.attribution_temporal_probability_floor
            ),
        )
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA particle filtering was requested but torch.cuda is unavailable")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.graph_edge_attributor = GraphEdgeAttributor(
            graph_attribution_config,
            self.device,
        )
        self.dtype = torch.float32
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.random_seed)
        self.particles: torch.Tensor | None = None
        self.velocities: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.route_indices: torch.Tensor | None = None
        self.initialized = np.zeros(2, dtype=bool)
        self.last_endpoints = np.full((2, 2, 3), np.nan, dtype=np.float32)
        self.last_endpoint_timestamps = np.full((2, 2), np.nan, dtype=np.float64)
        self.endpoint_velocities = np.zeros((2, 2, 3), dtype=np.float32)
        self.last_complete_timestamps = np.full(2, np.nan, dtype=np.float64)
        self.last_timestamp: float | None = None
        self._diagnostic_cache: list[_CachedCableDiagnostics | None] = [None, None]
        self._contact_dense_support: torch.Tensor | None = None
        self.lengths = torch.tensor(config.cable_lengths_m, device=self.device, dtype=self.dtype)
        self.segment_lengths = self.lengths / float(config.node_count - 1)
        positions = torch.linspace(0.0, 1.0, config.node_count, device=self.device)
        self._node_fraction = positions.view(1, 1, config.node_count, 1)
        modes = torch.arange(
            1, config.deformation_modes + 1, device=self.device, dtype=self.dtype
        )
        self.basis = torch.sin(math.pi * positions[:, None] * modes[None, :])
        self.basis = self.basis / torch.linalg.vector_norm(self.basis, dim=0, keepdim=True).clamp_min(1e-6)
        dense_count = (
            (config.node_count - 1) * config.dense_samples_per_segment + 1
        )
        self._particle_indices = torch.arange(
            config.particle_count, device=self.device, dtype=torch.int64
        )[None, :]
        self._dense_sample_indices = torch.arange(
            dense_count, device=self.device, dtype=torch.int64
        )
        self._node_dense_indices = (
            torch.arange(config.node_count, device=self.device, dtype=torch.int64)
            * config.dense_samples_per_segment
        )
        self._dense_alpha = torch.arange(
            config.dense_samples_per_segment,
            device=self.device,
            dtype=self.dtype,
        ) / float(config.dense_samples_per_segment)
        self._initial_noise_scale = torch.full(
            (2,), config.initial_noise_m, device=self.device, dtype=self.dtype
        )
        self._all_particles_mask = torch.ones(
            config.particle_count, device=self.device, dtype=torch.bool
        )
        self._interior_node_mask = torch.ones(
            (1, 1, config.node_count, 1),
            device=self.device,
            dtype=torch.bool,
        )
        self._interior_node_mask[:, :, 0, :] = False
        self._interior_node_mask[:, :, -1, :] = False
        self._unknown_visibility = torch.full(
            (dense_count,),
            VISIBILITY_UNKNOWN,
            device=self.device,
            dtype=torch.uint8,
        )
        self._negative_one_index = torch.full(
            (), -1, device=self.device, dtype=torch.int64
        )
        self._nan_scalar = torch.full(
            (), float("nan"), device=self.device, dtype=self.dtype
        )
        self._constraint_graphs: dict[str, dict[tuple, _ConstraintGraphEntry]] = {
            "prediction": {},
            "measurement": {},
            "estimate": {},
        }
        self._velocity_graphs: dict[str, dict[tuple, _VelocityGraphEntry]] = {
            "prediction": {},
            "measurement": {},
        }
        self._fused_constraints: FusedCudaConstraints | None = None
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                self._fused_constraints = FusedCudaConstraints(self.device)
                self._capture_stream = torch.cuda.Stream(device=self.device)
                self._graph_pools = {
                    "constraint_prediction": torch.cuda.graph_pool_handle(),
                    "constraint_measurement": torch.cuda.graph_pool_handle(),
                    "constraint_estimate": torch.cuda.graph_pool_handle(),
                    "velocity_prediction": torch.cuda.graph_pool_handle(),
                    "velocity_measurement": torch.cuda.graph_pool_handle(),
                }
                self._profile_events = {
                    name: torch.cuda.Event(enable_timing=True)
                    for name in (
                        "start",
                        "inputs",
                        "attribution",
                        "prediction",
                        "prediction_constraint",
                        "proposal",
                        "measurement_constraint",
                        "local_motion",
                        "route_score",
                        "visible_edge_score",
                        "regularization",
                        "weights",
                        "posterior",
                        "estimate_constraint",
                        "diagnostics",
                        "end",
                    )
                }
        else:
            self._capture_stream = None
            self._graph_pools = {}
            self._profile_events = {}

    def set_features(self, features: ParticleFeatures) -> None:
        temporal_identity_changed = (
            self.features.temporal_edge_identity
            != features.temporal_edge_identity
        )
        pf_feature_changed = any(
            getattr(self.features, name) != getattr(features, name)
            for name in ParticleFeatures.__dataclass_fields__
            if name != "graph_edge_attribution"
        )
        self.features = features
        if temporal_identity_changed:
            self.graph_edge_attributor.clear_temporal_history()
        if pf_feature_changed:
            self._diagnostic_cache = [None, None]

    def posterior_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return read-only references to the current CUDA posterior state.

        Downstream passive observers may evaluate these tensors after
        :meth:`update` completes. They must not modify them.
        """

        if (
            self.particles is None
            or self.velocities is None
            or self.weights is None
        ):
            raise RuntimeError("The cable posterior has not been allocated.")
        return self.particles, self.velocities, self.weights

    def contact_support_tensor(self) -> torch.Tensor:
        """Return the current read-only dense cable-arc support mask on CUDA."""

        if self._contact_dense_support is None:
            raise RuntimeError(
                "Current contact support was not requested from the PF update."
            )
        return self._contact_dense_support

    def _capture_constraint_graph(
        self,
        particles: torch.Tensor,
        endpoints: torch.Tensor,
        endpoint_visible: tuple[tuple[bool, bool], tuple[bool, bool]],
        slot: str,
    ) -> _ConstraintGraphEntry:
        assert self._capture_stream is not None
        static_particles = torch.empty_like(particles)
        static_endpoints = torch.empty_like(endpoints)
        static_particles.copy_(particles)
        static_endpoints.copy_(endpoints)
        capture_stream = self._capture_stream
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                _constrain_fixed_links(
                    static_particles,
                    static_endpoints,
                    endpoint_visible,
                    self.segment_lengths,
                    self.config.projection_iterations,
                )
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)
        with torch.cuda.device(self.device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=self._graph_pools[f"constraint_{slot}"],
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                output = _constrain_fixed_links(
                    static_particles,
                    static_endpoints,
                    endpoint_visible,
                    self.segment_lengths,
                    self.config.projection_iterations,
                )
        return _ConstraintGraphEntry(
            graph=graph,
            particles=static_particles,
            endpoints=static_endpoints,
            output=output,
        )

    def _constrain_links(
        self,
        particles: torch.Tensor,
        endpoints: torch.Tensor,
        endpoint_visible: tuple[tuple[bool, bool], tuple[bool, bool]],
        slot: str,
    ) -> torch.Tensor:
        # PyTorch's unanchored branch preserves the particle centroid with a
        # parallel reduction. Retain it whenever a cable has no endpoint so
        # the optimized and reference filters remain bit-identical over time.
        fused_position_supported = all(
            start_visible or end_visible
            for start_visible, end_visible in endpoint_visible
        )
        if (
            self.device.type == "cuda"
            and self.features.fused_constraint_kernels
            and fused_position_supported
        ):
            if self._fused_constraints is None:
                raise RuntimeError("Fused CUDA constraints were not initialized")
            return self._fused_constraints.constrain(
                particles,
                endpoints,
                endpoint_visible,
                self.segment_lengths,
                self.config.projection_iterations,
            )
        if self.device.type != "cuda" or not self.features.cuda_graph_replay:
            return _constrain_fixed_links(
                particles,
                endpoints,
                endpoint_visible,
                self.segment_lengths,
                self.config.projection_iterations,
            )
        cache = self._constraint_graphs[slot]
        entry = cache.get(endpoint_visible)
        if entry is None:
            entry = self._capture_constraint_graph(
                particles, endpoints, endpoint_visible, slot
            )
            cache[endpoint_visible] = entry
        entry.particles.copy_(particles)
        entry.endpoints.copy_(endpoints)
        entry.graph.replay()
        return entry.output

    def _capture_velocity_graph(
        self,
        velocities: torch.Tensor,
        particles: torch.Tensor,
        endpoint_visible: tuple[tuple[bool, bool], tuple[bool, bool]],
        slot: str,
    ) -> _VelocityGraphEntry:
        assert self._capture_stream is not None
        static_velocities = torch.empty_like(velocities)
        static_particles = torch.empty_like(particles)
        static_velocities.copy_(velocities)
        static_particles.copy_(particles)
        capture_stream = self._capture_stream
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                _project_link_velocities(
                    static_velocities,
                    static_particles,
                    endpoint_visible,
                    self.config.velocity_projection_iterations,
                )
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)
        with torch.cuda.device(self.device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph,
                pool=self._graph_pools[f"velocity_{slot}"],
                stream=capture_stream,
                capture_error_mode="thread_local",
            ):
                output = _project_link_velocities(
                    static_velocities,
                    static_particles,
                    endpoint_visible,
                    self.config.velocity_projection_iterations,
                )
        return _VelocityGraphEntry(
            graph=graph,
            velocities=static_velocities,
            particles=static_particles,
            output=output,
        )

    def _project_velocities(
        self,
        velocities: torch.Tensor,
        particles: torch.Tensor,
        endpoint_visible: tuple[tuple[bool, bool], tuple[bool, bool]],
        slot: str,
    ) -> torch.Tensor:
        if self.device.type == "cuda" and self.features.fused_constraint_kernels:
            if self._fused_constraints is None:
                raise RuntimeError("Fused CUDA constraints were not initialized")
            return self._fused_constraints.project_velocities(
                velocities,
                particles,
                endpoint_visible,
                self.config.velocity_projection_iterations,
            )
        if self.device.type != "cuda" or not self.features.cuda_graph_replay:
            return _project_link_velocities(
                velocities,
                particles,
                endpoint_visible,
                self.config.velocity_projection_iterations,
            )
        cache = self._velocity_graphs[slot]
        entry = cache.get(endpoint_visible)
        if entry is None:
            entry = self._capture_velocity_graph(
                velocities, particles, endpoint_visible, slot
            )
            cache[endpoint_visible] = entry
        entry.velocities.copy_(velocities)
        entry.particles.copy_(particles)
        entry.graph.replay()
        return entry.output

    def _allocate(self) -> None:
        if self.particles is not None:
            return
        shape = (2, self.config.particle_count, self.config.node_count, 3)
        self.particles = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.velocities = torch.zeros_like(self.particles)
        self.weights = torch.full(
            (2, self.config.particle_count),
            1.0 / self.config.particle_count,
            device=self.device,
            dtype=self.dtype,
        )
        self.route_indices = torch.zeros(
            (2, self.config.particle_count),
            device=self.device,
            dtype=torch.int64,
        )

    def _smooth_random_field(self, scale: torch.Tensor) -> torch.Tensor:
        coefficients = torch.randn(
            (2, self.config.particle_count, self.config.deformation_modes, 3),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        noise = torch.einsum("cpkd,nk->cpnd", coefficients, self.basis)
        return noise * scale.view(2, 1, 1, 1)

    def _edge_arc_samples(
        self,
        attribution: GraphAttributionBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized ordered arc samples and selected-edge flags.

        The attributor fits each observed edge to an interval of the predicted
        cable.  Its raw point order is retained, so a decreasing interval
        represents the same edge observed in the opposite orientation.
        """

        sample_fraction = torch.linspace(
            0.0,
            1.0,
            attribution.observed_count,
            device=self.device,
            dtype=self.dtype,
        )
        arc_m = (
            attribution.best_arc_start_m[:, :, None]
            + sample_fraction[None, None, :]
            * (
                attribution.best_arc_end_m
                - attribution.best_arc_start_m
            )[:, :, None]
        )
        normalized_arc = (
            arc_m / self.lengths[:, None, None].clamp_min(1e-9)
        )
        normalized_arc = torch.nan_to_num(
            normalized_arc,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        return normalized_arc, attribution.selected_mask()

    def _sample_particles_at_arc(
        self,
        particles: torch.Tensor,
        normalized_arc: torch.Tensor,
    ) -> torch.Tensor:
        """Sample all particles at common cable arc coordinates on CUDA."""

        target_index = normalized_arc * float(self.config.node_count - 1)
        lower = torch.floor(target_index).to(torch.int64).clamp(
            0,
            self.config.node_count - 1,
        )
        upper = (lower + 1).clamp_max(self.config.node_count - 1)
        alpha = target_index - lower.to(self.dtype)
        cable_count, particle_count = particles.shape[:2]
        sample_count = int(normalized_arc.shape[1] * normalized_arc.shape[2])
        lower_flat = lower.reshape(cable_count, sample_count)
        upper_flat = upper.reshape(cable_count, sample_count)
        lower_index = lower_flat[:, None, :, None].expand(
            -1,
            particle_count,
            -1,
            3,
        )
        upper_index = upper_flat[:, None, :, None].expand_as(lower_index)
        point_a = torch.gather(particles, 2, lower_index)
        point_b = torch.gather(particles, 2, upper_index)
        sampled = point_a + alpha.reshape(
            cable_count,
            1,
            sample_count,
            1,
        ) * (point_b - point_a)
        return sampled.reshape(
            cable_count,
            particle_count,
            normalized_arc.shape[1],
            normalized_arc.shape[2],
            3,
        )

    def _visible_edge_energy(
        self,
        particles: torch.Tensor,
        attribution: GraphAttributionBatch,
        normalized_arc: torch.Tensor,
        selected_edges: torch.Tensor,
    ) -> torch.Tensor:
        """Robust ordered line-to-line likelihood for attributed visible edges."""

        particle_samples = self._sample_particles_at_arc(
            particles,
            normalized_arc,
        )
        residual = torch.linalg.vector_norm(
            particle_samples
            - attribution.observed_xyz[None, None, :, :, :],
            dim=-1,
        )
        normalized = residual / self.config.trace_sigma_m
        delta = self.config.huber_delta
        huber = torch.where(
            normalized <= delta,
            0.5 * normalized.square(),
            delta * (normalized - 0.5 * delta),
        )
        edge_energy = huber.mean(dim=-1)
        edge_weight = (
            selected_edges.to(self.dtype)
            * attribution.selected_confidence[None, :]
            * attribution.edge_lengths_m[None, :].clamp_min(1e-6)
        )
        energy = torch.sum(
            edge_energy * edge_weight[:, None, :],
            dim=-1,
        ) / edge_weight.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return energy

    def _visible_edge_displacement(
        self,
        attribution: GraphAttributionBatch,
        normalized_arc: torch.Tensor,
        selected_edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Measure a local displacement and support flag for every cable node.

        Observed graph samples only influence nodes within a configurable arc
        radius.  This deliberately neither interpolates across an occlusion nor
        extrapolates a visible fragment's motion over the rest of the cable.
        """

        sample_arc = normalized_arc.reshape(2, -1)
        displacement = (
            attribution.observed_xyz[None, :, :, :]
            - attribution.best_model_xyz
        ).reshape(2, -1, 3)
        valid = selected_edges[:, :, None].expand(
            -1,
            -1,
            attribution.observed_count,
        ).reshape(2, -1)
        node_arc = torch.linspace(
            0.0,
            1.0,
            self.config.node_count,
            device=self.device,
            dtype=self.dtype,
        )[None, :, None]
        arc_distance = torch.abs(sample_arc[:, None, :] - node_arc)
        radius = (
            self.config.local_support_radius_nodes
            / float(self.config.node_count - 1)
        )
        inside = valid[:, None, :] & (arc_distance <= radius)
        confidence = (
            attribution.selected_confidence[None, :, None]
            .expand(2, -1, attribution.observed_count)
            .reshape(2, 1, -1)
        )
        weights = torch.where(
            inside,
            confidence * torch.exp(-0.5 * (arc_distance / radius).square()),
            torch.zeros_like(arc_distance),
        )
        weight_sum = weights.sum(dim=-1)
        field = torch.sum(
            weights[:, :, :, None] * displacement[:, None, :, :],
            dim=2,
        ) / weight_sum[:, :, None].clamp_min(1e-9)
        supported = weight_sum > 0.0
        field = torch.where(
            supported[:, :, None],
            field,
            torch.zeros_like(field),
        )
        return field, supported

    def _visible_edge_support(
        self,
        selected_particles: torch.Tensor,
        attribution: GraphAttributionBatch,
        normalized_arc: torch.Tensor,
        selected_edges: torch.Tensor,
        dense_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return current dense visibility and its edge residual inputs."""

        selected_samples = self._sample_particles_at_arc(
            selected_particles[:, None, :, :],
            normalized_arc,
        )[:, 0]
        residual = torch.linalg.vector_norm(
            selected_samples - attribution.observed_xyz[None, :, :, :],
            dim=-1,
        )
        flat_residual = residual.reshape(2, -1)
        flat_valid = selected_edges[:, :, None].expand(
            -1,
            -1,
            attribution.observed_count,
        ).reshape(2, -1)
        dense_arc = torch.linspace(
            0.0,
            1.0,
            dense_count,
            device=self.device,
            dtype=self.dtype,
        )
        interval_low = torch.minimum(
            normalized_arc[:, :, 0],
            normalized_arc[:, :, -1],
        )
        interval_high = torch.maximum(
            normalized_arc[:, :, 0],
            normalized_arc[:, :, -1],
        )
        covered = (
            selected_edges[:, None, :]
            & (dense_arc[None, :, None] >= interval_low[:, None, :])
            & (dense_arc[None, :, None] <= interval_high[:, None, :])
        ).any(dim=-1)
        sample_arc = normalized_arc.reshape(2, -1)
        sample_distance = torch.abs(
            dense_arc[None, :, None] - sample_arc[:, None, :]
        )
        sample_distance = torch.where(
            flat_valid[:, None, :],
            sample_distance,
            torch.full_like(sample_distance, torch.inf),
        )
        nearest_sample = torch.argmin(sample_distance, dim=-1)
        nearest_residual = torch.gather(
            flat_residual,
            1,
            nearest_sample,
        )
        supported = (
            covered
            & (nearest_residual <= self.config.support_distance_m)
        )
        visibility = torch.where(
            supported,
            torch.full(
                (),
                VISIBILITY_SUPPORTED,
                device=self.device,
                dtype=torch.uint8,
            ),
            torch.where(
                covered,
                torch.full(
                    (),
                    VISIBILITY_MISSING,
                    device=self.device,
                    dtype=torch.uint8,
                ),
                torch.full(
                    (),
                    VISIBILITY_UNKNOWN,
                    device=self.device,
                    dtype=torch.uint8,
                ),
            ),
        )
        return visibility, flat_residual, flat_valid

    def _visible_edge_trace(
        self,
        flat_residual: torch.Tensor,
        flat_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Summarize residuals only when viewer diagnostics are requested."""

        valid_count = flat_valid.sum(dim=1)
        denominator = valid_count.clamp_min(1).to(self.dtype)
        trace_mean = torch.sum(
            torch.where(flat_valid, flat_residual, 0.0),
            dim=1,
        ) / denominator
        trace_rms = torch.sqrt(
            torch.sum(
                torch.where(flat_valid, flat_residual.square(), 0.0),
                dim=1,
            )
            / denominator
        )
        sorted_residual = torch.sort(
            torch.where(
                flat_valid,
                flat_residual,
                torch.full_like(flat_residual, torch.inf),
            ),
            dim=1,
        ).values
        percentile_index = (
            torch.ceil(valid_count.to(self.dtype) * 0.95).to(torch.int64) - 1
        ).clamp_min(0)
        trace_p95 = torch.gather(
            sorted_residual,
            1,
            percentile_index[:, None],
        )[:, 0]
        trace_max = torch.where(
            flat_valid,
            flat_residual,
            torch.full_like(flat_residual, -torch.inf),
        ).amax(dim=1)
        trace = torch.stack(
            (trace_mean, trace_rms, trace_p95, trace_max),
            dim=1,
        )
        return torch.where(
            (valid_count > 0)[:, None],
            trace,
            self._nan_scalar.expand(2, 4),
        )

    def _systematic_resample_indices(self) -> torch.Tensor:
        """Low-variance batched resampling performed on the active device."""

        assert self.weights is not None
        count = self.config.particle_count
        cumulative = torch.cumsum(self.weights, dim=1)
        cumulative = cumulative / cumulative[:, -1:].clamp_min(1e-12)
        start = torch.rand(
            (2, 1),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        ) / float(count)
        positions = (
            start
            + self._particle_indices.to(self.dtype) / float(count)
        )
        return torch.searchsorted(
            cumulative.contiguous(),
            positions.contiguous(),
            right=False,
        ).clamp_max(count - 1)

    def _route_arrays(
        self, observation: FrameObservation
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[str],
    ]:
        endpoints = np.zeros((2, 2, 3), dtype=np.float32)
        endpoint_visible = np.zeros((2, 2), dtype=bool)
        maximum_routes = max(
            1,
            max(
                (len(cable.routes) for cable in observation.cables if cable.valid),
                default=0,
            ),
        )
        route_nodes = np.zeros(
            (2, maximum_routes, self.config.node_count, 3), dtype=np.float32
        )
        dense_count = (self.config.node_count - 1) * self.config.dense_samples_per_segment + 1
        route_dense = np.zeros(
            (2, maximum_routes, dense_count, 3), dtype=np.float32
        )
        route_lengths = np.zeros((2, maximum_routes), dtype=np.float32)
        route_valid = np.zeros((2, maximum_routes), dtype=bool)
        route_counts = np.zeros(2, dtype=np.int64)
        initializable = np.zeros(2, dtype=bool)
        reasons = []
        for cable_index, cable in enumerate(observation.cables):
            reason = cable.reason
            endpoint_visible[cable_index] = cable.endpoint_visible
            visible = endpoint_visible[cable_index]
            endpoints[cable_index, visible] = cable.endpoints_xyz[visible]
            routes_allowed = bool(visible.all())
            if routes_allowed:
                chord = float(np.linalg.norm(cable.endpoints_xyz[1] - cable.endpoints_xyz[0]))
                if chord > self.config.cable_lengths_m[cable_index] + 1e-6:
                    reason = (
                        f"endpoint chord {chord:.3f} m exceeds cable length "
                        f"{self.config.cable_lengths_m[cable_index]:.3f} m"
                    )
                    routes_allowed = False
            if routes_allowed:
                for route_index, route in enumerate(cable.routes[:maximum_routes]):
                    nodes = _resample_curve(route.route_xyz, self.config.node_count)
                    dense = _resample_curve(route.route_xyz, dense_count)
                    if len(nodes) != self.config.node_count or len(dense) != dense_count:
                        continue
                    route_nodes[cable_index, route_index] = nodes
                    route_dense[cable_index, route_index] = dense
                    route_lengths[cable_index, route_index] = route.length_m
                    route_valid[cable_index, route_index] = True
                route_counts[cable_index] = int(
                    np.count_nonzero(route_valid[cable_index])
                )
                initializable[cable_index] = route_counts[cable_index] > 0
            reasons.append(reason)
        return (
            initializable,
            endpoints,
            endpoint_visible,
            route_nodes,
            route_dense,
            route_lengths,
            route_valid,
            route_counts,
            reasons,
        )

    @torch.inference_mode()
    def update(
        self,
        observation: FrameObservation,
        timestamp: float,
        *,
        refresh_diagnostics: bool = True,
        require_contact_support: bool = False,
    ) -> ParticleFilterFrame:
        """Advance both filters and optionally refresh viewer-only diagnostics."""

        started = time.perf_counter()
        diagnostics_required = bool(
            refresh_diagnostics
            or any(cached is None for cached in self._diagnostic_cache)
        )
        self._contact_dense_support = None
        self._allocate()
        assert self.particles is not None
        assert self.velocities is not None
        assert self.weights is not None
        assert self.route_indices is not None
        (
            initializable_np,
            endpoints_np,
            endpoint_visible_np,
            route_nodes_np,
            route_dense_np,
            route_lengths_np,
            route_valid_np,
            route_counts_np,
            reasons,
        ) = self._route_arrays(observation)
        dt = 1.0 / 30.0 if self.last_timestamp is None else float(timestamp - self.last_timestamp)
        dt = float(np.clip(dt, 1.0 / 120.0, 0.25))
        self.last_timestamp = float(timestamp)

        effective_endpoint_visible = endpoint_visible_np.copy()
        if not self.features.single_endpoint_updates:
            for cable_index in range(2):
                if not effective_endpoint_visible[cable_index].all():
                    effective_endpoint_visible[cable_index] = False
        measured_endpoint_velocities = self.endpoint_velocities.copy()
        for cable_index in range(2):
            for endpoint_index in range(2):
                if not endpoint_visible_np[cable_index, endpoint_index]:
                    continue
                previous_time = self.last_endpoint_timestamps[cable_index, endpoint_index]
                previous_position = self.last_endpoints[cable_index, endpoint_index]
                if np.isfinite(previous_time) and np.all(np.isfinite(previous_position)):
                    elapsed = max(float(timestamp - previous_time), 1.0 / 120.0)
                    measured = (
                        endpoints_np[cable_index, endpoint_index] - previous_position
                    ) / elapsed
                    smoothing = self.config.endpoint_velocity_smoothing
                    measured_endpoint_velocities[cable_index, endpoint_index] = (
                        smoothing
                        * self.endpoint_velocities[cable_index, endpoint_index]
                        + (1.0 - smoothing) * measured
                    )
                else:
                    measured_endpoint_velocities[cable_index, endpoint_index] = 0.0

        complete_evidence_np = route_counts_np > 0

        route_input_finished = time.perf_counter()
        gpu_start = gpu_end = None
        if self.device.type == "cuda":
            if self.config.profile_stages:
                gpu_start = self._profile_events["start"]
                gpu_end = self._profile_events["end"]
            else:
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()

        endpoints = torch.as_tensor(endpoints_np, device=self.device, dtype=self.dtype)
        route_nodes = torch.as_tensor(route_nodes_np, device=self.device, dtype=self.dtype)
        route_dense = torch.as_tensor(route_dense_np, device=self.device, dtype=self.dtype)
        route_lengths = torch.as_tensor(
            route_lengths_np, device=self.device, dtype=self.dtype
        )
        route_valid = torch.as_tensor(
            route_valid_np, device=self.device, dtype=torch.bool
        )
        route_counts = torch.as_tensor(
            route_counts_np, device=self.device, dtype=torch.int64
        )
        initializable = torch.as_tensor(
            initializable_np, device=self.device, dtype=torch.bool
        )
        initialized = torch.as_tensor(self.initialized, device=self.device, dtype=torch.bool)
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["inputs"].record()

        attribution_batch = None
        attribution_curves = None
        attribution_spread = None
        attribution_initialized = self.initialized.copy()
        edge_feature_active = bool(
            self.features.visible_edge_scoring
            or self.features.visible_edge_transport
            or self.features.visible_edge_exploration
        )
        attribution_requested = bool(
            observation.graph_edges
            and (
                edge_feature_active
                or (
                    self.features.graph_edge_attribution
                    and diagnostics_required
                )
            )
        )
        damping = math.exp(-self.config.velocity_damping_per_second * dt)
        if attribution_requested:
            prior_velocity = (
                damping * self.velocities
                if self.features.temporal_prediction
                else torch.zeros_like(self.velocities)
            )
            attribution_particles = self.particles + prior_velocity * dt
            prior_weights = self.weights / self.weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
            attribution_curves = torch.sum(
                prior_weights[:, :, None, None] * attribution_particles,
                dim=1,
            )
            centered_prediction = (
                attribution_particles - attribution_curves[:, None]
            )
            attribution_spread = torch.sqrt(
                torch.sum(
                    prior_weights
                    * torch.mean(
                        torch.sum(centered_prediction.square(), dim=-1),
                        dim=-1,
                    ),
                    dim=1,
                ).clamp_min(0.0)
            )
            attribution_batch = self.graph_edge_attributor.attribute_batch(
                observation.graph_edges,
                attribution_curves,
                attribution_spread,
                attribution_initialized,
                use_temporal_identity=self.features.temporal_edge_identity,
            )
        if attribution_batch is not None:
            normalized_edge_arc, selected_edges = self._edge_arc_samples(
                attribution_batch
            )
            edge_available = selected_edges.any(dim=1) & initialized
            attributed_edge_counts = selected_edges.sum(dim=1)
        else:
            normalized_edge_arc = None
            selected_edges = None
            edge_available = torch.zeros(
                2,
                device=self.device,
                dtype=torch.bool,
            )
            attributed_edge_counts = torch.zeros(
                2,
                device=self.device,
                dtype=torch.int64,
            )
        route_available = route_counts > 0
        if self.features.visible_edge_scoring:
            edge_measurement = initialized & edge_available
            route_measurement = (~initialized) & route_available
            measurement_requested = edge_measurement | route_measurement
        else:
            edge_measurement = torch.zeros_like(initialized)
            route_measurement = route_available
            measurement_requested = route_available
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["attribution"].record()

        identity_parent_indices = self._particle_indices.expand(2, -1)
        if self.features.ess_resampling:
            sampled_parent_indices = self._systematic_resample_indices()
            prior_effective_sample_size = 1.0 / torch.sum(
                self.weights.square(),
                dim=1,
            )
            resample = (
                initialized
                & measurement_requested
                & (
                    prior_effective_sample_size
                    < (
                        self.config.ess_resample_fraction
                        * self.config.particle_count
                    )
                )
            )
        else:
            # Resampling-policy ablation: multinomial resampling is applied to
            # every real measurement, regardless of whether that measurement
            # is a complete route or a set of attributed visible edges.
            sampled_parent_indices = torch.multinomial(
                self.weights,
                self.config.particle_count,
                replacement=True,
                generator=self.generator,
            )
            resample = initialized & measurement_requested
        parent_indices = torch.where(
            resample[:, None], sampled_parent_indices, identity_parent_indices
        )
        gather_index = parent_indices[:, :, None, None].expand(
            -1, -1, self.config.node_count, 3
        )
        parent_particles = torch.gather(self.particles, 1, gather_index)
        parent_velocities = torch.gather(self.velocities, 1, gather_index)

        if self.features.temporal_prediction:
            deterministic_velocity = damping * parent_velocities
        else:
            deterministic_velocity = torch.zeros_like(parent_velocities)
        acceleration_sigma = np.full(
            2, self.config.acceleration_noise_mps2, dtype=np.float32
        )
        acceleration = self._smooth_random_field(
            torch.as_tensor(acceleration_sigma, device=self.device, dtype=self.dtype)
        )
        exploration_count = 0
        if self.features.global_particles and self.config.global_fraction > 0.0:
            exploration_count = max(
                1, int(round(self.config.particle_count * self.config.global_fraction))
            )
        incomplete_prediction = initialized & ~measurement_requested
        if exploration_count > 0:
            exploration_particle_mask = (
                self._particle_indices[0] < exploration_count
            )
        else:
            exploration_particle_mask = torch.zeros(
                self.config.particle_count,
                device=self.device,
                dtype=torch.bool,
            )
        acceleration_enabled = torch.where(
            incomplete_prediction[:, None],
            exploration_particle_mask[None, :],
            torch.ones(
                (2, self.config.particle_count),
                device=self.device,
                dtype=torch.bool,
            ),
        )
        acceleration = acceleration * acceleration_enabled[
            :, :, None, None
        ].to(self.dtype)
        predicted_velocities = deterministic_velocity + acceleration * dt
        proposals = (
            parent_particles
            + deterministic_velocity * dt
            + 0.5 * acceleration * (dt * dt)
        )
        if self.features.temporal_prediction:
            identity_velocity = damping * self.velocities
        else:
            identity_velocity = torch.zeros_like(self.velocities)
        prediction_only_velocities = identity_velocity + acceleration * dt
        prediction_only_proposals = (
            self.particles
            + identity_velocity * dt
            + 0.5 * acceleration * (dt * dt)
        )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["prediction"].record()
        if self.features.fixed_length:
            no_endpoint_anchors = ((False, False), (False, False))
            prediction_only_proposals = self._constrain_links(
                prediction_only_proposals,
                endpoints,
                no_endpoint_anchors,
                "prediction",
            )
            prediction_only_velocities = self._project_velocities(
                prediction_only_velocities,
                prediction_only_proposals,
                no_endpoint_anchors,
                "prediction",
            )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["prediction_constraint"].record()

        particle_route_slots = self._particle_indices % route_counts.clamp_min(1)[:, None]
        particle_center_index = particle_route_slots[:, :, None, None].expand(
            -1, -1, self.config.node_count, 3
        )
        particle_centers = torch.gather(
            route_nodes,
            1,
            particle_center_index,
        )

        newly_initialized_np = initializable_np & ~self.initialized
        newly_initialized = initializable & ~initialized
        if np.any(newly_initialized_np):
            initial = particle_centers.clone()
            initial = initial + self._smooth_random_field(
                self._initial_noise_scale
            )
            proposals = torch.where(newly_initialized[:, None, None, None], initial, proposals)
            predicted_velocities = torch.where(
                newly_initialized[:, None, None, None],
                torch.zeros_like(predicted_velocities),
                predicted_velocities,
            )

        edge_displacement = torch.zeros(
            (2, self.config.node_count, 3),
            device=self.device,
            dtype=self.dtype,
        )
        locally_supported_nodes = torch.zeros(
            (2, self.config.node_count),
            device=self.device,
            dtype=torch.bool,
        )
        if (
            self.features.visible_edge_transport
            and attribution_batch is not None
            and normalized_edge_arc is not None
            and selected_edges is not None
        ):
            edge_displacement, locally_supported_nodes = self._visible_edge_displacement(
                attribution_batch,
                normalized_edge_arc,
                selected_edges,
            )
            proposals = torch.where(
                edge_measurement[:, None, None, None],
                proposals + edge_displacement[:, None, :, :],
                proposals,
            )

        global_count = exploration_count
        if global_count > 0:
            global_proposals = particle_centers[:, :global_count].clone()
            coefficients = torch.randn(
                (2, global_count, self.config.deformation_modes, 3),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            global_noise = torch.einsum("cpkd,nk->cpnd", coefficients, self.basis)
            for cable_index in range(2):
                if route_counts_np[cable_index] > 0:
                    proposals[cable_index, :global_count] = (
                        global_proposals[cable_index]
                        + global_noise[cable_index]
                        * self.config.global_noise_m
                    )
                    predicted_velocities[cable_index, :global_count] = 0.0
            if (
                self.features.visible_edge_exploration
                and attribution_curves is not None
            ):
                disconnected_edge_measurement = (
                    edge_measurement & ~route_available
                )
                edge_center = (
                    attribution_curves + edge_displacement
                )[:, None, :, :]
                edge_global_proposals = (
                    edge_center
                    + global_noise[:, :global_count]
                    * self.config.global_noise_m
                )
                proposals[:, :global_count] = torch.where(
                    disconnected_edge_measurement[:, None, None, None],
                    edge_global_proposals,
                    proposals[:, :global_count],
                )

        if self.features.endpoint_motion_transport:
            endpoint_transport_mask = torch.as_tensor(
                effective_endpoint_visible,
                device=self.device,
                dtype=self.dtype,
            )
            start_residual = (
                endpoints[:, None, 0, :]
                - proposals[:, :, 0, :]
            ) * endpoint_transport_mask[:, None, 0, None]
            end_residual = (
                endpoints[:, None, 1, :]
                - proposals[:, :, -1, :]
            ) * endpoint_transport_mask[:, None, 1, None]
            # This is the minimum affine correction satisfying the measured
            # boundary displacement. A single visible endpoint naturally
            # fades to zero influence at the opposite end.
            proposals = (
                proposals
                + (1.0 - self._node_fraction) * start_residual[:, :, None, :]
                + self._node_fraction * end_residual[:, :, None, :]
            )

        measured_endpoint_velocity_tensor = torch.as_tensor(
            measured_endpoint_velocities, device=self.device, dtype=self.dtype
        )
        for cable_index in range(2):
            for endpoint_index, node_index in ((0, 0), (1, -1)):
                if not effective_endpoint_visible[cable_index, endpoint_index]:
                    continue
                proposals[cable_index, :, node_index, :] = endpoints[
                    cable_index, endpoint_index
                ]
                predicted_velocities[cable_index, :, node_index, :] = (
                    measured_endpoint_velocity_tensor[cable_index, endpoint_index]
                )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["proposal"].record()
        endpoint_anchor_pattern = tuple(
            tuple(bool(value) for value in cable)
            for cable in effective_endpoint_visible
        )
        if self.features.fixed_length:
            proposals = self._constrain_links(
                proposals,
                endpoints,
                endpoint_anchor_pattern,
                "measurement",
            )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["measurement_constraint"].record()

        # Update motion only where a currently attributed graph edge directly
        # supports the node.  The displacement is measured relative to the
        # damped constant-velocity prior, so displacement / dt is the local
        # velocity innovation.  Hidden nodes retain their own predicted
        # velocity; motion is never copied across an unobserved cable interval.
        velocity_prediction_before_correction = predicted_velocities
        local_innovation = torch.zeros_like(edge_displacement)
        local_velocity_update = (
            self.features.local_node_motion and attribution_batch is not None
        )
        if local_velocity_update:
            local_innovation = edge_displacement / dt
            corrected_velocity = (
                predicted_velocities
                + self.config.local_velocity_gain
                * local_innovation[:, None, :, :]
            )
            update_mask = (
                edge_measurement[:, None, None, None]
                & locally_supported_nodes[:, None, :, None]
                & self._interior_node_mask
            )
            predicted_velocities = torch.where(
                update_mask,
                corrected_velocity,
                predicted_velocities,
            )
        predicted_velocities = _clamp_vector_norm(
            predicted_velocities,
            self.config.maximum_node_velocity_mps,
        )

        # Enforce inextensibility once after local motion correction. This
        # removes only adjacent radial relative velocity; it does not impose a
        # shared translation or acceleration on the cable.
        if self.features.fixed_length:
            predicted_velocities = self._project_velocities(
                predicted_velocities,
                proposals,
                endpoint_anchor_pattern,
                "measurement",
            )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["local_motion"].record()

        dense = _dense_samples(
            proposals,
            self.config.dense_samples_per_segment,
            self._dense_alpha,
        )
        has_route = route_counts > 0

        if self.features.connected_trace:
            route_residual = torch.linalg.vector_norm(
                dense[:, :, None, :, :] - route_dense[:, None, :, :, :],
                dim=-1,
            )
            route_normalized = route_residual / self.config.trace_sigma_m
            delta = self.config.huber_delta
            route_huber = torch.where(
                route_normalized <= delta,
                0.5 * route_normalized.square(),
                delta * (route_normalized - 0.5 * delta),
            )
            route_energy = route_huber.mean(dim=-1)
            route_energy = torch.where(
                route_valid[:, None, :],
                route_energy,
                torch.full_like(route_energy, torch.inf),
            )
            best_route_indices = torch.argmin(route_energy, dim=-1)
            selected_route_index = best_route_indices[:, :, None, None].expand(
                -1, -1, 1, dense.shape[2]
            )
            residual = torch.gather(
                route_residual,
                2,
                selected_route_index,
            ).squeeze(2)
            best_route_energy = torch.gather(
                route_energy,
                2,
                best_route_indices[:, :, None],
            ).squeeze(2)
        else:
            route_residual = _unordered_graph_residual(dense, route_dense, route_valid)
            normalized = route_residual / self.config.trace_sigma_m
            delta = self.config.huber_delta
            huber = torch.where(
                normalized <= delta,
                0.5 * normalized.square(),
                delta * (normalized - 0.5 * delta),
            )
            first_valid_route = torch.argmax(route_valid.to(torch.int64), dim=-1)
            best_route_indices = first_valid_route[:, None].expand(
                -1, self.config.particle_count
            )
            best_route_energy = huber.mean(dim=-1)
            residual = route_residual
        best_route_indices = torch.where(
            has_route[:, None],
            best_route_indices,
            torch.full_like(best_route_indices, -1),
        )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["route_score"].record()

        visible_edge_energy = torch.zeros_like(best_route_energy)
        if (
            self.features.visible_edge_scoring
            and attribution_batch is not None
            and normalized_edge_arc is not None
            and selected_edges is not None
        ):
            visible_edge_energy = self._visible_edge_energy(
                proposals,
                attribution_batch,
                normalized_edge_arc,
                selected_edges,
            )
        energy = torch.where(
            edge_measurement[:, None],
            visible_edge_energy,
            torch.where(
                route_measurement[:, None],
                best_route_energy,
                torch.zeros_like(best_route_energy),
            ),
        )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["visible_edge_score"].record()

        route_supported = residual <= self.config.support_distance_m
        bad_samples = torch.zeros_like(route_supported)
        bad_samples = torch.where(
            has_route[:, None, None],
            ~route_supported,
            bad_samples,
        )
        supported_for_gap = ~bad_samples
        sample_indices = self._dense_sample_indices
        last_supported = torch.where(
            supported_for_gap,
            sample_indices.view(1, 1, -1),
            torch.full_like(sample_indices, -1).view(1, 1, -1),
        )
        last_supported = torch.cummax(last_supported, dim=-1).values
        bad_run_samples = torch.where(
            supported_for_gap,
            torch.zeros_like(last_supported),
            sample_indices.view(1, 1, -1) - last_supported,
        )
        longest_bad_samples = bad_run_samples.amax(dim=-1).to(self.dtype)
        sample_spacing = self.lengths / float(max(1, dense.shape[2] - 1))
        longest_unsupported = longest_bad_samples * sample_spacing[:, None]
        gap_fraction = longest_unsupported / max(
            self.config.maximum_unsupported_length_m,
            1e-6,
        )
        energy = energy + torch.where(
            route_measurement[:, None],
            self.config.unsupported_weight * gap_fraction.square(),
            torch.zeros_like(gap_fraction),
        )

        links = proposals[:, :, 1:, :] - proposals[:, :, :-1, :]
        link_lengths = torch.linalg.vector_norm(links, dim=-1)
        unit = links / link_lengths[..., None].clamp_min(1e-8)
        cosine = torch.sum(
            unit[:, :, :-1, :] * unit[:, :, 1:, :], dim=-1
        ).clamp(-1.0, 1.0)
        angles = torch.acos(cosine)
        if self.features.smoothness:
            energy = energy + self.config.smoothness_weight * torch.mean(
                angles.square(), dim=-1
            )
        link_error = torch.abs(link_lengths - self.segment_lengths[:, None, None]).amax(dim=-1)
        endpoint_distance = torch.stack(
            (
                torch.linalg.vector_norm(
                    proposals[:, :, 0, :] - endpoints[:, None, 0, :], dim=-1
                ),
                torch.linalg.vector_norm(
                    proposals[:, :, -1, :] - endpoints[:, None, 1, :], dim=-1
                ),
            ),
            dim=-1,
        )
        endpoint_mask = torch.as_tensor(
            effective_endpoint_visible, device=self.device, dtype=torch.bool
        )[:, None, :]
        endpoint_error = torch.where(
            endpoint_mask, endpoint_distance, torch.zeros_like(endpoint_distance)
        ).amax(dim=-1)
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["regularization"].record()

        trackable = initialized | initializable
        particle_valid = torch.isfinite(energy) & trackable[:, None]
        if self.features.fixed_length:
            particle_valid &= endpoint_error <= self.config.endpoint_tolerance_m
            particle_valid &= link_error <= self.config.endpoint_tolerance_m
        particle_valid &= torch.where(
            route_measurement[:, None],
            longest_unsupported <= self.config.maximum_unsupported_length_m,
            torch.ones_like(particle_valid),
        )

        measurement_available = measurement_requested

        particle_any_valid = torch.any(particle_valid, dim=1)
        eligible_np = self.initialized | initializable_np
        eligible = torch.as_tensor(
            eligible_np, device=self.device, dtype=torch.bool
        )
        accepted = eligible & measurement_available & particle_any_valid
        log_prior = torch.where(
            resample[:, None],
            torch.zeros_like(self.weights),
            torch.log(self.weights.clamp_min(1e-12)),
        )
        logits = torch.where(
            particle_valid,
            log_prior - energy,
            torch.full_like(energy, -torch.inf),
        )
        candidate_weights = torch.softmax(
            torch.where(
                accepted[:, None], logits, torch.zeros_like(logits)
            ),
            dim=1,
        )
        prediction_commit = (
            initialized
            & ~accepted
            & bool(self.features.prediction_without_measurement)
        )
        commit = accepted | prediction_commit
        committed_particles = torch.where(
            accepted[:, None, None, None], proposals, prediction_only_proposals
        )
        committed_velocities = torch.where(
            accepted[:, None, None, None],
            predicted_velocities,
            prediction_only_velocities,
        )
        self.particles = torch.where(
            commit[:, None, None, None], committed_particles, self.particles
        )
        self.velocities = torch.where(
            commit[:, None, None, None], committed_velocities, self.velocities
        )
        self.weights = torch.where(accepted[:, None], candidate_weights, self.weights)
        self.route_indices = torch.where(
            (accepted & route_measurement)[:, None],
            best_route_indices,
            self.route_indices,
        )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["weights"].record()

        route_masks: list[torch.Tensor] = []
        candidate_routes: list[torch.Tensor] = []
        selected_routes: list[torch.Tensor] = []
        conditional_weight_sets: list[torch.Tensor] = []
        posterior_means: list[torch.Tensor] = []
        for cable_index in range(2):
            measurement_accepted = accepted[cable_index]
            route_measurement_accepted = (
                measurement_accepted & route_measurement[cable_index]
            )
            selected_route = self._negative_one_index
            candidate_route = self._negative_one_index
            route_mask = self._all_particles_mask
            if route_counts_np[cable_index] > 0:
                route_count = int(route_counts_np[cable_index])
                route_mass = torch.zeros(
                    route_count,
                    device=self.device,
                    dtype=self.dtype,
                )
                particle_route_indices = self.route_indices[cable_index]
                valid_route_assignment = (
                    (particle_route_indices >= 0)
                    & (particle_route_indices < route_count)
                )
                safe_route_indices = particle_route_indices.clamp(
                    min=0, max=route_count - 1
                )
                route_mass.scatter_add_(
                    0,
                    safe_route_indices,
                    self.weights[cable_index]
                    * valid_route_assignment.to(self.dtype),
                )
                candidate_route = torch.argmax(route_mass)
                selected_route = torch.where(
                    route_measurement_accepted,
                    candidate_route,
                    self._negative_one_index,
                )
                route_mask = torch.where(
                    route_measurement_accepted,
                    valid_route_assignment
                    & (particle_route_indices == candidate_route),
                    self._all_particles_mask,
                )
            conditional_weights = self.weights[cable_index] * route_mask.to(
                self.dtype
            )
            conditional_weights = (
                conditional_weights
                / conditional_weights.sum().clamp_min(1e-12)
            )
            route_masks.append(route_mask)
            candidate_routes.append(candidate_route)
            selected_routes.append(selected_route)
            conditional_weight_sets.append(conditional_weights)
            posterior_means.append(
                torch.sum(
                    conditional_weights[:, None, None]
                    * self.particles[cable_index],
                    dim=0,
                )
            )

        posterior_mean_particles = torch.stack(posterior_means, dim=0)[:, None]
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["posterior"].record()
        if self.features.fixed_length:
            endpoint_anchor_pattern = tuple(
                tuple(bool(value) for value in cable)
                for cable in effective_endpoint_visible
            )
            selected_particles = self._constrain_links(
                posterior_mean_particles,
                endpoints,
                endpoint_anchor_pattern,
                "estimate",
            )[:, 0]
        else:
            selected_particles = posterior_mean_particles[:, 0]
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["estimate_constraint"].record()

        selected_dense_batch = _dense_samples(
            selected_particles[:, None, :, :],
            self.config.dense_samples_per_segment,
            self._dense_alpha,
        )[:, 0]
        top_count = min(self.config.top_particle_count, self.config.particle_count)
        visibility_required = bool(
            diagnostics_required or require_contact_support
        )
        selected_visibility_batch = self._unknown_visibility[None, :].expand(
            2, -1
        )
        edge_visibility_batch = selected_visibility_batch
        edge_trace_batch = self._nan_scalar.expand(2, 4)
        route_residuals: list[torch.Tensor | None] = [None, None]
        if visibility_required:
            if (
                attribution_batch is not None
                and normalized_edge_arc is not None
                and selected_edges is not None
            ):
                (
                    edge_visibility_batch,
                    flat_edge_residual,
                    flat_edge_valid,
                ) = self._visible_edge_support(
                    selected_particles,
                    attribution_batch,
                    normalized_edge_arc,
                    selected_edges,
                    selected_dense_batch.shape[1],
                )
                if diagnostics_required:
                    edge_trace_batch = self._visible_edge_trace(
                        flat_edge_residual,
                        flat_edge_valid,
                    )

            current_visibility: list[torch.Tensor] = []
            for cable_index in range(2):
                cable_visibility = self._unknown_visibility
                if route_counts_np[cable_index] > 0:
                    candidate_route = candidate_routes[cable_index]
                    if self.features.connected_trace:
                        selected_residual = torch.linalg.vector_norm(
                            selected_dense_batch[cable_index]
                            - route_dense[cable_index, candidate_route],
                            dim=-1,
                        )
                    else:
                        graph_points = route_dense[
                            cable_index, route_valid[cable_index]
                        ].reshape(-1, 3)
                        selected_residual = torch.cdist(
                            selected_dense_batch[cable_index][None, :, :],
                            graph_points[None, :, :],
                        )[0].amin(dim=-1)
                    route_residuals[cable_index] = selected_residual
                    route_visibility = torch.where(
                        selected_residual <= self.config.support_distance_m,
                        torch.full_like(
                            cable_visibility,
                            VISIBILITY_SUPPORTED,
                        ),
                        torch.full_like(
                            cable_visibility,
                            VISIBILITY_MISSING,
                        ),
                    )
                    cable_visibility = torch.where(
                        accepted[cable_index]
                        & route_measurement[cable_index],
                        route_visibility,
                        cable_visibility,
                    )
                cable_visibility = torch.where(
                    accepted[cable_index] & edge_measurement[cable_index],
                    edge_visibility_batch[cable_index],
                    cable_visibility,
                )
                current_visibility.append(cable_visibility)
            selected_visibility_batch = torch.stack(
                current_visibility,
                dim=0,
            )
            if require_contact_support:
                self._contact_dense_support = (
                    selected_visibility_batch == VISIBILITY_SUPPORTED
                )

        if diagnostics_required:
            route_mask_batch = torch.stack(route_masks, dim=0)
            conditional_weights_batch = torch.stack(
                conditional_weight_sets, dim=0
            )
            representative_distance = torch.mean(
                torch.sum(
                    (
                        self.particles
                        - selected_particles[:, None, :, :]
                    ).square(),
                    dim=-1,
                ),
                dim=-1,
            )
            representative_distance = torch.where(
                route_mask_batch,
                representative_distance,
                torch.full_like(representative_distance, torch.inf),
            )
            selected_indices = torch.argmin(representative_distance, dim=1)
            selected_links = selected_particles[:, 1:] - selected_particles[:, :-1]
            selected_link_lengths = torch.linalg.vector_norm(selected_links, dim=-1)
            selected_unit = selected_links / selected_link_lengths[..., None].clamp_min(
                1e-8
            )
            selected_cosine = torch.sum(
                selected_unit[:, :-1] * selected_unit[:, 1:], dim=-1
            ).clamp(-1.0, 1.0)
            selected_turn_rms_batch = torch.sqrt(
                torch.mean(torch.acos(selected_cosine).square(), dim=1)
            )
            maximum_link_error_batch = torch.abs(
                selected_link_lengths - self.segment_lengths[:, None]
            ).amax(dim=1)
            selected_endpoint_distance = torch.stack(
                (
                    torch.linalg.vector_norm(
                        selected_particles[:, 0] - endpoints[:, 0], dim=-1
                    ),
                    torch.linalg.vector_norm(
                        selected_particles[:, -1] - endpoints[:, 1], dim=-1
                    ),
                ),
                dim=1,
            )
            selected_endpoint_mask = torch.as_tensor(
                effective_endpoint_visible, device=self.device, dtype=torch.bool
            )
            endpoint_error_batch = torch.where(
                selected_endpoint_mask,
                selected_endpoint_distance,
                torch.zeros_like(selected_endpoint_distance),
            ).amax(dim=1)
            endpoint_error_batch = torch.where(
                selected_endpoint_mask.any(dim=1),
                endpoint_error_batch,
                self._nan_scalar,
            )

            if self.features.posterior_uncertainty:
                centered = self.particles - selected_particles[:, None]
                node_covariance_batch = torch.einsum(
                    "cp,cpni,cpnj->cnij",
                    conditional_weights_batch,
                    centered,
                    centered,
                )
                maximum_node_uncertainty_batch = torch.sqrt(
                    torch.linalg.eigvalsh(node_covariance_batch)
                    .clamp_min(0.0)
                    .amax(dim=(1, 2))
                )
            else:
                node_covariance_batch = torch.zeros(
                    (2, self.config.node_count, 3, 3),
                    device=self.device,
                    dtype=self.dtype,
                )
                maximum_node_uncertainty_batch = self._nan_scalar.expand(2)
            top_indices = torch.topk(self.weights, top_count, dim=1).indices
            top_particles_batch = torch.gather(
                self.particles,
                1,
                top_indices[:, :, None, None].expand(
                    -1, -1, self.config.node_count, 3
                ),
            )
            effective_sample_size_batch = 1.0 / torch.sum(
                self.weights.square(), dim=1
            )

        pending_outputs: list[_PendingCableOutput] = []
        for cable_index in range(2):
            measurement_accepted = accepted[cable_index]
            current_initialized = initialized[cable_index] | measurement_accepted
            selected_particle = selected_particles[cable_index]
            selected_dense = selected_dense_batch[cable_index]
            endpoint_visible_count = int(endpoint_visible_np[cable_index].sum())
            estimate_payload = torch.cat(
                (
                    selected_particle.reshape(-1),
                    selected_dense.reshape(-1),
                )
            )
            state_payload = torch.stack(
                (
                    measurement_accepted.to(self.dtype),
                    current_initialized.to(self.dtype),
                    measurement_available[cable_index].to(self.dtype),
                    particle_any_valid[cable_index].to(self.dtype),
                    edge_measurement[cable_index].to(self.dtype),
                    route_measurement[cable_index].to(self.dtype),
                    attributed_edge_counts[cable_index].to(self.dtype),
                )
            )
            diagnostic_float_payload = None
            diagnostic_visibility_payload = None
            if diagnostics_required:
                selected_route = selected_routes[cable_index]
                candidate_route = candidate_routes[cable_index]
                selected_index = selected_indices[cable_index]
                selected_visibility = selected_visibility_batch[cable_index]
                trace_values = self._nan_scalar.expand(4)
                if route_counts_np[cable_index] > 0:
                    selected_residual = route_residuals[cable_index]
                    assert selected_residual is not None
                    measured_trace_values = torch.stack(
                        (
                            selected_residual.mean(),
                            torch.sqrt(torch.mean(selected_residual.square())),
                            torch.quantile(selected_residual, 0.95),
                            selected_residual.max(),
                        )
                    )
                    trace_values = torch.where(
                        (
                            measurement_accepted
                            & route_measurement[cable_index]
                        ),
                        measured_trace_values,
                        self._nan_scalar.expand(4),
                    )
                trace_values = torch.where(
                    (
                        measurement_accepted
                        & edge_measurement[cable_index]
                    ),
                    edge_trace_batch[cable_index],
                    trace_values,
                )
                selected_supported = selected_visibility == VISIBILITY_SUPPORTED
                selected_missing = selected_visibility == VISIBILITY_MISSING
                selected_unknown = selected_visibility == VISIBILITY_UNKNOWN
                visibility_fractions = torch.stack(
                    (
                        selected_supported.float().mean(),
                        selected_missing.float().mean(),
                        selected_unknown.float().mean(),
                    )
                )
                sample_order = self._dense_sample_indices
                last_nonmissing = torch.where(
                    ~selected_missing,
                    sample_order,
                    torch.full_like(sample_order, -1),
                )
                last_nonmissing = torch.cummax(last_nonmissing, dim=0).values
                missing_runs = torch.where(
                    selected_missing,
                    sample_order - last_nonmissing,
                    torch.zeros_like(sample_order),
                )
                longest_missing_run = missing_runs.max().to(self.dtype)
                node_visibility = selected_visibility[self._node_dense_indices]
                observed_route_length = (
                    torch.where(
                        (
                            measurement_accepted
                            & route_measurement[cable_index]
                        ),
                        route_lengths[cable_index, candidate_route],
                        self._nan_scalar,
                    )
                    if route_counts_np[cable_index] > 0
                    else self._nan_scalar
                )
                predicted_node_speeds = torch.linalg.vector_norm(
                    velocity_prediction_before_correction[
                        cable_index, selected_index
                    ],
                    dim=-1,
                )
                corrected_node_speeds = torch.linalg.vector_norm(
                    predicted_velocities[cable_index, selected_index],
                    dim=-1,
                )
                supported_node_mask = (
                    locally_supported_nodes[cable_index]
                    & self._interior_node_mask[0, 0, :, 0]
                )
                supported_node_count = supported_node_mask.sum()
                local_node_motion_applied = (
                    measurement_accepted
                    & edge_measurement[cable_index]
                    & (supported_node_count > 0)
                    & bool(self.features.local_node_motion)
                )
                local_velocity_innovation_speed = torch.where(
                    local_node_motion_applied,
                    torch.sum(
                        torch.linalg.vector_norm(
                            local_innovation[cable_index],
                            dim=-1,
                        )
                        * supported_node_mask.to(self.dtype)
                    )
                    / supported_node_count.clamp_min(1).to(self.dtype),
                    self._nan_scalar,
                )
                locally_supported_fraction = (
                    supported_node_count.to(self.dtype)
                    / float(self.config.node_count)
                )
                scalar_values = torch.stack(
                    (
                        self.weights[cable_index, selected_index],
                        effective_sample_size_batch[cable_index],
                        resample[cable_index].to(self.dtype),
                        trace_values[0],
                        trace_values[1],
                        trace_values[2],
                        trace_values[3],
                        visibility_fractions[0],
                        visibility_fractions[1],
                        visibility_fractions[2],
                        longest_missing_run,
                        maximum_link_error_batch[cable_index],
                        endpoint_error_batch[cable_index],
                        observed_route_length,
                        selected_turn_rms_batch[cable_index],
                        maximum_node_uncertainty_batch[cable_index],
                        selected_route.to(self.dtype),
                        predicted_node_speeds.mean(),
                        local_velocity_innovation_speed,
                        corrected_node_speeds.mean(),
                        corrected_node_speeds.max(),
                        local_node_motion_applied.to(self.dtype),
                        locally_supported_fraction,
                    )
                )
                diagnostic_float_payload = torch.cat(
                    (
                        node_covariance_batch[cable_index].reshape(-1),
                        top_particles_batch[cable_index].reshape(-1),
                        scalar_values,
                    )
                )
                diagnostic_visibility_payload = torch.cat(
                    (selected_visibility, node_visibility)
                )
            pending_outputs.append(
                _PendingCableOutput(
                    estimate_payload=estimate_payload,
                    state_payload=state_payload,
                    diagnostic_float_payload=diagnostic_float_payload,
                    diagnostic_visibility_payload=diagnostic_visibility_payload,
                    endpoint_visible_count=endpoint_visible_count,
                    route_candidate_count=int(route_counts_np[cable_index]),
                    dense_count=len(selected_dense),
                    top_count=top_count,
                )
            )

        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["diagnostics"].record()

        if gpu_end is not None:
            gpu_end.record()
            gpu_end.synchronize()
            gpu_ms = float(gpu_start.elapsed_time(gpu_end))
        else:
            gpu_ms = 0.0

        stage_gpu_ms: dict[str, float] = {}
        if self.config.profile_stages and self.device.type == "cuda":
            events = self._profile_events

            def elapsed(first: str, second: str) -> float:
                return float(events[first].elapsed_time(events[second]))

            stage_gpu_ms = {
                "inputs": elapsed("start", "inputs"),
                "attribution": elapsed("inputs", "attribution"),
                "prediction": elapsed("attribution", "prediction"),
                "prediction_constraint": elapsed(
                    "prediction", "prediction_constraint"
                ),
                "proposal": elapsed("prediction_constraint", "proposal"),
                "measurement_constraint": elapsed(
                    "proposal", "measurement_constraint"
                ),
                "local_motion": elapsed(
                    "measurement_constraint", "local_motion"
                ),
                "route_score": elapsed(
                    "local_motion", "route_score"
                ),
                "visible_edge_score": elapsed(
                    "route_score", "visible_edge_score"
                ),
                "regularization": elapsed(
                    "visible_edge_score", "regularization"
                ),
                "weights": elapsed("regularization", "weights"),
                "posterior": elapsed("weights", "posterior"),
                "estimate_constraint": elapsed(
                    "posterior", "estimate_constraint"
                ),
                "diagnostics": elapsed(
                    "estimate_constraint", "diagnostics"
                ),
            }

        readback_started = time.perf_counter()

        outputs = []
        frame_initialized: list[bool] = []
        frame_measurement_used: list[bool] = []
        for cable_index, pending in enumerate(pending_outputs):
            estimate_payload = pending.estimate_payload.cpu().numpy()
            state_payload = pending.state_payload.cpu().numpy()
            offset = 0
            curve_size = self.config.node_count * 3
            curve = np.ascontiguousarray(
                estimate_payload[offset : offset + curve_size].reshape(
                    self.config.node_count, 3
                ),
                dtype=np.float32,
            )
            offset += curve_size
            dense_size = pending.dense_count * 3
            dense_curve = np.ascontiguousarray(
                estimate_payload[offset : offset + dense_size].reshape(
                    pending.dense_count, 3
                ),
                dtype=np.float32,
            )
            (
                measurement_accepted_value,
                current_initialized_value,
                measurement_available_value,
                particle_any_valid_value,
                edge_measurement_value,
                route_measurement_value,
                attributed_edge_count_value,
            ) = (float(value) for value in state_payload)
            measurement_accepted = bool(measurement_accepted_value)
            current_initialized = bool(current_initialized_value)
            frame_initialized.append(current_initialized)
            frame_measurement_used.append(measurement_accepted)
            measurement_available_value = bool(measurement_available_value)
            particle_any_valid_value = bool(particle_any_valid_value)
            edge_measurement_value = bool(edge_measurement_value)
            route_measurement_value = bool(route_measurement_value)
            attributed_edge_count = int(round(attributed_edge_count_value))
            if not eligible_np[cable_index]:
                status = "LOST: complete route required for initialization"
            elif not measurement_available_value:
                status = (
                    "no confidently attributed visible edge"
                    if self.features.visible_edge_scoring
                    else "no complete-route measurement"
                )
            elif not particle_any_valid_value:
                status = "all particles rejected by geometric/support constraints"
            else:
                status = "measurement accepted"
            measurement_source = (
                "visible edges"
                if edge_measurement_value
                else ("complete route" if route_measurement_value else "none")
            )

            self.initialized[cable_index] = current_initialized
            if measurement_accepted and complete_evidence_np[cable_index]:
                self.last_complete_timestamps[cable_index] = float(timestamp)
            for endpoint_index in range(2):
                if not endpoint_visible_np[cable_index, endpoint_index]:
                    continue
                self.last_endpoints[cable_index, endpoint_index] = endpoints_np[
                    cable_index, endpoint_index
                ]
                self.last_endpoint_timestamps[cable_index, endpoint_index] = float(
                    timestamp
                )
                self.endpoint_velocities[cable_index, endpoint_index] = (
                    measured_endpoint_velocities[cable_index, endpoint_index]
                )
            if not current_initialized:
                output = _empty_output(status, measurement_available_value)
                outputs.append(output)
                if diagnostics_required:
                    self._diagnostic_cache[cable_index] = _CachedCableDiagnostics(
                        dense_visibility=output.dense_visibility,
                        node_visibility=output.node_visibility,
                        node_covariance=output.node_covariance,
                        top_particles=output.top_particles,
                        diagnostics=output.diagnostics,
                    )
                continue

            if not diagnostics_required:
                cached = self._diagnostic_cache[cable_index]
                if cached is None:
                    raise RuntimeError("PF diagnostic cache was not initialized")
                outputs.append(
                    CableFilterOutput(
                        curve=curve,
                        dense_curve=dense_curve,
                        dense_visibility=cached.dense_visibility,
                        node_visibility=cached.node_visibility,
                        node_covariance=cached.node_covariance,
                        top_particles=cached.top_particles,
                        diagnostics=cached.diagnostics,
                    )
                )
                continue

            if (
                pending.diagnostic_float_payload is None
                or pending.diagnostic_visibility_payload is None
            ):
                raise RuntimeError("Fresh PF diagnostics are missing their payload")
            diagnostic_payload = pending.diagnostic_float_payload.cpu().numpy()
            visibility_payload = pending.diagnostic_visibility_payload.cpu().numpy()
            diagnostic_offset = 0
            covariance_size = self.config.node_count * 9
            node_covariance = np.ascontiguousarray(
                diagnostic_payload[
                    diagnostic_offset : diagnostic_offset + covariance_size
                ].reshape(self.config.node_count, 3, 3),
                dtype=np.float32,
            )
            diagnostic_offset += covariance_size
            top_size = pending.top_count * self.config.node_count * 3
            top_particles = np.ascontiguousarray(
                diagnostic_payload[
                    diagnostic_offset : diagnostic_offset + top_size
                ].reshape(pending.top_count, self.config.node_count, 3),
                dtype=np.float32,
            )
            diagnostic_offset += top_size
            (
                selected_weight,
                effective_sample_size,
                resampled_value,
                trace_mean,
                trace_rms,
                trace_p95,
                trace_max,
                support_fraction,
                missing_fraction,
                unknown_fraction,
                longest_missing_run,
                maximum_link_error,
                endpoint_error,
                observed_route_length,
                turn_rms_radians,
                maximum_node_uncertainty,
                selected_route_value,
                predicted_node_speed_mps,
                local_velocity_innovation_speed_mps,
                corrected_node_speed_mps,
                maximum_corrected_node_speed_mps,
                local_node_motion_applied_value,
                locally_supported_node_fraction,
            ) = (
                float(value)
                for value in diagnostic_payload[diagnostic_offset:]
            )
            selected_route = int(selected_route_value)
            dense_visibility = np.ascontiguousarray(
                visibility_payload[: pending.dense_count], dtype=np.uint8
            )
            node_visibility = np.ascontiguousarray(
                visibility_payload[pending.dense_count :], dtype=np.uint8
            )
            longest_unsupported_mm = (
                longest_missing_run
                * self.config.cable_lengths_m[cable_index]
                / max(1, pending.dense_count - 1)
                * 1000.0
            )
            trace_mean_mm = trace_mean * 1000.0
            trace_rms_mm = trace_rms * 1000.0
            trace_p95_mm = trace_p95 * 1000.0
            trace_max_mm = trace_max * 1000.0
            maximum_link_error_mm = maximum_link_error * 1000.0
            endpoint_error_mm = endpoint_error * 1000.0
            turn_rms_degrees = turn_rms_radians * 180.0 / math.pi
            maximum_node_uncertainty_mm = maximum_node_uncertainty * 1000.0
            route_length_error_mm = (
                abs(
                    observed_route_length
                    - self.config.cable_lengths_m[cable_index]
                )
                * 1000.0
                if measurement_accepted and selected_route >= 0
                else float("nan")
            )
            if measurement_accepted and complete_evidence_np[cable_index]:
                tracking_state = "FULL"
            elif measurement_accepted and (
                support_fraction > 0.0 or pending.endpoint_visible_count > 0
            ):
                tracking_state = "PARTIAL"
            else:
                tracking_state = "LOST"
            last_complete = self.last_complete_timestamps[cable_index]
            complete_age = (
                max(0.0, float(timestamp - last_complete))
                if np.isfinite(last_complete)
                else float("inf")
            )
            diagnostics = CableFilterDiagnostics(
                initialized=True,
                measurement_valid=measurement_accepted,
                status=status,
                selected_weight=selected_weight,
                effective_sample_size=effective_sample_size,
                resampled=bool(resampled_value),
                trace_mean_mm=trace_mean_mm,
                trace_rms_mm=trace_rms_mm,
                trace_p95_mm=trace_p95_mm,
                trace_max_mm=trace_max_mm,
                support_fraction=support_fraction,
                longest_unsupported_mm=longest_unsupported_mm,
                maximum_link_error_mm=maximum_link_error_mm,
                endpoint_error_mm=endpoint_error_mm,
                observed_route_length_m=observed_route_length,
                route_candidate_count=pending.route_candidate_count,
                selected_route_index=selected_route,
                route_length_error_mm=route_length_error_mm,
                turn_rms_degrees=turn_rms_degrees,
                tracking_state=tracking_state,
                endpoint_visible_count=pending.endpoint_visible_count,
                visible_fraction=support_fraction,
                missing_fraction=missing_fraction,
                unknown_fraction=unknown_fraction,
                maximum_node_uncertainty_mm=maximum_node_uncertainty_mm,
                complete_observation_age_s=complete_age,
                predicted_node_speed_mps=predicted_node_speed_mps,
                local_velocity_innovation_speed_mps=(
                    local_velocity_innovation_speed_mps
                ),
                corrected_node_speed_mps=corrected_node_speed_mps,
                maximum_corrected_node_speed_mps=(
                    maximum_corrected_node_speed_mps
                ),
                local_node_motion_applied=bool(
                    local_node_motion_applied_value
                ),
                locally_supported_node_fraction=locally_supported_node_fraction,
                attributed_edge_count=attributed_edge_count,
                measurement_source=measurement_source,
            )
            output = CableFilterOutput(
                curve=curve,
                dense_curve=dense_curve,
                dense_visibility=dense_visibility,
                node_visibility=node_visibility,
                node_covariance=node_covariance,
                top_particles=top_particles,
                diagnostics=diagnostics,
            )
            outputs.append(output)
            self._diagnostic_cache[cable_index] = _CachedCableDiagnostics(
                dense_visibility=output.dense_visibility,
                node_visibility=output.node_visibility,
                node_covariance=output.node_covariance,
                top_particles=output.top_particles,
                diagnostics=output.diagnostics,
            )
        readback_cpu_ms = float(
            (time.perf_counter() - readback_started) * 1000.0
        )
        graph_attribution = None
        graph_attribution_started = time.perf_counter()
        if (
            self.features.graph_edge_attribution
            and attribution_batch is not None
        ):
            graph_attribution = self.graph_edge_attributor.diagnostics(
                attribution_batch
            )
        graph_attribution_wall_ms = float(
            (time.perf_counter() - graph_attribution_started) * 1000.0
        )
        profile = None
        if self.config.profile_stages:
            profile = ParticleFilterProfile(
                route_input_cpu_ms=float(
                    (route_input_finished - started) * 1000.0
                ),
                inputs_gpu_ms=stage_gpu_ms.get("inputs", 0.0),
                graph_attribution_gpu_ms=stage_gpu_ms.get(
                    "attribution", 0.0
                ),
                prediction_gpu_ms=stage_gpu_ms.get("prediction", 0.0),
                prediction_constraint_gpu_ms=stage_gpu_ms.get(
                    "prediction_constraint", 0.0
                ),
                proposal_gpu_ms=stage_gpu_ms.get("proposal", 0.0),
                measurement_constraint_gpu_ms=stage_gpu_ms.get(
                    "measurement_constraint", 0.0
                ),
                local_motion_gpu_ms=stage_gpu_ms.get(
                    "local_motion", 0.0
                ),
                route_score_gpu_ms=stage_gpu_ms.get("route_score", 0.0),
                visible_edge_score_gpu_ms=stage_gpu_ms.get(
                    "visible_edge_score", 0.0
                ),
                regularization_gpu_ms=stage_gpu_ms.get("regularization", 0.0),
                weight_update_gpu_ms=stage_gpu_ms.get("weights", 0.0),
                posterior_gpu_ms=stage_gpu_ms.get("posterior", 0.0),
                estimate_constraint_gpu_ms=stage_gpu_ms.get(
                    "estimate_constraint", 0.0
                ),
                diagnostics_gpu_ms=stage_gpu_ms.get("diagnostics", 0.0),
                readback_cpu_ms=readback_cpu_ms,
                graph_attribution_wall_ms=graph_attribution_wall_ms,
            )
        return ParticleFilterFrame(
            cables=(outputs[0], outputs[1]),
            cable_radius_m=0.5 * self.config.cable_diameter_m,
            initialized=(frame_initialized[0], frame_initialized[1]),
            measurement_used=(
                frame_measurement_used[0],
                frame_measurement_used[1],
            ),
            processing_ms=float((time.perf_counter() - started) * 1000.0),
            gpu_ms=gpu_ms,
            active_features=self.features.summary(),
            profile=profile,
            diagnostics_refreshed=diagnostics_required,
            graph_attribution=graph_attribution,
        )
