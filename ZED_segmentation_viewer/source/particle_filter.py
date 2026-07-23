"""Batched fixed-length particle hypotheses for two endpoint-identified cables."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch
import torch.nn.functional as torch_functional

from cuda_constraints import FusedCudaConstraints
from observation import CameraModel, FrameObservation


VISIBILITY_UNKNOWN = 0
VISIBILITY_SUPPORTED = 1
VISIBILITY_OCCLUDED = 2
VISIBILITY_MISSING = 3


@dataclass(frozen=True)
class ParticleFeatures:
    endpoint_direction_proposal: bool = True
    connected_trace: bool = True
    route_length_score: bool = True
    endpoint_tangent_score: bool = True
    smoothness: bool = True
    fixed_length: bool = True
    temporal_prediction: bool = True
    motion_adaptive_noise: bool = True
    global_particles: bool = True
    measurement_velocity_update: bool = True
    global_particle_velocity_update: bool = True
    kink_limit: bool = False
    partial_observation: bool = True
    depth_occlusion: bool = True
    fragment_score: bool = True
    single_endpoint_updates: bool = True
    prediction_without_measurement: bool = True
    posterior_uncertainty: bool = True
    fused_constraint_kernels: bool = True
    cuda_graph_replay: bool = True
    crossing: bool = False

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
    dense_samples_per_segment: int = 4
    deformation_modes: int = 5
    initial_noise_m: float = 0.010
    global_noise_m: float = 0.018
    global_fraction: float = 0.10
    acceleration_noise_mps2: float = 0.45
    motion_acceleration_gain: float = 1.50
    maximum_acceleration_noise_mps2: float = 3.00
    velocity_damping_per_second: float = 1.25
    endpoint_velocity_smoothing: float = 0.50
    measurement_velocity_gain: float = 0.65
    maximum_node_velocity_mps: float = 2.50
    projection_iterations: int = 8
    velocity_projection_iterations: int = 2
    endpoint_tolerance_m: float = 0.003
    trace_sigma_m: float = 0.008
    huber_delta: float = 1.5
    route_length_sigma_m: float = 0.025
    route_length_weight: float = 1.0
    endpoint_tangent_weight: float = 0.35
    smoothness_weight: float = 2.0
    support_distance_m: float = 0.015
    maximum_unsupported_length_m: float = 0.080
    unsupported_weight: float = 1.0
    maximum_kink_degrees: float = 145.0
    kink_weight: float = 0.20
    occlusion_depth_margin_m: float = 0.012
    visibility_depth_sigma_m: float = 0.012
    visibility_depth_tolerance_m: float = 0.020
    visibility_probability_threshold: float = 0.50
    image_likelihood_weight: float = 1.00
    depth_likelihood_weight: float = 0.35
    fragment_likelihood_weight: float = 1.00
    top_particle_count: int = 12
    random_seed: int = 7
    profile_stages: bool = True

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ParticleFilterConfig":
        values = values or {}
        lengths = tuple(float(item) for item in values.get("cable_lengths_m", (0.515, 0.515)))
        if len(lengths) != 2 or any(not np.isfinite(item) or item <= 0.0 for item in lengths):
            raise ValueError("particle_filter.cable_lengths_m must contain two positive lengths")
        return cls(
            device=str(values.get("device", "cuda")),
            particle_count=max(32, int(values.get("particle_count", 1024))),
            node_count=max(4, int(values.get("node_count", 13))),
            cable_lengths_m=(lengths[0], lengths[1]),
            dense_samples_per_segment=max(1, int(values.get("dense_samples_per_segment", 4))),
            deformation_modes=max(1, int(values.get("deformation_modes", 5))),
            initial_noise_m=max(0.0, float(values.get("initial_noise_m", 0.010))),
            global_noise_m=max(0.0, float(values.get("global_noise_m", 0.018))),
            global_fraction=float(np.clip(values.get("global_fraction", 0.10), 0.0, 1.0)),
            acceleration_noise_mps2=max(
                0.0, float(values.get("acceleration_noise_mps2", 0.45))
            ),
            motion_acceleration_gain=max(
                0.0, float(values.get("motion_acceleration_gain", 1.50))
            ),
            maximum_acceleration_noise_mps2=max(
                0.0, float(values.get("maximum_acceleration_noise_mps2", 3.00))
            ),
            velocity_damping_per_second=max(
                0.0, float(values.get("velocity_damping_per_second", 1.25))
            ),
            endpoint_velocity_smoothing=float(
                np.clip(values.get("endpoint_velocity_smoothing", 0.50), 0.0, 0.999)
            ),
            measurement_velocity_gain=float(
                np.clip(values.get("measurement_velocity_gain", 0.65), 0.0, 1.0)
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
            route_length_sigma_m=max(
                1e-6, float(values.get("route_length_sigma_m", 0.025))
            ),
            route_length_weight=max(
                0.0, float(values.get("route_length_weight", 1.0))
            ),
            endpoint_tangent_weight=max(
                0.0, float(values.get("endpoint_tangent_weight", 0.35))
            ),
            smoothness_weight=max(
                0.0, float(values.get("smoothness_weight", 2.0))
            ),
            support_distance_m=max(1e-6, float(values.get("support_distance_m", 0.015))),
            maximum_unsupported_length_m=max(
                0.0, float(values.get("maximum_unsupported_length_m", 0.080))
            ),
            unsupported_weight=max(0.0, float(values.get("unsupported_weight", 1.0))),
            maximum_kink_degrees=float(
                np.clip(values.get("maximum_kink_degrees", 145.0), 1.0, 179.9)
            ),
            kink_weight=max(0.0, float(values.get("kink_weight", 0.20))),
            occlusion_depth_margin_m=max(
                0.0, float(values.get("occlusion_depth_margin_m", 0.012))
            ),
            visibility_depth_sigma_m=max(
                1e-6, float(values.get("visibility_depth_sigma_m", 0.012))
            ),
            visibility_depth_tolerance_m=max(
                1e-6, float(values.get("visibility_depth_tolerance_m", 0.020))
            ),
            visibility_probability_threshold=float(
                np.clip(values.get("visibility_probability_threshold", 0.50), 0.01, 0.99)
            ),
            image_likelihood_weight=max(
                0.0, float(values.get("image_likelihood_weight", 1.00))
            ),
            depth_likelihood_weight=max(
                0.0, float(values.get("depth_likelihood_weight", 0.35))
            ),
            fragment_likelihood_weight=max(
                0.0, float(values.get("fragment_likelihood_weight", 1.00))
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
    fragment_count: int
    visible_fraction: float
    occluded_fraction: float
    missing_fraction: float
    unknown_fraction: float
    maximum_node_uncertainty_mm: float
    complete_observation_age_s: float
    predicted_node_speed_mps: float
    measured_displacement_speed_mps: float
    corrected_node_speed_mps: float
    maximum_corrected_node_speed_mps: float
    velocity_correction_applied: bool


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
    processing_ms: float
    gpu_ms: float
    active_features: str
    profile: "ParticleFilterProfile | None" = None
    diagnostics_refreshed: bool = True


@dataclass(frozen=True)
class ParticleFilterProfile:
    """One-frame PF stage timings without additional CUDA synchronization."""

    route_input_cpu_ms: float
    inputs_gpu_ms: float
    prediction_gpu_ms: float
    prediction_constraint_gpu_ms: float
    proposal_gpu_ms: float
    measurement_constraint_gpu_ms: float
    velocity_correction_gpu_ms: float
    route_score_gpu_ms: float
    partial_score_gpu_ms: float
    weight_update_gpu_ms: float
    posterior_gpu_ms: float
    estimate_constraint_gpu_ms: float
    diagnostics_gpu_ms: float
    readback_cpu_ms: float


@dataclass(frozen=True)
class _PendingCableOutput:
    """Current estimate plus an optional fresh diagnostic payload."""

    estimate_payload: torch.Tensor
    state_payload: torch.Tensor
    diagnostic_float_payload: torch.Tensor | None
    diagnostic_visibility_payload: torch.Tensor | None
    endpoint_visible_count: int
    fragment_count: int
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
        fragment_count=0,
        visible_fraction=0.0,
        occluded_fraction=0.0,
        missing_fraction=0.0,
        unknown_fraction=1.0,
        maximum_node_uncertainty_mm=float("nan"),
        complete_observation_age_s=float("inf"),
        predicted_node_speed_mps=float("nan"),
        measured_displacement_speed_mps=float("nan"),
        corrected_node_speed_mps=float("nan"),
        maximum_corrected_node_speed_mps=float("nan"),
        velocity_correction_applied=False,
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
        camera_model: CameraModel | None = None,
    ):
        self.config = config
        self.features = features
        self.camera_model = camera_model
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA particle filtering was requested but torch.cuda is unavailable")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
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
        self.lengths = torch.tensor(config.cable_lengths_m, device=self.device, dtype=self.dtype)
        self.segment_lengths = self.lengths / float(config.node_count - 1)
        positions = torch.linspace(0.0, 1.0, config.node_count, device=self.device)
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
                        "prediction",
                        "prediction_constraint",
                        "proposal",
                        "measurement_constraint",
                        "velocity_correction",
                        "route_score",
                        "partial_score",
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
        self.features = features
        self._diagnostic_cache = [None, None]

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
        np.ndarray,
        np.ndarray,
        np.ndarray,
        list[str],
    ]:
        endpoints = np.zeros((2, 2, 3), dtype=np.float32)
        tangents = np.zeros((2, 2, 3), dtype=np.float32)
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
        route_alignment = np.zeros((2, maximum_routes), dtype=np.float32)
        route_valid = np.zeros((2, maximum_routes), dtype=bool)
        route_counts = np.zeros(2, dtype=np.int64)
        initializable = np.zeros(2, dtype=bool)
        fragment_counts = np.zeros(2, dtype=np.int64)
        reasons = []
        for cable_index, cable in enumerate(observation.cables):
            reason = cable.reason
            endpoint_visible[cable_index] = cable.endpoint_visible
            visible = endpoint_visible[cable_index]
            endpoints[cable_index, visible] = cable.endpoints_xyz[visible]
            tangents[cable_index, visible] = cable.endpoint_tangents[visible]
            fragment_counts[cable_index] = len(cable.fragments)
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
                    route_alignment[cable_index, route_index] = (
                        route.endpoint_alignment_error
                    )
                    route_valid[cable_index, route_index] = True
                route_counts[cable_index] = int(
                    np.count_nonzero(route_valid[cable_index])
                )
                initializable[cable_index] = route_counts[cable_index] > 0
            reasons.append(reason)
        return (
            initializable,
            endpoints,
            tangents,
            endpoint_visible,
            route_nodes,
            route_dense,
            route_lengths,
            route_alignment,
            route_valid,
            route_counts,
            fragment_counts,
            reasons,
        )

    def _visibility_likelihood(
        self,
        dense: torch.Tensor,
        observation: FrameObservation,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        energy = torch.zeros(
            dense.shape[:2], device=self.device, dtype=self.dtype
        )
        states = torch.full(
            dense.shape[:-1],
            VISIBILITY_UNKNOWN,
            device=self.device,
            dtype=torch.uint8,
        )
        if (
            self.camera_model is None
            or observation.cable_probability is None
            or observation.scene_depth is None
        ):
            return energy, states, False
        probability_tensor = observation.cable_probability
        depth_tensor = observation.scene_depth
        if (
            not isinstance(probability_tensor, torch.Tensor)
            or not isinstance(depth_tensor, torch.Tensor)
            or probability_tensor.ndim != 2
            or depth_tensor.shape != probability_tensor.shape
            or probability_tensor.numel() == 0
        ):
            return energy, states, False
        if probability_tensor.device != self.device or depth_tensor.device != self.device:
            raise ValueError(
                "Observation probability and depth must already reside on the PF device"
            )
        if probability_tensor.dtype != self.dtype or depth_tensor.dtype != self.dtype:
            raise ValueError("Observation probability and depth must be float32 tensors")
        height, width = probability_tensor.shape
        probability_map = probability_tensor.view(1, 1, height, width)
        depth_map = depth_tensor.view(1, 1, height, width)
        camera = self.camera_model
        predicted_depth = camera.forward_sign * dense[..., 2]
        safe_depth = predicted_depth.clamp_min(1e-6)
        pixel_x = camera.fx * dense[..., 0] / safe_depth + camera.cx
        pixel_y = camera.cy + camera.image_y_sign * camera.fy * dense[..., 1] / safe_depth
        if camera.width > 1 and width != camera.width:
            pixel_x = pixel_x * float(width - 1) / float(camera.width - 1)
        if camera.height > 1 and height != camera.height:
            pixel_y = pixel_y * float(height - 1) / float(camera.height - 1)
        in_view = (
            (predicted_depth > 1e-4)
            & (pixel_x >= 0.0)
            & (pixel_x <= float(width - 1))
            & (pixel_y >= 0.0)
            & (pixel_y <= float(height - 1))
        )
        grid_x = 2.0 * pixel_x / float(max(1, width - 1)) - 1.0
        grid_y = 2.0 * pixel_y / float(max(1, height - 1)) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            1, dense.shape[0] * dense.shape[1], dense.shape[2], 2
        )
        sampled_probability = torch_functional.grid_sample(
            probability_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(dense.shape[:-1])
        sampled_depth = torch_functional.grid_sample(
            depth_map,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(dense.shape[:-1])
        depth_valid = sampled_depth > 1e-4
        occluded = (
            in_view
            & depth_valid
            & (
                sampled_depth + self.config.occlusion_depth_margin_m
                < predicted_depth
            )
        )
        if not self.features.depth_occlusion:
            occluded = torch.zeros_like(occluded)
        measurable = in_view & depth_valid & ~occluded
        depth_error = torch.abs(sampled_depth - predicted_depth)
        supported = (
            measurable
            & (
                sampled_probability
                >= self.config.visibility_probability_threshold
            )
            & (depth_error <= self.config.visibility_depth_tolerance_m)
        )
        missing = measurable & ~supported
        states = torch.where(
            occluded,
            torch.full_like(states, VISIBILITY_OCCLUDED),
            states,
        )
        states = torch.where(
            supported,
            torch.full_like(states, VISIBILITY_SUPPORTED),
            states,
        )
        states = torch.where(
            missing,
            torch.full_like(states, VISIBILITY_MISSING),
            states,
        )
        probability_nll = -torch.log(sampled_probability.clamp_min(1e-4))
        depth_normalized = depth_error / self.config.visibility_depth_sigma_m
        delta = self.config.huber_delta
        depth_huber = torch.where(
            depth_normalized <= delta,
            0.5 * depth_normalized.square(),
            delta * (depth_normalized - 0.5 * delta),
        )
        sample_energy = (
            self.config.image_likelihood_weight * probability_nll
            + self.config.depth_likelihood_weight
            * sampled_probability.detach()
            * depth_huber
        )
        measurable_float = measurable.to(self.dtype)
        energy = torch.sum(sample_energy * measurable_float, dim=-1) / torch.sum(
            measurable_float, dim=-1
        ).clamp_min(1.0)
        return energy, states, True

    def _fragment_likelihood(
        self,
        dense: torch.Tensor,
        observation: FrameObservation,
        route_counts: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        energy = torch.zeros(
            dense.shape[:2], device=self.device, dtype=self.dtype
        )
        available = np.zeros(2, dtype=bool)
        for cable_index, cable in enumerate(observation.cables):
            if route_counts[cable_index] > 0 or not cable.fragments:
                continue
            weighted_energy = torch.zeros(
                dense.shape[1], device=self.device, dtype=self.dtype
            )
            total_length = 0.0
            for fragment in cable.fragments:
                span = int(
                    round(
                        fragment.length_m
                        / self.config.cable_lengths_m[cable_index]
                        * (dense.shape[2] - 1)
                    )
                ) + 1
                span = max(2, min(dense.shape[2], span))
                fragment_samples = _resample_curve(fragment.xyz, span)
                if len(fragment_samples) != span:
                    continue
                target = torch.as_tensor(
                    fragment_samples,
                    device=self.device,
                    dtype=self.dtype,
                )
                windows = dense[cable_index].unfold(1, span, 1).permute(0, 1, 3, 2)
                forward_distance = torch.linalg.vector_norm(
                    windows - target[None, None, :, :], dim=-1
                )
                reverse_distance = torch.linalg.vector_norm(
                    windows - target.flip(0)[None, None, :, :], dim=-1
                )
                delta = self.config.huber_delta
                forward_normalized = forward_distance / self.config.trace_sigma_m
                reverse_normalized = reverse_distance / self.config.trace_sigma_m
                forward_robust = torch.where(
                    forward_normalized <= delta,
                    0.5 * forward_normalized.square(),
                    delta * (forward_normalized - 0.5 * delta),
                ).mean(dim=-1)
                reverse_robust = torch.where(
                    reverse_normalized <= delta,
                    0.5 * reverse_normalized.square(),
                    delta * (reverse_normalized - 0.5 * delta),
                ).mean(dim=-1)
                robust = torch.minimum(forward_robust, reverse_robust).amin(dim=-1)
                weighted_energy += float(fragment.length_m) * robust
                total_length += float(fragment.length_m)
            if total_length > 0.0:
                energy[cable_index] = weighted_energy / total_length
                available[cable_index] = True
        return energy, available

    @torch.inference_mode()
    def update(
        self,
        observation: FrameObservation,
        timestamp: float,
        *,
        refresh_diagnostics: bool = True,
    ) -> ParticleFilterFrame:
        """Advance both filters and optionally refresh viewer-only diagnostics."""

        started = time.perf_counter()
        diagnostics_required = bool(
            refresh_diagnostics
            or any(cached is None for cached in self._diagnostic_cache)
        )
        self._allocate()
        assert self.particles is not None
        assert self.velocities is not None
        assert self.weights is not None
        assert self.route_indices is not None
        (
            initializable_np,
            endpoints_np,
            tangents_np,
            endpoint_visible_np,
            route_nodes_np,
            route_dense_np,
            route_lengths_np,
            route_alignment_np,
            route_valid_np,
            route_counts_np,
            fragment_counts_np,
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

        partial_map_available = bool(
            self.features.partial_observation
            and self.camera_model is not None
            and observation.cable_probability is not None
            and observation.scene_depth is not None
        )
        complete_evidence_np = route_counts_np > 0
        partial_identity_np = (
            effective_endpoint_visible.any(axis=1) | (fragment_counts_np > 0)
        )
        # Only cable-identified evidence triggers prior resampling. Generic cable
        # pixels may still update weights below, but cannot justify discarding
        # diversity for a fully hidden, indistinguishable cable.
        measurement_requested_np = complete_evidence_np | (
            self.features.partial_observation & partial_identity_np
        )

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
        tangents = torch.as_tensor(tangents_np, device=self.device, dtype=self.dtype)
        route_nodes = torch.as_tensor(route_nodes_np, device=self.device, dtype=self.dtype)
        route_dense = torch.as_tensor(route_dense_np, device=self.device, dtype=self.dtype)
        route_lengths = torch.as_tensor(
            route_lengths_np, device=self.device, dtype=self.dtype
        )
        route_alignment = torch.as_tensor(
            route_alignment_np, device=self.device, dtype=self.dtype
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

        sampled_parent_indices = torch.multinomial(
            self.weights,
            self.config.particle_count,
            replacement=True,
            generator=self.generator,
        )
        identity_parent_indices = self._particle_indices.expand(2, -1)
        # A complete route observes the cable's arc-length correspondence and
        # can safely replace the population. Partial evidence cannot constrain
        # the hidden section, so resampling it would turn unobserved process
        # noise into apparent cable motion.
        resample_np = self.initialized & complete_evidence_np
        resample = torch.as_tensor(resample_np, device=self.device, dtype=torch.bool)
        parent_indices = torch.where(
            resample[:, None], sampled_parent_indices, identity_parent_indices
        )
        gather_index = parent_indices[:, :, None, None].expand(
            -1, -1, self.config.node_count, 3
        )
        parent_particles = torch.gather(self.particles, 1, gather_index)
        parent_velocities = torch.gather(self.velocities, 1, gather_index)

        damping = math.exp(-self.config.velocity_damping_per_second * dt)
        if self.features.temporal_prediction:
            deterministic_velocity = damping * parent_velocities
        else:
            deterministic_velocity = torch.zeros_like(parent_velocities)
        endpoint_speed = np.zeros(2, dtype=np.float32)
        for cable_index in range(2):
            visible = endpoint_visible_np[cable_index]
            if np.any(visible):
                endpoint_speed[cable_index] = float(
                    np.mean(
                        np.linalg.norm(
                            measured_endpoint_velocities[cable_index, visible], axis=1
                        )
                    )
                )
        acceleration_sigma = np.full(
            2, self.config.acceleration_noise_mps2, dtype=np.float32
        )
        if self.features.motion_adaptive_noise:
            acceleration_sigma += (
                self.config.motion_acceleration_gain * endpoint_speed
            )
            acceleration_sigma = np.minimum(
                acceleration_sigma,
                self.config.maximum_acceleration_noise_mps2,
            )
        acceleration = self._smooth_random_field(
            torch.as_tensor(acceleration_sigma, device=self.device, dtype=self.dtype)
        )
        exploration_count = 0
        if self.features.global_particles and self.config.global_fraction > 0.0:
            exploration_count = max(
                1, int(round(self.config.particle_count * self.config.global_fraction))
            )
        incomplete_prediction_np = self.initialized & ~complete_evidence_np
        if np.any(incomplete_prediction_np):
            acceleration_enabled = torch.ones(
                (2, self.config.particle_count),
                device=self.device,
                dtype=self.dtype,
            )
            for cable_index in range(2):
                if not incomplete_prediction_np[cable_index]:
                    continue
                acceleration_enabled[cable_index] = 0.0
                if exploration_count > 0:
                    acceleration_enabled[cable_index, :exploration_count] = 1.0
            acceleration = acceleration * acceleration_enabled[:, :, None, None]
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

        proposal_centers = route_nodes.clone()
        if self.features.endpoint_direction_proposal:
            start_proposal = (
                endpoints[:, None, 0, :]
                + self.segment_lengths[:, None, None] * tangents[:, None, 0, :]
            )
            end_proposal = (
                endpoints[:, None, 1, :]
                + self.segment_lengths[:, None, None] * tangents[:, None, 1, :]
            )
            start_valid = torch.as_tensor(
                endpoint_visible_np[:, 0], device=self.device, dtype=torch.bool
            )[:, None, None]
            end_valid = torch.as_tensor(
                endpoint_visible_np[:, 1], device=self.device, dtype=torch.bool
            )[:, None, None]
            proposal_centers[:, :, 1, :] = torch.where(
                start_valid, start_proposal, proposal_centers[:, :, 1, :]
            )
            proposal_centers[:, :, -2, :] = torch.where(
                end_valid, end_proposal, proposal_centers[:, :, -2, :]
            )

        particle_route_slots = self._particle_indices % route_counts.clamp_min(1)[:, None]
        particle_center_index = particle_route_slots[:, :, None, None].expand(
            -1, -1, self.config.node_count, 3
        )
        particle_centers = torch.gather(
            proposal_centers,
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
                if route_counts_np[cable_index] <= 0:
                    continue
                proposals[cable_index, :global_count] = (
                    global_proposals[cable_index]
                    + global_noise[cable_index] * self.config.global_noise_m
                )
                predicted_velocities[cable_index, :global_count] = 0.0

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

        # A position observation must also update the velocity part of the
        # state. The previous implementation committed constrained positions
        # but retained their pre-correction velocities; route-centered retrack
        # particles were even assigned zero velocity. Arc-length node ordering
        # makes parent-to-proposal displacement a valid per-node measurement.
        velocity_prediction_before_correction = predicted_velocities
        velocity_update_particle_mask = torch.zeros(
            (2, self.config.particle_count),
            device=self.device,
            dtype=torch.bool,
        )
        velocity_update_cables_np = self.initialized & complete_evidence_np
        ordinary_velocity_update_np = (
            velocity_update_cables_np
            & bool(self.features.measurement_velocity_update)
        )
        global_velocity_update_np = (
            velocity_update_cables_np
            & bool(self.features.global_particle_velocity_update)
            & (global_count > 0)
        )
        velocity_update_requested = bool(
            np.any(ordinary_velocity_update_np)
            or np.any(global_velocity_update_np)
        )
        velocity_projection_complete = False
        if velocity_update_requested and np.any(velocity_update_cables_np):
            is_global_particle = self._particle_indices[0] < global_count
            if (
                self.features.measurement_velocity_update
                and self.features.global_particle_velocity_update
            ):
                enabled_particle_category = self._all_particles_mask
            elif self.features.measurement_velocity_update:
                enabled_particle_category = ~is_global_particle
            elif global_count > 0:
                enabled_particle_category = is_global_particle
            else:
                enabled_particle_category = torch.zeros_like(
                    self._all_particles_mask
                )
            velocity_update_cables = torch.as_tensor(
                velocity_update_cables_np,
                device=self.device,
                dtype=torch.bool,
            )
            velocity_update_particle_mask = (
                velocity_update_cables[:, None]
                & enabled_particle_category[None, :]
            )
            if (
                self.features.fixed_length
                and self.device.type == "cuda"
                and self.features.fused_constraint_kernels
            ):
                if self._fused_constraints is None:
                    raise RuntimeError("Fused CUDA constraints were not initialized")
                predicted_velocities = (
                    self._fused_constraints.correct_and_project_velocities(
                        predicted_velocities,
                        parent_particles,
                        proposals,
                        endpoint_anchor_pattern,
                        self.config.velocity_projection_iterations,
                        tuple(bool(value) for value in ordinary_velocity_update_np),
                        tuple(bool(value) for value in global_velocity_update_np),
                        global_count,
                        self.config.measurement_velocity_gain,
                        self.config.maximum_node_velocity_mps,
                        dt,
                    )
                )
                velocity_projection_complete = True
            else:
                measured_displacement_velocities = _clamp_vector_norm(
                    (proposals - parent_particles) / dt,
                    self.config.maximum_node_velocity_mps,
                )
                measured_blend = torch.lerp(
                    predicted_velocities,
                    measured_displacement_velocities,
                    self.config.measurement_velocity_gain,
                )
                velocity_update_node_mask = (
                    velocity_update_particle_mask[:, :, None, None]
                    & self._interior_node_mask
                )
                predicted_velocities = torch.where(
                    velocity_update_node_mask,
                    measured_blend,
                    predicted_velocities,
                )
                predicted_velocities = _clamp_vector_norm(
                    predicted_velocities,
                    self.config.maximum_node_velocity_mps,
                )

        # Reuse the existing projection exactly once, after measurement
        # correction, so the corrected velocity remains tangent to the fixed
        # link-length constraints.
        if self.features.fixed_length and not velocity_projection_complete:
            predicted_velocities = self._project_velocities(
                predicted_velocities,
                proposals,
                endpoint_anchor_pattern,
                "measurement",
            )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["velocity_correction"].record()

        dense = _dense_samples(
            proposals,
            self.config.dense_samples_per_segment,
            self._dense_alpha,
        )
        has_route = route_counts > 0
        route_prior = torch.zeros_like(route_lengths)
        if self.features.route_length_score:
            route_length_normalized = (
                route_lengths - self.lengths[:, None]
            ) / self.config.route_length_sigma_m
            route_prior = route_prior + (
                0.5
                * self.config.route_length_weight
                * route_length_normalized.square()
            )
        if self.features.endpoint_tangent_score:
            route_prior = route_prior + (
                self.config.endpoint_tangent_weight * route_alignment
            )
        route_prior = torch.where(
            route_valid,
            route_prior,
            torch.full_like(route_prior, torch.inf),
        )

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
            route_energy = route_huber.mean(dim=-1) + route_prior[:, None, :]
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
            best_prior_indices = torch.argmin(route_prior, dim=-1)
            best_route_indices = best_prior_indices[:, None].expand(
                -1, self.config.particle_count
            )
            best_route_energy = huber.mean(dim=-1) + torch.gather(
                route_prior,
                1,
                best_prior_indices[:, None],
            )
            residual = route_residual
        best_route_indices = torch.where(
            has_route[:, None],
            best_route_indices,
            torch.full_like(best_route_indices, -1),
        )
        energy = torch.where(
            has_route[:, None],
            best_route_energy,
            torch.zeros_like(best_route_energy),
        )
        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["route_score"].record()

        visibility_states = torch.full(
            dense.shape[:-1],
            VISIBILITY_UNKNOWN,
            device=self.device,
            dtype=torch.uint8,
        )
        visibility_available = False
        visibility_informative = torch.zeros(
            2, device=self.device, dtype=torch.bool
        )
        if self.features.partial_observation and not bool(
            np.all(complete_evidence_np)
        ):
            visibility_energy, visibility_states, visibility_available = (
                self._visibility_likelihood(dense, observation)
            )
            energy = energy + torch.where(
                has_route[:, None],
                torch.zeros_like(visibility_energy),
                visibility_energy,
            )
            if visibility_available:
                visibility_informative = (
                    (visibility_states == VISIBILITY_SUPPORTED)
                    | (visibility_states == VISIBILITY_MISSING)
                ).any(dim=(1, 2))
        fragment_available_np = np.zeros(2, dtype=bool)
        if self.features.partial_observation and self.features.fragment_score:
            fragment_energy, fragment_available_np = self._fragment_likelihood(
                dense, observation, route_counts_np
            )
            energy = energy + self.config.fragment_likelihood_weight * fragment_energy

        route_supported = residual <= self.config.support_distance_m
        bad_samples = torch.zeros_like(route_supported)
        bad_samples = torch.where(
            has_route[:, None, None],
            ~route_supported,
            bad_samples,
        )
        if visibility_available:
            partial_missing = visibility_states == VISIBILITY_MISSING
            bad_samples = torch.where(
                has_route[:, None, None], bad_samples, partial_missing
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
        evidence_for_gap = torch.as_tensor(
            complete_evidence_np | fragment_available_np,
            device=self.device,
            dtype=torch.bool,
        )
        if partial_map_available:
            evidence_for_gap = evidence_for_gap | visibility_informative
        energy = energy + torch.where(
            evidence_for_gap[:, None],
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
        if self.features.endpoint_tangent_score:
            endpoint_directions = torch.stack(
                (unit[:, :, 0, :], -unit[:, :, -1, :]),
                dim=2,
            )
            tangent_valid = torch.as_tensor(
                effective_endpoint_visible,
                device=self.device,
                dtype=self.dtype,
            ) * (torch.linalg.vector_norm(tangents, dim=-1) > 1e-6).to(self.dtype)
            tangent_dot = torch.sum(
                endpoint_directions * tangents[:, None, :, :], dim=-1
            ).clamp(-1.0, 1.0)
            tangent_error = torch.sum(
                (1.0 - tangent_dot) * tangent_valid[:, None, :], dim=-1
            ) / tangent_valid.sum(dim=-1)[:, None].clamp_min(1.0)
            energy = energy + self.config.endpoint_tangent_weight * tangent_error
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
        if self.features.kink_limit:
            limit = math.radians(self.config.maximum_kink_degrees)
            energy = energy + self.config.kink_weight * torch.relu(angles - limit).square().mean(
                dim=-1
            )

        if self.config.profile_stages and self.device.type == "cuda":
            self._profile_events["partial_score"].record()

        trackable = initialized | initializable
        particle_valid = torch.isfinite(energy) & trackable[:, None]
        if self.features.fixed_length:
            particle_valid &= endpoint_error <= self.config.endpoint_tolerance_m
            particle_valid &= link_error <= self.config.endpoint_tolerance_m
        particle_valid &= torch.where(
            evidence_for_gap[:, None],
            longest_unsupported <= self.config.maximum_unsupported_length_m,
            torch.ones_like(particle_valid),
        )

        measurement_available = torch.as_tensor(
            complete_evidence_np
            | (
                self.features.partial_observation
                & (
                    effective_endpoint_visible.any(axis=1)
                    | fragment_available_np
                )
            ),
            device=self.device,
            dtype=torch.bool,
        )
        if (
            self.features.partial_observation
            and observation.cable_pixel_count > 0
        ):
            measurement_available = measurement_available | visibility_informative

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
            accepted[:, None],
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
                    measurement_accepted,
                    candidate_route,
                    self._negative_one_index,
                )
                route_mask = torch.where(
                    measurement_accepted,
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
            cable_indices = torch.arange(2, device=self.device, dtype=torch.int64)
            if visibility_available:
                measured_visibility = visibility_states[
                    cable_indices, selected_indices
                ]
                use_visibility = accepted | (
                    prediction_commit
                    & ~torch.as_tensor(
                        measurement_requested_np,
                        device=self.device,
                        dtype=torch.bool,
                    )
                )
                selected_visibility_batch = torch.where(
                    use_visibility[:, None],
                    measured_visibility,
                    self._unknown_visibility[None, :],
                )
            else:
                selected_visibility_batch = self._unknown_visibility[None, :].expand(
                    2, -1
                )

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
                )
            )
            diagnostic_float_payload = None
            diagnostic_visibility_payload = None
            if diagnostics_required:
                selected_route = selected_routes[cable_index]
                candidate_route = candidate_routes[cable_index]
                selected_index = selected_indices[cable_index]
                selected_visibility = selected_visibility_batch[cable_index]
                if route_counts_np[cable_index] > 0:
                    if self.features.connected_trace:
                        selected_residual = torch.linalg.vector_norm(
                            selected_dense - route_dense[cable_index, candidate_route],
                            dim=-1,
                        )
                    else:
                        graph_points = route_dense[
                            cable_index, route_valid[cable_index]
                        ].reshape(-1, 3)
                        selected_residual = torch.cdist(
                            selected_dense[None, :, :],
                            graph_points[None, :, :],
                        )[0].amin(dim=-1)
                    route_support = selected_residual <= self.config.support_distance_m
                    if not visibility_available:
                        route_visibility = torch.where(
                            route_support,
                            torch.full_like(
                                selected_visibility, VISIBILITY_SUPPORTED
                            ),
                            torch.full_like(selected_visibility, VISIBILITY_MISSING),
                        )
                        selected_visibility = torch.where(
                            measurement_accepted,
                            route_visibility,
                            selected_visibility,
                        )
                    measured_trace_values = torch.stack(
                        (
                            selected_residual.mean(),
                            torch.sqrt(torch.mean(selected_residual.square())),
                            torch.quantile(selected_residual, 0.95),
                            selected_residual.max(),
                        )
                    )
                    trace_values = torch.where(
                        measurement_accepted,
                        measured_trace_values,
                        self._nan_scalar.expand(4),
                    )
                else:
                    trace_values = self._nan_scalar.expand(4)
                selected_supported = selected_visibility == VISIBILITY_SUPPORTED
                selected_occluded = selected_visibility == VISIBILITY_OCCLUDED
                selected_missing = selected_visibility == VISIBILITY_MISSING
                selected_unknown = selected_visibility == VISIBILITY_UNKNOWN
                visibility_fractions = torch.stack(
                    (
                        selected_supported.float().mean(),
                        selected_occluded.float().mean(),
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
                        measurement_accepted,
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
                velocity_correction_applied = velocity_update_particle_mask[
                    cable_index, selected_index
                ]
                selected_displacement_velocity = _clamp_vector_norm(
                    (
                        proposals[cable_index, selected_index]
                        - parent_particles[cable_index, selected_index]
                    )
                    / dt,
                    self.config.maximum_node_velocity_mps,
                )
                measured_displacement_speed = torch.where(
                    velocity_correction_applied,
                    torch.linalg.vector_norm(
                        selected_displacement_velocity,
                        dim=-1,
                    ).mean(),
                    self._nan_scalar,
                )
                scalar_values = torch.stack(
                    (
                        self.weights[cable_index, selected_index],
                        effective_sample_size_batch[cable_index],
                        trace_values[0],
                        trace_values[1],
                        trace_values[2],
                        trace_values[3],
                        visibility_fractions[0],
                        visibility_fractions[1],
                        visibility_fractions[2],
                        visibility_fractions[3],
                        longest_missing_run,
                        maximum_link_error_batch[cable_index],
                        endpoint_error_batch[cable_index],
                        observed_route_length,
                        selected_turn_rms_batch[cable_index],
                        maximum_node_uncertainty_batch[cable_index],
                        selected_route.to(self.dtype),
                        predicted_node_speeds.mean(),
                        measured_displacement_speed,
                        corrected_node_speeds.mean(),
                        corrected_node_speeds.max(),
                        velocity_correction_applied.to(self.dtype),
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
                    fragment_count=int(fragment_counts_np[cable_index]),
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
                "prediction": elapsed("inputs", "prediction"),
                "prediction_constraint": elapsed(
                    "prediction", "prediction_constraint"
                ),
                "proposal": elapsed("prediction_constraint", "proposal"),
                "measurement_constraint": elapsed(
                    "proposal", "measurement_constraint"
                ),
                "velocity_correction": elapsed(
                    "measurement_constraint", "velocity_correction"
                ),
                "route_score": elapsed(
                    "velocity_correction", "route_score"
                ),
                "partial_score": elapsed("route_score", "partial_score"),
                "weights": elapsed("partial_score", "weights"),
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
            ) = (float(value) for value in state_payload)
            measurement_accepted = bool(measurement_accepted_value)
            current_initialized = bool(current_initialized_value)
            measurement_available_value = bool(measurement_available_value)
            particle_any_valid_value = bool(particle_any_valid_value)
            if not eligible_np[cable_index]:
                status = "LOST: complete route required for initialization"
            elif not measurement_available_value:
                status = "no usable measurement"
            elif not particle_any_valid_value:
                status = "all particles rejected by geometric/support constraints"
            else:
                status = "measurement accepted"

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
                trace_mean,
                trace_rms,
                trace_p95,
                trace_max,
                support_fraction,
                occluded_fraction,
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
                measured_displacement_speed_mps,
                corrected_node_speed_mps,
                maximum_corrected_node_speed_mps,
                velocity_correction_applied_value,
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
            if measurement_accepted and selected_route >= 0:
                tracking_state = "FULL"
            elif measurement_accepted and (
                support_fraction > 0.0 or pending.endpoint_visible_count > 0
            ):
                tracking_state = "PARTIAL"
            elif occluded_fraction > 0.0:
                tracking_state = "OCCLUDED"
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
                fragment_count=pending.fragment_count,
                visible_fraction=support_fraction,
                occluded_fraction=occluded_fraction,
                missing_fraction=missing_fraction,
                unknown_fraction=unknown_fraction,
                maximum_node_uncertainty_mm=maximum_node_uncertainty_mm,
                complete_observation_age_s=complete_age,
                predicted_node_speed_mps=predicted_node_speed_mps,
                measured_displacement_speed_mps=measured_displacement_speed_mps,
                corrected_node_speed_mps=corrected_node_speed_mps,
                maximum_corrected_node_speed_mps=(
                    maximum_corrected_node_speed_mps
                ),
                velocity_correction_applied=bool(
                    velocity_correction_applied_value
                ),
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
        profile = None
        if self.config.profile_stages:
            profile = ParticleFilterProfile(
                route_input_cpu_ms=float(
                    (route_input_finished - started) * 1000.0
                ),
                inputs_gpu_ms=stage_gpu_ms.get("inputs", 0.0),
                prediction_gpu_ms=stage_gpu_ms.get("prediction", 0.0),
                prediction_constraint_gpu_ms=stage_gpu_ms.get(
                    "prediction_constraint", 0.0
                ),
                proposal_gpu_ms=stage_gpu_ms.get("proposal", 0.0),
                measurement_constraint_gpu_ms=stage_gpu_ms.get(
                    "measurement_constraint", 0.0
                ),
                velocity_correction_gpu_ms=stage_gpu_ms.get(
                    "velocity_correction", 0.0
                ),
                route_score_gpu_ms=stage_gpu_ms.get("route_score", 0.0),
                partial_score_gpu_ms=stage_gpu_ms.get("partial_score", 0.0),
                weight_update_gpu_ms=stage_gpu_ms.get("weights", 0.0),
                posterior_gpu_ms=stage_gpu_ms.get("posterior", 0.0),
                estimate_constraint_gpu_ms=stage_gpu_ms.get(
                    "estimate_constraint", 0.0
                ),
                diagnostics_gpu_ms=stage_gpu_ms.get("diagnostics", 0.0),
                readback_cpu_ms=float(
                    (time.perf_counter() - readback_started) * 1000.0
                ),
            )
        return ParticleFilterFrame(
            cables=(outputs[0], outputs[1]),
            processing_ms=float((time.perf_counter() - started) * 1000.0),
            gpu_ms=gpu_ms,
            active_features=self.features.summary(),
            profile=profile,
            diagnostics_refreshed=diagnostics_required,
        )
