"""Batched fixed-length particle hypotheses for two endpoint-identified cables."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from observation import FrameObservation


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
    kink_limit: bool = False
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
    process_noise_m: float = 0.0015
    global_noise_m: float = 0.018
    global_fraction: float = 0.10
    motion_noise_gain: float = 0.25
    maximum_process_noise_m: float = 0.015
    velocity_smoothing: float = 0.65
    projection_iterations: int = 8
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
    top_particle_count: int = 12
    random_seed: int = 7

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
            process_noise_m=max(0.0, float(values.get("process_noise_m", 0.0015))),
            global_noise_m=max(0.0, float(values.get("global_noise_m", 0.018))),
            global_fraction=float(np.clip(values.get("global_fraction", 0.10), 0.0, 1.0)),
            motion_noise_gain=max(0.0, float(values.get("motion_noise_gain", 0.25))),
            maximum_process_noise_m=max(
                0.0, float(values.get("maximum_process_noise_m", 0.015))
            ),
            velocity_smoothing=float(np.clip(values.get("velocity_smoothing", 0.65), 0.0, 0.999)),
            projection_iterations=max(1, int(values.get("projection_iterations", 8))),
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
            top_particle_count=max(1, int(values.get("top_particle_count", 12))),
            random_seed=int(values.get("random_seed", 7)),
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


@dataclass(frozen=True)
class CableFilterOutput:
    curve: np.ndarray
    dense_curve: np.ndarray
    dense_supported: np.ndarray
    top_particles: np.ndarray
    diagnostics: CableFilterDiagnostics


@dataclass(frozen=True)
class ParticleFilterFrame:
    cables: tuple[CableFilterOutput, CableFilterOutput]
    processing_ms: float
    gpu_ms: float
    active_features: str


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
    )
    return CableFilterOutput(
        curve=np.empty((0, 3), dtype=np.float32),
        dense_curve=np.empty((0, 3), dtype=np.float32),
        dense_supported=np.empty(0, dtype=bool),
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


def _dense_samples(particles: torch.Tensor, samples_per_segment: int) -> torch.Tensor:
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
    segment_lengths: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    output = particles.clone()
    epsilon = torch.finfo(output.dtype).eps
    segment = segment_lengths.view(-1, 1, 1)
    for _ in range(int(iterations)):
        output[:, :, -1, :] = endpoints[:, None, 1, :]
        for index in range(output.shape[2] - 2, -1, -1):
            direction = output[:, :, index, :] - output[:, :, index + 1, :]
            direction = direction / torch.linalg.vector_norm(
                direction, dim=-1, keepdim=True
            ).clamp_min(epsilon)
            output[:, :, index, :] = output[:, :, index + 1, :] + segment * direction
        output[:, :, 0, :] = endpoints[:, None, 0, :]
        for index in range(output.shape[2] - 1):
            direction = output[:, :, index + 1, :] - output[:, :, index, :]
            direction = direction / torch.linalg.vector_norm(
                direction, dim=-1, keepdim=True
            ).clamp_min(epsilon)
            output[:, :, index + 1, :] = output[:, :, index, :] + segment * direction
    return output


class BatchedCableParticleFilter:
    """Two independent filters evaluated together with a cable batch dimension."""

    def __init__(self, config: ParticleFilterConfig, features: ParticleFeatures):
        self.config = config
        self.features = features
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA particle filtering was requested but torch.cuda is unavailable")
        self.dtype = torch.float32
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(config.random_seed)
        self.particles: torch.Tensor | None = None
        self.velocities: torch.Tensor | None = None
        self.weights: torch.Tensor | None = None
        self.route_indices: torch.Tensor | None = None
        self.initialized = np.zeros(2, dtype=bool)
        self.last_endpoints = np.full((2, 2, 3), np.nan, dtype=np.float32)
        self.last_timestamp: float | None = None
        self.lengths = torch.tensor(config.cable_lengths_m, device=self.device, dtype=self.dtype)
        self.segment_lengths = self.lengths / float(config.node_count - 1)
        positions = torch.linspace(0.0, 1.0, config.node_count, device=self.device)
        modes = torch.arange(
            1, config.deformation_modes + 1, device=self.device, dtype=self.dtype
        )
        self.basis = torch.sin(math.pi * positions[:, None] * modes[None, :])
        self.basis = self.basis / torch.linalg.vector_norm(self.basis, dim=0, keepdim=True).clamp_min(1e-6)

    def set_features(self, features: ParticleFeatures) -> None:
        self.features = features

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

    def _smooth_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        coefficients = torch.randn(
            (2, self.config.particle_count, self.config.deformation_modes, 3),
            generator=self.generator,
            device=self.device,
            dtype=self.dtype,
        )
        noise = torch.einsum("cpkd,nk->cpnd", coefficients, self.basis)
        return noise * sigma.view(2, 1, 1, 1)

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
        list[str],
    ]:
        endpoints = np.zeros((2, 2, 3), dtype=np.float32)
        tangents = np.zeros((2, 2, 3), dtype=np.float32)
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
        valid = np.zeros(2, dtype=bool)
        reasons = []
        for cable_index, cable in enumerate(observation.cables):
            reason = cable.reason
            if cable.valid:
                chord = float(np.linalg.norm(cable.endpoints_xyz[1] - cable.endpoints_xyz[0]))
                if chord > self.config.cable_lengths_m[cable_index] + 1e-6:
                    reason = (
                        f"endpoint chord {chord:.3f} m exceeds cable length "
                        f"{self.config.cable_lengths_m[cable_index]:.3f} m"
                    )
                else:
                    endpoints[cable_index] = cable.endpoints_xyz
                    tangents[cable_index] = cable.endpoint_tangents
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
                    valid[cable_index] = route_counts[cable_index] > 0
                    if not valid[cable_index]:
                        reason = "no finite graph route candidates"
            reasons.append(reason)
        return (
            valid,
            endpoints,
            tangents,
            route_nodes,
            route_dense,
            route_lengths,
            route_alignment,
            route_valid,
            route_counts,
            reasons,
        )

    @torch.inference_mode()
    def update(self, observation: FrameObservation, timestamp: float) -> ParticleFilterFrame:
        started = time.perf_counter()
        self._allocate()
        assert self.particles is not None
        assert self.velocities is not None
        assert self.weights is not None
        assert self.route_indices is not None
        (
            valid_np,
            endpoints_np,
            tangents_np,
            route_nodes_np,
            route_dense_np,
            route_lengths_np,
            route_alignment_np,
            route_valid_np,
            route_counts_np,
            reasons,
        ) = self._route_arrays(observation)
        dt = 1.0 / 30.0 if self.last_timestamp is None else float(timestamp - self.last_timestamp)
        dt = float(np.clip(dt, 1.0 / 120.0, 0.25))
        self.last_timestamp = float(timestamp)

        gpu_start = gpu_end = None
        if self.device.type == "cuda":
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
        valid = torch.as_tensor(valid_np, device=self.device, dtype=torch.bool)
        initialized = torch.as_tensor(self.initialized, device=self.device, dtype=torch.bool)

        parent_indices = torch.multinomial(
            self.weights,
            self.config.particle_count,
            replacement=True,
            generator=self.generator,
        )
        gather_index = parent_indices[:, :, None, None].expand(
            -1, -1, self.config.node_count, 3
        )
        parent_particles = torch.gather(self.particles, 1, gather_index)
        parent_velocities = torch.gather(self.velocities, 1, gather_index)

        if self.features.temporal_prediction:
            proposals = parent_particles + parent_velocities * dt
        else:
            proposals = parent_particles.clone()

        endpoint_displacement = np.zeros(2, dtype=np.float32)
        for cable_index in range(2):
            if valid_np[cable_index] and np.all(np.isfinite(self.last_endpoints[cable_index])):
                endpoint_displacement[cable_index] = float(
                    np.mean(
                        np.linalg.norm(
                            endpoints_np[cable_index] - self.last_endpoints[cable_index], axis=1
                        )
                    )
                )
        process_sigma = np.full(2, self.config.process_noise_m, dtype=np.float32)
        if self.features.motion_adaptive_noise:
            process_sigma += self.config.motion_noise_gain * endpoint_displacement
            process_sigma = np.minimum(process_sigma, self.config.maximum_process_noise_m)
        proposals = proposals + self._smooth_noise(
            torch.as_tensor(process_sigma, device=self.device, dtype=self.dtype)
        )

        proposal_centers = route_nodes.clone()
        if self.features.endpoint_direction_proposal:
            proposal_centers[:, :, 1, :] = (
                endpoints[:, None, 0, :]
                + self.segment_lengths[:, None, None] * tangents[:, None, 0, :]
            )
            proposal_centers[:, :, -2, :] = (
                endpoints[:, None, 1, :]
                + self.segment_lengths[:, None, None] * tangents[:, None, 1, :]
            )

        particle_route_slots = (
            torch.arange(self.config.particle_count, device=self.device)[None, :]
            % route_counts.clamp_min(1)[:, None]
        )
        particle_center_index = particle_route_slots[:, :, None, None].expand(
            -1, -1, self.config.node_count, 3
        )
        particle_centers = torch.gather(
            proposal_centers,
            1,
            particle_center_index,
        )

        newly_initialized = valid & ~initialized
        if torch.any(newly_initialized):
            initial = particle_centers.clone()
            initial = initial + self._smooth_noise(
                torch.full((2,), self.config.initial_noise_m, device=self.device)
            )
            proposals = torch.where(newly_initialized[:, None, None, None], initial, proposals)
            parent_velocities = torch.where(
                newly_initialized[:, None, None, None],
                torch.zeros_like(parent_velocities),
                parent_velocities,
            )

        global_count = 0
        if self.features.global_particles and self.config.global_fraction > 0.0:
            global_count = max(
                1, int(round(self.config.particle_count * self.config.global_fraction))
            )
            global_proposals = particle_centers[:, :global_count].clone()
            coefficients = torch.randn(
                (2, global_count, self.config.deformation_modes, 3),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            global_noise = torch.einsum("cpkd,nk->cpnd", coefficients, self.basis)
            proposals[:, :global_count] = (
                global_proposals + global_noise * self.config.global_noise_m
            )
            parent_velocities[:, :global_count] = 0.0

        proposals[:, :, 0, :] = endpoints[:, None, 0, :]
        proposals[:, :, -1, :] = endpoints[:, None, 1, :]
        if self.features.fixed_length:
            proposals = _constrain_fixed_links(
                proposals,
                endpoints,
                self.segment_lengths,
                self.config.projection_iterations,
            )

        dense = _dense_samples(proposals, self.config.dense_samples_per_segment)
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
            trace_energy = torch.gather(
                route_energy,
                2,
                best_route_indices[:, :, None],
            ).squeeze(2)
        else:
            residual = _unordered_graph_residual(dense, route_dense, route_valid)
            normalized = residual / self.config.trace_sigma_m
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
            trace_energy = huber.mean(dim=-1) + torch.gather(
                route_prior,
                1,
                best_prior_indices[:, None],
            )

        supported = residual <= self.config.support_distance_m
        sample_indices = torch.arange(dense.shape[2], device=self.device, dtype=torch.int64)
        last_supported = torch.where(
            supported,
            sample_indices.view(1, 1, -1),
            torch.full_like(sample_indices, -1).view(1, 1, -1),
        )
        last_supported = torch.cummax(last_supported, dim=-1).values
        bad_run_samples = torch.where(
            supported,
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
        energy = trace_energy + self.config.unsupported_weight * gap_fraction.square()

        links = proposals[:, :, 1:, :] - proposals[:, :, :-1, :]
        link_lengths = torch.linalg.vector_norm(links, dim=-1)
        unit = links / link_lengths[..., None].clamp_min(1e-8)
        cosine = torch.sum(
            unit[:, :, :-1, :] * unit[:, :, 1:, :], dim=-1
        ).clamp(-1.0, 1.0)
        angles = torch.acos(cosine)
        turn_rms = torch.sqrt(torch.mean(angles.square(), dim=-1))
        if self.features.smoothness:
            energy = energy + self.config.smoothness_weight * torch.mean(
                angles.square(), dim=-1
            )
        if self.features.endpoint_tangent_score:
            endpoint_directions = torch.stack(
                (unit[:, :, 0, :], -unit[:, :, -1, :]),
                dim=2,
            )
            tangent_valid = (
                torch.linalg.vector_norm(tangents, dim=-1) > 1e-6
            ).to(self.dtype)
            tangent_dot = torch.sum(
                endpoint_directions * tangents[:, None, :, :], dim=-1
            ).clamp(-1.0, 1.0)
            tangent_error = torch.sum(
                (1.0 - tangent_dot) * tangent_valid[:, None, :], dim=-1
            ) / tangent_valid.sum(dim=-1)[:, None].clamp_min(1.0)
            energy = energy + self.config.endpoint_tangent_weight * tangent_error
        link_error = torch.abs(link_lengths - self.segment_lengths[:, None, None]).amax(dim=-1)
        endpoint_error = torch.maximum(
            torch.linalg.vector_norm(proposals[:, :, 0, :] - endpoints[:, None, 0, :], dim=-1),
            torch.linalg.vector_norm(proposals[:, :, -1, :] - endpoints[:, None, 1, :], dim=-1),
        )
        if self.features.kink_limit:
            limit = math.radians(self.config.maximum_kink_degrees)
            energy = energy + self.config.kink_weight * torch.relu(angles - limit).square().mean(
                dim=-1
            )

        particle_valid = torch.isfinite(energy) & valid[:, None]
        if self.features.fixed_length:
            particle_valid &= endpoint_error <= self.config.endpoint_tolerance_m
            particle_valid &= link_error <= self.config.endpoint_tolerance_m
        particle_valid &= (
            longest_unsupported <= self.config.maximum_unsupported_length_m
        )

        candidate_weights = torch.zeros_like(self.weights)
        accepted_update = np.zeros(2, dtype=bool)
        statuses = list(reasons)
        for cable_index in range(2):
            if not valid_np[cable_index]:
                continue
            if not bool(torch.any(particle_valid[cable_index]).item()):
                statuses[cable_index] = "all particles rejected by geometric/support constraints"
                continue
            logits = torch.where(
                particle_valid[cable_index],
                -energy[cable_index],
                torch.full_like(energy[cable_index], -torch.inf),
            )
            candidate_weights[cable_index] = torch.softmax(logits, dim=0)
            accepted_update[cable_index] = True
            statuses[cable_index] = "tracking"

        accepted = torch.as_tensor(accepted_update, device=self.device, dtype=torch.bool)
        displacement_velocity = (proposals - parent_particles) / max(dt, 1e-6)
        new_velocities = (
            self.config.velocity_smoothing * parent_velocities
            + (1.0 - self.config.velocity_smoothing) * displacement_velocity
        )
        if global_count:
            new_velocities[:, :global_count] = 0.0
        self.particles = torch.where(
            accepted[:, None, None, None], proposals, self.particles
        )
        self.velocities = torch.where(
            accepted[:, None, None, None], new_velocities, self.velocities
        )
        self.weights = torch.where(accepted[:, None], candidate_weights, self.weights)
        self.route_indices = torch.where(
            accepted[:, None],
            best_route_indices,
            self.route_indices,
        )
        self.initialized |= accepted_update
        for cable_index in range(2):
            if accepted_update[cable_index]:
                self.last_endpoints[cable_index] = endpoints_np[cable_index]

        if gpu_end is not None:
            gpu_end.record()
            gpu_end.synchronize()
            gpu_ms = float(gpu_start.elapsed_time(gpu_end))
        else:
            gpu_ms = 0.0

        outputs = []
        for cable_index in range(2):
            if not self.initialized[cable_index]:
                outputs.append(_empty_output(statuses[cable_index], bool(valid_np[cable_index])))
                continue
            measurement_accepted = bool(accepted_update[cable_index])
            selected_route = -1
            route_mask = torch.ones(
                self.config.particle_count,
                device=self.device,
                dtype=torch.bool,
            )
            if measurement_accepted:
                route_mass = torch.zeros(
                    max(1, int(route_counts_np[cable_index])),
                    device=self.device,
                    dtype=self.dtype,
                )
                route_mass.scatter_add_(
                    0,
                    self.route_indices[cable_index],
                    self.weights[cable_index],
                )
                selected_route = int(torch.argmax(route_mass).item())
                route_mask = self.route_indices[cable_index] == selected_route
            conditional_weights = self.weights[cable_index] * route_mask.to(self.dtype)
            conditional_weights = conditional_weights / conditional_weights.sum().clamp_min(
                1e-12
            )
            posterior_mean = torch.sum(
                conditional_weights[:, None, None] * self.particles[cable_index],
                dim=0,
            )
            representative_distance = torch.mean(
                torch.sum(
                    (self.particles[cable_index] - posterior_mean[None, :, :]).square(),
                    dim=-1,
                ),
                dim=-1,
            )
            representative_distance = torch.where(
                route_mask,
                representative_distance,
                torch.full_like(representative_distance, torch.inf),
            )
            selected_index = int(torch.argmin(representative_distance).item())
            selected_particle = self.particles[cable_index, selected_index]
            selected_dense = _dense_samples(
                selected_particle[None, None, :, :], self.config.dense_samples_per_segment
            )[0, 0]
            if measurement_accepted and selected_route >= 0:
                if self.features.connected_trace:
                    selected_residual = torch.linalg.vector_norm(
                        selected_dense - route_dense[cable_index, selected_route],
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
                selected_supported = selected_residual <= self.config.support_distance_m
                last_good = torch.where(
                    selected_supported,
                    torch.arange(len(selected_supported), device=self.device),
                    torch.full(
                        (len(selected_supported),),
                        -1,
                        device=self.device,
                        dtype=torch.int64,
                    ),
                )
                last_good = torch.cummax(last_good, dim=0).values
                bad_runs = torch.where(
                    selected_supported,
                    torch.zeros_like(last_good),
                    torch.arange(len(selected_supported), device=self.device) - last_good,
                )
                longest_mm = float(
                    bad_runs.max().item()
                    * self.config.cable_lengths_m[cable_index]
                    / max(1, len(selected_supported) - 1)
                    * 1000.0
                )
                trace_mean = float(selected_residual.mean().item() * 1000.0)
                trace_rms = float(
                    torch.sqrt(torch.mean(selected_residual.square())).item() * 1000.0
                )
                trace_p95 = float(torch.quantile(selected_residual, 0.95).item() * 1000.0)
                trace_max = float(selected_residual.max().item() * 1000.0)
                support_fraction = float(selected_supported.float().mean().item())
            else:
                selected_supported = torch.zeros(
                    len(selected_dense), device=self.device, dtype=torch.bool
                )
                longest_mm = trace_mean = trace_rms = trace_p95 = trace_max = float("nan")
                support_fraction = 0.0
            selected_links = selected_particle[1:] - selected_particle[:-1]
            selected_link_lengths = torch.linalg.vector_norm(selected_links, dim=-1)
            selected_unit = selected_links / selected_link_lengths[:, None].clamp_min(1e-8)
            selected_cosine = torch.sum(
                selected_unit[:-1] * selected_unit[1:], dim=-1
            ).clamp(-1.0, 1.0)
            selected_turn_rms = float(
                torch.sqrt(torch.mean(torch.acos(selected_cosine).square())).item()
                * 180.0
                / math.pi
            )
            maximum_link_error = float(
                torch.abs(
                    selected_link_lengths - self.segment_lengths[cable_index]
                ).max().item()
                * 1000.0
            )
            endpoint_error_mm = float(
                torch.maximum(
                    torch.linalg.vector_norm(selected_particle[0] - endpoints[cable_index, 0]),
                    torch.linalg.vector_norm(selected_particle[-1] - endpoints[cable_index, 1]),
                ).item()
                * 1000.0
            ) if measurement_accepted else float("nan")
            top_count = min(self.config.top_particle_count, self.config.particle_count)
            top_indices = torch.topk(self.weights[cable_index], top_count).indices
            top_particles = self.particles[cable_index, top_indices]
            ess = float((1.0 / torch.sum(self.weights[cable_index].square())).item())
            diagnostics = CableFilterDiagnostics(
                initialized=True,
                measurement_valid=bool(valid_np[cable_index]),
                status=statuses[cable_index],
                selected_weight=float(self.weights[cable_index, selected_index].item()),
                effective_sample_size=ess,
                trace_mean_mm=trace_mean,
                trace_rms_mm=trace_rms,
                trace_p95_mm=trace_p95,
                trace_max_mm=trace_max,
                support_fraction=support_fraction,
                longest_unsupported_mm=longest_mm,
                maximum_link_error_mm=maximum_link_error,
                endpoint_error_mm=endpoint_error_mm,
                observed_route_length_m=float(
                    route_lengths[cable_index, selected_route].item()
                    if measurement_accepted and selected_route >= 0
                    else float("nan")
                ),
                route_candidate_count=int(route_counts_np[cable_index]),
                selected_route_index=selected_route,
                route_length_error_mm=float(
                    abs(
                        route_lengths[cable_index, selected_route].item()
                        - self.config.cable_lengths_m[cable_index]
                    )
                    * 1000.0
                    if measurement_accepted and selected_route >= 0
                    else float("nan")
                ),
                turn_rms_degrees=selected_turn_rms,
            )
            outputs.append(
                CableFilterOutput(
                    curve=np.ascontiguousarray(
                        selected_particle.cpu().numpy(), dtype=np.float32
                    ),
                    dense_curve=np.ascontiguousarray(
                        selected_dense.cpu().numpy(), dtype=np.float32
                    ),
                    dense_supported=np.ascontiguousarray(
                        selected_supported.cpu().numpy(), dtype=bool
                    ),
                    top_particles=np.ascontiguousarray(top_particles.cpu().numpy(), dtype=np.float32),
                    diagnostics=diagnostics,
                )
            )
        return ParticleFilterFrame(
            cables=(outputs[0], outputs[1]),
            processing_ms=float((time.perf_counter() - started) * 1000.0),
            gpu_ms=gpu_ms,
            active_features=self.features.summary(),
        )
