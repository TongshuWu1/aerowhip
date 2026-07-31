"""Passive uncertainty-aware cable--cube contact inference on CUDA."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from cube_state_filter import CubeStateEstimate
from cube_tracker import CubeTrackingResult
from particle_filter import BatchedCableParticleFilter, ParticleFilterFrame


@dataclass(frozen=True)
class ContactEstimatorConfig:
    """Parameters of the passive binary contact observation model."""

    surface_noise_floor_m: float = 0.002
    normal_velocity_std_floor_mps: float = 0.040
    initial_contact_probability: float = 0.02
    contact_entry_rate_hz: float = 0.05
    contact_exit_rate_hz: float = 0.50
    observation_outlier_probability: float = 0.10
    interval_sigma_scale: float = 2.0
    comotion_min_speed_mps: float = 0.020
    comotion_relative_speed_std_mps: float = 0.040
    comotion_max_bayes_factor: float = 3.0

    @classmethod
    def from_mapping(cls, values: dict | None) -> "ContactEstimatorConfig":
        source = values or {}
        defaults = cls()
        config = cls(
            surface_noise_floor_m=float(
                source.get(
                    "surface_noise_floor_m",
                    defaults.surface_noise_floor_m,
                )
            ),
            normal_velocity_std_floor_mps=float(
                source.get(
                    "normal_velocity_std_floor_mps",
                    defaults.normal_velocity_std_floor_mps,
                )
            ),
            initial_contact_probability=float(
                source.get(
                    "initial_contact_probability",
                    defaults.initial_contact_probability,
                )
            ),
            contact_entry_rate_hz=float(
                source.get(
                    "contact_entry_rate_hz",
                    defaults.contact_entry_rate_hz,
                )
            ),
            contact_exit_rate_hz=float(
                source.get(
                    "contact_exit_rate_hz",
                    defaults.contact_exit_rate_hz,
                )
            ),
            observation_outlier_probability=float(
                source.get(
                    "observation_outlier_probability",
                    defaults.observation_outlier_probability,
                )
            ),
            interval_sigma_scale=float(
                source.get(
                    "interval_sigma_scale",
                    defaults.interval_sigma_scale,
                )
            ),
            comotion_min_speed_mps=float(
                source.get(
                    "comotion_min_speed_mps",
                    defaults.comotion_min_speed_mps,
                )
            ),
            comotion_relative_speed_std_mps=float(
                source.get(
                    "comotion_relative_speed_std_mps",
                    defaults.comotion_relative_speed_std_mps,
                )
            ),
            comotion_max_bayes_factor=float(
                source.get(
                    "comotion_max_bayes_factor",
                    defaults.comotion_max_bayes_factor,
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not all(math.isfinite(float(value)) for value in vars(self).values()):
            raise ValueError("All contact parameters must be finite.")
        if self.surface_noise_floor_m <= 0.0:
            raise ValueError("surface_noise_floor_m must be positive.")
        if self.normal_velocity_std_floor_mps <= 0.0:
            raise ValueError("normal_velocity_std_floor_mps must be positive.")
        if not 0.0 < self.initial_contact_probability < 1.0:
            raise ValueError("initial_contact_probability must be in (0, 1).")
        if self.contact_entry_rate_hz < 0.0:
            raise ValueError("contact_entry_rate_hz cannot be negative.")
        if self.contact_exit_rate_hz <= 0.0:
            raise ValueError("contact_exit_rate_hz must be positive.")
        if not 0.0 < self.observation_outlier_probability < 0.5:
            raise ValueError(
                "observation_outlier_probability must be in (0, 0.5)."
            )
        if self.interval_sigma_scale <= 0.0:
            raise ValueError("interval_sigma_scale must be positive.")
        if self.comotion_min_speed_mps <= 0.0:
            raise ValueError("comotion_min_speed_mps must be positive.")
        if self.comotion_relative_speed_std_mps <= 0.0:
            raise ValueError(
                "comotion_relative_speed_std_mps must be positive."
            )
        if self.comotion_max_bayes_factor < 1.0:
            raise ValueError("comotion_max_bayes_factor must be at least one.")


@dataclass(frozen=True)
class CableContactEstimate:
    """Passive contact posterior and its current geometric explanation."""

    initialized: bool
    geometry_valid: bool
    evidence_used: bool
    reason: str
    contact_probability: float
    evidence_age_s: float
    minimum_gap_m: float
    gap_std_m: float
    normal_velocity_mps: float
    comotion_score: float
    local_support_mass: float
    contact_arc_m: float
    arc_interval_m: tuple[float, float]
    closest_point_m: np.ndarray | None
    surface_normal: np.ndarray | None


@dataclass(frozen=True)
class ContactFrame:
    """Two independent cable--cube contact posteriors for one camera frame."""

    timestamp: float
    cables: tuple[CableContactEstimate, CableContactEstimate]
    processing_ms: float
    gpu_ms: float


@dataclass(frozen=True)
class _GeometryBatch:
    payload: np.ndarray
    gpu_ms: float


class CableCubeContactEstimator:
    """Read-only contact observer over the current cable and cube posteriors."""

    def __init__(
        self,
        config: ContactEstimatorConfig,
        *,
        cable_lengths_m: tuple[float, float],
        particle_count: int,
        node_count: int,
        dense_samples_per_segment: int,
        device: str | torch.device,
    ):
        config.validate()
        if len(cable_lengths_m) != 2 or any(
            not np.isfinite(value) or value <= 0.0
            for value in cable_lengths_m
        ):
            raise ValueError("Contact inference requires two positive cable lengths.")
        if particle_count <= 0 or node_count < 2:
            raise ValueError(
                "Contact inference requires particles with at least two nodes."
            )
        if dense_samples_per_segment <= 0:
            raise ValueError("dense_samples_per_segment must be positive.")

        self.config = config
        self.cable_lengths_m = tuple(float(value) for value in cable_lengths_m)
        self.particle_count = int(particle_count)
        self.node_count = int(node_count)
        self.dense_samples_per_segment = int(dense_samples_per_segment)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA contact inference was requested but torch.cuda is unavailable."
            )
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = torch.float32
        self.sample_count = (
            (self.node_count - 1) * self.dense_samples_per_segment + 1
        )
        self._alpha = torch.arange(
            self.dense_samples_per_segment,
            device=self.device,
            dtype=self.dtype,
        ) / float(self.dense_samples_per_segment)
        self._sample_arcs = torch.as_tensor(
            self.cable_lengths_m,
            device=self.device,
            dtype=self.dtype,
        )[:, None] * torch.linspace(
            0.0,
            1.0,
            self.sample_count,
            device=self.device,
            dtype=self.dtype,
        )[None, :]
        self._probabilities = np.full(
            2,
            self.config.initial_contact_probability,
            dtype=np.float64,
        )
        self._initialized = np.zeros(2, dtype=bool)
        self._last_evidence_timestamp = np.full(2, np.nan, dtype=np.float64)
        self._last_timestamp: float | None = None
        self._surface_noise_m = self.config.surface_noise_floor_m
        if self.device.type == "cuda":
            self._gpu_start = torch.cuda.Event(enable_timing=True)
            self._gpu_end = torch.cuda.Event(enable_timing=True)
        else:
            self._gpu_start = None
            self._gpu_end = None

    def _dense_samples(self, values: torch.Tensor) -> torch.Tensor:
        starts = values[:, :, :-1, None, :]
        differences = values[:, :, 1:, None, :] - starts
        samples = starts + differences * self._alpha[None, None, None, :, None]
        samples = samples.reshape(values.shape[0], values.shape[1], -1, 3)
        return torch.cat((samples, values[:, :, -1:, :]), dim=2)

    @staticmethod
    def _box_query(
        points: torch.Tensor,
        center_m: torch.Tensor,
        rotation: torch.Tensor,
        side_m: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        half_side = 0.5 * float(side_m)
        local = torch.matmul(points - center_m, rotation)
        relative = torch.abs(local) - half_side
        outside_vector = torch.clamp_min(relative, 0.0)
        outside_distance = torch.linalg.vector_norm(outside_vector, dim=-1)
        signed_distance = outside_distance + torch.minimum(
            torch.amax(relative, dim=-1),
            torch.zeros_like(outside_distance),
        )

        closest_local = torch.clamp(local, -half_side, half_side)
        outside_delta = local - closest_local
        outside_normal = outside_delta / outside_distance[..., None].clamp_min(
            1.0e-12
        )

        face_axis = torch.argmax(torch.abs(local), dim=-1, keepdim=True)
        face_coordinate = torch.gather(local, -1, face_axis)
        face_sign = torch.where(
            face_coordinate >= 0.0,
            torch.ones_like(face_coordinate),
            -torch.ones_like(face_coordinate),
        )
        inside_normal = torch.zeros_like(local).scatter(-1, face_axis, face_sign)
        inside_closest = closest_local.scatter(
            -1,
            face_axis,
            face_sign * half_side,
        )
        outside = outside_distance > 1.0e-12
        normal_local = torch.where(
            outside[..., None],
            outside_normal,
            inside_normal,
        )
        closest_local = torch.where(
            outside[..., None],
            closest_local,
            inside_closest,
        )
        closest_world = center_m + torch.matmul(closest_local, rotation.T)
        normal_world = torch.matmul(normal_local, rotation.T)
        return signed_distance, closest_world, normal_world

    @torch.inference_mode()
    def _evaluate_geometry(
        self,
        particles: torch.Tensor,
        velocities: torch.Tensor,
        weights: torch.Tensor,
        dense_support: torch.Tensor,
        *,
        cable_radius_m: float,
        cube_side_m: float,
        center_m: np.ndarray,
        rotation: np.ndarray,
        linear_velocity_mps: np.ndarray,
        angular_velocity_rps: np.ndarray,
        state_covariance: np.ndarray,
        surface_noise_m: float,
    ) -> _GeometryBatch:
        if particles.device != self.device or velocities.device != self.device:
            raise ValueError("Contact inputs must remain on the configured PF device.")
        if particles.shape != (
            2,
            self.particle_count,
            self.node_count,
            3,
        ):
            raise ValueError(f"Unexpected contact particle shape: {particles.shape}.")
        if velocities.shape != particles.shape:
            raise ValueError("Contact particle positions and velocities must align.")
        if weights.shape != (2, self.particle_count):
            raise ValueError(f"Unexpected contact weight shape: {weights.shape}.")
        if dense_support.shape != (2, self.sample_count):
            raise ValueError(
                f"Unexpected contact support shape: {dense_support.shape}."
            )
        if dense_support.device != self.device:
            raise ValueError("Contact support must remain on the PF device.")

        if self._gpu_start is not None:
            self._gpu_start.record()

        center = torch.as_tensor(center_m, device=self.device, dtype=self.dtype)
        cube_rotation = torch.as_tensor(
            rotation,
            device=self.device,
            dtype=self.dtype,
        )
        cube_linear_velocity = torch.as_tensor(
            linear_velocity_mps,
            device=self.device,
            dtype=self.dtype,
        )
        cube_angular_velocity = torch.as_tensor(
            angular_velocity_rps,
            device=self.device,
            dtype=self.dtype,
        )
        covariance = torch.as_tensor(
            state_covariance,
            device=self.device,
            dtype=self.dtype,
        )
        pose_covariance = covariance[:6, :6]
        velocity_covariance = covariance[6:12, 6:12]

        dense_points = self._dense_samples(particles)
        dense_velocities = self._dense_samples(velocities)
        signed_distance, closest_points, normals = self._box_query(
            dense_points,
            center,
            cube_rotation,
            cube_side_m,
        )
        gap = signed_distance - float(cable_radius_m)
        lever = closest_points - center
        rotation_jacobian = torch.cross(normals, lever, dim=-1)
        pose_jacobian = torch.cat((-normals, rotation_jacobian), dim=-1)
        pose_variance = torch.sum(
            torch.matmul(pose_jacobian, pose_covariance) * pose_jacobian,
            dim=-1,
        )
        gap_std = torch.sqrt(
            torch.clamp_min(
                pose_variance + float(surface_noise_m) ** 2,
                1.0e-12,
            )
        )

        minimum_gap, minimum_index = torch.min(gap, dim=-1)
        particle_support = torch.gather(
            dense_support[:, None, :].expand(
                -1,
                self.particle_count,
                -1,
            ),
            2,
            minimum_index[:, :, None],
        )[:, :, 0].to(self.dtype)
        sample_index_3 = minimum_index[..., None, None].expand(-1, -1, 1, 3)
        sample_index_1 = minimum_index[..., None]
        selected_closest = torch.gather(
            closest_points,
            2,
            sample_index_3,
        )[:, :, 0]
        selected_normal = torch.gather(normals, 2, sample_index_3)[:, :, 0]
        selected_cable_velocity = torch.gather(
            dense_velocities,
            2,
            sample_index_3,
        )[:, :, 0]
        selected_gap_std = torch.gather(gap_std, 2, sample_index_1)[:, :, 0]
        selected_lever = selected_closest - center
        surface_velocity = cube_linear_velocity + torch.cross(
            cube_angular_velocity.expand_as(selected_lever),
            selected_lever,
            dim=-1,
        )
        relative_velocity = selected_cable_velocity - surface_velocity
        normal_velocity = torch.sum(selected_normal * relative_velocity, dim=-1)

        velocity_rotation_jacobian = torch.cross(
            selected_normal,
            selected_lever,
            dim=-1,
        )
        velocity_jacobian = torch.cat(
            (-selected_normal, velocity_rotation_jacobian),
            dim=-1,
        )
        normal_velocity_variance = torch.sum(
            torch.matmul(velocity_jacobian, velocity_covariance)
            * velocity_jacobian,
            dim=-1,
        )
        normal_velocity_std = torch.sqrt(
            torch.clamp_min(
                normal_velocity_variance
                + self.config.normal_velocity_std_floor_mps**2,
                1.0e-12,
            )
        )

        normalized_gap = minimum_gap / selected_gap_std
        proximity_score = torch.exp(-0.5 * normalized_gap.square())
        normal_score = torch.exp(
            -0.5 * (normal_velocity / normal_velocity_std).square()
        )
        contact_score = proximity_score * normal_score
        free_score = 0.5 * (
            1.0 + torch.erf(normalized_gap / math.sqrt(2.0))
        )
        outlier = self.config.observation_outlier_probability
        contact_likelihood = outlier + (1.0 - outlier) * contact_score
        free_likelihood = outlier + (1.0 - outlier) * free_score

        normalized_weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            1.0e-12
        )
        cable_speed = torch.linalg.vector_norm(selected_cable_velocity, dim=-1)
        surface_speed = torch.linalg.vector_norm(surface_velocity, dim=-1)
        minimum_speed = torch.minimum(cable_speed, surface_speed)
        excitation = torch.clamp(
            (
                minimum_speed - self.config.comotion_min_speed_mps
            )
            / self.config.comotion_min_speed_mps,
            0.0,
            1.0,
        )
        relative_speed = torch.linalg.vector_norm(relative_velocity, dim=-1)
        comotion_compatibility = torch.exp(
            -0.5
            * (
                relative_speed
                / self.config.comotion_relative_speed_std_mps
            ).square()
        )
        comotion_particle = (
            proximity_score * excitation * comotion_compatibility
        )
        local_support_mass = torch.sum(
            normalized_weights * particle_support,
            dim=1,
        )
        comotion_score = torch.sum(
            normalized_weights * particle_support * comotion_particle,
            dim=1,
        )
        particle_comotion_bayes_factor = 1.0 + (
            self.config.comotion_max_bayes_factor - 1.0
        ) * comotion_particle
        gated_contact_likelihood = (
            particle_support
            * contact_likelihood
            * particle_comotion_bayes_factor
            + (1.0 - particle_support)
        )
        gated_free_likelihood = (
            particle_support * free_likelihood
            + (1.0 - particle_support)
        )
        cable_contact_likelihood = torch.sum(
            normalized_weights * gated_contact_likelihood,
            dim=1,
        )
        cable_free_likelihood = torch.sum(
            normalized_weights * gated_free_likelihood,
            dim=1,
        )

        map_particle = torch.argmax(
            normalized_weights * contact_score,
            dim=1,
        )
        cable_indices = torch.arange(2, device=self.device)
        map_sample = minimum_index[cable_indices, map_particle]
        map_gap = minimum_gap[cable_indices, map_particle]
        map_gap_std = selected_gap_std[cable_indices, map_particle]
        map_normal_velocity = normal_velocity[cable_indices, map_particle]
        map_point = selected_closest[cable_indices, map_particle]
        map_normal = selected_normal[cable_indices, map_particle]
        map_arc = self._sample_arcs[cable_indices, map_sample]

        map_gap_samples = gap[cable_indices, map_particle]
        map_std_samples = gap_std[cable_indices, map_particle]
        near_contact = torch.abs(map_gap_samples) <= (
            self.config.interval_sigma_scale * map_std_samples
        )
        sample_indices = torch.arange(self.sample_count, device=self.device)[
            None, :
        ]
        previous_false = torch.where(
            (~near_contact) & (sample_indices < map_sample[:, None]),
            sample_indices,
            torch.full_like(sample_indices, -1),
        ).amax(dim=1)
        next_false = torch.where(
            (~near_contact) & (sample_indices > map_sample[:, None]),
            sample_indices,
            torch.full_like(sample_indices, self.sample_count),
        ).amin(dim=1)
        interval_start_index = previous_false + 1
        interval_end_index = next_false - 1
        near_at_minimum = near_contact[cable_indices, map_sample]
        nan = torch.full((2,), float("nan"), device=self.device, dtype=self.dtype)
        interval_start = torch.where(
            near_at_minimum,
            self._sample_arcs[cable_indices, interval_start_index],
            nan,
        )
        interval_end = torch.where(
            near_at_minimum,
            self._sample_arcs[cable_indices, interval_end_index],
            nan,
        )

        payload = torch.cat(
            (
                cable_contact_likelihood[:, None],
                cable_free_likelihood[:, None],
                map_gap[:, None],
                map_gap_std[:, None],
                map_normal_velocity[:, None],
                comotion_score[:, None],
                local_support_mass[:, None],
                map_arc[:, None],
                interval_start[:, None],
                interval_end[:, None],
                map_point,
                map_normal,
            ),
            dim=1,
        )

        gpu_ms = 0.0
        if self._gpu_end is not None and self._gpu_start is not None:
            self._gpu_end.record()
            self._gpu_end.synchronize()
            gpu_ms = float(self._gpu_start.elapsed_time(self._gpu_end))
        return _GeometryBatch(
            payload=np.ascontiguousarray(payload.cpu().numpy()),
            gpu_ms=gpu_ms,
        )

    def warmup(self) -> None:
        """Compile/initialize the fixed-shape tensor path without changing state."""

        particles = torch.zeros(
            (2, self.particle_count, self.node_count, 3),
            device=self.device,
            dtype=self.dtype,
        )
        particles[..., 0] = torch.linspace(
            -0.05,
            0.05,
            self.node_count,
            device=self.device,
            dtype=self.dtype,
        )
        particles[..., 1] = 0.080
        velocities = torch.zeros_like(particles)
        dense_support = torch.ones(
            (2, self.sample_count),
            device=self.device,
            dtype=torch.bool,
        )
        weights = torch.full(
            (2, self.particle_count),
            1.0 / float(self.particle_count),
            device=self.device,
            dtype=self.dtype,
        )
        self._evaluate_geometry(
            particles,
            velocities,
            weights,
            dense_support,
            cable_radius_m=0.0045,
            cube_side_m=0.150,
            center_m=np.zeros(3, dtype=np.float64),
            rotation=np.eye(3, dtype=np.float64),
            linear_velocity_mps=np.zeros(3, dtype=np.float64),
            angular_velocity_rps=np.zeros(3, dtype=np.float64),
            state_covariance=np.eye(12, dtype=np.float64) * 1.0e-4,
            surface_noise_m=self.config.surface_noise_floor_m,
        )

    def _predict_probability(self, probability: float, dt: float) -> float:
        entry = self.config.contact_entry_rate_hz
        exit_rate = self.config.contact_exit_rate_hz
        total_rate = entry + exit_rate
        stationary = entry / total_rate
        return float(
            stationary
            + (float(probability) - stationary) * math.exp(-total_rate * dt)
        )

    @staticmethod
    def _empty_geometry() -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        tuple[float, float],
        np.ndarray | None,
        np.ndarray | None,
    ]:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            (float("nan"), float("nan")),
            None,
            None,
        )

    @torch.inference_mode()
    def update(
        self,
        particle_filter: BatchedCableParticleFilter,
        particle_frame: ParticleFilterFrame,
        cube_state: CubeStateEstimate | None,
        cube_measurement: CubeTrackingResult | None,
        timestamp: float,
    ) -> ContactFrame:
        started = time.perf_counter()
        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Contact timestamps must be monotonic.")
        dt = (
            0.0
            if self._last_timestamp is None
            else max(0.0, timestamp - self._last_timestamp)
        )
        self._last_timestamp = timestamp

        for cable_index in range(2):
            if self._initialized[cable_index]:
                self._probabilities[cable_index] = self._predict_probability(
                    self._probabilities[cable_index],
                    dt,
                )

        if (
            cube_state is not None
            and cube_state.measurement_used
            and cube_measurement is not None
            and cube_measurement.refinement_valid
            and np.isfinite(cube_measurement.refined_surface_rms_m)
        ):
            self._surface_noise_m = max(
                self.config.surface_noise_floor_m,
                float(cube_measurement.refined_surface_rms_m),
            )

        geometry_available = bool(
            cube_state is not None
            and cube_state.valid
            and cube_state.center_m is not None
            and cube_state.rotation is not None
            and cube_state.linear_velocity_mps is not None
            and cube_state.angular_velocity_rps is not None
            and cube_state.state_covariance is not None
            and any(particle_frame.initialized)
        )
        geometry: _GeometryBatch | None = None
        if geometry_available:
            assert cube_state is not None
            particles, velocities, weights = particle_filter.posterior_tensors()
            dense_support = particle_filter.contact_support_tensor()
            geometry = self._evaluate_geometry(
                particles,
                velocities,
                weights,
                dense_support,
                cable_radius_m=particle_frame.cable_radius_m,
                cube_side_m=cube_state.cube_side_m,
                center_m=cube_state.center_m,
                rotation=cube_state.rotation,
                linear_velocity_mps=cube_state.linear_velocity_mps,
                angular_velocity_rps=cube_state.angular_velocity_rps,
                state_covariance=cube_state.state_covariance,
                surface_noise_m=self._surface_noise_m,
            )

        outputs: list[CableContactEstimate] = []
        for cable_index in range(2):
            cable_geometry_valid = bool(
                geometry is not None and particle_frame.initialized[cable_index]
            )
            if cable_geometry_valid:
                assert geometry is not None
                row = geometry.payload[cable_index]
                contact_likelihood = max(float(row[0]), 1.0e-12)
                free_likelihood = max(float(row[1]), 1.0e-12)
                minimum_gap = float(row[2])
                gap_std = float(row[3])
                normal_velocity = float(row[4])
                comotion_score = float(np.clip(row[5], 0.0, 1.0))
                local_support_mass = float(np.clip(row[6], 0.0, 1.0))
                contact_arc = float(row[7])
                interval = (float(row[8]), float(row[9]))
                closest_point = np.ascontiguousarray(
                    row[10:13], dtype=np.float32
                )
                surface_normal = np.ascontiguousarray(
                    row[13:16], dtype=np.float32
                )
            else:
                (
                    minimum_gap,
                    gap_std,
                    normal_velocity,
                    comotion_score,
                    local_support_mass,
                    contact_arc,
                    interval,
                    closest_point,
                    surface_normal,
                ) = self._empty_geometry()
                contact_likelihood = free_likelihood = 1.0

            evidence_used = bool(
                cable_geometry_valid
                and cube_state is not None
                and cube_state.measurement_used
                and particle_frame.measurement_used[cable_index]
                and local_support_mass > 0.0
            )

            if evidence_used:
                prior = (
                    self._probabilities[cable_index]
                    if self._initialized[cable_index]
                    else self.config.initial_contact_probability
                )
                numerator = prior * contact_likelihood
                denominator = numerator + (1.0 - prior) * free_likelihood
                self._probabilities[cable_index] = float(
                    np.clip(numerator / max(denominator, 1.0e-12), 0.0, 1.0)
                )
                self._initialized[cable_index] = True
                self._last_evidence_timestamp[cable_index] = timestamp
                reason = "contact measurement update"
            elif not particle_frame.initialized[cable_index]:
                reason = "waiting for initialized cable posterior"
            elif cube_state is None or not cube_state.valid:
                reason = "waiting for valid temporal cube state"
            elif not cube_state.measurement_used:
                reason = "contact prediction only: cube measurement unavailable"
            elif not particle_frame.measurement_used[cable_index]:
                reason = "contact prediction only: cable measurement unavailable"
            elif local_support_mass <= 0.0:
                reason = "contact prediction only: closest cable arc unobserved"
            else:
                reason = "contact prediction only"

            last_evidence = self._last_evidence_timestamp[cable_index]
            evidence_age = (
                max(0.0, timestamp - last_evidence)
                if np.isfinite(last_evidence)
                else float("inf")
            )
            outputs.append(
                CableContactEstimate(
                    initialized=bool(self._initialized[cable_index]),
                    geometry_valid=cable_geometry_valid,
                    evidence_used=evidence_used,
                    reason=reason,
                    contact_probability=(
                        float(self._probabilities[cable_index])
                        if self._initialized[cable_index]
                        else float("nan")
                    ),
                    evidence_age_s=evidence_age,
                    minimum_gap_m=minimum_gap,
                    gap_std_m=gap_std,
                    normal_velocity_mps=normal_velocity,
                    comotion_score=comotion_score,
                    local_support_mass=local_support_mass,
                    contact_arc_m=contact_arc,
                    arc_interval_m=interval,
                    closest_point_m=closest_point,
                    surface_normal=surface_normal,
                )
            )

        return ContactFrame(
            timestamp=timestamp,
            cables=(outputs[0], outputs[1]),
            processing_ms=float((time.perf_counter() - started) * 1000.0),
            gpu_ms=(geometry.gpu_ms if geometry is not None else 0.0),
        )
