from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from ..shared.dder import DderModel, DderState

from .config import ParticleFilterSettings
from ..shared.model_artifact import DderArtifact


@dataclass(frozen=True, slots=True)
class ParticleFilterEstimate:
    positions_m: np.ndarray
    velocities_m_s: np.ndarray
    posterior_mean_m: np.ndarray
    node_covariance_m2: np.ndarray
    weights: np.ndarray
    ess: float
    residual_m: float
    body_updated: bool
    endpoint_count: int
    prediction_only: bool
    resampled: bool
    timing_ms: tuple[float, float, float, float]


def _arc_resample(points: np.ndarray, count: int) -> np.ndarray:
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.concatenate(([True], delta > 1.0e-6))]
    if len(points) < 2:
        raise ValueError("Visible metric curve has fewer than two distinct samples.")
    arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    target = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(target, arc, points[:, axis]) for axis in range(3)])


def _student_t_negative_log_likelihood(
    value: torch.Tensor, degrees_of_freedom: float
) -> torch.Tensor:
    return 0.5 * (degrees_of_freedom + 1.0) * torch.log1p(
        value.square() / degrees_of_freedom
    )


def _segment_spaced_indices(
    points_m: np.ndarray,
    valid: np.ndarray,
    segment_ids: np.ndarray,
    spacing_m: float,
) -> np.ndarray:
    """Select one observation per DDER segment length along each visible fragment."""

    selected: list[int] = []
    active = np.asarray(valid, dtype=bool) & np.all(np.isfinite(points_m), axis=1)
    for segment in np.unique(segment_ids[active]):
        indices = np.flatnonzero(active & (segment_ids == segment))
        if len(indices) < 1:
            continue
        points = points_m[indices]
        arc = np.concatenate(
            ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        )
        total = float(arc[-1])
        targets = (
            np.asarray((0.5 * total,))
            if total <= spacing_m
            else np.arange(0.5 * spacing_m, total, spacing_m)
        )
        selected.extend(int(indices[np.argmin(np.abs(arc - target))]) for target in targets)
    return np.asarray(sorted(set(selected)), dtype=np.int64)


class _CudaDderStep:
    def __init__(self, model: DderModel, particle_count: int, device: torch.device) -> None:
        self.model = model
        self.device = device
        shape = (particle_count, model.parameters.node_count, 3)
        self.q = torch.zeros(shape, dtype=torch.float32, device=device)
        self.q[:, :, 0] = torch.linspace(
            0.0,
            model.parameters.cable_length_m,
            model.parameters.node_count,
            device=device,
        )
        self.v = torch.zeros_like(self.q)
        self.boundary = self.q[:, (0, -1)].clone()
        self.dt = torch.full((particle_count,), 1.0 / 30.0, dtype=torch.float32, device=device)
        self.constants = model.runtime_constants(self.q)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: DderState | None = None
        if device.type == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.output = model.step_runtime(
                        DderState(self.q, self.v), self.boundary, self.dt, self.constants
                    )
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = model.step_runtime(
                    DderState(self.q, self.v), self.boundary, self.dt, self.constants
                )

    def __call__(
        self,
        state: DderState,
        boundary: torch.Tensor,
        dt_s: float,
    ) -> DderState:
        self.q.copy_(state.positions_m)
        self.v.copy_(state.velocities_m_s)
        self.boundary.copy_(boundary)
        self.dt.fill_(float(dt_s))
        if self.graph is None:
            return self.model.step_runtime(
                DderState(self.q, self.v), self.boundary, self.dt, self.constants
            )
        self.graph.replay()
        assert self.output is not None
        return self.output


class DderParticleFilter:
    """Causal single-cable PF whose transition is the identified DDER model."""

    def __init__(
        self,
        model: DderModel,
        settings: ParticleFilterSettings,
        artifact: DderArtifact,
    ) -> None:
        self.model = model
        self.settings = settings
        self.artifact = artifact
        self.device = torch.device(settings.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DDER particle filter requested CUDA, but CUDA is unavailable.")
        if model.maximum_stable_bending_stiffness(settings.maximum_dt_s) < (
            model.parameters.bending_stiffness_n_m2
        ):
            raise ValueError("The fitted DDER is unstable at the configured PF time step.")
        if model.maximum_stable_bending_damping(settings.maximum_dt_s) < (
            model.parameters.bending_damping_n_m2_s
        ):
            raise ValueError(
                "The fitted DDER curvature damping is unstable at the configured PF time step."
            )
        self.generator = torch.Generator(device=self.device).manual_seed(settings.random_seed)
        self.particle_count = settings.particle_count
        self.node_count = model.parameters.node_count
        self.positions: torch.Tensor | None = None
        self.velocities: torch.Tensor | None = None
        self.log_weights: torch.Tensor | None = None
        self.last_timestamp_ns: int | None = None
        self.stepper: _CudaDderStep | None = None

    @property
    def initialized(self) -> bool:
        return self.positions is not None

    def _frame_curve(
        self,
        metric_points_m: np.ndarray,
        metric_valid: np.ndarray,
        segment_ids: np.ndarray,
        endpoints_m: np.ndarray,
        endpoint_valid: np.ndarray,
        endpoint_sigma_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.count_nonzero(endpoint_valid) != 2:
            raise ValueError("Initialization requires both metric endpoints.")
        candidates = []
        for segment in np.unique(segment_ids[metric_valid]):
            selected = metric_valid & (segment_ids == segment)
            points = metric_points_m[selected]
            points = points[np.all(np.isfinite(points), axis=1)]
            if len(points) >= 2:
                candidates.append(points)
        if not candidates:
            raise ValueError("Initialization requires one sufficiently complete metric fragment.")
        points = max(candidates, key=len)
        padded = np.pad(points, ((2, 2), (0, 0)), mode="edge")
        points = sum(padded[offset : offset + len(points)] for offset in range(5)) / 5.0
        endpoints = endpoints_m[endpoint_valid]
        direct = np.linalg.norm(endpoints[0] - points[0]) + np.linalg.norm(endpoints[1] - points[-1])
        flipped = np.linalg.norm(endpoints[1] - points[0]) + np.linalg.norm(endpoints[0] - points[-1])
        direct_order = direct <= flipped
        ordered = endpoints if direct_order else endpoints[::-1]
        ordered_sigma = endpoint_sigma_m[endpoint_valid]
        if not direct_order:
            ordered_sigma = ordered_sigma[::-1].copy()
        points = np.vstack((ordered[0], points, ordered[1]))
        if np.linalg.norm(ordered[1] - ordered[0]) >= self.model.parameters.cable_length_m:
            raise ValueError("Observed endpoint separation exceeds cable length.")
        return _arc_resample(points, self.node_count), ordered.copy(), ordered_sigma

    def initialize(
        self,
        *,
        timestamp_ns: int,
        metric_points_m: np.ndarray,
        metric_valid: np.ndarray,
        segment_ids: np.ndarray,
        endpoints_m: np.ndarray,
        endpoint_valid: np.ndarray,
        metric_sigma_m: np.ndarray,
        endpoint_sigma_m: np.ndarray,
    ) -> ParticleFilterEstimate:
        curve, ordered_endpoints, ordered_endpoint_sigma = self._frame_curve(
            metric_points_m,
            metric_valid,
            segment_ids,
            endpoints_m,
            endpoint_valid,
            endpoint_sigma_m,
        )
        base = torch.as_tensor(
            curve,
            dtype=torch.float32,
            device=self.device,
        )
        q = base[None].expand(self.particle_count, -1, -1).clone()
        initial_sigma = float(np.median(metric_sigma_m[metric_valid]))
        q += initial_sigma * torch.randn(
            q.shape, generator=self.generator, device=self.device
        )
        endpoint_noise = torch.as_tensor(
            ordered_endpoint_sigma, dtype=q.dtype, device=q.device
        )[None, :, None] * torch.randn(
            (self.particle_count, 2, 3), generator=self.generator, device=self.device
        )
        observed = torch.as_tensor(ordered_endpoints, dtype=q.dtype, device=q.device)
        terminal = observed[None] + endpoint_noise
        boundary = terminal
        for _ in range(25):
            q = self.model.project_lengths(q, boundary)
            if float(self.model.maximum_segment_error_m(q).max().cpu()) <= 1.0e-6:
                break
        else:
            raise RuntimeError("PF initialization did not converge to the length constraint.")
        self.positions = q.detach()
        self.velocities = torch.zeros_like(q)
        self.log_weights = torch.full(
            (self.particle_count,), -math.log(self.particle_count), device=self.device
        )
        self.last_timestamp_ns = int(timestamp_ns)
        self.stepper = _CudaDderStep(self.model, self.particle_count, self.device)
        return self._estimate(
            residual_m=float("nan"), body_updated=False, endpoint_count=2,
            prediction_only=False, resampled=False, timing_ms=(0.0, 0.0, 0.0, 0.0),
        )

    def _sample_endpoint_boundary(
        self,
        endpoints: torch.Tensor,
        endpoint_valid: torch.Tensor,
        endpoint_sigma_m: torch.Tensor,
        dt_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the optimal Gaussian endpoint proposal and its evidence.

        Endpoint components form an unlabeled set.  The predictive likelihood
        marginalizes the two terminal assignments, and each particle samples
        its assignment from that posterior.  With no observation this reduces
        exactly to the learned constant-velocity acceleration prior.
        """

        assert self.positions is not None and self.velocities is not None
        prior_mean = (
            self.positions[:, (0, -1)]
            + dt_s * self.velocities[:, (0, -1)]
        )
        prior_sigma = 0.5 * dt_s * dt_s * (
            self.artifact.endpoint_acceleration_sigma_m_s2
        )
        prior_variance = prior_sigma * prior_sigma
        prior_normal = torch.randn(
            (self.particle_count, 2, 3), generator=self.generator, device=self.device
        )
        boundary = prior_mean + prior_sigma * prior_normal
        log_evidence = torch.zeros(
            self.particle_count, dtype=prior_mean.dtype, device=self.device
        )
        indices = torch.nonzero(endpoint_valid, as_tuple=False).squeeze(1)
        if len(indices) == 0:
            return boundary, log_evidence

        observations = endpoints[indices]
        observation_sigma = endpoint_sigma_m[indices]

        def log_predictive(point: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            variance = prior_variance + sigma.square()
            square = torch.sum((prior_mean - point[None, None]).square(), dim=-1)
            return -0.5 * (
                square / variance
                + 3.0 * math.log(2.0 * math.pi)
                + 3.0 * torch.log(variance)
            )

        if len(indices) == 1:
            assignment_log = log_predictive(
                observations[0], observation_sigma[0]
            )
            log_evidence = torch.logsumexp(assignment_log, dim=1) - math.log(2.0)
            probability_first = torch.softmax(assignment_log, dim=1)[:, 0]
            assigned_first = torch.rand(
                self.particle_count, generator=self.generator, device=self.device
            ) < probability_first
            ordered_observation = observations[0][None].expand(
                self.particle_count, -1
            )
            ordered_sigma = observation_sigma[0].expand(self.particle_count)
            assigned_endpoint = torch.where(
                assigned_first,
                torch.zeros_like(assigned_first, dtype=torch.long),
                torch.ones_like(assigned_first, dtype=torch.long),
            )
            particle = torch.arange(self.particle_count, device=self.device)
            variance = ordered_sigma.square()
            posterior_variance = prior_variance * variance / (
                prior_variance + variance
            )
            posterior_mean = (
                variance[:, None] * prior_mean[particle, assigned_endpoint]
                + prior_variance * ordered_observation
            ) / (prior_variance + variance)[:, None]
            posterior_normal = torch.randn(
                (self.particle_count, 3),
                generator=self.generator,
                device=self.device,
            )
            boundary[particle, assigned_endpoint] = posterior_mean + torch.sqrt(
                posterior_variance
            )[:, None] * posterior_normal
            return boundary, log_evidence

        if len(indices) != 2:
            raise ValueError("A cable frame can contain at most two endpoint observations.")
        first_log = log_predictive(observations[0], observation_sigma[0])
        second_log = log_predictive(observations[1], observation_sigma[1])
        direct_log = first_log[:, 0] + second_log[:, 1]
        reversed_log = second_log[:, 0] + first_log[:, 1]
        assignment_log = torch.stack((direct_log, reversed_log), dim=1)
        log_evidence = torch.logsumexp(assignment_log, dim=1) - math.log(2.0)
        direct = torch.rand(
            self.particle_count, generator=self.generator, device=self.device
        ) < torch.softmax(assignment_log, dim=1)[:, 0]
        ordered_observation = torch.stack(
            (
                torch.where(direct[:, None], observations[0], observations[1]),
                torch.where(direct[:, None], observations[1], observations[0]),
            ),
            dim=1,
        )
        ordered_sigma = torch.stack(
            (
                torch.where(direct, observation_sigma[0], observation_sigma[1]),
                torch.where(direct, observation_sigma[1], observation_sigma[0]),
            ),
            dim=1,
        )
        observation_variance = ordered_sigma.square()
        posterior_variance = prior_variance * observation_variance / (
            prior_variance + observation_variance
        )
        posterior_mean = (
            observation_variance[:, :, None] * prior_mean
            + prior_variance * ordered_observation
        ) / (prior_variance + observation_variance)[:, :, None]
        posterior_normal = torch.randn(
            boundary.shape, generator=self.generator, device=self.device
        )
        boundary = posterior_mean + torch.sqrt(posterior_variance)[:, :, None] * (
            posterior_normal
        )
        return boundary, log_evidence

    def _propagate(
        self,
        dt_s: float,
        endpoints: torch.Tensor,
        endpoint_valid: torch.Tensor,
        endpoint_sigma_m: torch.Tensor,
    ) -> torch.Tensor:
        assert self.positions is not None and self.velocities is not None and self.stepper is not None
        macrosteps = max(1, int(math.ceil(dt_s / self.settings.maximum_dt_s)))
        step_dt = dt_s / macrosteps
        start_boundary = self.positions[:, (0, -1)]
        final_boundary, endpoint_log_evidence = self._sample_endpoint_boundary(
            endpoints, endpoint_valid, endpoint_sigma_m, dt_s
        )
        for macrostep in range(macrosteps):
            acceleration = self.artifact.process_acceleration_sigma_m_s2 * torch.randn(
                self.velocities.shape, generator=self.generator, device=self.device
            )
            acceleration[:, (0, -1)] = 0.0
            self.velocities = self.velocities + step_dt * acceleration
            boundary = start_boundary + float(macrostep + 1) / macrosteps * (
                final_boundary - start_boundary
            )
            state = self.stepper(
                DderState(self.positions, self.velocities), boundary, step_dt
            )
            self.positions, self.velocities = state.positions_m, state.velocities_m_s
        return endpoint_log_evidence

    def _measurement_update(
        self,
        metric_points: torch.Tensor,
        selected: torch.Tensor,
        metric_sigma_m: torch.Tensor,
        endpoint_valid: torch.Tensor,
        endpoint_log_evidence: torch.Tensor,
    ) -> tuple[bool, int, float]:
        assert self.positions is not None and self.log_weights is not None
        nll = -endpoint_log_evidence
        body_updated = len(selected) > 0
        residual_m = float("nan")
        if body_updated:
            observed = metric_points[selected]
            coordinate = torch.linspace(
                0.0, self.node_count - 1.0, 2 * self.node_count - 1,
                device=self.device,
            )
            low = torch.floor(coordinate).long()
            high = torch.clamp(low + 1, max=self.node_count - 1)
            fraction = (coordinate - low)[None, :, None]
            curve = (1.0 - fraction) * self.positions[:, low] + fraction * self.positions[:, high]
            distance = torch.cdist(
                observed[None].expand(self.particle_count, -1, -1), curve
            ).amin(dim=2)
            sigma = metric_sigma_m[selected]
            nll += _student_t_negative_log_likelihood(
                distance / sigma[None],
                self.artifact.student_t_degrees_of_freedom,
            ).sum(dim=1)
            weights_before = torch.softmax(self.log_weights, dim=0)
            residual_m = float(
                torch.sqrt(torch.sum(weights_before * distance.square().mean(dim=1))).cpu()
            )
        # The endpoint proposal is conditioned on the measurement, so its
        # importance correction is the predictive endpoint evidence computed by
        # _sample_endpoint_boundary. This is not a second measurement score.
        endpoint_count = int(torch.count_nonzero(endpoint_valid))
        if body_updated or endpoint_count > 0:
            updated = self.log_weights - nll
            self.log_weights = updated - torch.logsumexp(updated, dim=0)
        return body_updated, endpoint_count, residual_m

    def _resample_if_needed(self) -> tuple[float, bool]:
        assert self.positions is not None and self.velocities is not None and self.log_weights is not None
        weights = torch.softmax(self.log_weights, dim=0)
        ess = float(torch.reciprocal(torch.sum(weights.square())).cpu())
        if ess >= self.settings.ess_resample_fraction * self.particle_count:
            return ess, False
        positions = (
            torch.rand((), generator=self.generator, device=self.device)
            + torch.arange(self.particle_count, device=self.device)
        ) / self.particle_count
        indices = torch.searchsorted(torch.cumsum(weights, dim=0), positions, right=False)
        self.positions = self.positions[indices]
        self.velocities = self.velocities[indices]
        self.log_weights = torch.full_like(self.log_weights, -math.log(self.particle_count))
        return ess, True

    def _estimate(
        self,
        *,
        residual_m: float,
        body_updated: bool,
        endpoint_count: int,
        prediction_only: bool,
        resampled: bool,
        timing_ms: tuple[float, float, float, float],
    ) -> ParticleFilterEstimate:
        assert self.positions is not None and self.velocities is not None and self.log_weights is not None
        weights = torch.softmax(self.log_weights, dim=0)
        mean = torch.sum(weights[:, None, None] * self.positions, dim=0)
        delta = self.positions - mean[None]
        covariance = torch.einsum("p,pni,pnj->nij", weights, delta, delta)
        index = int(torch.argmax(weights))
        ess = float(torch.reciprocal(torch.sum(weights.square())).cpu())
        return ParticleFilterEstimate(
            self.positions[index].detach().cpu().numpy(),
            self.velocities[index].detach().cpu().numpy(),
            mean.detach().cpu().numpy(),
            covariance.detach().cpu().numpy(),
            weights.detach().cpu().numpy(),
            ess,
            residual_m,
            body_updated,
            endpoint_count,
            prediction_only,
            resampled,
            timing_ms,
        )

    def update(
        self,
        *,
        timestamp_ns: int,
        metric_points_m: np.ndarray,
        metric_valid: np.ndarray,
        segment_ids: np.ndarray,
        metric_sigma_m: np.ndarray,
        endpoints_m: np.ndarray,
        endpoint_valid: np.ndarray,
        endpoint_sigma_m: np.ndarray,
    ) -> ParticleFilterEstimate:
        if not self.initialized or self.last_timestamp_ns is None:
            raise RuntimeError("Initialize the DDER particle filter before update().")
        dt = (int(timestamp_ns) - self.last_timestamp_ns) * 1.0e-9
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("PF timestamps must be strictly increasing.")
        if dt > self.settings.maximum_gap_s:
            raise ValueError(f"PF frame gap {dt:.3f}s exceeds the configured maximum.")
        metric_points = torch.as_tensor(metric_points_m, dtype=torch.float32, device=self.device)
        selected = torch.as_tensor(
            _segment_spaced_indices(
                metric_points_m,
                metric_valid,
                segment_ids,
                self.model.parameters.segment_length_m,
            ),
            dtype=torch.long,
            device=self.device,
        )
        metric_sigma = torch.as_tensor(metric_sigma_m, dtype=torch.float32, device=self.device)
        endpoints = torch.as_tensor(endpoints_m, dtype=torch.float32, device=self.device)
        endpoint_mask = torch.as_tensor(endpoint_valid, dtype=torch.bool, device=self.device)
        endpoint_sigma = torch.as_tensor(endpoint_sigma_m, dtype=torch.float32, device=self.device)
        start = time.perf_counter()
        boundary_done = time.perf_counter()
        endpoint_log_evidence = self._propagate(
            dt, endpoints, endpoint_mask, endpoint_sigma
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        transition_done = time.perf_counter()
        body_updated, endpoint_count, residual = self._measurement_update(
            metric_points,
            selected,
            metric_sigma,
            endpoint_mask,
            endpoint_log_evidence,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        measurement_done = time.perf_counter()
        # Preserve the weighted posterior in the returned estimate. Resampling
        # prepares only the next frame's particle population.
        estimate = self._estimate(
            residual_m=residual,
            body_updated=body_updated,
            endpoint_count=endpoint_count,
            prediction_only=not body_updated and endpoint_count == 0,
            resampled=False,
            timing_ms=(0.0, 0.0, 0.0, 0.0),
        )
        _ess, resampled = self._resample_if_needed()
        end = time.perf_counter()
        self.last_timestamp_ns = int(timestamp_ns)
        return ParticleFilterEstimate(
            estimate.positions_m,
            estimate.velocities_m_s,
            estimate.posterior_mean_m,
            estimate.node_covariance_m2,
            estimate.weights,
            estimate.ess,
            estimate.residual_m,
            estimate.body_updated,
            estimate.endpoint_count,
            estimate.prediction_only,
            resampled,
            (
                1000.0 * (boundary_done - start),
                1000.0 * (transition_done - boundary_done),
                1000.0 * (measurement_done - transition_done),
                1000.0 * (end - start),
            ),
        )
