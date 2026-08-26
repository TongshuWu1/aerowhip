"""Moving-history DDER observer from root and free-tip position measurements.

The observer keeps the complete one-attached DDER state, but parameterizes
only a smooth, low-dimensional correction to its recursive prior.  Candidate
past states are projected onto the same inextensible DDER manifold and replayed
forward under the root motion that actually occurred.  No plant interior state
or future observation is accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np
import torch

from cable_twin.shared.dder import (
    DderState,
    START_PINNED_FREE_END,
    momentum_project_lengths,
    momentum_project_velocities,
)

from .model import CableModelSnapshot


@dataclass(frozen=True, slots=True)
class EndpointHistoryObservation:
    """One causal sparse observation; deliberately contains no cable interior."""

    timestamp_s: float
    root_position_m: np.ndarray
    root_velocity_m_s: np.ndarray
    tip_position_m: np.ndarray

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s):
            raise ValueError("Observer timestamp must be finite.")
        for name in ("root_position_m", "root_velocity_m_s", "tip_position_m"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain three finite values.")
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class HistoryObserverSettings:
    """Numerical settings for one local moving-history correction."""

    history_duration_s: float = 0.30
    spatial_mode_count: int = 4
    position_correction_scale_m: float = 0.025
    velocity_correction_scale_m_s: float = 0.25
    finite_difference_step: float = 0.04
    prior_weight: float = 1.0e-4
    lm_damping: float = 1.0e-4
    maximum_iterations: int = 2
    maximum_step_norm: float = 2.0
    maximum_component_magnitude: float = 2.5
    line_search_steps: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125)
    minimum_objective_improvement: float = 1.0e-10
    singular_value_relative_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        positive = (
            self.history_duration_s,
            self.position_correction_scale_m,
            self.velocity_correction_scale_m_s,
            self.finite_difference_step,
            self.lm_damping,
            self.maximum_step_norm,
            self.maximum_component_magnitude,
            self.minimum_objective_improvement,
            self.singular_value_relative_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("History-observer scales must be finite and positive.")
        if not math.isfinite(self.prior_weight) or self.prior_weight < 0.0:
            raise ValueError("History-observer prior weight must be non-negative.")
        if self.spatial_mode_count < 1:
            raise ValueError("History observer requires at least one spatial mode.")
        if self.maximum_iterations not in (1, 2):
            raise ValueError("History observer supports one or two GN/LM iterations.")
        if not self.line_search_steps or any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in self.line_search_steps
        ):
            raise ValueError("Observer line-search steps must lie in (0, 1].")


@dataclass(frozen=True, slots=True)
class HistoryObserverUpdate:
    """Diagnostics from one causal observer correction."""

    ready: bool
    accepted: bool
    reason: str
    frame_count: int
    history_span_s: float
    iterations: int
    measurement_rmse_before_m: float
    measurement_rmse_after_m: float
    objective_before: float
    objective_after: float
    correction_norm: float
    maximum_correction_component: float
    singular_values: tuple[float, ...]
    numerical_rank: int
    condition_number: float
    selected_line_search_alphas: tuple[float, ...]
    trust_region_saturations: int
    component_bound_saturations: int
    finite_difference_rollout_time_s: float
    line_search_rollout_time_s: float
    linear_solve_time_s: float
    total_time_s: float


@dataclass(frozen=True, slots=True)
class _ReplayBatch:
    positions_m: torch.Tensor
    velocities_m_s: torch.Tensor
    tip_positions_m: np.ndarray
    elapsed_s: float


class _FixedHistoryReplay:
    """Fixed-shape, GPU-resident observer shooting operator.

    The operation is mathematically identical to the former Python replay:
    project one smooth correction at the window start, then apply the existing
    DDER transition under every measured root position.  CUDA captures the
    complete history so one replay replaces the per-frame Python launch loop.
    Full state histories remain on device; only the small free-tip history is
    transferred to the CPU for the unchanged NumPy GN/LM calculation.
    """

    def __init__(
        self,
        model: CableModelSnapshot,
        basis: torch.Tensor,
        *,
        batch_size: int,
        frame_count: int,
        position_scale_m: float,
        velocity_scale_m_s: float,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.basis = basis
        self.batch_size = int(batch_size)
        self.frame_count = int(frame_count)
        self.mode_count = int(basis.shape[1])
        self.position_scale_m = float(position_scale_m)
        self.velocity_scale_m_s = float(velocity_scale_m_s)
        self.dtype = dtype
        self.device = device
        node_count = model.node_count
        dimension = 2 * self.mode_count * 3
        self.q_values = torch.zeros(
            (batch_size, dimension), dtype=dtype, device=device
        )
        self.anchor_positions = torch.zeros(
            (1, node_count, 3), dtype=dtype, device=device
        )
        self.anchor_velocities = torch.zeros_like(self.anchor_positions)
        self.root_positions = torch.zeros(
            (frame_count, 1, 3), dtype=dtype, device=device
        )
        self.root_velocities = torch.zeros_like(self.root_positions)
        self.dt_s = torch.full(
            (max(frame_count - 1, 1),), 0.02, dtype=dtype, device=device
        )
        reference = self.anchor_positions.expand(batch_size, -1, -1)
        self.constants = model.model.runtime_constants(reference)
        self.output: tuple[torch.Tensor, torch.Tensor] | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

    def _capture(self) -> None:
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.output = self._execute()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.output = self._execute()

    def _project_initial(self, state: DderState) -> DderState:
        batch = self.batch_size
        boundary_position = self.root_positions[0:1].expand(batch, -1, -1)
        boundary_velocity = self.root_velocities[0:1].expand(batch, -1, -1)
        position = state.positions_m.clone()
        velocity = state.velocities_m_s.clone()
        position[:, 0] = boundary_position[:, 0]
        position = momentum_project_lengths(
            position,
            self.constants.rest_lengths_m,
            self.constants.masses_kg,
            boundary_position,
            iterations=self.model.model.parameters.constraint_iterations,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        velocity[:, 0] = boundary_velocity[:, 0]
        velocity = momentum_project_velocities(
            position,
            velocity,
            self.constants.masses_kg,
            boundary_velocity,
            validate=False,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        return DderState(position, velocity)

    def _execute(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch_size
        modes = self.mode_count
        position_coefficients = self.q_values[:, : 3 * modes].reshape(
            batch, modes, 3
        )
        velocity_coefficients = self.q_values[:, 3 * modes :].reshape(
            batch, modes, 3
        )
        position_delta = self.position_scale_m * torch.einsum(
            "nm,bmc->bnc", self.basis, position_coefficients
        )
        velocity_delta = self.velocity_scale_m_s * torch.einsum(
            "nm,bmc->bnc", self.basis, velocity_coefficients
        )
        state = self._project_initial(
            DderState(
                self.anchor_positions.expand(batch, -1, -1).clone()
                + position_delta,
                self.anchor_velocities.expand(batch, -1, -1).clone()
                + velocity_delta,
            )
        )
        positions = [state.positions_m]
        velocities = [state.velocities_m_s]
        for index in range(1, self.frame_count):
            boundary = self.root_positions[index : index + 1].expand(
                batch, -1, -1
            )
            dt = self.dt_s[index - 1 : index].expand(batch)
            state = self.model.model.step_runtime(
                state,
                boundary,
                dt,
                self.constants,
                iterative_damping=True,
                pinned_endpoints=START_PINNED_FREE_END,
            )
            positions.append(state.positions_m)
            velocities.append(state.velocities_m_s)
        return torch.stack(positions, dim=1), torch.stack(velocities, dim=1)

    def __call__(
        self,
        q_values: torch.Tensor,
        anchor: DderState,
        root_positions: torch.Tensor,
        root_velocities: torch.Tensor,
        dt_s: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.q_values.copy_(q_values)
        self.anchor_positions.copy_(anchor.positions_m)
        self.anchor_velocities.copy_(anchor.velocities_m_s)
        self.root_positions.copy_(root_positions)
        self.root_velocities.copy_(root_velocities)
        if self.frame_count > 1:
            self.dt_s[: self.frame_count - 1].copy_(dt_s)
        if self.device.type == "cuda" and self.graph is None:
            self._capture()
        if self.graph is None:
            self.output = self._execute()
        else:
            self.graph.replay()
        assert self.output is not None
        return self.output


class _FixedObserverStep:
    """Captured batch-one DDER step used by continuous recursive propagation."""

    def __init__(
        self,
        model: CableModelSnapshot,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.device = device
        self.positions = torch.zeros(
            (1, model.node_count, 3), dtype=dtype, device=device
        )
        self.velocities = torch.zeros_like(self.positions)
        self.boundary = torch.zeros((1, 1, 3), dtype=dtype, device=device)
        self.dt_s = torch.full((1,), 0.02, dtype=dtype, device=device)
        self.constants = model.model.runtime_constants(self.positions)
        self.output: DderState | None = None
        self.graph: torch.cuda.CUDAGraph | None = None

    def _execute(self) -> DderState:
        return self.model.model.step_runtime(
            DderState(self.positions, self.velocities),
            self.boundary,
            self.dt_s,
            self.constants,
            iterative_damping=True,
            pinned_endpoints=START_PINNED_FREE_END,
        )

    def _capture(self) -> None:
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.output = self._execute()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.output = self._execute()

    def __call__(
        self,
        state: DderState,
        boundary: torch.Tensor,
        dt_s: float,
    ) -> DderState:
        self.positions.copy_(state.positions_m)
        self.velocities.copy_(state.velocities_m_s)
        self.boundary.copy_(boundary)
        self.dt_s.fill_(float(dt_s))
        if self.graph is None:
            self._capture()
        self.graph.replay()
        assert self.output is not None
        return self.output


def spatial_correction_basis(
    node_count: int,
    mode_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return ``s, sin(pi*s), ...`` with an exactly fixed root."""

    if node_count < 3 or mode_count < 1:
        raise ValueError("Correction basis requires at least three nodes and one mode.")
    coordinate = torch.linspace(0.0, 1.0, node_count, dtype=dtype, device=device)
    modes = [coordinate]
    modes.extend(
        torch.sin(float(index) * math.pi * coordinate)
        for index in range(1, mode_count)
    )
    basis = torch.stack(modes, dim=1)
    basis[0] = 0.0
    return basis


class DderHistoryObserver:
    """Recursive moving-history observer with batched finite-difference GN/LM."""

    def __init__(
        self,
        model: CableModelSnapshot,
        initial_state: DderState,
        initial_observation: EndpointHistoryObservation,
        settings: HistoryObserverSettings | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.settings = settings or HistoryObserverSettings()
        self.device = (
            torch.device(device)
            if device is not None
            else initial_state.positions_m.device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA history observer selected but unavailable.")
        self.dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        if model.node_count != initial_state.positions_m.shape[1]:
            raise ValueError("Observer model and initial state node counts differ.")
        if initial_state.positions_m.shape != (1, model.node_count, 3):
            raise ValueError("History observer expects one unbatched cable state.")
        if self.settings.spatial_mode_count > model.node_count:
            raise ValueError("Observer mode count cannot exceed cable node count.")
        self._basis = spatial_correction_basis(
            model.node_count,
            self.settings.spatial_mode_count,
            dtype=self.dtype,
            device=self.device,
        )
        self._observations: list[EndpointHistoryObservation] = [initial_observation]
        self._root_positions: list[torch.Tensor] = [
            self._vector_tensor(initial_observation.root_position_m)
        ]
        self._root_velocities: list[torch.Tensor] = [
            self._vector_tensor(initial_observation.root_velocity_m_s)
        ]
        initial = DderState(
            initial_state.positions_m.to(device=self.device, dtype=self.dtype).clone(),
            initial_state.velocities_m_s.to(device=self.device, dtype=self.dtype).clone(),
        )
        self._states: list[DderState] = [
            self._project_state(
                initial,
                self._root_positions[0],
                self._root_velocities[0],
                batch_size=1,
            )
        ]
        self._runtime_constants = self.model.model.runtime_constants(
            self._states[0].positions_m
        )
        self._replay_operators: dict[tuple[int, int], _FixedHistoryReplay] = {}
        self._recursive_step = (
            _FixedObserverStep(
                self.model,
                dtype=self.dtype,
                device=self.device,
            )
            if self.device.type == "cuda"
            else None
        )
        self.updates: list[HistoryObserverUpdate] = []

    @property
    def correction_dimension(self) -> int:
        return 2 * self.settings.spatial_mode_count * 3

    @property
    def frame_count(self) -> int:
        return len(self._observations)

    @property
    def history_span_s(self) -> float:
        if len(self._observations) < 2:
            return 0.0
        return (
            self._observations[-1].timestamp_s
            - self._observations[0].timestamp_s
        )

    @property
    def ready(self) -> bool:
        return (
            len(self._observations) >= 3
            and self.history_span_s
            >= self.settings.history_duration_s - 1.0e-8
        )

    @property
    def current_state(self) -> DderState:
        state = self._states[-1]
        return DderState(state.positions_m.clone(), state.velocities_m_s.clone())

    @property
    def observations(self) -> tuple[EndpointHistoryObservation, ...]:
        return tuple(self._observations)

    def prewarm(self, sample_interval_s: float) -> None:
        """Capture fixed observer workloads before entering the control loop.

        This changes only when CUDA setup cost is paid.  The captured operators
        receive the real anchor, root history, and correction candidates on
        every later call.
        """

        interval = float(sample_interval_s)
        if self.device.type != "cuda":
            return
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError("Observer prewarm interval must be positive.")
        assert self._recursive_step is not None
        recursive_output = self._recursive_step(
            self._states[0], self._root_positions[0], interval
        )
        # Complete the replay before its static output buffer is reused.
        recursive_output.positions_m.clone()
        recursive_output.velocities_m_s.clone()
        dimension = self.correction_dimension
        nominal_frame_count = (
            int(math.ceil(self.settings.history_duration_s / interval)) + 1
        )
        # Float32 simulation timestamps can make the shortest retained suffix
        # alternate between the nominal count and one extra frame.  Capture
        # both fixed shapes during setup so neither first occurrence stalls a
        # live observer update.
        for frame_count in (nominal_frame_count, nominal_frame_count + 1):
            root_positions = self._root_positions[0].expand(
                frame_count, -1, -1
            ).clone()
            root_velocities = self._root_velocities[0].expand(
                frame_count, -1, -1
            ).clone()
            dt_s = torch.full(
                (frame_count - 1,), interval, dtype=self.dtype, device=self.device
            )
            for batch_size in (
                1 + 2 * dimension,
                len(self.settings.line_search_steps),
            ):
                q_values = torch.zeros(
                    (batch_size, dimension), dtype=self.dtype, device=self.device
                )
                operator = _FixedHistoryReplay(
                    self.model,
                    self._basis,
                    batch_size=batch_size,
                    frame_count=frame_count,
                    position_scale_m=self.settings.position_correction_scale_m,
                    velocity_scale_m_s=self.settings.velocity_correction_scale_m_s,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._replay_operators[(batch_size, frame_count)] = operator
                operator(
                    q_values,
                    self._states[0],
                    root_positions,
                    root_velocities,
                    dt_s,
                )
        torch.cuda.synchronize(self.device)

    def _vector_tensor(self, value: np.ndarray) -> torch.Tensor:
        return torch.tensor(
            np.asarray(value), dtype=self.dtype, device=self.device
        ).reshape(1, 1, 3)

    def _project_state(
        self,
        state: DderState,
        root_position: torch.Tensor,
        root_velocity: torch.Tensor,
        *,
        batch_size: int,
    ) -> DderState:
        root_position = root_position.expand(batch_size, -1, -1)
        root_velocity = root_velocity.expand(batch_size, -1, -1)
        position = state.positions_m.clone()
        velocity = state.velocities_m_s.clone()
        position[:, 0] = root_position[:, 0]
        position = self.model.model.project_lengths(
            position,
            root_position,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        velocity[:, 0] = root_velocity[:, 0]
        velocity = self.model.model.project_velocities(
            position,
            velocity,
            root_velocity,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        return DderState(position, velocity)

    def _step(
        self,
        state: DderState,
        root_position: torch.Tensor,
        dt_s: float,
    ) -> DderState:
        batch_size = state.positions_m.shape[0]
        boundary = root_position.expand(batch_size, -1, -1)
        dt = torch.full(
            (batch_size,), float(dt_s), dtype=self.dtype, device=self.device
        )
        constants = self._runtime_constants
        if constants.bending_stiffness_n_m2.shape[0] != batch_size:
            constants = self.model.model.runtime_constants(state.positions_m)
        return self.model.model.step_runtime(
            state,
            boundary,
            dt,
            constants,
            iterative_damping=True,
            pinned_endpoints=START_PINNED_FREE_END,
        )

    def ingest(self, observation: EndpointHistoryObservation) -> DderState:
        """Causally propagate the recursive prior to one newly arrived sample."""

        previous = self._observations[-1]
        dt = observation.timestamp_s - previous.timestamp_s
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("Observer samples must have strictly increasing timestamps.")
        root_position = self._vector_tensor(observation.root_position_m)
        root_velocity = self._vector_tensor(observation.root_velocity_m_s)
        if self._recursive_step is None:
            propagated = self._step(self._states[-1], root_position, dt)
        else:
            static_output = self._recursive_step(
                self._states[-1], root_position, dt
            )
            # The captured operator owns reusable output buffers.  History
            # states must remain immutable across later graph replays.
            propagated = DderState(
                static_output.positions_m.clone(),
                static_output.velocities_m_s.clone(),
            )
        self._observations.append(observation)
        self._root_positions.append(root_position)
        self._root_velocities.append(root_velocity)
        self._states.append(propagated)
        # Retain the shortest causal suffix that still spans the requested
        # duration.  Testing the *next* sample avoids a floating-point boundary
        # case that could otherwise leave only 0.28 s for a requested 0.30 s
        # window at 50 Hz.
        tolerance = 1.0e-8
        while len(self._observations) > 2 and (
            observation.timestamp_s - self._observations[1].timestamp_s
            >= self.settings.history_duration_s - tolerance
        ):
            self._observations.pop(0)
            self._root_positions.pop(0)
            self._root_velocities.pop(0)
            self._states.pop(0)
        return self.current_state

    def _correction(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.shape[0]
        modes = self.settings.spatial_mode_count
        position_coefficients = q[:, : 3 * modes].reshape(batch_size, modes, 3)
        velocity_coefficients = q[:, 3 * modes :].reshape(batch_size, modes, 3)
        position = self.settings.position_correction_scale_m * torch.einsum(
            "nm,bmc->bnc", self._basis, position_coefficients
        )
        velocity = self.settings.velocity_correction_scale_m_s * torch.einsum(
            "nm,bmc->bnc", self._basis, velocity_coefficients
        )
        return position, velocity

    def _history_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_positions = torch.cat(self._root_positions, dim=0)
        root_velocities = torch.cat(self._root_velocities, dim=0)
        timestamps = np.fromiter(
            (observation.timestamp_s for observation in self._observations),
            dtype=np.float64,
            count=self.frame_count,
        )
        dt_s = torch.as_tensor(
            np.diff(timestamps), dtype=self.dtype, device=self.device
        )
        return root_positions, root_velocities, dt_s

    def _replay_operator(self, batch_size: int) -> _FixedHistoryReplay:
        key = (int(batch_size), self.frame_count)
        operator = self._replay_operators.get(key)
        if operator is None:
            operator = _FixedHistoryReplay(
                self.model,
                self._basis,
                batch_size=batch_size,
                frame_count=self.frame_count,
                position_scale_m=self.settings.position_correction_scale_m,
                velocity_scale_m_s=self.settings.velocity_correction_scale_m_s,
                dtype=self.dtype,
                device=self.device,
            )
            self._replay_operators[key] = operator
        return operator

    def _replay(
        self,
        q_values: np.ndarray,
        history_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> _ReplayBatch:
        q_array = np.asarray(q_values, dtype=np.float64)
        if q_array.ndim != 2 or q_array.shape[1] != self.correction_dimension:
            raise ValueError("Observer correction batch has invalid shape.")
        if not np.all(np.isfinite(q_array)):
            raise ValueError("Observer corrections must be finite.")
        batch_size = len(q_array)
        q_tensor = torch.as_tensor(q_array, dtype=self.dtype, device=self.device)
        root_positions, root_velocities, dt_s = history_tensors
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            position, velocity = self._replay_operator(batch_size)(
                q_tensor,
                self._states[0],
                root_positions,
                root_velocities,
                dt_s,
            )
            end_event.record()
            # This is the sole replay result needed by the CPU GN/LM path.
            tip_positions = position[:, :, -1].detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
        else:
            started = time.perf_counter()
            position, velocity = self._replay_operator(batch_size)(
                q_tensor,
                self._states[0],
                root_positions,
                root_velocities,
                dt_s,
            )
            tip_positions = position[:, :, -1].detach().numpy().astype(
                np.float64, copy=False
            )
            elapsed = time.perf_counter() - started
        return _ReplayBatch(
            positions_m=position,
            velocities_m_s=velocity,
            tip_positions_m=tip_positions,
            elapsed_s=elapsed,
        )

    def _measurement_residual(self, replay: _ReplayBatch) -> np.ndarray:
        measured = np.stack(
            [observation.tip_position_m for observation in self._observations]
        )
        return replay.tip_positions_m - measured[None]

    def _objective(
        self,
        q_values: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray:
        measurement = np.mean(np.square(residual), axis=(1, 2))
        prior = self.settings.prior_weight * np.sum(np.square(q_values), axis=1)
        return measurement + prior

    def _not_ready_update(self) -> HistoryObserverUpdate:
        return HistoryObserverUpdate(
            ready=False,
            accepted=False,
            reason="collecting_history",
            frame_count=self.frame_count,
            history_span_s=self.history_span_s,
            iterations=0,
            measurement_rmse_before_m=math.nan,
            measurement_rmse_after_m=math.nan,
            objective_before=math.nan,
            objective_after=math.nan,
            correction_norm=0.0,
            maximum_correction_component=0.0,
            singular_values=(),
            numerical_rank=0,
            condition_number=math.inf,
            selected_line_search_alphas=(),
            trust_region_saturations=0,
            component_bound_saturations=0,
            finite_difference_rollout_time_s=0.0,
            line_search_rollout_time_s=0.0,
            linear_solve_time_s=0.0,
            total_time_s=0.0,
        )

    def correct(self) -> HistoryObserverUpdate:
        """Correct the window-start prior and publish its replayed final state."""

        if not self.ready:
            update = self._not_ready_update()
            self.updates.append(update)
            return update
        total_started = time.perf_counter()
        dimension = self.correction_dimension
        q = np.zeros(dimension, dtype=np.float64)
        finite_difference_time = 0.0
        line_search_time = 0.0
        linear_solve_time = 0.0
        accepted_iterations = 0
        last_singular_values = np.empty(0, dtype=np.float64)
        last_rank = 0
        last_condition = math.inf
        selected_alphas: list[float] = []
        trust_region_saturations = 0
        component_bound_saturations = 0
        history_tensors = self._history_tensors()
        initial_rmse = math.nan
        current_objective = math.inf
        initial_objective = math.nan
        final_rmse = math.nan
        accepted_history: tuple[torch.Tensor, torch.Tensor] | None = None

        for _iteration in range(self.settings.maximum_iterations):
            candidates = np.repeat(q[None], 1 + 2 * dimension, axis=0)
            indices = np.arange(dimension)
            candidates[1 + 2 * indices, indices] += self.settings.finite_difference_step
            candidates[2 + 2 * indices, indices] -= self.settings.finite_difference_step
            replay = self._replay(candidates, history_tensors)
            finite_difference_time += replay.elapsed_s
            residual = self._measurement_residual(replay)
            nominal_objective = float(self._objective(q[None], residual[:1])[0])
            if _iteration == 0:
                initial_rmse = float(
                    np.sqrt(np.mean(np.square(residual[0])))
                )
                current_objective = nominal_objective
                initial_objective = nominal_objective
            residual_size = residual.shape[1] * residual.shape[2]
            normalization = math.sqrt(float(residual_size))
            nominal_vector = residual[0].reshape(-1) / normalization
            plus = residual[1::2].reshape(dimension, -1) / normalization
            minus = residual[2::2].reshape(dimension, -1) / normalization
            jacobian = (
                (plus - minus) / (2.0 * self.settings.finite_difference_step)
            ).T
            last_singular_values = np.linalg.svd(jacobian, compute_uv=False)
            if len(last_singular_values) and last_singular_values[0] > 0.0:
                threshold = (
                    self.settings.singular_value_relative_tolerance
                    * last_singular_values[0]
                )
                last_rank = int(np.sum(last_singular_values > threshold))
                last_condition = float(
                    last_singular_values[0]
                    / max(last_singular_values[-1], np.finfo(np.float64).eps)
                )
            else:
                last_rank = 0
                last_condition = math.inf

            solve_started = time.perf_counter()
            hessian = jacobian.T @ jacobian + (
                self.settings.prior_weight + self.settings.lm_damping
            ) * np.eye(dimension)
            gradient = (
                jacobian.T @ nominal_vector + self.settings.prior_weight * q
            )
            try:
                step = -np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            component_bound_saturations += int(
                np.any(np.abs(step) > self.settings.maximum_component_magnitude)
            )
            step = np.clip(
                step,
                -self.settings.maximum_component_magnitude,
                self.settings.maximum_component_magnitude,
            )
            step_norm = float(np.linalg.norm(step))
            if step_norm > self.settings.maximum_step_norm:
                trust_region_saturations += 1
                step *= self.settings.maximum_step_norm / step_norm
            linear_solve_time += time.perf_counter() - solve_started

            line_candidates = np.stack(
                [
                    np.clip(
                        q + alpha * step,
                        -self.settings.maximum_component_magnitude,
                        self.settings.maximum_component_magnitude,
                    )
                    for alpha in self.settings.line_search_steps
                ]
            )
            line_replay = self._replay(line_candidates, history_tensors)
            line_search_time += line_replay.elapsed_s
            line_residual = self._measurement_residual(line_replay)
            line_objective = self._objective(line_candidates, line_residual)
            selected = int(np.argmin(line_objective))
            if (
                float(line_objective[selected])
                >= current_objective - self.settings.minimum_objective_improvement
            ):
                break
            q = line_candidates[selected]
            current_objective = float(line_objective[selected])
            selected_alphas.append(float(self.settings.line_search_steps[selected]))
            final_rmse = float(
                np.sqrt(np.mean(np.square(line_residual[selected])))
            )
            # Preserve the accepted device-resident state before a later
            # line-search replay reuses the fixed CUDA-graph output buffers.
            accepted_history = (
                line_replay.positions_m[selected].clone(),
                line_replay.velocities_m_s[selected].clone(),
            )
            accepted_iterations += 1

        accepted = accepted_iterations > 0
        if accepted:
            assert accepted_history is not None
            accepted_positions, accepted_velocities = accepted_history
            self._states = [
                DderState(
                    accepted_positions[index : index + 1].clone(),
                    accepted_velocities[index : index + 1].clone(),
                )
                for index in range(self.frame_count)
            ]
        total = time.perf_counter() - total_started
        update = HistoryObserverUpdate(
            ready=True,
            accepted=accepted,
            reason="accepted" if accepted else "no_improving_correction",
            frame_count=self.frame_count,
            history_span_s=self.history_span_s,
            iterations=accepted_iterations,
            measurement_rmse_before_m=initial_rmse,
            measurement_rmse_after_m=final_rmse if accepted else initial_rmse,
            objective_before=initial_objective,
            objective_after=current_objective,
            correction_norm=float(np.linalg.norm(q)),
            maximum_correction_component=float(np.max(np.abs(q))),
            singular_values=tuple(float(value) for value in last_singular_values),
            numerical_rank=last_rank,
            condition_number=last_condition,
            selected_line_search_alphas=tuple(selected_alphas),
            trust_region_saturations=trust_region_saturations,
            component_bound_saturations=component_bound_saturations,
            finite_difference_rollout_time_s=finite_difference_time,
            line_search_rollout_time_s=line_search_time,
            linear_solve_time_s=linear_solve_time,
            total_time_s=total,
        )
        self.updates.append(update)
        return update
