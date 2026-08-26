"""Differentiable closed-loop-drone translation and free-tip DER simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Callable

import numpy as np
import torch

from cable_twin.shared.dder import DderModel, DderState, START_PINNED_FREE_END

from .model import CableModelSnapshot


CancellationCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class RuntimeAcceleration:
    """Resolved forward-runtime implementation for one cable discretization.

    Every CUDA inference rollout can use the fixed-shape full-horizon graph.
    The experimentally optimized 11-node topology additionally uses the fused
    damping and projection kernels.  Keeping these tiers explicit prevents the
    online UI from silently falling back to the old per-step Python path.
    """

    captured_full_horizon: bool
    fused_cost: bool
    fused_mechanics: bool
    node_count: int

    @property
    def tier(self) -> str:
        if self.fused_mechanics:
            return "maximum (captured horizon + fused 11-node mechanics)"
        if self.captured_full_horizon and self.fused_cost:
            return "captured CUDA (arbitrary-node mechanics + fused cost)"
        return "reference"

    @property
    def online_ready(self) -> bool:
        return self.captured_full_horizon and self.fused_cost


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    """Numerical and low-level closed-loop drone assumptions."""

    horizon_s: float = 1.5
    simulation_dt_s: float = 0.01
    control_interval_s: float = 0.10
    attachment_drop_m: float = 0.10
    maximum_acceleration_m_s2: float = 6.0
    maximum_speed_m_s: float = 3.0

    def __post_init__(self) -> None:
        values = (
            self.horizon_s,
            self.simulation_dt_s,
            self.control_interval_s,
            self.attachment_drop_m,
            self.maximum_acceleration_m_s2,
            self.maximum_speed_m_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Simulation times, limits, and attachment drop must be positive.")
        ratio = self.control_interval_s / self.simulation_dt_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("Control interval must be an integer multiple of simulation dt.")
        horizon_ratio = self.horizon_s / self.control_interval_s
        if not math.isclose(
            horizon_ratio, round(horizon_ratio), rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValueError("Horizon must be an integer multiple of control interval.")

    @property
    def steps_per_control(self) -> int:
        return int(round(self.control_interval_s / self.simulation_dt_s))

    @property
    def control_count(self) -> int:
        return int(round(self.horizon_s / self.control_interval_s))


@dataclass(frozen=True, slots=True)
class DroneCableState:
    drone_position_m: torch.Tensor
    drone_velocity_m_s: torch.Tensor
    cable: DderState

    @property
    def batch_size(self) -> int:
        return int(self.drone_position_m.shape[0])


@dataclass(frozen=True, slots=True)
class TensorRollout:
    time_s: torch.Tensor
    drone_positions_m: torch.Tensor
    drone_velocities_m_s: torch.Tensor
    attachment_positions_m: torch.Tensor
    cable_positions_m: torch.Tensor
    cable_velocities_m_s: torch.Tensor
    accelerations_m_s2: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.drone_positions_m.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.drone_positions_m.shape[1])

    def final_state(self) -> DroneCableState:
        return DroneCableState(
            self.drone_positions_m[:, -1],
            self.drone_velocities_m_s[:, -1],
            DderState(
                self.cable_positions_m[:, -1],
                self.cable_velocities_m_s[:, -1],
            ),
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    time_s: np.ndarray
    drone_positions_m: np.ndarray
    drone_velocities_m_s: np.ndarray
    attachment_positions_m: np.ndarray
    cable_positions_m: np.ndarray
    cable_velocities_m_s: np.ndarray
    accelerations_m_s2: np.ndarray
    target_position_m: np.ndarray
    impact_direction: np.ndarray
    model_sha256: str

    @property
    def frame_count(self) -> int:
        return len(self.time_s)

    @property
    def free_tip_positions_m(self) -> np.ndarray:
        return self.cable_positions_m[:, -1]

    @property
    def free_tip_velocities_m_s(self) -> np.ndarray:
        return self.cable_velocities_m_s[:, -1]


class _RuntimeDderStep:
    """Fixed-shape DDER inference step with CUDA-graph replay.

    The differentiable fitting path intentionally retains the exact direct
    linear solve.  MPC needs many forward evaluations of an already identified
    model, so it uses the model's numerically equivalent fixed-iteration
    runtime solve.  Capturing one substep update removes the otherwise dominant
    Python and kernel-launch overhead without changing the rod equations.
    """

    def __init__(
        self,
        model: DderModel,
        batch_size: int,
        node_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (batch_size, node_count, 3)
        self.model = model
        self.device = device
        self.q = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros_like(self.q)
        self.boundary = torch.zeros((batch_size, 1, 3), dtype=dtype, device=device)
        self.dt = torch.full((batch_size,), 0.01, dtype=dtype, device=device)
        self.constants = model.runtime_constants(self.q)
        self.output: DderState | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        if device.type == "cuda":
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.output = model.step_runtime(
                        DderState(self.q, self.v),
                        self.boundary,
                        self.dt,
                        self.constants,
                        iterative_damping=True,
                        pinned_endpoints=START_PINNED_FREE_END,
                    )
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self.output = model.step_runtime(
                    DderState(self.q, self.v),
                    self.boundary,
                    self.dt,
                    self.constants,
                    iterative_damping=True,
                    pinned_endpoints=START_PINNED_FREE_END,
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
                DderState(self.q, self.v),
                self.boundary,
                self.dt,
                self.constants,
                iterative_damping=False,
                pinned_endpoints=START_PINNED_FREE_END,
            )
        self.graph.replay()
        assert self.output is not None
        return self.output


class _RuntimeDderRollout:
    """Capture a complete fixed-shape rollout as one CUDA graph.

    This preserves the exact production step and chronological operation order;
    it only removes the 35 Python-side one-step graph launches and inter-step
    input copies from the authoritative online workload.
    """

    def __init__(
        self,
        model: DderModel,
        settings: SimulationSettings,
        batch_size: int,
        control_count: int,
        node_count: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.model = model
        self.settings = settings
        self.device = device
        self.batch_size = batch_size
        self.control_count = control_count
        self.q = torch.zeros(
            (batch_size, node_count, 3), dtype=dtype, device=device
        )
        self.v = torch.zeros_like(self.q)
        self.drone_position = torch.zeros(
            (batch_size, 3), dtype=dtype, device=device
        )
        self.drone_velocity = torch.zeros_like(self.drone_position)
        self.controls = torch.zeros(
            (batch_size, control_count, 3), dtype=dtype, device=device
        )
        self.drop = torch.zeros((3,), dtype=dtype, device=device)
        self.drop[2] = -settings.attachment_drop_m
        self.constants = model.runtime_constants(self.q)
        self.output: TensorRollout | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.output = self._execute()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.output = self._execute()

    def _execute(self) -> TensorRollout:
        drone_position = self.drone_position
        drone_velocity = self.drone_velocity
        cable = DderState(self.q, self.v)
        drop = self.drop
        drone_positions = [drone_position]
        drone_velocities = [drone_velocity]
        attachments = [drone_position + drop]
        cable_positions = [cable.positions_m]
        cable_velocities = [cable.velocities_m_s]
        dt = self.settings.simulation_dt_s
        dt_tensor = torch.full(
            (self.batch_size,), dt, dtype=self.q.dtype, device=self.q.device
        )
        total_steps = self.control_count * self.settings.steps_per_control
        for step_index in range(total_steps):
            control_index = step_index // self.settings.steps_per_control
            acceleration = self.controls[:, control_index]
            next_position = (
                drone_position
                + dt * drone_velocity
                + 0.5 * dt * dt * acceleration
            )
            next_velocity = drone_velocity + dt * acceleration
            next_attachment = next_position + drop
            cable = self.model.step_runtime(
                cable,
                next_attachment[:, None],
                dt_tensor,
                self.constants,
                iterative_damping=True,
                pinned_endpoints=START_PINNED_FREE_END,
            )
            drone_position = next_position
            drone_velocity = next_velocity
            drone_positions.append(drone_position)
            drone_velocities.append(drone_velocity)
            attachments.append(next_attachment)
            cable_positions.append(cable.positions_m)
            cable_velocities.append(cable.velocities_m_s)
        time = torch.arange(
            total_steps + 1, dtype=self.q.dtype, device=self.q.device
        ) * dt
        return TensorRollout(
            time_s=time,
            drone_positions_m=torch.stack(drone_positions, dim=1),
            drone_velocities_m_s=torch.stack(drone_velocities, dim=1),
            attachment_positions_m=torch.stack(attachments, dim=1),
            cable_positions_m=torch.stack(cable_positions, dim=1),
            cable_velocities_m_s=torch.stack(cable_velocities, dim=1),
            accelerations_m_s2=self.controls,
        )

    def __call__(
        self,
        initial_state: DroneCableState,
        controls: torch.Tensor,
    ) -> TensorRollout:
        self.q.copy_(initial_state.cable.positions_m)
        self.v.copy_(initial_state.cable.velocities_m_s)
        self.drone_position.copy_(initial_state.drone_position_m)
        self.drone_velocity.copy_(initial_state.drone_velocity_m_s)
        self.controls.copy_(controls)
        assert self.graph is not None
        self.graph.replay()
        assert self.output is not None
        return self.output


def tensor_rollout_to_result(
    rollout: TensorRollout,
    *,
    batch_index: int,
    target_position_m: torch.Tensor,
    impact_direction: torch.Tensor,
    model_sha256: str,
) -> SimulationResult:
    def array(value: torch.Tensor) -> np.ndarray:
        result = value.detach().cpu().numpy().copy()
        result.setflags(write=False)
        return result

    return SimulationResult(
        time_s=array(rollout.time_s),
        drone_positions_m=array(rollout.drone_positions_m[batch_index]),
        drone_velocities_m_s=array(rollout.drone_velocities_m_s[batch_index]),
        attachment_positions_m=array(rollout.attachment_positions_m[batch_index]),
        cable_positions_m=array(rollout.cable_positions_m[batch_index]),
        cable_velocities_m_s=array(rollout.cable_velocities_m_s[batch_index]),
        accelerations_m_s2=array(rollout.accelerations_m_s2[batch_index]),
        target_position_m=array(target_position_m),
        impact_direction=array(impact_direction),
        model_sha256=model_sha256,
    )


class WhipSimulator:
    """A cable driven by an acceleration-controlled drone attachment.

    The drone translation is a double integrator whose acceleration is assumed
    to be tracked by a faster low-level attitude/thrust controller.  Cable force
    does not feed back into this first-stage drone model; the cable-to-drone mass
    ratio and tracking error must be checked experimentally before retaining
    that approximation in flight.
    """

    def __init__(
        self,
        snapshot: CableModelSnapshot,
        settings: SimulationSettings,
        *,
        device: str | torch.device = "cuda",
    ) -> None:
        self.snapshot = snapshot
        self.settings = settings
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is unavailable.")
        maximum_ei = snapshot.model.maximum_stable_bending_stiffness(
            settings.simulation_dt_s,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        if snapshot.bending_stiffness_n_m2 > maximum_ei:
            raise ValueError(
                f"Fitted EI is unstable at dt={settings.simulation_dt_s:g}s; "
                "reduce dt or increase the artifact solver substeps."
            )
        self._runtime_steps: dict[int, _RuntimeDderStep] = {}
        self._runtime_rollouts: dict[tuple[int, int], _RuntimeDderRollout] = {}

    @property
    def runtime_acceleration(self) -> RuntimeAcceleration:
        captured = (
            self.device.type == "cuda"
            and os.environ.get("CABLE_TWIN_FULL_HORIZON_GRAPH", "1") != "0"
        )
        fused_cost = (
            self.device.type == "cuda"
            and os.environ.get("DRONE_MPPI_FUSED_COST", "1") != "0"
        )
        fused_mechanics = (
            self.device.type == "cuda"
            and self.snapshot.node_count == 11
            and self.snapshot.model.parameters.substeps == 1
            and self.snapshot.model.parameters.constraint_iterations == 4
            and os.environ.get("CABLE_TWIN_FUSED_FIXED_DAMPING", "1") != "0"
            and os.environ.get("CABLE_TWIN_FUSED_FIXED_PROJECTION", "1") != "0"
        )
        return RuntimeAcceleration(
            captured_full_horizon=captured,
            fused_cost=fused_cost,
            fused_mechanics=fused_mechanics,
            node_count=self.snapshot.node_count,
        )

    def require_online_acceleration(self) -> RuntimeAcceleration:
        """Return the runtime tier or reject an obsolete slow online path."""

        acceleration = self.runtime_acceleration
        if not acceleration.online_ready:
            raise RuntimeError(
                "Online DDER-MPPI requires CUDA full-horizon capture and the "
                "fused CUDA MPPI cost. Enable CABLE_TWIN_FULL_HORIZON_GRAPH and "
                "DRONE_MPPI_FUSED_COST, or use the reference simulator only in "
                "offline numerical tests."
            )
        return acceleration

    @property
    def dtype(self) -> torch.dtype:
        # Identification remains float64.  The deployed DDER runtime uses
        # float32 on CUDA, as does the existing particle-filter inference path.
        return torch.float32 if self.device.type == "cuda" else torch.float64

    def _runtime_step(self, batch_size: int) -> _RuntimeDderStep:
        step = self._runtime_steps.get(batch_size)
        if step is None:
            step = _RuntimeDderStep(
                self.snapshot.model,
                batch_size,
                self.snapshot.node_count,
                self.dtype,
                self.device,
            )
            self._runtime_steps[batch_size] = step
        return step

    def _runtime_rollout(
        self, batch_size: int, control_count: int
    ) -> _RuntimeDderRollout:
        key = (batch_size, control_count)
        rollout = self._runtime_rollouts.get(key)
        if rollout is None:
            rollout = _RuntimeDderRollout(
                self.snapshot.model,
                self.settings,
                batch_size,
                control_count,
                self.snapshot.node_count,
                self.dtype,
                self.device,
            )
            self._runtime_rollouts[key] = rollout
        return rollout

    def initial_state(
        self,
        drone_position_m: tuple[float, float, float] | np.ndarray,
        drone_velocity_m_s: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> DroneCableState:
        position = torch.as_tensor(
            drone_position_m, dtype=self.dtype, device=self.device
        ).reshape(1, 3)
        velocity = torch.as_tensor(
            drone_velocity_m_s, dtype=self.dtype, device=self.device
        ).reshape(1, 3)
        if not bool(torch.all(torch.isfinite(position)).detach().cpu()):
            raise ValueError("Initial drone position must be finite.")
        if not bool(torch.all(torch.isfinite(velocity)).detach().cpu()):
            raise ValueError("Initial drone velocity must be finite.")
        attachment = position + torch.tensor(
            ((0.0, 0.0, -self.settings.attachment_drop_m),),
            dtype=self.dtype,
            device=self.device,
        )
        material = torch.tensor(
            self.snapshot.rod_material_coordinates_m,
            dtype=self.dtype,
            device=self.device,
        )
        cable_position = attachment[:, None].repeat(
            1, self.snapshot.node_count, 1
        )
        cable_position[:, :, 2] -= material[None]
        cable_velocity = torch.zeros_like(cable_position)
        return DroneCableState(
            position,
            velocity,
            DderState(cable_position, cable_velocity),
        )

    @staticmethod
    def _repeat_state(state: DroneCableState, batch_size: int) -> DroneCableState:
        if state.batch_size == batch_size:
            return state
        if state.batch_size != 1:
            raise ValueError("State batch must be one or match the control batch.")
        return DroneCableState(
            state.drone_position_m.repeat(batch_size, 1),
            state.drone_velocity_m_s.repeat(batch_size, 1),
            DderState(
                state.cable.positions_m.repeat(batch_size, 1, 1),
                state.cable.velocities_m_s.repeat(batch_size, 1, 1),
            ),
        )

    def rollout(
        self,
        initial_state: DroneCableState,
        accelerations_m_s2: torch.Tensor,
        *,
        create_graph: bool,
        cancelled: CancellationCallback | None = None,
    ) -> TensorRollout:
        controls = torch.as_tensor(
            accelerations_m_s2, dtype=self.dtype, device=self.device
        )
        if controls.ndim != 3 or controls.shape[1] < 1 or controls.shape[2] != 3:
            raise ValueError("Accelerations must have shape BxKx3 with K >= 1.")
        if not bool(torch.all(torch.isfinite(controls)).detach().cpu()):
            raise ValueError("Accelerations must be finite.")
        norms = torch.linalg.vector_norm(controls, dim=2)
        if bool(
            torch.any(norms > self.settings.maximum_acceleration_m_s2 * (1.0 + 1.0e-6))
            .detach()
            .cpu()
        ):
            raise ValueError("Acceleration control exceeds the configured hard limit.")

        state = self._repeat_state(initial_state, controls.shape[0])
        # The graph is specialized when it is first constructed, so node
        # count, substep count, and projection count may be arbitrary fixed
        # values.  The former 11-node/one-substep/four-projection gates were
        # historical restrictions from the profiling workload, not CUDA-graph
        # or DDER requirements.
        use_full_horizon_graph = (
            not create_graph
            and self.runtime_acceleration.captured_full_horizon
        )
        if use_full_horizon_graph:
            return self._runtime_rollout(
                controls.shape[0], controls.shape[1]
            )(state, controls)
        drone_position = state.drone_position_m
        drone_velocity = state.drone_velocity_m_s
        cable = state.cable
        drop = torch.tensor(
            (0.0, 0.0, -self.settings.attachment_drop_m),
            dtype=self.dtype,
            device=self.device,
        )
        drone_positions = [drone_position]
        drone_velocities = [drone_velocity]
        attachments = [drone_position + drop]
        cable_positions = [cable.positions_m]
        cable_velocities = [cable.velocities_m_s]
        dt = self.settings.simulation_dt_s
        dt_tensor = torch.full(
            (controls.shape[0],), dt, dtype=self.dtype, device=self.device
        )
        total_steps = controls.shape[1] * self.settings.steps_per_control
        for step_index in range(total_steps):
            if cancelled is not None and cancelled():
                raise RuntimeError("Simulation stopped by user.")
            control_index = step_index // self.settings.steps_per_control
            acceleration = controls[:, control_index]
            next_position = (
                drone_position
                + dt * drone_velocity
                + 0.5 * dt * dt * acceleration
            )
            next_velocity = drone_velocity + dt * acceleration
            next_attachment = next_position + drop
            if create_graph:
                cable = self.snapshot.model.step(
                    cable,
                    next_attachment[:, None],
                    dt_tensor,
                    create_graph=True,
                    _validate=False,
                    pinned_endpoints=START_PINNED_FREE_END,
                )
            else:
                cable = self._runtime_step(controls.shape[0])(
                    cable,
                    next_attachment[:, None],
                    dt,
                )
            drone_position = next_position
            drone_velocity = next_velocity
            drone_positions.append(drone_position)
            drone_velocities.append(drone_velocity)
            attachments.append(next_attachment)
            # A CUDA graph writes every step into the same static output
            # buffers.  Preserve the trajectory frames before the next replay.
            cable_positions.append(
                cable.positions_m if create_graph else cable.positions_m.clone()
            )
            cable_velocities.append(
                cable.velocities_m_s if create_graph else cable.velocities_m_s.clone()
            )

        time = torch.arange(
            total_steps + 1,
            dtype=self.dtype,
            device=self.device,
        ) * dt
        return TensorRollout(
            time_s=time,
            drone_positions_m=torch.stack(drone_positions, dim=1),
            drone_velocities_m_s=torch.stack(drone_velocities, dim=1),
            attachment_positions_m=torch.stack(attachments, dim=1),
            cable_positions_m=torch.stack(cable_positions, dim=1),
            cable_velocities_m_s=torch.stack(cable_velocities, dim=1),
            accelerations_m_s2=controls,
        )
