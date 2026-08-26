"""GPU-side profile of the fixed-topology forward DDER--MPPI runtime.

This is deliberately a diagnostic harness.  The production controller,
physics, CUDA graph, objective, and numerical methods are imported unchanged.
The primary benchmark calls :func:`optimize_mppi` directly.  A separate
numerically checked CUDA graph exposes timings inside one DDER step without
modifying the production graph.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Callable, Iterator

import numpy as np
import torch

import cable_twin.shared.dder as dder
from cable_twin.shared.dder import DderState, START_PINNED_FREE_END
from drone_mpc.model import CableModelSnapshot, load_cable_model
from drone_mpc.problem import MpcProblem
import drone_mpc.mppi as mppi_module
from drone_mpc.mppi import MppiSettings, interpolate_control_knots, optimize_mppi
from drone_mpc.reduced import build_controller_and_truth_models
from drone_mpc.simulator import (
    DroneCableState,
    SimulationSettings,
    TensorRollout,
    WhipSimulator,
)


SCHEMA = "forward_dder_mppi_gpu_profile_v1"
DEFAULT_PROFILE = Path("data/drone_mpc/settings_profiles/11node_tru_phys.json")
DEFAULT_OUTPUT = Path("data/drone_mpc/profiles/11node_forward_profile.json")


@dataclass(frozen=True, slots=True)
class Workload:
    source: CableModelSnapshot
    controller: CableModelSnapshot
    truth: CableModelSnapshot
    simulation: SimulationSettings
    problem: MpcProblem
    mppi: MppiSettings
    initial_xyz: tuple[float, float, float]
    warm_knots: np.ndarray
    profile_payload: dict[str, object]


def _readonly_knots(path: str | Path, knot_count: int) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    knots: np.ndarray | None = None
    metadata = source if source.suffix.lower() == ".json" else source.with_suffix(".json")
    if metadata.is_file():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        parameterization = payload.get("control_parameterization")
        if isinstance(parameterization, dict) and "knots_m_s2" in parameterization:
            knots = np.asarray(parameterization["knots_m_s2"], dtype=np.float32)
    if knots is None and source.suffix.lower() == ".npz":
        with np.load(source) as archive:
            for name in (
                "optimized_knots_m_s2",
                "control_knots_m_s2",
                "nominal_knots_m_s2",
            ):
                if name in archive:
                    values = np.asarray(archive[name], dtype=np.float32)
                    knots = values[-1] if values.ndim == 3 else values
                    break
    if knots is None or knots.ndim != 2 or knots.shape[1] != 3 or len(knots) < 2:
        raise ValueError(f"No valid Mx3 acceleration knots were found in {source}.")
    if len(knots) != knot_count:
        old = np.linspace(0.0, 1.0, len(knots))
        new = np.linspace(0.0, 1.0, knot_count)
        knots = np.stack(
            [np.interp(new, old, knots[:, axis]) for axis in range(3)], axis=1
        ).astype(np.float32)
    result = np.array(knots, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def _vector(text: object) -> tuple[float, float, float]:
    values = tuple(float(value.strip()) for value in str(text).split(","))
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Expected a finite XYZ vector, received {text!r}.")
    return values


def load_workload(profile_path: str | Path) -> Workload:
    profile = Path(profile_path).expanduser().resolve()
    payload = json.loads(profile.read_text(encoding="utf-8"))
    inputs = payload["inputs"]
    task = payload["task"]
    controller_settings = payload["controller"]
    truth_settings = payload["plant_truth"]
    fixed = payload["fixed_objective"]
    if not all(isinstance(value, dict) for value in (
        inputs, task, controller_settings, truth_settings, fixed
    )):
        raise ValueError("The settings profile is incomplete.")

    source = load_cable_model(inputs["model_path"])
    horizon = float(controller_settings["horizon_s"])
    physics_rate = float(controller_settings["physics_rate_hz"])
    control_rate = float(controller_settings["control_rate_hz"])
    node_count = int(controller_settings["simulation_nodes"])
    simulation = SimulationSettings(
        horizon_s=horizon,
        simulation_dt_s=1.0 / physics_rate,
        control_interval_s=1.0 / control_rate,
        attachment_drop_m=0.10,
        maximum_acceleration_m_s2=float(
            controller_settings["maximum_acceleration_m_s2"]
        ),
        maximum_speed_m_s=float(controller_settings["maximum_speed_m_s"]),
    )
    controller, truth = build_controller_and_truth_models(
        source,
        simulation_dt_s=simulation.simulation_dt_s,
        node_count=node_count,
        truth_bending_stiffness_scale=float(truth_settings["ei_scale"]),
        truth_bending_damping_scale=float(truth_settings["cb_scale"]),
    )
    initial = _vector(task["initial_drone_xyz"])
    problem = MpcProblem(
        target_position_m=_vector(task["target_xyz"]),
        impact_direction=_vector(task["impact_direction"]),
        minimum_impact_speed_m_s=float(task["minimum_directed_speed_m_s"]),
        maximum_tip_error_m=float(task["target_radius_m"]),
        maximum_impact_angle_deg=float(task["direction_half_angle_deg"]),
        maximum_drone_excursion_m=1.0,
        minimum_forward_stroke_m=0.0,
        minimum_recoil_stroke_m=0.0,
        drone_keepout_radius_m=0.30,
        drone_workspace_center_m=initial,
    )
    samples = int(controller_settings["samples"])
    mppi = MppiSettings(
        iterations=int(controller_settings["iterations"]),
        samples=samples,
        rollout_batch_size=samples,
        knot_count=int(controller_settings["acceleration_knots"]),
        temperature=float(controller_settings["temperature"]),
        acceleration_noise_sigma_m_s2=float(
            controller_settings["noise_sigma_m_s2"]
        ),
        noise_decay=float(controller_settings["noise_decay"]),
        seed=int(controller_settings["seed"]),
        objective_stage=str(fixed["objective_stage"]),
        position_sigma_m=float(fixed["position_sigma_m"]),
        velocity_gate_sigma_m=float(fixed["velocity_gate_sigma_m"]),
        position_weight=float(fixed["position_weight"]),
        speed_weight=float(fixed["speed_weight"]),
        predictive_speed_weight=0.0,
        direction_weight=float(fixed["direction_weight"]),
        success_cost=float(fixed["success_cost"]),
        drone_displacement_weight=float(fixed["drone_displacement_weight"]),
        safety_weight=float(fixed["safety_weight"]),
        enforce_workspace_limit=False,
        control_effort_weight=float(fixed["control_effort_weight"]),
        control_smoothness_weight=float(fixed["control_smoothness_weight"]),
        gradient_guidance_fraction=0.0,
    )
    warm = _readonly_knots(inputs["warm_start_path"], mppi.knot_count)
    return Workload(
        source=source,
        controller=controller,
        truth=truth,
        simulation=simulation,
        problem=problem,
        mppi=mppi,
        initial_xyz=initial,
        warm_knots=warm,
        profile_payload=payload,
    )


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(ordered),
        "mean_s": statistics.fmean(ordered),
        "median_s": statistics.median(ordered),
        "standard_deviation_s": (
            statistics.stdev(ordered) if len(ordered) > 1 else 0.0
        ),
        "minimum_s": ordered[0],
        "maximum_s": ordered[-1],
        "p95_s": ordered[p95_index],
    }


def _time_synchronized(function: Callable[[], object]) -> tuple[object, float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    result = function()
    end.record()
    torch.cuda.synchronize()
    return result, time.perf_counter() - wall_start, start.elapsed_time(end) * 1.0e-3


class _RegionRecorder:
    def __init__(self) -> None:
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.cpu_enqueue_s: dict[str, float] = {}

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.setdefault(name, []).append((start, end))
            self.cpu_enqueue_s[name] = self.cpu_enqueue_s.get(name, 0.0) + (
                time.perf_counter() - wall
            )

    def elapsed(self) -> dict[str, dict[str, float | int]]:
        torch.cuda.synchronize()
        result: dict[str, dict[str, float | int]] = {}
        for name, pairs in self.events.items():
            values = [start.elapsed_time(end) * 1.0e-3 for start, end in pairs]
            result[name] = {
                "calls": len(values),
                "gpu_s": float(sum(values)),
                "mean_gpu_s": float(statistics.fmean(values)),
                "cpu_enqueue_s": float(self.cpu_enqueue_s.get(name, 0.0)),
            }
        return result


@contextmanager
def _profile_mppi_functions() -> Iterator[_RegionRecorder]:
    """Time non-overlapping public MPPI phases without synchronizing per call."""

    recorder = _RegionRecorder()
    originals = {
        "sample": mppi_module._sample_perturbations,
        "interpolate": mppi_module.interpolate_control_knots,
        "cost": mppi_module._mppi_event_objective,
        "rollout": WhipSimulator.rollout,
    }

    def sample(*args, **kwargs):
        with recorder.region("noise_generation"):
            return originals["sample"](*args, **kwargs)

    def interpolate(*args, **kwargs):
        with recorder.region("knot_interpolation_and_control_clipping"):
            return originals["interpolate"](*args, **kwargs)

    def cost(*args, **kwargs):
        with recorder.region("contact_safety_and_cost"):
            return originals["cost"](*args, **kwargs)

    def rollout(self, initial_state, controls, **kwargs):
        batch = int(torch.as_tensor(controls).shape[0])
        name = "sampled_rollout" if batch > 1 else "winner_rollout"
        with recorder.region(name):
            return originals["rollout"](self, initial_state, controls, **kwargs)

    mppi_module._sample_perturbations = sample
    mppi_module.interpolate_control_knots = interpolate
    mppi_module._mppi_event_objective = cost
    WhipSimulator.rollout = rollout
    try:
        yield recorder
    finally:
        mppi_module._sample_perturbations = originals["sample"]
        mppi_module.interpolate_control_knots = originals["interpolate"]
        mppi_module._mppi_event_objective = originals["cost"]
        WhipSimulator.rollout = originals["rollout"]


def _candidate_controls(
    simulator: WhipSimulator,
    settings: MppiSettings,
    warm_knots: np.ndarray,
) -> torch.Tensor:
    nominal = torch.tensor(warm_knots, dtype=simulator.dtype, device=simulator.device)
    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(settings.seed)
    perturbations = mppi_module._sample_perturbations(
        settings.samples,
        settings.knot_count,
        settings.acceleration_noise_sigma_m_s2,
        generator=generator,
        dtype=simulator.dtype,
        device=simulator.device,
    )
    candidates = mppi_module._bound_vectors(
        nominal[None] + perturbations,
        simulator.settings.maximum_acceleration_m_s2,
    )
    return interpolate_control_knots(
        candidates,
        simulator.settings.control_count,
        simulator.settings.maximum_acceleration_m_s2,
    )


def _instrumented_rollout(
    simulator: WhipSimulator,
    initial_state: DroneCableState,
    controls: torch.Tensor,
) -> tuple[TensorRollout, dict[str, dict[str, float | int]]]:
    """Exact production rollout with timing around copies/replay/storage."""

    recorder = _RegionRecorder()
    batch = controls.shape[0]
    state = simulator._repeat_state(initial_state, batch)
    drone_position = state.drone_position_m
    drone_velocity = state.drone_velocity_m_s
    cable = state.cable
    drop = torch.tensor(
        (0.0, 0.0, -simulator.settings.attachment_drop_m),
        dtype=simulator.dtype,
        device=simulator.device,
    )
    drone_positions = [drone_position]
    drone_velocities = [drone_velocity]
    attachments = [drone_position + drop]
    cable_positions = [cable.positions_m]
    cable_velocities = [cable.velocities_m_s]
    dt = simulator.settings.simulation_dt_s
    runtime = simulator._runtime_step(batch)
    total_steps = controls.shape[1] * simulator.settings.steps_per_control
    for step_index in range(total_steps):
        acceleration = controls[:, step_index // simulator.settings.steps_per_control]
        with recorder.region("drone_and_attachment_integration"):
            next_position = (
                drone_position + dt * drone_velocity + 0.5 * dt * dt * acceleration
            )
            next_velocity = drone_velocity + dt * acceleration
            next_attachment = next_position + drop
        with recorder.region("dder_graph_input_copies"):
            runtime.q.copy_(cable.positions_m)
            runtime.v.copy_(cable.velocities_m_s)
            runtime.boundary.copy_(next_attachment[:, None])
            runtime.dt.fill_(float(dt))
        with recorder.region("dder_cuda_graph_replay"):
            assert runtime.graph is not None
            runtime.graph.replay()
        assert runtime.output is not None
        cable = runtime.output
        drone_position, drone_velocity = next_position, next_velocity
        drone_positions.append(drone_position)
        drone_velocities.append(drone_velocity)
        attachments.append(next_attachment)
        with recorder.region("candidate_trajectory_frame_clones"):
            cable_positions.append(cable.positions_m.clone())
            cable_velocities.append(cable.velocities_m_s.clone())
    with recorder.region("candidate_trajectory_stacking"):
        rollout = TensorRollout(
            time_s=torch.arange(
                total_steps + 1, dtype=simulator.dtype, device=simulator.device
            )
            * dt,
            drone_positions_m=torch.stack(drone_positions, dim=1),
            drone_velocities_m_s=torch.stack(drone_velocities, dim=1),
            attachment_positions_m=torch.stack(attachments, dim=1),
            cable_positions_m=torch.stack(cable_positions, dim=1),
            cable_velocities_m_s=torch.stack(cable_velocities, dim=1),
            accelerations_m_s2=controls,
        )
    return rollout, recorder.elapsed()


class _CapturedStepProfile:
    """Numerically identical one-attached DDER step with internal CUDA events."""

    def __init__(self, simulator: WhipSimulator, batch_size: int) -> None:
        self.simulator = simulator
        model = simulator.snapshot.model
        nodes = simulator.snapshot.node_count
        shape = (batch_size, nodes, 3)
        self.q = torch.zeros(shape, dtype=simulator.dtype, device=simulator.device)
        self.v = torch.zeros_like(self.q)
        self.boundary = torch.zeros(
            (batch_size, 1, 3), dtype=simulator.dtype, device=simulator.device
        )
        self.dt = torch.full(
            (batch_size,), simulator.settings.simulation_dt_s,
            dtype=simulator.dtype, device=simulator.device
        )
        self.constants = model.runtime_constants(self.q)
        self.event_pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
        self.output: DderState | None = None

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.output = self._step(record=False)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.output = self._step(record=True)

    def _events(self, name: str) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        self.event_pairs.setdefault(name, []).append((start, end))
        return start, end

    @contextmanager
    def _region(self, name: str, record: bool) -> Iterator[None]:
        if not record:
            yield
            return
        start, end = self._events(name)
        start.record()
        try:
            yield
        finally:
            end.record()

    def _pcg(
        self, system: torch.Tensor, rhs: torch.Tensor, iterations: int, record: bool
    ) -> torch.Tensor:
        with self._region("damping_float32_to_float64", record):
            matrix = system.to(dtype=torch.float64)
            vector = rhs.to(dtype=torch.float64)
        with self._region("damping_pcg_60_iterations", record):
            solution = torch.zeros_like(vector)
            residual = vector.clone()
            diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
            preconditioned = residual / diagonal
            direction = preconditioned.clone()
            product = torch.sum(residual * preconditioned, dim=-1, keepdim=True)
            tiny = torch.finfo(matrix.dtype).tiny
            for _ in range(iterations):
                applied = (matrix @ direction[..., None])[..., 0]
                denominator = torch.sum(direction * applied, dim=-1, keepdim=True)
                alpha = torch.where(
                    torch.abs(denominator) > tiny,
                    product / denominator,
                    torch.zeros_like(denominator),
                )
                solution = solution + alpha * direction
                residual = residual - alpha * applied
                preconditioned = residual / diagonal
                next_product = torch.sum(
                    residual * preconditioned, dim=-1, keepdim=True
                )
                beta = torch.where(
                    torch.abs(product) > tiny,
                    next_product / product,
                    torch.zeros_like(product),
                )
                direction = preconditioned + beta * direction
                product = next_product
        with self._region("damping_float64_to_float32", record):
            return solution.to(dtype=rhs.dtype)

    def _damping(
        self,
        q: torch.Tensor,
        undamped: torch.Tensor,
        boundary_velocity: torch.Tensor,
        substep_dt: torch.Tensor,
        record: bool,
    ) -> torch.Tensor:
        constants = self.constants
        batch, nodes, _ = q.shape
        with self._region("damping_curvature_rate_jacobian", record):
            jacobian, affine, weight = dder._curvature_rate_jacobian_impl(
                q, constants.rest_lengths_m, None, None
            )
        with self._region("damping_system_assembly", record):
            free_jacobian = jacobian[..., 3:]
            free_zeros = torch.zeros_like(undamped[:, 1:])
            boundary_only = dder._combine_boundary_and_free_values(
                free_zeros, boundary_velocity, START_PINNED_FREE_END
            )
            prescribed_rate = (
                jacobian @ boundary_only.reshape(batch, nodes * 3, 1)
            )[..., 0] + affine
            unit_damping = free_jacobian.transpose(-1, -2) @ (
                weight[..., None] * free_jacobian
            )
            free_mass = constants.masses_kg[1:].repeat_interleave(3)
            step = substep_dt.reshape(batch, 1)
            coefficient = step * constants.bending_damping_n_m2_s[:, None]
            system = torch.diag_embed(free_mass[None].expand(batch, -1))
            system = system + coefficient[..., None] * unit_damping
            damping_from_prescribed = -constants.bending_damping_n_m2_s[:, None] * (
                free_jacobian.transpose(-1, -2)
                @ (weight * prescribed_rate)[..., None]
            )[..., 0]
            rhs = (
                free_mass[None] * undamped[:, 1:].reshape(batch, -1)
                + step * damping_from_prescribed
            )
        free_solution = self._pcg(system, rhs, 6 * (nodes - 1), record)
        with self._region("damping_state_reassembly", record):
            free_velocity = free_solution.reshape(batch, nodes - 1, 3)
            return dder._combine_boundary_and_free_values(
                free_velocity, boundary_velocity, START_PINNED_FREE_END
            )

    def _position_projection(
        self,
        positions: torch.Tensor,
        boundary: torch.Tensor,
        record: bool,
    ) -> torch.Tensor:
        constants = self.constants
        masses = constants.masses_kg
        inverse_mass = torch.reciprocal(masses)
        inverse_mass = torch.cat(
            (torch.zeros_like(inverse_mass[:1]), inverse_mass[1:])
        )
        value = dder._replace_pinned_values(
            positions, boundary, START_PINNED_FREE_END
        )
        epsilon = 64.0 * torch.finfo(value.dtype).eps
        for _ in range(self.simulator.snapshot.model.parameters.constraint_iterations):
            with self._region("position_projection_assembly", record):
                delta = value[:, 1:] - value[:, :-1]
                distance = torch.linalg.vector_norm(delta, dim=-1)
                direction = delta / torch.clamp(distance[..., None], min=epsilon)
                diagonal = (inverse_mass[:-1] + inverse_mass[1:])[None].expand(
                    value.shape[0], -1
                )
                adjacent = torch.sum(direction[:, :-1] * direction[:, 1:], dim=-1)
                off_diagonal = -inverse_mass[1:-1][None] * adjacent
                regularization = 8.0 * torch.finfo(value.dtype).eps * torch.amax(
                    diagonal, dim=1, keepdim=True
                )
                rhs = -(distance - constants.rest_lengths_m[None])
            with self._region("position_projection_thomas_solve", record):
                multiplier = dder.solve_symmetric_tridiagonal(
                    diagonal + regularization, off_diagonal, rhs
                )
            with self._region("position_projection_correction", record):
                edge_correction = multiplier[..., None] * direction
                left = -inverse_mass[:-1][None, :, None] * edge_correction
                right = inverse_mass[1:][None, :, None] * edge_correction
                node_correction = torch.nn.functional.pad(left, (0, 0, 0, 1))
                node_correction = node_correction + torch.nn.functional.pad(
                    right, (0, 0, 1, 0)
                )
                value = dder._replace_pinned_values(
                    value + node_correction, boundary, START_PINNED_FREE_END
                )
        return value

    def _velocity_projection(
        self,
        q: torch.Tensor,
        velocity: torch.Tensor,
        boundary_velocity: torch.Tensor,
        record: bool,
    ) -> torch.Tensor:
        masses = self.constants.masses_kg
        inverse_mass = torch.reciprocal(masses)
        inverse_mass = torch.cat(
            (torch.zeros_like(inverse_mass[:1]), inverse_mass[1:])
        )
        with self._region("velocity_projection_assembly", record):
            value = dder._replace_pinned_values(
                velocity, boundary_velocity, START_PINNED_FREE_END
            )
            edges = q[:, 1:] - q[:, :-1]
            direction = dder._normalize_unchecked(edges)
            diagonal = (inverse_mass[:-1] + inverse_mass[1:])[None].expand(
                q.shape[0], -1
            )
            adjacent = torch.sum(direction[:, :-1] * direction[:, 1:], dim=-1)
            off_diagonal = -inverse_mass[1:-1][None] * adjacent
            constraint_velocity = torch.sum(
                direction * (value[:, 1:] - value[:, :-1]), dim=-1
            )
            regularization = 8.0 * torch.finfo(value.dtype).eps * torch.amax(
                diagonal, dim=1, keepdim=True
            )
        with self._region("velocity_projection_thomas_solve", record):
            multiplier = dder.solve_symmetric_tridiagonal(
                diagonal + regularization, off_diagonal, -constraint_velocity
            )
        with self._region("velocity_projection_correction", record):
            edge_correction = multiplier[..., None] * direction
            left = -inverse_mass[:-1][None, :, None] * edge_correction
            right = inverse_mass[1:][None, :, None] * edge_correction
            node_correction = torch.nn.functional.pad(left, (0, 0, 0, 1))
            node_correction = node_correction + torch.nn.functional.pad(
                right, (0, 0, 1, 0)
            )
            return dder._replace_pinned_values(
                value + node_correction, boundary_velocity, START_PINNED_FREE_END
            )

    def _step(self, record: bool) -> DderState:
        q, v = self.q, self.v
        start_boundary = q[:, :1]
        boundary_velocity = (self.boundary - start_boundary) / self.dt[:, None, None]
        substep_dt = (self.dt / self.simulator.snapshot.model.parameters.substeps)[
            :, None, None
        ]
        with self._region("bending_force", record):
            force = self.simulator.snapshot.model._runtime_internal_force(
                q, self.constants
            )
        with self._region("gravity_and_unconstrained_velocity", record):
            acceleration = (
                force / self.constants.masses_kg[None, :, None]
                + self.constants.gravity_m_s2
            )
            undamped = (v + substep_dt * acceleration) * torch.exp(
                -self.constants.external_drag_s_inv[:, None, None] * substep_dt
            )
        predicted_v = self._damping(
            q, undamped, boundary_velocity, substep_dt[:, 0, 0], record
        )
        with self._region("position_integration_and_attachment_clamp", record):
            predicted_q = q + substep_dt * predicted_v
            predicted_q = dder._replace_pinned_values(
                predicted_q, self.boundary, START_PINNED_FREE_END
            )
        next_q = self._position_projection(predicted_q, self.boundary, record)
        with self._region("provisional_velocity", record):
            provisional_v = (next_q - q) / substep_dt
        next_v = self._velocity_projection(
            next_q, provisional_v, boundary_velocity, record
        )
        return DderState(next_q, next_v)

    def replay(
        self, state: DderState, boundary: torch.Tensor, dt_s: float
    ) -> tuple[DderState, dict[str, float]]:
        self.q.copy_(state.positions_m)
        self.v.copy_(state.velocities_m_s)
        self.boundary.copy_(boundary)
        self.dt.fill_(float(dt_s))
        total_start = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        total_start.record()
        self.graph.replay()
        total_end.record()
        torch.cuda.synchronize()
        assert self.output is not None
        timings = {
            name: sum(start.elapsed_time(end) for start, end in pairs) * 1.0e-3
            for name, pairs in self.event_pairs.items()
        }
        timings["profiled_graph_total"] = total_start.elapsed_time(total_end) * 1.0e-3
        return self.output, timings


def _pcg_residual_diagnostic(
    simulator: WhipSimulator,
    state: DroneCableState,
    boundary: torch.Tensor,
) -> dict[str, object]:
    """Residual history for the unchanged 60-step Jacobi-PCG solve."""

    q = simulator._repeat_state(state, boundary.shape[0]).cable.positions_m
    v = simulator._repeat_state(state, boundary.shape[0]).cable.velocities_m_s
    constants = simulator.snapshot.model.runtime_constants(q)
    dt = torch.full(
        (q.shape[0],), simulator.settings.simulation_dt_s,
        dtype=simulator.dtype, device=simulator.device
    )
    boundary_velocity = (boundary - q[:, :1]) / dt[:, None, None]
    force = simulator.snapshot.model._runtime_internal_force(q, constants)
    acceleration = force / constants.masses_kg[None, :, None] + constants.gravity_m_s2
    undamped = (v + dt[:, None, None] * acceleration) * torch.exp(
        -constants.external_drag_s_inv[:, None, None] * dt[:, None, None]
    )
    jacobian, affine, weight = dder._curvature_rate_jacobian_impl(
        q, constants.rest_lengths_m, None, None
    )
    free_jacobian = jacobian[..., 3:]
    boundary_only = dder._combine_boundary_and_free_values(
        torch.zeros_like(undamped[:, 1:]),
        boundary_velocity,
        START_PINNED_FREE_END,
    )
    prescribed = (jacobian @ boundary_only.reshape(q.shape[0], -1, 1))[..., 0] + affine
    unit = free_jacobian.transpose(-1, -2) @ (weight[..., None] * free_jacobian)
    free_mass = constants.masses_kg[1:].repeat_interleave(3)
    coefficient = dt[:, None] * constants.bending_damping_n_m2_s[:, None]
    system = torch.diag_embed(free_mass[None].expand(q.shape[0], -1))
    system = system + coefficient[..., None] * unit
    damping_from_boundary = -constants.bending_damping_n_m2_s[:, None] * (
        free_jacobian.transpose(-1, -2) @ (weight * prescribed)[..., None]
    )[..., 0]
    rhs = free_mass[None] * undamped[:, 1:].reshape(q.shape[0], -1)
    rhs = rhs + dt[:, None] * damping_from_boundary
    matrix = system.double()
    vector = rhs.double()
    solution = torch.zeros_like(vector)
    residual = vector.clone()
    diagonal = torch.diagonal(matrix, dim1=-2, dim2=-1)
    preconditioned = residual / diagonal
    direction = preconditioned.clone()
    product = torch.sum(residual * preconditioned, dim=-1, keepdim=True)
    tiny = torch.finfo(matrix.dtype).tiny
    checkpoints = {0, 1, 2, 5, 10, 20, 30, 40, 50, 60}
    rhs_norm = torch.linalg.vector_norm(vector, dim=1)
    history: dict[int, torch.Tensor] = {
        0: torch.linalg.vector_norm(residual, dim=1)
        / torch.clamp(rhs_norm, min=tiny)
    }
    for iteration in range(1, 61):
        applied = (matrix @ direction[..., None])[..., 0]
        denominator = torch.sum(direction * applied, dim=-1, keepdim=True)
        alpha = torch.where(
            torch.abs(denominator) > tiny,
            product / denominator,
            torch.zeros_like(denominator),
        )
        solution = solution + alpha * direction
        residual = residual - alpha * applied
        preconditioned = residual / diagonal
        next_product = torch.sum(residual * preconditioned, dim=-1, keepdim=True)
        beta = torch.where(
            torch.abs(product) > tiny,
            next_product / product,
            torch.zeros_like(product),
        )
        direction = preconditioned + beta * direction
        product = next_product
        if iteration in checkpoints:
            history[iteration] = torch.linalg.vector_norm(residual, dim=1) / torch.clamp(
                rhs_norm, min=tiny
            )
    result: dict[str, object] = {}
    for iteration, values in history.items():
        array = values.detach().cpu().numpy()
        result[str(iteration)] = {
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "maximum": float(np.max(array)),
        }
    return result


def _kernel_profile(
    runtime,
    state: DderState,
    boundary: torch.Tensor,
    dt_s: float,
    trace_path: Path,
) -> dict[str, object]:
    runtime.q.copy_(state.positions_m)
    runtime.v.copy_(state.velocities_m_s)
    runtime.boundary.copy_(boundary)
    runtime.dt.fill_(float(dt_s))
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=False,
    ) as profiler:
        for _ in range(5):
            runtime.graph.replay()
        torch.cuda.synchronize()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(trace_path))
    rows: list[dict[str, object]] = []
    for item in profiler.key_averages():
        device_total = float(
            getattr(item, "self_device_time_total", 0.0)
            or getattr(item, "self_cuda_time_total", 0.0)
        )
        if device_total <= 0.0:
            continue
        rows.append(
            {
                "name": item.key,
                "calls": int(item.count),
                "self_device_time_us": device_total,
                "mean_self_device_time_us": device_total / max(1, int(item.count)),
                "self_device_memory_bytes": int(
                    getattr(item, "self_device_memory_usage", 0)
                ),
            }
        )
    rows.sort(key=lambda value: float(value["self_device_time_us"]), reverse=True)
    total_calls = sum(int(value["calls"]) for value in rows)
    total_device_time_us = sum(float(value["self_device_time_us"]) for value in rows)
    return {
        "five_replays_top_device_operations": rows[:40],
        "five_replays_device_operation_calls": total_calls,
        "device_operation_calls_per_replay": total_calls / 5.0,
        "five_replays_summed_device_operation_time_us": total_device_time_us,
        "mean_device_operation_duration_us": (
            total_device_time_us / max(1, total_calls)
        ),
        "trace_path": str(trace_path.resolve()),
        "occupancy_and_bandwidth": (
            "Not available from torch.profiler; Nsight Compute must target the "
            "dominant generated kernels separately."
        ),
    }


def _max_difference(first: TensorRollout, second: TensorRollout) -> dict[str, float]:
    names = (
        "drone_positions_m",
        "drone_velocities_m_s",
        "attachment_positions_m",
        "cable_positions_m",
        "cable_velocities_m_s",
    )
    return {
        name: float(torch.max(torch.abs(getattr(first, name) - getattr(second, name))).cpu())
        for name in names
    }


def _single_step_inputs(
    simulator: WhipSimulator,
    state: DroneCableState,
    controls: torch.Tensor,
) -> tuple[DderState, torch.Tensor]:
    repeated = simulator._repeat_state(state, controls.shape[0])
    dt = simulator.settings.simulation_dt_s
    acceleration = controls[:, 0]
    next_position = (
        repeated.drone_position_m
        + dt * repeated.drone_velocity_m_s
        + 0.5 * dt * dt * acceleration
    )
    drop = torch.tensor(
        (0.0, 0.0, -simulator.settings.attachment_drop_m),
        dtype=simulator.dtype,
        device=simulator.device,
    )
    return repeated.cable, (next_position + drop)[:, None]


def run_profile(arguments: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("This profile requires CUDA.")
    workload = load_workload(arguments.profile)
    device = torch.device("cuda")
    simulator = WhipSimulator(workload.controller, workload.simulation, device=device)
    state = simulator.initial_state(workload.initial_xyz)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile_path": str(Path(arguments.profile).resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "workload": {
            "nodes": workload.controller.node_count,
            "horizon_s": workload.simulation.horizon_s,
            "physics_rate_hz": 1.0 / workload.simulation.simulation_dt_s,
            "control_rate_hz": 1.0 / workload.simulation.control_interval_s,
            "physics_steps": workload.simulation.control_count
            * workload.simulation.steps_per_control,
            "acceleration_knots": workload.mppi.knot_count,
            "samples": workload.mppi.samples,
            "iterations": workload.mppi.iterations,
            "sampled_rollouts_per_update": workload.mppi.samples
            * workload.mppi.iterations,
            "dder_substeps": workload.controller.model.parameters.substeps,
            "position_projection_iterations": (
                workload.controller.model.parameters.constraint_iterations
            ),
            "pcg_iterations": 6 * (workload.controller.node_count - 1),
            "bending_stiffness_n_m2": workload.controller.bending_stiffness_n_m2,
            "bending_damping_n_m2_s": (
                workload.controller.bending_damping_n_m2_s
            ),
            "matched_truth": workload.controller is workload.truth,
            "gradient_guidance_fraction": workload.mppi.gradient_guidance_fraction,
        },
    }

    # The first call contains both 2048- and 1-batch CUDA graph capture.
    torch.cuda.reset_peak_memory_stats(device)
    cold_plan, cold_wall, cold_gpu = _time_synchronized(
        lambda: optimize_mppi(
            simulator,
            state,
            workload.problem,
            workload.mppi,
            warm_start_knots_m_s2=workload.warm_knots,
            objective_reference_state=state,
            progress=None,
        )
    )
    report["cold_update"] = {
        "wall_s": cold_wall,
        "gpu_span_s": cold_gpu,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "feasible": cold_plan.feasible,
    }
    del cold_plan

    warmed_wall: list[float] = []
    warmed_gpu: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(arguments.warm_repeats):
        plan, wall, gpu = _time_synchronized(
            lambda: optimize_mppi(
                simulator,
                state,
                workload.problem,
                workload.mppi,
                warm_start_knots_m_s2=workload.warm_knots,
                objective_reference_state=state,
                progress=None,
            )
        )
        warmed_wall.append(wall)
        warmed_gpu.append(gpu)
        del plan
    report["warmed_update"] = {
        "wall": _stats(warmed_wall),
        "gpu_span": _stats(warmed_gpu),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
    }

    with _profile_mppi_functions() as phase_recorder:
        _plan, instrumented_wall, instrumented_gpu = _time_synchronized(
            lambda: optimize_mppi(
                simulator,
                state,
                workload.problem,
                workload.mppi,
                warm_start_knots_m_s2=workload.warm_knots,
                objective_reference_state=state,
                progress=None,
            )
        )
    phases = phase_recorder.elapsed()
    phase_gpu = sum(float(value["gpu_s"]) for value in phases.values())
    report["instrumented_update_phases"] = {
        "wall_s": instrumented_wall,
        "gpu_span_s": instrumented_gpu,
        "phases": phases,
        "unclassified_or_host_idle_gpu_span_s": max(0.0, instrumented_gpu - phase_gpu),
        "note": (
            "Function-boundary CUDA events add profiling overhead. The untouched "
            "warmed benchmark is the authoritative update time."
        ),
    }
    del _plan

    controls = _candidate_controls(simulator, workload.mppi, workload.warm_knots)
    # The graph is already warm.  Compare an exact diagnostic rollout against production.
    with torch.no_grad():
        production = simulator.rollout(state, controls, create_graph=False)
        diagnostic, rollout_regions = _instrumented_rollout(simulator, state, controls)
    report["rollout_breakdown"] = rollout_regions
    report["rollout_diagnostic_max_abs_difference"] = _max_difference(
        production, diagnostic
    )

    theoretical_cable_bytes = int(
        workload.mppi.samples
        * (workload.simulation.control_count + 1)
        * workload.controller.node_count
        * 3
        * torch.tensor([], dtype=simulator.dtype).element_size()
    )
    report["trajectory_storage"] = {
        "cable_positions_bytes": theoretical_cable_bytes,
        "cable_velocities_bytes": theoretical_cable_bytes,
        "combined_cable_position_velocity_bytes": 2 * theoretical_cable_bytes,
        "frame_clone_gpu_s": rollout_regions["candidate_trajectory_frame_clones"][
            "gpu_s"
        ],
        "stack_gpu_s": rollout_regions["candidate_trajectory_stacking"]["gpu_s"],
        "note": (
            "The byte count excludes drone/attachment arrays, Python frame lists, "
            "stacking transients, cost intermediates, and allocator reservation."
        ),
    }

    cable_state, boundary = _single_step_inputs(simulator, state, controls)
    production_runtime = simulator._runtime_step(workload.mppi.samples)
    production_output, production_step_wall, production_step_gpu = _time_synchronized(
        lambda: production_runtime(cable_state, boundary, simulator.settings.simulation_dt_s)
    )
    profiled_step = _CapturedStepProfile(simulator, workload.mppi.samples)
    component_repetitions: list[dict[str, float]] = []
    profiled_output = None
    for _ in range(arguments.component_repeats):
        profiled_output, component_times = profiled_step.replay(
            cable_state, boundary, simulator.settings.simulation_dt_s
        )
        component_repetitions.append(component_times)
    assert profiled_output is not None
    component_names = tuple(component_repetitions[0])
    component_statistics = {
        name: _stats([values[name] for values in component_repetitions])
        for name in component_names
    }
    median_components = {
        name: float(component_statistics[name]["median_s"])
        for name in component_names
    }
    q_difference = float(
        torch.max(torch.abs(production_output.positions_m - profiled_output.positions_m)).cpu()
    )
    v_difference = float(
        torch.max(
            torch.abs(production_output.velocities_m_s - profiled_output.velocities_m_s)
        ).cpu()
    )
    report["one_step_component_profile"] = {
        "production_wall_s": production_step_wall,
        "production_gpu_s": production_step_gpu,
        "profiled_components_s": median_components,
        "profiled_component_statistics": component_statistics,
        "profiled_component_sum_s": sum(
            value for key, value in median_components.items()
            if key != "profiled_graph_total"
        ),
        "position_max_abs_difference_m": q_difference,
        "velocity_max_abs_difference_m_s": v_difference,
        "note": (
            "The profiling graph inserts external CUDA event nodes but evaluates "
            "the same equations and fixed iteration counts."
        ),
    }
    report["pcg_relative_residual"] = _pcg_residual_diagnostic(
        simulator, state, boundary
    )

    # The final selected trajectory is replayed at batch one after every MPPI solve.
    winner_controls = interpolate_control_knots(
        torch.tensor(
            workload.warm_knots[None], dtype=simulator.dtype, device=simulator.device
        ),
        simulator.settings.control_count,
        simulator.settings.maximum_acceleration_m_s2,
    )
    with torch.no_grad():
        _winner_rollout, winner_regions = _instrumented_rollout(
            simulator, state, winner_controls
        )
    report["winner_rollout_breakdown"] = winner_regions
    winner_state, winner_boundary = _single_step_inputs(
        simulator, state, winner_controls
    )
    winner_production_runtime = simulator._runtime_step(1)
    winner_production_output, winner_wall, winner_gpu = _time_synchronized(
        lambda: winner_production_runtime(
            winner_state, winner_boundary, simulator.settings.simulation_dt_s
        )
    )
    winner_profiled_step = _CapturedStepProfile(simulator, 1)
    winner_component_repetitions: list[dict[str, float]] = []
    winner_profiled_output = None
    for _ in range(arguments.component_repeats):
        winner_profiled_output, values = winner_profiled_step.replay(
            winner_state, winner_boundary, simulator.settings.simulation_dt_s
        )
        winner_component_repetitions.append(values)
    assert winner_profiled_output is not None
    winner_statistics = {
        name: _stats([values[name] for values in winner_component_repetitions])
        for name in winner_component_repetitions[0]
    }
    report["one_step_component_profile_batch_one"] = {
        "production_wall_s": winner_wall,
        "production_gpu_s": winner_gpu,
        "profiled_components_s": {
            name: float(statistics["median_s"])
            for name, statistics in winner_statistics.items()
        },
        "profiled_component_statistics": winner_statistics,
        "position_max_abs_difference_m": float(
            torch.max(
                torch.abs(
                    winner_production_output.positions_m
                    - winner_profiled_output.positions_m
                )
            ).cpu()
        ),
        "velocity_max_abs_difference_m_s": float(
            torch.max(
                torch.abs(
                    winner_production_output.velocities_m_s
                    - winner_profiled_output.velocities_m_s
                )
            ).cpu()
        ),
    }

    trace_path = Path(arguments.output).with_suffix(".torch_trace.json")
    report["production_graph_kernel_profile"] = _kernel_profile(
        production_runtime,
        cable_state,
        boundary,
        simulator.settings.simulation_dt_s,
        trace_path,
    )

    del (
        production,
        diagnostic,
        controls,
        profiled_output,
        production_output,
        _winner_rollout,
        winner_production_output,
        winner_profiled_output,
        winner_controls,
    )
    torch.cuda.synchronize()

    sweep: dict[str, object] = {}
    sweep_simulator = WhipSimulator(
        workload.controller, workload.simulation, device=device
    )
    sweep_state = sweep_simulator.initial_state(workload.initial_xyz)
    for samples in arguments.sample_counts:
        settings = replace(
            workload.mppi,
            samples=samples,
            rollout_batch_size=samples,
            iterations=1,
        )
        _cold, cold_wall_b, _cold_gpu_b = _time_synchronized(
            lambda settings=settings: optimize_mppi(
                sweep_simulator,
                sweep_state,
                workload.problem,
                settings,
                warm_start_knots_m_s2=workload.warm_knots,
                objective_reference_state=sweep_state,
                progress=None,
            )
        )
        del _cold
        walls: list[float] = []
        gpu_values: list[float] = []
        for _ in range(arguments.sweep_repeats):
            value, wall, gpu = _time_synchronized(
                lambda settings=settings: optimize_mppi(
                    sweep_simulator,
                    sweep_state,
                    workload.problem,
                    settings,
                    warm_start_knots_m_s2=workload.warm_knots,
                    objective_reference_state=sweep_state,
                    progress=None,
                )
            )
            walls.append(wall)
            gpu_values.append(gpu)
            del value
        sweep[str(samples)] = {
            "actual_sample_batch": samples,
            "actual_sampled_rollouts": samples,
            "physics_step_instances": samples
            * workload.simulation.control_count
            * workload.simulation.steps_per_control,
            "cold_wall_s": cold_wall_b,
            "warmed_wall": _stats(walls),
            "warmed_gpu_span": _stats(gpu_values),
        }
    report["sample_count_sweep_one_iteration"] = sweep

    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--warm-repeats", type=int, default=10)
    parser.add_argument("--sweep-repeats", type=int, default=5)
    parser.add_argument("--component-repeats", type=int, default=10)
    parser.add_argument(
        "--sample-counts", type=int, nargs="+", default=(256, 512, 1024, 2048)
    )
    arguments = parser.parse_args()
    if (
        arguments.warm_repeats < 2
        or arguments.sweep_repeats < 2
        or arguments.component_repeats < 2
    ):
        raise ValueError("Use at least two warmed repetitions.")
    output = run_profile(arguments)
    print(f"Forward DDER--MPPI profile: {output}")


if __name__ == "__main__":
    main()
