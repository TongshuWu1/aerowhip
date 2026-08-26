"""Profile the exact DDER action-gradient path without changing its algorithm.

The report separates an inference rollout, an eager no-outer-graph rollout,
autograd graph construction, the one scalar reverse-mode gradient, and a
projection-only chain with the same number of DDER substep projections.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import fields, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable, Iterator

import numpy as np
import torch

import cable_twin.shared.dder as dder
from cable_twin.shared.dder import DderState, START_PINNED_FREE_END
from drone_mpc.model import load_cable_model
from drone_mpc.problem import MpcProblem
from drone_mpc.mppi import interpolate_control_knots, smooth_strike_surrogate
from drone_mpc.perfect_model import PerfectMpcSettings
from drone_mpc.simulator import DroneCableState, TensorRollout, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH
from research_tools.mppi_ablation import resample_knots_in_time


DEFAULT_BASELINE = Path("data/drone_mpc/perfect_model_mppi_farther_faster.npz")
SCHEMA = "dder_gradient_profile_v1"


def _load_metadata(path: Path) -> dict[str, object]:
    payload = json.loads(path.expanduser().resolve().with_suffix(".json").read_text())
    if not isinstance(payload, dict):
        raise ValueError("Baseline metadata must be a JSON object.")
    return payload


def _settings(payload: dict[str, object]) -> PerfectMpcSettings:
    raw = payload.get("settings")
    if not isinstance(raw, dict):
        raise ValueError("Baseline metadata has no settings object.")
    known = {item.name for item in fields(PerfectMpcSettings)}
    return PerfectMpcSettings(**{key: value for key, value in raw.items() if key in known})


def _problem(payload: dict[str, object], minimum_speed: float) -> MpcProblem:
    raw = payload.get("problem")
    if not isinstance(raw, dict):
        raise ValueError("Baseline metadata has no problem object.")
    return replace(MpcProblem(**raw), minimum_impact_speed_m_s=minimum_speed)


def _nominal_knots(
    payload: dict[str, object], *, horizon_s: float, knot_count: int
) -> np.ndarray:
    controls = payload.get("control_parameterization")
    settings = payload.get("settings")
    if not isinstance(controls, dict) or not isinstance(settings, dict):
        raise ValueError("Baseline has no saved knot trajectory.")
    return resample_knots_in_time(
        np.asarray(controls["knots_m_s2"], dtype=np.float32),
        float(settings["horizon_s"]),
        horizon_s,
        knot_count,
    )


def _cuda_time(function: Callable[[], object]) -> tuple[object, float, float]:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    result = function()
    end.record()
    torch.cuda.synchronize()
    return result, time.perf_counter() - wall_start, start.elapsed_time(end) * 1.0e-3


class _AutogradCounter:
    def __init__(self) -> None:
        self.phase = "unassigned"
        self.calls: dict[str, int] = {}
        self.create_graph_calls: dict[str, int] = {}
        self.outer_shapes: list[dict[str, object]] = []
        self._original = torch.autograd.grad

    def __enter__(self) -> "_AutogradCounter":
        original = self._original

        def counted_grad(*args, **kwargs):
            phase = self.phase
            self.calls[phase] = self.calls.get(phase, 0) + 1
            if bool(kwargs.get("create_graph", False)):
                self.create_graph_calls[phase] = (
                    self.create_graph_calls.get(phase, 0) + 1
                )
            if phase in {"full_backward", "projection_backward"}:
                outputs = args[0] if args else kwargs.get("outputs")
                inputs = args[1] if len(args) > 1 else kwargs.get("inputs")
                output_values = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
                input_values = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                self.outer_shapes.append(
                    {
                        "phase": phase,
                        "output_shapes": [list(value.shape) for value in output_values],
                        "input_shapes": [list(value.shape) for value in input_values],
                    }
                )
            return original(*args, **kwargs)

        torch.autograd.grad = counted_grad  # type: ignore[assignment]
        return self

    def __exit__(self, *_args) -> None:
        torch.autograd.grad = self._original  # type: ignore[assignment]


@contextmanager
def _saved_tensor_audit() -> Iterator[dict[str, object]]:
    report: dict[str, object] = {
        "saved_by_device": {},
        "loaded_by_device": {},
        "pack_calls": 0,
        "unpack_calls": 0,
        "hook_wall_time_s": 0.0,
    }

    def update(name: str, tensor: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        key = str(tensor.device)
        values = report[name]
        assert isinstance(values, dict)
        values[key] = int(values.get(key, 0)) + 1
        call_name = "pack_calls" if name == "saved_by_device" else "unpack_calls"
        report[call_name] = int(report[call_name]) + 1
        report["hook_wall_time_s"] = float(report["hook_wall_time_s"]) + (
            time.perf_counter() - started
        )
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: update("saved_by_device", tensor),
        lambda tensor: update("loaded_by_device", tensor),
    ):
        yield report


def _eager_dder_rollout(
    simulator: WhipSimulator,
    initial_state: DroneCableState,
    controls: torch.Tensor,
) -> TensorRollout:
    """Same eager DDER step as the gradient path, without an outer graph."""

    state = simulator._repeat_state(initial_state, controls.shape[0])
    drone_position = state.drone_position_m
    drone_velocity = state.drone_velocity_m_s
    cable = state.cable
    drop = torch.tensor(
        (0.0, 0.0, -simulator.settings.attachment_drop_m),
        dtype=simulator.dtype,
        device=simulator.device,
    )
    positions = [drone_position]
    velocities = [drone_velocity]
    attachments = [drone_position + drop]
    cable_positions = [cable.positions_m]
    cable_velocities = [cable.velocities_m_s]
    dt = simulator.settings.simulation_dt_s
    dt_tensor = torch.full(
        (controls.shape[0],), dt, dtype=simulator.dtype, device=simulator.device
    )
    total_steps = controls.shape[1] * simulator.settings.steps_per_control
    with torch.no_grad():
        for step_index in range(total_steps):
            acceleration = controls[:, step_index // simulator.settings.steps_per_control]
            next_position = drone_position + dt * drone_velocity + 0.5 * dt * dt * acceleration
            next_velocity = drone_velocity + dt * acceleration
            attachment = next_position + drop
            cable = simulator.snapshot.model.step(
                cable,
                attachment[:, None],
                dt_tensor,
                create_graph=False,
                _validate=False,
                pinned_endpoints=START_PINNED_FREE_END,
            )
            drone_position, drone_velocity = next_position, next_velocity
            positions.append(drone_position)
            velocities.append(drone_velocity)
            attachments.append(attachment)
            cable_positions.append(cable.positions_m)
            cable_velocities.append(cable.velocities_m_s)
    return TensorRollout(
        time_s=torch.arange(
            total_steps + 1, dtype=simulator.dtype, device=simulator.device
        )
        * dt,
        drone_positions_m=torch.stack(positions, dim=1),
        drone_velocities_m_s=torch.stack(velocities, dim=1),
        attachment_positions_m=torch.stack(attachments, dim=1),
        cable_positions_m=torch.stack(cable_positions, dim=1),
        cable_velocities_m_s=torch.stack(cable_velocities, dim=1),
        accelerations_m_s2=controls,
    )


def _projection_only_chain(
    simulator: WhipSimulator,
    state: DroneCableState,
    controls: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat the exact length/velocity projections at the full rollout count."""

    q = state.cable.positions_m
    v = state.cable.velocities_m_s
    rest_lengths, masses = simulator.snapshot.model._constants(q)
    drone_position = state.drone_position_m
    drone_velocity = state.drone_velocity_m_s
    drop = torch.tensor(
        (0.0, 0.0, -simulator.settings.attachment_drop_m),
        dtype=simulator.dtype,
        device=simulator.device,
    )
    dt = simulator.settings.simulation_dt_s
    substeps = simulator.snapshot.model.parameters.substeps
    substep_dt = dt / substeps
    for step_index in range(
        controls.shape[1] * simulator.settings.steps_per_control
    ):
        acceleration = controls[:, step_index // simulator.settings.steps_per_control]
        next_position = drone_position + dt * drone_velocity + 0.5 * dt * dt * acceleration
        next_velocity = drone_velocity + dt * acceleration
        next_attachment = next_position + drop
        start_boundary = q[:, :1]
        boundary_velocity = (next_attachment[:, None] - start_boundary) / dt
        for substep in range(substeps):
            fraction = float(substep + 1) / substeps
            boundary = start_boundary + fraction * (
                next_attachment[:, None] - start_boundary
            )
            predicted_q = q + substep_dt * v
            next_q = dder.momentum_project_lengths(
                predicted_q,
                rest_lengths,
                masses,
                boundary,
                iterations=simulator.snapshot.model.parameters.constraint_iterations,
                pinned_endpoints=START_PINNED_FREE_END,
            )
            provisional_v = (next_q - q) / substep_dt
            v = dder.momentum_project_velocities(
                next_q,
                provisional_v,
                masses,
                boundary_velocity,
                validate=False,
                pinned_endpoints=START_PINNED_FREE_END,
            )
            q = next_q
        drone_position, drone_velocity = next_position, next_velocity
    return q, v


def profile(arguments: argparse.Namespace) -> Path:
    if arguments.device != "cuda":
        raise ValueError("This profiler is specifically for the CUDA gradient path.")
    metadata = _load_metadata(arguments.baseline_replay)
    snapshot = load_cable_model(arguments.model)
    if snapshot.sha256 != metadata.get("model_sha256"):
        raise ValueError("Baseline and selected model hashes do not match.")
    settings = replace(
        _settings(metadata), horizon_s=2.0, mppi_knot_count=16
    )
    problem = _problem(metadata, arguments.minimum_impact_speed_m_s)
    simulator = WhipSimulator(snapshot, settings.simulation_settings(), device="cuda")
    state = simulator.initial_state(problem.drone_workspace_center_m)
    knots_np = _nominal_knots(metadata, horizon_s=2.0, knot_count=16)
    knots = torch.tensor(knots_np, dtype=simulator.dtype, device=simulator.device)
    controls = interpolate_control_knots(
        knots[None],
        simulator.settings.control_count,
        simulator.settings.maximum_acceleration_m_s2,
    )

    # Warm only the CUDA-graph inference path so its capture cost is excluded.
    with torch.no_grad():
        simulator.rollout(state, controls, create_graph=False)

    report: dict[str, object] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(simulator.device),
        "device_name": torch.cuda.get_device_name(simulator.device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_sha256": snapshot.sha256,
        "node_count": snapshot.node_count,
        "physics_steps": simulator.settings.control_count
        * simulator.settings.steps_per_control,
        "dder_substeps": snapshot.model.parameters.substeps,
        "projection_iterations": snapshot.model.parameters.constraint_iterations,
        "knot_shape": [16, 3],
        "control_shape": list(controls.shape),
    }

    with _AutogradCounter() as counter:
        counter.phase = "cuda_graph_no_grad_forward"
        no_grad_rollout, wall, gpu = _cuda_time(
            lambda: simulator.rollout(state, controls, create_graph=False)
        )
        report["cuda_graph_no_grad_forward"] = {"wall_s": wall, "gpu_s": gpu}
        del no_grad_rollout

        counter.phase = "eager_no_outer_graph_forward"
        eager_rollout, wall, gpu = _cuda_time(
            lambda: _eager_dder_rollout(simulator, state, controls)
        )
        report["eager_no_outer_graph_forward"] = {"wall_s": wall, "gpu_s": gpu}
        del eager_rollout

        torch.cuda.reset_peak_memory_stats()
        nominal = knots.detach().clone().requires_grad_(True)
        counter.phase = "interpolation_graph"
        graph_controls, wall, gpu = _cuda_time(
            lambda: interpolate_control_knots(
                nominal[None],
                simulator.settings.control_count,
                simulator.settings.maximum_acceleration_m_s2,
            )
        )
        report["interpolation_graph"] = {"wall_s": wall, "gpu_s": gpu}

        saved_audit: dict[str, object]
        with _saved_tensor_audit() as saved_audit:
            counter.phase = "full_rollout_graph_forward"
            rollout, wall, gpu = _cuda_time(
                lambda: simulator.rollout(
                    state, graph_controls, create_graph=True
                )
            )
            report["full_rollout_graph_forward"] = {"wall_s": wall, "gpu_s": gpu}
            counter.phase = "smooth_surrogate_graph_forward"
            cost_and_terms, wall, gpu = _cuda_time(
                lambda: smooth_strike_surrogate(
                    rollout, problem, settings.mppi_settings()
                )
            )
            smooth_cost, _terms = cost_and_terms
            report["smooth_surrogate_graph_forward"] = {"wall_s": wall, "gpu_s": gpu}
            report["rollout_output_devices"] = {
                name: str(getattr(rollout, name).device)
                for name in (
                    "time_s",
                    "drone_positions_m",
                    "drone_velocities_m_s",
                    "attachment_positions_m",
                    "cable_positions_m",
                    "cable_velocities_m_s",
                    "accelerations_m_s2",
                )
            }
            counter.phase = "full_backward"
            gradient, wall, gpu = _cuda_time(
                lambda: torch.autograd.grad(
                    smooth_cost.sum(),
                    nominal,
                    create_graph=False,
                    retain_graph=False,
                )[0]
            )
            report["full_backward"] = {"wall_s": wall, "gpu_s": gpu}
            report["gradient_device"] = str(gradient.device)
            report["gradient_shape"] = list(gradient.shape)
            report["gradient_finite"] = bool(
                torch.all(torch.isfinite(gradient)).detach().cpu()
            )
            report["saved_tensor_audit"] = saved_audit
        report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
        del rollout, graph_controls, smooth_cost, gradient, nominal, cost_and_terms
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        projection_knots = knots.detach().clone().requires_grad_(True)
        projection_controls = interpolate_control_knots(
            projection_knots[None],
            simulator.settings.control_count,
            simulator.settings.maximum_acceleration_m_s2,
        )
        counter.phase = "projection_graph_forward"
        projection_output, wall, gpu = _cuda_time(
            lambda: _projection_only_chain(
                simulator, state, projection_controls
            )
        )
        report["projection_graph_forward"] = {"wall_s": wall, "gpu_s": gpu}
        projected_q, projected_v = projection_output
        projection_scalar = projected_q.square().sum() + projected_v.square().sum()
        counter.phase = "projection_backward"
        projection_gradient, wall, gpu = _cuda_time(
            lambda: torch.autograd.grad(
                projection_scalar,
                projection_knots,
                create_graph=False,
                retain_graph=False,
            )[0]
        )
        report["projection_backward"] = {"wall_s": wall, "gpu_s": gpu}
        report["projection_gradient_finite"] = bool(
            torch.all(torch.isfinite(projection_gradient)).detach().cpu()
        )
        report["autograd_grad_calls_by_phase"] = counter.calls
        report["autograd_create_graph_calls_by_phase"] = counter.create_graph_calls
        report["outer_autograd_shapes"] = counter.outer_shapes

    physics_steps = int(report["physics_steps"])
    substeps = int(report["dder_substeps"])
    report["expected_counts"] = {
        "elastic_force_autograd_grad_calls_during_full_graph_forward": (
            physics_steps * substeps
        ),
        "outer_scalar_reverse_mode_calls": 1,
        "length_projection_calls_per_rollout": physics_steps * substeps,
        "velocity_projection_calls_per_rollout": physics_steps * substeps,
        "length_projection_tridiagonal_solves_per_rollout": (
            physics_steps * substeps * int(report["projection_iterations"])
        ),
    }
    report["static_path_audit"] = {
        "full_jacobian_api_used": False,
        "one_backward_per_control_dimension": False,
        "outer_gradient_api": "one torch.autograd.grad(scalar, U[16,3]) call",
        "known_host_synchronizations": [
            "WhipSimulator.rollout finite-control validation (.detach().cpu())",
            "WhipSimulator.rollout acceleration-limit validation (.detach().cpu())",
            "compute_dder_guidance explicit CUDA synchronize before/after timing",
            "post-backward scalar extraction for cost/norm/finite checks",
        ],
        "known_host_to_device_construction": [
            "DderModel.step recreates four fitted-parameter CUDA scalars, gravity, "
            "rest lengths, and masses on each of 200 physics steps",
            "DderModel.bending_energy calls _constants again inside each of 600 "
            "elastic-force evaluations, transferring rest lengths and masses",
            "small target and impact-direction tensors in smooth_strike_surrogate",
            "small rollout drop, dt, and time tensors",
        ],
        "estimated_repeated_small_host_to_device_constructions": 2600,
        "per_step_numpy_conversion": False,
        "per_step_device_transfer_in_differentiable_loop": True,
        "python_loops": (
            "200 physics steps; 3 DDER substeps each; sequential length-projection "
            "Thomas solves over 20 edges"
        ),
        "graph_reconstruction": (
            "the full 200-step higher-order autograd graph is rebuilt once per "
            "guided MPPI iteration"
        ),
    }

    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--baseline-replay", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--minimum-impact-speed-m-s", type=float, default=3.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drone_mpc/ablations/dder_gradient_profile.json"),
    )
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    output = profile(arguments)
    print(f"DDER gradient profile: {output}")


if __name__ == "__main__":
    main()
