"""Freeze deterministic reference data for exact DDER--MPPI acceleration.

This tool deliberately calls the production simulator and objective without
modifying either.  The resulting bundle is the numerical and controller-level
oracle used to accept or reject optimized forward operators.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import cable_twin.shared.dder as dder
from cable_twin.shared.dder import DderState, START_PINNED_FREE_END
import drone_mpc.mppi as mppi_module
from drone_mpc.mppi import interpolate_control_knots
from drone_mpc.receding_mppi import perturb_cable_state
from drone_mpc.simulator import DroneCableState, TensorRollout, WhipSimulator
from research_tools.mppi_ablation import resample_knots_in_time
from research_tools.profile_mppi_forward import DEFAULT_PROFILE, load_workload


SCHEMA = "dder_mppi_exact_reference_v1"
DEFAULT_OUTPUT = Path("data/drone_mpc/optimization/reference")
BACKWARD_GLOB = Path(
    "data/drone_mpc/ablations/mppi_discovery_pilot"
).glob("initialization_continuation__T2__k11__vg0_12__pw0__pilot__i6__n128__b128__seed17.npz")


def _cpu(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_npz(path: Path, values: dict[str, object]) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            arrays[name] = _cpu(value)
        elif isinstance(value, np.ndarray):
            arrays[name] = np.array(value, copy=True)
        elif isinstance(value, (bool, int, float, str)) or value is None:
            metadata[name] = value
        else:
            arrays[name] = np.asarray(value)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return {
        "path": path.as_posix(),
        "sha256": _hash(path),
        "bytes": path.stat().st_size,
        "metadata": metadata,
    }


def _rollout_values(
    rollout: TensorRollout,
    cost: torch.Tensor,
    diagnostics: dict[str, torch.Tensor],
    *,
    controls_knots: torch.Tensor,
) -> dict[str, object]:
    values: dict[str, object] = {
        "time_s": rollout.time_s,
        "drone_positions_m": rollout.drone_positions_m,
        "drone_velocities_m_s": rollout.drone_velocities_m_s,
        "attachment_positions_m": rollout.attachment_positions_m,
        "cable_positions_m": rollout.cable_positions_m,
        "cable_velocities_m_s": rollout.cable_velocities_m_s,
        "accelerations_m_s2": rollout.accelerations_m_s2,
        "control_knots_m_s2": controls_knots,
        "cost": cost,
    }
    values.update({f"diagnostic__{name}": value for name, value in diagnostics.items()})
    return values


def _step_components(
    simulator: WhipSimulator,
    state: DroneCableState,
    next_drone_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, DderState]:
    """Return the reference undamped velocity, damping solution, and full step."""

    model = simulator.snapshot.model
    constants = model.runtime_constants(state.cable.positions_m)
    q = state.cable.positions_m
    v = state.cable.velocities_m_s
    dt = simulator.settings.simulation_dt_s
    dt_tensor = torch.full(
        (q.shape[0],), dt, dtype=simulator.dtype, device=simulator.device
    )
    boundary = next_drone_position[:, None] + torch.tensor(
        (0.0, 0.0, -simulator.settings.attachment_drop_m),
        dtype=simulator.dtype,
        device=simulator.device,
    )
    boundary_velocity = (boundary - q[:, :1]) / dt_tensor[:, None, None]
    substep_dt = (dt_tensor / model.parameters.substeps)[:, None, None]
    force = model._runtime_internal_force(q, constants, None, None)
    acceleration = force / constants.masses_kg[None, :, None] + constants.gravity_m_s2
    undamped = (v + substep_dt * acceleration) * torch.exp(
        -constants.external_drag_s_inv[:, None, None] * substep_dt
    )
    damped = dder._implicit_bending_damping_velocity(
        q,
        undamped,
        boundary_velocity,
        constants.rest_lengths_m,
        constants.masses_kg,
        substep_dt[:, 0, 0],
        constants.bending_damping_n_m2_s,
        None,
        None,
        conjugate_gradient_iterations=60,
        pinned_endpoints=START_PINNED_FREE_END,
    )
    output = model.step_runtime(
        state.cable,
        boundary,
        dt_tensor,
        constants,
        iterative_damping=True,
        pinned_endpoints=START_PINNED_FREE_END,
    )
    return undamped, damped, output


def _state_at(rollout: TensorRollout, frame: int) -> DroneCableState:
    return DroneCableState(
        rollout.drone_positions_m[:, frame],
        rollout.drone_velocities_m_s[:, frame],
        DderState(
            rollout.cable_positions_m[:, frame],
            rollout.cable_velocities_m_s[:, frame],
        ),
    )


def _load_backward_knots(target_horizon_s: float, target_count: int) -> np.ndarray:
    matches = list(BACKWARD_GLOB)
    if not matches:
        raise FileNotFoundError("Continuation/backward-whip reference trajectory was not found.")
    with np.load(matches[0]) as archive:
        for key in ("optimized_knots_m_s2", "control_knots_m_s2", "nominal_knots_m_s2"):
            if key in archive:
                values = np.asarray(archive[key], dtype=np.float32)
                if values.ndim == 3:
                    values = values[-1]
                return resample_knots_in_time(values, 2.0, target_horizon_s, target_count)
        if "accelerations_m_s2" in archive:
            # Older continuation artifacts persisted the interpolated control
            # sequence rather than its knot parameterization.  Sampling that
            # deterministic sequence on the new knot grid preserves the
            # archived maneuver over the shorter reference horizon.
            values = np.asarray(archive["accelerations_m_s2"], dtype=np.float32)
            return resample_knots_in_time(values, 2.0, target_horizon_s, target_count)
    raise ValueError(f"No knots in {matches[0]}")


def freeze(profile_path: Path, output: Path) -> Path:
    workload = load_workload(profile_path)
    if not torch.cuda.is_available():
        raise RuntimeError("The authoritative reference must be generated on CUDA.")
    output.mkdir(parents=True, exist_ok=True)
    simulator = WhipSimulator(workload.controller, workload.simulation, device="cuda")
    initial = simulator.initial_state(workload.initial_xyz)
    settings = workload.mppi
    count = simulator.settings.control_count
    maximum = simulator.settings.maximum_acceleration_m_s2
    nominal_knots = torch.tensor(
        workload.warm_knots, dtype=simulator.dtype, device=simulator.device
    )
    backward_knots = torch.tensor(
        _load_backward_knots(workload.simulation.horizon_s, settings.knot_count),
        dtype=simulator.dtype,
        device=simulator.device,
    )

    generator = torch.Generator(device=simulator.device)
    generator.manual_seed(settings.seed)
    noise = mppi_module._sample_perturbations(
        settings.samples,
        settings.knot_count,
        settings.acceleration_noise_sigma_m_s2,
        generator=generator,
        dtype=simulator.dtype,
        device=simulator.device,
    )
    candidates = mppi_module._bound_vectors(nominal_knots[None] + noise, maximum)
    candidate_controls = interpolate_control_knots(candidates, count, maximum)

    # This is exactly the first production MPPI iteration with a fixed seed.
    candidate_rollout = simulator.rollout(initial, candidate_controls, create_graph=False)
    costs, diagnostics = mppi_module._mppi_event_objective(
        candidate_rollout, initial, workload.problem, simulator, settings
    )
    minimum = torch.min(costs)
    importance = torch.exp(-(costs - minimum) / settings.temperature)
    weights = importance / torch.clamp(torch.sum(importance), min=1.0e-30)
    weighted_update = mppi_module._bound_vectors(
        nominal_knots + torch.sum(weights[:, None, None] * (candidates - nominal_knots[None]), dim=0),
        maximum,
    )
    best_index = int(torch.argmin(costs).detach().cpu())

    feasible = diagnostics["feasible"]
    non_tip = diagnostics["non_tip_contact_violation"] > 0
    infeasible_indices = torch.nonzero(~feasible, as_tuple=False).flatten()
    if len(infeasible_indices):
        near_index = int(
            infeasible_indices[
                torch.argmin(diagnostics["position_error_m"][infeasible_indices])
            ].detach().cpu()
        )
    else:
        near_index = int(torch.argsort(costs)[1].detach().cpu())
    non_tip_indices = torch.nonzero(non_tip, as_tuple=False).flatten()
    non_tip_index = (
        int(non_tip_indices[torch.argmin(costs[non_tip_indices])].detach().cpu())
        if len(non_tip_indices)
        else int(torch.argmax(diagnostics["non_tip_contact_violation"]).detach().cpu())
    )
    high_index = int(torch.argmax(costs).detach().cpu())
    random_index = min(17, settings.samples - 1)

    files: list[dict[str, object]] = []
    population_path = output / "mppi_iteration_seeded.npz"
    population_values: dict[str, object] = {
        "nominal_knots_m_s2": nominal_knots,
        "sampled_noise_m_s2": noise,
        "candidate_knots_m_s2": candidates,
        "candidate_costs": costs,
        "importance_weights": weights,
        "weighted_nominal_knots_m_s2": weighted_update,
        "best_sample_index": best_index,
        "best_sample_knots_m_s2": candidates[best_index],
    }
    population_values.update({f"diagnostic__{name}": value for name, value in diagnostics.items()})
    files.append(_save_npz(population_path, population_values))

    named_knots: dict[str, torch.Tensor] = {
        "forward_recoil_nominal": nominal_knots,
        "continuation_backward": backward_knots,
        "near_miss": candidates[near_index],
        "non_tip_first": candidates[non_tip_index],
        "random_candidate": candidates[random_index],
        "high_cost_candidate": candidates[high_index],
        "best_sample": candidates[best_index],
        "weighted_nominal": weighted_update,
    }
    nominal_rollout: TensorRollout | None = None
    for name, knots in named_knots.items():
        controls = interpolate_control_knots(knots[None], count, maximum)
        rollout = simulator.rollout(initial, controls, create_graph=False)
        rollout_cost, rollout_diagnostics = mppi_module._mppi_event_objective(
            rollout, initial, workload.problem, simulator, settings
        )
        if name == "forward_recoil_nominal":
            nominal_rollout = rollout
        files.append(
            _save_npz(
                output / f"rollout__{name}.npz",
                _rollout_values(
                    rollout,
                    rollout_cost,
                    rollout_diagnostics,
                    controls_knots=knots,
                ),
            )
        )

    assert nominal_rollout is not None
    disturbed = perturb_cable_state(
        _state_at(nominal_rollout, 0),
        profile="interior",
        velocity_delta_m_s=(0.0, 0.40, 0.0),
    )
    disturbed_controls = interpolate_control_knots(nominal_knots[None], count, maximum)
    disturbed_rollout = simulator.rollout(disturbed, disturbed_controls, create_graph=False)
    disturbed_cost, disturbed_diagnostics = mppi_module._mppi_event_objective(
        disturbed_rollout, initial, workload.problem, simulator, settings
    )
    files.append(
        _save_npz(
            output / "rollout__hidden_interior_velocity.npz",
            _rollout_values(
                disturbed_rollout,
                disturbed_cost,
                disturbed_diagnostics,
                controls_knots=nominal_knots,
            ),
        )
    )

    # Seven deterministic phase states. Reversal is identified from the root
    # X-velocity maximum, so it remains tied to the actual reference maneuver.
    root_x = nominal_rollout.drone_velocities_m_s[0, :, 0]
    reversal = int(torch.argmax(root_x).detach().cpu())
    closest = int(
        torch.argmin(
            torch.linalg.vector_norm(
                nominal_rollout.cable_positions_m[0, :, -1]
                - torch.tensor(
                    workload.problem.target_position_m,
                    dtype=simulator.dtype,
                    device=simulator.device,
                ),
                dim=1,
            )
        ).detach().cpu()
    )
    phase_frames = {
        "nominal_hanging": 0,
        "forward_stroke": max(1, reversal // 2),
        "near_reversal": max(1, reversal),
        "after_reversal": min(count, reversal + 3),
        "distal_lash": min(count, max(reversal + 6, closest - 4)),
        "near_target_impact": min(count, max(1, closest - 1)),
    }
    for name, frame in phase_frames.items():
        state = _state_at(nominal_rollout, frame)
        control_index = min(frame, count - 1)
        acceleration = nominal_rollout.accelerations_m_s2[:, control_index]
        dt = simulator.settings.simulation_dt_s
        next_drone = (
            state.drone_position_m
            + dt * state.drone_velocity_m_s
            + 0.5 * dt * dt * acceleration
        )
        undamped, damped, output_state = _step_components(simulator, state, next_drone)
        rest = simulator.snapshot.model.runtime_constants(state.cable.positions_m).rest_lengths_m
        edges = output_state.positions_m[:, 1:] - output_state.positions_m[:, :-1]
        residual = torch.linalg.vector_norm(edges, dim=2) - rest[None]
        files.append(
            _save_npz(
                output / f"step__{name}.npz",
                {
                    "frame": frame,
                    "input_drone_position_m": state.drone_position_m,
                    "input_drone_velocity_m_s": state.drone_velocity_m_s,
                    "input_positions_m": state.cable.positions_m,
                    "input_velocities_m_s": state.cable.velocities_m_s,
                    "control_acceleration_m_s2": acceleration,
                    "next_drone_position_m": next_drone,
                    "undamped_velocity_m_s": undamped,
                    "damping_solution_velocity_m_s": damped,
                    "output_positions_m": output_state.positions_m,
                    "output_velocities_m_s": output_state.velocities_m_s,
                    "edge_length_residual_m": residual,
                },
            )
        )

    hidden_state = disturbed
    hidden_acceleration = nominal_rollout.accelerations_m_s2[:, 0]
    dt = simulator.settings.simulation_dt_s
    hidden_next_drone = (
        hidden_state.drone_position_m
        + dt * hidden_state.drone_velocity_m_s
        + 0.5 * dt * dt * hidden_acceleration
    )
    undamped, damped, output_state = _step_components(
        simulator, hidden_state, hidden_next_drone
    )
    files.append(
        _save_npz(
            output / "step__hidden_interior_velocity.npz",
            {
                "input_drone_position_m": hidden_state.drone_position_m,
                "input_drone_velocity_m_s": hidden_state.drone_velocity_m_s,
                "input_positions_m": hidden_state.cable.positions_m,
                "input_velocities_m_s": hidden_state.cable.velocities_m_s,
                "control_acceleration_m_s2": hidden_acceleration,
                "next_drone_position_m": hidden_next_drone,
                "undamped_velocity_m_s": undamped,
                "damping_solution_velocity_m_s": damped,
                "output_positions_m": output_state.positions_m,
                "output_velocities_m_s": output_state.velocities_m_s,
            },
        )
    )

    torch.cuda.synchronize()
    manifest = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "profile_path": profile_path.resolve().as_posix(),
        "profile_sha256": _hash(profile_path.resolve()),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "workload": {
            "nodes": workload.controller.node_count,
            "horizon_s": workload.simulation.horizon_s,
            "physics_dt_s": workload.simulation.simulation_dt_s,
            "control_interval_s": workload.simulation.control_interval_s,
            "control_count": count,
            "samples": settings.samples,
            "iterations": settings.iterations,
            "knots": settings.knot_count,
            "seed": settings.seed,
            "simulation": asdict(workload.simulation),
            "mppi": asdict(settings),
        },
        "selected_candidate_indices": {
            "best": best_index,
            "near_miss": near_index,
            "non_tip_first": non_tip_index,
            "random": random_index,
            "high_cost": high_index,
        },
        "phase_frames": phase_frames,
        "files": files,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    manifest = freeze(arguments.profile, arguments.output)
    print(manifest.resolve())


if __name__ == "__main__":
    main()
