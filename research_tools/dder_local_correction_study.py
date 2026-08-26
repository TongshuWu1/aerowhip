"""Evaluate local DDER control gradients for two established whip families.

This is a diagnostic study only.  It imports the existing simulator, smooth
DDER surrogate, and hard MPPI evaluator without changing their definitions.
Short-horizon gradients begin H seconds before each nominal candidate impact,
so they represent receding local correction of an already-established whip.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as functional

from cable_twin.shared.dder import DderState
from drone_mpc.model import load_cable_model
from drone_mpc.problem import MpcProblem
from drone_mpc.mppi import (
    evaluate_mppi_rollout,
    interpolate_control_knots,
    smooth_strike_surrogate,
)
from drone_mpc.perfect_model import PerfectMpcSettings
from drone_mpc.simulator import DroneCableState, WhipSimulator
from optitrack_offline.config import DEFAULT_MODEL_PATH
from research_tools.mppi_ablation import resample_knots_in_time


SCHEMA = "dder_two_family_local_correction_v1"
HORIZONS_S = (0.1, 0.2, 0.3, 0.5, 1.0, 2.0)
CORRECTION_HORIZONS_S = (0.1, 0.2, 0.3, 0.5, 2.0)
ALPHA_SIGMA_RATIOS = (0.005, 0.010, 0.025)

FORWARD_SOURCES = (
    (
        "forward_success_seed11",
        Path(
            "data/drone_mpc/ablations/mppi_discovery_pilot/"
            "initialization_forward_recoil__T2__k11__vg0_12__pw0__"
            "pilot__i6__n128__b128__seed11.npz"
        ),
        "successful_seed",
    ),
    (
        "forward_success_seed17",
        Path(
            "data/drone_mpc/ablations/mppi_discovery_pilot/"
            "initialization_forward_recoil__T2__k11__vg0_12__pw0__"
            "pilot__i6__n128__b128__seed17.npz"
        ),
        "representative_success",
    ),
    (
        "forward_success_seed29",
        Path(
            "data/drone_mpc/ablations/mppi_discovery_pilot/"
            "initialization_forward_recoil__T2__k11__vg0_12__pw0__"
            "pilot__i6__n128__b128__seed29.npz"
        ),
        "successful_seed",
    ),
    (
        "forward_near_seed17",
        Path(
            "data/drone_mpc/ablations/mppi_far_fast/"
            "initialization_forward_recoil_seed_17.npz"
        ),
        "near_success",
    ),
)

BACKWARD_SOURCES = (
    (
        "backward_continuation_success",
        Path("data/drone_mpc/perfect_model_mppi_farther_faster.npz"),
        "representative_success",
    ),
    (
        "backward_near_seed11",
        Path(
            "data/drone_mpc/ablations/mppi_discovery_pilot/"
            "initialization_backward_forward__T2__k11__vg0_12__pw0__"
            "pilot__i6__n128__b128__seed11.npz"
        ),
        "near_success",
    ),
    (
        "backward_near_seed17",
        Path(
            "data/drone_mpc/ablations/mppi_discovery_pilot/"
            "initialization_backward_forward__T2__k11__vg0_12__pw0__"
            "pilot__i6__n128__b128__seed17.npz"
        ),
        "near_success",
    ),
    (
        "backward_near_seed29",
        Path(
            "data/drone_mpc/ablations/mppi_discovery_pilot/"
            "initialization_backward_forward__T2__k11__vg0_12__pw0__"
            "pilot__i6__n128__b128__seed29.npz"
        ),
        "near_success",
    ),
)


@dataclass(frozen=True, slots=True)
class TrajectoryCase:
    family: str
    label: str
    kind: str
    source: str
    reversal_time_s: float
    controls_m_s2: np.ndarray


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    names = sorted({name for row in rows for name in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _metadata(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve().with_suffix(".json")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata is not an object: {resolved}")
    return payload


def _settings(payload: dict[str, object]) -> PerfectMpcSettings:
    raw = payload.get("settings")
    if not isinstance(raw, dict):
        raise ValueError("Replay has no settings object.")
    known = {item.name for item in fields(PerfectMpcSettings)}
    return PerfectMpcSettings(**{key: value for key, value in raw.items() if key in known})


def _problem(payload: dict[str, object], minimum_speed: float) -> MpcProblem:
    raw = payload.get("problem")
    if not isinstance(raw, dict):
        raise ValueError("Replay has no problem object.")
    return replace(MpcProblem(**raw), minimum_impact_speed_m_s=minimum_speed)


def _source_knots(payload: dict[str, object]) -> tuple[np.ndarray, float]:
    controls = payload.get("control_parameterization")
    settings = payload.get("settings")
    if not isinstance(controls, dict) or not isinstance(settings, dict):
        raise ValueError("Replay has no saved control knots.")
    return (
        np.asarray(controls["knots_m_s2"], dtype=np.float32),
        float(settings["horizon_s"]),
    )


def _resampled_case(
    family: str,
    label: str,
    source: Path,
    kind: str,
) -> TrajectoryCase:
    payload = _metadata(source)
    archive = np.load(source.expanduser().resolve())
    controls = np.asarray(archive["accelerations_m_s2"], dtype=np.float32)
    required_controls = int(round(2.0 / 0.02))
    if controls.shape[1:] != (3,):
        raise ValueError(f"Replay does not contain 3-D controls: {source}")
    if controls.shape[0] < required_controls:
        controls = np.pad(
            controls,
            ((0, required_controls - controls.shape[0]), (0, 0)),
            mode="constant",
        )
    return TrajectoryCase(
        family=family,
        label=label,
        kind=kind,
        source=str(source.expanduser().resolve()),
        reversal_time_s=float(payload.get("injection_end_s", 0.4)),
        # Use the exact saved actuator sequence.  Reparameterizing a successful
        # whip through a different knot grid measurably changes its phase.
        controls_m_s2=controls[:required_controls].copy(),
    )


def _time_warp_reversal(
    knots: np.ndarray,
    old_reversal_s: float,
    new_reversal_s: float,
    horizon_s: float = 2.0,
) -> np.ndarray:
    old_reversal = float(np.clip(old_reversal_s, 0.05, horizon_s - 0.05))
    new_reversal = float(np.clip(new_reversal_s, 0.05, horizon_s - 0.05))
    target_time = np.linspace(0.0, horizon_s, knots.shape[0])
    source_time = np.where(
        target_time <= new_reversal,
        target_time * old_reversal / new_reversal,
        old_reversal
        + (target_time - new_reversal)
        * (horizon_s - old_reversal)
        / (horizon_s - new_reversal),
    )
    original_time = np.linspace(0.0, horizon_s, knots.shape[0])
    return np.stack(
        [np.interp(source_time, original_time, knots[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)


def _scale_phase(
    knots: np.ndarray,
    direction: np.ndarray,
    reversal_time_s: float,
    *,
    phase: str,
    scale: float,
) -> np.ndarray:
    result = np.asarray(knots, dtype=np.float64).copy()
    time_s = np.linspace(0.0, 2.0, result.shape[0])
    parallel = result @ direction
    lateral = result - parallel[:, None] * direction[None]
    if phase == "stroke":
        selected = time_s <= reversal_time_s
    elif phase == "recoil":
        selected = time_s > reversal_time_s
    else:
        raise ValueError(f"Unknown phase: {phase}")
    parallel[selected] *= scale
    result = lateral + parallel[:, None] * direction[None]
    return result.astype(np.float32)


def _bound_numpy_vectors(values: np.ndarray, maximum: float = 20.0) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    scale = np.minimum(1.0, maximum / np.maximum(norms, 1.0e-12))
    return (result * scale).astype(np.float32)


def build_cases(problem: MpcProblem) -> list[TrajectoryCase]:
    cases = [
        _resampled_case("forward_recoil", label, path, kind)
        for label, path, kind in FORWARD_SOURCES
    ]
    cases.extend(
        _resampled_case("backward_continuation", label, path, kind)
        for label, path, kind in BACKWARD_SOURCES
    )
    direction = np.asarray(problem.impact_direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    representatives = {
        family: next(
            case
            for case in cases
            if case.family == family and case.kind == "representative_success"
        )
        for family in ("forward_recoil", "backward_continuation")
    }
    for family, base in representatives.items():
        variants = (
            (
                "reversal_early_40ms",
                "timing_perturbation",
                _time_warp_reversal(
                    base.controls_m_s2,
                    base.reversal_time_s,
                    base.reversal_time_s - 0.04,
                ),
            ),
            (
                "reversal_late_40ms",
                "timing_perturbation",
                _time_warp_reversal(
                    base.controls_m_s2,
                    base.reversal_time_s,
                    base.reversal_time_s + 0.04,
                ),
            ),
            (
                "stroke_amplitude_85pct",
                "stroke_amplitude_perturbation",
                _scale_phase(
                    base.controls_m_s2,
                    direction,
                    base.reversal_time_s,
                    phase="stroke",
                    scale=0.85,
                ),
            ),
            (
                "recoil_amplitude_30pct",
                "recoil_amplitude_perturbation",
                _scale_phase(
                    base.controls_m_s2,
                    direction,
                    base.reversal_time_s,
                    phase="recoil",
                    scale=0.30,
                ),
            ),
            (
                "knot_magnitude_93pct",
                "knot_magnitude_perturbation",
                (0.93 * base.controls_m_s2).astype(np.float32),
            ),
            (
                "knot_magnitude_99pct",
                "knot_magnitude_perturbation",
                (0.99 * base.controls_m_s2).astype(np.float32),
            ),
        )
        for suffix, kind, knots in variants:
            cases.append(
                TrajectoryCase(
                    family=family,
                    label=f"{family}_{suffix}",
                    kind=kind,
                    source=base.source,
                    reversal_time_s=base.reversal_time_s,
                    controls_m_s2=_bound_numpy_vectors(knots),
                )
            )
    return cases


def _bound_vectors(values: torch.Tensor, maximum: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    return values * torch.clamp(
        maximum / torch.clamp(norms, min=1.0e-12), max=1.0
    )


def _cuda_time(function):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    result = function()
    end.record()
    torch.cuda.synchronize()
    return result, time.perf_counter() - wall_start, start.elapsed_time(end) * 1.0e-3


def _metrics_rows(
    rollout,
    initial_state: DroneCableState,
    problem: MpcProblem,
    simulator: WhipSimulator,
    settings,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    with torch.no_grad():
        real_cost, diagnostics, impact_frames = evaluate_mppi_rollout(
            rollout, initial_state, problem, simulator, settings
        )
    rows: list[dict[str, object]] = []
    for index in range(real_cost.shape[0]):
        geometric = bool(diagnostics["geometric_tip_contact"][index].cpu())
        non_tip = bool(diagnostics["non_tip_contact_violation"][index].cpu() > 0.0)
        rows.append(
            {
                "valid_strike": bool(diagnostics["feasible"][index].cpu()),
                "tip_first": geometric and not non_tip,
                "geometric_tip_contact": geometric,
                "non_tip_first_failure": non_tip,
                "target_error_m": float(diagnostics["position_error_m"][index].cpu()),
                "directed_tip_speed_m_s": float(
                    diagnostics["directional_speed_m_s"][index].cpu()
                ),
                "total_tip_speed_m_s": float(
                    diagnostics["tip_speed_m_s"][index].cpu()
                ),
                "direction_error_deg": float(
                    diagnostics["direction_error_deg"][index].cpu()
                ),
                "impact_time_s": float(diagnostics["impact_time_s"][index].cpu()),
                "non_tip_clearance_m": float(
                    diagnostics["minimum_non_tip_target_distance_m"][index].cpu()
                    - problem.maximum_tip_error_m
                ),
                "maximum_drone_speed_m_s": float(
                    diagnostics["maximum_drone_speed_m_s"][index].cpu()
                ),
                "maximum_drone_displacement_m": float(
                    diagnostics["maximum_drone_excursion_m"][index].cpu()
                ),
                "real_mppi_cost": float(real_cost[index].cpu()),
            }
        )
    return rows, impact_frames


def _failure_mode(row: dict[str, object], problem: MpcProblem) -> str:
    if bool(row["valid_strike"]):
        return "valid"
    if bool(row["non_tip_first_failure"]):
        return "non_tip_first"
    if float(row["target_error_m"]) > problem.maximum_tip_error_m:
        return "position_miss"
    if float(row["directed_tip_speed_m_s"]) < problem.minimum_impact_speed_m_s:
        return "wrong_speed"
    if float(row["direction_error_deg"]) > problem.maximum_impact_angle_deg:
        return "wrong_direction"
    return "safety_failure"


def _local_state(rollout, batch_index: int, frame: int) -> DroneCableState:
    return DroneCableState(
        rollout.drone_positions_m[batch_index : batch_index + 1, frame],
        rollout.drone_velocities_m_s[batch_index : batch_index + 1, frame],
        DderState(
            rollout.cable_positions_m[batch_index : batch_index + 1, frame],
            rollout.cable_velocities_m_s[batch_index : batch_index + 1, frame],
        ),
    )


def run_scaling(
    output: Path,
    snapshot,
    base_settings: PerfectMpcSettings,
    problem: MpcProblem,
    reference: TrajectoryCase,
    device: str,
) -> list[dict[str, object]]:
    path = output / "gradient_runtime_scaling.json"
    rows: list[dict[str, object]] = []
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows = loaded
    complete = {float(row["horizon_s"]) for row in rows}
    # Warm CUDA/autograd operator caches with the smallest exact problem.
    for horizon in HORIZONS_S:
        if horizon in complete:
            continue
        settings = replace(base_settings, horizon_s=horizon, mppi_knot_count=16)
        simulator = WhipSimulator(snapshot, settings.simulation_settings(), device=device)
        state = simulator.initial_state(problem.drone_workspace_center_m)
        knots_np = resample_knots_in_time(
            reference.controls_m_s2, 2.0, horizon, 16
        )
        nominal = torch.tensor(
            knots_np, dtype=simulator.dtype, device=simulator.device
        ).requires_grad_(True)
        controls = interpolate_control_knots(
            nominal[None],
            simulator.settings.control_count,
            simulator.settings.maximum_acceleration_m_s2,
        )
        torch.cuda.reset_peak_memory_stats()
        rollout, forward_wall, forward_gpu = _cuda_time(
            lambda: simulator.rollout(state, controls, create_graph=True)
        )
        cost_and_terms, surrogate_wall, surrogate_gpu = _cuda_time(
            lambda: smooth_strike_surrogate(
                rollout, problem, settings.mppi_settings()
            )
        )
        smooth_cost, _terms = cost_and_terms
        gradient, backward_wall, backward_gpu = _cuda_time(
            lambda: torch.autograd.grad(
                smooth_cost.sum(), nominal, create_graph=False, retain_graph=False
            )[0]
        )
        physics_steps = int(round(horizon / settings.physics_dt_s))
        substeps = snapshot.model.parameters.substeps
        length_solves = (
            physics_steps
            * substeps
            * snapshot.model.parameters.constraint_iterations
        )
        velocity_solves = physics_steps * substeps
        row = {
            "schema": SCHEMA,
            "horizon_s": horizon,
            "knot_count": 16,
            "control_dimension": 48,
            "physics_steps": physics_steps,
            "dder_substeps": substeps,
            "differentiable_forward_wall_s": forward_wall + surrogate_wall,
            "differentiable_forward_gpu_s": forward_gpu + surrogate_gpu,
            "rollout_graph_wall_s": forward_wall,
            "surrogate_wall_s": surrogate_wall,
            "backward_wall_s": backward_wall,
            "backward_gpu_s": backward_gpu,
            "total_gradient_wall_s": forward_wall + surrogate_wall + backward_wall,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            "internal_bending_force_autograd_calls": physics_steps * substeps,
            "length_projection_solves": length_solves,
            "velocity_projection_solves": velocity_solves,
            "total_projection_solves": length_solves + velocity_solves,
            "outer_scalar_autograd_calls": 1,
            "gradient_finite": bool(torch.all(torch.isfinite(gradient)).cpu()),
        }
        rows.append(row)
        rows.sort(key=lambda value: float(value["horizon_s"]))
        path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        _write_csv(output / "gradient_runtime_scaling.csv", rows)
        print(
            f"scaling H={horizon:g}s: forward={row['differentiable_forward_wall_s']:.3f}s "
            f"backward={backward_wall:.3f}s total={row['total_gradient_wall_s']:.3f}s"
        )
        del rollout, smooth_cost, gradient, nominal, controls, cost_and_terms
        torch.cuda.empty_cache()
    return rows


def _case_baselines(
    cases: list[TrajectoryCase],
    simulator: WhipSimulator,
    state: DroneCableState,
    problem: MpcProblem,
    settings,
):
    controls = torch.tensor(
        np.stack([case.controls_m_s2 for case in cases]),
        dtype=simulator.dtype,
        device=simulator.device,
    )
    with torch.no_grad():
        rollout = simulator.rollout(state, controls, create_graph=False)
        smooth, _ = smooth_strike_surrogate(rollout, problem, settings)
    metrics, impact_frames = _metrics_rows(
        rollout, state, problem, simulator, settings
    )
    rows: list[dict[str, object]] = []
    for index, (case, metric) in enumerate(zip(cases, metrics, strict=True)):
        row = {
            "family": case.family,
            "case": case.label,
            "kind": case.kind,
            "source": case.source,
            "reversal_time_s": case.reversal_time_s,
            "smooth_surrogate_cost": float(smooth[index].cpu()),
            "impact_frame": int(impact_frames[index].cpu()),
        }
        row.update(metric)
        row["failure_mode"] = _failure_mode(row, problem)
        rows.append(row)
    return controls, rollout, impact_frames, rows


def _stable_seed(*values: object) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def run_local_corrections(
    output: Path,
    cases: list[TrajectoryCase],
    snapshot,
    base_settings: PerfectMpcSettings,
    problem: MpcProblem,
    device: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    settings = replace(base_settings, horizon_s=2.0, mppi_knot_count=16)
    simulator = WhipSimulator(snapshot, settings.simulation_settings(), device=device)
    state = simulator.initial_state(problem.drone_workspace_center_m)
    controls, rollout, impact_frames, baseline_rows = _case_baselines(
        cases, simulator, state, problem, settings.mppi_settings()
    )
    (output / "trajectory_test_set.json").write_text(
        json.dumps(baseline_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / "trajectory_test_set.csv", baseline_rows)

    checkpoint = output / "local_correction_candidates.json"
    rows: list[dict[str, object]] = []
    if checkpoint.is_file():
        loaded = json.loads(checkpoint.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            rows = loaded
    complete = {
        (str(row["case"]), float(row["gradient_horizon_s"])) for row in rows
    }

    for case_index, case in enumerate(cases):
        for horizon in CORRECTION_HORIZONS_S:
            key = (case.label, horizon)
            if key in complete:
                print(f"skip local correction {case.label}, H={horizon:g}s")
                continue
            local_control_count = int(round(horizon / settings.control_interval_s))
            local_physics_steps = int(round(horizon / settings.physics_dt_s))
            impact_control = int(
                math.ceil(
                    int(impact_frames[case_index].cpu())
                    / simulator.settings.steps_per_control
                )
            )
            if horizon >= 2.0 - 1.0e-9:
                start_control = 0
            else:
                start_control = max(0, impact_control - local_control_count)
                start_control = min(
                    start_control,
                    simulator.settings.control_count - local_control_count,
                )
            start_frame = start_control * simulator.settings.steps_per_control
            local_state = _local_state(rollout, case_index, start_frame)
            base_segment = controls[
                case_index : case_index + 1,
                start_control : start_control + local_control_count,
            ].detach()
            full_knot_spacing = 2.0 / 15.0
            local_knot_count = max(2, int(round(horizon / full_knot_spacing)) + 1)
            local_settings = replace(
                base_settings,
                horizon_s=horizon,
                mppi_knot_count=local_knot_count,
            )
            local_simulator = WhipSimulator(
                snapshot, local_settings.simulation_settings(), device=device
            )
            delta_knots = torch.zeros(
                (local_knot_count, 3),
                dtype=simulator.dtype,
                device=simulator.device,
                requires_grad=True,
            )

            def local_controls_from_delta(delta: torch.Tensor) -> torch.Tensor:
                interpolated = functional.interpolate(
                    delta[None].transpose(1, 2),
                    size=local_control_count,
                    mode="linear",
                    align_corners=True,
                ).transpose(1, 2)
                return _bound_vectors(
                    base_segment + interpolated,
                    simulator.settings.maximum_acceleration_m_s2,
                )

            torch.cuda.reset_peak_memory_stats()
            corrected_controls = local_controls_from_delta(delta_knots)
            local_rollout, forward_wall, _forward_gpu = _cuda_time(
                lambda: local_simulator.rollout(
                    local_state, corrected_controls, create_graph=True
                )
            )
            cost_and_terms, surrogate_wall, _surrogate_gpu = _cuda_time(
                lambda: smooth_strike_surrogate(
                    local_rollout, problem, local_settings.mppi_settings()
                )
            )
            local_cost, _local_terms = cost_and_terms
            gradient, backward_wall, _backward_gpu = _cuda_time(
                lambda: torch.autograd.grad(
                    local_cost.sum(),
                    delta_knots,
                    create_graph=False,
                    retain_graph=False,
                )[0]
            )
            norm = float(torch.linalg.vector_norm(gradient).cpu())
            finite = bool(torch.all(torch.isfinite(gradient)).cpu())
            valid_gradient = finite and norm > 1.0e-10
            direction = (
                gradient / (torch.linalg.vector_norm(gradient) + 1.0e-10)
                if valid_gradient
                else torch.zeros_like(gradient)
            )
            candidates: list[tuple[str, float, float, torch.Tensor]] = [
                ("nominal", 0.0, 0.0, torch.zeros_like(delta_knots))
            ]
            generator = torch.Generator(device=simulator.device)
            generator.manual_seed(_stable_seed(case.label, horizon))
            dimension = local_knot_count * 3
            for ratio in ALPHA_SIGMA_RATIOS:
                alpha = (
                    ratio
                    * local_settings.mppi_noise_sigma_m_s2
                    * math.sqrt(dimension)
                )
                random_direction = torch.randn(
                    delta_knots.shape,
                    dtype=delta_knots.dtype,
                    device=delta_knots.device,
                    generator=generator,
                )
                random_direction = random_direction / (
                    torch.linalg.vector_norm(random_direction) + 1.0e-10
                )
                candidates.extend(
                    (
                        ("negative_gradient", ratio, alpha, -alpha * direction),
                        ("positive_gradient", ratio, alpha, alpha * direction),
                        ("random_equal_norm", ratio, alpha, alpha * random_direction),
                    )
                )

            local_candidate_controls = torch.cat(
                [local_controls_from_delta(delta.detach()) for *_head, delta in candidates],
                dim=0,
            )
            full_candidate_controls = controls[case_index : case_index + 1].repeat(
                len(candidates), 1, 1
            )
            full_candidate_controls[
                :, start_control : start_control + local_control_count
            ] = local_candidate_controls
            with torch.no_grad():
                candidate_local_rollout = local_simulator.rollout(
                    local_state,
                    local_candidate_controls,
                    create_graph=False,
                )
                candidate_smooth, _ = smooth_strike_surrogate(
                    candidate_local_rollout,
                    problem,
                    local_settings.mppi_settings(),
                )
                candidate_full_rollout = simulator.rollout(
                    state,
                    full_candidate_controls,
                    create_graph=False,
                )
            hard_rows, _candidate_impacts = _metrics_rows(
                candidate_full_rollout,
                state,
                problem,
                simulator,
                settings.mppi_settings(),
            )
            for candidate_index, (mode, ratio, alpha, _delta) in enumerate(candidates):
                row: dict[str, object] = {
                    "schema": SCHEMA,
                    "family": case.family,
                    "case": case.label,
                    "case_kind": case.kind,
                    "gradient_horizon_s": horizon,
                    "window_start_s": start_control * settings.control_interval_s,
                    "window_end_s": (
                        start_control + local_control_count
                    )
                    * settings.control_interval_s,
                    "local_knot_count": local_knot_count,
                    "correction_mode": mode,
                    "alpha_sigma_ratio": ratio,
                    "alpha_l2_m_s2": alpha,
                    "gradient_norm": norm,
                    "gradient_valid": valid_gradient,
                    "gradient_forward_wall_s": forward_wall + surrogate_wall,
                    "gradient_backward_wall_s": backward_wall,
                    "gradient_total_wall_s": (
                        forward_wall + surrogate_wall + backward_wall
                    ),
                    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                    "smooth_surrogate_cost": float(
                        candidate_smooth[candidate_index].cpu()
                    ),
                }
                row.update(hard_rows[candidate_index])
                row["failure_mode"] = _failure_mode(row, problem)
                rows.append(row)
            complete.add(key)
            checkpoint.write_text(
                json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
            )
            _write_csv(output / "local_correction_candidates.csv", rows)
            print(
                f"local {case.family}/{case.label}, H={horizon:g}s: "
                f"|g|={norm:.3g}, forward={forward_wall + surrogate_wall:.3f}s, "
                f"backward={backward_wall:.3f}s"
            )
            del (
                local_rollout,
                local_cost,
                gradient,
                candidate_local_rollout,
                candidate_full_rollout,
                local_candidate_controls,
                full_candidate_controls,
                cost_and_terms,
            )
            torch.cuda.empty_cache()
    return baseline_rows, rows


def write_test_set_only(
    output: Path,
    cases: list[TrajectoryCase],
    snapshot,
    base_settings: PerfectMpcSettings,
    problem: MpcProblem,
    device: str,
) -> list[dict[str, object]]:
    settings = replace(base_settings, horizon_s=2.0, mppi_knot_count=16)
    simulator = WhipSimulator(snapshot, settings.simulation_settings(), device=device)
    state = simulator.initial_state(problem.drone_workspace_center_m)
    _controls, _rollout, _impacts, rows = _case_baselines(
        cases, simulator, state, problem, settings.mppi_settings()
    )
    (output / "trajectory_test_set.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / "trajectory_test_set.csv", rows)
    return rows


def _percent(numerator: int, denominator: int) -> float | str:
    return 100.0 * numerator / denominator if denominator else ""


def summarize_local(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    families = sorted({str(row["family"]) for row in rows})
    for family in families:
        for horizon in CORRECTION_HORIZONS_S:
            selected = [
                row
                for row in rows
                if row["family"] == family
                and float(row["gradient_horizon_s"]) == horizon
            ]
            if not selected:
                continue
            cases = sorted({str(row["case"]) for row in selected})
            nominals = {
                str(row["case"]): row
                for row in selected
                if row["correction_mode"] == "nominal"
            }
            for mode in (
                "negative_gradient",
                "positive_gradient",
                "random_equal_norm",
            ):
                best: dict[str, dict[str, object]] = {}
                for case in cases:
                    candidates = [
                        row
                        for row in selected
                        if row["case"] == case and row["correction_mode"] == mode
                    ]
                    if candidates:
                        best[case] = min(
                            candidates, key=lambda row: float(row["real_mppi_cost"])
                        )
                paired = [
                    (nominals[case], best[case])
                    for case in cases
                    if case in nominals and case in best
                ]
                invalid = [(a, b) for a, b in paired if not bool(a["valid_strike"])]
                valid = [(a, b) for a, b in paired if bool(a["valid_strike"])]

                def mean_change(name: str) -> float:
                    return float(
                        np.mean([float(b[name]) - float(a[name]) for a, b in paired])
                    )

                summaries.append(
                    {
                        "family": family,
                        "gradient_horizon_s": horizon,
                        "correction_mode": mode,
                        "trajectories": len(paired),
                        "invalid_trajectories": len(invalid),
                        "already_valid_trajectories": len(valid),
                        "invalid_repaired_count": sum(
                            bool(b["valid_strike"]) for _a, b in invalid
                        ),
                        "invalid_repaired_pct": _percent(
                            sum(bool(b["valid_strike"]) for _a, b in invalid),
                            len(invalid),
                        ),
                        "already_valid_improved_count": sum(
                            bool(b["valid_strike"])
                            and float(b["real_mppi_cost"])
                            < float(a["real_mppi_cost"])
                            for a, b in valid
                        ),
                        "already_valid_improved_pct": _percent(
                            sum(
                                bool(b["valid_strike"])
                                and float(b["real_mppi_cost"])
                                < float(a["real_mppi_cost"])
                                for a, b in valid
                            ),
                            len(valid),
                        ),
                        "surrogate_improved_pct": _percent(
                            sum(
                                float(b["smooth_surrogate_cost"])
                                < float(a["smooth_surrogate_cost"])
                                for a, b in paired
                            ),
                            len(paired),
                        ),
                        "real_cost_improved_pct": _percent(
                            sum(
                                float(b["real_mppi_cost"])
                                < float(a["real_mppi_cost"])
                                for a, b in paired
                            ),
                            len(paired),
                        ),
                        "mean_target_error_change_mm": 1000.0
                        * mean_change("target_error_m"),
                        "mean_directed_speed_change_m_s": mean_change(
                            "directed_tip_speed_m_s"
                        ),
                        "mean_direction_error_change_deg": mean_change(
                            "direction_error_deg"
                        ),
                        "mean_non_tip_clearance_change_mm": 1000.0
                        * mean_change("non_tip_clearance_m"),
                        "non_tip_first_failures": sum(
                            bool(b["non_tip_first_failure"]) for _a, b in paired
                        ),
                        "median_gradient_norm": float(
                            np.median([float(a["gradient_norm"]) for a, _b in paired])
                        ),
                        "median_gradient_runtime_s": float(
                            np.median(
                                [float(a["gradient_total_wall_s"]) for a, _b in paired]
                            )
                        ),
                    }
                )
    return summaries


def profile_and_study(arguments: argparse.Namespace) -> Path:
    output = arguments.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_payload = _metadata(
        Path("data/drone_mpc/perfect_model_mppi_farther_faster.npz")
    )
    snapshot = load_cable_model(arguments.model)
    if snapshot.sha256 != reference_payload.get("model_sha256"):
        raise ValueError("Reference replay and cable model hashes do not match.")
    base_settings = replace(
        _settings(reference_payload),
        horizon_s=2.0,
        mppi_knot_count=16,
    )
    problem = _problem(reference_payload, arguments.minimum_impact_speed_m_s)
    cases = build_cases(problem)
    if arguments.mode in {"scaling", "all"}:
        reference = next(
            case for case in cases if case.label == "forward_success_seed17"
        )
        scaling = run_scaling(
            output,
            snapshot,
            base_settings,
            problem,
            reference,
            arguments.device,
        )
    else:
        scaling_path = output / "gradient_runtime_scaling.json"
        scaling = (
            json.loads(scaling_path.read_text(encoding="utf-8"))
            if scaling_path.is_file()
            else []
        )
    if arguments.mode in {"local", "all"}:
        test_set, local_rows = run_local_corrections(
            output,
            cases,
            snapshot,
            base_settings,
            problem,
            arguments.device,
        )
        summaries = summarize_local(local_rows)
        (output / "local_correction_summary.json").write_text(
            json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
        )
        _write_csv(output / "local_correction_summary.csv", summaries)
    elif arguments.mode == "testset":
        test_set = write_test_set_only(
            output,
            cases,
            snapshot,
            base_settings,
            problem,
            arguments.device,
        )
        local_rows = []
        summaries = []
    else:
        test_set = []
        local_rows = []
        summaries = []
    manifest = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": snapshot.sha256,
        "physics_unchanged": True,
        "hard_mppi_objective_unchanged": True,
        "problem": {
            "minimum_impact_speed_m_s": problem.minimum_impact_speed_m_s,
            "maximum_tip_error_m": problem.maximum_tip_error_m,
            "maximum_impact_angle_deg": problem.maximum_impact_angle_deg,
        },
        "horizons_s": list(HORIZONS_S),
        "correction_horizons_s": list(CORRECTION_HORIZONS_S),
        "alpha_sigma_ratios": list(ALPHA_SIGMA_RATIOS),
        "families": {
            "forward_recoil": [case.label for case in cases if case.family == "forward_recoil"],
            "backward_continuation": [
                case.label for case in cases if case.family == "backward_continuation"
            ],
        },
        "runtime_scaling_rows": len(scaling),
        "test_set_rows": len(test_set),
        "local_candidate_rows": len(local_rows),
        "summary_rows": len(summaries),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--minimum-impact-speed-m-s", type=float, default=3.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/drone_mpc/ablations/dder_two_family_local_correction_exact"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("testset", "scaling", "local", "all"),
        default="all",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    output = profile_and_study(arguments)
    print(f"DDER local-correction study: {output}")


if __name__ == "__main__":
    main()
