"""Matched-model, long-horizon optimizer verification for the drone whip.

This module deliberately removes the online-adaptation and estimator questions.
The optimizer and the independently replayed plant use the same immutable full
DDER snapshot and the complete simulated state.  It is therefore a numerical
feasibility test of the controls, objective, constraints, and numerical
optimizer--not a robustness claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .model import CableModelSnapshot
from .mppi import MppiPlan, MppiSettings, evaluate_mppi_rollout, optimize_mppi
from .mpc import (
    CostWeights,
    MpcPlan,
    MpcProblem,
    OptimizerSettings,
    optimize_controls,
    variable_impact_rollout_cost_terms,
)
from .simulator import (
    SimulationResult,
    SimulationSettings,
    WhipSimulator,
    tensor_rollout_to_result,
)


ProgressCallback = Callable[[str], None]
CancellationCallback = Callable[[], bool]
PERFECT_MPC_SCHEMA = "drone_whip_perfect_model_optimizer_v2"


@dataclass(frozen=True, slots=True)
class PerfectMpcSettings:
    """Numerical settings for one open-loop matched-model verification."""

    horizon_s: float = 3.0
    physics_dt_s: float = 0.01
    control_interval_s: float = 0.02
    attachment_drop_m: float = 0.10
    maximum_acceleration_m_s2: float = 20.0
    maximum_speed_m_s: float = 3.0
    solver: str = "mppi"
    mppi_iterations: int = 20
    mppi_samples: int = 128
    mppi_rollout_batch_size: int = 128
    mppi_knot_count: int = 12
    mppi_temperature: float = 1.0
    mppi_noise_sigma_m_s2: float = 8.0
    mppi_noise_decay: float = 0.92
    mppi_objective_stage: str = "full"
    mppi_position_sigma_m: float = 0.25
    mppi_velocity_gate_sigma_m: float = 0.15
    mppi_position_weight: float = 10.0
    mppi_speed_weight: float = 2.0
    mppi_predictive_speed_weight: float = 0.0
    mppi_predictive_velocity_gate_sigma_m: float = 0.45
    mppi_predictive_speed_ratio: float = 0.25
    mppi_direction_weight: float = 10.0
    mppi_success_cost: float = 100.0
    mppi_drone_displacement_weight: float = 2.0
    mppi_safety_weight: float = 100.0
    mppi_gradient_guidance_fraction: float = 0.0
    mppi_gradient_step_sigma_ratio: float = 0.025
    mppi_smooth_softmin_temperature_m: float = 0.05
    mppi_seed: int = 17
    ipopt_iterations: int = 120
    ipopt_tolerance: float = 1.0e-4
    ipopt_acceptable_tolerance: float = 1.0e-3
    ipopt_max_wall_time_s: float = 90.0
    finite_difference_step: float = 1.0e-3
    initial_samples: int = 256
    acceleration_effort_weight: float = 1.0
    acceleration_smoothness_weight: float = 1.0

    def __post_init__(self) -> None:
        # Reuse the production validators rather than maintaining a second set
        # of subtly different timing and solver rules.
        simulation = self.simulation_settings()
        weights = self.cost_weights()
        if self.solver not in {"mppi", "ipopt"}:
            raise ValueError("Perfect-model solver must be 'mppi' or 'ipopt'.")
        if self.solver == "mppi":
            optimizer = self.mppi_settings()
        else:
            optimizer = self.optimizer_settings()
        del simulation, optimizer, weights

    def simulation_settings(self) -> SimulationSettings:
        return SimulationSettings(
            horizon_s=self.horizon_s,
            simulation_dt_s=self.physics_dt_s,
            control_interval_s=self.control_interval_s,
            attachment_drop_m=self.attachment_drop_m,
            maximum_acceleration_m_s2=self.maximum_acceleration_m_s2,
            maximum_speed_m_s=self.maximum_speed_m_s,
        )

    def optimizer_settings(self) -> OptimizerSettings:
        return OptimizerSettings(
            iterations=self.ipopt_iterations,
            tolerance=self.ipopt_tolerance,
            acceptable_tolerance=self.ipopt_acceptable_tolerance,
            maximum_wall_time_s=self.ipopt_max_wall_time_s,
            finite_difference_step=self.finite_difference_step,
            # A solve-once verification does not replan.  This field is still
            # required by the shared optimizer settings type.
            replan_interval_s=self.control_interval_s,
            initial_samples=self.initial_samples,
        )

    def mppi_settings(self) -> MppiSettings:
        return MppiSettings(
            iterations=self.mppi_iterations,
            samples=self.mppi_samples,
            rollout_batch_size=self.mppi_rollout_batch_size,
            knot_count=self.mppi_knot_count,
            temperature=self.mppi_temperature,
            acceleration_noise_sigma_m_s2=self.mppi_noise_sigma_m_s2,
            noise_decay=self.mppi_noise_decay,
            objective_stage=self.mppi_objective_stage,
            position_sigma_m=self.mppi_position_sigma_m,
            velocity_gate_sigma_m=self.mppi_velocity_gate_sigma_m,
            position_weight=self.mppi_position_weight,
            speed_weight=self.mppi_speed_weight,
            predictive_speed_weight=self.mppi_predictive_speed_weight,
            predictive_velocity_gate_sigma_m=(
                self.mppi_predictive_velocity_gate_sigma_m
            ),
            predictive_speed_ratio=self.mppi_predictive_speed_ratio,
            direction_weight=self.mppi_direction_weight,
            success_cost=self.mppi_success_cost,
            drone_displacement_weight=self.mppi_drone_displacement_weight,
            safety_weight=self.mppi_safety_weight,
            gradient_guidance_fraction=self.mppi_gradient_guidance_fraction,
            gradient_step_sigma_ratio=self.mppi_gradient_step_sigma_ratio,
            smooth_softmin_temperature_m=(
                self.mppi_smooth_softmin_temperature_m
            ),
            seed=self.mppi_seed,
        )

    def cost_weights(self) -> CostWeights:
        return CostWeights(
            acceleration_effort=self.acceleration_effort_weight,
            acceleration_smoothness=self.acceleration_smoothness_weight,
        )


@dataclass(frozen=True, slots=True)
class EnergyDiagnostics:
    """Energy and phase traces computed from the independent full-model replay."""

    time_s: np.ndarray
    cable_kinetic_energy_j: np.ndarray
    relative_cable_kinetic_energy_j: np.ndarray
    bending_energy_j: np.ndarray
    gravitational_energy_change_j: np.ndarray
    directed_tip_energy_j: np.ndarray
    drone_forward_displacement_m: np.ndarray
    injection_end_s: float
    recoil_end_s: float

    @property
    def internal_motion_energy_j(self) -> np.ndarray:
        """Cable deformation/motion energy, excluding rigid attachment motion."""

        return self.relative_cable_kinetic_energy_j + self.bending_energy_j


@dataclass(frozen=True, slots=True)
class PerfectMpcResult:
    """Optimizer plan plus a fresh, independently created matched-model replay."""

    plan: MpcPlan | MppiPlan
    prediction: SimulationResult
    terms: dict[str, float]
    impact_frame: int
    energy: EnergyDiagnostics
    maximum_replay_position_difference_m: float
    maximum_replay_velocity_difference_m_s: float
    source_model_sha256: str
    source_model_provisional: bool

    @property
    def feasible(self) -> bool:
        return bool(self.terms["feasible"])

    @property
    def speed_amplification(self) -> float:
        denominator = max(self.terms["maximum_drone_speed_m_s"], 1.0e-9)
        return self.terms["directional_speed_m_s"] / denominator


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


def _finite_history(values: tuple[float, ...]) -> list[float | None]:
    """Represent disabled/invalid diagnostic values as portable JSON nulls."""

    return [float(value) if math.isfinite(value) else None for value in values]


def _energy_diagnostics(
    snapshot: CableModelSnapshot,
    prediction: SimulationResult,
    problem: MpcProblem,
    plan: MpcPlan | MppiPlan,
    settings: PerfectMpcSettings,
) -> EnergyDiagnostics:
    positions = torch.as_tensor(
        np.array(prediction.cable_positions_m, copy=True),
        dtype=torch.float64,
    )
    velocities = torch.as_tensor(
        np.array(prediction.cable_velocities_m_s, copy=True),
        dtype=torch.float64,
    )
    drone_velocities = torch.as_tensor(
        np.array(prediction.drone_velocities_m_s, copy=True),
        dtype=torch.float64,
    )
    masses = torch.as_tensor(
        snapshot.model.parameters.vertex_masses_kg,
        dtype=torch.float64,
    )
    kinetic = 0.5 * torch.sum(
        masses[None, :, None] * velocities.square(), dim=(1, 2)
    )
    relative_velocity = velocities - drone_velocities[:, None]
    relative_kinetic = 0.5 * torch.sum(
        masses[None, :, None] * relative_velocity.square(), dim=(1, 2)
    )
    with torch.no_grad():
        bending = snapshot.model.bending_energy(
            positions,
            bending_stiffness_n_m2=snapshot.bending_stiffness_n_m2,
            _validate=False,
        )
    gravity = torch.as_tensor(
        snapshot.model.parameters.gravity_camera_m_s2,
        dtype=torch.float64,
    )
    gravitational = -torch.sum(
        masses[None, :, None] * positions * gravity[None, None], dim=(1, 2)
    )
    gravitational = gravitational - gravitational[0]
    direction = torch.as_tensor(problem.impact_direction, dtype=torch.float64)
    directed_speed = torch.sum(velocities[:, -1] * direction[None], dim=1)
    tip_mass = masses[-1]
    directed_tip_energy = 0.5 * tip_mass * torch.relu(directed_speed).square()

    initial_drone = prediction.drone_positions_m[0]
    horizontal = np.asarray(problem.target_position_m) - initial_drone
    horizontal = horizontal.copy()
    horizontal[2] = 0.0
    horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1.0e-9:
        horizontal = np.asarray(problem.impact_direction, dtype=np.float64).copy()
        horizontal[2] = 0.0
        horizontal_norm = float(np.linalg.norm(horizontal))
    if horizontal_norm <= 1.0e-9:
        horizontal = np.asarray((1.0, 0.0, 0.0))
        horizontal_norm = 1.0
    forward = horizontal / horizontal_norm
    displacement = (prediction.drone_positions_m - initial_drone) @ forward

    if isinstance(plan, MpcPlan):
        maneuver_duration = plan.casting_action.motion_fraction * settings.horizon_s
        injection_end = plan.casting_action.reversal_fraction * maneuver_duration
    else:
        # MPPI has no predefined injection/release boundary.  For plotting only,
        # infer the reversal from peak forward drone displacement before impact.
        impact = int(np.clip(plan.impact_time_s / settings.physics_dt_s, 1, len(displacement) - 1))
        peak = int(np.argmax(displacement[: impact + 1]))
        injection_end = float(prediction.time_s[peak])
        maneuver_duration = float(plan.impact_time_s)
    return EnergyDiagnostics(
        time_s=_readonly(prediction.time_s),
        cable_kinetic_energy_j=_readonly(kinetic.numpy()),
        relative_cable_kinetic_energy_j=_readonly(relative_kinetic.numpy()),
        bending_energy_j=_readonly(bending.numpy()),
        gravitational_energy_change_j=_readonly(gravitational.numpy()),
        directed_tip_energy_j=_readonly(directed_tip_energy.numpy()),
        drone_forward_displacement_m=_readonly(displacement),
        injection_end_s=float(injection_end),
        recoil_end_s=float(maneuver_duration),
    )


def solve_perfect_model_mpc(
    snapshot: CableModelSnapshot,
    problem: MpcProblem,
    initial_drone_position_m: tuple[float, float, float],
    settings: PerfectMpcSettings = PerfectMpcSettings(),
    *,
    device: str = "cuda",
    mppi_warm_start_knots_m_s2: np.ndarray | torch.Tensor | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> PerfectMpcResult:
    """Solve once, then replay in a newly constructed exact plant."""

    report = progress if progress is not None else (lambda _message: None)
    simulation_settings = settings.simulation_settings()
    report(
        "Matched-model verification: full-state/full-DDER controller and plant; "
        f"horizon={settings.horizon_s:.2f}s, physics={1.0 / settings.physics_dt_s:.0f}Hz, "
        f"control={1.0 / settings.control_interval_s:.0f}Hz"
    )
    planner = WhipSimulator(snapshot, simulation_settings, device=device)
    planning_state = planner.initial_state(initial_drone_position_m)
    if settings.solver == "mppi":
        plan: MpcPlan | MppiPlan = optimize_mppi(
            planner,
            planning_state,
            problem,
            settings.mppi_settings(),
            warm_start_knots_m_s2=mppi_warm_start_knots_m_s2,
            progress=report,
            cancelled=cancelled,
        )
    else:
        plan = optimize_controls(
            planner,
            planning_state,
            problem,
            settings.optimizer_settings(),
            weights=settings.cost_weights(),
            progress=report,
            cancelled=cancelled,
            optimize_impact_time=True,
            truncate_at_impact=False,
        )

    if cancelled is not None and cancelled():
        raise RuntimeError("Perfect-model MPC verification stopped by user.")
    report("Independent replay: constructing a fresh full-DDER plant")
    plant = WhipSimulator(snapshot, simulation_settings, device=device)
    plant_state = plant.initial_state(initial_drone_position_m)
    controls = torch.as_tensor(
        np.array(plan.controls_m_s2, copy=True),
        dtype=plant.dtype,
        device=plant.device,
    )[None]
    with torch.no_grad():
        replay = plant.rollout(
            plant_state,
            controls,
            create_graph=False,
            cancelled=cancelled,
        )
        if isinstance(plan, MppiPlan):
            _, replay_terms, replay_impact_frames = evaluate_mppi_rollout(
                replay,
                plant_state,
                problem,
                plant,
                settings.mppi_settings(),
            )
        else:
            replay_terms, replay_impact_frames = variable_impact_rollout_cost_terms(
                replay,
                plant_state,
                problem,
                plant,
                settings.cost_weights(),
            )
    terms = {
        name: float(value[0].detach().cpu())
        for name, value in replay_terms.items()
    }
    impact_frame = int(replay_impact_frames[0].detach().cpu())
    prediction = tensor_rollout_to_result(
        replay,
        batch_index=0,
        target_position_m=torch.as_tensor(problem.target_position_m),
        impact_direction=torch.as_tensor(problem.impact_direction),
        model_sha256=snapshot.sha256,
    )
    position_difference = float(
        np.max(
            np.linalg.norm(
                prediction.cable_positions_m - plan.prediction.cable_positions_m,
                axis=2,
            )
        )
    )
    velocity_difference = float(
        np.max(
            np.linalg.norm(
                prediction.cable_velocities_m_s - plan.prediction.cable_velocities_m_s,
                axis=2,
            )
        )
    )
    energy = _energy_diagnostics(snapshot, prediction, problem, plan, settings)
    report(
        "Independent result: "
        f"feasible={'yes' if bool(terms['feasible']) else 'no'}, "
        f"tip error={1000.0 * terms['position_error_m']:.1f}mm, "
        f"directed speed={terms['directional_speed_m_s']:.2f}m/s, "
        f"direction error={terms['direction_error_deg']:.1f}deg"
    )
    report(
        "Replay agreement: "
        f"position={1.0e6 * position_difference:.3f}um, "
        f"velocity={1.0e6 * velocity_difference:.3f}um/s"
    )
    return PerfectMpcResult(
        plan=plan,
        prediction=prediction,
        terms=terms,
        impact_frame=impact_frame,
        energy=energy,
        maximum_replay_position_difference_m=position_difference,
        maximum_replay_velocity_difference_m_s=velocity_difference,
        source_model_sha256=snapshot.sha256,
        source_model_provisional=snapshot.provisional,
    )


def save_perfect_mpc_result(
    path: str | Path,
    result: PerfectMpcResult,
    settings: PerfectMpcSettings,
    problem: MpcProblem,
) -> Path:
    """Save replay arrays and a human-readable sidecar with full provenance."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    energy = result.energy
    np.savez_compressed(
        output,
        time_s=result.prediction.time_s,
        drone_positions_m=result.prediction.drone_positions_m,
        drone_velocities_m_s=result.prediction.drone_velocities_m_s,
        attachment_positions_m=result.prediction.attachment_positions_m,
        cable_positions_m=result.prediction.cable_positions_m,
        cable_velocities_m_s=result.prediction.cable_velocities_m_s,
        accelerations_m_s2=result.prediction.accelerations_m_s2,
        target_position_m=result.prediction.target_position_m,
        impact_direction=result.prediction.impact_direction,
        impact_frame=np.asarray(result.impact_frame),
        cable_kinetic_energy_j=energy.cable_kinetic_energy_j,
        relative_cable_kinetic_energy_j=energy.relative_cable_kinetic_energy_j,
        bending_energy_j=energy.bending_energy_j,
        gravitational_energy_change_j=energy.gravitational_energy_change_j,
        directed_tip_energy_j=energy.directed_tip_energy_j,
        drone_forward_displacement_m=energy.drone_forward_displacement_m,
    )
    metadata = {
        "schema": PERFECT_MPC_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "solver": (
            "low-frequency batched MPPI over smooth 3-D acceleration knots"
            if settings.solver == "mppi"
            else "CasADi IPOPT with batched central-difference full-DDER callbacks"
        ),
        "assumption": "controller model and independently replayed plant are identical",
        "model_sha256": result.source_model_sha256,
        "model_provisional": result.source_model_provisional,
        "settings": asdict(settings),
        "problem": asdict(problem),
        "control_parameterization": (
            {
                "type": "low_frequency_acceleration_knots",
                "knots_m_s2": result.plan.control_knots_m_s2.tolist(),
            }
            if isinstance(result.plan, MppiPlan)
            else {
                "type": "target_aligned_casting_action",
                "casting_action": asdict(result.plan.casting_action),
            }
        ),
        "mppi_diagnostics": (
            {
                "best_cost_history": list(result.plan.history),
                "effective_sample_size_history": list(
                    result.plan.effective_sample_size_history
                ),
                "sample_success_rate_history": list(
                    result.plan.sample_success_rate_history
                ),
                "minimum_tip_error_history_m": list(
                    result.plan.minimum_tip_error_history_m
                ),
                "gradient_valid_history": list(
                    result.plan.gradient_valid_history
                ),
                "gradient_reason_history": list(
                    result.plan.gradient_reason_history
                ),
                "smooth_surrogate_cost_history": _finite_history(
                    result.plan.smooth_surrogate_cost_history
                ),
                "gradient_norm_history": _finite_history(
                    result.plan.gradient_norm_history
                ),
                "gradient_computation_time_s_history": list(
                    result.plan.gradient_computation_time_s_history
                ),
            }
            if isinstance(result.plan, MppiPlan)
            else None
        ),
        "terms": result.terms,
        "impact_frame": result.impact_frame,
        "injection_end_s": energy.injection_end_s,
        "recoil_end_s": energy.recoil_end_s,
        "maximum_replay_position_difference_m": (
            result.maximum_replay_position_difference_m
        ),
        "maximum_replay_velocity_difference_m_s": (
            result.maximum_replay_velocity_difference_m_s
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output
