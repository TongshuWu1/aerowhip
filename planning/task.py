"""Canonical cable-whip task configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_TASK = PROJECT_ROOT / "config" / "tasks" / "canonical_whip_v1.json"


def _finite_tuple(payload: Any, length: int, name: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in payload)
    if len(values) != length or not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain {length} finite values.")
    return values


@dataclass(frozen=True, slots=True)
class CostWeights:
    event_position: float
    event_speed_deficiency: float
    event_direction_deficiency: float
    event_non_tip_proximity: float
    command_effort: float
    command_smoothness: float
    final_uav_speed: float
    uav_displacement_violation: float
    uav_speed_violation: float
    success_bonus: float
    invalid_cost: float


@dataclass(frozen=True, slots=True)
class MppiSettings:
    horizon_s: float
    knot_count: int
    sample_count: int
    maximum_iterations: int
    sigma_m_s2: float
    temperature: float
    adaptive_ess_target_fraction: float
    fixed_uav_evaluation_batch_size: int
    random_seed: int
    success_polishing_iterations: int
    hard_stop_s: float


@dataclass(frozen=True, slots=True)
class CanonicalWhipTask:
    task_id: str
    model_freeze: str
    initial_uav_position_m: tuple[float, float, float]
    initial_uav_velocity_m_s: tuple[float, float, float]
    initial_yaw_rad: float
    hover_preroll_s: float
    target_position_m: tuple[float, float, float]
    desired_direction: tuple[float, float, float]
    success_radius_m: float
    minimum_directed_speed_m_s: float
    maximum_direction_error_deg: float
    impact_window_s: tuple[float, float]
    non_tip_clearance_m: float
    maximum_uav_displacement_m: float
    maximum_uav_speed_m_s: float
    maximum_command_acceleration_m_s2: float
    mppi: MppiSettings
    weights: CostWeights
    initial_nominal_source: str
    initial_nominal_times_s: tuple[float, ...]
    initial_nominal_knots_m_s2: tuple[tuple[float, float, float], ...]
    authorization: str
    real_flight_authorized: bool
    protected_take_id: str
    protected_test_evaluation_allowed: bool
    source_path: Path

    def validate_for_dt(self, dt_s: float) -> None:
        if self.task_id not in ("canonical_whip_v1", "figure8_endpoint_whip_v1"):
            raise ValueError("Unsupported production whip task identifier.")
        if self.model_freeze != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
            raise ValueError("Canonical task does not reference the required model freeze.")
        positive = (
            self.hover_preroll_s,
            self.success_radius_m,
            self.minimum_directed_speed_m_s,
            self.non_tip_clearance_m,
            self.maximum_uav_displacement_m,
            self.maximum_uav_speed_m_s,
            self.maximum_command_acceleration_m_s2,
            self.mppi.horizon_s,
            self.mppi.sigma_m_s2,
            self.mppi.temperature,
            self.mppi.hard_stop_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("All task scales and horizons must be finite and positive.")
        if not 0.0 < self.mppi.adaptive_ess_target_fraction <= 1.0:
            raise ValueError("Adaptive ESS target fraction must lie in (0, 1].")
        if self.mppi.fixed_uav_evaluation_batch_size < self.mppi.sample_count:
            raise ValueError(
                "Fixed UAV evaluation batch must cover the MPPI population."
            )
        if self.impact_window_s[0] < 0.0 or self.impact_window_s[1] > self.mppi.horizon_s:
            raise ValueError("Impact window must lie inside the planning horizon.")
        if self.impact_window_s[0] >= self.impact_window_s[1]:
            raise ValueError("Impact window must have positive duration.")
        direction_norm = math.sqrt(sum(value * value for value in self.desired_direction))
        if abs(direction_norm - 1.0) > 1.0e-9:
            raise ValueError("Desired impact direction must be normalized.")
        if len(self.initial_nominal_knots_m_s2) != self.mppi.knot_count:
            raise ValueError("Initial nominal knot count does not match MPPI configuration.")
        if len(self.initial_nominal_times_s) != self.mppi.knot_count:
            raise ValueError("Initial nominal times do not match MPPI configuration.")
        expected_times = torch.linspace(
            0.0, self.mppi.horizon_s, self.mppi.knot_count, dtype=torch.float64
        )
        actual_times = torch.tensor(self.initial_nominal_times_s, dtype=torch.float64)
        if not torch.allclose(actual_times, expected_times, atol=1.0e-12, rtol=0.0):
            raise ValueError("Initial nominal times must be uniform over the horizon.")
        steps = self.mppi.horizon_s / dt_s
        preroll_steps = self.hover_preroll_s / dt_s
        if abs(steps - round(steps)) > 1.0e-9:
            raise ValueError("Planning horizon must contain an integer number of physics steps.")
        if abs(preroll_steps - round(preroll_steps)) > 1.0e-9:
            raise ValueError("Hover pre-roll must contain an integer number of physics steps.")
        if self.authorization != "SIMULATION_ONLY" or self.real_flight_authorized:
            raise ValueError("Milestone 4A artifacts must remain simulation-only.")
        if self.protected_test_evaluation_allowed:
            raise ValueError("Protected-test evaluation is forbidden in Milestone 4A.")

    def nominal_knots(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            self.initial_nominal_knots_m_s2, device=device, dtype=dtype
        )

    def snapshot(self) -> dict[str, Any]:
        return json.loads(self.source_path.read_text(encoding="utf-8"))


def load_canonical_whip_task(path: str | Path = DEFAULT_CANONICAL_TASK) -> CanonicalWhipTask:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "canonical_whip_task_v1":
        raise ValueError(f"Unsupported canonical whip task schema: {source}")
    initial = payload["initial_state"]
    target = payload["target"]
    constraints = payload["constraints"]
    planning = payload["planning"]
    weights = payload["cost_weights"]
    nominal = payload["initial_nominal"]
    policy = payload["artifact_policy"]
    task = CanonicalWhipTask(
        task_id=str(payload["task_id"]),
        model_freeze=str(payload["model_freeze"]),
        initial_uav_position_m=_finite_tuple(initial["uav_position_m"], 3, "initial position"),
        initial_uav_velocity_m_s=_finite_tuple(initial["uav_velocity_m_s"], 3, "initial velocity"),
        initial_yaw_rad=float(initial["yaw_rad"]),
        hover_preroll_s=float(initial["hover_preroll_s"]),
        target_position_m=_finite_tuple(target["position_m"], 3, "target position"),
        desired_direction=_finite_tuple(target["desired_impact_direction"], 3, "desired direction"),
        success_radius_m=float(target["success_radius_m"]),
        minimum_directed_speed_m_s=float(target["minimum_directed_tip_speed_m_s"]),
        maximum_direction_error_deg=float(target["maximum_impact_direction_error_deg"]),
        impact_window_s=_finite_tuple(target["impact_time_window_s"], 2, "impact window"),
        non_tip_clearance_m=float(target["non_tip_cost_clearance_m"]),
        maximum_uav_displacement_m=float(constraints["maximum_uav_displacement_m"]),
        maximum_uav_speed_m_s=float(constraints["maximum_uav_speed_m_s"]),
        maximum_command_acceleration_m_s2=float(constraints["maximum_command_acceleration_m_s2"]),
        mppi=MppiSettings(
            horizon_s=float(planning["horizon_s"]),
            knot_count=int(planning["acceleration_knots"]),
            sample_count=int(planning["samples"]),
            maximum_iterations=int(planning["maximum_iterations"]),
            sigma_m_s2=float(planning["perturbation_sigma_m_s2"]),
            temperature=float(planning["temperature_lambda"]),
            adaptive_ess_target_fraction=float(
                planning.get("adaptive_ess_target_fraction", 0.02)
            ),
            fixed_uav_evaluation_batch_size=int(
                planning.get("fixed_uav_evaluation_batch_size", planning["samples"])
            ),
            random_seed=int(planning["random_seed"]),
            success_polishing_iterations=int(planning["success_polishing_iterations"]),
            hard_stop_s=float(planning["hard_stop_s"]),
        ),
        weights=CostWeights(**{name: float(value) for name, value in weights.items()}),
        initial_nominal_source=str(nominal["source"]),
        initial_nominal_times_s=tuple(float(value) for value in nominal["times_s"]),
        initial_nominal_knots_m_s2=tuple(
            _finite_tuple(row, 3, "initial nominal knot")
            for row in nominal["acceleration_knots_m_s2"]
        ),
        authorization=str(policy["authorization"]),
        real_flight_authorized=bool(policy["real_flight_authorized"]),
        protected_take_id=str(policy["protected_take_id"]),
        protected_test_evaluation_allowed=bool(policy["protected_test_evaluation_allowed"]),
        source_path=source,
    )
    return task
