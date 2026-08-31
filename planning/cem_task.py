"""Frozen variable-duration whip task and CEM configuration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .task import CostWeights, PROJECT_ROOT, load_canonical_whip_task


DEFAULT_VARIABLE_DURATION_TASK = (
    PROJECT_ROOT / "config" / "tasks" / "canonical_whip_variable_duration_v1.json"
)


def _tuple(payload: Any, length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in payload)
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {length} finite values.")
    return result


@dataclass(frozen=True, slots=True)
class CemSettings:
    duration_min_s: float
    duration_max_initial_s: float
    duration_max_extended_s: float
    knot_count: int
    population: int
    logical_chunk_size: int
    elite_fraction: float
    maximum_iterations: int
    initial_duration_mean_s: float
    initial_acceleration_std_m_s2: float
    initial_duration_std_s: float
    old_distribution_weight: float
    elite_distribution_weight: float
    acceleration_std_floor_m_s2: float
    duration_std_floor_s: float
    covariance: str
    covariance_jitter: float
    maximum_seed_count: int
    seeds: tuple[int, ...]
    strong_tip_error_m: float
    strong_directed_speed_m_s: float
    strong_direction_error_deg: float
    strong_uav_displacement_m: float
    strong_uav_speed_m_s: float
    strong_stale_iterations: int
    hard_stop_s: float
    fixed_uav_evaluation_batch_size: int


@dataclass(frozen=True, slots=True)
class LegacyRunOnlineObjective:
    """Effective strike-cost settings used by the frozen legacy run_online GUI.

    The current production planner retains its own physical model, hard task
    gates, and feasibility-first candidate ordering.  This structure freezes
    the legacy reward shape and coefficients explicitly so a rerun cannot
    accidentally inherit later CEM defaults.
    """

    profile: str
    position_sigma_m: float
    velocity_gate_sigma_m: float
    predictive_velocity_gate_sigma_m: float
    position_weight: float
    speed_weight: float
    predictive_speed_weight: float
    predictive_speed_ratio: float
    directed_speed_shaping_m_s: float
    direction_weight: float
    direction_shaping_error_deg: float
    success_cost: float
    drone_displacement_weight: float
    safety_weight: float
    control_effort_weight: float
    control_smoothness_weight: float


@dataclass(frozen=True, slots=True)
class VariableDurationWhipTask:
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
    non_tip_clearance_m: float
    maximum_uav_displacement_m: float
    maximum_uav_speed_m_s: float
    maximum_command_acceleration_m_s2: float
    cem: CemSettings
    weights: CostWeights
    legacy_run_online_objective: LegacyRunOnlineObjective | None
    initial_nominal_source: str
    initial_nominal_artifact: Path | None
    initial_nominal_sha256: str | None
    source_path: Path
    authorization: str
    real_flight_authorized: bool
    protected_take_id: str
    protected_test_evaluation_allowed: bool

    @property
    def impact_window_s(self) -> tuple[float, float]:
        """Compatibility view; per-row duration masking supplies the upper limit."""

        return (0.0, self.cem.duration_max_extended_s)

    def validate_for_dt(self, dt_s: float) -> None:
        if self.task_id not in (
            "canonical_whip_variable_duration_v1",
            "canonical_whip_variable_duration_legacy_reward_v1",
            "canonical_whip_variable_duration_tuned_reward_v1",
            "figure8_endpoint_whip_variable_duration_v1",
        ):
            raise ValueError("Unsupported variable-duration task identifier.")
        if self.model_freeze != "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI":
            raise ValueError("Variable-duration task references the wrong freeze.")
        if self.cem.knot_count != 16 or self.cem.population != 8192:
            raise ValueError("Milestone 4C requires 16 knots and 8192 candidates.")
        if self.cem.logical_chunk_size != 2048:
            raise ValueError("Milestone 4C requires canonical 2048-row chunks.")
        if self.cem.population % self.cem.logical_chunk_size:
            raise ValueError("Population must divide exactly into logical chunks.")
        if not 0.0 < self.cem.elite_fraction < 1.0:
            raise ValueError("Elite fraction must lie in (0, 1).")
        if not (
            0.0 < self.cem.duration_min_s < self.cem.duration_max_initial_s
            <= self.cem.duration_max_extended_s
        ):
            raise ValueError("Invalid duration search bounds.")
        if abs(self.hover_preroll_s / dt_s - round(self.hover_preroll_s / dt_s)) > 1e-9:
            raise ValueError("Hover pre-roll must use an integer number of steps.")
        direction_norm = math.sqrt(sum(value * value for value in self.desired_direction))
        if abs(direction_norm - 1.0) > 1e-9:
            raise ValueError("Strike direction must be normalized.")
        if self.legacy_run_online_objective is not None and not (
            0.0 < self.legacy_run_online_objective.direction_shaping_error_deg
            <= self.maximum_direction_error_deg
        ):
            raise ValueError(
                "Direction shaping angle must be positive and no looser than the hard gate."
            )
        if (
            self.legacy_run_online_objective is not None
            and self.legacy_run_online_objective.directed_speed_shaping_m_s
            < self.minimum_directed_speed_m_s
        ):
            raise ValueError(
                "Directed-speed shaping target cannot be below the hard speed gate."
            )
        if self.authorization != "SIMULATION_ONLY" or self.real_flight_authorized:
            raise ValueError("Variable-duration planning is simulation-only.")
        if self.protected_test_evaluation_allowed:
            raise ValueError("Protected-test evaluation remains forbidden.")

    def nominal_knots(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Resample the accepted 11-knot nominal to 16 normalized-time knots."""

        if self.initial_nominal_artifact is not None:
            if not self.initial_nominal_artifact.is_file():
                raise FileNotFoundError(
                    f"Missing frozen CEM warm-start artifact: {self.initial_nominal_artifact}"
                )
            actual_sha = hashlib.sha256(self.initial_nominal_artifact.read_bytes()).hexdigest()
            if actual_sha != self.initial_nominal_sha256:
                raise RuntimeError("CEM warm-start artifact hash mismatch.")
            payload = json.loads(self.initial_nominal_artifact.read_text(encoding="utf-8"))
            knots = torch.as_tensor(payload["values"], device=device, dtype=dtype)
            if knots.shape != (self.cem.knot_count, 3) or not torch.isfinite(knots).all():
                raise ValueError("CEM warm-start artifact must contain finite [16,3] knots.")
            return knots

        baseline = load_canonical_whip_task().nominal_knots(device=device, dtype=dtype)
        source_s = torch.linspace(0.0, 1.0, baseline.shape[0], device=device, dtype=dtype)
        target_s = torch.linspace(0.0, 1.0, self.cem.knot_count, device=device, dtype=dtype)
        upper = torch.searchsorted(source_s, target_s, right=True).clamp(1, baseline.shape[0] - 1)
        lower = upper - 1
        fraction = (target_s - source_s[lower]) / (source_s[upper] - source_s[lower])
        return baseline[lower] * (1.0 - fraction[:, None]) + baseline[upper] * fraction[:, None]

    def snapshot(self) -> dict[str, Any]:
        return json.loads(self.source_path.read_text(encoding="utf-8"))


def load_variable_duration_task(
    path: str | Path = DEFAULT_VARIABLE_DURATION_TASK,
) -> VariableDurationWhipTask:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "variable_duration_whip_task_v1":
        raise ValueError(f"Unsupported variable-duration task schema: {source}")
    initial = payload["initial_state"]
    target = payload["target"]
    constraints = payload["constraints"]
    config = payload["cem"]
    policy = payload["artifact_policy"]
    task = VariableDurationWhipTask(
        task_id=str(payload["task_id"]),
        model_freeze=str(payload["model_freeze"]),
        initial_uav_position_m=_tuple(initial["uav_position_m"], 3, "initial position"),
        initial_uav_velocity_m_s=_tuple(initial["uav_velocity_m_s"], 3, "initial velocity"),
        initial_yaw_rad=float(initial["yaw_rad"]),
        hover_preroll_s=float(initial["hover_preroll_s"]),
        target_position_m=_tuple(target["position_m"], 3, "target position"),
        desired_direction=_tuple(target["desired_impact_direction"], 3, "direction"),
        success_radius_m=float(target["success_radius_m"]),
        minimum_directed_speed_m_s=float(target["minimum_directed_tip_speed_m_s"]),
        maximum_direction_error_deg=float(target["maximum_impact_direction_error_deg"]),
        non_tip_clearance_m=float(target["non_tip_cost_clearance_m"]),
        maximum_uav_displacement_m=float(constraints["maximum_uav_displacement_m"]),
        maximum_uav_speed_m_s=float(constraints["maximum_uav_speed_m_s"]),
        maximum_command_acceleration_m_s2=float(constraints["maximum_command_acceleration_m_s2"]),
        cem=CemSettings(
            duration_min_s=float(config["duration_min_s"]),
            duration_max_initial_s=float(config["duration_max_initial_s"]),
            duration_max_extended_s=float(config["duration_max_extended_s"]),
            knot_count=int(config["acceleration_knots"]),
            population=int(config["population"]),
            logical_chunk_size=int(config["logical_chunk_size"]),
            elite_fraction=float(config["elite_fraction"]),
            maximum_iterations=int(config["maximum_iterations"]),
            initial_duration_mean_s=float(config["initial_duration_mean_s"]),
            initial_acceleration_std_m_s2=float(config["initial_acceleration_std_m_s2"]),
            initial_duration_std_s=float(config["initial_duration_std_s"]),
            old_distribution_weight=float(config["old_distribution_weight"]),
            elite_distribution_weight=float(config["elite_distribution_weight"]),
            acceleration_std_floor_m_s2=float(config["acceleration_std_floor_m_s2"]),
            duration_std_floor_s=float(config["duration_std_floor_s"]),
            covariance=str(config["covariance"]),
            covariance_jitter=float(config["covariance_jitter"]),
            maximum_seed_count=int(config["maximum_seed_count"]),
            seeds=tuple(int(value) for value in config["seeds"]),
            strong_tip_error_m=float(config["strong_tip_error_m"]),
            strong_directed_speed_m_s=float(config["strong_directed_speed_m_s"]),
            strong_direction_error_deg=float(config["strong_direction_error_deg"]),
            strong_uav_displacement_m=float(config["strong_uav_displacement_m"]),
            strong_uav_speed_m_s=float(config["strong_uav_speed_m_s"]),
            strong_stale_iterations=int(config["strong_stale_iterations"]),
            hard_stop_s=float(config["hard_stop_s"]),
            fixed_uav_evaluation_batch_size=int(config["fixed_uav_evaluation_batch_size"]),
        ),
        weights=CostWeights(**{name: float(value) for name, value in payload["cost_weights"].items()}),
        legacy_run_online_objective=(
            None
            if payload.get("legacy_run_online_objective") is None
            else LegacyRunOnlineObjective(
                profile=str(payload["legacy_run_online_objective"]["profile"]),
                direction_shaping_error_deg=float(
                    payload["legacy_run_online_objective"].get(
                        "direction_shaping_error_deg",
                        target["maximum_impact_direction_error_deg"],
                    )
                ),
                directed_speed_shaping_m_s=float(
                    payload["legacy_run_online_objective"].get(
                        "directed_speed_shaping_m_s",
                        target["minimum_directed_tip_speed_m_s"],
                    )
                ),
                **{
                    name: float(value)
                    for name, value in payload["legacy_run_online_objective"].items()
                    if name
                    not in {
                        "profile",
                        "direction_shaping_error_deg",
                        "directed_speed_shaping_m_s",
                    }
                },
            )
        ),
        initial_nominal_source=str(payload["initial_nominal"]["source"]),
        initial_nominal_artifact=(
            None
            if payload["initial_nominal"].get("artifact") is None
            else (PROJECT_ROOT / str(payload["initial_nominal"]["artifact"])).resolve()
        ),
        initial_nominal_sha256=(
            None
            if payload["initial_nominal"].get("sha256") is None
            else str(payload["initial_nominal"]["sha256"])
        ),
        source_path=source,
        authorization=str(policy["authorization"]),
        real_flight_authorized=bool(policy["real_flight_authorized"]),
        protected_take_id=str(policy["protected_take_id"]),
        protected_test_evaluation_allowed=bool(policy["protected_test_evaluation_allowed"]),
    )
    return task
