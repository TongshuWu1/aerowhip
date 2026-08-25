"""Ordered-marker simulation testbed for nominal SAC cable control.

The module deliberately separates three objects:

* an independent, matched-grid cable plant used only to generate simulated truth;
* an OptiTrack-like measurement containing the drone and ordered cable markers;
* a nominal cable belief and SAC policy whose identities must match the policy
  checkpoint used during training.

The current SAC action is a commanded translational acceleration tracked by a
fast drone inner loop.  It is not a force-coupled drone/cable simulation: the
cable reaction is not applied back to the point-mass drone in this phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterator, Sequence
import uuid

import numpy as np
import torch

from cable_twin.shared.dder import (
    DderState,
    START_PINNED_FREE_END,
    momentum_project_lengths,
)
from cable_twin.shared.observation_data import sha256_file

from .model import CableModelSnapshot, load_cable_model
from .reduced import stable_controller_model
from .rl_env import TaskDistribution, VectorWhipEnvironment
from .sac import (
    SacAgent,
    SacSettings,
    is_supported_sac_checkpoint_schema,
    load_policy,
)
from .simulator import DroneCableState, SimulationSettings, TensorRollout, WhipSimulator


TESTBED_SCHEMA = "ordered_marker_hidden_plant_sac_testbed_v1"
FrameCallback = Callable[["TestbedFrame"], None]
CancellationCallback = Callable[[], bool]


def _readonly_vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values.")
    result.setflags(write=False)
    return result


def _readonly_points(value: np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape Nx3.")
    result.setflags(write=False)
    return result


def _unit_vector(value: Sequence[float], name: str) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values.")
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError(f"{name} must be non-zero.")
    return tuple(float(component) for component in vector / norm)


@dataclass(frozen=True, slots=True)
class TestbedSettings:
    """One deterministic fixed-target ordered-marker-feedback episode."""

    initial_drone_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_position_m: tuple[float, float, float] = (0.8, 0.0, -0.1)
    impact_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    episode_horizon_s: float | None = None
    hidden_ei_scale: float = 1.0
    hidden_cb_scale: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        initial = tuple(_readonly_vector(self.initial_drone_position_m, "initial position"))
        target = tuple(_readonly_vector(self.target_position_m, "target position"))
        direction = _unit_vector(self.impact_direction, "impact direction")
        if self.episode_horizon_s is not None and (
            not math.isfinite(self.episode_horizon_s) or self.episode_horizon_s <= 0.0
        ):
            raise ValueError("Episode horizon must be positive when supplied.")
        if (
            not math.isfinite(self.hidden_ei_scale)
            or self.hidden_ei_scale <= 0.0
            or not math.isfinite(self.hidden_cb_scale)
            or self.hidden_cb_scale <= 0.0
        ):
            raise ValueError("Hidden EI and Cb scales must be finite and positive.")
        if self.seed < 0:
            raise ValueError("Testbed seed must be non-negative.")
        object.__setattr__(self, "initial_drone_position_m", initial)
        object.__setattr__(self, "target_position_m", target)
        object.__setattr__(self, "impact_direction", direction)


@dataclass(frozen=True, slots=True)
class OptitrackMeasurement:
    """The complete ordered-marker measurement passed to the controller.

    Phase A assumes perfect marker association and ordering.  Marker velocities
    are causal backward differences computed by the measurement frontend, not
    hidden simulator velocities.
    """

    time_s: float
    drone_position_m: np.ndarray
    drone_velocity_m_s: np.ndarray
    attachment_position_m: np.ndarray
    marker_material_coordinates_m: np.ndarray
    marker_positions_m: np.ndarray
    marker_velocities_m_s: np.ndarray
    marker_valid: np.ndarray
    applied_acceleration_m_s2: np.ndarray

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("Measurement time must be finite and non-negative.")
        for name in (
            "drone_position_m",
            "drone_velocity_m_s",
            "attachment_position_m",
            "applied_acceleration_m_s2",
        ):
            object.__setattr__(self, name, _readonly_vector(getattr(self, name), name))
        coordinates = np.array(
            self.marker_material_coordinates_m, dtype=np.float64, copy=True
        )
        positions = _readonly_points(self.marker_positions_m, "marker positions")
        velocities = _readonly_points(self.marker_velocities_m_s, "marker velocities")
        valid = np.array(self.marker_valid, dtype=np.bool_, copy=True)
        marker_count = positions.shape[0]
        if (
            coordinates.shape != (marker_count,)
            or velocities.shape != positions.shape
            or valid.shape != (marker_count,)
            or marker_count < 3
            or not np.all(np.isfinite(coordinates))
            or np.any(np.diff(coordinates) <= 0.0)
        ):
            raise ValueError("OptiTrack marker arrays/material sites are incompatible.")
        coordinates.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "marker_material_coordinates_m", coordinates)
        object.__setattr__(self, "marker_positions_m", positions)
        object.__setattr__(self, "marker_velocities_m_s", velocities)
        object.__setattr__(self, "marker_valid", valid)

    @property
    def free_tip_position_m(self) -> np.ndarray:
        return self.marker_positions_m[-1]


@dataclass(frozen=True, slots=True)
class ControllerCommand:
    """One 10-Hz target-frame SAC action and its world acceleration."""

    normalized_action: np.ndarray
    acceleration_m_s2: np.ndarray

    def __post_init__(self) -> None:
        action = _readonly_vector(self.normalized_action, "normalized action")
        acceleration = _readonly_vector(self.acceleration_m_s2, "acceleration")
        if float(np.linalg.norm(action)) > 1.0 + 1.0e-6:
            raise ValueError("Normalized SAC action must lie in the unit ball.")
        object.__setattr__(self, "normalized_action", action)
        object.__setattr__(self, "acceleration_m_s2", acceleration)


@dataclass(frozen=True, slots=True)
class TestbedFrame:
    """One immutable 50-Hz renderer/evaluation frame.

    Hidden cable positions are present for visualization and scoring only.  A
    ``TestbedFrame`` is never accepted by ``NominalSacController.select_action``.
    Innovation values are evaluated immediately before the most recent 10-Hz
    measurement correction and held on the intervening 50-Hz display frames.
    """

    time_s: float
    hidden_cable_positions_m: np.ndarray
    belief_cable_positions_m: np.ndarray
    drone_position_m: np.ndarray
    drone_velocity_m_s: np.ndarray
    attachment_position_m: np.ndarray
    measured_marker_positions_m: np.ndarray
    measured_marker_velocities_m_s: np.ndarray
    measured_marker_valid: np.ndarray
    measured_tip_position_m: np.ndarray
    measurement_applied_acceleration_m_s2: np.ndarray
    target_position_m: np.ndarray
    impact_direction: np.ndarray
    innovation_norm_m: float
    marker_innovation_rms_m: float
    tip_target_distance_m: float
    directional_tip_speed_m_s: float
    commanded_acceleration_m_s2: np.ndarray
    action_updated: bool
    done: bool
    success: bool
    unsafe: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0.0:
            raise ValueError("Frame time must be finite and non-negative.")
        object.__setattr__(
            self,
            "hidden_cable_positions_m",
            _readonly_points(self.hidden_cable_positions_m, "hidden cable"),
        )
        object.__setattr__(
            self,
            "belief_cable_positions_m",
            _readonly_points(self.belief_cable_positions_m, "belief cable"),
        )
        for name in (
            "drone_position_m",
            "drone_velocity_m_s",
            "attachment_position_m",
            "measured_tip_position_m",
            "measurement_applied_acceleration_m_s2",
            "target_position_m",
            "impact_direction",
            "commanded_acceleration_m_s2",
        ):
            object.__setattr__(self, name, _readonly_vector(getattr(self, name), name))
        markers = _readonly_points(
            self.measured_marker_positions_m, "measured marker positions"
        )
        marker_velocities = _readonly_points(
            self.measured_marker_velocities_m_s, "measured marker velocities"
        )
        marker_valid = np.array(self.measured_marker_valid, dtype=np.bool_, copy=True)
        if (
            marker_velocities.shape != markers.shape
            or marker_valid.shape != (markers.shape[0],)
        ):
            raise ValueError("Measured marker velocity/validity arrays are incompatible.")
        marker_valid.setflags(write=False)
        object.__setattr__(self, "measured_marker_positions_m", markers)
        object.__setattr__(self, "measured_marker_velocities_m_s", marker_velocities)
        object.__setattr__(self, "measured_marker_valid", marker_valid)
        for name in (
            "innovation_norm_m",
            "marker_innovation_rms_m",
            "tip_target_distance_m",
            "directional_tip_speed_m_s",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite.")


@dataclass(frozen=True, slots=True)
class TestbedEpisode:
    schema: str
    settings: TestbedSettings
    sac_settings: SacSettings
    task: TaskDistribution
    frames: tuple[TestbedFrame, ...]
    marker_material_coordinates_m: np.ndarray
    nominal_source_path: str
    nominal_source_sha256: str
    nominal_controller_sha256: str
    hidden_source_path: str
    hidden_source_sha256: str
    hidden_plant_sha256: str
    policy_path: str
    policy_sha256: str
    action_semantics: str
    physics_rate_hz: float
    control_rate_hz: float
    cancelled: bool
    success: bool
    unsafe: bool
    termination_reason: str
    minimum_tip_error_m: float
    directional_tip_speed_at_minimum_error_m_s: float
    maximum_directional_tip_speed_m_s: float

    def __post_init__(self) -> None:
        if self.schema != TESTBED_SCHEMA:
            raise ValueError("Unexpected adaptation-testbed schema.")
        if not self.frames:
            raise ValueError("A testbed episode must contain at least one frame.")
        if any(
            not math.isfinite(value)
            for value in (
                self.physics_rate_hz,
                self.control_rate_hz,
                self.minimum_tip_error_m,
                self.directional_tip_speed_at_minimum_error_m_s,
                self.maximum_directional_tip_speed_m_s,
            )
        ):
            raise ValueError("Episode rates and metrics must be finite.")
        if self.termination_reason not in (
            "success",
            "unsafe",
            "timeout",
            "cancelled",
        ):
            raise ValueError("Unknown testbed termination reason.")
        object.__setattr__(self, "frames", tuple(self.frames))
        coordinates = np.array(
            self.marker_material_coordinates_m, dtype=np.float64, copy=True
        )
        if (
            coordinates.ndim != 1
            or len(coordinates) != self.frames[0].measured_marker_positions_m.shape[0]
            or not np.all(np.isfinite(coordinates))
            or np.any(np.diff(coordinates) <= 0.0)
        ):
            raise ValueError("Episode marker material coordinates are invalid.")
        coordinates.setflags(write=False)
        object.__setattr__(self, "marker_material_coordinates_m", coordinates)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def action_count(self) -> int:
        return sum(frame.action_updated for frame in self.frames)


def _interpolate_material_samples(
    values: np.ndarray,
    source_coordinates_m: np.ndarray,
    target_coordinates_m: np.ndarray,
) -> np.ndarray:
    """Piecewise-linearly sample vector data on a material coordinate grid."""

    if values.shape != (len(source_coordinates_m), 3):
        raise ValueError("Material samples must have shape Mx3.")
    if (
        np.any(np.diff(source_coordinates_m) <= 0.0)
        or np.any(np.diff(target_coordinates_m) <= 0.0)
        or target_coordinates_m[0] < source_coordinates_m[0] - 1.0e-12
        or target_coordinates_m[-1] > source_coordinates_m[-1] + 1.0e-12
    ):
        raise ValueError("Material grids must be ordered and cover the target grid.")
    return np.column_stack(
        [
            np.interp(target_coordinates_m, source_coordinates_m, values[:, axis])
            for axis in range(3)
        ]
    )


class _OrderedMarkerFrontend:
    """Causal perfect-association full-state OptiTrack baseline."""

    def __init__(
        self,
        plant_material_coordinates_m: np.ndarray,
        marker_material_coordinates_m: np.ndarray,
    ) -> None:
        self.plant_material_coordinates_m = np.array(
            plant_material_coordinates_m, dtype=np.float64, copy=True
        )
        self.marker_material_coordinates_m = np.array(
            marker_material_coordinates_m, dtype=np.float64, copy=True
        )
        self.previous_time_s: float | None = None
        self.previous_positions_m: np.ndarray | None = None
        self.previous_drone_position_m: np.ndarray | None = None
        self.second_previous_time_s: float | None = None
        self.second_previous_positions_m: np.ndarray | None = None
        self.second_previous_drone_position_m: np.ndarray | None = None

    @staticmethod
    def _causal_velocity(
        current: np.ndarray,
        previous: np.ndarray | None,
        second_previous: np.ndarray | None,
        time_s: float,
        previous_time_s: float | None,
        second_previous_time_s: float | None,
    ) -> np.ndarray:
        """Differentiate positions causally: BDF1 once, then BDF2.

        The three-point formula permits unequal positive frame intervals, even
        though the Phase-A simulator produces the intended uniform 50-Hz grid.
        """

        if previous is None:
            return np.zeros_like(current)
        assert previous_time_s is not None
        first_interval = time_s - previous_time_s
        if first_interval <= 0.0:
            raise RuntimeError("OptiTrack measurement time did not increase.")
        if second_previous is None:
            return (current - previous) / first_interval
        assert second_previous_time_s is not None
        second_interval = previous_time_s - second_previous_time_s
        if second_interval <= 0.0:
            raise RuntimeError("OptiTrack measurement history is not ordered.")
        current_weight = (2.0 * first_interval + second_interval) / (
            first_interval * (first_interval + second_interval)
        )
        previous_weight = -(
            first_interval + second_interval
        ) / (first_interval * second_interval)
        second_previous_weight = first_interval / (
            second_interval * (first_interval + second_interval)
        )
        return (
            current_weight * current
            + previous_weight * previous
            + second_previous_weight * second_previous
        )

    def measure(
        self,
        *,
        time_s: float,
        state: DroneCableState,
        attachment_drop_m: float,
        applied_acceleration_m_s2: np.ndarray,
    ) -> OptitrackMeasurement:
        hidden_positions = (
            state.cable.positions_m[0].detach().cpu().numpy().copy()
        )
        positions = _interpolate_material_samples(
            hidden_positions,
            self.plant_material_coordinates_m,
            self.marker_material_coordinates_m,
        )
        drone_position = (
            state.drone_position_m[0].detach().cpu().numpy().copy()
        )
        velocities = self._causal_velocity(
            positions,
            self.previous_positions_m,
            self.second_previous_positions_m,
            time_s,
            self.previous_time_s,
            self.second_previous_time_s,
        )
        drone_velocity = self._causal_velocity(
            drone_position,
            self.previous_drone_position_m,
            self.second_previous_drone_position_m,
            time_s,
            self.previous_time_s,
            self.second_previous_time_s,
        )
        self.second_previous_time_s = self.previous_time_s
        self.second_previous_positions_m = self.previous_positions_m
        self.second_previous_drone_position_m = self.previous_drone_position_m
        self.previous_time_s = float(time_s)
        self.previous_positions_m = positions.copy()
        self.previous_drone_position_m = drone_position.copy()
        attachment = drone_position + np.asarray((0.0, 0.0, -attachment_drop_m))
        # The first physical cable marker is the attachment in the one-pinned
        # model.  Retain both fields because the real system obtains the drone
        # rigid body and cable marker set through distinct OptiTrack assets.
        if not np.allclose(positions[0], attachment, atol=1.0e-6):
            raise RuntimeError("Hidden plant marker zero is not the drone attachment.")
        # The pinned material marker and attachment are one physical point.
        # Enforce the same causally measured velocity before controller use so
        # interpolation/projection cannot introduce a boundary inconsistency.
        velocities[0] = drone_velocity
        return OptitrackMeasurement(
            time_s=time_s,
            drone_position_m=drone_position,
            drone_velocity_m_s=drone_velocity,
            attachment_position_m=attachment,
            marker_material_coordinates_m=self.marker_material_coordinates_m,
            marker_positions_m=positions,
            marker_velocities_m_s=velocities,
            marker_valid=np.ones(len(positions), dtype=np.bool_),
            applied_acceleration_m_s2=applied_acceleration_m_s2,
        )


class NominalSacController:
    """SAC inference backed only by its nominal DDER belief."""

    def __init__(
        self,
        environment: VectorWhipEnvironment,
        agent: SacAgent,
        marker_material_coordinates_m: np.ndarray,
    ) -> None:
        if environment.environment_count != 1:
            raise ValueError("The Phase-A controller requires one nominal belief.")
        self.environment = environment
        self.agent = agent
        self.marker_material_coordinates_m = np.array(
            marker_material_coordinates_m, dtype=np.float64, copy=True
        )
        self.controller_material_coordinates_m = np.asarray(
            environment.simulator.snapshot.rod_material_coordinates_m,
            dtype=np.float64,
        )
        self.last_marker_innovation_rms_m = 0.0
        self.last_tip_innovation_norm_m = 0.0

    def _sample_belief_at_markers(self) -> np.ndarray:
        positions = (
            self.environment.state.cable.positions_m[0].detach().cpu().numpy()
        )
        return _interpolate_material_samples(
            positions,
            self.controller_material_coordinates_m,
            self.marker_material_coordinates_m,
        )

    def _reconstruct_state(self, measurement: OptitrackMeasurement) -> DderState:
        """Correct measured positions while retaining causal model velocity.

        Motive supplies marker positions, not the exact instantaneous DDER
        velocity used when this SAC policy was trained.  Directly substituting
        finite-difference marker velocities creates a controller-input shift
        even when the hidden and nominal models are identical.  The Phase-A
        observer therefore uses the standard predict/correct split: the full
        marker record corrects ``q`` and the nominal DDER prediction supplies
        ``v``.  The measured BDF velocity remains in the measurement/archive as
        an independent diagnostic and future observer input.
        """
        if not bool(np.all(measurement.marker_valid)):
            raise RuntimeError(
                "Phase-A full-marker feedback requires every ordered marker."
            )
        if not np.allclose(
            measurement.marker_material_coordinates_m,
            self.marker_material_coordinates_m,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("OptiTrack material sites do not match the controller setup.")
        sampled_positions = _interpolate_material_samples(
            measurement.marker_positions_m,
            self.marker_material_coordinates_m,
            self.controller_material_coordinates_m,
        )
        dtype = self.environment.dtype
        device = self.environment.device
        positions = torch.tensor(
            sampled_positions, dtype=dtype, device=device
        ).reshape(1, -1, 3)
        velocities = self.environment.state.cable.velocities_m_s.clone()
        rest_lengths = torch.as_tensor(
            self.environment.simulator.snapshot.model.parameters.rest_lengths_m,
            dtype=dtype,
            device=device,
        )
        masses = torch.as_tensor(
            self.environment.simulator.snapshot.model.parameters.vertex_masses_kg,
            dtype=dtype,
            device=device,
        )
        boundary = torch.tensor(
            measurement.attachment_position_m,
            dtype=dtype,
            device=device,
        ).reshape(1, 1, 3)
        positions = momentum_project_lengths(
            positions,
            rest_lengths,
            masses,
            boundary,
            iterations=32,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        # Re-project the predicted generalized velocity on the corrected curve,
        # but preserve the predictor's pinned-node velocity as the boundary.
        # The discrete DDER state stores an interval-average boundary velocity,
        # whereas the OptiTrack derivative estimates instantaneous velocity.
        # Mixing those definitions would inject a false impulse and shift the
        # SAC observation distribution.
        predicted_boundary_velocity = velocities[:, :1].clone()
        velocities = self.environment.simulator.snapshot.model.project_velocities(
            positions,
            velocities,
            predicted_boundary_velocity,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        return DderState(positions, velocities)

    def select_action(self, measurement: OptitrackMeasurement) -> ControllerCommand:
        """Select an action from ordered markers plus the nominal predictor.

        Hidden plant state cannot enter this method by construction.  In Phase A
        the drone double integrator is exact, so a disagreement between the
        measured and nominal drone state is an error rather than an unreported
        state-estimation fallback.
        """

        predicted_markers = self._sample_belief_at_markers()
        marker_residual = predicted_markers - measurement.marker_positions_m
        self.last_marker_innovation_rms_m = float(
            np.sqrt(np.mean(np.sum(np.square(marker_residual), axis=1)))
        )
        self.last_tip_innovation_norm_m = float(
            np.linalg.norm(marker_residual[-1])
        )
        reconstructed = self._reconstruct_state(measurement)
        self.environment.state = DroneCableState(
            torch.tensor(
                measurement.drone_position_m,
                dtype=self.environment.dtype,
                device=self.environment.device,
            ).reshape(1, 3),
            torch.tensor(
                measurement.drone_velocity_m_s,
                dtype=self.environment.dtype,
                device=self.environment.device,
            ).reshape(1, 3),
            reconstructed,
        )
        # SAC networks and their saved normalization statistics are float32;
        # the CPU simulator otherwise uses float64 for numerical development.
        observation = self.environment.observation().to(torch.float32)
        normalized = self.agent.act(observation, deterministic=True)
        norm = torch.linalg.vector_norm(normalized, dim=1, keepdim=True)
        normalized = normalized * torch.clamp(
            1.0 / torch.clamp(norm, min=1.0e-12), max=1.0
        )
        normalized = normalized.to(
            dtype=self.environment.dtype,
            device=self.environment.device,
        )
        normalized_world = normalized.to(
            dtype=self.environment.dtype,
            device=self.environment.device,
        )
        world = torch.einsum(
            "bij,bi->bj", self.environment.target_frame, normalized_world
        )
        acceleration = world * self.environment.simulator.settings.maximum_acceleration_m_s2
        return ControllerCommand(
            normalized_action=normalized[0].detach().cpu().numpy(),
            acceleration_m_s2=acceleration[0].detach().cpu().numpy(),
        )

    def advance(self, command: ControllerCommand) -> TensorRollout:
        acceleration = torch.tensor(
            command.acceleration_m_s2,
            dtype=self.environment.dtype,
            device=self.environment.device,
        ).reshape(1, 1, 3)
        with torch.no_grad():
            rollout = self.environment.simulator.rollout(
                self.environment.state,
                acceleration,
                create_graph=False,
            )
        self.environment.state = rollout.final_state()
        self.environment.step_index += 1
        self.environment.previous_action = torch.tensor(
            command.normalized_action,
            dtype=self.environment.dtype,
            device=self.environment.device,
        ).reshape(1, 3)
        return rollout


class AdaptationTestbedSession:
    """Loaded nominal controller and independent hidden-plant experiment."""

    def __init__(
        self,
        *,
        settings: TestbedSettings,
        sac_settings: SacSettings,
        task: TaskDistribution,
        nominal_source: CableModelSnapshot,
        nominal_controller: CableModelSnapshot,
        hidden_source: CableModelSnapshot,
        hidden_plant_snapshot: CableModelSnapshot,
        policy_path: Path,
        policy_sha256: str,
        agent: SacAgent,
        device: torch.device,
    ) -> None:
        self.settings = settings
        self.sac_settings = sac_settings
        self.task = task
        self.nominal_source = nominal_source
        self.nominal_controller = nominal_controller
        self.hidden_source = hidden_source
        self.hidden_plant = hidden_plant_snapshot
        self.policy_path = policy_path
        self.policy_sha256 = policy_sha256
        self.agent = agent
        self.device = device
        # Phase A is deliberately a full-state-feedback baseline: simulated
        # OptiTrack returns one perfectly associated point at every controller
        # material site.  Sparse physical marker reconstruction is a separate
        # later experiment and must not confound nominal-model mismatch here.
        self.marker_material_coordinates_m = np.asarray(
            nominal_controller.rod_material_coordinates_m,
            dtype=np.float64,
        ).copy()
        self.episode_horizon_s = (
            sac_settings.episode_horizon_s
            if settings.episode_horizon_s is None
            else settings.episode_horizon_s
        )
        if not math.isclose(
            self.episode_horizon_s,
            sac_settings.episode_horizon_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "Phase-A episode horizon must match the SAC checkpoint observation semantics."
            )
        target_delta = np.asarray(
            settings.target_position_m, dtype=np.float64
        ) - np.asarray(settings.initial_drone_position_m, dtype=np.float64)
        horizontal_radius = float(np.linalg.norm(target_delta[:2]))
        height_offset = float(target_delta[2])
        azimuth_deg = math.degrees(math.atan2(target_delta[1], target_delta[0]))
        tolerance = 1.0e-9
        if not (
            task.horizontal_distance_min_m - tolerance
            <= horizontal_radius
            <= task.horizontal_distance_max_m + tolerance
            and task.target_height_offset_min_m - tolerance
            <= height_offset
            <= task.target_height_offset_max_m + tolerance
            and task.target_azimuth_min_deg - tolerance
            <= azimuth_deg
            <= task.target_azimuth_max_deg + tolerance
        ):
            raise ValueError(
                "Testbed target lies outside the SAC checkpoint training "
                "distribution; use an in-support target so policy extrapolation "
                "is not confused with cable-model mismatch."
            )

    def _simulation_settings(self) -> SimulationSettings:
        return SimulationSettings(
            horizon_s=self.sac_settings.control_interval_s,
            simulation_dt_s=self.sac_settings.physics_dt_s,
            control_interval_s=self.sac_settings.control_interval_s,
            attachment_drop_m=self.sac_settings.attachment_drop_m,
            maximum_acceleration_m_s2=self.sac_settings.maximum_acceleration_m_s2,
            maximum_speed_m_s=self.sac_settings.maximum_speed_m_s,
        )

    def _target_frame(self, dtype: torch.dtype) -> torch.Tensor:
        direction = np.asarray(self.settings.impact_direction, dtype=np.float64)
        horizontal = direction.copy()
        horizontal[2] = 0.0
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm <= 1.0e-9:
            raise ValueError("SAC target frame requires a non-vertical impact direction.")
        forward = horizontal / horizontal_norm
        radial = (
            np.asarray(self.settings.target_position_m, dtype=np.float64)
            - np.asarray(self.settings.initial_drone_position_m, dtype=np.float64)
        )
        radial[2] = 0.0
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= 1.0e-9 or not np.allclose(
            forward,
            radial / radial_norm,
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError(
                "Horizontal impact direction must point radially from the "
                "initial drone position to the target, matching SAC training."
            )
        elevation_deg = math.degrees(math.atan2(direction[2], horizontal_norm))
        if not math.isclose(
            elevation_deg,
            self.task.desired_impact_elevation_deg,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "Impact-direction elevation must match the policy training task."
            )
        up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        lateral = np.cross(up, forward)
        frame = np.stack((forward, lateral, up), axis=0)
        return torch.as_tensor(frame, dtype=dtype, device=self.device).reshape(1, 3, 3)

    def _controller(self) -> NominalSacController:
        simulator = WhipSimulator(
            self.nominal_controller,
            self._simulation_settings(),
            device=self.device,
        )
        environment = VectorWhipEnvironment(
            simulator,
            1,
            self.episode_horizon_s,
            self.task,
            initial_drone_position_m=self.settings.initial_drone_position_m,
            seed=self.settings.seed,
        )
        environment.target_position_m[:] = torch.as_tensor(
            self.settings.target_position_m,
            dtype=environment.dtype,
            device=self.device,
        )
        environment.target_frame[:] = self._target_frame(environment.dtype)
        environment.minimum_impact_speed_m_s[:] = 0.5 * (
            self.task.minimum_impact_speed_min_m_s
            + self.task.minimum_impact_speed_max_m_s
        )
        environment.hit_tolerance_m[:] = self.task.hit_tolerance_m
        environment.impact_angle_deg[:] = self.task.impact_angle_deg
        environment.step_index.zero_()
        environment.previous_action.zero_()
        environment.peak_relative_cable_energy_j.zero_()
        return NominalSacController(
            environment,
            self.agent,
            self.marker_material_coordinates_m,
        )

    def _truth_metrics(
        self,
        state: DroneCableState,
    ) -> tuple[float, float, bool, bool]:
        tip = state.cable.positions_m[0, -1].detach().cpu().numpy()
        tip_velocity = state.cable.velocities_m_s[0, -1].detach().cpu().numpy()
        drone = state.drone_position_m[0].detach().cpu().numpy()
        drone_velocity = state.drone_velocity_m_s[0].detach().cpu().numpy()
        target = np.asarray(self.settings.target_position_m)
        direction = np.asarray(self.settings.impact_direction)
        distance = float(np.linalg.norm(tip - target))
        directed_speed = float(np.dot(tip_velocity, direction))
        lateral_speed = float(
            np.linalg.norm(tip_velocity - directed_speed * direction)
        )
        speed_threshold = 0.5 * (
            self.task.minimum_impact_speed_min_m_s
            + self.task.minimum_impact_speed_max_m_s
        )
        hit = (
            distance <= self.task.hit_tolerance_m
            and directed_speed >= speed_threshold
            and lateral_speed
            <= math.tan(math.radians(self.task.impact_angle_deg))
            * max(directed_speed, 0.0)
        )
        unsafe = (
            float(np.linalg.norm(drone - target)) < self.task.drone_keepout_radius_m
            or float(np.linalg.norm(drone_velocity))
            > self.sac_settings.maximum_speed_m_s
        )
        success = bool(hit and not unsafe)
        return distance, directed_speed, success, unsafe

    def _frame(
        self,
        *,
        time_s: float,
        hidden_state: DroneCableState,
        belief_state: DroneCableState,
        measurement: OptitrackMeasurement,
        command: ControllerCommand,
        marker_innovation_rms_m: float,
        tip_innovation_norm_m: float,
        action_updated: bool,
        timeout: bool,
    ) -> TestbedFrame:
        distance, directed_speed, success, unsafe = self._truth_metrics(hidden_state)
        return TestbedFrame(
            time_s=time_s,
            hidden_cable_positions_m=(
                hidden_state.cable.positions_m[0].detach().cpu().numpy()
            ),
            belief_cable_positions_m=(
                belief_state.cable.positions_m[0].detach().cpu().numpy()
            ),
            drone_position_m=measurement.drone_position_m,
            drone_velocity_m_s=measurement.drone_velocity_m_s,
            attachment_position_m=measurement.attachment_position_m,
            measured_marker_positions_m=measurement.marker_positions_m,
            measured_marker_velocities_m_s=measurement.marker_velocities_m_s,
            measured_marker_valid=measurement.marker_valid,
            measured_tip_position_m=measurement.free_tip_position_m,
            measurement_applied_acceleration_m_s2=(
                measurement.applied_acceleration_m_s2
            ),
            target_position_m=np.asarray(self.settings.target_position_m),
            impact_direction=np.asarray(self.settings.impact_direction),
            innovation_norm_m=tip_innovation_norm_m,
            marker_innovation_rms_m=marker_innovation_rms_m,
            tip_target_distance_m=distance,
            directional_tip_speed_m_s=directed_speed,
            commanded_acceleration_m_s2=command.acceleration_m_s2,
            action_updated=action_updated,
            done=bool(success or unsafe or timeout),
            success=bool(success),
            unsafe=bool(unsafe),
        )

    def iter_episode(
        self,
        *,
        cancelled: CancellationCallback | None = None,
    ) -> Iterator[TestbedFrame]:
        """Yield the causal 50-Hz episode record in time order."""

        controller = self._controller()
        hidden_simulator = WhipSimulator(
            self.hidden_plant,
            self._simulation_settings(),
            device=self.device,
        )
        hidden_state = hidden_simulator.initial_state(
            self.settings.initial_drone_position_m
        )
        frontend = _OrderedMarkerFrontend(
            np.asarray(
                self.hidden_plant.rod_material_coordinates_m, dtype=np.float64
            ),
            self.marker_material_coordinates_m,
        )
        previous_acceleration = np.zeros(3, dtype=np.float64)
        control_count = int(
            round(self.episode_horizon_s / self.sac_settings.control_interval_s)
        )
        time_s = 0.0
        measurement = frontend.measure(
            time_s=time_s,
            state=hidden_state,
            attachment_drop_m=self.sac_settings.attachment_drop_m,
            applied_acceleration_m_s2=previous_acceleration,
        )
        for control_index in range(control_count):
            command = controller.select_action(measurement)
            marker_innovation = controller.last_marker_innovation_rms_m
            tip_innovation = controller.last_tip_innovation_norm_m
            boundary_frame = self._frame(
                time_s=time_s,
                hidden_state=hidden_state,
                belief_state=controller.environment.state,
                measurement=measurement,
                command=command,
                marker_innovation_rms_m=marker_innovation,
                tip_innovation_norm_m=tip_innovation,
                action_updated=True,
                timeout=False,
            )
            yield boundary_frame
            if boundary_frame.done or (cancelled is not None and cancelled()):
                return

            nominal_rollout = controller.advance(command)
            acceleration = torch.tensor(
                command.acceleration_m_s2,
                dtype=hidden_simulator.dtype,
                device=self.device,
            ).reshape(1, 1, 3)
            with torch.no_grad():
                hidden_rollout = hidden_simulator.rollout(
                    hidden_state,
                    acceleration,
                    create_graph=False,
                )
            if hidden_rollout.frame_count != nominal_rollout.frame_count:
                raise RuntimeError("Hidden and nominal physics frame counts differ.")
            previous_acceleration = command.acceleration_m_s2
            last_control = control_index + 1 == control_count
            for frame_index in range(1, hidden_rollout.frame_count):
                final_in_block = frame_index + 1 == hidden_rollout.frame_count
                frame_time = time_s + float(
                    hidden_rollout.time_s[frame_index].detach().cpu()
                )
                hidden_frame_state = _rollout_state(hidden_rollout, frame_index)
                belief_frame_state = _rollout_state(nominal_rollout, frame_index)
                next_measurement = frontend.measure(
                    time_s=frame_time,
                    state=hidden_frame_state,
                    attachment_drop_m=self.sac_settings.attachment_drop_m,
                    applied_acceleration_m_s2=command.acceleration_m_s2,
                )
                # A nonterminal control boundary is emitted once, after the next
                # marker measurement has corrected the controller state and the
                # next action has been selected.
                if final_in_block and not last_control:
                    measurement = next_measurement
                    continue
                frame = self._frame(
                    time_s=frame_time,
                    hidden_state=hidden_frame_state,
                    belief_state=belief_frame_state,
                    measurement=next_measurement,
                    command=command,
                    marker_innovation_rms_m=marker_innovation,
                    tip_innovation_norm_m=tip_innovation,
                    action_updated=False,
                    timeout=bool(last_control and final_in_block),
                )
                yield frame
                if frame.done:
                    return
                if cancelled is not None and cancelled():
                    return
            hidden_state = hidden_rollout.final_state()
            time_s += self.sac_settings.control_interval_s

    def run_episode(
        self,
        *,
        frame_callback: FrameCallback | None = None,
        cancelled: CancellationCallback | None = None,
    ) -> TestbedEpisode:
        frames: list[TestbedFrame] = []
        was_cancelled = False
        iterator = self.iter_episode(cancelled=cancelled)
        for frame in iterator:
            frames.append(frame)
            if frame_callback is not None:
                frame_callback(frame)
        if cancelled is not None and cancelled() and frames and not frames[-1].done:
            was_cancelled = True
        if not frames:
            raise RuntimeError("Testbed produced no frames.")
        minimum_index = int(
            np.argmin([frame.tip_target_distance_m for frame in frames])
        )
        episode_success = any(frame.success for frame in frames)
        episode_unsafe = any(frame.unsafe for frame in frames)
        termination_reason = (
            "cancelled"
            if was_cancelled
            else "success"
            if episode_success
            else "unsafe"
            if episode_unsafe
            else "timeout"
        )
        return TestbedEpisode(
            schema=TESTBED_SCHEMA,
            settings=self.settings,
            sac_settings=self.sac_settings,
            task=self.task,
            frames=tuple(frames),
            marker_material_coordinates_m=self.marker_material_coordinates_m,
            nominal_source_path=str(self.nominal_source.source_path),
            nominal_source_sha256=self.nominal_source.sha256,
            nominal_controller_sha256=self.nominal_controller.sha256,
            hidden_source_path=str(self.hidden_source.source_path),
            hidden_source_sha256=self.hidden_source.sha256,
            hidden_plant_sha256=self.hidden_plant.sha256,
            policy_path=str(self.policy_path),
            policy_sha256=self.policy_sha256,
            action_semantics=(
                "world translational acceleration tracked by a point-mass drone; "
                "cable reaction is not fed back to the drone in Phase A"
            ),
            physics_rate_hz=1.0 / self.sac_settings.physics_dt_s,
            control_rate_hz=1.0 / self.sac_settings.control_interval_s,
            cancelled=was_cancelled,
            success=episode_success,
            unsafe=episode_unsafe,
            termination_reason=termination_reason,
            minimum_tip_error_m=frames[minimum_index].tip_target_distance_m,
            directional_tip_speed_at_minimum_error_m_s=(
                frames[minimum_index].directional_tip_speed_m_s
            ),
            maximum_directional_tip_speed_m_s=max(
                frame.directional_tip_speed_m_s for frame in frames
            ),
        )


def _rollout_state(rollout: TensorRollout, frame_index: int) -> DroneCableState:
    return DroneCableState(
        rollout.drone_positions_m[:, frame_index],
        rollout.drone_velocities_m_s[:, frame_index],
        DderState(
            rollout.cable_positions_m[:, frame_index],
            rollout.cable_velocities_m_s[:, frame_index],
        ),
    )


def _load_checkpoint_metadata(path: Path) -> tuple[SacSettings, TaskDistribution]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not is_supported_sac_checkpoint_schema(payload.get("schema")):
        raise ValueError("Selected file is not a compatible SAC checkpoint.")
    raw_settings = payload.get("settings")
    raw_task = payload.get("task")
    if not isinstance(raw_settings, dict) or not isinstance(raw_task, dict):
        raise ValueError("SAC checkpoint is missing settings or task metadata.")
    setting_names = {field.name for field in fields(SacSettings)}
    task_names = {field.name for field in fields(TaskDistribution)}
    legacy_setting_names = {"prior_replay_fraction"}
    legacy_task_names = {"minimum_extension_ratio"}
    if (
        not setting_names.issubset(raw_settings)
        or set(raw_settings) - setting_names - legacy_setting_names
        or not (task_names - legacy_task_names).issubset(raw_task)
        or set(raw_task) - task_names
    ):
        raise ValueError("SAC checkpoint settings/task schema does not match this code.")
    active_settings = {name: raw_settings[name] for name in setting_names}
    active_task = dict(raw_task)
    active_task.setdefault("minimum_extension_ratio", 0.0)
    return SacSettings(**active_settings), TaskDistribution(**active_task)


def load_adaptation_testbed(
    nominal_model_path: str | Path,
    policy_path: str | Path,
    *,
    hidden_model_path: str | Path | None = None,
    settings: TestbedSettings = TestbedSettings(),
    device: str | torch.device = "cuda",
) -> AdaptationTestbedSession:
    """Strictly load the nominal policy and an independent compatible plant."""

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA testbed requested but CUDA is unavailable.")
    policy = Path(policy_path).expanduser().resolve()
    if not policy.is_file():
        raise FileNotFoundError(f"SAC policy not found: {policy}")
    sac_settings, task = _load_checkpoint_metadata(policy)
    nominal_source = load_cable_model(nominal_model_path)
    nominal_controller = stable_controller_model(
        nominal_source,
        simulation_dt_s=sac_settings.physics_dt_s,
        node_count=sac_settings.controller_node_count,
        constraint_iterations=4,
    )
    simulation = SimulationSettings(
        horizon_s=sac_settings.control_interval_s,
        simulation_dt_s=sac_settings.physics_dt_s,
        control_interval_s=sac_settings.control_interval_s,
        attachment_drop_m=sac_settings.attachment_drop_m,
        maximum_acceleration_m_s2=sac_settings.maximum_acceleration_m_s2,
        maximum_speed_m_s=sac_settings.maximum_speed_m_s,
    )
    nominal_simulator = WhipSimulator(nominal_controller, simulation, device=target_device)
    nominal_environment = VectorWhipEnvironment(
        nominal_simulator,
        1,
        sac_settings.episode_horizon_s,
        task,
        initial_drone_position_m=settings.initial_drone_position_m,
        seed=settings.seed,
    )
    # This strict loader verifies both nominal model hashes before any hidden
    # plant is loaded; a hidden artifact can therefore never satisfy a policy
    # identity check on behalf of the nominal model.
    agent = load_policy(
        policy,
        sac_settings,
        task,
        nominal_environment.observation_size,
        nominal_environment.action_size,
        nominal_source,
        nominal_controller,
        device=target_device,
    )

    hidden_source = load_cable_model(
        nominal_model_path if hidden_model_path is None else hidden_model_path
    )
    if not math.isclose(
        hidden_source.cable_length_m,
        nominal_source.cable_length_m,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("Hidden and nominal cable lengths must match in Phase A.")
    if not np.allclose(
        hidden_source.model.parameters.gravity_camera_m_s2,
        nominal_source.model.parameters.gravity_camera_m_s2,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError("Hidden and nominal gravity vectors must match in Phase A.")
    hidden_snapshot = stable_controller_model(
        hidden_source,
        simulation_dt_s=sac_settings.physics_dt_s,
        # Primary mismatch experiments use the same numerical discretization
        # as the policy model.  This isolates EI/Cb mismatch from resolution.
        node_count=nominal_controller.node_count,
        constraint_iterations=4,
        bending_stiffness_scale=settings.hidden_ei_scale,
        bending_damping_scale=settings.hidden_cb_scale,
    )
    if not np.allclose(
        hidden_snapshot.rod_material_coordinates_m,
        nominal_controller.rod_material_coordinates_m,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Hidden and nominal reduced plants must share the controller material grid."
        )
    # Construct now to reject an unstable hidden plant during setup, not after
    # the GUI has started an episode.
    WhipSimulator(hidden_snapshot, simulation, device=target_device)
    return AdaptationTestbedSession(
        settings=settings,
        sac_settings=sac_settings,
        task=task,
        nominal_source=nominal_source,
        nominal_controller=nominal_controller,
        hidden_source=hidden_source,
        hidden_plant_snapshot=hidden_snapshot,
        policy_path=policy,
        policy_sha256=sha256_file(policy),
        agent=agent,
        device=target_device,
    )


def run_episode(
    nominal_model_path: str | Path,
    hidden_model_path: str | Path | None,
    policy_path: str | Path,
    settings: TestbedSettings = TestbedSettings(),
    *,
    device: str | torch.device = "cuda",
    frame_callback: FrameCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> TestbedEpisode:
    session = load_adaptation_testbed(
        nominal_model_path,
        policy_path,
        hidden_model_path=hidden_model_path,
        settings=settings,
        device=device,
    )
    return session.run_episode(
        frame_callback=frame_callback,
        cancelled=cancelled,
    )


def save_episode(episode: TestbedEpisode, path: str | Path) -> Path:
    """Atomically save one replayable ordered-marker episode as compressed NPZ."""

    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != ".npz":
        destination = destination.with_suffix(".npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    frames = episode.frames
    metadata = {
        "schema": episode.schema,
        "settings": asdict(episode.settings),
        "sac_settings": asdict(episode.sac_settings),
        "task": asdict(episode.task),
        "marker_material_coordinates_m": (
            episode.marker_material_coordinates_m.tolist()
        ),
        "nominal_source_path": episode.nominal_source_path,
        "nominal_source_sha256": episode.nominal_source_sha256,
        "nominal_controller_sha256": episode.nominal_controller_sha256,
        "hidden_source_path": episode.hidden_source_path,
        "hidden_source_sha256": episode.hidden_source_sha256,
        "hidden_plant_sha256": episode.hidden_plant_sha256,
        "policy_path": episode.policy_path,
        "policy_sha256": episode.policy_sha256,
        "action_semantics": episode.action_semantics,
        "physics_rate_hz": episode.physics_rate_hz,
        "control_rate_hz": episode.control_rate_hz,
        "cancelled": episode.cancelled,
        "success": episode.success,
        "unsafe": episode.unsafe,
        "termination_reason": episode.termination_reason,
        "minimum_tip_error_m": episode.minimum_tip_error_m,
        "directional_tip_speed_at_minimum_error_m_s": (
            episode.directional_tip_speed_at_minimum_error_m_s
        ),
        "maximum_directional_tip_speed_m_s": (
            episode.maximum_directional_tip_speed_m_s
        ),
        "frame_count": episode.frame_count,
        "action_count": episode.action_count,
        "measurement_contract": [
            "time_s",
            "drone_position_m",
            "drone_velocity_m_s",
            "attachment_position_m",
            "ordered_marker_material_coordinates_m",
            "ordered_marker_positions_m",
            "causal_marker_velocities_m_s",
            "marker_valid",
            "applied_acceleration_m_s2",
        ],
        "marker_association": "perfect ordered full-state association baseline",
        "target_support": "validated inside SAC checkpoint training distribution",
        "marker_grid": "one measured point per nominal controller material site",
        "velocity_estimator": "causal BDF1 for first interval, then causal BDF2",
        "controller_state_observer": (
            "full marker-position correction with nominal DDER predicted "
            "cable velocity; measured BDF velocity is diagnostic only"
        ),
    }
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, allow_nan=False)
                ),
                time_s=np.asarray([frame.time_s for frame in frames]),
                marker_material_coordinates_m=episode.marker_material_coordinates_m,
                hidden_cable_positions_m=np.stack(
                    [frame.hidden_cable_positions_m for frame in frames]
                ),
                belief_cable_positions_m=np.stack(
                    [frame.belief_cable_positions_m for frame in frames]
                ),
                drone_positions_m=np.stack([frame.drone_position_m for frame in frames]),
                drone_velocities_m_s=np.stack(
                    [frame.drone_velocity_m_s for frame in frames]
                ),
                attachment_positions_m=np.stack(
                    [frame.attachment_position_m for frame in frames]
                ),
                measured_tip_positions_m=np.stack(
                    [frame.measured_tip_position_m for frame in frames]
                ),
                measurement_applied_accelerations_m_s2=np.stack(
                    [
                        frame.measurement_applied_acceleration_m_s2
                        for frame in frames
                    ]
                ),
                measured_marker_positions_m=np.stack(
                    [frame.measured_marker_positions_m for frame in frames]
                ),
                measured_marker_velocities_m_s=np.stack(
                    [frame.measured_marker_velocities_m_s for frame in frames]
                ),
                measured_marker_valid=np.stack(
                    [frame.measured_marker_valid for frame in frames]
                ),
                target_positions_m=np.stack([frame.target_position_m for frame in frames]),
                impact_directions=np.stack([frame.impact_direction for frame in frames]),
                innovation_norm_m=np.asarray(
                    [frame.innovation_norm_m for frame in frames]
                ),
                marker_innovation_rms_m=np.asarray(
                    [frame.marker_innovation_rms_m for frame in frames]
                ),
                tip_target_distance_m=np.asarray(
                    [frame.tip_target_distance_m for frame in frames]
                ),
                directional_tip_speed_m_s=np.asarray(
                    [frame.directional_tip_speed_m_s for frame in frames]
                ),
                commanded_accelerations_m_s2=np.stack(
                    [frame.commanded_acceleration_m_s2 for frame in frames]
                ),
                action_updated=np.asarray(
                    [frame.action_updated for frame in frames], dtype=np.bool_
                ),
                done=np.asarray([frame.done for frame in frames], dtype=np.bool_),
                success=np.asarray([frame.success for frame in frames], dtype=np.bool_),
                unsafe=np.asarray([frame.unsafe for frame in frames], dtype=np.bool_),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = (
    "AdaptationTestbedSession",
    "ControllerCommand",
    "NominalSacController",
    "OptitrackMeasurement",
    "TESTBED_SCHEMA",
    "TestbedEpisode",
    "TestbedFrame",
    "TestbedSettings",
    "load_adaptation_testbed",
    "run_episode",
    "save_episode",
)
