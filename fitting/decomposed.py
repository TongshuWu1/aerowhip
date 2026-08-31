"""Milestone 3C decomposed full-episode identification.

This module orchestrates the existing production components without defining a
second simulator.  UAV parameters are identified from full physical episodes,
EI/Cb are identified with the measured Motive clamp as an external boundary,
and the frozen causal residual is trained only on causal suffix views.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable, Sequence

import numpy as np
import torch

from experimental_data.io import atomic_json, canonical_json_hash, sha256_file, utc_now
from simulator.cable.cuda_fixed_pcg import VALIDATED_OPTIMIZED_DAMPING_BACKEND
from simulator.cable.dder import DderRuntimeConstants, DderState
from simulator.parameters import (
    CableParameters,
    SimulatorParameters,
    SimulatorSettings,
    UAVResponseParameters,
)
from simulator.simulator import CoupledSimulator
from simulator.state import SimulatorState
from simulator.uav.model import FullStateUAVModel
from simulator.uav.residual import (
    RESIDUAL_FEATURE_NAMES,
    CausalTranslationalResidual,
    residual_feature,
)
from simulator.uav.state import (
    FullStateCommand,
    FullStateCommandSequence,
    ResidualHistoryState,
    UAVState,
)

from .artifacts import dataset_snapshot, write_csv
from .config import FitConfiguration
from .dataset import Dataset, ProcessedTake
from .episodes import PhysicalEpisode, build_physical_episodes, episode_manifest
from .initialization import initialize_uav_state, initialize_window
from .losses import pseudo_huber_distance, quaternion_geodesic_rad
from .production_support import (
    UAV_PARAMETER_NAMES,
    euler_xyz_from_quaternion as _euler_xyz_from_quaternion,
    measured_residual_history,
    wrap_angle as _wrap_angle,
)
from .windows import PredictionWindow


METHODOLOGY = "decomposed_full_episode_id_v1"
UAV_METHOD = "full_episode_sobol_adam_v1"
CABLE_METHOD = "measured_boundary_log_grid_v1"
RESIDUAL_METHOD = "full_episode_causal_residual_v1"
VALIDATION_METHOD = "full_episode_end_to_end_v1"
PHYSICAL_NAMES = (*UAV_PARAMETER_NAMES, "EI", "Cb")
INVALID_OBJECTIVE = 1.0e12
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "data" / "fit_results_decomposed"
DEFAULT_REPORT = (
    Path(__file__).resolve().parents[1]
    / "reports"
    / "MILESTONE3C_DECOMPOSED_REFIT_AND_VALIDATION_REPORT.md"
)
DEFAULT_FREEZE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "model_freezes"
    / "MODEL_FREEZE_DECOMPOSED_PRETEST"
)
DEFAULT_REMEASURED_RESULTS = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "fit_results_remeasured_geometry"
)
DEFAULT_REMEASURED_REPORT = (
    Path(__file__).resolve().parents[1]
    / "MILESTONE3C4_REMEASURED_GEOMETRY_PRE_MPPI_REPORT.md"
)
DEFAULT_REMEASURED_FREEZE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "model_freezes"
    / "MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI"
)
REMEASURED_GEOMETRY_VERSION = "remeasured_0p9525m_12node_v1"
REMEASURED_REST_LENGTHS_M = (
    0.0315,
    0.0315,
    0.0870,
    0.1000,
    0.1000,
    0.1000,
    0.1000,
    0.1000,
    0.1025,
    0.1000,
    0.1000,
)
PRE_MPPI_LEAD_TIMES_S = (0.10, 0.25, 0.50, 0.70, 1.00)
PRE_MPPI_THRESHOLDS = {
    "uav_position_rmse_m": 0.050,
    "uav_orientation_rmse_deg": 20.0,
    "distributed_marker_rmse_m": 0.150,
    "tip_rmse_m": 0.125,
}


@dataclass(frozen=True, slots=True)
class EpisodeView:
    """A full physical episode or a causally eligible suffix of one."""

    take: ProcessedTake
    parent: PhysicalEpisode
    start_index: int
    kind: str
    stop_index: int | None = None

    def __post_init__(self) -> None:
        if not self.parent.start_index <= self.start_index < self.parent.end_index:
            raise ValueError("Episode view start must lie inside its parent episode.")
        if self.kind not in {
            "physical_episode",
            "causal_residual_suffix",
            "task_horizon_prefix",
            "causal_residual_task_horizon_prefix",
        }:
            raise ValueError("Unsupported episode-view kind.")
        if self.stop_index is not None and not (
            self.start_index < self.stop_index <= self.parent.end_index
        ):
            raise ValueError("Episode view stop must lie after its start inside the parent.")

    @property
    def end_index(self) -> int:
        return self.parent.end_index if self.stop_index is None else self.stop_index

    @property
    def start_offset(self) -> int:
        return self.start_index - self.parent.start_index

    @property
    def step_count(self) -> int:
        return self.end_index - self.start_index

    @property
    def duration_s(self) -> float:
        time_s = self.take.arrays["time_s"]
        return float(time_s[self.end_index] - time_s[self.start_index])

    @property
    def view_id(self) -> str:
        suffix = {
            "physical_episode": "full",
            "causal_residual_suffix": "residual_suffix",
            "task_horizon_prefix": "task_horizon_prefix",
            "causal_residual_task_horizon_prefix": "residual_task_horizon_prefix",
        }[self.kind]
        return f"{self.parent.episode_id}__{suffix}"

    def initialization_window(self, config: FitConfiguration) -> PredictionWindow:
        return PredictionWindow(
            take_id=self.take.take_id,
            role=self.take.role,
            start_index=self.start_index,
            stop_index=self.end_index,
            history_start_index=self.start_index - config.initialization_history_frames + 1,
            start_s=float(self.take.arrays["time_s"][self.start_index]),
            end_s=float(self.take.arrays["time_s"][self.end_index]),
            valid_marker_fraction=float(
                np.mean(
                    self.parent.cable_observation_valid[
                        self.start_offset : self.start_offset + self.step_count + 1
                    ]
                )
            ),
        )


def physical_episode_views(
    takes: Iterable[ProcessedTake], config: FitConfiguration
) -> tuple[EpisodeView, ...]:
    return tuple(
        EpisodeView(take, episode, episode.start_index, "physical_episode")
        for take in takes
        for episode in build_physical_episodes(take, config)[0]
    )


def causal_residual_suffixes(
    takes: Iterable[ProcessedTake], config: FitConfiguration
) -> tuple[tuple[EpisodeView, ...], dict[str, object]]:
    views: list[EpisodeView] = []
    exclusions: list[dict[str, object]] = []
    physical_count = 0
    physical_duration = 0.0
    for take in takes:
        for episode in build_physical_episodes(take, config)[0]:
            physical_count += 1
            physical_duration += episode.duration_s
            start = episode.residual_eligible_start_index
            if start is None:
                exclusions.append(
                    {
                        "episode_id": episode.episode_id,
                        "reason": "no_causal_residual_suffix",
                        "excluded_duration_s": episode.duration_s,
                    }
                )
                continue
            view = EpisodeView(take, episode, start, "causal_residual_suffix")
            views.append(view)
            excluded = view.duration_s
            prefix = episode.duration_s - excluded
            if prefix > 1.0e-12:
                exclusions.append(
                    {
                        "episode_id": episode.episode_id,
                        "reason": "causal_history_prefix",
                        "excluded_duration_s": prefix,
                    }
                )
    eligible_duration = sum(view.duration_s for view in views)
    audit = {
        "schema": "causal_residual_coverage_v1",
        "physical_episode_count": physical_count,
        "residual_eligible_suffix_count": len(views),
        "physical_duration_s": physical_duration,
        "residual_eligible_duration_s": eligible_duration,
        "excluded_duration_s": physical_duration - eligible_duration,
        "eligibility_rule": "PhysicalEpisode.residual_eligible_start_index",
        "fixed_warmup_hard_coded": False,
        "exclusions": exclusions,
    }
    return tuple(views), audit


def task_horizon_views(
    views: Sequence[EpisodeView],
    *,
    dt_s: float,
    maximum_duration_s: float = 1.0,
) -> tuple[EpisodeView, ...]:
    """Truncate each existing causal view to one continuous task-horizon prefix.

    This changes only the terminal frame of validation.  It never moves the
    original initialization, creates rolling windows, or resets a trajectory.
    """

    if dt_s <= 0.0 or maximum_duration_s <= 0.0:
        raise ValueError("Task-horizon duration and frame interval must be positive.")
    maximum_steps = int(round(maximum_duration_s / dt_s))
    if maximum_steps < 1:
        raise ValueError("Task-horizon prefix must contain at least one step.")
    result: list[EpisodeView] = []
    for view in views:
        stop = min(view.end_index, view.start_index + maximum_steps)
        kind = (
            "causal_residual_task_horizon_prefix"
            if view.kind == "causal_residual_suffix"
            else "task_horizon_prefix"
        )
        result.append(EpisodeView(view.take, view.parent, view.start_index, kind, stop))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PreparedUAVEpisode:
    view: EpisodeView
    initial_state: UAVState
    command_position_m: torch.Tensor
    command_velocity_mps: torch.Tensor
    command_acceleration_mps2: torch.Tensor
    command_orientation_xyzw: torch.Tensor
    command_angular_velocity_body_rad_s: torch.Tensor
    measured_position_m: torch.Tensor
    measured_orientation_xyzw: torch.Tensor
    valid: torch.Tensor
    time_s: torch.Tensor

    @property
    def step_count(self) -> int:
        return self.view.step_count


@dataclass(frozen=True, slots=True)
class PreparedUAVTrainingBatch:
    """Padded execution batch; masks preserve each original episode exactly."""

    initial_state: UAVState
    commands: FullStateCommandSequence
    measured_position_m: torch.Tensor
    measured_orientation_xyzw: torch.Tensor
    observation_valid: torch.Tensor
    active_steps: torch.Tensor
    observation_weight: torch.Tensor
    residual_weight: torch.Tensor
    smoothness_weight: torch.Tensor

    @property
    def step_count(self) -> int:
        return self.commands.step_count


def _uav_simulator(
    settings: SimulatorSettings,
    *,
    device: torch.device,
    dtype: torch.dtype,
    residual_model: CausalTranslationalResidual | None = None,
) -> CoupledSimulator:
    return CoupledSimulator(
        settings.cable_configuration,
        settings.parameters,
        dt_s=settings.dt_s,
        device=device,
        dtype=dtype,
        uav_model=FullStateUAVModel(
            residual_model=residual_model,
        ),
        attachment_offset_body_m=settings.attachment_offset_body_m,
        attachment_tangent_body=settings.attachment_tangent_body,
    )


def _parameters(
    settings: SimulatorSettings,
    uav_values: torch.Tensor | Sequence[float],
    cable_values: tuple[torch.Tensor | float, torch.Tensor | float] | None = None,
) -> SimulatorParameters:
    values = torch.as_tensor(uav_values)
    if values.shape[-1] != 5:
        raise ValueError("UAV parameter vector must have five entries.")
    cable = settings.parameters.cable if cable_values is None else CableParameters(*cable_values)
    return SimulatorParameters(
        cable,
        # Independent scalar tensors avoid input-alias ambiguity in the
        # differentiable scan while preserving the exact values/gradients.
        UAVResponseParameters(*(values[..., index].clone() for index in range(5))),
    )


def _repeat_uav_state(state: UAVState, batch: int) -> UAVState:
    def repeat(value: torch.Tensor) -> torch.Tensor:
        return value.expand(batch, *value.shape[1:]).clone()

    history = state.residual_history
    return UAVState(
        repeat(state.position_m),
        repeat(state.velocity_m_s),
        repeat(state.orientation_xyzw),
        repeat(state.angular_velocity_world_rad_s),
        None if history is None else ResidualHistoryState(repeat(history.features)),
        (
            None
            if state.residual_acceleration_m_s2 is None
            else repeat(state.residual_acceleration_m_s2)
        ),
    )


def _finite_uav(state: UAVState) -> torch.Tensor:
    values = (
        state.position_m,
        state.velocity_m_s,
        state.orientation_xyzw,
        state.angular_velocity_world_rad_s,
    )
    finite = torch.stack(
        [torch.isfinite(value).reshape(value.shape[0], -1).all(dim=1) for value in values]
    ).all(dim=0)
    bounded = (
        (state.position_m.abs().amax(dim=1) < 1.0e3)
        & (state.velocity_m_s.abs().amax(dim=1) < 1.0e3)
        & (state.angular_velocity_world_rad_s.abs().amax(dim=1) < 1.0e4)
    )
    return finite & bounded


def _safe_uav(state: UAVState, fallback: UAVState, invalid: torch.Tensor) -> UAVState:
    def choose(value: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        mask = invalid.reshape(-1, *([1] * (value.ndim - 1)))
        return torch.where(mask, old, value)

    history = state.residual_history
    old_history = fallback.residual_history
    return UAVState(
        choose(state.position_m, fallback.position_m),
        choose(state.velocity_m_s, fallback.velocity_m_s),
        choose(state.orientation_xyzw, fallback.orientation_xyzw),
        choose(
            state.angular_velocity_world_rad_s,
            fallback.angular_velocity_world_rad_s,
        ),
        (
            None
            if history is None or old_history is None
            else ResidualHistoryState(choose(history.features, old_history.features))
        ),
        (
            None
            if state.residual_acceleration_m_s2 is None
            or fallback.residual_acceleration_m_s2 is None
            else choose(
                state.residual_acceleration_m_s2,
                fallback.residual_acceleration_m_s2,
            )
        ),
    )


def prepare_uav_episodes(
    views: Sequence[EpisodeView],
    config: FitConfiguration,
    simulator: CoupledSimulator,
    *,
    residual_history: bool = False,
) -> tuple[PreparedUAVEpisode, ...]:
    prepared: list[PreparedUAVEpisode] = []
    for view in views:
        take = view.take
        arrays = take.arrays
        window = view.initialization_window(config)
        initial = initialize_uav_state(
            take, window, device=simulator.device, dtype=simulator.dtype
        )
        if residual_history:
            if view.kind not in {
                "causal_residual_suffix",
                "causal_residual_task_horizon_prefix",
            }:
                raise ValueError("Residual history is permitted only for causal suffixes.")
            history = measured_residual_history(
                take,
                window,
                config,
                dtype=simulator.dtype,
                device=simulator.device,
            )
            initial = UAVState(
                initial.position_m,
                initial.velocity_m_s,
                initial.orientation_xyzw,
                initial.angular_velocity_world_rad_s,
                history,
                torch.zeros_like(initial.position_m),
            )

        def tensor(name: str, state_series: bool = False) -> torch.Tensor:
            stop = view.end_index + (1 if state_series else 0)
            return torch.as_tensor(
                arrays[name][view.start_index:stop],
                dtype=simulator.dtype,
                device=simulator.device,
            )

        offset = view.start_offset
        valid = torch.as_tensor(
            view.parent.uav_observation_valid[offset : offset + view.step_count + 1],
            dtype=torch.bool,
            device=simulator.device,
        )
        prepared.append(
            PreparedUAVEpisode(
                view,
                initial,
                tensor("command_position_m"),
                tensor("command_velocity_mps"),
                tensor("command_acceleration_mps2"),
                tensor("command_orientation_xyzw"),
                tensor("command_angular_velocity"),
                tensor("uav_position_m", True),
                tensor("uav_orientation_xyzw", True),
                valid,
                tensor("time_s", True),
            )
        )
    return tuple(prepared)


def prepare_uav_training_batch(
    episodes: Sequence[PreparedUAVEpisode],
) -> PreparedUAVTrainingBatch:
    """Pad episodes for parallel execution without changing their boundaries."""

    if not episodes:
        raise ValueError("A UAV training batch requires at least one episode.")
    device = episodes[0].initial_state.position_m.device
    dtype = episodes[0].initial_state.position_m.dtype
    maximum = max(episode.step_count for episode in episodes)
    batch = len(episodes)
    counts = _take_counts(episodes)
    take_count = len(counts)

    def padded(last_dimension: int, *, states: bool = False) -> torch.Tensor:
        length = maximum + (1 if states else 0)
        return torch.zeros((length, batch, last_dimension), dtype=dtype, device=device)

    command_position = padded(3)
    command_velocity = padded(3)
    command_acceleration = padded(3)
    command_orientation = padded(4)
    command_orientation[..., 3] = 1.0
    command_angular_velocity = padded(3)
    measured_position = padded(3, states=True)
    measured_orientation = padded(4, states=True)
    measured_orientation[..., 3] = 1.0
    observation_valid = torch.zeros(
        (maximum + 1, batch), dtype=torch.bool, device=device
    )
    active = torch.zeros((maximum, batch), dtype=torch.bool, device=device)
    observation_weight = torch.zeros(
        (maximum + 1, batch), dtype=dtype, device=device
    )
    residual_weight = torch.zeros((maximum, batch), dtype=dtype, device=device)
    smoothness_weight = torch.zeros((maximum, batch), dtype=dtype, device=device)

    for column, episode in enumerate(episodes):
        steps = episode.step_count
        command_position[:steps, column] = episode.command_position_m
        command_velocity[:steps, column] = episode.command_velocity_mps
        command_acceleration[:steps, column] = episode.command_acceleration_mps2
        command_orientation[:steps, column] = episode.command_orientation_xyzw
        command_angular_velocity[:steps, column] = (
            episode.command_angular_velocity_body_rad_s
        )
        measured_position[: steps + 1, column] = episode.measured_position_m
        measured_orientation[: steps + 1, column] = episode.measured_orientation_xyzw
        observation_valid[: steps + 1, column] = episode.valid
        active[:steps, column] = True
        denominators = counts[episode.view.take.take_id]
        observation_weight[: steps + 1, column] = (
            episode.valid.to(dtype)
            / max(denominators["observation"], 1.0)
            / take_count
        )
        residual_weight[:steps, column] = (
            1.0 / max(denominators["residual"], 1.0) / take_count
        )
        if steps > 1:
            smoothness_weight[1:steps, column] = (
                1.0 / max(denominators["smoothness"], 1.0) / take_count
            )

    initial_history = episodes[0].initial_state.residual_history
    if any(
        (episode.initial_state.residual_history is None) != (initial_history is None)
        for episode in episodes
    ):
        raise ValueError("Cannot mix residual and non-residual episodes in one batch.")
    initial_state = UAVState(
        torch.cat([episode.initial_state.position_m for episode in episodes]),
        torch.cat([episode.initial_state.velocity_m_s for episode in episodes]),
        torch.cat([episode.initial_state.orientation_xyzw for episode in episodes]),
        torch.cat(
            [episode.initial_state.angular_velocity_world_rad_s for episode in episodes]
        ),
        (
            None
            if initial_history is None
            else ResidualHistoryState(
                torch.cat(
                    [
                        episode.initial_state.residual_history.features  # type: ignore[union-attr]
                        for episode in episodes
                    ]
                )
            )
        ),
        (
            None
            if initial_history is None
            else torch.cat(
                [
                    episode.initial_state.residual_acceleration_m_s2
                    for episode in episodes
                    if episode.initial_state.residual_acceleration_m_s2 is not None
                ]
            )
        ),
    )
    return PreparedUAVTrainingBatch(
        initial_state,
        FullStateCommandSequence(
            command_position,
            command_velocity,
            command_acceleration,
            command_orientation,
            command_angular_velocity,
        ),
        measured_position,
        measured_orientation,
        observation_valid,
        active,
        observation_weight,
        residual_weight,
        smoothness_weight,
    )


def _command(episode: PreparedUAVEpisode, frame: int, batch: int) -> FullStateCommand:
    def expand(value: torch.Tensor) -> torch.Tensor:
        return value[frame].reshape(1, -1).expand(batch, -1)

    return FullStateCommand(
        expand(episode.command_position_m),
        expand(episode.command_velocity_mps),
        expand(episode.command_acceleration_mps2),
        expand(episode.command_orientation_xyzw),
        expand(episode.command_angular_velocity_body_rad_s),
    )


class _TracedUAVStep(torch.nn.Module):
    """Trace wrapper around the production model step for fitting throughput."""

    def __init__(self, model: FullStateUAVModel, dt_s: float) -> None:
        super().__init__()
        self.model = model
        self.dt_s = float(dt_s)
        if model.residual_model is not None:
            # Register the exact production residual module with TorchScript.
            self.residual_model = model.residual_model

    def forward(
        self,
        p: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        omega: torch.Tensor,
        history: torch.Tensor,
        last_residual: torch.Tensor,
        command_position: torch.Tensor,
        command_velocity: torch.Tensor,
        command_acceleration: torch.Tensor,
        command_orientation: torch.Tensor,
        command_angular_velocity: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        residual_enabled = self.model.residual_model is not None
        if residual_enabled:
            state = UAVState(
                p,
                v,
                q,
                omega,
                ResidualHistoryState(history),
                last_residual,
            )
        else:
            state = UAVState(p, v, q, omega)
        following = self.model.step(
            state,
            FullStateCommand(
                command_position,
                command_velocity,
                command_acceleration,
                command_orientation,
                command_angular_velocity,
            ),
            self.dt_s,
            UAVResponseParameters(*(values[index] for index in range(5))),
        )
        next_history = (
            following.residual_history.features
            if following.residual_history is not None
            else history
        )
        next_residual = (
            following.residual_acceleration_m_s2
            if following.residual_acceleration_m_s2 is not None
            else last_residual
        )
        return (
            following.position_m,
            following.velocity_m_s,
            following.orientation_xyzw,
            following.angular_velocity_world_rad_s,
            next_history,
            next_residual,
        )


def trace_uav_step(
    batch: PreparedUAVTrainingBatch,
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    values: torch.Tensor,
) -> torch.jit.ScriptModule:
    """Trace the authoritative step once; candidate values remain live inputs."""

    if not isinstance(simulator.uav_model, FullStateUAVModel):
        raise TypeError("Milestone 3C requires FullStateUAVModel.")
    state = batch.initial_state
    first = batch.commands.command_at(0)
    history = (
        state.residual_history.features
        if state.residual_history is not None
        else torch.zeros(
            (state.batch_size, 10, 9),
            dtype=state.position_m.dtype,
            device=state.position_m.device,
        )
    )
    last_residual = (
        state.residual_acceleration_m_s2
        if state.residual_acceleration_m_s2 is not None
        else torch.zeros_like(state.position_m)
    )
    inputs: tuple[torch.Tensor, ...] = (
        state.position_m,
        state.velocity_m_s,
        state.orientation_xyzw,
        state.angular_velocity_world_rad_s,
        history,
        last_residual,
        first.position_m,
        first.velocity_m_s,
        first.acceleration_m_s2,
        first.orientation_xyzw,
        first.angular_velocity_body_rad_s,
        values,
    )
    return torch.jit.trace(
        _TracedUAVStep(simulator.uav_model, settings.dt_s),
        inputs,
        check_trace=False,
        strict=True,
    )


class _UAVTrainingLoop(torch.nn.Module):
    __constants__ = [
        "residual_enabled",
        "position_factor",
        "orientation_factor",
        "magnitude_factor",
        "smoothness_factor",
    ]

    def __init__(
        self,
        stepper: torch.jit.ScriptModule,
        *,
        residual_enabled: bool,
        position_factor: float,
        orientation_factor: float,
        magnitude_factor: float,
        smoothness_factor: float,
    ) -> None:
        super().__init__()
        self.stepper = stepper
        self.residual_enabled = residual_enabled
        self.position_factor = position_factor
        self.orientation_factor = orientation_factor
        self.magnitude_factor = magnitude_factor
        self.smoothness_factor = smoothness_factor

    def forward(
        self,
        p: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor,
        omega: torch.Tensor,
        history: torch.Tensor,
        last_residual: torch.Tensor,
        command_position: torch.Tensor,
        command_velocity: torch.Tensor,
        command_acceleration: torch.Tensor,
        command_orientation: torch.Tensor,
        command_angular_velocity: torch.Tensor,
        measured_position: torch.Tensor,
        measured_orientation: torch.Tensor,
        observation_valid: torch.Tensor,
        active_steps: torch.Tensor,
        observation_weight: torch.Tensor,
        residual_weight: torch.Tensor,
        smoothness_weight: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        position_sum = torch.zeros((), dtype=p.dtype, device=p.device)
        orientation_sum = torch.zeros((), dtype=p.dtype, device=p.device)
        magnitude_sum = torch.zeros((), dtype=p.dtype, device=p.device)
        smoothness_sum = torch.zeros((), dtype=p.dtype, device=p.device)
        previous_residual = last_residual
        for frame in range(command_position.shape[0]):
            outputs = self.stepper(
                p,
                v,
                q,
                omega,
                history,
                last_residual,
                command_position[frame],
                command_velocity[frame],
                command_acceleration[frame],
                command_orientation[frame],
                command_angular_velocity[frame],
                values,
            )
            inactive = ~active_steps[frame]
            vector_mask = inactive[:, None]
            history_mask = inactive[:, None, None]
            next_p = torch.where(vector_mask, p, outputs[0])
            next_v = torch.where(vector_mask, v, outputs[1])
            next_q = torch.where(vector_mask, q, outputs[2])
            next_omega = torch.where(vector_mask, omega, outputs[3])
            next_history = torch.where(history_mask, history, outputs[4])
            next_residual = torch.where(vector_mask, last_residual, outputs[5])
            valid = observation_valid[frame + 1]
            weight = observation_weight[frame + 1]
            position_error2 = torch.sum(
                (next_p - measured_position[frame + 1]).square(), dim=1
            )
            predicted_q = next_q / torch.clamp(
                torch.linalg.vector_norm(next_q, dim=-1, keepdim=True), min=1.0e-12
            )
            measured_q = measured_orientation[frame + 1] / torch.clamp(
                torch.linalg.vector_norm(
                    measured_orientation[frame + 1], dim=-1, keepdim=True
                ),
                min=1.0e-12,
            )
            dot = torch.abs(torch.sum(predicted_q * measured_q, dim=-1))
            orientation_error2 = (
                2.0 * torch.acos(torch.clamp(dot, max=1.0 - 1.0e-12))
            ).square()
            position_sum = position_sum + torch.sum(
                torch.where(valid, position_error2 * weight, torch.zeros_like(weight))
            )
            orientation_sum = orientation_sum + torch.sum(
                torch.where(valid, orientation_error2 * weight, torch.zeros_like(weight))
            )
            if self.residual_enabled:
                magnitude_sum = magnitude_sum + torch.sum(
                    torch.sum(next_residual.square(), dim=1) * residual_weight[frame]
                )
                smoothness_sum = smoothness_sum + torch.sum(
                    torch.sum((next_residual - previous_residual).square(), dim=1)
                    * smoothness_weight[frame]
                )
            p, v, q, omega = next_p, next_v, next_q, next_omega
            history, last_residual = next_history, next_residual
            previous_residual = next_residual
        position = self.position_factor * position_sum
        orientation = self.orientation_factor * orientation_sum
        magnitude = self.magnitude_factor * magnitude_sum
        smoothness = self.smoothness_factor * smoothness_sum
        return position + orientation + magnitude + smoothness, position, orientation, magnitude, smoothness


def script_uav_training_loop(
    batch: PreparedUAVTrainingBatch,
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    values: torch.Tensor,
    *,
    residual_enabled: bool,
) -> torch.jit.ScriptModule:
    scale = float(config.uav_residual_ablation["acceleration_regularization_scale_mps2"])
    return torch.jit.script(
        _UAVTrainingLoop(
            trace_uav_step(batch, simulator, settings, values),
            residual_enabled=residual_enabled,
            position_factor=(
                config.joint_weights["uav_position"]
                / config.normalization_scales["uav_position_m"] ** 2
            ),
            orientation_factor=(
                config.joint_weights["uav_orientation"]
                / config.normalization_scales["uav_orientation_rad"] ** 2
            ),
            magnitude_factor=(
                float(config.uav_residual_ablation["lambda_magnitude"]) / scale**2
                if residual_enabled
                else 0.0
            ),
            smoothness_factor=(
                float(config.uav_residual_ablation["lambda_smoothness"]) / scale**2
                if residual_enabled
                else 0.0
            ),
        )
    )


def uav_population_objective(
    episodes: Sequence[PreparedUAVEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    population: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate a no-gradient UAV population with equal top-level take weight."""

    population = torch.as_tensor(
        population, dtype=simulator.dtype, device=simulator.device
    )
    if population.ndim == 1:
        population = population[None]
    batch = int(population.shape[0])
    parameters = _parameters(settings, population)
    invalid = torch.zeros(batch, dtype=torch.bool, device=simulator.device)
    take_sums: dict[str, dict[str, torch.Tensor]] = {}
    with torch.no_grad():
        for episode in episodes:
            state = _repeat_uav_state(episode.initial_state, batch)
            fallback = state
            values = take_sums.setdefault(
                episode.view.take.take_id,
                {
                    "position": torch.zeros(batch, dtype=simulator.dtype, device=simulator.device),
                    "orientation": torch.zeros(batch, dtype=simulator.dtype, device=simulator.device),
                    "count": torch.zeros((), dtype=simulator.dtype, device=simulator.device),
                },
            )
            for frame in range(episode.step_count):
                state = simulator.uav_model.step(
                    state, _command(episode, frame, batch), settings.dt_s, parameters.uav
                )
                invalid = invalid | (~_finite_uav(state))
                state = _safe_uav(state, fallback, invalid)
                if bool(episode.valid[frame + 1]):
                    position = torch.sum(
                        (state.position_m - episode.measured_position_m[frame + 1]).square(),
                        dim=1,
                    )
                    orientation = quaternion_geodesic_rad(
                        state.orientation_xyzw,
                        episode.measured_orientation_xyzw[frame + 1].expand(batch, -1),
                    ).square()
                    active = ~invalid
                    values["position"] += torch.where(active, position, torch.zeros_like(position))
                    values["orientation"] += torch.where(active, orientation, torch.zeros_like(orientation))
                    values["count"] += 1.0
    per_take = []
    for values in take_sums.values():
        count = torch.clamp(values["count"], min=1.0)
        per_take.append(
            config.joint_weights["uav_position"]
            * (values["position"] / count)
            / config.normalization_scales["uav_position_m"] ** 2
            + config.joint_weights["uav_orientation"]
            * (values["orientation"] / count)
            / config.normalization_scales["uav_orientation_rad"] ** 2
        )
    objective = torch.mean(torch.stack(per_take), dim=0)
    objective = torch.where(
        invalid, torch.full_like(objective, INVALID_OBJECTIVE), objective
    )
    return objective, invalid, {
        "take_count": torch.tensor(float(len(take_sums)), device=simulator.device),
    }


def _single_uav_episode_loss(
    episode: PreparedUAVEpisode,
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    values: torch.Tensor,
    *,
    include_residual_regularization: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    state = episode.initial_state
    parameters = _parameters(settings, values)
    position_sum = torch.zeros((), dtype=simulator.dtype, device=simulator.device)
    orientation_sum = torch.zeros_like(position_sum)
    magnitude_sum = torch.zeros_like(position_sum)
    smoothness_sum = torch.zeros_like(position_sum)
    observation_count = 0
    residual_count = 0
    smoothness_count = 0
    previous_residual: torch.Tensor | None = None
    for frame in range(episode.step_count):
        state = simulator.uav_model.step(
            state, _command(episode, frame, 1), settings.dt_s, parameters.uav
        )
        if not bool(_finite_uav(state).all()):
            raise FloatingPointError(f"Non-finite UAV state in {episode.view.view_id}.")
        if bool(episode.valid[frame + 1]):
            position_sum = position_sum + torch.sum(
                (state.position_m[0] - episode.measured_position_m[frame + 1]).square()
            )
            orientation_sum = orientation_sum + quaternion_geodesic_rad(
                state.orientation_xyzw[0], episode.measured_orientation_xyzw[frame + 1]
            ).square()
            observation_count += 1
        residual = state.residual_acceleration_m_s2
        if include_residual_regularization:
            if residual is None:
                raise RuntimeError("Residual-enabled episode did not produce Delta_a.")
            magnitude_sum = magnitude_sum + torch.sum(residual.square())
            residual_count += 1
            if previous_residual is not None:
                smoothness_sum = smoothness_sum + torch.sum(
                    (residual - previous_residual).square()
                )
                smoothness_count += 1
            previous_residual = residual
    return position_sum, {
        "orientation_sum": orientation_sum,
        "observation_count": torch.tensor(
            float(observation_count), dtype=simulator.dtype, device=simulator.device
        ),
        "magnitude_sum": magnitude_sum,
        "magnitude_count": torch.tensor(
            float(residual_count), dtype=simulator.dtype, device=simulator.device
        ),
        "smoothness_sum": smoothness_sum,
        "smoothness_count": torch.tensor(
            float(smoothness_count), dtype=simulator.dtype, device=simulator.device
        ),
    }


def _take_counts(episodes: Sequence[PreparedUAVEpisode]) -> dict[str, dict[str, float]]:
    counts: dict[str, dict[str, float]] = {}
    for episode in episodes:
        item = counts.setdefault(
            episode.view.take.take_id,
            {"observation": 0.0, "residual": 0.0, "smoothness": 0.0},
        )
        item["observation"] += float(episode.valid[1:].sum().cpu())
        item["residual"] += episode.step_count
        item["smoothness"] += max(0, episode.step_count - 1)
    return counts


def differentiable_uav_training_step(
    episodes: Sequence[PreparedUAVEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    values: torch.Tensor,
    *,
    residual_parameters: Iterable[torch.nn.Parameter] | None = None,
) -> tuple[float, dict[str, float]]:
    """Backpropagate the exact equal-take objective one episode at a time."""

    counts = _take_counts(episodes)
    take_count = len(counts)
    if take_count < 1:
        raise ValueError("Training requires at least one physical take.")
    totals = {
        "objective": 0.0,
        "position": 0.0,
        "orientation": 0.0,
        "magnitude": 0.0,
        "smoothness": 0.0,
    }
    residual_active = residual_parameters is not None
    scale = float(config.uav_residual_ablation["acceleration_regularization_scale_mps2"])
    value_gradient = torch.zeros_like(values) if not residual_active else None
    for episode in episodes:
        take_id = episode.view.take.take_id
        denom = counts[take_id]
        position_sum, components = _single_uav_episode_loss(
            episode,
            simulator,
            settings,
            config,
            values,
            include_residual_regularization=residual_active,
        )
        position = (
            config.joint_weights["uav_position"]
            * position_sum
            / max(denom["observation"], 1.0)
            / config.normalization_scales["uav_position_m"] ** 2
            / take_count
        )
        orientation = (
            config.joint_weights["uav_orientation"]
            * components["orientation_sum"]
            / max(denom["observation"], 1.0)
            / config.normalization_scales["uav_orientation_rad"] ** 2
            / take_count
        )
        objective = position + orientation
        magnitude = torch.zeros_like(objective)
        smoothness = torch.zeros_like(objective)
        if residual_active:
            magnitude = (
                float(config.uav_residual_ablation["lambda_magnitude"])
                * components["magnitude_sum"]
                / max(denom["residual"], 1.0)
                / scale**2
                / take_count
            )
            smoothness = (
                float(config.uav_residual_ablation["lambda_smoothness"])
                * components["smoothness_sum"]
                / max(denom["smoothness"], 1.0)
                / scale**2
                / take_count
            )
            objective = objective + magnitude + smoothness
        if residual_active:
            objective.backward()
        else:
            # Each long episode is backpropagated and released independently
            # to keep memory bounded.  ``values`` is the shared output of the
            # smooth bounded transform, so accumulate dJ/dvalues episode by
            # episode and traverse that small transform exactly once below.
            episode_gradient = torch.autograd.grad(objective, values)[0]
            assert value_gradient is not None
            value_gradient = value_gradient + episode_gradient
        totals["objective"] += float(objective.detach().cpu())
        totals["position"] += float(position.detach().cpu())
        totals["orientation"] += float(orientation.detach().cpu())
        totals["magnitude"] += float(magnitude.detach().cpu())
        totals["smoothness"] += float(smoothness.detach().cpu())
    if not residual_active:
        assert value_gradient is not None
        values.backward(value_gradient)
    return totals["objective"], totals


def differentiable_batched_uav_training_step(
    batch: PreparedUAVTrainingBatch,
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    values: torch.Tensor,
    *,
    residual_parameters: Iterable[torch.nn.Parameter] | None = None,
    stepper: torch.jit.ScriptModule | None = None,
    training_loop: torch.jit.ScriptModule | None = None,
) -> tuple[float, dict[str, float]]:
    """Exact masked full-episode objective with episodes executed in parallel."""

    state = batch.initial_state
    if training_loop is not None:
        history = (
            state.residual_history.features
            if state.residual_history is not None
            else torch.zeros(
                (state.batch_size, 10, 9),
                dtype=state.position_m.dtype,
                device=state.position_m.device,
            )
        )
        last_residual = (
            state.residual_acceleration_m_s2
            if state.residual_acceleration_m_s2 is not None
            else torch.zeros_like(state.position_m)
        )
        outputs = training_loop(
            state.position_m,
            state.velocity_m_s,
            state.orientation_xyzw,
            state.angular_velocity_world_rad_s,
            history,
            last_residual,
            batch.commands.positions_m,
            batch.commands.velocities_m_s,
            batch.commands.accelerations_m_s2,
            batch.commands.orientations_xyzw,
            batch.commands.angular_velocities_body_rad_s,
            batch.measured_position_m,
            batch.measured_orientation_xyzw,
            batch.observation_valid,
            batch.active_steps,
            batch.observation_weight,
            batch.residual_weight,
            batch.smoothness_weight,
            values,
        )
        if not bool(torch.isfinite(outputs[0])):
            raise FloatingPointError("Non-finite scripted full-episode UAV objective.")
        outputs[0].backward()
        components = {
            "objective": float(outputs[0].detach().cpu()),
            "position": float(outputs[1].detach().cpu()),
            "orientation": float(outputs[2].detach().cpu()),
            "magnitude": float(outputs[3].detach().cpu()),
            "smoothness": float(outputs[4].detach().cpu()),
        }
        return components["objective"], components
    parameters = _parameters(settings, values)
    position_sum = torch.zeros((), dtype=simulator.dtype, device=simulator.device)
    orientation_sum = torch.zeros_like(position_sum)
    magnitude_sum = torch.zeros_like(position_sum)
    smoothness_sum = torch.zeros_like(position_sum)
    previous_residual: torch.Tensor | None = None
    residual_active = residual_parameters is not None
    for frame in range(batch.step_count):
        current = state
        command = batch.commands.command_at(frame)
        if stepper is None:
            following = simulator.uav_model.step(
                current,
                command,
                settings.dt_s,
                parameters.uav,
            )
        else:
            history = (
                current.residual_history.features
                if current.residual_history is not None
                else torch.zeros(
                    (current.batch_size, 10, 9),
                    dtype=current.position_m.dtype,
                    device=current.position_m.device,
                )
            )
            last_residual = (
                current.residual_acceleration_m_s2
                if current.residual_acceleration_m_s2 is not None
                else torch.zeros_like(current.position_m)
            )
            outputs = stepper(
                current.position_m,
                current.velocity_m_s,
                current.orientation_xyzw,
                current.angular_velocity_world_rad_s,
                history,
                last_residual,
                command.position_m,
                command.velocity_m_s,
                command.acceleration_m_s2,
                command.orientation_xyzw,
                command.angular_velocity_body_rad_s,
                values,
            )
            following = UAVState(
                outputs[0],
                outputs[1],
                outputs[2],
                outputs[3],
                (
                    ResidualHistoryState(outputs[4])
                    if current.residual_history is not None
                    else None
                ),
                outputs[5] if current.residual_history is not None else None,
            )
        state = _safe_uav(following, current, ~batch.active_steps[frame])
        valid = batch.observation_valid[frame + 1]
        weight = batch.observation_weight[frame + 1]
        position_error2 = torch.sum(
            (state.position_m - batch.measured_position_m[frame + 1]).square(), dim=1
        )
        orientation_error2 = quaternion_geodesic_rad(
            state.orientation_xyzw,
            batch.measured_orientation_xyzw[frame + 1],
        ).square()
        position_sum = position_sum + torch.sum(
            torch.where(valid, position_error2 * weight, torch.zeros_like(weight))
        )
        orientation_sum = orientation_sum + torch.sum(
            torch.where(valid, orientation_error2 * weight, torch.zeros_like(weight))
        )
        if residual_active:
            residual = state.residual_acceleration_m_s2
            if residual is None:
                raise RuntimeError("Residual-enabled batch did not produce Delta_a.")
            magnitude_sum = magnitude_sum + torch.sum(
                torch.sum(residual.square(), dim=1) * batch.residual_weight[frame]
            )
            if previous_residual is not None:
                smoothness_sum = smoothness_sum + torch.sum(
                    torch.sum((residual - previous_residual).square(), dim=1)
                    * batch.smoothness_weight[frame]
                )
            previous_residual = residual
    if not bool(_finite_uav(state).all()):
        raise FloatingPointError("Non-finite state in batched full-episode UAV fit.")
    position = (
        config.joint_weights["uav_position"]
        * position_sum
        / config.normalization_scales["uav_position_m"] ** 2
    )
    orientation = (
        config.joint_weights["uav_orientation"]
        * orientation_sum
        / config.normalization_scales["uav_orientation_rad"] ** 2
    )
    magnitude = torch.zeros_like(position)
    smoothness = torch.zeros_like(position)
    if residual_active:
        scale = float(
            config.uav_residual_ablation["acceleration_regularization_scale_mps2"]
        )
        magnitude = (
            float(config.uav_residual_ablation["lambda_magnitude"])
            * magnitude_sum
            / scale**2
        )
        smoothness = (
            float(config.uav_residual_ablation["lambda_smoothness"])
            * smoothness_sum
            / scale**2
        )
    objective = position + orientation + magnitude + smoothness
    objective.backward()
    components = {
        "objective": float(objective.detach().cpu()),
        "position": float(position.detach().cpu()),
        "orientation": float(orientation.detach().cpu()),
        "magnitude": float(magnitude.detach().cpu()),
        "smoothness": float(smoothness.detach().cpu()),
    }
    return components["objective"], components


def fit_uav_physics(
    episodes: Sequence[PreparedUAVEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    output: Path,
    *,
    deadline: float,
    sobol_count: int = 64,
    refinements: int = 2,
    adam_updates: int = 300,
) -> tuple[torch.Tensor, dict[str, object]]:
    lower = torch.tensor(
        [config.bounds[name][0] for name in UAV_PARAMETER_NAMES],
        dtype=simulator.dtype,
        device=simulator.device,
    )
    upper = torch.tensor(
        [config.bounds[name][1] for name in UAV_PARAMETER_NAMES],
        dtype=simulator.dtype,
        device=simulator.device,
    )
    seed = int(config.optimizer["seed"])
    sobol = torch.quasirandom.SobolEngine(5, scramble=True, seed=seed)
    started = time.perf_counter()
    candidate_batches: list[torch.Tensor] = []
    objective_batches: list[torch.Tensor] = []
    invalid_batches: list[torch.Tensor] = []
    sobol_rows: list[dict[str, object]] = []
    # The first authoritative batch is exactly the prescribed 64-point Sobol
    # design.  Full-episode attitude-coupled trajectories can be unstable for
    # broad gain combinations.  If that initial batch supplies fewer than the
    # two finite Adam starts required by the protocol, continue the *same*
    # deterministic Sobol stream in 64-point blocks.  This does not narrow the
    # bounds, shorten episodes, or use Validation information.
    maximum_candidates = 512
    while len(sobol_rows) < maximum_candidates:
        samples = sobol.draw(sobol_count).to(
            dtype=simulator.dtype, device=simulator.device
        )
        batch_candidates = lower + samples * (upper - lower)
        batch_objectives, batch_invalid, _ = uav_population_objective(
            episodes, simulator, settings, config, batch_candidates
        )
        offset = len(sobol_rows)
        for local_index in range(sobol_count):
            row: dict[str, object] = {
                "candidate": offset + local_index,
                "objective": float(batch_objectives[local_index].cpu()),
                "invalid": bool(batch_invalid[local_index].cpu()),
            }
            row.update(
                {
                    name: float(batch_candidates[local_index, parameter_index].cpu())
                    for parameter_index, name in enumerate(UAV_PARAMETER_NAMES)
                }
            )
            sobol_rows.append(row)
        candidate_batches.append(batch_candidates)
        objective_batches.append(batch_objectives)
        invalid_batches.append(batch_invalid)
        write_csv(output / "uav_sobol.csv", sobol_rows)
        if sum(not bool(row["invalid"]) for row in sobol_rows) >= refinements:
            break
        if time.perf_counter() >= deadline:
            raise TimeoutError(
                "Milestone 3C deadline reached while finding finite Sobol starts."
            )
    candidates = torch.cat(candidate_batches)
    objectives = torch.cat(objective_batches)
    invalid = torch.cat(invalid_batches)
    finite_order = torch.argsort(objectives)
    initial_indices = [
        int(index.cpu())
        for index in finite_order
        if not bool(invalid[index].cpu())
    ][:refinements]
    if len(initial_indices) != refinements:
        raise RuntimeError(
            "Fewer than two finite UAV Sobol candidates were found after "
            f"{len(sobol_rows)} deterministic candidates."
        )
    trace: list[dict[str, object]] = []
    best_objective = math.inf
    best_values: torch.Tensor | None = None
    training_batch = prepare_uav_training_batch(episodes)
    for refinement, candidate_index in enumerate(initial_indices):
        fraction = torch.clamp(
            (candidates[candidate_index] - lower) / (upper - lower),
            1.0e-6,
            1.0 - 1.0e-6,
        )
        unconstrained = torch.log(fraction / (1.0 - fraction)).detach().clone()
        unconstrained.requires_grad_(True)
        optimizer = torch.optim.Adam([unconstrained], lr=0.01)
        refinement_best = math.inf
        stale_updates = 0
        for update in range(1, adam_updates + 1):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Milestone 3C 90-minute deadline reached during UAV fit.")
            optimizer.zero_grad(set_to_none=True)
            values = lower + torch.sigmoid(unconstrained) * (upper - lower)
            objective, components = differentiable_batched_uav_training_step(
                training_batch, simulator, settings, config, values
            )
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_([unconstrained], 10.0).detach().cpu()
            )
            optimizer.step()
            current = values.detach().clone()
            row: dict[str, object] = {
                "refinement": refinement,
                "update": update,
                "objective": objective,
                "position_component": components["position"],
                "orientation_component": components["orientation"],
                "gradient_norm_before_clip": gradient_norm,
            }
            row.update(
                {
                    name: float(current[index].cpu())
                    for index, name in enumerate(UAV_PARAMETER_NAMES)
                }
            )
            trace.append(row)
            if math.isfinite(objective) and objective < best_objective:
                best_objective = objective
                best_values = current
            meaningful = (not math.isfinite(refinement_best)) or (
                objective
                < refinement_best - 1.0e-6 * max(1.0, abs(refinement_best))
            )
            if meaningful:
                refinement_best = objective
                stale_updates = 0
            else:
                stale_updates += 1
            if update == 1 or update % 5 == 0:
                write_csv(output / "uav_fit_trace.csv", trace)
                atomic_json(
                    output / "uav_fit_progress.json",
                    {
                        "refinement": refinement,
                        "update": update,
                        "best_objective": best_objective,
                        "elapsed_s": time.perf_counter() - started,
                    },
                )
                print(
                    f"[Milestone 3C] UAV refinement {refinement + 1}/{refinements}, "
                    f"update {update}/{adam_updates}, objective {objective:.6g}",
                    flush=True,
                )
            if stale_updates >= 15:
                break
    if best_values is None:
        raise RuntimeError("UAV Adam refinement produced no finite objective.")
    write_csv(output / "uav_fit_trace.csv", trace)
    result = {
        "schema": "decomposed_uav_fit_v1",
        "method": UAV_METHOD,
        "initial_sobol_candidates": sobol_count,
        "sobol_candidates": len(sobol_rows),
        "sobol_extension_reason": (
            None
            if len(sobol_rows) == sobol_count
            else "initial 64-point batch contained fewer than two finite full-episode candidates"
        ),
        "refinements": refinements,
        "adam_updates_per_refinement": adam_updates,
        "early_stopping": {
            "patience_updates": 15,
            "relative_minimum_improvement": 1.0e-6,
            "selection_uses_validation": False,
        },
        "learning_rate": 0.01,
        "gradient_clip": 10.0,
        "seed": seed,
        "best_training_objective": best_objective,
        "runtime_s": time.perf_counter() - started,
        "fitted_parameters": {
            name: float(best_values[index].cpu())
            for index, name in enumerate(UAV_PARAMETER_NAMES)
        },
        "bound_proximity": {
            name: min(
                (float(best_values[index].cpu()) - config.bounds[name][0])
                / (config.bounds[name][1] - config.bounds[name][0]),
                (config.bounds[name][1] - float(best_values[index].cpu()))
                / (config.bounds[name][1] - config.bounds[name][0]),
            )
            for index, name in enumerate(UAV_PARAMETER_NAMES)
        },
    }
    atomic_json(output / "fitted_uav_parameters.json", result)
    return best_values, result


def _simulate_uav_episode(
    episode: PreparedUAVEpisode,
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    values: torch.Tensor,
) -> dict[str, torch.Tensor]:
    parameters = _parameters(settings, values)
    state = episode.initial_state
    position = [state.position_m[0]]
    velocity = [state.velocity_m_s[0]]
    orientation = [state.orientation_xyzw[0]]
    angular_velocity = [state.angular_velocity_world_rad_s[0]]
    residuals: list[torch.Tensor] = []
    with torch.no_grad():
        for frame in range(episode.step_count):
            state = simulator.uav_model.step(
                state, _command(episode, frame, 1), settings.dt_s, parameters.uav
            )
            if not bool(_finite_uav(state).all()):
                raise FloatingPointError(
                    f"Non-finite UAV validation state in {episode.view.view_id}."
                )
            position.append(state.position_m[0])
            velocity.append(state.velocity_m_s[0])
            orientation.append(state.orientation_xyzw[0])
            angular_velocity.append(state.angular_velocity_world_rad_s[0])
            if state.residual_acceleration_m_s2 is not None:
                residuals.append(state.residual_acceleration_m_s2[0])
    return {
        "position_m": torch.stack(position),
        "velocity_m_s": torch.stack(velocity),
        "orientation_xyzw": torch.stack(orientation),
        "angular_velocity_world_rad_s": torch.stack(angular_velocity),
        "residual_acceleration_m_s2": (
            torch.stack(residuals)
            if residuals
            else torch.empty((0, 3), dtype=simulator.dtype, device=simulator.device)
        ),
    }


def evaluate_uav_model(
    episodes: Sequence[PreparedUAVEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    values: torch.Tensor,
    output: Path,
    label: str,
    lead_times_s: Sequence[float],
) -> dict[str, object]:
    prediction_root = output / "prediction_arrays" / label
    prediction_root.mkdir(parents=True, exist_ok=True)
    take_accumulators: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    residual_vectors: dict[str, list[torch.Tensor]] = {
        "training": [],
        "validation": [],
    }
    for episode in episodes:
        try:
            prediction = _simulate_uav_episode(episode, simulator, settings, values)
        except FloatingPointError as error:
            failures.append(
                {
                    "take_id": episode.view.take.take_id,
                    "role": episode.view.take.role,
                    "view_id": episode.view.view_id,
                    "reason": "non_finite_open_loop_uav_state",
                    "detail": str(error),
                }
            )
            continue
        valid = episode.valid
        position_difference = prediction["position_m"] - episode.measured_position_m
        orientation_error = quaternion_geodesic_rad(
            prediction["orientation_xyzw"], episode.measured_orientation_xyzw
        )
        predicted_euler = _euler_xyz_from_quaternion(prediction["orientation_xyzw"])
        measured_euler = _euler_xyz_from_quaternion(episode.measured_orientation_xyzw)
        euler_difference = _wrap_angle(predicted_euler - measured_euler)
        item = take_accumulators.setdefault(
            episode.view.take.take_id,
            {
                "role": episode.view.take.role,
                "position_sum": 0.0,
                "orientation_sum": 0.0,
                "axis_position_sum": np.zeros(3),
                "axis_euler_sum": np.zeros(3),
                "count": 0,
                "terminal_position2": [],
                "terminal_orientation2": [],
                "lead": {float(lead): [] for lead in lead_times_s},
                "episodes": [],
            },
        )
        selected_position = position_difference[valid]
        selected_orientation = orientation_error[valid]
        selected_euler = euler_difference[valid]
        count = int(valid.sum().cpu())
        item["position_sum"] += float(torch.sum(selected_position.square()).cpu())
        item["orientation_sum"] += float(torch.sum(selected_orientation.square()).cpu())
        item["axis_position_sum"] += torch.sum(
            selected_position.square(), dim=0
        ).cpu().numpy()
        item["axis_euler_sum"] += torch.sum(selected_euler.square(), dim=0).cpu().numpy()
        item["count"] += count
        terminal = int(torch.nonzero(valid, as_tuple=False)[-1, 0].cpu())
        item["terminal_position2"].append(
            float(torch.sum(position_difference[terminal].square()).cpu())
        )
        item["terminal_orientation2"].append(
            float(orientation_error[terminal].square().cpu())
        )
        lead_payload: dict[str, object] = {}
        for lead in lead_times_s:
            step = int(round(float(lead) / settings.dt_s))
            if step <= episode.step_count and bool(valid[step]):
                p2 = float(torch.sum(position_difference[step].square()).cpu())
                r2 = float(orientation_error[step].square().cpu())
                item["lead"][float(lead)].append((p2, r2))
                lead_payload[str(lead)] = {
                    "position_error_m": math.sqrt(p2),
                    "orientation_error_deg": math.degrees(math.sqrt(r2)),
                }
        item["episodes"].append(
            {
                "view_id": episode.view.view_id,
                "parent_episode_id": episode.view.parent.episode_id,
                "kind": episode.view.kind,
                "start_index": episode.view.start_index,
                "duration_s": episode.view.duration_s,
                "terminal_position_error_m": math.sqrt(item["terminal_position2"][-1]),
                "terminal_orientation_error_deg": math.degrees(
                    math.sqrt(item["terminal_orientation2"][-1])
                ),
                "lead_time": lead_payload,
            }
        )
        residual = prediction["residual_acceleration_m_s2"]
        if residual.numel():
            residual_vectors[episode.view.take.role].append(residual)
        np.savez_compressed(
            prediction_root / f"{episode.view.view_id}.npz",
            time_s=(episode.time_s - episode.time_s[0]).cpu().numpy(),
            predicted_position_m=prediction["position_m"].cpu().numpy(),
            measured_position_m=episode.measured_position_m.cpu().numpy(),
            predicted_orientation_xyzw=prediction["orientation_xyzw"].cpu().numpy(),
            measured_orientation_xyzw=episode.measured_orientation_xyzw.cpu().numpy(),
            valid=valid.cpu().numpy(),
            position_error_m=torch.linalg.vector_norm(position_difference, dim=1).cpu().numpy(),
            orientation_error_deg=torch.rad2deg(orientation_error).cpu().numpy(),
            residual_acceleration_m_s2=residual.cpu().numpy(),
        )

    per_take: dict[str, object] = {}
    for take_id, raw in take_accumulators.items():
        count = max(int(raw["count"]), 1)
        lead = {}
        for lead_s, entries in raw["lead"].items():
            if entries:
                lead[str(lead_s)] = {
                    "episode_count": len(entries),
                    "position_rmse_m": math.sqrt(
                        sum(value[0] for value in entries) / len(entries)
                    ),
                    "orientation_rmse_deg": math.degrees(
                        math.sqrt(sum(value[1] for value in entries) / len(entries))
                    ),
                }
        axis_position = np.sqrt(np.asarray(raw["axis_position_sum"]) / count)
        axis_euler = np.degrees(np.sqrt(np.asarray(raw["axis_euler_sum"]) / count))
        per_take[take_id] = {
            "role": raw["role"],
            "episode_count": len(raw["episodes"]),
            "observation_count": count,
            "position_rmse_m": math.sqrt(float(raw["position_sum"]) / count),
            "orientation_rmse_deg": math.degrees(
                math.sqrt(float(raw["orientation_sum"]) / count)
            ),
            "position_axis_rmse_m": dict(zip(("x", "y", "z"), axis_position.tolist())),
            "roll_pitch_yaw_rmse_deg": dict(
                zip(("roll", "pitch", "yaw"), axis_euler.tolist())
            ),
            "terminal_position_rmse_m": math.sqrt(
                sum(raw["terminal_position2"]) / len(raw["terminal_position2"])
            ),
            "terminal_orientation_rmse_deg": math.degrees(
                math.sqrt(
                    sum(raw["terminal_orientation2"])
                    / len(raw["terminal_orientation2"])
                )
            ),
            "lead_time": lead,
            "episodes": raw["episodes"],
        }

    def aggregate(role: str) -> dict[str, object]:
        selected = [value for value in per_take.values() if value["role"] == role]
        attempted_takes = {
            episode.view.take.take_id
            for episode in episodes
            if episode.view.take.role == role
        }
        failed = [item for item in failures if item["role"] == role]
        if not selected:
            return {
                "take_count": 0,
                "attempted_take_count": len(attempted_takes),
                "failed_episode_count": len(failed),
                "complete": False,
            }
        keys = ("position_rmse_m", "orientation_rmse_deg", "terminal_position_rmse_m", "terminal_orientation_rmse_deg")
        result: dict[str, object] = {
            "take_count": len(selected),
            "attempted_take_count": len(attempted_takes),
            "episode_count": sum(int(value["episode_count"]) for value in selected),
            "failed_episode_count": len(failed),
            "complete": len(failed) == 0 and len(selected) == len(attempted_takes),
        }
        for key in keys:
            result[key] = math.sqrt(
                sum(float(value[key]) ** 2 for value in selected) / len(selected)
            )
        result["lead_time"] = {}
        for lead in lead_times_s:
            lead_key = str(float(lead))
            available = [value["lead_time"][lead_key] for value in selected if lead_key in value["lead_time"]]
            if available:
                result["lead_time"][lead_key] = {
                    "take_count": len(available),
                    "position_rmse_m": math.sqrt(
                        sum(float(value["position_rmse_m"]) ** 2 for value in available)
                        / len(available)
                    ),
                    "orientation_rmse_deg": math.sqrt(
                        sum(float(value["orientation_rmse_deg"]) ** 2 for value in available)
                        / len(available)
                    ),
                }
        return result

    residual_statistics: dict[str, object] | None = None
    if any(residual_vectors.values()):
        residual_statistics = {}
        for role, vectors in residual_vectors.items():
            if not vectors:
                continue
            residual = torch.cat(vectors)
            norms = torch.linalg.vector_norm(residual, dim=1)
            differences = [value[1:] - value[:-1] for value in vectors if len(value) > 1]
            delta = (
                torch.cat(differences)
                if differences
                else torch.empty((0, 3), dtype=residual.dtype, device=residual.device)
            )
            residual_statistics[role] = {
                "rms_mps2": float(
                    torch.sqrt(torch.mean(torch.sum(residual.square(), dim=1))).cpu()
                ),
                "axis_rms_mps2": dict(
                    zip(
                        ("x", "y", "z"),
                        torch.sqrt(torch.mean(residual.square(), dim=0)).cpu().tolist(),
                    )
                ),
                "p95_norm_mps2": float(torch.quantile(norms, 0.95).cpu()),
                "maximum_norm_mps2": float(norms.max().cpu()),
                "smoothness_rms_delta_mps2": (
                    float(
                        torch.sqrt(torch.mean(torch.sum(delta.square(), dim=1))).cpu()
                    )
                    if delta.numel()
                    else 0.0
                ),
            }
    result = {
        "schema": "decomposed_uav_metrics_v1",
        "label": label,
        "training": aggregate("training"),
        "validation": aggregate("validation"),
        "per_take": per_take,
        "failed_episodes": failures,
        "residual_statistics": residual_statistics,
        "lead_time_semantics": "elapsed time from the view's single causal initialization",
    }
    atomic_json(output / f"{label}.json", result)
    return result


@dataclass(frozen=True, slots=True)
class PreparedCableEpisode:
    view: EpisodeView
    initial_cable: DderState
    measured_boundary_positions_m: torch.Tensor
    measured_markers_m: torch.Tensor
    marker_valid: torch.Tensor
    time_s: torch.Tensor
    initialization_rmse_m: float

    @property
    def step_count(self) -> int:
        return self.view.step_count


def prepare_measured_boundary_episodes(
    views: Sequence[EpisodeView],
    config: FitConfiguration,
    settings: SimulatorSettings,
    simulator: CoupledSimulator,
    *,
    audit: list[dict[str, object]] | None = None,
) -> tuple[PreparedCableEpisode, ...]:
    prepared: list[PreparedCableEpisode] = []
    for view in views:
        arrays = view.take.arrays
        boundary_slice = slice(view.start_index, view.end_index + 1)
        raw_valid = arrays["uav_valid"][boundary_slice].astype(bool)
        finite = (
            np.isfinite(arrays["uav_position_m"][boundary_slice]).all(axis=1)
            & np.isfinite(arrays["uav_orientation_xyzw"][boundary_slice]).all(axis=1)
        )
        if not bool(np.all(raw_valid & finite)):
            raise ValueError(
                f"{view.view_id} has an invalid measured UAV boundary frame; "
                "the current implementation refuses to fabricate or bridge it."
            )
        position = torch.as_tensor(
            arrays["uav_position_m"][boundary_slice],
            dtype=simulator.dtype,
            device=simulator.device,
        )
        orientation = torch.as_tensor(
            arrays["uav_orientation_xyzw"][boundary_slice],
            dtype=simulator.dtype,
            device=simulator.device,
        )
        measured_uav = UAVState(
            position,
            torch.zeros_like(position),
            orientation,
            torch.zeros_like(position),
        )
        boundary = simulator.root_boundary.evaluate(
            measured_uav, settings.cable_configuration.rest_lengths_m[0]
        )
        try:
            initialized = initialize_window(
                view.take,
                view.initialization_window(config),
                config,
                settings,
                simulator,
            )
        except ValueError as error:
            if "initialization_rmse=" not in str(error):
                raise
            if audit is not None:
                audit.append(
                    {
                        "take_id": view.take.take_id,
                        "view_id": view.view_id,
                        "accepted": False,
                        "reason": "cable_initialization_rmse_gate",
                        "detail": str(error),
                    }
                )
            continue
        offset = view.start_offset
        marker_valid = torch.as_tensor(
            view.parent.cable_observation_valid[
                offset : offset + view.step_count + 1
            ],
            dtype=torch.bool,
            device=simulator.device,
        )
        prepared.append(
            PreparedCableEpisode(
                view,
                initialized.state.cable,
                boundary.prescribed_positions_m,
                torch.as_tensor(
                    arrays["cable_marker_positions_m"][boundary_slice],
                    dtype=simulator.dtype,
                    device=simulator.device,
                ),
                marker_valid,
                torch.as_tensor(
                    arrays["time_s"][boundary_slice],
                    dtype=simulator.dtype,
                    device=simulator.device,
                ),
                initialized.marker_initialization_rmse_m,
            )
        )
        if audit is not None:
            audit.append(
                {
                    "take_id": view.take.take_id,
                    "view_id": view.view_id,
                    "accepted": True,
                    "initialization_rmse_m": initialized.marker_initialization_rmse_m,
                }
            )
    return tuple(prepared)


def _repeat_cable(state: DderState, batch: int) -> DderState:
    return DderState(
        state.positions_m.expand(batch, -1, -1).clone(),
        state.velocities_m_s.expand(batch, -1, -1).clone(),
    )


def _finite_cable(state: DderState) -> torch.Tensor:
    return (
        torch.isfinite(state.positions_m).reshape(state.positions_m.shape[0], -1).all(dim=1)
        & torch.isfinite(state.velocities_m_s).reshape(state.velocities_m_s.shape[0], -1).all(dim=1)
        & (state.positions_m.abs().amax(dim=(1, 2)) < 1.0e3)
        & (state.velocities_m_s.abs().amax(dim=(1, 2)) < 1.0e4)
    )


def cable_population_objective(
    episodes: Sequence[PreparedCableEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    population: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    population = torch.as_tensor(population, dtype=simulator.dtype, device=simulator.device)
    if population.ndim != 2 or population.shape[1] != 2:
        raise ValueError("Cable population must have shape Px2 [EI,Cb].")
    batch = int(population.shape[0])
    marker_nodes = torch.as_tensor(
        settings.cable_configuration.marker_node_indices[1:],
        dtype=torch.long,
        device=simulator.device,
    )
    invalid = torch.zeros(batch, dtype=torch.bool, device=simulator.device)
    take_sums: dict[str, dict[str, torch.Tensor]] = {}
    dt = torch.full((batch,), settings.dt_s, dtype=simulator.dtype, device=simulator.device)
    with torch.no_grad():
        for episode in episodes:
            state = _repeat_cable(episode.initial_cable, batch)
            fallback = state
            constants = simulator.cable_model.runtime_constants(state.positions_m)
            constants = replace(
                constants,
                bending_stiffness_n_m2=population[:, 0],
                bending_damping_n_m2_s=population[:, 1],
            )
            values = take_sums.setdefault(
                episode.view.take.take_id,
                {
                    "sum": torch.zeros(batch, dtype=simulator.dtype, device=simulator.device),
                    "count": torch.zeros((), dtype=simulator.dtype, device=simulator.device),
                },
            )
            for frame in range(episode.step_count):
                boundary = episode.measured_boundary_positions_m[frame + 1].unsqueeze(0).expand(batch, -1, -1)
                state = simulator.cable_model.step_runtime(
                    state,
                    boundary,
                    dt,
                    constants,
                    iterative_damping=True,
                    damping_backend=VALIDATED_OPTIMIZED_DAMPING_BACKEND,
                    pinned_endpoints=simulator.root_boundary.pinned_endpoints,
                    create_graph=False,
                    functional_force_autograd=simulator.functional_force_autograd,
                )
                newly_invalid = ~_finite_cable(state)
                invalid = invalid | newly_invalid
                mask = invalid[:, None, None]
                state = DderState(
                    torch.where(mask, fallback.positions_m, state.positions_m),
                    torch.where(mask, fallback.velocities_m_s, state.velocities_m_s),
                )
                valid = episode.marker_valid[frame + 1]
                if bool(valid.any()):
                    predicted = state.positions_m.index_select(1, marker_nodes)
                    measured = episode.measured_markers_m[frame + 1].unsqueeze(0)
                    # Missing marker entries are represented by NaNs.  Mask the
                    # difference before evaluating the robust distance: applying
                    # a numeric mask after the loss would leave NaN * 0 == NaN
                    # and contaminate the complete candidate population.
                    difference = torch.where(
                        valid[None, :, None],
                        predicted - measured,
                        torch.zeros_like(predicted),
                    )
                    distance = torch.linalg.vector_norm(difference, dim=2)
                    robust = pseudo_huber_distance(distance, config.robust_cable_scale_m)
                    sample = torch.sum(robust, dim=1)
                    values["sum"] += torch.where(
                        invalid, torch.zeros_like(sample), sample
                    )
                    values["count"] += valid.sum()
    per_take = {
        take_id: values["sum"] / torch.clamp(values["count"], min=1.0)
        for take_id, values in take_sums.items()
    }
    objective = torch.mean(torch.stack(list(per_take.values())), dim=0)
    invalid = invalid | ~torch.isfinite(objective)
    objective = torch.where(
        invalid, torch.full_like(objective, INVALID_OBJECTIVE), objective
    )
    return objective, invalid, per_take


def _log_grid(bounds: tuple[float, float], count: int = 8) -> np.ndarray:
    return np.linspace(math.log10(bounds[0]), math.log10(bounds[1]), count)


def fit_cable_grid(
    episodes: Sequence[PreparedCableEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    output: Path,
    *,
    deadline: float,
) -> tuple[tuple[float, float], dict[str, object]]:
    ei_log = _log_grid(config.bounds["EI"])
    cb_log = _log_grid(config.bounds["Cb"])
    passes: list[dict[str, object]] = []
    started = time.perf_counter()
    best_ei = best_cb = math.nan
    best_loss = math.inf
    for pass_index in range(1, 4):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Milestone 3C 90-minute deadline reached during cable fit.")
        mesh_ei, mesh_cb = np.meshgrid(ei_log, cb_log, indexing="ij")
        physical = np.column_stack((10.0 ** mesh_ei.ravel(), 10.0 ** mesh_cb.ravel()))
        population = torch.as_tensor(
            physical, dtype=simulator.dtype, device=simulator.device
        )
        pass_started = time.perf_counter()
        objective, invalid, per_take = cable_population_objective(
            episodes, simulator, settings, config, population
        )
        elapsed = time.perf_counter() - pass_started
        objective_np = objective.cpu().numpy()
        best_index = int(np.argmin(objective_np))
        if bool(invalid[best_index].cpu()):
            raise RuntimeError(f"All Pass-{pass_index} EI/Cb candidates are invalid.")
        best_ei, best_cb = physical[best_index]
        best_loss = float(objective_np[best_index])
        best_i, best_j = np.unravel_index(best_index, (8, 8))
        rows = []
        for index, (ei, cb) in enumerate(physical):
            row: dict[str, object] = {
                "candidate": index,
                "log10_EI": math.log10(ei),
                "log10_Cb": math.log10(cb),
                "EI": ei,
                "Cb": cb,
                "objective": float(objective_np[index]),
                "invalid": bool(invalid[index].cpu()),
            }
            row.update(
                {
                    f"{take_id}_loss": float(values[index].cpu())
                    for take_id, values in per_take.items()
                }
            )
            rows.append(row)
        write_csv(output / f"cable_grid_pass{pass_index}.csv", rows)
        passes.append(
            {
                "pass": pass_index,
                "runtime_s": elapsed,
                "best_EI": float(best_ei),
                "best_Cb": float(best_cb),
                "best_objective": best_loss,
                "invalid_candidates": int(invalid.sum().cpu()),
                "log10_EI_range": [float(ei_log[0]), float(ei_log[-1])],
                "log10_Cb_range": [float(cb_log[0]), float(cb_log[-1])],
            }
        )
        if pass_index < 3:
            ei_log = np.linspace(
                ei_log[max(0, best_i - 1)], ei_log[min(7, best_i + 1)], 8
            )
            cb_log = np.linspace(
                cb_log[max(0, best_j - 1)], cb_log[min(7, best_j + 1)], 8
            )
        atomic_json(output / "cable_fit_progress.json", {"passes": passes})
        print(
            f"[Milestone 3C] cable grid pass {pass_index}/3: "
            f"EI={best_ei:.9g}, Cb={best_cb:.9g}, objective={best_loss:.9g}, "
            f"runtime={elapsed:.1f} s",
            flush=True,
        )
    result = {
        "schema": "decomposed_measured_boundary_cable_fit_v1",
        "method": CABLE_METHOD,
        "fitted_parameters": {"EI": float(best_ei), "Cb": float(best_cb)},
        "best_training_objective": best_loss,
        "passes": passes,
        "runtime_s": time.perf_counter() - started,
        "grid_refinement": "neighboring log-grid bracket, clamped to original bounds",
        "measured_boundary": True,
        "cable_measurement_reset_after_initialization": False,
    }
    atomic_json(output / "fitted_cable_parameters.json", result)
    return (float(best_ei), float(best_cb)), result


def _profile_grid_axis(
    objective: np.ndarray, axis_values: np.ndarray, *, axis: int
) -> tuple[list[float], list[float]]:
    profile = np.min(objective, axis=axis)
    return [float(value) for value in axis_values], [float(value) for value in profile]


def _boundary_improvement(profile: np.ndarray, index: int) -> float:
    adjacent = 1 if index == 0 else len(profile) - 2
    denominator = max(abs(float(profile[adjacent])), 1.0e-30)
    return max(0.0, (float(profile[adjacent]) - float(profile[index])) / denominator)


def _next_log_axis(
    values: np.ndarray,
    profile: np.ndarray,
    index: int,
    state: dict[str, bool],
) -> tuple[np.ndarray, str | None]:
    """Apply the frozen one-refinement/one-extension decision to one axis."""

    if 0 < index < len(values) - 1:
        if state["refined"]:
            return values, None
        state["refined"] = True
        return np.linspace(values[index - 1], values[index + 1], 8), "local_refinement"
    improvement = _boundary_improvement(profile, index)
    if improvement <= 0.01 or state["extended"]:
        return values, None
    state["extended"] = True
    step = float(values[1] - values[0])
    if index == 0:
        return np.linspace(values[0] - 6.0 * step, values[1], 8), "lower_extension"
    return np.linspace(values[-2], values[-1] + 6.0 * step, 8), "upper_extension"


def _profile_identifiability(
    rows: Sequence[dict[str, object]], parameter: str
) -> dict[str, object]:
    other = "Cb" if parameter == "EI" else "EI"
    grouped: dict[float, float] = {}
    for row in rows:
        value = float(row[parameter])
        grouped[value] = min(grouped.get(value, math.inf), float(row["objective"]))
    values = sorted(grouped)
    profile = [grouped[value] for value in values]
    minimum_index = int(np.argmin(profile))
    minimum = profile[minimum_index]
    neighbors: list[dict[str, float]] = []
    for index in (minimum_index - 1, minimum_index + 1):
        if 0 <= index < len(values):
            rise = (profile[index] - minimum) / max(abs(minimum), 1.0e-30)
            neighbors.append(
                {
                    parameter: values[index],
                    "profile_objective": profile[index],
                    "relative_rise_from_minimum": rise,
                }
            )
    boundary = minimum_index in (0, len(values) - 1)
    minimum_adjacent_rise = min(
        (item["relative_rise_from_minimum"] for item in neighbors), default=0.0
    )
    if boundary and minimum_adjacent_rise > 0.01:
        status = "boundary_seeking"
    elif minimum_adjacent_rise <= 0.01:
        status = "weak"
    else:
        status = "contained"
    return {
        "parameter": parameter,
        "profiled_over": other,
        "status": status,
        "best_value": values[minimum_index],
        "best_profile_objective": minimum,
        "best_is_evaluated_boundary": boundary,
        "neighboring_profile_values": neighbors,
        "minimum_adjacent_relative_rise": minimum_adjacent_rise,
        "full_profiled_span_relative_to_minimum": (
            max(profile) - minimum
        )
        / max(abs(minimum), 1.0e-30),
        "evaluated_values": values,
        "profile_objectives": profile,
        "one_percent_boundary_extension_rule": True,
    }


def fit_remeasured_cable_grid(
    episodes: Sequence[PreparedCableEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    output: Path,
    *,
    deadline: float,
) -> tuple[tuple[float, float], dict[str, object]]:
    """Run the frozen 3C.4 adaptive 8x8 measured-boundary cable search."""

    ei_log = _log_grid((2.0e-4, 3.0e-3))
    cb_log = _log_grid((1.0e-5, 1.5e-3))
    axis_state = {
        "EI": {"refined": False, "extended": False},
        "Cb": {"refined": False, "extended": False},
    }
    passes: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    best_row: dict[str, object] | None = None
    started = time.perf_counter()
    for pass_index in range(1, 4):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Milestone 3C.4 20-minute hard stop reached during EI/Cb fitting.")
        mesh_ei, mesh_cb = np.meshgrid(ei_log, cb_log, indexing="ij")
        physical = np.column_stack((10.0**mesh_ei.ravel(), 10.0**mesh_cb.ravel()))
        population = torch.as_tensor(
            physical, dtype=simulator.dtype, device=simulator.device
        )
        pass_started = time.perf_counter()
        objective, invalid, per_take = cable_population_objective(
            episodes, simulator, settings, config, population
        )
        if simulator.device.type == "cuda":
            torch.cuda.synchronize(simulator.device)
        elapsed = time.perf_counter() - pass_started
        objective_np = objective.detach().cpu().numpy().reshape(8, 8)
        invalid_np = invalid.detach().cpu().numpy().reshape(8, 8)
        best_flat = int(np.argmin(objective_np))
        best_i, best_j = np.unravel_index(best_flat, (8, 8))
        if bool(invalid_np[best_i, best_j]):
            raise RuntimeError(f"All Pass-{pass_index} EI/Cb candidates are invalid.")
        rows: list[dict[str, object]] = []
        for flat_index, (ei, cb) in enumerate(physical):
            i, j = np.unravel_index(flat_index, (8, 8))
            row: dict[str, object] = {
                "pass": pass_index,
                "candidate": flat_index,
                "log10_EI": math.log10(float(ei)),
                "log10_Cb": math.log10(float(cb)),
                "EI": float(ei),
                "Cb": float(cb),
                "objective": float(objective_np[i, j]),
                "invalid": bool(invalid_np[i, j]),
            }
            row.update(
                {
                    f"{take_id}_loss": float(values[flat_index].detach().cpu())
                    for take_id, values in per_take.items()
                }
            )
            rows.append(row)
        all_rows.extend(rows)
        candidate = rows[best_flat]
        if best_row is None or float(candidate["objective"]) < float(best_row["objective"]):
            best_row = dict(candidate)
        ei_profile = np.min(objective_np, axis=1)
        cb_profile = np.min(objective_np, axis=0)
        pass_result: dict[str, object] = {
            "pass": pass_index,
            "runtime_s": elapsed,
            "best_EI": float(physical[best_flat, 0]),
            "best_Cb": float(physical[best_flat, 1]),
            "best_objective": float(objective_np[best_i, best_j]),
            "best_ei_index": int(best_i),
            "best_cb_index": int(best_j),
            "invalid_candidates": int(np.count_nonzero(invalid_np)),
            "ei_values": [float(10.0**value) for value in ei_log],
            "cb_values": [float(10.0**value) for value in cb_log],
            "ei_profile": [float(value) for value in ei_profile],
            "cb_profile": [float(value) for value in cb_profile],
            "decision": {},
        }
        write_csv(output / f"ei_cb_grid_pass{pass_index}.csv", rows)
        atomic_json(
            output / f"ei_cb_grid_pass{pass_index}.json",
            {**pass_result, "candidates": rows},
        )
        passes.append(pass_result)
        print(
            f"[Milestone 3C.4] grid {pass_index}: "
            f"EI={physical[best_flat, 0]:.9g}, Cb={physical[best_flat, 1]:.9g}, "
            f"objective={objective_np[best_i, best_j]:.9g}, runtime={elapsed:.1f}s",
            flush=True,
        )
        if pass_index == 3:
            break
        next_ei, ei_action = _next_log_axis(
            ei_log, ei_profile, best_i, axis_state["EI"]
        )
        next_cb, cb_action = _next_log_axis(
            cb_log, cb_profile, best_j, axis_state["Cb"]
        )
        pass_result["decision"] = {
            "EI": ei_action or "stop",
            "Cb": cb_action or "stop",
            "EI_boundary_improvement": (
                _boundary_improvement(ei_profile, best_i)
                if best_i in (0, 7)
                else None
            ),
            "Cb_boundary_improvement": (
                _boundary_improvement(cb_profile, best_j)
                if best_j in (0, 7)
                else None
            ),
        }
        atomic_json(
            output / f"ei_cb_grid_pass{pass_index}.json",
            {**pass_result, "candidates": rows},
        )
        if ei_action is None and cb_action is None:
            break
        ei_log, cb_log = next_ei, next_cb
        if time.perf_counter() >= deadline:
            raise TimeoutError("Milestone 3C.4 hard stop reached after a completed grid.")
    assert best_row is not None
    identifiability = {
        "EI": _profile_identifiability(all_rows, "EI"),
        "Cb": _profile_identifiability(all_rows, "Cb"),
    }
    result = {
        "schema": "milestone3c4_remeasured_geometry_cable_fit_v1",
        "method": "adaptive_measured_boundary_log_grid_v1",
        "fitted_parameters": {
            "EI": float(best_row["EI"]),
            "Cb": float(best_row["Cb"]),
        },
        "best_training_objective": float(best_row["objective"]),
        "best_pass": int(best_row["pass"]),
        "passes": passes,
        "runtime_s": time.perf_counter() - started,
        "identifiability": identifiability,
        "measured_boundary": True,
        "full_physical_episode_fit": True,
        "cable_measurement_reset_after_initialization": False,
        "maximum_passes": 3,
        "boundary_extension_threshold_relative": 0.01,
    }
    atomic_json(output / "fitted_cable_parameters.json", result)
    atomic_json(output / "cable_parameter_identifiability.json", identifiability)
    _render_remeasured_cable_profiles(output, all_rows, int(best_row["pass"]))
    return (
        (float(best_row["EI"]), float(best_row["Cb"])),
        result,
    )


def _render_remeasured_cable_profiles(
    output: Path, rows: Sequence[dict[str, object]], best_pass: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if int(row["pass"]) == best_pass]
    x = np.asarray([float(row["log10_Cb"]) for row in selected]).reshape(8, 8)
    y = np.asarray([float(row["log10_EI"]) for row in selected]).reshape(8, 8)
    z = np.asarray([float(row["objective"]) for row in selected]).reshape(8, 8)
    figure, axis = plt.subplots(figsize=(6.4, 5.1), constrained_layout=True)
    image = axis.contourf(x, y, np.log10(np.maximum(z, 1.0e-30)), levels=20)
    axis.set_xlabel("log10 Cb [N m² s]")
    axis.set_ylabel("log10 EI [N m²]")
    axis.set_title(f"Remeasured-geometry EI/Cb loss, pass {best_pass}")
    figure.colorbar(image, ax=axis, label="log10 objective")
    figure.savefig(output / "ei_cb_loss_landscape.png", dpi=180)
    plt.close(figure)

    for parameter, filename, units in (
        ("EI", "ei_profile.png", "N m²"),
        ("Cb", "cb_profile.png", "N m² s"),
    ):
        profile = _profile_identifiability(rows, parameter)
        values = np.asarray(profile["evaluated_values"], dtype=np.float64)
        losses = np.asarray(profile["profile_objectives"], dtype=np.float64)
        figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        axis.semilogx(values, losses, marker="o")
        axis.set_xlabel(f"{parameter} [{units}]")
        axis.set_ylabel("Profiled training objective")
        axis.set_title(f"{parameter} profile over evaluated grids")
        axis.grid(True, which="both", alpha=0.25)
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)


def _cable_constants(
    simulator: CoupledSimulator,
    state: DderState,
    EI: float,
    Cb: float,
) -> DderRuntimeConstants:
    constants = simulator.cable_model.runtime_constants(state.positions_m)
    return replace(
        constants,
        bending_stiffness_n_m2=torch.full_like(
            constants.bending_stiffness_n_m2, EI
        ),
        bending_damping_n_m2_s=torch.full_like(
            constants.bending_damping_n_m2_s, Cb
        ),
    )


def evaluate_conditional_cable(
    episodes: Sequence[PreparedCableEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    EI: float,
    Cb: float,
    output: Path,
    label: str,
    lead_times_s: Sequence[float],
) -> dict[str, object]:
    prediction_root = output / "prediction_arrays" / label
    prediction_root.mkdir(parents=True, exist_ok=True)
    marker_nodes = torch.as_tensor(
        settings.cable_configuration.marker_node_indices[1:],
        dtype=torch.long,
        device=simulator.device,
    )
    take_accumulators: dict[str, dict[str, object]] = {}
    for episode in episodes:
        state = episode.initial_cable
        constants = _cable_constants(simulator, state, EI, Cb)
        dt = torch.tensor([settings.dt_s], dtype=simulator.dtype, device=simulator.device)
        predictions = [state.positions_m[0].index_select(0, marker_nodes)]
        with torch.no_grad():
            for frame in range(episode.step_count):
                state = simulator.cable_model.step_runtime(
                    state,
                    episode.measured_boundary_positions_m[frame + 1 : frame + 2],
                    dt,
                    constants,
                    iterative_damping=True,
                    damping_backend=VALIDATED_OPTIMIZED_DAMPING_BACKEND,
                    pinned_endpoints=simulator.root_boundary.pinned_endpoints,
                    create_graph=False,
                    functional_force_autograd=simulator.functional_force_autograd,
                )
                if not bool(_finite_cable(state).all()):
                    raise FloatingPointError(
                        f"Conditional cable validation diverged in {episode.view.view_id}."
                    )
                predictions.append(state.positions_m[0].index_select(0, marker_nodes))
        predicted = torch.stack(predictions)
        difference = predicted - episode.measured_markers_m
        distance2 = torch.sum(difference.square(), dim=2)
        valid = episode.marker_valid
        tip_valid = valid[:, -1]
        item = take_accumulators.setdefault(
            episode.view.take.take_id,
            {
                "role": episode.view.take.role,
                "marker_sum": 0.0,
                "marker_count": 0,
                "tip_sum": 0.0,
                "tip_count": 0,
                "terminal_tip2": [],
                "lead": {float(lead): [] for lead in lead_times_s},
                "lead_marker": {float(lead): [] for lead in lead_times_s},
                "episodes": [],
            },
        )
        item["marker_sum"] += float(distance2[valid].sum().cpu())
        item["marker_count"] += int(valid.sum().cpu())
        item["tip_sum"] += float(distance2[:, -1][tip_valid].sum().cpu())
        item["tip_count"] += int(tip_valid.sum().cpu())
        terminal = int(torch.nonzero(tip_valid, as_tuple=False)[-1, 0].cpu())
        terminal2 = float(distance2[terminal, -1].cpu())
        item["terminal_tip2"].append(terminal2)
        lead_payload = {}
        for lead in lead_times_s:
            step = int(round(float(lead) / settings.dt_s))
            if step <= episode.step_count and bool(tip_valid[step]):
                error2 = float(distance2[step, -1].cpu())
                item["lead"][float(lead)].append(error2)
                marker_valid = valid[step]
                marker_error2 = distance2[step][marker_valid]
                marker_mse = float(torch.mean(marker_error2).cpu())
                item["lead_marker"][float(lead)].append(marker_mse)
                lead_payload[str(lead)] = {
                    "tip_error_m": math.sqrt(error2),
                    "distributed_marker_rmse_m": math.sqrt(marker_mse),
                }
        item["episodes"].append(
            {
                "view_id": episode.view.view_id,
                "duration_s": episode.view.duration_s,
                "initialization_rmse_m": episode.initialization_rmse_m,
                "terminal_tip_error_m": math.sqrt(terminal2),
                "lead_time": lead_payload,
            }
        )
        np.savez_compressed(
            prediction_root / f"{episode.view.view_id}.npz",
            time_s=(episode.time_s - episode.time_s[0]).cpu().numpy(),
            predicted_markers_m=predicted.cpu().numpy(),
            measured_markers_m=episode.measured_markers_m.cpu().numpy(),
            marker_valid=valid.cpu().numpy(),
            marker_error_m=torch.sqrt(distance2).cpu().numpy(),
            measured_boundary_positions_m=episode.measured_boundary_positions_m.cpu().numpy(),
        )

    per_take = {}
    for take_id, raw in take_accumulators.items():
        lead = {
            str(lead): {
                "episode_count": len(entries),
                "tip_rmse_m": math.sqrt(sum(entries) / len(entries)),
                "distributed_marker_rmse_m": math.sqrt(
                    sum(raw["lead_marker"][lead])
                    / len(raw["lead_marker"][lead])
                ),
            }
            for lead, entries in raw["lead"].items()
            if entries
        }
        per_take[take_id] = {
            "role": raw["role"],
            "episode_count": len(raw["episodes"]),
            "marker_observation_count": raw["marker_count"],
            "distributed_marker_rmse_m": math.sqrt(
                raw["marker_sum"] / max(raw["marker_count"], 1)
            ),
            "tip_rmse_m": math.sqrt(raw["tip_sum"] / max(raw["tip_count"], 1)),
            "terminal_tip_rmse_m": math.sqrt(
                sum(raw["terminal_tip2"]) / len(raw["terminal_tip2"])
            ),
            "lead_time": lead,
            "episodes": raw["episodes"],
        }

    def aggregate(role: str) -> dict[str, object]:
        selected = [value for value in per_take.values() if value["role"] == role]
        if not selected:
            return {}
        result = {
            "take_count": len(selected),
            "episode_count": sum(int(value["episode_count"]) for value in selected),
        }
        for key in ("distributed_marker_rmse_m", "tip_rmse_m", "terminal_tip_rmse_m"):
            result[key] = math.sqrt(
                sum(float(value[key]) ** 2 for value in selected) / len(selected)
            )
        result["lead_time"] = {}
        for lead in lead_times_s:
            key = str(float(lead))
            available = [value["lead_time"][key] for value in selected if key in value["lead_time"]]
            if available:
                result["lead_time"][key] = {
                    "take_count": len(available),
                    "tip_rmse_m": math.sqrt(
                        sum(float(value["tip_rmse_m"]) ** 2 for value in available)
                        / len(available)
                    ),
                    "distributed_marker_rmse_m": math.sqrt(
                        sum(
                            float(value["distributed_marker_rmse_m"]) ** 2
                            for value in available
                        )
                        / len(available)
                    ),
                }
        return result

    result = {
        "schema": "decomposed_conditional_cable_metrics_v1",
        "label": label,
        "boundary": "measured Motive UAV pose through production rigid clamp",
        "EI": EI,
        "Cb": Cb,
        "training": aggregate("training"),
        "validation": aggregate("validation"),
        "per_take": per_take,
        "lead_time_semantics": "elapsed time from each episode's single initialization",
    }
    atomic_json(output / f"{label}.json", result)
    return result


def compute_residual_normalization(
    episodes: Sequence[PreparedUAVEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    values: torch.Tensor,
    std_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    features = []
    parameters = _parameters(settings, values)
    with torch.no_grad():
        for episode in episodes:
            state = episode.initial_state
            episode_features = []
            for frame in range(episode.step_count):
                episode_features.append(
                    residual_feature(
                        episode.command_position_m[frame : frame + 1],
                        episode.command_velocity_mps[frame : frame + 1],
                        episode.command_acceleration_mps2[frame : frame + 1],
                        state.position_m,
                        state.velocity_m_s,
                    )[0]
                )
                state = simulator.uav_model.step(
                    state,
                    _command(episode, frame, 1),
                    settings.dt_s,
                    parameters.uav,
                )
            features.append(torch.stack(episode_features))
    stacked = torch.cat(features)
    mean = torch.mean(stacked, dim=0)
    raw_std = torch.std(stacked, dim=0, correction=0)
    std = torch.clamp(raw_std, min=std_floor)
    payload = {
        "schema": "decomposed_residual_normalization_v1",
        "source": "new_fitted_nominal_physics_on_training_causal_suffixes_only",
        "sample_count": int(stacked.shape[0]),
        "feature_names": list(RESIDUAL_FEATURE_NAMES),
        "mean": mean.cpu().tolist(),
        "raw_standard_deviation": raw_std.cpu().tolist(),
        "standard_deviation": std.cpu().tolist(),
        "standard_deviation_floor": std_floor,
        "validation_used": False,
        "protected_test_used": False,
    }
    return mean, std, payload


def fit_residual(
    episodes: Sequence[PreparedUAVEpisode],
    simulator: CoupledSimulator,
    residual: CausalTranslationalResidual,
    settings: SimulatorSettings,
    config: FitConfiguration,
    values: torch.Tensor,
    output: Path,
    *,
    deadline: float,
) -> dict[str, object]:
    optimizer_config = config.uav_residual_ablation["optimizer"]
    configured_updates = int(optimizer_config["updates"])
    updates = min(configured_updates, 150)
    learning_rate = float(optimizer_config["learning_rate"])
    gradient_clip = float(optimizer_config["gradient_clip"])
    evaluation_interval = int(optimizer_config["full_training_evaluation_interval"])
    optimizer = torch.optim.Adam(residual.parameters(), lr=learning_rate)
    trace: list[dict[str, object]] = []
    best_objective = math.inf
    best_state = {name: value.detach().cpu().clone() for name, value in residual.state_dict().items()}
    started = time.perf_counter()
    training_batch = prepare_uav_training_batch(episodes)
    for update in range(1, updates + 1):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Milestone 3C 90-minute deadline reached during residual fit.")
        optimizer.zero_grad(set_to_none=True)
        objective, components = differentiable_batched_uav_training_step(
            training_batch,
            simulator,
            settings,
            config,
            values,
            residual_parameters=residual.parameters(),
        )
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(residual.parameters(), gradient_clip).cpu()
        )
        candidate_state = {
            name: value.detach().cpu().clone()
            for name, value in residual.state_dict().items()
        }
        optimizer.step()
        row = {
            "update": update,
            "objective": objective,
            "position_component": components["position"],
            "orientation_component": components["orientation"],
            "magnitude_component": components["magnitude"],
            "smoothness_component": components["smoothness"],
            "gradient_norm_before_clip": gradient_norm,
        }
        trace.append(row)
        if math.isfinite(objective) and objective < best_objective:
            best_objective = objective
            best_state = candidate_state
        if update == 1 or update % evaluation_interval == 0 or update == updates:
            write_csv(output / "residual_training_trace.csv", trace)
            torch.save(best_state, output / "residual_weights.pt")
            atomic_json(
                output / "residual_training_progress.json",
                {
                    "update": update,
                    "best_training_objective": best_objective,
                    "runtime_s": time.perf_counter() - started,
                    "selection_uses_validation": False,
                },
            )
            print(
                f"[Milestone 3C] residual update {update}/{updates}, "
                f"objective {objective:.6g}",
                flush=True,
            )
    residual.load_state_dict(best_state)
    torch.save(residual.state_dict(), output / "residual_weights.pt")
    result = {
        "schema": "decomposed_full_episode_residual_fit_v1",
        "method": RESIDUAL_METHOD,
        "architecture": "90 -> 32 -> SiLU -> 32 -> SiLU -> 3",
        "parameter_count": residual.parameter_count,
        "updates": updates,
        "configured_update_ceiling": configured_updates,
        "execution_ceiling_reason": (
            None
            if updates == configured_updates
            else "bounded by the milestone 90-minute total wall-clock policy"
        ),
        "learning_rate": learning_rate,
        "gradient_clip": gradient_clip,
        "best_training_objective": best_objective,
        "runtime_s": time.perf_counter() - started,
        "network_state_sha256": sha256_file(output / "residual_weights.pt"),
        "initialized_from_zero_output_layer": True,
        "historical_weights_reused": False,
        "validation_used_for_selection": False,
    }
    atomic_json(output / "residual_fit.json", result)
    return result


def _commands_for_episode(episode: PreparedUAVEpisode) -> FullStateCommandSequence:
    def batch(value: torch.Tensor) -> torch.Tensor:
        return value[:, None]

    return FullStateCommandSequence(
        batch(episode.command_position_m),
        batch(episode.command_velocity_mps),
        batch(episode.command_acceleration_mps2),
        batch(episode.command_orientation_xyzw),
        batch(episode.command_angular_velocity_body_rad_s),
    )


def evaluate_end_to_end(
    episodes: Sequence[PreparedUAVEpisode],
    simulator: CoupledSimulator,
    settings: SimulatorSettings,
    config: FitConfiguration,
    uav_values: torch.Tensor,
    EI: float,
    Cb: float,
    output: Path,
    label: str,
    lead_times_s: Sequence[float],
    *,
    residual_history: bool,
) -> dict[str, object]:
    prediction_root = output / "prediction_arrays" / label
    prediction_root.mkdir(parents=True, exist_ok=True)
    marker_nodes = torch.as_tensor(
        settings.cable_configuration.marker_node_indices[1:],
        dtype=torch.long,
        device=simulator.device,
    )
    parameters = _parameters(settings, uav_values, (EI, Cb))
    uav_take: dict[str, dict[str, float | int]] = {}
    cable_take: dict[str, dict[str, float | int]] = {}
    episode_rows = []
    failures: list[dict[str, object]] = []
    for episode in episodes:
        try:
            initialized = initialize_window(
                episode.view.take,
                episode.view.initialization_window(config),
                config,
                settings,
                simulator,
            )
        except ValueError as error:
            # A PhysicalEpisode remains authoritative even when its cable
            # initialization fails the frozen quality threshold.  Preserve the
            # episode boundary and record the complete episode as ineligible;
            # never rescue it by trimming a favorable sub-window.
            failures.append(
                {
                    "take_id": episode.view.take.take_id,
                    "view_id": episode.view.view_id,
                    "reason": "cable_initialization_ineligible",
                    "detail": str(error),
                }
            )
            continue
        initial_state = initialized.state
        if residual_history:
            history = episode.initial_state.residual_history
            if history is None:
                raise ValueError("Model PR requires a causal residual history.")
            initial_state = SimulatorState(
                initial_state.time_s,
                UAVState(
                    initial_state.uav.position_m,
                    initial_state.uav.velocity_m_s,
                    initial_state.uav.orientation_xyzw,
                    initial_state.uav.angular_velocity_world_rad_s,
                    history,
                    torch.zeros_like(initial_state.uav.position_m),
                ),
                initial_state.cable,
            )
        try:
            with torch.no_grad():
                trajectory = simulator.rollout(
                    initial_state,
                    _commands_for_episode(episode),
                    parameters,
                    create_graph=False,
                )
        except (FloatingPointError, RuntimeError) as error:
            failures.append(
                {
                    "take_id": episode.view.take.take_id,
                    "view_id": episode.view.view_id,
                    "reason": "coupled_rollout_error",
                    "detail": str(error),
                }
            )
            continue
        predicted_uav = trajectory.uav_positions_m[:, 0]
        predicted_orientation = trajectory.uav_orientations_xyzw[:, 0]
        predicted_markers = trajectory.cable_positions_m[:, 0].index_select(1, marker_nodes)
        if not bool(
            torch.isfinite(predicted_uav).all()
            and torch.isfinite(predicted_orientation).all()
            and torch.isfinite(predicted_markers).all()
        ):
            failures.append(
                {
                    "take_id": episode.view.take.take_id,
                    "view_id": episode.view.view_id,
                    "reason": "non_finite_coupled_rollout",
                }
            )
            continue
        uav_valid = episode.valid
        offset = episode.view.start_offset
        cable_valid = torch.as_tensor(
            episode.view.parent.cable_observation_valid[
                offset : offset + episode.step_count + 1
            ],
            dtype=torch.bool,
            device=simulator.device,
        )
        measured_markers = torch.as_tensor(
            episode.view.take.arrays["cable_marker_positions_m"][
                episode.view.start_index : episode.view.end_index + 1
            ],
            dtype=simulator.dtype,
            device=simulator.device,
        )
        uav_difference = predicted_uav - episode.measured_position_m
        orientation_error = quaternion_geodesic_rad(
            predicted_orientation, episode.measured_orientation_xyzw
        )
        cable_distance2 = torch.sum((predicted_markers - measured_markers).square(), dim=2)
        take_id = episode.view.take.take_id
        u = uav_take.setdefault(
            take_id,
            {"position_sum": 0.0, "orientation_sum": 0.0, "count": 0, "terminal_position2": 0.0, "terminal_orientation2": 0.0, "terminal_count": 0},
        )
        c = cable_take.setdefault(
            take_id,
            {"marker_sum": 0.0, "marker_count": 0, "tip_sum": 0.0, "tip_count": 0, "terminal_tip2": 0.0, "terminal_count": 0},
        )
        u["position_sum"] += float(torch.sum(uav_difference[uav_valid].square()).cpu())
        u["orientation_sum"] += float(torch.sum(orientation_error[uav_valid].square()).cpu())
        u["count"] += int(uav_valid.sum().cpu())
        u_terminal = int(torch.nonzero(uav_valid, as_tuple=False)[-1, 0].cpu())
        episode_terminal_uav_position2 = float(
            torch.sum(uav_difference[u_terminal].square()).cpu()
        )
        episode_terminal_uav_orientation2 = float(
            orientation_error[u_terminal].square().cpu()
        )
        u["terminal_position2"] += episode_terminal_uav_position2
        u["terminal_orientation2"] += episode_terminal_uav_orientation2
        u["terminal_count"] += 1
        c["marker_sum"] += float(cable_distance2[cable_valid].sum().cpu())
        c["marker_count"] += int(cable_valid.sum().cpu())
        tip_valid = cable_valid[:, -1]
        c["tip_sum"] += float(cable_distance2[:, -1][tip_valid].sum().cpu())
        c["tip_count"] += int(tip_valid.sum().cpu())
        c_terminal = int(torch.nonzero(tip_valid, as_tuple=False)[-1, 0].cpu())
        episode_terminal_tip2 = float(cable_distance2[c_terminal, -1].cpu())
        c["terminal_tip2"] += episode_terminal_tip2
        c["terminal_count"] += 1
        lead = {}
        for lead_s in lead_times_s:
            step = int(round(float(lead_s) / settings.dt_s))
            if step <= episode.step_count:
                lead[str(lead_s)] = {
                    "uav_position_error_m": float(torch.linalg.vector_norm(uav_difference[step]).cpu()),
                    "uav_orientation_error_deg": float(torch.rad2deg(orientation_error[step]).cpu()),
                    "distributed_marker_rmse_m": float(
                        torch.sqrt(torch.mean(cable_distance2[step][cable_valid[step]])).cpu()
                    ),
                    "tip_error_m": float(torch.sqrt(cable_distance2[step, -1]).cpu()),
                }
        episode_rows.append(
            {
                "take_id": take_id,
                "view_id": episode.view.view_id,
                "kind": episode.view.kind,
                "duration_s": episode.view.duration_s,
                "terminal_uav_position_error_m": math.sqrt(
                    episode_terminal_uav_position2
                ),
                "terminal_uav_orientation_error_deg": math.degrees(
                    math.sqrt(episode_terminal_uav_orientation2)
                ),
                "terminal_tip_error_m": math.sqrt(episode_terminal_tip2),
                "lead_time": lead,
            }
        )
        np.savez_compressed(
            prediction_root / f"{episode.view.view_id}.npz",
            time_s=(episode.time_s - episode.time_s[0]).cpu().numpy(),
            predicted_uav_position_m=predicted_uav.cpu().numpy(),
            measured_uav_position_m=episode.measured_position_m.cpu().numpy(),
            predicted_uav_orientation_xyzw=predicted_orientation.cpu().numpy(),
            measured_uav_orientation_xyzw=episode.measured_orientation_xyzw.cpu().numpy(),
            predicted_markers_m=predicted_markers.cpu().numpy(),
            measured_markers_m=measured_markers.cpu().numpy(),
            uav_valid=uav_valid.cpu().numpy(),
            cable_valid=cable_valid.cpu().numpy(),
        )

    per_take = {}
    for take_id in uav_take:
        u, c = uav_take[take_id], cable_take[take_id]
        per_take[take_id] = {
            "uav_position_rmse_m": math.sqrt(float(u["position_sum"]) / max(int(u["count"]), 1)),
            "uav_orientation_rmse_deg": math.degrees(math.sqrt(float(u["orientation_sum"]) / max(int(u["count"]), 1))),
            "terminal_uav_position_rmse_m": math.sqrt(float(u["terminal_position2"]) / max(int(u["terminal_count"]), 1)),
            "terminal_uav_orientation_rmse_deg": math.degrees(math.sqrt(float(u["terminal_orientation2"]) / max(int(u["terminal_count"]), 1))),
            "distributed_marker_rmse_m": math.sqrt(float(c["marker_sum"]) / max(int(c["marker_count"]), 1)),
            "tip_rmse_m": math.sqrt(float(c["tip_sum"]) / max(int(c["tip_count"]), 1)),
            "terminal_tip_rmse_m": math.sqrt(float(c["terminal_tip2"]) / max(int(c["terminal_count"]), 1)),
        }
    aggregate: dict[str, object] = {}
    if per_take:
        keys = (
            "uav_position_rmse_m",
            "uav_orientation_rmse_deg",
            "terminal_uav_position_rmse_m",
            "terminal_uav_orientation_rmse_deg",
            "distributed_marker_rmse_m",
            "tip_rmse_m",
            "terminal_tip_rmse_m",
        )
        aggregate = {
            key: math.sqrt(
                sum(float(value[key]) ** 2 for value in per_take.values())
                / len(per_take)
            )
            for key in keys
        }
        lead_aggregate: dict[str, object] = {}
        for lead_s in lead_times_s:
            lead_key = str(float(lead_s))
            rows_by_take: dict[str, list[dict[str, object]]] = {}
            for episode in episode_rows:
                if lead_key not in episode["lead_time"]:
                    continue
                rows_by_take.setdefault(str(episode["take_id"]), []).append(
                    episode["lead_time"][lead_key]
                )
            if rows_by_take:
                metric_names = (
                    "uav_position_error_m",
                    "uav_orientation_error_deg",
                    "tip_error_m",
                    "distributed_marker_rmse_m",
                )
                take_metrics = {
                    take_id: {
                        metric: math.sqrt(
                            sum(float(row[metric]) ** 2 for row in rows) / len(rows)
                        )
                        for metric in metric_names
                    }
                    for take_id, rows in rows_by_take.items()
                }
                lead_aggregate[lead_key] = {
                    "take_count": len(take_metrics),
                    "episode_count": sum(len(rows) for rows in rows_by_take.values()),
                    "uav_position_rmse_m": math.sqrt(
                        sum(
                            float(row["uav_position_error_m"]) ** 2
                            for row in take_metrics.values()
                        )
                        / len(take_metrics)
                    ),
                    "uav_orientation_rmse_deg": math.sqrt(
                        sum(
                            float(row["uav_orientation_error_deg"]) ** 2
                            for row in take_metrics.values()
                        )
                        / len(take_metrics)
                    ),
                    "tip_rmse_m": math.sqrt(
                        sum(
                            float(row["tip_error_m"]) ** 2
                            for row in take_metrics.values()
                        )
                        / len(take_metrics)
                    ),
                    "distributed_marker_rmse_m": math.sqrt(
                        sum(
                            float(row["distributed_marker_rmse_m"]) ** 2
                            for row in take_metrics.values()
                        )
                        / len(take_metrics)
                    ),
                    "per_take": take_metrics,
                }
        aggregate["lead_time"] = lead_aggregate
    result = {
        "schema": "decomposed_end_to_end_metrics_v1",
        "label": label,
        "method": VALIDATION_METHOD,
        "measured_uav_boundary_used": False,
        "measurement_reset_after_initialization": False,
        "residual_history_rule": (
            "PhysicalEpisode.residual_eligible_start_index"
            if residual_history
            else None
        ),
        "aggregate_equal_take_rmse": aggregate,
        "per_take": per_take,
        "episodes": episode_rows,
        "failed_episodes": failures,
        "complete": not failures,
    }
    atomic_json(output / f"{label}.json", result)
    return result


def _source_manifest(root: Path) -> dict[str, object]:
    relative_files = (
        "fitting/decomposed.py",
        "fitting/episodes.py",
        "fitting/initialization.py",
        "fitting/config.py",
        "fitting/default_fit.json",
        "simulator/simulator.py",
        "simulator/parameters.py",
        "simulator/uav/model.py",
        "simulator/uav/residual.py",
        "simulator/coupling/attachment.py",
        "simulator/coupling/root_boundary.py",
        "simulator/cable/dder.py",
        "simulator/cable/cuda_fixed_pcg.py",
        "simulator/production.py",
        "fitting/production_status.py",
        "config/default.json",
        "run_milestone3c.py",
    )
    files = {}
    for relative in relative_files:
        path = root / relative
        if not path.exists():
            continue
        files[relative] = sha256_file(path)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    return {
        "git_commit": commit,
        "working_tree_dirty": bool(status),
        "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "files": files,
        "aggregate_source_sha256": canonical_json_hash(files),
    }


def _render_cable_landscape(output: Path) -> None:
    import csv
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = output / "cable_grid_pass3.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = np.asarray([float(row["log10_Cb"]) for row in rows]).reshape(8, 8)
    y = np.asarray([float(row["log10_EI"]) for row in rows]).reshape(8, 8)
    z = np.asarray([float(row["objective"]) for row in rows]).reshape(8, 8)
    figure, axis = plt.subplots(figsize=(6.4, 5.1), constrained_layout=True)
    image = axis.contourf(x, y, np.log10(np.maximum(z, 1.0e-30)), levels=20)
    axis.set_xlabel("log10 Cb")
    axis.set_ylabel("log10 EI")
    axis.set_title("Measured-boundary cable loss, refinement pass 3")
    figure.colorbar(image, ax=axis, label="log10 objective")
    figure.savefig(output / "cable_loss_landscape.png", dpi=180)
    plt.close(figure)


def _table_parameters(
    uav_fit: dict[str, object], cable_fit: dict[str, object]
) -> str:
    uav = uav_fit["fitted_parameters"]
    cable = cable_fit["fitted_parameters"]
    assert isinstance(uav, dict) and isinstance(cable, dict)
    rows = ["| Parameter | Fitted value | Provenance |", "|---|---:|---|"]
    for name in UAV_PARAMETER_NAMES:
        rows.append(f"| `{name}` | {float(uav[name]):.9g} | command → measured UAV |")
    rows.append(f"| `EI` | {float(cable['EI']):.9g} | measured-boundary DDER |")
    rows.append(f"| `Cb` | {float(cable['Cb']):.9g} | measured-boundary DDER |")
    return "\n".join(rows)


def _fmt_metric(payload: dict[str, object], key: str, scale: float = 1.0) -> str:
    value = payload.get(key)
    return "n/a" if value is None else f"{scale * float(value):.3f}"


def render_report(output: Path, summary: dict[str, object], report: Path) -> None:
    uav_fit = summary["uav_fit"]
    cable_fit = summary["cable_fit"]
    coverage = summary["residual_coverage"]
    uav_physics = summary["uav_physics_metrics"]
    uav_same = summary["uav_same_suffix_physics"]
    uav_residual = summary["uav_same_suffix_residual"]
    conditional = summary["conditional_cable_metrics"]
    end_full = summary["end_to_end_full_physics"]
    end_suffix_p = summary["end_to_end_suffix_physics"]
    end_suffix_pr = summary["end_to_end_suffix_residual"]
    residual_fit = summary["residual_fit"]
    freeze = summary["freeze"]
    validation_u = uav_physics["validation"]
    validation_same_u = uav_same["validation"]
    validation_same_ur = uav_residual["validation"]
    conditional_validation = conditional["validation"]

    def end_rows(payload: dict[str, object]) -> list[dict[str, object]]:
        return list(payload["per_take"].values())

    def end_aggregate(payload: dict[str, object], key: str) -> float:
        rows = end_rows(payload)
        return math.sqrt(sum(float(row[key]) ** 2 for row in rows) / len(rows))

    p_marker = end_aggregate(end_full, "distributed_marker_rmse_m")
    pr_marker = end_aggregate(end_suffix_pr, "distributed_marker_rmse_m")
    same_p_marker = end_aggregate(end_suffix_p, "distributed_marker_rmse_m")
    conditional_marker = float(conditional_validation["distributed_marker_rmse_m"])
    uav_error = float(validation_u["position_rmse_m"])
    if conditional_marker > 0.05 and uav_error > 0.05:
        classification = "BOTH REMAIN LIMITING"
    elif conditional_marker > 0.05:
        classification = "CABLE MODEL REMAINS LIMITING"
    elif uav_error > 0.05:
        classification = "UAV MODEL REMAINS LIMITING"
    else:
        classification = "CURRENT MODEL SUFFICIENT"
    summary["model_classification"] = classification
    text = f"""# Milestone 3C — Decomposed Refit and Validation Report

**Repository:** `{Path(__file__).resolve().parents[1]}`  
**Methodology:** `{METHODOLOGY}`  
**Status:** **COMPLETE — PRE-PROTECTED-TEST FREEZE CREATED**

## A. Executive result

All prescribed stages completed within the 90-minute execution limit. The protected take was not evaluated. The provisional subsystem classification is:

**{classification}**

## B. Data

- Training: `osc_001`, `fig8_001`, `fig8_002`, `fig8vertical_001`, `osc_002`.
- Provisional Validation: `fig8_003`, `osc_003`.
- Protected Test: `fig8vertical_002` — not predictively evaluated.
- Training PhysicalEpisodes: {coverage['physical_episode_count']}.
- Training physical duration: {float(coverage['physical_duration_s']):.2f} s.

## C. UAV physics refit

- Method: `{UAV_METHOD}`.
- Sobol candidates: {uav_fit['sobol_candidates']}.
- Adam refinements: {uav_fit['refinements']} × {uav_fit['adam_updates_per_refinement']} updates.
- Runtime: {float(uav_fit['runtime_s']):.2f} s.
- Training objective: {float(uav_fit['best_training_objective']):.6g}.
- Full-episode Validation position RMSE: {_fmt_metric(validation_u, 'position_rmse_m', 1000)} mm.
- Full-episode Validation orientation RMSE: {_fmt_metric(validation_u, 'orientation_rmse_deg')} deg.
- Terminal position RMSE: {_fmt_metric(validation_u, 'terminal_position_rmse_m', 1000)} mm.
- Terminal orientation RMSE: {_fmt_metric(validation_u, 'terminal_orientation_rmse_deg')} deg.

## D. Cable physics fit

- Method: `{CABLE_METHOD}`.
- Boundary: measured Motive UAV position/orientation mapped through the production rigid clamp.
- Search: three deterministic 8×8 log-space grids.
- Runtime: {float(cable_fit['runtime_s']):.2f} s.
- Final training objective: {float(cable_fit['best_training_objective']):.6g}.
- Loss landscape: `cable_loss_landscape.png`.

## E. Conditional cable validation

These results are **conditional on the real measured UAV boundary** and are not end-to-end command predictions.

- Distributed marker RMSE: {1000 * conditional_marker:.3f} mm.
- Free-tip RMSE: {_fmt_metric(conditional_validation, 'tip_rmse_m', 1000)} mm.
- Terminal tip RMSE: {_fmt_metric(conditional_validation, 'terminal_tip_rmse_m', 1000)} mm.

## F. Causal residual coverage

Residual results apply only to causally eligible suffixes determined by `PhysicalEpisode.residual_eligible_start_index`.

- PhysicalEpisodes: {coverage['physical_episode_count']}.
- Residual-eligible suffixes: {coverage['residual_eligible_suffix_count']}.
- Full duration: {float(coverage['physical_duration_s']):.2f} s.
- Residual-eligible duration: {float(coverage['residual_eligible_duration_s']):.2f} s.
- Excluded duration: {float(coverage['excluded_duration_s']):.2f} s.
- No fixed offset was hard-coded, no history was fabricated, and no command gap was crossed.

Physics-only full-episode metrics remain separately available.

## G. UAV residual refit

- Method: `{RESIDUAL_METHOD}`.
- Architecture: 90 → 32 → SiLU → 32 → SiLU → 3.
- Historical weights reused: **NO**.
- Runtime: {float(residual_fit['runtime_s']):.2f} s.
- Best Training objective: {float(residual_fit['best_training_objective']):.6g}.
- Weight SHA-256: `{residual_fit['network_state_sha256']}`.
- Same-suffix Validation position RMSE, physics: {_fmt_metric(validation_same_u, 'position_rmse_m', 1000)} mm.
- Same-suffix Validation position RMSE, physics+residual: {_fmt_metric(validation_same_ur, 'position_rmse_m', 1000)} mm.
- Same-suffix Validation orientation RMSE, physics: {_fmt_metric(validation_same_u, 'orientation_rmse_deg')} deg.
- Same-suffix Validation orientation RMSE, physics+residual: {_fmt_metric(validation_same_ur, 'orientation_rmse_deg')} deg.

## H. End-to-end validation

No measured UAV boundary or measured state reset was used after initialization.

| Configuration | UAV position RMSE [mm] | Distributed cable RMSE [mm] | Tip RMSE [mm] |
|---|---:|---:|---:|
| Full-episode Model P | {1000 * end_aggregate(end_full, 'uav_position_rmse_m'):.3f} | {1000 * p_marker:.3f} | {1000 * end_aggregate(end_full, 'tip_rmse_m'):.3f} |
| Same-suffix Model P | {1000 * end_aggregate(end_suffix_p, 'uav_position_rmse_m'):.3f} | {1000 * same_p_marker:.3f} | {1000 * end_aggregate(end_suffix_p, 'tip_rmse_m'):.3f} |
| Same-suffix Model PR | {1000 * end_aggregate(end_suffix_pr, 'uav_position_rmse_m'):.3f} | {1000 * pr_marker:.3f} | {1000 * end_aggregate(end_suffix_pr, 'tip_rmse_m'):.3f} |

## I. Error localization

- Full-episode UAV position RMSE: {1000 * uav_error:.3f} mm.
- Conditional measured-boundary DDER marker RMSE: {1000 * conditional_marker:.3f} mm.
- Full end-to-end Model-P marker RMSE: {1000 * p_marker:.3f} mm.
- Same-suffix Model-P marker RMSE: {1000 * same_p_marker:.3f} mm.
- Same-suffix Model-PR marker RMSE: {1000 * pr_marker:.3f} mm.

These comparisons localize whether error is already present in boundary-conditioned cable physics or is introduced mainly by command-to-UAV boundary prediction. They do not establish a new missing-physics mechanism by themselves.

## J. Final seven parameters

{_table_parameters(uav_fit, cable_fit)}

## K. Should more physics be added?

**{classification}**

No additional physics was implemented. This classification is provisional and based only on the frozen Validation results.

## L. Freeze

- Freeze: `{freeze['path']}`.
- Aggregate source SHA-256: `{freeze['source']['aggregate_source_sha256']}`.
- Artifact manifest SHA-256: `{freeze['artifact_manifest_sha256']}`.
- The working-tree state is explicitly recorded because the repository contains the intentional legacy restructuring.

## M. Protected test

`fig8vertical_002` **WAS NOT PREDICTIVELY EVALUATED**.
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")


def create_freeze(
    output: Path,
    root: Path,
    summary: dict[str, object],
    residual_weights: Path,
    residual_normalization: Path,
    *,
    destination: Path = DEFAULT_FREEZE,
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(
            f"Pretest freeze already exists and will not be overwritten: {destination}"
        )
    destination.mkdir(parents=True)
    copied = {
        "physical_parameters.json": output / "physical_parameters.json",
        "residual_weights.pt": residual_weights,
        "residual_normalization.json": residual_normalization,
        "dataset_snapshot.json": output / "dataset_snapshot.json",
        "episode_manifest.json": output / "episode_manifest.json",
        "verification.json": output / "verification.json",
    }
    for name, source in copied.items():
        shutil.copy2(source, destination / name)
    source = _source_manifest(root)
    atomic_json(destination / "source_manifest.json", source)
    protocol = {
        "schema": "decomposed_pretest_freeze_v1",
        "model_version": "aerial_cable_attitude_coupled_v1",
        "methodology": METHODOLOGY,
        "residual_eligibility_rule": "PhysicalEpisode.residual_eligible_start_index",
        "protected_take_id": "fig8vertical_002",
        "protected_test_predictively_evaluated": False,
        "validation_protocol": {
            "physics_only": "complete original PhysicalEpisodes",
            "residual_comparison": "same causal residual suffix for U and UR",
            "end_to_end_residual_comparison": "same suffix for P and PR",
        },
        "source": source,
    }
    atomic_json(destination / "freeze_protocol.json", protocol)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest_hash = canonical_json_hash(hashes)
    manifest = {
        **protocol,
        "created_utc": utc_now(),
        "artifact_hashes": hashes,
        "artifact_manifest_sha256": manifest_hash,
        "fit_directory": str(output),
    }
    atomic_json(destination / "manifest.json", manifest)
    return {
        "path": str(destination),
        "source": source,
        "artifact_manifest_sha256": manifest_hash,
    }


def _remeasured_geometry_and_mass_audit(
    settings: SimulatorSettings,
    root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    processing = json.loads(
        (root / "experimental_data" / "default_processing.json").read_text(
            encoding="utf-8"
        )
    )
    transform = processing["source_to_simulator"]
    assert isinstance(transform, dict)
    cable = settings.cable_configuration
    geometry = {
        "schema": "milestone3c4_updated_geometry_manifest_v1",
        "geometry_version": REMEASURED_GEOMETRY_VERSION,
        "attachment_reference_audit": {
            "result": "PASS",
            "motive_rigid_body_label": processing["uav_rigid_body_label"],
            "motive_position_semantics": "rigid-body reference position",
            "source_coordinate_space": transform["source_world"],
            "source_to_simulator_rotation": transform["rotation"],
            "source_to_simulator_translation_m": transform["translation_m"],
            "simulator_pose_source": "processed uav_position_m/uav_orientation_xyzw from the same Motive rigid body",
            "attachment_equation": "p_connector = p_rigid_body + R(q) d_attachment_body",
            "alternate_pose_origin_or_hidden_translation_found": False,
            "conclusion": "The simulator origin is the Motive rigid-body reference from which the physical offset was measured.",
        },
        "attachment_offset_body_m": list(settings.attachment_offset_body_m),
        "attachment_tangent_body": list(settings.attachment_tangent_body),
        "node_count": cable.node_count,
        "edge_count": cable.edge_count,
        "topology": {
            "node_0": "connector/root, prescribed",
            "node_1": "artificial clamp-support node, prescribed",
            "nodes_2_to_11": "c1...c10, dynamic/observed",
            "free_tip_node": 11,
        },
        "marker_node_indices": list(cable.marker_node_indices[1:]),
        "marker_labels": [f"c{index}" for index in range(1, 11)],
        "marker_interval_lengths_m": list(cable.marker_interval_lengths_m),
        "rest_lengths_m": list(cable.rest_lengths_m),
        "rest_length_sum_m": cable.length_m,
        "connector_to_c1_m": cable.marker_interval_lengths_m[0],
        "clamp_support_split_m": list(cable.rest_lengths_m[:2]),
    }
    if settings.attachment_offset_body_m != (0.0, 0.0, -0.055):
        raise RuntimeError("Remeasured attachment offset is not active.")
    if cable.node_count != 12 or cable.edge_count != 11:
        raise RuntimeError("Remeasured production topology must be 12 nodes/11 edges.")
    if cable.marker_node_indices[1:] != tuple(range(2, 12)):
        raise RuntimeError("c1...c10 must map exactly to nodes 2...11.")
    if any(
        abs(actual - expected) > 1.0e-12
        for actual, expected in zip(
            cable.rest_lengths_m, REMEASURED_REST_LENGTHS_M, strict=True
        )
    ) or abs(cable.length_m - 0.9525) > 1.0e-12:
        raise RuntimeError("Remeasured rest-length vector is not active.")

    old_audit_path = (
        root
        / "data"
        / "fit_results_12node"
        / "2026-08-28T221032.685038+0000_0eea8215"
        / "topology_mass_observation_audit.json"
    )
    old_audit = json.loads(old_audit_path.read_text(encoding="utf-8"))
    old_mass = old_audit["new_12node"]
    marker_total = sum(cable.moving_marker_masses_kg)
    total = sum(cable.vertex_masses_kg)
    bare_from_vertices = list(cable.vertex_masses_kg)
    for node, marker_mass in zip(
        cable.marker_node_indices[1:], cable.moving_marker_masses_kg, strict=True
    ):
        bare_from_vertices[node] -= marker_mass
    mass = {
        "schema": "milestone3c4_mass_audit_v1",
        "edge_dual_volume_convention_recomputed": True,
        "old_geometry_length_m": old_audit["authoritative_physical_length_m"],
        "new_geometry_length_m": cable.length_m,
        "old_total_bare_cable_mass_kg": old_mass["bare_cable_mass_kg"],
        "new_total_bare_cable_mass_kg": sum(bare_from_vertices),
        "configured_bare_cable_mass_kg": cable.bare_cable_mass_kg,
        "old_marker_mass_total_kg": old_mass["marker_mass_total_kg"],
        "new_marker_mass_total_kg": marker_total,
        "old_total_modeled_mass_kg": old_mass["total_modeled_mass_kg"],
        "new_total_modeled_mass_kg": total,
        "old_vertex_masses_kg": old_mass["vertex_masses_kg"],
        "new_vertex_masses_kg": list(cable.vertex_masses_kg),
        "new_bare_cable_vertex_contributions_kg": bare_from_vertices,
        "new_linear_bare_mass_density_kg_m": cable.bare_cable_mass_kg / cable.length_m,
        "mass_conservation": {
            "bare_cable": abs(float(old_mass["bare_cable_mass_kg"]) - sum(bare_from_vertices)) <= 1.0e-12,
            "markers": abs(float(old_mass["marker_mass_total_kg"]) - marker_total) <= 1.0e-12,
            "total": abs(float(old_mass["total_modeled_mass_kg"]) - total) <= 1.0e-12,
        },
        "copied_old_node_masses": False,
    }
    if not all(bool(value) for value in mass["mass_conservation"].values()):
        raise RuntimeError("Remeasured geometry does not conserve physical cable mass.")
    return geometry, mass


def _create_remeasured_output(
    dataset: Dataset,
    config: FitConfiguration,
    output_root: Path,
) -> Path:
    fit_configuration = {
        "schema": "milestone3c4_remeasured_geometry_fit_config_v1",
        "workflow": "cable-refit-pre-mppi",
        "uav_physics": "frozen_not_refitted",
        "uav_residual": "frozen_not_retrained",
        "cable_fit": {
            "parameters": ["EI", "Cb"],
            "pass_1": {"EI": [2.0e-4, 3.0e-3], "Cb": [1.0e-5, 1.5e-3]},
            "grid_shape": [8, 8],
            "maximum_passes": 3,
            "boundary_extension_threshold_relative": 0.01,
            "objective": "equal-take pseudo-Huber over complete Training PhysicalEpisodes",
            "robust_scale_m": config.robust_cable_scale_m,
        },
        "validation": {
            "takes": ["fig8_003", "osc_003"],
            "continuous_prefix_s": 1.0,
            "lead_times_s": list(PRE_MPPI_LEAD_TIMES_S),
            "thresholds_at_0p7_s": PRE_MPPI_THRESHOLDS,
        },
        "backend": {
            "device": "CUDA",
            "dtype": "float32",
            "damping": VALIDATED_OPTIMIZED_DAMPING_BACKEND,
            "substeps": 3,
            "position_projections": 4,
        },
        "hard_stop_s": 1200.0,
        "protected_take": "fig8vertical_002",
    }
    snapshot = dataset_snapshot(dataset, fit_configuration)
    fit_id = (
        utc_now().replace(":", "").replace("+00:00", "Z")
        + "_"
        + canonical_json_hash(snapshot)[:8]
    )
    output = output_root / fit_id
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "dataset_snapshot.json", snapshot)
    atomic_json(output / "fit_config.json", fit_configuration)
    return output


def _load_frozen_pr(
    root: Path,
    simulator: CoupledSimulator,
) -> tuple[dict[str, float], CausalTranslationalResidual, Path, str]:
    freeze = root / "data" / "model_freezes" / "MODEL_FREEZE_DECOMPOSED_PRETEST"
    physical = json.loads((freeze / "physical_parameters.json").read_text(encoding="utf-8"))
    expected = {
        "K_p": 4.020097778647703,
        "K_v": 12.05728865003419,
        "k_a": 0.7327301468333923,
        "K_R": 69.18419375930429,
        "K_omega": 11.456583174321011,
    }
    for name, value in expected.items():
        if abs(float(physical[name]) - value) > 1.0e-12:
            raise RuntimeError(f"Frozen UAV parameter {name} changed unexpectedly.")
    weights = freeze / "residual_weights.pt"
    weight_hash = sha256_file(weights)
    if weight_hash != "f39b9e63c90f550bd07b6c9c2f91aec23469762aee9df6abc391334a5c75a18c":
        raise RuntimeError("Frozen residual weight hash changed unexpectedly.")
    normalization = json.loads(
        (freeze / "residual_normalization.json").read_text(encoding="utf-8")
    )
    residual = CausalTranslationalResidual(
        torch.as_tensor(
            normalization["mean"], dtype=simulator.dtype, device=simulator.device
        ),
        torch.as_tensor(
            normalization["standard_deviation"],
            dtype=simulator.dtype,
            device=simulator.device,
        ),
        seed=42,
    ).to(device=simulator.device, dtype=simulator.dtype)
    residual.load_state_dict(
        torch.load(weights, map_location=simulator.device, weights_only=True)
    )
    residual.eval()
    return expected, residual, freeze, weight_hash


def _create_remeasured_freeze(
    output: Path,
    root: Path,
    *,
    geometry: dict[str, object],
    mass: dict[str, object],
    cable_fit: dict[str, object],
    conditional: dict[str, object],
    end_to_end: dict[str, object],
    acceptance: dict[str, object],
    component_freeze: Path,
    active_manifest: dict[str, object],
    destination: Path,
) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"PRE_MPPI freeze already exists: {destination}")
    destination.mkdir(parents=True)
    for name, payload in (
        ("geometry_manifest.json", geometry),
        ("mass_audit.json", mass),
        ("fitted_cable_parameters.json", cable_fit),
        ("conditional_task_horizon_validation.json", conditional),
        ("pr_end_to_end_task_horizon_validation.json", end_to_end),
        ("pre_mppi_acceptance.json", acceptance),
        ("active_model_manifest.json", active_manifest),
    ):
        atomic_json(destination / name, payload)
    uav = json.loads(
        (component_freeze / "physical_parameters.json").read_text(encoding="utf-8")
    )
    fitted = cable_fit["fitted_parameters"]
    physical = {
        "schema": "remeasured_geometry_pre_mppi_parameters_v1",
        **{name: float(uav[name]) for name in UAV_PARAMETER_NAMES},
        "EI": float(fitted["EI"]),
        "Cb": float(fitted["Cb"]),
        "uav_physics_refitted": False,
        "uav_residual_retrained": False,
    }
    atomic_json(destination / "physical_parameters.json", physical)
    for name in (
        "residual_weights.pt",
        "residual_normalization.json",
        "dataset_snapshot.json",
        "episode_manifest.json",
        "source_hash_manifest.json",
    ):
        source = (
            component_freeze / name
            if name.startswith("residual_")
            else output / name
        )
        shutil.copy2(source, destination / name)
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(destination.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema": "remeasured_geometry_pre_mppi_freeze_v1",
        "created_utc": utc_now(),
        "fit_directory": str(output),
        "geometry_version": REMEASURED_GEOMETRY_VERSION,
        "model_status": "MODEL_FROZEN_FOR_MPPI",
        "production_predictor": "UAV Physics + UAV Residual + 12-node DDER",
        "residual_eligibility_rule": "PhysicalEpisode.residual_eligible_start_index",
        "protected_take_id": "fig8vertical_002",
        "protected_test_predictively_evaluated": False,
        "artifact_hashes": hashes,
        "artifact_manifest_sha256": canonical_json_hash(hashes),
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def _render_remeasured_report(
    report_path: Path,
    *,
    output: Path,
    geometry: dict[str, object],
    mass: dict[str, object],
    cable_fit: dict[str, object],
    conditional: dict[str, object],
    end_to_end: dict[str, object],
    acceptance: dict[str, object],
    runtime_s: float,
    freeze_created: bool,
    source: dict[str, object],
) -> None:
    fitted = cable_fit["fitted_parameters"]
    ident = cable_fit["identifiability"]
    conditional_07 = conditional["validation"]["lead_time"]["0.7"]
    end_07 = end_to_end["aggregate_equal_take_rmse"]["lead_time"]["0.7"]
    passes = "\n".join(
        f"| {item['pass']} | {item['best_EI']:.9g} | {item['best_Cb']:.9g} | "
        f"{item['best_objective']:.9g} | {item['runtime_s']:.2f} |"
        for item in cable_fit["passes"]
    )
    status = "PASS" if bool(acceptance["pass"]) else "FAIL"
    freeze_text = (
        "`MODEL_FREEZE_REMEASURED_GEOMETRY_PRE_MPPI` was created and selected explicitly."
        if freeze_created
        else "No PRE_MPPI freeze was created."
    )
    text = f"""# Milestone 3C.4 — Remeasured-Geometry Cable Refit and PRE_MPPI Report

**Repository:** `{Path(__file__).resolve().parents[1]}`  
**Status:** **{status}**  
**Fit artifact:** `{output}`

## 1. Geometry and reference audit

The audit passed. Motive rigid body `{geometry['attachment_reference_audit']['motive_rigid_body_label']}` is copied into `uav_position_m/uav_orientation_xyzw` with an identity source-to-simulator transform. The production clamp then applies `p_connector = p_rigid_body + R(q)d_body`. No alternate pose origin or hidden translation was found.

- Attachment offset: `{geometry['attachment_offset_body_m']}` m (55 mm body -Z).
- Cable length, connector to c10: **{1000*float(geometry['rest_length_sum_m']):.1f} mm**.
- Rest lengths [m]: `{geometry['rest_lengths_m']}`.
- Topology: 12 nodes, 11 edges; node 0 and node 1 prescribed; c1...c10 map to nodes `{geometry['marker_node_indices']}`.
- Connector-to-c1 is 63 mm, split into two 31.5-mm clamp edges. Node 1 is not an observation.

## 2. Mass audit

- Bare cable: old/new = {float(mass['old_total_bare_cable_mass_kg']):.8f} / {float(mass['new_total_bare_cable_mass_kg']):.8f} kg.
- Marker total: old/new = {float(mass['old_marker_mass_total_kg']):.8f} / {float(mass['new_marker_mass_total_kg']):.8f} kg.
- Total modeled: old/new = {float(mass['old_total_modeled_mass_kg']):.8f} / {float(mass['new_total_modeled_mass_kg']):.8f} kg.
- The edge-dual-volume bare masses were recomputed from the new rest lengths; old node masses were not copied.

## 3. Frozen upstream model and backend

- UAV Physics was **not refitted**.
- UAV Residual was **not retrained**.
- Physics-only Model P was **not evaluated**.
- Production path: CUDA float32, PCG32, fused DDER/projection, 3 substeps, 4 position projections.

## 4. EI/Cb search

The fit used all accepted Training PhysicalEpisodes, each initialized once at its original causal start and propagated continuously to its original end. The loss used all valid c1...c10 observations, the frozen pseudo-Huber scale, and equal top-level take weighting.

| Pass | Best EI [N m²] | Best Cb [N m² s] | Objective | Runtime [s] |
|---:|---:|---:|---:|---:|
{passes}

- Final EI: **{float(fitted['EI']):.12g} N m²** — **{ident['EI']['status']}**.
- Final Cb: **{float(fitted['Cb']):.12g} N m² s** — **{ident['Cb']['status']}**.
- Best objective: {float(cable_fit['best_training_objective']):.12g}.
- EI adjacent profile evidence: `{ident['EI']['neighboring_profile_values']}`.
- Cb adjacent profile evidence: `{ident['Cb']['neighboring_profile_values']}`.
- EI full profiled span: {100*float(ident['EI']['full_profiled_span_relative_to_minimum']):.3f}%.
- Cb full profiled span: {100*float(ident['Cb']['full_profiled_span_relative_to_minimum']):.3f}%.

## 5. Conditional task-horizon validation

Measured Motive UAV pose drove the production rigid clamp. Each eligible provisional-validation PhysicalEpisode was initialized once and propagated as one continuous prefix for at most 1.0 s; there were no rolling windows or resets.

At 0.70 s:

- Distributed c1...c10 RMSE: **{1000*float(conditional_07['distributed_marker_rmse_m']):.3f} mm**.
- c10 tip RMSE: **{1000*float(conditional_07['tip_rmse_m']):.3f} mm**.

Full lead-time data are in `conditional_task_horizon_validation.json`.

## 6. PR-only end-to-end task-horizon validation

The only end-to-end predictor evaluated was recorded FullState → frozen UAV Physics + frozen UAV Residual → rigid clamp → fitted 12-node DDER. No measured UAV or cable state entered after the one causal initialization.

Lead-time aggregation first computes RMSE across eligible episodes within each physical take and then gives `fig8_003` and `osc_003` equal top-level weight. Final verification corrected an initially episode-weighted summary before the freeze was finalized; no trajectory, fitted parameter, or per-episode metric changed.

At 0.70 s:

- UAV position RMSE: **{1000*float(end_07['uav_position_rmse_m']):.3f} mm**.
- UAV orientation RMSE: **{float(end_07['uav_orientation_rmse_deg']):.3f} deg**.
- Distributed c1...c10 RMSE: **{1000*float(end_07['distributed_marker_rmse_m']):.3f} mm**.
- c10 tip RMSE: **{1000*float(end_07['tip_rmse_m']):.3f} mm**.

## 7. Saved historical comparison

No old-geometry trajectory was rerun.

| Metric at 0.70 s | Old 0.943-m saved | New 0.9525-m |
|---|---:|---:|
| Conditional distributed | 42.108 mm | {1000*float(conditional_07['distributed_marker_rmse_m']):.3f} mm |
| Conditional tip | 66.287 mm | {1000*float(conditional_07['tip_rmse_m']):.3f} mm |
| PR UAV position | 11.491 mm | {1000*float(end_07['uav_position_rmse_m']):.3f} mm |
| PR UAV orientation | 4.112 deg | {float(end_07['uav_orientation_rmse_deg']):.3f} deg |
| PR distributed cable | 53.286 mm | {1000*float(end_07['distributed_marker_rmse_m']):.3f} mm |
| PR tip | 103.716 mm | {1000*float(end_07['tip_rmse_m']):.3f} mm |

## 8. Frozen acceptance decision

The predeclared 0.70-s checks were: `{acceptance['threshold_checks']}`. Fatal rollout failures: `{acceptance['fatal_failures']}`. Initialization-ineligible episodes remain explicit exclusions: `{acceptance['initialization_ineligible_episodes']}`.

**PRE_MPPI = {status}.** {freeze_text}

## 9. Production status and protected test

- Active-model status after the run: `{acceptance['resulting_active_status']}`.
- Identification UI source: the explicit active-model/result status API; no newest-timestamp selection.
- `fig8vertical_002`: **PROTECTED — NOT EVALUATED**.

## 10. Runtime and reproducibility

- Total workflow runtime: {runtime_s:.2f} s.
- Git commit: `{source['git_commit']}`.
- Aggregate source SHA-256: `{source['aggregate_source_sha256']}`.
- Working tree dirty state was recorded: `{source['working_tree_dirty']}`.
- Detailed source/config hashes: `source_hash_manifest.json`.

## 11. Final decision

    EI = {float(fitted['EI']):.12g}
    status = {ident['EI']['status']}

    Cb = {float(fitted['Cb']):.12g}
    status = {ident['Cb']['status']}

    Conditional @ 0.70 s:
        distributed = {1000*float(conditional_07['distributed_marker_rmse_m']):.3f} mm
        tip = {1000*float(conditional_07['tip_rmse_m']):.3f} mm

    PR End-to-End @ 0.70 s:
        UAV position = {1000*float(end_07['uav_position_rmse_m']):.3f} mm
        UAV orientation = {float(end_07['uav_orientation_rmse_deg']):.3f} deg
        distributed cable = {1000*float(end_07['distributed_marker_rmse_m']):.3f} mm
        tip = {1000*float(end_07['tip_rmse_m']):.3f} mm

    PRE_MPPI = {status}
"""
    if bool(acceptance["pass"]):
        text += "\nGeneric system identification is now frozen. The next scientific milestone is task formulation plus MPPI.\n"
    else:
        text += f"\nFailure reason: {acceptance['failure_reason']}\n"
    report_path.write_text(text, encoding="utf-8")


def run_remeasured_geometry_pre_mppi(
    dataset: Dataset,
    config: FitConfiguration,
    settings: SimulatorSettings,
    *,
    root: Path | None = None,
    output_root: Path = DEFAULT_REMEASURED_RESULTS,
    report_path: Path = DEFAULT_REMEASURED_REPORT,
    freeze_path: Path = DEFAULT_REMEASURED_FREEZE,
    wall_clock_limit_s: float = 1200.0,
) -> Path:
    """Run only the authoritative 3C.4 cable refit and PRE_MPPI gate."""

    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    started = time.perf_counter()
    deadline = started + min(float(wall_clock_limit_s), 1200.0)
    expected_roles = {
        "training": {"osc_001", "fig8_001", "fig8_002", "fig8vertical_001", "osc_002"},
        "validation": {"fig8_003", "osc_003"},
        "untouched_test": {"fig8vertical_002"},
    }
    actual_roles = {
        "training": {take.take_id for take in dataset.training},
        "validation": {take.take_id for take in dataset.validation},
        "untouched_test": {take.take_id for take in dataset.untouched_test},
    }
    if actual_roles != expected_roles:
        raise RuntimeError(f"Dataset roles differ from the frozen contract: {actual_roles}")
    if settings.torch_device().type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Milestone 3C.4 requires the production CUDA backend.")
    output = _create_remeasured_output(dataset, config, output_root)
    status: dict[str, object] = {
        "schema": "milestone3c4_execution_status_v1",
        "status": "running",
        "fit_directory": str(output),
        "started_utc": utc_now(),
        "wall_clock_limit_s": min(float(wall_clock_limit_s), 1200.0),
        "uav_physics_refitted": False,
        "uav_residual_retrained": False,
        "physics_only_evaluated": False,
        "protected_test_predictively_evaluated": False,
        "stages": {},
    }

    def stage(name: str, value: str, **details: object) -> None:
        stages = status["stages"]
        assert isinstance(stages, dict)
        stages[name] = {"status": value, **details}
        status["elapsed_s"] = time.perf_counter() - started
        atomic_json(output / "execution_status.json", status)
        print(f"[Milestone 3C.4] {name}: {value}", flush=True)

    geometry, mass = _remeasured_geometry_and_mass_audit(settings, root)
    atomic_json(output / "updated_geometry_manifest.json", geometry)
    atomic_json(output / "mass_audit.json", mass)
    stage("geometry_mass_audit", "complete")

    simulator = _uav_simulator(
        settings, device=settings.torch_device(), dtype=torch.float32
    )
    backend = simulator.forward_backend_audit()
    expected_backend = {
        "node_count": 12,
        "substeps": 3,
        "position_projection_iterations": 4,
        "damping_backend": VALIDATED_OPTIMIZED_DAMPING_BACKEND,
        "production_fused_forward_active": True,
    }
    for key, value in expected_backend.items():
        if backend[key] != value:
            raise RuntimeError(f"Production backend audit failed for {key}: {backend[key]!r}")
    uav_parameters, _unused_residual, component_freeze, residual_hash = _load_frozen_pr(
        root, simulator
    )
    del _unused_residual
    atomic_json(
        output / "preflight_audit.json",
        {
            "geometry": geometry,
            "mass": mass,
            "backend": backend,
            "frozen_uav_parameters": uav_parameters,
            "uav_physics_refitted": False,
            "residual_sha256": residual_hash,
            "uav_residual_retrained": False,
            "physics_only_evaluated": False,
            "protected_test_predictively_evaluated": False,
        },
    )

    train_views = physical_episode_views(dataset.training, config)
    validation_views = physical_episode_views(dataset.validation, config)
    residual_validation_views, residual_coverage = causal_residual_suffixes(
        dataset.validation, config
    )
    conditional_prefix_views = task_horizon_views(
        validation_views, dt_s=settings.dt_s, maximum_duration_s=1.0
    )
    residual_prefix_views = task_horizon_views(
        residual_validation_views, dt_s=settings.dt_s, maximum_duration_s=1.0
    )
    manifest = episode_manifest((*dataset.training, *dataset.validation), config)
    manifest.update(
        {
            "milestone": "3C.4",
            "training_fit_uses_complete_physical_episodes": True,
            "conditional_validation_prefix_s": 1.0,
            "pr_validation_prefix_s": 1.0,
            "residual_validation_coverage": residual_coverage,
            "protected_test_predictively_evaluated": False,
        }
    )
    atomic_json(output / "episode_manifest.json", manifest)

    training_audit: list[dict[str, object]] = []
    conditional_audit: list[dict[str, object]] = []
    cable_train = prepare_measured_boundary_episodes(
        train_views, config, settings, simulator, audit=training_audit
    )
    cable_validation = prepare_measured_boundary_episodes(
        conditional_prefix_views,
        config,
        settings,
        simulator,
        audit=conditional_audit,
    )
    atomic_json(
        output / "cable_episode_eligibility.json",
        {
            "schema": "milestone3c4_cable_episode_eligibility_v1",
            "training": training_audit,
            "validation": conditional_audit,
            "initialization_threshold_m": config.maximum_initialization_rmse_m,
            "training_episode_boundaries_modified": False,
            "validation_is_single_continuous_prefix": True,
        },
    )
    if not cable_train:
        raise RuntimeError("No Training PhysicalEpisode passed the frozen initialization gate.")
    # Cheap finite production CUDA smoke step, using a real measured boundary.
    smoke = cable_train[0]
    smoke_state = smoke.initial_cable
    smoke_constants = _cable_constants(
        simulator,
        smoke_state,
        float(settings.parameters.cable.EI),
        float(settings.parameters.cable.Cb),
    )
    with torch.no_grad():
        smoke_state = simulator.cable_model.step_runtime(
            smoke_state,
            smoke.measured_boundary_positions_m[1:2],
            torch.tensor([settings.dt_s], dtype=simulator.dtype, device=simulator.device),
            smoke_constants,
            iterative_damping=True,
            damping_backend=VALIDATED_OPTIMIZED_DAMPING_BACKEND,
            pinned_endpoints=simulator.root_boundary.pinned_endpoints,
            create_graph=False,
            functional_force_autograd=simulator.functional_force_autograd,
        )
    torch.cuda.synchronize(simulator.device)
    if not bool(_finite_cable(smoke_state).all()):
        raise FloatingPointError("Production CUDA pre-fit smoke rollout is non-finite.")
    stage(
        "preflight",
        "complete",
        training_episodes=len(cable_train),
        validation_eligible_prefixes=len(cable_validation),
        backend=backend,
    )

    stage("ei_cb_fit", "running")
    (EI, Cb), cable_fit = fit_remeasured_cable_grid(
        cable_train,
        simulator,
        settings,
        config,
        output,
        deadline=deadline,
    )
    stage("ei_cb_fit", "complete", EI=EI, Cb=Cb, runtime_s=cable_fit["runtime_s"])
    if time.perf_counter() >= deadline:
        raise TimeoutError("Milestone 3C.4 hard stop reached after fitting.")

    stage("conditional_validation", "running")
    conditional = evaluate_conditional_cable(
        cable_validation,
        simulator,
        settings,
        EI,
        Cb,
        output,
        "conditional_task_horizon_validation",
        PRE_MPPI_LEAD_TIMES_S,
    )
    stage("conditional_validation", "complete")

    # Build Model PR from the unchanged frozen residual. No Model-P branch is created.
    _uav, residual, component_freeze_check, residual_hash_check = _load_frozen_pr(
        root, simulator
    )
    assert component_freeze_check == component_freeze and residual_hash_check == residual_hash
    pr_simulator = _uav_simulator(
        settings,
        device=settings.torch_device(),
        dtype=torch.float32,
        residual_model=residual,
    )
    prepared_pr = prepare_uav_episodes(
        residual_prefix_views,
        config,
        pr_simulator,
        residual_history=True,
    )
    uav_values = torch.tensor(
        [uav_parameters[name] for name in UAV_PARAMETER_NAMES],
        dtype=pr_simulator.dtype,
        device=pr_simulator.device,
    )
    stage("pr_end_to_end_validation", "running")
    end_to_end = evaluate_end_to_end(
        prepared_pr,
        pr_simulator,
        settings,
        config,
        uav_values,
        EI,
        Cb,
        output,
        "pr_end_to_end_task_horizon_validation",
        PRE_MPPI_LEAD_TIMES_S,
        residual_history=True,
    )
    stage("pr_end_to_end_validation", "complete")

    horizon = end_to_end.get("aggregate_equal_take_rmse", {}).get(
        "lead_time", {}
    ).get("0.7")
    if not isinstance(horizon, dict):
        raise RuntimeError("PR validation produced no aggregate 0.70-s metric.")
    checks = {
        key: bool(math.isfinite(float(horizon[key])) and float(horizon[key]) <= limit)
        for key, limit in PRE_MPPI_THRESHOLDS.items()
    }
    ineligible = [
        item
        for item in end_to_end["failed_episodes"]
        if item["reason"] == "cable_initialization_ineligible"
    ]
    fatal = [
        item
        for item in end_to_end["failed_episodes"]
        if item["reason"] != "cable_initialization_ineligible"
    ]
    passed = not fatal and all(checks.values()) and bool(end_to_end["episodes"])
    failed_metrics = [key for key, value in checks.items() if not value]
    failure_reasons = [
        *(f"0.70-s {key} exceeded its frozen threshold" for key in failed_metrics),
        *(f"fatal rollout: {item}" for item in fatal),
    ]
    if not end_to_end["episodes"]:
        failure_reasons.append("no eligible PR validation rollout")
    failure_reason = "none" if passed else "; ".join(failure_reasons)
    acceptance: dict[str, object] = {
        "schema": "milestone3c4_pre_mppi_acceptance_v1",
        "task_horizon_s": 0.7,
        "thresholds": PRE_MPPI_THRESHOLDS,
        "observed": {key: float(horizon[key]) for key in PRE_MPPI_THRESHOLDS},
        "threshold_checks": checks,
        "finite_rollouts": not fatal,
        "measurement_reset_after_initialization": False,
        "physics_only_evaluated": False,
        "fatal_failures": fatal,
        "initialization_ineligible_episodes": ineligible,
        "pass": passed,
        "failure_reason": failure_reason,
        "resulting_active_status": (
            "MODEL_FROZEN_FOR_MPPI" if passed else "MODEL_NOT_YET_FROZEN_FOR_MPPI"
        ),
        "protected_test_predictively_evaluated": False,
    }
    atomic_json(output / "pre_mppi_acceptance.json", acceptance)

    freeze_created = False
    if passed:
        config_path = root / "config" / "default.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload["parameters"]["cable"] = {"EI": EI, "Cb": Cb}
        atomic_json(config_path, config_payload)
        fit_relative = output.relative_to(root).as_posix()
        freeze_relative = freeze_path.relative_to(root).as_posix()
        active_manifest = {
            "schema": "aerial_cable_active_model_v1",
            "model_name": "PR_12NODE_DDER_REMEASURED_PRE_MPPI",
            "status": "MODEL_FROZEN_FOR_MPPI",
            "ready_for_mppi": True,
            "reason_not_ready": "none",
            "geometry_version": REMEASURED_GEOMETRY_VERSION,
            "simulator_configuration": "config/default.json",
            "uav_residual_freeze": "data/model_freezes/MODEL_FREEZE_DECOMPOSED_PRETEST",
            "development_cable_fit": fit_relative,
            "development_cable_fit_geometry_version": REMEASURED_GEOMETRY_VERSION,
            "production_freeze": freeze_relative,
            "production_predictor": "UAV Physics + UAV Residual + 12-node Cable DDER",
            "damping_backend": VALIDATED_OPTIMIZED_DAMPING_BACKEND,
            "precision": "float32",
            "projection_backend": "fused",
            "model_integrity": "Verified",
            "protected_take_id": "fig8vertical_002",
            "protected_test_predictively_evaluated": False,
            "selection_policy": "explicit_manifest_only_never_newest_timestamp",
        }
        source = _source_manifest(root)
        atomic_json(output / "source_hash_manifest.json", source)
        freeze_manifest = _create_remeasured_freeze(
            output,
            root,
            geometry=geometry,
            mass=mass,
            cable_fit=cable_fit,
            conditional=conditional,
            end_to_end=end_to_end,
            acceptance=acceptance,
            component_freeze=component_freeze,
            active_manifest=active_manifest,
            destination=freeze_path,
        )
        atomic_json(root / "config" / "active_model.json", active_manifest)
        freeze_created = True
        stage(
            "pre_mppi_freeze",
            "complete",
            freeze=str(freeze_path),
            artifact_manifest_sha256=freeze_manifest["artifact_manifest_sha256"],
        )
    else:
        source = _source_manifest(root)
        atomic_json(output / "source_hash_manifest.json", source)
        stage("pre_mppi_freeze", "not_created", reason=failure_reason)

    runtime_s = time.perf_counter() - started
    _render_remeasured_report(
        report_path,
        output=output,
        geometry=geometry,
        mass=mass,
        cable_fit=cable_fit,
        conditional=conditional,
        end_to_end=end_to_end,
        acceptance=acceptance,
        runtime_s=runtime_s,
        freeze_created=freeze_created,
        source=source,
    )
    status.update(
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "elapsed_s": runtime_s,
            "pre_mppi_pass": passed,
            "freeze_created": freeze_created,
            "report": str(report_path),
        }
    )
    atomic_json(output / "execution_status.json", status)
    atomic_json(
        output_root / "latest.json",
        {
            "schema": "milestone3c4_latest_pointer_v1",
            "fit_directory": str(output),
            "report": str(report_path),
            "pre_mppi_pass": passed,
            "freeze": str(freeze_path) if freeze_created else None,
        },
    )
    return output


def _create_output_directory(
    dataset: Dataset,
    config: FitConfiguration,
    root: Path,
    output_root: Path,
) -> Path:
    fit_configuration = {
        "schema": "decomposed_full_episode_fit_config_v1",
        "methodology": METHODOLOGY,
        "model_version": "aerial_cable_attitude_coupled_v1",
        "submethods": {
            "uav": UAV_METHOD,
            "cable": CABLE_METHOD,
            "residual": RESIDUAL_METHOD,
            "validation": VALIDATION_METHOD,
        },
        "uav": {
            "sobol_candidates": 64,
            "selected_refinements": 2,
            "configured_adam_update_ceiling_per_refinement": 300,
            "execution_adam_update_ceiling_per_refinement": 60,
            "learning_rate": 0.01,
            "gradient_clip": 10.0,
            "execution_note": "bounded by the milestone 90-minute wall-clock policy",
            "bounds": {name: list(config.bounds[name]) for name in UAV_PARAMETER_NAMES},
        },
        "cable": {
            "passes": 3,
            "grid_shape": [8, 8],
            "space": "log10",
            "bounds": {name: list(config.bounds[name]) for name in ("EI", "Cb")},
            "boundary": "measured Motive UAV pose",
        },
        "residual": {
            **config.uav_residual_ablation,
            "configured_update_ceiling": int(
                config.uav_residual_ablation["optimizer"]["updates"]
            ),
            "execution_update_ceiling": min(
                int(config.uav_residual_ablation["optimizer"]["updates"]), 150
            ),
            "execution_note": "bounded by the milestone 90-minute wall-clock policy",
        },
        "lead_times_s": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "wall_clock_limit_s": 5400.0,
        "protected_take": "fig8vertical_002",
    }
    snapshot = dataset_snapshot(dataset, fit_configuration)
    fit_id = utc_now().replace(":", "").replace("+00:00", "Z") + "_" + canonical_json_hash(snapshot)[:8]
    output = output_root / fit_id
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "dataset_snapshot.json", snapshot)
    atomic_json(output / "fit_config.json", fit_configuration)
    atomic_json(output / "uav_fit_config.json", fit_configuration["uav"])
    atomic_json(output / "cable_fit_config.json", fit_configuration["cable"])
    atomic_json(output / "residual_config.json", fit_configuration["residual"])
    return output


def run_decomposed_identification(
    dataset: Dataset,
    config: FitConfiguration,
    settings: SimulatorSettings,
    *,
    root: Path | None = None,
    output_root: Path = DEFAULT_RESULTS,
    report_path: Path = DEFAULT_REPORT,
    freeze_path: Path = DEFAULT_FREEZE,
    wall_clock_limit_s: float = 5400.0,
    resume_output: Path | None = None,
) -> Path:
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    role_expectation = {
        "training": {"osc_001", "fig8_001", "fig8_002", "fig8vertical_001", "osc_002"},
        "validation": {"fig8_003", "osc_003"},
        "untouched_test": {"fig8vertical_002"},
    }
    actual_roles = {
        "training": {take.take_id for take in dataset.training},
        "validation": {take.take_id for take in dataset.validation},
        "untouched_test": {take.take_id for take in dataset.untouched_test},
    }
    if actual_roles != role_expectation:
        raise RuntimeError(
            f"Dataset roles differ from the frozen Milestone 3C contract: {actual_roles}."
        )
    output = (
        _create_output_directory(dataset, config, root, output_root)
        if resume_output is None
        else resume_output.resolve()
    )
    if not output.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {output}")
    consumed_runtime_s = 0.0
    fitted_uav_path = output / "fitted_uav_parameters.json"
    if resume_output is not None and fitted_uav_path.exists():
        consumed_runtime_s = float(
            json.loads(fitted_uav_path.read_text(encoding="utf-8"))["runtime_s"]
        )
        # A failed/interrupted stage still consumes the user-specified total
        # wall-clock budget.  Account for already completed cable grid passes
        # when resuming after an evaluator/instrumentation failure.
        cable_progress_path = output / "cable_fit_progress.json"
        if cable_progress_path.exists() and not (output / "fitted_cable_parameters.json").exists():
            cable_progress = json.loads(cable_progress_path.read_text(encoding="utf-8"))
            consumed_runtime_s += sum(
                float(pass_result.get("runtime_s", 0.0))
                for pass_result in cable_progress.get("passes", [])
            )
        fitted_cable_path = output / "fitted_cable_parameters.json"
        if fitted_cable_path.exists():
            consumed_runtime_s += float(
                json.loads(fitted_cable_path.read_text(encoding="utf-8"))["runtime_s"]
            )
        residual_fit_path = output / "residual_fit.json"
        if residual_fit_path.exists():
            consumed_runtime_s += float(
                json.loads(residual_fit_path.read_text(encoding="utf-8"))["runtime_s"]
            )
    started = time.perf_counter()
    deadline = started + max(0.0, wall_clock_limit_s - consumed_runtime_s)
    stage_status: dict[str, object] = {
        "schema": "milestone3c_execution_status_v1",
        "fit_directory": str(output),
        "started_utc": utc_now(),
        "wall_clock_limit_s": wall_clock_limit_s,
        "resumed": resume_output is not None,
        "runtime_consumed_before_resume_s": consumed_runtime_s,
        "stages": {},
        "protected_test_predictively_evaluated": False,
    }

    def stage(name: str, status: str, **values: object) -> None:
        stages = stage_status["stages"]
        assert isinstance(stages, dict)
        stages[name] = {"status": status, **values}
        stage_status["elapsed_s"] = time.perf_counter() - started
        atomic_json(output / "execution_status.json", stage_status)
        print(
            f"[Milestone 3C] {name}: {status} "
            f"(elapsed {float(stage_status['elapsed_s']):.1f} s)",
            flush=True,
        )

    lead_times = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    train_views = physical_episode_views(dataset.training, config)
    validation_views = physical_episode_views(dataset.validation, config)
    residual_training_views, training_coverage = causal_residual_suffixes(
        dataset.training, config
    )
    residual_validation_views, validation_coverage = causal_residual_suffixes(
        dataset.validation, config
    )
    coverage = {"training": training_coverage, "validation": validation_coverage}
    atomic_json(output / "causal_residual_coverage.json", coverage)
    manifest = episode_manifest((*dataset.training, *dataset.validation), config)
    manifest["methodology"] = METHODOLOGY
    manifest["protected_test_predictively_evaluated"] = False
    atomic_json(output / "episode_manifest.json", manifest)
    stage("preflight", "complete", training_episodes=len(train_views), validation_episodes=len(validation_views))

    # A. Full-episode attitude-coupled UAV refit.
    # Small UAV episode batches are CPU-latency limited on the RTX path.  CPU
    # execution is numerically gated against the same production model step;
    # DDER population work remains on the configured CUDA device.
    uav_device = torch.device("cpu")
    uav_simulator = _uav_simulator(
        settings, device=uav_device, dtype=torch.float64
    )
    prepared_train = prepare_uav_episodes(train_views, config, uav_simulator)
    prepared_validation = prepare_uav_episodes(validation_views, config, uav_simulator)
    if fitted_uav_path.exists():
        uav_fit = json.loads(fitted_uav_path.read_text(encoding="utf-8"))
        fitted = uav_fit["fitted_parameters"]
        uav_values = torch.tensor(
            [float(fitted[name]) for name in UAV_PARAMETER_NAMES],
            dtype=uav_simulator.dtype,
            device=uav_simulator.device,
        )
        stage("uav_physics_fit", "resumed_complete", runtime_s=uav_fit["runtime_s"])
    else:
        stage("uav_physics_fit", "running")
        uav_values, uav_fit = fit_uav_physics(
            prepared_train,
            uav_simulator,
            settings,
            config,
            output,
            deadline=deadline,
            adam_updates=60,
        )
    uav_metrics = evaluate_uav_model(
        (*prepared_train, *prepared_validation),
        uav_simulator,
        settings,
        uav_values,
        output,
        "uav_physics_metrics",
        lead_times,
    )
    atomic_json(output / "uav_training_metrics.json", uav_metrics["training"])
    atomic_json(output / "uav_validation_physics.json", uav_metrics["validation"])
    stage("uav_physics_fit", "complete", runtime_s=uav_fit["runtime_s"])

    # B/C. Conditional measured-boundary EI/Cb fit and validation.
    cable_simulator = _uav_simulator(
        settings, device=settings.torch_device(), dtype=torch.float32
    )
    cable_training_audit: list[dict[str, object]] = []
    cable_validation_audit: list[dict[str, object]] = []
    cable_train = prepare_measured_boundary_episodes(
        train_views,
        config,
        settings,
        cable_simulator,
        audit=cable_training_audit,
    )
    cable_validation = prepare_measured_boundary_episodes(
        validation_views,
        config,
        settings,
        cable_simulator,
        audit=cable_validation_audit,
    )
    atomic_json(
        output / "cable_episode_eligibility.json",
        {
            "schema": "decomposed_cable_episode_eligibility_v1",
            "training": cable_training_audit,
            "validation": cable_validation_audit,
            "threshold_m": config.maximum_initialization_rmse_m,
            "physical_episode_boundaries_modified": False,
        },
    )
    atomic_json(
        output / "measured_boundary_metadata.json",
        {
            "schema": "measured_uav_boundary_v1",
            "source": "Motive UAV position and xyzw orientation",
            "attachment_offset_body_m": list(settings.attachment_offset_body_m),
            "attachment_tangent_body": list(settings.attachment_tangent_body),
            "first_edge_length_m": settings.cable_configuration.rest_lengths_m[0],
            "definition": "r0=p+R*d; r1=r0+l0*normalize(R*t)",
            "interpolation": "production DDER linear substep interpolation between Motive frames",
            "measured_cable_reset_after_initialization": False,
        },
    )
    fitted_cable_path = output / "fitted_cable_parameters.json"
    if fitted_cable_path.exists():
        cable_fit = json.loads(fitted_cable_path.read_text(encoding="utf-8"))
        fitted_cable = cable_fit["fitted_parameters"]
        EI, Cb = float(fitted_cable["EI"]), float(fitted_cable["Cb"])
        stage("cable_physics_fit", "resumed_complete", runtime_s=cable_fit["runtime_s"])
    else:
        stage("cable_physics_fit", "running")
        (EI, Cb), cable_fit = fit_cable_grid(
            cable_train,
            cable_simulator,
            settings,
            config,
            output,
            deadline=deadline,
        )
    _render_cable_landscape(output)
    conditional_cable = evaluate_conditional_cable(
        (*cable_train, *cable_validation),
        cable_simulator,
        settings,
        EI,
        Cb,
        output,
        "cable_conditional_metrics",
        lead_times,
    )
    atomic_json(
        output / "cable_conditional_training_metrics.json",
        conditional_cable["training"],
    )
    atomic_json(
        output / "cable_conditional_validation.json",
        conditional_cable["validation"],
    )
    stage("cable_physics_fit", "complete", runtime_s=cable_fit["runtime_s"])

    # D. New residual from zero on causal suffixes only.
    nominal_suffix_train = prepare_uav_episodes(
        residual_training_views, config, uav_simulator, residual_history=True
    )
    nominal_suffix_validation = prepare_uav_episodes(
        residual_validation_views, config, uav_simulator, residual_history=True
    )
    normalization_path = output / "residual_normalization.json"
    if normalization_path.exists():
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        mean = torch.as_tensor(
            normalization["mean"], dtype=uav_simulator.dtype, device=uav_simulator.device
        )
        std = torch.as_tensor(
            normalization["standard_deviation"],
            dtype=uav_simulator.dtype,
            device=uav_simulator.device,
        )
    else:
        mean, std, normalization = compute_residual_normalization(
            nominal_suffix_train,
            uav_simulator,
            settings,
            uav_values,
            float(config.uav_residual_ablation["normalization_std_floor"]),
        )
        atomic_json(normalization_path, normalization)
    residual = CausalTranslationalResidual(
        mean,
        std,
        seed=int(config.uav_residual_ablation["optimizer"]["seed"]),
    ).to(device=uav_simulator.device, dtype=uav_simulator.dtype)
    residual_simulator = _uav_simulator(
        settings,
        device=uav_device,
        dtype=torch.float64,
        residual_model=residual,
    )
    residual_train = prepare_uav_episodes(
        residual_training_views, config, residual_simulator, residual_history=True
    )
    residual_validation = prepare_uav_episodes(
        residual_validation_views, config, residual_simulator, residual_history=True
    )
    residual_fit_path = output / "residual_fit.json"
    residual_weights_path = output / "residual_weights.pt"
    if residual_fit_path.exists() and residual_weights_path.exists():
        residual.load_state_dict(
            torch.load(
                residual_weights_path,
                map_location=uav_simulator.device,
                weights_only=True,
            )
        )
        residual_fit = json.loads(residual_fit_path.read_text(encoding="utf-8"))
        stage("uav_residual_fit", "resumed_complete", runtime_s=residual_fit["runtime_s"])
    else:
        stage("uav_residual_fit", "running")
        residual_fit = fit_residual(
            residual_train,
            residual_simulator,
            residual,
            settings,
            config,
            uav_values,
            output,
            deadline=deadline,
        )
    same_suffix_physics = evaluate_uav_model(
        (*nominal_suffix_train, *nominal_suffix_validation),
        uav_simulator,
        settings,
        uav_values,
        output,
        "uav_same_suffix_physics",
        lead_times,
    )
    same_suffix_residual = evaluate_uav_model(
        (*residual_train, *residual_validation),
        residual_simulator,
        settings,
        uav_values,
        output,
        "residual_validation",
        lead_times,
    )
    stage("uav_residual_fit", "complete", runtime_s=residual_fit["runtime_s"])

    # E. Genuine command-to-UAV-to-DDER validation.  Model P is reported both
    # on complete episodes and the same suffix used by Model PR.
    fitted_uav = [float(uav_values[index].cpu()) for index in range(5)]
    p_simulator = _uav_simulator(
        settings, device=settings.torch_device(), dtype=torch.float32
    )
    p_full = prepare_uav_episodes(validation_views, config, p_simulator)
    p_suffix = prepare_uav_episodes(
        residual_validation_views, config, p_simulator, residual_history=True
    )
    residual32 = CausalTranslationalResidual(
        torch.as_tensor(mean, dtype=torch.float32, device=p_simulator.device),
        torch.as_tensor(std, dtype=torch.float32, device=p_simulator.device),
        seed=int(config.uav_residual_ablation["optimizer"]["seed"]),
    ).to(device=p_simulator.device, dtype=torch.float32)
    residual32.load_state_dict(residual.state_dict())
    pr_simulator = _uav_simulator(
        settings,
        device=settings.torch_device(),
        dtype=torch.float32,
        residual_model=residual32,
    )
    pr_suffix = prepare_uav_episodes(
        residual_validation_views, config, pr_simulator, residual_history=True
    )
    stage("end_to_end_validation", "running")
    end_full = evaluate_end_to_end(
        p_full,
        p_simulator,
        settings,
        config,
        torch.tensor(fitted_uav, dtype=torch.float32, device=p_simulator.device),
        EI,
        Cb,
        output,
        "end_to_end_physics_validation",
        lead_times,
        residual_history=False,
    )
    end_suffix_p = evaluate_end_to_end(
        p_suffix,
        p_simulator,
        settings,
        config,
        torch.tensor(fitted_uav, dtype=torch.float32, device=p_simulator.device),
        EI,
        Cb,
        output,
        "end_to_end_same_suffix_physics",
        lead_times,
        residual_history=False,
    )
    end_suffix_pr = evaluate_end_to_end(
        pr_suffix,
        pr_simulator,
        settings,
        config,
        torch.tensor(fitted_uav, dtype=torch.float32, device=pr_simulator.device),
        EI,
        Cb,
        output,
        "end_to_end_physics_residual_validation",
        lead_times,
        residual_history=True,
    )
    stage("end_to_end_validation", "complete")

    physical_parameters = {
        "schema": "decomposed_physical_parameters_v1",
        "model_version": "aerial_cable_attitude_coupled_v1",
        "methodology": METHODOLOGY,
        **uav_fit["fitted_parameters"],
        "EI": EI,
        "Cb": Cb,
        "provenance": {
            "uav": "fitted from FullState command to measured UAV motion",
            "EI_Cb": "fitted conditional on measured Motive UAV boundary",
        },
    }
    atomic_json(output / "physical_parameters.json", physical_parameters)
    verification = {
        "schema": "milestone3c_verification_v1",
        "uav_step_is_common_coupled_simulator_model": True,
        "measured_boundary_uses_production_root_boundary": True,
        "cable_measurement_reset_after_initialization": False,
        "conditional_validation_measured_boundary": True,
        "end_to_end_measured_boundary": False,
        "residual_state_after_initialization": "simulated_only",
        "same_residual_implementation_subsystem_and_end_to_end": True,
        "same_dder_implementation_subsystem_and_end_to_end": True,
        "protected_test_predictively_evaluated": False,
    }
    atomic_json(output / "verification.json", verification)
    provisional_summary: dict[str, object] = {
        "schema": "milestone3c_decomposed_result_v1",
        "created_utc": utc_now(),
        "fit_directory": str(output),
        "elapsed_s": time.perf_counter() - started,
        "uav_fit": uav_fit,
        "uav_physics_metrics": uav_metrics,
        "cable_fit": cable_fit,
        "conditional_cable_metrics": conditional_cable,
        "residual_coverage": training_coverage,
        "validation_residual_coverage": validation_coverage,
        "residual_fit": residual_fit,
        "uav_same_suffix_physics": same_suffix_physics,
        "uav_same_suffix_residual": same_suffix_residual,
        "end_to_end_full_physics": end_full,
        "end_to_end_suffix_physics": end_suffix_p,
        "end_to_end_suffix_residual": end_suffix_pr,
        "protected_test": {
            "take_id": "fig8vertical_002",
            "predictively_evaluated": False,
        },
    }
    atomic_json(output / "comparison_summary.json", provisional_summary)
    freeze = create_freeze(
        output,
        root,
        provisional_summary,
        output / "residual_weights.pt",
        output / "residual_normalization.json",
        destination=freeze_path,
    )
    provisional_summary["freeze"] = freeze
    provisional_summary["elapsed_s"] = time.perf_counter() - started
    render_report(output, provisional_summary, report_path)
    provisional_summary["report"] = str(report_path)
    atomic_json(output / "comparison_summary.json", provisional_summary)
    stage_status["completed_utc"] = utc_now()
    stage_status["elapsed_s"] = time.perf_counter() - started
    stage_status["status"] = "complete"
    stage_status["freeze"] = freeze
    atomic_json(output / "execution_status.json", stage_status)
    atomic_json(
        output_root / "latest.json",
        {
            "schema": "milestone3c_latest_pointer_v1",
            "fit_directory": str(output),
            "result": str(output / "comparison_summary.json"),
            "report": str(report_path),
            "freeze": str(freeze_path),
        },
    )
    return output
