"""Explicit pivot and centerline-clamped DDER root boundary models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch

from ..cable.config import CableConfiguration
from ..cable.dder import (
    CLAMPED_START_FREE_END,
    START_PINNED_FREE_END,
    DderModel,
    DderState,
    PinnedEndpointMask,
)
from ..cable.initialization import (
    clamped_hanging_cable_state,
    hanging_cable_state,
)
from ..uav.state import UAVState
from .attachment import attachment_state, rigid_attachment_state


@dataclass(frozen=True, slots=True)
class RootBoundaryState:
    prescribed_positions_m: torch.Tensor
    analytic_velocities_m_s: torch.Tensor
    root_tangent_world: torch.Tensor | None

    @property
    def attachment_position_m(self) -> torch.Tensor:
        return self.prescribed_positions_m[:, 0]

    @property
    def attachment_velocity_analytic_m_s(self) -> torch.Tensor:
        return self.analytic_velocities_m_s[:, 0]


class RootBoundary(ABC):
    """Maps a simulated UAV state to an explicit DDER boundary."""

    pinned_endpoints: PinnedEndpointMask

    @abstractmethod
    def evaluate(
        self,
        uav_state: UAVState,
        first_edge_length_m: torch.Tensor | float,
    ) -> RootBoundaryState:
        raise NotImplementedError

    @abstractmethod
    def initialize_cable(
        self,
        model: DderModel,
        uav_state: UAVState,
        configuration: CableConfiguration,
    ) -> DderState:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class PivotRootBoundary(RootBoundary):
    """Legacy position-only pivot: root prescribed, centerline tangent free."""

    attachment_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pinned_endpoints: PinnedEndpointMask = START_PINNED_FREE_END

    def __post_init__(self) -> None:
        if len(self.attachment_offset_body_m) != 3 or not all(
            math.isfinite(value) for value in self.attachment_offset_body_m
        ):
            raise ValueError("Pivot attachment offset must contain three finite values.")

    def evaluate(
        self,
        uav_state: UAVState,
        first_edge_length_m: torch.Tensor | float,
    ) -> RootBoundaryState:
        del first_edge_length_m
        attachment = attachment_state(
            uav_state,
            self.attachment_offset_body_m,
        )
        return RootBoundaryState(
            attachment.position_m[:, None],
            attachment.analytic_velocity_m_s[:, None],
            None,
        )

    def initialize_cable(
        self,
        model: DderModel,
        uav_state: UAVState,
        configuration: CableConfiguration,
    ) -> DderState:
        boundary = self.evaluate(uav_state, configuration.rest_lengths_m[0])
        return hanging_cable_state(
            model,
            boundary.attachment_position_m,
            configuration,
            boundary.attachment_velocity_analytic_m_s,
        )


@dataclass(frozen=True, slots=True)
class ClampedRootBoundary(RootBoundary):
    """Position+tangent clamp using the first physical DDER edge.

    This constrains the centerline only.  Material roll and torsion remain
    absent from the reduced isotropic cable model.
    """

    attachment_offset_body_m: tuple[float, float, float]
    attachment_tangent_body: tuple[float, float, float]
    pinned_endpoints: PinnedEndpointMask = CLAMPED_START_FREE_END

    def __post_init__(self) -> None:
        if len(self.attachment_offset_body_m) != 3 or not all(
            math.isfinite(value) for value in self.attachment_offset_body_m
        ):
            raise ValueError("Clamped attachment offset must contain three finite values.")
        if len(self.attachment_tangent_body) != 3 or not all(
            math.isfinite(value) for value in self.attachment_tangent_body
        ):
            raise ValueError("Clamped attachment tangent must contain three finite values.")
        norm = math.sqrt(sum(value * value for value in self.attachment_tangent_body))
        if norm <= 0.0:
            raise ValueError("Clamped attachment tangent must be nonzero.")

    def evaluate(
        self,
        uav_state: UAVState,
        first_edge_length_m: torch.Tensor | float,
    ) -> RootBoundaryState:
        rigid = rigid_attachment_state(
            uav_state,
            self.attachment_offset_body_m,
            self.attachment_tangent_body,
            first_edge_length_m,
        )
        return RootBoundaryState(
            torch.stack(
                (rigid.root.position_m, rigid.first_edge_position_m), dim=1
            ),
            torch.stack(
                (
                    rigid.root.analytic_velocity_m_s,
                    rigid.first_edge_analytic_velocity_m_s,
                ),
                dim=1,
            ),
            rigid.tangent_world,
        )

    def initialize_cable(
        self,
        model: DderModel,
        uav_state: UAVState,
        configuration: CableConfiguration,
    ) -> DderState:
        boundary = self.evaluate(uav_state, configuration.rest_lengths_m[0])
        return clamped_hanging_cable_state(
            model,
            boundary.prescribed_positions_m,
            configuration,
            boundary.analytic_velocities_m_s,
        )
