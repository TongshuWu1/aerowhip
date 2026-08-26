"""Controller-independent definition of the cable strike task.

This module intentionally depends only on NumPy.  The supported receding-
horizon MPPI application must not import the legacy CasADi/IPOPT optimizer
merely to describe its target and safety limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True, slots=True)
class MpcProblem:
    """Immutable target, impact, and geometric-safety contract."""

    target_position_m: tuple[float, float, float]
    impact_direction: tuple[float, float, float]
    minimum_impact_speed_m_s: float = 1.5
    drone_keepout_radius_m: float = 0.30
    maximum_drone_excursion_m: float = 0.15
    minimum_forward_stroke_m: float = 0.0
    minimum_recoil_stroke_m: float = 0.0
    drone_workspace_center_m: tuple[float, float, float] | None = None
    maximum_tip_error_m: float = 0.05
    planning_tip_error_margin_m: float = 0.0
    maximum_impact_angle_deg: float = 20.0
    planning_impact_angle_margin_deg: float = 0.0

    def __post_init__(self) -> None:
        target = np.asarray(self.target_position_m, dtype=np.float64)
        direction = np.asarray(self.impact_direction, dtype=np.float64)
        workspace_center = (
            None
            if self.drone_workspace_center_m is None
            else np.asarray(self.drone_workspace_center_m, dtype=np.float64)
        )
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("Target position must be a finite XYZ vector.")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("Impact direction must be a finite XYZ vector.")
        if workspace_center is not None and (
            workspace_center.shape != (3,) or not np.all(np.isfinite(workspace_center))
        ):
            raise ValueError("Drone workspace center must be a finite XYZ vector.")
        norm = float(np.linalg.norm(direction))
        if norm <= 1.0e-9:
            raise ValueError("Impact direction cannot be zero.")
        if (
            not math.isfinite(self.minimum_impact_speed_m_s)
            or self.minimum_impact_speed_m_s <= 0.0
            or not math.isfinite(self.drone_keepout_radius_m)
            or self.drone_keepout_radius_m <= 0.0
            or not math.isfinite(self.maximum_drone_excursion_m)
            or self.maximum_drone_excursion_m <= 0.0
            or not math.isfinite(self.minimum_forward_stroke_m)
            or not 0.0 <= self.minimum_forward_stroke_m
            <= self.maximum_drone_excursion_m
            or not math.isfinite(self.minimum_recoil_stroke_m)
            or not 0.0 <= self.minimum_recoil_stroke_m
            <= 2.0 * self.maximum_drone_excursion_m
            or not math.isfinite(self.maximum_tip_error_m)
            or self.maximum_tip_error_m <= 0.0
            or not math.isfinite(self.planning_tip_error_margin_m)
            or self.planning_tip_error_margin_m < 0.0
            or self.planning_tip_error_margin_m >= self.maximum_tip_error_m
            or not math.isfinite(self.maximum_impact_angle_deg)
            or not 0.0 < self.maximum_impact_angle_deg < 90.0
            or not math.isfinite(self.planning_impact_angle_margin_deg)
            or self.planning_impact_angle_margin_deg < 0.0
            or self.planning_impact_angle_margin_deg
            >= self.maximum_impact_angle_deg
        ):
            raise ValueError(
                "Impact speed, hit tolerance, keepout radius, drone excursion, and "
                "forward/recoil stroke thresholds must be non-negative; the planning "
                "margin must be non-negative and smaller than the hit tolerance; the "
                "impact angle must be between 0 and 90 degrees and its planning margin "
                "must leave a positive cone."
            )
        object.__setattr__(self, "target_position_m", tuple(float(v) for v in target))
        object.__setattr__(
            self,
            "impact_direction",
            tuple(float(v) for v in direction / norm),
        )
        if workspace_center is not None:
            object.__setattr__(
                self,
                "drone_workspace_center_m",
                tuple(float(v) for v in workspace_center),
            )

    @property
    def planning_tip_error_limit_m(self) -> float:
        """Tightened internal constraint; physical success uses the full tolerance."""

        return self.maximum_tip_error_m - self.planning_tip_error_margin_m

    @property
    def planning_impact_angle_limit_deg(self) -> float:
        """Tightened internal cone; physical success uses the full angle."""

        return self.maximum_impact_angle_deg - self.planning_impact_angle_margin_deg


__all__ = ["MpcProblem"]
