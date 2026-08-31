"""Differentiable aerial-cable simulation foundation."""

from .parameters import (
    CableParameters,
    SimulatorParameters,
    SimulatorSettings,
    UAVResponseParameters,
)
from .simulator import CoupledSimulator
from .state import SimulatorState, SimulatorTrajectory, UAVTrajectory
from .observation import (
    OPTITRACK_CABLE_MARKER_NODE_INDICES,
    OptiTrackObservation,
    observe_simulator_state,
    observe_simulator_trajectory,
)
from .coupling import ClampedRootBoundary, PivotRootBoundary
from .uav import (
    FullStateCommand,
    FullStateCommandSequence,
    FullStateUAVModel,
    PrescribedRootModel,
)

__all__ = [
    "CableParameters",
    "ClampedRootBoundary",
    "CoupledSimulator",
    "FullStateCommand",
    "FullStateCommandSequence",
    "FullStateUAVModel",
    "OPTITRACK_CABLE_MARKER_NODE_INDICES",
    "OptiTrackObservation",
    "PivotRootBoundary",
    "PrescribedRootModel",
    "SimulatorParameters",
    "SimulatorSettings",
    "SimulatorState",
    "SimulatorTrajectory",
    "UAVResponseParameters",
    "UAVTrajectory",
    "observe_simulator_state",
    "observe_simulator_trajectory",
]
