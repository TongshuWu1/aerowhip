"""Explicit physical model reduction for real-time cable MPC."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

from cable_twin.shared.dder import (
    DderModel,
    DderParameters,
    DderState,
    START_PINNED_FREE_END,
    momentum_project_lengths,
)

from .model import CableModelSnapshot


REDUCTION_SCHEMA = "mass_conserving_der_reduction_v2"
STATE_TRANSFER_PROJECTION_ITERATIONS = 32
MAXIMUM_CONTROLLER_SUBSTEPS = 32


def _project_vertex_masses(
    source_coordinates_m: np.ndarray,
    source_masses_kg: np.ndarray,
    target_coordinates_m: np.ndarray,
) -> np.ndarray:
    """Linearly deposit material point masses onto the reduced grid."""

    target = np.zeros(len(target_coordinates_m), dtype=np.float64)
    for coordinate, mass in zip(source_coordinates_m, source_masses_kg):
        upper = int(np.searchsorted(target_coordinates_m, coordinate))
        if upper <= 0:
            target[0] += mass
        elif upper >= len(target_coordinates_m):
            target[-1] += mass
        else:
            lower = upper - 1
            span = target_coordinates_m[upper] - target_coordinates_m[lower]
            fraction = (coordinate - target_coordinates_m[lower]) / span
            target[lower] += mass * (1.0 - fraction)
            target[upper] += mass * fraction
    return target


def reduce_cable_model(
    source: CableModelSnapshot,
    *,
    node_count: int = 7,
    substeps: int = 1,
    constraint_iterations: int = 4,
    bending_stiffness_scale: float = 1.0,
    bending_damping_scale: float = 1.0,
) -> CableModelSnapshot:
    """Build the coarse controller DER from an immutable full-order plant model."""

    if not 6 <= node_count <= source.node_count:
        raise ValueError("Reduced DER node count must be between 6 and the fitted count.")
    if substeps < 1 or constraint_iterations < 1:
        raise ValueError("Reduced DER solver counts must be positive.")
    if (
        not math.isfinite(bending_stiffness_scale)
        or bending_stiffness_scale <= 0.0
        or not math.isfinite(bending_damping_scale)
        or bending_damping_scale <= 0.0
    ):
        raise ValueError("Controller EI and Cb scales must be finite and positive.")
    length = source.cable_length_m
    source_coordinates = np.asarray(source.rod_material_coordinates_m, dtype=np.float64)
    source_masses = np.asarray(
        source.model.parameters.vertex_masses_kg,
        dtype=np.float64,
    )
    if node_count == source.node_count:
        # The full-resolution comparison must preserve the fitted material grid
        # exactly.  Re-meshing it uniformly would mix discretization error into
        # the resolution study.
        coordinates = source_coordinates.copy()
        masses = source_masses.copy()
        rest_lengths = np.asarray(
            source.model.parameters.rest_lengths_m, dtype=np.float64
        )
    else:
        coordinates = np.linspace(0.0, length, node_count, dtype=np.float64)
        masses = _project_vertex_masses(
            source_coordinates, source_masses, coordinates
        )
        rest_lengths = np.diff(coordinates)
    if np.any(masses <= 0.0):
        raise ValueError("Physical reduction produced a non-positive vertex mass.")
    if not np.isclose(np.sum(masses), np.sum(source_masses), rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Physical reduction did not conserve cable mass.")
    parameters = source.model.parameters
    controller_ei = source.bending_stiffness_n_m2 * bending_stiffness_scale
    controller_cb = source.bending_damping_n_m2_s * bending_damping_scale
    model = DderModel(
        DderParameters(
            node_count=node_count,
            cable_length_m=length,
            cable_mass_kg=float(np.sum(masses)),
            cable_diameter_m=parameters.cable_diameter_m,
            bending_stiffness_n_m2=controller_ei,
            bending_damping_n_m2_s=controller_cb,
            torsional_stiffness_n_m2=0.0,
            external_drag_s_inv=parameters.external_drag_s_inv,
            gravity_camera_m_s2=parameters.gravity_camera_m_s2,
            rest_lengths_m=tuple(float(value) for value in rest_lengths),
            vertex_masses_kg=tuple(float(value) for value in masses),
            substeps=substeps,
            constraint_iterations=constraint_iterations,
        )
    )
    identity = (
        f"{source.sha256}:{REDUCTION_SCHEMA}:nodes={node_count}:"
        f"substeps={substeps}:constraints={constraint_iterations}:"
        f"ei_scale={bending_stiffness_scale:.17g}:"
        f"cb_scale={bending_damping_scale:.17g}"
    )
    effective_sha = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    note = (
        f"{source.provenance_note}; controller reduction={REDUCTION_SCHEMA}, "
        f"{source.node_count}->{node_count} nodes, total mass conserved, "
        f"EI scale={bending_stiffness_scale:g}, Cb scale={bending_damping_scale:g}"
    )
    # Preserve the observation provenance of the source artifact.  A reduced
    # simulation grid is not a new set of physical OptiTrack markers: map each
    # original marker to its nearest reduced material node instead of falsely
    # claiming that every reduced node is a measured marker.  State transfer
    # itself continues to use material-coordinate interpolation below.
    source_marker_coordinates = source_coordinates[
        np.asarray(source.marker_node_indices, dtype=np.int64)
    ]
    reduced_marker_indices = tuple(
        int(np.argmin(np.abs(coordinates - marker_coordinate)))
        for marker_coordinate in source_marker_coordinates
    )
    return CableModelSnapshot(
        source_path=source.source_path,
        sha256=effective_sha,
        model=model,
        marker_node_indices=reduced_marker_indices,
        rod_material_coordinates_m=tuple(float(value) for value in coordinates),
        bending_stiffness_n_m2=controller_ei,
        bending_damping_n_m2_s=controller_cb,
        payload=source.payload,
        provisional=source.provisional,
        provenance_note=note,
    )


def stable_controller_model(
    source: CableModelSnapshot,
    *,
    simulation_dt_s: float,
    node_count: int = 7,
    constraint_iterations: int = 4,
    bending_stiffness_scale: float = 1.0,
    bending_damping_scale: float = 1.0,
) -> CableModelSnapshot:
    """Use the smallest explicit-solver substep count stable at the MPC dt.

    Node resolution changes the largest bending eigenfrequency.  Holding the
    solver at one substep would make a fine controller unstable and would turn
    the resolution benchmark into a numerical-integration comparison.  The
    selected count is deterministic and is included in the model identity.
    """

    if not math.isfinite(simulation_dt_s) or simulation_dt_s <= 0.0:
        raise ValueError("Controller simulation dt must be finite and positive.")
    for substeps in range(1, MAXIMUM_CONTROLLER_SUBSTEPS + 1):
        candidate = reduce_cable_model(
            source,
            node_count=node_count,
            substeps=substeps,
            constraint_iterations=constraint_iterations,
            bending_stiffness_scale=bending_stiffness_scale,
            bending_damping_scale=bending_damping_scale,
        )
        maximum_ei = candidate.model.maximum_stable_bending_stiffness(
            simulation_dt_s,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        if candidate.bending_stiffness_n_m2 <= maximum_ei:
            return candidate
    raise ValueError(
        "No stable controller integration was found within the explicit "
        f"substep limit ({MAXIMUM_CONTROLLER_SUBSTEPS})."
    )


def build_controller_and_truth_models(
    source: CableModelSnapshot,
    *,
    simulation_dt_s: float,
    node_count: int,
    truth_bending_stiffness_scale: float = 1.0,
    truth_bending_damping_scale: float = 1.0,
) -> tuple[CableModelSnapshot, CableModelSnapshot]:
    """Build a nominal fitted model and a parameter-mismatched truth model.

    The two returned models always use the same material grid, mass, geometry,
    constraint solver, and explicit substep count.  Only EI and Cb may differ.
    This keeps a truth-model experiment attributable to physical-parameter
    mismatch instead of quietly mixing in discretization or solver mismatch.
    """

    if (
        not math.isfinite(truth_bending_stiffness_scale)
        or truth_bending_stiffness_scale <= 0.0
        or not math.isfinite(truth_bending_damping_scale)
        or truth_bending_damping_scale <= 0.0
    ):
        raise ValueError("Truth EI and Cb scales must be finite and positive.")

    constraint_iterations = source.model.parameters.constraint_iterations
    if node_count == source.node_count:
        nominal_candidate = source
    else:
        nominal_candidate = stable_controller_model(
            source,
            simulation_dt_s=simulation_dt_s,
            node_count=node_count,
            constraint_iterations=constraint_iterations,
        )
    # The fixed nominal controller determines numerical resolution.  Never use
    # a hidden truth setting to change the controller's integration scheme.
    controller = nominal_candidate
    common_substeps = controller.model.parameters.substeps
    maximum_controller_ei = controller.model.maximum_stable_bending_stiffness(
        simulation_dt_s,
        pinned_endpoints=START_PINNED_FREE_END,
    )
    if controller.bending_stiffness_n_m2 > maximum_controller_ei:
        raise ValueError(
            "The fitted controller model is unstable at the selected physics "
            f"rate ({common_substeps} fixed substeps at dt={simulation_dt_s:g}s). "
            "Increase the physics rate."
        )

    if (
        truth_bending_stiffness_scale == 1.0
        and truth_bending_damping_scale == 1.0
    ):
        # Matched mode should be exactly identical, including model identity.
        truth = controller
    else:
        truth = reduce_cable_model(
            source,
            node_count=node_count,
            substeps=common_substeps,
            constraint_iterations=constraint_iterations,
            bending_stiffness_scale=truth_bending_stiffness_scale,
            bending_damping_scale=truth_bending_damping_scale,
        )
        maximum_truth_ei = truth.model.maximum_stable_bending_stiffness(
            simulation_dt_s,
            pinned_endpoints=START_PINNED_FREE_END,
        )
        if truth.bending_stiffness_n_m2 > maximum_truth_ei:
            raise ValueError(
                "Truth EI is unstable with the fitted controller's fixed solver "
                f"({common_substeps} substeps at dt={simulation_dt_s:g}s). "
                "Reduce the truth EI ratio or increase the physics rate."
            )
    return controller, truth


def transfer_dder_state(
    state: DderState,
    source: CableModelSnapshot,
    controller: CableModelSnapshot,
) -> DderState:
    """Map a full plant state onto the controller material grid.

    Positions and velocities are first interpolated at the controller material
    coordinates.  A mass-weighted projection then restores the controller's
    exact inextensible edge lengths while preserving the measured attachment.
    Velocity is projected onto the corresponding tangent constraint.  This is
    the deterministic observation adapter later replaced by OptiTrack/RGB-D.
    """

    positions = state.positions_m
    velocities = state.velocities_m_s
    if positions.ndim != 3 or positions.shape[1:] != (source.node_count, 3):
        raise ValueError("Plant cable state does not match the source model grid.")
    if velocities.shape != positions.shape:
        raise ValueError("Plant cable velocities must match its positions.")
    if state.endpoint_orientations is not None or state.endpoint_twist_rad is not None:
        raise ValueError("The one-attachment controller state must be torsion free.")
    if not math.isclose(
        source.cable_length_m,
        controller.cable_length_m,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Plant and controller cable lengths must be identical.")

    source_coordinates = np.asarray(
        source.rod_material_coordinates_m, dtype=np.float64
    )
    target_coordinates = np.asarray(
        controller.rod_material_coordinates_m, dtype=np.float64
    )
    if (
        len(source_coordinates) != source.node_count
        or len(target_coordinates) != controller.node_count
        or np.any(np.diff(source_coordinates) <= 0.0)
        or np.any(np.diff(target_coordinates) <= 0.0)
    ):
        raise ValueError("Cable material coordinates must be strictly increasing.")
    upper = np.searchsorted(source_coordinates, target_coordinates, side="left")
    upper = np.clip(upper, 1, source.node_count - 1)
    lower = upper - 1
    exact_start = target_coordinates <= source_coordinates[0]
    exact_end = target_coordinates >= source_coordinates[-1]
    lower[exact_start] = 0
    upper[exact_start] = 0
    lower[exact_end] = source.node_count - 1
    upper[exact_end] = source.node_count - 1
    denominator = source_coordinates[upper] - source_coordinates[lower]
    fraction = np.divide(
        target_coordinates - source_coordinates[lower],
        denominator,
        out=np.zeros_like(target_coordinates),
        where=denominator > 0.0,
    )
    lower_index = torch.as_tensor(lower, dtype=torch.long, device=positions.device)
    upper_index = torch.as_tensor(upper, dtype=torch.long, device=positions.device)
    blend = torch.as_tensor(
        fraction, dtype=positions.dtype, device=positions.device
    )[None, :, None]
    sampled_positions = (
        positions[:, lower_index] * (1.0 - blend)
        + positions[:, upper_index] * blend
    )
    sampled_velocities = (
        velocities[:, lower_index] * (1.0 - blend)
        + velocities[:, upper_index] * blend
    )

    rest_lengths = torch.as_tensor(
        controller.model.parameters.rest_lengths_m,
        dtype=positions.dtype,
        device=positions.device,
    )
    masses = torch.as_tensor(
        controller.model.parameters.vertex_masses_kg,
        dtype=positions.dtype,
        device=positions.device,
    )
    projected_positions = momentum_project_lengths(
        sampled_positions,
        rest_lengths,
        masses,
        sampled_positions[:, :1],
        iterations=STATE_TRANSFER_PROJECTION_ITERATIONS,
        pinned_endpoints=START_PINNED_FREE_END,
    )
    projected_velocities = controller.model.project_velocities(
        projected_positions,
        sampled_velocities,
        sampled_velocities[:, :1],
        pinned_endpoints=START_PINNED_FREE_END,
    )
    return DderState(projected_positions, projected_velocities)
