"""Isaac Lab runtime for the floating 6-DoF drone--cable plant.

This module must be imported only after :class:`isaaclab.app.AppLauncher`
starts Kit.  :mod:`isaac_whip.launcher` enforces that import order.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import matrix_from_quat, quat_apply
from isaacsim.core.cloner import GridCloner

from .asset import author_drone_cable, export_drone_cable_usd
from .config import build_cable_spec, load_drone_config


@dataclass(slots=True)
class RuntimeOptions:
    """Options resolved by the launcher before entering the Kit runtime."""

    model_path: Path
    drone_config_path: Path
    link_count: int = 20
    num_envs: int = 1
    env_spacing_m: float = 2.0
    physics_dt_s: float = 0.002
    render_hz: float = 60.0
    control_hz: float = 50.0
    duration_s: float = 8.0
    mode: str = "hover"
    device: str = "cuda:0"
    headless: bool = False
    output_json: Path | None = None
    export_usd: Path | None = None


def _desired_translation(
    mode: str,
    time_s: float,
    origins: torch.Tensor,
    initial_position: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame position and velocity references for every clone."""

    desired_position = origins + initial_position.unsqueeze(0)
    desired_velocity = torch.zeros_like(desired_position)
    if mode == "excite":
        # A smooth, bounded 3-D root excitation for visually checking the
        # reaction-coupled cable.  It is a diagnostic input, not a whip policy.
        omega_x = 2.0 * math.pi * 0.28
        omega_y = 2.0 * math.pi * 0.21
        omega_z = 2.0 * math.pi * 0.33
        desired_position[:, 0] += 0.18 * math.sin(omega_x * time_s)
        desired_position[:, 1] += 0.10 * math.sin(omega_y * time_s + 0.35)
        desired_position[:, 2] += 0.035 * math.sin(omega_z * time_s)
        desired_velocity[:, 0] = 0.18 * omega_x * math.cos(omega_x * time_s)
        desired_velocity[:, 1] = 0.10 * omega_y * math.cos(omega_y * time_s + 0.35)
        desired_velocity[:, 2] = 0.035 * omega_z * math.cos(
            omega_z * time_s
        )
    return desired_position, desired_velocity


def _geometric_wrench(
    robot: Articulation,
    *,
    desired_position_w: torch.Tensor,
    desired_velocity_w: torch.Tensor,
    total_mass_kg: float,
    drone_cfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute bounded collective thrust and body torque.

    Translation is controlled through collective thrust along the current body
    +Z axis.  A geometric attitude loop tilts that axis toward the requested
    world force.  Thus the plant remains genuinely 6-DoF; horizontal force is
    never injected directly in world coordinates.
    """

    device = robot.device
    dtype = robot.data.root_pos_w.dtype
    kp_position = torch.tensor(drone_cfg.position_kp_n_m, device=device, dtype=dtype)
    kd_position = torch.tensor(drone_cfg.position_kd_n_s_m, device=device, dtype=dtype)
    kp_attitude = torch.tensor(drone_cfg.attitude_kp_n_m_rad, device=device, dtype=dtype)
    kd_attitude = torch.tensor(drone_cfg.attitude_kd_n_m_s_rad, device=device, dtype=dtype)
    torque_limit = torch.tensor(drone_cfg.maximum_body_torque_n_m, device=device, dtype=dtype)

    position_error = desired_position_w - robot.data.root_pos_w
    velocity_error = desired_velocity_w - robot.data.root_lin_vel_w
    gravity_force = torch.zeros_like(position_error)
    gravity_force[:, 2] = total_mass_kg * 9.81
    requested_force_w = gravity_force + kp_position * position_error + kd_position * velocity_error

    force_norm = torch.linalg.vector_norm(requested_force_w, dim=-1, keepdim=True).clamp_min(1.0e-8)
    desired_b3_w = requested_force_w / force_norm
    desired_b1_heading = torch.zeros_like(desired_b3_w)
    desired_b1_heading[:, 0] = 1.0
    desired_b2_w = torch.linalg.cross(desired_b3_w, desired_b1_heading, dim=-1)
    poor_heading = torch.linalg.vector_norm(desired_b2_w, dim=-1, keepdim=True) < 1.0e-5
    fallback_heading = torch.zeros_like(desired_b3_w)
    fallback_heading[:, 1] = 1.0
    fallback_b2 = torch.linalg.cross(desired_b3_w, fallback_heading, dim=-1)
    desired_b2_w = torch.where(poor_heading, fallback_b2, desired_b2_w)
    desired_b2_w = desired_b2_w / torch.linalg.vector_norm(
        desired_b2_w, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    desired_b1_w = torch.linalg.cross(desired_b2_w, desired_b3_w, dim=-1)
    desired_rotation_wb = torch.stack((desired_b1_w, desired_b2_w, desired_b3_w), dim=-1)

    rotation_wb = matrix_from_quat(robot.data.root_quat_w)
    rotation_error_matrix = (
        desired_rotation_wb.transpose(-1, -2) @ rotation_wb
        - rotation_wb.transpose(-1, -2) @ desired_rotation_wb
    )
    attitude_error = 0.5 * torch.stack(
        (
            rotation_error_matrix[:, 2, 1],
            rotation_error_matrix[:, 0, 2],
            rotation_error_matrix[:, 1, 0],
        ),
        dim=-1,
    )
    body_torque = -kp_attitude * attitude_error - kd_attitude * robot.data.root_ang_vel_b
    body_torque = torch.maximum(torch.minimum(body_torque, torque_limit), -torque_limit)

    current_b3_w = rotation_wb[:, :, 2]
    collective = torch.sum(requested_force_w * current_b3_w, dim=-1).clamp(
        0.0, drone_cfg.maximum_collective_thrust_n
    )
    body_force = torch.zeros_like(body_torque)
    body_force[:, 2] = collective
    return body_force.unsqueeze(1), body_torque.unsqueeze(1)


def _material_stations(
    robot: Articulation,
    link_body_ids: list[int],
    rest_lengths: tuple[float, ...],
) -> torch.Tensor:
    """Recover the link-chain material stations from rigid-body poses."""

    positions = robot.data.body_pos_w[:, link_body_ids]
    orientations = robot.data.body_quat_w[:, link_body_ids]
    lengths = torch.tensor(rest_lengths, device=robot.device, dtype=positions.dtype)
    local_half = torch.zeros(robot.num_instances, len(link_body_ids), 3, device=robot.device)
    local_half[:, :, 0] = 0.5 * lengths.unsqueeze(0)
    distal = positions + quat_apply(orientations.reshape(-1, 4), local_half.reshape(-1, 3)).reshape_as(
        positions
    )
    proximal_first = positions[:, 0] + quat_apply(orientations[:, 0], -local_half[:, 0])
    return torch.cat((proximal_first.unsqueeze(1), distal), dim=1)


def _constraint_gaps(
    robot: Articulation,
    *,
    drone_body_id: int,
    link_body_ids: list[int],
    rest_lengths: tuple[float, ...],
    attachment_offset_body_m: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = robot.data.body_pos_w[:, link_body_ids]
    orientations = robot.data.body_quat_w[:, link_body_ids]
    lengths = torch.tensor(rest_lengths, device=robot.device, dtype=positions.dtype)
    half = torch.zeros(robot.num_instances, len(link_body_ids), 3, device=robot.device)
    half[:, :, 0] = 0.5 * lengths.unsqueeze(0)
    proximal = positions + quat_apply(orientations.reshape(-1, 4), (-half).reshape(-1, 3)).reshape_as(
        positions
    )
    distal = positions + quat_apply(orientations.reshape(-1, 4), half.reshape(-1, 3)).reshape_as(
        positions
    )
    internal = torch.linalg.vector_norm(distal[:, :-1] - proximal[:, 1:], dim=-1)

    drone_position = robot.data.body_pos_w[:, drone_body_id]
    drone_orientation = robot.data.body_quat_w[:, drone_body_id]
    offset = torch.tensor(
        attachment_offset_body_m, device=robot.device, dtype=drone_position.dtype
    ).expand_as(drone_position)
    attachment = drone_position + quat_apply(drone_orientation, offset)
    root = torch.linalg.vector_norm(attachment - proximal[:, 0], dim=-1)
    return root, internal


def _authored_parameter_check(
    stage, authored: dict[str, object], cable, drone_cfg
) -> dict[str, object]:
    """Check actual USD gains, topology, and boundary-condition schemas."""

    joint_paths = tuple(str(value) for value in authored["joint_paths"])
    link_paths = tuple(str(value) for value in authored["link_paths"])
    drone_path = str(authored["drone_path"])
    max_stiffness_error = 0.0
    max_damping_error = 0.0
    twist_drive_count = 0
    twist_limit_count = 0
    locked_translation_axis_count = 0
    bending_drive_axis_count = 0
    unexpected_bending_limit_count = 0
    nonzero_authored_friction_count = 0
    drive_configuration_ok = True
    internal_body_pairs_ok = True
    maximum_joint_frame_error = 0.0

    def update_vector_error(actual, expected: tuple[float, ...]) -> None:
        nonlocal maximum_joint_frame_error
        maximum_joint_frame_error = max(
            maximum_joint_frame_error,
            *(abs(float(value) - target) for value, target in zip(actual, expected, strict=True)),
        )

    def quaternion_components(value) -> tuple[float, float, float, float]:
        imaginary = value.GetImaginary()
        return (float(value.GetReal()), *(float(imaginary[index]) for index in range(3)))

    for index, path in enumerate(joint_paths[1:]):
        prim = stage.GetPrimAtPath(path)
        schemas = tuple(str(value) for value in prim.GetAppliedSchemas())
        twist_drive_count += sum(value.endswith("DriveAPI:rotX") for value in schemas)
        twist_limit_count += sum(value.endswith("LimitAPI:rotX") for value in schemas)
        for axis in (UsdPhysics.Tokens.transX, UsdPhysics.Tokens.transY, UsdPhysics.Tokens.transZ):
            if f"PhysicsLimitAPI:{axis}" not in schemas:
                continue
            limit = UsdPhysics.LimitAPI.Get(prim, axis)
            if float(limit.GetLowAttr().Get()) > float(limit.GetHighAttr().Get()):
                locked_translation_axis_count += 1
        bending_drive_axis_count += sum(
            f"PhysicsDriveAPI:{axis}" in schemas
            for axis in (UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ)
        )
        unexpected_bending_limit_count += sum(
            f"PhysicsLimitAPI:{axis}" in schemas
            for axis in (UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ)
        )
        for attribute in prim.GetAuthoredAttributes():
            if "friction" not in str(attribute.GetName()).lower():
                continue
            value = attribute.Get()
            if isinstance(value, (int, float)) and abs(float(value)) > 0.0:
                nonzero_authored_friction_count += 1
        joint = UsdPhysics.Joint(prim)
        internal_body_pairs_ok = internal_body_pairs_ok and tuple(
            str(value) for value in joint.GetBody0Rel().GetTargets()
        ) == (link_paths[index],)
        internal_body_pairs_ok = internal_body_pairs_ok and tuple(
            str(value) for value in joint.GetBody1Rel().GetTargets()
        ) == (link_paths[index + 1],)
        update_vector_error(
            joint.GetLocalPos0Attr().Get(), (0.5 * cable.rest_lengths_m[index], 0.0, 0.0)
        )
        update_vector_error(
            joint.GetLocalPos1Attr().Get(), (-0.5 * cable.rest_lengths_m[index + 1], 0.0, 0.0)
        )
        update_vector_error(quaternion_components(joint.GetLocalRot0Attr().Get()), (1.0, 0.0, 0.0, 0.0))
        update_vector_error(quaternion_components(joint.GetLocalRot1Attr().Get()), (1.0, 0.0, 0.0, 0.0))
        for axis in (UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
            drive = UsdPhysics.DriveAPI.Get(prim, axis)
            stiffness = float(drive.GetStiffnessAttr().Get())
            damping = float(drive.GetDampingAttr().Get())
            drive_configuration_ok = drive_configuration_ok and drive.GetTypeAttr().Get() == UsdPhysics.Tokens.force
            drive_configuration_ok = drive_configuration_ok and math.isclose(
                float(drive.GetTargetPositionAttr().Get()), 0.0, abs_tol=1.0e-12
            )
            drive_configuration_ok = drive_configuration_ok and math.isclose(
                float(drive.GetTargetVelocityAttr().Get()), 0.0, abs_tol=1.0e-12
            )
            max_stiffness_error = max(
                max_stiffness_error,
                abs(stiffness - cable.usd_hinge_stiffness_n_m_deg[index]),
            )
            max_damping_error = max(
                max_damping_error,
                abs(damping - cable.usd_hinge_damping_n_m_s_deg[index]),
            )

    root_prim = stage.GetPrimAtPath(joint_paths[0])
    root_joint = UsdPhysics.FixedJoint(root_prim)
    root_fixed_ok = root_prim.IsA(UsdPhysics.FixedJoint)
    root_fixed_ok = root_fixed_ok and tuple(
        str(value) for value in root_joint.GetBody0Rel().GetTargets()
    ) == (drone_path,)
    root_fixed_ok = root_fixed_ok and tuple(
        str(value) for value in root_joint.GetBody1Rel().GetTargets()
    ) == (link_paths[0],)
    update_vector_error(root_joint.GetLocalPos0Attr().Get(), drone_cfg.attachment_offset_body_m)
    update_vector_error(
        root_joint.GetLocalPos1Attr().Get(), (-0.5 * cable.rest_lengths_m[0], 0.0, 0.0)
    )
    half_sqrt_two = math.sqrt(0.5)
    update_vector_error(
        quaternion_components(root_joint.GetLocalRot0Attr().Get()),
        (half_sqrt_two, 0.0, half_sqrt_two, 0.0),
    )
    update_vector_error(
        quaternion_components(root_joint.GetLocalRot1Attr().Get()), (1.0, 0.0, 0.0, 0.0)
    )

    distal_joint_count = 0
    all_joint_bodies_defined = True
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = tuple(
            str(value)
            for value in (*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets())
        )
        all_joint_bodies_defined = all_joint_bodies_defined and len(targets) == 2
        distal_joint_count += targets.count(link_paths[-1])

    expected_internal = cable.link_count - 1
    schema_pass = (
        root_fixed_ok
        and internal_body_pairs_ok
        and all_joint_bodies_defined
        and locked_translation_axis_count == 3 * expected_internal
        and bending_drive_axis_count == 2 * expected_internal
        and twist_drive_count == 0
        and twist_limit_count == 0
        and unexpected_bending_limit_count == 0
        and nonzero_authored_friction_count == 0
        and drive_configuration_ok
        and maximum_joint_frame_error < 1.0e-6
        and distal_joint_count == 1
    )
    return {
        "maximum_authored_stiffness_error_n_m_deg": max_stiffness_error,
        "maximum_authored_damping_error_n_m_s_deg": max_damping_error,
        "root_fixed_joint_pass": root_fixed_ok,
        "internal_body_pairs_pass": internal_body_pairs_ok,
        "all_joint_bodies_defined": all_joint_bodies_defined,
        "locked_translation_axis_count": locked_translation_axis_count,
        "expected_locked_translation_axis_count": 3 * expected_internal,
        "bending_drive_axis_count": bending_drive_axis_count,
        "expected_bending_drive_axis_count": 2 * expected_internal,
        "drive_configuration_pass": drive_configuration_ok,
        "twist_drive_count": twist_drive_count,
        "twist_limit_count": twist_limit_count,
        "unexpected_bending_limit_count": unexpected_bending_limit_count,
        "nonzero_authored_friction_count": nonzero_authored_friction_count,
        "maximum_joint_frame_error": maximum_joint_frame_error,
        "distal_joint_count": distal_joint_count,
        "free_tip_pass": distal_joint_count == 1,
        "schema_pass": schema_pass,
        # USD stores the drive attributes as single-precision values.
        "pass": max_stiffness_error < 1.0e-9
        and max_damping_error < 1.0e-9
        and schema_pass,
    }


def _resolved_body_property_check(
    robot: Articulation,
    *,
    drone_body_id: int,
    link_body_ids: list[int],
    drone_cfg,
    cable,
) -> dict[str, object]:
    """Compare PhysX-resolved mass, COM, and inertia against the authored plant."""

    ordered_body_ids = [drone_body_id, *link_body_ids]
    resolved_mass = robot.data.default_mass[:, ordered_body_ids]
    resolved_com = robot.data.body_com_pos_b[:, ordered_body_ids]
    resolved_inertia = robot.data.default_inertia[:, ordered_body_ids]
    expected_mass = torch.tensor(
        [drone_cfg.mass_kg, *cable.link_masses_kg],
        device=resolved_mass.device,
        dtype=resolved_mass.dtype,
    )
    expected_com = torch.tensor(
        [(0.0, 0.0, 0.0), *((value, 0.0, 0.0) for value in cable.link_com_x_m)],
        device=resolved_com.device,
        dtype=resolved_com.dtype,
    )
    expected_diagonal_inertia = torch.tensor(
        [drone_cfg.diagonal_inertia_kg_m2, *cable.link_diagonal_inertia_kg_m2],
        device=resolved_inertia.device,
        dtype=resolved_inertia.dtype,
    )
    resolved_diagonal = resolved_inertia[:, :, [0, 4, 8]]
    resolved_off_diagonal = resolved_inertia[:, :, [1, 2, 3, 5, 6, 7]]
    maximum_mass_error = float(torch.max(torch.abs(resolved_mass - expected_mass)).item())
    maximum_com_error = float(torch.max(torch.abs(resolved_com - expected_com)).item())
    maximum_diagonal_inertia_error = float(
        torch.max(torch.abs(resolved_diagonal - expected_diagonal_inertia)).item()
    )
    maximum_off_diagonal_inertia = float(torch.max(torch.abs(resolved_off_diagonal)).item())
    passed = (
        maximum_mass_error < 1.0e-7
        and maximum_com_error < 1.0e-6
        and maximum_diagonal_inertia_error < 1.0e-9
        and maximum_off_diagonal_inertia < 1.0e-9
    )
    return {
        "maximum_mass_error_kg": maximum_mass_error,
        "maximum_com_error_m": maximum_com_error,
        "maximum_diagonal_inertia_error_kg_m2": maximum_diagonal_inertia_error,
        "maximum_off_diagonal_inertia_kg_m2": maximum_off_diagonal_inertia,
        "pass": passed,
    }


def run(options: RuntimeOptions, simulation_app) -> dict[str, object]:
    """Build, run, and summarize one or many cloned plants."""

    if options.mode not in {"hover", "excite", "free"}:
        raise ValueError("mode must be one of: hover, excite, free")
    if options.num_envs < 1:
        raise ValueError("num_envs must be positive")
    if (
        options.physics_dt_s <= 0.0
        or options.control_hz <= 0.0
        or options.render_hz <= 0.0
    ):
        raise ValueError("physics_dt_s, control_hz, and render_hz must be positive")
    if options.duration_s < 0.0:
        raise ValueError("duration_s must be nonnegative; use zero for continuous")
    if options.env_spacing_m <= 0.0:
        raise ValueError("env_spacing_m must be positive")

    drone_cfg = load_drone_config(options.drone_config_path)
    _snapshot, cable = build_cable_spec(options.model_path, link_count=options.link_count)
    if options.export_usd is not None:
        options.export_usd.parent.mkdir(parents=True, exist_ok=True)
        export_drone_cable_usd(
            str(options.export_usd), drone_cfg=drone_cfg, cable=cable
        )

    render_decimation = max(1, round(1.0 / (options.render_hz * options.physics_dt_s)))
    physics_cfg = sim_utils.PhysxCfg(
        solver_type=1,
        enable_external_forces_every_iteration=True,
        min_position_iteration_count=1,
        max_position_iteration_count=255,
        min_velocity_iteration_count=0,
        max_velocity_iteration_count=255,
    )
    simulation_cfg = sim_utils.SimulationCfg(
        dt=options.physics_dt_s,
        # Keep Kit's own interval at one physics step.  We decimate rendering
        # explicitly below; otherwise the full GUI experience advances
        # ``render_interval`` physics substeps per Python iteration while the
        # controller and experiment clock advance by only one.
        render_interval=1,
        gravity=(0.0, 0.0, -9.81),
        device=options.device,
        use_fabric=True,
        physx=physics_cfg,
    )
    sim = SimulationContext(simulation_cfg)
    stage = sim_utils.get_current_stage()
    stage.DefinePrim("/World/envs/env_0", "Xform")
    cloner = GridCloner(spacing=options.env_spacing_m, stage=stage)
    cloner.define_base_env("/World/envs")

    authored = author_drone_cable(
        stage,
        "/World/envs/env_0/DroneCable",
        drone_cfg=drone_cfg,
        cable=cable,
    )
    authored_check = _authored_parameter_check(stage, authored, cable, drone_cfg)
    env_paths = cloner.generate_paths("/World/envs/env", num_paths=options.num_envs)
    env_positions = cloner.clone(
        source_prim_path="/World/envs/env_0",
        prim_paths=env_paths,
        replicate_physics=True,
        copy_from_source=False,
        enable_env_ids=options.device != "cpu",
    )

    ground_cfg = sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    ground_cfg.func("/World/Ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.85, 0.88, 0.92))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/DroneCable/Drone",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=drone_cfg.initial_position_m,
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators={},
    )
    robot = Articulation(robot_cfg)

    grid_span = max(2.0, options.env_spacing_m * math.sqrt(options.num_envs))
    sim.set_camera_view(
        eye=[0.72 * grid_span, 0.72 * grid_span, 0.58 * grid_span + 1.5],
        target=[0.0, 0.0, 0.65],
    )
    sim.reset()
    # Isaac Sim's full GUI experience may advance the timeline while its UI and
    # RTX extensions finish starting.  Pause first, then impose the experiment
    # initial condition explicitly so GUI and headless runs begin identically.
    sim.pause()
    robot.update(0.0)
    print("[isaac_whip] articulation initialized; resolving bodies", flush=True)

    drone_body_ids, _ = robot.find_bodies("Drone")
    link_names = [f"Link_{index:02d}" for index in range(cable.link_count)]
    link_body_ids, resolved_link_names = robot.find_bodies(link_names, preserve_order=True)
    if len(drone_body_ids) != 1:
        raise RuntimeError(f"Expected one drone body, found {drone_body_ids}.")
    if resolved_link_names != link_names:
        raise RuntimeError(
            f"Cable body order mismatch: expected {link_names}, received {resolved_link_names}."
        )
    drone_body_id = drone_body_ids[0]

    expected_bodies = cable.link_count + 1
    expected_dofs = 3 * (cable.link_count - 1)
    topology_ok = robot.num_instances == options.num_envs and robot.num_bodies == expected_bodies
    topology_ok = topology_ok and robot.num_joints == expected_dofs
    resolved_property_check = _resolved_body_property_check(
        robot,
        drone_body_id=drone_body_id,
        link_body_ids=link_body_ids,
        drone_cfg=drone_cfg,
        cable=cable,
    )

    origins = torch.tensor(env_positions, device=robot.device, dtype=torch.float32)
    initial_position = torch.tensor(
        drone_cfg.initial_position_m, device=robot.device, dtype=torch.float32
    )
    initial_root_pose = torch.zeros(
        options.num_envs, 7, device=robot.device, dtype=torch.float32
    )
    initial_root_pose[:, :3] = origins + initial_position
    initial_root_pose[:, 3] = 1.0  # identity quaternion in Isaac Lab's wxyz convention
    initial_root_velocity = torch.zeros(
        options.num_envs, 6, device=robot.device, dtype=torch.float32
    )
    initial_joint_position = torch.zeros(
        options.num_envs, robot.num_joints, device=robot.device, dtype=torch.float32
    )
    initial_joint_velocity = torch.zeros_like(initial_joint_position)
    robot.write_root_pose_to_sim(initial_root_pose)
    robot.write_root_velocity_to_sim(initial_root_velocity)
    robot.write_joint_state_to_sim(initial_joint_position, initial_joint_velocity)
    robot.reset()
    robot.update(0.0)
    if not options.headless:
        # Give the renderer a few paused frames to present the reset state.  No
        # physics time elapses here.
        for _ in range(3):
            simulation_app.update()
        robot.update(0.0)

    total_mass = drone_cfg.mass_kg + cable.total_mass_kg
    control_decimation = max(1, round(1.0 / (options.control_hz * options.physics_dt_s)))
    achieved_control_hz = 1.0 / (control_decimation * options.physics_dt_s)
    achieved_render_hz = 1.0 / (render_decimation * options.physics_dt_s)

    body_ids_tensor = torch.tensor([drone_body_id], device=robot.device, dtype=torch.int32)
    root_gap_max = 0.0
    internal_gap_max = 0.0
    root_error_squared = 0.0
    root_error_samples = 0
    finite = True
    initial_stations = _material_stations(robot, link_body_ids, cable.rest_lengths_m).clone()
    initial_tip = initial_stations[:, -1].clone()
    previous_tip = initial_tip.clone()
    maximum_tip_displacement = torch.zeros(options.num_envs, device=robot.device)
    tip_path_length = torch.zeros(options.num_envs, device=robot.device)
    initial_tip_env_local = initial_tip - origins
    tip_min_env_local = initial_tip_env_local.clone()
    tip_max_env_local = initial_tip_env_local.clone()
    trace: list[dict[str, object]] = []
    steps = 0
    simulated_time = 0.0
    wall_start = time.perf_counter()
    maximum_steps = None
    if options.duration_s > 0.0:
        maximum_steps = max(1, math.ceil(options.duration_s / options.physics_dt_s))
    print(
        f"[isaac_whip] duration={options.duration_s:g}s dt={options.physics_dt_s:g}s "
        f"maximum_steps={maximum_steps}",
        flush=True,
    )

    # Preload the first control wrench while the timeline is still paused.
    # In the GUI, play() may render several extension-startup frames before it
    # returns; without this preload the plant can free-fall before iteration 0.
    initial_desired_position, initial_desired_velocity = _desired_translation(
        options.mode, 0.0, origins, initial_position
    )
    if options.mode == "free":
        initial_forces = torch.zeros(options.num_envs, 1, 3, device=robot.device)
        initial_torques = torch.zeros_like(initial_forces)
    else:
        initial_forces, initial_torques = _geometric_wrench(
            robot,
            desired_position_w=initial_desired_position,
            desired_velocity_w=initial_desired_velocity,
            total_mass_kg=total_mass,
            drone_cfg=drone_cfg,
        )
    robot.permanent_wrench_composer.set_forces_and_torques(
        forces=initial_forces,
        torques=initial_torques,
        body_ids=body_ids_tensor,
        is_global=False,
    )
    robot.write_data_to_sim()
    sim.play()
    while simulation_app.is_running() and (maximum_steps is None or steps < maximum_steps):
        desired_position, desired_velocity = _desired_translation(
            options.mode, simulated_time, origins, initial_position
        )
        if steps % control_decimation == 0:
            if options.mode == "free":
                forces = torch.zeros(options.num_envs, 1, 3, device=robot.device)
                torques = torch.zeros_like(forces)
            else:
                forces, torques = _geometric_wrench(
                    robot,
                    desired_position_w=desired_position,
                    desired_velocity_w=desired_velocity,
                    total_mass_kg=total_mass,
                    drone_cfg=drone_cfg,
                )
            robot.permanent_wrench_composer.set_forces_and_torques(
                forces=forces,
                torques=torques,
                body_ids=body_ids_tensor,
                is_global=False,
            )

        robot.write_data_to_sim()
        render_this_step = not options.headless and (steps + 1) % render_decimation == 0
        sim.step(render=render_this_step)
        robot.update(options.physics_dt_s)
        simulated_time += options.physics_dt_s
        steps += 1

        stations = _material_stations(robot, link_body_ids, cable.rest_lengths_m)
        tip = stations[:, -1]
        tip_env_local = tip - origins
        tip_min_env_local = torch.minimum(tip_min_env_local, tip_env_local)
        tip_max_env_local = torch.maximum(tip_max_env_local, tip_env_local)
        maximum_tip_displacement = torch.maximum(
            maximum_tip_displacement,
            torch.linalg.vector_norm(tip - initial_tip, dim=-1),
        )
        tip_path_length += torch.linalg.vector_norm(tip - previous_tip, dim=-1)
        previous_tip = tip.clone()
        root_gap, internal_gap = _constraint_gaps(
            robot,
            drone_body_id=drone_body_id,
            link_body_ids=link_body_ids,
            rest_lengths=cable.rest_lengths_m,
            attachment_offset_body_m=drone_cfg.attachment_offset_body_m,
        )
        root_gap_max = max(root_gap_max, float(root_gap.max().item()))
        if internal_gap.numel():
            internal_gap_max = max(internal_gap_max, float(internal_gap.max().item()))
        position_error = robot.data.root_pos_w - desired_position
        root_error_squared += float(torch.sum(position_error * position_error).item())
        root_error_samples += int(position_error.shape[0])
        finite = finite and bool(torch.isfinite(robot.data.body_state_w).all().item())
        if not finite:
            break
        if steps % max(1, round(0.1 / options.physics_dt_s)) == 0:
            trace.append(
                {
                    "time_s": simulated_time,
                    "drone_position_m": robot.data.root_pos_w[0].detach().cpu().tolist(),
                    "tip_position_m": stations[0, -1].detach().cpu().tolist(),
                    "desired_drone_position_m": desired_position[0].detach().cpu().tolist(),
                }
            )

    print(f"[isaac_whip] simulation loop complete after {steps} step(s)", flush=True)

    wall_time = time.perf_counter() - wall_start
    final_stations = _material_stations(robot, link_body_ids, cable.rest_lengths_m)
    root_rmse = math.sqrt(root_error_squared / max(root_error_samples, 1))
    tip_final_displacement = torch.linalg.vector_norm(
        final_stations[:, -1] - initial_stations[:, -1], dim=-1
    )
    gap_tolerance = max(0.00025, 0.005 * min(cable.rest_lengths_m))
    tracking_tolerance = 0.08 if options.mode == "excite" else 0.01
    pass_tracking = options.mode == "free" or root_rmse <= tracking_tolerance
    passed = (
        finite
        and topology_ok
        and bool(authored_check["pass"])
        and bool(resolved_property_check["pass"])
        and root_gap_max <= gap_tolerance
        and internal_gap_max <= gap_tolerance
        and pass_tracking
    )
    tip_final_displacement_min = float(tip_final_displacement.min().item())
    tip_final_displacement_max = float(tip_final_displacement.max().item())
    maximum_tip_displacement_min = float(maximum_tip_displacement.min().item())
    maximum_tip_displacement_max = float(maximum_tip_displacement.max().item())
    tip_path_length_min = float(tip_path_length.min().item())
    tip_path_length_max = float(tip_path_length.max().item())
    tip_coordinate_min_env_local = tip_min_env_local.min(dim=0).values.detach().cpu().tolist()
    tip_coordinate_max_env_local = tip_max_env_local.max(dim=0).values.detach().cpu().tolist()
    result: dict[str, object] = {
        "schema": "isaac_whip_physics_run_v1",
        "root_reference_schema": "isaac_whip_root_reference_v1",
        "status": "PASS" if passed else "FAIL",
        "mode": options.mode,
        "num_envs": options.num_envs,
        "device": options.device,
        "physics_dt_s": options.physics_dt_s,
        "requested_control_hz": options.control_hz,
        "achieved_control_hz": achieved_control_hz,
        "requested_render_hz": options.render_hz,
        "achieved_render_hz": achieved_render_hz,
        "simulated_time_s": simulated_time,
        "wall_time_s": wall_time,
        "real_time_factor": simulated_time / max(wall_time, 1.0e-9),
        "topology": {
            "bodies": robot.num_bodies,
            "expected_bodies": expected_bodies,
            "dofs": robot.num_joints,
            "expected_dofs": expected_dofs,
            "instances": robot.num_instances,
            "pass": topology_ok,
        },
        "authored_parameter_check": authored_check,
        "resolved_body_property_check": resolved_property_check,
        "finite": finite,
        "maximum_root_constraint_gap_m": root_gap_max,
        "maximum_internal_constraint_gap_m": internal_gap_max,
        "constraint_gap_tolerance_m": gap_tolerance,
        "drone_tracking_rmse_m": root_rmse,
        "tracking_tolerance_m": tracking_tolerance,
        "tip_final_displacement_m": {
            "minimum": tip_final_displacement_min,
            "maximum": tip_final_displacement_max,
        },
        "tip_maximum_displacement_m": {
            "minimum": maximum_tip_displacement_min,
            "maximum": maximum_tip_displacement_max,
        },
        "tip_path_length_m": {
            "minimum": tip_path_length_min,
            "maximum": tip_path_length_max,
        },
        "tip_coordinate_min_env_local_m": tip_coordinate_min_env_local,
        "tip_coordinate_max_env_local_m": tip_coordinate_max_env_local,
        "cable": cable.to_manifest(),
        "drone": {
            "config_path": str(options.drone_config_path.resolve()),
            "config_sha256": hashlib.sha256(options.drone_config_path.read_bytes()).hexdigest(),
            "mass_kg": drone_cfg.mass_kg,
            "total_system_mass_kg": total_mass,
            "maximum_collective_thrust_n": drone_cfg.maximum_collective_thrust_n,
            "hover_thrust_fraction": total_mass * 9.81 / drone_cfg.maximum_collective_thrust_n,
            "provisional": drone_cfg.provisional,
            "note": drone_cfg.note,
        },
        "trace_env_0_10_hz": trace,
    }
    if options.export_usd is not None:
        result["exported_usd"] = str(options.export_usd.resolve())
    if options.output_json is not None:
        options.output_json.parent.mkdir(parents=True, exist_ok=True)
        options.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[isaac_whip] report: {options.output_json.resolve()}", flush=True)
    print(
        "[isaac_whip] "
        f"{result['status']} | envs={options.num_envs} links={cable.link_count} "
        f"mode={options.mode} simulated={simulated_time:.3f}s "
        f"root_gap={root_gap_max * 1000.0:.3f}mm "
        f"internal_gap={internal_gap_max * 1000.0:.3f}mm "
        f"tracking={root_rmse * 1000.0:.3f}mm RTF={result['real_time_factor']:.2f}",
        flush=True,
    )
    return result
