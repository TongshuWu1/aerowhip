"""Procedural USD authoring for the floating drone--cable articulation.

Import this module only after :class:`isaaclab.app.AppLauncher` has started
Isaac Sim.  The pure physical conversion lives in :mod:`isaac_whip.config`.
"""

from __future__ import annotations

import math

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from .config import CablePlantSpec, DronePhysicalConfig


DRONE_BODY_NAME = "Drone"
LINK_NAME_PREFIX = "Link_"


def _path(value: str | Sdf.Path) -> Sdf.Path:
    return value if isinstance(value, Sdf.Path) else Sdf.Path(value)


def _identity_quat() -> Gf.Quatf:
    return Gf.Quatf(1.0, 0.0, 0.0, 0.0)


def _x_to_negative_z_quat() -> Gf.Quatf:
    # Positive 90 degrees around local Y maps +X onto world -Z.
    return Gf.Quatf(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), 90.0).GetQuat())


def _set_body_properties(
    prim: Usd.Prim,
    *,
    mass_kg: float,
    center_of_mass_m: tuple[float, float, float],
    diagonal_inertia_kg_m2: tuple[float, float, float],
) -> None:
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(float(mass_kg))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass_m))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*diagonal_inertia_kg_m2))
    mass.CreatePrincipalAxesAttr(_identity_quat())
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx.CreateLinearDampingAttr(0.0)
    physx.CreateAngularDampingAttr(0.0)
    physx.CreateMaxAngularVelocityAttr(1000.0)
    physx.CreateSleepThresholdAttr(0.0)


def _add_box(
    stage: Usd.Stage,
    path: Sdf.Path,
    *,
    size_m: tuple[float, float, float],
    color: tuple[float, float, float],
    collision: bool,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*size_m))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        collision_api.CreateContactOffsetAttr(0.002)
        collision_api.CreateRestOffsetAttr(0.0)
    return cube


def _author_drone(
    stage: Usd.Stage,
    root_path: Sdf.Path,
    drone_cfg: DronePhysicalConfig,
) -> Sdf.Path:
    drone_path = root_path.AppendChild(DRONE_BODY_NAME)
    drone = UsdGeom.Xform.Define(stage, drone_path)
    drone.AddTranslateOp().Set(Gf.Vec3d(*drone_cfg.initial_position_m))
    _set_body_properties(
        drone.GetPrim(),
        mass_kg=drone_cfg.mass_kg,
        center_of_mass_m=(0.0, 0.0, 0.0),
        diagonal_inertia_kg_m2=drone_cfg.diagonal_inertia_kg_m2,
    )
    UsdPhysics.ArticulationRootAPI.Apply(drone.GetPrim())
    articulation = PhysxSchema.PhysxArticulationAPI.Apply(drone.GetPrim())
    articulation.CreateEnabledSelfCollisionsAttr(False)
    articulation.CreateSolverPositionIterationCountAttr(16)
    articulation.CreateSolverVelocityIterationCountAttr(4)
    articulation.CreateSleepThresholdAttr(0.0)
    articulation.CreateStabilizationThresholdAttr(0.0)

    _add_box(
        stage,
        drone_path.AppendChild("Body"),
        size_m=drone_cfg.body_size_m,
        color=(0.12, 0.18, 0.25),
        collision=True,
    )
    arm_width = 0.008
    arm_height = 0.006
    arm_length = 2.0 * drone_cfg.arm_length_m
    for name, angle in (("ArmA", 45.0), ("ArmB", -45.0)):
        arm = _add_box(
            stage,
            drone_path.AppendChild(name),
            size_m=(arm_length, arm_width, arm_height),
            color=(0.35, 0.38, 0.42),
            collision=False,
        )
        arm.AddRotateZOp().Set(angle)

    rotor_radius = min(0.0381, 0.55 * drone_cfg.arm_length_m)
    rotor_offset = drone_cfg.arm_length_m / math.sqrt(2.0)
    rotor_positions = (
        (rotor_offset, rotor_offset, 0.01),
        (-rotor_offset, rotor_offset, 0.01),
        (-rotor_offset, -rotor_offset, 0.01),
        (rotor_offset, -rotor_offset, 0.01),
    )
    for index, position in enumerate(rotor_positions):
        rotor = UsdGeom.Cylinder.Define(stage, drone_path.AppendChild(f"Rotor_{index}"))
        rotor.CreateAxisAttr(UsdGeom.Tokens.z)
        rotor.CreateRadiusAttr(rotor_radius)
        rotor.CreateHeightAttr(0.0015)
        rotor.AddTranslateOp().Set(Gf.Vec3f(*position))
        rotor.CreateDisplayColorAttr().Set([Gf.Vec3f(0.15, 0.48, 0.72)])
        rotor.CreateDisplayOpacityAttr().Set([0.45])
    return drone_path


def _author_cable_link(
    stage: Usd.Stage,
    link_path: Sdf.Path,
    *,
    center_m: tuple[float, float, float],
    length_m: float,
    radius_m: float,
    visual_radius_m: float,
    mass_kg: float,
    com_x_m: float,
    diagonal_inertia_kg_m2: tuple[float, float, float],
    index: int,
) -> None:
    link = UsdGeom.Xform.Define(stage, link_path)
    link.AddTranslateOp().Set(Gf.Vec3d(*center_m))
    link.AddOrientOp().Set(_x_to_negative_z_quat())
    _set_body_properties(
        link.GetPrim(),
        mass_kg=mass_kg,
        center_of_mass_m=(com_x_m, 0.0, 0.0),
        diagonal_inertia_kg_m2=diagonal_inertia_kg_m2,
    )

    collision = UsdGeom.Capsule.Define(stage, link_path.AppendChild("Collision"))
    collision.CreateAxisAttr(UsdGeom.Tokens.x)
    collision.CreateRadiusAttr(radius_m)
    collision.CreateHeightAttr(max(length_m - 2.0 * radius_m, 1.0e-6))
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
    collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision.GetPrim())
    collision_api.CreateContactOffsetAttr(max(2.0 * radius_m, 0.002))
    collision_api.CreateRestOffsetAttr(0.0)

    visual = UsdGeom.Capsule.Define(stage, link_path.AppendChild("Visual"))
    visual.CreateAxisAttr(UsdGeom.Tokens.x)
    visual.CreateRadiusAttr(visual_radius_m)
    visual.CreateHeightAttr(max(length_m - 2.0 * visual_radius_m, 1.0e-6))
    shade = 0.76 + 0.10 * (index % 2)
    visual.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, shade * 0.48, 0.06)])


def _lock_axis(joint_prim: Usd.Prim, axis: str) -> None:
    limit = UsdPhysics.LimitAPI.Apply(joint_prim, axis)
    # USD/PhysX encodes a locked D6 axis with low > high.
    limit.CreateLowAttr(1.0)
    limit.CreateHighAttr(-1.0)


def _author_root_joint(
    stage: Usd.Stage,
    joint_path: Sdf.Path,
    *,
    drone_path: Sdf.Path,
    first_link_path: Sdf.Path,
    drone_cfg: DronePhysicalConfig,
    first_length_m: float,
) -> None:
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([drone_path])
    joint.CreateBody1Rel().SetTargets([first_link_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*drone_cfg.attachment_offset_body_m))
    joint.CreateLocalRot0Attr().Set(_x_to_negative_z_quat())
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5 * first_length_m, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(_identity_quat())
    joint.CreateCollisionEnabledAttr(False)


def _author_bending_joint(
    stage: Usd.Stage,
    joint_path: Sdf.Path,
    *,
    parent_path: Sdf.Path,
    child_path: Sdf.Path,
    parent_length_m: float,
    child_length_m: float,
    stiffness_n_m_deg: float,
    damping_n_m_s_deg: float,
) -> None:
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([parent_path])
    joint.CreateBody1Rel().SetTargets([child_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.5 * parent_length_m, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(_identity_quat())
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.5 * child_length_m, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(_identity_quat())
    joint.CreateCollisionEnabledAttr(False)
    prim = joint.GetPrim()
    for axis in (UsdPhysics.Tokens.transX, UsdPhysics.Tokens.transY, UsdPhysics.Tokens.transZ):
        _lock_axis(prim, axis)
    # rotX is deliberately left completely free.  Locking it would impose
    # infinite torsional stiffness, which is not part of the identified model.
    for axis in (UsdPhysics.Tokens.rotY, UsdPhysics.Tokens.rotZ):
        drive = UsdPhysics.DriveAPI.Apply(prim, axis)
        drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(float(stiffness_n_m_deg))
        drive.CreateDampingAttr(float(damping_n_m_s_deg))


def _add_station_marker(
    stage: Usd.Stage,
    parent_link_path: Sdf.Path,
    *,
    name: str,
    local_x_m: float,
    radius_m: float,
    physical_marker: bool,
) -> None:
    sphere = UsdGeom.Sphere.Define(stage, parent_link_path.AppendChild(name))
    sphere.CreateRadiusAttr(radius_m)
    sphere.AddTranslateOp().Set(Gf.Vec3f(local_x_m, 0.0, 0.0))
    color = Gf.Vec3f(0.96, 0.96, 0.96) if physical_marker else Gf.Vec3f(0.25, 0.72, 0.92)
    sphere.CreateDisplayColorAttr().Set([color])


def author_drone_cable(
    stage: Usd.Stage,
    root_path: str | Sdf.Path,
    *,
    drone_cfg: DronePhysicalConfig,
    cable: CablePlantSpec,
) -> dict[str, object]:
    """Author one SI-unit, floating-base drone--cable articulation."""

    root = _path(root_path)
    UsdGeom.Xform.Define(stage, root)
    drone_path = _author_drone(stage, root, drone_cfg)
    cable_scope = root.AppendChild("Cable")
    joints_scope = root.AppendChild("Joints")
    UsdGeom.Scope.Define(stage, cable_scope)
    UsdGeom.Scope.Define(stage, joints_scope)

    start = drone_cfg.initial_position_m
    offset = drone_cfg.attachment_offset_body_m
    radius = 0.5 * cable.diameter_m
    visual_radius = max(2.2 * radius, 0.004)
    link_paths: list[Sdf.Path] = []
    for index, length in enumerate(cable.rest_lengths_m):
        material_midpoint = 0.5 * (
            cable.material_coordinates_m[index] + cable.material_coordinates_m[index + 1]
        )
        center = (
            start[0] + offset[0],
            start[1] + offset[1],
            start[2] + offset[2] - material_midpoint,
        )
        link_path = cable_scope.AppendChild(f"{LINK_NAME_PREFIX}{index:02d}")
        _author_cable_link(
            stage,
            link_path,
            center_m=center,
            length_m=length,
            radius_m=radius,
            visual_radius_m=visual_radius,
            mass_kg=cable.link_masses_kg[index],
            com_x_m=cable.link_com_x_m[index],
            diagonal_inertia_kg_m2=cable.link_diagonal_inertia_kg_m2[index],
            index=index,
        )
        link_paths.append(link_path)

    _author_root_joint(
        stage,
        joints_scope.AppendChild("RootFixed"),
        drone_path=drone_path,
        first_link_path=link_paths[0],
        drone_cfg=drone_cfg,
        first_length_m=cable.rest_lengths_m[0],
    )
    for index in range(1, cable.link_count):
        _author_bending_joint(
            stage,
            joints_scope.AppendChild(f"Bend_{index:02d}"),
            parent_path=link_paths[index - 1],
            child_path=link_paths[index],
            parent_length_m=cable.rest_lengths_m[index - 1],
            child_length_m=cable.rest_lengths_m[index],
            stiffness_n_m_deg=cable.usd_hinge_stiffness_n_m_deg[index - 1],
            damping_n_m_s_deg=cable.usd_hinge_damping_n_m_s_deg[index - 1],
        )

    # Show all numerical stations, while distinguishing the 11 measured
    # stations of the 20-link artifact when that exact discretization is used.
    measured_indices = set(range(0, cable.node_count, 2)) if cable.node_count == 21 else set()
    marker_radius = max(1.6 * visual_radius, 0.006)
    _add_station_marker(
        stage,
        link_paths[0],
        name="Station_00",
        local_x_m=-0.5 * cable.rest_lengths_m[0],
        radius_m=marker_radius,
        physical_marker=0 in measured_indices,
    )
    for index, link_path in enumerate(link_paths):
        station = index + 1
        _add_station_marker(
            stage,
            link_path,
            name=f"Station_{station:02d}",
            local_x_m=0.5 * cable.rest_lengths_m[index],
            radius_m=marker_radius,
            physical_marker=station in measured_indices,
        )

    return {
        "root_path": str(root),
        "drone_path": str(drone_path),
        "link_paths": tuple(str(value) for value in link_paths),
        "joint_paths": (
            str(joints_scope.AppendChild("RootFixed")),
            *(str(joints_scope.AppendChild(f"Bend_{index:02d}")) for index in range(1, cable.link_count)),
        ),
    }


def export_drone_cable_usd(
    output_path: str,
    *,
    drone_cfg: DronePhysicalConfig,
    cable: CablePlantSpec,
) -> dict[str, object]:
    """Write the procedural plant as one portable, SI-unit USD asset.

    The same authoring function is used for the exported artifact and the live
    simulation.  This avoids a second, silently different cable definition.
    """

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    paths = author_drone_cable(stage, "/DroneCable", drone_cfg=drone_cfg, cable=cable)
    root = stage.GetPrimAtPath("/DroneCable")
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    return paths
