"""USD scene construction. Import only after Isaac Lab AppLauncher starts."""

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from .dynamics import cable_masses
from .rod_material import rod_mass_properties


def transform(prim, position=None, scale=None):
    xform = UsdGeom.Xformable(prim)
    if position is not None:
        xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def color(shape, rgb):
    shape.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    stage = shape.GetPrim().GetStage()
    path = "/World/Materials/Color_" + "_".join(str(round(v * 1000)) for v in rgb)
    material = UsdShade.Material.Get(stage, path)
    if not material:
        material = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*rgb)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
    UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)


def cube(stage, path, position, size, rgb, collision=False):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.0)
    transform(shape.GetPrim(), position, size)
    color(shape, rgb)
    if collision:
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return shape


def sphere(stage, path, position, radius, rgb):
    shape = UsdGeom.Sphere.Define(stage, path)
    shape.CreateRadiusAttr(radius)
    transform(shape.GetPrim(), position)
    color(shape, rgb)
    return shape


def rigid_body(prim, mass, inertia, gyroscopic=True, com=(0.0, 0.0, 0.0)):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, com)))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
    api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    px.CreateLinearDampingAttr(0.0)
    px.CreateAngularDampingAttr(0.0)
    px.CreateMaxAngularVelocityAttr(10000.0)
    px.CreateEnableGyroscopicForcesAttr(gyroscopic)
    px.CreateSleepThresholdAttr(0.0)


def visual_drone(stage, body_path, config):
    """Reuse NVIDIA's Crazyflie mesh only; plant mass/inertia live on our body."""
    scale = config["drone"]["visual_scale"]
    path = body_path + "/CrazyflieVisual"
    visual = UsdGeom.Xform.Define(stage, path).GetPrim()
    source = f"{ISAAC_NUCLEUS_DIR}/Robots/Bitcraze/Crazyflie/cf2x.usd"
    visual.GetReferences().AddReference(source)
    if not list(visual.GetChildren()):
        raise RuntimeError(
            "Crazyflie USD did not load. Check Isaac asset-server access; no placeholder substituted."
        )
    # Disable source instances before overriding their referenced physics APIs.
    for prim in Usd.PrimRange(visual):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in list(Usd.PrimRange(visual)):
        if prim.IsA(UsdPhysics.Joint):
            prim.SetActive(False)
            continue
        for api in (
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.CollisionAPI,
            PhysxSchema.PhysxArticulationAPI,
            PhysxSchema.PhysxRigidBodyAPI,
            PhysxSchema.PhysxCollisionAPI,
        ):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    UsdGeom.Xformable(visual).AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
    # Tracking beads and red/green navigation lights have no independent dynamics.
    for i, xy in enumerate(((-0.025, -0.02), (0.025, -0.02), (0, 0.027))):
        sphere(
            stage,
            body_path + f"/TrackingMarker{i}",
            (*xy, config["drone"]["tracked_origin_body_m"][2]),
            0.006,
            (0.9, 0.95, 1.0),
        )
    sphere(stage, body_path + "/FrontLED", (0.045, 0, 0.014), 0.004, (0.05, 1.0, 0.5))
    sphere(stage, body_path + "/RearLED", (-0.045, 0, 0.014), 0.004, (1.0, 0.08, 0.12))
    return source


def polyline(stage, path, points, rgb, width=0.008):
    line = UsdGeom.BasisCurves.Define(stage, path)
    line.CreateTypeAttr(UsdGeom.Tokens.linear)
    line.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    line.CreateCurveVertexCountsAttr([len(points)])
    line.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    line.CreateWidthsAttr([width])
    line.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    color(line, rgb)
    return line


def build_scene(config, flight_path):
    stage = sim_utils.get_current_stage()
    cube(
        stage,
        "/World/Ground",
        (0, 0, -0.05),
        (200.0, 200.0, 0.10),
        (0.035, 0.055, 0.08),
        True,
    )
    light = sim_utils.DomeLightCfg(intensity=450.0, color=(0.78, 0.86, 1.0))
    light.func("/World/Softbox", light)
    sun = sim_utils.DistantLightCfg(intensity=700.0, color=(1.0, 0.87, 0.72), angle=0.6)
    sun.func("/World/KeyLight", sun, orientation=(0.92388, 0.38268, 0.0, 0.0))
    # A clean research bay: floor scale, perimeter posts and a launch pad.
    for i in range(-6, 7):
        shade = (0.20, 0.25, 0.32) if i else (0.28, 0.39, 0.50)
        cube(stage, f"/World/GridX{i+6}", (i * 0.5, 0, 0.002), (0.006, 6, 0.002), shade)
        cube(stage, f"/World/GridY{i+6}", (0, i * 0.5, 0.002), (6, 0.006, 0.002), shade)
    for i, (x, y) in enumerate(((-2, -2), (-2, 2), (2, -2), (2, 2))):
        cube(
            stage,
            f"/World/Post{i}",
            (x, y, 1.6),
            (0.035, 0.035, 3.2),
            (0.18, 0.23, 0.30),
        )
        cube(
            stage,
            f"/World/PostAccent{i}",
            (x, y, 2.9),
            (0.039, 0.039, 0.08),
            (0.08, 0.65, 0.95),
        )
    pad = UsdGeom.Cylinder.Define(stage, "/World/LaunchPad")
    pad.CreateRadiusAttr(0.30)
    pad.CreateHeightAttr(0.008)
    transform(pad.GetPrim(), (0, 0, 0.008))
    color(pad, (0.13, 0.27, 0.34))
    planned = [
        flight_path.sample(t).position
        for t in np.linspace(flight_path.hold, flight_path.end, 500)
    ]
    polyline(stage, "/World/Reference", planned, (0.10, 0.58, 0.92), 0.006)
    trace = polyline(
        stage, "/World/ActualTrail", [planned[0], planned[0]], (1.0, 0.36, 0.10), 0.005
    )
    goal = sphere(
        stage, "/World/CurrentReference", planned[0], 0.022, (0.10, 0.85, 1.0)
    )

    system = UsdGeom.Xform.Define(stage, "/World/FlightRig").GetPrim()
    # Root API must be on the drone rigid body. Putting it on the container lets
    # PhysX choose a central cable segment as root and changes reset coordinates.
    body_path = "/World/FlightRig/Drone"
    body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(body)
    articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(body)
    articulation_api.CreateEnabledSelfCollisionsAttr(config["cable"]["self_collision"])
    articulation_api.CreateSolverPositionIterationCountAttr(
        config["physics"]["position_iterations"]
    )
    articulation_api.CreateSolverVelocityIterationCountAttr(
        config["physics"]["velocity_iterations"]
    )
    articulation_api.CreateSleepThresholdAttr(0.0)
    articulation_api.CreateStabilizationThresholdAttr(0.0)
    origin = np.array(config["trajectory"]["start_position_m"], dtype=float) - np.array(
        config["drone"]["tracked_origin_body_m"]
    )
    transform(body, origin)
    visual_source = visual_drone(stage, body_path, config)
    rigid_body(body, config["drone"]["mass_kg"], config["drone"]["inertia_kg_m2"])
    collision = cube(
        stage,
        body_path + "/CollisionHull",
        (0, 0, 0),
        (0.115, 0.10, 0.035),
        (0.1, 0.1, 0.1),
        True,
    )
    collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    cable = config["cable"]
    n = cable["segments"]
    length = cable["length_m"] / n
    radius = cable["diameter_m"] / 2
    masses = cable_masses(config)
    enhanced = "rod_material" in cable
    if enhanced:
        masses, com_z, inertias, _ = rod_mass_properties(cable)
        contact_material = UsdShade.Material.Define(
            stage, "/World/Materials/ParacordContact"
        )
        contact_api = UsdPhysics.MaterialAPI.Apply(contact_material.GetPrim())
        contact_api.CreateStaticFrictionAttr(cable["contact_friction"])
        contact_api.CreateDynamicFrictionAttr(cable["contact_friction"])
        contact_api.CreateRestitutionAttr(0.0)
    attach = np.array(config["drone"]["attachment_body_m"], dtype=float)
    paths = []
    for i, mass in enumerate(masses):
        path = f"/World/FlightRig/Segment{i:02d}"
        paths.append(path)
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        transform(prim, origin + attach + np.array([0.0, 0.0, -(i + 0.5) * length]))
        transverse = mass * (length * length / 12 + radius * radius / 4)
        rigid_body(
            prim,
            mass,
            (
                inertias[i]
                if enhanced
                else [transverse, transverse, 0.5 * mass * radius * radius]
            ),
            com=(0, 0, -com_z[i]) if enhanced else (0, 0, 0),
        )
        shape = UsdGeom.Capsule.Define(stage, path + "/Cable")
        shape.CreateAxisAttr("Z")
        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(max(0.001, length - 2 * radius))
        color(shape, (0.95, 0.65, 0.15))
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
        if enhanced:
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
                contact_material, materialPurpose="physics"
            )
        collision_api = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        collision_api.CreateContactOffsetAttr(0.001)
        collision_api.CreateRestOffsetAttr(0.0)
        joint = UsdPhysics.Joint.Define(stage, f"/World/FlightRig/Joints/Cable{i:02d}")
        # D6 with translation locked is a ball joint, with supported implicit
        # angular damping. USD SphericalJoint silently ignores its drive data.
        for axis in ("transX", "transY", "transZ"):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)
        for axis in ("rotX", "rotY", "rotZ"):
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(0.0)
            drive.CreateTargetVelocityAttr(0.0)
            drive.CreateMaxForceAttr(1.0)
        joint.CreateBody0Rel().SetTargets(
            [Sdf.Path(body_path if i == 0 else paths[i - 1])]
        )
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        joint.CreateLocalPos0Attr(
            Gf.Vec3f(*attach) if i == 0 else Gf.Vec3f(0, 0, -length / 2)
        )
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, length / 2))
        joint.CreateCollisionEnabledAttr(False)
    marker_bindings = []
    for i, distance in enumerate(cable["marker_distances_m"]):
        segment = min(n - 1, int(distance / length))
        z = (segment + 0.5) * length - distance
        marker = sphere(
            stage,
            paths[segment] + f"/Marker{i+1:02d}",
            (0, 0, z),
            cable.get("marker_radius_m", 0.006),
            (0.92, 0.94, 0.96) if i < 9 else (1.0, 0.12, 0.08),
        )
        if enhanced:
            UsdPhysics.CollisionAPI.Apply(marker.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(
                contact_material, materialPurpose="physics"
            )
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(marker.GetPrim())
            collision_api.CreateContactOffsetAttr(0.0005)
            collision_api.CreateRestOffsetAttr(0.0)
        marker_bindings.append((segment, z))
    visual = dict(
        trace=trace,
        goal=goal,
        marker_bindings=marker_bindings,
        asset_source=visual_source,
    )
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/FlightRig",
            spawn=None,
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(origin), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
            ),
            actuators={
                "passive_cable": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness=0.0, damping=0.0
                )
            },
        )
    )
    return robot, visual
