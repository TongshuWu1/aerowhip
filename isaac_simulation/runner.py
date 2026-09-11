"""Independent plant runtime; no import of the research simulator or policy."""

import csv
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
import torch
from pxr import Gf, UsdGeom
import isaaclab.sim as sim_utils
from .dynamics import (
    FlightPath,
    QuadrotorController,
    cable_masses,
    cable_loads,
    rotation_from_wxyz,
    tracked_kinematics,
    validate_config,
)
from .scene import build_scene
from .rod_material import rod_coefficients, rod_mass_properties, aerodynamic_loads


class FlightSimulationContext(sim_utils.SimulationContext):
    """Let our runner save and exit when Kit stops the timeline.

    The installed Lab STOP callback renders in a loop after STOP, and PhysX
    tensor handles cannot be resumed after that event. Override only this
    lifecycle hook; integration and rendering still use SimulationContext.
    """

    stop_requested = False

    def _app_control_on_stop_handle_fn(self, event):
        if not self._disable_app_control_on_stop_handle:
            self.stop_requested = True


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(app, args):
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for option, key in [
        ("period", "period_s"),
        ("radius", "radius_m"),
        ("hold", "hold_s"),
        ("cycles", "cycles"),
    ]:
        value = getattr(args, option)
        if value is not None:
            config["trajectory"][key] = value
    validate_config(config)
    clock = config["physics"]
    hz = clock["rate_hz"]
    dt = 1 / hz
    path = FlightPath(config["trajectory"], args.trajectory)
    duration = args.duration if args.duration is not None else path.duration
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be finite and positive")
    output = args.output or Path(__file__).resolve().parents[
        1
    ] / "runs/isaac_physx" / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plant.json", config)
    source = Path(__file__).resolve().parents[1]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for p in [
        source / "run_isaac_simulation.py",
        *(source / "isaac_simulation").glob("*.py"),
    ]:
        relative = p.relative_to(source)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()
        target.write_bytes(data)
        hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    write_json(
        output / "provenance.json",
        dict(
            source_sha256=hashes,
            python=sys.executable,
            torch=torch.__version__,
            physics_device=args.device,
            trajectory=args.trajectory,
        ),
    )
    sim = FlightSimulationContext(
        sim_utils.SimulationCfg(
            dt=dt,
            gravity=(0.0, 0.0, -9.80665),
            render_interval=hz // clock["render_rate_hz"],
            device=args.device,
            physx=sim_utils.PhysxCfg(
                solver_type=1,
                enable_external_forces_every_iteration=True,
                min_position_iteration_count=clock["position_iterations"],
                min_velocity_iteration_count=clock["velocity_iterations"],
            ),
        )
    )
    sim.set_camera_view(eye=(1.65, -2.1, 1.95), target=(-0.25, 0, 1.12))
    robot, visual = build_scene(config, path)
    print("[PhysX] Scene authored; initializing articulation.", flush=True)
    sim.reset()
    # Initialization only: reset startup gravity drift before the first command.
    robot.write_root_state_to_sim(robot.data.default_root_state.clone())
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
    )
    robot.reset()
    sim.forward()
    robot.update(dt)
    names = robot.body_names
    enhanced = "rod_material" in config["cable"]
    stiffness = torch.zeros_like(robot.data.joint_stiffness)
    if enhanced:
        material = rod_coefficients(config["cable"])
        _, rod_com_z, _, _ = rod_mass_properties(config["cable"])
    # D6 passive drives integrate isotropic angular resistance implicitly.
    # Keep the three top-pivot axes entirely free of damping/restoring drive.
    damping = torch.full_like(
        robot.data.joint_damping,
        0.0 if enhanced else config["cable"]["joint_angular_damping_n_m_s"],
    )
    for i, name in enumerate(robot.joint_names):
        if name.startswith("Cable00"):
            damping[:, i] = 0.0
        elif enhanced:
            mode = "twist" if name.endswith(":2") or name.endswith("rotZ") else "bend"
            stiffness[:, i] = material[mode + "_stiffness"]
            damping[:, i] = material[mode + "_damping"]
    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)
    robot.write_joint_effort_limit_to_sim(torch.full_like(damping, 1.0))
    if not torch.allclose(
        robot.root_physx_view.get_dof_dampings().cpu(), damping.cpu()
    ):
        raise RuntimeError(
            "PhysX passive joint damping did not match the configured values"
        )
    body = names.index("Drone")
    if body != 0:
        raise RuntimeError("PhysX articulation root must be the Drone body")
    cable_ids = np.array(
        [names.index(f"Segment{i:02d}") for i in range(config["cable"]["segments"])]
    )
    if enhanced:
        _, expected_com, expected_inertia, _ = rod_mass_properties(config["cable"])
        actual_com = robot.root_physx_view.get_coms()[0, cable_ids, :3].cpu().numpy()
        actual_inertia = (
            robot.root_physx_view.get_inertias()[0, cable_ids]
            .cpu()
            .numpy()
            .reshape(-1, 3, 3)
        )
        expected_com_xyz = np.zeros_like(actual_com)
        expected_com_xyz[:, 2] = -expected_com
        if not np.allclose(actual_com, expected_com_xyz, atol=1e-8) or not np.allclose(
            actual_inertia,
            np.eye(3)[None, :, :] * expected_inertia[:, None, :],
            rtol=1e-5,
            atol=1e-12,
        ):
            raise RuntimeError(
                "PhysX cable COM/inertia does not match authored marker geometry"
            )
    masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
    measured_drone_mass = float(masses[body])
    measured_cable_mass = float(masses[cable_ids].sum())
    if not np.isclose(
        measured_drone_mass, config["drone"]["mass_kg"], rtol=1e-5
    ) or not np.isclose(measured_cable_mass, cable_masses(config).sum(), rtol=1e-5):
        raise RuntimeError(
            f"PhysX mass mismatch: drone {measured_drone_mass}, cable {measured_cable_mass}"
        )
    controller = QuadrotorController(config)
    force = np.zeros((len(names), 3))
    torque = np.zeros_like(force)
    commands = []
    records = []
    trails = []
    saturation_steps = 0
    max_gap = 0.0
    max_tilt = 0.0
    failure = None
    wall_start = time.perf_counter()
    frame_wall = wall_start
    step = 0
    published = False
    reference = path.sample(0.0)
    ui_state = {"pause": False, "close": False}
    ui_label = None
    if not args.headless:
        import omni.ui as ui

        panel = ui.Window("PhysX drone + cable", width=355, height=250)
        with panel.frame:
            with ui.VStack(spacing=8):
                ui.Label("Independent plant | " + args.trajectory)
                ui.Label(
                    f"{1000*measured_drone_mass:.0f} g drone + {1000*measured_cable_mass:.0f} g passive cable"
                )
                ui.Label("Cyan: command    Orange: measured path")
                ui_label = ui.Label("Starting...", word_wrap=True)
                with ui.HStack():
                    ui.Button(
                        "Pause / resume",
                        clicked_fn=lambda: ui_state.update(pause=not ui_state["pause"]),
                    )
                    ui.Button(
                        "Finish and save",
                        clicked_fn=lambda: ui_state.update(close=True),
                    )
                ui.Label(
                    "PhysX motion, rotor limits and cable reaction.\nNo policy, DDER or fitted residual loaded.",
                    word_wrap=True,
                )
    print(
        f"[PhysX] Initialized {len(names)} rigid bodies / {robot.num_joints} DOFs; mass {measured_drone_mass:.6f} + {measured_cable_mass:.6f} kg.",
        flush=True,
    )

    def state():
        values = robot.data.body_link_state_w[0].detach().cpu().numpy().astype(float)
        return values, rotation_from_wxyz(values[:, 3:7])

    def tracked(values, rotations):
        return tracked_kinematics(
            values[body, :3],
            values[body, 7:10],
            rotations[body],
            values[body, 10:13],
            config["drone"]["tracked_origin_body_m"],
        )

    def geometry(values, rotations):
        positions = values[cable_ids, :3]
        r = rotations[cable_ids]
        half = config["cable"]["length_m"] / len(cable_ids) / 2
        top = positions + r[:, :, 2] * half
        bottom = positions - r[:, :, 2] * half
        attachment = values[body, :3] + rotations[body] @ np.array(
            config["drone"]["attachment_body_m"]
        )
        gaps = np.r_[
            np.linalg.norm(top[0] - attachment),
            np.linalg.norm(top[1:] - bottom[:-1], axis=1),
        ]
        markers = np.array(
            [positions[i] + r[i, :, 2] * z for i, z in visual["marker_bindings"]]
        )
        return gaps, markers, bottom[-1]

    def save_result():
        if not records:
            return
        array = np.asarray(records)
        header = [
            "time_s",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "qw",
            "qx",
            "qy",
            "qz",
            "wx",
            "wy",
            "wz",
            "cmd_x",
            "cmd_y",
            "cmd_z",
            "cmd_vx",
            "cmd_vy",
            "cmd_vz",
            "cmd_ax",
            "cmd_ay",
            "cmd_az",
            "tip_x",
            "tip_y",
            "tip_z",
        ]
        header += [f"c{i+1}_{axis}" for i in range(10) for axis in "xyz"]
        header += [f"motor{i+1}_thrust_n" for i in range(4)] + ["joint_gap_max_m"]
        with (output / "ground_truth_100hz.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(records)
        with (output / "fullstate_30hz.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time_s",
                    "px_m",
                    "py_m",
                    "pz_m",
                    "vx_m_s",
                    "vy_m_s",
                    "vz_m_s",
                    "ax_m_s2",
                    "ay_m_s2",
                    "az_m_s2",
                    "yaw_rad",
                    "yaw_rate_rad_s",
                ]
            )
            writer.writerows(commands)
        errors = array[:, 1:4] - array[:, 14:17]
        moving = (array[:, 0] >= path.hold) & (array[:, 0] <= path.end)
        rms = lambda e: (
            float(np.sqrt(np.mean(np.sum(e * e, axis=1)))) if len(e) else None
        )
        summary = dict(
            schema="physx_independent_flight_validation_v1",
            trajectory=args.trajectory,
            physics_engine="NVIDIA PhysX articulation; independent of DDER",
            asset=visual["asset_source"],
            hardware="153 g experimental-rig approximation, Crazyflie mesh scaled; not stock Crazyflie dynamics",
            plant_parameters="Engineering estimates; not fitted or hardware validated",
            cable_model=(
                "Inextensible braided-nylon rod; separate implicit bend/twist and distributed marker/air loads"
                if enhanced
                else "Legacy rigid-link cable"
            ),
            drone_mass_kg=measured_drone_mass,
            cable_mass_kg=measured_cable_mass,
            rigid_bodies=len(names),
            physics_rate_hz=hz,
            controller_rate_hz=clock["controller_rate_hz"],
            command_rate_hz=clock["command_rate_hz"],
            logging_rate_hz=clock["logging_rate_hz"],
            physics_device=args.device,
            rendering=(
                "NVIDIA RTX GPU"
                if not args.headless or args.screenshot
                else "headless, no rendering"
            ),
            simulated_duration_s=float(array[-1, 0]),
            wall_seconds=time.perf_counter() - wall_start,
            tracking_rms_m=rms(errors),
            motion_tracking_rms_m=rms(errors[moving]),
            peak_tracking_error_m=float(np.linalg.norm(errors, axis=1).max()),
            minimum_drone_height_m=float(array[:, 3].min()),
            maximum_drone_height_m=float(array[:, 3].max()),
            maximum_tilt_deg=max_tilt,
            maximum_joint_gap_m=max_gap,
            motor_saturation_fraction=saturation_steps / max(1, step),
            failure=failure,
            records=len(records),
            command_packets=len(commands),
            sensor_contract="Native synthetic ground truth, not yet a Motive/controller-log compatibility adapter",
        )
        summary["position_reference"] = (
            "Top tracked origin; rotated offset and angular-velocity lever arm included"
        )
        summary["quaternion_order"] = "wxyz"
        summary["logged_angular_velocity_frame"] = "world"
        summary["tracked_origin_to_attachment_body_m"] = (
            np.array(config["drone"]["attachment_body_m"])
            - np.array(config["drone"]["tracked_origin_body_m"])
        ).tolist()
        write_json(output / "summary.json", summary)
        print("[PhysX] RESULT " + json.dumps(summary), flush=True)

    try:
        values, rotations = state()
        while app.is_running() and not ui_state["close"]:
            if sim.stop_requested or sim.is_stopped():
                break
            if ui_state["pause"] or not sim.is_playing():
                sim.render()
                time.sleep(0.01)
                frame_wall = time.perf_counter()
                continue
            t = step * dt
            measured_position, measured_velocity = tracked(values, rotations)
            if step % (hz // clock["command_rate_hz"]) == 0:
                reference = path.sample(t)
                if not published:
                    commands.append(
                        np.r_[
                            t,
                            reference.position,
                            reference.velocity,
                            reference.acceleration,
                            reference.yaw,
                            0.0,
                        ].tolist()
                    )
            if step % (hz // clock["logging_rate_hz"]) == 0 and not published:
                gaps, markers, tip = geometry(values, rotations)
                max_gap = max(max_gap, float(gaps.max()))
                records.append(
                    np.r_[
                        t,
                        measured_position,
                        measured_velocity,
                        values[body, 3:7],
                        values[body, 10:13],
                        reference.position,
                        reference.velocity,
                        reference.acceleration,
                        tip,
                        markers.ravel(),
                        controller.motor_thrust,
                        gaps.max(),
                    ].tolist()
                )
            if t >= duration and not published:
                save_result()
                published = True
                if args.screenshot:
                    import omni.kit.viewport.utility as viewport

                    for _ in range(6):
                        sim.render()
                    viewport.capture_viewport_to_file(
                        viewport.get_active_viewport(), str(output / "scene.png")
                    )
                    # app.update() alone also advances physics without our
                    # controller/loads; render() explicitly suppresses that.
                    for _ in range(40):
                        sim.render()
                if args.headless or args.exit_after_trajectory:
                    break
            if step % (hz // clock["controller_rate_hz"]) == 0:
                controller.update(
                    measured_position,
                    measured_velocity,
                    rotations[body],
                    values[body, 10:13],
                    reference,
                    1 / clock["controller_rate_hz"],
                )
            wrench = controller.actuator_step(dt)
            force.fill(0)
            torque.fill(0)
            force[body] = (
                rotations[body, :, 2] * wrench[0]
                - np.array(config["drone"]["linear_drag_n_s_m"]) * values[body, 7:10]
            )
            torque[body] = rotations[body] @ wrench[1:]
            if enhanced:
                r_down = rotations[cable_ids] @ np.diag([1.0, -1.0, -1.0])
                omega = values[cable_ids, 10:13]
                com_velocity = values[cable_ids, 7:10] + np.cross(
                    omega, r_down[:, :, 2] * rod_com_z[:, None]
                )
                f, q = aerodynamic_loads(
                    values[cable_ids, :3],
                    r_down,
                    com_velocity,
                    omega,
                    rod_com_z,
                    config["cable"],
                )
            else:
                f, q = cable_loads(
                    rotations[cable_ids], values[cable_ids, 7:10], config["cable"]
                )
            force[cable_ids] = f
            torque[cable_ids] = q
            # The passive articulation needs no actuator-target writes. Apply
            # already-world-frame physical loads directly through PhysX tensors.
            robot.root_physx_view.apply_forces_and_torques_at_position(
                force_data=torch.as_tensor(
                    force, dtype=torch.float32, device=robot.device
                ),
                torque_data=torch.as_tensor(
                    torque, dtype=torch.float32, device=robot.device
                ),
                # Tensor API defaults to the link origin, not its COM. Marker
                # masses offset the cable COM; aerodynamic moments above are
                # already computed about that COM, so apply force there too.
                position_data=robot.data.body_com_state_w[0, :, :3].contiguous(),
                indices=robot._ALL_INDICES,
                is_global=True,
            )
            sim.step(render=False)
            robot.update(dt)
            step += 1
            values, rotations = state()
            saturation_steps += int(controller.saturated)
            measured_position, measured_velocity = tracked(values, rotations)
            tilt = float(np.degrees(np.arccos(np.clip(rotations[body, 2, 2], -1, 1))))
            max_tilt = max(max_tilt, tilt)
            if (
                not np.isfinite(values).all()
                or np.max(np.abs(values[:, :3])) > 10
                or values[body, 2] < 0.15
                or tilt > 80
            ):
                failure = "Flight left its finite position/altitude/tilt envelope"
                raise RuntimeError(failure)
            if step % (hz // clock["render_rate_hz"]) == 0 and (
                not args.headless or args.screenshot
            ):
                trails.append(measured_position.copy())
                trails = trails[-3000:]
                if len(trails) > 1:
                    visual["trace"].GetPointsAttr().Set(
                        [Gf.Vec3f(*map(float, p)) for p in trails]
                    )
                    visual["trace"].GetCurveVertexCountsAttr().Set([len(trails)])
                UsdGeom.Xformable(visual["goal"]).GetOrderedXformOps()[0].Set(
                    Gf.Vec3d(*reference.position)
                )
                if ui_label is not None:
                    ui_label.text = (
                        f"Time {step*dt:.1f} s | tracking error {np.linalg.norm(measured_position-reference.position)*100:.1f} cm\nAltitude {measured_position[2]:.2f} m | tilt {tilt:.1f} deg\n"
                        + str(output)
                    )
                sim.render()
                # Visible runs never race ahead of wall time. Slow rendering
                # does not change fixed-step physics or command/log timestamps.
                if not args.headless:
                    time.sleep(
                        max(
                            0.0,
                            1 / clock["render_rate_hz"]
                            - (time.perf_counter() - frame_wall),
                        )
                    )
                frame_wall = time.perf_counter()
            if step % (hz * 5) == 0:
                print(
                    f"[PhysX] t={step*dt:.1f}s z={measured_position[2]:.3f} error={np.linalg.norm(measured_position-reference.position):.4f} m",
                    flush=True,
                )
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        save_result()
        published = True
        raise
    finally:
        if not published:
            save_result()
    return 0
