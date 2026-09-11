"""Controller, actuators and physical loads; PhysX integrates all body motion.

No fitted-model or DDER imports. All vectors use world XYZ metres, Z up;
rotation matrices map body vectors into world coordinates.
"""

from dataclasses import dataclass
import math
import numpy as np


def validate_config(config):
    """Reject invalid plant settings before starting the physics scene."""
    physics = config["physics"]
    for key in (
        "rate_hz",
        "controller_rate_hz",
        "command_rate_hz",
        "logging_rate_hz",
        "render_rate_hz",
    ):
        rate = physics[key]
        if not isinstance(rate, int) or rate <= 0 or physics["rate_hz"] % rate:
            raise ValueError(
                f"{key} must be a positive integer divisor of physics rate"
            )
    # The saved schema intentionally promises these two interface rates.
    if physics["command_rate_hz"] != 30 or physics["logging_rate_hz"] != 100:
        raise ValueError("This runner uses 30 Hz FullState and 100 Hz ground truth")
    drone = config["drone"]
    cable = config["cable"]
    for key in (
        "mass_kg",
        "motor_xy_m",
        "maximum_motor_thrust_n",
        "motor_time_constant_s",
        "yaw_moment_per_thrust_m",
    ):
        if not np.isfinite(drone[key]) or drone[key] <= 0:
            raise ValueError(f"drone.{key} must be finite and positive")
    inertia = np.asarray(drone["inertia_kg_m2"])
    if (
        inertia.shape != (3,)
        or not np.isfinite(inertia).all()
        or np.any(inertia <= 0)
        or 2 * inertia.max() > inertia.sum()
    ):
        raise ValueError(
            "Drone inertia must be positive and satisfy the triangle inequalities"
        )
    if not isinstance(cable["segments"], int) or cable["segments"] < 2:
        raise ValueError("Cable needs at least two segments")
    for key in ("length_m", "diameter_m", "bare_mass_kg"):
        if not np.isfinite(cable[key]) or cable[key] <= 0:
            raise ValueError(f"cable.{key} must be finite and positive")
    for key in (
        "marker_mass_kg",
        "crossflow_drag_coefficient",
    ):
        if not np.isfinite(cable[key]) or cable[key] < 0:
            raise ValueError(f"cable.{key} must be finite and nonnegative")
    if "rod_material" in cable:
        from .rod_material import rod_coefficients

        rod_coefficients(cable)
        for key in (
            "marker_radius_m",
            "air_density_kg_m3",
            "axial_skin_drag_coefficient",
            "marker_drag_coefficient",
            "contact_friction",
        ):
            if not np.isfinite(cable[key]) or cable[key] < 0:
                raise ValueError(f"cable.{key} must be finite and nonnegative")
        wind = np.asarray(cable["wind_world_m_s"])
        if wind.shape != (3,) or not np.isfinite(wind).all():
            raise ValueError("Wind must be a finite world XYZ vector")
    else:
        for key in ("bending_stiffness_n_m2", "joint_angular_damping_n_m_s"):
            if not np.isfinite(cable[key]) or cable[key] < 0:
                raise ValueError(f"cable.{key} must be finite and nonnegative")
    markers = np.asarray(cable["marker_distances_m"])
    if (
        len(markers) != 10
        or not np.isfinite(markers).all()
        or np.any(np.diff(markers) <= 0)
        or markers[0] <= 0
        or markers[-1] > cable["length_m"]
    ):
        raise ValueError("Specify ten ordered cable markers within the cable length")
    if (
        4 * drone["maximum_motor_thrust_n"]
        <= (drone["mass_kg"] + cable_masses(config).sum()) * 9.80665
    ):
        raise ValueError("Loaded rig exceeds maximum collective thrust")


def rotation_from_wxyz(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


def cable_masses(config):
    cable = config["cable"]
    n = cable["segments"]
    length = cable["length_m"] / n
    masses = np.full(n, cable["bare_mass_kg"] / n)
    for distance in cable["marker_distances_m"]:
        masses[min(n - 1, int(distance / length))] += cable["marker_mass_kg"]
    return masses


def tracked_kinematics(position, velocity, rotation, omega_world, offset_body):
    arm = rotation @ np.asarray(offset_body)
    return position + arm, velocity + np.cross(omega_world, arm)


@dataclass
class Reference:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    yaw: float = 0.0


class FlightPath:
    """Analytic PVA with smooth angular-speed ramps and a closed final loop."""

    def __init__(self, config, kind="figure8"):
        if kind not in ("hover", "circle", "figure8", "vertical8"):
            raise ValueError("Unknown trajectory")
        self.config = config
        self.kind = kind
        self.hold = float(config["hold_s"])
        self.ramp = float(config["ramp_s"])
        self.period = float(config["period_s"])
        self.cycles = int(config["cycles"])
        for key in ("hold_s", "final_hold_s", "radius_m"):
            if not np.isfinite(config[key]) or config[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not np.isfinite(self.period) or self.period <= 0 or self.cycles < 1:
            raise ValueError("Period and number of cycles must be positive")
        if not 0 < self.ramp < self.period * self.cycles:
            raise ValueError(
                "Ramp must be positive and shorter than the periodic motion"
            )
        self.motion_duration = self.period * self.cycles + self.ramp
        self.end = self.hold + self.motion_duration
        self.duration = self.end + float(config["final_hold_s"])

    def phase(self, t):
        s = float(np.clip(t - self.hold, 0, self.motion_duration))
        omega = 2 * math.pi / self.period

        def ramp_values(time):
            u = time / self.ramp
            integral = self.ramp * (2.5 * u**4 - 3 * u**5 + u**6)
            rate = 10 * u**3 - 15 * u**4 + 6 * u**5
            accel = (30 * u**2 - 60 * u**3 + 30 * u**4) / self.ramp
            return integral, rate, accel

        if s < self.ramp:
            distance, rate, accel = ramp_values(s)
        elif s > self.motion_duration - self.ramp:
            distance, rate, accel = ramp_values(self.motion_duration - s)
            distance = self.period * self.cycles - distance
            accel = -accel
        else:
            distance, rate, accel = s - 0.5 * self.ramp, 1.0, 0.0
        return omega * distance, omega * rate, omega * accel

    def sample(self, t):
        start = np.array(self.config["start_position_m"], dtype=float)
        zero = np.zeros(3)
        if self.kind == "hover" or t <= self.hold or t >= self.end:
            return Reference(start, zero.copy(), zero.copy())
        theta, rate, accel = self.phase(t)
        r = float(self.config["radius_m"])
        if self.kind == "circle":
            p = r * np.array([np.cos(theta) - 1, np.sin(theta), 0.0])
            dp = r * np.array([-np.sin(theta), np.cos(theta), 0.0])
            ddp = r * np.array([-np.cos(theta), -np.sin(theta), 0.0])
        else:
            p = r * np.array([np.sin(theta), 0.5 * np.sin(2 * theta), 0.0])
            dp = r * np.array([np.cos(theta), np.cos(2 * theta), 0.0])
            ddp = r * np.array([-np.sin(theta), -2 * np.sin(2 * theta), 0.0])
            if self.kind == "vertical8":
                p = p[[0, 2, 1]]
                dp = dp[[0, 2, 1]]
                ddp = ddp[[0, 2, 1]]
        return Reference(start + p, dp * rate, ddp * rate**2 + dp * accel)


class QuadrotorController:
    """PVA position loop, geometric attitude PD and bounded four-motor mixer."""

    def __init__(self, config):
        self.config = config
        d = config["drone"]
        self.mass = d["mass_kg"] + cable_masses(config).sum()
        self.inertia = np.array(d["inertia_kg_m2"])
        arm = d["motor_xy_m"]
        self.motor_xy = np.array([[arm, arm], [-arm, arm], [-arm, -arm], [arm, -arm]])
        self.mixer = np.array(
            [
                np.ones(4),
                self.motor_xy[:, 1],
                -self.motor_xy[:, 0],
                d["yaw_moment_per_thrust_m"] * np.array([1, -1, 1, -1]),
            ]
        )
        self.inverse_mixer = np.linalg.inv(self.mixer)
        self.integral = np.zeros(3)
        self.motor_thrust = np.full(4, self.mass * 9.80665 / 4)
        self.motor_target = self.motor_thrust.copy()
        self.saturated = False

    def update(self, p, v, rotation, omega_world, reference, dt):
        c = self.config["controller"]
        d = self.config["drone"]
        error = reference.position - p
        # Freeze the integrator during actuator saturation (anti-windup).
        if not self.saturated:
            self.integral = np.clip(
                self.integral + error * dt,
                -c["integral_limit_m_s"],
                c["integral_limit_m_s"],
            )
        acceleration = (
            reference.acceleration
            + np.array(c["position_kp"]) * error
            + np.array(c["velocity_kd"]) * (reference.velocity - v)
            + np.array(c["position_ki"]) * self.integral
        )
        force = self.mass * (acceleration + np.array([0.0, 0.0, 9.80665]))
        force[2] = max(force[2], 0.2 * self.mass * 9.80665)
        horizontal = np.linalg.norm(force[:2])
        limit = force[2] * math.tan(math.radians(c["maximum_tilt_deg"]))
        if horizontal > limit:
            force[:2] *= limit / horizontal
        z = force / np.linalg.norm(force)
        heading = np.array([math.cos(reference.yaw), math.sin(reference.yaw), 0.0])
        y = np.cross(z, heading)
        y /= np.linalg.norm(y)
        desired = np.column_stack((np.cross(y, z), y, z))
        skew = 0.5 * (desired.T @ rotation - rotation.T @ desired)
        attitude_error = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        omega = rotation.T @ omega_world
        frequency = c["attitude_frequency_rad_s"]
        torque = (
            -self.inertia * frequency**2 * attitude_error
            - 2 * c["attitude_damping_ratio"] * frequency * self.inertia * omega
            + np.cross(omega, self.inertia * omega)
        )
        thrust = max(0.0, float(force @ rotation[:, 2]))
        unconstrained = self.inverse_mixer @ np.r_[thrust, torque]
        self.motor_target = np.clip(unconstrained, 0.0, d["maximum_motor_thrust_n"])
        self.saturated = bool(np.any(np.abs(unconstrained - self.motor_target) > 1e-8))

    def actuator_step(self, dt):
        tau = self.config["drone"]["motor_time_constant_s"]
        self.motor_thrust += -math.expm1(-dt / tau) * (
            self.motor_target - self.motor_thrust
        )
        return self.mixer @ self.motor_thrust


def cable_loads(rotations, velocities, cable):
    """Crossflow drag and equal/opposite bending torques between physical links.

    The top attachment is a free pivot. No tension force or position projection
    is supplied here: those are solved by the PhysX articulation constraints.
    Passive joint angular damping is integrated implicitly inside PhysX; do
    not also apply explicit damping torques here.
    """
    tangent = -rotations[:, :, 2]
    normal_v = velocities - np.sum(velocities * tangent, axis=1)[:, None] * tangent
    length = cable["length_m"] / cable["segments"]
    area = cable["diameter_m"] * length
    force = (
        -0.5
        * cable["air_density_kg_m3"]
        * cable["crossflow_drag_coefficient"]
        * area
        * np.linalg.norm(normal_v, axis=1)[:, None]
        * normal_v
    )
    torque = np.zeros_like(force)
    pair = (
        -cable["bending_stiffness_n_m2"] / length * np.cross(tangent[:-1], tangent[1:])
    )
    torque[1:] += pair
    torque[:-1] -= pair
    return force, torque
