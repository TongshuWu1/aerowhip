#!/usr/bin/env python3

import math
import signal

import numpy as np
import rclpy
import transforms3d

from crazyflie_py import Crazyswarm


# ============================================================
# Experiment configuration
# ============================================================

DRONE_NAME = "cf_7"

RATE = 50.0

TAKEOFF_HEIGHT = 1.50
TAKEOFF_DURATION = 3.0

NORMAL_FULLSTATE_TIME = 10.0
RECOVERY_TIME = 2.0

LAND_HEIGHT = 0.04
LAND_DURATION = 3.0

# Physical/calibrated platform values.
REAL_MASS = 0.157
EXPECTED_MASS_THRUST = 17700.0

# The script verifies that the firmware values are close to these.
MASS_TOLERANCE = 0.005
MASS_THRUST_TOLERANCE = 500.0


# ============================================================
# Force sequence
#
# Each stage has:
#
#   ([Fx, Fy, Fz], transition_time, hold_time)
#
# Units:
#
#   force           -> Newtons
#   transition_time -> seconds
#   hold_time       -> seconds
#
# Semantics:
#
# 1. Smoothly transition from the previous force to the target
#    force during transition_time.
#
# 2. Keep the target force constant for hold_time.
#
# 3. Continue to the next force.
#
# At the end, the script smoothly returns to [0, 0, 0] N.
#
# IMPORTANT:
# These are EXTRA / NET force commands around gravity hover.
#
# Because Mellinger already adds gravity internally:
#
#   a_ff = F / mass
#
# Hover feedforward therefore corresponds to:
#
#   F = [0, 0, 0] N
#   a_ff = [0, 0, 0] m/s^2
# ============================================================

# FORCE_SEQUENCE = [
#     ([+0.3, 0.0, 0.0], 0.00, 0.50),
#     ([-0.3, 0.0, 0.0], 0.00, 0.50),
#     ([ 0.00, 0.0, 0.0], 0.00, 0.50),
# ]

FORCE_SEQUENCE = [
    ([+1.6372172832489014, +0.0026519554667174816, +0.8400952538940427],  0.0, 0.05),
    ([+1.6377307176589966, +0.0028367023915052414, +0.8400478562805176],  0.0, 0.05),
    ([+1.1874630451202393, +0.0031924769282341003, +0.7645069798919677],  0.0, 0.05),
    ([+1.1865592002868652, +0.0036306560505181550, +0.7640752515289304],  0.0, 0.05),
    ([+0.4155618250370026, +0.0040971282869577410, +0.6847747048828123],  0.0, 0.05),
    ([+0.4093304574489594, +0.0043132258579134940, +0.6834413728210447],  0.0, 0.05),
    ([-0.5218799710273743, +0.0043710065074265000, +0.6002678117248534],  0.0, 0.05),
    ([-0.5333175063133240, +0.0040557859465479850, +0.5981303891632079],  0.0, 0.05),
    ([-1.1853652000427246, +0.0035374567378312350, +0.4416278323623655],  0.0, 0.05),
    ([-1.1957759857177734, +0.0027776004280894995, +0.4388173303100584],  0.0, 0.05),
    ([-1.3542249202728271, +0.0019253581995144486, -0.0130972424057008],  0.0, 0.05),
    ([-1.3624105453491210, +0.0010368883376941085, -0.0160097756889346],  0.0, 0.05),
    ([-1.4927613735198975, -0.0000414824899053201, -0.4390033998992922],  0.0, 0.05),
    ([-1.4990038871765137, -0.0013078856281936169, -0.4418004789855961],  0.0, 0.05),
    ([-1.6039720773696900, -0.0028293586801737547, -0.7825691023376466],  0.0, 0.05),
    ([-1.6078566312789917, -0.0047986046411097050, -0.7842618742492677],  0.0, 0.05),
]




RETURN_TO_ZERO_TIME = 0.10


# ============================================================
# Safety limits
# ============================================================

MAX_FORCE_N = 2.0

MIN_SAFE_Z = 0.50
MAX_SAFE_Z = 2.50

MAX_HORIZONTAL_DISPLACEMENT = 1.00


# ============================================================
# Mellinger position gains that will be disabled
#
# Attitude gains (kR, kw, etc.) are deliberately NOT changed.
# ============================================================

P_D_PARAMS = [
    "ctrlMel.kp_xy",
    "ctrlMel.kd_xy",
    "ctrlMel.kp_z",
    "ctrlMel.kd_z",
]

KI_PARAMS = [
    "ctrlMel.ki_xy",
    "ctrlMel.ki_z",
]

INTEGRAL_RANGE_PARAMS = [
    "ctrlMel.i_range_xy",
    "ctrlMel.i_range_z",
]

ALL_POSITION_PARAMS = P_D_PARAMS + KI_PARAMS


# ============================================================
# Global state
# ============================================================

abort_requested = False
emergency_sent = False


# ============================================================
# Ctrl+C
# ============================================================

def sigint_handler(signum, frame):
    global abort_requested

    if not abort_requested:
        print()
        print("CTRL+C -> emergency requested")

    abort_requested = True


def emergency(cf, time_helper):
    global emergency_sent

    if emergency_sent:
        return

    emergency_sent = True

    print()
    print("================================")
    print("!!! EMERGENCY STOP !!!")
    print("================================")

    cf.emergency()

    end_time = time_helper.time() + 0.5

    while rclpy.ok() and time_helper.time() < end_time:
        rclpy.spin_once(
            time_helper.node,
            timeout_sec=0.01,
        )


# ============================================================
# Parameter helpers
# ============================================================

def read_param(cf, name):
    value = cf.getParam(name)

    if not np.isfinite(value):
        raise RuntimeError(
            f"Could not read firmware parameter: {name}"
        )

    return float(value)


def read_params(cf, names):
    values = {}

    for name in names:
        value = read_param(cf, name)
        values[name] = value
        print(f"{name} = {value}")

    return values


def set_params(cf, values, time_helper, settle_time=0.30):
    for name, value in values.items():
        print(f"set {name} = {value}")
        cf.setParam(name, value)

    end_time = time_helper.time() + settle_time

    while time_helper.time() < end_time:
        rclpy.spin_once(
            time_helper.node,
            timeout_sec=0.01,
        )


def disable_position_feedback(cf, time_helper):
    print()
    print("Disabling Mellinger translational feedback...")

    set_params(
        cf,
        {name: 0.0 for name in P_D_PARAMS},
        time_helper,
    )

    set_params(
        cf,
        {name: 0.0 for name in KI_PARAMS},
        time_helper,
    )

    set_params(
        cf,
        {name: 0.0 for name in INTEGRAL_RANGE_PARAMS},
        time_helper,
    )


def restore_position_feedback(
    cf,
    time_helper,
    saved_position_params,
    saved_integral_ranges,
):
    print()
    print("Restoring Mellinger translational feedback...")

    # Restore P/D first so damping returns before integral action.
    set_params(
        cf,
        {
            name: saved_position_params[name]
            for name in P_D_PARAMS
        },
        time_helper,
    )

    set_params(
        cf,
        saved_integral_ranges,
        time_helper,
    )

    # Restore Ki last.
    set_params(
        cf,
        {
            name: saved_position_params[name]
            for name in KI_PARAMS
        },
        time_helper,
    )


# ============================================================
# Pose helpers
# ============================================================

def get_yaw(cf):
    pose = cf.get_pose()

    if not pose:
        raise RuntimeError("No pose available.")

    q = pose["orientation"]

    _, _, yaw = transforms3d.euler.quat2euler(
        [q.w, q.x, q.y, q.z]
    )

    return yaw


# ============================================================
# Seventh-order smooth interpolation
# ============================================================

def smooth_profile(t, duration):
    """Return a smooth scalar from 0 to 1."""

    if duration <= 0.0:
        return 1.0

    u = np.clip(t / duration, 0.0, 1.0)

    return (
        35.0 * u**4
        - 84.0 * u**5
        + 70.0 * u**6
        - 20.0 * u**7
    )


# ============================================================
# Force conversion
# ============================================================

def force_to_acceleration(force, firmware_mass):
    """
    Convert desired extra/net force [N] to cmdFullState
    feedforward acceleration [m/s^2].

    With calibrated Mellinger mass equal to real mass:

        a_ff = F / m

    Gravity is NOT added here. Mellinger handles gravity.
    """

    force = np.array(force, dtype=float)
    return force / firmware_mass


# ============================================================
# Safety monitoring
# ============================================================

def safety_violation(cf, reference_position):
    position = np.array(cf.get_position(), dtype=float)

    if not np.all(np.isfinite(position)):
        return "position estimate is not finite"

    if position[2] < MIN_SAFE_Z:
        return (
            f"z={position[2]:.3f} m "
            f"< MIN_SAFE_Z={MIN_SAFE_Z:.3f} m"
        )

    if position[2] > MAX_SAFE_Z:
        return (
            f"z={position[2]:.3f} m "
            f"> MAX_SAFE_Z={MAX_SAFE_Z:.3f} m"
        )

    horizontal_error = np.linalg.norm(
        position[:2] - reference_position[:2]
    )

    if horizontal_error > MAX_HORIZONTAL_DISPLACEMENT:
        return (
            "horizontal displacement="
            f"{horizontal_error:.3f} m "
            "> limit="
            f"{MAX_HORIZONTAL_DISPLACEMENT:.3f} m"
        )

    return None


# ============================================================
# Normal full-state hover
# ============================================================

def stream_hover_with_feedback(
    cf,
    time_helper,
    position,
    yaw,
    duration,
):
    zero = np.zeros(3)
    start_time = time_helper.time()

    while time_helper.time() - start_time < duration:
        if abort_requested:
            emergency(cf, time_helper)
            return "emergency"

        cf.cmdFullState(
            position,
            zero,
            zero,
            yaw,
            zero,
        )

        time_helper.sleepForRate(RATE)

    return "ok"


# ============================================================
# One force setpoint
# ============================================================

def send_force_setpoint(
    cf,
    reference_position,
    yaw,
    force,
    firmware_mass,
):
    zero = np.zeros(3)

    acceleration_ff = force_to_acceleration(
        force,
        firmware_mass,
    )

    cf.cmdFullState(
        reference_position,
        zero,
        acceleration_ff,
        yaw,
        zero,
    )


# ============================================================
# Smooth force transition
# ============================================================

def stream_force_transition(
    cf,
    time_helper,
    reference_position,
    yaw,
    force_start,
    force_target,
    firmware_mass,
    duration,
):
    force_start = np.array(force_start, dtype=float)
    force_target = np.array(force_target, dtype=float)

    if duration <= 0.0:

        send_force_setpoint(
            cf,
            reference_position,
            yaw,
            force_target,
            firmware_mass,
        )

        return "ok"

    start_time = time_helper.time()

    while True:
        if abort_requested:
            emergency(cf, time_helper)
            return "emergency"

        t = time_helper.time() - start_time

        if t >= duration:
            break

        violation = safety_violation(
            cf,
            reference_position,
        )

        if violation is not None:
            print()
            print("SAFETY LIMIT -> " + violation)
            return "safety"

        s = smooth_profile(t, duration)

        force = (
            force_start
            + (force_target - force_start) * s
        )

        send_force_setpoint(
            cf,
            reference_position,
            yaw,
            force,
            firmware_mass,
        )

        time_helper.sleepForRate(RATE)

    send_force_setpoint(
        cf,
        reference_position,
        yaw,
        force_target,
        firmware_mass,
    )

    return "ok"


# ============================================================
# Hold constant force
# ============================================================

def stream_force_hold(
    cf,
    time_helper,
    reference_position,
    yaw,
    force,
    firmware_mass,
    duration,
):
    force = np.array(force, dtype=float)
    start_time = time_helper.time()

    while time_helper.time() - start_time < duration:
        if abort_requested:
            emergency(cf, time_helper)
            return "emergency"

        violation = safety_violation(
            cf,
            reference_position,
        )

        if violation is not None:
            print()
            print("SAFETY LIMIT -> " + violation)
            return "safety"

        send_force_setpoint(
            cf,
            reference_position,
            yaw,
            force,
            firmware_mass,
        )

        time_helper.sleepForRate(RATE)

    return "ok"


# ============================================================
# Complete force sequence
# ============================================================

def execute_force_sequence(
    cf,
    time_helper,
    reference_position,
    yaw,
    firmware_mass,
):
    current_force = np.zeros(3)

    print()
    print("================================")
    print("FORCE SEQUENCE")
    print("================================")

    for index, stage in enumerate(FORCE_SEQUENCE, start=1):
        force_target, transition_time, hold_time = stage

        force_target = np.array(
            force_target,
            dtype=float,
        )

        print()
        print(
            f"Stage {index}/{len(FORCE_SEQUENCE)}"
        )
        print(f"  from force: {current_force} N")
        print(f"  to force:   {force_target} N")
        print(f"  transition: {transition_time:.3f} s")
        print(f"  hold:       {hold_time:.3f} s")
        print(
            "  target a_ff: "
            f"{force_target / firmware_mass} m/s^2"
        )

        result = stream_force_transition(
            cf,
            time_helper,
            reference_position,
            yaw,
            current_force,
            force_target,
            firmware_mass,
            transition_time,
        )

        if result != "ok":
            return result

        result = stream_force_hold(
            cf,
            time_helper,
            reference_position,
            yaw,
            force_target,
            firmware_mass,
            hold_time,
        )

        if result != "ok":
            return result

        current_force = force_target.copy()

    # Always finish at zero extra force.
    if not np.allclose(current_force, np.zeros(3)):
        print()
        print("Returning smoothly to [0, 0, 0] N...")

        result = stream_force_transition(
            cf,
            time_helper,
            reference_position,
            yaw,
            current_force,
            np.zeros(3),
            firmware_mass,
            RETURN_TO_ZERO_TIME,
        )

        if result != "ok":
            return result

    return "ok"


# ============================================================
# Configuration validation
# ============================================================

def validate_force_sequence():
    if not FORCE_SEQUENCE:
        raise RuntimeError(
            "FORCE_SEQUENCE cannot be empty."
        )

    for index, stage in enumerate(FORCE_SEQUENCE, start=1):
        if len(stage) != 3:
            raise RuntimeError(
                f"Stage {index}: expected "
                "([Fx,Fy,Fz], transition_time, hold_time)"
            )

        force, transition_time, hold_time = stage
        force = np.array(force, dtype=float)

        if force.shape != (3,):
            raise RuntimeError(
                f"Stage {index}: force must contain "
                "exactly Fx, Fy, Fz."
            )

        if not np.all(np.isfinite(force)):
            raise RuntimeError(
                f"Stage {index}: force contains "
                "non-finite values."
            )

        force_norm = np.linalg.norm(force)

        if force_norm > MAX_FORCE_N:
            raise RuntimeError(
                f"Stage {index}: |F|={force_norm:.3f} N "
                f"exceeds MAX_FORCE_N={MAX_FORCE_N:.3f} N."
            )

        if transition_time < 0.0:
            raise RuntimeError(
                f"Stage {index}: transition_time "
                "must be >= 0."
            )

        if hold_time < 0.0:
            raise RuntimeError(
                f"Stage {index}: hold_time must be >= 0."
            )


# ============================================================
# Main
# ============================================================

def main():
    raise RuntimeError(
        'Draft integration only: four-motor PWM logging fields and the firmware '
        'mapping are not configured. Use the original controller separately; '
        'this copy has not been flight-validated.'
    )
    validate_force_sequence()

    swarm = Crazyswarm()

    allcfs = swarm.allcfs
    time_helper = swarm.timeHelper

    signal.signal(
        signal.SIGINT,
        sigint_handler,
    )

    if DRONE_NAME not in allcfs.crazyfliesByName:
        raise RuntimeError(
            f"{DRONE_NAME} not found."
        )

    cf = allcfs.crazyfliesByName[DRONE_NAME]

    print()
    print("================================")
    print("Mellinger force-sequence test")
    print("================================")
    print()
    print(f"Drone: {DRONE_NAME}")
    print(f"Streaming rate: {RATE:.1f} Hz")

    # --------------------------------------------------------
    # Verify controller and calibrated platform parameters
    # --------------------------------------------------------

    controller = read_param(
        cf,
        "stabilizer.controller",
    )

    print()
    print(
        f"stabilizer.controller = {controller}"
    )

    if int(controller) != 2:
        raise RuntimeError(
            "This experiment requires "
            "stabilizer.controller = 2 (Mellinger)."
        )

    firmware_mass = read_param(
        cf,
        "ctrlMel.mass",
    )

    mass_thrust = read_param(
        cf,
        "ctrlMel.massThrust",
    )

    print()
    print(
        f"ctrlMel.mass       = {firmware_mass:.6f} kg"
    )
    print(
        f"ctrlMel.massThrust = {mass_thrust:.3f}"
    )

    if abs(firmware_mass - REAL_MASS) > MASS_TOLERANCE:
        raise RuntimeError(
            "Firmware mass does not match the calibrated "
            "physical mass. "
            f"Expected about {REAL_MASS:.3f} kg, "
            f"got {firmware_mass:.3f} kg."
        )

    if (
        abs(mass_thrust - EXPECTED_MASS_THRUST)
        > MASS_THRUST_TOLERANCE
    ):
        raise RuntimeError(
            "ctrlMel.massThrust is outside the expected "
            "calibrated range. "
            f"Expected about {EXPECTED_MASS_THRUST:.0f}, "
            f"got {mass_thrust:.0f}."
        )

    # --------------------------------------------------------
    # Save current gains
    # --------------------------------------------------------

    print()
    print("Current position gains:")

    saved_position_params = read_params(
        cf,
        ALL_POSITION_PARAMS,
    )

    print()
    print("Current integral ranges:")

    saved_integral_ranges = read_params(
        cf,
        INTEGRAL_RANGE_PARAMS,
    )

    # --------------------------------------------------------
    # Print force sequence before arming
    # --------------------------------------------------------

    print()
    print("Configured force sequence:")

    for index, (
        force,
        transition_time,
        hold_time,
    ) in enumerate(FORCE_SEQUENCE, start=1):
        force = np.array(force, dtype=float)

        print(
            f"  {index}: "
            f"F={force} N, "
            f"transition={transition_time:.3f}s, "
            f"hold={hold_time:.3f}s, "
            f"a_ff={force / firmware_mass} m/s^2"
        )

    flight_active = False
    gains_modified = False

    try:
        # ====================================================
        # Arm
        # ====================================================

        print()
        print("Arming...")

        cf.arm(True)
        flight_active = True

        time_helper.sleep(1.0)

        # ====================================================
        # Normal high-level takeoff
        # ====================================================

        print(
            f"Taking off to {TAKEOFF_HEIGHT:.2f} m..."
        )

        cf.takeoff(
            targetHeight=TAKEOFF_HEIGHT,
            duration=TAKEOFF_DURATION,
        )

        time_helper.sleep(
            TAKEOFF_DURATION + 1.0
        )

        if abort_requested:
            emergency(cf, time_helper)
            return

        # ====================================================
        # Capture post-takeoff hover reference
        # ====================================================

        hover_position = np.array(
            cf.get_position(),
            dtype=float,
        )

        hover_yaw = get_yaw(cf)

        print()
        print(
            f"Hover position: {hover_position}"
        )
        print(
            f"Hover yaw: {math.degrees(hover_yaw):.1f} deg"
        )

        # ====================================================
        # Enter cmdFullState with NORMAL feedback first
        # ====================================================

        print()
        print(
            "Entering full-state mode "
            "with normal Mellinger feedback..."
        )

        result = stream_hover_with_feedback(
            cf,
            time_helper,
            hover_position,
            hover_yaw,
            NORMAL_FULLSTATE_TIME,
        )

        if result != "ok":
            return

        # ====================================================
        # Disable translational P/I/D only
        # ====================================================

        disable_position_feedback(
            cf,
            time_helper,
        )

        gains_modified = True

        # ====================================================
        # Execute force sequence
        # ====================================================

        result = execute_force_sequence(
            cf,
            time_helper,
            hover_position,
            hover_yaw,
            firmware_mass,
        )

        # ====================================================
        # Restore feedback immediately after force experiment
        # ====================================================

        restore_position_feedback(
            cf,
            time_helper,
            saved_position_params,
            saved_integral_ranges,
        )

        gains_modified = False

        # Ctrl+C already sent emergency.
        if result == "emergency":
            return

        # ====================================================
        # Recover at CURRENT position
        # ====================================================

        recovery_position = np.array(
            cf.get_position(),
            dtype=float,
        )

        recovery_yaw = get_yaw(cf)

        print()
        print(
            f"Recovery position: {recovery_position}"
        )
        print(
            "Recovering with normal feedback..."
        )

        recovery_result = stream_hover_with_feedback(
            cf,
            time_helper,
            recovery_position,
            recovery_yaw,
            RECOVERY_TIME,
        )

        if recovery_result != "ok":
            return

        # ====================================================
        # Transition back to high-level commander
        # ====================================================

        print()
        print("Switching to high-level landing...")

        cf.notifySetpointsStop(
            remainValidMillisecs=200
        )

        # No sleep here.
        cf.land(
            targetHeight=LAND_HEIGHT,
            duration=LAND_DURATION,
        )

        time_helper.sleep(
            LAND_DURATION + 1.0
        )

        cf.arm(False)
        flight_active = False

        print()
        print("Experiment complete.")

        if result == "safety":
            print(
                "The force sequence was stopped "
                "by a safety limit."
            )

    except Exception:
        # Best effort: restore feedback before emergency.
        if gains_modified:
            print()
            print("Attempting to restore gains...")

            try:
                restore_position_feedback(
                    cf,
                    time_helper,
                    saved_position_params,
                    saved_integral_ranges,
                )
                gains_modified = False

            except Exception as restore_error:
                print("Gain restore failed:")
                print(restore_error)

        if flight_active:
            emergency(cf, time_helper)

        raise


if __name__ == "__main__":
    main()
