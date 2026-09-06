#!/usr/bin/env python3

import signal

import numpy as np
import rclpy
import transforms3d

from crazyflie_py import Crazyswarm


DRONE_NAME = "cf_7"

RATE = 50.0

TAKEOFF_HEIGHT = 0.40
TAKEOFF_DURATION = 2.5

NORMAL_FULLSTATE_TIME = 1.0

# Start conservatively.
FEEDFORWARD_ONLY_TIME = 0.2

RECOVERY_TIME = 1.5

LAND_HEIGHT = 0.04
LAND_DURATION = 2.5


POSITION_PARAMS = [
    "ctrlMel.kp_xy",
    "ctrlMel.kd_xy",
    "ctrlMel.ki_xy",
    "ctrlMel.kp_z",
    "ctrlMel.kd_z",
    "ctrlMel.ki_z",
]

INTEGRAL_RANGE_PARAMS = [
    "ctrlMel.i_range_xy",
    "ctrlMel.i_range_z",
]


abort_requested = False


def sigint_handler(signum, frame):

    global abort_requested

    if not abort_requested:
        print("\nCTRL+C -> emergency requested")

    abort_requested = True


def emergency(cf, time_helper):

    print("\n!!! EMERGENCY STOP !!!")

    cf.emergency()

    end = time_helper.time() + 0.5

    while (
        rclpy.ok()
        and time_helper.time() < end
    ):
        rclpy.spin_once(
            time_helper.node,
            timeout_sec=0.01,
        )


def get_yaw(cf):

    pose = cf.get_pose()

    if not pose:
        raise RuntimeError(
            "No pose available."
        )

    q = pose["orientation"]

    _, _, yaw = transforms3d.euler.quat2euler(
        [
            q.w,
            q.x,
            q.y,
            q.z,
        ]
    )

    return yaw


def stream_hover(
    cf,
    time_helper,
    position,
    yaw,
    duration,
):

    zero = np.zeros(3)

    start = time_helper.time()

    while (
        time_helper.time() - start
        < duration
    ):

        if abort_requested:
            emergency(
                cf,
                time_helper,
            )
            return False

        cf.cmdFullState(
            position,
            zero,
            zero,   # a_ff = 0
            yaw,
            zero,
        )

        time_helper.sleepForRate(
            RATE
        )

    return True


def read_params(cf, names):

    result = {}

    for name in names:

        value = cf.getParam(
            name
        )

        result[name] = value

        print(
            f"{name} = {value}"
        )

    return result


def set_params(
    cf,
    values,
    time_helper,
):

    for name, value in values.items():

        print(
            f"set {name} = {value}"
        )

        cf.setParam(
            name,
            value,
        )

    # setParam() is asynchronous through ROS/Crazyswarm2.
    # Give the parameter writes a little time to propagate.
    time_helper.sleep(
        0.4
    )


def main():

    swarm = Crazyswarm()

    allcfs = swarm.allcfs
    time_helper = swarm.timeHelper

    signal.signal(
        signal.SIGINT,
        sigint_handler,
    )


    if (
        DRONE_NAME
        not in allcfs.crazyfliesByName
    ):
        raise RuntimeError(
            f"{DRONE_NAME} not found."
        )


    cf = allcfs.crazyfliesByName[
        DRONE_NAME
    ]


    print()
    print("==============================")
    print("Mellinger feedforward test")
    print("==============================")
    print()


    # --------------------------------------------------------
    # Confirm Mellinger
    # --------------------------------------------------------

    controller = cf.getParam(
        "stabilizer.controller"
    )

    print(
        f"stabilizer.controller = "
        f"{controller}"
    )


    if not np.isfinite(controller):
        raise RuntimeError(
            "Could not read stabilizer.controller."
        )

    if int(controller) != 2:
        raise RuntimeError(
            f"Mellinger required, but "
            f"stabilizer.controller={controller}"
        )


    # --------------------------------------------------------
    # Save current gains
    #
    # Do NOT hard-code defaults.
    # Restore exactly what the vehicle currently uses.
    # --------------------------------------------------------

    print()
    print("Current position gains:")

    saved_position_params = read_params(
        cf,
        POSITION_PARAMS
    )


    print()
    print("Current integral ranges:")

    saved_integral_ranges = read_params(
        cf,
        INTEGRAL_RANGE_PARAMS
    )


    # --------------------------------------------------------
    # Check mass
    # --------------------------------------------------------

    mass = cf.getParam(
        "ctrlMel.mass"
    )

    print()
    print(
        f"ctrlMel.mass = {mass} kg"
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

        time_helper.sleep(
            1.0
        )


        # ====================================================
        # Normal high-level takeoff
        # ====================================================

        print(
            f"Taking off to "
            f"{TAKEOFF_HEIGHT:.2f} m..."
        )

        cf.takeoff(
            targetHeight=TAKEOFF_HEIGHT,
            duration=TAKEOFF_DURATION,
        )

        time_helper.sleep(
            TAKEOFF_DURATION + 1.0
        )


        if abort_requested:
            emergency(
                cf,
                time_helper,
            )
            return


        # ====================================================
        # Capture actual hover state
        # ====================================================

        hover_position = np.array(
            cf.get_position(),
            dtype=float,
        )

        hover_yaw = get_yaw(
            cf
        )


        print()
        print(
            f"Hover position: "
            f"{hover_position}"
        )

        print(
            f"Hover yaw: "
            f"{np.degrees(hover_yaw):.1f} deg"
        )


        # ====================================================
        # Enter cmdFullState with NORMAL Mellinger gains
        # ====================================================

        print()
        print(
            "Entering full-state mode "
            "with normal feedback..."
        )


        if not stream_hover(
            cf,
            time_helper,
            hover_position,
            hover_yaw,
            NORMAL_FULLSTATE_TIME,
        ):
            return


        # ====================================================
        # Disable POSITION feedback only
        # ====================================================

        print()
        print(
            "Disabling Mellinger "
            "position feedback..."
        )


        zero_position_params = {
            name: 0.0
            for name in POSITION_PARAMS
        }


        set_params(
            cf,
            zero_position_params,
            time_helper,
        )


        # Prevent integral wind-up while position feedback
        # is disabled.
        set_params(
            cf,
            {
                "ctrlMel.i_range_xy": 0.0,
                "ctrlMel.i_range_z": 0.0,
            },
            time_helper,
        )


        gains_modified = True


        # ====================================================
        # FEEDFORWARD-ONLY TEST
        #
        # cmdFullState acceleration = [0,0,0]
        #
        # With position gains zero:
        #
        # F_target =
        #
        #     [0]
        #     [0]
        #     [m*g]
        #
        # Attitude feedback remains enabled.
        # ====================================================

        print()
        print("==============================")
        print("FEEDFORWARD-ONLY HOVER")
        print("==============================")

        print(
            f"Duration: "
            f"{FEEDFORWARD_ONLY_TIME:.2f} s"
        )

        print(
            "a_ff = [0, 0, 0] m/s^2"
        )


        if not stream_hover(
            cf,
            time_helper,
            hover_position,
            hover_yaw,
            FEEDFORWARD_ONLY_TIME,
        ):
            return


        # ====================================================
        # Restore normal position feedback
        # ====================================================

        print()
        print(
            "Restoring Mellinger "
            "position feedback..."
        )


        # Restore integral ranges first.
        set_params(
            cf,
            saved_integral_ranges,
            time_helper,
        )


        set_params(
            cf,
            saved_position_params,
            time_helper,
        )


        gains_modified = False


        # ====================================================
        # Recover hover with normal feedback
        # ====================================================

        print(
            "Recovering hover..."
        )


        if not stream_hover(
            cf,
            time_helper,
            hover_position,
            hover_yaw,
            RECOVERY_TIME,
        ):
            return


        # ====================================================
        # Return to high-level commander
        # ====================================================

        print()
        print(
            "Switching to high-level landing..."
        )


        cf.notifySetpointsStop(
            remainValidMillisecs=200
        )


        # No delay here.
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
        print(
            "Experiment complete."
        )


    except Exception:

        # Restore gains if possible before emergency.
        if gains_modified:

            print()
            print(
                "Attempting to restore gains..."
            )

            set_params(
                cf,
                saved_integral_ranges,
                time_helper,
            )

            set_params(
                cf,
                saved_position_params,
                time_helper,
            )


        if flight_active:

            emergency(
                cf,
                time_helper,
            )


        raise


if __name__ == "__main__":
    main()