#!/usr/bin/env python3

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy

from crazyflie_interfaces.msg import FullState
from crazyflie_py import Crazyswarm
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


DEFAULT_OUTPUT_DIR = "/workspace/logs"

COMMAND_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Log Crazyflie position/velocity from Crazyswarm together "
            "with the latest cmdFullState P/V/A/yaw/yaw_rate."
        )
    )

    parser.add_argument(
        "--drone",
        default="cf_7",
        help="Crazyflie name (default: cf_7).",
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=100.0,
        help="Logger sampling rate in Hz (default: 100).",
    )

    parser.add_argument(
        "--cmd-timeout",
        type=float,
        default=0.20,
        help=(
            "Maximum age [s] of the latest cmdFullState command "
            "before command fields are written as NaN (default: 0.20)."
        ),
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "CSV filename or output directory. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="Flush CSV every N rows (default: 20).",
    )

    return parser.parse_args()


def resolve_output_path(value):
    path = Path(value)

    if path.suffix.lower() == ".csv":
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        return path

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return path / f"pva_log_{timestamp}.csv"


def finite_or_nan(value):
    value = float(value)

    if math.isfinite(value):
        return value

    return float("nan")


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (
        q.w * q.z
        + q.x * q.y
    )

    cosy_cosp = 1.0 - 2.0 * (
        q.y * q.y
        + q.z * q.z
    )

    return math.atan2(
        siny_cosp,
        cosy_cosp,
    )


class CommandCache:
    def __init__(
        self,
        time_helper,
    ):
        self.time_helper = time_helper

        self.msg = None
        self.receive_time = None

    def callback(
        self,
        msg,
    ):
        self.msg = msg
        self.receive_time = (
            self.time_helper.time()
        )

    def values(
        self,
        timeout,
    ):
        """
        Returns:
            cmd_age
            cmd_valid
            cmd_x, cmd_y, cmd_z
            cmd_vx, cmd_vy, cmd_vz
            cmd_ax, cmd_ay, cmd_az
            cmd_yaw
            cmd_yaw_rate
        """

        nan = float("nan")

        if (
            self.msg is None
            or self.receive_time is None
        ):
            return [
                nan,
                0,
                nan, nan, nan,
                nan, nan, nan,
                nan, nan, nan,
                nan,
                nan,
            ]

        age = (
            self.time_helper.time()
            - self.receive_time
        )

        if (
            not math.isfinite(age)
            or age > timeout
        ):
            return [
                finite_or_nan(age),
                0,
                nan, nan, nan,
                nan, nan, nan,
                nan, nan, nan,
                nan,
                nan,
            ]

        msg = self.msg

        p = msg.pose.position
        q = msg.pose.orientation
        v = msg.twist.linear
        omega = msg.twist.angular
        a = msg.acc

        return [
            finite_or_nan(age),
            1,

            finite_or_nan(p.x),
            finite_or_nan(p.y),
            finite_or_nan(p.z),

            finite_or_nan(v.x),
            finite_or_nan(v.y),
            finite_or_nan(v.z),

            finite_or_nan(a.x),
            finite_or_nan(a.y),
            finite_or_nan(a.z),

            finite_or_nan(
                quaternion_to_yaw(q)
            ),

            finite_or_nan(
                omega.z
            ),
        ]


def main():
    args = parse_args()

    if args.rate <= 0.0:
        raise ValueError(
            "--rate must be > 0"
        )

    if args.cmd_timeout <= 0.0:
        raise ValueError(
            "--cmd-timeout must be > 0"
        )

    if args.flush_every < 1:
        raise ValueError(
            "--flush-every must be >= 1"
        )

    output_path = resolve_output_path(
        args.output
    )

    swarm = Crazyswarm()

    allcfs = swarm.allcfs
    time_helper = swarm.timeHelper

    if (
        args.drone
        not in allcfs.crazyfliesByName
    ):
        raise RuntimeError(
            f"{args.drone} not found."
        )

    cf = allcfs.crazyfliesByName[
        args.drone
    ]

    cmd_topic = (
        f"/{args.drone}/cmd_full_state"
    )

    command_cache = CommandCache(
        time_helper
    )

    command_subscription = (
        time_helper.node.create_subscription(
            FullState,
            cmd_topic,
            command_cache.callback,
            COMMAND_QOS,
        )
    )

    # Keep the subscription alive.
    _ = command_subscription

    csv_file = open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    )

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            # Logger time.
            "time_s",

            # Measured state from Crazyswarm.
            "x",
            "y",
            "z",

            "vx",
            "vy",
            "vz",

            # Latest cmdFullState.
            "cmd_age",
            "cmd_valid",

            "cmd_x",
            "cmd_y",
            "cmd_z",

            "cmd_vx",
            "cmd_vy",
            "cmd_vz",

            "cmd_ax",
            "cmd_ay",
            "cmd_az",

            "cmd_yaw",
            "cmd_yaw_rate",
        ]
    )

    print()
    print("================================")
    print("Crazyswarm PVA logger")
    print("================================")
    print()
    print(
        f"Drone:      {args.drone}"
    )
    print(
        f"State:      cf.get_position()"
    )
    print(
        f"cmd topic:  {cmd_topic}"
    )
    print(
        f"Rate:       {args.rate:.1f} Hz"
    )
    print(
        f"Output:     {output_path}"
    )
    print()
    print(
        "Velocity vx/vy/vz is computed from consecutive "
        "Crazyswarm position samples."
    )
    print(
        "Press Ctrl+C to stop logging."
    )
    print()

    previous_position = None
    previous_time = None

    row_count = 0

    start_time = (
        time_helper.time()
    )

    try:
        while rclpy.ok():
            # Process /cmd_full_state callbacks.
            rclpy.spin_once(
                time_helper.node,
                timeout_sec=0.0,
            )

            now = (
                time_helper.time()
            )

            position = np.asarray(
                cf.get_position(),
                dtype=float,
            )

            if (
                position.shape != (3,)
                or not np.all(
                    np.isfinite(position)
                )
            ):
                x = y = z = float("nan")
                vx = vy = vz = float("nan")

                previous_position = None
                previous_time = None

            else:
                x, y, z = position

                velocity = np.array(
                    [
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                    dtype=float,
                )

                if (
                    previous_position
                    is not None
                    and previous_time
                    is not None
                ):
                    dt = (
                        now
                        - previous_time
                    )

                    if dt > 1e-9:
                        velocity = (
                            position
                            - previous_position
                        ) / dt

                vx, vy, vz = velocity

                previous_position = (
                    position.copy()
                )

                previous_time = now

            cmd_values = (
                command_cache.values(
                    args.cmd_timeout
                )
            )

            writer.writerow(
                [
                    now - start_time,

                    finite_or_nan(x),
                    finite_or_nan(y),
                    finite_or_nan(z),

                    finite_or_nan(vx),
                    finite_or_nan(vy),
                    finite_or_nan(vz),

                    *cmd_values,
                ]
            )

            row_count += 1

            if (
                row_count
                % args.flush_every
                == 0
            ):
                csv_file.flush()

            if (
                row_count
                % int(max(1, args.rate * 10))
                == 0
            ):
                print(
                    f"{row_count} rows written"
                )

            time_helper.sleepForRate(
                args.rate
            )

    except KeyboardInterrupt:
        pass

    finally:
        csv_file.flush()
        csv_file.close()

    print()
    print(
        f"CSV saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
