#!/usr/bin/env python3

import argparse
import csv
import math
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import rclpy
from crazyflie_interfaces.msg import FullState
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
import yaml


DEFAULT_CONFIG = "/workspace/config/cable_markers.yaml"
DEFAULT_NATNET_PATH = "/opt/optitrack_natnet"
DEFAULT_OUTPUT_DIR = "/workspace/logs"

COMMAND_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Log one OptiTrack rigid body, fixed-ID cable markers, "
            "and cmdFullState references directly to CSV."
        )
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Configuration YAML (default: {DEFAULT_CONFIG})",
    )

    parser.add_argument(
        "--natnet-path",
        default=os.environ.get(
            "OPTITRACK_NATNET_PATH",
            DEFAULT_NATNET_PATH,
        ),
        help=(
            "Directory containing NatNetClient.py, "
            "MoCapData.py and DataDescriptions.py."
        ),
    )

    parser.add_argument(
        "--cmd-topic",
        default=None,
        help=(
            "cmdFullState topic. By default it is derived from "
            "tracking.drone_name, e.g. /cf_7/cmd_full_state."
        ),
    )

    parser.add_argument(
        "--cmd-timeout",
        type=float,
        default=0.2,
        help=(
            "Maximum age [s] of a cmdFullState reference. "
            "Older commands are written as NaN (default: 0.2)."
        ),
    )

    parser.add_argument(
        "--poll-rate",
        type=float,
        default=500.0,
        help=(
            "Rate used to poll the latest decoded NatNet frame [Hz]. "
            "Only one row is written per new NatNet frame."
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
        default=10,
        help="Flush CSV every N rows (default: 10).",
    )

    parser.add_argument(
        "--max-command-history",
        type=int,
        default=500,
        help="Maximum cmdFullState messages retained for matching.",
    )

    return parser.parse_known_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(
            "Configuration root must be a YAML mapping"
        )

    natnet = config.get("natnet", {})
    tracking = config.get("tracking", {})
    markers = config.get("markers", {})

    if not markers:
        raise ValueError(
            "No cable markers configured under 'markers:'"
        )

    if "drone_rigid_body_id" not in tracking:
        raise ValueError(
            "Missing tracking.drone_rigid_body_id"
        )

    marker_ids = OrderedDict()
    seen_ids = set()

    for name, marker_id in markers.items():
        marker_id = int(marker_id)

        if marker_id in seen_ids:
            raise ValueError(
                f"Duplicate NatNet marker ID: {marker_id}"
            )

        seen_ids.add(marker_id)
        marker_ids[str(name)] = marker_id

    return {
        "server_address": str(
            natnet.get(
                "server_address",
                "192.168.0.4",
            )
        ),
        "client_address": str(
            natnet.get(
                "client_address",
                "192.168.0.32",
            )
        ),
        "use_multicast": bool(
            natnet.get(
                "use_multicast",
                True,
            )
        ),
        "drone_name": str(
            tracking.get(
                "drone_name",
                "cf_1",
            )
        ),
        "drone_rigid_body_id": int(
            tracking["drone_rigid_body_id"]
        ),
        "marker_ids": marker_ids,
    }


def import_natnet_client(natnet_path):
    path = Path(natnet_path)

    required = [
        "NatNetClient.py",
        "MoCapData.py",
        "DataDescriptions.py",
    ]

    missing = [
        filename
        for filename in required
        if not (path / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"NatNet library not found in {path}. "
            f"Missing: {', '.join(missing)}"
        )

    sys.path.insert(
        0,
        str(path),
    )

    from NatNetClient import NatNetClient

    return NatNetClient


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

    return path / f"experiment_{timestamp}.csv"


def stamp_to_ns(stamp):
    return (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
    )


def stamp_to_seconds(stamp):
    return (
        float(stamp.sec)
        + float(stamp.nanosec) * 1e-9
    )


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


def extract_natnet_frame(mocap_data):
    if mocap_data is None:
        return None

    prefix = getattr(
        mocap_data,
        "prefix_data",
        None,
    )

    suffix = getattr(
        mocap_data,
        "suffix_data",
        None,
    )

    labeled = getattr(
        mocap_data,
        "labeled_marker_data",
        None,
    )

    rigid_data = getattr(
        mocap_data,
        "rigid_body_data",
        None,
    )

    if (
        prefix is None
        or suffix is None
        or labeled is None
        or rigid_data is None
    ):
        return None

    markers = {
        int(marker.id_num): tuple(
            float(value)
            for value in marker.pos
        )
        for marker
        in labeled.labeled_marker_list
    }

    rigid_bodies = {
        int(rigid_body.id_num): rigid_body
        for rigid_body
        in rigid_data.rigid_body_list
    }

    return (
        int(prefix.frame_number),
        float(suffix.timestamp),
        markers,
        rigid_bodies,
    )


class ExperimentLogger(Node):

    def __init__(
        self,
        config,
        NatNetClient,
        natnet_path,
        cmd_topic,
        cmd_timeout,
        poll_rate,
        output_path,
        flush_every,
        max_command_history,
    ):
        super().__init__("experiment_logger")

        if cmd_timeout <= 0.0:
            raise ValueError(
                "--cmd-timeout must be > 0"
            )

        if poll_rate <= 0.0:
            raise ValueError(
                "--poll-rate must be > 0"
            )

        if flush_every < 1:
            raise ValueError(
                "--flush-every must be >= 1"
            )

        if max_command_history < 1:
            raise ValueError(
                "--max-command-history must be >= 1"
            )

        self.config = config
        self.marker_items = list(
            config["marker_ids"].items()
        )

        self.cmd_timeout_ns = int(
            cmd_timeout * 1e9
        )

        self.flush_every = flush_every
        self.max_command_history = (
            max_command_history
        )

        self.command_cache = OrderedDict()

        self.previous_drone_position = None
        self.previous_drone_motive_time = None

        self.last_frame_number = None
        self.row_count = 0

        self.output_path = output_path

        self.csv_file = open(
            output_path,
            "w",
            encoding="utf-8",
            newline="",
        )

        self.writer = csv.writer(
            self.csv_file
        )

        self.writer.writerow(
            self.make_header()
        )

        self.command_subscription = (
            self.create_subscription(
                FullState,
                cmd_topic,
                self.on_command,
                COMMAND_QOS,
            )
        )

        self.client = NatNetClient()

        self.client.set_client_address(
            config["client_address"]
        )

        self.client.set_server_address(
            config["server_address"]
        )

        self.client.set_use_multicast(
            config["use_multicast"]
        )

        if not self.client.run():
            raise RuntimeError(
                "NatNetClient could not start"
            )

        self.timer = self.create_timer(
            1.0 / poll_rate,
            self.process_latest_natnet_frame,
        )

        self.get_logger().info(
            f"Logging to {output_path}"
        )

        self.get_logger().info(
            f"NatNet server: "
            f"{config['server_address']}"
        )

        self.get_logger().info(
            f"NatNet client: "
            f"{config['client_address']}"
        )

        self.get_logger().info(
            f"Drone: {config['drone_name']} "
            f"(rigid body ID "
            f"{config['drone_rigid_body_id']})"
        )

        self.get_logger().info(
            f"Cable markers: "
            f"{len(self.marker_items)}"
        )

        self.get_logger().info(
            f"cmdFullState topic: "
            f"{cmd_topic}"
        )

        self.get_logger().info(
            "Waiting for NatNet frames..."
        )

    def destroy_node(self):
        try:
            if hasattr(self, "client"):
                self.client.shutdown()

            if hasattr(self, "csv_file"):
                self.csv_file.flush()
                self.csv_file.close()

        finally:
            super().destroy_node()

    def make_header(self):
        header = [
            "ros_time",
            "motive_time",
            "natnet_frame",

            "drone_x",
            "drone_y",
            "drone_z",

            "drone_vx",
            "drone_vy",
            "drone_vz",

            "drone_qx",
            "drone_qy",
            "drone_qz",
            "drone_qw",

            "drone_valid",

            "cmd_ros_time",
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

            "cmd_qx",
            "cmd_qy",
            "cmd_qz",
            "cmd_qw",

            "cmd_yaw",

            "cmd_omega_x",
            "cmd_omega_y",
            "cmd_omega_z",
        ]

        for name, _ in self.marker_items:
            header.extend(
                [
                    f"{name}_x",
                    f"{name}_y",
                    f"{name}_z",
                    f"{name}_valid",
                ]
            )

        return header

    def on_command(self, msg):
        command_time_ns = stamp_to_ns(
            msg.header.stamp
        )

        self.command_cache[
            command_time_ns
        ] = msg

        while (
            len(self.command_cache)
            > self.max_command_history
        ):
            self.command_cache.popitem(
                last=False
            )

    def process_latest_natnet_frame(self):
        frame = extract_natnet_frame(
            self.client.mocap_data
        )

        if frame is None:
            return

        (
            frame_number,
            motive_time,
            received_markers,
            rigid_bodies,
        ) = frame

        if frame_number == self.last_frame_number:
            return

        self.last_frame_number = (
            frame_number
        )

        # ROS timestamp assigned when this NatNet frame is consumed.
        ros_stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        measurement_time_ns = (
            stamp_to_ns(
                ros_stamp
            )
        )

        command_msg, command_age = (
            self.find_command_for_measurement(
                measurement_time_ns
            )
        )

        self.write_frame(
            ros_stamp=ros_stamp,
            motive_time=motive_time,
            frame_number=frame_number,
            received_markers=received_markers,
            rigid_bodies=rigid_bodies,
            command_msg=command_msg,
            command_age=command_age,
        )

    def find_command_for_measurement(
        self,
        measurement_time_ns,
    ):
        selected_time_ns = None
        selected_msg = None

        for command_time_ns in reversed(
            self.command_cache
        ):
            if (
                command_time_ns
                <= measurement_time_ns
            ):
                selected_time_ns = (
                    command_time_ns
                )

                selected_msg = (
                    self.command_cache[
                        command_time_ns
                    ]
                )

                break

        if selected_msg is None:
            return None, None

        age_ns = (
            measurement_time_ns
            - selected_time_ns
        )

        age = age_ns * 1e-9

        if age_ns > self.cmd_timeout_ns:
            return None, age

        return selected_msg, age

    def drone_values(
        self,
        rigid_bodies,
        motive_time,
    ):
        rigid_body = rigid_bodies.get(
            self.config[
                "drone_rigid_body_id"
            ]
        )

        tracking_valid = (
            rigid_body is not None
            and bool(
                getattr(
                    rigid_body,
                    "tracking_valid",
                    True,
                )
            )
        )

        if not tracking_valid:
            self.previous_drone_position = None
            self.previous_drone_motive_time = None

            nan = float("nan")

            return [
                nan, nan, nan,
                nan, nan, nan,
                nan, nan, nan, nan,
                0,
            ]

        position = tuple(
            float(value)
            for value in rigid_body.pos
        )

        rotation = tuple(
            float(value)
            for value in rigid_body.rot
        )

        velocity = (
            float("nan"),
            float("nan"),
            float("nan"),
        )

        if (
            self.previous_drone_position
            is not None
            and
            self.previous_drone_motive_time
            is not None
        ):
            dt = (
                motive_time
                - self.previous_drone_motive_time
            )

            if dt > 1e-9:
                velocity = tuple(
                    (
                        position[index]
                        - self.previous_drone_position[
                            index
                        ]
                    )
                    / dt
                    for index in range(3)
                )

        self.previous_drone_position = (
            position
        )

        self.previous_drone_motive_time = (
            motive_time
        )

        return [
            finite_or_nan(position[0]),
            finite_or_nan(position[1]),
            finite_or_nan(position[2]),

            finite_or_nan(velocity[0]),
            finite_or_nan(velocity[1]),
            finite_or_nan(velocity[2]),

            finite_or_nan(rotation[0]),
            finite_or_nan(rotation[1]),
            finite_or_nan(rotation[2]),
            finite_or_nan(rotation[3]),

            1,
        ]

    def command_values(
        self,
        command_msg,
        command_age,
    ):
        if command_msg is None:
            nan = float("nan")

            return [
                nan,
                (
                    finite_or_nan(
                        command_age
                    )
                    if command_age
                    is not None
                    else nan
                ),
                0,

                nan, nan, nan,
                nan, nan, nan,
                nan, nan, nan,
                nan, nan, nan, nan,
                nan,
                nan, nan, nan,
            ]

        p = command_msg.pose.position
        q = command_msg.pose.orientation
        v = command_msg.twist.linear
        omega = command_msg.twist.angular
        a = command_msg.acc

        return [
            stamp_to_seconds(
                command_msg.header.stamp
            ),

            finite_or_nan(
                command_age
            ),

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

            finite_or_nan(q.x),
            finite_or_nan(q.y),
            finite_or_nan(q.z),
            finite_or_nan(q.w),

            finite_or_nan(
                quaternion_to_yaw(q)
            ),

            finite_or_nan(omega.x),
            finite_or_nan(omega.y),
            finite_or_nan(omega.z),
        ]

    def cable_values(
        self,
        received_markers,
    ):
        values = []

        for _, marker_id in self.marker_items:
            position = (
                received_markers.get(
                    marker_id
                )
            )

            if position is None:
                values.extend(
                    [
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        0,
                    ]
                )

            else:
                values.extend(
                    [
                        finite_or_nan(
                            position[0]
                        ),
                        finite_or_nan(
                            position[1]
                        ),
                        finite_or_nan(
                            position[2]
                        ),
                        1,
                    ]
                )

        return values

    def write_frame(
        self,
        ros_stamp,
        motive_time,
        frame_number,
        received_markers,
        rigid_bodies,
        command_msg,
        command_age,
    ):
        row = [
            stamp_to_seconds(
                ros_stamp
            ),
            motive_time,
            frame_number,
        ]

        row.extend(
            self.drone_values(
                rigid_bodies,
                motive_time,
            )
        )

        row.extend(
            self.command_values(
                command_msg,
                command_age,
            )
        )

        row.extend(
            self.cable_values(
                received_markers
            )
        )

        self.writer.writerow(row)

        self.row_count += 1

        if (
            self.row_count
            % self.flush_every
            == 0
        ):
            self.csv_file.flush()

        if self.row_count == 1:
            self.get_logger().info(
                "First NatNet frame written"
            )

        if self.row_count % 1000 == 0:
            self.get_logger().info(
                f"{self.row_count} rows written"
            )


def main():
    args, ros_args = parse_args()

    config = load_config(
        args.config
    )

    NatNetClient = import_natnet_client(
        args.natnet_path
    )

    cmd_topic = (
        args.cmd_topic
        if args.cmd_topic is not None
        else (
            f"/{config['drone_name']}"
            "/cmd_full_state"
        )
    )

    output_path = resolve_output_path(
        args.output
    )

    rclpy.init(
        args=ros_args
    )

    node = ExperimentLogger(
        config=config,
        NatNetClient=NatNetClient,
        natnet_path=args.natnet_path,
        cmd_topic=cmd_topic,
        cmd_timeout=args.cmd_timeout,
        poll_rate=args.poll_rate,
        output_path=output_path,
        flush_every=args.flush_every,
        max_command_history=(
            args.max_command_history
        ),
    )

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    print(
        f"CSV saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
