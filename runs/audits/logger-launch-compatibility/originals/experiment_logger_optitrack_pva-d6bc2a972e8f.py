#!/usr/bin/env python3

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import rclpy
from crazyflie_interfaces.msg import FullState
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_msgs.msg import TFMessage


DEFAULT_OUTPUT_DIR = "/workspace/logs"

COMMAND_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

TF_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=100,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Log OptiTrack position/velocity from /tf together with "
            "the latest cmdFullState P/V/A/yaw/yaw_rate."
        )
    )
    parser.add_argument("--drone", default="cf_3")
    parser.add_argument(
        "--mocap-frame",
        default=None,
        help="Default: <drone>_mocap",
    )
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--cmd-topic", default=None)
    parser.add_argument("--cmd-timeout", type=float, default=0.20)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--flush-every", type=int, default=20)
    return parser.parse_args()


def resolve_output_path(value):
    path = Path(value)
    if path.suffix.lower() == ".csv":
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path / f"optitrack_pva_log_{timestamp}.csv"


def finite_or_nan(value):
    value = float(value)
    return value if math.isfinite(value) else float("nan")


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OptiTrackPVALogger(Node):
    def __init__(
        self,
        drone,
        mocap_frame,
        tf_topic,
        cmd_topic,
        cmd_timeout,
        output_path,
        flush_every,
    ):
        super().__init__("optitrack_pva_logger")

        if cmd_timeout <= 0.0:
            raise ValueError("--cmd-timeout must be > 0")
        if flush_every < 1:
            raise ValueError("--flush-every must be >= 1")

        self.drone = drone
        self.mocap_frame = mocap_frame.lstrip("/")
        self.cmd_timeout = float(cmd_timeout)
        self.flush_every = int(flush_every)

        self.command_msg = None
        self.command_receive_time = None

        self.previous_position = None
        self.previous_mocap_stamp_ns = None
        self.last_mocap_stamp_ns = None
        self.first_mocap_stamp_ns = None

        self.row_count = 0

        self.csv_file = open(
            output_path,
            "w",
            encoding="utf-8",
            newline="",
        )
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow(
            [
                "time_s",
                "x", "y", "z",
                "vx", "vy", "vz",
                "cmd_age", "cmd_valid",
                "cmd_x", "cmd_y", "cmd_z",
                "cmd_vx", "cmd_vy", "cmd_vz",
                "cmd_ax", "cmd_ay", "cmd_az",
                "cmd_yaw", "cmd_yaw_rate",
            ]
        )

        self.command_subscription = self.create_subscription(
            FullState,
            cmd_topic,
            self.on_command,
            COMMAND_QOS,
        )

        self.tf_subscription = self.create_subscription(
            TFMessage,
            tf_topic,
            self.on_tf,
            TF_QOS,
        )

        self.get_logger().info(f"Drone: {drone}")
        self.get_logger().info(f"OptiTrack child frame: {self.mocap_frame}")
        self.get_logger().info(f"TF topic: {tf_topic}")
        self.get_logger().info(f"cmdFullState topic: {cmd_topic}")
        self.get_logger().info(f"Logging to: {output_path}")
        self.get_logger().info("One CSV row per new OptiTrack TF sample.")

    def destroy_node(self):
        try:
            if hasattr(self, "csv_file"):
                self.csv_file.flush()
                self.csv_file.close()
        finally:
            super().destroy_node()

    def on_command(self, msg):
        self.command_msg = msg
        self.command_receive_time = self.get_clock().now().nanoseconds * 1e-9

    def command_values(self):
        nan = float("nan")

        if self.command_msg is None or self.command_receive_time is None:
            return [
                nan, 0,
                nan, nan, nan,
                nan, nan, nan,
                nan, nan, nan,
                nan, nan,
            ]

        now = self.get_clock().now().nanoseconds * 1e-9
        age = now - self.command_receive_time

        if not math.isfinite(age) or age > self.cmd_timeout:
            return [
                finite_or_nan(age), 0,
                nan, nan, nan,
                nan, nan, nan,
                nan, nan, nan,
                nan, nan,
            ]

        msg = self.command_msg
        p = msg.pose.position
        q = msg.pose.orientation
        v = msg.twist.linear
        omega = msg.twist.angular
        a = msg.acc

        return [
            finite_or_nan(age), 1,
            finite_or_nan(p.x),
            finite_or_nan(p.y),
            finite_or_nan(p.z),
            finite_or_nan(v.x),
            finite_or_nan(v.y),
            finite_or_nan(v.z),
            finite_or_nan(a.x),
            finite_or_nan(a.y),
            finite_or_nan(a.z),
            finite_or_nan(quaternion_to_yaw(q)),
            finite_or_nan(omega.z),
        ]

    def on_tf(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id.lstrip("/") != self.mocap_frame:
                continue
            self.process_mocap_transform(transform)

    def process_mocap_transform(self, transform):
        stamp_ns = stamp_to_ns(transform.header.stamp)

        if stamp_ns <= 0:
            return

        if self.last_mocap_stamp_ns == stamp_ns:
            return

        self.last_mocap_stamp_ns = stamp_ns

        if self.first_mocap_stamp_ns is None:
            self.first_mocap_stamp_ns = stamp_ns

        time_s = (stamp_ns - self.first_mocap_stamp_ns) * 1e-9

        t = transform.transform.translation
        position = (
            float(t.x),
            float(t.y),
            float(t.z),
        )

        velocity = (
            float("nan"),
            float("nan"),
            float("nan"),
        )

        if (
            self.previous_position is not None
            and self.previous_mocap_stamp_ns is not None
        ):
            dt = (stamp_ns - self.previous_mocap_stamp_ns) * 1e-9

            if dt > 1e-9:
                velocity = (
                    (position[0] - self.previous_position[0]) / dt,
                    (position[1] - self.previous_position[1]) / dt,
                    (position[2] - self.previous_position[2]) / dt,
                )

        self.previous_position = position
        self.previous_mocap_stamp_ns = stamp_ns

        self.writer.writerow(
            [
                finite_or_nan(time_s),
                finite_or_nan(position[0]),
                finite_or_nan(position[1]),
                finite_or_nan(position[2]),
                finite_or_nan(velocity[0]),
                finite_or_nan(velocity[1]),
                finite_or_nan(velocity[2]),
                *self.command_values(),
            ]
        )

        self.row_count += 1

        if self.row_count % self.flush_every == 0:
            self.csv_file.flush()

        if self.row_count == 1:
            self.get_logger().info("First OptiTrack sample written.")

        if self.row_count % 1000 == 0:
            self.get_logger().info(
                f"{self.row_count} OptiTrack samples written."
            )


def main():
    args = parse_args()

    mocap_frame = (
        args.mocap_frame
        if args.mocap_frame is not None
        else f"{args.drone}_mocap"
    )

    cmd_topic = (
        args.cmd_topic
        if args.cmd_topic is not None
        else f"/{args.drone}/cmd_full_state"
    )

    output_path = resolve_output_path(args.output)

    rclpy.init()

    node = OptiTrackPVALogger(
        drone=args.drone,
        mocap_frame=mocap_frame,
        tf_topic=args.tf_topic,
        cmd_topic=cmd_topic,
        cmd_timeout=args.cmd_timeout,
        output_path=output_path,
        flush_every=args.flush_every,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print()
    print(f"CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
