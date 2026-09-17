#!/usr/bin/env python3
"""Record received mocap TF samples and FullState commands without controlling flight.

time_s and cmd_age use the SAME host monotonic clock. Native TF timestamps are
stored separately. TF timestamps are not assumed to be Motive capture timestamps.
Keep the Motive export for cable markers and independent capture-time evidence.
Requires the same ROS 2 environment as the previous TF logger; no numpy dependency.
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


COMMAND_FIELDS = [
    'cmd_x', 'cmd_y', 'cmd_z', 'cmd_vx', 'cmd_vy', 'cmd_vz',
    'cmd_ax', 'cmd_ay', 'cmd_az', 'cmd_yaw', 'cmd_yaw_rate',
]
FRAME_FIELDS = [
    'time_s', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'cmd_age', 'cmd_valid',
    *COMMAND_FIELDS,
    'tf_stamp_ns', 'tf_time_s', 'receive_ros_ns', 'receive_monotonic_ns',
    'sample_index', 'tf_dt_s', 'tracking_gap', 'pose_valid', 'velocity_valid',
    'qx', 'qy', 'qz', 'qw', 'cmd_sequence', 'cmd_header_stamp_ns',
    'cmd_receive_ros_ns', 'cmd_receive_monotonic_ns',
]
EVENT_FIELDS = [
    'time_s', 'cmd_sequence', 'cmd_header_stamp_ns', 'receive_ros_ns',
    'receive_monotonic_ns', 'cmd_valid', *COMMAND_FIELDS,
    'cmd_qx', 'cmd_qy', 'cmd_qz', 'cmd_qw', 'cmd_omega_x', 'cmd_omega_y',
]


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else math.nan


def stamp_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def valid_quaternion(q):
    norm2 = sum(v*v for v in q)
    return all(math.isfinite(v) for v in q) and abs(norm2-1.) <= .05


def command_payload(msg):
    p, v, a = msg.pose.position, msg.twist.linear, msg.acc
    rotation, omega = msg.pose.orientation, msg.twist.angular
    q = [finite(getattr(rotation, k)) for k in ('x', 'y', 'z', 'w')]
    yaw = math.atan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2))
    values = [finite(getattr(obj, k)) for obj in (p, v, a) for k in ('x', 'y', 'z')]
    values += [finite(yaw), finite(omega.z)]
    extra = q + [finite(omega.x), finite(omega.y)]
    valid = all(math.isfinite(x) for x in values+extra) and valid_quaternion(q)
    return values, extra, valid


class Recorder:
    """Single-executor recording core; independent of ROS for offline tests."""

    def __init__(self, output, *, drone='cf_3', frame='cf_3_mocap', parent_frame=None,
                 expected_rate=100., cmd_timeout=.2, flush_every=20, start_ns=None):
        if not math.isfinite(expected_rate) or expected_rate <= 0:
            raise ValueError('expected-rate must be finite and positive')
        if not math.isfinite(cmd_timeout) or cmd_timeout <= 0 or flush_every < 1:
            raise ValueError('cmd-timeout and flush-every must be positive')
        self.output = Path(output)
        self.command_path = self.output.with_suffix('.commands.csv')
        self.meta_path = self.output.with_suffix('.metadata.json')
        self.start_ns = time.monotonic_ns() if start_ns is None else start_ns
        self.frame = frame.lstrip('/')
        self.parent = parent_frame.lstrip('/') if parent_frame else None
        if not self.frame:
            raise ValueError('mocap-frame cannot be empty')
        self.expected_rate, self.cmd_timeout, self.flush_every = expected_rate, cmd_timeout, flush_every
        self.command = None
        self.first_stamp = self.last_stamp = self.previous_position = None
        self.first_receive = self.last_receive = None
        self.last_callback_ns = self.start_ns
        self.closed = False
        self.stats = dict(samples=0, commands=0, duplicate_samples=0, out_of_order_samples=0,
                          bad_stamps=0, invalid_poses=0, tracking_gaps=0)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.streams = []
        # Exclusive creation: never overwrite a recording or one of its sidecars.
        try:
            for path in (self.output, self.command_path, self.meta_path):
                self.streams.append(path.open('x', encoding='utf-8', newline=''))
        except Exception:
            for stream in self.streams:
                stream.close()
            raise
        self.frames = csv.writer(self.streams[0])
        self.events = csv.writer(self.streams[1])
        self.frames.writerow(FRAME_FIELDS)
        self.events.writerow(EVENT_FIELDS)
        self.metadata = dict(schema='optitrack_tf_pva_v2', created_utc=datetime.now(timezone.utc).isoformat(),
            drone=drone, child_frame=self.frame, parent_frame=self.parent,
            expected_rate_hz=expected_rate, command_timeout_s=cmd_timeout,
            files=dict(samples=self.output.name, commands=self.command_path.name),
            time_s='Host monotonic callback time minus logger start; shared with command-event time_s.',
            cmd_age='Sample callback monotonic time minus most recent command callback monotonic time.',
            tf_stamp_ns='Unmodified TF source timestamp; its clock and capture-time meaning require publisher verification.',
            tf_time_s='TF source time relative to first accepted TF sample.',
            command_header='Publisher timestamp retained separately; not assumed synchronized with receiver.',
            command_snapshot='Latest command received when the TF callback runs, not necessarily the command at TF capture time. Use event log for reconstruction.',
            velocity='Backward difference of adjacent valid TF samples using TF timestamps; NaN across detected gaps.',
            measurement='Selected TF translation and quaternion, unchanged. No coordinate transform.',
            motive_frame_ids_available=False, cable_markers_recorded=False,
            clock_verified=False, logger_start_monotonic_ns=self.start_ns,
            status='recording')
        self.save_metadata()
        self.flush()

    def save_metadata(self):
        self.metadata.update(parent_frame=self.parent, counts=self.stats.copy(),
                             first_tf_stamp_ns=self.first_stamp, last_tf_stamp_ns=self.last_stamp)
        stream = self.streams[2]
        stream.seek(0)
        json.dump(self.metadata, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.truncate()
        stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def check_receive_time(self, receive_ns):
        if self.closed:
            raise RuntimeError('Recorder is closed')
        if receive_ns < self.last_callback_ns:
            raise RuntimeError('Host monotonic time moved backwards')
        self.last_callback_ns = receive_ns

    def on_command(self, msg, receive_ns, ros_ns):
        self.check_receive_time(receive_ns)
        values, extra, valid = command_payload(msg)
        self.stats['commands'] += 1
        seq = self.stats['commands']
        source = stamp_ns(msg.header.stamp)
        self.command = dict(values=values, valid=valid, sequence=seq, source=source,
                            receive_ns=receive_ns, ros_ns=ros_ns)
        # Every delivered callback is recorded, including repeated commands and invalid data.
        self.events.writerow([(receive_ns-self.start_ns)*1e-9, seq, source, ros_ns,
                              receive_ns, int(valid), *values, *extra])
        if seq % self.flush_every == 0:
            self.flush()

    def on_transform(self, transform, receive_ns, ros_ns):
        if transform.child_frame_id.lstrip('/') != self.frame:
            return
        self.check_receive_time(receive_ns)
        parent = transform.header.frame_id.lstrip('/')
        if not parent or (self.parent is not None and parent != self.parent):
            raise RuntimeError(f'Unexpected TF parent {parent!r}; expected {self.parent!r}. No frame conversion applied.')
        source = stamp_ns(transform.header.stamp)
        if source <= 0:
            self.stats['bad_stamps'] += 1
            return
        if self.last_stamp is not None:
            if source == self.last_stamp:
                self.stats['duplicate_samples'] += 1
                return
            if source < self.last_stamp:
                self.stats['out_of_order_samples'] += 1
                if self.last_stamp-source >= 1_000_000_000:
                    raise RuntimeError('TF timestamp jumped backwards by at least 1 s. Start a new recording after resolving the clock/publisher reset.')
                return
        if self.parent is None:
            self.parent = parent
        if self.first_stamp is None:
            self.first_stamp = source
            self.first_receive = receive_ns
        dt = (source-self.last_stamp)*1e-9 if self.last_stamp is not None else math.nan
        gap = math.isfinite(dt) and dt > 1.5/self.expected_rate
        p = tuple(finite(getattr(transform.transform.translation, k)) for k in ('x', 'y', 'z'))
        q = tuple(finite(getattr(transform.transform.rotation, k)) for k in ('x', 'y', 'z', 'w'))
        valid = all(math.isfinite(x) for x in p) and valid_quaternion(q)
        velocity_valid = valid and self.previous_position is not None and not gap and math.isfinite(dt) and dt > 0
        velocity = [(p[j]-self.previous_position[j])/dt for j in range(3)] if velocity_valid else [math.nan]*3
        self.previous_position = p if valid else None
        self.last_stamp, self.last_receive = source, receive_ns
        self.stats['samples'] += 1
        self.stats['invalid_poses'] += int(not valid)
        self.stats['tracking_gaps'] += int(gap)
        command = self.command
        age = (receive_ns-command['receive_ns'])*1e-9 if command else math.nan
        command_valid = command is not None and command['valid'] and 0 <= age <= self.cmd_timeout
        values = command['values'] if command_valid else [math.nan]*len(COMMAND_FIELDS)
        identity = [command[k] for k in ('sequence', 'source', 'ros_ns', 'receive_ns')] if command else [0, 0, 0, 0]
        self.frames.writerow([(receive_ns-self.start_ns)*1e-9, *p, *velocity, age, int(command_valid), *values,
            source, (source-self.first_stamp)*1e-9, ros_ns, receive_ns, self.stats['samples'], dt,
            int(gap), int(valid), int(velocity_valid), *q, *identity])
        if self.stats['samples'] == 1:
            self.save_metadata()
        if self.stats['samples'] % self.flush_every == 0:
            self.flush()

    def close(self, status='complete'):
        if self.closed:
            return
        self.closed = True
        self.metadata['status'] = status
        self.metadata['closed_utc'] = datetime.now(timezone.utc).isoformat()
        try:
            self.save_metadata()
        finally:
            for stream in self.streams:
                stream.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drone', default='cf_3')
    parser.add_argument('--mocap-frame', help='Default: <drone>_mocap')
    parser.add_argument('--parent-frame', help='Require this parent frame; otherwise lock to the first received parent and report it.')
    parser.add_argument('--tf-topic', default='/tf')
    parser.add_argument('--cmd-topic', help='Default: /<drone>/cmd_full_state')
    parser.add_argument('--expected-rate', '--rate', dest='expected_rate', type=float, default=100.,
                        help='Expected TF rate for gap diagnostics (--rate retained for existing launch commands); does not change the publisher rate.')
    parser.add_argument('--cmd-timeout', type=float, default=.2)
    parser.add_argument('--flush-every', type=int, default=20)
    parser.add_argument('--output', default='/workspace/logs', help='New CSV filename or output directory; existing files are never overwritten.')
    return parser.parse_known_args(argv)


def main():
    args, ros_args = parse_args()
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from crazyflie_interfaces.msg import FullState
    from tf2_msgs.msg import TFMessage

    output = Path(args.output)
    if output.suffix.lower() != '.csv':
        prefix = 'pva_log_' if Path(__file__).stem == 'experiment_logger_pva' else 'optitrack_pva_log_'
        output /= prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv'
    rclpy.init(args=ros_args)
    node = Node('optitrack_pva_logger')
    recorder = None
    status = 'complete'
    try:
        recorder = Recorder(output, drone=args.drone, frame=args.mocap_frame or args.drone+'_mocap',
            parent_frame=args.parent_frame, expected_rate=args.expected_rate,
            cmd_timeout=args.cmd_timeout, flush_every=args.flush_every)
        topic = args.cmd_topic or f'/{args.drone}/cmd_full_state'
        recorder.metadata.update(tf_topic=args.tf_topic, command_topic=topic)
        recorder.save_metadata()
        def on_command(msg):
            received = time.monotonic_ns()
            recorder.on_command(msg, received, node.get_clock().now().nanoseconds)
        def on_tf(msg):
            for transform in msg.transforms:
                if transform.child_frame_id.lstrip('/') == recorder.frame:
                    # Timestamp handling of each matching transform, including
                    # publishers that batch multiple source samples in a TFMessage.
                    received = time.monotonic_ns()
                    recorder.on_transform(transform, received, node.get_clock().now().nanoseconds)
        command_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
            reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        tf_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
            reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        subscriptions = [node.create_subscription(FullState, topic, on_command, command_qos),
                         node.create_subscription(TFMessage, args.tf_topic, on_tf, tf_qos)]
        previous = [time.monotonic_ns(), 0, 0]
        def report():
            now = time.monotonic_ns()
            elapsed = (now-previous[0])*1e-9
            frames, commands = recorder.stats['samples'], recorder.stats['commands']
            hz = (frames-previous[1])/elapsed
            cmd_hz = (commands-previous[2])/elapsed
            note = (f'TF {hz:.1f} Hz, commands {cmd_hz:.1f} Hz; '
                    f'{frames} samples; parent={recorder.parent}; gaps={recorder.stats["tracking_gaps"]}; '
                    f'old/duplicate={recorder.stats["out_of_order_samples"]}/{recorder.stats["duplicate_samples"]}')
            if frames == previous[1]:
                node.get_logger().warning('No new matching TF samples. Check the topic and child frame. '+note)
            else:
                node.get_logger().info(note)
            previous[:] = [now, frames, commands]
            recorder.save_metadata()
            recorder.flush()
        timer = node.create_timer(5., report)
        node.get_logger().info(f'Recording {recorder.frame} from {args.tf_topic}; commands from {topic}.')
        node.get_logger().info(f'Output: {output}. Keep the Motive cable-marker export. Ctrl+C stops logging.')
        # A single executor serializes command/TF callbacks and their file writes.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        status = f'error: {type(error).__name__}: {error}'
        raise
    finally:
        if recorder is not None:
            recorder.close(status)
            print(f'Saved: {recorder.output}\nCommands: {recorder.command_path}\nMetadata: {recorder.meta_path}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
