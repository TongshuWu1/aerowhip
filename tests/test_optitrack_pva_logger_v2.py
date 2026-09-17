"""Offline tests of recording semantics; no ROS or flight hardware required."""
import csv
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest

from deployment.experiment_logger_optitrack_pva_v2 import Recorder, FRAME_FIELDS, EVENT_FIELDS, parse_args


def stamp(seconds):
    ns = round(seconds*1e9)
    return S(sec=ns//10**9, nanosec=ns % 10**9)


def transform(seconds, x=0., parent='world', child='cf_3_mocap'):
    return S(child_frame_id=child, header=S(stamp=stamp(seconds), frame_id=parent),
             transform=S(translation=S(x=x, y=0., z=1.), rotation=S(x=0., y=0., z=0., w=1.)))


def command(seconds=9., x=0.):
    return S(header=S(stamp=stamp(seconds)), pose=S(position=S(x=x, y=0., z=1.),
             orientation=S(x=0., y=0., z=0., w=1.)),
             twist=S(linear=S(x=0., y=0., z=0.), angular=S(x=0., y=0., z=0.)),
             acc=S(x=0., y=0., z=0.))


class LoggerTests(unittest.TestCase):
    def test_original_launch_arguments(self):
        args, remaining = parse_args(['--drone', 'cf_3', '--rate', '100',
            '--cmd-timeout', '.2', '--flush-every', '20', '--output', '/workspace/logs/experiment_M0_001.csv'])
        self.assertEqual(remaining, [])
        self.assertEqual(args.expected_rate, 100.)
        self.assertEqual(args.output, '/workspace/logs/experiment_M0_001.csv')
        args, remaining = parse_args(['--drone', 'cf_3', '--mocap-frame', 'cf_3_mocap',
            '--tf-topic', '/tf', '--cmd-topic', '/cf_3/cmd_full_state'])
        self.assertEqual(remaining, [])
        self.assertEqual(args.mocap_frame, 'cf_3_mocap')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'recording.csv'
        self.r = Recorder(self.path, start_ns=0)

    def tearDown(self):
        self.r.close()
        self.temp.cleanup()

    def rows(self, commands=False):
        self.r.flush()
        path = self.r.command_path if commands else self.path
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    def test_delayed_tf_does_not_shift_command_receipt(self):
        self.r.on_transform(transform(100.), 1_000_000_000, 100_000_000_000)
        self.r.on_command(command(), 1_990_000_000, 100_990_000_000)
        self.r.on_transform(transform(101.), 2_050_000_000, 101_050_000_000)
        row = self.rows()[-1]
        self.assertAlmostEqual(float(row['time_s'])-float(row['cmd_age']), 1.99)
        self.assertEqual(int(row['tf_stamp_ns']), 101_000_000_000)
        self.assertAlmostEqual(float(row['tf_time_s']), 1.)
        self.assertEqual(int(row['cmd_receive_ros_ns']), 100_990_000_000)

    def test_velocity_uses_source_interval_not_arrival_jitter(self):
        self.r.on_transform(transform(100., 0.), 1_000_000_000, 100_000_000_000)
        self.r.on_transform(transform(100.01, .01), 1_020_000_000, 100_020_000_000)
        row = self.rows()[-1]
        self.assertAlmostEqual(float(row['vx']), 1.)
        self.assertEqual(row['velocity_valid'], '1')

    def test_every_command_recorded_without_tf_and_stale_commands_invalid(self):
        for i in range(4):
            self.r.on_command(command(20.+i*.03, x=i), 1_000_000_000+i*30_000_000, 10+i)
        events = self.rows(commands=True)
        self.assertEqual(len(events), 4)
        self.assertEqual(list(events[0]), EVENT_FIELDS)
        self.r.on_transform(transform(100.), 2_000_000_000, 200)
        row = self.rows()[0]
        self.assertEqual(row['cmd_valid'], '0')
        self.assertTrue(math.isnan(float(row['cmd_x'])))
        self.assertEqual(row['cmd_sequence'], '4')

    def test_older_and_duplicate_samples_do_not_reenter_csv(self):
        for i, source in enumerate([100., 100.01, 100.02, 100.01, 100.02, 100.03]):
            self.r.on_transform(transform(source), 1_000_000_000+i*10_000_000, i)
        rows = self.rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(self.r.stats['out_of_order_samples'], 1)
        self.assertEqual(self.r.stats['duplicate_samples'], 1)
        with self.assertRaisesRegex(RuntimeError, 'jumped backwards'):
            self.r.on_transform(transform(90.), 2_000_000_000, 2)

    def test_frame_checks(self):
        self.r.on_transform(transform(100., child='other'), 1_000_000_000, 1)
        self.assertEqual(self.r.stats['samples'], 0)
        self.r.on_transform(transform(100.), 1_000_000_000, 1)
        with self.assertRaisesRegex(RuntimeError, 'Unexpected TF parent'):
            self.r.on_transform(transform(100.01, parent='odom'), 1_010_000_000, 2)

    def test_gap_and_invalid_pose_reset_velocity(self):
        for i, (source, x) in enumerate([(100., 0.), (100.1, .1), (100.11, math.nan), (100.12, .12), (100.13, .13)]):
            self.r.on_transform(transform(source, x), 1_000_000_000+i*100_000_000, i)
        rows = self.rows()
        self.assertEqual(rows[1]['tracking_gap'], '1')
        self.assertEqual(rows[2]['pose_valid'], '0')
        self.assertTrue(all(row['velocity_valid'] == '0' for row in rows[:4]))
        self.assertEqual(rows[4]['velocity_valid'], '1')

    def test_monotonic_clock_is_independent_of_ros_clock_jump(self):
        self.r.on_command(command(), 1_000_000_000, 100_000_000_000)
        self.r.on_transform(transform(10.), 1_020_000_000, 90_000_000_000)
        self.assertAlmostEqual(float(self.rows()[0]['cmd_age']), .02)
        self.assertEqual(self.rows()[0]['cmd_valid'], '1')

    def test_100hz_preserved_metadata_and_no_overwrite(self):
        for i in range(100):
            self.r.on_transform(transform(100.+i*.01, i*.01), 1_000_000_000+i*10_000_000, 10+i)
        rows = self.rows()
        self.assertEqual(len(rows), 100)
        self.assertEqual(list(rows[0]), FRAME_FIELDS)
        self.assertTrue(all(None not in row for row in rows))
        self.assertAlmostEqual(float(rows[-1]['tf_time_s']), .99)
        with self.assertRaises(FileExistsError):
            Recorder(self.path, start_ns=0)
        self.r.close()
        metadata = json.loads(self.r.meta_path.read_text())
        self.assertEqual(metadata['counts']['samples'], 100)
        self.assertEqual(metadata['parent_frame'], 'world')
        self.assertFalse(metadata['clock_verified'])


if __name__ == '__main__':
    unittest.main()
