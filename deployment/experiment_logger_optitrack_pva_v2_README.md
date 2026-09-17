# OptiTrack TF / PVA logger v2

Replace the old script under its original filename and run your existing command in the same sourced ROS 2 environment. Both original filenames are supplied in Downloads: `experiment_logger_optitrack_pva.py` and `experiment_logger_pva.py`. The previous options are retained, including `--rate 100` for the earlier PVA logger. No new launch option is required. For example:

```bash
python3 experiment_logger_optitrack_pva.py --drone cf_3 --mocap-frame cf_3_mocap --output /workspace/logs
```

Use Ctrl+C to stop. The logger prints the received TF and command rates every five seconds. A TF publisher delivering 100 Hz should produce approximately 100 samples per second; `--expected-rate 100` is a diagnostic expectation, not a publisher setting. If there are no samples, verify the child frame and `/tf` publisher. `--parent-frame world` can enforce a known parent; otherwise the first parent is reported, saved, and required for the rest of the recording. No coordinate conversion is performed.

Three files are written with a unique timestamped name:

- `optitrack_pva_log_....csv` (or `pva_log_....csv` under the earlier filename): one row per accepted matching TF sample, original XYZ/quaternion and TF timestamps, command snapshot, receive timestamps, and validity/gap flags.
- `<same stem>.commands.csv`: one event per received FullState callback, including repeats and commands received during tracking gaps.
- `<same stem>.metadata.json`: coordinate frames, timestamp meanings, counts and completion/error status.

An explicit `--output /workspace/logs/experiment_TAKE.csv` is also supported. Existing files are never overwritten. Keep all three files together, plus the original Motive export containing cable markers. The selected TF only supplies the drone pose; it does not supply Motive frame IDs or the cable markers. TF timestamps are preserved without claiming they are Motive capture timestamps.

The first twenty sample columns retain the previous CSV layout. `time_s` now uses host monotonic receive/handling time relative to logger startup, and `cmd_age` uses the same clock. Their difference reconstructs the recorded command callback time consistently. Absolute TF, ROS receive, command-header and monotonic timestamps are stored separately. Commands in a sample row are the latest received at handling time; use the command event file when reconstructing the sequence. Header timestamps from another publisher are not automatically synchronized.

Velocity uses the TF source time difference and is invalid across gaps or invalid poses. Identical/older TF samples are counted and excluded. A backward jump of at least one second stops the logger with an error, rather than silently mixing clock epochs. Tracking gaps remain gaps; the logger does not fabricate frames. Source validity is limited to the available TF values and timestamp checks because TF does not include a Motive tracking-validity flag.

Offline validation: nine tests cover original launch arguments, command timing with delayed TF, source-time velocity, command recording through tracking gaps, duplicate/out-of-order/reset handling, coordinate-frame checks, invalid poses, ROS clock jumps, 100 Hz synthetic samples and overwrite protection. Live ROS, publisher rates and hardware have not been tested on this Windows machine. Record a short test and check the rates and files before the next batch.
