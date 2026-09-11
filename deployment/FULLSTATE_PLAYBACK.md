# Recorded full-state trajectory playback

Current sequence: take off, hold at the initial reference position until settled,
execute the entire CSV, then use the vehicle's established landing procedure.

The v3 CSV comes from the **existing simulation recording**: original PPO force
whip, original PID recovery, and settled hover. It adds no new recovery controller
or synthetic braking/return path. Takeoff and initial settling are not in the CSV.
The duration is the actual recorded whip-plus-recovery duration.

## Load the entire reference

The supplied colleague script has a shortened hard-coded list. Replace that list
with the provided reader; never copy just the first rows. Validate the extracted
bundle without connecting to a vehicle:

    python fullstate_playback.py /path/to/extracted/bundle

`load_reference(directory)` returns `(rows, metadata)` after checking the CSV hash,
count, sample indices, timestamps and terminal values. V3 phase labels are `whip`,
`pid_recovery` and `hover_hold`. `whip_end_s` / `cutoff_s` is only the whip
boundary. **Consume all rows through total_duration_s.**

The reference remains in simulated cable-attachment coordinates, world Z up.
Verify its mapping to the vehicle's controlled reference point. Kinematic
accelerations receive no gravity subtraction or mass division. Keep normal
position/velocity feedback enabled. The supplied cmdFullState signature maps as:

```python
def send_sample(row):
    cf.cmdFullState(
        np.array([row[k] for k in ('px_m', 'py_m', 'pz_m')]),
        np.array([row[k] for k in ('vx_m_s', 'vy_m_s', 'vz_m_s')]),
        np.array([row[k] for k in ('ax_m_s2', 'ay_m_s2', 'az_m_s2')]),
        row['yaw_rad'],
        np.array([0.0, 0.0, row['yaw_rate_rad_s']]),
    )
```

Yaw and angular-rate feedforward are zero in these exports. This mapping does
not validate the vehicle reference point or firmware configuration.

`play_reference(rows, send_sample, time_helper.time, time_helper.sleep, ...)`
uses one absolute start-relative timeline. Supply the integration's abort callback
and preserve its runtime bounds, telemetry checks and flight recovery handling.
The helper never arms, takes off, chooses emergency behavior or lands.

Follow each timestamp: regular samples are at 30 Hz, with exact whip/end boundaries
where necessary. Do not sleep an extra interval after the terminal row. Do not
replace the remaining reference with measured-position hold at the whip boundary.

## Endpoint and landing

V3 preserves the actual recorded terminal position, velocity and acceleration.
The original simulation's hover criterion uses tolerances, so terminal values
need not be exactly the selected hover position or mathematical zeros. They are
recorded in metadata; do not silently overwrite them. PID acceleration changes
are retained by the original Hermite export convention.

The colleague's established landing transition belongs after the full reference
and after checking actual vehicle state. This offline simulation does not validate
real controller tracking or establish physical cable settling.

## Logging

Record trajectory hash, sample index, phase, planned time and actual host send
time. The optional `on_send` callback supplies index, phase, relative send time and
lateness. These are not vehicle execution acknowledgements. Keep logger callback
reception timestamps and OptiTrack timestamps separate.

Use timestamped OptiTrack positions for velocity estimation; the original
logger differences repeatedly cached positions at 100 Hz and produces spikes.

Historical v2 polynomial-recovery bundles remain readable for provenance, but
they are not the current export workflow.
