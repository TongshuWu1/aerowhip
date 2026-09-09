# Complete full-state reference playback

Flight sequence: take off, hold at the first reference position until settled,
play the entire CSV, then use the vehicle's established landing procedure.
Takeoff and the initial 10-second settling stage are not in this CSV.

The complete reference contains the unchanged force-policy whip, a smooth braking
segment, return to the selected starting hover position, and a final stationary hold.
All p/v/a values use the simulated cable-attachment reference point in world Z-up meters.
Acceleration is kinematic: do not subtract gravity or divide it by mass.
Verify the mapping to the vehicle's controlled reference point before flight.

## Replace the shortened hard-coded list

The supplied `full_state_pva.py` has only the first 20 of 25 old whip samples.
Do not copy another partial list. Use the companion `fullstate_playback.py` to load
`fullstate_30hz.csv` together with `fullstate.json`. It checks the CSV hash, sample
count, ordering, endpoint time and final hover state. Its command-line mode only
validates files and never connects to a drone:

    python fullstate_playback.py /path/to/extracted/bundle

`load_reference(directory)` returns `(rows, metadata)`. Keep position/velocity
feedback enabled. The existing send adapter can consume each row by name:

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

The export uses zero yaw and zero angular-rate feedforward. This example only maps
the already supplied cmdFullState signature; it does not resolve the vehicle's
tracked-origin/attachment transformation or validate firmware settings.

`play_reference(rows, send_sample, time_helper.time, time_helper.sleep, ...)`
uses absolute start-relative deadlines. Supply the integration's abort callback
and handle any timing/cancellation exception with its established flight recovery.
Keep runtime vehicle bounds and telemetry freshness checks in that integration.
The helper does not arm, take off, choose emergency behavior, or land.

Consume **all** rows. `whip_end_s` / legacy `cutoff_s` is a phase boundary, not the
end of playback. `total_duration_s` is the end of the entire CSV. Timestamps are on
the 30 Hz grid with extra exact phase-boundary samples where needed; follow the
timestamps rather than assuming every interval is 1/30 s. Never restart a timer
at each phase, repeat the terminal sample for another interval, or switch to
`recovery_position = cf.get_position()` at the whip boundary.

After the final row, its reference velocity and acceleration are zero at the
selected hover position. The existing script's high-level landing handover belongs
here, after the full reference, using the colleague's verified interface. Final
hold time is configurable and is not a guarantee that the real cable has settled.

## Record the actual execution

Record the trajectory hash, sample index, phase, planned time, and actual send time.
The optional `on_send` callback supplies index, phase, relative send time and lateness.
These are host send records, not vehicle execution acknowledgements. Keep the
original logger's command reception records and OptiTrack timestamps as separate clocks.
Use recorded OptiTrack positions to estimate velocities; the existing logger's
100 Hz differences of repeatedly cached positions produce velocity spikes.

The desktop 3D preview uses the source force simulation for the whip and an ideal
prescribed attachment for recovery. It does not model the real full-state tracker.
Review peak PVA, workspace excursion and final predicted cable speed. Neither the
CSV nor this transport-neutral helper establishes hardware flight readiness.
