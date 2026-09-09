> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Full-state reference testing

## Current workflow: unchanged whip with gentle CSV recovery

The latest user request replaces the abrupt PID recovery in the CSV while preserving the whip. The Full-state page now exports `gentle_recovery_fullstate_v4`: a continuous-acceleration transition, smooth braking to rest, at least 10 seconds to return, and 3 seconds at stationary hover. The return duration increases if needed to keep its analytic peak acceleration below 0.3 m/s² and speed below 0.4 m/s. These are reference-shaping choices, not measured vehicle limits. Exact zero tilt cannot produce a horizontal return; the slow return is nearly level, while initial braking still needs tilt.

The existing GPU rehearsal and force policy are unchanged. After the original source rehearsal finishes, the exporter preserves whip rows and replaces only the CSV recovery using analytic position/velocity/acceleration, with no second cable rollout. **The live 3D view still shows the source PID recovery. Use Open CSV trajectory plot to inspect the new recovery reference.** Cable and attitude tracking of the new recovery are not simulated. New CSV cutoffs explicitly use the left-hand acceleration derivative from the final whip interval; timestamp rounding must not select the first PID acceleration.

For the supplied CSV, the separate corrected bundle is `policies/PPO-fullstate-gentle-recovery-20260907/`: all original 25 whip rows are byte-identical, braking takes 1.39 s, return 10 s, hold 3 s, total 15.192 s / 457 samples. First recovery reference-direction change is 2.96° instead of 116.39°. Return feedforward tilt peaks at 0.45°; braking reaches x=-1.334 m before returning to [0,0,1.5]. This requires that space. The old full CSV is retained as `recorded_pid_reference.csv` for provenance only; use `fullstate_30hz.csv` for the corrected reference. Original Downloads files and original rehearsal bundles were not modified.

21 targeted tests passed, including analytic derivatives, phase continuity, cutoff-side selection at multiple clock origins, exact supplied-whip preservation, playback and UI. Two end-to-end Windows/RTX 4080 UI runs completed; the final one after the cutoff correction is `runs/rehearsals/20260908-000327-330343`. This validates export/runtime integration, not vehicle tracking. No policy retraining, controller/logger edits or flight commands were performed. Restart the app for new exports.

## Historical v3 workflow: export the original rehearsal unchanged

The user clarified that this is an **export-only change**. The simulation, PPO
force plan, normal PID recovery and settling behavior remain unchanged. The added
braking/return/hold controls and second recovery simulation are no longer used.

1. Open **Full-state 30 Hz**, select the PPO and start rehearsal.
2. Wait for the original 10-second settled hover and normal force-plan generation.
3. Click **Execute rehearsal & create CSV**. Watch the same GPU simulation execute
   the whip and its original PID recovery to hover.
4. Once that rehearsal finishes, the complete CSV appears. Save the full-state bundle.

The exporter reads existing recorded states; it does not simulate anything. The
CSV starts at the recorded strike launch and ends after original recovery/settling.
Takeoff, initial hold and landing remain external. Its duration depends on that
recorded recovery, rather than the superseded fixed 7.3-second construction.

Schema `recorded_rehearsal_fullstate_v3` includes `whip`, `pid_recovery`, and
`hover_hold` phase labels. `fullstate_source.npz` contains the exact recorded node
positions/velocities for that interval. Position/velocity resampling and analytic
acceleration use the original cubic Hermite convention. Actual terminal p/v/a are
preserved, not replaced with ideal zeros. PID accelerations can change at updates;
export does not smooth or redesign that controller.

The reader checks all row counts, hash, timestamps and recorded terminal values.
`whip_end_s` is NOT the end of playback; use `total_duration_s` and consume all rows.
A saved original GPU rehearsal exported in about 0.048 seconds: 613 rows / 20.4 s.
Stop/restart an already running app to load this correction.

Everything below describes superseded exports retained as history.

## Complete export — 7 September 2026

The current **Full-state 30 Hz** page exports **whip + recovery + final hover**.
The separate force-control Testing tab was removed at the user's request; use
Full-state 30 Hz for the experiment workflow.
The vehicle performs takeoff and initial settling before the CSV, then lands after
the entire CSV. PPO weights and historical force exports are unchanged.

1. Select the saved PPO, hover position and target.
2. Set **Braking**, **Return to hover**, and **Final hold** durations. Defaults are
   0.5, 3.0 and 3.0 seconds. Edits persist in `config/fullstate_recovery.json` and
   are recorded in each new experiment setup. They do not rewrite existing exports.
3. Start rehearsal. After initial hover, preparation preserves the full original
   force-driven whip and constructs a recovery matching its terminal p/v/a. A
   braking segment ends at rest, followed by a smooth return and stationary hold.
4. Inspect the entire CSV duration, phase labels, peak reference speed/acceleration
   and residual cable speed. **Preview complete trajectory** shows the saved whip
   and exported recovery, with orange whip trails and blue recovery/hold trails.
5. **Save full-state bundle** writes a new ZIP for transfer.

The complete export uses schema `complete_fullstate_reference_v2`:

- `fullstate_30hz.csv`: original twelve numeric columns plus `sample_index` and
  `phase` (`whip`, `recovery_brake`, `recovery_return`, `hover_hold`). All original
  whip rows are preserved. Recovery follows the global 30 Hz grid, with additional
  exact phase-boundary timestamps when necessary.
- `fullstate_source.npz`: unchanged source whip states.
- `fullstate_preview.npz`: complete cable states, with recovery generated by
  prescribing the attachment trajectory in the cable model. This is ideal tracking,
  not a simulation of the real full-state controller.
- `fullstate.json`: source hashes, whip boundary, total duration, recovery polynomial
  coefficients/settings, endpoint, phase boundaries and numerical diagnostics.
- `fullstate_playback.py` and `FULLSTATE_PLAYBACK.md`: a transport-neutral reader,
  deadline player and integration instructions. They do not arm or fly a vehicle.

The companion reader rejects truncated or changed CSVs by hash and sample count.
`cutoff_s` and `whip_end_s` identify the whip boundary; **`total_duration_s` is the
end of playback**. No extra 1/30-second wait is added after the final row.
The final row is at the selected hover position with zero reference velocity and
acceleration. Final hold duration does not establish that the cable has settled.

Reference positions remain cable-attachment coordinates in world Z up. Kinematic
accelerations receive no gravity subtraction or mass division. The colleague must
verify the mapping to the vehicle's controlled reference point and motion limits.
The supplied hard-coded `full_state_pva.py` must consume this full CSV instead of
its shortened list; it was not edited or run against hardware.

Validation on Windows/RTX 4080: the preserved selected 0.80 s plan produced 220
rows over 7.30 s. All 25 original whip rows match exactly. Recovery ended at
`[0,0,1.5]` m with zero reference v/a; max cable-length error was below 5e-16 m.
The final predicted maximum cable speed was 0.144 m/s, so the default final hold
does not imply a motionless cable. Eager GPU preparation took about 122 seconds;
this is offline preparation, not a real-time planner. Native VTK preview and Qt
table/phase playback were inspected. Audit: `runs/audits/20260907-complete-fullstate`.

## Historical strike-only export

The description below applies to earlier v1 files, which remain preserved. Newly
generated full-state UI references use the complete v2 workflow above.

The **Full-state 30 Hz** page is separate from the original **Testing** force page.
Select a saved PPO, set the initial attachment position and target, and start a
rehearsal. Normal simulated hover settles before planning. The existing 20 Hz
PPO generates its frozen force plan; a separate GPU physical rollout executes
that exact plan from its saved initial state. No training or adaptation occurs.

The table presents `fullstate_30hz.csv` before execution. **Preview source
maneuver** runs the existing force-driven simulation with hover/recovery. It is
not a simulation of Mellinger tracking or a 30 Hz force controller. The CSV has
30 Hz reference samples; the source physics and policy clocks are unchanged.
**Save full-state bundle** writes a new ZIP containing the CSV, metadata, source
trajectory, and force-plan provenance. Existing bundles are not overwritten.

Files are saved under the new rehearsal's `plan_NNN` folder:

- `fullstate_30hz.csv`: time, world XYZ position (m), velocity (m/s), kinematic
  acceleration (m/s²), constant zero yaw (rad), and zero yaw rate (rad/s).
- `fullstate_source.npz`: all cable positions and velocities at source physics
  times, including initial state and exact terminal state.
- `fullstate.json`: coordinate/acceleration conventions, source hash, cutoff,
  peak reference speed/acceleration, and limitations.

Position and velocity are interpolated with a piecewise cubic Hermite curve.
Velocity and acceleration are its analytic derivatives, not independent linear
interpolations. Acceleration may jump at source-physics knots. This describes a
numerical reference; it does not impose vehicle acceleration/jerk/tilt limits.
The last row is the exact terminal boundary, not another held action. A 0.80 s
maneuver has 24 regular command intervals and 25 rows. A cutoff between 30 Hz
ticks has a shorter final interval; do not round up or repeat the sequence.

For a physical adapter, keep normal position/velocity feedback enabled and
schedule against elapsed time from one monotonic start. Do not subtract gravity
or divide these kinematic accelerations by mass. Verify the installed firmware's
loaded-vehicle feedforward and gravity convention. This is different from the
older `controller_acceleration.csv`, which converts residual force to an input
for feedback-disabled force execution.

The reference point is the simulated cable attachment, not automatically the
tracked vehicle origin or center of mass. The model has no vehicle attitude or
body-rate state. Zero yaw is an explicit heading choice, not an inferred initial
heading; yaw rate is not a prediction of roll/pitch body rates. Vehicle frame,
point-offset transformation, cmdFullState body-rate arguments, and a continuous
recovery from terminal position/velocity require the actual vehicle interface.
No ROS sender or physical flight validation is included. The preview's recovery
is not appended to the exported strike CSV.
