# Legacy whip recordings: command phases and geometry

The three adaptation0 takes contain motion before and after CSV playback.
Only the maneuver was generated from the simulated force-policy trajectory.
Surrounding commands must not be called policy output or replaced with a newly
generated recovery trajectory. This revision prepares data only; no fit or PPO
training is performed.

## What the supplied controller and logs establish

`full_state_pva.py` takes off through the high-level command, streams the first
position with zero velocity/acceleration for 10 s, sends its hard-coded 20 PVA
rows, then streams a zero-V/A hold at `cf.get_position()` sampled at sequence
end. It later requests high-level landing. **That post-maneuver hold target is
the measured end position, not the starting hover position.** A moving drone
can still brake or drift while its command is a position hold.

All 20 rows are observed in each controller log in order, exactly matching the
first 20 rows of the supplied 25-row reference. The five reference rows from
0.666667 through 0.8 s were not observed. Do not append those rows, assign them
to the real drone, or score the intended later hit as though playback reached it.

| Take | First CSV observation (controller s) | First post-hold observation (s) | Observed CSV span | Approx. post-hold callback coverage |
|---|---:|---:|---:|---:|
| whip1_001 | 18.810240 | 19.480232 | 0.669992 s | 1.496 s |
| whip1_002 | 18.860148 | 19.530174 | 0.670025 s | 1.493 s |
| whip1_003 | 16.640152 | 17.300195 | 0.660043 s | 4.999 s |

The user confirmed that take 003 intentionally used a different post-hold.
This is recorded in `collection_clarifications/` and per-take review notes.
The supplied script specifies 5 s, whereas takes 001/002 show about 1.5 s of
observed publishing coverage. This difference is provenance, not grounds for
rejecting a take. Source inspection does not establish identical script
settings across takes. Do not reconstruct an extra hold from the source constant.

The logger subscribes to FullState and samples its latest cached message. It
does not record high-level takeoff/landing setpoints, mode changes, radio
acceptance or actual thrust. Cached commands may remain fresh briefly after
publishing stops. No fresh FullState observation is labelled **unobserved
command**, not hover, zero force, a return trajectory, or confirmed landing.

## New processed version

Use **Real-world Updates → Recording rounds → adaptation0**, then the latest
processed version. The historical archive decision stays on disk; the user's
later preparation request is recorded separately in
`PREPARATION_AUTHORIZATION.json`. This enables processing/review only and does
not assign new fit roles or alter historical study splits.

`execution_profile.json` binds this round to immutable copies of the supplied
controller/logger and the existing active geometry model. Processing verifies
their hashes and snapshots them with the processing code. It parses controller
literals with Python AST; it never imports or runs the flight program.
Subsequent **Process round into new version** operations reuse this explicit
profile. Other rounds do not automatically inherit the legacy interpretation.

All 1,484 OptiTrack frames in each trimmed file are retained, along with the
entire native controller table (3,323 / 2,940 / 2,930 rows). Tracking covers
different portions of each take: it ends about 3.435 / 1.675 / 4.465 s after
CSV onset. Native controller-only portions remain available without fabricated
cable positions. Original files and old processed versions remain unchanged.

New trial outputs:

- `execution_phases.json`: native phase intervals, CSV row identity and timing,
  post-hold targets, source/log discrepancies and tracking coverage.
- `phases.csv`: phase and CSV row index at every OptiTrack sample.
- `controller_phases.csv`: phases across the entire controller recording.
- `dataset.npz`: existing measurements/commands plus the fields below.
- `review.png`: measured versus commanded positions, marker availability,
  phase shading, and an overview of the complete controller timeline.

| New NPZ field | Meaning |
|---|---|
| `execution_phase` | pre-maneuver FullState hold, CSV maneuver, post-maneuver FullState hold, other FullState, or unobserved command |
| `csv_sample_index` | Exact supplied sequence row; -1 outside observed CSV playback |
| `time_from_csv_onset_s` | Aligned controller time minus first observed CSV row; includes negative pre-hover times |
| `csv_maneuver_mask` | Observed CSV command samples only; not a fit/contact/success mask |
| `non_csv_fullstate_mask` | Fresh surrounding FullState observations, separate from policy-generated commands |
| `phase_boundary_uncertain` | Tracking samples within the previous-to-first logger-row bracket around a phase transition |
| `attachment_position_m`, `attachment_valid` | Measured attachment reconstructed from tracked origin and orientation |
| `attachment_velocity_m_s` | Offline centered differences of reconstructed attachment; not causal policy input |
| `controller_native_execution_phase`, `controller_native_csv_sample_index` | Native-clock labels, retaining portions outside OptiTrack coverage |
| `controller_native_command_age_s`, `controller_native_cmd_valid` | Original logger freshness fields |
| `controller_native_logged_columns`, `controller_native_logged_values` | Entire original numeric controller table, including its original velocity estimates |

Phases are matched against all P/V/A/yaw/rate values and sequence order, rather
than using nonzero acceleration as the definition of a whip. Missing,
reordered, repeated/ambiguous sequences or command gaps fail for review.
The existing 100 ms freshness limit is applied both natively and after causal
hold onto the tracking clock. Commands are never interpolated across unknown
intervals. A hold label describes the commanded target, not settled flight.

## Timing and coordinates for later modeling

Clock alignment still compares **measured drone positions** between logs;
it never aligns desired positions to actual positions to remove tracking lag.
The offsets remain +1.175202478, -0.534629138 and +0.035524219 s. The approximately
10 Hz position update in the controller logger is not a 100 Hz fresh state
measurement. Its finite-difference velocity is retained as raw data but is not
used as the OptiTrack velocity estimate.

CSV boundaries use first observed logger rows, with roughly one 100 Hz logger
tick of observation uncertainty. `row_time - cmd_age` is an approximate logger
callback timestamp, included only as a diagnostic because the logger reads the
two times separately. It is not a vehicle actuation timestamp. The alignment
also includes measurement/logging latency; the phase boundary mask does not
represent that additional unknown uncertainty.

**Do not retrospectively translate the recorded PVA commands.** They were sent
unchanged while the external position reference was the top cf_7 tracking
origin, according to the user's confirmed no-offset setup. Reconstruct only
the measured cable boundary as `p_attachment = p_cf7 + R(t) r`. The archived
active offset is `[0.0066549972854827175, -0.01287427254333901, -0.055]` m.
Its lateral components are existing fitted geometry, not new measurements.
The flexible 63 mm attachment-to-C1 span is not part of that rigid offset.

For the redesigned model, use measured attachment plus cable markers for cable
replay, and actual logged command PVA plus measured tracking-origin motion for
the drone response. Choose phase-specific training/validation windows explicitly
after model design is agreed. Hover, maneuver and post-hold are not interchangeable
samples of force-policy execution. Contact/marker-quality masks are separate
from phase labels; no flight is discarded based on tracking error.

Historical fitted input snapshots and their legacy fitting loader retain their
original pinned version for reproducibility. The future redesigned fitting
pipeline must explicitly consume this new phase-aware version instead of
reusing the old hover-dominated window selection. All new outputs remain
`training_ready=false`; this task does not refit or certify a model.
