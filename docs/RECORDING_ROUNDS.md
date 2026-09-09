# Preparing recorded adaptation rounds

For the current inventory, temporary legacy treatment and full next-session
recording procedure, use [Data lifecycle and recording guide](DATA_LIFECYCLE_AND_RECORDING_GUIDE.md).
The current UI saves review intervals but does not apply them to the new pose
loader's fitting masks. It supports one reference CSV per round; per-take
references and explicit playback-event import remain upcoming work. Keep the
full original captures and each take's exact reference even before that support
is added. Recording preparation is not automatic acceptance for model fitting.

For adaptation0 legacy CSV-only playback, see
[the phase-aware processing contract](LEGACY_WHIP_DATA_PROCESSING.md). Its new
version retains the full take and separates pre-hold, the actually observed
20-row CSV maneuver, post-hold, and unobserved commands. The generic motion
heuristic described below remains only for rounds without an execution profile.

Open **Real-world Updates → Recording rounds**. The older force-log fitting
tools remain under **Legacy force-log tools**; full-state recordings are not
converted into force logs or passed to those tools.

`adaptation0` contains the three whip1 flights collected on 6 September 2026,
before any physical-model adaptation. The experiment ran PPO in simulation and
executed its precomputed full-state reference through cmdFullState. This is
different from deploying the policy on the vehicle or executing open-loop force
control. Round numbers describe collection history; they do not establish that
a model or policy update succeeded.

## Folder structure

```
data/adaptation_rounds/adaptation0/
  round.json                  # identity, parent, collection notes, original hashes
  raw/whip1_001/              # original OptiTrack and controller CSVs, byte-for-byte
  raw/whip1_002/
  raw/whip1_003/
  reference/                 # supplied full-state CSV for comparison
  import_context/            # workspace configs at import, NOT verified flight configs
  processed/<version>/
    processor_snapshot.py
    processing.json
    whip1_001/
      dataset.npz
      tracking.csv
      fullstate_aligned.csv
      quality.json
      review.png
    whip1_002/...
    whip1_003/...
  reviews/<revision>.json
```

Original raw files and versions are not overwritten by the application.
Processing checks raw SHA-256 hashes before reading. Failures are included in
the processing report, never silently dropped. A new processing action writes
a new version. Review changes write separate revisions tied to the exact
processed version and trial. Recordings reside inside the project workspace;
the existing `/data` Git exclusion remains, so transferring the source-code
repository alone does not transfer private experiment recordings.

## Next collection

1. Put each pair in a source folder: `experiment_<trial>.csv` for controller
   logging and `<trial>.csv` for OptiTrack. Keep unsuccessful attempts.
2. Choose the next round number and parent round. Add collection/controller
   notes. Optionally select the exact full-state reference CSV used for that
   collection; do not substitute a newly generated plan.
3. Click **Import CSV pairs**. Parsing and processing run in a background job.
4. Select the round, processed version, and trial. Inspect plots, missing samples,
   motion intervals, alignment RMS, and reference comparison.
5. If needed, enter a source-to-controller time offset for the selected trial
   and process again. Blank means automatic alignment for all trials; an override
   applies only to the selected trial in that new version.
6. Record the outcome and intended adaptation/validation role. Mark alignment
   reviewed only after inspecting it. Optional pre-contact start/end use the
   controller log's time axis; record supporting contact/intervention evidence
   in the notes. Save a review revision. These fields do not launch fitting.

The marker identity rule is exactly `cf_7` plus `cable1:c1` through `cable1:c10`.
All unlabeled markers and individual drone markers are ignored. Cable columns
are sorted numerically, not lexicographically. Numeric names alone do not prove
which endpoint is closest to the attachment; physical endpoint mapping remains
a review item. No attachment offset or axis transformation is guessed.

## Processed contract

Load `dataset.npz` with `numpy.load(path, allow_pickle=False)`. Arrays include:

| Array | Meaning |
|---|---|
| `optitrack_time_s`, `frame` | Original trimmed-file time and original frame numbers |
| `controller_time_s` | OptiTrack time plus estimated/reviewed constant offset |
| `drone_position_m` | T×3 global rigid-body position |
| `drone_quaternion_xyzw` | T×4 source orientation; body-frame mapping unverified |
| `cable_position_m` | T×10×3 named cable marker positions |
| `drone_position_valid`, `drone_orientation_valid`, `cable_valid` | Explicit finite-value masks |
| `drone_velocity_m_s`, `cable_velocity_m_s` | Offline centered differences; NaN at edges/gaps |
| `reference_fullstate` | T×11 causal-held command fields, ordered by `reference_columns` |
| `reference_valid`, `reference_age_s` | Validity/freshness at each aligned tracking sample |
| `controller_native_*` | Original controller timestamps, measured position, full-state commands and finite/valid flags |
| `cable_names`, `reference_columns` | Machine-readable labels |

`tracking.csv` contains positions, quaternion and validity flags in the same
numeric cable order. `fullstate_aligned.csv` contains aligned desired p/v/a/yaw
and validity. Missing observations remain NaN; no interpolation fills marker
gaps. Derivatives do not cross missing neighbors or gaps larger than 1.5 times
the median timestep. They are numerical offline estimates, not ground-truth
instantaneous velocities. Position trajectories are not resampled.

Commands are held causally from the latest logged row and rejected when their
age exceeds 100 ms. Native samples remain separately available. Logged desired
acceleration is kinematic feedforward, not total thrust, residual force, or
measured acceleration. No gravity subtraction or mass multiplication creates
fictional measured force.

Automatic alignment minimizes measured XYZ disagreement with one time offset,
without rotation, translation, or time warping. The result includes logging
latency; it is not verified hardware-clock synchronization. Static clips and
clips extending outside controller coverage are rejected for review. RMS here
compares two measured logs, not actual-versus-desired tracking performance.
Motion intervals are candidates detected from nonzero desired velocity or
acceleration, not certified strike/contact boundaries.

These datasets support later measured-attachment cable replay, tracking-error
analysis and model/policy improvement work. That later integration still needs
reviewed frame/attachment geometry, contact-free intervals, and explicit
training/validation separation. Prepared files remain `training_ready=false`;
this preparation step makes no fitting, performance or flight-readiness claim.
