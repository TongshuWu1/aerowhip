> Historical document archived on 9 September 2026. For current work, read [HANDOFF.md](../../../HANDOFF.md). Old running-job and launch instructions below are historical.

# Between-flight adaptation: first working version

This workflow is ready for **offline flight-log import, replay, candidate cable fitting, and simulation-only force correction**. It does not connect to a Crazyflie, change the active baseline, or release a command sequence for real flight. The Lee-controller interface and measured force envelope are still pending. The running PPO/SAC comparison uses its frozen source and configuration and is unaffected.

## Tomorrow's sequence

1. Open **Real-world Updates** in the restarted UI and create a flight-log template. This creates empty CSV headers, `trial.json`, and copies the current model/task snapshots. Ensure those snapshots are the exact versions used to generate the flown maneuver.
2. Record 100 Hz tracking, the actual timestamped sent force commands, and original controller/IMU logs. Save the initial state actually used to plan and the original planned sequence alongside them. Keep at least 0.1 seconds of tracking before launch. Log failed flights and interventions as well as successes.
3. Fill the explicit coordinate and clock information in `trial.json`. Set the frozen launch/cutoff times and review a conservative pre-contact end, before any possible target contact or intervention. Import the completed folder. Imports are immutable and separate from preliminary recordings.
4. Select a flight and click **Compare selected flight**. Measured-attachment replay checks cable prediction. Independent force-driven replay also includes aircraft-force-response mismatch. Inspect all-marker and tip errors and the attachment error before changing parameters.
5. Include whole flights marked `adaptation` and separate flights marked `validation`, then click **Fit cable drag**. The first version fits only positive cable drag in a bounded 0.5–2× neighborhood of its prior; measured geometry/mass and the audited EI/internal damping stay fixed. Validation does not train the optimizer. With no validation flight, the result is explicitly **not validated**.
6. Choose the resulting candidate in **Model to use** and replay held-out flights. Inspect candidate plots and metrics. No candidate is automatically applied to the shared training baseline.
7. **Refine selected force sequence** produces a local correction to that flight's recorded sequence, using its recorded launch state. It leaves actor weights unchanged. It saves a 20 Hz force CSV, a frozen cutoff, the initial state and model, and strict nominal hit/recovery results. A sequence for another launch must be generated from that launch's measured initial state.

The new UI is available after restarting the UI. The separate training queue can continue running.

## If ROS runs alongside the simulator

The colleague's supplied script uses ROS 2/Crazyswarm2 and onboard Mellinger.
Its force mapping and unresolved transition issues are documented in the
[controller review](../../CONTROLLER_INTERFACE_REVIEW.md). The exact installed stack
and tracking/controller log messages still need confirmation. The recording core
is transport-independent: `experimental_data.flight_recorder.FlightRecorder`
accepts callback data and produces the same importable trial folder. It needs
no ROS dependency and does not publish commands. A small adapter can call these
methods after explicitly converting frames/units:

```python
recorder = FlightRecorder(new_folder, exact_model_snapshot, exact_task_snapshot, metadata=trial_metadata)
# Start recording during hover, before launch, to retain causal prehistory.
recorder.record_tracking(tracking_stamp, drone_xyz_m, quaternion_xyzw, ten_marker_xyz_m,
                         drone_valid=valid_body, marker_valid=ten_valid_flags)
recorder.record_sent_force(command_stamp, sent_world_force_n)
recorder.record_controller(controller_stamp, telemetry_dictionary,
                           clock="controller_boot", frame="body")
# After the attempt, review event times and clock calibration; close flushes files.
recorder.close(strike_start_s=launch, planned_cutoff_s=cutoff,
               precontact_end_s=reviewed_precontact_end, contact_evidence=evidence)
```

The arguments are callback values, not prescribed ROS topics. Clock offsets and verification remain explicit in metadata. Raw ROS bags should also be retained. A concrete subscriber adapter, bag converter and command publisher are not implemented until the actual ROS version, messages, timestamps and controller contract are known. Online logging does not change the open-loop strike: collect observations during the strike, adapt between attempts.

## Data contract

The import folder contains `trial.json`, `model.json`, `task.json`, `tracking.csv`, and `commands.csv`. Extra files, including original controller/IMU and planned-command logs, are copied and hashed without modification. The current importer does **not** interpret arbitrary controller log dialects or identify actuator dynamics from them.

All positions and forces must already share a right-handed **Z-up world frame**, meters and newtons. Quaternion columns are `qx,qy,qz,qw`, body-to-world. Tracking contains the OptiTrack rigid-body origin, not a guessed center of mass. The attachment is computed as `origin + R(body_offset)` using the saved model offset. c1–c10 are ordered from the attachment towards the tip; the 63 mm first interval has its existing interpolated simulation midpoint.

`tracking.csv`:

```text
time_s,drone_x,drone_y,drone_z,qx,qy,qz,qw,drone_valid,c1_x,c1_y,c1_z,c1_valid,...,c10_x,c10_y,c10_z,c10_valid
```

The actual template writes every column, without the ellipsis. Validity flags are 0 or 1. This version requires a complete valid prehistory and pre-contact interval at the model's 100 Hz timestep (5% timestamp tolerance). It rejects gaps and invalid markers rather than filling them silently. Launch must coincide with a tracking frame. It does not read future strike frames when estimating launch velocity.

`commands.csv`:

```text
time_s,fx_n,fy_n,fz_n
```

Each timestamp begins a zero-order hold. Include the last hover command before launch and every sent command through the frozen cutoff. Store the original planned sequence separately if it differs from what was sent. A sent command is **not measured applied thrust**.

For every clock, explicitly record:

```text
common_time = source_time + recorded_offset_s
```

Event times use the common clock. Set `clock_alignment_verified` only after checking the synchronization. Constant offsets are this version's clock model; if clock drift matters, correct it upstream with documented calibration. Do not align clocks by optimizing away trajectory error.

The force convention is the existing simulator convention: **total commanded world force, including hover support**. The simulator separately applies gravity. The real Lee controller must interpret the command consistently and must not add hover support a second time. Whether the controller uses total force, acceleration, or gravity-compensated force must be verified with its code before deployment.

Set roles before importing: `adaptation` fits the model; `validation` evaluates complete held-out flights. Protected tests are refused by this development workflow. Do not split the same flight into fit and validation fragments.

## What is implemented and what remains

Implemented:

- Immutable raw-log and normalized-snapshot hashes, explicit clocks/units, causal launch estimate, length/velocity projection, pre-contact exclusion.
- Measured-attachment DDER replay and independent coupled replay; tip XYZ PNG/PDF figures and all-marker/tip/root RMSE.
- Equal-trial robust fitting with bounded log parameters, SciPy trust-region reflective least squares, and autodifferentiated Jacobian-vector products. A small prior discourages unsupported changes.
- Local differentiable force refinement with up to four XYZ knots, ±0.15 N component correction, fixed command timing/cutoff, simulation force bounds, and nominal strict task/recovery rescoring. The objective is a smooth pre-contact surrogate; task success is checked separately.

Still required before an adapted sequence is released for flight:

- Verify the actual Lee-controller frame, gravity treatment, units, saturation, timing, and handoff; map its real log columns and identify/validate force response. This code retains the instantaneous point-force assumption.
- Regenerate from the next causal launch state and run independent uncertainty validation with a documented acceptance rule. Current refinement is nominal and every export is marked `flight_ready: false`.
- Check preliminary-data prediction after fitting to avoid forgetting, then validate longer held-out flights. A good short cable replay alone does not validate reaction forces or an entire strike.
- Check directional derivatives and forward agreement over full representative strikes. Short synthetic derivative tests do not validate contact gradients or real-world improvement.
- Fit a neural residual only after repeatable held-out motion error remains. No new residual network is trained by this workflow, and no NN contribution is claimed.

Tomorrow can establish logging, initialization and replay quality with the first recorded flights. Those measurements will determine whether cable drag, aircraft force response, initial-state error, or another model limitation deserves the next update. Do not interpret an optimizer's convergence or a simulation hit as real-flight validation.

## Output locations and reproducibility

- Imported flights: `data/flight_trials/<trial_id>/`.
- UI jobs: `data/adaptation_jobs/<timestamp>-<operation>/result/`.
- Synthetic development checks: `data/adaptation_preflight/`; these are not real-flight results.
- Replay: `replay.npz`, `metrics.json`, `tip_replay.png`, `tip_replay.pdf`.
- Fitting: baseline and candidate model JSON, full objective history, per-flight before/after plots and `result.json`.
- Refinement: `candidate_forces.csv`, `initial_state.npz`, frozen model/task, objective history and `result.json`. The final force hold ends at `cutoff_s`, even if shorter than 50 ms. Contact feedback does not alter execution.

The command-line interface mirrors the page:

```powershell
.venv/Scripts/python.exe tools/adapt_flight.py template --output path/to/new_template
.venv/Scripts/python.exe tools/adapt_flight.py import --source path/to/completed_flight
.venv/Scripts/python.exe tools/adapt_flight.py replay --source data/flight_trials/flight_001 --output path/to/new_replay
.venv/Scripts/python.exe tools/adapt_flight.py fit --trials data/flight_trials/flight_001 data/flight_trials/flight_002 --output path/to/new_fit
.venv/Scripts/python.exe tools/adapt_flight.py refine --source data/flight_trials/flight_001 --model path/to/new_fit/model.json --output path/to/new_refinement
```

The CLI template starts without model/task snapshots; copy the actual planning snapshots into it. Output paths must be new so old evidence is preserved. Training does not need to be stopped for these CPU offline checks, although fitting and differentiation take additional wall time.
