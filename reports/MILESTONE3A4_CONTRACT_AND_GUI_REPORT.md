# Milestone 3A.4 — Command/Data Contract and Production GUI Report

Date: 2026-08-28  
Repository: `C:\Users\wts28\Documents\PHD\particle_filter_cable_project`

## Outcome

Milestone 3A.4 is complete.

The command and timeline contracts both passed. The accepted simulator already used the correct effective Crazyflie abstraction, and the four authoritative processed takes already followed the manually trimmed Motive exports exactly. Consequently, no model retraining or scientific-data replacement occurred.

The GUI now has three production tabs:

1. Simulator
2. Dataset
3. Identification

The next physical take can be assigned the protected `Untouched Test` role and evaluated through one immutable Model A versus Model C backend. Stage B and Stage C remain locked. No EI/Cb fit, joint fit, MPPI run, model change, or residual experiment occurred.

## A. Command contract audit

Status: **PASS — semantic/metadata finalization only**.

The production command-to-UAV path already does the following:

- externally supplied command fields are world-frame position, velocity and acceleration plus yaw;
- `a_cmd` is used explicitly in `a_nom`;
- the recorded command quaternion is used only to extract yaw;
- desired roll/pitch are generated from `a_nom + gravity` and yaw;
- generic omega input is body-frame rad/s;
- current `omega_cmd` is identically zero;
- effective attitude parameters describe the onboard-generated attitude response, not tracking of an externally commanded roll/pitch quaternion.

The authoritative implementation remains [`simulator/uav/model.py`](../simulator/uav/model.py) and [`simulator/uav/quaternion.py`](../simulator/uav/quaternion.py). No numerical equation changed.

The processed-data contract now states:

| Scientific name | Existing immutable NPZ field |
|---|---|
| `p_cmd_m` | `command_position_m` |
| `v_cmd_mps` | `command_velocity_mps` |
| `a_cmd_mps2` | `command_acceleration_mps2` |
| `yaw_cmd_rad` | `command_yaw` |
| q_cmd provenance | `command_orientation_xyzw` |
| omega provenance | `command_angular_velocity` |

Aliases are metadata/loader semantics, not duplicate NPZ arrays. This preserves every processed hash.

Stored hardware/controller provenance:

- controller: Mellinger;
- estimator: Kalman;
- robot: `bolt_3in_2s`;
- motion capture: vendor tracking;
- firmware: stock/unmodified, exact version not archived.

The simulator remains an effective command-to-motion model. No firmware, motor, inertia, thrust or controller-internal parameters were added.

## B. Timeline contract audit

Status: **PASS — Motive is exactly authoritative**.

The processor builds scientific arrays only on Motive frames and defines:

`time_s = motive.source_time_s - motive.source_time_s[0]`.

For every audited take, stored time, source time and frame arrays exactly equal a fresh parse of the manually trimmed Motive CSV. The maximum timestamp difference is zero.

| Take | Motive range [s] | Duration [s] | Processed range [s] | Command coverage | Logger pre-history [s] | Logger post-history excluded [s] |
|---|---:|---:|---:|---:|---:|---:|
| osc_001 | 12.74–32.04 | 19.30 | 0.00–19.30 | 92.49% | 11.19 | 7.47 |
| fig8_001 | 6.39–49.23 | 42.84 | 0.00–42.84 | 94.52% | 10.27 | 6.82 |
| fig8_002 | 9.49–39.83 | 30.34 | 0.00–30.34 | 91.73% | 9.37 | 13.23 |
| fig8_003 | 9.58–66.87 | 57.29 | 0.00–57.29 | 98.46% | 10.01 | 16.42 |

Logger-only pre-history is recorded for causal initialization and never extends scientific duration. Logger-only post-history is explicitly excluded. Playback, manual segments, quality masks, windows and duration statistics all operate on the Motive-defined `time_s` array.

Metadata now explicitly records:

- `scientific_timeline_source = motive_manual_trim`;
- Motive start/end/duration;
- logger file and synchronized coverage;
- command coverage within the Motive interval;
- causal logger pre-history;
- excluded logger post-history.

The reusable audit/migration backend is [`experimental_data/contracts.py`](../experimental_data/contracts.py). It refuses metadata migration unless exact Motive identity is proven and rechecks the NPZ hash afterward.

During the final idempotence check, four additional raw pairs present in `data/raw_takes` were processed through the same production contract: `fig8vertical_001`, `fig8vertical_002`, `osc_002`, and `osc_003`. They were automatically added as disabled `Ignore` takes. They were not assigned to Training, Validation, or Untouched Test, and no fit/evaluation used them.

## C. Freeze integrity

The pre-untouched-test freeze remains valid.

| Artifact | Verified value |
|---|---|
| residual weight SHA-256 | `8feb4b18ce130641e65fa2bef23801fd3e9b5febfa95277faa550ad32d88c08b` |
| normalization canonical SHA-256 | `4649a6ba1794ceceab75b1917b09667dafd42f5fa76cc9c288e567a860b8ba95` |
| evaluation protocol canonical SHA-256 | `33d14d40ff285c8398c322fa1c0b79323398ebbbbebde23b8d8016a1b70f51c9` |
| original four processed NPZ hashes | unchanged |

No file inside `data/model_freezes/UAV_MODEL_FREEZE_PRE_UNTOUCHED_TEST` was rewritten.

Persistent scientific state is stored in [`data/scientific_state.json`](../data/scientific_state.json), currently:

- `UAV_BASELINE_COMPLETE`;
- `UAV_RESIDUAL_PROVISIONAL`;
- `UAV_PRETEST_FROZEN`;
- `UNTOUCHED_TEST_PENDING`;
- no accepted UAV boundary model;
- Stage B locked pending explicit user decision;
- Stage C locked;
- MPPI not started.

## D. Dataset GUI

The old Takes & Dataset panel was replaced by a production master/detail Dataset tab.

Implemented workflow:

- compact take table with Take, Role, Motive Duration, Command Coverage, UAV Valid, Cable Valid, Accepted Windows and Status;
- roles: Training, Provisional Validation, protected Untouched Test, and Ignore;
- selected-take summary with explicit “Motive manual trim” duration provenance;
- controller/estimator and p/v/a/yaw command provenance;
- one coherent Motive-time timeline with command, UAV, cable, Use/Exclude and accepted-window tracks;
- current playback cursor on that scientific timeline;
- manual Use/Exclude segments as secondary masks inside the Motive trim;
- processed-data-only 3D playback with play/pause, scrubber, current time and speed;
- minimal processing controls: Refresh/Scan, Process Selected, Process All Changed and Save Roles/Segments;
- processing through the existing CLI/backend in `QProcess`, outside the Qt UI thread;
- no raw-CSV parser in the GUI.

The protected role is implemented in [`fitting/dataset.py`](../fitting/dataset.py). `Dataset.fitting_takes` includes only Training and Provisional Validation; `Dataset.untouched_test` is separate and cannot enter standard fitting/normalization paths.

## E. Identification GUI

The old experiment-rerun-oriented Fit & Validate panel was replaced by a frozen-state Identification tab.

Implemented:

- persistent stage strip for UAV Physics, UAV Residual, Untouched Test, Stage B, Stage C and MPPI;
- frozen five-parameter physics baseline;
- residual history/features/architecture/freeze/hash summary;
- concise command semantics preventing q_cmd/full-attitude confusion;
- Milestone 3A.3 transfer card, including the `osc_001` orientation warning and `MIXED TRANSFER` classification;
- Model A / delay-ablation / Model C comparison selector;
- saved measured-versus-predicted position/orientation overlay with window selection and RMSE summary;
- artifact browser showing stage, model, dataset/freeze, status, key metric and path;
- protected Untouched Test preflight and evaluation status;
- explicit post-evaluation user decisions: accept Physics + Residual or retain Physics-Only;
- no automatic acceptance threshold and no automatic Stage B unlock.

## F. Frozen untouched-test workflow

The shared backend is [`fitting/untouched_evaluation.py`](../fitting/untouched_evaluation.py). The CLI entry point is [`run_frozen_uav_evaluation.py`](../run_frozen_uav_evaluation.py). The GUI invokes that same CLI/backend rather than duplicating evaluation equations.

First evaluation behavior:

1. require exactly one enabled `Untouched Test` take;
2. verify every frozen-file hash, residual weight hash, normalization hash and protocol hash;
3. use frozen fit configuration, normalization, parameters and weights;
4. evaluate exactly Model A and Model C on frozen residual-eligible windows;
5. save aggregate, lead-time, X/Y/Z and residual-acceleration metrics plus full prediction arrays;
6. atomically create a permanent artifact keyed by take, weight hash and protocol hash;
7. mark the scientific state as `UNTOUCHED_TEST_COMPLETE` without selecting a model.

Subsequent normal calls load the permanent result. A diagnostic replay is possible only after the original result and is written separately as `REPLAY_OF_FROZEN_EVALUATION`; it cannot overwrite or modify the original.

No untouched test was assigned or evaluated in this milestone.

## G. Stage B GUI readiness

Stage B now has a serious preflight section while execution remains locked.

It shows:

- purpose: identify EI and Cb;
- current EI/Cb initial values and configured bounds;
- fitting horizon and stride;
- Training takes only;
- accepted windows and cable coverage;
- selected UAV boundary model or the explicit absence of one;
- lock reason.

Stage B execution is intentionally disabled in Milestone 3A.4. Stage C is visible as Joint Refinement and remains locked. No EI/Cb or joint optimization was run.

## H. Verification

| Check | Result |
|---|---|
| command-semantics contract | PASS |
| exact Motive timeline, four authoritative takes | PASS, max Δt = 0 |
| processed hashes preserved | PASS |
| frozen residual/protocol/normalization hashes | PASS |
| Untouched Test excluded from fitting roles | PASS |
| CLI/GUI shared frozen evaluator routing | PASS |
| Python compilation | PASS |
| `git diff --check` | PASS; only pre-existing line-ending warnings in unrelated `.gitignore`/`README.md` |
| repository tests | **67 passed in 43.63 s** |
| real Qt/VTK GUI launch smoke | **PASS**, exit 0 |

The real GUI launch returned:

`GUI_SMOKE_PASS 0 ['Simulator', 'Dataset', 'Identification']`

Launch command:

```powershell
.\.venv\Scripts\python.exe run_simulator.py
```

## Scientific stop

Milestone 3A.4 stops here as required.

- UAV method unchanged.
- Residual unchanged.
- No delay adopted.
- No new model experiment.
- No EI/Cb fit.
- No Stage C.
- No MPPI.
- Untouched evaluation pending user-provided role assignment.
- UAV boundary-model acceptance pending explicit user decision after that result.

