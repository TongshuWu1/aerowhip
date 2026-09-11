# Preliminary1: repaired 145 g drone, 17 g cable

User-authorized fresh M0 identification, 9 September 2026. Job:
`runs/adaptation/20260909-preliminary1-M0-v2`. **Completed; all workers finished.**
The saved candidate is provisional and remains unselected because coupled cable
accuracy is insufficient for precise strike prediction. This is one fit, with no automatic folds, PPO, MPPI
optimization or flight. All earlier system artifacts remain in the external archive.

## Raw observations and preparation

The five user-supplied OptiTrack hand trims are preserved. The vertical pair is
now `vertical_figure8_001.csv` / `experiment_vertical_figure8_001.csv` in the source
and imported copy. Native CSV contents, including historical Motive Take Name,
are unchanged. `runs/data_review/preliminary1-intake/renaming.json` preserves old
names, new paths and verified hashes. The original intake audit is unchanged.

Use native cf_3 pose/quaternion and cable1:c1-c10 at 100 Hz for observations.
Controller XYZ is a cached OptiTrack copy (~10 Hz changes), not an independent
measurement. Measured-to-measured XYZ matching estimates one clock offset per
take; commands do not enter this alignment objective. Controller time equals
native time plus the saved offset. Approximately 2–4 mm stream agreement does
not establish actuator latency; cache/transport delay remains a nuisance.
Four temporal chunk checks and per-take offsets are in `preparation.json`.

The logger records cached PVA commands at ~100 Hz with actual packet arrivals
near 30 Hz. Receipt time is `time_s - cmd_age`; repeated cached timestamps are
grouped before enforcing strictly increasing packet times. PVA values are
unchanged. Invalid/stale command coverage excludes affected windows. A discrete
0–120 ms effective response delay is selected using training data only.

There is no Z normalization, command-based coordinate correction, raw-array
interpolation overwrite, destructive trimming or missing-tail reconstruction.
The supplied geometry and rotated tracked-origin-to-attachment offset are used
consistently. Individual masses remain proportionally scaled to the measured
17 g assembly total; they were not individually remeasured.

Full arrays and explicit masks are saved per take in `inputs/*/data.npz`.
Masks flag nonfinite observations, conservative native position jumps and
marker chords longer than saved interval length plus 15 mm. These data-quality
flags are not vehicle limits. Interpolation requires adjacent finite samples.
The vertical take's final 342 all-marker-missing samples remain missing.
Valid drone data can still be used when cable observations are unavailable.

## Roles, windows and initialization

Training: figure8_001, osci_001, osci_002, vertical_figure8_001.
Validation: whole figure8_002, excluded from optimization and weight selection.
Longer takes do not dominate: training losses have equal total weight per take.
This is a same-session held-out take, not an independent hardware or prospective
flight test. Alignment and state estimation on the validation take still use its
own recorded measurements.

Fixed two-second drone windows with two-second spacing span the recordings.
There are 69 training and 26 validation windows. Their preceding 0.4 s supplies
initial motion estimates; window outputs never supply initial state. No stationary
pre-hover was recorded, so rest must not be invented. Endpoint quadratic fits
estimate initial linear/angular velocity. Mean acceleration over preceding
history minus mean nominal feedback/feedforward estimates the fixed compensation
state. Mean preceding orientation and driven acceleration estimate effective
attitude alignment. Both are nuisance estimates, not firmware I-term readings or
independent mounting calibration. Short windows limit the frozen-memory assumption.

Each drone window supplies up to two one-second cable windows. Cable state uses
only past marker history, projects onto the saved inextensible model and records
projection magnitude. Missing history or observations exclude the cable window;
15 of 190 proposed windows are rejected, leaving 175. Cable fitting uses measured
rotated attachment; coupled diagnostics use predicted attachment after initialization.
Initialization histories can overlap an earlier window's observations, so windows
are not independent samples. The whole-take split prevents train/validation overlap.

## Model, acceleration and stopping

No old fitted parameters or weights are loaded. Fresh engineering nominal values
and zero-output residual heads initialize the saved `source_candidate`.
Fit order: drone translation gains/effective delay, attitude response, bounded
drone acceleration residual, attitude refinement, cable EI/Cb, cable damping plus
bounded acceleration residual. These are effective loaded-drone/cable dynamics;
they do not identify firmware gains, thrust curves or battery compensation.

Windows, parameter candidates and residual computation are batched on the RTX 4080
using CUDA graphs. Cable fitting uses the verified tridiagonal constraint solver
and vectorized geometry. Float64 preserves the existing simulator's numerical
contract. Small trust-region parameter decisions stay on CPU; rollout and
sensitivity computations run on GPU. Training is not made faster by reducing
physics substeps or detaching recurrent states. Residuals receive full temporal
gradients over their saved windows.

Practical plateau stopping is training-only. Drone checks every 5 updates after
at least 40, patience 5, relative threshold 0.5%, safety ceiling 400. Cable checks
every 3 after at least 12, patience 4, threshold 0.5%, safety ceiling 120. Best
weights, current weights, optimizer and stopping history are saved. Solver
termination, plateau and safety ceiling are distinct outcomes. Parameter/delay
search boundaries are not proof of physical identification or global convergence.

## Verification and artifacts

Windows/roles/protocol, source CSVs, input arrays, seed assets and the 265-file
fit source snapshot have saved SHA-256 hashes. The worker checks protected bytes
before publication. The preparation-only first attempt is preserved; it failed
on cached receipt timestamp jitter before any fitting.

Eight focused preparation/import/UI tests and 22 pose-response tests pass on Windows. GPU smoke checks:
translation matches the reference exactly; six analytic sensitivity columns
agree with finite differences within 3.84e-8; attitude differs by <1.5e-15;
drone residual loss/gradient matches exactly; cable fast/direct difference is
<2.1e-14 m and differentiable/inference outputs match with finite gradients.
These checks used a small window sample and are recorded in
`preflight_verification.json`; final candidate checks assess the fitted model.

Candidate publication requires loadable assets, finite recursive rollouts,
training/execution consistency and unchanged protected source bytes. Results
include all-window cold/fitted errors and one deterministically selected coupled
two-second replay per take with saved arrays. Invalid cable observations remain
masked. These are retrospective short-window diagnostics, not complete 50-second
open-loop predictions or demonstrated whip/flight improvement.

The native **Models & fitting** page recognizes this job, shows its stage,
update bar and loss curves, and distinguishes training from validation rows.
Another fit requires a new reviewed job; the fresh experiment cannot fall back
to the historical normalized adp0 fitter.

## Fit result and limitations

Drone residual: best update 400, safety ceiling reached before plateau; selected
training objective 0.93844 versus initial 2.91625. Cable residual: best update 12,
practical plateau, objective 0.98632 versus initial 0.99070. A plateau here denotes
lack of meaningful training progress, not verified parameter identifiability.
Cable gradients were finite but sometimes extremely large (up to 1.44e16 before
norm clipping); the cable residual provides only a small objective improvement.
Do not call the cable residual well converged or independently validated.

Median update times on Windows/RTX 4080: 0.368 s drone, 5.215 s cable. The 400/12
updates took 147.1/62.3 s respectively, excluding preparation, graph capture,
selection checks and final replays. Observed GPU utilization reached 100%, with
about 5.4–5.9 GB allocated by the device across active processes. These are local
measurements, not a CPU speedup benchmark or another-platform validation.

The selected effective delay is 20 ms. Cable EI is about 1.00e-8 N·m², Cb about
1.00e-4 N·m²·s. The physical solver terminated on parameter-step tolerance near
the selected grid point; that grid point lies at the initial grid extremes.
Attitude Z response scale is near its search upper bound (1.498 versus 1.5).
Treat these as provisional effective values, not measured material/controller
constants. Cable initialization projection has median max-coordinate displacement
7.49 mm and maximum 14.15 mm; raw marker observations are not moved.

Held-out figure8_002 retrospective errors, unchanged by post-fit review:

| Prediction | Cold seed | Fitted M0 |
|---|---:|---:|
| Drone position, 2 s windows | 32.83 cm | 11.99 cm |
| Cable tip, 1 s windows with measured attachment | 12.83 cm | 11.11 cm |

Residual optimization and nominal selection never used this held-out error. Cable
improvement is modest and some training takes worsen slightly in tip RMSE. These
numbers do not demonstrate complete-system whip accuracy or real adaptation.

The initial publication gate failed because it required exact equality between
two slightly different integration partitions. The training evaluator agrees
exactly with `predict_pose`; production removes floating-point ceiling
over-subdivision at exact 5 ms boundaries. Its initial difference was 1.34 µm.
The saved post-fit review checks exact reference agreement (<1e-10 m) and separately
requires production agreement within 0.1 mm, recording actual differences.
`review_amendment.json` and `review_snapshot/` preserve this change. No weights or
training choices were changed, and the failed publication artifacts remain in
`candidate_failed_check/` with `fit_complete=false`.

Final review passed: all five training/reference translation comparisons are
exact; production differences are at most 6.81 µm. Ten coupled predictions are
finite. The 292 protected input/source files, review snapshot and imported raw
CSV copies passed SHA-256 verification. `candidate/artifact_manifest.json` and
`final_verification.json` record the published result. All best/current weights
and optimizer histories remain saved. No training was repeated after validation.

The coupled check exposes a larger limitation than measured-root cable windows:

| Representative 2 s coupled cable-tip RMS | Cold seed | Fitted M0 |
|---|---:|---:|
| figure8_001 (training) | 76.29 cm | 66.26 cm |
| figure8_002 (validation) | 101.54 cm | 64.62 cm |
| osci_001 (training) | 53.34 cm | 7.56 cm |
| osci_002 (training) | 154.73 cm | 31.07 cm |
| vertical_figure8_001 (training) | 43.12 cm | 4.04 cm |

These are one predetermined representative window per take, not whole-take
averages. Fitted predictions improve over the cold seed but remain poor for the
figure-eight cable motion. The model is saved for inspection; `flight_ready=false`
and the active experiment's model selection remains empty. The UI displays both
measured-root and coupled errors. A lower training loss must not be substituted
for this combined-system result. Further model/initialization investigation is
needed before a precise MPPI strike claim; no new fit or planner was started.
